//! Modifiers to influence the output (rendering) of each particle.

use std::hash::Hash;

use bevy::prelude::*;
use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::{
    impl_mod_render, Attribute, BoxedModifier, CpuValue, EvalContext, ExprError, ExprHandle,
    Gradient, Modifier, ModifierContext, Module, RenderContext, RenderModifier, ShaderCode,
    ShaderWriter, ToWgslString,
};

/// Mapping of the sample read from a texture image to the base particle color.
///
/// This defines the way the texture image of [`ParticleTextureModifier`] blends
/// with the base particle color to define the final render color of the
/// particle.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum ImageSampleMapping {
    /// Modulate the particle's base color with the full RGBA sample of the
    /// texture image.
    ///
    /// ```wgsl
    /// color = baseColor * texColor;
    /// ```
    #[default]
    Modulate,

    /// Modulate the particle's base color with the RGB sample of the texture
    /// image, leaving the alpha component unmodified.
    ///
    /// ```wgsl
    /// color.rgb = baseColor.rgb * texColor.rgb;
    /// ```
    ModulateRGB,

    /// Modulate the alpha component (opacity) of the particle's base color with
    /// the red component of the sample of the texture image.
    ///
    /// ```wgsl
    /// color.a = baseColor.a * texColor.r;
    /// ```
    ModulateOpacityFromR,
}

impl ImageSampleMapping {
    /// Convert this mapping to shader code for the given texture slot.
    pub fn to_shader_code(&self, texture_slot: &str) -> String {
        match *self {
            ImageSampleMapping::Modulate => format!("color = color * texColor{texture_slot};"),
            ImageSampleMapping::ModulateRGB => {
                format!("color = vec4<f32>(color.rgb * texColor{texture_slot}.rgb, color.a);")
            }
            ImageSampleMapping::ModulateOpacityFromR => {
                format!("color.a = color.a * texColor{texture_slot}.r;")
            }
        }
    }
}

/// A modifier modulating each particle's color by sampling a texture.
///
/// # Attributes
///
/// This modifier does not require any specific particle attribute.
#[derive(Debug, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize)]
pub struct ParticleTextureModifier {
    /// Index of the texture slot containing the texture to use. The slot is
    /// defined in the [`Module`], and the actual texture is bound via the
    /// [`EffectMaterial`] component.
    ///
    /// [`EffectMaterial`]: crate::EffectMaterial
    pub texture_slot: ExprHandle,

    /// The mapping of the texture image samples to the base particle color.
    pub sample_mapping: ImageSampleMapping,
}

impl ParticleTextureModifier {
    /// Create a new modifier with the default [`ImageSampleMapping`].
    pub fn new(texture_slot: ExprHandle) -> Self {
        Self {
            texture_slot,
            sample_mapping: default(),
        }
    }
}

impl_mod_render!(ParticleTextureModifier, &[]); // TODO - should require some UV maybe?

#[cfg_attr(feature = "serde", typetag::serde)]
impl RenderModifier for ParticleTextureModifier {
    fn apply_render(
        &self,
        module: &mut Module,
        context: &mut RenderContext,
    ) -> Result<(), ExprError> {
        context.set_needs_uv();
        let code = self.eval(module, context)?;
        context.fragment_code += &code;
        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(*self)
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}

impl ParticleTextureModifier {
    /// Evaluate the modifier to generate the shader code.
    pub fn eval(
        &self,
        module: &Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        let texture_slot = module.try_get(self.texture_slot)?;
        let texture_slot = texture_slot.eval(module, context)?;
        let sample_mapping = self.sample_mapping.to_shader_code(&texture_slot[..]);

        let sample_mapping_name = format!("{:?}", self.sample_mapping);

        // Build a switch statement to select the texture/sampler.
        // FIXME - Ideally with bindless (texture/sampler arrays with dynamic indices)
        // we don't need this. But bindless is not available on Web anyway, so this is a
        // safe fallback.
        let mut code = String::with_capacity(1024);
        code += &format!(
            "    // ParticleTextureModifier
    var texColor{texture_slot}: vec4<f32>;
    switch ({texture_slot}) {{\n"
        );
        let count = module.texture_layout().layout.len() as u32;
        for index in 0..count {
            let wgsl_index = index.to_wgsl_string();
            code += &format!("      case {wgsl_index}: {{ texColor{texture_slot} = textureSample(material_texture_{index}, material_sampler_{index}, uv); }}\n");
        }
        code += &format!("      default: {{ texColor{texture_slot} = vec4<f32>(0.0); }}\n");
        code += &format!(
            "    }}
    // Sample mapping: {sample_mapping_name}
    {sample_mapping}"
        );
        Ok(code)
    }
}

/// Color blending modes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum ColorBlendMode {
    /// Overwrite the destination color with the source one.
    #[default]
    Overwrite,
    /// Add the source color to the destination one.
    Add,
    /// Multiply the source color by the destination one.
    Modulate,
}

impl ColorBlendMode {
    /// Convert the blend mode to the string representation of its operator.
    pub fn to_assign_operator(&self) -> String {
        match *self {
            ColorBlendMode::Overwrite => "=",
            ColorBlendMode::Add => "+=",
            ColorBlendMode::Modulate => "*=",
        }
        .to_string()
    }
}

/// Color component write mask for blending colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct ColorBlendMask(u8);

bitflags! {
    impl ColorBlendMask: u8 {
        /// First component (red).
        const R = 0b0001;
        /// Second component (green).
        const G = 0b0010;
        /// Third component (blue).
        const B = 0b0100;
        /// Last component (alpha).
        const A = 0b1000;

        /// RGB mask (skip alpha).
        const RGB = 0b0111;

        /// RGBA mask (all components).
        const RGBA = 0b1111;
    }
}

impl Default for ColorBlendMask {
    fn default() -> Self {
        Self::RGBA
    }
}

impl ColorBlendMask {
    /// Convert the mask to a string of components, e.g. "rgb" or "ra".
    pub fn to_components(&self) -> String {
        let cmp = ['r', 'g', 'b', 'a'];
        (0..=3).fold(String::with_capacity(4), |mut acc, i| {
            let mask = ColorBlendMask::from_bits_truncate(1u8 << i);
            if self.contains(mask) {
                acc.push(cmp[i]);
            }
            acc
        })
    }
}

/// A modifier to set the rendering color of all particles.
///
/// This modifier assigns a _single_ color to all particles. That color can be
/// determined by the user with [`CpuValue::Single`], or left randomized with
/// [`CpuValue::Uniform`], but will be the same color for all particles.
///
/// # Attributes
///
/// This modifier does not require any specific particle attribute.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct SetColorModifier {
    /// The particle color.
    pub color: CpuValue<Vec4>,
    /// The color blend mode.
    pub blend: ColorBlendMode,
    /// The blend mask.
    pub mask: ColorBlendMask,
}

impl SetColorModifier {
    /// Create a new modifier with the default color blend and mask.
    pub fn new<C>(color: C) -> Self
    where
        C: Into<CpuValue<Vec4>>,
    {
        Self {
            color: color.into(),
            blend: default(),
            mask: default(),
        }
    }
}

impl_mod_render!(SetColorModifier, &[]);

#[cfg_attr(feature = "serde", typetag::serde)]
impl RenderModifier for SetColorModifier {
    fn apply_render(
        &self,
        _module: &mut Module,
        context: &mut RenderContext,
    ) -> Result<(), ExprError> {
        let op = self.blend.to_assign_operator();
        let col = self.color.to_wgsl_string();
        let s = if self.mask == ColorBlendMask::RGBA {
            format!("color {op} {col};\n")
        } else {
            let mask = self.mask.to_components();
            format!("color.{mask} {op} ({col}).{mask};\n")
        };
        context.vertex_code += &s;
        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(*self)
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}

/// A modifier modulating each particle's color over its lifetime with a
/// gradient curve.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::AGE`]
/// - [`Attribute::LIFETIME`]
#[derive(Debug, Default, Clone, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub struct ColorOverLifetimeModifier {
    /// The color gradient defining the particle color based on its lifetime.
    pub gradient: Gradient<Vec4>,
    /// The color blend mode.
    pub blend: ColorBlendMode,
    /// The blend mask.
    pub mask: ColorBlendMask,
}

impl ColorOverLifetimeModifier {
    /// Create a new modifier from a given gradient.
    pub fn new(gradient: Gradient<Vec4>) -> Self {
        Self {
            gradient,
            blend: default(),
            mask: default(),
        }
    }
}

impl_mod_render!(
    ColorOverLifetimeModifier,
    &[Attribute::AGE, Attribute::LIFETIME]
);

#[cfg_attr(feature = "serde", typetag::serde)]
impl RenderModifier for ColorOverLifetimeModifier {
    fn apply_render(
        &self,
        _module: &mut Module,
        context: &mut RenderContext,
    ) -> Result<(), ExprError> {
        let func_name = context.add_color_gradient(self.gradient.clone());
        context.render_extra += &format!(
            r#"fn {0}(key: f32) -> vec4<f32> {{
    {1}
}}

"#,
            func_name,
            self.gradient.to_shader_code("key")
        );

        let op = self.blend.to_assign_operator();
        let col = format!(
            "{0}(particle.{1} / particle.{2})",
            func_name,
            Attribute::AGE.name(),
            Attribute::LIFETIME.name()
        );
        let s = if self.mask == ColorBlendMask::RGBA {
            format!("color {op} {col};\n")
        } else {
            let mask = self.mask.to_components();
            let non_mask = match self.blend {
                ColorBlendMode::Overwrite => {
                    format!("color.{}", self.mask.complement().to_components())
                }
                ColorBlendMode::Add => "0".to_string(),
                ColorBlendMode::Modulate => "1".to_string(),
            };
            format!("color {op} vec4<f32>(({col}).{mask}, {non_mask});\n")
        };

        context.vertex_code += &s;
        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(self.clone())
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}

/// A modifier modulating each particle's color based on its velocity using a
/// gradient curve.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Default, Clone, PartialEq, Reflect, serde::Serialize, serde::Deserialize)]
pub struct ColorByVelocityModifier {
    /// The color gradient defining the particle color based on its lifetime.
    pub gradient: Gradient<Vec4>,
    /// The color blend mode.
    pub blend: ColorBlendMode,
    /// The blend mask.
    pub mask: ColorBlendMask,
    /// the velocity range to use for the color gradient
    pub velocity_range: std::ops::Range<f32>,
}

impl ColorByVelocityModifier {
    /// Create a new modifier from a given gradient.
    pub fn new(gradient: Gradient<Vec4>) -> Self {
        Self {
            gradient,
            blend: default(),
            mask: default(),
            velocity_range: 0.0..1.0,
        }
    }
}

impl_mod_render!(ColorByVelocityModifier, &[Attribute::VELOCITY]);

#[cfg_attr(feature = "serde", typetag::serde)]
impl RenderModifier for ColorByVelocityModifier {
    fn apply_render(
        &self,
        _module: &mut Module,
        context: &mut RenderContext,
    ) -> Result<(), ExprError> {
        let func_name = context.add_color_gradient(self.gradient.clone());
        context.render_extra += &format!(
            r#"fn {0}(key: f32) -> vec4<f32> {{
    {1}
}}

"#,
            func_name,
            self.gradient.to_shader_code("key")
        );

        let op = self.blend.to_assign_operator();
        let col = format!(
            "{func}(clamp(length(particle.{vel}), {min}, {max}) / {range});\n",
            func = func_name,
            vel = Attribute::VELOCITY.name(),
            min = self.velocity_range.start.to_wgsl_string(),
            max = self.velocity_range.end.to_wgsl_string(),
            range = (self.velocity_range.end - self.velocity_range.start).to_wgsl_string()
        );
        let s = if self.mask == ColorBlendMask::RGBA {
            format!("color {op} {col};\n")
        } else {
            let mask = self.mask.to_components();
            format!("color.{mask} {op} ({col}).{mask};\n")
        };

        context.vertex_code += &s;
        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(self.clone())
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}

/// A modifier to set the size of all particles.
///
/// This modifier assigns a _single_ size to all particles. That size can be
/// determined by the user with [`CpuValue::Single`], or left randomized with
/// [`CpuValue::Uniform`], but will be the same size for all particles.
///
/// # Attributes
///
/// This modifier does not require any specific particle attribute. The size of
/// the particle is extracted from the [`Attribute::SIZE`],
/// [`Attribute::SIZE2`], or [`Attribute::SIZE3`] if any, but even if they're
/// absent this modifier acts on the default particle size.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct SetSizeModifier {
    /// The 3D particle size.
    pub size: CpuValue<Vec3>,
}

impl_mod_render!(SetSizeModifier, &[]);

#[cfg_attr(feature = "serde", typetag::serde)]
impl RenderModifier for SetSizeModifier {
    fn apply_render(
        &self,
        _module: &mut Module,
        context: &mut RenderContext,
    ) -> Result<(), ExprError> {
        context.vertex_code += &format!("size = {0};\n", self.size.to_wgsl_string());
        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(*self)
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}

/// A modifier modulating each particle's size over its lifetime with a gradient
/// curve.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::AGE`]
/// - [`Attribute::LIFETIME`]
#[derive(Debug, Default, Clone, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub struct SizeOverLifetimeModifier {
    /// The size gradient defining the particle size based on its lifetime.
    pub gradient: Gradient<Vec3>,
    /// Is the particle size in screen-space logical pixel? If `true`, the size
    /// is in screen-space logical pixels, and not affected by the camera
    /// projection. If `false`, the particle size is in world units.
    pub screen_space_size: bool,
}

impl_mod_render!(
    SizeOverLifetimeModifier,
    &[Attribute::AGE, Attribute::LIFETIME]
);

#[cfg_attr(feature = "serde", typetag::serde)]
impl RenderModifier for SizeOverLifetimeModifier {
    fn apply_render(
        &self,
        _module: &mut Module,
        context: &mut RenderContext,
    ) -> Result<(), ExprError> {
        let func_name = context.add_size_gradient(self.gradient.clone());
        context.render_extra += &format!(
            r#"fn {0}(key: f32) -> vec3<f32> {{
    {1}
}}

"#,
            func_name,
            self.gradient.to_shader_code("key")
        );

        context.vertex_code += &format!(
            "size = {0}(particle.{1} / particle.{2});\n",
            func_name,
            Attribute::AGE.name(),
            Attribute::LIFETIME.name()
        );

        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(self.clone())
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}

/// Mode of orientation of a particle's local frame.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum OrientMode {
    /// Orient a particle such that its local XY plane is parallel to the
    /// camera's near and far planes (depth planes).
    ///
    /// By default the local X axis is (1,0,0) in camera space, and the local Y
    /// axis is (0,1,0). The local Z axis is (0,0,1), perpendicular to the
    /// camera depth planes, pointing toward the camera. If an
    /// [`OrientModifier::rotation`] is provided, it defines a rotation in the
    /// local X-Y plane, relative to that default.
    ///
    /// This mode is a bit cheaper to calculate than [`FaceCameraPosition`], and
    /// should be preferred to it unless the particle absolutely needs to have
    /// its Z axis pointing to the camera position.
    ///
    /// This is the default variant.
    ///
    /// [`FaceCameraPosition`]: crate::modifier::output::OrientMode::FaceCameraPosition
    #[default]
    ParallelCameraDepthPlane,

    /// Orient a particle to face the camera's position.
    ///
    /// The local Z axis of the particle points directly at the camera position.
    /// The X and Y axes form an orthonormal frame with it, where Y is roughly
    /// upward. If an [`OrientModifier::rotation`] is provided, it defines a
    /// rotation in the local X-Y plane, relative to that default.
    ///
    /// This mode is a bit more costly to calculate than
    /// [`ParallelCameraDepthPlane`], and should be used only when the
    /// particle absolutely needs to have its Z axis pointing to the camera
    /// position.
    ///
    /// [`ParallelCameraDepthPlane`]: crate::modifier::output::OrientMode::ParallelCameraDepthPlane
    FaceCameraPosition,

    /// Orient a particle alongside its velocity.
    ///
    /// The local X axis points alongside the velocity. The local Y axis is
    /// derived as the cross product of the camera direction with that local X
    /// axis. The Z axis completes the orthonormal basis. This allows flat
    /// particles (quads) to roughly face the camera position (as long as
    /// velocity is not perpendicular to the camera depth plane), while having
    /// their X axis always pointing alongside the velocity.
    ///
    /// With this mode, any provided [`OrientModifier::rotation`] is ignored.
    AlongVelocity,
}

/// Orients the particle's local frame.
///
/// The orientation is calculated during the rendering of each particle. An
/// additional in-plane rotation can be optionally specified; its meaning
/// depends on the [`OrientMode`] in use.
///
/// # Attributes
///
/// The required attribute(s) depend on the orientation [`mode`]:
/// - [`OrientMode::ParallelCameraDepthPlane`]: This modifier does not require
///   any specific particle attribute.
/// - [`OrientMode::FaceCameraPosition`]: This modifier requires the
///   [`Attribute::POSITION`] attribute.
/// - [`OrientMode::AlongVelocity`]: This modifier requires the
///   [`Attribute::POSITION`] and [`Attribute::VELOCITY`] attributes.
///
/// [`mode`]: crate::modifier::output::OrientModifier::mode
/// [`Attribute::POSITION`]: crate::attributes::Attribute::POSITION
#[derive(Debug, Default, Clone, Copy, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub struct OrientModifier {
    /// Orientation mode for the particles.
    pub mode: OrientMode,
    /// Optional in-plane rotation expression, as a single `f32` angle in
    /// radians.
    ///
    /// The actual meaning depends on [`OrientMode`], and the rotation may be
    /// ignored for some mode(s).
    pub rotation: Option<ExprHandle>,
}

impl OrientModifier {
    /// Create a new instance of this modifier with the given orient mode.
    pub fn new(mode: OrientMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Set the rotation expression for the particles.
    pub fn with_rotation(mut self, rotation: ExprHandle) -> Self {
        self.rotation = Some(rotation);
        self
    }
}

#[cfg_attr(feature = "serde", typetag::serde)]
impl Modifier for OrientModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Render
    }

    fn as_render(&self) -> Option<&dyn RenderModifier> {
        Some(self)
    }

    fn as_render_mut(&mut self) -> Option<&mut dyn RenderModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[Attribute] {
        // Note: don't required AXIS_X/Y/Z, they're written at the last minute in the
        // render shader alone, so don't need to be stored as part of the particle's
        // layout for simulation.
        match self.mode {
            OrientMode::ParallelCameraDepthPlane => &[],
            OrientMode::FaceCameraPosition => &[Attribute::POSITION],
            OrientMode::AlongVelocity => &[Attribute::POSITION, Attribute::VELOCITY],
        }
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn apply(&self, _module: &mut Module, _context: &mut ShaderWriter) -> Result<(), ExprError> {
        Err(ExprError::TypeError("Wrong modifier context".to_string()))
    }
}

#[cfg_attr(feature = "serde", typetag::serde)]
impl RenderModifier for OrientModifier {
    fn apply_render(
        &self,
        module: &mut Module,
        context: &mut RenderContext,
    ) -> Result<(), ExprError> {
        match self.mode {
            OrientMode::ParallelCameraDepthPlane => {
                if let Some(rotation) = self.rotation {
                    let rotation = context.eval(module, rotation)?;
                    context.vertex_code += &format!(
                        r#"let cam_rot = get_camera_rotation_effect_space();
let particle_rot_in_cam_space = {};
let particle_rot_in_cam_space_cos = cos(particle_rot_in_cam_space);
let particle_rot_in_cam_space_sin = sin(particle_rot_in_cam_space);
axis_x = cam_rot[0].xyz * particle_rot_in_cam_space_cos + cam_rot[1].xyz * particle_rot_in_cam_space_sin;
axis_y = cam_rot[0].xyz * particle_rot_in_cam_space_sin - cam_rot[1].xyz * particle_rot_in_cam_space_cos;
axis_z = cam_rot[2].xyz;
"#,
                        rotation
                    );
                } else {
                    context.vertex_code += r#"let cam_rot = get_camera_rotation_effect_space();
axis_x = cam_rot[0].xyz;
axis_y = cam_rot[1].xyz;
axis_z = cam_rot[2].xyz;
"#;
                }
            }
            OrientMode::FaceCameraPosition => {
                if let Some(rotation) = self.rotation {
                    let rotation = context.eval(module, rotation)?;
                    context.vertex_code += &format!(
                        r#"axis_z = normalize(get_camera_position_effect_space() - position);
let particle_rot_in_cam_space = {};
let particle_rot_in_cam_space_cos = cos(particle_rot_in_cam_space);
let particle_rot_in_cam_space_sin = sin(particle_rot_in_cam_space);
let axis_x0 = normalize(cross(view.world_from_view[1].xyz, axis_z));
let axis_y0 = cross(axis_z, axis_x0);
axis_x = axis_x0 * particle_rot_in_cam_space_cos + axis_y0 * particle_rot_in_cam_space_sin;
axis_y = axis_x0 * particle_rot_in_cam_space_sin - axis_y0 * particle_rot_in_cam_space_cos;
"#,
                        rotation
                    );
                } else {
                    context.vertex_code += r#"axis_z = normalize(get_camera_position_effect_space() - position);
axis_x = normalize(cross(view.world_from_view[1].xyz, axis_z));
axis_y = cross(axis_z, axis_x);
"#;
                }
            }
            OrientMode::AlongVelocity => {
                context.vertex_code += r#"let dir = normalize(position - get_camera_position_effect_space());
axis_x = normalize(particle.velocity);
axis_y = cross(dir, axis_x);
axis_z = cross(axis_x, axis_y);
"#;
            }
        }

        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(*self)
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}

/// A modifier to render particles using flipbook animation.
///
/// Flipbook animation renders multiple still images at interactive framerate
/// (generally 10+ FPS) to give the illusion of animation. The still images are
/// sprites, taken from a single texture acting as a sprite sheet. This requires
/// using the [`ParticleTextureModifier`] to specify the source texture
/// containing the sprite sheet image. The [`FlipbookModifier`] itself only
/// activates flipbook rendering, specifying how to slice the texture into a
/// sprite sheet (list of sprites).
///
/// The flipbook renderer reads the [`Attribute::SPRITE_INDEX`] of the particle
/// and selects a region of the particle texture specified via
/// [`ParticleTextureModifier`]. Note that [`FlipbookModifier`] by itself
/// doesn't animate anything; instead, the animation comes from a varying value
/// of [`Attribute::SPRITE_INDEX`].
///
/// There's no built-in modifier to update the [`Attribute::SPRITE_INDEX`];
/// instead you should use a [`SetAttributeModifier`] with a suitable
/// expression. A common example is to base the sprite index on the particle
/// age, accessed from [`Attribute::AGE`]. Note that in that case the
/// [`Attribute::AGE`] being a floating point value must be cast to an integer
/// to be assigned to [`Attribute::SPRITE_INDEX`].
///
/// # Example
///
/// ```
/// # use bevy::prelude::*;
/// # use bevy_hanabi::*;
/// # let texture = Handle::<Image>::default();
/// let writer = ExprWriter::new();
///
/// let lifetime = writer.lit(5.).expr();
/// let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);
///
/// // Age goes from 0 to LIFETIME=5s
/// let age = writer.lit(0.).expr();
/// let init_age = SetAttributeModifier::new(Attribute::AGE, age);
///
/// // sprite_index = i32(particle.age) % 4;
/// let sprite_index = writer
///     .attr(Attribute::AGE)
///     .cast(ScalarType::Int)
///     .rem(writer.lit(4i32))
///     .expr();
/// let update_sprite_index = SetAttributeModifier::new(Attribute::SPRITE_INDEX, sprite_index);
///
/// let texture_slot = writer.lit(0u32).expr();
///
/// let asset = EffectAsset::new(32768, SpawnerSettings::once(32.0.into()), writer.finish())
///     .with_name("flipbook")
///     .init(init_age)
///     .init(init_lifetime)
///     .update(update_sprite_index)
///     .render(ParticleTextureModifier {
///         texture_slot,
///         sample_mapping: ImageSampleMapping::ModulateOpacityFromR,
///     })
///     .render(FlipbookModifier {
///         sprite_grid_size: UVec2::new(2, 2), // 4 frames
///     });
/// ```
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::SPRITE_INDEX`]
///
/// [`SetAttributeModifier`]: crate::modifier::attr::SetAttributeModifier
#[derive(Debug, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize)]
pub struct FlipbookModifier {
    /// Flipbook sprite sheet grid size.
    ///
    /// The grid size defines the number of sprites in the texture, and
    /// implicitly their size as the total texture size divided by the grid
    /// size.
    ///
    /// To animate the rendered sprite index, modify the
    /// [`Attribute::SPRITE_INDEX`] property. The value of that attribute should
    /// ideally be in the `[0:N-1]` range where `N = grid.x * grid.y` is the
    /// total number of sprites. However values outside that range will not
    /// produce any error, but will yield texture UV coordinates outside the
    /// `[0:1]` range.
    pub sprite_grid_size: UVec2,
}

impl Default for FlipbookModifier {
    fn default() -> Self {
        // Default to something which animates, to help debug mistakes.
        Self {
            sprite_grid_size: UVec2::ONE * 2,
        }
    }
}

impl_mod_render!(FlipbookModifier, &[Attribute::SPRITE_INDEX]);

#[cfg_attr(feature = "serde", typetag::serde)]
impl RenderModifier for FlipbookModifier {
    fn apply_render(
        &self,
        _module: &mut Module,
        context: &mut RenderContext,
    ) -> Result<(), ExprError> {
        context.sprite_grid_size = Some(self.sprite_grid_size);
        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(*self)
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}

/// A modifier to interpret the size of all particles in screen-space pixels.
///
/// This modifier assigns a pixel size to particles in screen space, ignoring
/// the distance to the camera and perspective. It effectively scales the
/// existing [`Attribute::SIZE`] of each particle to negate the perspective
/// correction usually applied to rendered objects based on their distance to
/// the camera.
///
/// Note that this modifier should generally be placed last in the stack, or at
/// least after any modifier which might modify the particle position or its
/// size. Otherwise the scaling will be incorrect.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
///
/// If the [`Attribute::SIZE`], [`Attribute::SIZE2`], or [`Attribute::SIZE3`]
/// are present, they're used to initialize the particle's size. Otherwise the
/// default size is used. So this modifier doesn't require any size attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct ScreenSpaceSizeModifier;

impl_mod_render!(
    ScreenSpaceSizeModifier,
    &[Attribute::POSITION, Attribute::SIZE]
);

#[cfg_attr(feature = "serde", typetag::serde)]
impl RenderModifier for ScreenSpaceSizeModifier {
    fn apply_render(
        &self,
        _module: &mut Module,
        context: &mut RenderContext,
    ) -> Result<(), ExprError> {
        // Get perspective divide factor from clip space position. This is the "average"
        // factor for the entire particle, taken at its position (mesh origin),
        // and applied uniformly for all vertices. Scale size by w_cs to negate
        // the perspective divide which will happen later after the vertex shader.
        // The 2.0 factor is because clip space is in [-1:1] so we need to divide by the
        // half screen size only.
        // Note: here "size" is the built-in render size, which is always defined and
        // called "size", and which may or may not be the Attribute::SIZE/2
        // attribute(s).
        context.vertex_code += &format!(
            "let w_cs = transform_position_simulation_to_clip(particle.{0}).w;\n
            let screen_size_pixels = view.viewport.zw;\n
            let projection_scale = vec2<f32>(view.clip_from_view[0][0], view.clip_from_view[1][1]);\n
            size = (size * w_cs * 2.0) / min(screen_size_pixels.x * projection_scale.x, screen_size_pixels.y * projection_scale.y);\n",
            Attribute::POSITION.name());
        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(*self)
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}

/// Makes particles round.
///
/// The shape of each particle is a [squircle] (like a rounded rectangle, but
/// faster to evaluate). The `roundness` parameter specifies how round the shape
/// is. At 0.0, the particle is a rectangle; at 1.0, the particle is an
/// ellipse.
///
/// Given x and y from (-1, 1), the equation of the shape of the particle is
/// |x|ⁿ + |y|ⁿ = 1, where n = 2 / `roundness``.
///
/// Note that this modifier is presently incompatible with the
/// [`FlipbookModifier`]. Attempts to use them together will produce unexpected
/// results.
///
/// [squircle]: https://en.wikipedia.org/wiki/Squircle
#[derive(Debug, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize)]
pub struct RoundModifier {
    /// How round the particle is.
    ///
    /// This ranges from 0.0 for a perfect rectangle to 1.0 for a perfect
    /// ellipse. 1/3 produces a nice rounded rectangle shape.
    ///
    /// n in the squircle formula is calculated as (2 / roundness).
    pub roundness: ExprHandle,
}

impl_mod_render!(RoundModifier, &[]);

#[cfg_attr(feature = "serde", typetag::serde)]
impl RenderModifier for RoundModifier {
    fn apply_render(
        &self,
        module: &mut Module,
        context: &mut RenderContext,
    ) -> Result<(), ExprError> {
        context.set_needs_uv();

        let roundness = context.eval(module, self.roundness)?;
        context.fragment_code += &format!(
            "let roundness = {};
            if (roundness > 0.0f) {{
                let n = 2.0f / roundness;
                if (pow(abs(1.0f - 2.0f * in.uv.x), n) +
                        pow(abs(1.0f - 2.0f * in.uv.y), n) > 1.0f) {{
                    discard;
                }}
            }}",
            roundness
        );

        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(*self)
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}

impl RoundModifier {
    /// Creates a new [`RoundModifier`] with the given roundness.
    ///
    /// The `roundness` parameter varies from 0.0 to 1.0.
    pub fn constant(module: &mut Module, roundness: f32) -> RoundModifier {
        RoundModifier {
            roundness: module.lit(roundness),
        }
    }

    /// Creates a new [`RoundModifier`] that describes an ellipse.
    #[doc(alias = "circle")]
    pub fn ellipse(module: &mut Module) -> RoundModifier {
        RoundModifier::constant(module, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn color_blend_mask() {
        assert_eq!(ColorBlendMask::RGB.to_components(), "rgb");
        assert_eq!(ColorBlendMask::RGBA.to_components(), "rgba");

        let m = ColorBlendMask::R | ColorBlendMask::B;
        assert_eq!(m.to_components(), "rb");
        assert_eq!(m.complement(), ColorBlendMask::G | ColorBlendMask::A);
        assert_eq!(m.complement().to_components(), "ga");
    }

    #[test]
    fn mod_particle_texture() {
        let mut module = Module::default();
        let slot = module.lit(42u32);
        // let texture = Handle::<Image>::default();
        let modifier = ParticleTextureModifier::new(slot);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let texture_layout = module.texture_layout();
        let mut context = RenderContext::new(&property_layout, &particle_layout, &texture_layout);
        modifier.apply_render(&mut module, &mut context).unwrap();

        // TODO - no validation at the minute
        assert_eq!(context.texture_layout.layout.len(), 0); // we "forgot" to add the slot to Module
        assert_eq!(context.textures.len(), 0); // we "forgot" the EffectMaterial
    }

    #[test]
    fn mod_flipbook() {
        let modifier = FlipbookModifier {
            sprite_grid_size: UVec2::new(3, 4),
        };

        let mut module = Module::default();
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let texture_layout = module.texture_layout();
        let mut context = RenderContext::new(&property_layout, &particle_layout, &texture_layout);
        modifier.apply_render(&mut module, &mut context).unwrap();

        assert!(context.sprite_grid_size.is_some());
        assert_eq!(context.sprite_grid_size.unwrap(), UVec2::new(3, 4));
    }

    #[test]
    fn mod_color_over_lifetime() {
        let red: Vec4 = Vec4::new(1., 0., 0., 1.);
        let blue: Vec4 = Vec4::new(0., 0., 1., 1.);
        let mut gradient = Gradient::new();
        gradient.add_key(0.5, red);
        gradient.add_key(0.8, blue);
        let modifier = ColorOverLifetimeModifier {
            gradient: gradient.clone(),
            blend: ColorBlendMode::Overwrite,
            mask: ColorBlendMask::RGBA,
        };

        let mut module = Module::default();
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let texture_layout = module.texture_layout();
        let mut context = RenderContext::new(&property_layout, &particle_layout, &texture_layout);
        modifier.apply_render(&mut module, &mut context).unwrap();

        assert!(context
            .render_extra
            .contains(&gradient.to_shader_code("key")));
    }

    #[test]
    fn mod_size_over_lifetime() {
        let x = Vec3::new(1., 0., 1.);
        let y = Vec3::new(0., 1., 1.);
        let mut gradient = Gradient::new();
        gradient.add_key(0.5, x);
        gradient.add_key(0.8, y);
        let modifier = SizeOverLifetimeModifier {
            gradient: gradient.clone(),
            screen_space_size: false,
        };

        let mut module = Module::default();
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let texture_layout = module.texture_layout();
        let mut context = RenderContext::new(&property_layout, &particle_layout, &texture_layout);
        modifier.apply_render(&mut module, &mut context).unwrap();

        assert!(context
            .render_extra
            .contains(&gradient.to_shader_code("key")));
    }

    #[test]
    fn mod_set_color() {
        let mut modifier = SetColorModifier::default();
        assert_eq!(modifier.context(), ModifierContext::Render);
        assert!(modifier.as_render().is_some());
        assert!(modifier.as_render_mut().is_some());
        assert_eq!(modifier.boxed_clone().context(), ModifierContext::Render);

        let mut module = Module::default();
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let texture_layout = module.texture_layout();
        let mut context = RenderContext::new(&property_layout, &particle_layout, &texture_layout);
        modifier.apply_render(&mut module, &mut context).unwrap();

        assert_eq!(modifier.color, CpuValue::from(Vec4::ZERO));
        assert_eq!(context.vertex_code, "color = vec4<f32>(0.,0.,0.,0.);\n");
    }

    #[test]
    fn mod_set_size() {
        let mut modifier = SetSizeModifier::default();
        assert_eq!(modifier.context(), ModifierContext::Render);
        assert!(modifier.as_render().is_some());
        assert!(modifier.as_render_mut().is_some());
        assert_eq!(modifier.boxed_clone().context(), ModifierContext::Render);

        let mut module = Module::default();
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let texture_layout = module.texture_layout();
        let mut context = RenderContext::new(&property_layout, &particle_layout, &texture_layout);
        modifier.apply_render(&mut module, &mut context).unwrap();

        assert_eq!(modifier.size, CpuValue::from(Vec3::ZERO));
        assert_eq!(context.vertex_code, "size = vec3<f32>(0.,0.,0.);\n");
    }

    #[test]
    fn mod_orient() {
        let mut modifier = OrientModifier::default();
        assert_eq!(modifier.context(), ModifierContext::Render);
        assert!(modifier.as_render().is_some());
        assert!(modifier.as_render_mut().is_some());
        assert_eq!(modifier.boxed_clone().context(), ModifierContext::Render);
    }

    #[test]
    fn mod_orient_default() {
        let mut module = Module::default();
        let modifier = OrientModifier::default();
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let texture_layout = module.texture_layout();
        let mut context = RenderContext::new(&property_layout, &particle_layout, &texture_layout);
        modifier.apply_render(&mut module, &mut context).unwrap();

        // TODO - less weak test...
        assert!(context
            .vertex_code
            .contains("get_camera_rotation_effect_space"));
        assert!(!context
            .vertex_code
            .contains("cos(particle_rot_in_cam_space)"));
        assert!(!context.vertex_code.contains("let axis_x0 ="));
    }

    #[test]
    fn mod_orient_rotation() {
        let mut module = Module::default();
        let modifier = OrientModifier::default().with_rotation(module.lit(1.));
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let texture_layout = module.texture_layout();
        let mut context = RenderContext::new(&property_layout, &particle_layout, &texture_layout);
        modifier.apply_render(&mut module, &mut context).unwrap();

        // TODO - less weak test...
        assert!(context
            .vertex_code
            .contains("get_camera_rotation_effect_space"));
        assert!(context
            .vertex_code
            .contains("cos(particle_rot_in_cam_space)"));
        assert!(!context.vertex_code.contains("let axis_x0 ="));
    }

    #[test]
    fn mod_orient_rotation_face_camera() {
        let mut module = Module::default();
        let modifier =
            OrientModifier::new(OrientMode::FaceCameraPosition).with_rotation(module.lit(1.));
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let texture_layout = module.texture_layout();
        let mut context = RenderContext::new(&property_layout, &particle_layout, &texture_layout);
        modifier.apply_render(&mut module, &mut context).unwrap();

        // TODO - less weak test...
        assert!(context
            .vertex_code
            .contains("get_camera_position_effect_space"));
        assert!(context
            .vertex_code
            .contains("cos(particle_rot_in_cam_space)"));
        assert!(context.vertex_code.contains("let axis_x0 ="));
    }
}
