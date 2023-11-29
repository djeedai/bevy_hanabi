//! Modifiers to influence the output (rendering) of each particle.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::hash::Hash;

use crate::{
    impl_mod_render, Attribute, BoxedModifier, CpuValue, EvalContext, ExprHandle, Gradient,
    Modifier, ModifierContext, Module, RenderContext, RenderModifier, ShaderCode, ToWgslString,
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

impl ToWgslString for ImageSampleMapping {
    fn to_wgsl_string(&self) -> String {
        match *self {
            ImageSampleMapping::Modulate => "color = color * texColor;",
            ImageSampleMapping::ModulateRGB => {
                "color = vec4<f32>(color.rgb * texColor.rgb, color.a);"
            }
            ImageSampleMapping::ModulateOpacityFromR => "color.a = color.a * texColor.r;",
        }
        .to_string()
    }
}

/// A modifier modulating each particle's color by sampling a texture.
///
/// # Attributes
///
/// This modifier does not require any specific particle attribute.
#[derive(Default, Debug, Clone, PartialEq, Reflect, Serialize, Deserialize)]
pub struct ParticleTextureModifier {
    /// The texture image to modulate the particle color with.
    #[serde(skip)]
    // TODO - Clarify if Modifier needs to be serializable, or we need another on-disk
    // representation... NOTE - Need to keep a strong handle here, nothing else will keep that
    // texture loaded currently.
    pub texture: Handle<Image>,

    /// The mapping of the texture image samples to the base particle color.
    pub sample_mapping: ImageSampleMapping,
}

impl_mod_render!(ParticleTextureModifier, &[]); // TODO - should require some UV maybe?

#[typetag::serde]
impl RenderModifier for ParticleTextureModifier {
    fn apply_render(&self, _module: &mut Module, context: &mut RenderContext) {
        context.set_particle_texture(self.texture.clone());
        context.image_sample_mapping_code = self.sample_mapping.to_wgsl_string();
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
}

impl_mod_render!(SetColorModifier, &[]);

#[typetag::serde]
impl RenderModifier for SetColorModifier {
    fn apply_render(&self, _module: &mut Module, context: &mut RenderContext) {
        context.vertex_code += &format!("color = {0};\n", self.color.to_wgsl_string());
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
}

impl_mod_render!(
    ColorOverLifetimeModifier,
    &[Attribute::AGE, Attribute::LIFETIME]
);

#[typetag::serde]
impl RenderModifier for ColorOverLifetimeModifier {
    fn apply_render(&self, _module: &mut Module, context: &mut RenderContext) {
        let func_name = context.add_color_gradient(self.gradient.clone());
        context.render_extra += &format!(
            r#"fn {0}(key: f32) -> vec4<f32> {{
    {1}
}}

"#,
            func_name,
            self.gradient.to_shader_code("key")
        );

        context.vertex_code += &format!(
            "color = {0}(particle.{1} / particle.{2});\n",
            func_name,
            Attribute::AGE.name(),
            Attribute::LIFETIME.name()
        );
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
/// This modifier does not require any specific particle attribute.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct SetSizeModifier {
    /// The 2D particle (quad) size.
    pub size: CpuValue<Vec2>,
    /// Is the particle size in screen-space logical pixel? If `true`, the size
    /// is in screen-space logical pixels, and not affected by the camera
    /// projection. If `false`, the particle size is in world units.
    pub screen_space_size: bool,
}

impl_mod_render!(SetSizeModifier, &[]);

#[typetag::serde]
impl RenderModifier for SetSizeModifier {
    fn apply_render(&self, _module: &mut Module, context: &mut RenderContext) {
        context.vertex_code += &format!("size = {0};\n", self.size.to_wgsl_string());
        context.screen_space_size = self.screen_space_size;
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
    pub gradient: Gradient<Vec2>,
    /// Is the particle size in screen-space logical pixel? If `true`, the size
    /// is in screen-space logical pixels, and not affected by the camera
    /// projection. If `false`, the particle size is in world units.
    pub screen_space_size: bool,
}

impl_mod_render!(
    SizeOverLifetimeModifier,
    &[Attribute::AGE, Attribute::LIFETIME]
);

#[typetag::serde]
impl RenderModifier for SizeOverLifetimeModifier {
    fn apply_render(&self, _module: &mut Module, context: &mut RenderContext) {
        let func_name = context.add_size_gradient(self.gradient.clone());
        context.render_extra += &format!(
            r#"fn {0}(key: f32) -> vec2<f32> {{
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

        context.screen_space_size = self.screen_space_size;
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

#[typetag::serde]
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
}

#[typetag::serde]
impl RenderModifier for OrientModifier {
    fn apply_render(&self, module: &mut Module, context: &mut RenderContext) {
        match self.mode {
            OrientMode::ParallelCameraDepthPlane => {
                if let Some(rotation) = self.rotation {
                    let rotation = context.eval(module, rotation).unwrap();
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
                    let rotation = context.eval(module, rotation).unwrap();
                    context.vertex_code += &format!(
                        r#"axis_z = normalize(get_camera_position_effect_space() - position);
let particle_rot_in_cam_space = {};
let particle_rot_in_cam_space_cos = cos(particle_rot_in_cam_space);
let particle_rot_in_cam_space_sin = sin(particle_rot_in_cam_space);
let axis_x0 = normalize(cross(view.view[1].xyz, axis_z));
let axis_y0 = cross(axis_z, axis_x0);
axis_x = axis_x0 * particle_rot_in_cam_space_cos + axis_y0 * particle_rot_in_cam_space_sin;
axis_y = axis_x0 * particle_rot_in_cam_space_sin - axis_y0 * particle_rot_in_cam_space_cos;
"#,
                        rotation
                    );
                } else {
                    context.vertex_code += r#"axis_z = normalize(get_camera_position_effect_space() - position);
axis_x = normalize(cross(view.view[1].xyz, axis_z));
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
/// let sprite_index = writer.attr(Attribute::AGE).cast(ScalarType::Int).rem(writer.lit(4i32)).expr();
/// let update_sprite_index = SetAttributeModifier::new(Attribute::SPRITE_INDEX, sprite_index);
///
/// let asset = EffectAsset::new(32768, Spawner::once(32.0.into(), true), writer.finish())
///     .with_name("flipbook")
///     .init(init_age)
///     .init(init_lifetime)
///     .update(update_sprite_index)
///     .render(ParticleTextureModifier {
///         texture,
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

#[typetag::serde]
impl RenderModifier for FlipbookModifier {
    fn apply_render(&self, _module: &mut Module, context: &mut RenderContext) {
        context.sprite_grid_size = Some(self.sprite_grid_size);
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    use super::*;

    #[test]
    fn mod_particle_texture() {
        let texture = Handle::<Image>::default();
        let modifier = ParticleTextureModifier {
            texture: texture.clone(),
            ..default()
        };

        let mut module = Module::default();
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut context = RenderContext::new(&property_layout, &particle_layout);
        modifier.apply_render(&mut module, &mut context);

        assert!(context.particle_texture.is_some());
        assert_eq!(context.particle_texture.unwrap(), texture);
    }

    #[test]
    fn mod_flipbook() {
        let modifier = FlipbookModifier {
            sprite_grid_size: UVec2::new(3, 4),
        };

        let mut module = Module::default();
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut context = RenderContext::new(&property_layout, &particle_layout);
        modifier.apply_render(&mut module, &mut context);

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
        };

        let mut module = Module::default();
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut context = RenderContext::new(&property_layout, &particle_layout);
        modifier.apply_render(&mut module, &mut context);

        assert!(context
            .render_extra
            .contains(&gradient.to_shader_code("key")));
    }

    #[test]
    fn mod_size_over_lifetime() {
        let x = Vec2::new(1., 0.);
        let y = Vec2::new(0., 1.);
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
        let mut context = RenderContext::new(&property_layout, &particle_layout);
        modifier.apply_render(&mut module, &mut context);

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
        let mut context = RenderContext::new(&property_layout, &particle_layout);
        modifier.apply_render(&mut module, &mut context);

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
        let mut context = RenderContext::new(&property_layout, &particle_layout);
        modifier.apply_render(&mut module, &mut context);

        assert_eq!(modifier.size, CpuValue::from(Vec2::ZERO));
        assert_eq!(context.vertex_code, "size = vec2<f32>(0.,0.);\n");
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
        let mut context = RenderContext::new(&property_layout, &particle_layout);
        modifier.apply_render(&mut module, &mut context);

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
        let mut context = RenderContext::new(&property_layout, &particle_layout);
        modifier.apply_render(&mut module, &mut context);

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
        let mut context = RenderContext::new(&property_layout, &particle_layout);
        modifier.apply_render(&mut module, &mut context);

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
