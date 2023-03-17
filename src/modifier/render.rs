//! Modifiers to influence the rendering of each particle.

use bevy::{
    prelude::*,
    utils::{FloatOrd, HashMap},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use crate::{
    Attribute, BoxedModifier, Gradient, Modifier, ModifierContext, ShaderCode, ToWgslString, Value,
};

/// Calculate a function ID by hashing the given value representative of the
/// function.
fn calc_func_id<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Particle rendering shader code generation context.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RenderContext {
    /// Main particle rendering code for the vertex shader.
    pub vertex_code: String,
    /// Main particle rendering code for the fragment shader.
    pub fragment_code: String,
    /// Extra functions emitted at top level, which `vertex_code` and
    /// `fragment_code` can call.
    pub render_extra: String,
    /// Texture modulating the particle color.
    pub particle_texture: Option<Handle<Image>>,
    /// Color gradients.
    pub gradients: HashMap<u64, Gradient<Vec4>>,
    /// Size gradients.
    pub size_gradients: HashMap<u64, Gradient<Vec2>>,
}

impl RenderContext {
    /// Set the main texture used to color particles.
    fn set_particle_texture(&mut self, handle: Handle<Image>) {
        self.particle_texture = Some(handle);
    }

    /// Add a color gradient.
    ///
    /// # Returns
    ///
    /// Returns the unique name of the gradient, to be used as function name in
    /// the shader code.
    fn add_color_gradient(&mut self, gradient: Gradient<Vec4>) -> String {
        let func_id = calc_func_id(&gradient);
        self.gradients.insert(func_id, gradient);
        let func_name = format!("color_gradient_{0:016X}", func_id);
        func_name
    }

    /// Add a size gradient.
    ///
    /// # Returns
    ///
    /// Returns the unique name of the gradient, to be used as function name in
    /// the shader code.
    fn add_size_gradient(&mut self, gradient: Gradient<Vec2>) -> String {
        let func_id = calc_func_id(&gradient);
        self.size_gradients.insert(func_id, gradient);
        let func_name = format!("size_gradient_{0:016X}", func_id);
        func_name
    }
}

/// Trait to customize the rendering of alive particles each frame.
#[typetag::serde]
pub trait RenderModifier: Modifier {
    /// Apply the rendering code.
    fn apply(&self, context: &mut RenderContext);
}

/// Macro to implement the [`Modifier`] trait for a render modifier.
macro_rules! impl_mod_render {
    ($t:ty, $attrs:expr) => {
        #[typetag::serde]
        impl Modifier for $t {
            fn context(&self) -> ModifierContext {
                ModifierContext::Render
            }

            fn as_render(&self) -> Option<&dyn RenderModifier> {
                Some(self)
            }

            fn as_render_mut(&mut self) -> Option<&mut dyn RenderModifier> {
                Some(self)
            }

            fn attributes(&self) -> &[&'static Attribute] {
                $attrs
            }

            fn boxed_clone(&self) -> BoxedModifier {
                Box::new(self.clone())
            }
        }
    };
}

/// A modifier modulating each particle's color by sampling a texture.
///
/// # Attributes
///
/// This modifier does not require any specific particle attribute.
#[derive(Default, Debug, Clone, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct ParticleTextureModifier {
    /// The texture image to modulate the particle color with.
    #[serde(skip)]
    // TODO - Clarify if Modifier needs to be serializable, or we need another on-disk
    // representation... NOTE - Need to keep a strong handle here, nothing else will keep that
    // texture loaded currently.
    pub texture: Handle<Image>,
}

impl_mod_render!(ParticleTextureModifier, &[]); // TODO - should require some UV maybe?

#[typetag::serde]
impl RenderModifier for ParticleTextureModifier {
    fn apply(&self, context: &mut RenderContext) {
        context.set_particle_texture(self.texture.clone());
    }
}

/// A modifier to set each particle's rendering color.
///
/// # Attributes
///
/// This modifier does not require any specific particle attribute.
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct SetColorModifier {
    /// The particle color.
    pub color: Value<Vec4>,
}

// TODO - impl Hash for Value<T>
// SAFETY: This is consistent with the derive, but we can't derive due to
// FloatOrd.
#[allow(clippy::derive_hash_xor_eq)]
impl Hash for SetColorModifier {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.color {
            Value::Single(v) => {
                FloatOrd(v.x).hash(state);
                FloatOrd(v.y).hash(state);
                FloatOrd(v.z).hash(state);
                FloatOrd(v.w).hash(state);
            }
            Value::Uniform((a, b)) => {
                FloatOrd(a.x).hash(state);
                FloatOrd(a.y).hash(state);
                FloatOrd(a.z).hash(state);
                FloatOrd(a.w).hash(state);
                FloatOrd(b.x).hash(state);
                FloatOrd(b.y).hash(state);
                FloatOrd(b.z).hash(state);
                FloatOrd(b.w).hash(state);
            }
        }
    }
}

impl_mod_render!(SetColorModifier, &[]);

#[typetag::serde]
impl RenderModifier for SetColorModifier {
    fn apply(&self, context: &mut RenderContext) {
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
#[derive(Debug, Default, Clone, PartialEq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
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
    fn apply(&self, context: &mut RenderContext) {
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

/// A modifier to set each particle's size.
///
/// # Attributes
///
/// This modifier does not require any specific particle attribute.
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct SetSizeModifier {
    /// The particle color.
    pub size: Value<Vec2>,
}

// TODO - impl Hash for Value<T>
// SAFETY: This is consistent with the derive, but we can't derive due to
// FloatOrd.
#[allow(clippy::derive_hash_xor_eq)]
impl Hash for SetSizeModifier {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.size {
            Value::Single(v) => {
                FloatOrd(v.x).hash(state);
                FloatOrd(v.y).hash(state);
            }
            Value::Uniform((a, b)) => {
                FloatOrd(a.x).hash(state);
                FloatOrd(a.y).hash(state);
                FloatOrd(b.x).hash(state);
                FloatOrd(b.y).hash(state);
            }
        }
    }
}

impl_mod_render!(SetSizeModifier, &[]);

#[typetag::serde]
impl RenderModifier for SetSizeModifier {
    fn apply(&self, context: &mut RenderContext) {
        context.vertex_code += &format!("size = {0};\n", self.size.to_wgsl_string());
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
#[derive(Debug, Default, Clone, PartialEq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub struct SizeOverLifetimeModifier {
    /// The size gradient defining the particle size based on its lifetime.
    pub gradient: Gradient<Vec2>,
}

impl_mod_render!(
    SizeOverLifetimeModifier,
    &[Attribute::AGE, Attribute::LIFETIME]
);

#[typetag::serde]
impl RenderModifier for SizeOverLifetimeModifier {
    fn apply(&self, context: &mut RenderContext) {
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
    }
}

/// Reorients the vertices to always face the camera when rendering.
///
/// # Attributes
///
/// This modifier does not require any specific particle attribute.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Hash, Reflect, FromReflect, Serialize, Deserialize,
)]
pub struct BillboardModifier;

impl_mod_render!(BillboardModifier, &[]);

#[typetag::serde]
impl RenderModifier for BillboardModifier {
    fn apply(&self, context: &mut RenderContext) {
        context.vertex_code += "axis_x = view.view[0].xyz;\naxis_y = view.view[1].xyz;\n";
    }
}

/// A modifier orienting each particle alongside its velocity.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Hash, Reflect, FromReflect, Serialize, Deserialize,
)]
pub struct OrientAlongVelocityModifier;

impl_mod_render!(
    OrientAlongVelocityModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

#[typetag::serde]
impl RenderModifier for OrientAlongVelocityModifier {
    fn apply(&self, context: &mut RenderContext) {
        context.vertex_code += r#"let dir = normalize(particle.position - view.view[3].xyz);
    axis_x = normalize(particle.velocity);
    axis_y = cross(dir, axis_x);
    axis_z = cross(axis_x, axis_y);
"#;
    }
}

#[cfg(test)]
mod tests {
    use crate::ParticleLayout;

    use super::*;

    use naga::front::wgsl::Parser;

    #[test]
    fn mod_particle_texture() {
        let texture = Handle::<Image>::default();
        let modifier = ParticleTextureModifier {
            texture: texture.clone(),
        };

        let mut context = RenderContext::default();
        modifier.apply(&mut context);

        assert!(context.particle_texture.is_some());
        assert_eq!(context.particle_texture.unwrap(), texture);
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

        let mut context = RenderContext::default();
        modifier.apply(&mut context);

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
        };

        let mut context = RenderContext::default();
        modifier.apply(&mut context);

        assert!(context
            .render_extra
            .contains(&gradient.to_shader_code("key")));
    }

    #[test]
    fn mod_billboard() {
        let modifier = BillboardModifier;
        let mut context = RenderContext::default();
        modifier.apply(&mut context);
        assert!(context.vertex_code.contains("view.view"));
    }

    #[test]
    fn validate() {
        let modifiers: &[&dyn RenderModifier] = &[
            &ParticleTextureModifier::default(),
            &ColorOverLifetimeModifier::default(),
            &SizeOverLifetimeModifier::default(),
            &BillboardModifier::default(),
            &OrientAlongVelocityModifier::default(),
        ];
        for &modifier in modifiers.iter() {
            let mut context = RenderContext::default();
            modifier.apply(&mut context);
            let vertex_code = context.vertex_code;
            let fragment_code = context.fragment_code;
            let render_extra = context.render_extra;

            let mut particle_layout = ParticleLayout::new();
            for &attr in modifier.attributes() {
                particle_layout = particle_layout.append(attr);
            }
            let particle_layout = particle_layout.build();
            let attributes_code = particle_layout.generate_code();

            let code = format!(
                r##"
struct View {{
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    world_position: vec3<f32>,
    width: f32,
    height: f32,
}};

fn rand() -> f32 {{
    return 0.0;
}}

const tau: f32 = 6.283185307179586476925286766559;

struct Particle {{
    {attributes_code}
}};

struct VertexOutput {{
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}};

@group(0) @binding(0) var<uniform> view: View;

{render_extra}

@compute @workgroup_size(64)
fn main() {{
    var particle = Particle();
    var size = vec2<f32>(1.0, 1.0);
    var axis_x = vec3<f32>(1.0, 0.0, 0.0);
    var axis_y = vec3<f32>(0.0, 1.0, 0.0);
    var axis_z = vec3<f32>(0.0, 0.0, 1.0);
    var color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
{vertex_code}
    var out: VertexOutput;
    return out;
}}


@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {{
{fragment_code}
    return vec4<f32>(1.0);
}}"##
            );

            let mut parser = Parser::new();
            let res = parser.parse(&code);
            if let Err(err) = &res {
                println!("Modifier: {:?}", modifier.type_name());
                println!("Code: {:?}", code);
                println!("Err: {:?}", err);
            }
            assert!(res.is_ok());
        }
    }
}
