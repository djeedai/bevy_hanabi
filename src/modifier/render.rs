//! Modifiers to influence the rendering of each particle.

use bevy::{prelude::*, utils::HashMap};
use serde::{Deserialize, Serialize};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use crate::{Attribute, BoxedModifier, Gradient, Modifier, ModifierContext, ShaderCode};

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

/// A modifier modulating each particle's color over its lifetime with a
/// gradient curve.
#[derive(Debug, Default, Clone, PartialEq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub struct ColorOverLifetimeModifier {
    /// The color gradient defining the particle color based on its lifetime.
    pub gradient: Gradient<Vec4>,
}

impl_mod_render!(ColorOverLifetimeModifier, &[]);

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
            "color = {0}(particle.age / particle.lifetime);\n",
            func_name
        );
    }
}

/// A modifier modulating each particle's size over its lifetime with a gradient
/// curve.
#[derive(Debug, Default, Clone, PartialEq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub struct SizeOverLifetimeModifier {
    /// The size gradient defining the particle size based on its lifetime.
    pub gradient: Gradient<Vec2>,
}

impl_mod_render!(SizeOverLifetimeModifier, &[]);

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

        context.vertex_code +=
            &format!("size = {0}(particle.age / particle.lifetime);\n", func_name);
    }
}

/// Reorients the vertices to always face the camera when rendering.
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
    use super::*;

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
}
