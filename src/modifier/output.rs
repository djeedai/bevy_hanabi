//! Modifiers to influence the output (rendering) of each particle.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::hash::Hash;

use crate::{
    impl_mod_render, Attribute, CpuValue, Gradient, RenderContext, RenderModifier, ShaderCode,
    ToWgslString,
};

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
}

impl_mod_render!(ParticleTextureModifier, &[]); // TODO - should require some UV maybe?

#[typetag::serde]
impl RenderModifier for ParticleTextureModifier {
    fn apply_render(&self, context: &mut RenderContext) {
        context.set_particle_texture(self.texture.clone());
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
    fn apply_render(&self, context: &mut RenderContext) {
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
    fn apply_render(&self, context: &mut RenderContext) {
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
    /// The particle color.
    pub size: CpuValue<Vec2>,
    /// Is the particle size in screen-space logical pixel? If `true`, the size
    /// is in screen-space logical pixels, and not affected by the camera
    /// projection. If `false`, the particle size is in world units.
    pub screen_space_size: bool,
}

impl_mod_render!(SetSizeModifier, &[]);

#[typetag::serde]
impl RenderModifier for SetSizeModifier {
    fn apply_render(&self, context: &mut RenderContext) {
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
    fn apply_render(&self, context: &mut RenderContext) {
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

/// Reorients the vertices to always face the camera when rendering.
///
/// # Attributes
///
/// This modifier does not require any specific particle attribute.
#[derive(Debug, Default, Clone, Copy, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub struct BillboardModifier;

impl_mod_render!(BillboardModifier, &[]);

#[typetag::serde]
impl RenderModifier for BillboardModifier {
    fn apply_render(&self, context: &mut RenderContext) {
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
#[derive(Debug, Default, Clone, Copy, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub struct OrientAlongVelocityModifier;

impl_mod_render!(
    OrientAlongVelocityModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

#[typetag::serde]
impl RenderModifier for OrientAlongVelocityModifier {
    fn apply_render(&self, context: &mut RenderContext) {
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
        modifier.apply_render(&mut context);

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
        modifier.apply_render(&mut context);

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

        let mut context = RenderContext::default();
        modifier.apply_render(&mut context);

        assert!(context
            .render_extra
            .contains(&gradient.to_shader_code("key")));
    }

    #[test]
    fn mod_billboard() {
        let modifier = BillboardModifier;
        let mut context = RenderContext::default();
        modifier.apply_render(&mut context);
        assert!(context.vertex_code.contains("view.view"));
    }
}
