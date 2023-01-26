//! Modifiers to influence the rendering of each particle.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{Attribute, BoxedModifier, Gradient, Modifier, ModifierContext, RenderLayout};

/// Trait to customize the rendering of alive particles each frame.
#[typetag::serde]
pub trait RenderModifier: Modifier {
    /// Apply the modifier to the render layout of the effect instance.
    fn apply(&self, render_layout: &mut RenderLayout);
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

#[typetag::serde]
impl Modifier for ParticleTextureModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Render
    }

    fn render(&self) -> Option<&dyn RenderModifier> {
        Some(self)
    }

    fn render_mut(&mut self) -> Option<&mut dyn RenderModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[&'static Attribute] {
        &[] // TODO - should require some UV maybe?
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(self.clone())
    }
}

#[typetag::serde]
impl RenderModifier for ParticleTextureModifier {
    fn apply(&self, render_layout: &mut RenderLayout) {
        render_layout.particle_texture = Some(self.texture.clone());
    }
}

/// A modifier modulating each particle's color over its lifetime with a
/// gradient curve.
#[derive(Debug, Default, Clone, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct ColorOverLifetimeModifier {
    /// The color gradient defining the particle color based on its lifetime.
    pub gradient: Gradient<Vec4>,
}

#[typetag::serde]
impl Modifier for ColorOverLifetimeModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Render
    }

    fn render(&self) -> Option<&dyn RenderModifier> {
        Some(self)
    }

    fn render_mut(&mut self) -> Option<&mut dyn RenderModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[&'static Attribute] {
        &[]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(self.clone())
    }
}

#[typetag::serde]
impl RenderModifier for ColorOverLifetimeModifier {
    fn apply(&self, render_layout: &mut RenderLayout) {
        render_layout.lifetime_color_gradient = Some(self.gradient.clone());
    }
}

/// A modifier modulating each particle's size over its lifetime with a gradient
/// curve.
#[derive(Debug, Default, Clone, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct SizeOverLifetimeModifier {
    /// The size gradient defining the particle size based on its lifetime.
    pub gradient: Gradient<Vec2>,
}

#[typetag::serde]
impl Modifier for SizeOverLifetimeModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Render
    }

    fn render(&self) -> Option<&dyn RenderModifier> {
        Some(self)
    }

    fn render_mut(&mut self) -> Option<&mut dyn RenderModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[&'static Attribute] {
        &[]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(self.clone())
    }
}

#[typetag::serde]
impl RenderModifier for SizeOverLifetimeModifier {
    fn apply(&self, render_layout: &mut RenderLayout) {
        render_layout.lifetime_size_gradient = Some(self.gradient.clone());
    }
}

/// Reorients the vertices to always face the camera when rendering.
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct BillboardModifier;

#[typetag::serde]
impl Modifier for BillboardModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Render
    }

    fn render(&self) -> Option<&dyn RenderModifier> {
        Some(self)
    }

    fn render_mut(&mut self) -> Option<&mut dyn RenderModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::POSITION]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }
}

#[typetag::serde]
impl RenderModifier for BillboardModifier {
    fn apply(&self, render_layout: &mut RenderLayout) {
        render_layout.billboard = true;
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

        let mut layout = RenderLayout::default();
        modifier.apply(&mut layout);

        assert!(layout.particle_texture.is_some());
        assert_eq!(layout.particle_texture.unwrap(), texture);
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

        let mut layout = RenderLayout::default();
        modifier.apply(&mut layout);

        assert!(layout.lifetime_color_gradient.is_some());
        assert_eq!(layout.lifetime_color_gradient.unwrap(), gradient);
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

        let mut layout = RenderLayout::default();
        modifier.apply(&mut layout);

        assert!(layout.lifetime_size_gradient.is_some());
        assert_eq!(layout.lifetime_size_gradient.unwrap(), gradient);
    }

    #[test]
    fn mod_billboard() {
        let modifier = BillboardModifier;
        let mut layout = RenderLayout::default();
        modifier.apply(&mut layout);
        assert!(layout.billboard);
    }
}
