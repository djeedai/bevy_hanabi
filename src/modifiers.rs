use bevy::prelude::*;

use crate::{
    asset::{RenderLayout, UpdateLayout},
    gradient::Gradient,
};

pub trait Modifier {
    fn apply(&self, render_layout: &mut RenderLayout);
}

pub trait UpdateModifier {
    fn apply(&self, update_modifier: &mut UpdateLayout);
}

pub trait RenderModifier: Modifier {}

/// A modifier modulating each particle's color by sampling a texture.
#[derive(Default, Clone)]
pub struct ParticleTextureModifier {
    pub texture: Handle<Image>,
}

impl Modifier for ParticleTextureModifier {
    fn apply(&self, render_layout: &mut RenderLayout) {
        render_layout.particle_texture = Some(self.texture.clone());
    }
}
impl RenderModifier for ParticleTextureModifier {}

/// A modifier modulating each particle's color over its lifetime with a gradient curve.
#[derive(Default, Clone)]
pub struct ColorOverLifetimeModifier {
    pub gradient: Gradient,
}

impl Modifier for ColorOverLifetimeModifier {
    fn apply(&self, render_layout: &mut RenderLayout) {
        render_layout.lifetime_color_gradient = Some(self.gradient.clone());
    }
}
impl RenderModifier for ColorOverLifetimeModifier {}

#[derive(Default, Clone, Copy)]
pub struct AccelModifier {
    pub accel: Vec3,
}

impl UpdateModifier for AccelModifier {
    fn apply(&self, layout: &mut UpdateLayout) {
        layout.accel = self.accel;
    }
}
