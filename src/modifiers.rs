use bevy::prelude::*;

use crate::{
    asset::{InitLayout, RenderLayout, UpdateLayout},
    gradient::Gradient,
    ToWgslFloat,
};

pub trait InitModifier {
    fn apply(&self, init_layout: &mut InitLayout);
}

pub trait UpdateModifier {
    fn apply(&self, update_layout: &mut UpdateLayout);
}

pub trait RenderModifier {
    fn apply(&self, render_layout: &mut RenderLayout);
}

///
#[derive(Default, Clone, Copy)]
pub struct PositionSphereModifier {
    pub center: Vec3,
    pub radius: f32,
}

impl InitModifier for PositionSphereModifier {
    fn apply(&self, init_layout: &mut InitLayout) {
        init_layout.position_code = format!(
            r##"
    // >>> [PositionSphereModifier]
    // Sphere center
    var c = vec3<f32>({0}, {1}, {2});
    // Sphere radius
    var r = {3};
    // Radial speed
    var speed = 2.0;
    // Spawn randomly along the sphere surface
    var dir = rand3() * 2. - 1.;
    dir = normalize(dir);
    ret.pos = c + dir * r;
    // Radial speed away from sphere center
    ret.vel = dir * speed;
    // <<< [PositionSphereModifier]
"##,
            self.center.x.to_float_string(),
            self.center.y.to_float_string(),
            self.center.z.to_float_string(),
            self.radius.to_float_string()
        )
        .to_string();
    }
}

/// A modifier modulating each particle's color by sampling a texture.
#[derive(Default, Clone)]
pub struct ParticleTextureModifier {
    pub texture: Handle<Image>,
}

impl RenderModifier for ParticleTextureModifier {
    fn apply(&self, render_layout: &mut RenderLayout) {
        render_layout.particle_texture = Some(self.texture.clone());
    }
}

/// A modifier modulating each particle's color over its lifetime with a gradient curve.
#[derive(Default, Clone)]
pub struct ColorOverLifetimeModifier {
    pub gradient: Gradient,
}

impl RenderModifier for ColorOverLifetimeModifier {
    fn apply(&self, render_layout: &mut RenderLayout) {
        render_layout.lifetime_color_gradient = Some(self.gradient.clone());
    }
}

#[derive(Default, Clone, Copy)]
pub struct AccelModifier {
    pub accel: Vec3,
}

impl UpdateModifier for AccelModifier {
    fn apply(&self, layout: &mut UpdateLayout) {
        layout.accel = self.accel;
    }
}
