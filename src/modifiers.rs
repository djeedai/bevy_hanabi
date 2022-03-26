use bevy::prelude::*;

use crate::{
    asset::{InitLayout, RenderLayout, UpdateLayout},
    gradient::Gradient,
    ToWgslString,
};

/// Trait to customize the initializing of newly spawned particles.
pub trait InitModifier {
    /// Apply the modifier to the init layout of the effect instance.
    fn apply(&self, init_layout: &mut InitLayout);
}

/// Trait to customize the updating of existing particles each frame.
pub trait UpdateModifier {
    /// Apply the modifier to the update layout of the effect instance.
    fn apply(&self, update_layout: &mut UpdateLayout);
}

/// Trait to customize the rendering of alive particles each frame.
pub trait RenderModifier {
    /// Apply the modifier to the render layout of the effect instance.
    fn apply(&self, render_layout: &mut RenderLayout);
}

/// The dimension of a shape to consider.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ShapeDimension {
    /// Consider the surface of the shape only.
    Surface,
    /// Consider the entire shape volume.
    Volume,
}

impl Default for ShapeDimension {
    fn default() -> Self {
        ShapeDimension::Surface
    }
}

/// An initialization modifier spawning particles on a circle/disc.
#[derive(Clone, Copy)]
pub struct PositionCircleModifier {
    /// The circle center, relative to the emitter position.
    pub center: Vec3,
    /// The circle axis, which is the normalized normal of the circle's plane.
    /// Set this to `Vec3::Z` for a 2D game.
    pub axis: Vec3,
    /// The circle radius.
    pub radius: f32,
    /// The radial speed of the particles on spawn.
    pub speed: f32,
    /// The shape dimension to spawn from.
    pub dimension: ShapeDimension,
}

impl Default for PositionCircleModifier {
    fn default() -> Self {
        Self {
            center: Default::default(),
            axis: Vec3::Z,
            radius: Default::default(),
            speed: Default::default(),
            dimension: Default::default(),
        }
    }
}

impl InitModifier for PositionCircleModifier {
    fn apply(&self, init_layout: &mut InitLayout) {
        let (tangent, bitangent) = self.axis.any_orthonormal_pair();

        let radius_code = match self.dimension {
            ShapeDimension::Surface => {
                // Constant radius
                format!("let r = {};", self.radius.to_wgsl_string())
            }
            ShapeDimension::Volume => {
                // Radius uniformly distributed in [0:1], then square-rooted
                // to account for the increased perimeter covered by increased radii.
                format!("let r = sqrt(rand()) * {};", self.radius.to_wgsl_string())
            }
        };

        init_layout.position_code = format!(
            r##"
    // >>> [PositionCircleModifier]
    // Circle center
    let c = {};
    // Circle basis
    let tangent = {};
    let bitangent = {};
    // Circle radius
    {}
    // Radial speed
    let speed = {};
    // Spawn random point on/in circle
    let theta = rand() * tau;
    let dir = tangent * cos(theta) + bitangent * sin(theta);
    ret.pos = c + r * dir;
    // Velocity away from center
    ret.vel = dir * speed;
    // <<< [PositionCircleModifier]
            "##,
            self.center.to_wgsl_string(),
            tangent.to_wgsl_string(),
            bitangent.to_wgsl_string(),
            radius_code,
            self.speed.to_wgsl_string()
        )
        .to_string();
    }
}

/// An initialization modifier spawning particles on a sphere.
#[derive(Default, Clone, Copy)]
pub struct PositionSphereModifier {
    /// The sphere center, relative to the emitter position.
    pub center: Vec3,
    /// The sphere radius.
    pub radius: f32,
    /// The radial speed of the particles on spawn.
    pub speed: f32,
    /// The shape dimension to spawn from.
    pub dimension: ShapeDimension,
}

impl InitModifier for PositionSphereModifier {
    fn apply(&self, init_layout: &mut InitLayout) {
        let radius_code = match self.dimension {
            ShapeDimension::Surface => {
                // Constant radius
                format!("let r = {};", self.radius.to_wgsl_string())
            }
            ShapeDimension::Volume => {
                // Radius uniformly distributed in [0:1], then scaled by ^(1/3) in 3D
                // to account for the increased surface covered by increased radii.
                // https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere
                format!(
                    "var r = pow(rand(), 1./3.) * {};",
                    self.radius.to_wgsl_string()
                )
            }
        };
        init_layout.position_code = format!(
            r##"
    // >>> [PositionSphereModifier]
    // Sphere center
    let c = {0};
    // Sphere radius
    {1}
    // Radial speed
    let speed = {2};
    // Spawn randomly along the sphere surface using Archimedes's theorem
    var theta = rand() * tau;
    var z = rand() * 2. - 1.;
    var phi = acos(z);
    var sinphi = sin(phi);
    var x = sinphi * cos(theta);
    var y = sinphi * sin(theta);
    var dir = vec3<f32>(x, y, z);
    ret.pos = c + r * dir;
    // Radial velocity away from sphere center
    ret.vel = dir * speed;
    // <<< [PositionSphereModifier]
"##,
            self.center.to_wgsl_string(),
            radius_code,
            self.speed.to_wgsl_string()
        )
        .to_string();
    }
}

/// A modifier modulating each particle's color by sampling a texture.
#[derive(Default, Clone)]
pub struct ParticleTextureModifier {
    /// The texture image to modulate the particle color with.
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
    /// The color gradient defining the particle color based on its lifetime.
    pub gradient: Gradient<Vec4>,
}

impl RenderModifier for ColorOverLifetimeModifier {
    fn apply(&self, render_layout: &mut RenderLayout) {
        render_layout.lifetime_color_gradient = Some(self.gradient.clone());
    }
}

/// A modifier modulating each particle's size over its lifetime with a gradient curve.
#[derive(Default, Clone)]
pub struct SizeOverLifetimeModifier {
    /// The size gradient defining the particle size based on its lifetime.
    pub gradient: Gradient<Vec2>,
}

impl RenderModifier for SizeOverLifetimeModifier {
    fn apply(&self, render_layout: &mut RenderLayout) {
        render_layout.size_color_gradient = Some(self.gradient.clone());
    }
}

/// A modifier to apply a constant acceleration to all particles each frame.
///
/// This is typically used to apply some kind of gravity.
#[derive(Default, Clone, Copy)]
pub struct AccelModifier {
    /// The constant acceleration to apply to all particles in the effect each frame.
    pub accel: Vec3,
}

impl UpdateModifier for AccelModifier {
    fn apply(&self, layout: &mut UpdateLayout) {
        layout.accel = self.accel;
    }
}
