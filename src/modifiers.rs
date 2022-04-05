use bevy::prelude::*;

use crate::{
    asset::{InitLayout, RenderLayout, UpdateLayout},
    gradient::Gradient,
    render::FFNUM,
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

/// The [`PullingForceFieldParam`] allows for either a constant force accross all of space, or a
/// pulling/repulsing force originating from a point source.
#[derive(Clone, Copy)]
pub enum PullingForceType {
    /// Constant field in all space. The position_or_direction field is interpreted as a direction.
    /// For this particular case, the radii field of [`PullingForceFieldParam`] are ignored.
    Constant,

    /// Linear pulling force. The position_or_direction field is interpreted as the source position.
    Linear,

    /// Quadratic pulling force. The position_or_direction field is interpreted as the source position.
    Quadratic,

    /// Cubic pulling force. The position_or_direction field is interpreted as the source position.
    Cubic,
}

impl Default for PullingForceType {
    fn default() -> Self {
        PullingForceType::Constant
    }
}

impl PullingForceType {
    /// Converts the instance of PullingForceType into an integer, preparing to be sent to the GPU.
    pub fn to_int(self) -> i32 {
        match self {
            PullingForceType::Constant => 0,
            PullingForceType::Linear => 1,
            PullingForceType::Quadratic => 2,
            PullingForceType::Cubic => 3,
        }
    }
}

/// Parameters for the components making the force field.
#[derive(Clone, Copy)]
pub struct PullingForceFieldParam {
    /// The particle_update.wgsl shader interprets this field as a position when the force type is set
    /// to either [`PullingForceType::Linear`], [`PullingForceType::Quadratic`] or [`PullingForceType::Cubic`].
    /// The interpretation is direction in the case of a [`PullingForceType::Constant`].
    /// [`PullingForceType::Constant`]
    pub position_or_direction: Vec3,
    /// For a pulling/repulsing force, this is the maximum radius of the sphere of influence, outside of which
    /// the force field is null.
    pub max_radius: f32,
    /// For a pulling/repulsing force, this is the minimum radius of the sphere of influence, inside of which
    /// the force field is null, avoiding the singularity at the source position.
    pub min_radius: f32,
    /// Mass is proportional to the intensity of the force. Note that the particles_update.wgsl shader will
    /// ignore all subsequent force field components after it encounters a component with a mass of zero.
    pub mass: f32,
    /// Defines whether the field is constant or a pulling/repulsing force with an exponent of either one,
    /// two or three.
    pub force_type: PullingForceType,
}

impl Default for PullingForceFieldParam {
    fn default() -> Self {
        // defaults to no force field (a mass of 0)
        PullingForceFieldParam {
            position_or_direction: Vec3::new(0., 0., 0.),
            min_radius: 0.1,
            max_radius: 0.0,
            mass: 0.,
            force_type: PullingForceType::Constant,
        }
    }
}

/// A modifier to apply a force field to all particles each frame. The force field is made up of
/// point sources and/or constant fields. The maximum number of components is set with [`FFNUM`].
#[derive(Default, Clone, Copy)]
pub struct PullingForceFieldModifier {
    /// Force Field.
    pub force_field: [PullingForceFieldParam; FFNUM],
}

impl PullingForceFieldModifier {
    /// Instantiate a PullingForceFieldModifier.
    ///
    /// # Panics
    ///
    /// Panics if the number of attractors exceeds [`FFNUM`].
    pub fn new(point_attractors: Vec<PullingForceFieldParam>) -> Self {
        let mut force_field = [PullingForceFieldParam::default(); FFNUM];

        for (i, p_attractor) in point_attractors.iter().enumerate() {
            if i > FFNUM {
                panic!("Too many point attractors");
            }
            force_field[i] = *p_attractor;
        }

        Self { force_field }
    }

    /// Perhaps will be deleted in the future.
    // Delete me?
    pub fn add_or_replace(&mut self, point_attractor: PullingForceFieldParam, index: usize) {
        self.force_field[index] = point_attractor;
    }
}

impl UpdateModifier for PullingForceFieldModifier {
    fn apply(&self, layout: &mut UpdateLayout) {
        layout.force_field = self.force_field;
    }
}
