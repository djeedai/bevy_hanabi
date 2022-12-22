//! Modifiers to initialize particles when they spawn.

use bevy::prelude::*;

use crate::{
    asset::InitLayout, modifier::ShapeDimension, Attribute, Modifier, ToWgslString, Value,
};

/// Trait to customize the initializing of newly spawned particles.
pub trait InitModifier: Modifier {
    /// Apply the modifier to the init layout of the effect instance.
    fn apply(&self, init_layout: &mut InitLayout);
}

/// An initialization modifier spawning particles on a circle/disc.
#[derive(Debug, Clone, Copy, Reflect, FromReflect)]
pub struct PositionCircleModifier {
    /// The circle center, relative to the emitter position.
    pub center: Vec3,
    /// The circle axis, which is the normalized normal of the circle's plane.
    /// Set this to `Vec3::Z` for a 2D game.
    pub axis: Vec3,
    /// The circle radius.
    pub radius: f32,
    /// The radial speed of the particles on spawn.
    pub speed: Value<f32>,
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

impl Modifier for PositionCircleModifier {
    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::POSITION, Attribute::VELOCITY]
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
    ret.position = c + r * dir + transform[3].xyz;
    // Velocity away from center
    ret.velocity = dir * speed;
    // <<< [PositionCircleModifier]
            "##,
            self.center.to_wgsl_string(),
            tangent.to_wgsl_string(),
            bitangent.to_wgsl_string(),
            radius_code,
            self.speed.to_wgsl_string()
        );
    }
}

/// An initialization modifier spawning particles on a sphere.
#[derive(Debug, Default, Clone, Copy, Reflect, FromReflect)]
pub struct PositionSphereModifier {
    /// The sphere center, relative to the emitter position.
    pub center: Vec3,
    /// The sphere radius.
    pub radius: f32,
    /// The radial speed of the particles on spawn.
    pub speed: Value<f32>,
    /// The shape dimension to spawn from.
    pub dimension: ShapeDimension,
}

impl Modifier for PositionSphereModifier {
    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::POSITION, Attribute::VELOCITY]
    }
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
    ret.position = c + r * dir + transform[3].xyz;
    // Radial velocity away from sphere center
    ret.velocity = dir * speed;
    // <<< [PositionSphereModifier]
"##,
            self.center.to_wgsl_string(),
            radius_code,
            self.speed.to_wgsl_string()
        );
    }
}

/// An initialization modifier spawning particles inside a truncated 3D cone.
///
/// The 3D cone is oriented along the Y axis, with its origin at the center of
/// the top circle truncating the cone. The center of the base circle of the
/// cone is located at a positive Y.
///
/// Particles are spawned somewhere inside the volume or on the surface of a
/// truncated 3D cone defined by its base radius, its top radius, and the height
/// of the cone section.
///
/// The particle velocity is initialized to a random speed along the direction
/// going from the cone apex to the particle position.
#[derive(Debug, Default, Clone, Copy, Reflect, FromReflect)]
pub struct PositionCone3dModifier {
    /// The cone height along its axis, between the base and top radii.
    pub height: f32,
    /// The cone radius at its base, perpendicularly to its axis.
    pub base_radius: f32,
    /// The cone radius at its truncated top, perpendicularly to its axis.
    /// This can be set to zero to get a non-truncated cone.
    pub top_radius: f32,
    /// The speed of the particles on spawn.
    pub speed: Value<f32>,
    /// The shape dimension to spawn from.
    pub dimension: ShapeDimension,
}

impl Modifier for PositionCone3dModifier {
    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::POSITION, Attribute::VELOCITY]
    }
}

impl InitModifier for PositionCone3dModifier {
    fn apply(&self, init_layout: &mut InitLayout) {
        if matches!(self.dimension, ShapeDimension::Surface) {
            unimplemented!("TODO");
        }
        // let radius_code = match self.dimension {
        //     ShapeDimension::Surface => {
        //         // Constant radius
        //         format!("let r = {};", self.radius.to_wgsl_string())
        //     }
        //     ShapeDimension::Volume => {
        //         // Radius uniformly distributed in [0:1], then scaled by ^(1/3) in 3D
        //         // to account for the increased surface covered by increased radii.
        //         // https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere
        //         format!(
        //             "var r = pow(rand(), 1./3.) * {};",
        //             self.radius.to_wgsl_string()
        //         )
        //     }
        // };
        init_layout.position_code = format!(
            r##"
    // >>> [PositionCone3dModifier]
    // Truncated cone height
    let h0 = {0};
    // Random height ratio
    let alpha_h = pow(rand(), 1.0 / 3.0);
    // Random delta height from top
    let h = h0 * alpha_h;
    // Top radius
    let rt = {1};
    // Bottom radius
    let rb = {2};
    // Radius at height h
    let r0 = rt + (rb - rt) * alpha_h;
    // Random delta radius
    let alpha_r = sqrt(rand());
    // Random radius at height h
    let r = r0 * alpha_r;
    // Random base angle
    let theta = rand() * tau;
    let cost = cos(theta);
    let sint = sin(theta);
    // Random position relative to truncated cone origin (not apex)
    let x = r * cost;
    let y = h;
    let z = r * sint;
    let p = vec3<f32>(x, y, z);
    let p2 = transform * vec4<f32>(p, 1.0);
    ret.position = p2.xyz;
    // Emit direction
    let rb2 = rb * alpha_r;
    let pb = vec3<f32>(rb2 * cost, h0, rb2 * sint);
    let dir = transform * vec4<f32>(normalize(pb - p), 0.0);
    // Emit speed
    let speed = {3};
    // Velocity away from cone top/apex
    ret.velocity = dir.xyz * speed;
    // <<< [PositionCone3dModifier]
"##,
            self.height.to_wgsl_string(),
            self.top_radius.to_wgsl_string(),
            self.base_radius.to_wgsl_string(),
            self.speed.to_wgsl_string(),
        );
    }
}

/// A modifier to set the lifetime of all particles.
///
/// Particles with a lifetime are aged each frame by the frame's delta time, and
/// are despawned once their age is greater than or equal to their lifetime.
#[derive(Debug, Default, Clone, Copy, Reflect, FromReflect)]
pub struct ParticleLifetimeModifier {
    /// The lifetime of all particles when they spawn, in seconds.
    pub lifetime: f32,
}

impl Modifier for ParticleLifetimeModifier {
    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::AGE, Attribute::LIFETIME]
    }
}

impl InitModifier for ParticleLifetimeModifier {
    fn apply(&self, init_layout: &mut InitLayout) {
        init_layout.lifetime_code = format!(
            r##"
    // >>> [ParticleLifetimeModifier]
    ret = {0};
    // <<< [ParticleLifetimeModifier]
"##,
            self.lifetime.to_wgsl_string(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use naga::front::wgsl::Parser;

    #[test]
    fn validate() {
        let modifiers: &[&dyn InitModifier] = &[
            &PositionCircleModifier::default(),
            &PositionSphereModifier::default(),
            &PositionCone3dModifier {
                dimension: ShapeDimension::Volume,
                ..Default::default()
            },
        ];
        for modifier in modifiers.iter() {
            let mut layout = InitLayout::default();
            modifier.apply(&mut layout);

            let code = format!(
                r##"fn rand() -> f32 {{
    return 0.0;
}}

let tau: f32 = 6.283185307179586476925286766559;

struct PosVel {{
    position: vec3<f32>,
    velocity: vec3<f32>,
}};

@compute @workgroup_size(64)
fn init_pos_vel(index: u32, transform: mat4x4<f32>) -> PosVel {{
    var ret : PosVel;
{0}
    return ret;
}}"##,
                layout.position_code
            );
            //println!("code: {:?}", code);

            let mut parser = Parser::new();
            let res = parser.parse(&code);
            assert!(res.is_ok());
            // if let Err(err) = res {
            //     println!("Err: {:?}", err);
            // }
        }
    }
}
