//! Modifiers to initialize particles when they spawn.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    modifier::ShapeDimension, Attribute, BoxedModifier, DimValue, Modifier, ModifierContext,
    ToWgslString, Value,
};

/// Particle initializing shader code generation context.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InitContext {
    /// Main particle initializing code, which needs to assign the fields of the
    /// `particle` struct instance.
    pub init_code: String,
    /// Extra functions emitted at top level, which `init_code` can call.
    pub init_extra: String,
}

/// Trait to customize the initializing of newly spawned particles.
#[typetag::serde]
pub trait InitModifier: Modifier {
    /// Append the initializing code.
    fn apply(&self, context: &mut InitContext);
}

/// An initialization modifier spawning particles on a circle/disc.
#[derive(Debug, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
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

#[typetag::serde]
impl Modifier for PositionCircleModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Init
    }

    fn init(&self) -> Option<&dyn InitModifier> {
        Some(self)
    }

    fn init_mut(&mut self) -> Option<&mut dyn InitModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::POSITION, Attribute::VELOCITY]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }
}

#[typetag::serde]
impl InitModifier for PositionCircleModifier {
    fn apply(&self, context: &mut InitContext) {
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

        context.init_extra += &format!(
            r##"fn position_circle(particle: ptr<function, Particle>) {{
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
    (*particle).{} = c + r * dir;
    // Velocity away from center
    (*particle).{} = dir * speed;
}}
"##,
            self.center.to_wgsl_string(),
            tangent.to_wgsl_string(),
            bitangent.to_wgsl_string(),
            radius_code,
            self.speed.to_wgsl_string(),
            Attribute::POSITION.name(),
            Attribute::VELOCITY.name(),
        );

        context.init_code += "position_circle(&particle);\n";
    }
}

/// An initialization modifier spawning particles on a sphere.
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
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

#[typetag::serde]
impl Modifier for PositionSphereModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Init
    }

    fn init(&self) -> Option<&dyn InitModifier> {
        Some(self)
    }

    fn init_mut(&mut self) -> Option<&mut dyn InitModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::POSITION, Attribute::VELOCITY]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }
}

#[typetag::serde]
impl InitModifier for PositionSphereModifier {
    fn apply(&self, context: &mut InitContext) {
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

        context.init_extra += &format!(
            r##"fn position_sphere(particle: ptr<function, Particle>) {{
    // Sphere center
    let c = {};
    // Sphere radius
    {}
    // Radial speed
    let speed = {};
    // Spawn randomly along the sphere surface using Archimedes's theorem
    var theta = rand() * tau;
    var z = rand() * 2. - 1.;
    var phi = acos(z);
    var sinphi = sin(phi);
    var x = sinphi * cos(theta);
    var y = sinphi * sin(theta);
    var dir = vec3<f32>(x, y, z);
    (*particle).{} = c + r * dir;
    // Radial velocity away from sphere center
    (*particle).{} = dir * speed;
}}
"##,
            self.center.to_wgsl_string(),
            radius_code,
            self.speed.to_wgsl_string(),
            Attribute::POSITION.name(),
            Attribute::VELOCITY.name(),
        );

        context.init_code += "position_sphere(&particle);\n";
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
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
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

#[typetag::serde]
impl Modifier for PositionCone3dModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Init
    }

    fn init(&self) -> Option<&dyn InitModifier> {
        Some(self)
    }

    fn init_mut(&mut self) -> Option<&mut dyn InitModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::POSITION, Attribute::VELOCITY]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }
}

#[typetag::serde]
impl InitModifier for PositionCone3dModifier {
    fn apply(&self, context: &mut InitContext) {
        context.init_extra += &format!(
            r##"fn position_cone3d(particle: ptr<function, Particle>) {{
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
    (*particle).{3} = p2.xyz;
    // Emit direction
    let rb2 = rb * alpha_r;
    let pb = vec3<f32>(rb2 * cost, h0, rb2 * sint);
    let dir = transform * vec4<f32>(normalize(pb - p), 0.0);
    // Emit speed
    let speed = {4};
    // Velocity away from cone top/apex
    (*particle).{5} = dir.xyz * speed;
}}
"##,
            self.height.to_wgsl_string(),
            self.top_radius.to_wgsl_string(),
            self.base_radius.to_wgsl_string(),
            Attribute::POSITION.name(),
            self.speed.to_wgsl_string(),
            Attribute::VELOCITY.name(),
        );

        context.init_code += "position_cone3d(&particle);\n";
    }
}

/// A modifier to set the lifetime of all particles.
///
/// Particles with a lifetime are aged each frame by the frame's delta time, and
/// are despawned once their age is greater than or equal to their lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct ParticleLifetimeModifier {
    /// The lifetime of all particles when they spawn, in seconds.
    pub lifetime: f32,
}

impl Default for ParticleLifetimeModifier {
    fn default() -> Self {
        Self { lifetime: 5. }
    }
}

#[typetag::serde]
impl Modifier for ParticleLifetimeModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Init
    }

    fn init(&self) -> Option<&dyn InitModifier> {
        Some(self)
    }

    fn init_mut(&mut self) -> Option<&mut dyn InitModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::AGE, Attribute::LIFETIME]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }
}

#[typetag::serde]
impl InitModifier for ParticleLifetimeModifier {
    fn apply(&self, context: &mut InitContext) {
        context.init_code += &format!(
            "particle.{} = {};\n",
            Attribute::LIFETIME.name(),
            self.lifetime.to_wgsl_string()
        );
    }
}

/// Modifier to initialize the per-particle size attribute.
///
/// The particle is initialized with a fixed or randomly distributed size value,
/// and will retain that size unless another modifier (like
/// [`SizeOverLifetimeModifier`]) changes its size after spawning.
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct InitSizeModifier {
    /// The size to initialize each particle with.
    ///
    /// Only [`DimValue::D1`] and [`DimValue::D2`] are valid. The former
    /// requires the [`Attribute::SIZE`] attribute in the particle layout, while
    /// the latter requires the [`Attribute::SIZE2`] attribute.
    pub size: DimValue,
}

#[typetag::serde]
impl Modifier for InitSizeModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Init
    }

    fn init(&self) -> Option<&dyn InitModifier> {
        Some(self)
    }

    fn init_mut(&mut self) -> Option<&mut dyn InitModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[&'static Attribute] {
        match self.size {
            DimValue::D1(_) => &[Attribute::SIZE],
            DimValue::D2(_) => &[Attribute::SIZE2],
            _ => panic!("Invalid dimension for InitSizeModifier; only 1D and 2D values are valid."),
        }
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }
}

#[typetag::serde]
impl InitModifier for InitSizeModifier {
    fn apply(&self, context: &mut InitContext) {
        context.init_code += &format!(
            "particle.{} = {};\n",
            Attribute::SIZE.name(),
            self.size.to_wgsl_string(),
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
            let mut context = InitContext::default();
            modifier.apply(&mut context);

            let code = format!(
                r##"fn rand() -> f32 {{
    return 0.0;
}}

let tau: f32 = 6.283185307179586476925286766559;

struct PosVel {{
    position: vec3<f32>,
    velocity: vec3<f32>,
}};

{}

@compute @workgroup_size(64)
fn main() {{
    var particle = Particle();
{}
}}"##,
                context.init_extra, context.init_code,
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
