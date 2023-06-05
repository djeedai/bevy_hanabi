//! Modifiers to initialize particles when they spawn.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    graph::{EvalContext, ExprError},
    modifier::ShapeDimension,
    Attribute, BoxedModifier, DimValue, Expr, ExprHandle, Modifier, ModifierContext, Module,
    PropertyLayout, ToWgslString, Value,
};

/// Particle initializing shader code generation context.
#[derive(Debug, PartialEq)]
pub struct InitContext<'a> {
    /// Module being populated with new expressions from modifiers.
    pub module: &'a mut Module,
    /// Main particle initializing code, which needs to assign the fields of the
    /// `particle` struct instance.
    pub init_code: String,
    /// Extra functions emitted at top level, which `init_code` can call.
    pub init_extra: String,
    /// Layout of properties for the current effect.
    pub property_layout: &'a PropertyLayout,
}

impl<'a> InitContext<'a> {
    /// Create a new init context.
    pub fn new(module: &'a mut Module, property_layout: &'a PropertyLayout) -> Self {
        Self {
            module,
            init_code: String::new(),
            init_extra: String::new(),
            property_layout,
        }
    }
}

impl<'a> EvalContext for InitContext<'a> {
    fn module(&self) -> &Module {
        self.module
    }

    fn property_layout(&self) -> &PropertyLayout {
        self.property_layout
    }

    fn expr(&self, expr: ExprHandle) -> Result<&Expr, ExprError> {
        self.module
            .get(expr)
            .ok_or(ExprError::InvalidExprHandleError(format!(
                "Cannot find expression with handle {:?} in the current module. Check that the Module used to build the expression was the same used in the EvalContext or the original EffectAsset.",
                expr
            )))
    }

    fn eval(&self, handle: ExprHandle) -> Result<String, ExprError> {
        self.expr(handle)?.eval(self)
    }
}

/// Trait to customize the initializing of newly spawned particles.
#[typetag::serde]
pub trait InitModifier: Modifier {
    /// Append the initializing code.
    fn apply(&self, context: &mut InitContext) -> Result<(), ExprError>;
}

/// Macro to implement the [`Modifier`] trait for an init modifier.
macro_rules! impl_mod_init {
    ($t:ty, $attrs:expr) => {
        #[typetag::serde]
        impl Modifier for $t {
            fn context(&self) -> ModifierContext {
                ModifierContext::Init
            }

            fn as_init(&self) -> Option<&dyn InitModifier> {
                Some(self)
            }

            fn as_init_mut(&mut self) -> Option<&mut dyn InitModifier> {
                Some(self)
            }

            fn attributes(&self) -> &[Attribute] {
                $attrs
            }

            fn boxed_clone(&self) -> BoxedModifier {
                Box::new(*self)
            }
        }
    };
}

/// A modifier to set the initial value of any particle attribute.
///
/// This modifier initializes an [`Attribute`] of the particle of a system to a
/// given graph expression when the particle spawns.
///
/// This is a basic building block to create any complex effect. Most other init
/// modifiers are convenience helpers to achieve a behavior which can otherwise
/// be produced, more verbosely, by setting the individual attributes of a
/// particle with instances of this modifier.
///
/// # Warning
///
/// At the minute there is no validation that the type of the value is the same
/// as the type of the attribute. Users are advised to be careful, until more
/// safeguards are added.
///
/// # Attributes
///
/// This modifier requires the attribute specified in the `attribute` field.
#[derive(Debug, Clone, Copy, Reflect, FromReflect, Serialize, Deserialize)]
pub struct InitAttributeModifier {
    /// The name of the attribute to initialize.
    pub attribute: Attribute,
    /// The initial value of the attribute.
    pub value: ExprHandle,
}

impl InitAttributeModifier {
    /// Create a new instance of an [`InitAttributeModifier`].
    pub fn new(attribute: Attribute, value: ExprHandle) -> Self {
        Self { attribute, value }
    }
}

#[typetag::serde]
impl Modifier for InitAttributeModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Init
    }

    fn as_init(&self) -> Option<&dyn InitModifier> {
        Some(self)
    }

    fn as_init_mut(&mut self) -> Option<&mut dyn InitModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[Attribute] {
        std::slice::from_ref(&self.attribute)
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(self.clone())
    }
}

#[typetag::serde]
impl InitModifier for InitAttributeModifier {
    fn apply(&self, context: &mut InitContext) -> Result<(), ExprError> {
        assert!(context.module.get(self.value).is_some());
        let attr = context.module.attr(self.attribute);
        let attr = context.eval(attr)?;
        let expr = context.eval(self.value)?;
        context.init_code += &format!("{} = {};\n", attr, expr);
        Ok(())
    }
}

/// An initialization modifier spawning particles on a circle/disc.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
#[derive(Debug, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct InitPositionCircleModifier {
    /// The circle center, relative to the emitter position.
    pub center: Vec3,
    /// The circle axis, which is the normalized normal of the circle's plane.
    /// Set this to `Vec3::Z` for a 2D game.
    pub axis: Vec3,
    /// The circle radius.
    pub radius: f32,
    /// The shape dimension to spawn from.
    pub dimension: ShapeDimension,
}

impl Default for InitPositionCircleModifier {
    fn default() -> Self {
        Self {
            center: Default::default(),
            axis: Vec3::Z,
            radius: Default::default(),
            dimension: Default::default(),
        }
    }
}

impl_mod_init!(InitPositionCircleModifier, &[Attribute::POSITION]);

#[typetag::serde]
impl InitModifier for InitPositionCircleModifier {
    fn apply(&self, context: &mut InitContext) -> Result<(), ExprError> {
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
            r##"fn init_position_circle(particle: ptr<function, Particle>) {{
    // Circle center
    let c = {};
    // Circle basis
    let tangent = {};
    let bitangent = {};
    // Circle radius
    {}
    // Spawn random point on/in circle
    let theta = rand() * tau;
    let dir = tangent * cos(theta) + bitangent * sin(theta);
    (*particle).{} = c + r * dir;
}}
"##,
            self.center.to_wgsl_string(),
            tangent.to_wgsl_string(),
            bitangent.to_wgsl_string(),
            radius_code,
            Attribute::POSITION.name(),
        );

        context.init_code += "init_position_circle(&particle);\n";

        Ok(())
    }
}

/// An initialization modifier spawning particles on a sphere.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct InitPositionSphereModifier {
    /// The sphere center, relative to the emitter position.
    pub center: Vec3,
    /// The sphere radius.
    pub radius: f32,
    /// The shape dimension to spawn from.
    pub dimension: ShapeDimension,
}

impl_mod_init!(InitPositionSphereModifier, &[Attribute::POSITION]);

#[typetag::serde]
impl InitModifier for InitPositionSphereModifier {
    fn apply(&self, context: &mut InitContext) -> Result<(), ExprError> {
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
                    "let r = pow(rand(), 1./3.) * {};",
                    self.radius.to_wgsl_string()
                )
            }
        };

        context.init_extra += &format!(
            r##"fn init_position_sphere(particle: ptr<function, Particle>) {{
    // Sphere center
    let c = {};

    // Sphere radius
    {}

    // Spawn randomly along the sphere surface using Archimedes's theorem
    let theta = rand() * tau;
    let z = rand() * 2. - 1.;
    let phi = acos(z);
    let sinphi = sin(phi);
    let x = sinphi * cos(theta);
    let y = sinphi * sin(theta);
    let dir = vec3<f32>(x, y, z);
    (*particle).{} = c + r * dir;
}}
"##,
            self.center.to_wgsl_string(),
            radius_code,
            Attribute::POSITION.name(),
        );

        context.init_code += "init_position_sphere(&particle);\n";

        Ok(())
    }
}

/// An initialization modifier spawning particles inside a truncated 3D cone.
///
/// The 3D cone is oriented along the Y axis, with its origin at the center of
/// the base circle of the cone. The center of the top circle truncating the
/// cone is located at a positive Y.
///
/// Particles are spawned somewhere inside the volume or on the surface of a
/// truncated 3D cone defined by its base radius, its top radius, and the height
/// of the cone section.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct InitPositionCone3dModifier {
    /// The cone height along its axis, between the base and top radii.
    pub height: f32,
    /// The cone radius at its base, perpendicularly to its axis.
    pub base_radius: f32,
    /// The cone radius at its truncated top, perpendicularly to its axis.
    /// This can be set to zero to get a non-truncated cone.
    pub top_radius: f32,
    /// The shape dimension to spawn from.
    pub dimension: ShapeDimension,
}

impl_mod_init!(InitPositionCone3dModifier, &[Attribute::POSITION]);

#[typetag::serde]
impl InitModifier for InitPositionCone3dModifier {
    fn apply(&self, context: &mut InitContext) -> Result<(), ExprError> {
        context.init_extra += &format!(
            r##"fn init_position_cone3d(transform: mat4x4<f32>, particle: ptr<function, Particle>) {{
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
    let r0 = rb + (rt - rb) * alpha_h;
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
    let p2 = transform * vec4<f32>(p, 0.0);
    (*particle).{3} = p2.xyz;
}}
"##,
            self.height.to_wgsl_string(),
            self.top_radius.to_wgsl_string(),
            self.base_radius.to_wgsl_string(),
            Attribute::POSITION.name(),
        );

        context.init_code += "init_position_cone3d(transform, &particle);\n";

        Ok(())
    }
}

/// A modifier to set the initial velocity of particles radially on a circle.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct InitVelocityCircleModifier {
    /// The circle center, relative to the emitter position.
    pub center: Vec3,
    /// The circle axis, which is the normalized normal of the circle's plane.
    /// Set this to `Vec3::Z` for a 2D game.
    pub axis: Vec3,
    /// The initial speed distribution of a particle when it spawns.
    pub speed: Value<f32>,
}

impl_mod_init!(
    InitVelocityCircleModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

#[typetag::serde]
impl InitModifier for InitVelocityCircleModifier {
    fn apply(&self, context: &mut InitContext) -> Result<(), ExprError> {
        context.init_extra += &format!(
            r##"fn init_velocity_circle(transform: mat4x4<f32>, particle: ptr<function, Particle>) {{
    let delta = (*particle).{0} - {1};
    let radial = normalize(delta - dot(delta, {2}) * {2});
    let radial_vec4 = transform * vec4<f32>(radial.xyz, 0.0);
    (*particle).{3} = radial_vec4.xyz * {4};
}}
"##,
            Attribute::POSITION.name(),
            self.center.to_wgsl_string(),
            self.axis.to_wgsl_string(),
            Attribute::VELOCITY.name(),
            self.speed.to_wgsl_string(),
        );

        context.init_code += "init_velocity_circle(transform, &particle);\n";

        Ok(())
    }
}

/// A modifier to set the initial velocity of particles to a spherical
/// distribution.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct InitVelocitySphereModifier {
    /// Center of the sphere. The radial direction of the velocity is the
    /// direction from the sphere center to the particle position.
    pub center: Vec3,
    /// The initial speed distribution of a particle when it spawns.
    pub speed: Value<f32>,
}

impl_mod_init!(
    InitVelocitySphereModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

#[typetag::serde]
impl InitModifier for InitVelocitySphereModifier {
    fn apply(&self, context: &mut InitContext) -> Result<(), ExprError> {
        context.init_code += &format!(
            "particle.{} = normalize(particle.{} - {}) * ({});\n",
            Attribute::VELOCITY.name(),
            Attribute::POSITION.name(),
            self.center.to_wgsl_string(),
            self.speed.to_wgsl_string()
        );
        Ok(())
    }
}

/// A modifier to set the initial velocity of particles along the tangent to an
/// axis.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct InitVelocityTangentModifier {
    /// Origin from which to derive the radial axis based on the particle
    /// position.
    pub origin: Vec3,
    /// Axis defining the normal to the plane containing the radial and tangent
    /// axes.
    pub axis: Vec3,
    /// The initial speed distribution of a particle when it spawns.
    pub speed: Value<f32>,
}

impl_mod_init!(
    InitVelocityTangentModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

#[typetag::serde]
impl InitModifier for InitVelocityTangentModifier {
    fn apply(&self, context: &mut InitContext) -> Result<(), ExprError> {
        context.init_extra += &format!(
            r##"fn init_velocity_tangent(transform: mat4x4<f32>, particle: ptr<function, Particle>) {{
    let radial = (*particle).{0} - {1};
    let tangent = normalize(cross({2}, radial));
    let tangent_vec4 = transform * vec4<f32>(tangent.xyz, 0.0);
    (*particle).{3} = tangent_vec4.xyz * {4};
}}
"##,
            Attribute::POSITION.name(),
            self.origin.to_wgsl_string(),
            self.axis.to_wgsl_string(),
            Attribute::VELOCITY.name(),
            self.speed.to_wgsl_string(),
        );

        context.init_code += "init_velocity_tangent(transform, &particle);\n";

        Ok(())
    }
}

/// Modifier to initialize the per-particle size attribute.
///
/// The particle is initialized with a fixed or randomly distributed size value,
/// and will retain that size unless another modifier (like
/// [`SizeOverLifetimeModifier`]) changes its size after spawning.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::SIZE`] or [`Attribute::SIZE2`]
///
/// [`SizeOverLifetimeModifier`]: crate::SizeOverLifetimeModifier
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

    fn as_init(&self) -> Option<&dyn InitModifier> {
        Some(self)
    }

    fn as_init_mut(&mut self) -> Option<&mut dyn InitModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[Attribute] {
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
    fn apply(&self, context: &mut InitContext) -> Result<(), ExprError> {
        context.init_code += &format!(
            "particle.{} = {};\n",
            Attribute::SIZE.name(),
            self.size.to_wgsl_string(),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParticleLayout;

    use naga::front::wgsl::Parser;

    #[test]
    fn validate() {
        let mut module = Module::default();
        let modifiers: &[&dyn InitModifier] = &[
            &InitPositionCircleModifier::default(),
            &InitPositionSphereModifier::default(),
            &InitPositionCone3dModifier {
                dimension: ShapeDimension::Volume,
                ..Default::default()
            },
            &InitVelocityCircleModifier::default(),
            &InitVelocitySphereModifier::default(),
            &InitVelocityTangentModifier::default(),
        ];
        for &modifier in modifiers.iter() {
            let property_layout = PropertyLayout::default();
            let mut context = InitContext::new(&mut module, &property_layout);
            assert!(modifier.apply(&mut context).is_ok());
            let init_code = context.init_code;
            let init_extra = context.init_extra;

            let mut particle_layout = ParticleLayout::new();
            for &attr in modifier.attributes() {
                particle_layout = particle_layout.append(attr);
            }
            let particle_layout = particle_layout.build();
            let attributes_code = particle_layout.generate_code();

            let code = format!(
                r##"fn rand() -> f32 {{
    return 0.0;
}}

const tau: f32 = 6.283185307179586476925286766559;

struct Particle {{
    {attributes_code}
}};

{init_extra}

@compute @workgroup_size(64)
fn main() {{
    var particle = Particle();
    var transform: mat4x4<f32> = mat4x4<f32>();
{init_code}
}}"##
            );
            // println!("code: {:?}", code);

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

    // #[test]
    // fn naga() {
    //     let mut parser = Parser::new();
    //     let res = parser.parse(&"let x = sim_params.deltaTime;");
    //     let module = res.unwrap();
    //     println!("{:?}", module);
    // }
}
