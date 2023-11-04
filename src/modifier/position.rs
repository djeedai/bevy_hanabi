//! Modifiers to set the position of particles.
//!
//! These modifiers directly manipulate a particle's position. They're generally
//! useful to initialize the position at spawn time, but can occasionally be
//! used during simulation update to enforce a particular position.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    calc_func_id, graph::ExprError, impl_mod_init_update, modifier::ShapeDimension, Attribute,
    EvalContext, ExprHandle, InitContext, InitModifier, Module, UpdateContext, UpdateModifier,
};

/// A modifier to set the position of particles on or inside a circle/disc,
/// randomly.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct SetPositionCircleModifier {
    /// The circle center, relative to the emitter position.
    ///
    /// Expression type: `Vec3`
    pub center: ExprHandle,
    /// The circle axis, which is the normalized normal of the circle's plane.
    /// Set this to `Vec3::Z` for a 2D game.
    ///
    /// Expression type: `Vec3`
    pub axis: ExprHandle,
    /// The circle radius.
    ///
    /// Expression type: `f32`
    pub radius: ExprHandle,
    /// The shape dimension to set the position to.
    ///
    /// Note the particular interpretation of the dimension for this shape,
    /// which unlike other shapes is a 2D one to begin with:
    /// - [`ShapeDimension::Volume`] randomly position the particle anywhere on
    /// the "volume" of the shape, which here is understood to be the 2D disc
    /// surface including its origin (`dist <= r`).
    /// - [`ShapeDimension::Surface`] randomly position the particle
    /// anywhere on the "surface" of the shape, which here is understood to
    /// be the perimeter circle, the set of points at a distance from the center
    /// exactly equal to the radius (`dist == r`).
    pub dimension: ShapeDimension,
}

impl_mod_init_update!(SetPositionCircleModifier, &[Attribute::POSITION]);

impl SetPositionCircleModifier {
    fn eval(
        &self,
        module: &mut Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("set_position_circle_{0:016X}", func_id);

        context.make_fn(
            &func_name,
            "particle: ptr<function, Particle>",
            module,
            &mut |m: &mut Module, ctx: &mut dyn EvalContext| -> Result<String, ExprError> {
                let center = ctx.eval(m, self.center)?;
                let axis = ctx.eval(m, self.axis)?;

                let radius = match self.dimension {
                    ShapeDimension::Surface => {
                        // Constant radius
                        format!("let r = {};", ctx.eval(m, self.radius)?)
                    }
                    ShapeDimension::Volume => {
                        // Radius uniformly distributed in [0:1], then square-rooted
                        // to account for the increased perimeter covered by increased radii.
                        format!("let r = sqrt(frand()) * ({});", ctx.eval(m, self.radius)?)
                    }
                };

                Ok(format!(
                    r##"    // Circle center
    let c = {};
    // Circle basis
    let n = {};
    let sign = step(0.0, n.z) * 2.0 - 1.0;
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let tangent = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitangent = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    // Circle radius
    {}
    // Spawn random point on/in circle
    let theta = frand() * tau;
    let dir = tangent * cos(theta) + bitangent * sin(theta);
    (*particle).{} = c + r * dir;
"##,
                    center,
                    axis,
                    radius,
                    Attribute::POSITION.name(),
                ))
            },
        )?;

        let code = format!("{}(&particle);\n", func_name);

        Ok(code)
    }
}

#[typetag::serde]
impl InitModifier for SetPositionCircleModifier {
    fn apply_init(&self, module: &mut Module, context: &mut InitContext) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.init_code += &code;
        Ok(())
    }
}

#[typetag::serde]
impl UpdateModifier for SetPositionCircleModifier {
    fn apply_update(
        &self,
        module: &mut Module,
        context: &mut UpdateContext,
    ) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.update_code += &code;
        Ok(())
    }
}

/// A modifier to set the position of particles on or inside a sphere, randomly.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
#[derive(Debug, Clone, Copy, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub struct SetPositionSphereModifier {
    /// The sphere center, relative to the emitter position.
    ///
    /// Expression type: `Vec3`
    pub center: ExprHandle,
    /// The sphere radius.
    ///
    /// Expression type: `f32`
    pub radius: ExprHandle,
    /// The shape dimension to set the position to.
    pub dimension: ShapeDimension,
}

impl_mod_init_update!(SetPositionSphereModifier, &[Attribute::POSITION]);

impl SetPositionSphereModifier {
    fn eval(
        &self,
        module: &mut Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("set_position_sphere_{0:016X}", func_id);

        context.make_fn(
            &func_name,
            "particle: ptr<function, Particle>",
            module,
            &mut |m: &mut Module, ctx: &mut dyn EvalContext| -> Result<String, ExprError> {
                let center = ctx.eval(m, self.center)?;

                let radius = match self.dimension {
                    ShapeDimension::Surface => {
                        // Constant radius
                        format!("let r = {};", ctx.eval(m, self.radius)?)
                    }
                    ShapeDimension::Volume => {
                        // Radius uniformly distributed in [0:1], then scaled by ^(1/3) in 3D
                        // to account for the increased surface covered by increased radii.
                        // https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere
                        format!(
                            "let r = pow(frand(), 1./3.) * ({});",
                            ctx.eval(m, self.radius)?
                        )
                    }
                };

                Ok(format!(
                    r##"    // Sphere center
    let c = {};

    // Sphere radius
    {}

    // Spawn randomly along the sphere surface using Archimedes's theorem
    let theta = frand() * tau;
    let z = frand() * 2. - 1.;
    let phi = acos(z);
    let sinphi = sin(phi);
    let x = sinphi * cos(theta);
    let y = sinphi * sin(theta);
    let dir = vec3<f32>(x, y, z);
    (*particle).{} = c + r * dir;
"##,
                    center,
                    radius,
                    Attribute::POSITION.name(),
                ))
            },
        )?;

        let code = format!("{}(&particle);\n", func_name);

        Ok(code)
    }
}

#[typetag::serde]
impl InitModifier for SetPositionSphereModifier {
    fn apply_init(&self, module: &mut Module, context: &mut InitContext) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.init_code += &code;
        Ok(())
    }
}

#[typetag::serde]
impl UpdateModifier for SetPositionSphereModifier {
    fn apply_update(
        &self,
        module: &mut Module,
        context: &mut UpdateContext,
    ) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.update_code += &code;
        Ok(())
    }
}

/// A modifier to set the position of particles on a truncated 3D cone.
///
/// The 3D cone is oriented along the Y axis, with its origin at the center of
/// the base circle of the cone. The center of the top circle truncating the
/// cone is located at a positive Y.
///
/// Particles are moved somewhere inside the volume or on the surface of a
/// truncated 3D cone defined by its base radius, its top radius, and the height
/// of the cone section.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct SetPositionCone3dModifier {
    /// The cone height along its axis, between the base and top radii.
    ///
    /// Expression type: `f32`
    pub height: ExprHandle,
    /// The cone radius at its base, perpendicularly to its axis.
    ///
    /// Expression type: `f32`
    pub base_radius: ExprHandle,
    /// The cone radius at its truncated top, perpendicularly to its axis.
    /// This can be set to zero to get a non-truncated cone.
    ///
    /// Expression type: `f32`
    pub top_radius: ExprHandle,
    /// The shape dimension to set the position to.
    pub dimension: ShapeDimension,
}

impl_mod_init_update!(SetPositionCone3dModifier, &[Attribute::POSITION]);

impl SetPositionCone3dModifier {
    fn eval(
        &self,
        module: &mut Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("set_position_cone3d_{0:016X}", func_id);

        context.make_fn(
            &func_name,
            "transform: mat4x4<f32>, particle: ptr<function, Particle>",
            module,
            &mut |m: &mut Module, ctx: &mut dyn EvalContext| -> Result<String, ExprError> {
                let height = ctx.eval(m, self.height)?;
                let top_radius = ctx.eval(m, self.top_radius)?;
                let base_radius = ctx.eval(m, self.base_radius)?;

                Ok(format!(
                    r##"    // Truncated cone height
    let h0 = {0};
    // Random height ratio
    let alpha_h = pow(frand(), 1.0 / 3.0);
    // Random delta height from top
    let h = h0 * alpha_h;
    // Top radius
    let rt = {1};
    // Bottom radius
    let rb = {2};
    // Radius at height h
    let r0 = rb + (rt - rb) * alpha_h;
    // Random delta radius
    let alpha_r = sqrt(frand());
    // Random radius at height h
    let r = r0 * alpha_r;
    // Random base angle
    let theta = frand() * tau;
    let cost = cos(theta);
    let sint = sin(theta);
    // Random position relative to truncated cone origin (not apex)
    let x = r * cost;
    let y = h;
    let z = r * sint;
    let p = vec3<f32>(x, y, z);
    let p2 = transform * vec4<f32>(p, 0.0);
    (*particle).{3} = p2.xyz;
"##,
                    height,
                    top_radius,
                    base_radius,
                    Attribute::POSITION.name(),
                ))
            },
        )?;

        let code = format!("{}(transform, &particle);\n", func_name);

        Ok(code)
    }
}

#[typetag::serde]
impl InitModifier for SetPositionCone3dModifier {
    fn apply_init(&self, module: &mut Module, context: &mut InitContext) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.init_code += &code;
        Ok(())
    }
}

#[typetag::serde]
impl UpdateModifier for SetPositionCone3dModifier {
    fn apply_update(
        &self,
        module: &mut Module,
        context: &mut UpdateContext,
    ) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.update_code += &code;
        Ok(())
    }
}
