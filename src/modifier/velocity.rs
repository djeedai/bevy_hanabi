//! Modifiers to set the velocity of particles.
//!
//! These modifiers directly manipulate a particle's velocity. They're generally
//! useful to initialize the velocity at spawn time, but can occasionally be
//! used during simulation update to enforce a particular velocity. The velocity
//! in turn drives the position update during Euler motion integration.
//!
//! ```txt
//! particle.position += particle.velocity * simulation.delta_time;
//! ```

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    calc_func_id, graph::ExprError, impl_mod_init_update, Attribute, EvalContext, ExprHandle,
    InitContext, InitModifier, Module, UpdateContext, UpdateModifier,
};

/// A modifier to set the velocity of particles radially on a circle.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct SetVelocityCircleModifier {
    /// The circle center, relative to the emitter position.
    ///
    /// Expression type: `Vec3`
    pub center: ExprHandle,
    /// The circle axis, which is the normalized normal of the circle's plane.
    /// Set this to `Vec3::Z` for a 2D game.
    ///
    /// Expression type: `Vec3`
    pub axis: ExprHandle,
    /// The initial speed distribution of a particle when it spawns.
    ///
    /// Expression type: `f32`
    pub speed: ExprHandle,
}

impl_mod_init_update!(
    SetVelocityCircleModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

impl SetVelocityCircleModifier {
    fn eval(
        &self,
        module: &mut Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("set_velocity_circle_{0:016X}", func_id);

        context.make_fn(
            &func_name,
            "transform: mat4x4<f32>, particle: ptr<function, Particle>",
            module,
            &mut |m: &mut Module, ctx: &mut dyn EvalContext| -> Result<String, ExprError> {
                let center = ctx.eval(m, self.center)?;
                let axis = ctx.eval(m, self.axis)?;
                let speed = ctx.eval(m, self.speed)?;

                Ok(format!(
                    r##"    let delta = (*particle).{0} - ({1});
    let radial = normalize(delta - dot(delta, {2}) * ({2}));
    let radial_vec4 = transform * vec4<f32>(radial.xyz, 0.0);
    (*particle).{3} = radial_vec4.xyz * ({4});
"##,
                    Attribute::POSITION.name(),
                    center,
                    axis,
                    Attribute::VELOCITY.name(),
                    speed,
                ))
            },
        )?;

        let code = format!("{}(transform, &particle);\n", func_name);

        Ok(code)
    }
}

#[typetag::serde]
impl InitModifier for SetVelocityCircleModifier {
    fn apply_init(&self, module: &mut Module, context: &mut InitContext) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.init_code += &code;
        Ok(())
    }
}

#[typetag::serde]
impl UpdateModifier for SetVelocityCircleModifier {
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

/// A modifier to set the velocity of particles to a spherical distribution.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct SetVelocitySphereModifier {
    /// Center of the sphere. The radial direction of the velocity is the
    /// direction from the sphere center to the particle position.
    ///
    /// Expression type: `Vec3`
    pub center: ExprHandle,
    /// The initial speed distribution of a particle when it spawns.
    ///
    /// Expression type: `f32`
    pub speed: ExprHandle,
}

impl_mod_init_update!(
    SetVelocitySphereModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

impl SetVelocitySphereModifier {
    fn eval(
        &self,
        module: &mut Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        let center = context.eval(module, self.center)?;
        let speed = context.eval(module, self.speed)?;
        let code = format!(
            "particle.{} = normalize(particle.{} - ({})) * ({});\n",
            Attribute::VELOCITY.name(),
            Attribute::POSITION.name(),
            center,
            speed
        );
        Ok(code)
    }
}

#[typetag::serde]
impl InitModifier for SetVelocitySphereModifier {
    fn apply_init(&self, module: &mut Module, context: &mut InitContext) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.init_code += &code;
        Ok(())
    }
}

#[typetag::serde]
impl UpdateModifier for SetVelocitySphereModifier {
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

/// A modifier to set the velocity of particles along the tangent to an axis.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct SetVelocityTangentModifier {
    /// Origin from which to derive the radial axis based on the particle
    /// position.
    ///
    /// Expression type: `Vec3`
    pub origin: ExprHandle,
    /// Axis defining the normal to the plane containing the radial and tangent
    /// axes.
    ///
    /// Expression type: `Vec3`
    pub axis: ExprHandle,
    /// The initial speed distribution of a particle when it spawns.
    ///
    /// Expression type: `f32`
    pub speed: ExprHandle,
}

impl_mod_init_update!(
    SetVelocityTangentModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

impl SetVelocityTangentModifier {
    fn eval(
        &self,
        module: &mut Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("set_velocity_tangent_{0:016X}", func_id);

        context.make_fn(
            &func_name,
            "transform: mat4x4<f32>, particle: ptr<function, Particle>",
            module,
            &mut |m: &mut Module, ctx: &mut dyn EvalContext| -> Result<String, ExprError> {
                let origin = ctx.eval(m, self.origin)?;
                let axis = ctx.eval(m, self.axis)?;
                let speed = ctx.eval(m, self.speed)?;

                Ok(format!(
                    r##"    let radial = (*particle).{0} - ({1});
    let tangent = normalize(cross({2}, radial));
    let tangent_vec4 = transform * vec4<f32>(tangent.xyz, 0.0);
    (*particle).{3} = tangent_vec4.xyz * ({4});
"##,
                    Attribute::POSITION.name(),
                    origin,
                    axis,
                    Attribute::VELOCITY.name(),
                    speed,
                ))
            },
        )?;

        let code = format!("{}(transform, &particle);\n", func_name);

        Ok(code)
    }
}

#[typetag::serde]
impl InitModifier for SetVelocityTangentModifier {
    fn apply_init(&self, module: &mut Module, context: &mut InitContext) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.init_code += &code;
        Ok(())
    }
}

#[typetag::serde]
impl UpdateModifier for SetVelocityTangentModifier {
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
