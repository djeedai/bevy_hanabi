//! Modifiers influencing the acceleration of particles.
//!
//! The particle acceleration directly drives their velocity. It's applied each
//! frame during simulation update.
/// ```txt
/// particle.velocity += acceleration * simulation.delta_time;
/// ```
use std::hash::Hash;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    calc_func_id,
    graph::{BuiltInExpr, EvalContext, ExprError},
    impl_mod_update, Attribute, BoxedModifier, ExprHandle, Modifier, ModifierContext, Module,
    UpdateContext, UpdateModifier,
};

/// A modifier to apply a uniform acceleration to all particles each frame, to
/// simulate gravity or any other global force.
///
/// The acceleration is the same for all particles of the effect, and is applied
/// each frame to modify the particle's velocity based on the simulation
/// timestep.
///
/// ```txt
/// particle.velocity += acceleration * simulation.delta_time;
/// ```
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct AccelModifier {
    /// The acceleration to apply to all particles in the effect each frame.
    ///
    /// Expression type: `Vec3`
    accel: ExprHandle,
}

impl AccelModifier {
    /// Create a new modifier from an acceleration expression.
    pub fn new(accel: ExprHandle) -> Self {
        Self { accel }
    }

    /// Create a new modifier with an acceleration derived from a property.
    pub fn via_property(module: &mut Module, property_name: impl Into<String>) -> Self {
        Self {
            accel: module.prop(property_name),
        }
    }

    /// Create a new modifier with a constant acceleration.
    pub fn constant(module: &mut Module, acceleration: Vec3) -> Self {
        Self {
            accel: module.lit(acceleration),
        }
    }
}

#[typetag::serde]
impl Modifier for AccelModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Update
    }

    fn as_update(&self) -> Option<&dyn UpdateModifier> {
        Some(self)
    }

    fn as_update_mut(&mut self) -> Option<&mut dyn UpdateModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[Attribute] {
        &[Attribute::VELOCITY]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }
}

#[typetag::serde]
impl UpdateModifier for AccelModifier {
    fn apply_update(
        &self,
        module: &mut Module,
        context: &mut UpdateContext,
    ) -> Result<(), ExprError> {
        let attr = module.attr(Attribute::VELOCITY);
        let attr = context.eval(module, attr)?;
        let expr = context.eval(module, self.accel)?;
        let dt = BuiltInExpr::new(crate::graph::BuiltInOperator::DeltaTime).eval(context)?;
        context.update_code += &format!("{} += ({}) * {};", attr, expr, dt);
        Ok(())
    }
}

/// A modifier to apply a radial acceleration to all particles each frame.
///
/// The acceleration is the same for all particles of the effect, and is applied
/// each frame to modify the particle's velocity based on the simulation
/// timestep.
///
/// ```txt
/// particle.velocity += acceleration * simulation.delta_time;
/// ```
///
/// In the absence of other modifiers, the radial acceleration alone, if
/// oriented toward the center point, makes particles move toward that center
/// point. The radial direction is calculated as the direction from the modifier
/// origin to the particle position.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct RadialAccelModifier {
    /// The center point the radial direction is calculated from.
    ///
    /// Expression type: `Vec3`
    origin: ExprHandle,
    /// The acceleration to apply to all particles in the effect each frame.
    ///
    /// Expression type: `f32`
    accel: ExprHandle,
}

impl RadialAccelModifier {
    /// Create a new modifier from an origin expression and an acceleration
    /// expression.
    pub fn new(origin: ExprHandle, accel: ExprHandle) -> Self {
        Self { origin, accel }
    }

    /// Create a new modifier with an acceleration derived from a property.
    ///
    /// The origin of the sphere defining the radial direction is constant.
    pub fn via_property(
        module: &mut Module,
        origin: Vec3,
        property_name: impl Into<String>,
    ) -> Self {
        Self {
            origin: module.lit(origin),
            accel: module.prop(property_name),
        }
    }

    /// Create a new modifier with a constant radial origin and acceleration.
    pub fn constant(module: &mut Module, origin: Vec3, acceleration: f32) -> Self {
        Self {
            origin: module.lit(origin),
            accel: module.lit(acceleration),
        }
    }
}

impl_mod_update!(
    RadialAccelModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

#[typetag::serde]
impl UpdateModifier for RadialAccelModifier {
    fn apply_update(
        &self,
        module: &mut Module,
        context: &mut UpdateContext,
    ) -> Result<(), ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("radial_accel_{0:016X}", func_id);

        let origin = context.eval(module, self.origin)?;
        let accel = context.eval(module, self.accel)?;

        context.update_extra += &format!(
            r##"fn {}(particle: ptr<function, Particle>) {{
    let radial = normalize((*particle).{} - {});
    (*particle).{} += radial * (({}) * sim_params.delta_time);
}}
"##,
            func_name,
            Attribute::POSITION.name(),
            origin,
            Attribute::VELOCITY.name(),
            accel,
        );

        context.update_code += &format!("{}(&particle);\n", func_name);

        Ok(())
    }
}

/// A modifier to apply a tangential acceleration to all particles each frame.
///
/// The acceleration is the same for all particles of the effect, and is applied
/// each frame to modify the particle's velocity based on the simulation
/// timestep.
///
/// ```txt
/// particle.velocity += acceleration * simulation.delta_time;
/// ```
///
/// In the absence of other modifiers, the tangential acceleration alone makes
/// particles rotate around the center point. The tangent direction is
/// calculated as the cross product of the rotation plane axis and the direction
/// from the modifier origin to the particle position. The effect is undefined
/// if those two directions align.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct TangentAccelModifier {
    /// The center point the tangent direction is calculated from.
    ///
    /// Expression type: `Vec3`
    origin: ExprHandle,
    /// The axis defining the rotation plane and orientation.
    ///
    /// Expression type: `Vec3`
    axis: ExprHandle,
    /// The acceleration to apply to all particles in the effect each frame.
    ///
    /// Expression type: `f32`
    accel: ExprHandle,
}

impl TangentAccelModifier {
    /// Create a new modifier from origin and axis expressions, and an
    /// acceleration expression.
    pub fn new(origin: ExprHandle, axis: ExprHandle, accel: ExprHandle) -> Self {
        Self {
            origin,
            axis,
            accel,
        }
    }

    /// Create a new modifier with an acceleration derived from a property.
    ///
    /// The origin and axis are constant.
    pub fn via_property(
        module: &mut Module,
        origin: Vec3,
        axis: Vec3,
        property_name: impl Into<String>,
    ) -> Self {
        Self {
            origin: module.lit(origin),
            axis: module.lit(axis),
            accel: module.prop(property_name),
        }
    }

    /// Create a new modifier with constant values.
    pub fn constant(module: &mut Module, origin: Vec3, axis: Vec3, acceleration: f32) -> Self {
        Self {
            origin: module.lit(origin),
            axis: module.lit(axis),
            accel: module.lit(acceleration),
        }
    }
}

impl_mod_update!(
    TangentAccelModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

#[typetag::serde]
impl UpdateModifier for TangentAccelModifier {
    fn apply_update(
        &self,
        module: &mut Module,
        context: &mut UpdateContext,
    ) -> Result<(), ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("tangent_accel_{0:016X}", func_id);

        let origin = context.eval(module, self.origin)?;
        let axis = context.eval(module, self.axis)?;
        let accel = context.eval(module, self.accel)?;

        context.update_extra += &format!(
            r##"fn {}(particle: ptr<function, Particle>) {{
    let radial = normalize((*particle).{} - {});
    let tangent = normalize(cross({}, radial));
    (*particle).{} += tangent * (({}) * sim_params.delta_time);
}}
"##,
            func_name,
            Attribute::POSITION.name(),
            origin,
            axis,
            Attribute::VELOCITY.name(),
            accel,
        );

        context.update_code += &format!("{}(&particle);\n", func_name);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{ParticleLayout, PropertyLayout, ToWgslString};

    use super::*;

    #[test]
    fn mod_accel() {
        let mut module = Module::default();
        let accel = Vec3::new(1., 2., 3.);
        let modifier = AccelModifier::constant(&mut module, accel);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut context = UpdateContext::new(&property_layout, &particle_layout);
        assert!(modifier.apply_update(&mut module, &mut context).is_ok());

        assert!(context.update_code.contains(&accel.to_wgsl_string()));
    }

    #[test]
    fn mod_radial_accel() {
        let mut module = Module::default();
        let origin = Vec3::new(-1.2, 5.3, -8.5);
        let accel = 6.;
        let modifier = RadialAccelModifier::constant(&mut module, origin, accel);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut context = UpdateContext::new(&property_layout, &particle_layout);
        assert!(modifier.apply_update(&mut module, &mut context).is_ok());

        // TODO: less weak check...
        assert!(context.update_extra.contains(&accel.to_wgsl_string()));
    }

    #[test]
    fn mod_tangent_accel() {
        let mut module = Module::default();
        let origin = Vec3::new(-1.2, 5.3, -8.5);
        let axis = Vec3::Y;
        let accel = 6.;
        let modifier = TangentAccelModifier::constant(&mut module, origin, axis, accel);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut context = UpdateContext::new(&property_layout, &particle_layout);
        assert!(modifier.apply_update(&mut module, &mut context).is_ok());

        // TODO: less weak check...
        assert!(context.update_extra.contains(&accel.to_wgsl_string()));
    }
}
