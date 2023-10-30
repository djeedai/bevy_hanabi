//! Modifiers to kill particles under specific conditions.
//!
//! These modifiers control the despawning (killing) of particles meeting
//! specific conditions, like entering or leaving an area in space.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    graph::{EvalContext, ExprError},
    impl_mod_update, Attribute, ExprHandle, Module, UpdateContext, UpdateModifier,
};

/// A modifier killing all particles that enter or exit a sphere.
///
/// This enables confining particles to a region in space, or preventing
/// particles to enter that region.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
#[derive(Debug, Clone, Copy, Reflect, Serialize, Deserialize)]
pub struct KillSphereModifier {
    /// Center of the sphere.
    ///
    /// Expression type: `Vec3`
    pub center: ExprHandle,
    /// Squared radius of the sphere.
    ///
    /// This is the sphere radius multiplied with itself. Storing the squared
    /// radius makes it more performant for simulation.
    ///
    /// Expression type: `f32`
    pub sqr_radius: ExprHandle,
    /// If `true`, invert the kill condition and kill all particles inside the
    /// AABB. If `false` (default), kill all particles outside the AABB.
    pub kill_inside: bool,
}

impl KillSphereModifier {
    /// Create a new instance of an [`KillSphereModifier`] from a sphere center
    /// and squared radius.
    ///
    /// The e
    ///
    /// The created instance has a default `kill_inside = false` value.
    pub fn new(center: ExprHandle, sqr_radius: ExprHandle) -> Self {
        Self {
            center,
            sqr_radius,
            kill_inside: false,
        }
    }

    /// Set whether particles are killed when inside the AABB or not.
    pub fn with_kill_inside(mut self, kill_inside: bool) -> Self {
        self.kill_inside = kill_inside;
        self
    }
}

impl_mod_update!(KillSphereModifier, &[Attribute::POSITION]);

#[typetag::serde]
impl UpdateModifier for KillSphereModifier {
    fn apply_update(
        &self,
        module: &mut Module,
        context: &mut UpdateContext,
    ) -> Result<(), ExprError> {
        let pos = module.attr(Attribute::POSITION);
        let diff = module.sub(pos, self.center);
        let sqr_dist = module.dot(diff, diff);
        let cmp = if self.kill_inside {
            module.lt(sqr_dist, self.sqr_radius)
        } else {
            module.gt(sqr_dist, self.sqr_radius)
        };
        let expr = context.eval(module, cmp)?;

        context.update_code += &format!(
            r#"if ({}) {{
    is_alive = false;
}}
"#,
            expr
        );

        Ok(())
    }
}

/// A modifier killing all particles that enter or exit an AABB.
///
/// This enables confining particles to a region in space, or preventing
/// particles to enter that region.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
#[derive(Debug, Clone, Copy, Reflect, Serialize, Deserialize)]
pub struct KillAabbModifier {
    /// Center of the AABB.
    ///
    /// Expression type: `Vec3`
    pub center: ExprHandle,
    /// Half-size of the AABB.
    ///
    /// Expression type: `Vec3`
    pub half_size: ExprHandle,
    /// If `true`, invert the kill condition and kill all particles inside the
    /// AABB. If `false` (default), kill all particles outside the AABB.
    pub kill_inside: bool,
}

impl KillAabbModifier {
    /// Create a new instance of an [`KillAabbModifier`] from an AABB center and
    /// half extents.
    ///
    /// The created instance has a default `kill_inside = false` value.
    pub fn new(center: impl Into<ExprHandle>, half_size: impl Into<ExprHandle>) -> Self {
        Self {
            center: center.into(),
            half_size: half_size.into(),
            kill_inside: false,
        }
    }

    /// Set whether particles are killed when inside the AABB or not.
    pub fn with_kill_inside(mut self, kill_inside: bool) -> Self {
        self.kill_inside = kill_inside;
        self
    }
}

impl_mod_update!(KillAabbModifier, &[Attribute::POSITION]);

#[typetag::serde]
impl UpdateModifier for KillAabbModifier {
    fn apply_update(
        &self,
        module: &mut Module,
        context: &mut UpdateContext,
    ) -> Result<(), ExprError> {
        let pos = module.attr(Attribute::POSITION);
        let diff = module.sub(pos, self.center);
        let dist = module.abs(diff);
        let cmp = if self.kill_inside {
            module.lt(dist, self.half_size)
        } else {
            module.gt(dist, self.half_size)
        };
        let reduce = if self.kill_inside {
            module.all(cmp)
        } else {
            module.any(cmp)
        };
        let expr = context.eval(module, reduce)?;

        context.update_code += &format!(
            r#"if ({}) {{
    is_alive = false;
}}
"#,
            expr
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Module, ParticleLayout, PropertyLayout};

    use super::*;

    #[test]
    fn mod_kill_aabb() {
        let mut module = Module::default();
        let center = module.lit(Vec3::ZERO);
        let half_size = module.lit(Vec3::ONE);
        let modifier = KillAabbModifier::new(center, half_size);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut context = UpdateContext::new(&property_layout, &particle_layout);
        assert!(modifier.apply_update(&mut module, &mut context).is_ok());

        assert!(context.update_code.contains("is_alive = false")); // TODO - less weak check
    }
}
