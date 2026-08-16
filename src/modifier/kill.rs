//! Modifiers to kill particles under specific conditions.
//!
//! These modifiers control the despawning (killing) of particles meeting
//! specific conditions, like entering or leaving an area in space.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    calc_func_id,
    graph::{EvalContext, ExprError},
    Attribute, BoxedModifier, ExprHandle, Modifier, ModifierContext, Module, ShaderWriter,
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

impl Modifier for KillSphereModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Update
    }

    fn attributes(&self) -> &[Attribute] {
        &[Attribute::POSITION]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn apply(&self, module: &mut Module, context: &mut ShaderWriter) -> Result<(), ExprError> {
        let pos = module.attr(Attribute::POSITION);
        let diff = module.sub(pos, self.center);
        let sqr_dist = module.dot(diff, diff);
        let cmp = if self.kill_inside {
            module.lt(sqr_dist, self.sqr_radius)
        } else {
            module.gt(sqr_dist, self.sqr_radius)
        };
        let expr = context.eval(module, cmp)?;

        context.main_code += &format!(
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

impl Modifier for KillAabbModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Update
    }

    fn attributes(&self) -> &[Attribute] {
        &[Attribute::POSITION]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn apply(&self, module: &mut Module, context: &mut ShaderWriter) -> Result<(), ExprError> {
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

        context.main_code += &format!(
            r#"if ({}) {{
    is_alive = false;
}}
"#,
            expr
        );

        Ok(())
    }
}

/// A modifier killing all particles that exit the camera's frustum.
///
/// This ensures camera moving off-screen (not visible) are immediately killed,
/// to avoid simulating them for nothing. The camera frustum is read from the
/// (unique) camera tagged with the [`HanabiMainCamera`] component; if this
/// component is missing, this modifier does nothing.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
#[derive(Debug, Default, Clone, Copy, Hash, Reflect, Serialize, Deserialize)]
pub struct KillFrustumModifier {
    /// Optional distance threshold (defaults to 0) the particle is allowed to
    /// be at outside the frustum planes before being killed. This ensures that
    /// _e.g._ a shaking camera won't kill particles at the edge of the screen,
    /// before moving back to the area where the particle was, which could
    /// effectively make the user "see" that they disappeared. The value can be
    /// negative, but this will shrink the frustum inside the view, and will
    /// kill particles still visible on screen, so is strongly discouraged.
    pub threshold: Option<ExprHandle>,
}

impl KillFrustumModifier {
    /// Create a new instance of an [`KillFrustumModifier`].
    ///
    /// The created instance has a default `threshold = 0.0` value.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the distance threshold from the camera frustum plances, that is the
    /// distance outside of the screen that particles are allowed to be before
    /// they're killed.
    pub fn with_threshold(mut self, threshold: impl Into<ExprHandle>) -> Self {
        self.threshold = Some(threshold.into());
        self
    }
}

impl Modifier for KillFrustumModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Update
    }

    fn attributes(&self) -> &[Attribute] {
        &[Attribute::POSITION]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn apply(&self, module: &mut Module, context: &mut ShaderWriter) -> Result<(), ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("kill_frustum_{0:016X}", func_id);

        context.make_fn(
            &func_name,
            "particle: ptr<function, Particle>",
            Some("bool"),
            module,
            &mut |m: &mut Module, ctx: &mut dyn EvalContext| -> Result<String, ExprError> {
                let threshold = if let Some(threshold) = self.threshold {
                    ctx.eval(m, threshold)?
                } else {
                    "0.0".to_string()
                };

                Ok(format!(
                    r##"    let p = transform_position_simulation_to_world((*particle).{0}).xyz;
    let threshold = {1};
    for (var i = 0; i < 6; i += 1) {{
        if (dot(sim_params.frustum[i].xyz, p) + sim_params.frustum[i].w + threshold <= 0) {{
            return true;
        }}
    }}
    return false;
"##,
                    Attribute::POSITION.name(),
                    threshold
                ))
            },
        )?;

        context.main_code +=
            &format!("if ({func_name}(&particle)) {{\n    is_alive = false;\n}}\n");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParticleLayout, PropertyLayout, TextureLayout};

    #[test]
    fn mod_kill_aabb() {
        let mut module = Module::default();
        let center = module.lit(Vec3::ZERO);
        let half_size = module.lit(Vec3::ONE);
        let modifier = KillAabbModifier::new(center, half_size);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let texture_layout = TextureLayout::default();
        let mut context = ShaderWriter::new(
            ModifierContext::Update,
            &property_layout,
            &particle_layout,
            &texture_layout,
        );
        assert!(modifier.apply(&mut module, &mut context).is_ok());

        assert!(context.main_code.contains("is_alive = false")); // TODO - less
                                                                 // weak check
    }
}
