//! Modifiers to apply forces to the particles.
//!
//! The forces are applied as accelerations times unit mass, as particles
//! currently do not have a mass.

use std::hash::Hash;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    calc_func_id,
    graph::{BuiltInOperator, EvalContext, ExprError},
    Attribute, BoxedModifier, ExprHandle, Modifier, ModifierContext, Module, ShaderWriter,
};

/// A modifier to apply a force to the particle which makes it conform ("stick")
/// to the surface of a sphere.
///
/// This modifies the [`Attribute::VELOCITY`] of the particle to make it
/// converge to the surface of the sphere, while keeping its tangent velocity
/// component unchanged. This modifier gives best results when it's the last
/// modifier affecting the particle velocity; otherwise subsequent modifiers
/// might interfere and break the correction factor calculated by this modifier.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
///
/// # Functioning
///
/// The modifier applies a correction to the radial component of the particle's
/// velocity, either to draw it toward the surface if it's within the zone of
/// influence of the modifier but too far from the surface, or to make it stick
/// to the surface if it's already in its vicinity.
///
/// ![Conform to sphere](https://raw.githubusercontent.com/djeedai/bevy_hanabi/fa23246e8430e0f29ed0a180d8ea6916b3534789/docs/conform-to-sphere.png)
///
/// When inside the area of influence defined by [`influence_dist`], particles
/// are accelerated by the [`attraction_accel`] alongside the radial direction
/// toward the sphere center. This effectively makes the particle move closer to
/// the sphere surface. The particle's velocity is clamped by
/// [`max_attraction_speed`] to prevent it from accelerating infinitely.
///
/// Once the particle arrives in the vicinity of the sphere, an more precisely
/// inside a "shell" area around the sphere surface, the behavior is tweaked a
/// bit. That shell area is defined as the area at a distance of
/// [`shell_half_thickness`] from the sphere surface. The attraction
/// acceleration in the shell area is increased by a [`sticky_factor`], which
/// prevents fast moving particles from overshooting their target and passing
/// through the sphere surface to the other side, oscillating around the surface
/// with each new correction. The shell area effectively acts as a tolerance
/// area, where particles inside it are considered to be "around" the sphere
/// surface. This prevents numerical instabilities trying to conform exactly to
/// the (infinitely thin) surface while also moving tangentially to it. For this
/// reason, particles inside the shell area are on average located on the
/// surface of the sphere, but may wobble around this average. For this reason,
/// it's best to keep the shell thickness tiny, so this wobbling is not visible.
///
/// Particles outside of the area of influence are entirely unaffected.
///
/// [`influence_dist`]: crate::ConformToSphereModifier::influence_dist
/// [`attraction_accel`]: crate::ConformToSphereModifier::attraction_accel
/// [`max_attraction_speed`]: crate::ConformToSphereModifier::max_attraction_speed
/// [`shell_half_thickness`]: crate::ConformToSphereModifier::shell_half_thickness
/// [`sticky_factor`]: crate::ConformToSphereModifier::sticky_factor
#[derive(Debug, Clone, Copy, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub struct ConformToSphereModifier {
    /// The sphere origin (`Vec3`), in [simulation
    /// space](crate::SimulationSpace).
    ///
    /// This is the point toward which particles are attracted.
    pub origin: ExprHandle,
    /// The sphere radius (`f32`).
    ///
    /// This is the distance from the sphere origin, and defines the surface to
    /// which particles conform ("stick").
    pub radius: ExprHandle,
    /// The influence distance of this attractor (`f32`).
    ///
    /// Particles located at a distance greater than this from the sphere
    /// surface are not affected by this modifier. This effectively limits
    /// the effect to a sphere of influence with a radius equal to the
    /// sphere radius plus this influence distance.
    ///
    /// Note that particles located inside the sphere are always influenced.
    ///
    /// [`radius`]: ConformToSphereModifier::radius
    pub influence_dist: ExprHandle,
    /// Acceleration applied to particles to attract them (`f32`).
    ///
    /// Particles in the area of influence of the effect are accelerated by this
    /// value toward the sphere surface.
    ///
    /// Note that because this is an acceleration, all particles are accelerated
    /// identically. If you want the per-particle mass to have an effect on the
    /// attraction effect, assign to this acceleration a constant force divided
    /// by the per-particle mass. That way heavier particles will accelerate
    /// more slowly, having more inertia.
    pub attraction_accel: ExprHandle,
    /// Maximum speed of attraction toward the sphere surface (`f32`).
    ///
    /// This value clamps the radial speed of particles being attracted toward
    /// the sphere surface. Any particle with a radial velocity larger than this
    /// gets immediately slowed down to this value. This prevents particles from
    /// accelerating infinitely. This effectively simulates some kind of drag
    /// effect.
    pub max_attraction_speed: ExprHandle,
    /// Optional shell half-thickness defining the conforming tolerance (`f32`).
    ///
    /// This is an advanced use parameter. Most users can ignore it. If not
    /// specified (`None`), a default value is assigned which works in most
    /// cases.
    ///
    /// Particles located close to the sphere surface tend to oscillate around
    /// it, due to the nature of the linear correction applied to the particle
    /// velocity, and numerical errors. This can cause numerical instabilities
    /// and visual jitter. To prevent this, the correction factor is smoothed
    /// out toward zero in a shell around the sphere surface. Inside that shell,
    /// the velocity correction factor is reduced, which allows particle to not
    /// strictly conform to the sphere as long as they remain inside this area.
    pub shell_half_thickness: Option<ExprHandle>,
    /// Optional "stickiness" acceleration factor (`f32`).
    ///
    /// This is an advanced use parameter. Most users can ignore it. If not
    /// specified (`None`), a default value is assigned which works in most
    /// cases.
    ///
    /// When particles enter the tolerance shell area, the attraction
    /// acceleration is amplified by this factor to improve the "stickiness" to
    /// the ideal sphere surface. The default value if not specified is `2.0`.
    /// Increasing this value makes the particles instantly "stick" to the
    /// sphere surface, which may be desirable to avoid an "overshoot" effect
    /// when particles approach the conforming sphere surface too fast, but
    /// may also look un-natural depending on the effect.
    pub sticky_factor: Option<ExprHandle>,
}

impl ConformToSphereModifier {
    /// Create a new modifier.
    pub fn new(
        origin: ExprHandle,
        radius: ExprHandle,
        influence_dist: ExprHandle,
        attraction_accel: ExprHandle,
        max_attraction_speed: ExprHandle,
    ) -> Self {
        Self {
            origin,
            radius,
            influence_dist,
            attraction_accel,
            max_attraction_speed,
            shell_half_thickness: None,
            sticky_factor: None,
        }
    }
}

#[cfg_attr(feature = "serde", typetag::serde)]
impl Modifier for ConformToSphereModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Update
    }

    fn attributes(&self) -> &[Attribute] {
        &[Attribute::POSITION, Attribute::VELOCITY]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn apply(&self, module: &mut Module, context: &mut ShaderWriter) -> Result<(), ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("force_field_{0:016X}", func_id);

        context.make_fn(
            &func_name,
            "particle: ptr<function, Particle>",
            module,
            &mut |m: &mut Module, ctx: &mut dyn EvalContext| -> Result<String, ExprError> {
                let origin = ctx.eval(m, self.origin)?;
                let radius = ctx.eval(m, self.radius)?;
                let influence_dist = ctx.eval(m, self.influence_dist)?;
                let shell_half_thickness = if let Some(shell_half_thickness) = self.shell_half_thickness { ctx.eval(m, shell_half_thickness)? } else { "0.1".to_string() };
                let max_attraction_speed = ctx.eval(m, self.max_attraction_speed)?;
                let attraction_accel = ctx.eval(m, self.attraction_accel)?;
                let sticky_factor = if let Some(sticky_factor) = self.sticky_factor { ctx.eval(m, sticky_factor)? } else { "2.0".to_string() };

                let attr_pos = format!("(*particle).{}", Attribute::POSITION.name());
                let attr_vel = format!("(*particle).{}", Attribute::VELOCITY.name());

                Ok(format!(
                    r##"    // Sphere center
    let c = {origin};
    // Sphere radius
    let r = {radius};
    // Distance and direction to origin (sphere center)
    let rel_pos = c - {attr_pos};
    let origin_dist = length(rel_pos);
    let origin_dir = normalize(rel_pos);
    // Signed distance to sphere surface, negative if inside sphere
    let surface_dist = origin_dist - r;
    // Influence distance
    let influence_dist = {influence_dist};
    if (surface_dist > influence_dist) {{
        return;
    }}
    // Current signed radial speed (normal to sphere surface) toward the sphere center, which needs to be
    // corrected to conform to the sphere.
    let cur_radial_speed = dot({attr_vel}, origin_dir);
    // Signed radial speed (toward the sphere center) at which we'd like to move to stick to the surface.
    // This is smoothed out to zero as the distance to the surface gets close to zero, to prevent numerical
    // oscillations around the surface.
    let shell_half_thickness = {shell_half_thickness};
    let shell_factor = smoothstep(0., shell_half_thickness, abs(surface_dist));
    let max_attraction_speed = {max_attraction_speed};
    let max_radial_speed = sign(surface_dist) * shell_factor * max_attraction_speed;
    // Delta radial speed to reach the ideal value
    let delta_speed = max_radial_speed - cur_radial_speed;
    // Conforming delta speed from attraction acceleration
    let attraction_accel = {attraction_accel};
    let sticky_accel = attraction_accel * {sticky_factor};
    let conforming_accel = mix(sticky_accel, attraction_accel, shell_factor);
    let conforming_delta_speed = sim_params.delta_time * conforming_accel;
    // Final impulse clamped by the maximum acceleration speed
    {attr_vel} += sign(delta_speed) * min(abs(delta_speed), conforming_delta_speed) * origin_dir;
"##
                ))
            },
        )?;

        context.main_code += &format!("{}(&particle);\n", func_name);

        Ok(())
    }
}

/// A modifier to apply a linear drag force to all particles each frame. The
/// force slows down the particles without changing their direction.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Clone, Copy, Reflect, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LinearDragModifier {
    /// Drag coefficient. Higher values increase the drag force, and
    /// consequently decrease the particle's speed faster.
    ///
    /// Expression type: `f32`
    pub drag: ExprHandle,
}

impl LinearDragModifier {
    /// Create a new modifier from a drag expression.
    pub fn new(drag: ExprHandle) -> Self {
        Self { drag }
    }

    /// Instantiate a [`LinearDragModifier`] with a constant drag value.
    pub fn constant(module: &mut Module, drag: f32) -> Self {
        Self {
            drag: module.lit(drag),
        }
    }
}

#[cfg_attr(feature = "serde", typetag::serde)]
impl Modifier for LinearDragModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Update
    }

    fn attributes(&self) -> &[Attribute] {
        &[Attribute::VELOCITY]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn apply(&self, module: &mut Module, context: &mut ShaderWriter) -> Result<(), ExprError> {
        let m = module;
        let attr = m.attr(Attribute::VELOCITY);
        let dt = m.builtin(BuiltInOperator::DeltaTime);
        let drag_dt = m.mul(self.drag, dt);
        let one = m.lit(1.);
        let one_minus_drag_dt = m.sub(one, drag_dt);
        let zero = m.lit(0.);
        let expr = m.max(zero, one_minus_drag_dt);
        let attr = context.eval(m, attr)?;
        let expr = context.eval(m, expr)?;
        context.main_code += &format!("{} *= {};", attr, expr);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParticleLayout, PropertyLayout};

    #[test]
    fn mod_drag() {
        let mut module = Module::default();
        let modifier = LinearDragModifier::constant(&mut module, 3.5);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut context =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
        assert!(modifier.apply(&mut module, &mut context).is_ok());

        assert!(context.main_code.contains("3.5")); // TODO - less weak check
    }
}
