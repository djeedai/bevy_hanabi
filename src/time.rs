use bevy::prelude::*;

/// The effect simulation clock.
///
/// This is a specialization of the [`Time`] structure and uses the virtual
/// clock [`Time<Virtual>`](Virtual) as its base.
///
/// The speed of this clock is therefore the product of the speed of the virtual
/// clock and the speed of this clock itself. To change the speed of the entire
/// app (including the effects), use [`Time<Virtual>`](Virtual). To influence
/// only the speed of the effects, use
/// [`Time<EffectSimulation>`](EffectSimulation).
///
/// # Example
///
/// ```
/// # use bevy_hanabi::*;
/// # use bevy::prelude::*;
/// fn my_system(mut time: ResMut<Time<EffectSimulation>>) {
///     // Pause the effects
///     time.pause();
///
///     // Unpause the effects
///     time.unpause();
///
///     // Set the speed to 2.0
///     time.set_relative_speed(2.0);
/// }
/// ```
#[derive(Debug, Copy, Clone, Reflect)]
pub struct EffectSimulation {
    paused: bool,
    relative_speed: f64,
    effective_speed: f64,
}

impl Default for EffectSimulation {
    fn default() -> Self {
        Self {
            paused: false,
            relative_speed: 1.0,
            effective_speed: 1.0,
        }
    }
}

/// All methods for the [`Time<EffectSimulation>`](EffectSimulation) clock.
pub trait EffectSimulationTime {
    /// Returns the speed the clock advances relative to the virtual clock, as
    /// [`f32`].
    fn relative_speed(&self) -> f32;

    /// Returns the speed the clock advances relative to the virtual clock, as
    /// [`f64`].
    fn relative_speed_f64(&self) -> f64;

    /// Returns the speed the clock advanced relative to your system clock in
    /// this update, as [`f32`].
    ///
    /// Returns `0.0` if either the [`Time<Virtual>`](Virtual) or the
    /// [`Time<EffectSimulationTime>`](EffectSimulationTime) was paused
    /// and otherwise the product of the `relative_speed` of the clocks at the
    /// start of the update.
    fn effective_speed(&self) -> f32;

    /// Returns the speed the clock advanced relative to your system clock in
    /// this update, as [`f64`].
    ///
    /// Returns `0.0` if either the [`Time<Virtual>`](Virtual) or the
    /// [`Time<EffectSimulationTime>`](EffectSimulationTime) was paused
    /// and otherwise the product of the `relative_speed` of the clocks at the
    /// start of the update.
    fn effective_speed_f64(&self) -> f64;

    /// Sets the speed the clock advances relative to the virtual clock, given
    /// as an [`f32`].
    ///
    /// For example, setting this to `2.0` will make the clock advance twice as
    /// fast as the virtual clock.
    ///
    /// # Panics
    ///
    /// Panics if `ratio` is negative or not finite.
    fn set_relative_speed(&mut self, ratio: f32);

    /// Sets the speed the clock advances relative to the virtual clock, given
    /// as an [`f64`].
    ///
    /// For example, setting this to `2.0` will make the clock advance twice as
    /// fast as the virtual clock.
    ///
    /// # Panics
    ///
    /// Panics if `ratio` is negative or not finite.
    fn set_relative_speed_f64(&mut self, ratio: f64);

    /// Stops the clock, preventing it from advancing until resumed.
    fn pause(&mut self);

    /// Resumes the clock if paused.
    fn unpause(&mut self);

    /// Returns `true` if the clock is currently paused.
    fn is_paused(&self) -> bool;

    /// Returns `true` if the clock was paused at the start of this update.
    fn was_paused(&self) -> bool;
}

impl EffectSimulationTime for Time<EffectSimulation> {
    #[inline]
    fn relative_speed(&self) -> f32 {
        self.relative_speed_f64() as f32
    }

    #[inline]
    fn relative_speed_f64(&self) -> f64 {
        self.context().relative_speed
    }

    #[inline]
    fn effective_speed(&self) -> f32 {
        self.context().effective_speed as f32
    }

    #[inline]
    fn effective_speed_f64(&self) -> f64 {
        self.context().effective_speed
    }

    #[inline]
    fn set_relative_speed(&mut self, ratio: f32) {
        self.set_relative_speed_f64(ratio as f64);
    }

    #[inline]
    fn set_relative_speed_f64(&mut self, ratio: f64) {
        assert!(ratio.is_finite(), "tried to go infinitely fast");
        assert!(ratio >= 0.0, "tried to go back in time");
        self.context_mut().relative_speed = ratio;
    }

    #[inline]
    fn pause(&mut self) {
        self.context_mut().paused = true;
    }

    #[inline]
    fn unpause(&mut self) {
        self.context_mut().paused = false;
    }

    #[inline]
    fn is_paused(&self) -> bool {
        self.context().paused
    }

    #[inline]
    fn was_paused(&self) -> bool {
        self.context().effective_speed == 0.0
    }
}

pub(crate) fn effect_simulation_time_system(
    virt: Res<Time<Virtual>>,
    mut effect_simulation: ResMut<Time<EffectSimulation>>,
) {
    let virt_delta = virt.delta();
    let effective_speed = if effect_simulation.context().paused {
        0.0
    } else {
        effect_simulation.context().relative_speed
    };
    let delta = if effective_speed != 1.0 {
        virt_delta.mul_f64(effective_speed)
    } else {
        // avoid rounding when at normal speed
        virt_delta
    };
    effect_simulation.context_mut().effective_speed = effective_speed * virt.effective_speed_f64();
    effect_simulation.advance_by(delta);
}

#[cfg(test)]
mod tests {
    use std::{thread::sleep, time::Duration};

    use bevy::time::{time_system, TimePlugin, TimeSystem};

    use super::*;

    fn make_test_app() -> App {
        let mut app = App::new();

        app.add_plugins(TimePlugin);
        app.init_resource::<Time<EffectSimulation>>();
        app.add_systems(
            First,
            effect_simulation_time_system
                .after(time_system)
                .in_set(TimeSystem),
        );

        app
    }

    #[test]
    #[allow(clippy::suboptimal_flops)]
    fn test_effect_simulation_time() {
        // This is only used for floating point comparisons, inaccurate sleep times are
        // always fine, as we only test for the relative values between the
        // clocks.
        const EPSILON: f32 = 0.000001;

        let mut app = make_test_app();
        app.update();

        // Update with default speed
        sleep(Duration::from_millis(1));
        app.update();
        let real = app.world().resource::<Time<Real>>();
        let virt = app.world().resource::<Time<Virtual>>();
        let effect_simulation = app.world().resource::<Time<EffectSimulation>>();
        assert!(f32::abs(virt.delta_secs() - real.delta_secs()) < EPSILON);
        assert!(f32::abs(effect_simulation.delta_secs() - real.delta_secs()) < EPSILON);

        // Update with virtual speed 2.0
        app.world_mut()
            .resource_mut::<Time<Virtual>>()
            .set_relative_speed(2.0);
        sleep(Duration::from_millis(1));
        app.update();
        let real = app.world().resource::<Time<Real>>();
        let virt = app.world().resource::<Time<Virtual>>();
        let effect_simulation = app.world().resource::<Time<EffectSimulation>>();
        assert!(f32::abs(virt.delta_secs() - 2.0 * real.delta_secs()) < EPSILON);
        assert!(f32::abs(effect_simulation.delta_secs() - 2.0 * real.delta_secs()) < EPSILON);
        assert!(f32::abs(virt.effective_speed() - 2.0) < EPSILON);
        assert!(f32::abs(effect_simulation.effective_speed() - 2.0) < EPSILON);

        // Update with virtual speed 2.0 and effect speed 3.0
        app.world_mut()
            .resource_mut::<Time<EffectSimulation>>()
            .set_relative_speed(3.0);
        sleep(Duration::from_millis(1));
        app.update();
        let real = app.world().resource::<Time<Real>>();
        let virt = app.world().resource::<Time<Virtual>>();
        let effect_simulation = app.world().resource::<Time<EffectSimulation>>();
        assert!(f32::abs(virt.delta_secs() - 2.0 * real.delta_secs()) < EPSILON);
        assert!(f32::abs(effect_simulation.delta_secs() - 6.0 * real.delta_secs()) < EPSILON);
        assert!(f32::abs(virt.effective_speed() - 2.0) < EPSILON);
        assert!(f32::abs(effect_simulation.effective_speed() - 6.0) < EPSILON);
    }
}
