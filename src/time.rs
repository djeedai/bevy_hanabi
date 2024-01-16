use bevy::prelude::*;

/// The effect simulation clock.
///
/// This is a specialization of the [`Time`] structure and uses the virtual clock
/// [`Time<Virtual>`](Virtual) as its base.
///
/// The speed of this clock is therefore the product of the speed of the virtual clock and the
/// speed of this clock itself. To change the speed of the entire app (including the effects),
/// use [`Time<Virtual>`](Virtual). To influence only the speed of the effects, use
/// [`Time<EffectSimulation>`](EffectSimulation).
///
/// # Example
///
/// ```
/// # use bevy_hanabi::*;
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

/// All methods for the effect simulation clock.
pub trait EffectSimulationTime {
    /// Returns the speed the clock advances relative to the virtual clock, as [`f32`].
    fn relative_speed(&self) -> f32;

    /// Returns the speed the clock advances relative to the virtual clock, as [`f64`].
    fn relative_speed_f64(&self) -> f64;

    /// Returns the speed the clock advanced relative to the virtual clock in
    /// this update, as [`f32`].
    ///
    /// Returns `0.0` if the game was paused or what the `relative_speed` value
    /// was at the start of this update.
    fn effective_speed(&self) -> f32;

    /// Returns the speed the clock advanced relative to the virtual clock in
    /// this update, as [`f64`].
    ///
    /// Returns `0.0` if the game was paused or what the `relative_speed` value
    /// was at the start of this update.
    fn effective_speed_f64(&self) -> f64;

    /// Sets the speed the clock advances relative to the virtual clock, given as an [`f32`].
    ///
    /// For example, setting this to `2.0` will make the clock advance twice as fast as the virtual
    /// clock.
    ///
    /// # Panics
    ///
    /// Panics if `ratio` is negative or not finite.
    fn set_relative_speed(&mut self, ratio: f32);

    /// Sets the speed the clock advances relative to the virtual clock, given as an [`f64`].
    ///
    /// For example, setting this to `2.0` will make the clock advance twice as fast as the virtual
    /// clock.
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
    effect_simulation.context_mut().effective_speed = effective_speed;
    effect_simulation.advance_by(delta);
}
