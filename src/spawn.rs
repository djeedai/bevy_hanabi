use std::hash::{Hash, Hasher};

use bevy::{ecs::resource::Resource, log::trace, math::FloatOrd, prelude::*, reflect::Reflect};
use rand::{
    distr::{uniform::SampleUniform, Distribution, Uniform},
    SeedableRng,
};
use rand_pcg::Pcg32;
use serde::{Deserialize, Serialize};

use crate::{
    CompiledParticleEffect, EffectAsset, EffectSimulation, ParticleEffect, SimulationCondition,
};

/// An RNG to be used in the CPU for the particle system engine
pub(crate) fn new_rng() -> Pcg32 {
    let mut rng = rand::rng();
    let mut seed = [0u8; 16];
    seed.copy_from_slice(
        &Uniform::new_inclusive(0, u128::MAX)
            .unwrap()
            .sample(&mut rng)
            .to_le_bytes(),
    );
    Pcg32::from_seed(seed)
}

/// An RNG resource
#[derive(Resource)]
pub struct Random(pub Pcg32);

/// Utility trait to help implementing [`std::hash::Hash`] for [`CpuValue`] of
/// floating-point type.
pub trait FloatHash: PartialEq {
    fn hash_f32<H: Hasher>(&self, state: &mut H);
}

impl FloatHash for f32 {
    fn hash_f32<H: Hasher>(&self, state: &mut H) {
        FloatOrd(*self).hash(state);
    }
}

impl FloatHash for Vec2 {
    fn hash_f32<H: Hasher>(&self, state: &mut H) {
        FloatOrd(self.x).hash(state);
        FloatOrd(self.y).hash(state);
    }
}

impl FloatHash for Vec3 {
    fn hash_f32<H: Hasher>(&self, state: &mut H) {
        FloatOrd(self.x).hash(state);
        FloatOrd(self.y).hash(state);
        FloatOrd(self.z).hash(state);
    }
}

impl FloatHash for Vec4 {
    fn hash_f32<H: Hasher>(&self, state: &mut H) {
        FloatOrd(self.x).hash(state);
        FloatOrd(self.y).hash(state);
        FloatOrd(self.z).hash(state);
        FloatOrd(self.w).hash(state);
    }
}

/// A constant or random value evaluated on CPU.
///
/// This enum represents a value which is either constant, or randomly sampled
/// according to a given probability distribution.
///
/// Not to be confused with [`graph::Value`]. This [`CpuValue`] is a legacy type
/// that will be eventually replaced with a [`graph::Value`] once evaluation of
/// the latter can be emulated on CPU, which is required for use
/// with the [`SpawnerSettings`].
///
/// [`graph::Value`]: crate::graph::Value
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Reflect)]
#[non_exhaustive]
pub enum CpuValue<T: Copy + FromReflect> {
    /// Single constant value.
    Single(T),
    /// Random value distributed uniformly between two inclusive bounds.
    ///
    /// The minimum bound must be less than or equal to the maximum one,
    /// otherwise some methods like [`sample()`] will panic.
    ///
    /// [`sample()`]: crate::CpuValue::sample
    Uniform((T, T)),
}

impl<T: Copy + FromReflect + Default> Default for CpuValue<T> {
    fn default() -> Self {
        Self::Single(T::default())
    }
}

impl<T: Copy + FromReflect + SampleUniform> CpuValue<T> {
    /// Sample the value.
    /// - For [`CpuValue::Single`], always return the same single value.
    /// - For [`CpuValue::Uniform`], use the given pseudo-random number
    ///   generator to generate a random sample.
    pub fn sample(&self, rng: &mut Pcg32) -> T {
        match self {
            Self::Single(x) => *x,
            Self::Uniform((a, b)) => Uniform::new_inclusive(*a, *b).unwrap().sample(rng),
        }
    }
}

impl<T: Copy + FromReflect + PartialOrd> CpuValue<T> {
    /// Returns the range of allowable values in the form `[minimum, maximum]`.
    /// For [`CpuValue::Single`], both values are the same.
    pub fn range(&self) -> [T; 2] {
        match self {
            Self::Single(x) => [*x; 2],
            Self::Uniform((a, b)) => {
                if a <= b {
                    [*a, *b]
                } else {
                    [*b, *a]
                }
            }
        }
    }
}

impl<T: Copy + FromReflect> From<T> for CpuValue<T> {
    fn from(t: T) -> Self {
        Self::Single(t)
    }
}

impl<T: Copy + FromReflect> From<[T; 2]> for CpuValue<T> {
    fn from(t: [T; 2]) -> Self {
        Self::Uniform((t[0], t[1]))
    }
}

impl<T: Copy + FromReflect> From<(T, T)> for CpuValue<T> {
    fn from(t: (T, T)) -> Self {
        Self::Uniform(t)
    }
}

impl<T: Copy + FromReflect + FloatHash> Eq for CpuValue<T> {}

impl<T: Copy + FromReflect + FloatHash> Hash for CpuValue<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            CpuValue::Single(f) => {
                1_u8.hash(state);
                f.hash_f32(state);
            }
            CpuValue::Uniform((a, b)) => {
                2_u8.hash(state);
                a.hash_f32(state);
                b.hash_f32(state);
            }
        }
    }
}

/// Settings for an [`EffectSpawner`].
///
/// A [`SpawnerSettings`] represents the settings of an [`EffectSpawner`].
///
/// The spawning logic is based around the concept of _cycles_. A spawner
/// defines a pattern of particle spawning as the repetition of a number of unit
/// cycles. Each cycle is composed of a period of emission, followed by a period
/// of rest (idling). Both periods can be of zero duration.
///
/// The settings are:
///
/// - `count` is the number of particles to spawn over a single cycle, during
///   the emission period. It can evaluate to negative or zero random values, in
///   which case no particle is spawned during the current cycle.
/// - `spawn_duration` is the duration of the spawn part of a single cycle. If
///   this is <= 0, then the particles spawn all at once exactly at the same
///   instant at the beginning of the cycle.
/// - `period` is the period of a cycle. If this is <= `spawn_duration`, then
///   the spawner spawns a steady stream of particles. This is ignored if
///   `cycle_count == 1`.
/// - `cycle_count` is the number of cycles, that is the number of times this
///   spawn-rest pattern occurs. Set this to `0` to repeat forever.
///
/// ```txt
///  <----------- period ----------->
///  <- spawn_duration ->
/// |********************|-----------|
///      spawn 'count'        wait
///        particles
/// ```
///
/// Most settings are stored as a [`CpuValue`] to allow some randomizing. If
/// using a random distribution, the value is resampled each cycle.
///
/// Note that the "burst" semantic here doesn't strictly mean a one-off
/// emission, since that emission is spread over a number of simulation
/// frames that total a duration of `spawn_duration`. If you want a strict
/// single-frame burst, simply set the `spawn_duration` to zero; this is
/// what [`once()`] does.
///
/// [`once()`]: Self::once
#[derive(Debug, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect(Default)]
pub struct SpawnerSettings {
    /// Number of particles to spawn over [`spawn_duration`].
    ///
    /// [`spawn_duration`]: Self::spawn_duration
    count: CpuValue<f32>,

    /// Time over which to spawn [`count`], in seconds.
    ///
    /// [`count`]: Self::count
    spawn_duration: CpuValue<f32>,

    /// Time between bursts of the particle system, in seconds.
    ///
    /// If this is [`spawn_duration`] or less, the system spawns a steady stream
    /// of particles.
    ///
    /// [`spawn_duration`]: Self::spawn_duration
    period: CpuValue<f32>,

    /// Number of cycles the spawner is active before completing.
    ///
    /// Each cycle lasts for `period`. A value of `0` means "infinite", that is
    /// the spanwe emits particle forever as long as it's active.
    cycle_count: u32,

    /// Whether the [`EffectSpawner`] is active at startup.
    ///
    /// The value is used to initialize [`EffectSpawner::active`].
    ///
    /// [`EffectSpawner::active`]: crate::EffectSpawner::active
    starts_active: bool,

    /// Whether the [`EffectSpawner`] immediately starts emitting particles.
    emit_on_start: bool,
}

impl Default for SpawnerSettings {
    fn default() -> Self {
        Self::once(1.0f32.into())
    }
}

impl SpawnerSettings {
    /// Create settings from individual values.
    ///
    /// This is the _raw_ constructor. In general you should prefer using one of
    /// the utility constructors [`once()`], [`burst()`], or [`rate()`],
    /// which will ensure the control parameters are set consistently relative
    /// to each other.
    ///
    /// # Panics
    ///
    /// Panics if `period` can produce a negative number (the sample range lower
    /// bound is negative), unless the cycle count is exactly 1, in which case
    /// `period` is ignored.
    ///
    /// Panics if `period` can only produce 0 (the sample range upper bound
    /// is not strictly positive), unless the cycle count is exactly 1, in which
    /// case `period` is ignored.
    ///
    /// Panics if any value is infinite.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::SpawnerSettings;
    /// // Spawn 32 particles over 3 seconds, then pause for 7 seconds (10 - 3),
    /// // doing that 5 times in total.
    /// let spawner = SpawnerSettings::new(32.0.into(), 3.0.into(), 10.0.into(), 5);
    /// ```
    ///
    /// [`once()`]: Self::once
    /// [`burst()`]: Self::burst
    /// [`rate()`]: Self::rate
    pub fn new(
        count: CpuValue<f32>,
        spawn_duration: CpuValue<f32>,
        period: CpuValue<f32>,
        cycle_count: u32,
    ) -> Self {
        assert!(
            cycle_count == 1 || period.range()[0] >= 0.,
            "`period` must not generate negative numbers (period.min was {}, expected >= 0).",
            period.range()[0]
        );
        assert!(
            cycle_count == 1 || period.range()[1] > 0.,
            "`period` must be able to generate a positive number (period.max was {}, expected > 0).",
            period.range()[1]
        );
        assert!(
            period.range()[0].is_finite() && period.range()[1].is_finite(),
            "`period` {:?} has an infinite bound. If upgrading from a previous version, use `cycle_count = 1` instead for a single-cycle burst.",
            period
        );

        Self {
            count,
            spawn_duration,
            period,
            cycle_count,
            starts_active: true,
            emit_on_start: true,
        }
    }

    /// Set whether the [`EffectSpawner`] immediately starts to emit particle
    /// when the [`ParticleEffect`] is spawned into the ECS world.
    ///
    /// If set to `false`, then [`EffectSpawner::has_completed()`] will return
    /// `true` after spawning the component, and the spawner needs to be
    /// [`EffectSpawner::reset()`] before it can spawn particles. This is
    /// useful to spawn a particle effect instance immediately, but only start
    /// emitting particles when an event occurs (collision, user input, any
    /// other game logic...).
    ///
    /// Because a spawner repeating forever never completes, this has no effect
    /// if [`is_forever()`] is `true`. To start/stop spawning with those
    /// effects, use [`EffectSpawner::active`] instead.
    ///
    /// [`is_forever()`]: Self::is_forever
    pub fn with_emit_on_start(mut self, emit_on_start: bool) -> Self {
        self.emit_on_start = emit_on_start;
        self
    }

    /// Create settings to spawn a burst of particles once.
    ///
    /// The burst of particles is spawned all at once in the same frame. After
    /// that, the spawner idles, waiting to be manually reset via
    /// [`EffectSpawner::reset()`].
    ///
    /// This is a convenience for:
    ///
    /// ```
    /// # use bevy_hanabi::{SpawnerSettings, CpuValue};
    /// # let count = CpuValue::Single(1.);
    /// SpawnerSettings::new(count, 0.0.into(), 0.0.into(), 1);
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::SpawnerSettings;
    /// // Spawn 32 particles in a burst once immediately on creation.
    /// let spawner = SpawnerSettings::once(32.0.into());
    /// ```
    pub fn once(count: CpuValue<f32>) -> Self {
        Self::new(count, 0.0.into(), 0.0.into(), 1)
    }

    /// Get whether the spawner has a single cycle.
    ///
    /// This is true if the cycle count is exactly equal to 1.
    pub fn is_once(&self) -> bool {
        self.cycle_count == 1
    }

    /// Get whether the spawner has an infinite number of cycles.
    ///
    /// This is true if the cycle count is exactly equal to 0.
    pub fn is_forever(&self) -> bool {
        self.cycle_count == 0
    }

    /// Create settings to spawn a continuous stream of particles.
    ///
    /// The particle spawn `rate` is expressed in particles per second.
    /// Fractional values are accumulated each frame.
    ///
    /// This is a convenience for:
    ///
    /// ```
    /// # use bevy_hanabi::{SpawnerSettings, CpuValue};
    /// # let rate = CpuValue::Single(1.);
    /// SpawnerSettings::new(rate, 1.0.into(), 1.0.into(), 0);
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::SpawnerSettings;
    /// // Spawn 10 particles per second, indefinitely.
    /// let spawner = SpawnerSettings::rate(10.0.into());
    /// ```
    pub fn rate(rate: CpuValue<f32>) -> Self {
        Self::new(rate, 1.0.into(), 1.0.into(), 0)
    }

    /// Create settings to spawn particles in bursts.
    ///
    /// The settings define an infinite number of cycles where `count` particles
    /// are spawned at the beginning of the cycle, then the spawner waits
    /// `period` seconds, and repeats forever.
    ///
    /// This is a convenience for:
    ///
    /// ```
    /// # use bevy_hanabi::{SpawnerSettings, CpuValue};
    /// # let count = CpuValue::Single(1.);
    /// # let period = CpuValue::Single(1.);
    /// SpawnerSettings::new(count, 0.0.into(), period, 0);
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::SpawnerSettings;
    /// // Spawn a burst of 5 particles every 3 seconds, indefinitely.
    /// let spawner = SpawnerSettings::burst(5.0.into(), 3.0.into());
    /// ```
    pub fn burst(count: CpuValue<f32>, period: CpuValue<f32>) -> Self {
        Self::new(count, 0.0.into(), period, 0)
    }

    /// Set the number of particles that are spawned each cycle.
    pub fn with_count(mut self, count: CpuValue<f32>) -> Self {
        self.count = count;
        self
    }

    /// Set the number of particles that are spawned each cycle.
    pub fn set_count(&mut self, count: CpuValue<f32>) {
        self.count = count;
    }

    /// Get the number of particles that are spawned each cycle.
    pub fn count(&self) -> CpuValue<f32> {
        self.count
    }

    /// Set the duration, in seconds, of the spawn part each cycle.
    pub fn with_spawn_duration(mut self, spawn_duration: CpuValue<f32>) -> Self {
        self.spawn_duration = spawn_duration;
        self
    }

    /// Set the duration, in seconds, of the spawn part each cycle.
    pub fn set_spawn_duration(&mut self, spawn_duration: CpuValue<f32>) {
        self.spawn_duration = spawn_duration;
    }

    /// Get the duration, in seconds, of the spawn part each cycle.
    pub fn spawn_duration(&self) -> CpuValue<f32> {
        self.spawn_duration
    }

    /// Set the duration of a single spawn cycle, in seconds.
    ///
    /// A spawn cycle includes the [`spawn_duration()`] value, and any extra
    /// wait time (if larger than spawn time).
    ///
    /// # Panics
    ///
    /// Panics if the period is infinite.
    ///
    /// [`spawn_duration()`]: Self::spawn_duration
    pub fn with_period(mut self, period: CpuValue<f32>) -> Self {
        assert!(
            period.range()[0].is_finite() && period.range()[1].is_finite(),
            "`period` {:?} has an infinite bound. If upgrading from a previous version, use `cycle_count = 1` instead for a single-cycle burst.",
            period
        );
        self.period = period;
        self
    }

    /// Set the duration of a single spawn cycle, in seconds.
    ///
    /// A spawn cycle includes the [`spawn_duration()`] value, and any extra
    /// wait time (if larger than spawn time).
    ///
    /// # Panics
    ///
    /// Panics if the period is infinite.
    ///
    /// [`spawn_duration()`]: Self::spawn_duration
    pub fn set_period(&mut self, period: CpuValue<f32>) {
        assert!(
            period.range()[0].is_finite() && period.range()[1].is_finite(),
            "`period` {:?} has an infinite bound. If upgrading from a previous version, use `cycle_count = 1` instead for a single-cycle burst.",
            period
        );
        self.period = period;
    }

    /// Get the duration of a single spawn cycle, in seconds.
    ///
    /// A spawn cycle includes the [`spawn_duration()`] value, and any extra
    /// wait time (if larger than spawn time).
    ///
    /// [`spawn_duration()`]: Self::spawn_duration
    pub fn period(&self) -> CpuValue<f32> {
        self.period
    }

    /// Set the number of cycles to spawn for.
    ///
    /// A spawn cycle includes the [`spawn_duration()`] value, and any extra
    /// wait time (if larger than spawn time). It lasts for [`period()`].
    ///
    /// [`spawn_duration()`]: Self::spawn_duration
    /// [`period()`]: Self::period
    pub fn with_cycle_count(mut self, cycle_count: u32) -> Self {
        self.cycle_count = cycle_count;
        self
    }

    /// Set the number of cycles to spawn for.
    ///
    /// A spawn cycle includes the [`spawn_duration()`] value, and any extra
    /// wait time (if larger than spawn time). It lasts for [`period()`].
    ///
    /// [`spawn_duration()`]: Self::spawn_duration
    /// [`period()`]: Self::period
    pub fn set_cycle_count(&mut self, cycle_count: u32) {
        self.cycle_count = cycle_count;
    }

    /// Get the number of cycles to spawn for.
    ///
    /// A spawn cycle includes the [`spawn_duration()`] value, and any extra
    /// wait time (if larger than spawn time). It lasts for [`period()`].
    ///
    /// [`spawn_duration()`]: Self::spawn_duration
    /// [`period()`]: Self::period
    pub fn cycle_count(&self) -> u32 {
        self.cycle_count
    }

    /// Sets whether the spawner starts active when the effect is instantiated.
    ///
    /// This value will be transfered to the active state of the
    /// [`EffectSpawner`] once it's instantiated. Inactive spawners do not spawn
    /// any particle.
    pub fn with_starts_active(mut self, starts_active: bool) -> Self {
        self.starts_active = starts_active;
        self
    }

    /// Set whether the spawner starts active when the effect is instantiated.
    ///
    /// This value will be transfered to the active state of the
    /// [`EffectSpawner`] once it's instantiated. Inactive spawners do not spawn
    /// any particle.
    pub fn set_starts_active(&mut self, starts_active: bool) {
        self.starts_active = starts_active;
    }

    /// Get whether the spawner starts active when the effect is instantiated.
    ///
    /// This value will be transfered to the active state of the
    /// [`EffectSpawner`] once it's instantiated. Inactive spawners do not spawn
    /// any particle.
    pub fn starts_active(&self) -> bool {
        self.starts_active
    }
}

/// Runtime state machine for CPU particle spawning.
///
/// The spawner defines how new particles are emitted and when. Each time the
/// spawner ticks, it calculates a number of particles to emit for this frame,
/// based on its [`SpawnerSettings`]. This spawn count is passed to the GPU for
/// the init compute pass to actually allocate the new particles and initialize
/// them. The number of particles to spawn is stored as a floating-point number,
/// and any remainder accumulates for the next tick.
///
/// Spawners are used to control from CPU when particles are spawned. To use GPU
/// spawn events instead, and spawn particles based on events occurring on
/// existing particles in other effects, see [`EffectParent`]. Those two
/// mechanisms (CPU and GPU spawner) are mutually exclusive.
///
/// Once per frame the [`tick_spawners()`] system will add the [`EffectSpawner`]
/// component if it's missing, cloning the [`SpawnerSettings`] from the source
/// [`EffectAsset`] to initialize it. After that, it ticks the
/// [`SpawnerSettings`] stored in the component. The resulting number of
/// particles to spawn for the frame is then stored into
/// [`EffectSpawner::spawn_count`]. You can override that value to manually
/// control each frame how many particles are spawned, instead of using the
/// logic of [`SpawnerSettings`].
///
/// [`EffectParent`]: crate::EffectParent
#[derive(Debug, Default, Clone, Copy, PartialEq, Component, Reflect)]
#[reflect(Component)]
pub struct EffectSpawner {
    /// The spawner settings extracted from the [`EffectAsset`], or directly
    /// overriden by the user.
    pub settings: SpawnerSettings,

    /// Accumulated time for the current (partial) cycle, in seconds.
    cycle_time: f32,

    /// Number of cycles already completed.
    completed_cycle_count: u32,

    /// Sampled value of `spawn_duration` until `period` is reached. This is the
    /// duration of the "active" period during which we spawn particles, as
    /// opposed to the "wait" period during which we do nothing until the next
    /// spawn cycle.
    sampled_spawn_duration: f32,

    /// Sampled value of the time period, in seconds, until the next spawn
    /// cycle.
    sampled_period: f32,

    /// Sampled value of the number of particles to spawn per `spawn_duration`.
    sampled_count: f32,

    /// Number of particles to spawn this frame.
    ///
    /// This value is normally updated by calling [`tick()`], which
    /// automatically happens once per frame when the [`tick_spawners()`]
    /// system runs in the [`PostUpdate`] schedule.
    ///
    /// You can manually assign this value to override the one calculated by
    /// [`tick()`]. Note in this case that you need to override the value after
    /// the automated one was calculated, by ordering your system
    /// after [`tick_spawners()`] or [`EffectSystems::TickSpawners`].
    ///
    /// [`tick()`]: crate::EffectSpawner::tick
    /// [`EffectSystems::TickSpawners`]: crate::EffectSystems::TickSpawners
    pub spawn_count: u32,

    /// Fractional remainder of particle count to spawn.
    ///
    /// This is accumulated each tick, and the integral part is added to
    /// `spawn_count`. The reminder gets saved for next frame.
    spawn_remainder: f32,

    /// Whether the spawner is active. Defaults to
    /// [`SpawnerSettings::starts_active()`]. An inactive spawner
    /// doesn't tick (no particle spawned, no internal state updated).
    pub active: bool,
}

impl EffectSpawner {
    /// Create a new spawner.
    pub fn new(settings: &SpawnerSettings) -> Self {
        Self {
            settings: *settings,
            cycle_time: 0.,
            completed_cycle_count: if settings.emit_on_start || settings.is_forever() {
                // Infinitely repeating effects always start at cycle #0.
                0
            } else {
                // Start at last cycle. This means has_completed() is true.
                settings.cycle_count()
            },
            sampled_spawn_duration: 0.,
            sampled_period: 0.,
            sampled_count: 0.,
            spawn_count: 0,
            spawn_remainder: 0.,
            active: settings.starts_active(),
        }
    }

    /// Set whether the spawner is active.
    ///
    /// Inactive spawners do not tick, and therefore do not spawn any particle.
    /// Their internal state do not update.
    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Get the time relative to the beginning of the current cycle.
    #[inline]
    pub fn cycle_time(&self) -> f32 {
        self.cycle_time
    }

    /// Get the spawn duration for the current cycle.
    ///
    /// This value can change every cycle if [`SpawnerSettings::spawn_duration`]
    /// is a randomly distributed value.
    #[inline]
    pub fn cycle_spawn_duration(&self) -> f32 {
        self.sampled_spawn_duration
    }

    /// Get the period of the current cycle.
    ///
    /// This value can change every cycle if [`SpawnerSettings::period`] is a
    /// randomly distributed value. If the effect spawns only once, and
    /// therefore its cycle period is ignored, this returns `0`.
    #[inline]
    pub fn cycle_period(&self) -> f32 {
        if self.settings.is_once() {
            0.
        } else {
            self.sampled_period
        }
    }

    /// Get the progress ratio in 0..1 of the current cycle.
    ///
    /// This is the ratio of the [`cycle_time()`] over [`cycle_period()`]. If
    /// the effect spawns only once, and therefore its cycle period is
    /// ignored, this returns `0`.
    ///
    /// [`cycle_time()`]: Self::cycle_time
    /// [`cycle_period()`]: Self::cycle_period
    #[inline]
    pub fn cycle_ratio(&self) -> f32 {
        if self.settings.is_once() {
            0.
        } else {
            self.cycle_time / self.sampled_period
        }
    }

    /// Get the number of particles to spawn during the current cycle
    ///
    /// This value can change every cycle if [`SpawnerSettings::count`] is a
    /// randomly distributed value.
    #[inline]
    pub fn cycle_spawn_count(&self) -> f32 {
        self.sampled_count
    }

    /// Get the number of completed cycles since last [`reset()`].
    ///
    /// The value loops back if the pattern repeats forever
    /// ([`SpawnerSettings::is_forever()`] is `true`).
    ///
    /// [`reset()`]: Self::reset
    #[inline]
    pub fn completed_cycle_count(&self) -> u32 {
        self.completed_cycle_count
    }

    /// Get whether the spawner has completed.
    ///
    /// A spawner has completed if it already ticked through its maximum number
    /// of cycles. It can be reset back to its original state with [`reset()`].
    /// A spawner repeating forever never completes.
    ///
    /// [`reset()`]: Self::reset
    #[inline]
    pub fn has_completed(&self) -> bool {
        !self.settings.is_forever() && (self.completed_cycle_count >= self.settings.cycle_count())
    }

    /// Reset the spawner state.
    ///
    /// This resets the internal spawner time and cycle count to zero.
    ///
    /// Use this, for example, to immediately spawn some particles in a spawner
    /// constructed with [`SpawnerSettings::once`].
    ///
    /// [`SpawnerSettings::once`]: crate::SpawnerSettings::once
    pub fn reset(&mut self) {
        self.cycle_time = 0.;
        self.completed_cycle_count = 0;
        self.sampled_spawn_duration = 0.;
        self.sampled_period = 0.;
        self.sampled_count = 0.;
        self.spawn_count = 0;
        self.spawn_remainder = 0.;
    }

    /// Tick the spawner to calculate the number of particles to spawn this
    /// frame.
    ///
    /// The frame delta time `dt` is added to the current spawner time, before
    /// the spawner calculates the number of particles to spawn.
    ///
    /// This method is called automatically by [`tick_spawners()`] during the
    /// [`PostUpdate`], so you normally don't have to call it yourself
    /// manually.
    ///
    /// # Returns
    ///
    /// The integral number of particles to spawn this frame. Any fractional
    /// remainder is saved for the next call.
    pub fn tick(&mut self, mut dt: f32, rng: &mut Pcg32) -> u32 {
        // If inactive, or if the finite number of cycles has been completed, then we're
        // done.
        if !self.active
            || (!self.settings.is_forever()
                && (self.completed_cycle_count >= self.settings.cycle_count()))
        {
            self.spawn_count = 0;
            return 0;
        }

        // Use a loop in case the timestep dt spans multiple cycles
        loop {
            // Check if this is a new cycle which needs resampling
            if self.sampled_period == 0.0 {
                if self.settings.is_once() {
                    self.sampled_spawn_duration = self.settings.spawn_duration.sample(rng);
                    // Period is unchecked, should be ignored (could sample to <= 0). Use the spawn
                    // duration, but ensure we have something > 0 as a marker that we've resampled.
                    self.sampled_period = self.sampled_spawn_duration.max(1e-12);
                } else {
                    self.sampled_period = self.settings.period.sample(rng);
                    assert!(self.sampled_period > 0.);
                    self.sampled_spawn_duration = self
                        .settings
                        .spawn_duration
                        .sample(rng)
                        .clamp(0., self.sampled_period);
                }
                self.sampled_spawn_duration = self.settings.spawn_duration.sample(rng);
                self.sampled_count = self.settings.count.sample(rng).max(0.);
            }

            let new_time = self.cycle_time + dt;

            // If inside the spawn period, accumulate some particle spawn count
            if self.cycle_time <= self.sampled_spawn_duration {
                // If the spawn time is very small, close to zero, spawn all particles
                // immediately in one burst over a single frame.
                self.spawn_remainder += if self.sampled_spawn_duration < 1e-5f32.max(dt / 100.0) {
                    self.sampled_count
                } else {
                    // Spawn an amount of particles equal to the fraction of time the current frame
                    // spans compared to the total burst duration.
                    let ratio = ((new_time.min(self.sampled_spawn_duration) - self.cycle_time)
                        / self.sampled_spawn_duration)
                        .clamp(0., 1.);
                    self.sampled_count * ratio
                };
            }

            // Increment current time
            self.cycle_time = new_time;

            // Check for cycle completion
            if self.cycle_time >= self.sampled_period {
                dt = self.cycle_time - self.sampled_period;
                self.cycle_time = 0.0;
                self.completed_cycle_count += 1;

                // Mark as "need resampling"
                self.sampled_period = 0.0;

                // If this was the last cycle, we're done
                if !self.settings.is_forever()
                    && (self.completed_cycle_count >= self.settings.cycle_count())
                {
                    // Don't deactivate quite yet, otherwise we'll miss the spawns for this frame
                    break;
                }
            } else {
                // We're done for this frame
                break;
            }
        }

        // Extract integral number of particles to spawn this frame, keep remainder for
        // next one
        let count = self.spawn_remainder.floor();
        self.spawn_remainder -= count;
        self.spawn_count = count as u32;

        self.spawn_count
    }
}

/// Tick all the [`EffectSpawner`] components.
///
/// This system runs in the [`PostUpdate`] stage, after the visibility system
/// has updated the [`InheritedVisibility`] of each effect instance (see
/// [`VisibilitySystems::VisibilityPropagate`]). Hidden instances are not
/// updated, unless the [`EffectAsset::simulation_condition`]
/// is set to [`SimulationCondition::Always`]. If no [`InheritedVisibility`] is
/// present, the effect is assumed to be visible.
///
/// Note that by that point the [`ViewVisibility`] is not yet calculated, and it
/// may happen that spawners are ticked but no effect is visible in any view
/// even though some are "visible" (active) in the [`World`]. The actual
/// per-view culling of invisible (not in view) effects is performed later on
/// the render world.
///
/// Once the system determined that the effect instance needs to be simulated
/// this frame, it ticks the effect's spawner by calling
/// [`EffectSpawner::tick()`], adding a new [`EffectSpawner`] component if it
/// doesn't already exist on the same entity as the [`ParticleEffect`].
///
/// [`VisibilitySystems::VisibilityPropagate`]: bevy::camera::visibility::VisibilitySystems::VisibilityPropagate
/// [`EffectAsset::simulation_condition`]: crate::EffectAsset::simulation_condition
pub fn tick_spawners(
    mut commands: Commands,
    time: Res<Time<EffectSimulation>>,
    effects: Res<Assets<EffectAsset>>,
    mut rng: ResMut<Random>,
    mut query: Query<(
        Entity,
        &ParticleEffect,
        &CompiledParticleEffect,
        &InheritedVisibility,
        Option<&mut EffectSpawner>,
    )>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("tick_spawners").entered();
    trace!("tick_spawners()");

    let dt = time.delta_secs();

    for (entity, effect, compiled_effect, inherited_visibility, maybe_spawner) in query.iter_mut() {
        // Skip effect which are not ready; this prevents ticking the spawner for an
        // effect not ready to consume those spawn commands.
        let mut can_tick = if compiled_effect.is_ready() {
            true
        } else {
            trace!("[Effect {entity:?}] Not ready; skipped spawner tick.");
            false
        };

        let Some(asset) = effects.get(&effect.handle) else {
            trace!(
                "Effect asset with handle {:?} is not available; skipped initializers tick.",
                effect.handle
            );
            continue;
        };

        if asset.simulation_condition == SimulationCondition::WhenVisible
            && !inherited_visibility.get()
        {
            trace!(
                "Effect asset with handle {:?} is not visible, and simulates only WhenVisible; skipped initializers tick.",
                effect.handle
            );
            can_tick = false;
        }

        if let Some(mut effect_spawner) = maybe_spawner {
            if can_tick {
                effect_spawner.tick(dt, &mut rng.0);
            }
        } else {
            let mut effect_spawner = EffectSpawner::new(&asset.spawner);
            if can_tick {
                effect_spawner.tick(dt, &mut rng.0);
            }
            commands.entity(entity).insert(effect_spawner);
        }
    }
}

#[cfg(test)]
mod test {
    use std::time::Duration;

    use bevy::{
        asset::{
            io::{
                memory::{Dir, MemoryAssetReader},
                AssetSourceBuilder, AssetSourceBuilders, AssetSourceId,
            },
            AssetServerMode, UnapprovedPathMode,
        },
        camera::visibility::{VisibilityPlugin, VisibilitySystems},
        tasks::{IoTaskPool, TaskPoolBuilder},
    };

    use super::*;
    use crate::Module;

    #[test]
    fn test_range_single() {
        let value = CpuValue::Single(1.0);
        assert_eq!(value.range(), [1.0, 1.0]);
    }

    #[test]
    fn test_range_uniform() {
        let value = CpuValue::Uniform((1.0, 3.0));
        assert_eq!(value.range(), [1.0, 3.0]);
    }

    #[test]
    fn test_range_uniform_reverse() {
        let value = CpuValue::Uniform((3.0, 1.0));
        assert_eq!(value.range(), [1.0, 3.0]);
    }

    #[test]
    fn test_new() {
        let rng = &mut new_rng();
        // 3 particles over 3 seconds, pause 7 seconds (total 10 seconds period). 2
        // cycles.
        let spawner = SpawnerSettings::new(3.0.into(), 3.0.into(), 10.0.into(), 2);
        let mut spawner = EffectSpawner::new(&spawner);
        let count = spawner.tick(2., rng); // t = 2s
        assert_eq!(count, 2);
        assert!(spawner.active);
        assert_eq!(spawner.cycle_time(), 2.);
        assert_eq!(spawner.cycle_spawn_duration(), 3.);
        assert_eq!(spawner.cycle_period(), 10.);
        assert_eq!(spawner.cycle_ratio(), 0.2); // 2s / 10s
        assert_eq!(spawner.cycle_spawn_count(), 3.);
        assert_eq!(spawner.completed_cycle_count(), 0);
        let count = spawner.tick(5., rng); // t = 7s
        assert_eq!(count, 1);
        assert!(spawner.active);
        assert_eq!(spawner.cycle_time(), 7.);
        assert_eq!(spawner.cycle_spawn_duration(), 3.);
        assert_eq!(spawner.cycle_period(), 10.);
        assert_eq!(spawner.cycle_ratio(), 0.7); // 7s / 10s
        assert_eq!(spawner.cycle_spawn_count(), 3.);
        assert_eq!(spawner.completed_cycle_count(), 0);
        let count = spawner.tick(8., rng); // t = 15s
        assert_eq!(count, 3);
        assert!(spawner.active);
        assert_eq!(spawner.cycle_time(), 5.); // 15. mod 10.
        assert_eq!(spawner.cycle_spawn_duration(), 3.);
        assert_eq!(spawner.cycle_period(), 10.);
        assert_eq!(spawner.cycle_ratio(), 0.5); // 5s / 10s
        assert_eq!(spawner.cycle_spawn_count(), 3.);
        assert_eq!(spawner.completed_cycle_count(), 1);
        let count = spawner.tick(10., rng); // t = 25s
        assert_eq!(count, 0);
        assert!(spawner.active);
        assert_eq!(spawner.completed_cycle_count(), 2);
        let count = spawner.tick(0.1, rng); // t = 25.1s
        assert_eq!(count, 0);
        assert!(spawner.active);
        assert_eq!(spawner.completed_cycle_count(), 2);
    }

    #[test]
    #[should_panic]
    fn test_new_panic_negative_period() {
        let _ = SpawnerSettings::new(3.0.into(), 1.0.into(), CpuValue::Uniform((-1., 1.)), 0);
    }

    #[test]
    #[should_panic]
    fn test_new_panic_zero_period() {
        let _ = SpawnerSettings::new(3.0.into(), 1.0.into(), CpuValue::Uniform((0., 0.)), 0);
    }

    #[test]
    fn test_once() {
        let rng = &mut new_rng();
        let spawner = SpawnerSettings::once(5.0.into());
        assert!(spawner.is_once());
        let mut spawner = EffectSpawner::new(&spawner);
        assert!(spawner.active);
        let count = spawner.tick(0.001, rng);
        assert_eq!(count, 5);
        let count = spawner.tick(100.0, rng);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_once_reset() {
        let rng = &mut new_rng();
        let spawner = SpawnerSettings::once(5.0.into());
        assert!(spawner.is_once());
        assert!(spawner.starts_active());
        let mut spawner = EffectSpawner::new(&spawner);
        spawner.tick(1.0, rng);
        spawner.reset();
        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_once_start_inactive() {
        let rng = &mut new_rng();

        let spawner = SpawnerSettings::once(5.0.into()).with_starts_active(false);
        assert!(spawner.is_once());
        assert!(!spawner.starts_active());
        let mut spawner = EffectSpawner::new(&spawner);
        assert!(!spawner.has_completed());

        // Inactive; no-op
        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 0);
        assert!(!spawner.has_completed());

        spawner.active = true;

        // Active; spawns
        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 5);
        assert!(spawner.active);
        assert!(spawner.has_completed()); // once(), so completes on first tick()

        // Completed; no-op
        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 0);
        assert!(spawner.active);
        assert!(spawner.has_completed());

        // Reset internal state, still active
        spawner.reset();
        assert!(spawner.active);
        assert!(!spawner.has_completed());

        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 5);
        assert!(spawner.active);
        assert!(spawner.has_completed());
    }

    #[test]
    fn test_rate() {
        let rng = &mut new_rng();
        let spawner = SpawnerSettings::rate(5.0.into());
        assert!(!spawner.is_once());
        assert!(spawner.is_forever());
        let mut spawner = EffectSpawner::new(&spawner);
        // Slightly over 1.0 to avoid edge case
        let count = spawner.tick(1.01, rng);
        assert_eq!(count, 5);
        let count = spawner.tick(0.4, rng);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_rate_active() {
        let rng = &mut new_rng();
        let spawner = SpawnerSettings::rate(5.0.into());
        assert!(!spawner.is_once());
        let mut spawner = EffectSpawner::new(&spawner);
        spawner.tick(1.01, rng);
        spawner.active = false;
        assert!(!spawner.active);
        let count = spawner.tick(0.4, rng);
        assert_eq!(count, 0);
        spawner.active = true;
        assert!(spawner.active);
        let count = spawner.tick(0.4, rng);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_rate_accumulate() {
        let rng = &mut new_rng();
        let spawner = SpawnerSettings::rate(5.0.into());
        assert!(!spawner.is_once());
        let mut spawner = EffectSpawner::new(&spawner);
        // 13 ticks instead of 12 to avoid edge case
        let count = (0..13).map(|_| spawner.tick(1.0 / 60.0, rng)).sum::<u32>();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_burst() {
        let rng = &mut new_rng();
        let spawner = SpawnerSettings::burst(5.0.into(), 2.0.into());
        assert!(!spawner.is_once());
        assert!(spawner.is_forever());
        let mut spawner = EffectSpawner::new(&spawner);
        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 5);
        let count = spawner.tick(4.0, rng);
        assert_eq!(count, 10);
        let count = spawner.tick(0.1, rng);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_with_active() {
        let rng = &mut new_rng();
        let spawner = SpawnerSettings::rate(5.0.into()).with_starts_active(false);
        let mut spawner = EffectSpawner::new(&spawner);
        assert!(!spawner.active);
        let count = spawner.tick(1., rng);
        assert_eq!(count, 0);
        spawner.active = false; // no-op
        let count = spawner.tick(1., rng);
        assert_eq!(count, 0);
        spawner.active = true;
        assert!(spawner.active);
        let count = spawner.tick(1., rng);
        assert_eq!(count, 5);
    }

    fn make_test_app() -> App {
        IoTaskPool::get_or_init(|| {
            TaskPoolBuilder::default()
                .num_threads(1)
                .thread_name("Hanabi test IO Task Pool".to_string())
                .build()
        });

        let mut app = App::new();

        let watch_for_changes = false;
        let mut builders = app
            .world_mut()
            .get_resource_or_insert_with::<AssetSourceBuilders>(Default::default);
        let dir = Dir::default();
        let dummy_builder =
            AssetSourceBuilder::new(move || Box::new(MemoryAssetReader { root: dir.clone() }));
        builders.insert(AssetSourceId::Default, dummy_builder);
        let sources = builders.build_sources(watch_for_changes, false);
        let asset_server = AssetServer::new(
            sources.into(),
            AssetServerMode::Unprocessed,
            watch_for_changes,
            UnapprovedPathMode::Forbid,
        );

        app.insert_resource(asset_server);
        // app.add_plugins(DefaultPlugins);
        app.init_asset::<Mesh>();
        app.add_plugins(VisibilityPlugin);
        app.init_resource::<Time<EffectSimulation>>();
        app.insert_resource(Random(new_rng()));
        app.init_asset::<EffectAsset>();
        app.add_systems(
            PostUpdate,
            tick_spawners.after(VisibilitySystems::CheckVisibility),
        );

        app
    }

    /// Test case for `tick_spawners()`.
    struct TestCase {
        /// Initial entity visibility on spawn. If `None`, do not add a
        /// [`Visibility`] component.
        visibility: Option<Visibility>,

        /// Spawner settings assigned to the `EffectAsset`.
        asset_spawner: SpawnerSettings,
    }

    impl TestCase {
        fn new(visibility: Option<Visibility>, asset_spawner: SpawnerSettings) -> Self {
            Self {
                visibility,
                asset_spawner,
            }
        }
    }

    #[test]
    fn test_tick_spawners() {
        let asset_spawner = SpawnerSettings::once(32.0.into());

        for test_case in &[
            TestCase::new(None, asset_spawner),
            TestCase::new(Some(Visibility::Hidden), asset_spawner),
            TestCase::new(Some(Visibility::Visible), asset_spawner),
        ] {
            let mut app = make_test_app();

            let (effect_entity, handle) = {
                let world = app.world_mut();

                // Add effect asset
                let mut assets = world.resource_mut::<Assets<EffectAsset>>();
                let mut asset = EffectAsset::new(64, test_case.asset_spawner, Module::default());
                asset.simulation_condition = if test_case.visibility.is_some() {
                    SimulationCondition::WhenVisible
                } else {
                    SimulationCondition::Always
                };
                let handle = assets.add(asset);

                // Spawn particle effect
                let entity = if let Some(visibility) = test_case.visibility {
                    world
                        .spawn((
                            visibility,
                            InheritedVisibility::default(),
                            ParticleEffect {
                                handle: handle.clone(),
                                ..default()
                            },
                            // Force-ready the effect as those tests don't initialize the render
                            // world (headless), so the effect would never get ready otherwise.
                            CompiledParticleEffect::default().with_ready_for_tests(),
                        ))
                        .id()
                } else {
                    world
                        .spawn((
                            ParticleEffect {
                                handle: handle.clone(),
                                ..default()
                            },
                            // Force-ready the effect as those tests don't initialize the render
                            // world (headless), so the effect would never get ready otherwise.
                            CompiledParticleEffect::default().with_ready_for_tests(),
                        ))
                        .id()
                };

                // Spawn a camera, otherwise ComputedVisibility stays at HIDDEN
                world.spawn(Camera3d::default());

                (entity, handle)
            };

            // Tick once
            let _cur_time = {
                // Make sure to increment the current time so that the spawners spawn something.
                // Note that `Time` has this weird behavior where the common quantities like
                // `Time::delta_secs()` only update after the *second* update. So we tick the
                // `Time` twice here to enforce this.
                let mut time = app.world_mut().resource_mut::<Time<EffectSimulation>>();
                time.advance_by(Duration::from_millis(16));
                time.elapsed()
            };
            app.update();

            let world = app.world_mut();

            // Check the state of the components after `tick_spawners()` ran
            if let Some(test_visibility) = test_case.visibility {
                // Simulated-when-visible effect (SimulationCondition::WhenVisible)

                let (entity, visibility, inherited_visibility, particle_effect, effect_spawner) =
                    world
                        .query::<(
                            Entity,
                            &Visibility,
                            &InheritedVisibility,
                            &ParticleEffect,
                            Option<&EffectSpawner>,
                        )>()
                        .iter(world)
                        .next()
                        .unwrap();
                assert_eq!(entity, effect_entity);
                assert_eq!(visibility, test_visibility);
                assert_eq!(
                    inherited_visibility.get(),
                    test_visibility == Visibility::Visible
                );
                assert_eq!(particle_effect.handle, handle);

                // The EffectSpawner component is always spawned, even if not visible.
                assert!(effect_spawner.is_some());
                let effect_spawner = effect_spawner.unwrap();
                let actual_spawner = effect_spawner.settings;
                assert_eq!(actual_spawner, test_case.asset_spawner);
                assert!(effect_spawner.active);
                assert_eq!(effect_spawner.spawn_remainder, 0.);
                assert_eq!(effect_spawner.cycle_time, 0.);

                if inherited_visibility.get() {
                    // Check the spawner ticked
                    assert_eq!(effect_spawner.completed_cycle_count, 1);
                    assert_eq!(effect_spawner.spawn_count, 32);
                } else {
                    // Didn't tick
                    assert_eq!(effect_spawner.completed_cycle_count, 0);
                    assert_eq!(effect_spawner.spawn_count, 0);
                }
            } else {
                // Always-simulated effect (SimulationCondition::Always)

                let (entity, particle_effect, effect_spawners) = world
                    .query::<(Entity, &ParticleEffect, Option<&EffectSpawner>)>()
                    .iter(world)
                    .next()
                    .unwrap();
                assert_eq!(entity, effect_entity);
                assert_eq!(particle_effect.handle, handle);

                assert!(effect_spawners.is_some());
                let effect_spawner = effect_spawners.unwrap();
                let actual_spawner = effect_spawner.settings;

                // Check the spawner ticked
                assert!(effect_spawner.active); // will get deactivated next tick()
                assert_eq!(effect_spawner.spawn_remainder, 0.);
                assert_eq!(effect_spawner.cycle_time, 0.);
                assert_eq!(effect_spawner.completed_cycle_count, 1);
                assert_eq!(effect_spawner.spawn_count, 32);

                assert_eq!(actual_spawner, test_case.asset_spawner);
            }
        }
    }
}
