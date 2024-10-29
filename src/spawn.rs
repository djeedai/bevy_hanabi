use std::hash::{Hash, Hasher};

use bevy::{ecs::system::Resource, math::FloatOrd, prelude::*, reflect::Reflect};
use rand::{
    distributions::{uniform::SampleUniform, Distribution, Uniform},
    SeedableRng,
};
use rand_pcg::Pcg32;
use serde::{Deserialize, Serialize};

use crate::{EffectAsset, EffectSimulation, ParticleEffect, SimulationCondition};

/// An RNG to be used in the CPU for the particle system engine
pub(crate) fn new_rng() -> Pcg32 {
    let mut rng = rand::thread_rng();
    let mut seed = [0u8; 16];
    seed.copy_from_slice(&Uniform::from(0..=u128::MAX).sample(&mut rng).to_le_bytes());
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
/// with the [`Spawner`].
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
            Self::Uniform((a, b)) => Uniform::new_inclusive(*a, *b).sample(rng),
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

#[derive(Clone, Copy, PartialEq, Debug, Reflect, Serialize, Deserialize)]
#[reflect(Serialize, Deserialize)]
pub enum Initializer {
    Spawner(Spawner),
    Cloner(Cloner),
}

impl From<Spawner> for Initializer {
    #[inline]
    fn from(value: Spawner) -> Self {
        Self::Spawner(value)
    }
}

impl From<Cloner> for Initializer {
    #[inline]
    fn from(value: Cloner) -> Self {
        Self::Cloner(value)
    }
}

impl Initializer {
    #[cfg(test)]
    fn get_spawner(&self) -> Option<&Spawner> {
        match *self {
            Initializer::Spawner(ref spawner) => Some(spawner),
            Initializer::Cloner(_) => None,
        }
    }
}

/// Spawner defining how new particles are emitted.
///
/// The spawner defines how new particles are emitted and when. Each time the
/// spawner ticks, it calculates a number of particles to emit for this frame.
/// This spawn count is passed to the GPU for the init compute pass to actually
/// allocate the new particles and initialize them. The number of particles to
/// spawn is stored as a floating-point number, and any remainder accumulates
/// for the next emitting.
///
/// The spawner itself is embedded into the [`EffectInitializers`] component.
/// Once per frame the [`tick_spawners()`] system will add the component if
/// it's missing, cloning the [`Spawner`] from the source [`EffectAsset`], then
/// tick the [`Spawner`] stored in the [`EffectInitializers`].
#[derive(Debug, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect(Default)]
pub struct Spawner {
    /// Number of particles to spawn over [`spawn_time`].
    ///
    /// [`spawn_time`]: Spawner::spawn_time
    num_particles: CpuValue<f32>,

    /// Time over which to spawn `num_particles`, in seconds.
    spawn_time: CpuValue<f32>,

    /// Time between bursts of the particle system, in seconds.
    /// If this is infinity, there's only one burst.
    /// If this is `spawn_time`, the system spawns a steady stream of particles.
    period: CpuValue<f32>,

    /// Whether the spawner is active at startup. The value is used to initialize
    /// [`EffectSpawner::active`].
    ///
    /// [`EffectSpawner::active`]: crate::EffectSpawner::active
    starts_active: bool,

    /// Whether the burst of a once-style spawner triggers immediately when the
    /// spawner becomes active. If `false`, the spawner doesn't do anything
    /// until [`EffectSpawner::reset()`] is called.
    starts_immediately: bool,
}

impl Default for Spawner {
    fn default() -> Self {
        Self::once(1.0f32.into(), true)
    }
}

impl Spawner {
    /// Create a spawner with a given count, time, and period.
    ///
    /// This is the _raw_ constructor. In general you should prefer using one of
    /// the utility constructors [`once()`], [`burst()`], or [`rate()`],
    /// which will ensure the control parameters are set consistently relative
    /// to each other.
    ///
    /// The control parameters are:
    ///
    /// - `count` is the number of particles to spawn over `time` in a burst. It
    ///   can generate negative or zero random values, in which case no particle
    ///   is spawned during the current frame.
    /// - `time` is how long to spawn particles for. If this is <= 0, then the
    ///   particles spawn all at once exactly at the same instant.
    /// - `period` is the amount of time between bursts of particles. If this is
    ///   <= `time`, then the spawner spawns a steady stream of particles. If
    ///   this is infinity, then there is a single burst.
    ///
    /// Note that the "burst" semantic here doesn't strictly mean a one-off
    /// emission, since that emission is spread over a number of simulation
    /// frames that total a duration of `time`. If you want a strict
    /// single-frame burst, simply set the `time` to zero; this is what
    /// [`once()`] does.
    ///
    /// # Panics
    ///
    /// Panics if `period` can be a negative number (the sample range lower
    /// bound is negative), or can only be 0 (the sample range upper bound is
    /// not strictly positive).
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::Spawner;
    /// // Spawn 32 particles over 3 seconds, then pause for 7 seconds (10 - 3).
    /// let spawner = Spawner::new(32.0.into(), 3.0.into(), 10.0.into());
    /// ```
    ///
    /// [`once()`]: crate::Spawner::once
    /// [`burst()`]: crate::Spawner::burst
    /// [`rate()`]: crate::Spawner::rate
    pub fn new(count: CpuValue<f32>, time: CpuValue<f32>, period: CpuValue<f32>) -> Self {
        assert!(
            period.range()[0] >= 0.,
            "`period` must not generate negative numbers (period.min was {}, expected >= 0).",
            period.range()[0]
        );
        assert!(
            period.range()[1] > 0.,
            "`period` must be able to generate a positive number (period.max was {}, expected > 0).",
            period.range()[1]
        );

        Self {
            num_particles: count,
            spawn_time: time,
            period,
            starts_active: true,
            starts_immediately: true,
        }
    }

    /// Create a spawner that spawns `count` particles, then waits until reset.
    ///
    /// If `spawn_immediately` is `false`, this waits until
    /// [`EffectSpawner::reset()`] before spawning a burst of particles.
    ///
    /// When `spawn_immediately == true`, this is a convenience for:
    ///
    /// ```
    /// # use bevy_hanabi::{Spawner, CpuValue};
    /// # let count = CpuValue::Single(1.);
    /// Spawner::new(count, 0.0.into(), f32::INFINITY.into());
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::Spawner;
    /// // Spawn 32 particles in a burst once immediately on creation.
    /// let spawner = Spawner::once(32.0.into(), true);
    /// ```
    ///
    /// [`reset()`]: crate::Spawner::reset
    pub fn once(count: CpuValue<f32>, spawn_immediately: bool) -> Self {
        let mut spawner = Self::new(count, 0.0.into(), f32::INFINITY.into());
        spawner.starts_immediately = spawn_immediately;
        spawner
    }

    /// Get whether this spawner emits a single burst.
    pub fn is_once(&self) -> bool {
        if let CpuValue::Single(f) = self.period {
            f.is_infinite()
        } else {
            false
        }
    }

    /// Create a spawner that spawns particles at `rate`, accumulated each
    /// frame. `rate` is in particles per second.
    ///
    /// This is a convenience for:
    ///
    /// ```
    /// # use bevy_hanabi::{Spawner, CpuValue};
    /// # let rate = CpuValue::Single(1.);
    /// Spawner::new(rate, 1.0.into(), 1.0.into());
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::Spawner;
    /// // Spawn 10 particles per second, indefinitely.
    /// let spawner = Spawner::rate(10.0.into());
    /// ```
    pub fn rate(rate: CpuValue<f32>) -> Self {
        Self::new(rate, 1.0.into(), 1.0.into())
    }

    /// Create a spawner that spawns `count` particles, waits `period` seconds,
    /// and repeats forever.
    ///
    /// This is a convenience for:
    ///
    /// ```
    /// # use bevy_hanabi::{Spawner, CpuValue};
    /// # let count = CpuValue::Single(1.);
    /// # let period = CpuValue::Single(1.);
    /// Spawner::new(count, 0.0.into(), period);
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::Spawner;
    /// // Spawn a burst of 5 particles every 3 seconds, indefinitely.
    /// let spawner = Spawner::burst(5.0.into(), 3.0.into());
    /// ```
    pub fn burst(count: CpuValue<f32>, period: CpuValue<f32>) -> Self {
        Self::new(count, 0.0.into(), period)
    }

    /// Set the number of particles that are spawned each cycle.
    pub fn with_count(mut self, count: CpuValue<f32>) -> Self {
        self.num_particles = count;
        self
    }

    /// Set the number of particles that are spawned each cycle.
    pub fn set_count(&mut self, count: CpuValue<f32>) {
        self.num_particles = count;
    }

    /// Get the number of particles that are spawned each cycle.
    pub fn count(&self) -> CpuValue<f32> {
        self.num_particles
    }

    /// Set the length of the spawn time each cycle.
    pub fn with_spawn_time(mut self, spawn_time: CpuValue<f32>) -> Self {
        self.spawn_time = spawn_time;
        self
    }

    /// Set the length of the spawn time each cycle.
    pub fn set_spawn_time(&mut self, spawn_time: CpuValue<f32>) {
        self.spawn_time = spawn_time;
    }

    /// Get the length of spawn time each cycle.
    pub fn spawn_time(&self) -> CpuValue<f32> {
        self.spawn_time
    }

    /// Set the wait time between spawn cycles.
    pub fn with_period(mut self, period: CpuValue<f32>) -> Self {
        self.period = period;
        self
    }

    /// Set the wait time between spawn cycles.
    pub fn set_period(&mut self, period: CpuValue<f32>) {
        self.period = period;
    }

    /// Get the wait time between spawn cycles
    pub fn period(&self) -> CpuValue<f32> {
        self.period
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

/// Defines how particle trails are to be constructed.
///
/// Particle trails are constructed by cloning the particles from a group into
/// a different group on a fixed interval. Each time the cloner ticks, it
/// clones all the particles from the source group into the destination group.
/// Hanabi then runs the initialization modifiers on the newly-cloned
/// particles. Particle clones that would overflow the destination group
/// (exceed its capacity) are dropped.
///
/// The cloner itself is embedded into the [`EffectInitializers`] component.
/// Once per frame the [`tick_spawners()`] system will add the component if
/// it's missing, copying fields from the [`Cloner`] to the [`EffectCloner`].
#[derive(Default, Clone, Copy, Debug, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect(Serialize, Deserialize)]
pub struct Cloner {
    /// The group from which the cloner copies.
    pub src_group_index: u32,

    /// Time between clone operations, in seconds.
    pub period: CpuValue<f32>,

    /// Time that the particles persist, in seconds.
    ///
    /// Unlike spawned particles, cloned particles don't use the
    /// [`crate::attributes::Attribute::LIFETIME`] attribute and instead track
    /// lifetime themselves, using this value. This is because, internally,
    /// their lifetimes must follow last-in-first-out (LIFO) order.
    pub lifetime: f32,

    /// Whether the system is active at startup. The value is used to initialize
    /// [`EffectCloner::active`].
    ///
    /// [`EffectCloner::active`]: crate::EffectCloner::active
    pub starts_active: bool,
}

impl Cloner {
    /// Creates a cloner with the given source group index, period, and lifetime.
    ///
    /// This is the raw constructor. A more convenient way to create cloners is
    /// to use [`EffectAsset::with_trails`] or [`EffectAsset::with_ribbons`].
    pub fn new(src_group_index: u32, period: impl Into<CpuValue<f32>>, lifetime: f32) -> Self {
        Self {
            src_group_index,
            period: period.into(),
            lifetime,
            starts_active: true,
        }
    }

    /// Sets whether the cloner starts active when the effect is instantiated.
    ///
    /// This value will be transfered to the active state of the
    /// [`EffectCloner`] once it's instantiated. Inactive cloners do not clone
    /// any particle.
    pub fn with_starts_active(mut self, starts_active: bool) -> Self {
        self.starts_active = starts_active;
        self
    }

    /// Set whether the cloner starts active when the effect is instantiated.
    ///
    /// This value will be transfered to the active state of the
    /// [`EffectCloner`] once it's instantiated. Inactive cloners do not clone
    /// any particle.
    pub fn set_starts_active(&mut self, starts_active: bool) {
        self.starts_active = starts_active;
    }

    /// Get whether the cloner starts active when the effect is instantiated.
    ///
    /// This value will be transfered to the active state of the
    /// [`EffectCloner`] once it's instantiated. Inactive cloners do not clone
    /// any particle.
    pub fn starts_active(&self) -> bool {
        self.starts_active
    }
}

/// A runtime component maintaining the state of all initializers for an effect.
///
/// This component is automatically added to the same [`Entity`] as the
/// [`ParticleEffect`] it's associated with, during [`tick_spawners()`], if not
/// already present on the entity. In that case, the initializer configurations
/// are cloned from the underlying [`EffectAsset`] associated with the particle
/// effect instance.
///
/// You can manually add this component in advance to override its [`Spawner`]s
/// and/or [`Cloner`]s. In that case [`tick_spawners()`] will use the existing
/// component you added.
///
/// Each frame, for spawners, the component will automatically calculate the
/// number of particles to spawn, via its internal [`Spawner`], and store it
/// into [`spawn_count`]. You can manually override that value if you want, to
/// create more complex spawning sequences. For cloners, the component sets the
/// [`spawn_this_frame`] flag as appropriate. You can likewise manually override
/// that value if you want in order to clone on different schedules.
///
/// [`spawn_count`]: crate::EffectSpawner::spawn_count
/// [`spawn_count`]: crate::EffectCloner::spawn_this_frame
#[derive(Default, Clone, Component, PartialEq, Reflect, Debug, Deref, DerefMut)]
#[reflect(Component)]
pub struct EffectInitializers(pub Vec<EffectInitializer>);

impl EffectInitializers {
    /// Resets the initializer state.
    ///
    /// This resets the internal time for all initializers to zero, and restarts
    /// any internal particle counters that they might possess.
    ///
    /// Use this, for example, to immediately spawn some particles in a spawner
    /// constructed with [`Spawner::once`].
    ///
    /// [`Spawner::once`]: crate::Spawner::once
    pub fn reset(&mut self) {
        for initializer in &mut self.0 {
            initializer.reset();
        }
    }

    /// Marks all initializers as either active or inactive.
    ///
    /// Inactive initializers don't spawn any particles.
    pub fn set_active(&mut self, active: bool) {
        for initializer in &mut self.0 {
            initializer.set_active(active);
        }
    }
}

/// Holds the runtime state for the initializer of a single particle group on a
/// particle effect.
#[derive(Clone, Copy, PartialEq, Reflect, Debug)]
pub enum EffectInitializer {
    /// The group uses a spawner.
    Spawner(EffectSpawner),
    /// The group uses a cloner (i.e. is a trail or ribbon).
    Cloner(EffectCloner),
}

impl EffectInitializer {
    /// If this initializer is a spawner, returns an immutable reference to it.
    pub fn get_spawner(&self) -> Option<&EffectSpawner> {
        match *self {
            EffectInitializer::Spawner(ref spawner) => Some(spawner),
            _ => None,
        }
    }

    /// Resets the initializer state.
    ///
    /// This resets the internal time for this initializer to zero, and
    /// restarts any internal particle counters that it might possess.
    ///
    /// Use this, for example, to immediately spawn some particles in a spawner
    /// constructed with [`Spawner::once`].
    ///
    /// [`Spawner::once`]: crate::Spawner::once
    pub fn reset(&mut self) {
        match self {
            EffectInitializer::Spawner(effect_spawner) => effect_spawner.reset(),
            EffectInitializer::Cloner(effect_cloner) => effect_cloner.reset(),
        }
    }

    /// Marks this initializer as either active or inactive.
    ///
    /// Inactive initializers don't spawn any particles.
    pub fn set_active(&mut self, active: bool) {
        match self {
            EffectInitializer::Spawner(effect_spawner) => effect_spawner.set_active(active),
            EffectInitializer::Cloner(effect_cloner) => effect_cloner.set_active(active),
        }
    }
}

/// Runtime structure maintaining the state of the spawner for a particle group.
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect)]
pub struct EffectSpawner {
    /// The spawner configuration extracted either from the [`EffectAsset`], or
    /// from any overriden value provided by the user on the [`ParticleEffect`].
    spawner: Spawner,

    /// Accumulated time since last spawn.
    time: f32,

    /// Sampled value of `spawn_time` until `limit` is reached.
    curr_spawn_time: f32,

    /// Time limit until next spawn.
    limit: f32,

    /// Number of particles to spawn this frame.
    ///
    /// This value is normally updated by calling [`tick()`], which
    /// automatically happens once per frame when the [`tick_spawners()`] system
    /// runs in the [`PostUpdate`] schedule.
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
    /// This will be accumulated with the new count next tick, and the integral
    /// part will be stored in `spawn_count`.
    spawn_remainder: f32,

    /// Whether the system is active. Defaults to `true`.
    active: bool,
}

impl EffectSpawner {
    /// Create a new spawner state from an asset definition.
    pub fn new(spawner: &Spawner) -> Self {
        Self {
            spawner: *spawner,
            time: if spawner.is_once() && !spawner.starts_immediately {
                1. // anything > 0
            } else {
                0.
            },
            curr_spawn_time: 0.,
            limit: 0.,
            spawn_count: 0,
            spawn_remainder: 0.,
            active: spawner.starts_active(),
        }
    }

    /// Set whether the spawner is active.
    ///
    /// Inactive spawners do not spawn any particle.
    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Set whether the spawner is active.
    ///
    /// Inactive spawners do not spawn any particle.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// Get whether the spawner is active.
    ///
    /// Inactive spawners do not spawn any particle.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Get the spawner configuration in use.
    ///
    /// The effective [`Spawner`] used is either the override specified in the
    /// associated [`ParticleEffect`] instance, or the fallback one specified in
    /// underlying [`EffectAsset`].
    pub fn spawner(&self) -> &Spawner {
        &self.spawner
    }

    /// Reset the spawner state.
    ///
    /// This resets the internal spawner time to zero, and restarts any internal
    /// particle counter.
    ///
    /// Use this, for example, to immediately spawn some particles in a spawner
    /// constructed with [`Spawner::once`].
    ///
    /// [`Spawner::once`]: crate::Spawner::once
    pub fn reset(&mut self) {
        self.time = 0.;
        self.limit = 0.;
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
        if !self.active {
            self.spawn_count = 0;
            return 0;
        }

        // The limit can be reached multiple times, so use a loop
        loop {
            if self.limit == 0.0 {
                self.resample(rng);
                continue;
            }

            let new_time = self.time + dt;
            if self.time <= self.curr_spawn_time {
                // If the spawn time is very small, close to zero, spawn all particles
                // immediately in one burst over a single frame.
                self.spawn_remainder += if self.curr_spawn_time < 1e-5f32.max(dt / 100.0) {
                    self.spawner.num_particles.sample(rng)
                } else {
                    // Spawn an amount of particles equal to the fraction of time the current frame
                    // spans compared to the total burst duration.
                    self.spawner.num_particles.sample(rng)
                        * (new_time.min(self.curr_spawn_time) - self.time)
                        / self.curr_spawn_time
                };
            }

            let old_time = self.time;
            self.time = new_time;

            if self.time >= self.limit {
                dt -= self.limit - old_time;
                self.time = 0.0; // dt will be added on in the next iteration
                self.resample(rng);
            } else {
                break;
            }
        }

        let count = self.spawn_remainder.floor();
        self.spawn_remainder -= count;
        self.spawn_count = count as u32;

        self.spawn_count
    }

    /// Resamples the spawn time and period.
    fn resample(&mut self, rng: &mut Pcg32) {
        self.limit = self.spawner.period.sample(rng);
        self.curr_spawn_time = self.spawner.spawn_time.sample(rng).clamp(0.0, self.limit);
    }
}

/// A runtime structure maintaining the state of the cloner for a particle
/// group.
#[derive(Default, Clone, Copy, PartialEq, Reflect, Debug)]
pub struct EffectCloner {
    /// The cloner configuration extracted either from the [`EffectAsset`] or
    /// overridden manually.
    pub cloner: Cloner,
    /// Accumulated time since last spawn.
    time: f32,
    /// Time limit until next spawn.
    limit: f32,
    /// The capacity of the group.
    capacity: u32,
    /// Whether the cloner is to spawn this frame.
    pub spawn_this_frame: bool,
    /// Whether the cloner is active. Defaults to true.
    pub active: bool,
}

impl EffectCloner {
    pub(crate) fn new(cloner: Cloner, capacity: u32) -> EffectCloner {
        EffectCloner {
            cloner,
            time: 0.0,
            limit: 0.0,
            capacity,
            spawn_this_frame: false,
            active: cloner.starts_active(),
        }
    }

    /// Reset the spawner state.
    ///
    /// This resets the internal spawner time to zero, and restarts any internal
    /// particle counter.
    ///
    /// [`Spawner::once`]: crate::Spawner::once
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.limit = 0.0;
    }

    /// Returns true if we should spawn this frame.
    pub fn tick(&mut self, dt: f32, rng: &mut Pcg32) {
        if !self.active {
            self.spawn_this_frame = false;
            return;
        }

        if self.limit == 0.0 {
            self.resample(rng);
        }

        let new_time = self.time + dt;
        self.time = new_time;

        self.spawn_this_frame = self.time >= self.limit;

        if self.spawn_this_frame {
            self.time = 0.0;
            self.resample(rng);
        }
    }

    fn resample(&mut self, rng: &mut Pcg32) {
        self.limit = self.cloner.period.sample(rng);
    }

    /// Marks this cloner as either active or inactive.
    ///
    /// Inactive cloners don't clone any particles.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }
}

/// Tick all the [`EffectSpawner`] components of the simulated
/// [`ParticleEffect`] components.
///
/// This system runs in the [`PostUpdate`] stage, after the visibility system
/// has updated the [`InheritedVisibility`] of each effect instance (see
/// [`VisibilitySystems::VisibilityPropagate`]). Hidden instances are not
/// updated, unless the [`EffectAsset::simulation_condition`]
/// is set to [`SimulationCondition::Always`].
///
/// Note that by that point the [`ViewVisibility`] is not yet calculated, and it
/// may happen that spawners are ticked but no effect is visible in any view
/// even though some are "visible" (active) in the [`World`]. The actual
/// per-view culling of invisible (not in view) effects is performed later on
/// the render world.
///
/// Once the system determined that the effect instance needs to be simulated
/// this frame, it ticks the spawner by calling [`EffectSpawner::tick()`],
/// adding a new [`EffectSpawner`] component if it doesn't already exist on the
/// same entity as the [`ParticleEffect`].
///
/// [`VisibilitySystems::VisibilityPropagate`]: bevy::render::view::VisibilitySystems::VisibilityPropagate
/// [`EffectAsset::simulation_condition`]: crate::EffectAsset::simulation_condition
pub fn tick_spawners(
    mut commands: Commands,
    time: Res<Time<EffectSimulation>>,
    effects: Res<Assets<EffectAsset>>,
    mut rng: ResMut<Random>,
    mut query: Query<(
        Entity,
        &ParticleEffect,
        Option<&InheritedVisibility>,
        Option<&mut EffectInitializers>,
    )>,
) {
    trace!("tick_spawners");

    let dt = time.delta_seconds();

    for (entity, effect, maybe_inherited_visibility, maybe_initializers) in query.iter_mut() {
        // TODO - maybe cache simulation_condition so we don't need to unconditionally
        // query the asset?
        let Some(asset) = effects.get(&effect.handle) else {
            continue;
        };

        if asset.simulation_condition == SimulationCondition::WhenVisible
            && !maybe_inherited_visibility
                .map(|iv| iv.get())
                .unwrap_or(true)
        {
            continue;
        }

        if let Some(mut initializers) = maybe_initializers {
            for initializer in &mut **initializers {
                match initializer {
                    EffectInitializer::Spawner(effect_spawner) => {
                        effect_spawner.tick(dt, &mut rng.0);
                    }
                    EffectInitializer::Cloner(effect_cloner) => {
                        effect_cloner.tick(dt, &mut rng.0);
                    }
                }
            }
            continue;
        }

        let initializers = asset
            .init
            .iter()
            .enumerate()
            .map(|(group_index, init)| match *init {
                Initializer::Spawner(spawner) => {
                    let mut effect_spawner = EffectSpawner::new(&spawner);
                    effect_spawner.tick(dt, &mut rng.0);
                    EffectInitializer::Spawner(effect_spawner)
                }
                Initializer::Cloner(cloner) => {
                    let mut effect_cloner =
                        EffectCloner::new(cloner, asset.capacities()[group_index]);
                    effect_cloner.tick(dt, &mut rng.0);
                    EffectInitializer::Cloner(effect_cloner)
                }
            })
            .collect();
        commands
            .entity(entity)
            .insert(EffectInitializers(initializers));
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
            AssetServerMode,
        },
        render::view::{VisibilityPlugin, VisibilitySystems},
        tasks::{IoTaskPool, TaskPoolBuilder},
    };

    use super::*;
    use crate::Module;

    /// Make an `EffectSpawner` wrapping a `Spawner`.
    fn make_effect_spawner(spawner: Spawner) -> EffectSpawner {
        EffectSpawner::new(
            EffectAsset::new(256, spawner, Module::default()).init[0]
                .get_spawner()
                .expect("Expected the first group to have a spawner"),
        )
    }

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
        // 3 particles over 3 seconds, pause 7 seconds (total 10 seconds period).
        let spawner = Spawner::new(3.0.into(), 3.0.into(), 10.0.into());
        let mut spawner = make_effect_spawner(spawner);
        let count = spawner.tick(2.0, rng); // t = 2s
        assert_eq!(count, 2);
        let count = spawner.tick(5.0, rng); // t = 7s
        assert_eq!(count, 1);
        let count = spawner.tick(8.0, rng); // t = 15s
        assert_eq!(count, 3);
    }

    #[test]
    #[should_panic]
    fn test_new_panic_negative_period() {
        let _ = Spawner::new(3.0.into(), 1.0.into(), CpuValue::Uniform((-1., 1.)));
    }

    #[test]
    #[should_panic]
    fn test_new_panic_zero_period() {
        let _ = Spawner::new(3.0.into(), 1.0.into(), CpuValue::Uniform((0., 0.)));
    }

    #[test]
    fn test_once() {
        let rng = &mut new_rng();
        let spawner = Spawner::once(5.0.into(), true);
        let mut spawner = make_effect_spawner(spawner);
        let count = spawner.tick(0.001, rng);
        assert_eq!(count, 5);
        let count = spawner.tick(100.0, rng);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_once_reset() {
        let rng = &mut new_rng();
        let spawner = Spawner::once(5.0.into(), true);
        let mut spawner = make_effect_spawner(spawner);
        spawner.tick(1.0, rng);
        spawner.reset();
        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_once_not_immediate() {
        let rng = &mut new_rng();
        let spawner = Spawner::once(5.0.into(), false);
        let mut spawner = make_effect_spawner(spawner);
        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 0);
        spawner.reset();
        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_rate() {
        let rng = &mut new_rng();
        let spawner = Spawner::rate(5.0.into());
        let mut spawner = make_effect_spawner(spawner);
        // Slightly over 1.0 to avoid edge case
        let count = spawner.tick(1.01, rng);
        assert_eq!(count, 5);
        let count = spawner.tick(0.4, rng);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_rate_active() {
        let rng = &mut new_rng();
        let spawner = Spawner::rate(5.0.into());
        let mut spawner = make_effect_spawner(spawner);
        spawner.tick(1.01, rng);
        spawner.set_active(false);
        assert!(!spawner.is_active());
        let count = spawner.tick(0.4, rng);
        assert_eq!(count, 0);
        spawner.set_active(true);
        assert!(spawner.is_active());
        let count = spawner.tick(0.4, rng);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_rate_accumulate() {
        let rng = &mut new_rng();
        let spawner = Spawner::rate(5.0.into());
        let mut spawner = make_effect_spawner(spawner);
        // 13 ticks instead of 12 to avoid edge case
        let count = (0..13).map(|_| spawner.tick(1.0 / 60.0, rng)).sum::<u32>();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_burst() {
        let rng = &mut new_rng();
        let spawner = Spawner::burst(5.0.into(), 2.0.into());
        let mut spawner = make_effect_spawner(spawner);
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
        let spawner = Spawner::rate(5.0.into()).with_starts_active(false);
        let mut spawner = make_effect_spawner(spawner);
        assert!(!spawner.is_active());
        let count = spawner.tick(1., rng);
        assert_eq!(count, 0);
        spawner.set_active(false); // no-op
        let count = spawner.tick(1., rng);
        assert_eq!(count, 0);
        spawner.set_active(true);
        assert!(spawner.is_active());
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
        let dummy_builder = AssetSourceBuilder::default()
            .with_reader(move || Box::new(MemoryAssetReader { root: dir.clone() }));
        builders.insert(AssetSourceId::Default, dummy_builder);
        let sources = builders.build_sources(watch_for_changes, false);
        let asset_server =
            AssetServer::new(sources, AssetServerMode::Unprocessed, watch_for_changes);

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

        /// Spawner assigned to the `EffectAsset`.
        asset_spawner: Spawner,
    }

    impl TestCase {
        fn new(visibility: Option<Visibility>, asset_spawner: Spawner) -> Self {
            Self {
                visibility,
                asset_spawner,
            }
        }
    }

    #[test]
    fn test_tick_spawners() {
        let asset_spawner = Spawner::once(32.0.into(), true);

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
                                #[cfg(feature = "2d")]
                                z_layer_2d: None,
                            },
                        ))
                        .id()
                } else {
                    world
                        .spawn((ParticleEffect {
                            handle: handle.clone(),
                            #[cfg(feature = "2d")]
                            z_layer_2d: None,
                        },))
                        .id()
                };

                // Spawn a camera, otherwise ComputedVisibility stays at HIDDEN
                world.spawn(Camera3dBundle::default());

                (entity, handle)
            };

            // Tick once
            let cur_time = {
                // Make sure to increment the current time so that the spawners spawn something.
                // Note that `Time` has this weird behavior where the common quantities like
                // `Time::delta_seconds()` only update after the *second* update. So we tick the
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

                let (entity, visibility, inherited_visibility, particle_effect, effect_spawners) =
                    world
                        .query::<(
                            Entity,
                            &Visibility,
                            &InheritedVisibility,
                            &ParticleEffect,
                            Option<&EffectInitializers>,
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
                if inherited_visibility.get() {
                    // If visible, `tick_spawners()` spawns the EffectSpawner and ticks it
                    assert!(effect_spawners.is_some());
                    let effect_spawner = effect_spawners.unwrap()[0].get_spawner().unwrap();
                    let actual_spawner = effect_spawner.spawner;

                    // Check the spawner ticked
                    assert!(effect_spawner.active);
                    assert_eq!(effect_spawner.spawn_remainder, 0.);
                    assert_eq!(effect_spawner.time, cur_time.as_secs_f32());

                    assert_eq!(actual_spawner, test_case.asset_spawner);
                    assert_eq!(effect_spawner.spawn_count, 32);
                } else {
                    // If not visible, `tick_spawners()` skips the effect entirely so won't spawn an
                    // `EffectSpawner` for it
                    assert!(effect_spawners.is_none());
                }
            } else {
                // Always-simulated effect (SimulationCondition::Always)

                let (entity, particle_effect, effect_spawners) = world
                    .query::<(Entity, &ParticleEffect, Option<&EffectInitializers>)>()
                    .iter(world)
                    .next()
                    .unwrap();
                assert_eq!(entity, effect_entity);
                assert_eq!(particle_effect.handle, handle);

                assert!(effect_spawners.is_some());
                let effect_spawner = effect_spawners.unwrap()[0].get_spawner().unwrap();
                let actual_spawner = effect_spawner.spawner;

                // Check the spawner ticked
                assert!(effect_spawner.active);
                assert_eq!(effect_spawner.spawn_remainder, 0.);
                assert_eq!(effect_spawner.time, cur_time.as_secs_f32());

                assert_eq!(actual_spawner, test_case.asset_spawner);
                assert_eq!(effect_spawner.spawn_count, 32);
            }
        }
    }
}
