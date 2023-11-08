use std::hash::{Hash, Hasher};

use bevy::{ecs::system::Resource, prelude::*, reflect::Reflect, utils::FloatOrd};
use rand::{
    distributions::{uniform::SampleUniform, Distribution, Uniform},
    SeedableRng,
};
use rand_pcg::Pcg32;
use serde::{Deserialize, Serialize};

use crate::{EffectAsset, ParticleEffect, SimulationCondition};

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

/// Spawner defining how new particles are emitted.
///
/// The spawner defines how new particles are emitted and when. Each time the
/// spawner ticks, once per frame by the [`tick_spawners()`] system, it
/// calculates a number of particles to emit for this frame. This spawn count is
/// passed to the GPU for the init compute pass to actually allocate the new
/// particles and initialize them. The number of particles to spawn is stored as
/// a floating-point number, and any remainder accumulates for the next
/// emitting.
#[derive(Debug, Copy, Clone, PartialEq, Reflect, Serialize, Deserialize)]
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

    /// Whether the system is active at startup. The value is used to initialize
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

/// Runtime component maintaining the state of the spawner for an effect.
///
/// This component is automatically added to the same [`Entity`] as the
/// [`ParticleEffect`] it's associated with, during [`tick_spawners()`], if not
/// already present on the entity. The spawer configuration is derived from the
/// [`ParticleEffect`] itself, or as fallback from the underlying
/// [`EffectAsset`] associated with the particle effect instance.
#[derive(Default, Clone, Copy, PartialEq, Component)]
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

    /// Number of particles to spawn, as calculated by last [`tick()`] call.
    ///
    /// [`tick()`]: crate::EffectSpawner::tick
    spawn_count: u32,

    /// Fractional remainder of particle count to spawn.
    spawn_remainder: f32,

    /// Whether the system is active. Defaults to `true`.
    active: bool,
}

impl EffectSpawner {
    /// Create a new spawner state from an asset and an instance.
    ///
    /// The spawner data is cloned from the instance if the instance has an
    /// override. Otherwise it's cloned from the asset.
    pub fn new(asset: &EffectAsset, instance: &ParticleEffect) -> Self {
        let spawner = *instance.spawner.as_ref().unwrap_or(&asset.spawner);
        Self {
            spawner,
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

    /// Get the particle spawn count calculated by the last [`tick()`] call.
    ///
    /// This corresponds to the number of particles that will be (or have been,
    /// depending on the instant at which this is called inside the frame)
    /// spawned this frame.
    ///
    /// [`tick()`]: crate::EffectSpawner::tick
    #[inline]
    pub fn spawn_count(&self) -> u32 {
        self.spawn_count
    }

    /// Resamples the spawn time and period.
    fn resample(&mut self, rng: &mut Pcg32) {
        self.limit = self.spawner.period.sample(rng);
        self.curr_spawn_time = self.spawner.spawn_time.sample(rng).clamp(0.0, self.limit);
    }
}

/// Tick all the spawners of the visible [`ParticleEffect`] components.
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
/// [`VisibilitySystems::VisibilityPropagate`]: bevy::render::view::VisibilitySystems::VisibilityPropagate
/// [`EffectAsset::simulation_condition`]: crate::EffectAsset::simulation_condition
pub fn tick_spawners(
    mut commands: Commands,
    time: Res<Time>,
    effects: Res<Assets<EffectAsset>>,
    mut rng: ResMut<Random>,
    mut query: Query<(
        Entity,
        &ParticleEffect,
        Option<&InheritedVisibility>,
        Option<&mut EffectSpawner>,
    )>,
) {
    trace!("tick_spawners");

    let dt = time.delta_seconds();

    for (entity, effect, maybe_inherited_visibility, maybe_spawner) in query.iter_mut() {
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

        if let Some(mut spawner) = maybe_spawner {
            spawner.tick(dt, &mut rng.0);
        } else {
            let mut spawner = EffectSpawner::new(asset, effect);
            spawner.tick(dt, &mut rng.0);
            commands.entity(entity).insert(spawner);
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
            AssetServerMode,
        },
        render::view::{VisibilityPlugin, VisibilitySystems},
        tasks::{IoTaskPool, TaskPoolBuilder},
    };

    use crate::Module;

    use super::*;

    /// Make an `EffectSpawner` wrapping a `Spawner`.
    fn make_effect_spawner(spawner: Spawner) -> EffectSpawner {
        EffectSpawner::new(
            &EffectAsset::new(256, spawner, Module::default()),
            &ParticleEffect::default(),
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
            .world
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
        app.init_resource::<Time>();
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

        /// Optional spawner assigned to the `ParticleEffect` instance, which
        /// overrides the asset one.
        instance_spawner: Option<Spawner>,
    }

    impl TestCase {
        fn new(
            visibility: Option<Visibility>,
            asset_spawner: Spawner,
            instance_spawner: Option<Spawner>,
        ) -> Self {
            Self {
                visibility,
                asset_spawner,
                instance_spawner,
            }
        }
    }

    #[test]
    fn test_tick_spawners() {
        let asset_spawner = Spawner::once(32.0.into(), true);
        let instance_spawner = Spawner::once(64.0.into(), true);

        for test_case in &[
            TestCase::new(None, asset_spawner, None),
            TestCase::new(Some(Visibility::Hidden), asset_spawner, None),
            TestCase::new(Some(Visibility::Visible), asset_spawner, None),
            TestCase::new(
                Some(Visibility::Visible),
                asset_spawner,
                Some(instance_spawner),
            ),
        ] {
            let mut app = make_test_app();

            let (effect_entity, handle) = {
                let world = &mut app.world;

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
                                spawner: test_case.instance_spawner,
                                ..default()
                            },
                        ))
                        .id()
                } else {
                    world
                        .spawn((ParticleEffect {
                            handle: handle.clone(),
                            spawner: test_case.instance_spawner,
                            ..default()
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
                let mut time = app.world.resource_mut::<Time>();
                time.advance_by(Duration::from_millis(16));
                time.elapsed()
            };
            app.update();

            let world = &mut app.world;

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
                if inherited_visibility.get() {
                    // If visible, `tick_spawners()` spawns the EffectSpawner and ticks it
                    assert!(effect_spawner.is_some());
                    let effect_spawner = effect_spawner.unwrap();
                    let actual_spawner = effect_spawner.spawner();

                    // Check the spawner ticked
                    assert!(effect_spawner.active);
                    assert_eq!(effect_spawner.spawn_remainder, 0.);
                    assert_eq!(effect_spawner.time, cur_time.as_secs_f32());

                    // Check the spawner is actually the one we expect from the override rule
                    if let Some(instance_spawner) = &test_case.instance_spawner {
                        // If there's a per-instance spawner override, it should be the one used
                        assert_eq!(*actual_spawner, *instance_spawner);
                        assert_eq!(effect_spawner.spawn_count, 64);
                    } else {
                        // Otherwise the asset spawner should be used
                        assert_eq!(*actual_spawner, test_case.asset_spawner);
                        assert_eq!(effect_spawner.spawn_count, 32);
                    }
                } else {
                    // If not visible, `tick_spawners()` skips the effect entirely so won't spawn an
                    // `EffectSpawner` for it
                    assert!(effect_spawner.is_none());
                }
            } else {
                // Always-simulated effect (SimulationCondition::Always)

                let (entity, particle_effect, effect_spawner) = world
                    .query::<(Entity, &ParticleEffect, Option<&EffectSpawner>)>()
                    .iter(world)
                    .next()
                    .unwrap();
                assert_eq!(entity, effect_entity);
                assert_eq!(particle_effect.handle, handle);

                assert!(effect_spawner.is_some());
                let effect_spawner = effect_spawner.unwrap();
                let actual_spawner = effect_spawner.spawner();

                // Check the spawner ticked
                assert!(effect_spawner.active);
                assert_eq!(effect_spawner.spawn_remainder, 0.);
                assert_eq!(effect_spawner.time, cur_time.as_secs_f32());

                // Check the spawner is actually the one we expect from the override rule
                if let Some(instance_spawner) = &test_case.instance_spawner {
                    // If there's a per-instance spawner override, it should be the one used
                    assert_eq!(*actual_spawner, *instance_spawner);
                    assert_eq!(effect_spawner.spawn_count, 64);
                } else {
                    // Otherwise the asset spawner should be used
                    assert_eq!(*actual_spawner, test_case.asset_spawner);
                    assert_eq!(effect_spawner.spawn_count, 32);
                }
            }
        }
    }
}
