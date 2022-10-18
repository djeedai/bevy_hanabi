use bevy::{reflect::{Reflect, FromReflect}, prelude::Resource};
use rand::{
    distributions::{uniform::SampleUniform, Distribution, Uniform},
    SeedableRng,
};
use rand_pcg::Pcg32;
use serde::{Deserialize, Serialize};

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

/// A constant or random value.
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Value<T: Copy> {
    /// Single constant value.
    Single(T),
    /// Random value distributed uniformly between two bounds.
    Uniform((T, T)),
}

impl<T: Copy + Default> Default for Value<T> {
    fn default() -> Self {
        Self::Single(T::default())
    }
}

impl<T: Copy + SampleUniform> Value<T> {
    /// Sample the value.
    pub fn sample(&self, rng: &mut Pcg32) -> T {
        match self {
            Value::Single(x) => *x,
            Value::Uniform((a, b)) => Uniform::new_inclusive(*a, *b).sample(rng),
        }
    }
}

impl<T: Copy + PartialOrd> Value<T> {
    /// Returns the range of values this can be
    /// in the form `[minimum, maximum]`
    pub fn range(&self) -> [T; 2] {
        match self {
            Value::Single(x) => [*x; 2],
            Value::Uniform((a, b)) => {
                if a <= b {
                    [*a, *b]
                } else {
                    [*b, *a]
                }
            }
        }
    }
}

impl<T: Copy> From<T> for Value<T> {
    fn from(t: T) -> Self {
        Self::Single(t)
    }
}

/// Spawner defining how new particles are created.
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Reflect, FromReflect)]
pub struct Spawner {
    /// Number of particles to spawn over `spawn_time`
    #[reflect(ignore)] // TODO
    num_particles: Value<f32>,

    /// Time over which to spawn `num_particles`, in seconds
    #[reflect(ignore)] // TODO
    spawn_time: Value<f32>,

    /// Time between bursts of the particle system, in seconds.
    /// If this is infinity, there's only one burst.
    /// If this is `spawn_time`, the system spawns a steady stream of particles.
    #[reflect(ignore)] // TODO
    period: Value<f32>,

    /// Time since last spawn.
    time: f32,

    /// Sampled value of `spawn_time` until `limit` is reached
    curr_spawn_time: f32,

    /// Time limit until next spawn.
    limit: f32,

    /// Fractional remainder of particle count to spawn.
    spawn: f32,

    /// Whether the system is active
    active: bool,
}

impl Default for Spawner {
    fn default() -> Self {
        Spawner::once(1.0f32.into(), true)
    }
}

impl Spawner {
    /// Create a spawner with a given count, time, and period.
    ///
    /// - `count` is the number of particles to spawn over `time` in a burst
    /// - `time` is how long to spawn particles for. If this is <= 0, then the
    ///   particles spawn all at once.
    /// - `period` is the amount of time between bursts of particles. If this is
    ///   >= `time`, then the spawner spawns a steady stream of particles. If
    ///   this is infinity, then there is only 1 burst.
    ///
    /// # Panics:
    /// Panics if `period` can be a negative number, or can only be 0.
    pub fn new(count: Value<f32>, time: Value<f32>, period: Value<f32>) -> Self {
        assert!(
            period.range()[0] >= 0. && period.range()[1] > 0.,
            "`period` must be able to generate a positive number and no negative numbers"
        );

        Spawner {
            num_particles: count,
            spawn_time: time,
            period,
            time: 0.,
            curr_spawn_time: 0.,
            limit: 0.,
            spawn: 0.,
            active: true,
        }
    }

    /// Sets whether the spawner starts active.
    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Create a spawner that spawns `count` particles, then waits until reset.
    /// If `spawn_immediately` is false, this waits until reset before
    /// spawning its first burst of particles.
    pub fn once(count: Value<f32>, spawn_immediately: bool) -> Self {
        let mut spawner = Self::new(count, 0.0.into(), f32::INFINITY.into());
        if !spawn_immediately {
            spawner.time = 1.0;
        }
        spawner
    }

    /// Create a spawner that spawns particles at `rate`, accumulated each
    /// frame. `rate` is in particles per second.
    pub fn rate(rate: Value<f32>) -> Self {
        Self::new(rate, 1.0.into(), 1.0.into())
    }

    /// Create a spawner that spawns `count` particles, waits `period` seconds,
    /// and repeats forever.
    pub fn burst(count: Value<f32>, period: Value<f32>) -> Self {
        Self::new(count, 0.0.into(), period)
    }

    /// Resets the spawner.
    /// Use this, for example, to immediately spawn some particles
    /// in a spawner constructed with `Spawner::once`.
    pub fn reset(&mut self) {
        self.time = 0.;
        self.limit = 0.;
        self.spawn = 0.;
    }

    /// Sets whether the spawner is active.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// Gets whether the spawner is active.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Resamples the spawn time and period.
    fn resample(&mut self, rng: &mut Pcg32) {
        self.limit = self.period.sample(rng);
        self.curr_spawn_time = self.spawn_time.sample(rng).clamp(0.0, self.limit);
    }

    pub(crate) fn tick(&mut self, mut dt: f32, rng: &mut Pcg32) -> u32 {
        if !self.active {
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
                self.spawn += if self.curr_spawn_time < 1e-5f32.max(dt / 100.0) {
                    self.num_particles.sample(rng)
                } else {
                    self.num_particles.sample(rng)
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

        let count = self.spawn.floor();
        self.spawn -= count;
        count as u32
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_range_single() {
        let value = Value::Single(1.0);
        assert_eq!(value.range(), [1.0, 1.0]);
    }

    #[test]
    fn test_range_uniform() {
        let value = Value::Uniform((1.0, 3.0));
        assert_eq!(value.range(), [1.0, 3.0]);
    }

    #[test]
    fn test_range_uniform_reverse() {
        let value = Value::Uniform((3.0, 1.0));
        assert_eq!(value.range(), [1.0, 3.0]);
    }

    #[test]
    fn test_once() {
        let rng = &mut new_rng();
        let mut spawner = Spawner::once(5.0.into(), true);
        let count = spawner.tick(0.001, rng);
        assert_eq!(count, 5);
        let count = spawner.tick(100.0, rng);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_once_reset() {
        let rng = &mut new_rng();
        let mut spawner = Spawner::once(5.0.into(), true);
        spawner.tick(1.0, rng);
        spawner.reset();
        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_once_not_immediate() {
        let rng = &mut new_rng();
        let mut spawner = Spawner::once(5.0.into(), false);
        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 0);
        spawner.reset();
        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_rate() {
        let rng = &mut new_rng();
        let mut spawner = Spawner::rate(5.0.into());
        // Slightly over 1.0 to avoid edge case
        let count = spawner.tick(1.01, rng);
        assert_eq!(count, 5);
        let count = spawner.tick(0.4, rng);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_rate_active() {
        let rng = &mut new_rng();
        let mut spawner = Spawner::rate(5.0.into());
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
        let mut spawner = Spawner::rate(5.0.into());
        // 13 ticks instead of 12 to avoid edge case
        let count = (0..13).map(|_| spawner.tick(1.0 / 60.0, rng)).sum::<u32>();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_burst() {
        let rng = &mut new_rng();
        let mut spawner = Spawner::burst(5.0.into(), 2.0.into());
        let count = spawner.tick(1.0, rng);
        assert_eq!(count, 5);
        let count = spawner.tick(4.0, rng);
        assert_eq!(count, 10);
        let count = spawner.tick(0.1, rng);
        assert_eq!(count, 0);
    }
}
