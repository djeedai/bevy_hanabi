use serde::{Deserialize, Serialize};

/// A constant or random value.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Value<T: Copy> {
    /// Single constant value.
    Single(T),
    /// Random value distributed uniformly between two bounds.
    Uniform((T, T)),
}

impl<T: Copy> Value<T> {
    /// Sample the value.
    pub fn sample(&self) -> T {
        match self {
            Value::Single(x) => *x,
            Value::Uniform((a, _b)) => *a, // TODO rand_uniform(*a, *b)
        }
    }
}

impl<T: Copy> From<T> for Value<T> {
    fn from(t: T) -> Self {
        Self::Single(t)
    }
}

/// Spawn rate, in particles per second.
pub type SpawnRate = Value<f32>;

/// Spawn count, in particles per second.
///
/// This is fractional, but only emit one particle once the accumulated spawn count
/// reaches an integral number.
pub type SpawnCount = Value<f32>;

/// Mode of spawning new particles.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum SpawnMode {
    /// Emit once, then idle until effect is reset.
    Once(SpawnCount),
    /// Emit at constant or variable rate, accumulated each frame.
    Rate(SpawnRate),
    /// Emit a number of particles by bursts every given delay.
    Burst((SpawnCount, SpawnRate)),
}

impl SpawnMode {
    /// Create a [`SpawnMode::Once`] with a constant (non-random) value.
    pub fn once(count: f32) -> Self {
        SpawnMode::Once(SpawnCount::Single(count))
    }

    /// Create a [`SpawnMode::Rate`] with a constant (non-random) value.
    pub fn rate(rate: f32) -> Self {
        SpawnMode::Rate(SpawnRate::Single(rate))
    }

    /// Create a [`SpawnMode::Burst`] with constant (non-random) values.
    pub fn burst(count: f32, rate: f32) -> Self {
        SpawnMode::Burst((SpawnCount::Single(count), SpawnRate::Single(rate)))
    }
}

/// Spawner defining how new particles are created.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Spawner {
    /// Number of particles to spawn over `spawn_time`
    num_particles: Value<f32>,

    /// Time over which to spawn `num_particles`, in seconds
    spawn_time: Value<f32>,

    /// Time between bursts of the particle system, in seconds.
    /// If this is infinity, there's only one burst.
    /// If this is `spawn_time`, the system spawns a steady stream of particles.
    period: Value<f32>,

    /// Time since last spawn.
    time: f32,

    /// Sampled value of `spawn_time` until `limit` is reached
    curr_spawn_time: f32,

    /// Time limit until next spawn.
    limit: f32,

    /// Fractional remainder of particle count to spawn.
    spawn: f32,
}

impl Default for Spawner {
    fn default() -> Self {
        Spawner::new(SpawnMode::once(1.))
    }
}

impl Spawner {
    /// Create a new spawner with a given spawn mode.
    pub fn new(mode: SpawnMode) -> Self {
        let (num_particles, spawn_time, period) = match mode {
            SpawnMode::Once(count) => (count, 0.0f32.into(), f32::INFINITY.into()),
            SpawnMode::Rate(rate) => (rate, 1.0f32.into(), 1.0f32.into()),
            SpawnMode::Burst((count, rate)) => (count, 0.0f32.into(), rate),
        };

        Spawner {
            num_particles,
            spawn_time,
            period,
            time: 0.,
            curr_spawn_time: 0.,
            limit: 0.,
            spawn: 0.5,
        }
    }

    /// Create a new spawner that spawns `count` particles,
    /// but doesn't spawn them yet until reset.
    pub fn new_inactive(count: Value<f32>) -> Self {
        Spawner {
            num_particles: count,
            spawn_time: 0.0f32.into(),
            period: f32::INFINITY.into(),
            time: 1.,
            curr_spawn_time: 0.,
            limit: f32::INFINITY,
            spawn: 0.5,
        }
    }

    /// Resets the spawner.
    /// Use this, for example, to immediately spawn some particles.
    pub fn reset(&mut self) {
        self.time = 0.;
        self.limit = 0.;
        self.spawn = 0.5;
    }

    /// Resamples the spawn time and period.
    fn resample(&mut self) {
        self.curr_spawn_time = self.spawn_time.sample();
        self.limit = self.period.sample();
    }

    pub(crate) fn tick(&mut self, mut dt: f32) -> u32 {
        // The limit can be reached multiple times, so use a loop
        loop {
            if self.limit == 0.0 {
                self.resample();
                continue;
            }

            let new_time = self.time + dt;
            if self.time <= self.curr_spawn_time {
                self.spawn += if self.curr_spawn_time == 0.0 {
                    self.num_particles.sample()
                } else if new_time <= self.curr_spawn_time {
                    self.num_particles.sample() * dt / self.curr_spawn_time
                } else {
                    self.num_particles.sample() * (1.0 - self.time / self.curr_spawn_time)
                };
            }

            let old_time = self.time;
            self.time = new_time;

            if self.time >= self.limit {
                dt -= self.limit - old_time;
                self.time = 0.0; // dt will be added on in the next iteration
                self.resample();
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
    fn test_once() {
        let mut spawner = Spawner::new(SpawnMode::once(5.0));
        let count = spawner.tick(0.001);
        assert_eq!(count, 5);
        let count = spawner.tick(100.0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_once_reset() {
        let mut spawner = Spawner::new(SpawnMode::once(5.0));
        spawner.tick(0.001);
        spawner.reset();
        let count = spawner.tick(100.0);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_rate() {
        let mut spawner = Spawner::new(SpawnMode::rate(5.0));
        let count = spawner.tick(1.0);
        assert_eq!(count, 5);
        let count = spawner.tick(0.4);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_rate_accumulate() {
        let mut spawner = Spawner::new(SpawnMode::rate(5.0));

        let count = (0..12).map(|_| spawner.tick(1.0 / 60.0)).sum::<u32>();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_burst() {
        let mut spawner = Spawner::new(SpawnMode::burst(5.0, 2.0));
        let count = spawner.tick(1.0);
        assert_eq!(count, 5);
        let count = spawner.tick(4.0);
        assert_eq!(count, 10);
        let count = spawner.tick(0.1);
        assert_eq!(count, 0);
    }
}