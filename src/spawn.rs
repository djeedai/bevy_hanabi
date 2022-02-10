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
    /// Spawn mode.
    mode: SpawnMode,

    /// Time since last spawn.
    time: f32,

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
        Spawner {
            mode,
            time: 0.,
            limit: 0.,
            spawn: 0.,
        }
    }

    pub(crate) fn tick(&mut self, dt: f32) -> u32 {
        match &self.mode {
            SpawnMode::Once(count) => {
                if self.limit > -0.5 {
                    self.spawn += count.sample();
                    self.limit = -1.;
                }
            }
            SpawnMode::Rate(rate) => {
                self.spawn += rate.sample() * dt;
            }
            SpawnMode::Burst((count, rate)) => {
                self.time += dt;
                if self.time >= self.limit {
                    self.spawn += count.sample();
                    self.time = 0.;
                    self.limit = rate.sample();
                }
            }
        }
        let count = self.spawn.floor();
        self.spawn -= count;
        count as u32
    }
}
