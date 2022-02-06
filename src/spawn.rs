use serde::{Deserialize,Serialize};

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Value<T: Copy> {
    /// Single constant value.
    Single(T),
    /// Random value distributed uniformly between two bounds.
    Uniform((T, T)),
}

impl<T: Copy> Value<T> {
    pub fn sample(&self) -> T {
        match self {
            Value::Single(x) => *x,
            Value::Uniform((a, _b)) => *a, // TODO rand_uniform(*a, *b)
        }
    }
}

pub type SpawnRate = Value<f32>;
pub type SpawnCount = Value<f32>;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum SpawnMode {
    /// Emit once, then idle until effect is reset.
    Once(SpawnCount),
    /// Emit at constant or variable rate, accumulated each frame.
    Rate(SpawnRate),
    /// Emit a number of particles by bursts every given delay.
    Burst((SpawnCount, SpawnRate)),
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Spawner {
    /// Spawn mode.
    mode: SpawnMode,

    time: f32,
    limit: f32,
    spawn: f32,
}

impl Spawner {
    pub fn new(mode: SpawnMode) -> Self {
        Spawner {
            mode,
            time: 0.,
            limit: 0.,
            spawn: 0.,
        }
    }

    pub fn tick(&mut self, dt: f32) -> u32 {
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
