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

    /// Whether the system is active
    active: bool,
}

impl Default for Spawner {
    fn default() -> Self {
        Spawner::once(1.0f32.into())
    }
}

impl Spawner {
    /// Create a spawner with a given count, time, and period.
    ///
    /// - `count` is the number of particles to spawn over `time` in a burst
    /// - `time` is how long to spawn particles for. If this is
    ///   <= 0, then the particles spawn all at once.
    /// - `period` is the amount of time between bursts of particles.
    ///   If this is >= `time`, then the spawner spawns a steady stream of particles.
    ///   If this is infinity, then there is only 1 burst.
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
            spawn: 0.5,
            active: true,
        }
    }

    /// Sets whether the spawner starts active.
    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Create a spawner that spawns `count` particles, then waits until reset.
    pub fn once(count: Value<f32>) -> Self {
        Self::new(count, 0.0.into(), f32::INFINITY.into())
    }

    /// Create a spawner that spawns particles at `rate`, accumulated each frame.
    /// `rate` is in particles per second.
    pub fn rate(rate: Value<f32>) -> Self {
        Self::new(rate, 1.0.into(), 1.0.into())
    }

    /// Create a spawner that spawns `count` particles, waits `period` seconds,
    /// and repeats forever.
    pub fn burst(count: Value<f32>, period: Value<f32>) -> Self {
        Self::new(count, 0.0.into(), period)
    }

    /// Resets and activates the spawner.
    /// Use this, for example, to immediately spawn some particles.
    pub fn reset(&mut self) {
        self.time = 0.;
        self.limit = 0.;
        self.spawn = 0.5;
        self.active = true;
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
    fn resample(&mut self) {
        self.limit = self.period.sample();
        self.curr_spawn_time = self.spawn_time.sample().clamp(0.0, self.limit);
    }

    pub(crate) fn tick(&mut self, mut dt: f32) -> u32 {
        if !self.active {
            return 0;
        }

        // The limit can be reached multiple times, so use a loop
        loop {
            if self.limit == 0.0 {
                self.resample();
                continue;
            }

            let new_time = self.time + dt;
            if self.time <= self.curr_spawn_time {
                self.spawn += if self.curr_spawn_time < dt / 100.0 {
                    self.num_particles.sample()
                } else {
                    self.num_particles.sample() * (new_time.min(self.curr_spawn_time) - self.time)
                        / self.curr_spawn_time
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
        let mut spawner = Spawner::once(5.0.into());
        let count = spawner.tick(0.001);
        assert_eq!(count, 5);
        let count = spawner.tick(100.0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_once_reset() {
        let mut spawner = Spawner::once(5.0.into());
        spawner.tick(1.0);
        spawner.reset();
        let count = spawner.tick(1.0);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_once_inactive() {
        let mut spawner = Spawner::once(5.0.into()).with_active(false);
        let count = spawner.tick(1.0);
        assert_eq!(count, 0);
        spawner.reset();
        let count = spawner.tick(1.0);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_rate() {
        let mut spawner = Spawner::rate(5.0.into());
        let count = spawner.tick(1.0);
        assert_eq!(count, 5);
        let count = spawner.tick(0.4);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_rate_active() {
        let mut spawner = Spawner::rate(5.0.into());
        spawner.tick(1.0);
        spawner.set_active(false);
        assert!(!spawner.is_active());
        let count = spawner.tick(0.4);
        assert_eq!(count, 0);
        spawner.set_active(true);
        assert!(spawner.is_active());
        let count = spawner.tick(0.4);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_rate_accumulate() {
        let mut spawner = Spawner::rate(5.0.into());

        let count = (0..12).map(|_| spawner.tick(1.0 / 60.0)).sum::<u32>();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_burst() {
        let mut spawner = Spawner::burst(5.0.into(), 2.0.into());
        let count = spawner.tick(1.0);
        assert_eq!(count, 5);
        let count = spawner.tick(4.0);
        assert_eq!(count, 10);
        let count = spawner.tick(0.1);
        assert_eq!(count, 0);
    }
}
