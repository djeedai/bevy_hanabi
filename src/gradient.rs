use bevy::{
    core::FloatOrd,
    math::{Quat, Vec2, Vec3, Vec3A, Vec4},
};
use std::vec::Vec;

/// Describes a type that can be linearly interpolated between two keys.
///
/// This trait is used for values in a gradient, which are primitive types and are
/// therefore copyable.
pub trait Lerp: Copy {
    fn lerp(self, other: Self, ratio: f32) -> Self;
}

impl Lerp for f32 {
    #[inline]
    fn lerp(self, other: Self, ratio: f32) -> Self {
        self * (1. - ratio) + other * ratio
    }
}

impl Lerp for f64 {
    #[inline]
    fn lerp(self, other: Self, ratio: f32) -> Self {
        self * (1. - ratio) as f64 + other * ratio as f64
    }
}

macro_rules! impl_lerp_vecn {
    ($t:ty) => {
        impl Lerp for $t {
            #[inline]
            fn lerp(self, other: Self, ratio: f32) -> Self {
                // Force use of type's own lerp() to disambiguate and prevent infinite recursion
                <$t>::lerp(self, other, ratio)
            }
        }
    };
}

impl_lerp_vecn!(Vec2);
impl_lerp_vecn!(Vec3);
impl_lerp_vecn!(Vec3A);
impl_lerp_vecn!(Vec4);

impl Lerp for Quat {
    fn lerp(self, other: Self, ratio: f32) -> Self {
        // We use slerp() instead of lerp() as conceptually we want a smooth interpolation
        // and we expect Quat to be used to represent a rotation. lerp() would produce an
        // interpolation with varying speed, which feels non-natural.
        self.slerp(other, ratio)
    }
}

/// A single key point for a [`Gradient`].
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct GradientKey<T: Lerp> {
    /// Ratio in \[0:1\] where the key is located.
    ratio: f32,

    /// Value associated with the key.
    ///
    /// The value is uploaded as is to the render shader. For colors, this means
    /// the value does not imply any particular color space by itself.
    pub value: T,
}

impl<T: Lerp> GradientKey<T> {
    /// Get the ratio where the key point is located, in \[0:1\].
    pub fn ratio(&self) -> f32 {
        self.ratio
    }
}

/// A gradient curve made of keypoints and associated values.
///
/// The gradient can be sampled anywhere, and will return a linear interpolation
/// of the values of its closest keys. Sampling before 0 or after 1 returns a
/// constant value equal to the one of the closest bound.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Gradient<T: Lerp> {
    keys: Vec<GradientKey<T>>,
}

impl<T: Default + Lerp> Gradient<T> {
    /// Create a new empty gradient.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a constant gradient.
    /// Inserts the value at 0.0 and nowhere else.
    pub fn constant(value: T) -> Self {
        let mut grad = Self::default();
        grad.add_key(0.0, value);
        grad
    }
}

impl<T: Lerp> Gradient<T> {
    /// Add a key point to the gradient.
    ///
    /// If one or more duplicate ratios already exist, append the new key after all
    /// the existing keys with same ratio.
    ///
    /// The ratio must be a finite floating point value.
    ///
    /// # Panics
    ///
    /// This method panics if `ratio` is not in the \[0:1\] range.
    pub fn add_key(&mut self, ratio: f32, value: T) {
        assert!(ratio >= 0.0);
        assert!(ratio <= 1.0);
        let index = match self
            .keys
            .binary_search_by(|key| FloatOrd(key.ratio).cmp(&FloatOrd(ratio)))
        {
            Ok(mut index) => {
                // When there are duplicate keys, binary_search_by() returns the index of an unspecified
                // one. Make sure we insert always as the last duplicate one, for determinism.
                let len = self.keys.len();
                while index + 1 < len && self.keys[index].ratio == self.keys[index + 1].ratio {
                    index += 1;
                }
                index + 1 // insert after last duplicate
            }
            Err(upper_index) => upper_index,
        };
        self.keys.insert(index, GradientKey { ratio, value });
    }

    /// Get the gradient keys.
    pub fn keys(&self) -> &[GradientKey<T>] {
        &self.keys[..]
    }

    /// Get mutable access to the gradient keys.
    pub fn keys_mut(&mut self) -> &mut [GradientKey<T>] {
        &mut self.keys[..]
    }

    /// Sample the gradient at the given ratio.
    ///
    /// If the ratio is exactly equal to those of one or more keys, sample the first key
    /// in the collection. If the ratio falls between two keys, return a linear interpolation
    /// of their values. If the ratio is before the first key or after the last one, return
    /// the first and last value, respectively.
    ///
    /// # Panics
    ///
    /// This method panics if the gradient is empty (has no key point).
    pub fn sample(&self, ratio: f32) -> T {
        assert!(!self.keys.is_empty());
        match self
            .keys
            .binary_search_by(|key| FloatOrd(key.ratio).cmp(&FloatOrd(ratio)))
        {
            Ok(mut index) => {
                // When there are duplicate keys, binary_search_by() returns the index of an
                // unspecified one. Make sure we sample the first duplicate, for determinism.
                while index > 0 && self.keys[index - 1].ratio == self.keys[index].ratio {
                    index -= 1;
                }
                self.keys[index].value
            }
            Err(upper_index) => {
                if upper_index > 0 {
                    if upper_index < self.keys.len() {
                        let key0 = &self.keys[upper_index - 1];
                        let key1 = &self.keys[upper_index];
                        let t = (ratio - key0.ratio) / (key1.ratio - key0.ratio);
                        key0.value.lerp(key1.value, t)
                    } else {
                        // post: sampling point located after the last key
                        self.keys[upper_index - 1].value
                    }
                } else {
                    // pre: sampling point located before the first key
                    self.keys[upper_index].value
                }
            }
        }
    }

    /// Sample the gradient at regular intervals.
    ///
    /// Create a list of sample points starting at ratio `start` and spaced with `inc`
    /// delta ratio. The number of samples is equal to the length of the `dst` slice.
    /// Sample the gradient at all those points, and fill the `dst` slice with the
    /// resulting values.
    ///
    /// This is equivalent to calling [`sample()`] in a loop, but is more efficient.
    ///
    /// [`sample()`]: Gradient::sample
    pub fn sample_by(&self, start: f32, inc: f32, dst: &mut [T]) {
        let count = dst.len();
        assert!(!self.keys.is_empty());
        let mut ratio = start;
        // pre: sampling points located before the first key
        let first_ratio = self.keys[0].ratio;
        let first_col = self.keys[0].value;
        let mut idst = 0;
        while idst < count && ratio <= first_ratio {
            dst[idst] = first_col;
            idst += 1;
            ratio += inc;
        }
        // main: sampling points located on or after the first key
        let mut ikey = 1;
        let len = self.keys.len();
        for i in idst..count {
            // Find the first key after the ratio
            while ikey < len && ratio > self.keys[ikey].ratio {
                ikey += 1;
            }
            if ikey >= len {
                // post: sampling points located after the last key
                let last_col = self.keys[len - 1].value;
                for d in &mut dst[i..] {
                    *d = last_col;
                }
                return;
            }
            if self.keys[ikey].ratio == ratio {
                dst[i] = self.keys[ikey].value;
            } else {
                let k0 = &self.keys[ikey - 1];
                let k1 = &self.keys[ikey];
                let t = (ratio - k0.ratio) / (k1.ratio - k0.ratio);
                dst[i] = k0.value.lerp(k1.value, t);
            }
            ratio += inc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn color_approx_eq(c0: Vec4, c1: Vec4, tol: f32) -> bool {
        ((c0.x - c1.x).abs() < tol)
            && ((c0.y - c1.y).abs() < tol)
            && ((c0.z - c1.z).abs() < tol)
            && ((c0.w - c1.w).abs() < tol)
    }

    #[test]
    fn constant() {
        let grad = Gradient::constant(3.0);
        assert_eq!(grad.sample(0.0), 3.0);
        assert_eq!(grad.sample(0.3), 3.0);
        assert_eq!(grad.sample(1.0), 3.0);
    }

    #[test]
    fn add_key() {
        let red: Vec4 = Vec4::new(1., 0., 0., 1.);
        let blue: Vec4 = Vec4::new(0., 0., 1., 1.);
        let green: Vec4 = Vec4::new(0., 1., 0., 1.);
        let mut g = Gradient::new();
        g.add_key(0.3, red);
        // duplicate keys allowed
        g.add_key(0.3, red);
        // duplicate ratios stored in order they're inserted
        g.add_key(0.7, blue);
        g.add_key(0.7, green);
        let keys = g.keys();
        assert_eq!(4, keys.len());
        assert!(color_approx_eq(red, keys[0].value, 1e-5));
        assert!(color_approx_eq(red, keys[1].value, 1e-5));
        assert!(color_approx_eq(blue, keys[2].value, 1e-5));
        assert!(color_approx_eq(green, keys[3].value, 1e-5));
    }

    #[test]
    fn sample() {
        let red: Vec4 = Vec4::new(1., 0., 0., 1.);
        let blue: Vec4 = Vec4::new(0., 0., 1., 1.);
        let green: Vec4 = Vec4::new(0., 1., 0., 1.);
        let mut g = Gradient::new();
        g.add_key(0.5, red);
        assert_eq!(red, g.sample(0.0));
        assert_eq!(red, g.sample(0.5));
        assert_eq!(red, g.sample(1.0));
        g.add_key(0.8, blue);
        g.add_key(0.8, green);
        assert_eq!(red, g.sample(0.0));
        assert_eq!(red, g.sample(0.499));
        assert_eq!(red, g.sample(0.5));
        let expected = red.lerp(blue, 1. / 3.);
        let actual = g.sample(0.6);
        assert!(color_approx_eq(actual, expected, 1e-5));
        assert_eq!(blue, g.sample(0.8));
        assert_eq!(green, g.sample(0.801));
        assert_eq!(green, g.sample(1.0));
    }

    #[test]
    fn sample_by() {
        let red: Vec4 = Vec4::new(1., 0., 0., 1.);
        let blue: Vec4 = Vec4::new(0., 0., 1., 1.);
        let mut g = Gradient::new();
        g.add_key(0.5, red);
        g.add_key(0.8, blue);
        const COUNT: usize = 256;
        let mut data: [Vec4; COUNT] = [Vec4::ZERO; COUNT];
        let start = 0.;
        let inc = 1. / COUNT as f32;
        g.sample_by(start, inc, &mut data[..]);
        for (i, &d) in data.iter().enumerate() {
            let ratio = start + inc * i as f32;
            let expected = g.sample(ratio);
            assert!(color_approx_eq(expected, d, 1e-5));
        }
    }
}
