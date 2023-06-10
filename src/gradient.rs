use bevy::{
    math::{Quat, Vec2, Vec3, Vec3A, Vec4},
    reflect::{FromReflect, Reflect},
    utils::FloatOrd,
};
use serde::{Deserialize, Serialize};
use std::{
    hash::{Hash, Hasher},
    vec::Vec,
};

/// Describes a type that can be linearly interpolated between two keys.
///
/// This trait is used for values in a gradient, which are primitive types and
/// are therefore copyable.
pub trait Lerp: Copy {
    fn lerp(self, other: Self, ratio: f32) -> Self;
}

impl Lerp for f32 {
    #[inline]
    fn lerp(self, other: Self, ratio: f32) -> Self {
        self.mul_add(1. - ratio, other * ratio)
    }
}

impl Lerp for f64 {
    #[inline]
    fn lerp(self, other: Self, ratio: f32) -> Self {
        self.mul_add((1. - ratio) as f64, other * ratio as f64)
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
        // We use slerp() instead of lerp() as conceptually we want a smooth
        // interpolation and we expect Quat to be used to represent a rotation.
        // lerp() would produce an interpolation with varying speed, which feels
        // non-natural.
        self.slerp(other, ratio)
    }
}

/// A single key point for a [`Gradient`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct GradientKey<T: Lerp + FromReflect> {
    /// Ratio in \[0:1\] where the key is located.
    ratio: f32,

    /// Value associated with the key.
    ///
    /// The value is uploaded as is to the render shader. For colors, this means
    /// the value does not imply any particular color space by itself.
    pub value: T,
}

impl<T: Lerp + FromReflect> GradientKey<T> {
    /// Get the ratio where the key point is located, in \[0:1\].
    pub fn ratio(&self) -> f32 {
        self.ratio
    }
}

impl Hash for GradientKey<f32> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        FloatOrd(self.ratio).hash(state);
        FloatOrd(self.value).hash(state);
    }
}

impl Hash for GradientKey<Vec2> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        FloatOrd(self.ratio).hash(state);
        FloatOrd(self.value.x).hash(state);
        FloatOrd(self.value.y).hash(state);
    }
}

impl Hash for GradientKey<Vec3> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        FloatOrd(self.ratio).hash(state);
        FloatOrd(self.value.x).hash(state);
        FloatOrd(self.value.y).hash(state);
        FloatOrd(self.value.z).hash(state);
    }
}

impl Hash for GradientKey<Vec4> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        FloatOrd(self.ratio).hash(state);
        FloatOrd(self.value.x).hash(state);
        FloatOrd(self.value.y).hash(state);
        FloatOrd(self.value.z).hash(state);
        FloatOrd(self.value.w).hash(state);
    }
}

/// A gradient curve made of keypoints and associated values.
///
/// The gradient can be sampled anywhere, and will return a linear interpolation
/// of the values of its closest keys. Sampling before 0 or after 1 returns a
/// constant value equal to the one of the closest bound.
#[derive(Debug, Default, Clone, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct Gradient<T: Lerp + FromReflect> {
    keys: Vec<GradientKey<T>>,
}

// SAFETY: This is consistent with the derive, but we can't derive due to trait
// bounds.
#[allow(clippy::derived_hash_with_manual_eq)]
impl<T> Hash for Gradient<T>
where
    T: Default + Lerp + FromReflect,
    GradientKey<T>: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.keys.hash(state);
    }
}

impl<T: Default + Lerp + FromReflect> Gradient<T> {
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

impl<T: Lerp + FromReflect> Gradient<T> {
    /// Add a key point to the gradient.
    ///
    /// If one or more duplicate ratios already exist, append the new key after
    /// all the existing keys with same ratio.
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
                // When there are duplicate keys, binary_search_by() returns the index of an
                // unspecified one. Make sure we insert always as the last
                // duplicate one, for determinism.
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
    /// If the ratio is exactly equal to those of one or more keys, sample the
    /// first key in the collection. If the ratio falls between two keys,
    /// return a linear interpolation of their values. If the ratio is
    /// before the first key or after the last one, return the first and
    /// last value, respectively.
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
    /// Create a list of sample points starting at ratio `start` and spaced with
    /// `inc` delta ratio. The number of samples is equal to the length of
    /// the `dst` slice. Sample the gradient at all those points, and fill
    /// the `dst` slice with the resulting values.
    ///
    /// This is equivalent to calling [`sample()`] in a loop, but is more
    /// efficient.
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
    use std::collections::hash_map::DefaultHasher;

    use bevy::reflect::{ReflectRef, Struct};

    use crate::test_utils::*;

    use super::*;

    const RED: Vec4 = Vec4::new(1., 0., 0., 1.);
    const BLUE: Vec4 = Vec4::new(0., 0., 1., 1.);
    const GREEN: Vec4 = Vec4::new(0., 1., 0., 1.);

    fn make_test_gradient() -> Gradient<Vec4> {
        let mut g = Gradient::new();
        g.add_key(0.5, RED);
        g.add_key(0.8, BLUE);
        g.add_key(0.8, GREEN);
        g
    }

    fn color_approx_eq(c0: Vec4, c1: Vec4, tol: f32) -> bool {
        ((c0.x - c1.x).abs() < tol)
            && ((c0.y - c1.y).abs() < tol)
            && ((c0.z - c1.z).abs() < tol)
            && ((c0.w - c1.w).abs() < tol)
    }

    #[test]
    fn lerp_test() {
        assert_approx_eq!(Lerp::lerp(3_f32, 5_f32, 0.1), 3.2_f32);
        assert_approx_eq!(Lerp::lerp(3_f32, 5_f32, 0.5), 4.0_f32);
        assert_approx_eq!(Lerp::lerp(3_f32, 5_f32, 0.9), 4.8_f32);
        assert_approx_eq!(Lerp::lerp(5_f32, 3_f32, 0.1), 4.8_f32);
        assert_approx_eq!(Lerp::lerp(5_f32, 3_f32, 0.5), 4.0_f32);
        assert_approx_eq!(Lerp::lerp(5_f32, 3_f32, 0.9), 3.2_f32);

        assert_approx_eq!(Lerp::lerp(3_f64, 5_f64, 0.1), 3.2_f64);
        assert_approx_eq!(Lerp::lerp(3_f64, 5_f64, 0.5), 4.0_f64);
        assert_approx_eq!(Lerp::lerp(3_f64, 5_f64, 0.9), 4.8_f64);
        assert_approx_eq!(Lerp::lerp(5_f64, 3_f64, 0.1), 4.8_f64);
        assert_approx_eq!(Lerp::lerp(5_f64, 3_f64, 0.5), 4.0_f64);
        assert_approx_eq!(Lerp::lerp(5_f64, 3_f64, 0.9), 3.2_f64);

        let s = Quat::IDENTITY;
        let e = Quat::from_rotation_x(90_f32.to_radians());
        assert_approx_eq!(Lerp::lerp(s, e, 0.1), s.slerp(e, 0.1));
        assert_approx_eq!(Lerp::lerp(s, e, 0.5), s.slerp(e, 0.5));
        assert_approx_eq!(Lerp::lerp(s, e, 0.9), s.slerp(e, 0.9));
        assert_approx_eq!(Lerp::lerp(e, s, 0.1), s.slerp(e, 0.9));
        assert_approx_eq!(Lerp::lerp(e, s, 0.5), s.slerp(e, 0.5));
        assert_approx_eq!(Lerp::lerp(e, s, 0.9), s.slerp(e, 0.1));
    }

    #[test]
    fn constant() {
        let grad = Gradient::constant(3.0);
        for r in [
            -1e5, -0.5, -0.0001, 0., 0.0001, 0.3, 0.5, 0.9, 0.9999, 1., 1.0001, 100., 1e5,
        ] {
            assert_approx_eq!(grad.sample(r), 3.0);
        }
    }

    #[test]
    fn add_key() {
        let mut g = Gradient::new();
        g.add_key(0.3, RED);
        // duplicate keys allowed
        g.add_key(0.3, RED);
        // duplicate ratios stored in order they're inserted
        g.add_key(0.7, BLUE);
        g.add_key(0.7, GREEN);
        let keys = g.keys();
        assert_eq!(4, keys.len());
        assert!(color_approx_eq(RED, keys[0].value, 1e-5));
        assert!(color_approx_eq(RED, keys[1].value, 1e-5));
        assert!(color_approx_eq(BLUE, keys[2].value, 1e-5));
        assert!(color_approx_eq(GREEN, keys[3].value, 1e-5));
    }

    #[test]
    fn sample() {
        let mut g = Gradient::new();
        g.add_key(0.5, RED);
        assert_eq!(RED, g.sample(0.0));
        assert_eq!(RED, g.sample(0.5));
        assert_eq!(RED, g.sample(1.0));
        g.add_key(0.8, BLUE);
        g.add_key(0.8, GREEN);
        assert_eq!(RED, g.sample(0.0));
        assert_eq!(RED, g.sample(0.499));
        assert_eq!(RED, g.sample(0.5));
        let expected = RED.lerp(BLUE, 1. / 3.);
        let actual = g.sample(0.6);
        assert!(color_approx_eq(actual, expected, 1e-5));
        assert_eq!(BLUE, g.sample(0.8));
        assert_eq!(GREEN, g.sample(0.801));
        assert_eq!(GREEN, g.sample(1.0));
    }

    #[test]
    fn sample_by() {
        let mut g = Gradient::new();
        g.add_key(0.5, RED);
        g.add_key(0.8, BLUE);
        const COUNT: usize = 256;
        let mut data: [Vec4; COUNT] = [Vec4::ZERO; COUNT];
        let start = 0.;
        let inc = 1. / COUNT as f32;
        g.sample_by(start, inc, &mut data[..]);
        for (i, &d) in data.iter().enumerate() {
            let ratio = inc.mul_add(i as f32, start);
            let expected = g.sample(ratio);
            assert!(color_approx_eq(expected, d, 1e-5));
        }
    }

    #[test]
    fn reflect() {
        let g = make_test_gradient();

        // Reflect
        let reflect: &dyn Reflect = &g;
        assert!(reflect.is::<Gradient<Vec4>>());
        let g_reflect = reflect.downcast_ref::<Gradient<Vec4>>();
        assert!(g_reflect.is_some());
        let g_reflect = g_reflect.unwrap();
        assert_eq!(*g_reflect, g);

        // FromReflect
        let g_from = Gradient::<Vec4>::from_reflect(reflect).unwrap();
        assert_eq!(g_from, g);

        // Struct
        assert!(g
            .type_name()
            .starts_with("bevy_hanabi::gradient::Gradient<")); // the Vec4 type name depends on platform
        let keys = g.field("keys").unwrap();
        let ReflectRef::List(keys) = keys.reflect_ref() else { panic!("Invalid type"); };
        assert_eq!(keys.len(), 3);
        for (i, (r, v)) in [(0.5, RED), (0.8, BLUE), (0.8, GREEN)].iter().enumerate() {
            let k = keys.get(i).unwrap();
            let gk = k.downcast_ref::<GradientKey<Vec4>>().unwrap();
            assert_approx_eq!(gk.ratio(), r);
            assert_approx_eq!(gk.value, v);

            let ReflectRef::Struct(k) = k.reflect_ref() else { panic!("Invalid type"); };
            assert!(k.type_name().contains("GradientKey"));
        }
    }

    #[test]
    fn serde() {
        let g = make_test_gradient();

        let s = ron::to_string(&g).unwrap();
        // println!("gradient: {:?}", s);
        let g_serde: Gradient<Vec4> = ron::from_str(&s).unwrap();
        assert_eq!(g, g_serde);
    }

    #[test]
    fn hash() {
        let g = make_test_gradient();
        let mut hasher = DefaultHasher::default();
        g.hash(&mut hasher);
        let h = hasher.finish();
        println!("gradient: {:?}\nhash: {:016X}", g, h);
    }
}
