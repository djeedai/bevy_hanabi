#![deny(
    warnings,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications,
    missing_docs
)]
#![allow(dead_code)] // TEMP
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

//! Hanabi -- a GPU particle system plugin for the Bevy game engine.
//!
//! This library provides a GPU-based particle system for the Bevy game engine.
//!
//! # Example
//!
//! Add the Hanabi plugin to your app:
//!
//! ```no_run
//! # use bevy::prelude::*;
//! # use bevy_hanabi::*;
//! App::default()
//!     .add_plugins(DefaultPlugins)
//!     .add_plugin(HanabiPlugin)
//!     .run();
//! ```
//!
//! Create an [`EffectAsset`] describing a visual effect, then add an
//! instance of that effect to an entity:
//!
//! ```
//! # use bevy::prelude::*;
//! # use bevy_hanabi::*;
//! fn setup(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
//!     // Define a color gradient from red to transparent black
//!     let mut gradient = Gradient::new();
//!     gradient.add_key(0.0, Vec4::new(1., 0., 0., 1.));
//!     gradient.add_key(1.0, Vec4::splat(0.));
//!
//!     // Create the effect asset
//!     let effect = effects.add(EffectAsset {
//!         name: "MyEffect".to_string(),
//!         // Maximum number of particles alive at a time
//!         capacity: 32768,
//!         // Spawn at a rate of 5 particles per second
//!         spawner: Spawner::rate(5.0.into()),
//!         ..Default::default()
//!     }
//!     // On spawn, randomly initialize the position and velocity
//!     // of the particle over a sphere of radius 2 units, with a
//!     // radial initial velocity of 6 units/sec away from the
//!     // sphere center.
//!     .init(PositionSphereModifier {
//!         center: Vec3::ZERO,
//!         radius: 2.,
//!         dimension: ShapeDimension::Surface,
//!         speed: 6.0.into(),
//!     })
//!     // Every frame, add a gravity-like acceleration downward
//!     .update(AccelModifier {
//!         accel: Vec3::new(0., -3., 0.),
//!     })
//!     // Render the particles with a color gradient over their
//!     // lifetime.
//!     .render(ColorOverLifetimeModifier { gradient })
//!     );
//!
//!     commands
//!         .spawn()
//!         .insert(Name::new("MyEffectInstance"))
//!         .insert_bundle(ParticleEffectBundle {
//!             effect: ParticleEffect::new(effect),
//!             transform: Transform::from_translation(Vec3::new(0., 1., 0.)),
//!             ..Default::default()
//!         });
//! }
//! ```

use bevy::{prelude::*, reflect::TypeUuid};

mod asset;
mod bundle;
mod gradient;
mod modifiers;
mod plugin;
mod render;
mod spawn;

#[cfg(test)]
mod test_utils;

pub use asset::EffectAsset;
pub use bundle::ParticleEffectBundle;
pub use gradient::{Gradient, GradientKey};
pub use modifiers::{
    AccelModifier, ColorOverLifetimeModifier, ForceFieldModifier, ForceFieldParam, InitModifier,
    ParticleTextureModifier, PositionCircleModifier, PositionSphereModifier, RenderModifier,
    ShapeDimension, SizeOverLifetimeModifier, UpdateModifier, FFNUM,
};
pub use plugin::HanabiPlugin;
pub use render::EffectCacheId;
pub use spawn::{Spawner, Value};

#[cfg(not(any(feature = "2d", feature = "3d")))]
compile_error!("Enable either the '2d' or '3d' feature.");

/// Extension trait to write a floating point scalar or vector constant in a format
/// matching the WGSL grammar.
///
/// This is required because WGSL doesn't support a floating point constant without
/// a decimal separator (e.g. `0.` instead of `0`), which would be what a regular float
/// to string formatting produces, but is interpreted as an integral type by WGSL.
///
/// # Example
///
/// ```
/// # use bevy_hanabi::ToWgslString;
/// let x = 2.0_f32;
/// assert_eq!("let x = 2.;", format!("let x = {};", x.to_wgsl_string()));
/// ```
pub trait ToWgslString {
    /// Convert a floating point scalar or vector to a string representing a WGSL constant.
    fn to_wgsl_string(&self) -> String;
}

impl ToWgslString for f32 {
    fn to_wgsl_string(&self) -> String {
        let s = format!("{:.6}", self);
        s.trim_end_matches('0').to_string()
    }
}

impl ToWgslString for f64 {
    fn to_wgsl_string(&self) -> String {
        let s = format!("{:.15}", self);
        s.trim_end_matches('0').to_string()
    }
}

impl ToWgslString for Vec2 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec2<f32>({0}, {1})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string()
        )
    }
}

impl ToWgslString for Vec3 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec3<f32>({0}, {1}, {2})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string(),
            self.z.to_wgsl_string()
        )
    }
}

impl ToWgslString for Vec4 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec4<f32>({0}, {1}, {2}, {3})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string(),
            self.z.to_wgsl_string(),
            self.w.to_wgsl_string()
        )
    }
}

impl ToWgslString for Value<f32> {
    fn to_wgsl_string(&self) -> String {
        match self {
            Self::Single(x) => x.to_wgsl_string(),
            Self::Uniform((a, b)) => format!(
                "rand() * ({1} - {0}) + {0}",
                a.to_wgsl_string(),
                b.to_wgsl_string(),
            ),
        }
    }
}

/// Visual effect made of particles.
///
/// The particle effect component represent a single instance of a visual effect. The
/// visual effect itself is described by a handle to an [`EffectAsset`]. This instance
/// is associated to an [`Entity`], inheriting its [`Transform`] as the origin frame
/// for its particle spawning.
#[derive(Debug, Clone, Component, TypeUuid)]
#[uuid = "c48df8b5-7eca-4d25-831e-513c2575cf6c"]
pub struct ParticleEffect {
    /// Handle of the effect to instantiate.
    handle: Handle<EffectAsset>,
    /// Internal effect cache ID of the effect once allocated.
    effect: EffectCacheId,
    /// Particle spawning descriptor.
    spawner: Option<Spawner>,
}

impl ParticleEffect {
    /// Create a new particle effect without a spawner or any modifier.
    pub fn new(handle: Handle<EffectAsset>) -> Self {
        ParticleEffect {
            handle,
            effect: EffectCacheId::INVALID,
            spawner: None,
        }
    }

    /// Sets the spawner of this particle effect.
    pub fn set_spawner(&mut self, spawner: Spawner) {
        self.spawner = Some(spawner);
    }

    /// Configure the spawner of a new particle effect.
    ///
    /// The call returns a reference to the added spawner, allowing to chain
    /// adding modifiers to the effect.
    pub fn spawner(&mut self, spawner: &Spawner) -> &mut Spawner {
        if self.spawner.is_none() {
            self.spawner = Some(*spawner);
        }
        self.spawner.as_mut().unwrap()
    }

    /// Get the spawner of this particle effect.
    ///
    /// Returns None if `with_spawner` was not called
    /// and the effect has not rendered yet.
    pub fn maybe_spawner(&mut self) -> Option<&mut Spawner> {
        self.spawner.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_wgsl_f32() {
        let s = 1.0_f32.to_wgsl_string();
        assert_eq!(s, "1.");
        let s = (-1.0_f32).to_wgsl_string();
        assert_eq!(s, "-1.");
        let s = 1.5_f32.to_wgsl_string();
        assert_eq!(s, "1.5");
        let s = 0.5_f32.to_wgsl_string();
        assert_eq!(s, "0.5");
        let s = 0.123_456_78_f32.to_wgsl_string();
        assert_eq!(s, "0.123457"); // 6 digits
    }

    #[test]
    fn to_wgsl_f64() {
        let s = 1.0_f64.to_wgsl_string();
        assert_eq!(s, "1.");
        let s = (-1.0_f64).to_wgsl_string();
        assert_eq!(s, "-1.");
        let s = 1.5_f64.to_wgsl_string();
        assert_eq!(s, "1.5");
        let s = 0.5_f64.to_wgsl_string();
        assert_eq!(s, "0.5");
        let s = 0.123_456_789_012_345_67_f64.to_wgsl_string();
        assert_eq!(s, "0.123456789012346"); // 15 digits
    }

    #[test]
    fn to_wgsl_vec() {
        let s = Vec2::new(1., 2.).to_wgsl_string();
        assert_eq!(s, "vec2<f32>(1., 2.)");
        let s = Vec3::new(1., 2., -1.).to_wgsl_string();
        assert_eq!(s, "vec3<f32>(1., 2., -1.)");
        let s = Vec4::new(1., 2., -1., 2.).to_wgsl_string();
        assert_eq!(s, "vec4<f32>(1., 2., -1., 2.)");
    }

    #[test]
    fn to_wgsl_value_f32() {
        let s = Value::Single(1.0_f32).to_wgsl_string();
        assert_eq!(s, "1.");
        let s = Value::Uniform((1.0_f32, 2.0_f32)).to_wgsl_string();
        assert_eq!(s, "rand() * (2. - 1.) + 1.");
    }
}
