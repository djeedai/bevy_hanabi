#![deny(
    //warnings,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    //unused_import_braces,
    unused_qualifications,
    //missing_docs
)]
#![allow(dead_code)] // TEMP

//! Hanabi -- a particle system plugin for the Bevy game engine.
//!
//! This library provides a particle system for the Bevy game engine.
//!
//! # Example
//!
//! Add the Hanabi plugin to your app:
//!
//! ```rust
//! # use bevy::prelude::*;
//! # use bevy_hanabi::*;
//! App::default()
//!     .add_plugins(DefaultPlugins)
//!     .add_plugin(HanabiPlugin)
//!     .run();
//! ```

use bevy::{prelude::*, reflect::TypeUuid};

mod asset;
mod bundle;
mod gradient;
mod modifiers;
mod plugin;
mod render;
mod spawn;

pub use asset::EffectAsset;
pub use bundle::ParticleEffectBundle;
pub use gradient::{Gradient, GradientKey};
pub use modifiers::{ColorOverLifetimeModifier, Modifier, ParticleTextureModifier, RenderModifier};
pub use plugin::HanabiPlugin;
pub use render::EffectCacheId;
pub use spawn::{SpawnCount, SpawnMode, SpawnRate, Spawner, Value};

#[derive(Debug, Clone, Component, TypeUuid)]
#[uuid = "c48df8b5-7eca-4d25-831e-513c2575cf6c"]
pub struct ParticleEffect {
    /// Handle of the effect to instantiate.
    handle: Handle<EffectAsset>,
    /// Internal effect cache ID of the effect once allocated.
    effect: EffectCacheId,
    ///
    spawner: Option<Spawner>,
}

impl ParticleEffect {
    pub fn new(handle: Handle<EffectAsset>) -> Self {
        ParticleEffect {
            handle,
            effect: EffectCacheId::INVALID,
            spawner: None,
        }
    }

    pub fn spawner(&mut self, spawner: &Spawner) -> &mut Spawner {
        if self.spawner.is_none() {
            self.spawner = Some(spawner.clone());
        }
        self.spawner.as_mut().unwrap()
    }
}
