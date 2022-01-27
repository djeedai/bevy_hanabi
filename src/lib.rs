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
//! AppBuilder::default()
//!     .add_default_plugins()
//!     .add_plugin(HanabiPlugin)
//!     .run();
//! ```
//!
//! Animate the position ([`Transform::translation`]) of an [`Entity`]:
//!
//! ```rust
//! # use bevy_tweening::*;
//! # use std::time::Duration;
//! commands
//!     // Spawn a Sprite entity to animate the position of
//!     .spawn_bundle(SpriteBundle {
//!         material: materials.add(Color::RED.into()),
//!         sprite: Sprite {
//!             size: Vec2::new(size, size),
//!             ..Default::default()
//!         },
//!         ..Default::default()
//!     })
//!     // Add an Animator component to perform the animation
//!     .insert(Animator::new(
//!         // Use a quadratic easing on both endpoints
//!         EaseFunction::QuadraticInOut,
//!         // Loop animation back and forth over 1 second, with a 0.5 second
//!         // pause after each cycle (start -> end -> start).
//!         TweeningType::PingPong {
//!             duration: Duration::from_secs(1),
//!             pause: Some(Duration::from_millis(500)),
//!         },
//!         // The lens gives access to the Transform component of the Sprite,
//!         // for the Animator to animate it. It also contains the start and
//!         // end values associated with the animation ratios 0. and 1.
//!         TransformPositionLens {
//!             start: Vec3::new(0., 0., 0.),
//!             end: Vec3::new(1., 2., -4.),
//!         },
//!     ));
//! ```
//!
//! # Animators and lenses
//!
//! Bevy components and assets are animated with tweening animator components. Those animators determine
//! the fields to animate using lenses.
//!
//! ## Components animation
//!
//! Components are animated with the [`Animator`] component, which is generic over the type of component
//! it animates. This is a restriction imposed by Bevy, to access the animated component as a mutable
//! reference via a [`Query`] and comply with the ECS rules.
//!
//! The [`Animator`] itself is not generic over the subset of fields of the components it animates.
//! This limits the proliferation of generic types when animating e.g. both the position and rotation
//! of an entity.
//!
//! ## Assets animation
//!
//! Assets are animated in a similar way to component, via the [`AssetAnimator`] component. Because assets
//! are typically shared, and the animation applies to the asset itself, all users of the asset see the
//! animation. For example, animating the color of a [`ColorMaterial`] will change the color of all [`Sprite`]
//! components using that material.
//!
//! ## Lenses
//!
//! Both [`Animator`] and [`AssetAnimator`] access the field(s) to animate via a lens, a type that implements
//! the [`Lens`] trait. Several predefined lenses are provided for the most commonly animated fields, like the
//! components of a [`Transform`]. A custom lens can also be created by implementing the trait, allowing to
//! animate virtually any field of any Bevy component or asset.
//!
//! [`Transform::translation`]: bevy::transform::components::Transform::translation
//! [`Entity`]: bevy::ecs::entity::Entity
//! [`Query`]: bevy::ecs::system::Query
//! [`ColorMaterial`]: bevy::sprite::ColorMaterial
//! [`Sprite`]: bevy::sprite::Sprite
//! [`Transform`]: bevy::transform::components::Transform

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
