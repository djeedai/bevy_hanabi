use crate::{EffectAsset, ParticleEffect, Spawner};
use bevy::prelude::*;

/// A component bundle for a particle effect.
///
/// This bundle contains all necessary components for a [`ParticleEffect`] to
/// function correctly, and is the preferred method for spawning a new
/// [`ParticleEffect`].
#[derive(Bundle, Clone)]
pub struct ParticleEffectBundle {
    /// The particle effect instance itself.
    pub effect: ParticleEffect,
    /// Transform of the entity, representing the frame of reference for the
    /// particle emission.
    ///
    /// New particles are emitted relative to this transform, ignoring the
    /// scale.
    pub transform: Transform,
    /// Computed global transform.
    ///
    /// Users should not interact with this component manually, but it is
    /// required by Bevy's built-in transform system.
    pub global_transform: GlobalTransform,
    /// User indication of whether an entity is visible.
    ///
    /// Invisible entities do not process any particles, making it efficient to
    /// temporarily disable an effect instance.
    pub visibility: Visibility,
    /// Algorithmically-computed indication of whether an entity is visible and
    /// should be extracted for rendering.
    ///
    /// Users should not interact with this component manually, but it is
    /// required by Bevy's built-in visibility system.
    pub computed_visibility: ComputedVisibility,
}

impl Default for ParticleEffectBundle {
    fn default() -> Self {
        Self::new(Handle::<EffectAsset>::default())
    }
}

impl ParticleEffectBundle {
    /// Create a new particle effect bundle from an effect description.
    pub fn new(handle: Handle<EffectAsset>) -> Self {
        ParticleEffectBundle {
            effect: ParticleEffect::new(handle),
            transform: Default::default(),
            global_transform: Default::default(),
            visibility: Default::default(),
            computed_visibility: Default::default(),
        }
    }

    /// Gives the particle effect a spawner.
    pub fn with_spawner(mut self, spawner: Spawner) -> Self {
        self.effect.set_spawner(spawner);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{asset::HandleId, reflect::TypeUuid};

    #[test]
    fn bundle_default() {
        let handle = Handle::<EffectAsset>::default();
        let bundle = ParticleEffectBundle::default();
        assert_eq!(bundle.effect.handle, handle);
    }

    #[test]
    fn bundle_new() {
        let handle = Handle::weak(HandleId::new(EffectAsset::TYPE_UUID, 42));
        let bundle = ParticleEffectBundle::new(handle.clone());
        assert_eq!(bundle.effect.handle, handle);
    }

    #[test]
    fn bundle_with_spawner() {
        let spawner = Spawner::once(5.0.into(), true);
        let mut bundle = ParticleEffectBundle::default().with_spawner(spawner);
        assert!(bundle.effect.maybe_spawner().is_some());
        assert_eq!(*bundle.effect.maybe_spawner().unwrap(), spawner);
    }
}
