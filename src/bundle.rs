use crate::{EffectAsset, ParticleEffect};
use bevy::prelude::*;

/// A component bundle for a particle effect.
#[derive(Bundle, Clone)]
pub struct ParticleEffectBundle {
    pub effect: ParticleEffect,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    /// User indication of whether an entity is visible.
    pub visibility: Visibility,
    /// Algorithmically-computed indication of whether an entity is visible and should be extracted for rendering.
    pub computed_visibility: ComputedVisibility,
}

impl Default for ParticleEffectBundle {
    fn default() -> Self {
        Self::new(Handle::<EffectAsset>::default())
    }
}

impl ParticleEffectBundle {
    pub fn new(handle: Handle<EffectAsset>) -> Self {
        ParticleEffectBundle {
            effect: ParticleEffect::new(handle),
            transform: Default::default(),
            global_transform: Default::default(),
            visibility: Default::default(),
            computed_visibility: Default::default(),
        }
    }
}
