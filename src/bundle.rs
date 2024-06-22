use bevy::prelude::*;

use crate::{CompiledParticleEffect, EffectAsset, EffectProperties, ParticleEffect};

/// A component bundle for a particle effect.
///
/// This bundle contains all necessary components for a [`ParticleEffect`] to
/// function correctly, and is the preferred method for spawning a new
/// [`ParticleEffect`].
///
/// [`EffectProperties`]: crate::EffectProperties
#[derive(Default, Bundle, Clone)]
pub struct ParticleEffectBundle {
    /// The particle effect instance itself.
    pub effect: ParticleEffect,
    /// A compiled version of the particle effect, managed automatically.
    ///
    /// You don't need to interact with this component, but it must be present
    /// for the effect to work. This is split from the [`ParticleEffect`] itself
    /// mainly for change detection reasons, as well as for semantic.
    pub compiled_effect: CompiledParticleEffect,
    /// Runtime storage for effect properties.
    ///
    /// Stores the current property values, before they're uploaded to GPU. Can
    /// be modified manually to set a new property value.
    pub effect_properties: EffectProperties,
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
    /// Invisible entities do not process any particles if their
    /// [`EffectAsset::simulation_condition`] is set to
    /// [`SimulationCondition::WhenVisible`], which is the default. This makes
    /// it efficient to temporarily disable an effect instance.
    ///
    /// If your effect uses [`SimulationCondition::Always`] then this component
    /// is not necessary and you can remove it (spawn components manually
    /// instead of using this bundle).
    ///
    /// [`EffectAsset::simulation_condition`]: crate::EffectAsset::simulation_condition
    /// [`SimulationCondition::WhenVisible`]: crate::SimulationCondition::WhenVisible
    /// [`SimulationCondition::Always`]: crate::SimulationCondition::Always
    pub visibility: Visibility,
    /// Algorithmically-computed indication of whether an entity is visible in
    /// the entity hierarchy and should be extracted for rendering.
    ///
    /// If your effect uses [`SimulationCondition::Always`] then this component
    /// is not necessary and you can remove it (spawn components manually
    /// instead of using this bundle).
    ///
    /// Users should not interact with this component manually, but it is
    /// required by Bevy's built-in visibility system.
    ///
    /// [`SimulationCondition::Always`]: crate::SimulationCondition::Always
    pub inherited_visibility: InheritedVisibility,
    /// Algorithmically-computed indication of whether an entity is visible in
    /// the current camera view and should be extracted for rendering.
    ///
    /// If your effect uses [`SimulationCondition::Always`] then this component
    /// is not necessary and you can remove it (spawn components manually
    /// instead of using this bundle).
    ///
    /// Users should not interact with this component manually, but it is
    /// required by Bevy's built-in visibility system.
    ///
    /// [`SimulationCondition::Always`]: crate::SimulationCondition::Always
    pub view_visibility: ViewVisibility,
}

impl ParticleEffectBundle {
    /// Create a new particle effect bundle from an effect description.
    pub fn new(handle: Handle<EffectAsset>) -> Self {
        Self {
            effect: ParticleEffect::new(handle),
            compiled_effect: CompiledParticleEffect::default(),
            effect_properties: EffectProperties::default(),
            transform: Default::default(),
            global_transform: Default::default(),
            visibility: Default::default(),
            inherited_visibility: InheritedVisibility::default(),
            view_visibility: ViewVisibility::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_default() {
        let handle = Handle::<EffectAsset>::default();
        let bundle = ParticleEffectBundle::default();
        assert_eq!(bundle.effect.handle, handle);
    }

    #[test]
    fn bundle_new() {
        let handle = Handle::default();
        let bundle = ParticleEffectBundle::new(handle.clone());
        assert_eq!(bundle.effect.handle, handle);
    }
}
