use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    reflect::{FromReflect, Reflect, TypeUuid},
    utils::{BoxedFuture, HashSet},
};
use serde::{Deserialize, Serialize};

use crate::{
    graph::Value,
    modifier::{init::InitModifier, render::RenderModifier, update::UpdateModifier},
    BoxedModifier, ParticleLayout, Property, PropertyLayout, SimulationSpace, Spawner,
};

/// Type of motion integration applied to the particles of a system.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, FromReflect, Serialize, Deserialize,
)]
pub enum MotionIntegration {
    /// No motion integration. The [`Attribute::POSITION`] of the particles
    /// needs to be explicitly assigned by a modifier for the particles to move.
    ///
    /// [`Attribute::POSITION`]: crate::Attribute::POSITION
    None,

    /// Apply Euler motion integration each simulation update before all
    /// modifiers are applied.
    ///
    /// Not to be confused with Bevy's `PreUpdate` phase. Here "update" refers
    /// to the particle update on the GPU via a compute shader.
    PreUpdate,

    /// Apply Euler motion integration each simulation update after all
    /// modifiers are applied. This is the default.
    ///
    /// Not to be confused with Bevy's `PostUpdate` phase. Here "update" refers
    /// to the particle update on the GPU via a compute shader.
    #[default]
    PostUpdate,
}

/// Simulation condition for an effect.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, FromReflect, Serialize, Deserialize,
)]
pub enum SimulationCondition {
    /// Simulate the effect only when visible.
    ///
    /// The visibility is determined by the [`Visibility`] and
    /// [`ComputedVisibility`] components.
    ///
    /// This is the default for all assets, and is the most performant option,
    /// allowing to have many effects in the scene without the need to simulate
    /// all of them if they're not visible.
    ///
    /// Note: AABB culling is not currently available. Only boolean ON/OFF
    /// visibility is used.
    ///
    /// [`Visibility`]: bevy::render::view::Visibility
    /// [`ComputedVisibility`]: bevy::render::view::ComputedVisibility
    #[default]
    WhenVisible,

    /// Always simulate the effect, whether visible or not.
    ///
    /// For performance reasons, it's recommended to only simulate visible
    /// particle effects (that is, use [`SimulationCondition::WhenVisible`]).
    /// However occasionally it may be needed to continue the simulation
    /// when the effect is not visible, to ensure some temporal continuity when
    /// the effect is made visible again. This is an uncommon case, and you
    /// should be aware of the performance implications of using this
    /// condition, and only use it when strictly necessary.
    ///
    /// Any [`Visibility`] or [`ComputedVisibility`] component is ignored. You
    /// may want to spawn the particle effect components manually instead of
    /// using the [`ParticleEffectBundle`] to avoid adding those components.
    ///
    /// [`Visibility`]: bevy::render::view::Visibility
    /// [`ComputedVisibility`]: bevy::render::view::ComputedVisibility
    /// [`ParticleEffectBundle`]: crate::ParticleEffectBundle
    Always,
}

/// Asset describing a visual effect.
///
/// The effect can be instanciated with a [`ParticleEffect`] component, or a
/// [`ParticleEffectBundle`].
///
/// [`ParticleEffect`]: crate::ParticleEffect
/// [`ParticleEffectBundle`]: crate::ParticleEffectBundle
#[derive(Default, Clone, TypeUuid, Reflect, FromReflect, Serialize, Deserialize)]
#[uuid = "249aefa4-9b8e-48d3-b167-3adf6c081c34"]
pub struct EffectAsset {
    /// Display name of the effect.
    ///
    /// This has no internal use, and is mostly for the user to identify an
    /// effect or for display in some tool UI.
    pub name: String,
    /// Maximum number of concurrent particles.
    ///
    /// The capacity is the maximum number of particles that can be alive at the
    /// same time. It determines the size of various GPU resources, most notably
    /// the particle buffer itself. To prevent wasting GPU resources, users
    /// should keep this quantity as close as possible to the maximum number of
    /// particles they expect to render.
    pub capacity: u32,
    /// Spawner.
    pub spawner: Spawner,
    /// For 2D rendering, the Z coordinate used as the sort key.
    ///
    /// This value is passed to the render pipeline and used when sorting
    /// transparent items to render, to order them. As a result, effects
    /// with different Z values cannot be batched together, which may
    /// negatively affect performance.
    ///
    /// Ignored for 3D rendering.
    pub z_layer_2d: f32,
    /// Particle simulation space.
    pub simulation_space: SimulationSpace,
    /// Condition under which the effect is simulated.
    pub simulation_condition: SimulationCondition,
    /// Modifiers defining the effect.
    #[reflect(ignore)]
    // TODO - Can't manage to implement FromReflect for BoxedModifier in a nice way yet
    pub modifiers: Vec<BoxedModifier>,
    /// Properties of the effect.
    ///
    /// Properties must have a unique name. Manually adding two or more
    /// properties with the same name will result in an invalid asset and
    /// undefined behavior are runtime. Prefer using the [`with_property()`] and
    /// [`add_property()`] methods for safety.
    ///
    /// [`with_property()`]: crate::EffectAsset::with_property
    /// [`add_property()`]: crate::EffectAsset::add_property
    pub properties: Vec<Property>,
    /// Type of motion integration applied to the particles of a system.
    pub motion_integration: MotionIntegration,
}

impl EffectAsset {
    /// Add a new property to the asset.
    ///
    /// # Panics
    ///
    /// Panics if a property with the same name already exists.
    pub fn with_property(mut self, name: impl Into<String>, default_value: Value) -> Self {
        self.add_property(name, default_value);
        self
    }

    /// Add a new property to the asset.
    ///
    /// # Panics
    ///
    /// Panics if a property with the same name already exists.
    pub fn add_property(&mut self, name: impl Into<String>, default_value: Value) {
        let name = name.into();
        assert!(!self.properties.iter().any(|p| p.name() == name));
        self.properties.push(Property::new(name, default_value));
    }

    /// Get the list of existing properties.
    pub fn properties(&self) -> &[Property] {
        &self.properties
    }

    /// Add an initialization modifier to the effect.
    ///
    /// # Panics
    ///
    /// Panics if the modifier references a property which doesn't exist.
    /// You should declare an effect property first with [`with_property()`]
    /// or [`add_property()`], before adding any modifier referencing it.
    ///
    /// [`with_property()`]: crate::EffectAsset::with_property
    /// [`add_property()`]: crate::EffectAsset::add_property
    pub fn init<M>(mut self, mut modifier: M) -> Self
    where
        M: InitModifier + Send + Sync + 'static,
    {
        modifier.resolve_properties(&self.properties);
        self.modifiers.push(Box::new(modifier));
        self
    }

    /// Add an update modifier to the effect.
    ///
    /// # Panics
    ///
    /// Panics if the modifier references a property which doesn't exist.
    /// You should declare an effect property first with [`with_property()`]
    /// or [`add_property()`], before adding any modifier referencing it.
    ///
    /// [`with_property()`]: crate::EffectAsset::with_property
    /// [`add_property()`]: crate::EffectAsset::add_property
    pub fn update<M>(mut self, mut modifier: M) -> Self
    where
        M: UpdateModifier + Send + Sync + 'static,
    {
        modifier.resolve_properties(&self.properties);
        self.modifiers.push(Box::new(modifier));
        self
    }

    /// Add a render modifier to the effect.
    ///
    /// # Panics
    ///
    /// Panics if the modifier references a property which doesn't exist.
    /// You should declare an effect property first with [`with_property()`]
    /// or [`add_property()`], before adding any modifier referencing it.
    ///
    /// [`with_property()`]: crate::EffectAsset::with_property
    /// [`add_property()`]: crate::EffectAsset::add_property
    pub fn render<M>(mut self, mut modifier: M) -> Self
    where
        M: RenderModifier + Send + Sync + 'static,
    {
        modifier.resolve_properties(&self.properties);
        self.modifiers.push(Box::new(modifier));
        self
    }

    /// Build the particle layout of the asset based on its modifiers.
    pub fn particle_layout(&self) -> ParticleLayout {
        // Build the set of unique attributes required for all modifiers
        let mut set = HashSet::new();
        for modifier in &self.modifiers {
            for &attr in modifier.attributes() {
                set.insert(attr);
            }
        }

        // Build the layout
        let mut layout = ParticleLayout::new();
        for attr in set {
            layout = layout.append(attr);
        }
        layout.build()
    }

    /// Build the property layout of the asset based on its properties.
    pub fn property_layout(&self) -> PropertyLayout {
        PropertyLayout::new(self.properties.iter())
    }
}

#[derive(Default)]
pub struct EffectAssetLoader;

impl AssetLoader for EffectAssetLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<(), anyhow::Error>> {
        Box::pin(async move {
            let custom_asset = ron::de::from_bytes::<EffectAsset>(bytes)?;
            load_context.set_default_asset(LoadedAsset::new(custom_asset));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["effect"]
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    use super::*;

    #[test]
    fn property() {
        let mut effect = EffectAsset {
            name: "Effect".into(),
            capacity: 4096,
            spawner: Spawner::rate(30.0.into()),
            ..Default::default()
        }
        .with_property("my_prop", 345_u32.into());

        effect.add_property(
            "other_prop",
            graph::Value::Float3(Vec3::new(3., -7.5, 42.42)),
        );

        assert!(effect.properties().iter().any(|p| p.name() == "my_prop"));
        assert!(effect.properties().iter().any(|p| p.name() == "other_prop"));
        assert!(!effect
            .properties()
            .iter()
            .any(|p| p.name() == "do_not_exist"));

        let layout = effect.property_layout();
        assert_eq!(layout.size(), 16);
        assert_eq!(layout.align(), 16);
        assert_eq!(layout.offset("my_prop"), Some(12));
        assert_eq!(layout.offset("other_prop"), Some(0));
        assert_eq!(layout.offset("unknown"), None);
    }

    #[test]
    fn test_apply_modifiers() {
        let effect = EffectAsset {
            name: "Effect".into(),
            capacity: 4096,
            spawner: Spawner::rate(30.0.into()),
            ..Default::default()
        }
        .init(InitPositionSphereModifier::default())
        .init(InitVelocitySphereModifier::default())
        .init(InitAgeModifier::default())
        .init(InitLifetimeModifier::default())
        //.update(AccelModifier::default())
        .update(LinearDragModifier::default())
        .update(ForceFieldModifier::default())
        .render(ParticleTextureModifier::default())
        .render(ColorOverLifetimeModifier::default())
        .render(SizeOverLifetimeModifier::default())
        .render(BillboardModifier::default());

        assert_eq!(effect.capacity, 4096);

        let mut init_context = InitContext::default();
        InitPositionSphereModifier::default().apply(&mut init_context);
        InitVelocitySphereModifier::default().apply(&mut init_context);
        InitAgeModifier::default().apply(&mut init_context);
        InitLifetimeModifier::default().apply(&mut init_context);
        // assert_eq!(effect., init_context.init_code);

        let mut update_context = UpdateContext::default();
        AccelModifier::constant(Vec3::ONE).apply(&mut update_context);
        LinearDragModifier::default().apply(&mut update_context);
        ForceFieldModifier::default().apply(&mut update_context);
        // assert_eq!(effect.update_layout, update_layout);

        let mut render_context = RenderContext::default();
        ParticleTextureModifier::default().apply(&mut render_context);
        ColorOverLifetimeModifier::default().apply(&mut render_context);
        SizeOverLifetimeModifier::default().apply(&mut render_context);
        BillboardModifier::default().apply(&mut render_context);
        // assert_eq!(effect.render_layout, render_layout);
    }

    #[test]
    fn test_serde_ron() {
        let effect = EffectAsset {
            name: "Effect".into(),
            capacity: 4096,
            spawner: Spawner::rate(30.0.into()),
            ..Default::default()
        };

        let s = ron::to_string(&effect).unwrap();
        let _effect_serde: EffectAsset = ron::from_str(&s).unwrap();
        // assert_eq!(effect, effect_serde);
    }
}
