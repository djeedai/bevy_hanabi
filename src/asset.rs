use bevy::{
    asset::{AssetLoader, Handle, LoadContext, LoadedAsset},
    math::{Vec2, Vec4},
    reflect::{FromReflect, Reflect, TypeUuid},
    render::texture::Image,
    utils::{BoxedFuture, HashSet},
};
use serde::{Deserialize, Serialize};

use crate::{
    graph::Value,
    modifier::{init::InitModifier, render::RenderModifier, update::UpdateModifier},
    Attribute, BoxedModifier, Gradient, ParticleLayout, Property, PropertyLayout, SimulationSpace,
    Spawner,
};

/// Struct containing data and snippets of WSGL code that can be used
/// to render the particles every frame on the GPU.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RenderLayout {
    /// If set, defines the PARTICLE_TEXTURE shader key and extend the vertex
    /// format to contain UV coordinates. Also make available the image as a
    /// 2D texture and sampler in the render shaders.
    pub particle_texture: Option<Handle<Image>>,
    /// Optional color gradient used to vary the particle color over its
    /// lifetime.
    pub lifetime_color_gradient: Option<Gradient<Vec4>>,
    /// Optional size gradient used to vary the particle size over its lifetime.
    pub lifetime_size_gradient: Option<Gradient<Vec2>>,
    /// If true, renders sprites as "billboards", that is, they will always face
    /// the camera when rendered.
    pub billboard: bool,
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
    /// Layout describing the particle rendering code.
    ///
    /// The render layout determines how alive particles are rendered.
    /// Compatible layouts increase the chance of batching together effects.
    #[serde(skip)] // TODO
    #[reflect(ignore)] // TODO?
    pub render_layout: RenderLayout,
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
    /// Modifiers defining the effect.
    #[reflect(ignore)]
    // TODO - Can't manage to implement FromReflect for BoxedModifier in a nice way yet
    pub modifiers: Vec<BoxedModifier>,
    /// Properties of the effect.
    pub(crate) properties: Vec<Property>,
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
        assert!(self.properties.iter().find(|&p| p.name() == name).is_none());
        self.properties.push(Property::new(name, default_value));
    }

    /// Get the list of existing properties.
    pub(crate) fn properties(&self) -> &[Property] {
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
        modifier.apply(&mut self.render_layout);
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

        // For legacy compatibility reasons, and because both the motion integration and
        // the particle aging are currently non-optional, add some default attributes.
        set.insert(Attribute::POSITION);
        set.insert(Attribute::AGE);
        set.insert(Attribute::VELOCITY);
        set.insert(Attribute::LIFETIME);

        // Build the layout
        let mut layout = ParticleLayout::new();
        for attr in set {
            layout = layout.add(attr);
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
    fn test_apply_modifiers() {
        let effect = EffectAsset {
            name: "Effect".into(),
            capacity: 4096,
            spawner: Spawner::rate(30.0.into()),
            ..Default::default()
        }
        //.init(PositionCircleModifier::default())
        .init(PositionSphereModifier::default())
        //.init(PositionCone3dModifier::default())
        .init(ParticleLifetimeModifier::default())
        //.update(AccelModifier::default())
        .update(LinearDragModifier::default())
        .update(ForceFieldModifier::default())
        .render(ParticleTextureModifier::default())
        .render(ColorOverLifetimeModifier::default())
        .render(SizeOverLifetimeModifier::default())
        .render(BillboardModifier::default());

        assert_eq!(effect.capacity, 4096);

        let mut init_context = InitContext::default();
        PositionSphereModifier::default().apply(&mut init_context);
        ParticleLifetimeModifier::default().apply(&mut init_context);
        //assert_eq!(effect., init_context.init_code);

        let mut update_context = UpdateContext::default();
        //AccelModifier::default().apply(&mut update_context);
        LinearDragModifier::default().apply(&mut update_context);
        ForceFieldModifier::default().apply(&mut update_context);
        //assert_eq!(effect.update_layout, update_layout);

        let mut render_layout = RenderLayout::default();
        ParticleTextureModifier::default().apply(&mut render_layout);
        ColorOverLifetimeModifier::default().apply(&mut render_layout);
        SizeOverLifetimeModifier::default().apply(&mut render_layout);
        BillboardModifier::default().apply(&mut render_layout);
        assert_eq!(effect.render_layout, render_layout);
    }

    // #[test]
    // fn test_serde_ron() {
    //     let effect = EffectAsset {
    //         name: "Effect".into(),
    //         capacity: 4096,
    //         spawner: Spawner::rate(30.0.into()),
    //         ..Default::default()
    //     };

    //     let s = ron::to_string(&effect).unwrap();
    //     let effect_serde: EffectAsset = ron::from_str(&s).unwrap();
    //     assert_eq!(effect, effect_serde);
    // }
}
