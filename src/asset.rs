use bevy::{
    asset::{AssetLoader, Handle, LoadContext, LoadedAsset},
    math::{Vec2, Vec3, Vec4},
    reflect::TypeUuid,
    render::texture::Image,
    utils::BoxedFuture,
};
use serde::{Deserialize, Serialize};

use crate::{
    modifiers::{ForceFieldParam, FFNUM},
    Gradient, InitModifier, RenderModifier, Spawner, UpdateModifier,
};

#[derive(Default, Clone)]
pub struct InitLayout {
    pub position_code: String,
    pub force_field_code: String,
}

#[derive(Default, Clone, Copy)]
pub struct UpdateLayout {
    /// Constant accelereation to apply to all particles.
    /// Generally used to simulate some kind of gravity.
    pub accel: Vec3,
    /// Array of force field components with a maximum number of components determined by [`FFNUM`].
    pub force_field: [ForceFieldParam; FFNUM],
}

#[derive(Default, Clone)]
pub struct RenderLayout {
    /// If set, defines the PARTICLE_TEXTURE shader key and extend the vertex format to contain
    /// UV coordinates. Also make available the image as a 2D texture and sampler in the render
    /// shaders.
    pub particle_texture: Option<Handle<Image>>,

    pub lifetime_color_gradient: Option<Gradient<Vec4>>,

    pub size_color_gradient: Option<Gradient<Vec2>>,
}

/// Asset describing a visual effect.
///
/// The effect can be instanciated with a [`ParticleEffect`] component, or a [`ParticleEffectBundle`].
///
/// [`ParticleEffect`]: crate::ParticleEffect
/// [`ParticleEffectBundle`]: crate::ParticleEffectBundle
#[derive(Default, Serialize, Deserialize, TypeUuid)]
#[uuid = "249aefa4-9b8e-48d3-b167-3adf6c081c34"]
pub struct EffectAsset {
    /// Display name of the effect.
    pub name: String,
    /// Maximum number of concurrent particles.
    pub capacity: u32,
    /// Spawner.
    pub spawner: Spawner,
    ///
    #[serde(skip)] // TODO
    pub init_layout: InitLayout,
    ///
    #[serde(skip)] // TODO
    pub update_layout: UpdateLayout,
    ///
    #[serde(skip)] // TODO
    pub render_layout: RenderLayout,
}
///
//#[serde(skip)] // TODO
//modifiers: Vec<Box<dyn Modifier + Send + Sync + 'static>>,

impl EffectAsset {
    /// Add an initialization modifier to the effect.
    pub fn init<M: InitModifier + Send + Sync + 'static>(mut self, modifier: M) -> Self {
        modifier.apply(&mut self.init_layout);
        //self.modifiers.push(Box::new(modifier));
        self
    }

    /// Add an update modifier to the effect.
    pub fn update<M: UpdateModifier + Send + Sync + 'static>(mut self, modifier: M) -> Self {
        modifier.apply(&mut self.update_layout);
        //self.modifiers.push(Box::new(modifier));
        self
    }

    /// Add a render modifier to the effect.
    pub fn render<M: RenderModifier + Send + Sync + 'static>(mut self, modifier: M) -> Self {
        modifier.apply(&mut self.render_layout);
        //self.modifiers.push(Box::new(modifier));
        self
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
