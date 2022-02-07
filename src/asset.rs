use bevy::{
    asset::{AssetLoader, Handle, LoadContext, LoadedAsset},
    reflect::TypeUuid,
    render::texture::Image,
    utils::BoxedFuture,
};
use serde::{Deserialize, Serialize};

use crate::{Gradient, Modifier, Spawner};

#[derive(Default, Clone)]
pub struct RenderLayout {
    /// If set, defines the PARTICLE_TEXTURE shader key and extend the vertex format to contain
    /// UV coordinates. Also make available the image as a 2D texture and sampler in the render
    /// shaders.
    pub particle_texture: Option<Handle<Image>>,

    pub lifetime_color_gradient: Option<Gradient>,
}

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
    pub render_layout: RenderLayout,
}
///
//#[serde(skip)] // TODO
//modifiers: Vec<Box<dyn Modifier + Send + Sync + 'static>>,

impl EffectAsset {
    pub fn with<M: Modifier + Send + Sync + 'static>(mut self, modifier: M) -> Self {
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
