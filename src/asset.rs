use bevy::{
    asset::{AssetLoader, Handle, LoadContext, LoadedAsset},
    ecs::reflect::ReflectResource,
    math::{Vec2, Vec3, Vec4},
    prelude::Resource,
    reflect::{Reflect, TypeUuid},
    render::texture::Image,
    utils::BoxedFuture,
};
use serde::{Deserialize, Serialize};

use crate::{
    modifier::{
        init::InitModifier,
        render::RenderModifier,
        update::{ForceFieldSource, UpdateModifier},
    },
    Gradient, Spawner,
};

/// Struct containing snippets of WSGL code that can be used
/// to define the initial conditions of particles on the GPU.
#[derive(Default, Clone)]
pub struct InitLayout {
    /// Code to define the initial position of particles.
    pub position_code: String,
    /// WSGL code to initialize interactions with force fields. (Unused?)
    pub force_field_code: String,
    /// WSGL code to set the initial lifetime of the particle.
    pub lifetime_code: String,
}

/// Struct containing snippets of WSGL code that can be used
/// to update the particles every frame on the GPU.
#[derive(Default, Clone, Copy)]
pub struct UpdateLayout {
    /// Constant acceleration to apply to all particles.
    /// Generally used to simulate some kind of gravity.
    pub accel: Vec3,
    /// Array of force field components with a maximum number of components
    /// determined by [`ForceFieldSource::MAX_SOURCES`].
    pub force_field: [ForceFieldSource; ForceFieldSource::MAX_SOURCES],
}

/// Struct containing data and snippets of WSGL code that can be used
/// to render the particles every frame on the GPU.
#[derive(Default, Clone)]
pub struct RenderLayout {
    /// If set, defines the PARTICLE_TEXTURE shader key and extend the vertex
    /// format to contain UV coordinates. Also make available the image as a
    /// 2D texture and sampler in the render shaders.
    pub particle_texture: Option<Handle<Image>>,
    /// Optional color gradient used to vary the particle color over its
    /// lifetime.
    pub lifetime_color_gradient: Option<Gradient<Vec4>>,
    /// Optional size gradient used to vary the particle size over its lifetime.
    pub size_color_gradient: Option<Gradient<Vec2>>,
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
#[derive(Default, Clone, Serialize, Deserialize, Resource, Reflect, TypeUuid)]
#[reflect(Resource)]
#[uuid = "249aefa4-9b8e-48d3-b167-3adf6c081c34"]
pub struct EffectAsset {
    /// Display name of the effect.
    pub name: String,
    /// Maximum number of concurrent particles.
    pub capacity: u32,
    /// Spawner.
    pub spawner: Spawner,
    /// Layout describing the particle initialize code.
    #[serde(skip)] // TODO
    #[reflect(ignore)] // TODO?
    pub init_layout: InitLayout,
    /// Layout describing the particle update code.
    #[serde(skip)] // TODO
    #[reflect(ignore)] // TODO?
    pub update_layout: UpdateLayout,
    /// Layout describing the particle rendering code.
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
    //#[serde(skip)] // TODO
    //modifiers: Vec<Box<dyn Modifier + Send + Sync + 'static>>,
}

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
