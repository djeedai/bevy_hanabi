use bevy::{
    asset::{AssetLoader, Handle, LoadContext, LoadedAsset},
    ecs::{reflect::ReflectResource, system::Resource},
    math::{Vec2, Vec3, Vec4},
    reflect::{Reflect, TypeUuid},
    render::texture::Image,
    utils::BoxedFuture,
};
use serde::{Deserialize, Serialize};

use crate::{
    attributes::ParticleLayout,
    modifier::{
        init::InitModifier,
        render::RenderModifier,
        update::{ForceFieldSource, UpdateModifier},
    },
    Gradient, Spawner,
};

/// Struct containing snippets of WSGL code that can be used
/// to define the initial conditions of particles on the GPU.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InitLayout {
    /// Code to define the initial position of particles.
    pub position_code: String,
    /// WSGL code to set the initial lifetime of the particle.
    pub lifetime_code: String,
}

/// Struct containing snippets of WSGL code that can be used
/// to update the particles every frame on the GPU.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct UpdateLayout {
    /// Constant acceleration to apply to all particles.
    /// Generally used to simulate some kind of gravity.
    pub accel: Vec3,
    /// Linear drag coefficient.
    ///
    /// Amount of (linear) drag force applied to the particles each frame, as a
    /// fraction of the particle's acceleration. Higher values slow down the
    /// particle in a shorter amount of time. Defaults to zero (disabled; no
    /// drag force).
    pub drag_coefficient: f32,
    /// Array of force field components with a maximum number of components
    /// determined by [`ForceFieldSource::MAX_SOURCES`].
    pub force_field: [ForceFieldSource; ForceFieldSource::MAX_SOURCES],
}

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
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize, Resource, Reflect, TypeUuid)]
#[reflect(Resource)]
#[uuid = "249aefa4-9b8e-48d3-b167-3adf6c081c34"]
pub struct EffectAsset {
    /// Display name of the effect.
    ///
    /// This has no internal use, and is mostly for the user to identify an
    /// effect or for display is some tool UI.
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
    /// Layout describing the attributes used by particles.
    #[serde(skip)] // TODO
    #[reflect(ignore)] // TODO?
    particle_layout: ParticleLayout,
    /// Layout describing the particle initialize code.
    ///
    /// The initialize layout determines how new particles are initialized when
    /// spawned. Compatible layouts increase the chance of batching together
    /// effects.
    #[serde(skip)] // TODO
    #[reflect(ignore)] // TODO?
    pub init_layout: InitLayout,
    /// Layout describing the particle update code.
    ///
    /// The update layout determines how all alive particles are updated each
    /// frame. Compatible layouts increase the chance of batching together
    /// effects.
    #[serde(skip)] // TODO
    #[reflect(ignore)] // TODO?
    pub update_layout: UpdateLayout,
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
}

impl EffectAsset {
    /// Add an initialization modifier to the effect.
    pub fn init<M: InitModifier + Send + Sync + 'static>(mut self, modifier: M) -> Self {
        self.particle_layout = self.particle_layout.merged_with(modifier.attributes());
        modifier.apply(&mut self.init_layout);
        //self.modifiers.push(Box::new(modifier));
        self
    }

    /// Add an update modifier to the effect.
    pub fn update<M: UpdateModifier + Send + Sync + 'static>(mut self, modifier: M) -> Self {
        self.particle_layout = self.particle_layout.merged_with(modifier.attributes());
        modifier.apply(&mut self.update_layout);
        //self.modifiers.push(Box::new(modifier));
        self
    }

    /// Add a render modifier to the effect.
    pub fn render<M: RenderModifier + Send + Sync + 'static>(mut self, modifier: M) -> Self {
        self.particle_layout = self.particle_layout.merged_with(modifier.attributes());
        modifier.apply(&mut self.render_layout);
        //self.modifiers.push(Box::new(modifier));
        self
    }

    /// Get the particle layout of this effect.
    pub fn particle_layout(&self) -> &ParticleLayout {
        &self.particle_layout
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
    use crate::{attributes::Attribute, *};

    use super::*;

    #[test]
    fn test_apply_modifiers() {
        let effect = EffectAsset {
            name: "Effect".into(),
            capacity: 4096,
            spawner: Spawner::rate(30.0.into()),
            particle_layout: ParticleLayout::new()
                .add(Attribute::POSITION)
                .add(Attribute::AGE)
                .add(Attribute::LIFETIME)
                .build(),
            ..Default::default()
        }
        .init(PositionSphereModifier::default())
        .init(ParticleLifetimeModifier::default())
        .update(AccelModifier::default())
        .update(LinearDragModifier::default())
        .update(ForceFieldModifier::default())
        .render(ParticleTextureModifier::default())
        .render(ColorOverLifetimeModifier::default())
        .render(SizeOverLifetimeModifier::default())
        .render(BillboardModifier::default());

        assert_eq!(effect.capacity, 4096);

        let mut init_layout = InitLayout::default();
        PositionSphereModifier::default().apply(&mut init_layout);
        ParticleLifetimeModifier::default().apply(&mut init_layout);
        assert_eq!(effect.init_layout, init_layout);

        let mut update_layout = UpdateLayout::default();
        AccelModifier::default().apply(&mut update_layout);
        LinearDragModifier::default().apply(&mut update_layout);
        ForceFieldModifier::default().apply(&mut update_layout);
        assert_eq!(effect.update_layout, update_layout);

        let mut render_layout = RenderLayout::default();
        ParticleTextureModifier::default().apply(&mut render_layout);
        ColorOverLifetimeModifier::default().apply(&mut render_layout);
        SizeOverLifetimeModifier::default().apply(&mut render_layout);
        BillboardModifier::default().apply(&mut render_layout);
        assert_eq!(effect.render_layout, render_layout);
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
        let effect_serde: EffectAsset = ron::from_str(&s).unwrap();
        assert_eq!(effect, effect_serde);
    }
}
