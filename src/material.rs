use bevy::{AlphaMode, PbrPipeline, EffectMaterialFlags};
use bevy_app::{App, Plugin};
use bevy_asset::{AddAsset, Handle, HandleUntyped};
use bevy_ecs::system::{lifetimeless::SRes, SystemParamItem};
use bevy_math::Vec4;
use bevy_reflect::TypeUuid;
use bevy_render::{
    color::Color,
    render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssets},
    render_resource::{
        std140::{AsStd140, Std140},
        BindGroup, Buffer, BufferInitDescriptor, BufferUsages,
    },
    renderer::RenderDevice,
    texture::Image,
};
use wgpu::{BindGroupDescriptor, BindGroupEntry, BindingResource};

/// A material for a [`ParticleEffect`].
///
/// May be created directly from a [`Color`] or an [`Image`].
#[derive(Debug, Clone, TypeUuid)]
#[uuid = "1ebefa44-80b6-46bc-939d-5bf39ff15f53"]
pub struct EffectMaterial {
    /// The bsae color of the particles.
    ///
    /// This is multiplied by the sampling of the [`base_color_texture`], if any.
    ///
    /// [`base_color_texture`]: EffectMaterial::base_color_texture
    pub base_color: Color,
    /// The bsae color texture of the particles.
    ///
    /// This is multiplied by the scalar [`base_color`].
    ///
    /// [`base_color`]: EffectMaterial::base_color
    pub base_color_texture: Option<Handle<Image>>,
    /// Enable double-sided rendering.
    pub double_sided: bool,
    /// Disable lighting in rendering, use a full-bright model.
    pub unlit: bool,
    /// Alpha blending mode for rendering the particles.
    pub alpha_mode: AlphaMode,
}

impl Default for EffectMaterial {
    fn default() -> Self {
        EffectMaterial {
            base_color: Color::WHITE,
            base_color_texture: None,
            double_sided: false,
            unlit: false,
            alpha_mode: AlphaMode::Opaque,
        }
    }
}

impl From<Color> for EffectMaterial {
    fn from(color: Color) -> Self {
        EffectMaterial {
            base_color: color,
            ..Default::default()
        }
    }
}

impl From<Handle<Image>> for EffectMaterial {
    fn from(texture: Handle<Image>) -> Self {
        EffectMaterial {
            base_color_texture: Some(texture),
            ..Default::default()
        }
    }
}

// NOTE: These must match the bit flags in effect.wgsl!
bitflags::bitflags! {
    #[repr(transparent)]
    pub struct EffectMaterialFlags: u32 {
        const BASE_COLOR_TEXTURE         = (1 << 0);
        const DOUBLE_SIDED               = (1 << 4);
        const UNLIT                      = (1 << 5);
        const ALPHA_MODE_OPAQUE          = (1 << 6);
        const ALPHA_MODE_MASK            = (1 << 7);
        const ALPHA_MODE_BLEND           = (1 << 8);
        const NONE                       = 0;
        const UNINITIALIZED              = 0xFFFF;
    }
}

/// The GPU representation of the uniform data of a [`EffectMaterial`].
#[derive(Clone, Default, AsStd140)]
pub struct EffectMaterialUniformData {
    pub base_color: Vec4,
    pub flags: u32,
    pub alpha_cutoff: f32,
}

/// This plugin adds the [`EffectMaterial`] asset to the app.
pub struct EffectMaterialPlugin;

impl Plugin for EffectMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(RenderAssetPlugin::<EffectMaterial>::default())
            .add_asset::<EffectMaterial>();
    }
}

/// The GPU representation of an [`EffectMaterial`].
#[derive(Debug, Clone)]
pub struct GpuEffectMaterial {
    /// A buffer containing the [`EffectMaterialUniformData`] of the material.
    pub buffer: Buffer,
    /// The bind group specifying how the [`EffectMaterialUniformData`] and
    /// all the textures of the material are bound.
    pub bind_group: BindGroup,
    pub flags: EffectMaterialFlags,
    pub base_color_texture: Option<Handle<Image>>,
    pub alpha_mode: AlphaMode,
}

impl RenderAsset for EffectMaterial {
    type ExtractedAsset = EffectMaterial;
    type PreparedAsset = GpuEffectMaterial;
    type Param = (
        SRes<RenderDevice>,
        SRes<PbrPipeline>,
        SRes<RenderAssets<Image>>,
    );

    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        material: Self::ExtractedAsset,
        (render_device, pbr_pipeline, gpu_images): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let (base_color_texture_view, base_color_sampler) = if let Some(result) = pbr_pipeline
            .mesh_pipeline
            .get_image_texture(gpu_images, &material.base_color_texture)
        {
            result
        } else {
            return Err(PrepareAssetError::RetryNextUpdate(material));
        };

        let mut flags = EffectMaterialFlags::NONE;
        if material.base_color_texture.is_some() {
            flags |= EffectMaterialFlags::BASE_COLOR_TEXTURE;
        }
        if material.double_sided {
            flags |= EffectMaterialFlags::DOUBLE_SIDED;
        }
        if material.unlit {
            flags |= EffectMaterialFlags::UNLIT;
        }
        // NOTE: 0.5 is from the glTF default - do we want this?
        let mut alpha_cutoff = 0.5;
        match material.alpha_mode {
            AlphaMode::Opaque => flags |= EffectMaterialFlags::ALPHA_MODE_OPAQUE,
            AlphaMode::Mask(c) => {
                alpha_cutoff = c;
                flags |= EffectMaterialFlags::ALPHA_MODE_MASK
            }
            AlphaMode::Blend => flags |= EffectMaterialFlags::ALPHA_MODE_BLEND,
        };

        let value = EffectMaterialUniformData {
            base_color: material.base_color.as_linear_rgba_f32().into(),
            flags: flags.bits(),
            alpha_cutoff,
        };
        let value_std140 = value.as_std140();

        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("hanabi:effect_material_uniform_buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: value_std140.as_bytes(),
        });
        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(base_color_texture_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(base_color_sampler),
                },
            ],
            label: Some("hanabi:effect_material_bind_group"),
            layout: &pbr_pipeline.material_layout,
        });

        Ok(GpuEffectMaterial {
            buffer,
            bind_group,
            flags,
            has_normal_map,
            base_color_texture: material.base_color_texture,
            alpha_mode: material.alpha_mode,
        })
    }
}
