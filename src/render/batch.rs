use std::{
    fmt::Debug,
    ops::{Index, Range},
};

#[cfg(feature = "2d")]
use bevy::math::FloatOrd;
use bevy::{
    prelude::*,
    render::render_resource::{Buffer, CachedComputePipelineId},
};

use super::{
    effect_cache::{DispatchBufferIndices, EffectSlices},
    EffectCacheId, GpuCompressedTransform, LayoutFlags,
};
use crate::{
    spawn::EffectInitializer, AlphaMode, EffectAsset, EffectShader, ParticleLayout, PropertyLayout,
    TextureLayout,
};

/// Data needed to render all batches pertaining to a specific effect.
#[derive(Debug, Component)]
pub(crate) struct EffectBatches {
    /// Handle of the underlying effect asset describing the effect.
    pub handle: Handle<EffectAsset>,
    /// One batch per particle group.
    pub group_batches: Vec<EffectBatch>,
    /// Index of the buffer.
    pub buffer_index: u32,
    /// Index of the first Spawner of the effects in the batch.
    pub spawner_base: u32,
    /// The initializer (spawner or cloner) for each particle group.
    pub initializers: Vec<EffectInitializer>,
    /// The effect cache ID.
    pub effect_cache_id: EffectCacheId,
    /// The indices within the various indirect dispatch buffers.
    pub dispatch_buffer_indices: DispatchBufferIndices,
    /// The index of the first [`GpuParticleGroup`] structure in the global
    /// [`EffectsMeta::particle_group_buffer`] buffer. The buffer is currently
    /// re-created each frame, so the rows for multiple groups of an effect are
    /// guaranteed to be contiguous.
    pub first_particle_group_buffer_index: u32,
    /// Particle layout.
    pub particle_layout: ParticleLayout,
    /// Flags describing the render layout.
    pub layout_flags: LayoutFlags,
    /// Asset handle of the effect mesh to draw.
    pub mesh: Handle<Mesh>,
    /// GPU buffer storing the [`mesh`] of the effect.
    pub mesh_buffer: Buffer,
    /// Slice inside the GPU buffer for the effect mesh.
    pub mesh_slice: Range<u32>,
    /// Texture layout.
    pub texture_layout: TextureLayout,
    /// Textures.
    pub textures: Vec<Handle<Image>>,
    /// Alpha mode.
    pub alpha_mode: AlphaMode,
    /// Entities holding the source [`ParticleEffect`] instances which were
    /// batched into this single batch. Used to determine visibility per view.
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    pub entities: Vec<u32>,
    /// Configured shaders used for the particle rendering of this batch.
    /// Note that we don't need to keep the init/update shaders alive because
    /// their pipeline specialization is doing it via the specialization key.
    pub render_shaders: Vec<Handle<Shader>>,
    /// Init and update compute pipelines specialized for this batch.
    pub init_and_update_pipeline_ids: Vec<InitAndUpdatePipelineIds>,
    /// The order in which we evaluate groups.
    pub group_order: Vec<u32>,
}

impl Index<u32> for EffectBatches {
    type Output = EffectBatch;

    fn index(&self, index: u32) -> &Self::Output {
        &self.group_batches[index as usize]
    }
}

/// Single effect batch to drive rendering.
///
/// This component is spawned into the render world during the prepare phase
/// ([`prepare_effects()`]), once per effect batch per group. In turns it
/// references an [`EffectBatches`] component containing all the shared data for
/// all the groups of the effect.
#[derive(Debug, Component)]
pub(crate) struct EffectDrawBatch {
    /// Group index of the batch.
    pub group_index: u32,
    /// Entity holding the [`EffectBatches`] this batch is part of.
    pub batches_entity: Entity,
    /// For 2D rendering, the Z coordinate used as the sort key. Ignored for 3D
    /// rendering.
    #[cfg(feature = "2d")]
    pub z_sort_key_2d: FloatOrd,
    /// For 3d rendering, the position of the emitter so we can compute distance
    /// to camera. Ignored for 2D rendering.
    #[cfg(feature = "3d")]
    pub translation_3d: Vec3,
}

/// Batch data specific to a single particle group.
#[derive(Debug)]
pub(crate) struct EffectBatch {
    /// Slice of particles in the GPU effect buffer for the entire batch.
    pub slice: Range<u32>,
}

impl EffectBatches {
    /// Create a new batch from a single input.
    pub fn from_input(
        input: BatchesInput,
        spawner_base: u32,
        effect_cache_id: EffectCacheId,
        init_and_update_pipeline_ids: Vec<InitAndUpdatePipelineIds>,
        dispatch_buffer_indices: DispatchBufferIndices,
        first_particle_group_buffer_index: u32,
    ) -> EffectBatches {
        EffectBatches {
            buffer_index: input.effect_slices.buffer_index,
            spawner_base,
            initializers: input.initializers.clone(),
            particle_layout: input.effect_slices.particle_layout,
            effect_cache_id,
            dispatch_buffer_indices,
            first_particle_group_buffer_index,
            group_batches: input
                .effect_slices
                .slices
                .windows(2)
                .map(|range| EffectBatch {
                    slice: range[0]..range[1],
                })
                .collect(),
            handle: input.handle,
            layout_flags: input.layout_flags,
            mesh: input.mesh.clone(),
            mesh_buffer: input.mesh_buffer,
            mesh_slice: input.mesh_slice,
            texture_layout: input.texture_layout,
            textures: input.textures,
            alpha_mode: input.alpha_mode,
            render_shaders: input
                .effect_shaders
                .iter()
                .map(|shaders| shaders.render.clone())
                .collect(),
            init_and_update_pipeline_ids,
            entities: vec![input.entity.index()],
            group_order: input.group_order,
        }
    }
}

/// Effect batching input, obtained from extracted effects.
#[derive(Debug)]
pub(crate) struct BatchesInput {
    /// Handle of the underlying effect asset describing the effect.
    pub handle: Handle<EffectAsset>,
    /// Entity index excluding generation ([`Entity::index()`]). This is
    /// transient for a single frame, so the generation is useless.
    pub entity: Entity,
    /// Effect slices.
    pub effect_slices: EffectSlices,
    /// Layout of the effect properties.
    pub property_layout: PropertyLayout,
    /// Effect shaders.
    pub effect_shaders: Vec<EffectShader>,
    /// Various flags related to the effect.
    pub layout_flags: LayoutFlags,
    /// Asset handle of the effect mesh to draw.
    pub mesh: Handle<Mesh>,
    /// GPU buffer storing the [`mesh`] of the effect.
    pub mesh_buffer: Buffer,
    /// Slice inside the GPU buffer for the effect mesh.
    pub mesh_slice: Range<u32>,
    /// Texture layout.
    pub texture_layout: TextureLayout,
    /// Textures.
    pub textures: Vec<Handle<Image>>,
    /// Alpha mode.
    pub alpha_mode: AlphaMode,
    pub particle_layout: ParticleLayout,
    pub initializers: Vec<EffectInitializer>,
    /// The order in which we evaluate groups.
    pub group_order: Vec<u32>,
    /// Emitter transform.
    pub transform: GpuCompressedTransform,
    /// Emitter inverse transform.
    pub inverse_transform: GpuCompressedTransform,
    /// GPU buffer where properties for this batch need to be written.
    pub property_buffer: Option<Buffer>,
    /// Serialized property data.
    // FIXME - Contains a single effect's data; should handle multiple ones.
    pub property_data: Option<Vec<u8>>,
    /// Sort key, for 2D only.
    #[cfg(feature = "2d")]
    pub z_sort_key_2d: FloatOrd,
}

#[derive(Debug)]
pub(crate) struct InitAndUpdatePipelineIds {
    pub(crate) init: CachedComputePipelineId,
    pub(crate) update: CachedComputePipelineId,
}
