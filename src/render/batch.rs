use std::ops::{Index, Range};

use bevy::{
    prelude::*,
    render::render_resource::{Buffer, CachedComputePipelineId},
};

#[cfg(feature = "2d")]
use bevy::math::FloatOrd;

use crate::{EffectAsset, EffectShader, ParticleLayout, PropertyLayout};

use super::{
    effect_cache::{DispatchBufferIndices, EffectSlices},
    EffectCacheId, GpuCompressedTransform, LayoutFlags,
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
    /// Number of particles to spawn/init this frame.
    pub spawn_count: u32,
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
    /// Entities holding the source [`ParticleEffect`] instances which were
    /// batched into this single batch. Used to determine visibility per view.
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    pub entities: Vec<u32>,
    /// Texture to modulate the particle color.
    ///
    /// TODO: We should support more than one of these.
    pub image_handle: Handle<Image>,
    /// Configured shaders used for the particle rendering of this batch.
    /// Note that we don't need to keep the init/update shaders alive because
    /// their pipeline specialization is doing it via the specialization key.
    pub render_shaders: Vec<Handle<Shader>>,
    /// Init compute pipeline specialized for this batch.
    pub init_pipeline_id: CachedComputePipelineId,
    /// Update compute pipeline specialized for this batch.
    pub update_pipeline_ids: Vec<CachedComputePipelineId>,
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
        init_pipeline_id: CachedComputePipelineId,
        update_pipeline_ids: Vec<CachedComputePipelineId>,
        dispatch_buffer_indices: DispatchBufferIndices,
        first_particle_group_buffer_index: u32,
    ) -> EffectBatches {
        EffectBatches {
            buffer_index: input.effect_slices.buffer_index,
            spawner_base,
            spawn_count: input.spawn_count,
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
            image_handle: input.image_handle,
            render_shaders: input.effect_shader.render,
            init_pipeline_id,
            update_pipeline_ids,
            entities: vec![input.entity.index()],
        }
    }
}

/// Effect batching input, obtained from extracted effects.
#[derive(Debug, Clone)]
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
    /// Effect shader.
    pub effect_shader: EffectShader,
    /// Various flags related to the effect.
    pub layout_flags: LayoutFlags,
    /// Texture to modulate the particle color.
    pub image_handle: Handle<Image>,
    /// Number of particles to spawn for this effect.
    pub spawn_count: u32,
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
