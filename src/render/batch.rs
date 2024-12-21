use std::{
    fmt::Debug,
    ops::{Index, Range},
};

#[cfg(feature = "2d")]
use bevy::math::FloatOrd;
use bevy::{
    prelude::*,
    render::{
        render_resource::{Buffer, CachedComputePipelineId},
        sync_world::MainEntity,
    },
};

use super::{
    effect_cache::{DispatchBufferIndices, EffectSlices},
    CachedMesh, LayoutFlags,
};
use crate::{
    spawn::EffectInitializer, AlphaMode, EffectAsset, EffectShader, ParticleLayout, TextureLayout,
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
    /// Asset ID of the effect mesh to draw.
    pub mesh: AssetId<Mesh>,
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
        cached_mesh: &CachedMesh,
        input: &mut BatchesInput,
        spawner_base: u32,
        init_and_update_pipeline_ids: Vec<InitAndUpdatePipelineIds>,
        dispatch_buffer_indices: DispatchBufferIndices,
        first_particle_group_buffer_index: u32,
    ) -> EffectBatches {
        EffectBatches {
            buffer_index: input.effect_slices.buffer_index,
            spawner_base,
            initializers: input.initializers.clone(),
            particle_layout: input.effect_slices.particle_layout.clone(),
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
            handle: input.handle.clone(),
            layout_flags: input.layout_flags,
            mesh: cached_mesh.mesh,
            mesh_buffer: cached_mesh.buffer.clone(),
            mesh_slice: cached_mesh.range.clone(),
            texture_layout: input.texture_layout.clone(),
            textures: input.textures.clone(),
            alpha_mode: input.alpha_mode,
            render_shaders: input
                .effect_shaders
                .iter()
                .map(|shaders| shaders.render.clone())
                .collect(),
            init_and_update_pipeline_ids,
            entities: vec![input.main_entity.id().index()],
            group_order: input.group_order.clone(),
        }
    }
}

/// Effect batching input, obtained from extracted effects.
#[derive(Debug, Component)]
pub(crate) struct BatchesInput {
    /// Handle of the underlying effect asset describing the effect.
    pub handle: Handle<EffectAsset>,
    /// Main entity of the [`ParticleEffect`], used for visibility.
    pub main_entity: MainEntity,
    /// Render entity of the [`CachedEffect`]. FIXME - doesn't work with
    /// batching!
    #[allow(dead_code)]
    pub entity: Entity,
    /// Effect slices.
    pub effect_slices: EffectSlices,
    /// Effect shaders.
    pub effect_shaders: Vec<EffectShader>,
    /// Various flags related to the effect.
    pub layout_flags: LayoutFlags,
    /// Texture layout.
    pub texture_layout: TextureLayout,
    /// Textures.
    pub textures: Vec<Handle<Image>>,
    /// Alpha mode.
    pub alpha_mode: AlphaMode,
    #[allow(dead_code)]
    pub particle_layout: ParticleLayout,
    pub initializers: Vec<EffectInitializer>,
    /// The order in which we evaluate groups.
    pub group_order: Vec<u32>,
    /// Emitter position.
    #[cfg(feature = "3d")]
    pub position: Vec3,
    /// Sort key, for 2D only.
    #[cfg(feature = "2d")]
    pub z_sort_key_2d: FloatOrd,
}

#[derive(Debug)]
pub(crate) struct InitAndUpdatePipelineIds {
    pub init: CachedComputePipelineId,
    pub update: CachedComputePipelineId,
}

#[derive(Debug, Component)]
pub(crate) struct CachedGroups {
    pub spawner_base: u32,
    pub first_particle_group_buffer_index: Option<u32>,
    // Note: Stolen each frame, so invalid if not re-extracted each frame. This is how we tell if
    // an effect is active for the current frame.
    pub init_and_update_pipeline_ids: Vec<InitAndUpdatePipelineIds>,
}
