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
    CachedMesh, LayoutFlags, PropertyBindGroupKey,
};
use crate::{
    spawn::EffectInitializer, AlphaMode, EffectAsset, EffectShader, ParticleLayout, TextureLayout,
};

/// Batch data specific to a single particle group.
#[derive(Debug)]
pub(crate) struct GroupBatch {
    /// Slice of particles in the GPU effect buffer referenced by
    /// [`EffectBatch::buffer_index`]. The GPU buffer is shared by all groups.
    pub slice: Range<u32>,
    /// Initializer for the group.
    pub initializer: EffectInitializer,
    /// Init and update compute pipelines specialized for this group.
    pub init_and_update_pipeline_ids: InitAndUpdatePipelineIds,
    /// Configured shader used for the particle rendering of this group.
    /// Note that we don't need to keep the init/update shaders alive because
    /// their pipeline specialization is doing it via the specialization key.
    pub render_shader: Handle<Shader>,
}

/// Batch of effects dispatched and rendered together.
#[derive(Debug, Component)]
pub(crate) struct EffectBatch {
    /// Handle of the underlying effect asset describing the effect.
    pub handle: Handle<EffectAsset>,
    /// Index of the buffer.
    pub buffer_index: u32,
    /// One batch per particle group.
    pub group_batches: Vec<GroupBatch>,
    /// Index of the buffer of the parent effect, if any.
    pub parent_buffer_index: Option<u32>,
    /// Index of the event buffer, if this effect consumes GPU spawn events.
    pub event_buffer_index: Option<u32>,
    /// Child effects, if any.
    pub child_effects: Vec<Entity>,
    /// Index of the property buffer, if any.
    pub property_key: Option<PropertyBindGroupKey>,
    /// Offset in bytes into the property buffer where the Property struct is
    /// located for this effect.
    // FIXME: This is a per-instance value which prevents batching :(
    pub property_offset: Option<u32>,
    /// Index of the first [`GpuSpawnerParams`] entry of the effects in the
    /// batch. Subsequent batched effects have their entries following linearly
    /// after that one.
    ///
    /// [`GpuSpawnerParams`]: super::GpuSpawnerParams
    pub spawner_base: u32,
    /// The indices within the various indirect dispatch buffers.
    pub dispatch_buffer_indices: DispatchBufferIndices,
    /// The index of the first [`GpuParticleGroup`] structure in the global
    /// [`EffectsMeta::particle_group_buffer`] buffer. The buffer is currently
    /// re-created each frame, so the rows for multiple groups of an effect are
    /// guaranteed to be contiguous.
    ///
    /// [`GpuParticleGroup`]: super::GpuParticleGroup
    /// [`EffectsMeta::particle_group_buffer`]: super::EffectsMeta::particle_group_buffer
    pub first_particle_group_buffer_index: u32,
    /// Particle layout shared by all batched effects and groups.
    pub particle_layout: ParticleLayout,
    /// Flags describing the render layout.
    pub layout_flags: LayoutFlags,
    /// Asset ID of the effect mesh to draw.
    pub mesh: AssetId<Mesh>,
    /// GPU buffer storing the [`mesh`] of the effect.
    ///
    /// [`mesh`]: Self::mesh
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
    /// The order in which we evaluate groups.
    pub group_order: Vec<u32>,
    /// Index of the [`GpuInitDispatchIndirect`] struct into the init indirect
    /// dispatch buffer, if using indirect init dispatch only.
    pub init_indirect_dispatch_index: Option<u32>,
}

impl Index<u32> for EffectBatch {
    type Output = GroupBatch;

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

impl EffectBatch {
    /// Create a new batch from a single input.
    pub fn from_input(
        cached_mesh: &CachedMesh,
        input: &mut BatchesInput,
        spawner_base: u32,
        mut init_and_update_pipeline_ids: Vec<InitAndUpdatePipelineIds>,
        dispatch_buffer_indices: DispatchBufferIndices,
        first_particle_group_buffer_index: u32,
        property_key: Option<PropertyBindGroupKey>,
        property_offset: Option<u32>,
    ) -> EffectBatch {
        let group_batches = input
            .effect_slices
            .slices
            .windows(2)
            .map(|range| range[0]..range[1])
            .zip(input.groups.iter())
            .zip(init_and_update_pipeline_ids.drain(..))
            .map(|((slice, group_input), ids)| GroupBatch {
                slice,
                initializer: group_input.initializer,
                init_and_update_pipeline_ids: ids,
                render_shader: group_input.shaders.render.clone(),
            })
            .collect();

        assert_eq!(property_key.is_some(), property_offset.is_some());
        EffectBatch {
            handle: input.handle.clone(),
            group_batches,
            buffer_index: input.effect_slices.buffer_index,
            parent_buffer_index: input.parent_buffer_index,
            event_buffer_index: input.event_buffer_index,
            child_effects: input.child_effects.clone(),
            property_key,
            property_offset,
            spawner_base,
            particle_layout: input.effect_slices.particle_layout.clone(),
            dispatch_buffer_indices,
            first_particle_group_buffer_index,
            layout_flags: input.layout_flags,
            mesh: cached_mesh.mesh,
            mesh_buffer: cached_mesh.buffer.clone(),
            mesh_slice: cached_mesh.range.clone(),
            texture_layout: input.texture_layout.clone(),
            textures: input.textures.clone(),
            alpha_mode: input.alpha_mode,
            entities: vec![input.main_entity.id().index()],
            init_indirect_dispatch_index: input.init_indirect_dispatch_index,
            group_order: input.group_order.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GroupInput {
    /// Group shaders.
    pub shaders: EffectShader,
    /// Group initializer.
    pub initializer: EffectInitializer,
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
    // FIXME - Contains a single effect's data (multiple groups); should handle multiple ones.
    pub effect_slices: EffectSlices,
    /// Particle layout of the parent effect, if any.
    pub parent_particle_layout: Option<ParticleLayout>,
    /// Index of the buffer of the parent effect, if any.
    pub parent_buffer_index: Option<u32>,
    /// Index of the event buffer, if this effect consumes GPU spawn events.
    pub event_buffer_index: Option<u32>,
    /// Child effects, if any.
    pub child_effects: Vec<Entity>,
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
    pub groups: Vec<GroupInput>,
    /// The order in which we evaluate groups.
    pub group_order: Vec<u32>,
    /// Emitter position, for 3D sorting.
    #[cfg(feature = "3d")]
    pub position: Vec3,
    /// Index of the init indirect dispatch struct, if any.
    // FIXME - Contains a single effect's data; should handle multiple ones.
    pub init_indirect_dispatch_index: Option<u32>,
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
