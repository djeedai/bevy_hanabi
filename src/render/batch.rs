use std::{fmt::Debug, ops::Range};

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
    effect_cache::{DispatchBufferIndices, EffectSlice},
    CachedMesh, LayoutFlags, PropertyBindGroupKey,
};
use crate::{AlphaMode, EffectAsset, EffectShader, ParticleLayout, TextureLayout};

/// Batch of effects dispatched and rendered together.
#[derive(Debug, Component)]
pub(crate) struct EffectBatch {
    /// Handle of the underlying effect asset describing the effect.
    pub handle: Handle<EffectAsset>,
    /// Index of the buffer.
    pub buffer_index: u32,
    /// Slice of particles in the GPU effect buffer referenced by
    /// [`EffectBatch::buffer_index`]. The GPU buffer is shared by all groups.
    pub slice: Range<u32>,
    /// Number of particles to spawn for this effect.
    pub spawn_count: u32,
    /// Init and update compute pipelines specialized for this group.
    pub init_and_update_pipeline_ids: InitAndUpdatePipelineIds,
    /// Configured shader used for the particle rendering of this group.
    /// Note that we don't need to keep the init/update shaders alive because
    /// their pipeline specialization is doing it via the specialization key.
    pub render_shader: Handle<Shader>,
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
    /// Index of the [`GpuInitDispatchIndirect`] struct into the init indirect
    /// dispatch buffer, if using indirect init dispatch only.
    pub init_indirect_dispatch_index: Option<u32>,
}

/// Single effect batch to drive rendering.
///
/// This component is spawned into the render world during the prepare phase
/// ([`prepare_effects()`]), once per effect batch per group. In turns it
/// references an [`EffectBatch`] component containing all the shared data for
/// all the groups of the effect.
#[derive(Debug, Component)]
pub(crate) struct EffectDrawBatch {
    /// Entity holding the [`EffectBatch`] this batch is part of.
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
        dispatch_buffer_indices: DispatchBufferIndices,
        property_key: Option<PropertyBindGroupKey>,
        property_offset: Option<u32>,
    ) -> EffectBatch {
        assert_eq!(property_key.is_some(), property_offset.is_some());
        EffectBatch {
            handle: input.handle.clone(),
            buffer_index: input.effect_slice.buffer_index,
            slice: input.effect_slice.slice.clone(),
            spawn_count: input.spawn_count,
            init_and_update_pipeline_ids: input.init_and_update_pipeline_ids,
            render_shader: input.shaders.render.clone(),
            parent_buffer_index: input.parent_buffer_index,
            event_buffer_index: input.event_buffer_index,
            child_effects: input.child_effects.clone(),
            property_key,
            property_offset,
            spawner_base: input.spawner_base,
            particle_layout: input.effect_slice.particle_layout.clone(),
            dispatch_buffer_indices,
            layout_flags: input.layout_flags,
            mesh: cached_mesh.mesh,
            mesh_buffer: cached_mesh.buffer.clone(),
            mesh_slice: cached_mesh.range.clone(),
            texture_layout: input.texture_layout.clone(),
            textures: input.textures.clone(),
            alpha_mode: input.alpha_mode,
            entities: vec![input.main_entity.id().index()],
            init_indirect_dispatch_index: input.init_indirect_dispatch_index,
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
    /// Render entity of the [`CachedEffect`].
    #[allow(dead_code)]
    pub entity: Entity,
    /// Effect slices.
    pub effect_slice: EffectSlice,
    /// Compute pipeline IDs of the specialized and cached pipelines.
    pub init_and_update_pipeline_ids: InitAndUpdatePipelineIds,
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
    /// Effect shaders.
    pub shaders: EffectShader,
    /// Index of the [`GpuSpawnerParams`] in the
    /// [`EffectsCache::spawner_buffer`].
    pub spawner_base: u32,
    /// Number of particles to spawn for this effect.
    pub spawn_count: u32,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct InitAndUpdatePipelineIds {
    pub init: CachedComputePipelineId,
    pub update: CachedComputePipelineId,
}
