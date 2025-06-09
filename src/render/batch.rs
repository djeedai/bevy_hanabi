use std::{fmt::Debug, num::NonZeroU32, ops::Range};

use bevy::{
    ecs::entity::EntityHashMap,
    prelude::*,
    render::{render_resource::CachedComputePipelineId, sync_world::MainEntity},
};
use fixedbitset::FixedBitSet;

use super::{
    effect_cache::{DispatchBufferIndices, EffectSlice},
    event::{CachedChildInfo, CachedEffectEvents},
    BufferBindingSource, CachedMesh, LayoutFlags, PropertyBindGroupKey,
};
use crate::{AlphaMode, EffectAsset, EffectShader, ParticleLayout, TextureLayout};

#[derive(Debug, Clone, Copy)]
pub(crate) enum BatchSpawnInfo {
    /// Spawn a number of particles uploaded from CPU each frame.
    CpuSpawner {
        /// Total number of particles to spawn for the batch. This is only used
        /// to calculate the number of compute workgroups to dispatch.
        total_spawn_count: u32,
    },

    /// Spawn a number of particles calculated on GPU from "spawn events", which
    /// generally emitted by another effect.
    GpuSpawner {
        /// Index into the init indirect dispatch buffer of the
        /// [`GpuDispatchIndirect`] instance for this batch.
        ///
        /// [`GpuDispatchIndirect`]: super::GpuDispatchIndirect
        init_indirect_dispatch_index: u32,
        /// Index of the [`EventBuffer`] where the GPU spawn events consumed by
        /// this batch are stored.
        ///
        /// [`EventBuffer`]: super::event::EventBuffer
        #[allow(dead_code)]
        event_buffer_index: u32,
    },
}

/// Batch of effects dispatched and rendered together.
#[derive(Debug, Clone)]
pub(crate) struct EffectBatch {
    /// Handle of the underlying effect asset describing the effect.
    pub handle: Handle<EffectAsset>,
    /// Index of the [`EffectBuffer`].
    ///
    /// [`EffectBuffer`]: super::effect_cache::EffectBuffer
    pub buffer_index: u32,
    /// Slice of particles in the GPU effect buffer referenced by
    /// [`EffectBatch::buffer_index`].
    pub slice: Range<u32>,
    /// Spawn info for this batch
    pub spawn_info: BatchSpawnInfo,
    /// Specialized init and update compute pipelines.
    pub init_and_update_pipeline_ids: InitAndUpdatePipelineIds,
    /// Configured shader used for the particle rendering of this group.
    /// Note that we don't need to keep the init/update shaders alive because
    /// their pipeline specialization is doing it via the specialization key.
    pub render_shader: Handle<Shader>,
    pub parent_min_binding_size: Option<NonZeroU32>,
    pub parent_binding_source: Option<BufferBindingSource>,
    /// Event buffers of child effects, if any.
    pub child_event_buffers: Vec<(Entity, BufferBindingSource)>,
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
    pub cached_effect_events: Option<CachedEffectEvents>,
    pub sort_fill_indirect_dispatch_index: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EffectBatchIndex(pub u32);

#[derive(Debug, Default, Resource)]
pub(crate) struct SortedEffectBatches {
    /// Effect batches in the order they were inserted by [`push()`], indexed by
    /// the returned [`EffectBatchIndex`].
    ///
    /// [`push()`]: Self::push
    batches: Vec<EffectBatch>,
    /// Index of the dispatch queue used for indirect fill dispatch and
    /// submitted to [`GpuBufferOperations`].
    pub(super) dispatch_queue_index: Option<u32>,
}

impl SortedEffectBatches {
    pub fn clear(&mut self) {
        self.batches.clear();
        self.dispatch_queue_index = None;
    }

    pub fn push(&mut self, effect_batch: EffectBatch) -> EffectBatchIndex {
        let index = self.batches.len() as u32;
        self.batches.push(effect_batch);
        EffectBatchIndex(index)
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.batches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    /// Get an iterator over the sorted sequence of effect batches.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &EffectBatch> {
        self.batches.iter()
    }

    pub fn get(&self, index: EffectBatchIndex) -> Option<&EffectBatch> {
        if index.0 < self.batches.len() as u32 {
            Some(&self.batches[index.0 as usize])
        } else {
            None
        }
    }
}

/// Sorts effects into the proper order for batching.
///
/// This places parents before children and also tries to place effects in the
/// same buffer together.
pub(crate) struct EffectSorter {
    /// Information that we keep about each effect.
    pub(crate) effects: Vec<EffectToBeSorted>,
    /// A mapping from a child to its parent, if it has one.
    pub(crate) child_to_parent: EntityHashMap<Entity>,
}

/// Information that the [`EffectSorter`] maintains in order to sort each
/// effect into the proper order.
pub(crate) struct EffectToBeSorted {
    /// The render-world entity of the effect.
    pub(crate) entity: Entity,
    /// The index of the buffer that the indirect indices for this effect are
    /// stored in.
    pub(crate) buffer_index: u32,
    /// The offset within the buffer described above at which the indirect
    /// indices for this effect start.
    ///
    /// This is in elements, not bytes.
    pub(crate) base_instance: u32,
}

/// The key that we sort effects by for optimum batching.
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord)]
struct EffectSortKey {
    /// The level in the dependency graph.
    ///
    /// Parents always have lower levels than their children.
    level: u32,
    /// The index of the buffer that the indirect indices for this effect are
    /// stored in.
    buffer_index: u32,
    /// The offset within the buffer described above at which the indirect
    /// indices for this effect start.
    base_instance: u32,
}

impl EffectSorter {
    /// Creates a new [`EffectSorter`].
    pub(crate) fn new() -> EffectSorter {
        EffectSorter {
            effects: vec![],
            child_to_parent: EntityHashMap::default(),
        }
    }

    /// Sorts all the effects into the optimal order for batching.
    pub(crate) fn sort(&mut self) {
        // First, create a map of entity to index.
        let mut entity_to_index = EntityHashMap::default();
        for (index, effect) in self.effects.iter().enumerate() {
            entity_to_index.insert(effect.entity, index);
        }

        // Next, create a map of parents to children.
        let mut parent_to_children: Vec<_> = (0..self.effects.len()).map(|_| vec![]).collect();
        for (kid, parent) in self.child_to_parent.iter() {
            parent_to_children[entity_to_index[parent]].push(entity_to_index[kid]);
        }

        // Now topologically sort the graph to determine the level of each node.
        // This is a modification of the Tarjan algorithm that computes the
        // depth of each node in the tree, not just the ordering.
        // https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
        let mut levels = vec![0; self.effects.len()];
        let mut visiting = FixedBitSet::with_capacity(self.effects.len());
        let mut visited = FixedBitSet::with_capacity(self.effects.len());
        while let Some(effect_index) = visited.zeroes().next() {
            visit(
                &mut levels,
                &mut visiting,
                &mut visited,
                &parent_to_children,
                effect_index,
                0,
            );
        }

        // Now sort the result.
        self.effects.sort_unstable_by_key(|effect| EffectSortKey {
            level: levels[entity_to_index[&effect.entity]],
            buffer_index: effect.buffer_index,
            base_instance: effect.base_instance,
        });

        // A helper function for topologically sorting the effect dependency
        // graph.
        fn visit(
            levels: &mut Vec<u32>,
            visiting: &mut FixedBitSet,
            visited: &mut FixedBitSet,
            parent_to_children: &[Vec<usize>],
            effect_index: usize,
            current_level: u32,
        ) {
            if visited.contains(effect_index) {
                return;
            }
            debug_assert!(
                !visiting.contains(effect_index),
                "Parent-child effect relation contains a cycle"
            );

            visiting.insert(effect_index);

            for &kid in &parent_to_children[effect_index] {
                visit(
                    levels,
                    visiting,
                    visited,
                    parent_to_children,
                    kid,
                    current_level + 1,
                );
            }

            visited.insert(effect_index);
            levels[effect_index] = current_level;
        }
    }
}

/// Single effect batch to drive rendering.
///
/// This component is spawned into the render world during the prepare phase
/// ([`prepare_effects()`]), once per effect batch per group. In turns it
/// references an [`EffectBatch`] component containing all the shared data for
/// all the groups of the effect.
#[derive(Debug, Component)]
pub(crate) struct EffectDrawBatch {
    /// Index of the [`EffectBatch`] in the [`SortedEffectBatches`] this draw
    /// batch is part of.
    ///
    /// Note: currently there's a 1:1 mapping between effect batch and draw
    /// batch.
    pub effect_batch_index: EffectBatchIndex,
    /// Position of the emitter so we can compute distance to camera.
    pub translation: Vec3,
    /// The main-world entity that contains this effect.
    #[allow(dead_code)]
    pub main_entity: MainEntity,
}

impl EffectBatch {
    /// Create a new batch from a single input.
    pub fn from_input(
        cached_mesh: &CachedMesh,
        cached_effect_events: Option<&CachedEffectEvents>,
        cached_child_info: Option<&CachedChildInfo>,
        input: &mut BatchInput,
        dispatch_buffer_indices: DispatchBufferIndices,
        property_key: Option<PropertyBindGroupKey>,
        property_offset: Option<u32>,
    ) -> EffectBatch {
        assert_eq!(property_key.is_some(), property_offset.is_some());
        assert_eq!(
            input.event_buffer_index.is_some(),
            input.init_indirect_dispatch_index.is_some()
        );

        let spawn_info = if let Some(event_buffer_index) = input.event_buffer_index {
            BatchSpawnInfo::GpuSpawner {
                init_indirect_dispatch_index: input.init_indirect_dispatch_index.unwrap(),
                event_buffer_index,
            }
        } else {
            BatchSpawnInfo::CpuSpawner {
                total_spawn_count: input.spawn_count,
            }
        };

        EffectBatch {
            handle: input.handle.clone(),
            buffer_index: input.effect_slice.buffer_index,
            slice: input.effect_slice.slice.clone(),
            spawn_info,
            init_and_update_pipeline_ids: input.init_and_update_pipeline_ids,
            render_shader: input.shaders.render.clone(),
            parent_min_binding_size: cached_child_info
                .map(|cci| cci.parent_particle_layout.min_binding_size32()),
            parent_binding_source: cached_child_info
                .map(|cci| cci.parent_buffer_binding_source.clone()),
            child_event_buffers: input.child_effects.clone(),
            property_key,
            property_offset,
            spawner_base: input.spawner_index,
            particle_layout: input.effect_slice.particle_layout.clone(),
            dispatch_buffer_indices,
            layout_flags: input.layout_flags,
            mesh: cached_mesh.mesh,
            texture_layout: input.texture_layout.clone(),
            textures: input.textures.clone(),
            alpha_mode: input.alpha_mode,
            entities: vec![input.main_entity.id().index()],
            cached_effect_events: cached_effect_events.cloned(),
            sort_fill_indirect_dispatch_index: None, // set later as needed
        }
    }
}

/// Effect batching input, obtained from extracted effects.
#[derive(Debug, Component)]
pub(crate) struct BatchInput {
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
    /// Index of the event buffer, if this effect consumes GPU spawn events.
    pub event_buffer_index: Option<u32>,
    /// Child effects, if any.
    pub child_effects: Vec<(Entity, BufferBindingSource)>,
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
    pub spawner_index: u32,
    /// Number of particles to spawn for this effect.
    pub spawn_count: u32,
    /// Emitter position.
    pub position: Vec3,
    /// Index of the init indirect dispatch struct, if any.
    // FIXME - Contains a single effect's data; should handle multiple ones.
    pub init_indirect_dispatch_index: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct InitAndUpdatePipelineIds {
    pub init: CachedComputePipelineId,
    pub update: CachedComputePipelineId,
}
