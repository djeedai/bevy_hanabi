use std::{fmt::Debug, num::NonZeroU32};

use bevy::{
    ecs::entity::EntityHashMap,
    prelude::*,
    render::{
        render_resource::{Buffer, BufferVec, CachedComputePipelineId},
        renderer::{RenderDevice, RenderQueue},
        sync_world::MainEntity,
    },
};
use fixedbitset::FixedBitSet;
use wgpu::BufferUsages;

use super::{
    effect_cache::EffectSlice,
    event::{CachedChildInfo, CachedEffectEvents},
    BufferBindingSource, ExtractedEffectMesh, LayoutFlags, PropertyBindGroupKey,
};
use crate::{
    render::{
        aligned_buffer_vec::AlignedBufferVec, buffer_table::BufferTableId, effect_cache::SlabId,
        ExtractedEffect, ExtractedSpawner, GpuBatchInfo, GpuSpawnerParams,
    },
    AlphaMode, EffectAsset, ParticleLayout, TextureLayout,
};

/// Info about particle spawning for an entire batch of effects.
#[derive(Debug, Clone, Copy)]
pub(crate) enum BatchSpawnInfo {
    /// Spawn a number of particles uploaded from CPU each frame.
    CpuSpawner {
        /// Total number of particles to spawn for the batch. This is only used
        /// to calculate the number of compute workgroups to dispatch. This is
        /// the sum of all spawn counts for all effects in the batch.
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

impl BatchSpawnInfo {
    /// Check if this batch uses CPU-based spawning.
    ///
    /// # Returns
    ///
    /// Returns `true` if this instance is a `BatchSpawnInfo::CpuSpawner`.
    #[inline]
    #[must_use]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::CpuSpawner { .. })
    }

    /// Check if this batch uses GPU-based spawning.
    ///
    /// # Returns
    ///
    /// Returns `true` if this instance is a `BatchSpawnInfo::GpuSpawner`.
    #[inline]
    #[must_use]
    #[allow(dead_code)]
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::GpuSpawner { .. })
    }

    /// Retrieve the CPU spawn count, if this batch is CPU-based.
    ///
    /// # Returns
    ///
    /// Returns `Some(count)` with the total spawn count if this instance is a
    /// `BatchSpawnInfo::CpuSpawner`. Otherwise returns `None`.
    #[inline]
    #[must_use]
    #[allow(dead_code)]
    pub fn as_cpu(&self) -> Option<&u32> {
        if let Self::CpuSpawner { total_spawn_count } = self {
            Some(total_spawn_count)
        } else {
            None
        }
    }

    /// Retrieve the CPU spawn count or a default value.
    ///
    /// This variant is used as a shortcut to
    /// `.as_cpu().unwrap_or(<unspecified>)` when filling out GPU buffers,
    /// where we need a value whatever the case, but will ignore it if GPU
    /// based.
    ///
    /// # Returns
    ///
    /// Returns the total spawn count if this instance is a
    /// `BatchSpawnInfo::CpuSpawner`. Otherwise returns an unspecified value,
    /// which should be ignored.
    #[inline]
    #[must_use]
    #[allow(unused)]
    pub fn cpu_spawn_count(&self) -> u32 {
        if let Self::CpuSpawner { total_spawn_count } = self {
            *total_spawn_count
        } else {
            u32::MAX
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BatchEffectData {
    /// Main [`Entity`] this effect instance was extracted from.
    pub entity: u32,
    /// Offset into the GPU buffer where the particles for this effect instance
    /// are located. This is uploaded to GPU as [`GpuBatchInfo::base_particle`].
    pub slab_offset: u32,
    pub draw_indirect_buffer_row_index: BufferTableId,
    pub metadata_table_id: BufferTableId,
    pub sort_fill_indirect_dispatch_index: Option<u32>,
    /// Offset in bytes of the [`GpuBatchInfo`] into the
    /// [`Batcher::batch_info_buffer`], which contains the location of the
    /// particles to render for this effect instance. Ready for bind group
    /// dynamic offset usage.
    pub render_batch_info_offset: u32,
}

/// Batch of effects dispatched and rendered together.
#[derive(Debug, Clone)]
pub(crate) struct EffectBatch {
    /// ID of the [`GpuBatchInfo`] in the global shared array for this batch.
    pub batch_info_id: u32,
    /// Handle of the underlying effect asset describing the effect. The batch
    /// only contains effect instances of the same asset.
    pub handle: Handle<EffectAsset>,
    /// ID of the particle slab in the [`EffectBuffer`] where all the batched
    /// effects are stored.
    ///
    /// [`EffectBuffer`]: super::effect_cache::EffectBuffer
    pub slab_id: SlabId,
    /// Spawn info for this batch
    pub spawn_info: BatchSpawnInfo,
    /// Specialized init and update compute pipelines.
    pub init_and_update_pipeline_ids: InitAndUpdatePipelineIds,
    /// Configured shader used for the particle rendering of this group.
    /// Note that we don't need to keep the init/update shaders alive because
    /// their pipeline specialization is doing it via the specialization key.
    pub render_shader: Handle<Shader>,
    /// ID of the particle slab where the parent effect is stored, if any. If a
    /// parent exists, its particle buffer is made available (read-only) for
    /// a child effect to read its attributes.
    #[allow(dead_code)]
    pub parent_slab_id: Option<SlabId>,
    pub parent_min_binding_size: Option<NonZeroU32>,
    pub parent_binding_source: Option<BufferBindingSource>,
    /// Event buffers of child effects, if any.
    pub child_event_buffers: Vec<(Entity, BufferBindingSource)>,
    /// Index of the property buffer, if any.
    pub property_key: Option<PropertyBindGroupKey>,
    /// Index of the first [`GpuSpawnerParams`] entry of the effects in the
    /// batch. Subsequent batched effects have their entries following linearly
    /// after that one.
    ///
    /// [`GpuSpawnerParams`]: super::GpuSpawnerParams
    pub spawner_base: u32,
    /// Total number of effect instances batched together in this batch.
    //pub effect_count: u32,
    /// Per-effect metadata used for draw and sorting. The number of elements
    /// equals the number of effects batched together inside this batch.
    pub effect_data: Vec<BatchEffectData>,
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
    pub cached_effect_events: Option<CachedEffectEvents>,
}

impl EffectBatch {
    /// Try to merge another batch into this one.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if merged successfully. Returns `Err(input)` if the
    /// input couldn't be merged.
    #[allow(clippy::result_large_err)]
    pub fn try_merge(&mut self, input: EffectBatch) -> Result<(), EffectBatch> {
        // Keep merging conservative; parent/child/event-linked effects require
        // additional per-effect bindings and aren't safe to merge yet.
        if self.handle != input.handle
            || self.slab_id != input.slab_id
            || self.init_and_update_pipeline_ids != input.init_and_update_pipeline_ids
            || self.mesh != input.mesh
            || self.alpha_mode != input.alpha_mode
            || self.texture_layout != input.texture_layout
            || self.textures != input.textures
            || self.property_key != input.property_key
            || self.parent_slab_id != input.parent_slab_id
            || self.parent_binding_source != input.parent_binding_source
            || self.child_event_buffers != input.child_event_buffers
            || self.cached_effect_events.is_some()
            || input.cached_effect_events.is_some()
            || !self.spawn_info.is_cpu()
            || !input.spawn_info.is_cpu()
        {
            return Err(input);
        }

        self.effect_data.extend(input.effect_data);
        if let (
            BatchSpawnInfo::CpuSpawner {
                total_spawn_count: self_count,
            },
            BatchSpawnInfo::CpuSpawner {
                total_spawn_count: input_count,
            },
        ) = (&mut self.spawn_info, input.spawn_info)
        {
            *self_count += input_count;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EffectBatchIndex(pub u32);

#[derive(Resource)]
pub(crate) struct Batcher {
    /// Effect batches in the order they were inserted by [`push()`], indexed by
    /// the returned [`EffectBatchIndex`].
    ///
    /// [`push()`]: Self::push
    batches: Vec<EffectBatch>,
    /// Index of the dispatch queue used for indirect fill dispatch and
    /// submitted to [`GpuBufferOperations`].
    pub(super) dispatch_queue_index: Option<u32>,
    /// Index of the queue which copies post-update alive counts into the prefix
    /// sum buffer before the ribbon sort prefix sum pass.
    pub(super) sort_fill_prefix_sum_queue_index: Option<u32>,
    /// Global shared GPU buffer storing the various `BatchInfo` structs for the
    /// active batches. This is dynamically updated each frame based on current
    /// batching, with one entry per batch (= one entry per dispatch/draw).
    batch_info_buffer: AlignedBufferVec<GpuBatchInfo>,
    /// Debug: was begin_batch() called without end_batch()?
    is_batch_open: bool,
    /// Buffer containing the prefix sums for all batches.
    prefix_sum_buffer: BufferVec<u32>,
    /// Current running prefix sum counter of CPU values passed to
    /// [`Self::push()`], and used to initialize the prefix sum of each batch
    /// for the next init pass.
    cpu_prefix_sum_value: u32,
}

impl FromWorld for Batcher {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let item_align =
            NonZeroU32::new(device.limits().min_storage_buffer_offset_alignment).unwrap();
        let batch_info_buffer = AlignedBufferVec::new(
            BufferUsages::STORAGE,
            Some(item_align.into()),
            Some("hanabi:buffer:batch_info".to_string()),
        );
        let mut prefix_sum_buffer = BufferVec::new(BufferUsages::STORAGE);
        prefix_sum_buffer.set_label(Some("prefix_sum_buffer"));
        Self {
            batches: vec![],
            dispatch_queue_index: None,
            sort_fill_prefix_sum_queue_index: None,
            batch_info_buffer,
            is_batch_open: false,
            prefix_sum_buffer,
            cpu_prefix_sum_value: 0,
        }
    }
}

impl Batcher {
    #[inline]
    pub fn batch_info_buffer(&self) -> Option<&Buffer> {
        self.batch_info_buffer.buffer()
    }

    #[inline]
    pub fn batch_info_buffer_aligned_size(&self) -> u32 {
        self.batch_info_buffer.aligned_size() as u32
    }

    #[inline]
    pub fn prefix_sum_buffer(&self) -> Option<&Buffer> {
        self.prefix_sum_buffer.buffer()
    }

    /// Return the prefix sum buffer slot that the next pushed effect instance
    /// will occupy.
    #[inline]
    pub fn next_effect_prefix_sum_index(&self) -> u32 {
        u32::try_from(self.prefix_sum_buffer.len())
            .expect("Prefix sum buffer contains more than u32::MAX entries")
    }

    pub fn clear(&mut self) {
        self.batches.clear();
        self.dispatch_queue_index = None;
        self.sort_fill_prefix_sum_queue_index = None;
        self.prefix_sum_buffer.clear();
        self.batch_info_buffer.clear();
    }

    /// Begin a new batch of effects.
    fn begin_batch(&mut self, base_particle: u32, spawner_base: u32) -> u32 {
        assert!(!self.is_batch_open, "Duplicate call to begin_batch()");

        let prefix_sum_offset = self.prefix_sum_buffer.len() as u32;

        let batch_info_base = self.batch_info_buffer.len() as u32;
        let batch_info = GpuBatchInfo {
            total_spawn_count: 0,  // set in end_batch()
            total_update_count: 0, // calculated on GPU
            spawner_base,
            base_particle,
            prefix_sum_offset,
            prefix_sum_count: u32::MAX, // invalid; set in end_batch()
        };
        trace!("batch info = {:?}", batch_info);
        self.batch_info_buffer.push(batch_info);
        self.is_batch_open = true;

        batch_info_base
    }

    /// Add a single effect instance entry to the current batch prefix array.
    fn add_effect_to_batch(&mut self, prefix_value: u32) {
        assert!(
            self.is_batch_open,
            "Cannot add effect before calling begin_batch()"
        );
        self.prefix_sum_buffer.push(prefix_value);
    }

    /// Try to end the current batch, if any. Does nothing if no batch is
    /// pending.
    ///
    /// # Returns
    ///
    /// Returns `true` if a batch was closed, or `false` otherwise.
    pub fn try_end_batch(&mut self) -> bool {
        if self.is_batch_open {
            self.end_batch();
            true
        } else {
            false
        }
    }

    /// End the current batch.
    ///
    /// # Panics
    ///
    /// Panics if no batch is pending.
    fn end_batch(&mut self) {
        assert!(
            self.is_batch_open,
            "Call to end_batch() without begin_batch()"
        );

        // Get the open batch
        let batch = self
            .batch_info_buffer
            .last_mut()
            .expect("No open batch. Missing begin_batch() call?");

        // Record prefix sum
        let end = self.prefix_sum_buffer.len() as u32;
        assert!(end >= batch.prefix_sum_offset);
        batch.prefix_sum_count = end - batch.prefix_sum_offset;

        // Record total number of CPU spawn, to clamp the number of GPU threads
        batch.total_spawn_count = self.cpu_prefix_sum_value;

        self.is_batch_open = false;
    }

    /// Insert a new batch into the collection.
    ///
    /// Try to merge the `effect_batch` into the last pushed batch, or create a
    /// new standalone batch if not possible (incompatible effects, or first
    /// one). The `instance_spawn_count` is the number of particles to spawn
    /// from CPU, and is used to initialize the prefix sum buffer, for use by
    /// the init pass (CPU spawn). The update pass' prefix sum is recomputed
    /// inside the same buffer between the init and update passes, directly on
    /// GPU.
    ///
    /// # Returns
    ///
    /// This returns the index of the new batch if the inserted one couldn't be
    /// merged with a previous batch. Otherwise the input batch was merged with
    /// an existing one, and therefore share its index; in that case `None` is
    /// returned.
    pub fn push(
        &mut self,
        effect_batch: EffectBatch,
        instance_spawn_count: u32,
    ) -> Option<EffectBatchIndex> {
        assert!(effect_batch.effect_data.len() == 1);

        let effect_batch = if let Some(batch) = self.batches.last_mut() {
            let Err(effect_batch) = batch.try_merge(effect_batch) else {
                // Successfully batched
                self.add_effect_to_batch(self.cpu_prefix_sum_value);
                self.cpu_prefix_sum_value += instance_spawn_count;
                return None;
            };
            // Failed to merge incompatible batches
            effect_batch
        } else {
            // No prior batch to merge with
            effect_batch
        };

        // Close the previous batch if any
        self.try_end_batch();

        // Start a new batch
        let index = self.batches.len() as u32;
        let base_particle = effect_batch.effect_data[0].slab_offset;
        self.batches.push(effect_batch);

        // Begin a new batch with this new effect instance
        let batch_info_id = self.begin_batch(base_particle, self.last().unwrap().spawner_base);
        self.last_mut().unwrap().batch_info_id = batch_info_id;
        self.cpu_prefix_sum_value = 0;

        self.add_effect_to_batch(self.cpu_prefix_sum_value);
        self.cpu_prefix_sum_value += instance_spawn_count;

        Some(EffectBatchIndex(index))
    }

    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<&EffectBatch> {
        self.batches.last()
    }

    #[inline]
    #[must_use]
    pub fn last_mut(&mut self) -> Option<&mut EffectBatch> {
        self.batches.last_mut()
    }

    pub fn len(&self) -> usize {
        self.batches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    /// Get an iterator over the sorted sequence of effect batches.
    #[inline]
    pub fn iter(&self) -> &[EffectBatch] {
        &self.batches
    }

    pub fn get(&self, index: EffectBatchIndex) -> Option<&EffectBatch> {
        if index.0 < self.batches.len() as u32 {
            Some(&self.batches[index.0 as usize])
        } else {
            None
        }
    }

    /// Allocate the render batches.
    ///
    /// Allocate one render batch info entry per effect instance after compute
    /// batching is finalized, to avoid nesting begin_batch()/end_batch() calls.
    /// Currently there's no rendering batching, so we allocate one GpuBatchInfo
    /// per effect instance.
    pub fn allocate_render_batches(&mut self) {
        let batch_info_aligned_size = self.batch_info_buffer_aligned_size();
        let num_batches = self.batches.len();
        for batch_index in 0..num_batches {
            let num_effects = self.batches[batch_index].effect_data.len();
            for effect_index in 0..num_effects {
                let effect_batch = &self.batches[batch_index];
                let effect_data = &effect_batch.effect_data[effect_index];
                if effect_data.render_batch_info_offset != u32::MAX {
                    continue;
                }
                let spawner_index = effect_batch.spawner_base + effect_index as u32;

                let base_particle = effect_data.slab_offset;
                let render_batch_info_id = self.begin_batch(base_particle, spawner_index);
                // Render-only batch infos are not processed by vfx_prefix_sum; keep
                // a relative offset of 0 for the single effect in that batch.
                self.add_effect_to_batch(0);
                self.end_batch();
                let render_batch_info_offset = render_batch_info_id
                    .checked_mul(batch_info_aligned_size)
                    .unwrap();

                self.batches[batch_index].effect_data[effect_index].render_batch_info_offset =
                    render_batch_info_offset;
            }
        }
    }

    #[inline]
    pub fn write_batch_info_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) -> bool {
        self.batch_info_buffer.write_buffer(device, queue)
    }

    pub fn write_prefix_sum_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) -> bool {
        let mut reallocated = false;
        let cpu_len = self.prefix_sum_buffer.len();
        if cpu_len > 0 {
            let gpu_capacity = self.prefix_sum_buffer.capacity();
            if cpu_len > gpu_capacity {
                self.prefix_sum_buffer.reserve(cpu_len, device);
                reallocated = true;
            }
            assert!(self.prefix_sum_buffer.buffer().is_some());
            self.prefix_sum_buffer.write_buffer(device, queue);
        }
        reallocated
    }
}

/// Information that the [`EffectSorter`] maintains in order to sort each
/// effect into the proper order.
pub(crate) struct EffectToBeSorted {
    /// The render-world entity of the effect.
    pub(crate) entity: Entity,
    /// The index of the buffer that the indirect indices for this effect are
    /// stored in.
    pub(crate) slab_id: SlabId,
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
    slab_id: SlabId,
    /// The offset within the buffer described above at which the indirect
    /// indices for this effect start.
    base_instance: u32,
}

/// Sorts effects into the proper order for batching.
///
/// This places parents before children and also tries to place effects in the
/// same slab together.
pub(crate) struct EffectSorter {
    /// Information that we keep about each effect.
    pub effects: Vec<EffectToBeSorted>,
    /// A mapping from a child to its parent, if it has one.
    pub child_to_parent: EntityHashMap<Entity>,
}

impl EffectSorter {
    /// Creates a new [`EffectSorter`].
    pub fn new() -> EffectSorter {
        EffectSorter {
            effects: vec![],
            child_to_parent: default(),
        }
    }

    /// Insert an effect to be sorted.
    pub fn insert(
        &mut self,
        entity: Entity,
        slab_id: SlabId,
        base_instance: u32,
        parent: Option<Entity>,
    ) {
        self.effects.push(EffectToBeSorted {
            entity,
            slab_id,
            base_instance,
        });
        if let Some(parent) = parent {
            self.child_to_parent.insert(entity, parent);
        }
    }

    /// Sorts all the effects into the optimal order for batching.
    pub fn sort(&mut self) {
        // trace!("Sorting {} effects...", self.effects.len());
        // for effect in &self.effects {
        //     trace!(
        //         "+ {}: slab={:?} base_instance={:?}",
        //         effect.entity,
        //         effect.slab_id,
        //         effect.base_instance
        //     );
        // }
        // trace!("child->parent:");
        // for (k, v) in &self.child_to_parent {
        //     trace!("+ c[{k}] -> p[{v}]");
        // }

        // First, create a map of entity to index.
        let mut entity_to_index = EntityHashMap::default();
        for (index, effect) in self.effects.iter().enumerate() {
            entity_to_index.insert(effect.entity, index);
        }

        // Next, create a map of children to their parents.
        let mut children_to_parent: Vec<_> = (0..self.effects.len()).map(|_| vec![]).collect();
        for (kid, parent) in self.child_to_parent.iter() {
            let (parent_index, kid_index) = (entity_to_index[parent], entity_to_index[kid]);
            children_to_parent[kid_index].push(parent_index);
        }

        // Now topologically sort the graph. Create an ordering that places
        // children before parents.
        // https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
        let mut ordering = vec![0; self.effects.len()];
        let mut visiting = FixedBitSet::with_capacity(self.effects.len());
        let mut visited = FixedBitSet::with_capacity(self.effects.len());
        while let Some(effect_index) = visited.zeroes().next() {
            visit(
                &mut ordering,
                &mut visiting,
                &mut visited,
                &children_to_parent,
                effect_index,
            );
        }

        // Compute levels.
        let mut levels = vec![0; self.effects.len()];
        for effect_index in ordering.into_iter().rev() {
            let level = levels[effect_index];
            for &parent in &children_to_parent[effect_index] {
                levels[parent] = levels[parent].max(level + 1);
            }
        }

        // Now sort the result.
        self.effects.sort_unstable_by_key(|effect| EffectSortKey {
            level: levels[entity_to_index[&effect.entity]],
            slab_id: effect.slab_id,
            base_instance: effect.base_instance,
        });

        // Helper function for topologically sorting the effect dependency graph
        fn visit(
            ordering: &mut Vec<usize>,
            visiting: &mut FixedBitSet,
            visited: &mut FixedBitSet,
            children_to_parent: &[Vec<usize>],
            effect_index: usize,
        ) {
            if visited.contains(effect_index) {
                return;
            }
            debug_assert!(
                !visiting.contains(effect_index),
                "Parent-child effect relation contains a cycle"
            );

            visiting.insert(effect_index);

            for &parent in &children_to_parent[effect_index] {
                visit(ordering, visiting, visited, children_to_parent, parent);
            }

            visited.insert(effect_index);
            ordering.push(effect_index);
        }
    }

    /// Iterate over the effects. This only iterates in sorted order if
    /// [`Self::sort()`] was called beforehand.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = Entity> + use<'_> {
        self.effects.iter().map(|e| e.entity)
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
        main_entity: Entity,
        extracted_effect: &ExtractedEffect,
        extracted_spawner: &ExtractedSpawner,
        cached_mesh: &ExtractedEffectMesh,
        cached_effect_events: Option<&CachedEffectEvents>,
        cached_child_info: Option<&CachedChildInfo>,
        spawner_index: u32,
        input: &mut BatchInput,
        draw_indirect_buffer_row_index: BufferTableId,
        metadata_table_id: BufferTableId,
        property_key: Option<PropertyBindGroupKey>,
    ) -> EffectBatch {
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
                total_spawn_count: extracted_spawner.spawn_count,
            }
        };

        EffectBatch {
            batch_info_id: u32::MAX, // allocated later once the batch is completed
            handle: extracted_effect.handle.clone(),
            slab_id: input.effect_slice.slab_id,
            spawn_info,
            init_and_update_pipeline_ids: input.init_and_update_pipeline_ids,
            render_shader: extracted_effect.effect_shaders.render.clone(),
            parent_slab_id: input.parent_slab_id,
            parent_min_binding_size: cached_child_info
                .map(|cci| cci.parent_particle_layout.min_binding_size32()),
            parent_binding_source: cached_child_info
                .map(|cci| cci.parent_buffer_binding_source.clone()),
            child_event_buffers: input.child_effects.clone(),
            property_key,
            spawner_base: spawner_index,
            effect_data: vec![BatchEffectData {
                entity: main_entity.index_u32(),
                slab_offset: input.effect_slice.slice.start,
                draw_indirect_buffer_row_index,
                metadata_table_id,
                sort_fill_indirect_dispatch_index: None,
                render_batch_info_offset: u32::MAX,
            }],
            particle_layout: input.effect_slice.particle_layout.clone(),
            layout_flags: extracted_effect.layout_flags,
            mesh: cached_mesh.mesh,
            texture_layout: extracted_effect.texture_layout.clone(),
            textures: extracted_effect.textures.clone(),
            alpha_mode: extracted_effect.alpha_mode,
            cached_effect_events: cached_effect_events.cloned(),
        }
    }
}

/// Effect batching input, obtained from extracted effects.
#[derive(Debug, Component)]
pub(crate) struct BatchInput {
    /// Effect slices.
    pub effect_slice: EffectSlice,
    /// Compute pipeline IDs of the specialized and cached pipelines.
    pub init_and_update_pipeline_ids: InitAndUpdatePipelineIds,
    /// ID of the particle slab of the parent effect, if any.
    pub parent_slab_id: Option<SlabId>,
    /// Index of the event buffer, if this effect consumes GPU spawn events.
    pub event_buffer_index: Option<u32>,
    /// Child effects, if any.
    pub child_effects: Vec<(Entity, BufferBindingSource)>,
    /// [`GpuSpawnerParams`] for this instance.
    pub gpu_spawner_params: GpuSpawnerParams,
    /// Index of the init indirect dispatch struct, if any.
    // FIXME - Contains a single effect's data; should handle multiple ones.
    pub init_indirect_dispatch_index: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct InitAndUpdatePipelineIds {
    pub init: CachedComputePipelineId,
    pub update: CachedComputePipelineId,
}

#[cfg(test)]
mod tests {
    use bevy::ecs::entity::Entity;

    use super::*;

    fn insert_entry(
        sorter: &mut EffectSorter,
        entity: Entity,
        slab_id: SlabId,
        base_instance: u32,
        parent: Option<Entity>,
    ) {
        sorter.effects.push(EffectToBeSorted {
            entity,
            base_instance,
            slab_id,
        });
        if let Some(parent) = parent {
            sorter.child_to_parent.insert(entity, parent);
        }
    }

    #[test]
    fn toposort_batches() {
        let mut sorter = EffectSorter::new();

        // Some "parent" effect
        let e1 = Entity::from_raw_u32(1).unwrap();
        insert_entry(&mut sorter, e1, SlabId::new(42), 0, None);
        assert_eq!(sorter.effects.len(), 1);
        assert_eq!(sorter.effects[0].entity, e1);
        assert!(sorter.child_to_parent.is_empty());

        // Some "child" effect in a different buffer
        let e2 = Entity::from_raw_u32(2).unwrap();
        insert_entry(&mut sorter, e2, SlabId::new(5), 30, Some(e1));
        assert_eq!(sorter.effects.len(), 2);
        assert_eq!(sorter.effects[0].entity, e1);
        assert_eq!(sorter.effects[1].entity, e2);
        assert_eq!(sorter.child_to_parent.len(), 1);
        assert_eq!(sorter.child_to_parent[&e2], e1);

        sorter.sort();
        assert_eq!(sorter.effects.len(), 2);
        assert_eq!(sorter.effects[0].entity, e2); // child first
        assert_eq!(sorter.effects[1].entity, e1); // parent after
        assert_eq!(sorter.child_to_parent.len(), 1); // unchanged
        assert_eq!(sorter.child_to_parent[&e2], e1); // unchanged

        // Some "child" effect in the same buffer as its parent
        let e3 = Entity::from_raw_u32(3).unwrap();
        insert_entry(&mut sorter, e3, SlabId::new(42), 20, Some(e1));
        assert_eq!(sorter.effects.len(), 3);
        assert_eq!(sorter.effects[0].entity, e2); // from previous sort
        assert_eq!(sorter.effects[1].entity, e1); // from previous sort
        assert_eq!(sorter.effects[2].entity, e3); // simply appended
        assert_eq!(sorter.child_to_parent.len(), 2);
        assert_eq!(sorter.child_to_parent[&e2], e1);
        assert_eq!(sorter.child_to_parent[&e3], e1);

        sorter.sort();
        assert_eq!(sorter.effects.len(), 3);
        assert_eq!(sorter.effects[0].entity, e2); // child first
        assert_eq!(sorter.effects[1].entity, e3); // other child next (in same buffer as parent)
        assert_eq!(sorter.effects[2].entity, e1); // finally, parent
        assert_eq!(sorter.child_to_parent.len(), 2);
        assert_eq!(sorter.child_to_parent[&e2], e1);
        assert_eq!(sorter.child_to_parent[&e3], e1);
    }
}
