use std::{collections::VecDeque, fmt::Debug, num::NonZeroU32, ops::Range};

use bevy::{
    platform::collections::HashMap,
    prelude::*,
    render::{render_resource::CachedComputePipelineId, sync_world::MainEntity},
};

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
    /// Index of the buffer of the parent effect, if any. If a parent exists,
    /// its particle buffer is made available (read-only) for a child effect to
    /// read its attributes.
    pub parent_buffer_index: Option<u32>,
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

pub(crate) struct SortedEffectBatchesIter<'a> {
    batches: &'a [EffectBatch],
    sorted_indices: &'a [u32],
    next: u32,
}

impl<'a> SortedEffectBatchesIter<'a> {
    pub fn new(source: &'a SortedEffectBatches) -> Self {
        assert_eq!(source.batches.len(), source.sorted_indices.len());
        Self {
            batches: &source.batches[..],
            sorted_indices: &source.sorted_indices[..],
            next: 0,
        }
    }
}

impl<'a> Iterator for SortedEffectBatchesIter<'a> {
    type Item = &'a EffectBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next < self.sorted_indices.len() as u32 {
            let index = self.sorted_indices[self.next as usize];
            let batch = &self.batches[index as usize];
            self.next += 1;
            Some(batch)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for SortedEffectBatchesIter<'_> {
    fn len(&self) -> usize {
        self.sorted_indices.len()
    }
}

#[derive(Debug, Default, Resource)]
pub(crate) struct SortedEffectBatches {
    /// Effect batches in the order they were inserted by [`push()`], indexed by
    /// the returned [`EffectBatchIndex`].
    ///
    /// [`push()`]: Self::push
    batches: Vec<EffectBatch>,
    /// Indices into [`batches`] defining the sorted order batches need to be
    /// processed in. Calculated by [`sort()`].
    ///
    /// [`batches`]: Self::batches
    /// [`sort()`]: Self::sort
    sorted_indices: Vec<u32>,
    /// Index of the dispatch queue used for indirect fill dispatch and
    /// submitted to [`GpuBufferOperations`].
    pub(super) dispatch_queue_index: Option<u32>,
}

impl SortedEffectBatches {
    pub fn clear(&mut self) {
        self.batches.clear();
        self.sorted_indices.clear();
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
    pub fn iter(&self) -> SortedEffectBatchesIter {
        assert_eq!(
            self.batches.len(),
            self.sorted_indices.len(),
            "Invalid sorted size. Did you call sort() beforehand?"
        );
        SortedEffectBatchesIter::new(self)
    }

    pub fn get(&self, index: EffectBatchIndex) -> Option<&EffectBatch> {
        if index.0 < self.batches.len() as u32 {
            Some(&self.batches[index.0 as usize])
        } else {
            None
        }
    }

    /// Sort the effect batches.
    pub fn sort(&mut self) {
        self.sorted_indices.clear();
        self.sorted_indices.reserve_exact(self.batches.len());

        // Kahn’s algorithm for topological sorting.

        // Note: we sort by particle buffer index. In theory with batching this is
        // incorrect, because a parent and child could be batched together in the same
        // buffer, in the wrong order. However currently batching is broken and we
        // allocate one effect instance per buffer, so this works. Ideally we'd take
        // care of sorting earlier during batching.

        // Build a map from buffer index to batch index.
        let batch_index_from_buffer_index = self
            .batches
            .iter()
            .enumerate()
            .map(|(batch_index, effect_batch)| (effect_batch.buffer_index, batch_index))
            .collect::<HashMap<_, _>>();
        // In theory with batching we could have multiple batches referencing the same
        // buffer if we failed to batch some effect instances together which
        // otherwise share a same particle buffer. In practice this currently doesn't
        // happen because batching is disabled, so we always create one buffer
        // per effect instance. But this will need to be fixed later.
        assert_eq!(
            batch_index_from_buffer_index.len(),
            self.batches.len(),
            "FIXME: Duplicate buffer index in batches. This is not implemented yet."
        );

        // Build a map from the batch index of a child to the batch index of its
        // parent.
        let mut parent_batch_index_from_batch_index = HashMap::with_capacity(self.batches.len());
        for (batch_index, effect_batch) in self.batches.iter().enumerate() {
            if let Some(parent_buffer_index) = effect_batch.parent_buffer_index {
                let parent_batch_index = batch_index_from_buffer_index
                    .get(&parent_buffer_index)
                    .unwrap();
                parent_batch_index_from_batch_index.insert(batch_index as u32, *parent_batch_index);
            }
        }

        // Store the number of children per batch; we need to decrement it below
        // HACK - during tests we don't want to create Buffers so grab the count another
        // (slower) way
        #[cfg(test)]
        let mut child_count = {
            let mut counts = vec![0; self.batches.len()];
            for (_, parent_batch_index) in &parent_batch_index_from_batch_index {
                counts[*parent_batch_index] += 1;
            }
            counts
        };
        #[cfg(not(test))]
        let mut child_count = self
            .batches
            .iter()
            .map(|effect_batch| effect_batch.child_event_buffers.len() as u32)
            .collect::<Vec<_>>();

        // Insert in queue all effects without any child
        let mut queue = VecDeque::new();
        for (batch_index, count) in child_count.iter().enumerate() {
            if *count == 0 {
                queue.push_back(batch_index as u32);
            }
        }

        // Process queue
        while let Some(batch_index) = queue.pop_front() {
            // The batch has no unprocessed child, so it can be inserted in the final result
            assert!(child_count[batch_index as usize] == 0);
            self.sorted_indices.push(batch_index);

            // If it has a parent, that parent has one less child to be processed, so is one
            // step closer to being inserted itself in the final result.
            let Some(parent_batch_index) = parent_batch_index_from_batch_index.get(&batch_index)
            else {
                continue;
            };
            assert!(child_count[*parent_batch_index] > 0);
            child_count[*parent_batch_index] -= 1;

            // If this was the last child effect of that parent, then the parent is ready
            // and can be inserted itself.
            if child_count[*parent_batch_index] == 0 {
                queue.push_back(*parent_batch_index as u32);
            }
        }

        assert_eq!(
            self.sorted_indices.len(),
            self.batches.len(),
            "Cycle detected in effects"
        );
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
            parent_buffer_index: input.parent_buffer_index,
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
    /// Index of the buffer of the parent effect, if any.
    pub parent_buffer_index: Option<u32>,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_batch(buffer_index: u32, parent_buffer_index: Option<u32>) -> EffectBatch {
        EffectBatch {
            handle: default(),
            buffer_index,
            slice: 0..0,
            spawn_info: BatchSpawnInfo::CpuSpawner {
                total_spawn_count: 0,
            },
            init_and_update_pipeline_ids: InitAndUpdatePipelineIds {
                init: CachedComputePipelineId::INVALID,
                update: CachedComputePipelineId::INVALID,
            },
            render_shader: default(),
            parent_buffer_index,
            parent_min_binding_size: default(),
            parent_binding_source: default(),
            child_event_buffers: default(),
            property_key: default(),
            property_offset: default(),
            spawner_base: default(),
            dispatch_buffer_indices: default(),
            particle_layout: ParticleLayout::empty(),
            layout_flags: LayoutFlags::NONE,
            mesh: default(),
            texture_layout: default(),
            textures: default(),
            alpha_mode: default(),
            entities: default(),
            cached_effect_events: default(),
            sort_fill_indirect_dispatch_index: default(),
        }
    }

    #[test]
    fn toposort_batches() {
        let mut seb = SortedEffectBatches::default();
        assert!(seb.is_empty());
        assert_eq!(seb.len(), 0);

        seb.push(make_batch(42, None));
        assert!(!seb.is_empty());
        assert_eq!(seb.len(), 1);

        seb.push(make_batch(5, Some(42)));
        assert!(!seb.is_empty());
        assert_eq!(seb.len(), 2);

        seb.sort();
        assert!(!seb.is_empty());
        assert_eq!(seb.len(), 2);
        let sorted_batches = seb.iter().collect::<Vec<_>>();
        assert_eq!(sorted_batches.len(), 2);
        assert_eq!(sorted_batches[0].buffer_index, 5);
        assert_eq!(sorted_batches[1].buffer_index, 42);

        seb.push(make_batch(6, Some(42)));
        assert!(!seb.is_empty());
        assert_eq!(seb.len(), 3);

        seb.sort();
        assert!(!seb.is_empty());
        assert_eq!(seb.len(), 3);
        let sorted_batches = seb.iter().collect::<Vec<_>>();
        assert_eq!(sorted_batches.len(), 3);
        assert_eq!(sorted_batches[0].buffer_index, 5);
        assert_eq!(sorted_batches[1].buffer_index, 6);
        assert_eq!(sorted_batches[2].buffer_index, 42);

        seb.push(make_batch(55, Some(5)));
        assert!(!seb.is_empty());
        assert_eq!(seb.len(), 4);

        seb.sort();
        assert!(!seb.is_empty());
        assert_eq!(seb.len(), 4);
        let sorted_batches = seb.iter().collect::<Vec<_>>();
        assert_eq!(sorted_batches.len(), 4);
        assert_eq!(sorted_batches[0].buffer_index, 6);
        assert_eq!(sorted_batches[1].buffer_index, 55);
        assert_eq!(sorted_batches[2].buffer_index, 5);
        assert_eq!(sorted_batches[3].buffer_index, 42);
    }
}
