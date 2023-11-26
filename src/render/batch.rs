use std::ops::Range;

use bevy::{
    prelude::*,
    render::render_resource::{Buffer, CachedComputePipelineId},
};

#[cfg(feature = "2d")]
use bevy::utils::FloatOrd;

use crate::{EffectAsset, EffectShader, ForceFieldSource, ParticleLayout, PropertyLayout};

use super::{EffectSlice, GpuCompressedTransform, LayoutFlags};

/// A batch of multiple instances of the same effect, rendered all together to
/// reduce GPU shader permutations and draw call overhead.
#[derive(Debug, Component)]
pub(crate) struct EffectBatch {
    /// Index of the GPU effect buffer effects in this batch are contained in.
    pub buffer_index: u32,
    /// Index of the first Spawner of the effects in the batch.
    pub spawner_base: u32,
    /// Number of particles to spawn/init this frame.
    pub spawn_count: u32,
    /// Particle layout.
    pub particle_layout: ParticleLayout,
    /// Slice of particles in the GPU effect buffer for the entire batch.
    pub slice: Range<u32>,
    /// Handle of the underlying effect asset describing the effect.
    pub handle: Handle<EffectAsset>,
    /// Flags describing the render layout.
    pub layout_flags: LayoutFlags,
    /// Texture to modulate the particle color.
    pub image_handle: Handle<Image>,
    /// Configured shader used for the particle rendering of this batch.
    /// Note that we don't need to keep the init/update shaders alive because
    /// their pipeline specialization is doing it via the specialization key.
    pub render_shader: Handle<Shader>,
    /// Init compute pipeline specialized for this batch.
    pub init_pipeline_id: CachedComputePipelineId,
    /// Update compute pipeline specialized for this batch.
    pub update_pipeline_id: CachedComputePipelineId,
    /// For 2D rendering, the Z coordinate used as the sort key. Ignored for 3D
    /// rendering.
    #[cfg(feature = "2d")]
    pub z_sort_key_2d: FloatOrd,
    /// Entities holding the source [`ParticleEffect`] instances which were
    /// batched into this single batch. Used to determine visibility per view.
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    pub entities: Vec<u32>,
}

impl EffectBatch {
    /// Create a new batch from a single input.
    pub fn from_input(
        input: BatchInput,
        spawner_base: u32,
        init_pipeline_id: CachedComputePipelineId,
        update_pipeline_id: CachedComputePipelineId,
    ) -> EffectBatch {
        EffectBatch {
            buffer_index: input.effect_slice.group_index,
            spawner_base,
            spawn_count: input.spawn_count,
            particle_layout: input.effect_slice.particle_layout,
            slice: input.effect_slice.slice,
            handle: input.handle,
            layout_flags: input.layout_flags,
            image_handle: input.image_handle,
            render_shader: input.effect_shader.render,
            init_pipeline_id,
            update_pipeline_id,
            #[cfg(feature = "2d")]
            z_sort_key_2d: input.z_sort_key_2d,
            entities: vec![input.entity_index],
        }
    }
}

/// Effect batching input, obtained from extracted effects.
#[derive(Debug, Clone)]
pub(crate) struct BatchInput {
    /// Handle of the underlying effect asset describing the effect.
    pub handle: Handle<EffectAsset>,
    /// Entity index excluding generation ([`Entity::index()`]). This is
    /// transient for a single frame, so the generation is useless.
    pub entity_index: u32,
    /// Effect slice.
    pub effect_slice: EffectSlice,
    /// Layout of the effect properties.
    pub property_layout: PropertyLayout,
    /// Effect shader.
    pub effect_shader: EffectShader,
    /// Various flags related to the effect.
    pub layout_flags: LayoutFlags,
    /// Texture to modulate the particle color.
    pub image_handle: Handle<Image>,
    /// Force field sources.
    pub force_field: [ForceFieldSource; ForceFieldSource::MAX_SOURCES],
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

impl BatchInput {
    /// Check if the batch contains any property data.
    pub fn has_property_data(&self) -> bool {
        self.property_data
            .as_ref()
            .map(|data| !data.is_empty())
            .unwrap_or(false)
    }
}

/// Batching state, data not actually emitted in the effect batch but useful to
/// merge individual extracted effects.
#[derive(Debug, Clone)]
pub(crate) struct BatchState {
    // FIXME - why is that not in EffectBatch?!
    pub property_layout: PropertyLayout,
    /// The original init shader. The batch contains the cached pipeline ID
    /// instead of the shader handle, so doesn't need this.
    pub init_shader: Handle<Shader>,
    /// The original update shader. The batch contains the cached pipeline ID
    /// instead of the shader handle, so doesn't need this.
    pub update_shader: Handle<Shader>,
    /// Did the batch already emit property data? Batching currently doesn't
    /// handle multiple property values, so forces a batch split if more than
    /// one effect has properties.
    pub has_property_data: bool,
}

impl BatchState {
    /// Create a new batch from a single input.
    pub fn from_input(input: &mut BatchInput) -> BatchState {
        BatchState {
            property_layout: std::mem::take(&mut input.property_layout),
            init_shader: std::mem::take(&mut input.effect_shader.init),
            update_shader: std::mem::take(&mut input.effect_shader.update),
            has_property_data: input.has_property_data(),
        }
    }
}

/// Trait representing an item which can be batched together with other
/// compatible items.
pub(crate) trait Batchable<S, B>: Sized {
    /// Try to merge the current batchable item into the given batch based on a
    /// merge state.
    ///
    /// The `state` argument represents the state associated with the current
    /// batch, and can be used to determine if the current item is compatible
    /// with the batch.
    ///
    /// Return `Ok` if the item is successfully merged, or return `Err(self)` if
    /// the item couldn't be merged. In the latter case, the batch is not
    /// modified.
    fn try_merge(self, state: &mut S, batch: &mut B) -> Result<(), Self>;
}

impl Batchable<BatchState, EffectBatch> for BatchInput {
    fn try_merge(self, state: &mut BatchState, batch: &mut EffectBatch) -> Result<(), BatchInput> {
        // 2D effects need the same sort key; we never batch across sort keys because
        // they represent the drawing order, so effects shouldn't be reordered past
        // them.
        #[cfg(feature = "2d")]
        let is_2d_compatible = self.z_sort_key_2d == batch.z_sort_key_2d;
        #[cfg(not(feature = "2d"))]
        let is_2d_compatible = true;

        let has_property_data = self.has_property_data();

        let is_compatible = self.effect_slice.group_index == batch.buffer_index
            && self.effect_slice.slice.start == batch.slice.end  // continuous
            && self.effect_slice.particle_layout == batch.particle_layout
            && self.property_layout == state.property_layout
            && self.effect_shader.init == state.init_shader
            && self.effect_shader.update == state.update_shader
            && self.effect_shader.render == batch.render_shader
            && self.layout_flags == batch.layout_flags
            && self.image_handle == batch.image_handle
            && is_2d_compatible
            && (!has_property_data || !state.has_property_data);

        if !is_compatible {
            return Err(self);
        }

        // Merge self into batch
        batch.slice.end = self.effect_slice.slice.end;
        batch.entities.push(self.entity_index);
        state.has_property_data = has_property_data;
        // TODO - add per-effect spawner stuffs etc. which are "batched" but remain
        // per-effect

        Ok(())
    }
}

/// Utility to batch items together.
///
/// The batcher iterates over an ordered sequence of items, trying to merge each
/// item into the current batch, or creating a new batch for an incompatible
/// item.
///
/// The batcher maintains a batch state, additional information associated with
/// the current batch but not necessarily useful as part of the batch itself.
/// This is generally used to determine if the next item can be merged with the
/// batch, when that condition is based on data not explicitly saved in the
/// batch. The batch state is transient, only alive while an associated batch is
/// being built. Once a batch is emitted, its associated state is dropped. The
/// batcher never exposes nor returns a state.
///
/// Each time a batch is completed, the batcher invokes an emit callback to
/// yield the newly-created batch.
pub(crate) struct Batcher<'a, S, B, I: Batchable<S, B>> {
    /// Convert a batchable item into a batch merge state and a newly created
    /// batch containing that item alone.
    into_batch: Box<dyn FnMut(I) -> (S, B) + 'a>,
    /// Emit a completed batch.
    emit: Box<dyn FnMut(B) + 'a>,
}

impl<'a, S, B, I: Batchable<S, B>> Batcher<'a, S, B, I> {
    /// Create a new batcher.
    ///
    /// The batcher takes two callbacks:
    /// - `into_batch` converts an item into a new batch and associated merge
    ///   state.
    /// - `emit` is used to emit a completed batch.
    pub fn new(into_batch: impl FnMut(I) -> (S, B) + 'a, emit: impl FnMut(B) + 'a) -> Self {
        Self {
            into_batch: Box::new(into_batch),
            emit: Box::new(emit),
        }
    }

    /// Batch a sequence of items.
    ///
    /// The batcher loop on items in order, batch them together, and call the
    /// emit callback each time a batch is completed. The set of batches
    /// generated forms a partition of the input item sequence: each item is
    /// part of one batch, and one batch only, and each batch contains at least
    /// one item (no empty batch).
    pub fn batch(&mut self, items: impl IntoIterator<Item = I>) {
        // Loop over items in order, trying to merge them into the current batch (if
        // any)
        let mut current: Option<(S, B)> = None;
        for item in items.into_iter() {
            if let Some((mut state, mut batch)) = current {
                match item.try_merge(&mut state, &mut batch) {
                    Ok(_) => current = Some((state, batch)),
                    Err(item) => {
                        // Emit current batch, which is now completed since the new item cannot be
                        // merged.
                        self.do_emit(batch);

                        // Create a new batch from the incompatible item. That batch becomes the new
                        // current batch.
                        current = Some(self.do_into_batch(item));
                    }
                }
            } else {
                // First item, create a new batch
                current = Some(self.do_into_batch(item));
            }
        }

        // Emit the last batch if any
        if let Some((_, batch)) = current {
            self.do_emit(batch);
        }
    }

    #[inline]
    fn do_emit(&mut self, batch: B) {
        (self.emit)(batch);
    }

    #[inline]
    fn do_into_batch(&mut self, item: I) -> (S, B) {
        (self.into_batch)(item)
    }
}

#[cfg(test)]
mod tests {
    use crate::EffectShader;

    use super::*;

    // Test item to batch
    struct Item {
        pub range: Range<i32>,
        pub layer: i32,
    }

    // State is current batching layer
    struct State(pub i32);

    // Batch is contiguous range
    struct Batch(pub Range<i32>);

    impl Batchable<State, Batch> for Item {
        fn try_merge(self, state: &mut State, batch: &mut Batch) -> Result<(), Self> {
            // Can't batch items on different layers
            if self.layer != state.0 {
                return Err(self);
            }

            // Need contiguous ranges, in order
            if self.range.start != batch.0.end {
                return Err(self);
            }

            // Merge: Extend batch range with new item's
            batch.0.end = self.range.end;
            Ok(())
        }
    }

    #[test]
    fn batch_empty() {
        let mut batches = vec![];
        {
            let mut batcher = Batcher::new(
                |item: Item| (State(item.layer), Batch(item.range)),
                |b| batches.push(b),
            );
            batcher.batch([]);
        }
        assert!(batches.is_empty());
    }

    #[test]
    fn batch_single() {
        let mut batches = vec![];
        {
            let mut batcher = Batcher::new(
                |item: Item| (State(item.layer), Batch(item.range)),
                |b| batches.push(b),
            );
            batcher.batch([
                Item {
                    range: 0..5,
                    layer: 0,
                },
                Item {
                    range: 5..10,
                    layer: 0,
                },
                Item {
                    range: 10..15,
                    layer: 0,
                },
            ]);
        }
        assert_eq!(1, batches.len());
        assert_eq!(0..15, batches[0].0);
    }

    #[test]
    fn batch_hole() {
        let mut batches = vec![];
        {
            let mut batcher = Batcher::new(
                |item: Item| (State(item.layer), Batch(item.range)),
                |b| batches.push(b),
            );
            batcher.batch([
                Item {
                    range: 0..5,
                    layer: 0,
                },
                Item {
                    range: 5..10,
                    layer: 0,
                },
                Item {
                    range: 12..15,
                    layer: 0,
                },
            ]);
        }
        assert_eq!(2, batches.len());
        assert_eq!(0..10, batches[0].0);
        assert_eq!(12..15, batches[1].0);
    }

    #[test]
    fn batch_not_sorted() {
        let mut batches = vec![];
        {
            let mut batcher = Batcher::new(
                |item: Item| (State(item.layer), Batch(item.range)),
                |b| batches.push(b),
            );
            batcher.batch([
                Item {
                    range: 0..5,
                    layer: 0,
                },
                Item {
                    range: 10..15,
                    layer: 0,
                },
                Item {
                    range: 5..10,
                    layer: 0,
                },
            ]);
        }
        assert_eq!(3, batches.len());
        assert_eq!(0..5, batches[0].0);
        assert_eq!(10..15, batches[1].0);
        assert_eq!(5..10, batches[2].0);
    }

    #[test]
    fn batch_two_states() {
        let mut batches = vec![];
        {
            let mut batcher = Batcher::new(
                |item: Item| (State(item.layer), Batch(item.range)),
                |b| batches.push(b),
            );
            batcher.batch([
                Item {
                    range: 0..5,
                    layer: 0,
                },
                Item {
                    range: 5..10,
                    layer: 1,
                },
                Item {
                    range: 10..15,
                    layer: 1,
                },
            ]);
        }
        assert_eq!(2, batches.len());
        assert_eq!(0..5, batches[0].0);
        assert_eq!(5..15, batches[1].0);
    }

    #[test]
    fn batch_two_states_last() {
        let mut batches = vec![];
        {
            let mut batcher = Batcher::new(
                |item: Item| (State(item.layer), Batch(item.range)),
                |b| batches.push(b),
            );
            batcher.batch([
                Item {
                    range: 0..5,
                    layer: 0,
                },
                Item {
                    range: 5..10,
                    layer: 0,
                },
                Item {
                    range: 10..15,
                    layer: 1,
                },
            ]);
        }
        assert_eq!(2, batches.len());
        assert_eq!(0..10, batches[0].0);
        assert_eq!(10..15, batches[1].0);
    }

    #[test]
    fn batch_dual_hole() {
        let mut batches = vec![];
        {
            let mut batcher = Batcher::new(
                |item: Item| (State(item.layer), Batch(item.range)),
                |b| batches.push(b),
            );
            batcher.batch([
                Item {
                    range: 0..5,
                    layer: 0,
                },
                Item {
                    range: 10..15,
                    layer: 1,
                },
            ]);
        }
        assert_eq!(2, batches.len());
        assert_eq!(0..5, batches[0].0);
        assert_eq!(10..15, batches[1].0);
    }

    #[test]
    fn batch_restart_overlap() {
        let mut batches = vec![];
        {
            let mut batcher = Batcher::new(
                |item: Item| (State(item.layer), Batch(item.range)),
                |b| batches.push(b),
            );
            batcher.batch([
                Item {
                    range: 0..5,
                    layer: 0,
                },
                Item {
                    range: 0..5,
                    layer: 1,
                },
            ]);
        }
        assert_eq!(2, batches.len());
        assert_eq!(0..5, batches[0].0);
        assert_eq!(0..5, batches[1].0);
    }

    fn make_test_item() -> BatchInput {
        let handle = Handle::<EffectAsset>::default();
        let particle_layout = ParticleLayout::empty();
        let effect_shader = EffectShader::default();
        let image_handle = Handle::<Image>::default();
        let property_layout = PropertyLayout::empty();

        BatchInput {
            handle,
            entity_index: 0,
            effect_slice: EffectSlice {
                slice: 0..100,
                group_index: 0,
                particle_layout,
            },
            property_layout,
            effect_shader,
            layout_flags: LayoutFlags::NONE,
            image_handle,
            force_field: [ForceFieldSource::default(); ForceFieldSource::MAX_SOURCES],
            spawn_count: 32,
            transform: GpuCompressedTransform::default(),
            inverse_transform: GpuCompressedTransform::default(),
            property_buffer: None,
            property_data: None,
            #[cfg(feature = "2d")]
            z_sort_key_2d: FloatOrd(0.),
        }
    }

    #[test]
    fn effect_batch_same() {
        let mut batches = vec![];

        {
            let mut spawner_base = 0;
            let mut batcher: Batcher<'_, BatchState, EffectBatch, BatchInput> = Batcher::new(
                |mut item: BatchInput| {
                    spawner_base += 1;
                    (
                        BatchState::from_input(&mut item),
                        EffectBatch::from_input(
                            item,
                            spawner_base,
                            CachedComputePipelineId::INVALID,
                            CachedComputePipelineId::INVALID,
                        ),
                    )
                },
                |b| batches.push(b),
            );

            let item1 = make_test_item();

            let mut item2 = item1.clone();
            item2.effect_slice.slice = 100..200;

            batcher.batch([item1, item2]);
        }

        assert_eq!(1, batches.len());
        assert_eq!(0..200, batches[0].slice);
    }

    // FIXME - Currently we don't support per-effect property block in a batch, so
    // two effects with property blocks cannot be batched together.
    #[test]
    fn effect_batch_single_property_block() {
        let mut batches = vec![];

        {
            let mut spawner_base = 0;
            let mut batcher: Batcher<'_, BatchState, EffectBatch, BatchInput> = Batcher::new(
                |mut item: BatchInput| {
                    spawner_base += 1;
                    (
                        BatchState::from_input(&mut item),
                        EffectBatch::from_input(
                            item,
                            spawner_base,
                            CachedComputePipelineId::INVALID,
                            CachedComputePipelineId::INVALID,
                        ),
                    )
                },
                |b| batches.push(b),
            );

            let mut item1 = make_test_item();
            // Has property data, and so will item2 after cloning, so can't batch them
            // together
            item1.property_data = Some(vec![1, 2]);

            let mut item2 = item1.clone();
            item2.effect_slice.slice = 100..200;

            batcher.batch([item1, item2]);
        }

        assert_eq!(2, batches.len());
        assert_eq!(0..100, batches[0].slice);
        assert_eq!(100..200, batches[1].slice);
    }

    #[test]
    fn effect_batch_texture_same() {
        let mut batches = vec![];

        {
            let mut spawner_base = 0;
            let mut batcher: Batcher<'_, BatchState, EffectBatch, BatchInput> = Batcher::new(
                |mut item: BatchInput| {
                    spawner_base += 1;
                    (
                        BatchState::from_input(&mut item),
                        EffectBatch::from_input(
                            item,
                            spawner_base,
                            CachedComputePipelineId::INVALID,
                            CachedComputePipelineId::INVALID,
                        ),
                    )
                },
                |b| batches.push(b),
            );

            let mut item1 = make_test_item();
            item1.image_handle = Handle::<Image>::default();

            let mut item2 = item1.clone();
            item2.effect_slice.slice = 100..200;

            batcher.batch([item1, item2]);
        }

        assert_eq!(1, batches.len());
        assert_eq!(0..200, batches[0].slice);
    }

    #[test]
    fn effect_batch_texture_different() {
        let mut batches = vec![];

        let mut images = Assets::<Image>::default();
        let image1 = images.add(Image::default());
        let image2 = images.add(Image::default());
        assert_ne!(image1, image2);

        {
            let mut spawner_base = 0;
            let mut batcher: Batcher<'_, BatchState, EffectBatch, BatchInput> = Batcher::new(
                |mut item: BatchInput| {
                    spawner_base += 1;
                    (
                        BatchState::from_input(&mut item),
                        EffectBatch::from_input(
                            item,
                            spawner_base,
                            CachedComputePipelineId::INVALID,
                            CachedComputePipelineId::INVALID,
                        ),
                    )
                },
                |b| batches.push(b),
            );

            let mut item1 = make_test_item();
            item1.image_handle = image1;

            let mut item2 = item1.clone();
            item2.effect_slice.slice = 100..200;
            item2.image_handle = image2;

            assert_ne!(item1.image_handle, item2.image_handle);

            batcher.batch([item1, item2]);
        }

        assert_eq!(2, batches.len());
        assert_eq!(0..100, batches[0].slice);
        assert_eq!(100..200, batches[1].slice);
    }

    // Regression test - #181 Spawning effects with and without textures causes
    // problems
    #[test]
    fn effect_batch_texture_mixed() {
        let mut batches = vec![];

        let mut images = Assets::<Image>::default();
        let image1 = images.add(Image::default());

        {
            let mut spawner_base = 0;
            let mut batcher: Batcher<'_, BatchState, EffectBatch, BatchInput> = Batcher::new(
                |mut item: BatchInput| {
                    spawner_base += 1;
                    (
                        BatchState::from_input(&mut item),
                        EffectBatch::from_input(
                            item,
                            spawner_base,
                            CachedComputePipelineId::INVALID,
                            CachedComputePipelineId::INVALID,
                        ),
                    )
                },
                |b| batches.push(b),
            );

            let item1 = make_test_item();

            let mut item2 = item1.clone();
            item2.effect_slice.slice = 100..200;
            // Has texture, while item1 doesn't
            item2.image_handle = image1;

            batcher.batch([item1, item2]);
        }

        assert_eq!(2, batches.len());
        assert_eq!(0..100, batches[0].slice);
        assert_eq!(100..200, batches[1].slice);
    }

    #[cfg(feature = "2d")]
    #[test]
    fn effect_batch_zsortkey_different() {
        let mut batches = vec![];

        {
            let mut spawner_base = 0;
            let mut batcher: Batcher<'_, BatchState, EffectBatch, BatchInput> = Batcher::new(
                |mut item: BatchInput| {
                    spawner_base += 1;
                    (
                        BatchState::from_input(&mut item),
                        EffectBatch::from_input(
                            item,
                            spawner_base,
                            CachedComputePipelineId::INVALID,
                            CachedComputePipelineId::INVALID,
                        ),
                    )
                },
                |b| batches.push(b),
            );

            let item1 = make_test_item();

            let mut item2 = item1.clone();
            item2.effect_slice.slice = 100..200;
            item2.z_sort_key_2d = FloatOrd(32.5);

            assert_ne!(item1.z_sort_key_2d, item2.z_sort_key_2d);

            batcher.batch([item1, item2]);
        }

        assert_eq!(2, batches.len());
        assert_eq!(0..100, batches[0].slice);
        assert_eq!(100..200, batches[1].slice);
    }
}
