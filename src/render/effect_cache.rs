use bevy::{
    asset::Handle,
    log::{trace, warn},
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
    },
    utils::HashMap,
};
use bytemuck::cast_slice_mut;
use std::{
    cmp::Ordering,
    num::NonZeroU64,
    ops::Range,
    sync::atomic::{AtomicU64, Ordering as AtomicOrdering},
};

use crate::asset::EffectAsset;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectSlice {
    /// Slice into the underlying BufferVec of the group.
    pub slice: Range<u32>,
    /// Index of the group containing the BufferVec.
    pub group_index: u32,
    /// Size of a single item in the slice.
    pub item_size: u32,
}

impl Ord for EffectSlice {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.group_index.cmp(&other.group_index) {
            Ordering::Equal => self.slice.start.cmp(&other.slice.start),
            ord => ord,
        }
    }
}

impl PartialOrd for EffectSlice {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A reference to a slice allocated inside an [`EffectBuffer`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SliceRef {
    /// Range into an [`EffectBuffer`], in item count.
    range: Range<u32>,
    /// Size of a single item in the slice. Currently equal to the unique size
    /// of all items in an [`EffectBuffer`] (no mixed size supported in same
    /// buffer), so cached only for convenience.
    item_size: u32,
}

impl SliceRef {
    /// The length of the slice, in number of items.
    pub fn len(&self) -> u32 {
        self.range.end - self.range.start
    }

    /// The size in bytes of the slice.
    pub fn byte_size(&self) -> usize {
        (self.len() as usize) * (self.item_size as usize)
    }
}

/// Storage for a single kind of effects, sharing the same buffer(s).
///
/// Currently only accepts a single unique item size (particle size), fixed at
/// creation. Also currently only accepts instances of a unique of effect asset.
pub struct EffectBuffer {
    /// GPU buffer holding all particles for the entire group of effects.
    particle_buffer: Buffer,
    /// GPU buffer holding the indirection indices for the entire group of
    /// effects. This is a triple buffer containing:
    /// - the ping-pong alive particles and render indirect indices at offsets 0
    ///   and 1
    /// - the dead particle indices at offset 2
    indirect_buffer: Buffer,
    /// Size of each particle, in bytes.
    item_size: u32,
    /// Total buffer capacity, in number of particles.
    capacity: u32,
    /// Used buffer size, in number of particles, either from allocated slices
    /// or from slices in the free list.
    used_size: u32,
    /// Array of free slices for new allocations, sorted in increasing order in
    /// the buffer.
    free_slices: Vec<Range<u32>>,
    /// Compute pipeline for the effect update pass.
    //pub compute_pipeline: ComputePipeline, // FIXME - ComputePipelineId, to avoid duplicating per
    // instance!
    /// Handle of all effects common in this buffer. TODO - replace with
    /// compatible layout.
    asset: Handle<EffectAsset>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    Used,
    Free,
}

impl EffectBuffer {
    /// Minimum buffer capacity to allocate, in number of particles.
    // FIXME - Batching is broken due to binding a single GpuSpawnerParam instead of
    // N, and inability for a particle index to tell which Spawner it should
    // use. Setting this to 1 effectively ensures that all new buffers just fit
    // the effect, so batching never occurs.
    pub const MIN_CAPACITY: u32 = 1; //65536; // at least 64k particles

    /// Create a new group and a GPU buffer to back it up.
    ///
    /// The buffer cannot contain less than [`MIN_CAPACITY`] particles. If
    /// `capacity` is smaller, it's rounded up to [`MIN_CAPACITY`].
    ///
    /// [`MIN_CAPACITY`]: EffectBuffer::MIN_CAPACITY
    pub fn new(
        asset: Handle<EffectAsset>,
        capacity: u32,
        item_size: u32,
        //compute_pipeline: ComputePipeline,
        render_device: &RenderDevice,
        label: Option<&str>,
    ) -> Self {
        trace!(
            "EffectBuffer::new(capacity={}, item_size={}B)",
            capacity,
            item_size
        );
        let capacity = capacity.max(Self::MIN_CAPACITY);
        let particle_capacity_bytes: BufferAddress = capacity as u64 * item_size as u64;
        let particle_buffer = render_device.create_buffer(&BufferDescriptor {
            label,
            size: particle_capacity_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let capacity_bytes: BufferAddress = capacity as u64 * 4;

        let indirect_label = if let Some(label) = label {
            format!("{}_indirect", label)
        } else {
            "hanabi:effect_buffer_indirect".to_owned()
        };
        let indirect_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some(&indirect_label),
            size: capacity_bytes * 3, // ping-pong + deadlist
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        // Set content
        {
            // Scope get_mapped_range_mut() to force a drop before unmap()
            {
                let slice = &mut indirect_buffer.slice(..).get_mapped_range_mut()
                    [..capacity_bytes as usize * 3];
                let slice: &mut [u32] = cast_slice_mut(slice);
                for index in 0..capacity {
                    slice[3 * index as usize + 2] = capacity - 1 - index;
                }
            }
            indirect_buffer.unmap();
        }

        Self {
            particle_buffer,
            indirect_buffer,
            item_size,
            capacity,
            used_size: 0,
            free_slices: vec![],
            //compute_pipeline,
            asset,
        }
    }

    pub fn particle_buffer(&self) -> &Buffer {
        &self.particle_buffer
    }

    pub fn indirect_buffer(&self) -> &Buffer {
        &self.indirect_buffer
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Return a binding for the entire particle buffer.
    pub fn max_binding(&self) -> BindingResource {
        let capacity_bytes = self.capacity as u64 * self.item_size as u64;
        BindingResource::Buffer(BufferBinding {
            buffer: &self.particle_buffer,
            offset: 0,
            size: Some(NonZeroU64::new(capacity_bytes).unwrap()),
        })
    }

    /// Return a binding of the buffer for a starting range of a given size (in
    /// bytes).
    pub fn binding(&self, size: u32) -> BindingResource {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.particle_buffer,
            offset: 0,
            size: Some(NonZeroU64::new(size as u64).unwrap()),
        })
    }

    /// Return a binding for the entire indirect buffer associated with the
    /// current effect buffer.
    pub fn indirect_max_binding(&self) -> BindingResource {
        let capacity_bytes = self.capacity as u64 * 4;
        BindingResource::Buffer(BufferBinding {
            buffer: &self.indirect_buffer,
            offset: 0,
            size: Some(NonZeroU64::new(capacity_bytes * 3).unwrap()),
        })
    }

    /// Try to recycle a free slice to store `size` items.
    fn pop_free_slice(&mut self, size: u32) -> Option<Range<u32>> {
        if self.free_slices.is_empty() {
            return None;
        }

        struct BestRange {
            range: Range<u32>,
            capacity: u32,
            index: usize,
        }

        let mut result = BestRange {
            range: 0..0, // marker for "invalid"
            capacity: u32::MAX,
            index: usize::MAX,
        };
        for (index, slice) in self.free_slices.iter().enumerate() {
            let capacity = slice.end - slice.start;
            if size > capacity {
                continue;
            }
            if capacity < result.capacity {
                result = BestRange {
                    range: slice.clone(),
                    capacity,
                    index,
                };
            }
        }
        if !result.range.is_empty() {
            if result.capacity > size {
                // split
                let start = result.range.start;
                let used_end = start + size;
                let free_end = result.range.end;
                let range = start..used_end;
                self.free_slices[result.index] = used_end..free_end;
                Some(range)
            } else {
                // recycle entirely
                self.free_slices.remove(result.index);
                Some(result.range)
            }
        } else {
            None
        }
    }

    /// Allocate a new slice in the buffer to store the particles of a single
    /// effect.
    pub fn allocate_slice(&mut self, capacity: u32, item_size: u32) -> Option<SliceRef> {
        trace!(
            "EffectBuffer::allocate_slice: capacity={} item_size={}",
            capacity,
            item_size
        );

        if capacity > self.capacity {
            return None;
        }

        let range = if let Some(range) = self.pop_free_slice(capacity) {
            range
        } else {
            let new_size = self.used_size.checked_add(capacity).unwrap();
            if new_size <= self.capacity {
                let range = self.used_size..new_size;
                self.used_size = new_size;
                range
            } else {
                if self.used_size == 0 {
                    warn!(
                        "Cannot allocate slice of size {} in effect cache buffer of capacity {}.",
                        capacity, self.capacity
                    );
                }
                return None;
            }
        };

        Some(SliceRef { range, item_size })
    }

    /// Free an allocated slice, and if this was the last allocated slice also
    /// free the buffer.
    pub fn free_slice(&mut self, slice: SliceRef) -> BufferState {
        // If slice is at the end of the buffer, reduce total used size
        if slice.range.end == self.used_size {
            self.used_size = slice.range.start;
            // Check other free slices to further reduce used size and drain the free slice
            // list
            while let Some(free_slice) = self.free_slices.last() {
                if free_slice.end == self.used_size {
                    self.used_size = free_slice.start;
                    self.free_slices.pop();
                } else {
                    break;
                }
            }
            if self.used_size == 0 {
                assert!(self.free_slices.is_empty());
                // The buffer is not used anymore, free it too
                BufferState::Free
            } else {
                // There are still some slices used, the last one of which ends at
                // self.used_size
                BufferState::Used
            }
        } else {
            // Free slice is not at end; insert it in free list
            let range = slice.range;
            match self.free_slices.binary_search_by(|s| {
                if s.end <= range.start {
                    Ordering::Less
                } else if s.start >= range.end {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            }) {
                Ok(_) => warn!("Range {:?} already present in free list!", range),
                Err(index) => self.free_slices.insert(index, range),
            }
            BufferState::Used
        }
    }

    pub fn is_compatible(&self, handle: &Handle<EffectAsset>) -> bool {
        // TODO - replace with check particle layout is compatible to allow tighter
        // packing in less buffers, and update in the less dispatch calls
        *handle == self.asset
    }
}

/// Identifier referencing an effect cached in an internal effect cache.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct EffectCacheId(/* TEMP */ pub(crate) u64);

impl EffectCacheId {
    /// An invalid handle, corresponding to nothing.
    pub const INVALID: Self = Self(u64::MAX);

    /// Generate a new valid effect cache identifier.
    pub fn new() -> Self {
        static NEXT_EFFECT_CACHE_ID: AtomicU64 = AtomicU64::new(0);
        Self(NEXT_EFFECT_CACHE_ID.fetch_add(1, AtomicOrdering::Relaxed))
    }

    /// Check if the ID is valid.
    pub fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }
}

/// Cache for effect instances sharing common GPU data structures.
pub(crate) struct EffectCache {
    /// Render device the GPU resources (buffers) are allocated from.
    device: RenderDevice,
    /// Collection of effect buffers managed by this cache. Some buffers might
    /// be `None` if the entry is not used. Since the buffers are referenced
    /// by index, we cannot move them once they're allocated.
    buffers: Vec<Option<EffectBuffer>>,
    /// Map from an effect cache ID to the index of the buffer and the slice
    /// into that buffer.
    effects: HashMap<EffectCacheId, (usize, SliceRef)>,
}

impl EffectCache {
    pub fn new(device: RenderDevice) -> Self {
        Self {
            device,
            buffers: vec![],
            effects: HashMap::default(),
        }
    }

    pub fn buffers(&self) -> &[Option<EffectBuffer>] {
        &self.buffers
    }

    pub fn buffers_mut(&mut self) -> &mut [Option<EffectBuffer>] {
        &mut self.buffers
    }

    pub fn insert(
        &mut self,
        asset: Handle<EffectAsset>,
        capacity: u32,
        item_size: u32,
        //pipeline: ComputePipeline,
        _queue: &RenderQueue,
    ) -> EffectCacheId {
        let (buffer_index, slice) = self
            .buffers
            .iter_mut()
            .enumerate()
            .find_map(|(buffer_index, buffer)| {
                if let Some(buffer) = buffer {
                    // The buffer must be compatible with the effect layout, to allow the update pass
                    // to update all particles at once from all compatible effects in a single dispatch.
                    if !buffer.is_compatible(&asset) {
                        return None;
                    }

                    // Try to allocate a slice into the buffer
                    buffer
                        .allocate_slice(capacity, item_size)
                        .map(|slice| (buffer_index, slice))
                } else {
                    None
                }
            })
            .or_else(|| {
                // Cannot find any suitable buffer; allocate a new one
                let buffer_index = self.buffers.iter().position(|buf| buf.is_none()).unwrap_or(self.buffers.len());
                let byte_size = capacity.checked_mul(item_size).unwrap_or_else(|| panic!(
                    "Effect size overflow: capacity={} item_size={}",
                    capacity, item_size
                ));
                trace!(
                    "Creating new effect buffer #{} for effect {:?} (capacity={}, item_size={}, byte_size={})",
                    buffer_index,
                    asset,
                    capacity,
                    item_size,
                    byte_size
                );
                let mut buffer = EffectBuffer::new(
                    asset,
                    capacity,
                    item_size,
                    //pipeline,
                    &self.device,
                    Some(&format!("hanabi:effect_buffer{}", buffer_index)),
                );
                let slice_ref = buffer.allocate_slice(capacity, item_size).unwrap();
                if buffer_index >= self.buffers.len() {
                    self.buffers.push(Some(buffer));
                } else {
                    assert!(self.buffers[buffer_index].is_none());
                    self.buffers[buffer_index] = Some(buffer);
                }
                // Newly-allocated buffers are not cleared to zero, and currently we eagerly render the entire buffer
                // since we don't have an indirection buffer to tell us how many particles are alive. So clear the buffer
                // to zero to mark all particles as invalid and prevent rendering them.
                //queue.cle
                Some((buffer_index, slice_ref))
            })
            .unwrap();
        let id = EffectCacheId::new();
        trace!(
            "Insert effect id={:?} buffer_index={} slice={:?}x{}B",
            id,
            buffer_index,
            slice.range,
            slice.item_size
        );
        self.effects.insert(id, (buffer_index, slice));
        id
    }

    pub fn get_slice(&self, id: EffectCacheId) -> EffectSlice {
        self.effects
            .get(&id)
            .map(|(buffer_index, slice_ref)| EffectSlice {
                slice: slice_ref.range.clone(),
                group_index: *buffer_index as u32,
                item_size: slice_ref.item_size,
            })
            .unwrap()
    }

    /// Remove an effect from the cache. If this was the last effect, drop the
    /// underlying buffer and return the index of the dropped buffer.
    pub fn remove(&mut self, id: EffectCacheId) -> Option<u32> {
        if let Some((buffer_index, slice)) = self.effects.remove(&id) {
            if let Some(buffer) = &mut self.buffers[buffer_index] {
                if buffer.free_slice(slice) == BufferState::Free {
                    self.buffers[buffer_index] = None;
                    return Some(buffer_index as u32);
                }
            }
        }
        None
    }
}

#[cfg(all(test, feature = "gpu_tests"))]
mod gpu_tests {
    use bevy::asset::HandleId;

    use crate::test_utils::MockRenderer;

    use super::*;

    #[test]
    fn effect_slice_ord() {
        let slice1 = EffectSlice {
            slice: 0..32,
            group_index: 1,
            item_size: 48,
        };
        let slice2 = EffectSlice {
            slice: 32..64,
            group_index: 1,
            item_size: 48,
        };
        assert!(slice1 < slice2);
        assert!(slice1 <= slice2);
        assert!(slice2 > slice1);
        assert!(slice2 >= slice1);

        let slice3 = EffectSlice {
            slice: 0..32,
            group_index: 0,
            item_size: 48,
        };
        assert!(slice3 < slice1);
        assert!(slice3 < slice2);
        assert!(slice1 > slice3);
        assert!(slice2 > slice3);
    }

    #[test]
    fn slice_ref() {
        for (range, item_size, len, byte_size) in [
            (0..0, 0, 0, 0),
            (0..0, 16, 0, 0),
            (0..16, 0, 16, 0),
            (0..16, 32, 16, 16 * 32),
            (240..256, 48, 16, 16 * 48),
        ] {
            let sr = SliceRef { range, item_size };
            assert_eq!(sr.len(), len);
            assert_eq!(sr.byte_size(), byte_size);
        }
    }

    #[test]
    fn effect_buffer() {
        let renderer = MockRenderer::new();
        let render_device = renderer.device();
        //let render_queue = renderer.queue();

        let asset = Handle::weak(HandleId::random::<EffectAsset>());
        let capacity = 4096;
        let item_size = 64;
        let mut buffer = EffectBuffer::new(
            asset,
            capacity,
            item_size,
            &render_device,
            Some("my_buffer"),
        );

        assert_eq!(buffer.capacity, capacity.max(EffectBuffer::MIN_CAPACITY));
        assert_eq!(64, buffer.item_size);
        assert_eq!(0, buffer.used_size);
        assert!(buffer.free_slices.is_empty());

        assert_eq!(None, buffer.allocate_slice(buffer.capacity + 1, 64));

        let mut offset = 0;
        let mut slices = vec![];
        for size in [32, 128, 55, 148, 1, 2048, 42] {
            let slice = buffer.allocate_slice(size, 64);
            assert!(slice.is_some());
            let slice = slice.unwrap();
            assert_eq!(64, slice.item_size);
            assert_eq!(offset..offset + size, slice.range);
            slices.push(slice);
            offset += size;
        }
        assert_eq!(offset, buffer.used_size);

        assert_eq!(BufferState::Used, buffer.free_slice(slices[2].clone()));
        assert_eq!(1, buffer.free_slices.len());
        let free_slice = &buffer.free_slices[0];
        assert_eq!(160..215, *free_slice);
        assert_eq!(offset, buffer.used_size); // didn't move

        assert_eq!(BufferState::Used, buffer.free_slice(slices[3].clone()));
        assert_eq!(BufferState::Used, buffer.free_slice(slices[4].clone()));
        assert_eq!(BufferState::Used, buffer.free_slice(slices[5].clone()));
        assert_eq!(4, buffer.free_slices.len());
        assert_eq!(offset, buffer.used_size); // didn't move

        // this will collapse all the way to slices[1], the highest allocated
        assert_eq!(BufferState::Used, buffer.free_slice(slices[6].clone()));
        assert_eq!(0, buffer.free_slices.len()); // collapsed
        assert_eq!(160, buffer.used_size); // collapsed

        assert_eq!(BufferState::Used, buffer.free_slice(slices[0].clone()));
        assert_eq!(1, buffer.free_slices.len());
        assert_eq!(160, buffer.used_size); // didn't move

        // collapse all, and free buffer
        assert_eq!(BufferState::Free, buffer.free_slice(slices[1].clone()));
        assert_eq!(0, buffer.free_slices.len());
        assert_eq!(0, buffer.used_size); // collapsed and empty
    }

    #[test]
    fn pop_free_slice() {
        let renderer = MockRenderer::new();
        let render_device = renderer.device();
        //let render_queue = renderer.queue();

        let asset = Handle::weak(HandleId::random::<EffectAsset>());
        let capacity = 2048; //EffectBuffer::MIN_CAPACITY;
        assert!(capacity >= 2048); // otherwise the logic below breaks
        let item_size = 64;
        let mut buffer = EffectBuffer::new(
            asset,
            capacity,
            item_size,
            &render_device,
            Some("my_buffer"),
        );

        let slice0 = buffer.allocate_slice(32, item_size);
        assert!(slice0.is_some());
        let slice0 = slice0.unwrap();
        assert_eq!(slice0.range, 0..32);
        assert!(buffer.free_slices.is_empty());

        let slice1 = buffer.allocate_slice(1024, item_size);
        assert!(slice1.is_some());
        let slice1 = slice1.unwrap();
        assert_eq!(slice1.range, 32..1056);
        assert!(buffer.free_slices.is_empty());

        let state = buffer.free_slice(slice0);
        assert_eq!(state, BufferState::Used);
        assert_eq!(buffer.free_slices.len(), 1);
        assert_eq!(buffer.free_slices[0], 0..32);

        // Try to allocate a slice larger than slice0, such that slice0 cannot be
        // recycled, and instead the new slice has to be appended after all
        // existing ones.
        let slice2 = buffer.allocate_slice(64, item_size);
        assert!(slice2.is_some());
        let slice2 = slice2.unwrap();
        assert_eq!(slice2.range.start, slice1.range.end); // after slice1
        assert_eq!(slice2.range, 1056..1120);
        assert_eq!(buffer.free_slices.len(), 1);

        // Now allocate a small slice that fits, to recycle (part of) slice0.
        let slice3 = buffer.allocate_slice(16, item_size);
        assert!(slice3.is_some());
        let slice3 = slice3.unwrap();
        assert_eq!(slice3.range, 0..16);
        assert_eq!(buffer.free_slices.len(), 1); // split
        assert_eq!(buffer.free_slices[0], 16..32);

        // Allocate a second small slice that fits exactly the left space, completely
        // recycling
        let slice4 = buffer.allocate_slice(16, item_size);
        assert!(slice4.is_some());
        let slice4 = slice4.unwrap();
        assert_eq!(slice4.range, 16..32);
        assert!(buffer.free_slices.is_empty()); // recycled
    }

    #[test]
    fn effect_cache() {
        let renderer = MockRenderer::new();
        let render_device = renderer.device();
        let render_queue = renderer.queue();

        let mut effect_cache = EffectCache::new(render_device);
        assert_eq!(effect_cache.buffers().len(), 0);

        let asset = Handle::weak(HandleId::random::<EffectAsset>());
        let capacity = EffectBuffer::MIN_CAPACITY;
        let item_size = 32;

        let id1 = effect_cache.insert(asset.clone(), capacity, item_size, &render_queue);
        assert!(id1.is_valid());
        let slice1 = effect_cache.get_slice(id1);
        assert_eq!(slice1.item_size, item_size);
        assert_eq!(slice1.slice, 0..capacity);
        assert_eq!(effect_cache.buffers().len(), 1);

        let id2 = effect_cache.insert(asset.clone(), capacity, item_size, &render_queue);
        assert!(id2.is_valid());
        let slice2 = effect_cache.get_slice(id2);
        assert_eq!(slice2.item_size, item_size);
        assert_eq!(slice2.slice, 0..capacity);
        assert_eq!(effect_cache.buffers().len(), 2);

        let buffer_index = effect_cache.remove(id1);
        assert!(buffer_index.is_some());
        assert_eq!(buffer_index.unwrap(), 0);
        assert_eq!(effect_cache.buffers().len(), 2);
        {
            let buffers = effect_cache.buffers();
            assert!(buffers[0].is_none());
            assert!(buffers[1].is_some()); // id2
        }

        // Regression #60
        let id3 = effect_cache.insert(asset, capacity, item_size, &render_queue);
        assert!(id3.is_valid());
        let slice3 = effect_cache.get_slice(id3);
        assert_eq!(slice3.item_size, item_size);
        assert_eq!(slice3.slice, 0..capacity);
        assert_eq!(effect_cache.buffers().len(), 2);
        {
            let buffers = effect_cache.buffers();
            assert!(buffers[0].is_some()); // id3
            assert!(buffers[1].is_some()); // id2
        }
    }
}
