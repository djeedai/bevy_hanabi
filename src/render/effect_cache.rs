use std::{
    cmp::Ordering,
    num::{NonZeroU32, NonZeroU64},
    ops::Range,
};

use bevy::{
    asset::Handle,
    ecs::{component::Component, system::Resource},
    log::{trace, warn},
    render::{render_resource::*, renderer::RenderDevice},
    utils::{default, HashMap},
};
use bytemuck::cast_slice_mut;

use super::{buffer_table::BufferTableId, BufferBindingSource};
use crate::{
    asset::EffectAsset,
    render::{
        calc_hash, event::GpuChildInfo, GpuEffectMetadata, GpuSpawnerParams, LayoutFlags,
        StorageType as _, INDIRECT_INDEX_SIZE,
    },
    ParticleLayout,
};

/// Describes all particle slices of particles in the particle buffer
/// for a single effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectSlice {
    /// Slice into the underlying [`BufferVec`].
    ///
    /// This is measured in items, not bytes.
    pub slice: Range<u32>,
    /// Index of the buffer in the [`EffectCache`].
    pub buffer_index: u32,
    /// Particle layout of the effect.
    pub particle_layout: ParticleLayout,
}

impl Ord for EffectSlice {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.buffer_index.cmp(&other.buffer_index) {
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
    pub(crate) particle_layout: ParticleLayout,
}

impl SliceRef {
    /// The length of the slice, in number of items.
    #[allow(dead_code)]
    pub fn len(&self) -> u32 {
        self.range.end - self.range.start
    }

    /// The size in bytes of the slice.
    #[allow(dead_code)]
    pub fn byte_size(&self) -> usize {
        (self.len() as usize) * (self.particle_layout.min_binding_size().get() as usize)
    }

    pub fn range(&self) -> Range<u32> {
        self.range.clone()
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct SimBindGroupKey {
    buffer: Option<BufferId>,
    offset: u32,
    size: u32,
}

impl SimBindGroupKey {
    /// Invalid key, often used as placeholder.
    pub const INVALID: Self = Self {
        buffer: None,
        offset: u32::MAX,
        size: 0,
    };
}

impl From<&BufferBindingSource> for SimBindGroupKey {
    fn from(value: &BufferBindingSource) -> Self {
        Self {
            buffer: Some(value.buffer.id()),
            offset: value.offset,
            size: value.size.get(),
        }
    }
}

impl From<Option<&BufferBindingSource>> for SimBindGroupKey {
    fn from(value: Option<&BufferBindingSource>) -> Self {
        if let Some(bbs) = value {
            Self {
                buffer: Some(bbs.buffer.id()),
                offset: bbs.offset,
                size: bbs.size.get(),
            }
        } else {
            Self::INVALID
        }
    }
}

/// Storage for a single kind of effects, sharing the same buffer(s).
///
/// Currently only accepts a single unique item size (particle size), fixed at
/// creation.
///
/// Also currently only accepts instances of a unique effect asset, although
/// this restriction is purely for convenience and may be relaxed in the future
/// to improve batching.
#[derive(Debug)]
pub struct EffectBuffer {
    /// GPU buffer holding all particles for the entire group of effects.
    particle_buffer: Buffer,
    /// GPU buffer holding the indirection indices for the entire group of
    /// effects. This is a triple buffer containing:
    /// - the ping-pong alive particles and render indirect indices at offsets 0
    ///   and 1
    /// - the dead particle indices at offset 2
    indirect_index_buffer: Buffer,
    /// Layout of particles.
    particle_layout: ParticleLayout,
    /// Layout of the particle@1 bind group for the render pass.
    render_particles_buffer_layout: BindGroupLayout,
    /// Total buffer capacity, in number of particles.
    capacity: u32,
    /// Used buffer size, in number of particles, either from allocated slices
    /// or from slices in the free list.
    used_size: u32,
    /// Array of free slices for new allocations, sorted in increasing order in
    /// the buffer.
    free_slices: Vec<Range<u32>>,
    /// Compute pipeline for the effect update pass.
    // pub compute_pipeline: ComputePipeline, // FIXME - ComputePipelineId, to avoid duplicating per
    // instance!
    /// Handle of all effects common in this buffer. TODO - replace with
    /// compatible layout.
    asset: Handle<EffectAsset>,
    /// Bind group particle@1 of the simulation passes (init and udpate).
    sim_bind_group: Option<BindGroup>,
    /// Key the `sim_bind_group` was created from.
    sim_bind_group_key: SimBindGroupKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    /// The buffer is in use, with allocated resources.
    Used,
    /// Like `Used`, but the buffer was resized, so any bind group is
    /// nonetheless invalid.
    Resized,
    /// The buffer is free (its resources were deallocated).
    Free,
}

impl EffectBuffer {
    /// Minimum buffer capacity to allocate, in number of particles.
    // FIXME - Batching is broken due to binding a single GpuSpawnerParam instead of
    // N, and inability for a particle index to tell which Spawner it should
    // use. Setting this to 1 effectively ensures that all new buffers just fit
    // the effect, so batching never occurs.
    pub const MIN_CAPACITY: u32 = 1; // 65536; // at least 64k particles

    /// Create a new group and a GPU buffer to back it up.
    ///
    /// The buffer cannot contain less than [`MIN_CAPACITY`] particles. If
    /// `capacity` is smaller, it's rounded up to [`MIN_CAPACITY`].
    ///
    /// [`MIN_CAPACITY`]: EffectBuffer::MIN_CAPACITY
    pub fn new(
        buffer_index: u32,
        asset: Handle<EffectAsset>,
        capacity: u32,
        particle_layout: ParticleLayout,
        layout_flags: LayoutFlags,
        render_device: &RenderDevice,
    ) -> Self {
        trace!(
            "EffectBuffer::new(buffer_index={}, capacity={}, particle_layout={:?}, layout_flags={:?}, item_size={}B)",
            buffer_index,
            capacity,
            particle_layout,
            layout_flags,
            particle_layout.min_binding_size().get(),
        );

        // Calculate the clamped capacity of the group, in number of particles.
        let capacity = capacity.max(Self::MIN_CAPACITY);
        debug_assert!(
            capacity > 0,
            "Attempted to create a zero-sized effect buffer."
        );

        // Allocate the particle buffer itself, containing the attributes of each
        // particle.
        let particle_capacity_bytes: BufferAddress =
            capacity as u64 * particle_layout.min_binding_size().get();
        let particle_label = format!("hanabi:buffer:vfx{buffer_index}_particle");
        let particle_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some(&particle_label),
            size: particle_capacity_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Each indirect buffer stores 3 arrays of u32, of length the number of
        // particles.
        let capacity_bytes: BufferAddress = capacity as u64 * 4 * 3;

        let indirect_label = format!("hanabi:buffer:vfx{buffer_index}_indirect");
        let indirect_index_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some(&indirect_label),
            size: capacity_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        // Set content
        {
            // Scope get_mapped_range_mut() to force a drop before unmap()
            {
                let slice: &mut [u8] = &mut indirect_index_buffer
                    .slice(..capacity_bytes)
                    .get_mapped_range_mut();
                let slice: &mut [u32] = cast_slice_mut(slice);
                for index in 0..capacity {
                    slice[3 * index as usize + 2] = capacity - 1 - index;
                }
            }
            indirect_index_buffer.unmap();
        }

        // Create the render layout.
        let spawner_params_size = GpuSpawnerParams::aligned_size(
            render_device.limits().min_storage_buffer_offset_alignment,
        );
        let entries = [
            // @group(1) @binding(0) var<storage, read> particle_buffer : ParticleBuffer;
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(particle_layout.min_binding_size()),
                },
                count: None,
            },
            // @group(1) @binding(1) var<storage, read> indirect_buffer : IndirectBuffer;
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(INDIRECT_INDEX_SIZE as u64).unwrap()),
                },
                count: None,
            },
            // @group(1) @binding(2) var<storage, read> spawner : Spawner;
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: Some(spawner_params_size),
                },
                count: None,
            },
        ];
        let label = format!("hanabi:bind_group_layout:render:particles@1:vfx{buffer_index}");
        trace!(
            "Creating render layout '{}' with {} entries (flags: {:?})",
            label,
            entries.len(),
            layout_flags
        );
        let render_particles_buffer_layout =
            render_device.create_bind_group_layout(&label[..], &entries[..]);

        Self {
            particle_buffer,
            indirect_index_buffer,
            particle_layout,
            render_particles_buffer_layout,
            capacity,
            used_size: 0,
            free_slices: vec![],
            asset,
            sim_bind_group: None,
            sim_bind_group_key: SimBindGroupKey::INVALID,
        }
    }

    pub fn render_particles_buffer_layout(&self) -> &BindGroupLayout {
        &self.render_particles_buffer_layout
    }

    #[inline]
    pub fn particle_buffer(&self) -> &Buffer {
        &self.particle_buffer
    }

    #[inline]
    pub fn indirect_index_buffer(&self) -> &Buffer {
        &self.indirect_index_buffer
    }

    #[inline]
    pub fn particle_offset(&self, row: u32) -> u32 {
        self.particle_layout.min_binding_size().get() as u32 * row
    }

    #[inline]
    pub fn indirect_index_offset(&self, row: u32) -> u32 {
        row * 12
    }

    /// Return a binding for the entire particle buffer.
    pub fn max_binding(&self) -> BindingResource {
        let capacity_bytes = self.capacity as u64 * self.particle_layout.min_binding_size().get();
        BindingResource::Buffer(BufferBinding {
            buffer: &self.particle_buffer,
            offset: 0,
            size: Some(NonZeroU64::new(capacity_bytes).unwrap()),
        })
    }

    /// Return a binding source for the entire particle buffer.
    pub fn max_binding_source(&self) -> BufferBindingSource {
        let capacity_bytes = self.capacity * self.particle_layout.min_binding_size32().get();
        BufferBindingSource {
            buffer: self.particle_buffer.clone(),
            offset: 0,
            size: NonZeroU32::new(capacity_bytes).unwrap(),
        }
    }

    /// Return a binding for the entire indirect buffer associated with the
    /// current effect buffer.
    pub fn indirect_index_max_binding(&self) -> BindingResource {
        let capacity_bytes = self.capacity as u64 * 12;
        BindingResource::Buffer(BufferBinding {
            buffer: &self.indirect_index_buffer,
            offset: 0,
            size: Some(NonZeroU64::new(capacity_bytes).unwrap()),
        })
    }

    /// Create the "particle" bind group @1 for the init and update passes if
    /// needed.
    ///
    /// The `buffer_index` must be the index of the current [`EffectBuffer`]
    /// inside the [`EffectCache`].
    pub fn create_particle_sim_bind_group(
        &mut self,
        layout: &BindGroupLayout,
        buffer_index: u32,
        render_device: &RenderDevice,
        parent_binding_source: Option<&BufferBindingSource>,
    ) {
        let key: SimBindGroupKey = parent_binding_source.into();
        if self.sim_bind_group.is_some() && self.sim_bind_group_key == key {
            return;
        }

        let label = format!("hanabi:bind_group:sim:particle@1:vfx{}", buffer_index);
        let entries: &[BindGroupEntry] =
            if let Some(parent_binding) = parent_binding_source.as_ref().map(|bbs| bbs.binding()) {
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.max_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.indirect_index_max_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: parent_binding,
                    },
                ]
            } else {
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.max_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.indirect_index_max_binding(),
                    },
                ]
            };

        trace!(
            "Create particle simulation bind group '{}' with {} entries (has_parent:{})",
            label,
            entries.len(),
            parent_binding_source.is_some(),
        );
        let bind_group = render_device.create_bind_group(Some(&label[..]), layout, entries);
        self.sim_bind_group = Some(bind_group);
        self.sim_bind_group_key = key;
    }

    /// Invalidate any existing simulate bind group.
    ///
    /// Invalidate any existing bind group previously created by
    /// [`create_particle_sim_bind_group()`], generally because a buffer was
    /// re-allocated. This forces a re-creation of the bind group
    /// next time [`create_particle_sim_bind_group()`] is called.
    ///
    /// [`create_particle_sim_bind_group()`]: self::EffectBuffer::create_particle_sim_bind_group
    #[allow(dead_code)] // FIXME - review this...
    fn invalidate_particle_sim_bind_group(&mut self) {
        self.sim_bind_group = None;
        self.sim_bind_group_key = SimBindGroupKey::INVALID;
    }

    /// Return the cached particle@1 bind group for the simulation (init and
    /// update) passes.
    ///
    /// This is the per-buffer bind group at binding @1 which binds all
    /// per-buffer resources shared by all effect instances batched in a single
    /// buffer. The bind group is created by
    /// [`create_particle_sim_bind_group()`], and cached until a call to
    /// [`invalidate_particle_sim_bind_groups()`] clears the
    /// cached reference.
    ///
    /// [`create_particle_sim_bind_group()`]: self::EffectBuffer::create_particle_sim_bind_group
    /// [`invalidate_particle_sim_bind_groups()`]: self::EffectBuffer::invalidate_particle_sim_bind_groups
    pub fn particle_sim_bind_group(&self) -> Option<&BindGroup> {
        self.sim_bind_group.as_ref()
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
    pub fn allocate_slice(
        &mut self,
        capacity: u32,
        particle_layout: &ParticleLayout,
    ) -> Option<SliceRef> {
        trace!(
            "EffectBuffer::allocate_slice: capacity={} particle_layout={:?} item_size={}",
            capacity,
            particle_layout,
            particle_layout.min_binding_size().get(),
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

        trace!("-> allocated slice {:?}", range);
        Some(SliceRef {
            range,
            particle_layout: particle_layout.clone(),
        })
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

/// A single cached effect (all groups) in the [`EffectCache`].
#[derive(Debug, Component)]
pub(crate) struct CachedEffect {
    /// Index into the [`EffectCache::buffers`] of the buffer storing the
    /// particles for this effect.
    pub buffer_index: u32,
    /// The effect slice within that buffer.
    pub slice: SliceRef,
}

/// The indices in the indirect dispatch buffers for a single effect, as well as
/// that of the metadata buffer.
#[derive(Debug, Default, Clone, Copy, Component)]
pub(crate) struct DispatchBufferIndices {
    /// The index of the [`GpuDispatchIndirect`] in
    /// [`EffectsMeta::update_dispatch_indirect_buffer`].
    ///
    /// [`EffectsMeta::update_dispatch_indirect_buffer`]: super::EffectsMeta::update_dispatch_indirect_buffer
    pub(crate) update_dispatch_indirect_buffer_table_id: BufferTableId,

    /// The index of the [`GpuEffectMetadata`] in
    /// [`EffectsMeta::effect_metadata_buffer`].
    ///
    /// [`EffectsMeta::effect_metadata_buffer`]: super::EffectsMeta::effect_metadata_buffer
    pub(crate) effect_metadata_buffer_table_id: BufferTableId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ParticleBindGroupLayoutKey {
    pub min_binding_size: NonZeroU32,
    pub parent_min_binding_size: Option<NonZeroU32>,
}

/// Cache for effect instances sharing common GPU data structures.
#[derive(Resource)]
pub struct EffectCache {
    /// Render device the GPU resources (buffers) are allocated from.
    render_device: RenderDevice,
    /// Collection of effect buffers managed by this cache. Some buffers might
    /// be `None` if the entry is not used. Since the buffers are referenced
    /// by index, we cannot move them once they're allocated.
    buffers: Vec<Option<EffectBuffer>>,
    /// Cache of bind group layouts for the particle@1 bind groups of the
    /// simulation passes (init and update). Since all bindings depend only
    /// on buffers managed by the [`EffectCache`], we also cache the layouts
    /// here for convenience.
    particle_bind_group_layouts: HashMap<ParticleBindGroupLayoutKey, BindGroupLayout>,
    /// Cache of bind group layouts for the metadata@3 bind group of the init
    /// pass.
    metadata_init_bind_group_layout: [Option<BindGroupLayout>; 2],
    /// Cache of bind group layouts for the metadata@3 bind group of the
    /// updatepass.
    metadata_update_bind_group_layouts: HashMap<u32, BindGroupLayout>,
}

impl EffectCache {
    pub fn new(device: RenderDevice) -> Self {
        Self {
            render_device: device,
            buffers: vec![],
            particle_bind_group_layouts: default(),
            metadata_init_bind_group_layout: [None, None],
            metadata_update_bind_group_layouts: default(),
        }
    }

    /// Get all the buffer slots. Unallocated slots are `None`. This can be
    /// indexed by the buffer index.
    #[allow(dead_code)]
    #[inline]
    pub fn buffers(&self) -> &[Option<EffectBuffer>] {
        &self.buffers
    }

    /// Get all the buffer slots. Unallocated slots are `None`. This can be
    /// indexed by the buffer index.
    #[allow(dead_code)]
    #[inline]
    pub fn buffers_mut(&mut self) -> &mut [Option<EffectBuffer>] {
        &mut self.buffers
    }

    /// Fetch a specific buffer by index.
    #[allow(dead_code)]
    #[inline]
    pub fn get_buffer(&self, buffer_index: u32) -> Option<&EffectBuffer> {
        self.buffers.get(buffer_index as usize)?.as_ref()
    }

    /// Fetch a specific buffer by index.
    #[allow(dead_code)]
    #[inline]
    pub fn get_buffer_mut(&mut self, buffer_index: u32) -> Option<&mut EffectBuffer> {
        self.buffers.get_mut(buffer_index as usize)?.as_mut()
    }

    /// Invalidate all the particle@1 bind group for all buffers.
    ///
    /// This iterates over all valid buffers and calls
    /// [`EffectBuffer::invalidate_particle_sim_bind_group()`] on each one.
    #[allow(dead_code)] // FIXME - review this...
    pub fn invalidate_particle_sim_bind_groups(&mut self) {
        for buffer in self.buffers.iter_mut().flatten() {
            buffer.invalidate_particle_sim_bind_group();
        }
    }

    /// Insert a new effect instance in the cache.
    pub fn insert(
        &mut self,
        asset: Handle<EffectAsset>,
        capacity: u32,
        particle_layout: &ParticleLayout,
        layout_flags: LayoutFlags,
    ) -> CachedEffect {
        trace!("Inserting new effect into cache: capacity={capacity}");
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
                        .allocate_slice(capacity, particle_layout)
                        .map(|slice| (buffer_index, slice))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                // Cannot find any suitable buffer; allocate a new one
                let buffer_index = self.buffers.iter().position(|buf| buf.is_none()).unwrap_or(self.buffers.len());
                let byte_size = capacity.checked_mul(particle_layout.min_binding_size().get() as u32).unwrap_or_else(|| panic!(
                    "Effect size overflow: capacities={:?} particle_layout={:?} item_size={}",
                    capacity, particle_layout, particle_layout.min_binding_size().get()
                ));
                trace!(
                    "Creating new effect buffer #{} for effect {:?} (capacities={:?}, particle_layout={:?} item_size={}, byte_size={})",
                    buffer_index,
                    asset,
                    capacity,
                    particle_layout,
                    particle_layout.min_binding_size().get(),
                    byte_size
                );
                let mut buffer = EffectBuffer::new(
                    buffer_index as u32,
                    asset,
                    capacity,
                    particle_layout.clone(),
                    layout_flags,
                    &self.render_device,
                );
                let slice_ref = buffer.allocate_slice(capacity, particle_layout).unwrap();
                if buffer_index >= self.buffers.len() {
                    self.buffers.push(Some(buffer));
                } else {
                    debug_assert!(self.buffers[buffer_index].is_none());
                    self.buffers[buffer_index] = Some(buffer);
                }
                (buffer_index, slice_ref)
            });

        let slice = SliceRef {
            range: slice.range.clone(),
            particle_layout: slice.particle_layout,
        };

        trace!(
            "Insert effect buffer_index={} slice={}B particle_layout={:?}",
            buffer_index,
            slice.particle_layout.min_binding_size().get(),
            slice.particle_layout,
        );
        CachedEffect {
            buffer_index: buffer_index as u32,
            slice,
        }
    }

    /// Remove an effect from the cache. If this was the last effect, drop the
    /// underlying buffer and return the index of the dropped buffer.
    pub fn remove(&mut self, cached_effect: &CachedEffect) -> Result<BufferState, ()> {
        // Resolve the buffer by index
        let Some(maybe_buffer) = self.buffers.get_mut(cached_effect.buffer_index as usize) else {
            return Err(());
        };
        let Some(buffer) = maybe_buffer.as_mut() else {
            return Err(());
        };

        // Free the slice inside the resolved buffer
        if buffer.free_slice(cached_effect.slice.clone()) == BufferState::Free {
            *maybe_buffer = None;
            return Ok(BufferState::Free);
        }

        Ok(BufferState::Used)
    }

    //
    // Bind group layouts
    //

    /// Ensure a bind group layout exists for the bind group @1 ("particles")
    /// for use with the given min binding sizes.
    pub fn ensure_particle_bind_group_layout(
        &mut self,
        min_binding_size: NonZeroU32,
        parent_min_binding_size: Option<NonZeroU32>,
    ) -> &BindGroupLayout {
        // FIXME - This "ensure" pattern means we never de-allocate entries. This is
        // probably fine, because there's a limited number of realistic combinations,
        // but could cause wastes if e.g. loading widely different scenes.
        let key = ParticleBindGroupLayoutKey {
            min_binding_size,
            parent_min_binding_size,
        };
        self.particle_bind_group_layouts
            .entry(key)
            .or_insert_with(|| {
                trace!("Creating new particle sim bind group @1 for min_binding_size={} parent_min_binding_size={:?}", min_binding_size, parent_min_binding_size);
                create_particle_sim_bind_group_layout(
                    &self.render_device,
                    min_binding_size,
                    parent_min_binding_size,
                )
            })
    }

    /// Get the bind group layout for the bind group @1 ("particles") for use
    /// with the given min binding sizes.
    pub fn particle_bind_group_layout(
        &self,
        min_binding_size: NonZeroU32,
        parent_min_binding_size: Option<NonZeroU32>,
    ) -> Option<&BindGroupLayout> {
        let key = ParticleBindGroupLayoutKey {
            min_binding_size,
            parent_min_binding_size,
        };
        self.particle_bind_group_layouts.get(&key)
    }

    /// Ensure a bind group layout exists for the metadata@3 bind group of
    /// the init pass.
    pub fn ensure_metadata_init_bind_group_layout(&mut self, consume_gpu_spawn_events: bool) {
        let layout = &mut self.metadata_init_bind_group_layout[consume_gpu_spawn_events as usize];
        if layout.is_none() {
            *layout = Some(create_metadata_init_bind_group_layout(
                &self.render_device,
                consume_gpu_spawn_events,
            ));
        }
    }

    /// Get the bind group layout for the metadata@3 bind group of the init
    /// pass.
    pub fn metadata_init_bind_group_layout(
        &self,
        consume_gpu_spawn_events: bool,
    ) -> Option<&BindGroupLayout> {
        self.metadata_init_bind_group_layout[consume_gpu_spawn_events as usize].as_ref()
    }

    /// Ensure a bind group layout exists for the metadata@3 bind group of
    /// the update pass.
    pub fn ensure_metadata_update_bind_group_layout(&mut self, num_event_buffers: u32) {
        self.metadata_update_bind_group_layouts
            .entry(num_event_buffers)
            .or_insert_with(|| {
                create_metadata_update_bind_group_layout(&self.render_device, num_event_buffers)
            });
    }

    /// Get the bind group layout for the metadata@3 bind group of the
    /// update pass.
    pub fn metadata_update_bind_group_layout(
        &self,
        num_event_buffers: u32,
    ) -> Option<&BindGroupLayout> {
        self.metadata_update_bind_group_layouts
            .get(&num_event_buffers)
    }

    //
    // Bind groups
    //

    /// Get the "particle" bind group for the simulation (init and update)
    /// passes a cached effect stored in a given GPU particle buffer.
    pub fn particle_sim_bind_group(&self, buffer_index: u32) -> Option<&BindGroup> {
        self.buffers[buffer_index as usize]
            .as_ref()
            .and_then(|eb| eb.particle_sim_bind_group())
    }

    pub fn create_particle_sim_bind_group(
        &mut self,
        buffer_index: u32,
        render_device: &RenderDevice,
        min_binding_size: NonZeroU32,
        parent_min_binding_size: Option<NonZeroU32>,
        parent_binding_source: Option<&BufferBindingSource>,
    ) -> Result<(), ()> {
        // Create the bind group
        let layout = self
            .ensure_particle_bind_group_layout(min_binding_size, parent_min_binding_size)
            .clone();
        let slot = self.buffers.get_mut(buffer_index as usize).ok_or(())?;
        let effect_buffer = slot.as_mut().ok_or(())?;
        effect_buffer.create_particle_sim_bind_group(
            &layout,
            buffer_index,
            render_device,
            parent_binding_source,
        );
        Ok(())
    }
}

/// Create the bind group layout for the "particle" group (@1) of the init and
/// update passes.
fn create_particle_sim_bind_group_layout(
    render_device: &RenderDevice,
    particle_layout_min_binding_size: NonZeroU32,
    parent_particle_layout_min_binding_size: Option<NonZeroU32>,
) -> BindGroupLayout {
    let mut entries = Vec::with_capacity(3);

    // @group(1) @binding(0) var<storage, read_write> particle_buffer :
    // ParticleBuffer
    entries.push(BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(particle_layout_min_binding_size.into()),
        },
        count: None,
    });

    // @group(1) @binding(1) var<storage, read_write> indirect_buffer :
    // IndirectBuffer
    entries.push(BindGroupLayoutEntry {
        binding: 1,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(NonZeroU64::new(INDIRECT_INDEX_SIZE as _).unwrap()),
        },
        count: None,
    });

    // @group(1) @binding(2) var<storage, read> parent_particle_buffer :
    // ParentParticleBuffer;
    if let Some(min_binding_size) = parent_particle_layout_min_binding_size {
        entries.push(BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(min_binding_size.into()),
            },
            count: None,
        });
    }

    let hash = calc_hash(&entries);
    let label = format!("hanabi:bind_group_layout:sim:particles_{:016X}", hash);
    trace!(
        "Creating particle bind group layout '{}' for init pass with {} entries. (parent_buffer:{})",
        label,
        entries.len(),
        parent_particle_layout_min_binding_size.is_some(),
    );
    render_device.create_bind_group_layout(&label[..], &entries)
}

/// Create the bind group layout for the metadata@3 bind group of the init pass.
fn create_metadata_init_bind_group_layout(
    render_device: &RenderDevice,
    consume_gpu_spawn_events: bool,
) -> BindGroupLayout {
    let storage_alignment = render_device.limits().min_storage_buffer_offset_alignment;
    let effect_metadata_size = GpuEffectMetadata::aligned_size(storage_alignment);

    let mut entries = Vec::with_capacity(3);

    // @group(3) @binding(0) var<storage, read_write> effect_metadata :
    // EffectMetadata;
    entries.push(BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            // This WGSL struct is manually padded, so the Rust type GpuEffectMetadata doesn't
            // reflect its true min size.
            min_binding_size: Some(effect_metadata_size),
        },
        count: None,
    });

    if consume_gpu_spawn_events {
        // @group(3) @binding(1) var<storage, read> child_info_buffer : ChildInfoBuffer;
        entries.push(BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(GpuChildInfo::min_size()),
            },
            count: None,
        });

        // @group(3) @binding(2) var<storage, read> event_buffer : EventBuffer;
        entries.push(BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            },
            count: None,
        });
    }

    let hash = calc_hash(&entries);
    let label = format!(
        "hanabi:bind_group_layout:init:metadata@3_{}{:016X}",
        if consume_gpu_spawn_events {
            "events"
        } else {
            "noevent"
        },
        hash
    );
    trace!(
        "Creating metadata@3 bind group layout '{}' for init pass with {} entries. (consume_gpu_spawn_events:{})",
        label,
        entries.len(),
        consume_gpu_spawn_events,
    );
    render_device.create_bind_group_layout(&label[..], &entries)
}

/// Create the bind group layout for the metadata@3 bind group of the update
/// pass.
fn create_metadata_update_bind_group_layout(
    render_device: &RenderDevice,
    num_event_buffers: u32,
) -> BindGroupLayout {
    let storage_alignment = render_device.limits().min_storage_buffer_offset_alignment;
    let effect_metadata_size = GpuEffectMetadata::aligned_size(storage_alignment);

    let mut entries = Vec::with_capacity(num_event_buffers as usize + 2);

    // @group(3) @binding(0) var<storage, read_write> effect_metadata :
    // EffectMetadata;
    entries.push(BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            // This WGSL struct is manually padded, so the Rust type GpuEffectMetadata doesn't
            // reflect its true min size.
            min_binding_size: Some(effect_metadata_size),
        },
        count: None,
    });

    if num_event_buffers > 0 {
        // @group(3) @binding(1) var<storage, read_write> child_infos : array<ChildInfo,
        // N>;
        entries.push(BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: Some(GpuChildInfo::min_size()),
            },
            count: None,
        });

        for i in 0..num_event_buffers {
            // @group(3) @binding(2+i) var<storage, read_write> event_buffer_#i :
            // EventBuffer;
            entries.push(BindGroupLayoutEntry {
                binding: 2 + i,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                },
                count: None,
            });
        }
    }

    let hash = calc_hash(&entries);
    let label = format!("hanabi:bind_group_layout:update:metadata_{:016X}", hash);
    trace!(
        "Creating particle bind group layout '{}' for init update with {} entries. (num_event_buffers:{})",
        label,
        entries.len(),
        num_event_buffers,
    );
    render_device.create_bind_group_layout(&label[..], &entries)
}

#[cfg(all(test, feature = "gpu_tests"))]
mod gpu_tests {
    use std::borrow::Cow;

    use bevy::math::Vec4;

    use super::*;
    use crate::{
        graph::{Value, VectorValue},
        test_utils::MockRenderer,
        Attribute, AttributeInner,
    };

    #[test]
    fn effect_slice_ord() {
        let particle_layout = ParticleLayout::new().append(Attribute::POSITION).build();
        let slice1 = EffectSlice {
            slice: 0..32,
            buffer_index: 1,
            particle_layout: particle_layout.clone(),
        };
        let slice2 = EffectSlice {
            slice: 32..64,
            buffer_index: 1,
            particle_layout: particle_layout.clone(),
        };
        assert!(slice1 < slice2);
        assert!(slice1 <= slice2);
        assert!(slice2 > slice1);
        assert!(slice2 >= slice1);

        let slice3 = EffectSlice {
            slice: 0..32,
            buffer_index: 0,
            particle_layout,
        };
        assert!(slice3 < slice1);
        assert!(slice3 < slice2);
        assert!(slice1 > slice3);
        assert!(slice2 > slice3);
    }

    const F4A_INNER: &AttributeInner = &AttributeInner::new(
        Cow::Borrowed("F4A"),
        Value::Vector(VectorValue::new_vec4(Vec4::ONE)),
    );
    const F4B_INNER: &AttributeInner = &AttributeInner::new(
        Cow::Borrowed("F4B"),
        Value::Vector(VectorValue::new_vec4(Vec4::ONE)),
    );
    const F4C_INNER: &AttributeInner = &AttributeInner::new(
        Cow::Borrowed("F4C"),
        Value::Vector(VectorValue::new_vec4(Vec4::ONE)),
    );
    const F4D_INNER: &AttributeInner = &AttributeInner::new(
        Cow::Borrowed("F4D"),
        Value::Vector(VectorValue::new_vec4(Vec4::ONE)),
    );

    const F4A: Attribute = Attribute(F4A_INNER);
    const F4B: Attribute = Attribute(F4B_INNER);
    const F4C: Attribute = Attribute(F4C_INNER);
    const F4D: Attribute = Attribute(F4D_INNER);

    #[test]
    fn slice_ref() {
        let l16 = ParticleLayout::new().append(F4A).build();
        assert_eq!(16, l16.size());
        let l32 = ParticleLayout::new().append(F4A).append(F4B).build();
        assert_eq!(32, l32.size());
        let l48 = ParticleLayout::new()
            .append(F4A)
            .append(F4B)
            .append(F4C)
            .build();
        assert_eq!(48, l48.size());
        for (range, particle_layout, len, byte_size) in [
            (0..0, &l16, 0, 0),
            (0..16, &l16, 16, 16 * 16),
            (0..16, &l32, 16, 16 * 32),
            (240..256, &l48, 16, 16 * 48),
        ] {
            let sr = SliceRef {
                range,
                particle_layout: particle_layout.clone(),
            };
            assert_eq!(sr.len(), len);
            assert_eq!(sr.byte_size(), byte_size);
        }
    }

    #[test]
    fn effect_buffer() {
        let renderer = MockRenderer::new();
        let render_device = renderer.device();

        let l64 = ParticleLayout::new()
            .append(F4A)
            .append(F4B)
            .append(F4C)
            .append(F4D)
            .build();
        assert_eq!(64, l64.size());

        let asset = Handle::<EffectAsset>::default();
        let capacity = 4096;
        let mut buffer = EffectBuffer::new(
            42,
            asset,
            capacity,
            l64.clone(),
            LayoutFlags::NONE,
            &render_device,
        );

        assert_eq!(buffer.capacity, capacity.max(EffectBuffer::MIN_CAPACITY));
        assert_eq!(64, buffer.particle_layout.size());
        assert_eq!(64, buffer.particle_layout.min_binding_size().get());
        assert_eq!(0, buffer.used_size);
        assert!(buffer.free_slices.is_empty());

        assert_eq!(None, buffer.allocate_slice(buffer.capacity + 1, &l64));

        let mut offset = 0;
        let mut slices = vec![];
        for size in [32, 128, 55, 148, 1, 2048, 42] {
            let slice = buffer.allocate_slice(size, &l64);
            assert!(slice.is_some());
            let slice = slice.unwrap();
            assert_eq!(64, slice.particle_layout.size());
            assert_eq!(64, buffer.particle_layout.min_binding_size().get());
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

        let l64 = ParticleLayout::new()
            .append(F4A)
            .append(F4B)
            .append(F4C)
            .append(F4D)
            .build();
        assert_eq!(64, l64.size());

        let asset = Handle::<EffectAsset>::default();
        let capacity = 2048; // EffectBuffer::MIN_CAPACITY;
        assert!(capacity >= 2048); // otherwise the logic below breaks
        let mut buffer = EffectBuffer::new(
            42,
            asset,
            capacity,
            l64.clone(),
            LayoutFlags::NONE,
            &render_device,
        );

        let slice0 = buffer.allocate_slice(32, &l64);
        assert!(slice0.is_some());
        let slice0 = slice0.unwrap();
        assert_eq!(slice0.range, 0..32);
        assert!(buffer.free_slices.is_empty());

        let slice1 = buffer.allocate_slice(1024, &l64);
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
        let slice2 = buffer.allocate_slice(64, &l64);
        assert!(slice2.is_some());
        let slice2 = slice2.unwrap();
        assert_eq!(slice2.range.start, slice1.range.end); // after slice1
        assert_eq!(slice2.range, 1056..1120);
        assert_eq!(buffer.free_slices.len(), 1);

        // Now allocate a small slice that fits, to recycle (part of) slice0.
        let slice3 = buffer.allocate_slice(16, &l64);
        assert!(slice3.is_some());
        let slice3 = slice3.unwrap();
        assert_eq!(slice3.range, 0..16);
        assert_eq!(buffer.free_slices.len(), 1); // split
        assert_eq!(buffer.free_slices[0], 16..32);

        // Allocate a second small slice that fits exactly the left space, completely
        // recycling
        let slice4 = buffer.allocate_slice(16, &l64);
        assert!(slice4.is_some());
        let slice4 = slice4.unwrap();
        assert_eq!(slice4.range, 16..32);
        assert!(buffer.free_slices.is_empty()); // recycled
    }

    #[test]
    fn effect_cache() {
        let renderer = MockRenderer::new();
        let render_device = renderer.device();

        let l32 = ParticleLayout::new().append(F4A).append(F4B).build();
        assert_eq!(32, l32.size());

        let mut effect_cache = EffectCache::new(render_device);
        assert_eq!(effect_cache.buffers().len(), 0);

        let asset = Handle::<EffectAsset>::default();
        let capacity = EffectBuffer::MIN_CAPACITY;
        let item_size = l32.size();

        // Insert an effect
        let effect1 = effect_cache.insert(asset.clone(), capacity, &l32, LayoutFlags::NONE);
        //assert!(effect1.is_valid());
        let slice1 = &effect1.slice;
        assert_eq!(slice1.len(), capacity);
        assert_eq!(
            slice1.particle_layout.min_binding_size().get() as u32,
            item_size
        );
        assert_eq!(slice1.range, 0..capacity);
        assert_eq!(effect_cache.buffers().len(), 1);

        // Insert a second copy of the same effect
        let effect2 = effect_cache.insert(asset.clone(), capacity, &l32, LayoutFlags::NONE);
        //assert!(effect2.is_valid());
        let slice2 = &effect2.slice;
        assert_eq!(slice2.len(), capacity);
        assert_eq!(
            slice2.particle_layout.min_binding_size().get() as u32,
            item_size
        );
        assert_eq!(slice2.range, 0..capacity);
        assert_eq!(effect_cache.buffers().len(), 2);

        // Remove the first effect instance
        let buffer_state = effect_cache.remove(&effect1).unwrap();
        // Note: currently batching is disabled, so each instance has its own buffer,
        // which becomes unused once the instance is destroyed.
        assert_eq!(buffer_state, BufferState::Free);
        assert_eq!(effect_cache.buffers().len(), 2);
        {
            let buffers = effect_cache.buffers();
            assert!(buffers[0].is_none());
            assert!(buffers[1].is_some()); // id2
        }

        // Regression #60
        let effect3 = effect_cache.insert(asset, capacity, &l32, LayoutFlags::NONE);
        //assert!(effect3.is_valid());
        let slice3 = &effect3.slice;
        assert_eq!(slice3.len(), capacity);
        assert_eq!(
            slice3.particle_layout.min_binding_size().get() as u32,
            item_size
        );
        assert_eq!(slice3.range, 0..capacity);
        // Note: currently batching is disabled, so each instance has its own buffer.
        assert_eq!(effect_cache.buffers().len(), 2);
        {
            let buffers = effect_cache.buffers();
            assert!(buffers[0].is_some()); // id3
            assert!(buffers[1].is_some()); // id2
        }
    }
}
