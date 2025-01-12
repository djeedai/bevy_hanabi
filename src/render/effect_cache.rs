use std::{cmp::Ordering, num::NonZeroU64, ops::Range};

use bevy::{
    asset::Handle,
    ecs::{component::Component, system::Resource},
    log::{trace, warn},
    render::{render_resource::*, renderer::RenderDevice},
    utils::{default, HashMap},
};
use bytemuck::cast_slice_mut;

use super::{buffer_table::BufferTableId, AddedEffectGroup};
use crate::{
    asset::EffectAsset,
    render::{
        GpuDispatchIndirect, GpuParticleGroup, GpuSpawnerParams, LayoutFlags, StorageType as _,
    },
    ParticleLayout,
};

/// Describes all particle groups' slices of particles in the particle buffer
/// for a single effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectSlices {
    /// Slices into the underlying [`BufferVec`]` of the group.
    ///
    /// The length of this vector is the number of particle groups plus one.
    /// The range of the first group is (slices[0]..slices[1]), the index of
    /// the second group is (slices[1]..slices[2]), etc.
    ///
    /// This is measured in items, not bytes.
    pub slices: Vec<u32>,
    /// Index of the buffer in the [`EffectCache`].
    pub buffer_index: u32,
    /// Particle layout of the effect.
    pub particle_layout: ParticleLayout,
}

impl Ord for EffectSlices {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.buffer_index.cmp(&other.buffer_index) {
            Ordering::Equal => self.slices.first().cmp(&other.slices.first()),
            ord => ord,
        }
    }
}

impl PartialOrd for EffectSlices {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Describes all particle groups' slices of particles in the particle buffer
/// for a single effect, as well as the [`DispatchBufferIndices`].
#[derive(Debug)]
pub struct SlicesRef {
    pub ranges: Vec<u32>,
    /// Size of a single item in the slice. Currently equal to the unique size
    /// of all items in an [`EffectBuffer`] (no mixed size supported in same
    /// buffer), so cached only for convenience.
    particle_layout: ParticleLayout,
}

impl SlicesRef {
    pub fn group_count(&self) -> u32 {
        debug_assert!(self.ranges.len() >= 2);
        (self.ranges.len() - 1) as u32
    }

    #[allow(dead_code)]
    pub fn group_capacity(&self, group_index: u32) -> u32 {
        assert!(group_index + 1 < self.ranges.len() as u32);
        let start = self.ranges[group_index as usize];
        let end = self.ranges[group_index as usize + 1];
        end - start
    }

    #[allow(dead_code)]
    pub fn total_capacity(&self) -> u32 {
        if self.ranges.is_empty() {
            0
        } else {
            debug_assert!(self.ranges.len() >= 2);
            let start = self.ranges[0];
            let end = self.ranges[self.ranges.len() - 1];
            end - start
        }
    }

    pub fn particle_layout(&self) -> &ParticleLayout {
        &self.particle_layout
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
    particle_layout: ParticleLayout,
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
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct SimulateBindGroupKey {
    buffer: Option<BufferId>,
    offset: u32,
    size: u32,
}

impl SimulateBindGroupKey {
    /// Invalid key, often used as placeholder.
    pub const INVALID: Self = Self {
        buffer: None,
        offset: u32::MAX,
        size: 0,
    };
}

/// Storage for a single kind of effects, sharing the same buffer(s).
///
/// Currently only accepts a single unique item size (particle size), fixed at
/// creation. Also currently only accepts instances of a unique effect asset,
/// although this restriction is purely for convenience and may be relaxed in
/// the future to improve batching.
#[derive(Debug)]
pub struct EffectBuffer {
    /// GPU buffer holding all particles for the entire group of effects.
    particle_buffer: Buffer,
    /// GPU buffer holding the indirection indices for the entire group of
    /// effects. This is a triple buffer containing:
    /// - the ping-pong alive particles and render indirect indices at offsets 0
    ///   and 1
    /// - the dead particle indices at offset 2
    indirect_buffer: Buffer,
    /// Layout of particles.
    particle_layout: ParticleLayout,
    /// Flags
    layout_flags: LayoutFlags,
    /// -
    particles_buffer_layout_sim: BindGroupLayout,
    /// -
    particles_buffer_layout_with_dispatch: BindGroupLayout,
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
    /// Bind group for the per-buffer data (group @1) of the init and update
    /// passes.
    simulate_bind_group: Option<BindGroup>,
    /// Key the `simulate_bind_group` was created from.
    simulate_bind_group_key: SimulateBindGroupKey,
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
        asset: Handle<EffectAsset>,
        capacity: u32,
        particle_layout: ParticleLayout,
        layout_flags: LayoutFlags,
        render_device: &RenderDevice,
        label: Option<&str>,
    ) -> Self {
        trace!(
            "EffectBuffer::new(capacity={}, particle_layout={:?}, layout_flags={:?}, item_size={}B)",
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
        let particle_label = label
            .map(|s| format!("hanabi:buffer:effect{s}_particle"))
            .unwrap_or("hanabi:buffer:effect_particle".to_owned());
        let particle_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some(&particle_label),
            size: particle_capacity_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let capacity_bytes: BufferAddress = capacity as u64 * 4;

        let indirect_label = label
            .map(|s| format!("hanabi:buffer:effect{s}_indirect"))
            .unwrap_or("hanabi:buffer:effect_indirect".to_owned());
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

        // TODO - Cache particle_layout and associated bind group layout, instead of
        // creating one bind group layout per buffer using that layout...
        // FIXME - the layout is duplicated in ParticlesInitPipeline and
        // ParticlesUpdatePipeline.
        let particle_group_size = GpuParticleGroup::aligned_size(
            render_device.limits().min_storage_buffer_offset_alignment,
        );
        let entries = [
            // @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(particle_layout.min_binding_size()),
                },
                count: None,
            },
            // @binding(1) var<storage, read_write> indirect_buffer : IndirectBuffer
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(12).unwrap()),
                },
                count: None,
            },
            // @binding(2) var<storage, read> particle_groups : array<ParticleGroup>
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    // Despite no dynamic offset, we do bind a non-zero offset sometimes,
                    // so keep this aligned
                    min_binding_size: Some(particle_group_size),
                },
                count: None,
            },
        ];
        let bgl_label = label
            .map(|s| format!("hanabi:bind_group_layout:effect{s}"))
            .unwrap_or("hanabi:bind_group_layout:effect".to_owned());
        trace!(
            "Creating particle bind group layout '{}' for simulation passes with {} entries.",
            bgl_label,
            entries.len(),
        );
        let particles_buffer_layout_sim =
            render_device.create_bind_group_layout(Some(&bgl_label[..]), &entries);

        // Create the render layout.
        let dispatch_indirect_size = GpuDispatchIndirect::aligned_size(
            render_device.limits().min_storage_buffer_offset_alignment,
        );
        let mut entries = vec![
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
                    min_binding_size: BufferSize::new(std::mem::size_of::<u32>() as u64),
                },
                count: None,
            },
            // @group(1) @binding(2) var<storage, read> dispatch_indirect : DispatchIndirect;
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: Some(dispatch_indirect_size),
                },
                count: None,
            },
        ];
        if layout_flags.contains(LayoutFlags::LOCAL_SPACE_SIMULATION) {
            // @group(1) @binding(3) var<storage, read> spawner : Spawner;
            entries.push(BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: Some(GpuSpawnerParams::min_size()), // TODO - array
                },
                count: None,
            });
        }
        trace!(
            "Creating render layout with {} entries (flags: {:?})",
            entries.len(),
            layout_flags
        );
        let particles_buffer_layout_with_dispatch =
            render_device.create_bind_group_layout("hanabi:buffer_layout_render", &entries);

        Self {
            particle_buffer,
            indirect_buffer,
            particle_layout,
            layout_flags,
            particles_buffer_layout_sim,
            particles_buffer_layout_with_dispatch,
            capacity,
            used_size: 0,
            free_slices: vec![],
            asset,
            simulate_bind_group: None,
            simulate_bind_group_key: SimulateBindGroupKey::INVALID,
        }
    }

    pub fn particle_layout(&self) -> &ParticleLayout {
        &self.particle_layout
    }

    pub fn layout_flags(&self) -> LayoutFlags {
        self.layout_flags
    }

    pub fn particle_layout_bind_group_sim(&self) -> &BindGroupLayout {
        &self.particles_buffer_layout_sim
    }

    pub fn particle_layout_bind_group_with_dispatch(&self) -> &BindGroupLayout {
        &self.particles_buffer_layout_with_dispatch
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

    /// Return a binding of the buffer for a starting range of a given size (in
    /// bytes).
    #[allow(dead_code)]
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

    /// Create the bind group for the init and update passes if needed.
    ///
    /// The `buffer_index` must be the index of the current [`EffectBuffer`]
    /// inside the [`EffectCache`]. The `group_binding` is the binding resource
    /// for the particle groups of this buffer.
    pub fn create_sim_bind_group(
        &mut self,
        buffer_index: u32,
        render_device: &RenderDevice,
        particle_group_buffer: &Buffer,
        particle_group_offset: u64,
        particle_group_size: NonZeroU64,
    ) {
        let key = SimulateBindGroupKey {
            buffer: Some(particle_group_buffer.id()),
            offset: particle_group_offset as u32,
            size: particle_group_size.get() as u32,
        };

        if self.simulate_bind_group.is_some() && self.simulate_bind_group_key == key {
            return;
        }

        let group_binding = BufferBinding {
            buffer: particle_group_buffer,
            offset: particle_group_offset,
            size: Some(particle_group_size),
        };

        let layout = self.particle_layout_bind_group_sim();
        let label = format!("hanabi:bind_group_sim_batch{}", buffer_index);
        let bindings = [
            BindGroupEntry {
                binding: 0,
                resource: self.max_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: self.indirect_max_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(group_binding),
            },
        ];
        trace!(
            "Create simulate bind group '{}' with {} entries",
            label,
            bindings.len()
        );
        let bind_group = render_device.create_bind_group(Some(&label[..]), layout, &bindings);
        self.simulate_bind_group = Some(bind_group);
        self.simulate_bind_group_key = key;
    }

    /// Invalidate any existing simulate bind group.
    ///
    /// Invalidate any existing bind group previously created by
    /// [`create_sim_bind_group()`], generally because a buffer was
    /// re-allocated. This forces a re-creation of the bind group
    /// next time [`create_sim_bind_group()`] is called.
    ///
    /// [`create_sim_bind_group()`]: self::EffectBuffer::create_sim_bind_group
    fn invalidate_sim_bind_group(&mut self) {
        self.simulate_bind_group = None;
        self.simulate_bind_group_key = SimulateBindGroupKey::INVALID;
    }

    /// Return the cached bind group for the init and update passes.
    ///
    /// This is the per-buffer bind group at binding @1 which binds all
    /// per-buffer resources shared by all effect instances batched in a single
    /// buffer. The bind group is created by [`create_sim_bind_group()`], and
    /// cached until a call to [`invalidate_sim_bind_group()`] clears the cached
    /// reference.
    ///
    /// [`create_sim_bind_group()`]: self::EffectBuffer::create_sim_bind_group
    /// [`invalidate_sim_bind_group()`]: self::EffectBuffer::invalidate_sim_bind_group
    pub fn sim_bind_group(&self) -> Option<&BindGroup> {
        self.simulate_bind_group.as_ref()
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

/// Identifier referencing an effect cached in an internal effect cache.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct EffectCacheId {
    pub index: u32,
}

impl EffectCacheId {
    /// An invalid handle, corresponding to nothing.
    pub const INVALID: Self = Self { index: u32::MAX };

    /// Generate a new valid effect cache identifier.
    #[allow(dead_code)]
    pub fn new(index: u32) -> Self {
        Self { index }
    }

    /// Check if the ID is valid.
    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }
}

/// Stores various data, including the buffer index and slice boundaries within
/// the buffer for all groups in a single effect.
#[derive(Debug, Component)]
pub(crate) struct CachedEffect {
    /// The index of the [`EffectBuffer`].
    pub(crate) buffer_index: u32,
    /// The slices within that buffer.
    pub(crate) slices: SlicesRef,
    /// The order in which we evaluate groups.
    pub(crate) group_order: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingEffectGroup {
    pub capacity: u32,
    pub src_group_index_if_trail: Option<u32>,
}

impl From<&AddedEffectGroup> for PendingEffectGroup {
    fn from(value: &AddedEffectGroup) -> Self {
        Self {
            capacity: value.capacity,
            src_group_index_if_trail: value.src_group_index_if_trail,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum RenderGroupDispatchIndices {
    Pending {
        groups: Box<[PendingEffectGroup]>,
    },
    Allocated {
        /// The index of the first render group indirect dispatch buffer.
        ///
        /// There will be one such dispatch buffer for each particle group.
        first_render_group_dispatch_buffer_index: BufferTableId,
        /// Map from a group index to its source and destination rows into the
        /// render group dispatch buffer.
        trail_dispatch_buffer_indices: HashMap<u32, TrailDispatchBufferIndices>,
    },
}

impl Default for RenderGroupDispatchIndices {
    fn default() -> Self {
        Self::Pending {
            groups: Box::new([]),
        }
    }
}

/// The indices in the indirect dispatch buffers for a single effect, as well as
/// that of the metadata buffer.
#[derive(Debug, Clone, Component)]
pub(crate) struct DispatchBufferIndices {
    /// The index of the first update group indirect dispatch buffer.
    ///
    /// There will be one such dispatch buffer for each particle group.
    pub(crate) first_update_group_dispatch_buffer_index: BufferTableId,
    /// The index of the render indirect metadata buffer.
    pub(crate) render_effect_metadata_buffer_table_id: BufferTableId,
    /// Render group dispatch indirect indices for all groups of the effect.
    pub(crate) render_group_dispatch_indices: RenderGroupDispatchIndices,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TrailDispatchBufferIndices {
    pub(crate) dest: BufferTableId,
    pub(crate) src: BufferTableId,
}

impl Default for DispatchBufferIndices {
    // For testing purposes only.
    fn default() -> Self {
        DispatchBufferIndices {
            first_update_group_dispatch_buffer_index: BufferTableId::INVALID,
            render_effect_metadata_buffer_table_id: BufferTableId::INVALID,
            render_group_dispatch_indices: default(),
        }
    }
}

/// Cache for effect instances sharing common GPU data structures.
#[derive(Resource)]
pub struct EffectCache {
    /// Render device the GPU resources (buffers) are allocated from.
    device: RenderDevice,
    /// Collection of effect buffers managed by this cache. Some buffers might
    /// be `None` if the entry is not used. Since the buffers are referenced
    /// by index, we cannot move them once they're allocated.
    buffers: Vec<Option<EffectBuffer>>,
}

impl EffectCache {
    pub fn new(device: RenderDevice) -> Self {
        Self {
            device,
            buffers: vec![],
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub fn buffers(&self) -> &[Option<EffectBuffer>] {
        &self.buffers
    }

    #[allow(dead_code)]
    #[inline]
    pub fn buffers_mut(&mut self) -> &mut [Option<EffectBuffer>] {
        &mut self.buffers
    }

    #[allow(dead_code)]
    #[inline]
    pub fn get_buffer(&self, buffer_index: u32) -> Option<&EffectBuffer> {
        self.buffers.get(buffer_index as usize)?.as_ref()
    }

    #[allow(dead_code)]
    #[inline]
    pub fn get_buffer_mut(&mut self, buffer_index: u32) -> Option<&mut EffectBuffer> {
        self.buffers.get_mut(buffer_index as usize)?.as_mut()
    }

    /// Invalidate all the "simulation" bind groups for all buffers.
    ///
    /// This iterates over all valid buffers and calls
    /// [`EffectBuffer::invalidate_sim_bind_group()`] on each one.
    pub fn invalidate_sim_bind_groups(&mut self) {
        for buffer in self.buffers.iter_mut().flatten() {
            buffer.invalidate_sim_bind_group();
        }
    }

    pub fn insert(
        &mut self,
        asset: Handle<EffectAsset>,
        capacities: Vec<u32>,
        particle_layout: &ParticleLayout,
        layout_flags: LayoutFlags,
        group_order: Vec<u32>,
    ) -> CachedEffect {
        let total_capacity = capacities.iter().cloned().sum();
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
                        .allocate_slice(total_capacity, particle_layout)
                        .map(|slice| (buffer_index, slice))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                // Cannot find any suitable buffer; allocate a new one
                let buffer_index = self.buffers.iter().position(|buf| buf.is_none()).unwrap_or(self.buffers.len());
                let byte_size = total_capacity.checked_mul(particle_layout.min_binding_size().get() as u32).unwrap_or_else(|| panic!(
                    "Effect size overflow: capacities={:?} particle_layout={:?} item_size={}",
                    capacities, particle_layout, particle_layout.min_binding_size().get()
                ));
                trace!(
                    "Creating new effect buffer #{} for effect {:?} (capacities={:?}, particle_layout={:?} item_size={}, byte_size={})",
                    buffer_index,
                    asset,
                    capacities,
                    particle_layout,
                    particle_layout.min_binding_size().get(),
                    byte_size
                );
                let mut buffer = EffectBuffer::new(
                    asset,
                    total_capacity,
                    particle_layout.clone(),
                    layout_flags,
                    &self.device,
                    Some(&format!("{buffer_index}")),
                );
                let slice_ref = buffer.allocate_slice(total_capacity, particle_layout).unwrap();
                if buffer_index >= self.buffers.len() {
                    self.buffers.push(Some(buffer));
                } else {
                    debug_assert!(self.buffers[buffer_index].is_none());
                    self.buffers[buffer_index] = Some(buffer);
                }
                (buffer_index, slice_ref)
            });

        let mut ranges = vec![slice.range.start];
        let group_count = capacities.len();
        for capacity in capacities {
            let start_index = ranges.last().unwrap();
            ranges.push(start_index + capacity);
        }
        debug_assert_eq!(ranges.len(), group_count + 1);

        let slices = SlicesRef {
            ranges,
            particle_layout: slice.particle_layout,
        };

        trace!(
            "Insert effect buffer_index={} slice={}B particle_layout={:?}",
            buffer_index,
            slices.particle_layout.min_binding_size().get(),
            slices.particle_layout,
        );
        CachedEffect {
            buffer_index: buffer_index as u32,
            slices,
            group_order,
        }
    }

    /// Get the init bind group for a cached effect.
    pub fn init_bind_group(&self, buffer_index: u32) -> Option<&BindGroup> {
        self.buffers[buffer_index as usize]
            .as_ref()
            .and_then(|eb| eb.sim_bind_group())
    }

    /// Get the update bind group for a cached effect.
    #[inline]
    pub fn update_bind_group(&self, buffer_index: u32) -> Option<&BindGroup> {
        self.init_bind_group(buffer_index)
    }

    pub fn create_sim_bind_group(
        &mut self,
        buffer_index: u32,
        render_device: &RenderDevice,
        particle_group_buffer: &Buffer,
        particle_group_offset: u64,
        particle_group_size: NonZeroU64,
    ) -> Result<(), ()> {
        // Create the bind group
        let effect_buffer: &mut Option<EffectBuffer> =
            self.buffers.get_mut(buffer_index as usize).ok_or(())?;
        let effect_buffer = effect_buffer.as_mut().ok_or(())?;
        effect_buffer.create_sim_bind_group(
            buffer_index,
            render_device,
            particle_group_buffer,
            particle_group_offset,
            particle_group_size,
        );
        Ok(())
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

        // Reconstruct the original slice
        let slice = SliceRef {
            range: cached_effect.slices.ranges[0]..*cached_effect.slices.ranges.last().unwrap(),
            // FIXME: clone() needed to return CachedEffectIndices, but really we don't care about
            // returning the ParticleLayout, so should split...
            particle_layout: cached_effect.slices.particle_layout.clone(),
        };

        // Free the slice inside the resolved buffer
        if buffer.free_slice(slice) == BufferState::Free {
            *maybe_buffer = None;
            return Ok(BufferState::Free);
        }

        Ok(BufferState::Used)
    }
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
        let slice1 = EffectSlices {
            slices: vec![0, 32],
            buffer_index: 1,
            particle_layout: particle_layout.clone(),
        };
        let slice2 = EffectSlices {
            slices: vec![32, 64],
            buffer_index: 1,
            particle_layout: particle_layout.clone(),
        };
        assert!(slice1 < slice2);
        assert!(slice1 <= slice2);
        assert!(slice2 > slice1);
        assert!(slice2 >= slice1);

        let slice3 = EffectSlices {
            slices: vec![0, 32],
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
            asset,
            capacity,
            l64.clone(),
            LayoutFlags::NONE,
            &render_device,
            Some("my_buffer"),
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
            asset,
            capacity,
            l64.clone(),
            LayoutFlags::NONE,
            &render_device,
            Some("my_buffer"),
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
        let capacities = vec![capacity];
        let group_order = vec![0];
        let item_size = l32.size();

        // Insert an effect
        let effect1 = effect_cache.insert(
            asset.clone(),
            capacities.clone(),
            &l32,
            LayoutFlags::NONE,
            group_order.clone(),
        );
        //assert!(effect1.is_valid());
        let slice1 = &effect1.slices;
        assert_eq!(slice1.group_count(), 1);
        assert_eq!(slice1.group_capacity(0), capacity);
        assert_eq!(slice1.total_capacity(), capacity);
        assert_eq!(
            slice1.particle_layout.min_binding_size().get() as u32,
            item_size
        );
        assert_eq!(slice1.ranges, vec![0, capacity]);
        assert_eq!(effect_cache.buffers().len(), 1);

        // Insert a second copy of the same effect
        let effect2 = effect_cache.insert(
            asset.clone(),
            capacities.clone(),
            &l32,
            LayoutFlags::NONE,
            group_order.clone(),
        );
        //assert!(effect2.is_valid());
        let slice2 = &effect2.slices;
        assert_eq!(slice2.group_count(), 1);
        assert_eq!(slice2.group_capacity(0), capacity);
        assert_eq!(slice2.total_capacity(), capacity);
        assert_eq!(
            slice2.particle_layout.min_binding_size().get() as u32,
            item_size
        );
        assert_eq!(slice2.ranges, vec![0, capacity]);
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
        let effect3 = effect_cache.insert(asset, capacities, &l32, LayoutFlags::NONE, group_order);
        //assert!(effect3.is_valid());
        let slice3 = &effect3.slices;
        assert_eq!(slice3.group_count(), 1);
        assert_eq!(slice3.group_capacity(0), capacity);
        assert_eq!(slice3.total_capacity(), capacity);
        assert_eq!(
            slice3.particle_layout.min_binding_size().get() as u32,
            item_size
        );
        assert_eq!(slice3.ranges, vec![0, capacity]);
        // Note: currently batching is disabled, so each instance has its own buffer.
        assert_eq!(effect_cache.buffers().len(), 2);
        {
            let buffers = effect_cache.buffers();
            assert!(buffers[0].is_some()); // id3
            assert!(buffers[1].is_some()); // id2
        }
    }
}
