use std::{
    cmp::Ordering,
    num::{NonZeroU32, NonZeroU64},
    ops::Range,
};

use bevy::{
    asset::Handle,
    ecs::{component::Component, resource::Resource},
    log::{trace, warn},
    platform::collections::HashMap,
    render::{mesh::allocator::MeshBufferSlice, render_resource::*, renderer::RenderDevice},
    utils::default,
};
use bytemuck::cast_slice_mut;

use super::{buffer_table::BufferTableId, BufferBindingSource};
use crate::{
    asset::EffectAsset,
    render::{
        calc_hash, event::GpuChildInfo, GpuDrawIndexedIndirectArgs, GpuDrawIndirectArgs,
        GpuEffectMetadata, GpuSpawnerParams, StorageType as _, INDIRECT_INDEX_SIZE,
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
    /// ID of the particle slab in the [`EffectCache`].
    pub slab_id: SlabId,
    /// Particle layout of the effect.
    pub particle_layout: ParticleLayout,
}

impl Ord for EffectSlice {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.slab_id.cmp(&other.slab_id) {
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

/// A reference to a slice allocated inside an [`ParticleSlab`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SlabSliceRef {
    /// Range into a [`ParticleSlab`], in item count.
    range: Range<u32>,
    /// Particle layout for the effect stored in that slice.
    pub(crate) particle_layout: ParticleLayout,
}

impl SlabSliceRef {
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

/// State of a [`ParticleSlab`] after an insertion or removal operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlabState {
    /// The slab is in use, with allocated resources.
    Used,
    /// Like `Used`, but the slab was resized, so any bind group is
    /// nonetheless invalid.
    Resized,
    /// The slab is free (its resources were deallocated).
    Free,
}

/// ID of a [`ParticleSlab`] inside an [`EffectCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SlabId(u32);

impl SlabId {
    /// An invalid value, often used as placeholder.
    pub const INVALID: SlabId = SlabId(u32::MAX);

    /// Create a new slab ID from its underlying index.
    pub const fn new(index: u32) -> Self {
        assert!(index != u32::MAX);
        Self(index)
    }

    /// Check if the current ID is valid, that is, is different from
    /// [`INVALID`].
    ///
    /// [`INVALID`]: Self::INVALID
    #[inline]
    #[allow(dead_code)]
    pub const fn is_valid(&self) -> bool {
        self.0 != Self::INVALID.0
    }

    /// Get the raw underlying index.
    ///
    /// This is mostly used for debugging / logging.
    #[inline]
    pub const fn index(&self) -> u32 {
        self.0
    }
}

impl Default for SlabId {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Storage for the per-particle data of effects sharing compatible layouts.
///
/// Currently only accepts a single unique particle layout, fixed at creation.
/// If an effect has a different particle layout, it needs to be stored in a
/// different slab.
///
/// Also currently only accepts instances of a unique effect asset, although
/// this restriction is purely for convenience and may be relaxed in the future
/// to improve batching.
#[derive(Debug)]
pub struct ParticleSlab {
    /// GPU buffer storing all particles for the entire slab of effects.
    ///
    /// Each particle is a collection of attributes arranged according to
    /// [`Self::particle_layout`]. The buffer contains storage for exactly
    /// [`Self::capacity`] particles.
    particle_buffer: Buffer,
    /// GPU buffer storing the indirection indices for the entire slab of
    /// effects.
    ///
    /// Each indirection item contains 3 values:
    /// - the ping-pong alive particles and render indirect indices at offsets 0
    ///   and 1
    /// - the dead particle indices at offset 2
    ///
    /// The buffer contains storage for exactly [`Self::capacity`] items.
    indirect_index_buffer: Buffer,
    /// Layout of particles.
    particle_layout: ParticleLayout,
    /// Total slab capacity, in number of particles.
    capacity: u32,
    /// Used slab size, in number of particles, either from allocated slices
    /// or from slices in the free list.
    used_size: u32,
    /// Array of free slices for new allocations, sorted in increasing order
    /// inside the slab buffers.
    free_slices: Vec<Range<u32>>,

    /// Handle of all effects common in this slab. TODO - replace with
    /// compatible layout.
    asset: Handle<EffectAsset>,
    /// Layout of the particle@1 bind group for the render pass.
    // TODO - move; this only depends on the particle and spawner layouts, can be shared across
    // slabs
    render_particles_buffer_layout: BindGroupLayout,
    /// Bind group particle@1 of the simulation passes (init and udpate).
    sim_bind_group: Option<BindGroup>,
    /// Key the `sim_bind_group` was created from.
    sim_bind_group_key: SimBindGroupKey,
}

impl ParticleSlab {
    /// Minimum buffer capacity to allocate, in number of particles.
    pub const MIN_CAPACITY: u32 = 65536; // at least 64k particles

    /// Create a new slab and the GPU resources to back it up.
    ///
    /// The slab cannot contain less than [`MIN_CAPACITY`] particles. If the
    /// input `capacity` is smaller, it's rounded up to [`MIN_CAPACITY`].
    ///
    /// # Panics
    ///
    /// This panics if the `capacity` is zero.
    ///
    /// [`MIN_CAPACITY`]: Self::MIN_CAPACITY
    pub fn new(
        slab_id: SlabId,
        asset: Handle<EffectAsset>,
        capacity: u32,
        particle_layout: ParticleLayout,
        render_device: &RenderDevice,
    ) -> Self {
        trace!(
            "ParticleSlab::new(slab_id={}, capacity={}, particle_layout={:?}, item_size={}B)",
            slab_id.0,
            capacity,
            particle_layout,
            particle_layout.min_binding_size().get(),
        );

        // Calculate the clamped capacity of the group, in number of particles.
        let capacity = capacity.max(Self::MIN_CAPACITY);
        assert!(
            capacity > 0,
            "Attempted to create a zero-sized effect buffer."
        );

        // Allocate the particle buffer itself, containing the attributes of each
        // particle.
        #[cfg(debug_assertions)]
        let mapped_at_creation = true;
        #[cfg(not(debug_assertions))]
        let mapped_at_creation = false;
        let particle_capacity_bytes: BufferAddress =
            capacity as u64 * particle_layout.min_binding_size().get();
        let particle_label = format!("hanabi:buffer:slab{}:particle", slab_id.0);
        let particle_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some(&particle_label),
            size: particle_capacity_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation,
        });
        // Set content
        #[cfg(debug_assertions)]
        {
            // Scope get_mapped_range_mut() to force a drop before unmap()
            {
                let slice: &mut [u8] = &mut particle_buffer
                    .slice(..particle_capacity_bytes)
                    .get_mapped_range_mut();
                let slice: &mut [u32] = cast_slice_mut(slice);
                slice.fill(0xFFFFFFFF);
            }
            particle_buffer.unmap();
        }

        // Each indirect buffer stores 3 arrays of u32, of length the number of
        // particles.
        let indirect_capacity_bytes: BufferAddress = capacity as u64 * 4 * 3;
        let indirect_label = format!("hanabi:buffer:slab{}:indirect", slab_id.0);
        let indirect_index_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some(&indirect_label),
            size: indirect_capacity_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        // Set content
        {
            // Scope get_mapped_range_mut() to force a drop before unmap()
            {
                let slice: &mut [u8] = &mut indirect_index_buffer
                    .slice(..indirect_capacity_bytes)
                    .get_mapped_range_mut();
                let slice: &mut [u32] = cast_slice_mut(slice);
                for index in 0..capacity {
                    slice[3 * index as usize + 2] = index;
                }
            }
            indirect_index_buffer.unmap();
        }

        // Create the render layout.
        // TODO - move; this only depends on the particle and spawner layouts, can be
        // shared across slabs
        let spawner_params_size = GpuSpawnerParams::aligned_size(
            render_device.limits().min_storage_buffer_offset_alignment,
        );
        let entries = [
            // @group(1) @binding(0) var<storage, read> particle_buffer : ParticleBuffer;
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
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
        let label = format!(
            "hanabi:bind_group_layout:render:particles@1:slab{}",
            slab_id.0
        );
        trace!(
            "Creating particles@1 layout '{}' for render pass with {} entries",
            label,
            entries.len(),
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

    // TODO - move; this only depends on the particle and spawner layouts, can be
    // shared across slabs
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

    /// Return a binding for the entire particle buffer.
    pub fn as_entire_binding_particle(&self) -> BindingResource<'_> {
        let capacity_bytes = self.capacity as u64 * self.particle_layout.min_binding_size().get();
        BindingResource::Buffer(BufferBinding {
            buffer: &self.particle_buffer,
            offset: 0,
            size: Some(NonZeroU64::new(capacity_bytes).unwrap()),
        })
        //self.particle_buffer.as_entire_binding()
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
    pub fn as_entire_binding_indirect(&self) -> BindingResource<'_> {
        let capacity_bytes = self.capacity as u64 * 12;
        BindingResource::Buffer(BufferBinding {
            buffer: &self.indirect_index_buffer,
            offset: 0,
            size: Some(NonZeroU64::new(capacity_bytes).unwrap()),
        })
        //self.indirect_index_buffer.as_entire_binding()
    }

    /// Create the "particle" bind group @1 for the init and update passes if
    /// needed.
    ///
    /// The `slab_id` must be the ID of the current [`ParticleSlab`] inside the
    /// [`EffectCache`].
    pub fn create_particle_sim_bind_group(
        &mut self,
        layout: &BindGroupLayout,
        slab_id: &SlabId,
        render_device: &RenderDevice,
        parent_binding_source: Option<&BufferBindingSource>,
    ) {
        let key: SimBindGroupKey = parent_binding_source.into();
        if self.sim_bind_group.is_some() && self.sim_bind_group_key == key {
            return;
        }

        let label = format!("hanabi:bind_group:sim:particle@1:vfx{}", slab_id.index());
        let entries: &[BindGroupEntry] = if let Some(parent_binding) =
            parent_binding_source.as_ref().map(|bbs| bbs.as_binding())
        {
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.as_entire_binding_particle(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.as_entire_binding_indirect(),
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
                    resource: self.as_entire_binding_particle(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.as_entire_binding_indirect(),
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
    /// [`create_particle_sim_bind_group()`]: self::ParticleSlab::create_particle_sim_bind_group
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
    /// [`create_particle_sim_bind_group()`]: self::ParticleSlab::create_particle_sim_bind_group
    /// [`invalidate_particle_sim_bind_groups()`]: self::ParticleSlab::invalidate_particle_sim_bind_groups
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

    /// Allocate a new entry in the slab to store the particles of a single
    /// effect.
    pub fn allocate(&mut self, capacity: u32) -> Option<SlabSliceRef> {
        trace!("ParticleSlab::allocate(capacity={})", capacity);

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
                        "Cannot allocate slice of size {} in particle slab of capacity {}.",
                        capacity, self.capacity
                    );
                }
                return None;
            }
        };

        trace!("-> allocated slice {:?}", range);
        Some(SlabSliceRef {
            range,
            particle_layout: self.particle_layout.clone(),
        })
    }

    /// Free an allocated slice, and if this was the last allocated slice also
    /// free the buffer.
    pub fn free_slice(&mut self, slice: SlabSliceRef) -> SlabState {
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
                SlabState::Free
            } else {
                // There are still some slices used, the last one of which ends at
                // self.used_size
                SlabState::Used
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
            SlabState::Used
        }
    }

    /// Check whether this slab is compatible with the given asset.
    ///
    /// This allows determining whether an instance of the effect can be stored
    /// inside this slab.
    pub fn is_compatible(
        &self,
        handle: &Handle<EffectAsset>,
        _particle_layout: &ParticleLayout,
    ) -> bool {
        // TODO - replace with check particle layout is compatible to allow tighter
        // packing in less buffers, and update in the less dispatch calls
        *handle == self.asset
    }
}

/// A single cached effect in the [`EffectCache`].
#[derive(Debug, Component)]
pub(crate) struct CachedEffect {
    /// ID of the slab of the slab storing the particles for this effect in the
    /// [`EffectCache`].
    pub slab_id: SlabId,
    /// The allocated effect slice within that slab.
    pub slice: SlabSliceRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnyDrawIndirectArgs {
    /// Args of a non-indexed draw call.
    NonIndexed(GpuDrawIndirectArgs),
    /// Args of an indexed draw call.
    Indexed(GpuDrawIndexedIndirectArgs),
}

impl AnyDrawIndirectArgs {
    /// Create from a vertex buffer slice and an optional index buffer one.
    pub fn from_slices(
        vertex_slice: &MeshBufferSlice<'_>,
        index_slice: Option<&MeshBufferSlice<'_>>,
    ) -> Self {
        if let Some(index_slice) = index_slice {
            Self::Indexed(GpuDrawIndexedIndirectArgs {
                index_count: index_slice.range.len() as u32,
                instance_count: 0,
                first_index: index_slice.range.start,
                base_vertex: vertex_slice.range.start as i32,
                first_instance: 0,
            })
        } else {
            Self::NonIndexed(GpuDrawIndirectArgs {
                vertex_count: vertex_slice.range.len() as u32,
                instance_count: 0,
                first_vertex: vertex_slice.range.start,
                first_instance: 0,
            })
        }
    }

    /// Check if this args are for an indexed draw call.
    #[inline(always)]
    #[allow(dead_code)]
    pub fn is_indexed(&self) -> bool {
        matches!(*self, Self::Indexed(..))
    }

    /// Bit-cast the args to the row entry of the GPU buffer.
    ///
    /// If non-indexed, this returns an indexed struct bit-cast from the actual
    /// non-indexed one, ready for GPU upload.
    pub fn bitcast_to_row_entry(&self) -> GpuDrawIndexedIndirectArgs {
        match self {
            AnyDrawIndirectArgs::NonIndexed(args) => GpuDrawIndexedIndirectArgs {
                index_count: args.vertex_count,
                instance_count: args.instance_count,
                first_index: args.first_vertex,
                base_vertex: args.first_instance as i32,
                first_instance: 0,
            },
            AnyDrawIndirectArgs::Indexed(args) => *args,
        }
    }
}

impl From<GpuDrawIndirectArgs> for AnyDrawIndirectArgs {
    fn from(args: GpuDrawIndirectArgs) -> Self {
        Self::NonIndexed(args)
    }
}

impl From<GpuDrawIndexedIndirectArgs> for AnyDrawIndirectArgs {
    fn from(args: GpuDrawIndexedIndirectArgs) -> Self {
        Self::Indexed(args)
    }
}

/// Index of a row (entry) into the [`BufferTable`] storing the indirect draw
/// args of a single draw call.
#[derive(Debug, Clone, Copy, Component)]
pub(crate) struct CachedDrawIndirectArgs {
    pub row: BufferTableId,
    pub args: AnyDrawIndirectArgs,
}

impl Default for CachedDrawIndirectArgs {
    fn default() -> Self {
        Self {
            row: BufferTableId::INVALID,
            args: AnyDrawIndirectArgs::NonIndexed(default()),
        }
    }
}

impl CachedDrawIndirectArgs {
    /// Check if the index is valid.
    ///
    /// An invalid index doesn't correspond to any allocated args entry. A valid
    /// one may, but note that the args entry in the buffer may have been freed
    /// already with this index. There's no mechanism to detect reuse either.
    #[inline(always)]
    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool {
        self.get_row_raw().is_valid()
    }

    /// Check if this row index refers to an indexed draw args entry.
    #[inline(always)]
    #[allow(dead_code)]
    pub fn is_indexed(&self) -> bool {
        self.args.is_indexed()
    }

    /// Get the raw index value.
    ///
    /// Retrieve the raw index value, losing the discriminant between indexed
    /// and non-indexed draw. This is useful when storing the index value into a
    /// GPU buffer. The rest of the time, prefer retaining the typed enum for
    /// safety.
    ///
    /// # Panics
    ///
    /// Panics if the index is invalid, whether indexed or non-indexed.
    pub fn get_row(&self) -> BufferTableId {
        let idx = self.get_row_raw();
        assert!(idx.is_valid());
        idx
    }

    #[inline(always)]
    fn get_row_raw(&self) -> BufferTableId {
        self.row
    }
}

/// The indices in the indirect dispatch buffers for a single effect, as well as
/// that of the metadata buffer.
#[derive(Debug, Default, Clone, Copy, Component)]
pub(crate) struct DispatchBufferIndices {
    /// The index of the [`GpuDispatchIndirect`] row in the GPU buffer
    /// [`EffectsMeta::update_dispatch_indirect_buffer`].
    ///
    /// [`GpuDispatchIndirect`]: super::GpuDispatchIndirect
    /// [`EffectsMeta::update_dispatch_indirect_buffer`]: super::EffectsMeta::dispatch_indirect_buffer
    pub(crate) update_dispatch_indirect_buffer_row_index: u32,
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
    /// Collection of particle slabs managed by this cache. Some slabs might be
    /// `None` if the entry is not used. Since the slabs are referenced
    /// by index, we cannot move them once they're allocated.
    particle_slabs: Vec<Option<ParticleSlab>>,
    /// Cache of bind group layouts for the particle@1 bind groups of the
    /// simulation passes (init and update). Since all bindings depend only
    /// on buffers managed by the [`EffectCache`], we also cache the layouts
    /// here for convenience.
    particle_bind_group_layout_descs:
        HashMap<ParticleBindGroupLayoutKey, BindGroupLayoutDescriptor>,
    /// Cache of bind group layouts for the metadata@3 bind group of the init
    /// pass.
    metadata_init_bind_group_layout_desc: [Option<BindGroupLayoutDescriptor>; 2],
    /// Cache of bind group layouts for the metadata@3 bind group of the
    /// updatepass.
    metadata_update_bind_group_layout_descs: HashMap<u32, BindGroupLayoutDescriptor>,
}

impl EffectCache {
    /// Create a new empty cache.
    pub fn new(device: RenderDevice) -> Self {
        Self {
            render_device: device,
            particle_slabs: vec![],
            particle_bind_group_layout_descs: default(),
            metadata_init_bind_group_layout_desc: [None, None],
            metadata_update_bind_group_layout_descs: default(),
        }
    }

    /// Get all the particle slab slots. Unallocated slots are `None`. This can
    /// be indexed by the slab index.
    #[allow(dead_code)]
    #[inline]
    pub fn slabs(&self) -> &[Option<ParticleSlab>] {
        &self.particle_slabs
    }

    /// Get all the particle slab slots. Unallocated slots are `None`. This can
    /// be indexed by the slab ID.
    #[allow(dead_code)]
    #[inline]
    pub fn slabs_mut(&mut self) -> &mut [Option<ParticleSlab>] {
        &mut self.particle_slabs
    }

    /// Fetch a specific slab by ID.
    #[inline]
    pub fn get_slab(&self, slab_id: &SlabId) -> Option<&ParticleSlab> {
        self.particle_slabs.get(slab_id.0 as usize)?.as_ref()
    }

    /// Fetch a specific buffer by ID.
    #[allow(dead_code)]
    #[inline]
    pub fn get_slab_mut(&mut self, slab_id: &SlabId) -> Option<&mut ParticleSlab> {
        self.particle_slabs.get_mut(slab_id.0 as usize)?.as_mut()
    }

    /// Invalidate all the particle@1 bind group for all buffers.
    ///
    /// This iterates over all valid buffers and calls
    /// [`ParticleSlab::invalidate_particle_sim_bind_group()`] on each one.
    #[allow(dead_code)] // FIXME - review this...
    pub fn invalidate_particle_sim_bind_groups(&mut self) {
        for buffer in self.particle_slabs.iter_mut().flatten() {
            buffer.invalidate_particle_sim_bind_group();
        }
    }

    /// Insert a new effect instance in the cache.
    pub fn insert(
        &mut self,
        asset: Handle<EffectAsset>,
        capacity: u32,
        particle_layout: &ParticleLayout,
    ) -> CachedEffect {
        trace!("Inserting new effect into cache: capacity={capacity}");
        let (slab_id, slice) = self
            .particle_slabs
            .iter_mut()
            .enumerate()
            .find_map(|(slab_index, maybe_slab)| {
                // Ignore empty (non-allocated) entries as we're trying to fit the new allocation inside an existing slab.
                let Some(slab) = maybe_slab else { return None; };

                // The slab must be compatible with the effect's layout, otherwise ignore it.
                if !slab.is_compatible(&asset, particle_layout) {
                    return None;
                }

                // Try to allocate a slice into the slab
                slab
                    .allocate(capacity)
                    .map(|slice| (SlabId::new(slab_index as u32), slice))
            })
            .unwrap_or_else(|| {
                // Cannot find any suitable slab; allocate a new one
                let index = self.particle_slabs.iter().position(|buf| buf.is_none()).unwrap_or(self.particle_slabs.len());
                let byte_size = capacity.checked_mul(particle_layout.min_binding_size().get() as u32).unwrap_or_else(|| panic!(
                    "Effect size overflow: capacity={:?} particle_layout={:?} item_size={}",
                    capacity, particle_layout, particle_layout.min_binding_size().get()
                ));
                trace!(
                    "Creating new particle slab #{} for effect {:?} (capacity={:?}, particle_layout={:?} item_size={}, byte_size={})",
                    index,
                    asset,
                    capacity,
                    particle_layout,
                    particle_layout.min_binding_size().get(),
                    byte_size
                );
                let slab_id = SlabId::new(index as u32);
                let mut slab = ParticleSlab::new(
                    slab_id,
                    asset,
                    capacity,
                    particle_layout.clone(),
                    &self.render_device,
                );
                let slice_ref = slab.allocate(capacity).unwrap();
                if index >= self.particle_slabs.len() {
                    self.particle_slabs.push(Some(slab));
                } else {
                    debug_assert!(self.particle_slabs[index].is_none());
                    self.particle_slabs[index] = Some(slab);
                }
                (slab_id, slice_ref)
            });

        let slice = SlabSliceRef {
            range: slice.range.clone(),
            particle_layout: slice.particle_layout,
        };

        trace!(
            "Insert effect slab_id={} slice={}B particle_layout={:?}",
            slab_id.0,
            slice.particle_layout.min_binding_size().get(),
            slice.particle_layout,
        );
        CachedEffect { slab_id, slice }
    }

    /// Remove an effect from the cache. If this was the last effect, drop the
    /// underlying buffer and return the index of the dropped buffer.
    pub fn remove(&mut self, cached_effect: &CachedEffect) -> Result<SlabState, ()> {
        // Resolve the buffer by index
        let Some(maybe_buffer) = self
            .particle_slabs
            .get_mut(cached_effect.slab_id.0 as usize)
        else {
            return Err(());
        };
        let Some(buffer) = maybe_buffer.as_mut() else {
            return Err(());
        };

        // Free the slice inside the resolved buffer
        if buffer.free_slice(cached_effect.slice.clone()) == SlabState::Free {
            *maybe_buffer = None;
            return Ok(SlabState::Free);
        }

        Ok(SlabState::Used)
    }

    //
    // Bind group layouts
    //

    /// Ensure a bind group layout exists for the bind group @1 ("particles")
    /// for use with the given min binding sizes.
    pub fn ensure_particle_bind_group_layout_desc(
        &mut self,
        min_binding_size: NonZeroU32,
        parent_min_binding_size: Option<NonZeroU32>,
    ) -> &BindGroupLayoutDescriptor {
        // FIXME - This "ensure" pattern means we never de-allocate entries. This is
        // probably fine, because there's a limited number of realistic combinations,
        // but could cause wastes if e.g. loading widely different scenes.
        let key = ParticleBindGroupLayoutKey {
            min_binding_size,
            parent_min_binding_size,
        };
        self.particle_bind_group_layout_descs
            .entry(key)
            .or_insert_with(|| {
                trace!("Creating new particle sim bind group @1 for min_binding_size={} parent_min_binding_size={:?}", min_binding_size, parent_min_binding_size);
                create_particle_sim_bind_group_layout_desc(
                    min_binding_size,
                    parent_min_binding_size,
                )
            })
    }

    /// Get the bind group layout for the bind group @1 ("particles") for use
    /// with the given min binding sizes.
    pub fn particle_bind_group_layout_desc(
        &self,
        min_binding_size: NonZeroU32,
        parent_min_binding_size: Option<NonZeroU32>,
    ) -> Option<&BindGroupLayoutDescriptor> {
        let key = ParticleBindGroupLayoutKey {
            min_binding_size,
            parent_min_binding_size,
        };
        self.particle_bind_group_layout_descs.get(&key)
    }

    /// Ensure a bind group layout exists for the metadata@3 bind group of
    /// the init pass.
    pub fn ensure_metadata_init_bind_group_layout_desc(&mut self, consume_gpu_spawn_events: bool) {
        let layout =
            &mut self.metadata_init_bind_group_layout_desc[consume_gpu_spawn_events as usize];
        if layout.is_none() {
            *layout = Some(create_metadata_init_bind_group_layout_desc(
                &self.render_device,
                consume_gpu_spawn_events,
            ));
        }
    }

    /// Get the bind group layout for the metadata@3 bind group of the init
    /// pass.
    pub fn metadata_init_bind_group_layout_desc(
        &self,
        consume_gpu_spawn_events: bool,
    ) -> Option<&BindGroupLayoutDescriptor> {
        self.metadata_init_bind_group_layout_desc[consume_gpu_spawn_events as usize].as_ref()
    }

    /// Ensure a bind group layout exists for the metadata@3 bind group of
    /// the update pass.
    pub fn ensure_metadata_update_bind_group_layout_desc(&mut self, num_event_buffers: u32) {
        self.metadata_update_bind_group_layout_descs
            .entry(num_event_buffers)
            .or_insert_with(|| {
                create_metadata_update_bind_group_layout_desc(
                    &self.render_device,
                    num_event_buffers,
                )
            });
    }

    /// Get the bind group layout for the metadata@3 bind group of the
    /// update pass.
    pub fn metadata_update_bind_group_layout_desc(
        &self,
        num_event_buffers: u32,
    ) -> Option<&BindGroupLayoutDescriptor> {
        self.metadata_update_bind_group_layout_descs
            .get(&num_event_buffers)
    }

    //
    // Bind groups
    //

    /// Get the "particle" bind group for the simulation (init and update)
    /// passes a cached effect stored in a given GPU particle buffer.
    pub fn particle_sim_bind_group(&self, slab_id: &SlabId) -> Option<&BindGroup> {
        self.get_slab(slab_id)
            .and_then(|slab| slab.particle_sim_bind_group())
    }

    pub fn create_particle_sim_bind_group(
        &mut self,
        slab_id: &SlabId,
        render_device: &RenderDevice,
        min_binding_size: NonZeroU32,
        parent_min_binding_size: Option<NonZeroU32>,
        parent_binding_source: Option<&BufferBindingSource>,
        pipeline_cache: &PipelineCache,
    ) -> Result<(), ()> {
        // Create the bind group
        let layout = self
            .ensure_particle_bind_group_layout_desc(min_binding_size, parent_min_binding_size)
            .clone();
        let slot = self
            .particle_slabs
            .get_mut(slab_id.index() as usize)
            .ok_or(())?;
        let effect_buffer = slot.as_mut().ok_or(())?;
        effect_buffer.create_particle_sim_bind_group(
            &pipeline_cache.get_bind_group_layout(&layout),
            slab_id,
            render_device,
            parent_binding_source,
        );
        Ok(())
    }
}

/// Create the bind group layout for the "particle" group (@1) of the init and
/// update passes.
fn create_particle_sim_bind_group_layout_desc(
    particle_layout_min_binding_size: NonZeroU32,
    parent_particle_layout_min_binding_size: Option<NonZeroU32>,
) -> BindGroupLayoutDescriptor {
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
    BindGroupLayoutDescriptor::new(label, &entries)
}

/// Create the bind group layout for the metadata@3 bind group of the init pass.
fn create_metadata_init_bind_group_layout_desc(
    render_device: &RenderDevice,
    consume_gpu_spawn_events: bool,
) -> BindGroupLayoutDescriptor {
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
    BindGroupLayoutDescriptor::new(label, &entries)
}

/// Create the bind group layout for the metadata@3 bind group of the update
/// pass.
fn create_metadata_update_bind_group_layout_desc(
    render_device: &RenderDevice,
    num_event_buffers: u32,
) -> BindGroupLayoutDescriptor {
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
    BindGroupLayoutDescriptor::new(label, &entries)
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
            slab_id: SlabId::new(1),
            particle_layout: particle_layout.clone(),
        };
        let slice2 = EffectSlice {
            slice: 32..64,
            slab_id: SlabId::new(1),
            particle_layout: particle_layout.clone(),
        };
        assert!(slice1 < slice2);
        assert!(slice1 <= slice2);
        assert!(slice2 > slice1);
        assert!(slice2 >= slice1);

        let slice3 = EffectSlice {
            slice: 0..32,
            slab_id: SlabId::new(0),
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
            let sr = SlabSliceRef {
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
        let mut buffer = ParticleSlab::new(
            SlabId::new(42),
            asset,
            capacity,
            l64.clone(),
            &render_device,
        );

        assert_eq!(buffer.capacity, capacity.max(ParticleSlab::MIN_CAPACITY));
        assert_eq!(64, buffer.particle_layout.size());
        assert_eq!(64, buffer.particle_layout.min_binding_size().get());
        assert_eq!(0, buffer.used_size);
        assert!(buffer.free_slices.is_empty());

        assert_eq!(None, buffer.allocate(buffer.capacity + 1));

        let mut offset = 0;
        let mut slices = vec![];
        for size in [32, 128, 55, 148, 1, 2048, 42] {
            let slice = buffer.allocate(size);
            assert!(slice.is_some());
            let slice = slice.unwrap();
            assert_eq!(64, slice.particle_layout.size());
            assert_eq!(64, buffer.particle_layout.min_binding_size().get());
            assert_eq!(offset..offset + size, slice.range);
            slices.push(slice);
            offset += size;
        }
        assert_eq!(offset, buffer.used_size);

        assert_eq!(SlabState::Used, buffer.free_slice(slices[2].clone()));
        assert_eq!(1, buffer.free_slices.len());
        let free_slice = &buffer.free_slices[0];
        assert_eq!(160..215, *free_slice);
        assert_eq!(offset, buffer.used_size); // didn't move

        assert_eq!(SlabState::Used, buffer.free_slice(slices[3].clone()));
        assert_eq!(SlabState::Used, buffer.free_slice(slices[4].clone()));
        assert_eq!(SlabState::Used, buffer.free_slice(slices[5].clone()));
        assert_eq!(4, buffer.free_slices.len());
        assert_eq!(offset, buffer.used_size); // didn't move

        // this will collapse all the way to slices[1], the highest allocated
        assert_eq!(SlabState::Used, buffer.free_slice(slices[6].clone()));
        assert_eq!(0, buffer.free_slices.len()); // collapsed
        assert_eq!(160, buffer.used_size); // collapsed

        assert_eq!(SlabState::Used, buffer.free_slice(slices[0].clone()));
        assert_eq!(1, buffer.free_slices.len());
        assert_eq!(160, buffer.used_size); // didn't move

        // collapse all, and free buffer
        assert_eq!(SlabState::Free, buffer.free_slice(slices[1].clone()));
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
        let capacity = 2048; // ParticleSlab::MIN_CAPACITY;
        assert!(capacity >= 2048); // otherwise the logic below breaks
        let mut buffer = ParticleSlab::new(
            SlabId::new(42),
            asset,
            capacity,
            l64.clone(),
            &render_device,
        );

        let slice0 = buffer.allocate(32);
        assert!(slice0.is_some());
        let slice0 = slice0.unwrap();
        assert_eq!(slice0.range, 0..32);
        assert!(buffer.free_slices.is_empty());

        let slice1 = buffer.allocate(1024);
        assert!(slice1.is_some());
        let slice1 = slice1.unwrap();
        assert_eq!(slice1.range, 32..1056);
        assert!(buffer.free_slices.is_empty());

        let state = buffer.free_slice(slice0);
        assert_eq!(state, SlabState::Used);
        assert_eq!(buffer.free_slices.len(), 1);
        assert_eq!(buffer.free_slices[0], 0..32);

        // Try to allocate a slice larger than slice0, such that slice0 cannot be
        // recycled, and instead the new slice has to be appended after all
        // existing ones.
        let slice2 = buffer.allocate(64);
        assert!(slice2.is_some());
        let slice2 = slice2.unwrap();
        assert_eq!(slice2.range.start, slice1.range.end); // after slice1
        assert_eq!(slice2.range, 1056..1120);
        assert_eq!(buffer.free_slices.len(), 1);

        // Now allocate a small slice that fits, to recycle (part of) slice0.
        let slice3 = buffer.allocate(16);
        assert!(slice3.is_some());
        let slice3 = slice3.unwrap();
        assert_eq!(slice3.range, 0..16);
        assert_eq!(buffer.free_slices.len(), 1); // split
        assert_eq!(buffer.free_slices[0], 16..32);

        // Allocate a second small slice that fits exactly the left space, completely
        // recycling
        let slice4 = buffer.allocate(16);
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
        assert_eq!(effect_cache.slabs().len(), 0);

        let asset = Handle::<EffectAsset>::default();
        let capacity = ParticleSlab::MIN_CAPACITY;
        let item_size = l32.size();

        // Insert an effect
        let effect1 = effect_cache.insert(asset.clone(), capacity, &l32);
        //assert!(effect1.is_valid());
        let slice1 = &effect1.slice;
        assert_eq!(slice1.len(), capacity);
        assert_eq!(
            slice1.particle_layout.min_binding_size().get() as u32,
            item_size
        );
        assert_eq!(slice1.range, 0..capacity);
        assert_eq!(effect_cache.slabs().len(), 1);

        // Insert a second copy of the same effect
        let effect2 = effect_cache.insert(asset.clone(), capacity, &l32);
        //assert!(effect2.is_valid());
        let slice2 = &effect2.slice;
        assert_eq!(slice2.len(), capacity);
        assert_eq!(
            slice2.particle_layout.min_binding_size().get() as u32,
            item_size
        );
        assert_eq!(slice2.range, 0..capacity);
        assert_eq!(effect_cache.slabs().len(), 2);

        // Remove the first effect instance
        let buffer_state = effect_cache.remove(&effect1).unwrap();
        // Note: currently batching is disabled, so each instance has its own buffer,
        // which becomes unused once the instance is destroyed.
        assert_eq!(buffer_state, SlabState::Free);
        assert_eq!(effect_cache.slabs().len(), 2);
        {
            let slabs = effect_cache.slabs();
            assert!(slabs[0].is_none());
            assert!(slabs[1].is_some()); // id2
        }

        // Regression #60
        let effect3 = effect_cache.insert(asset, capacity, &l32);
        //assert!(effect3.is_valid());
        let slice3 = &effect3.slice;
        assert_eq!(slice3.len(), capacity);
        assert_eq!(
            slice3.particle_layout.min_binding_size().get() as u32,
            item_size
        );
        assert_eq!(slice3.range, 0..capacity);
        // Note: currently batching is disabled, so each instance has its own buffer.
        assert_eq!(effect_cache.slabs().len(), 2);
        {
            let slabs = effect_cache.slabs();
            assert!(slabs[0].is_some()); // id3
            assert!(slabs[1].is_some()); // id2
        }
    }
}
