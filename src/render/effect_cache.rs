use std::{
    cmp::Ordering,
    hash::Hash,
    num::NonZeroU64,
    ops::Range,
    sync::atomic::{AtomicU64, Ordering as AtomicOrdering},
};

use bevy::{
    asset::Handle,
    ecs::system::Resource,
    log::{trace, warn},
    prelude::{Entity, ResMut},
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
    },
    utils::{default, HashMap},
};
use bytemuck::cast_slice_mut;

use super::{
    aligned_buffer_vec::HybridAlignedBufferVec, buffer_table::BufferTableId, AddedEffectGroup,
    EffectBindGroups,
};
use crate::{
    asset::EffectAsset,
    render::{
        calc_hash, GpuChildInfo, GpuDispatchIndirect, GpuInitDispatchIndirect, GpuParticleGroup,
        GpuSpawnerParams, LayoutFlags, StorageType as _,
    },
    ParticleLayout, PropertyLayout,
};

/// Describes all particle groups' slices of particles in the particle buffer
/// for a single effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectSlices {
    /// Slices into the underlying [`BufferVec`] of the group.
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
pub struct SlicesRef {
    pub ranges: Vec<u32>,
    particle_layout: ParticleLayout,
    pub dispatch_buffer_indices: DispatchBufferIndices,
}

/// A reference to a slice allocated inside an [`EffectBuffer`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SliceRef {
    /// Range into an [`EffectBuffer`], in item count.
    range: Range<u32>,
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
    indirect_buffer: Buffer,
    /// GPU buffer holding the properties of the effect(s), if any. This is
    /// always `None` if the property layout is empty.
    properties_buffer: Option<Buffer>,
    /// Layout of particles.
    particle_layout: ParticleLayout,
    /// Layout of properties of the effect(s), if using properties.
    property_layout: PropertyLayout,
    /// Flags
    layout_flags: LayoutFlags,
    /// Bind group layout for the init pass.
    particles_buffer_layout_init: BindGroupLayout,
    /// Bind group layout for the update pass, once created. Unlike the init
    /// pass, this one is created lazily because it depends on the number of
    /// child effects for which to write GPU spawn events, which is known later
    /// and can vary at runtime after this effect is allocated.
    particles_buffer_layout_update: Option<(u32, BindGroupLayout)>,
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
    /// Bind group for the per-buffer data (group @1) of the init pass.
    init_bind_group: Option<BindGroup>,
    /// Bind group for the per-buffer data (group @1) of the update pass.
    // update_bind_group: Option<BindGroup>,
    update_bind_groups: HashMap<Box<[EffectCacheId]>, BindGroup>,
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
    pub const MIN_CAPACITY: u32 = 1; // 65536; // at least 64k particles

    /// Helper to create the bind group layout for an effect.
    pub fn make_init_layout(
        render_device: &RenderDevice,
        particle_layout_min_binding_size: NonZeroU64,
        property_layout_min_binding_size: Option<NonZeroU64>,
        parent_particle_layout_min_binding_size: Option<NonZeroU64>,
        has_event_buffer: bool,
    ) -> BindGroupLayout {
        let mut entries = Vec::with_capacity(5);

        // @group(1) @binding(0) var<storage, read_write> particle_buffer :
        // ParticleBuffer
        entries.push(BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: Some(particle_layout_min_binding_size),
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
                min_binding_size: BufferSize::new(12),
            },
            count: None,
        });

        // @group(1) @binding(2) var<storage, read> particle_groups :
        // array<ParticleGroup>
        // FIXME - I don't think we need to align the min_binding_size itself...
        let particle_group_size = GpuParticleGroup::aligned_size(
            render_device.limits().min_storage_buffer_offset_alignment,
        );
        entries.push(BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(particle_group_size),
            },
            count: None,
        });

        // @group(1) @binding(3) var<storage, read> properties : Properties
        if let Some(min_binding_size) = property_layout_min_binding_size {
            entries.push(BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, // TODO
                    min_binding_size: Some(min_binding_size),
                },
                count: None,
            });
        }

        if has_event_buffer {
            // @group(1) @binding(4) var<storage, read> child_info : array<ChildInfo>
            entries.push(BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(8), // sizeof(ChildInfo)
                },
                count: None,
            });

            // @group(1) @binding(5) var<storage, read_write> event_buffer : EventBuffer
            entries.push(BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(4), // sizeof(SpawnEvent)
                },
                count: None,
            });
        }

        // @group(1) @binding(6) var<storage, read> parent_particle_buffer :
        // ParentParticleBuffer;
        if let Some(min_binding_size) = parent_particle_layout_min_binding_size {
            entries.push(BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(min_binding_size),
                },
                count: None,
            });
        }

        let hash = calc_hash(&entries);
        let label = format!("hanabi:buffer_layout:init_particles_{:08X}", hash);
        trace!(
            "Creating particle bind group layout '{}' for init passes with {} entries. (event_buffer:{}, properties:{}, parent_buffer:{})",
            label,
            entries.len(),
            has_event_buffer,
            property_layout_min_binding_size.is_some(),
            parent_particle_layout_min_binding_size.is_some(),
        );
        // FIXME - This duplicates the layout created in EffectBuffer::new()!!!
        // Likely this one should go away, because we can't cache from inside
        // specialize() (non-mut access)
        render_device.create_bind_group_layout(&label[..], &entries)
    }

    /// Helper to create the bind group layout for an effect.
    pub fn make_update_layout(
        render_device: &RenderDevice,
        particle_layout_min_binding_size: NonZeroU64,
        property_layout_min_binding_size: Option<NonZeroU64>,
        num_event_buffers: u32,
    ) -> BindGroupLayout {
        let mut entries = Vec::with_capacity(4 + 2 * num_event_buffers as usize);

        // @group(1) @binding(0) var<storage, read_write> particle_buffer :
        // ParticleBuffer
        entries.push(BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: Some(particle_layout_min_binding_size),
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
                min_binding_size: BufferSize::new(12),
            },
            count: None,
        });

        // @group(1) @binding(2) var<storage, read> particle_groups :
        // array<ParticleGroup>
        // FIXME - I don't think we need to align the min_binding_size itself...
        let particle_group_size = GpuParticleGroup::aligned_size(
            render_device.limits().min_storage_buffer_offset_alignment,
        );
        entries.push(BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(particle_group_size),
            },
            count: None,
        });

        // @group(1) @binding(3) var<storage, read> properties : Properties
        if let Some(min_binding_size) = property_layout_min_binding_size {
            entries.push(BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, // TODO
                    min_binding_size: Some(min_binding_size),
                },
                count: None,
            });
        }

        if num_event_buffers > 0 {
            let child_infos_size = GpuChildInfo::SHADER_SIZE.get() * num_event_buffers as u64;

            // @group(1) @binding(4) var<storage, read_write> child_infos :
            // array<ChildInfo>; N>
            entries.push(BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(BufferSize::new(child_infos_size).unwrap()), // sizeof(array<ChildInfos; N>)
                },
                count: None,
            });

            // N times: @group(1) @binding(...) var<storage, read_write> event_buffer_N :
            // EventBuffer
            let mut next_binding_index = 5;
            for _ in 0..num_event_buffers {
                entries.push(BindGroupLayoutEntry {
                    binding: next_binding_index,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(4), // sizeof(SpawnEvent)
                    },
                    count: None,
                });
                next_binding_index += 1;
            }
        }

        let hash = calc_hash(&entries);
        let label = format!("hanabi:buffer_layout:update_particles_{:08X}", hash);
        trace!(
            "Creating particle bind group layout '{}' for update passes with {} entries. (num_event_buffers:{}, properties:{})\n{:?}",
            label,
            entries.len(),
            num_event_buffers,
            property_layout_min_binding_size.is_some(),
            entries
        );
        // FIXME - This duplicates the layout created in EffectBuffer::new()!!!
        // Likely this one should go away, because we can't cache from inside
        // specialize() (non-mut access)
        render_device.create_bind_group_layout(&label[..], &entries)
    }

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
        property_layout: PropertyLayout,
        parent_particle_layout: Option<ParticleLayout>,
        layout_flags: LayoutFlags,
        render_device: &RenderDevice,
        label: Option<&str>,
    ) -> Self {
        trace!(
            "EffectBuffer::new(capacity={}, particle_layout={:?}, property_layout={:?}, layout_flags={:?}, item_size={}B, properties_size={}B)",
            capacity,
            particle_layout,
            property_layout,
            layout_flags,
            particle_layout.min_binding_size().get(),
            if property_layout.is_empty() { 0 } else { property_layout.min_binding_size().get() },
        );

        // Calculate the clamped capacity of the group, in number of particles
        let capacity = capacity.max(Self::MIN_CAPACITY);
        debug_assert!(
            capacity > 0,
            "Attempted to create a zero-sized effect buffer."
        );

        // Allocate the particle buffer itself, containing the attributes of each
        // particle
        let particle_capacity_bytes: BufferAddress =
            capacity as u64 * particle_layout.min_binding_size().get();
        let particle_buffer = render_device.create_buffer(&BufferDescriptor {
            label,
            size: particle_capacity_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let capacity_bytes: BufferAddress = capacity as u64 * 4;

        let indirect_label = if let Some(label) = label {
            format!("{label}_indirect")
        } else {
            "hanabi:buffer:effect_indirect".to_owned()
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

        // Allocate the buffer to store effect properties, which are uploaded each time
        // they change from the CPU. This is only ever read from GPU.
        let properties_buffer = if property_layout.is_empty() {
            None
        } else {
            let properties_label = if let Some(label) = label {
                format!("{}_properties", label)
            } else {
                "hanabi:buffer:effect_properties".to_owned()
            };
            let size = property_layout.min_binding_size().get(); // TODO: * num_effects_in_buffer (once batching works again)
            let properties_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some(&properties_label),
                size,
                usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            Some(properties_buffer)
        };

        // TODO - Cache particle_layout and associated bind group layout, instead of
        // creating one bind group layout per buffer using that layout...
        // FIXME - This duplicates the layout created in
        // ParticlesInitPipeline::specialize()!!! Likely this one should be
        // kept, because we can't cache from inside specialize() (non-mut access)
        let property_layout_min_binding_size = if !property_layout.is_empty() {
            Some(property_layout.min_binding_size())
        } else {
            None
        };
        // init
        let uses_event_buffer = layout_flags.intersects(LayoutFlags::CONSUME_GPU_SPAWN_EVENTS);
        let particles_buffer_layout_init = EffectBuffer::make_init_layout(
            render_device,
            particle_layout.min_binding_size(),
            property_layout_min_binding_size,
            parent_particle_layout
                .as_ref()
                .map(|layout| layout.min_binding_size()),
            uses_event_buffer,
        );

        // Create the render layout.
        let dispatch_indirect_size = GpuDispatchIndirect::aligned_size(
            render_device.limits().min_storage_buffer_offset_alignment,
        );
        let mut entries = vec![
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
            properties_buffer,
            particle_layout,
            property_layout,
            layout_flags,
            particles_buffer_layout_init,
            particles_buffer_layout_update: None,
            particles_buffer_layout_with_dispatch,
            capacity,
            used_size: 0,
            free_slices: vec![],
            asset,
            init_bind_group: None,
            update_bind_groups: default(),
        }
    }

    pub fn properties_buffer(&self) -> Option<&Buffer> {
        self.properties_buffer.as_ref()
    }

    pub fn particle_layout(&self) -> &ParticleLayout {
        &self.particle_layout
    }

    pub fn property_layout(&self) -> &PropertyLayout {
        &self.property_layout
    }

    pub fn layout_flags(&self) -> LayoutFlags {
        self.layout_flags
    }

    pub fn particle_layout_bind_group_init(&self) -> &BindGroupLayout {
        &self.particles_buffer_layout_init
    }

    pub fn ensure_particle_update_bind_group_layout(
        &mut self,
        num_event_buffers: u32,
        render_device: &RenderDevice,
        particle_layout_min_binding_size: NonZeroU64,
        property_layout_min_binding_size: Option<NonZeroU64>,
    ) {
        trace!(
            "Ensuring bind group layout for update pass with {} event buffers...",
            num_event_buffers
        );

        // If there's already a layout for that particular number of event buffers,
        // we're done
        if let Some((num, _)) = self.particles_buffer_layout_update.as_ref() {
            if *num == num_event_buffers {
                trace!(
                    "-> Bind group layout already exists for {} event buffers.",
                    num_event_buffers
                );
                return;
            }
        }

        // Otherwise create a new layout, overwritting any existing one
        let particles_buffer_layout_update = EffectBuffer::make_update_layout(
            render_device,
            particle_layout_min_binding_size,
            property_layout_min_binding_size,
            num_event_buffers,
        );
        self.particles_buffer_layout_update =
            Some((num_event_buffers, particles_buffer_layout_update));
        trace!(
            "-> Created new bind group layout for {} event buffers.",
            num_event_buffers
        );
    }

    /// Get the particle layout bind group of the update pass with the given
    /// number of child event buffers.
    ///
    /// # Returns
    ///
    /// Returns `Some` if and only if a bind group layout exists which was
    /// created for the same number of event buffers as `num_event_buffers`. To
    /// ensure such a layout exists ahead of time, use
    /// [`ensure_particle_update_bind_group_layout()`].
    ///
    /// [`ensure_particle_update_bind_group_layout()`]: crate::EffectBuffer::ensure_particle_update_bind_group_layout
    pub fn particle_layout_bind_group_update(
        &self,
        num_event_buffers: u32,
    ) -> Option<&BindGroupLayout> {
        self.particles_buffer_layout_update
            .as_ref()
            .filter(|(num, _)| *num == num_event_buffers)
            .map(|(_, layout)| layout)
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

    /// Return a binding for the entire properties buffer associated with the
    /// current effect buffer, if any.
    pub fn properties_max_binding(&self) -> Option<BindingResource> {
        self.properties_buffer.as_ref().map(|buffer| {
            let capacity_bytes = self.property_layout.min_binding_size().get();
            BindingResource::Buffer(BufferBinding {
                buffer,
                offset: 0,
                size: Some(NonZeroU64::new(capacity_bytes).unwrap()),
            })
        })
    }

    /// Return the cached bind group for the init pass.
    ///
    /// This is the per-buffer bind group at binding @1 which binds all
    /// per-buffer resources shared by all effect instances batched in a single
    /// buffer.
    pub fn init_bind_group(&self) -> Option<&BindGroup> {
        self.init_bind_group.as_ref()
    }

    pub fn set_init_bind_group(&mut self, init_bind_group: BindGroup) {
        assert!(self.init_bind_group.is_none());
        self.init_bind_group = Some(init_bind_group);
    }

    /// Return the cached bind group for the update pass.
    ///
    /// This is the per-buffer bind group at binding @1 which binds all
    /// per-buffer resources shared by all effect instances batched in a single
    /// buffer. It's keyed by the (possibly empty) list of child effects. Each
    /// combination of child effects produces a different bind group.
    pub fn update_bind_group(&self, child_ids: &[EffectCacheId]) -> Option<&BindGroup> {
        self.update_bind_groups.get(child_ids)
    }

    pub fn set_update_bind_group(
        &mut self,
        child_ids: &[EffectCacheId],
        update_bind_group: BindGroup,
    ) {
        let key = child_ids.into();
        let prev_value = self.update_bind_groups.insert(key, update_bind_group);
        assert!(prev_value.is_none());
    }

    pub fn invalidate_all_bind_groups(&mut self) {
        self.init_bind_group = None;
        self.update_bind_groups.clear();
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

/// Identifier referencing an effect cached in an internal effect cache.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }
}

pub(crate) struct InitIndirect {
    /// Index of the dispatch entry into the [`init_indirect_dispatch_buffer`]
    /// array.
    ///
    /// [`init_indirect_dispatch_buffer`]: self::EffectCache::init_indirect_dispatch_buffer
    pub(crate) dispatch_index: u32,
    pub(crate) event_buffer_ref: EventBufferRef,
}

pub(crate) struct CachedChildren {
    /// Cache entry IDs of the child effects.
    pub effect_cache_ids: Vec<EffectCacheId>,
    /// Indices in bytes into the global [`EffectCache::child_infos_buffer`] of
    /// the child infos for all the child effects of this parent effect. The
    /// child effects are always allocated as a single contiguous block,
    /// which needs to be mapped into a shader binding point.
    pub byte_range: Range<u32>,
}

impl CachedChildren {
    /// Get a binding of the given underlying child info buffer spanning over
    /// the range of this child effect entry.
    pub fn binding<'a>(&self, buffer: &'a Buffer) -> BufferBinding<'a> {
        BufferBinding {
            buffer,
            offset: self.byte_range.start as u64,
            size: Some(
                NonZeroU64::new((self.byte_range.end - self.byte_range.start) as u64).unwrap(),
            ),
        }
    }

    pub fn max_binding<'a>(&self, buffer: &'a Buffer) -> BufferBinding<'a> {
        BufferBinding {
            buffer,
            offset: 0,
            size: None,
        }
    }

    /// Base offset of the first child into the global
    /// [`EffectCache::child_infos_buffer`], in number of element.
    pub fn first_child_index(&self) -> u32 {
        self.byte_range.start / size_of::<GpuChildInfo>() as u32
    }
}

pub(crate) struct CachedChildInfo {
    pub(crate) parent_cache_id: EffectCacheId,
    /// Index of this child effect into its parent's ChildInfo array
    /// ([`EffectChildren::effect_cache_ids`] and its associated GPU
    /// array). This starts at zero for the first child of each effect, and is
    /// only unique per parent, not globally. Only available if this effect is a
    /// child of another effect (i.e. if it has a parent).
    pub(crate) local_child_index: u32,
    /// Global index of this child effect into the shared global
    /// [`EffectCache::child_infos_buffer`] array. This is a unique index across
    /// all effects.
    pub(crate) global_child_index: u32,
    /// Data for the indirect dispatch of the init pass, if using GPU spawn
    /// events.
    pub(crate) init_indirect: InitIndirect,
}

/// A single cached effect (all groups) in the [`EffectCache`].
pub(crate) struct CachedEffect {
    /// Index into the [`EffectCache::buffers`] of the buffer storing the
    /// particles for this effect. This includes all particles of all effect
    /// groups.
    pub(crate) buffer_index: u32,
    /// The group slices within that buffer. All groups for an effect are always
    /// stored contiguously inside the same GPU buffer.
    pub(crate) slices: SlicesRef,
    /// The order in which we evaluate groups.
    pub(crate) group_order: Vec<u32>,
    /// Info about this effect as a child of another effect. Only available if
    /// this effect has a parent effect.
    pub(crate) cached_child_info: Option<CachedChildInfo>,
    /// Child effect(s), if any. Only available if this effect is the parent
    /// effect for one or more child effects.
    pub(crate) children: Option<CachedChildren>,
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
#[derive(Debug, Clone)]
pub(crate) struct DispatchBufferIndices {
    /// The index of the first update group indirect dispatch buffer.
    ///
    /// There will be one such dispatch buffer for each particle group.
    pub(crate) first_update_group_dispatch_buffer_index: BufferTableId,
    /// The index of the render indirect metadata buffer.
    pub(crate) render_effect_metadata_buffer_index: BufferTableId,
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
            render_effect_metadata_buffer_index: BufferTableId::INVALID,
            render_group_dispatch_indices: default(),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EventSlice {
    slice: Range<u32>,
}

/// GPU buffer storing the spawn events emitted by parent effects for their
/// children.
///
/// The event buffer contains for each child effect the number of particles to
/// spawn this frame. That number is incremented by its parent effect when it
/// emits a spawn event, and reset to zero on next frame after the indirect init
/// pass spawned new child particles, and before the new update pass of the
/// parent optionally emits more spawn events. GPU spawn events are never
/// accumulated over frames; if a parent emits too many events and the child
/// cannot spawn that many particles, for example because it reached its
/// capacity, then the extra events are discarded. This is consistent with the
/// CPU behavior of [`Spawner::spawn_count`].
pub struct EventBuffer {
    /// Owner child effect which consumes those events.
    effect_cache_id: EffectCacheId,
    /// GPU buffer storing the spawn events.
    buffer: Buffer,
    /// Buffer capacity, in words (4 bytes).
    capacity: u32,
    /// Allocated (used) buffer size, in words (4 bytes).
    size: u32,
    /// Slices into the GPU buffer where event sub-allocations for each effect
    /// are located. Slices are stored ordered by location in the buffer, for
    /// convenience of allocation.
    slices: Vec<EventSlice>,
}

impl EventBuffer {
    /// Create a new event buffer to store the spawn events of the specified
    /// child effect.
    pub fn new(effect_cache_id: EffectCacheId, buffer: Buffer, capacity: u32) -> Self {
        Self {
            effect_cache_id,
            buffer,
            capacity,
            size: 0,
            slices: vec![],
        }
    }

    /// Get the ID in the [`EffectCache`] of the child effect this event buffer
    /// is associated with.
    pub fn effect_cache_id(&self) -> EffectCacheId {
        self.effect_cache_id
    }

    /// Get a reference to the underlying GPU buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Allocate a new slice for a child effect.
    pub fn allocate(&mut self, size: u32) -> Option<EventSlice> {
        if self.size + size > self.capacity {
            return None;
        }

        if self.slices.is_empty() {
            let slice = EventSlice { slice: 0..size };
            self.slices.push(slice.clone());
            self.size += size;
            return Some(slice);
        }

        let mut start = 0;
        for (idx, es) in self.slices.iter().enumerate() {
            let avail_size = es.slice.start - start;
            if size > avail_size {
                start = es.slice.end;
                continue;
            }

            let slice = EventSlice {
                slice: start..start + size,
            };
            self.slices.insert(idx, slice.clone());
            self.size += size;
            return Some(slice);
        }

        if start + size <= self.capacity {
            let slice = EventSlice {
                slice: start..start + size,
            };
            self.slices.push(slice.clone());
            self.size += size;
            Some(slice)
        } else {
            None
        }
    }

    /// Free the slice of a child effect once that effect is deallocated.
    #[allow(dead_code)] // FIXME
    pub fn free(&mut self, slice: &EventSlice) {
        // Note: could use binary search, but likely not enough elements to be worth it
        if let Some(idx) = self.slices.iter().position(|es| es == slice) {
            self.slices.remove(idx);
            return;
        }
    }
}

pub(crate) struct EventBufferRef {
    /// Index into [`EffectCache::event_buffers`] of the [`EventBuffer`] storing
    /// the referenced events.
    pub(crate) buffer_index: u32,
    /// Slice into the [`EventBuffer`] where the events are allocated.
    pub(crate) slice: Range<u32>,
}

pub(crate) struct EventBufferBinding {
    buffer: Buffer,
    /// Offset in number of u32 elements (4 bytes).
    offset: u32,
    /// Size in number of u32 elements (4 bytes).
    size: u32,
}

impl EventBufferBinding {
    pub fn binding(&self) -> BindingResource {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.buffer,
            offset: self.offset as u64 * 4,
            size: Some(NonZeroU64::new(self.size as u64 * 4).unwrap()),
        })
    }

    pub fn max_binding(&self) -> BindingResource {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: None,
        })
    }
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
    /// Map from an effect cache ID to various buffer indices.
    effects: HashMap<EffectCacheId, CachedEffect>,
    /// Map from the [`Entity`] of the main [`World`] owning the
    /// [`ParticleEffect`] component, to the effect cache entry for that
    /// effect instance.
    effect_from_entity: HashMap<Entity, EffectCacheId>,
    /// Single shared GPU buffer storing all the [`GpuInitDispatchIndirect`]
    /// structs for all the indirect init passes.
    // Note: we abuse BufferVec but never copy anything from CPU
    init_indirect_dispatch_buffer: BufferVec<GpuInitDispatchIndirect>,
    /// Single shared GPU buffer storing all the [`GpuChildInfo`] structs
    /// for all the parent effects.
    child_infos_buffer: HybridAlignedBufferVec,
    /// Bind group layout for the indirect dispatch pass, which clears the GPU
    /// event counts.
    child_infos_bgl: BindGroupLayout,
    /// Bind group for the indirect dispatch pass, which clears the GPU event
    /// counts.
    child_infos_bind_group: Option<BindGroup>,
    /// Buffers for GPU events.
    event_buffers: Vec<Option<EventBuffer>>,
}

impl EffectCache {
    pub fn new(device: RenderDevice) -> Self {
        let mut init_indirect_dispatch_buffer =
            BufferVec::new(BufferUsages::STORAGE | BufferUsages::INDIRECT);
        init_indirect_dispatch_buffer.set_label(Some("hanabi:buffer:init_indirect_dispatch"));

        let child_infos_bgl = device.create_bind_group_layout(
            "hanabi:bind_group_layout:indirect_child_infos",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(GpuChildInfo::min_size()),
                },
                count: None,
            }],
        );

        Self {
            render_device: device,
            buffers: vec![],
            effects: HashMap::default(),
            effect_from_entity: HashMap::default(),
            init_indirect_dispatch_buffer,
            child_infos_buffer: HybridAlignedBufferVec::new(
                BufferUsages::STORAGE,
                Some(NonZeroU64::new(4).unwrap()),
                Some("hanabi:buffer:child_infos".to_string()),
            ),
            child_infos_bgl,
            // Can't create until the buffer is ready
            child_infos_bind_group: None,
            event_buffers: vec![],
        }
    }

    #[allow(dead_code)]
    pub fn buffers(&self) -> &[Option<EffectBuffer>] {
        &self.buffers
    }

    #[allow(dead_code)]
    pub fn buffers_mut(&mut self) -> &mut [Option<EffectBuffer>] {
        &mut self.buffers
    }

    #[allow(dead_code)]
    pub fn get_buffer(&self, index: u32) -> Option<&EffectBuffer> {
        self.buffers
            .get(index as usize)
            .map(|buf| buf.as_ref())
            .flatten()
    }

    /// Ensure the bind group for the init pass exists.
    ///
    /// The `buffer_index` must be the index of the [`EffectBuffer`] inside this
    /// [`EffectCache`]. The `group_binding` is the binding resource for the
    /// particle groups of this buffer. The parent buffer binding is an optional
    /// binding in case the effect has a parent effect.
    pub fn ensure_init_bind_group(
        &mut self,
        effect_cache_id: EffectCacheId,
        buffer_index: u32,
        parent_buffer_index: Option<u32>,
        group_binding: BufferBinding,
        event_buffer_binding: Option<BindingResource>,
    ) -> bool {
        let Some(effect_buffer) = &self.buffers[buffer_index as usize] else {
            return false;
        };
        if effect_buffer.init_bind_group().is_some() {
            return true;
        }

        let parent_buffer_binding = if let Some(parent_buffer_index) = parent_buffer_index {
            let Some(parent_effect_buffer) = self.buffers.get(parent_buffer_index as usize) else {
                return false;
            };
            let Some(binding) = parent_effect_buffer
                .as_ref()
                .map(|buffer| buffer.max_binding())
            else {
                return false;
            };
            Some(binding)
        } else {
            None
        };

        let layout = effect_buffer.particle_layout_bind_group_init();

        let mut bindings = vec![
            BindGroupEntry {
                binding: 0,
                resource: effect_buffer.max_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: effect_buffer.indirect_max_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(group_binding),
            },
        ];
        if let Some(property_binding) = effect_buffer.properties_max_binding() {
            bindings.push(BindGroupEntry {
                binding: 3,
                resource: property_binding,
            });
        }
        // FIXME - this binding depends on whether the particular effect is a child, but
        // the buffer itself should in theory be common to multiple effects (batching)
        if let Some(child_info_binding) = self.get_child_info_max_binding(effect_cache_id) {
            bindings.push(BindGroupEntry {
                binding: 4,
                resource: BindingResource::Buffer(child_info_binding),
            });
        }
        if let Some(event_buffer_binding) = event_buffer_binding {
            bindings.push(BindGroupEntry {
                binding: 5,
                resource: event_buffer_binding,
            });
        }
        if let Some(parent_buffer) = parent_buffer_binding {
            bindings.push(BindGroupEntry {
                binding: 6,
                resource: parent_buffer,
            });
        }

        let label = format!("hanabi:bind_group:init_batch{}", buffer_index);
        trace!(
            "Create init bind group '{}' with {} entries",
            label,
            bindings.len()
        );
        let init_bind_group =
            self.render_device
                .create_bind_group(Some(&label[..]), layout, &bindings);

        // Now that the bind group is created, and the borrows to the various buffers
        // are released, we can mutably borrow the main EffectBuffer to assign the newly
        // created bind group.
        self.buffers[buffer_index as usize]
            .as_mut()
            .unwrap()
            .set_init_bind_group(init_bind_group);

        return true;
    }

    /// Ensure the bind group for the update pass exists.
    ///
    /// The `buffer_index` must be the index of the [`EffectBuffer`] inside this
    /// [`EffectCache`]. The `group_binding` is the binding resource for the
    /// particle groups of this buffer. The child buffer indices is an optional
    /// slice of buffer indices in case the effect has one or more child
    /// effects.
    ///
    /// This lazy bind group creation is necessary because the bind group not
    /// only depends on the current effect, but also on its children. To avoid
    /// complex management when a child changes, we lazily ensure at the last
    /// minute that the bind group is compatible with the current parent/child
    /// hierarchy, and if not re-create it.
    pub fn ensure_update_bind_group(
        &mut self,
        effect_cache_id: EffectCacheId,
        buffer_index: u32,
        group_binding: BufferBinding,
        child_ids: &[EffectCacheId],
    ) -> bool {
        let Some(buffer) = &self.buffers[buffer_index as usize] else {
            return false;
        };
        if buffer.update_bind_group(child_ids).is_some() {
            return true;
        }

        let num_event_buffers = child_ids.len() as u32;
        let Some(layout) = buffer.particle_layout_bind_group_update(num_event_buffers) else {
            warn!(
                "Failed to find particle layout bind group for update pass with {} event buffers.",
                num_event_buffers
            );
            return false;
        };

        let mut bindings = vec![
            BindGroupEntry {
                binding: 0,
                resource: buffer.max_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buffer.indirect_max_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(group_binding),
            },
        ];
        if let Some(property_binding) = buffer.properties_max_binding() {
            bindings.push(BindGroupEntry {
                binding: 3,
                resource: property_binding,
            });
        }
        if let (Some(buffer), Some(_children_effect_entry)) = (
            self.child_infos_buffer.buffer(),
            self.effects
                .get(&effect_cache_id)
                .map(|cei| cei.children.as_ref())
                .flatten(),
        ) {
            bindings.push(BindGroupEntry {
                binding: 4,
                //resource: BindingResource::Buffer(children_effect_entry.binding(buffer)),
                resource: BindingResource::Buffer(BufferBinding {
                    buffer,
                    offset: 0,
                    size: None,
                }),
            });
        }
        // The buffer has one or more child effect(s), and those effects each own an
        // event buffer we need to bind in order to write events to.
        // Note: we need to store those bindings in a second array (Vec) to keep them
        // alive while the first array (`bindings`) references them.
        let mut child_bindings = Vec::with_capacity(child_ids.len());
        let mut next_binding_index = 5;
        for buffer_id in child_ids {
            if let (Some(event_buffer_binding), Some(_event_dispatch_index)) = (
                self.get_event_buffer_binding(*buffer_id),
                self.get_event_dispatch_index(*buffer_id),
            ) {
                child_bindings.push((next_binding_index, event_buffer_binding));
            }

            // Increment always, even if a buffer is not available, as the index binding is
            // mapped 1:1 with children. We will just miss a buffer and not be able to write
            // events for that particular child effect.
            next_binding_index += 1;
        }
        for (binding_index, ebb) in &child_bindings[..] {
            bindings.push(BindGroupEntry {
                binding: *binding_index,
                resource: ebb.binding(),
            });
        }

        let label = format!("hanabi:bind_group:update_batch{}", buffer_index);
        trace!(
            "Create update bind group '{}' with {} entries",
            label,
            bindings.len()
        );
        let update_bind_group =
            self.render_device
                .create_bind_group(Some(&label[..]), layout, &bindings);

        // Now that the bind group is created, and the borrows to the various buffers
        // are released, we can mutably borrow the main EffectBuffer to assign the newly
        // created bind group.
        self.buffers[buffer_index as usize]
            .as_mut()
            .unwrap()
            .set_update_bind_group(child_ids, update_bind_group);

        return true;
    }

    pub fn insert(
        &mut self,
        entity: Entity,
        asset: Handle<EffectAsset>,
        capacities: Vec<u32>,
        particle_layout: &ParticleLayout,
        parent_particle_layout: Option<&ParticleLayout>,
        property_layout: &PropertyLayout,
        layout_flags: LayoutFlags,
        event_capacity: u32,
        dispatch_buffer_indices: DispatchBufferIndices,
        group_order: Vec<u32>,
    ) -> EffectCacheId {
        let total_capacity = capacities.iter().cloned().sum();
        trace!("Inserting new effect into cache: entity={entity:?}, total_capacity={total_capacity} event_capacity={event_capacity}");
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
            .or_else(|| {
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
                    property_layout.clone(),
                    parent_particle_layout.cloned(),
                    layout_flags,
                    &self.render_device,
                    Some(&format!("hanabi:buffer:effect{buffer_index}_particles")),
                );
                let slice_ref = buffer.allocate_slice(total_capacity, particle_layout).unwrap();
                if buffer_index >= self.buffers.len() {
                    self.buffers.push(Some(buffer));
                } else {
                    debug_assert!(self.buffers[buffer_index].is_none());
                    self.buffers[buffer_index] = Some(buffer);
                }
                Some((buffer_index, slice_ref))
            })
            .unwrap();
        let effect_cached_id = EffectCacheId::new();

        // Calculate sub-allocations for each group within the total slice allocated
        // above
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
            dispatch_buffer_indices,
        };

        // Consistency check
        assert_eq!(
            layout_flags.contains(LayoutFlags::CONSUME_GPU_SPAWN_EVENTS),
            event_capacity > 0
        );

        let init_indirect_dispatch_index =
            if layout_flags.contains(LayoutFlags::CONSUME_GPU_SPAWN_EVENTS) {
                // The value pushed is a dummy; see allocate_frame_buffers().
                let init_indirect_dispatch_index = self
                    .init_indirect_dispatch_buffer
                    .push(GpuInitDispatchIndirect::default());
                Some(init_indirect_dispatch_index as u32)
            } else {
                None
            };

        let event_buffer_ref = if event_capacity > 0 {
            Some(self.insert_event_buffer(effect_cached_id, event_capacity))
        } else {
            None
        };

        trace!(
            "Insert effect effect_cached_id={:?} buffer_index={} slice={}B particle_layout={:?} indirect_init_index={:?}",
            effect_cached_id,
            buffer_index,
            slices.particle_layout.min_binding_size().get(),
            slices.particle_layout,
            init_indirect_dispatch_index,
        );
        self.effects.insert(
            effect_cached_id,
            CachedEffect {
                buffer_index: buffer_index as u32,
                slices,
                group_order,
                cached_child_info: event_buffer_ref.map(|event_buffer_ref| CachedChildInfo {
                    init_indirect: InitIndirect {
                        dispatch_index: init_indirect_dispatch_index.unwrap(),
                        event_buffer_ref,
                    },
                    // Resolved later in resolve_parents()
                    parent_cache_id: EffectCacheId::INVALID,
                    local_child_index: u32::MAX,
                    global_child_index: u32::MAX,
                }),
                // At this point we didn't resolve parents, so we don't know how many children this
                // effect has. Leave this empty for now, until resolve_parents() fix it up if
                // needed.
                children: None,
            },
        );
        self.effect_from_entity.insert(entity, effect_cached_id);
        effect_cached_id
    }

    /// Re-/allocate any buffer for the current frame.
    pub fn prepare_buffers(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        effect_bind_groups: &mut ResMut<EffectBindGroups>,
    ) {
        // Note: we abuse AlignedBufferVec for its ability to manage the GPU buffer, but
        // we don't use its CPU side capabilities. So we only need the GPU buffer to be
        // correctly allocated, using reserve(). No data is copied from CPU.
        self.init_indirect_dispatch_buffer
            .reserve(self.init_indirect_dispatch_buffer.len(), render_device);

        let old_buffer = self.child_infos_buffer.buffer().map(|b| b.id());
        self.child_infos_buffer
            .write_buffer(render_device, render_queue);
        let new_buffer = self.child_infos_buffer.buffer().map(|b| b.id());
        if old_buffer != new_buffer {
            // If the child infos buffer changed, all init bind groups of children and all
            // update bind groups of parents are invalid because they all use
            // that globally shared buffer.
            // FIXME - it's quite hard to tell for now; just invalidate everything
            for buffer in &mut self.buffers {
                if let Some(buffer) = buffer {
                    buffer.invalidate_all_bind_groups();
                }
            }

            effect_bind_groups.init_fill_dispatch.clear();
        }
    }

    pub fn get_slices(&self, id: EffectCacheId) -> EffectSlices {
        self.effects
            .get(&id)
            .map(|indices| EffectSlices {
                slices: indices.slices.ranges.clone(),
                buffer_index: indices.buffer_index,
                // parent_buffer_index: indices.parent_buffer_index,
                particle_layout: indices.slices.particle_layout.clone(),
                // parent_particle_layout: indices.slices.parent_particle_layout.clone(),
            })
            .unwrap()
    }

    pub fn get_init_indirect_dispatch_index(&self, id: EffectCacheId) -> Option<u32> {
        Some(
            self.effects
                .get(&id)?
                .cached_child_info
                .as_ref()?
                .init_indirect
                .dispatch_index,
        )
    }

    pub fn get_child_info_max_binding(&self, child_id: EffectCacheId) -> Option<BufferBinding> {
        // Find the parent effect; it's the one storing the list of children and their
        // ChildInfo
        let parent_cache_id = self
            .effects
            .get(&child_id)?
            .cached_child_info
            .as_ref()?
            .parent_cache_id;
        let children_effect_entry = self.effects.get(&parent_cache_id)?.children.as_ref()?;
        Some(children_effect_entry.max_binding(self.child_infos_buffer.buffer()?))
    }

    pub fn get_event_buffer_binding(&self, id: EffectCacheId) -> Option<EventBufferBinding> {
        let cached_effect = self.effects.get(&id)?;
        let cached_parent = cached_effect.cached_child_info.as_ref()?;
        let event_buffer_ref = &cached_parent.init_indirect.event_buffer_ref;
        let event_buffer = self.event_buffers[event_buffer_ref.buffer_index as usize].as_ref()?;
        Some(EventBufferBinding {
            buffer: event_buffer.buffer.clone(),
            offset: event_buffer_ref.slice.start,
            size: event_buffer_ref.slice.end - event_buffer_ref.slice.start,
        })
    }

    /// Get an iterator over all the valid event buffers and their index.
    ///
    /// This skips all deallocated / empty buffers in the underlying linear
    /// storage, and only returns buffers with at least one allocated slice.
    pub fn event_buffers(&self) -> impl Iterator<Item = (u32, &EventBuffer)> {
        self.event_buffers
            .iter()
            .enumerate()
            .filter_map(|(idx, buf)| buf.as_ref().map(|buf| (idx as u32, buf)))
    }

    pub fn get_event_slice(&self, effect_cache_id: EffectCacheId) -> Option<&EventBufferRef> {
        Some(
            &self
                .effects
                .get(&effect_cache_id)?
                .cached_child_info
                .as_ref()?
                .init_indirect
                .event_buffer_ref,
        )
    }

    pub fn child_infos(&self) -> &HybridAlignedBufferVec {
        &self.child_infos_buffer
    }

    pub fn child_infos_buffer(&self) -> Option<&Buffer> {
        self.child_infos_buffer.buffer()
    }

    pub fn invalidate_child_infos_bind_group(&mut self) {
        self.child_infos_bind_group = None;
    }

    pub fn ensure_child_infos_bind_group(&mut self, device: &RenderDevice) {
        let Some(buffer) = self.child_infos_buffer() else {
            return;
        };
        self.child_infos_bind_group = Some(device.create_bind_group(
            "hanabi:bind_group:indirect_child_infos",
            &self.child_infos_bgl,
            &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        ));
    }

    pub fn child_infos_bind_group(&self) -> Option<&BindGroup> {
        self.child_infos_bind_group.as_ref()
    }

    pub fn get_local_child_index(&self, effect_cache_id: EffectCacheId) -> Option<u32> {
        let cached_effect = self.effects.get(&effect_cache_id)?;
        let cached_child_info = cached_effect.cached_child_info.as_ref()?;
        Some(cached_child_info.local_child_index)
    }

    pub fn get_global_child_index(&self, effect_cache_id: EffectCacheId) -> Option<u32> {
        let cached_effect = self.effects.get(&effect_cache_id)?;
        let cached_child_info = cached_effect.cached_child_info.as_ref()?;
        Some(cached_child_info.global_child_index)
    }

    pub fn get_first_child_index(&self, effect_cache_id: EffectCacheId) -> Option<u32> {
        let cached_effect = self.effects.get(&effect_cache_id)?;
        let children = cached_effect.children.as_ref()?;
        Some(children.first_child_index())
    }

    pub fn get_event_dispatch_index(&self, effect_cache_id: EffectCacheId) -> Option<u32> {
        Some(
            self.effects
                .get(&effect_cache_id)?
                .cached_child_info
                .as_ref()?
                .init_indirect
                .dispatch_index,
        )
    }

    #[inline]
    pub fn init_indirect_dispatch_buffer(&self) -> Option<&Buffer> {
        self.init_indirect_dispatch_buffer.buffer()
    }

    #[inline]
    pub fn init_indirect_dispatch_buffer_binding(&self) -> Option<BindingResource> {
        self.init_indirect_dispatch_buffer.binding()
    }

    pub(crate) fn get_dispatch_buffer_indices(&self, id: EffectCacheId) -> &DispatchBufferIndices {
        &self.effects[&id].slices.dispatch_buffer_indices
    }

    pub(crate) fn get_dispatch_buffer_indices_mut(
        &mut self,
        id: EffectCacheId,
    ) -> &mut DispatchBufferIndices {
        &mut self
            .effects
            .get_mut(&id)
            .unwrap()
            .slices
            .dispatch_buffer_indices
    }

    pub(crate) fn get_group_order(&self, id: EffectCacheId) -> &[u32] {
        &self.effects[&id].group_order
    }

    /// Given an iterator over an effect cached ID and its parent's entity,
    /// resolve parent-child relationships and update the internal cache fields
    /// for both parent and children, optionally (re-)allocating the
    /// [`GpuChildInfo`]s associated with each parent.
    pub(crate) fn resolve_parents(&mut self, pairs: impl Iterator<Item = (EffectCacheId, Entity)>) {
        // For each child cache entry with a known parent entity, resolve the parent
        // cache entry, and patch up the child entry's cache info. Also remember the
        // list of children for each parent.
        let mut children =
            HashMap::<EffectCacheId, Vec<EffectCacheId>>::with_capacity(self.effects.len());
        for (child_cache_id, parent_entity) in pairs {
            // Resolve the cached parent effect. This should always exist.
            let Some(parent_cache_id) = self.effect_from_entity.get(&parent_entity) else {
                trace!("Unknown effect for parent entity {:?}", parent_entity);
                continue;
            };

            // Resolve the cached child effect.
            let Some(child_entry) = self.effects.get_mut(&child_cache_id) else {
                trace!(
                    "Invalid child effect with unknown cache ID {:?}",
                    child_cache_id
                );
                continue;
            };
            let Some(cached_child_info) = child_entry.cached_child_info.as_mut() else {
                trace!(
                    "Missing cached child info for child effect {:?} referenced by parent {:?}.",
                    child_cache_id,
                    parent_cache_id,
                );
                continue;
            };

            // Push the child cache ID into the children list
            let child_vec = children.entry(*parent_cache_id).or_default();
            let local_child_index = child_vec.len() as u32;
            child_vec.push(child_cache_id);

            // Update the cached child info. The global child index cannot yet be
            // calculated, as we might need to re-allocate the child list in the global
            // buffer, which would move it.
            cached_child_info.parent_cache_id = *parent_cache_id;
            cached_child_info.local_child_index = local_child_index;
        }

        // Once all parents are resolved, resolve all children and (re-)allocate their
        // GpuChildInfo if needed.
        let mut changed_parents = vec![];
        for (parent_cache_id, child_cache_ids) in children {
            let child_infos = child_cache_ids
                .iter()
                .map(|id| {
                    let init_indirect_dispatch_index =
                        self.get_init_indirect_dispatch_index(*id).expect(&format!(
                            "Failed to find init indirect dispatch index of child effect #{:?}.",
                            *id
                        ));
                    GpuChildInfo {
                        init_indirect_dispatch_index,
                        event_count: 0,
                    }
                })
                .collect::<Vec<_>>();

            let Some(parent_entry) = self.effects.get_mut(&parent_cache_id) else {
                continue;
            };

            // Check if any child changed
            let mut changed = false;
            if let Some(children) = &parent_entry.children {
                if children.effect_cache_ids.len() != child_cache_ids.len() {
                    trace!(
                        "ChildInfo changed for effect {:?}: old #{} != new #{}",
                        parent_cache_id,
                        children.effect_cache_ids.len(),
                        child_cache_ids.len()
                    );
                    changed = true;
                } else {
                    let mut old = children.effect_cache_ids.clone();
                    // We cannot sort the original array, we already assigned the child index
                    let mut new = child_cache_ids.clone();
                    old.sort();
                    new.sort();
                    if old != new {
                        trace!(
                            "ChildInfo changed for effect {:?}: old {:?} != new {:?}",
                            parent_cache_id,
                            old,
                            new
                        );
                        changed = true;
                    }
                }
            } else {
                trace!(
                    "ChildInfo changed for effect {:?}: old (none) != new #{}",
                    parent_cache_id,
                    child_cache_ids.len()
                );

                changed = true;
            }

            // (Re-)allocate the child info array for this parent effect
            if changed {
                let mut prev_start = None;
                if let Some(cached_children) = &parent_entry.children {
                    trace!(
                        "De-allocating old ChildInfo[] entry at range {:?}",
                        cached_children.byte_range
                    );
                    self.child_infos_buffer
                        .remove(cached_children.byte_range.clone());
                    prev_start = Some(cached_children.byte_range.start);
                }

                // FIXME - what about the case where child_infos[] is empty! This doesn't happen
                // here, because we skip those effects, but that means we're not de-allocating
                // old child effects!

                let byte_range = self.child_infos_buffer.push_many(&child_infos[..]);
                assert_eq!(byte_range.start as usize % size_of::<GpuChildInfo>(), 0);
                trace!(
                    "Allocated new ChildInfo[] entry at byte range {:?}",
                    byte_range
                );

                // Only if the parent still has at least one child, and the reallocation changed
                // its base index
                if let Some(prev_start) = prev_start {
                    if prev_start != byte_range.start {
                        trace!(
                            "Parent {:?}: moved ChildInfo[] array from +{} to +{}",
                            parent_cache_id,
                            prev_start,
                            byte_range.start
                        );
                        changed_parents.push(parent_cache_id);
                    }
                } else {
                    trace!(
                        "Parent {:?}: newly allocated ChildInfo[] array at +{}",
                        parent_cache_id,
                        byte_range.start
                    );
                    changed_parents.push(parent_cache_id);
                }

                parent_entry.children = Some(CachedChildren {
                    effect_cache_ids: child_cache_ids,
                    byte_range,
                });
            }
        }

        // Once all parents are re-allocated, fix up the global index of all
        // children if the parent base index changed.
        // TODO: this is the worst code ever, so many useless lookup because of
        // self.effects being a HashMap...
        trace!(
            "Updating the global index of children of {} parent effects...",
            changed_parents.len()
        );
        for parent_cache_id in changed_parents.drain(..) {
            let (base_index, mut children_cache_ids) = {
                let Some(parent_entry) = self.effects.get(&parent_cache_id) else {
                    continue;
                };
                let Some(cached_children) = &parent_entry.children else {
                    continue;
                };
                (
                    cached_children.byte_range.start / 4,
                    cached_children.effect_cache_ids.clone(),
                )
            };
            trace!(
                "Updating {} children of parent effect {:?} with base child index {}...",
                children_cache_ids.len(),
                parent_cache_id,
                base_index
            );
            for child_cache_id in children_cache_ids.drain(..) {
                let Some(child_entry) = self.effects.get_mut(&child_cache_id) else {
                    continue;
                };
                let Some(cached_child_info) = &mut child_entry.cached_child_info else {
                    continue;
                };
                cached_child_info.global_child_index =
                    base_index + cached_child_info.local_child_index;
                trace!(
                    "+ Updated global index for child ID {:?} of parent {:?}: local={}, global={}",
                    child_cache_id,
                    parent_cache_id,
                    cached_child_info.local_child_index,
                    cached_child_info.global_child_index
                );
            }
        }
    }

    /// Get the init bind group for a cached effect.
    pub fn init_bind_group(&self, id: EffectCacheId) -> Option<&BindGroup> {
        if let Some(indices) = self.effects.get(&id) {
            if let Some(effect_buffer) = &self.buffers[indices.buffer_index as usize] {
                return effect_buffer.init_bind_group();
            }
        }
        None
    }

    /// Get the update bind group for a cached effect.
    #[inline]
    pub fn update_bind_group(
        &self,
        id: EffectCacheId,
        child_ids: &[EffectCacheId],
    ) -> Option<&BindGroup> {
        if let Some(indices) = self.effects.get(&id) {
            if let Some(effect_buffer) = &self.buffers[indices.buffer_index as usize] {
                return effect_buffer.update_bind_group(child_ids);
            }
        }
        None
    }

    pub fn get_property_buffer(&self, id: EffectCacheId) -> Option<&Buffer> {
        if let Some(cached_effect_indices) = self.effects.get(&id) {
            let buffer_index = cached_effect_indices.buffer_index as usize;
            self.buffers[buffer_index]
                .as_ref()
                .and_then(|eb| eb.properties_buffer())
        } else {
            None
        }
    }

    /// Remove an effect from the cache. If this was the last effect, drop the
    /// underlying buffer and return the index of the dropped buffer.
    pub fn remove(&mut self, id: EffectCacheId) -> Option<CachedEffect> {
        let indices = self.effects.remove(&id)?;
        let &mut Some(ref mut buffer) = &mut self.buffers[indices.buffer_index as usize] else {
            return None;
        };

        let slice = SliceRef {
            range: indices.slices.ranges[0]..*indices.slices.ranges.last().unwrap(),
            // FIXME: clone() needed to return CachedEffectIndices, but really we don't care about
            // returning the ParticleLayout, so should split...
            particle_layout: indices.slices.particle_layout.clone(),
        };

        if buffer.free_slice(slice) == BufferState::Free {
            self.buffers[indices.buffer_index as usize] = None;
            return Some(indices);
        }

        None
    }

    /// Insert a new event buffer in the cache.
    ///
    /// The event buffer is allocated to store up to `event_count` GPU events
    /// per frame. If possible, a part of an existing GPU buffer is reused as
    /// storage, or if not possible a new GPU buffer is allocated.
    fn insert_event_buffer(
        &mut self,
        effect_cached_id: EffectCacheId,
        event_count: u32,
    ) -> EventBufferRef {
        assert!(event_count > 0, "Cannot allocate event buffer for 0 event.");
        assert!(
            event_count < 65536,
            "Cannot allocate event buffer with 64k or more events. Reduce the event count in effect #{:?}.",
            effect_cached_id
        );

        // Each buffer stores 1 event count + N events, each of which is a single u32 (4
        // bytes). We need each sub-allocation to be aligned such that we can bind it
        // individually to a shader binding point.
        let min_storage_buffer_offset_alignment = self
            .render_device
            .limits()
            .min_storage_buffer_offset_alignment;
        assert!(min_storage_buffer_offset_alignment % 4 == 0); // FIXME
        let size = ((event_count + 1) as usize)
            .next_multiple_of(min_storage_buffer_offset_alignment as usize / 4)
            as u32;
        trace!(
            "Allocating event buffer for effect #{:?} with {} events, rounded up to {}",
            effect_cached_id,
            event_count,
            size
        );

        // Try to allocate inside an existing buffer.
        let mut empty_index = None;
        for (event_buffer_index, event_buffer) in self.event_buffers.iter_mut().enumerate() {
            if let Some(event_buffer) = event_buffer.as_mut() {
                if let Some(slice) = event_buffer.allocate(size) {
                    trace!(
                        "-> reusing event buffer slice {:?} from buffer index #{}",
                        slice.slice,
                        event_buffer_index,
                    );
                    return EventBufferRef {
                        buffer_index: event_buffer_index as u32,
                        slice: slice.slice,
                    };
                }
            } else if empty_index.is_none() {
                empty_index = Some(event_buffer_index);
            }
        }

        // Allocate a new buffer
        let alloc_size = size.next_multiple_of(256).clamp(4096, 65536);
        trace!(
            "-> allocating new event buffer for effect #{:?} with {} events ({} bytes)",
            effect_cached_id,
            alloc_size,
            alloc_size * 4
        );
        // In debug, pad the buffer with some debug marker
        #[cfg(debug_assertions)]
        let buffer = {
            let mut contents: Vec<u32> = Vec::with_capacity(alloc_size as usize);
            contents.resize(alloc_size as usize, 0xDEADBEEF);
            self.render_device
                .create_buffer_with_data(&BufferInitDescriptor {
                    label: Some(&format!("hanabi:buffer:event_buffer{}", effect_cached_id.0)),
                    usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
                    contents: bytemuck::cast_slice(contents.as_slice()),
                })
        };
        // In release, don't initialize the buffer for performance
        #[cfg(not(debug_assertions))]
        let buffer = self.render_device.create_buffer(&BufferDescriptor {
            label: Some(&format!("hanabi:buffer:event_buffer{}", effect_cached_id.0)),
            size: alloc_size as u64 * 4,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut event_buffer = EventBuffer::new(effect_cached_id, buffer, alloc_size);
        let slice = event_buffer.allocate(size).unwrap();
        let buffer_index = if let Some(buffer_index) = empty_index {
            self.event_buffers[buffer_index] = Some(event_buffer);
            buffer_index as u32
        } else {
            let buffer_index = self.event_buffers.len() as u32;
            self.event_buffers.push(Some(event_buffer));
            buffer_index
        };
        trace!(
            "-> event buffer for effect #{:?} inserted at index #{}, slice {:?}",
            effect_cached_id,
            buffer_index,
            slice.slice
        );
        EventBufferRef {
            buffer_index,
            slice: slice.slice,
        }
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
            PropertyLayout::empty(), // not using properties
            None,
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
            PropertyLayout::empty(), // not using properties
            None,
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

        let empty_property_layout = PropertyLayout::empty(); // not using properties

        let l32 = ParticleLayout::new().append(F4A).append(F4B).build();
        assert_eq!(32, l32.size());

        let mut effect_cache = EffectCache::new(render_device);
        assert_eq!(effect_cache.buffers().len(), 0);

        let asset = Handle::<EffectAsset>::default();
        let capacity = EffectBuffer::MIN_CAPACITY;
        let capacities = vec![capacity];
        let group_order = vec![0];
        let item_size = l32.size();

        let id1 = effect_cache.insert(
            Entity::PLACEHOLDER,
            asset.clone(),
            capacities.clone(),
            &l32,
            None,
            &empty_property_layout,
            LayoutFlags::NONE,
            0,
            DispatchBufferIndices::default(),
            group_order.clone(),
        );
        assert!(id1.is_valid());
        let slice1 = effect_cache.get_slices(id1);
        assert_eq!(
            slice1.particle_layout.min_binding_size().get() as u32,
            item_size
        );
        assert_eq!(slice1.slices, vec![0, capacity]);
        assert_eq!(effect_cache.buffers().len(), 1);

        let id2 = effect_cache.insert(
            Entity::PLACEHOLDER,
            asset.clone(),
            capacities.clone(),
            &l32,
            None,
            &empty_property_layout,
            LayoutFlags::NONE,
            0,
            DispatchBufferIndices::default(),
            group_order.clone(),
        );
        assert!(id2.is_valid());
        let slice2 = effect_cache.get_slices(id2);
        assert_eq!(
            slice2.particle_layout.min_binding_size().get() as u32,
            item_size
        );
        assert_eq!(slice2.slices, vec![0, capacity]);
        assert_eq!(effect_cache.buffers().len(), 2);

        let cached_effect_indices = effect_cache.remove(id1).unwrap();
        assert_eq!(cached_effect_indices.buffer_index, 0);
        assert_eq!(effect_cache.buffers().len(), 2);
        {
            let buffers = effect_cache.buffers();
            assert!(buffers[0].is_none());
            assert!(buffers[1].is_some()); // id2
        }

        // Regression #60
        let id3 = effect_cache.insert(
            Entity::PLACEHOLDER,
            asset,
            capacities,
            &l32,
            None,
            &empty_property_layout,
            LayoutFlags::NONE,
            0,
            DispatchBufferIndices::default(),
            group_order,
        );
        assert!(id3.is_valid());
        let slice3 = effect_cache.get_slices(id3);
        assert_eq!(
            slice3.particle_layout.min_binding_size().get() as u32,
            item_size
        );
        assert_eq!(slice3.slices, vec![0, capacity]);
        assert_eq!(effect_cache.buffers().len(), 2);
        {
            let buffers = effect_cache.buffers();
            assert!(buffers[0].is_some()); // id3
            assert!(buffers[1].is_some()); // id2
        }
    }
}
