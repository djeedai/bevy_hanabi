use std::{num::NonZeroU64, ops::Range};

use bevy::{
    log::{trace, warn},
    prelude::{Component, Entity, ResMut, Resource},
    render::{
        render_resource::{
            BindGroup, BindGroupLayout, Buffer, BufferVec, ShaderSize as _, ShaderType,
        },
        renderer::{RenderDevice, RenderQueue},
        sync_world::MainEntity,
    },
};
use bytemuck::{Pod, Zeroable};
use thiserror::Error;
#[cfg(debug_assertions)]
use wgpu::util::BufferInitDescriptor;
#[cfg(not(debug_assertions))]
use wgpu::BufferDescriptor;
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType, BufferBinding,
    BufferBindingType, BufferUsages, ShaderStages,
};

use super::{
    aligned_buffer_vec::HybridAlignedBufferVec, effect_cache::BufferState, BufferBindingSource,
    EffectBindGroups, GpuDispatchIndirect,
};
use crate::ParticleLayout;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EventSlice {
    slice: Range<u32>,
}

/// GPU buffer storing the spawn events emitted by parent effects for their
/// children.
///
/// The event buffer contains for each effect the number of particles to spawn
/// this frame. That number is incremented by another effect when it
/// emits a spawn event, and reset to zero on next frame after the indirect init
/// pass spawned new particles, and before the new update pass of the
/// source effect optionally emits more spawn events. GPU spawn events are never
/// accumulated over frames; if a source emits too many events and the target
/// effect cannot spawn that many particles, for example because it reached its
/// capacity, then the extra events are discarded. This is consistent with the
/// CPU behavior of [`Spawner::spawn_count`].
pub struct EventBuffer {
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
    pub fn new(buffer: Buffer, capacity: u32) -> Self {
        Self {
            buffer,
            capacity,
            size: 0,
            slices: vec![],
        }
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

    /// Free the slice of a consumer effect once that effect is deallocated.
    pub fn free(&mut self, slice: &EventSlice) -> BufferState {
        // Note: could use binary search, but likely not enough elements to be worth it
        if let Some(idx) = self.slices.iter().position(|es| es == slice) {
            self.slices.remove(idx);
        }
        if self.slices.is_empty() {
            BufferState::Free
        } else {
            BufferState::Used
        }
    }
}

/// Data about the child effect(s) of this effect. This component is only
/// present on an effect instance if that effect is the parent effect for at
/// least one child effect.
#[derive(Debug, Component)]
pub(crate) struct CachedParentInfo {
    /// Render world entities of the child effects, and their associated event
    /// buffer binding source.
    pub children: Vec<(Entity, BufferBindingSource)>,
    /// Indices in bytes into the global [`EffectCache::child_infos_buffer`] of
    /// the [`GpuChildInfo`]s for all the child effects of this parent effect.
    /// The child effects are always allocated as a single contiguous block,
    /// which needs to be mapped into a shader binding point.
    pub byte_range: Range<u32>,
}

impl CachedParentInfo {
    /// Get a binding of the given underlying child info buffer spanning over
    /// the range of this child effect entry.
    #[allow(dead_code)]
    pub fn binding<'a>(&self, buffer: &'a Buffer) -> BufferBinding<'a> {
        BufferBinding {
            buffer,
            offset: self.byte_range.start as u64,
            size: Some(
                NonZeroU64::new((self.byte_range.end - self.byte_range.start) as u64).unwrap(),
            ),
        }
    }

    /// Base offset of the first child into the global
    /// [`EventCache::child_infos_buffer`], in number of element.
    #[allow(dead_code)]
    pub fn first_child_index(&self) -> u32 {
        self.byte_range.start / size_of::<GpuChildInfo>() as u32
    }
}

/// Reference to the parent of an effect instance.
///
/// This is a weak reference to the parent while pending resolving. This
/// component is always present on an effect instance if that effect is the
/// child effect for another effect (that is, this effect has a parent effect),
/// even if the parent is invalid. Once the parent is resolved,
/// [`CachedChildInfo`] is also spawned.
#[derive(Debug, Component)]
pub(crate) struct CachedParentRef {
    /// The main entity of the parent of this effect instance, as declared by
    /// the effect.
    pub entity: MainEntity,
}

/// Data about this effect as a child of another effect.
///
/// This component is only present on an effect instance if that effect is the
/// child effect for another effect (that is, this effect has a parent effect).
/// However, unlike [`CachedParentRef`], if the parent could no be resolved then
/// this component is absent.
#[derive(Debug, Component)]
pub(crate) struct CachedChildInfo {
    /// Render entity of the parent effect. This entity is resolved and always
    /// valid, otherwise this component is removed.
    pub parent: Entity,
    /// Parent's particle layout.
    pub parent_particle_layout: ParticleLayout,
    /// Parent's buffer.
    pub parent_buffer_binding_source: BufferBindingSource,
    /// Index of this child effect into its parent's [`GpuChildInfo`] array.
    /// This starts at zero for the first child of each effect, and is only
    /// unique per parent only, not globally.
    pub local_child_index: u32,
    /// Global index of this child effect into the shared global
    /// [`EventCache::child_infos_buffer`] array. This is a unique index across
    /// all effects.
    ///
    /// [`EventCache::child_infos_buffer`]: super::EventCache::child_infos_buffer
    pub global_child_index: u32,
    /// Index of the [`GpuDispatchIndirect`] entry into the
    /// [`init_indirect_dispatch_buffer`] array.
    ///
    /// [`init_indirect_dispatch_buffer`]: super::EventCache::init_indirect_dispatch_buffer
    pub init_indirect_dispatch_index: u32,
}

/// GPU representation of the child info data structure storing some data for a
/// child effect. The associated CPU representation is [`CachedEffectEvents`].
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct GpuChildInfo {
    /// Index of the [`GpuDispatchIndirect`] inside the
    /// [`EventCache::init_indirect_dispatch_buffer`] used to dispatch the init
    /// pass of this child effect.
    pub init_indirect_dispatch_index: u32,
    /// Number of events currently stored inside the [`EventBuffer`] slice
    /// associated with this child effect. This is updated atomically by the
    /// GPU while stored in the [`EventCache::child_infos_buffer`].
    pub event_count: i32,
}

#[derive(Debug, Clone, Component)]
pub struct CachedEffectEvents {
    /// Index of the [`EventBuffer`] inside the [`EventCache::buffers`]
    /// collection where a slice of events is allocated, for this effect to
    /// consume.
    pub buffer_index: u32,
    /// Range, in items (4 bytes), where the events are stored inside the
    /// [`EventBuffer`]. This determines the capacity, in event count, for this
    /// effect. The number of used events is stored on the GPU in
    /// [`GpuChildInfo::event_count`].
    pub range: Range<u32>,
    /// Index of the [`GpuDispatchIndirect`] inside the
    /// [`EventCache::init_indirect_dispatch_buffer`].
    pub init_indirect_dispatch_index: u32,
}

impl CachedEffectEvents {
    /// Capacity of this allocation, in number of GPU events. The number of used
    /// events is stored on the GPU in [`GpuChildInfo::event_count`].
    #[allow(dead_code)]
    pub fn capacity(&self) -> u32 {
        self.range.len() as u32
    }
}

/// Error code for [`EventCache::free()`].
#[derive(Debug, Error)]
pub enum CachedEventsError {
    /// The given buffer index is invalid. The [`EventCache`] doesn't contain
    /// any buffer with such index.
    #[error("Invalid buffer index #{0}.")]
    InvalidBufferIndex(u32),
    /// The given buffer index corresponds to a [`EventCache`] buffer which
    /// was already deallocated.
    #[error("Buffer at index #{0} was deallocated.")]
    BufferDeallocated(u32),
}

/// Cache for effect events.
#[derive(Resource)]
pub struct EventCache {
    /// Render device to allocate GPU resources as needed.
    device: RenderDevice,
    /// Single shared GPU buffer storing all the [`GpuChildInfo`] structs
    /// for all the parent effects.
    child_infos_buffer: HybridAlignedBufferVec,
    /// Collection of event buffers managed by this cache. Some buffers might
    /// be `None` if the entry is not used. Since the buffers are referenced
    /// by index, we cannot move them once they're allocated.
    buffers: Vec<Option<EventBuffer>>,
    /// Single shared GPU buffer storing all the [`GpuDispatchIndirect`]
    /// structs for all the indirect init passes. Any effect allocating storage
    /// for GPU events also get an entry into this buffer, to allow consuming
    /// the events from an init pass indirectly dispatched (GPU-driven).
    // Note: we abuse BufferVec but never copy anything from CPU
    // FIXME - merge with the update pass one, we don't need 2 buffers storing the same type
    init_indirect_dispatch_buffer: BufferVec<GpuDispatchIndirect>,
    /// Bind group layout for the indirect dispatch pass, which clears the GPU
    /// event counts ([`GpuChildInfo::event_count`]).
    indirect_child_info_buffer_bind_group_layout: BindGroupLayout,
    /// Bind group for the indirect dispatch pass, which clears the GPU event
    /// counts ([`GpuChildInfo::event_count`]).
    indirect_child_info_buffer_bind_group: Option<BindGroup>,
}

impl EventCache {
    /// Create a new event cache.
    pub fn new(device: RenderDevice) -> Self {
        let mut init_indirect_dispatch_buffer =
            BufferVec::new(BufferUsages::STORAGE | BufferUsages::INDIRECT);
        init_indirect_dispatch_buffer.set_label(Some("hanabi:buffer:init_indirect_dispatch"));

        let child_infos_bind_group_layout = device.create_bind_group_layout(
            "hanabi:bind_group_layout:indirect:child_infos@3",
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
            device,
            child_infos_buffer: HybridAlignedBufferVec::new(
                BufferUsages::STORAGE,
                Some(NonZeroU64::new(4).unwrap()),
                Some("hanabi:buffer:child_infos".to_string()),
            ),
            buffers: vec![],
            init_indirect_dispatch_buffer,
            indirect_child_info_buffer_bind_group_layout: child_infos_bind_group_layout,
            // Can't create until the buffer is ready
            indirect_child_info_buffer_bind_group: None,
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub fn buffers(&self) -> &[Option<EventBuffer>] {
        &self.buffers
    }

    #[allow(dead_code)]
    #[inline]
    pub fn buffers_mut(&mut self) -> &mut [Option<EventBuffer>] {
        &mut self.buffers
    }

    #[inline]
    pub fn get_buffer(&self, index: u32) -> Option<&Buffer> {
        self.buffers
            .get(index as usize)
            .and_then(|opt_eb| opt_eb.as_ref().map(|eb| eb.buffer()))
    }

    #[inline]
    pub fn child_infos_buffer(&self) -> Option<&Buffer> {
        self.child_infos_buffer.buffer()
    }

    /// Allocate a memory block to store the given number of GPU events.
    ///
    /// The allocation always succeeds, allocating a new GPU event buffer if
    /// none of the existing ones can store the requested number of events.
    ///
    /// # Returns
    ///
    /// The [`CachedEffectEvents`] component representing the allocation.
    ///
    /// # Panics
    ///
    /// Panics if the number of events `num_events` is zero.
    pub fn allocate(&mut self, num_events: u32) -> CachedEffectEvents {
        assert!(num_events > 0);

        // Allocate an entry into the indirect dispatch buffer
        // The value pushed is a dummy; see allocate_frame_buffers().
        let init_indirect_dispatch_index = self
            .init_indirect_dispatch_buffer
            .push(GpuDispatchIndirect::default()) as u32;

        // Try to find an allocated GPU buffer with enough capacity
        let mut empty_index = None;
        for (buffer_index, buffer) in self.buffers.iter_mut().enumerate() {
            let Some(buffer) = buffer.as_mut() else {
                // Remember the first empty slot in case we need to allocate a new GPU buffer
                if empty_index.is_none() {
                    empty_index = Some(buffer_index);
                }
                continue;
            };

            // Try to allocate a slice into the buffer
            if let Some(event_slice) = buffer.allocate(num_events) {
                trace!("Allocate new slice in event buffer #{buffer_index} for {num_events} events: range={event_slice:?}");
                return CachedEffectEvents {
                    buffer_index: buffer_index as u32,
                    init_indirect_dispatch_index,
                    range: event_slice.slice,
                };
            }
        }

        // Cannot find any suitable GPU event buffer; allocate a new one

        // Compute the slot where to store the new buffer
        let buffer_index = empty_index.unwrap_or(self.buffers.len());

        // Create the GPU bfufer
        let label = format!("hanabi:buffer:event_buffer{buffer_index}");
        let align = self.device.limits().min_storage_buffer_offset_alignment;
        let capacity = num_events.max(16 * 1024); // min capacity
        let byte_size = (capacity as u64 * 4).next_multiple_of(align as u64);
        let capacity = (byte_size / 4) as u32;
        // In debug, fill the buffer with some debug marker
        #[cfg(debug_assertions)]
        let buffer = {
            let mut contents: Vec<u32> = Vec::with_capacity(capacity as usize);
            contents.resize(capacity as usize, 0xDEADBEEF);
            self.device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(&label[..]),
                usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(contents.as_slice()),
            })
        };
        // In release, don't initialize the buffer for performance
        #[cfg(not(debug_assertions))]
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some(&label[..]),
            size: byte_size,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        trace!("Created new event buffer #{buffer_index} '{label}' with {byte_size} bytes ({capacity} events; align={align}B)");
        let mut buffer = EventBuffer::new(buffer, capacity);

        // Allocate a slice from the new event buffer
        let event_slice = buffer.allocate(num_events).expect("Failed to allocate event slice inside new buffer specifically created for this allocation.");
        trace!("Allocate new slice in event buffer #{buffer_index} for {num_events} events: range={event_slice:?}");

        // Store the event buffer at the selected slot
        if buffer_index >= self.buffers.len() {
            self.buffers.push(Some(buffer));
        } else {
            debug_assert!(self.buffers[buffer_index].is_none());
            self.buffers[buffer_index] = Some(buffer);
        }

        CachedEffectEvents {
            buffer_index: buffer_index as u32,
            init_indirect_dispatch_index,
            range: event_slice.slice,
        }
    }

    /// Deallocated and remove an event block allocation from the cache.
    pub fn free(
        &mut self,
        cached_effect_events: &CachedEffectEvents,
    ) -> Result<BufferState, CachedEventsError> {
        trace!(
            "Removing cached event {:?} from cache.",
            cached_effect_events
        );

        // FIXME - free() not implemented in BufferVec!
        warn!("free() not implemented in BufferVec, cannot free GpuInitDispatchInfo entry");
        // self.init_indirect_dispatch_buffer
        //     .free(cached_effect_events.init_indirect_dispatch_index);

        let entry = self
            .buffers
            .get_mut(cached_effect_events.buffer_index as usize)
            .ok_or(CachedEventsError::InvalidBufferIndex(
                cached_effect_events.buffer_index,
            ))?;
        let buffer = entry.as_mut().ok_or(CachedEventsError::BufferDeallocated(
            cached_effect_events.buffer_index,
        ))?;
        if buffer.free(&EventSlice {
            slice: cached_effect_events.range.clone(),
        }) == BufferState::Free
        {
            let buffer = entry.take().unwrap();
            buffer.buffer.destroy();
            Ok(BufferState::Free)
        } else {
            Ok(BufferState::Used)
        }
    }

    /// Allocate a new block of [`GpuChildInfo`] structures for a list of
    /// children.
    pub fn allocate_child_infos(
        &mut self,
        parent_entity: Entity,
        children: Vec<(Entity, BufferBindingSource)>,
        child_infos: &[GpuChildInfo],
    ) -> CachedParentInfo {
        assert_eq!(children.len(), child_infos.len());
        assert!(!children.is_empty());

        let byte_range = self.child_infos_buffer.push_many(child_infos);
        assert_eq!(byte_range.start as usize % size_of::<GpuChildInfo>(), 0);
        trace!(
            "Parent {:?}: newly allocated ChildInfo[] array at +{}",
            parent_entity,
            byte_range.start
        );

        CachedParentInfo {
            children,
            byte_range,
        }
    }

    /// Re-allocate a block of [`GpuChildInfo`] structures for a modified list
    /// of children.
    pub fn reallocate_child_infos(
        &mut self,
        parent_entity: Entity,
        children: Vec<(Entity, BufferBindingSource)>,
        child_infos: &[GpuChildInfo],
        cached_parent_info: &mut CachedParentInfo,
    ) {
        trace!(
            "Parent {:?}: De-allocating old ChildInfo[] entry at range {:?}",
            parent_entity,
            cached_parent_info.byte_range
        );
        self.child_infos_buffer
            .remove(cached_parent_info.byte_range.clone());

        let byte_range = self.child_infos_buffer.push_many(child_infos);
        assert_eq!(
            byte_range.start as usize % GpuChildInfo::SHADER_SIZE.get() as usize,
            0
        );
        trace!(
            "Parent {:?}: Allocated new ChildInfo[] entry at byte range {:?}",
            parent_entity,
            byte_range
        );

        cached_parent_info.children = children;
        cached_parent_info.byte_range = byte_range;
    }

    /// Re-/allocate any buffer for the current frame.
    pub fn prepare_buffers(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        // FIXME
        _effect_bind_groups: &mut ResMut<EffectBindGroups>,
    ) {
        // Note: we abuse BufferVec for its ability to manage the GPU buffer, but
        // we don't use its CPU side capabilities. So we only need the GPU buffer to be
        // correctly allocated, using reserve(). No data is copied from CPU.
        self.init_indirect_dispatch_buffer
            .reserve(self.init_indirect_dispatch_buffer.len(), render_device);

        let old_buffer = self.child_infos_buffer.buffer().map(|b| b.id());
        self.child_infos_buffer
            .write_buffer(render_device, render_queue);
        let new_buffer = self.child_infos_buffer.buffer().map(|b| b.id());
        if old_buffer != new_buffer && old_buffer.is_some() {
            // If the child infos buffer changed, all init bind groups of children and all
            // update bind groups of parents are invalid because they all use that globally
            // shared buffer.

            todo!("Invalidate ChildInfo bind groups");

            // Init pass
            // FIXME group@3 if CONSUME_GPU_SPAWN_EVENTS

            // Indirect pass
            // FIXME group@3 if HAS_GPU_SPAWN_EVENTS

            // FIXME - it's quite hard to tell for now; just invalidate
            // everything for buffer in &mut self.buffers {
            //     if let Some(buffer) = buffer {
            //         buffer.invalidate_all_bind_groups();
            //     }
            // }

            //effect_bind_groups.init_fill_dispatch.clear();
        }
    }

    #[inline]
    pub fn init_indirect_dispatch_buffer(&self) -> Option<&Buffer> {
        self.init_indirect_dispatch_buffer.buffer()
    }

    #[inline]
    pub fn child_infos(&self) -> &HybridAlignedBufferVec {
        &self.child_infos_buffer
    }

    #[inline]
    pub fn init_indirect_dispatch_binding_resource(&self) -> Option<BindingResource> {
        self.init_indirect_dispatch_buffer.binding()
    }

    // pub fn get_init_indirect_dispatch_index(&self, id: EffectCacheId) ->
    // Option<u32> {     Some(
    //         self.effects
    //             .get(&id)?
    //             .cached_child_info
    //             .as_ref()?
    //             .init_indirect
    //             .dispatch_index,
    //     )
    // }

    pub fn ensure_indirect_child_info_buffer_bind_group(
        &mut self,
        device: &RenderDevice,
    ) -> Option<&BindGroup> {
        let buffer = self.child_infos_buffer()?;
        // TODO - stop re-creating each frame...
        self.indirect_child_info_buffer_bind_group = Some(device.create_bind_group(
            "hanabi:bind_group:indirect:child_infos@3",
            &self.indirect_child_info_buffer_bind_group_layout,
            &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        ));
        self.indirect_child_info_buffer_bind_group.as_ref()
    }

    pub fn indirect_child_info_buffer_bind_group(&self) -> Option<&BindGroup> {
        self.indirect_child_info_buffer_bind_group.as_ref()
    }
}
