use std::{
    num::{NonZeroU32, NonZeroU64},
    ops::Range,
};

use bevy::{
    ecs::{lifecycle::Remove, observer::On, system::Commands, world::Mut},
    log::{error, trace},
    platform::collections::HashMap,
    prelude::{Component, Entity, Query, Res, ResMut, Resource},
    render::{
        render_resource::{
            binding_types::{storage_buffer_read_only, storage_buffer_read_only_sized},
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            PipelineCache, ShaderSize,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::{cast_slice, Pod};
use wgpu::{
    BindingResource, BufferAddress, BufferBinding, BufferDescriptor, BufferUsages, ShaderStages,
};

use super::effect_cache::SlabState;
use crate::{
    render::{ExtractedProperties, GpuBatchInfo, GpuSpawnerParams, StorageType},
    PropertyLayout,
};

/// Allocation into the [`PropertyCache`] for an effect instance. This component
/// is only present on an effect instance if that effect uses properties.
#[derive(Debug, Clone, PartialEq, Eq, Component)]
pub struct CachedEffectProperties {
    /// Property layout.
    pub property_layout: PropertyLayout,
    /// Index of the [`PropertyBuffer`] inside the [`PropertyCache`].
    pub buffer_index: u32,
    /// Array index inside the GPU buffer where the properties struct with the
    /// current layout is allocated.
    pub array_index: u32,
}

impl CachedEffectProperties {
    /// Convert this allocation into a bind group key, used for bind group
    /// re-creation when a change is detected in the key.
    pub fn to_key(&self) -> PropertyBindGroupKey {
        let binding_size = self.property_layout.min_binding_size().get() as u32;
        PropertyBindGroupKey {
            buffer_index: self.buffer_index,
            binding_size,
        }
    }
}

/// Error code for [`PropertyCache::remove_properties()`].
#[derive(Debug)]
pub enum CachedPropertiesError {
    /// The given buffer index is invalid. The [`PropertyCache`] doesn't contain
    /// any buffer with such index.
    InvalidBufferIndex(u32),
    /// The given buffer index corresponds to a [`PropertyCache`] buffer which
    /// was already deallocated.
    BufferDeallocated(u32),
}

/// Single free byte range inside a [`PropertyBuffer`].
#[derive(Debug, Clone, PartialEq, Eq)]
struct FreeRange(pub Range<u32>);

impl PartialOrd for FreeRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FreeRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.start.cmp(&other.0.start)
    }
}

#[derive(Debug)]
pub(crate) struct PropertyBuffer {
    /// Pending values accumulated on CPU and not yet written to GPU.
    values: Vec<u8>,
    /// GPU buffer if already allocated, or `None` otherwise.
    buffer: Option<Buffer>,
    /// Capacity of the buffer, in bytes.
    capacity: usize,
    /// GPU buffer name, for debugging.
    label: String,
    /// Free ranges available for re-allocation (in bytes).
    free_ranges: Vec<FreeRange>,
    /// Is the GPU buffer stale and the CPU one need to be re-uploaded?
    is_stale: bool,
    buffer_usages: BufferUsages,
}

impl PropertyBuffer {
    /// Create a new property buffer.
    ///
    /// The GPU resources are not yet allocated.
    pub fn new(label: Option<String>, buffer_usages: BufferUsages) -> Self {
        let label = label.unwrap_or("hanabi:buffer:properties".to_string());
        trace!("PropertyBuffer['{}']::new()", label);
        Self {
            values: vec![],
            buffer: None,
            capacity: 0,
            label,
            free_ranges: vec![],
            is_stale: true,
            buffer_usages,
        }
    }

    /// Get the GPU buffer, if allocated.
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Get a binding for the entire buffer.
    #[allow(dead_code)]
    #[inline]
    pub fn max_binding(&self) -> Option<BindingResource<'_>> {
        // FIXME - Return a Buffer wrapper first, which can be unwrapped, then from that
        // wrapper implement all the xxx_binding() helpers. That avoids a bunch of "if
        // let Some()" everywhere when we know the buffer is valid. The only reason the
        // buffer might not be valid is if it was not created, and in that case
        // we wouldn't be calling the xxx_bindings() helpers, we'd have earlied out
        // before.
        let buffer = self.buffer()?;
        Some(BindingResource::Buffer(BufferBinding {
            buffer,
            offset: 0,
            size: None, // entire buffer
        }))
    }

    /// Capacity of the allocated GPU buffer, in bytes.
    ///
    /// This may be zero if the buffer was not allocated yet. In general, this
    /// can differ from the actual data size cached on CPU and waiting to be
    /// uploaded to GPU.
    #[inline]
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current buffer size, in bytes.
    ///
    /// This represents the size of the CPU data uploaded to GPU. Pending a GPU
    /// buffer re-allocation or re-upload, this size might differ from the
    /// actual GPU buffer size. But they're eventually consistent.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Allocate storage for a property struct of the given size.
    ///
    /// The allocation is guaranteed to be aligned to the size itself. That
    /// means that the returned byte offset is such that `offset % size == 0`.
    /// This ensures `offset / size` forms a valid integral array index.
    fn alloc_aligned(&mut self, size: u32) -> u32 {
        self.is_stale = true;

        // Try to find a free block which can accomodate it, and pick the
        // smallest one in order to limit wasted space.
        let mut best_slot: Option<(u32, usize)> = None;
        for (index, range) in self.free_ranges.iter().enumerate() {
            // Align the expected alloc start
            let aligned_start = range.0.start.next_multiple_of(size);
            if aligned_start >= range.0.end {
                continue;
            }
            let align_padding = aligned_start - range.0.start;

            // Check the remaining size
            let free_size = range.0.end - aligned_start;
            if free_size < size {
                continue;
            }

            // If we found a slot with the exact size, just use it already
            let wasted_size = (free_size - size) + align_padding;
            if wasted_size == 0 {
                best_slot = Some((0, index));
                break;
            }

            // Otherwise try to find the smallest oversized slot to reduce wasted space
            if let Some(best_slot) = best_slot.as_mut() {
                if wasted_size < best_slot.0 {
                    *best_slot = (wasted_size, index);
                }
            } else {
                best_slot = Some((wasted_size, index));
            }
        }

        // Insert into existing space
        if let Some((_, index)) = best_slot {
            let range = self.free_ranges[index].0.clone();

            let aligned_start = range.start.next_multiple_of(size);
            let free_size = range.end - aligned_start;
            assert!(size <= free_size);

            let padding_bytes = aligned_start - range.start;
            if padding_bytes > 0 {
                // If the allocation doesn't span the entire free space, splice it.
                self.free_ranges[index].0.end = aligned_start;
            } else if size < free_size {
                self.free_ranges[index].0.start += size;
            } else {
                // Otherwise, steal the entire block
                self.free_ranges.remove(index);
            }

            debug_assert!(aligned_start.is_multiple_of(size));
            aligned_start
        }
        // Insert at end of vector, after resizing it
        else {
            // Calculate new aligned insertion offset and new capacity
            let old_len = self.values.len() as u32;
            let aligned_start = old_len.next_multiple_of(size);
            let new_capacity = aligned_start + size;
            if new_capacity > old_len {
                self.values.resize(new_capacity as usize, 0);
            }

            // Insert padding if needed (and mark as free)
            if aligned_start > old_len {
                self.free_ranges.push(FreeRange(old_len..aligned_start));
            }

            aligned_start
        }
    }

    /// Remove a range of bytes previously added.
    ///
    /// Remove a range of bytes previously returned by adding one or more
    /// elements with [`push()`] or [`push_many()`].
    ///
    /// # Returns
    ///
    /// Returns `true` if the range was valid and the corresponding data was
    /// removed, or `false` otherwise. In that case, the buffer is not modified.
    ///
    /// [`push()`]: Self::push
    /// [`push_many()`]: Self::push_many
    fn remove(&mut self, range: Range<u32>) -> bool {
        // Check for out of bounds argument
        let end = self.values.len() as u32;
        if range.start >= end || range.end > end {
            return false;
        }

        if range.end == end {
            // Walk the (sorted) free list to also dequeue any range which is now at the end
            // of the buffer
            let mut new_end = range.start;
            while let Some(free_range) = self.free_ranges.pop() {
                if free_range.0.end == new_end {
                    new_end = free_range.0.start;
                } else {
                    self.free_ranges.push(free_range);
                    break;
                }
            }

            // Note: we can't really recover any padding here because we don't know the
            // exact size of that allocation, only its row-aligned size.
            if new_end > 0 {
                self.values.truncate(new_end as usize);
            } else {
                self.values.clear();
            }
        } else {
            // Otherwise, save the row into the free list.
            let free_range = FreeRange(range);

            // Insert as sorted
            if self.free_ranges.is_empty() {
                // Special case to simplify below, and to avoid binary_search()
                self.free_ranges.push(free_range);
            } else if let Err(index) = self.free_ranges.binary_search(&free_range) {
                if index >= self.free_ranges.len() {
                    // insert at end
                    let prev = self.free_ranges.last_mut().unwrap(); // known
                    if prev.0.end == free_range.0.start {
                        // merge with last value
                        prev.0.end = free_range.0.end;
                    } else {
                        // insert last, with gap
                        self.free_ranges.push(free_range);
                    }
                } else if index == 0 {
                    // insert at start
                    let next = &mut self.free_ranges[0];
                    if free_range.0.end == next.0.start {
                        // merge with next
                        next.0.start = free_range.0.start;
                    } else {
                        // insert first, with gap
                        self.free_ranges.insert(0, free_range);
                    }
                } else {
                    // insert between 2 existing elements
                    let prev = &mut self.free_ranges[index - 1];
                    if prev.0.end == free_range.0.start {
                        // merge with previous value
                        prev.0.end = free_range.0.end;

                        let prev = self.free_ranges[index - 1].clone();
                        let next = &mut self.free_ranges[index];
                        if prev.0.end == next.0.start {
                            // also merge prev with next, and remove prev
                            next.0.start = prev.0.start;
                            self.free_ranges.remove(index - 1);
                        }
                    } else {
                        let next = &mut self.free_ranges[index];
                        if free_range.0.end == next.0.start {
                            // merge with next value
                            next.0.start = free_range.0.start;
                        } else {
                            // insert between 2 values, with gaps on both sides
                            self.free_ranges.insert(0, free_range);
                        }
                    }
                }
            } else {
                // The range exists in the free list, this means it's already removed. This is a
                // duplicate; ignore it.
                return false;
            }
        }
        self.is_stale = true;
        true
    }

    /// Update an allocated entry with a new value.
    #[allow(dead_code)]
    #[inline]
    pub fn update<T: Pod + ShaderSize>(&mut self, offset: u32, value: &T) {
        let data: &[u8] = cast_slice(std::slice::from_ref(value));
        assert_eq!(value.size().get() as usize, data.len());
        self.update_raw(offset, data);
    }

    /// Update an allocated entry with new data.
    pub fn update_raw(&mut self, offset: u32, data: &[u8]) {
        // Check for out of bounds argument
        let end = self.values.len() as u32;
        let data_end = offset + data.len() as u32;
        if offset >= end || data_end > end {
            return;
        }

        let dst: &mut [u8] = &mut self.values[offset as usize..data_end as usize];
        dst.copy_from_slice(data);

        self.is_stale = true;
    }

    /// Reserve some capacity into the buffer.
    ///
    /// If the buffer is reallocated, the old content (on the GPU) is lost, and
    /// needs to be re-uploaded to the newly-created buffer. This is done with
    /// [`write_buffer()`].
    ///
    /// # Returns
    ///
    /// `true` if the buffer was (re)allocated, or `false` if an existing buffer
    /// was reused which already had enough capacity.
    ///
    /// [`write_buffer()`]: crate::AlignedBufferVec::write_buffer
    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) -> bool {
        if capacity > self.capacity {
            trace!(
                "reserve: increase capacity from {} to {} bytes",
                self.capacity,
                capacity,
            );
            self.capacity = capacity;
            if let Some(old_buffer) = self.buffer.take() {
                trace!("reserve: forgetting old buffer #{:?}", old_buffer.id());
                // Do not explicitly destroy the old buffer here; let the
                // backend drop it safely.
            }
            self.buffer = Some(device.create_buffer(&BufferDescriptor {
                label: Some(&self.label[..]),
                size: capacity as BufferAddress,
                usage: BufferUsages::COPY_DST | self.buffer_usages,
                mapped_at_creation: false,
            }));
            self.is_stale = !self.values.is_empty();
            // FIXME - this discards the old content if any!!!
            true
        } else {
            false
        }
    }

    /// Schedule the buffer write to GPU.
    ///
    /// # Returns
    ///
    /// `true` if the buffer was (re)allocated, `false` otherwise. If the buffer
    /// was reallocated, all bind groups referencing the old buffer should be
    /// destroyed.
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) -> bool {
        if self.values.is_empty() || !self.is_stale {
            return false;
        }
        // Round GPU allocations to 256 bytes; we don't want to keep doing
        // micro-allocations all the time.
        let size = self.values.len();
        let capacity = size.next_multiple_of(256);
        trace!(
            "property buffer: write_buffer: size={}B capacity={}B",
            size,
            capacity
        );
        let buffer_changed = self.reserve(capacity, device);
        if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, self.values.as_slice());
            self.is_stale = false;
        }
        buffer_changed
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        if !self.values.is_empty() {
            self.is_stale = true;
        }
        self.values.clear();
    }

    /// Allocate storage for a single property struct with the given layout.
    ///
    /// # Returns
    ///
    /// On success, this returns the index in the GPU buffer where the struct
    /// was allocated. On failure, this returns `None`.
    #[inline]
    pub fn allocate(&mut self, layout: &PropertyLayout) -> Option<u32> {
        // Note: allocate with min_binding_size() and not cpu_size(), because the buffer
        // needs to be large enough to host at least one struct when bound to a shader,
        // and in WGSL the struct is padded to its align size.
        let size = layout.min_binding_size().get() as u32;

        // For now, we expand a single buffer infinitely. TODO to add a limit...
        let offset = self.alloc_aligned(size);
        assert!(offset.is_multiple_of(size)); // by design of alloc_aligned()
        Some(offset / size) // array index
    }

    #[inline]
    pub fn free(&mut self, offset: u32, size: u32) -> SlabState {
        let id = self
            .buffer
            .as_ref()
            .map(|buf| {
                let id: NonZeroU32 = buf.id().into();
                id.get()
            })
            .unwrap_or(u32::MAX);
        let buffer_size = self.len();
        if self.remove(offset..(offset + size)) {
            if self.is_empty() {
                SlabState::Free
            } else if self.len() != buffer_size
                || self
                    .buffer
                    .as_ref()
                    .map(|buf| {
                        let id: NonZeroU32 = buf.id().into();
                        id.get()
                    })
                    .unwrap_or(u32::MAX)
                    != id
            {
                SlabState::Resized
            } else {
                SlabState::Used
            }
        } else {
            SlabState::Used
        }
    }

    pub fn write(&mut self, offset: u32, data: &[u8]) {
        self.update_raw(offset, data);
    }
}

/// Cache for effect properties.
#[derive(Resource)]
pub struct PropertyCache {
    /// Collection of property buffers managed by this cache. Some buffers might
    /// be `None` if the entry is not used. Since the buffers are referenced
    /// by index, we cannot move them once they're allocated.
    buffers: Vec<Option<PropertyBuffer>>,
    /// Map from a binding size in bytes to its bind group layout. The binding
    /// size zero is valid, and corresponds to the variant without properties,
    /// which by abuse is stored here even though it's not related to properties
    /// (contains only the spawner binding).
    bind_group_layout_descs: HashMap<u32, BindGroupLayoutDescriptor>,
}

impl PropertyCache {
    pub fn new(device: &RenderDevice) -> Self {
        let align = device.limits().min_storage_buffer_offset_alignment;
        let spawner_min_binding_size = GpuSpawnerParams::aligned_size(align);

        // Create the default bind group layout when no properties are present
        let bgl = BindGroupLayoutDescriptor::new(
            "hanabi:bgl:no_property",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE | ShaderStages::VERTEX,
                (
                    // @group(2) @binding(0) var<storage, read> spawners : array<Spawner>;
                    //storage_buffer_read_only::<GpuSpawnerParams>(false), // TODO - vfx_sort_fill
                    // still needs align
                    storage_buffer_read_only_sized(false, Some(spawner_min_binding_size)),
                    // @group(2) @binding(1) var<storage, read> prefix_sum : array<u32>;
                    storage_buffer_read_only::<u32>(false),
                    // @group(2) @binding(2) var<storage, read> batch_info : BatchInfo;
                    storage_buffer_read_only_sized(true, Some(GpuBatchInfo::aligned_size(align))),
                ),
            ),
        );
        trace!(
            "-> created bind group layout desc for no-property variant: {:?}",
            bgl
        );
        let mut bind_group_layout_descs = HashMap::with_capacity_and_hasher(1, Default::default());
        bind_group_layout_descs.insert(0, bgl);

        Self {
            buffers: vec![],
            bind_group_layout_descs,
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub fn buffers(&self) -> &[Option<PropertyBuffer>] {
        &self.buffers
    }

    #[allow(dead_code)]
    #[inline]
    pub fn buffers_mut(&mut self) -> &mut [Option<PropertyBuffer>] {
        &mut self.buffers
    }

    pub fn bind_group_layout_desc(
        &self,
        min_binding_size: Option<NonZeroU64>,
    ) -> Option<&BindGroupLayoutDescriptor> {
        let key = min_binding_size.map(NonZeroU64::get).unwrap_or(0) as u32;
        self.bind_group_layout_descs.get(&key)
    }

    pub fn allocate(&mut self, property_layout: &PropertyLayout) -> CachedEffectProperties {
        assert!(!property_layout.is_empty());

        // Use the no-property layout descriptor as the base, that we'll customize. It's
        // always available. But since it's stored inside the hash map, we need to
        // eagerly clone in advance, for the borrow checker.
        let mut bgl = self.bind_group_layout_descs.get(&0).unwrap().clone();

        // Ensure there's a bind group layout for the property variant with that binding
        // size.
        let properties_min_binding_size = property_layout.min_binding_size();
        self.bind_group_layout_descs
            .entry(properties_min_binding_size.get() as u32)
            .or_insert_with(|| {
                let label = format!(
                    "hanabi:bgl:property_size{}",
                    properties_min_binding_size.get()
                );
                trace!(
                    "Create new property bind group layout '{}' for binding size {} bytes.",
                    label,
                    properties_min_binding_size.get()
                );
                // Append the Properties array binding
                bgl.label = label.into();
                bgl.entries.push(
                    storage_buffer_read_only_sized(false, Some(properties_min_binding_size))
                        .build(3, ShaderStages::COMPUTE | ShaderStages::VERTEX),
                );
                trace!(
                    "-> created bind group layout desc for size {}: {:?}",
                    properties_min_binding_size.get(),
                    bgl
                );
                bgl
            });

        self.buffers
            .iter_mut()
            .enumerate()
            .find_map(|(buffer_index, buffer)| {
                let buffer = buffer.as_mut()?;
                let array_index= buffer.allocate(property_layout)?;
                trace!("Allocated new slice in property buffer #{buffer_index} for layout {property_layout:?}: array_index={array_index}");
                Some(CachedEffectProperties {
                    property_layout: property_layout.clone(),
                    buffer_index: buffer_index as u32,
                    array_index,
                })
            })
            .unwrap_or_else(|| {
                // Cannot find any suitable buffer; allocate a new one
                let buffer_index = self
                    .buffers
                    .iter()
                    .position(|buf| buf.is_none())
                    .unwrap_or(self.buffers.len());
                let label = format!("hanabi:buffer:properties{buffer_index}");
                trace!("Creating new property buffer #{buffer_index} '{label}'");
                let mut buffer = PropertyBuffer::new(Some(label), BufferUsages::STORAGE);
                // FIXME - Currently PropertyBuffer::allocate() always succeeds and grows
                // indefinitely
                let array_index = buffer.allocate(property_layout).unwrap();
                if buffer_index >= self.buffers.len() {
                    self.buffers.push(Some(buffer));
                } else {
                    debug_assert!(self.buffers[buffer_index].is_none());
                    self.buffers[buffer_index] = Some(buffer);
                }
                CachedEffectProperties {
                    property_layout: property_layout.clone(),
                    buffer_index: buffer_index as u32,
                    array_index,
                }
            })
    }

    pub fn get_buffer(&self, buffer_index: u32) -> Option<&Buffer> {
        self.buffers
            .get(buffer_index as usize)
            .and_then(|opt_pb| opt_pb.as_ref().map(|pb| pb.buffer()))
            .flatten()
    }

    /// Deallocated and remove properties from the cache.
    pub fn free(
        &mut self,
        cached_effect_properties: &CachedEffectProperties,
    ) -> Result<SlabState, CachedPropertiesError> {
        trace!(
            "Removing cached properties {:?} from cache.",
            cached_effect_properties
        );
        let entry = self
            .buffers
            .get_mut(cached_effect_properties.buffer_index as usize)
            .ok_or(CachedPropertiesError::InvalidBufferIndex(
                cached_effect_properties.buffer_index,
            ))?;
        let buffer = entry
            .as_mut()
            .ok_or(CachedPropertiesError::BufferDeallocated(
                cached_effect_properties.buffer_index,
            ))?;
        let size = cached_effect_properties
            .property_layout
            .min_binding_size()
            .get() as u32;
        let offset = cached_effect_properties.array_index * size;
        Ok(buffer.free(offset, size))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PropertyBindGroupKey {
    pub buffer_index: u32,
    pub binding_size: u32,
}

#[derive(Default, Resource)]
pub struct PropertyBindGroups {
    /// Map from a [`PropertyBuffer`] index and a binding size to the
    /// corresponding bind group.
    property_bind_groups: HashMap<PropertyBindGroupKey, BindGroup>,
    /// Bind group for the variant without any property.
    no_property_bind_group: Option<BindGroup>,
}

impl PropertyBindGroups {
    /// Clear all bind groups.
    ///
    /// If `with_no_property` is `true`, also clear the no-property bind group,
    /// which doesn't depend on any property buffer.
    pub fn clear(&mut self, with_no_property: bool) {
        self.property_bind_groups.clear();
        if with_no_property {
            self.no_property_bind_group = None;
        }
    }

    /// Ensure the bind group for the given key exists, creating it if needed.
    pub fn ensure_exists(
        &mut self,
        property_key: &PropertyBindGroupKey,
        property_cache: &PropertyCache,
        spawner_buffer: &Buffer,
        prefix_sum_buffer: &Buffer,
        batch_info_buffer: &Buffer,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
    ) -> Result<(), ()> {
        let Some(property_buffer) = property_cache.get_buffer(property_key.buffer_index) else {
            error!(
                "Missing property buffer #{}, referenced by effect batch.",
                property_key.buffer_index,
            );
            return Err(());
        };

        // This should always be non-zero if the property key is Some().
        let property_binding_size = NonZeroU64::new(property_key.binding_size as u64).unwrap();
        let Some(layout_desc) = property_cache.bind_group_layout_desc(Some(property_binding_size))
        else {
            error!(
                "Missing property bind group layout for binding size {}, referenced by effect batch.",
                property_binding_size.get(),
            );
            return Err(());
        };

        let align = render_device.limits().min_storage_buffer_offset_alignment;
        self.property_bind_groups
            .entry(*property_key)
            .or_insert_with(|| {
                trace!(
                    "Creating new spawner@2 bind group for property buffer #{} and binding size {}",
                    property_key.buffer_index,
                    property_key.binding_size
                );
                render_device.create_bind_group(
                    Some(
                        &format!(
                            "hanabi:bg:spawner@2:property{}_size{}",
                            property_key.buffer_index, property_key.binding_size
                        )[..],
                    ),
                    &pipeline_cache.get_bind_group_layout(layout_desc),
                    &BindGroupEntries::sequential((
                        spawner_buffer.as_entire_binding(),
                        prefix_sum_buffer.as_entire_binding(),
                        BufferBinding {
                            buffer: batch_info_buffer,
                            offset: 0,
                            size: Some(GpuBatchInfo::aligned_size(align)),
                        },
                        property_buffer.as_entire_binding(),
                    )),
                )
            });
        Ok(())
    }

    /// Ensure the bind group for the given key exists, creating it if needed.
    pub fn ensure_exists_no_property(
        &mut self,
        property_cache: &PropertyCache,
        spawner_buffer: &Buffer,
        prefix_sum_buffer: &Buffer,
        batch_info_buffer: &Buffer,
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
    ) -> Result<(), ()> {
        let Some(layout_desc) = property_cache.bind_group_layout_desc(None) else {
            error!(
                "Missing property bind group layout for no-property variant, referenced by effect batch.",
            );
            return Err(());
        };

        if self.no_property_bind_group.is_none() {
            let align = render_device.limits().min_storage_buffer_offset_alignment;
            trace!("Creating new spawner@2 bind group for no-property variant");
            self.no_property_bind_group = Some(render_device.create_bind_group(
                Some("hanabi:bg:spawner@2:no-property"),
                &pipeline_cache.get_bind_group_layout(layout_desc),
                &BindGroupEntries::sequential((
                    spawner_buffer.as_entire_binding(),
                    prefix_sum_buffer.as_entire_binding(),
                    BufferBinding {
                        buffer: batch_info_buffer,
                        offset: 0,
                        size: Some(GpuBatchInfo::aligned_size(align)),
                    },
                )),
            ));
        }

        Ok(())
    }

    /// Get the bind group for the given key.
    pub fn get(&self, key: Option<&PropertyBindGroupKey>) -> Option<&BindGroup> {
        if let Some(key) = key {
            self.property_bind_groups.get(key)
        } else {
            self.no_property_bind_group.as_ref()
        }
    }
}

fn upload_properties(
    extracted_properties: &ExtractedProperties,
    cached_effect_properties: &CachedEffectProperties,
    mut property_cache: Mut<'_, PropertyCache>,
) {
    if let Some(property_data) = &extracted_properties.property_data {
        let size = cached_effect_properties
            .property_layout
            .min_binding_size()
            .get() as u32;
        trace!(
            "Properties changed; (re-)uploading to GPU... New data: {} bytes. Capacity: {} bytes.",
            property_data.len(),
            size,
        );
        if property_data.len() <= size as usize {
            let property_buffer = property_cache.buffers_mut()
                [cached_effect_properties.buffer_index as usize]
                .as_mut()
                .unwrap();
            property_buffer.write(cached_effect_properties.array_index * size, property_data);
        } else {
            error!(
                "Cannot upload properties: existing property slice in property buffer #{} is too small ({} bytes) for the new data ({} bytes).",
                cached_effect_properties.buffer_index,
                size,
                property_data.len()
            );
        }
    }
}

/// Allocate GPU resources to store an effect's properties.
///
/// This insert or updates the [`CachedEffectProperties`] component with the
/// allocation details.
pub(crate) fn allocate_properties(
    mut commands: Commands,
    mut property_cache: ResMut<PropertyCache>,
    mut q_effects: Query<(
        Entity,
        &ExtractedProperties,
        Option<&mut CachedEffectProperties>,
    )>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("allocate_properties").entered();
    trace!("allocate_properties");

    for (entity, extracted_properties, maybe_cached_effect_properties) in &mut q_effects {
        if let Some(mut cached_effect_properties) = maybe_cached_effect_properties {
            // Note: Technically we should compare the entire layout, but in practice the
            // allocation only cares about the size of the layout to store all properties,
            // and not about the actual layout itself (where each property is). So to reduce
            // updates and maximize the chance of reuse, compare the allocation size only.
            if cached_effect_properties.property_layout.min_binding_size()
                != extracted_properties.property_layout.min_binding_size()
            {
                // Free the old allocations. We're sure there's one because the component
                // exists.
                // TODO - handle SlabState return value to invalidate property bind groups!!
                if let Err(err) = property_cache.free(cached_effect_properties.as_ref()) {
                    error!("Error while freeing cached properties for effect {entity:?}: {err:?}");
                }

                // If the layout is not empty, which means we need to store at least one
                // property, then allocate a new storate. Otherwise remove the component.
                if extracted_properties.property_layout.is_empty() {
                    commands.entity(entity).remove::<CachedEffectProperties>();
                } else {
                    *cached_effect_properties =
                        property_cache.allocate(&extracted_properties.property_layout);
                }
            }

            // Re-upload new properties
            if !extracted_properties.property_layout.is_empty() {
                debug_assert_eq!(
                    extracted_properties.property_layout,
                    cached_effect_properties.property_layout
                );
                upload_properties(
                    extracted_properties,
                    cached_effect_properties.as_ref(),
                    property_cache.reborrow(),
                );
            }
        } else {
            let cached_effect_properties =
                property_cache.allocate(&extracted_properties.property_layout);
            trace!("First-time properties, allocated a new CachedEffectProperties : {cached_effect_properties:?}");
            upload_properties(
                extracted_properties,
                &cached_effect_properties,
                property_cache.reborrow(),
            );
            commands.entity(entity).insert(cached_effect_properties);
        }
    }
}

/// Observer raised when the [`CachedEffectProperties`] component is removed,
/// which indicates that the effect doesn't use properties anymore (including,
/// when the effect itself is despawned).
pub(crate) fn on_remove_cached_properties(
    trigger: On<Remove, CachedEffectProperties>,
    query: Query<(Entity, &CachedEffectProperties)>,
    mut property_cache: ResMut<PropertyCache>,
    mut property_bind_groups: ResMut<PropertyBindGroups>,
) {
    // FIXME - review this Observer pattern; this triggers for each event one by
    // one, which could kill performance if many effects are removed.

    let Ok((render_entity, cached_effect_properties)) = query.get(trigger.event().entity) else {
        return;
    };

    match property_cache.free(cached_effect_properties) {
        Err(err) => match err {
            CachedPropertiesError::InvalidBufferIndex(buffer_index)
                => error!("Failed to remove cached properties of render entity {render_entity:?} from buffer #{buffer_index}: the index is invalid."),
            CachedPropertiesError::BufferDeallocated(buffer_index)
                => error!("Failed to remove cached properties of render entity {render_entity:?} from buffer #{buffer_index}: the buffer is not allocated."),
        }
        Ok(buffer_state) => if buffer_state != SlabState::Used {
            // The entire buffer was deallocated, or it was resized; destroy all bind groups referencing it
            let key = cached_effect_properties.to_key();
            trace!("Destroying property bind group for key {key:?} due to property buffer deallocated.");
            property_bind_groups
                .property_bind_groups
                .retain(|&k, _| k.buffer_index != key.buffer_index);
        }
    }
}

/// Prepare GPU buffers storing effect properties.
pub(crate) fn prepare_property_buffers(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut property_cache: ResMut<PropertyCache>,
    mut bind_groups: ResMut<PropertyBindGroups>,
) {
    // Allocate all the property buffer(s) as needed, before we move to the next
    // step which will need those buffers to schedule data copies from CPU.
    for (buffer_index, buffer_slot) in property_cache.buffers_mut().iter_mut().enumerate() {
        let Some(property_buffer) = buffer_slot.as_mut() else {
            continue;
        };
        let changed = property_buffer.write_buffer(&render_device, &render_queue);
        if changed {
            trace!("Destroying all bind groups for property buffer #{buffer_index}");
            bind_groups
                .property_bind_groups
                .retain(|&k, _| k.buffer_index != buffer_index as u32);
        }
    }
}

#[cfg(all(test, feature = "gpu_tests"))]
mod gpu_tests {
    use bevy::math::Vec3;
    use wgpu::BufferView;

    use super::*;
    use crate::{test_utils::MockRenderer, Property};

    fn submit_and_wait(device: &RenderDevice, queue: &RenderQueue) {
        // Create a dummy CommandBuffer to force the write_buffer() call to have any
        // effect.
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test"),
        });
        let command_buffer = encoder.finish();

        queue.submit([command_buffer]);
        let (tx, rx) = futures::channel::oneshot::channel();
        queue.on_submitted_work_done(move || {
            tx.send(()).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let _ = futures::executor::block_on(rx);
    }

    fn gpu_read_back_and_wait(device: &RenderDevice, buffer: &Buffer) -> BufferView {
        let buffer = buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        futures::executor::block_on(rx).unwrap().unwrap();
        buffer.get_mapped_range()
    }

    #[test]
    fn pb_alloc_free() {
        let mut pb = PropertyBuffer::new(None, BufferUsages::STORAGE | BufferUsages::MAP_READ);

        let layout16 = PropertyLayout::new(&[
            Property::new("my_property", Vec3::new(1., 4., 9.)),
            Property::new("other_property", 3.4_f32),
        ]);
        let size16 = layout16.min_binding_size().get() as u32;
        let layout4 = PropertyLayout::new(&[Property::new("prop", 3.4_f32)]);
        let size4 = layout4.min_binding_size().get() as u32;

        // Alloc 16 + 4
        let index1 = pb.allocate(&layout16).unwrap();
        let index2 = pb.allocate(&layout4).unwrap();
        assert_eq!(index1, 0);
        assert_eq!(index2, size16 / size4);
        assert!(pb.free_ranges.is_empty());

        // [1111][2]

        // Alloc 16; makes a gap
        let index3 = pb.allocate(&layout16).unwrap();
        assert_eq!(index3, (size16 + size4).div_ceil(size16));
        assert_eq!(pb.free_ranges.len(), 1);

        // [1111][2---][3333]

        // Free first; buffer stay in full use (with a gap)
        let slab_state = pb.free(index1 * size16, size16);
        assert_eq!(slab_state, SlabState::Used);

        // [----][2---][3333]

        // Alloc 4 (x2); fit inside start of buffer (16B), but gap after #2 is smaller
        // (12B)
        let index4 = pb.allocate(&layout4).unwrap();
        let index5 = pb.allocate(&layout4).unwrap();
        let end2 = (index2 + 1) * size4;
        assert_eq!(index4, end2.div_ceil(size4));
        assert_eq!(index5, index4 + 1);

        // [----][245-][3333]

        // Free #2 and #4; should merge with start
        assert_eq!(pb.free(index2 * size4, size4), SlabState::Used);
        assert_eq!(pb.free(index4 * size4, size4), SlabState::Used);
        assert_eq!(pb.free_ranges.len(), 2);

        // [----][--5-][3333]

        // Free #5; merge 2 free ranges
        assert_eq!(pb.free(index5 * size4, size4), SlabState::Used);
        assert_eq!(pb.free_ranges.len(), 1);

        // [----][----][3333]

        // Free last
        assert_eq!(pb.free(index3 * size16, size16), SlabState::Free);
        assert!(pb.free_ranges.is_empty());
        assert!(pb.is_empty());

        // []

        // Alloc 16 + 16
        let index6 = pb.allocate(&layout16).unwrap();
        let index7 = pb.allocate(&layout16).unwrap();
        assert_eq!(index6, 0);
        assert_eq!(index7, 1);
        assert!(pb.free_ranges.is_empty());

        // [6666][7777]

        // Free #75; resize
        assert_eq!(pb.free(index7 * size16, size16), SlabState::Resized);
        assert!(pb.free_ranges.is_empty());

        // [6666]
    }

    #[test]
    fn pb_write() {
        let renderer = MockRenderer::new();
        let device = renderer.device();
        let queue = renderer.queue();

        let mut pb = PropertyBuffer::new(None, BufferUsages::STORAGE | BufferUsages::MAP_READ);
        assert!(pb.is_empty());
        assert_eq!(pb.len(), 0);
        assert!(pb.buffer().is_none());
        assert_eq!(pb.capacity(), 0);

        let layout1 = PropertyLayout::new(&[
            Property::new("my_property", Vec3::new(1., 4., 9.)),
            Property::new("other_property", 3.4_f32),
        ]);
        assert_eq!(layout1.min_binding_size().get(), 16);
        let offset1 = pb.allocate(&layout1).unwrap();
        assert_eq!(offset1, 0);
        assert!(!pb.is_empty());
        assert_eq!(pb.len(), 16);
        // GPU buffer not yet allocated
        assert!(pb.buffer().is_none());
        assert_eq!(pb.capacity(), 0);

        let layout2 = PropertyLayout::new(&[Property::new("prop", 3.4_f32)]);
        let offset2 = pb.allocate(&layout2).unwrap();
        assert_eq!(offset2, 4); // 16 B / size == 4
        assert!(!pb.is_empty());
        assert_eq!(pb.len(), 20);
        // GPU buffer not yet allocated
        assert!(pb.buffer().is_none());
        assert_eq!(pb.capacity(), 0);

        pb.write(offset2 * 4, bytemuck::cast_slice(&[24.99f32; 1]));
        pb.write(
            offset1 * 16,
            bytemuck::cast_slice(&[55.2f32, -32.1f32, 99.07f32, 34.5f32]),
        );
        // GPU buffer not yet allocated
        assert!(pb.buffer().is_none());
        assert_eq!(pb.capacity(), 0);

        let buffer_changed = pb.write_buffer(&device, &queue);
        // GPU buffer now allocated
        assert!(buffer_changed);
        assert!(pb.buffer().is_some());
        assert!(pb.capacity() >= 20);

        submit_and_wait(&device, &queue);
        println!("Buffer written");

        // Read back (GPU -> CPU)
        let buffer = pb.buffer().unwrap();
        let view = gpu_read_back_and_wait(&device, buffer);

        // Validate content
        assert!(view.len() >= 20);
        let v: &[f32] = bytemuck::cast_slice(&view[..256]);
        assert_eq!(v[0], 55.2f32);
        assert_eq!(v[1], -32.1f32);
        assert_eq!(v[2], 99.07f32);
        assert_eq!(v[3], 34.5f32);
        assert_eq!(v[4], 24.99f32);
        drop(view);
        buffer.unmap();

        pb.write(offset1 * 16, bytemuck::cast_slice(&[0f32, 1f32, 2f32]));
        // GPU buffer still allocated
        assert!(pb.buffer().is_some());
        assert!(pb.capacity() >= 20);

        let buffer_changed = pb.write_buffer(&device, &queue);
        // GPU buffer NOT re-allocated (only content changed)
        assert!(!buffer_changed);
        assert!(pb.buffer().is_some());
        assert!(pb.capacity() >= 20);

        submit_and_wait(&device, &queue);
        println!("Buffer written");

        // Read back (GPU -> CPU)
        let buffer = pb.buffer().unwrap();
        let view = gpu_read_back_and_wait(&device, buffer);

        // Validate content
        assert!(view.len() >= 20);
        let v: &[f32] = bytemuck::cast_slice(&view[..256]);
        assert_eq!(v[0], 0f32);
        assert_eq!(v[1], 1f32);
        assert_eq!(v[2], 2f32);
        assert_eq!(v[3], 34.5f32);
        assert_eq!(v[4], 24.99f32);
        drop(view);
        buffer.unmap();
    }
}
