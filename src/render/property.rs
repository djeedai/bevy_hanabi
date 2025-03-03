use std::{
    num::{NonZeroU32, NonZeroU64},
    ops::Range,
};

use bevy::{
    log::{error, trace},
    prelude::{Component, Entity, OnRemove, Query, Res, ResMut, Resource, Trigger},
    render::{
        render_resource::{BindGroup, BindGroupLayout, Buffer},
        renderer::{RenderDevice, RenderQueue},
    },
    utils::HashMap,
};
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType, BufferBinding,
    BufferBindingType, BufferUsages, ShaderStages,
};

use super::{aligned_buffer_vec::HybridAlignedBufferVec, effect_cache::BufferState};
use crate::{
    render::{GpuSpawnerParams, StorageType},
    PropertyLayout,
};

/// Allocation into the [`PropertyCache`] for an effect instance. This component
/// is only present on an effect instance if that effect uses properties.
#[derive(Debug, Clone, PartialEq, Eq, Component)]
pub struct CachedEffectProperties {
    /// Index of the [`PropertyBuffer`] inside the [`PropertyCache`].
    pub buffer_index: u32,
    /// Slice of GPU buffer where the storage for properties is allocated.
    pub range: Range<u32>,
}

impl CachedEffectProperties {
    /// Convert this allocation into a bind group key, used for bind group
    /// re-creation when a change is detected in the key.
    pub fn to_key(&self) -> PropertyBindGroupKey {
        PropertyBindGroupKey {
            buffer_index: self.buffer_index,
            binding_size: self.range.len() as u32,
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

#[derive(Debug)]
pub(crate) struct PropertyBuffer {
    /// GPU buffer holding the properties of some effect(s).
    buffer: HybridAlignedBufferVec,
    // Layout of properties of the effect(s), if using properties.
    //property_layout: PropertyLayout,
}

impl PropertyBuffer {
    pub fn new(align: u32, label: Option<String>) -> Self {
        let align = NonZeroU64::new(align as u64).unwrap();
        let label = label.unwrap_or("hanabi:buffer:properties".to_string());
        Self {
            buffer: HybridAlignedBufferVec::new(BufferUsages::STORAGE, Some(align), Some(label)),
        }
    }

    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.buffer()
    }

    #[inline]
    pub fn allocate(&mut self, layout: &PropertyLayout) -> Range<u32> {
        // Note: allocate with min_binding_size() and not cpu_size(), because the buffer
        // needs to be large enough to host at least one struct when bound to a shader,
        // and in WGSL the struct is padded to its align size.
        let size = layout.min_binding_size().get() as usize;
        // FIXME - allocate(size) instead of push(data) so we don't need to allocate an
        // empty vector just to read its size.
        self.buffer.push_raw(&vec![0u8; size][..])
    }

    #[allow(dead_code)]
    #[inline]
    pub fn free(&mut self, range: Range<u32>) -> BufferState {
        let id = self
            .buffer
            .buffer()
            .map(|buf| {
                let id: NonZeroU32 = buf.id().into();
                id.get()
            })
            .unwrap_or(u32::MAX);
        let size = self.buffer.len();
        if self.buffer.remove(range) {
            if self.buffer.is_empty() {
                BufferState::Free
            } else if self.buffer.len() != size
                || self
                    .buffer
                    .buffer()
                    .map(|buf| {
                        let id: NonZeroU32 = buf.id().into();
                        id.get()
                    })
                    .unwrap_or(u32::MAX)
                    != id
            {
                BufferState::Resized
            } else {
                BufferState::Used
            }
        } else {
            BufferState::Used
        }
    }

    pub fn write(&mut self, offset: u32, data: &[u8]) {
        self.buffer.update(offset, data);
    }

    #[inline]
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) -> bool {
        self.buffer.write_buffer(device, queue)
    }
}

/// Cache for effect properties.
#[derive(Resource)]
pub struct PropertyCache {
    /// Render device to allocate GPU buffers and bind group layouts as needed.
    device: RenderDevice,
    /// Collection of property buffers managed by this cache. Some buffers might
    /// be `None` if the entry is not used. Since the buffers are referenced
    /// by index, we cannot move them once they're allocated.
    buffers: Vec<Option<PropertyBuffer>>,
    /// Map from a binding size in bytes to its bind group layout. The binding
    /// size zero is valid, and corresponds to the variant without properties,
    /// which by abuse is stored here even though it's not related to properties
    /// (contains only the spawner binding).
    bind_group_layouts: HashMap<u32, BindGroupLayout>,
}

impl PropertyCache {
    pub fn new(device: RenderDevice) -> Self {
        let spawner_min_binding_size =
            GpuSpawnerParams::aligned_size(device.limits().min_storage_buffer_offset_alignment);
        let bgl = device.create_bind_group_layout(
            "hanabi:bind_group_layout:no_property",
            // @group(2) @binding(0) var<storage, read> spawner: Spawner;
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: Some(spawner_min_binding_size),
                },
                count: None,
            }],
        );
        trace!(
            "-> created bind group layout #{:?} for no-property variant",
            bgl.id()
        );
        let mut bind_group_layouts = HashMap::with_capacity(1);
        bind_group_layouts.insert(0, bgl);

        Self {
            device,
            buffers: vec![],
            bind_group_layouts,
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

    pub fn bind_group_layout(
        &self,
        min_binding_size: Option<NonZeroU64>,
    ) -> Option<&BindGroupLayout> {
        let key = min_binding_size.map(NonZeroU64::get).unwrap_or(0) as u32;
        self.bind_group_layouts.get(&key)
    }

    pub fn insert(&mut self, property_layout: &PropertyLayout) -> CachedEffectProperties {
        assert!(!property_layout.is_empty());

        // Ensure there's a bind group layout for the property variant with that binding
        // size
        let properties_min_binding_size = property_layout.min_binding_size();
        let spawner_min_binding_size = GpuSpawnerParams::aligned_size(
            self.device.limits().min_storage_buffer_offset_alignment,
        );
        self.bind_group_layouts
            .entry(properties_min_binding_size.get() as u32)
            .or_insert_with(|| {
                let label = format!(
                    "hanabi:bind_group_layout:property_size{}",
                    properties_min_binding_size.get()
                );
                trace!(
                    "Create new property bind group layout '{}' for binding size {} bytes.",
                    label,
                    properties_min_binding_size.get()
                );
                let bgl = self.device.create_bind_group_layout(
                    Some(&label[..]),
                    &[
                        // @group(2) @binding(0) var<storage, read> spawner: Spawner;
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: true,
                                min_binding_size: Some(spawner_min_binding_size),
                            },
                            count: None,
                        },
                        // @group(2) @binding(1) var<storage, read> properties : Properties;
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: true,
                                min_binding_size: Some(properties_min_binding_size),
                            },
                            count: None,
                        },
                    ],
                );
                trace!("-> created bind group layout #{:?}", bgl.id());
                bgl
            });

        self.buffers
            .iter_mut()
            .enumerate()
            .find_map(|(buffer_index, buffer)| {
                if let Some(buffer) = buffer {
                    // Try to allocate a slice into the buffer
                    // FIXME - Currently PropertyBuffer::allocate() always succeeds and
                    // grows indefinitely
                    let range = buffer.allocate(property_layout);
                    trace!("Allocate new slice in property buffer #{buffer_index} for layout {property_layout:?}: range={range:?}");
                    Some(CachedEffectProperties {
                        buffer_index: buffer_index as u32,
                        range,
                    })
                } else {
                    None
                }
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
                let align = self.device.limits().min_storage_buffer_offset_alignment;
                let mut buffer = PropertyBuffer::new(align, Some(label));
                // FIXME - Currently PropertyBuffer::allocate() always succeeds and grows
                // indefinitely
                let range = buffer.allocate(property_layout);
                if buffer_index >= self.buffers.len() {
                    self.buffers.push(Some(buffer));
                } else {
                    debug_assert!(self.buffers[buffer_index].is_none());
                    self.buffers[buffer_index] = Some(buffer);
                }
                CachedEffectProperties {
                    buffer_index: buffer_index as u32,
                    range,
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
    pub fn remove_properties(
        &mut self,
        cached_effect_properties: &CachedEffectProperties,
    ) -> Result<BufferState, CachedPropertiesError> {
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
        Ok(buffer.free(cached_effect_properties.range.clone()))
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
        spawner_buffer_binding_size: NonZeroU64,
        render_device: &RenderDevice,
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
        let Some(layout) = property_cache.bind_group_layout(Some(property_binding_size)) else {
            error!(
                "Missing property bind group layout for binding size {}, referenced by effect batch.",
                property_binding_size.get(),
            );
            return Err(());
        };

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
                            "hanabi:bind_group:spawner@2:property{}_size{}",
                            property_key.buffer_index, property_key.binding_size
                        )[..],
                    ),
                    layout,
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: spawner_buffer,
                                offset: 0,
                                size: Some(spawner_buffer_binding_size),
                            }),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: property_buffer,
                                offset: 0,
                                size: Some(property_binding_size),
                            }),
                        },
                    ],
                )
            });
        Ok(())
    }

    /// Ensure the bind group for the given key exists, creating it if needed.
    pub fn ensure_exists_no_property(
        &mut self,
        property_cache: &PropertyCache,
        spawner_buffer: &Buffer,
        spawner_buffer_binding_size: NonZeroU64,
        render_device: &RenderDevice,
    ) -> Result<(), ()> {
        let Some(layout) = property_cache.bind_group_layout(None) else {
            error!(
                "Missing property bind group layout for no-property variant, referenced by effect batch.",
            );
            return Err(());
        };

        if self.no_property_bind_group.is_none() {
            trace!("Creating new spawner@2 bind group for no-property variant");
            self.no_property_bind_group = Some(render_device.create_bind_group(
                Some("hanabi:bind_group:spawner@2:no-property"),
                layout,
                &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: spawner_buffer,
                        offset: 0,
                        size: Some(spawner_buffer_binding_size),
                    }),
                }],
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

/// Observer raised when the [`CachedEffectProperties`] component is removed,
/// which indicates that the effect doesn't use properties anymore (including,
/// when the effect itself is despawned).
pub(crate) fn on_remove_cached_properties(
    trigger: Trigger<OnRemove, CachedEffectProperties>,
    query: Query<(Entity, &CachedEffectProperties)>,
    mut property_cache: ResMut<PropertyCache>,
    mut property_bind_groups: ResMut<PropertyBindGroups>,
) {
    // FIXME - review this Observer pattern; this triggers for each event one by
    // one, which could kill performance if many effects are removed.

    let Ok((render_entity, cached_effect_properties)) = query.get(trigger.entity()) else {
        return;
    };

    match property_cache.remove_properties(cached_effect_properties) {
        Err(err) => match err {
            CachedPropertiesError::InvalidBufferIndex(buffer_index)
                => error!("Failed to remove cached properties of render entity {render_entity:?} from buffer #{buffer_index}: the index is invalid."),
            CachedPropertiesError::BufferDeallocated(buffer_index)
                => error!("Failed to remove cached properties of render entity {render_entity:?} from buffer #{buffer_index}: the buffer is not allocated."),
        }
        Ok(buffer_state) => if buffer_state != BufferState::Used {
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
///
/// This system runs after the new effects have been registered by
/// [`add_effects()`], and all effects using properties are known for this
/// frame. It (re-)allocate any property buffer, and schedule buffer writes to
/// them, in anticipation of [`prepare_bind_groups()`] referencing those buffers
/// to create bind groups.
pub(crate) fn prepare_property_buffers(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut cache: ResMut<PropertyCache>,
    mut bind_groups: ResMut<PropertyBindGroups>,
) {
    // Allocate all the property buffer(s) as needed, before we move to the next
    // step which will need those buffers to schedule data copies from CPU.
    for (buffer_index, buffer_slot) in cache.buffers_mut().iter_mut().enumerate() {
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
