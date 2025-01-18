use std::{
    num::{NonZeroU32, NonZeroU64},
    ops::Range,
};

use bevy::{
    log::trace,
    prelude::{Component, Resource},
    render::{
        render_resource::{BindGroupLayout, Buffer, ShaderType as _},
        renderer::{RenderDevice, RenderQueue},
    },
    utils::HashMap,
};
use wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, ShaderStages};

use super::{
    aligned_buffer_vec::HybridAlignedBufferVec, effect_cache::BufferState, PropertyBindGroupKey,
};
use crate::{render::GpuSpawnerParams, PropertyLayout};

#[derive(Debug, Clone, PartialEq, Eq, Component)]
pub struct CachedEffectProperties {
    pub buffer_index: u32,
    pub range: Range<u32>,
}

impl CachedEffectProperties {
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
        let bgl = device.create_bind_group_layout(
            "hanabi:bind_group_layout:no_property",
            // @group(2) @binding(0) var<storage, read> spawner: Spawner;
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: Some(GpuSpawnerParams::min_size()),
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
        let min_binding_size = property_layout.min_binding_size();
        self.bind_group_layouts
            .entry(min_binding_size.get() as u32)
            .or_insert_with(|| {
                let label = format!(
                    "hanabi:bind_group_layout:property_size{}",
                    min_binding_size.get()
                );
                trace!(
                    "Create new property bind group layout '{}' for binding size {} bytes.",
                    label,
                    min_binding_size.get()
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
                                min_binding_size: Some(GpuSpawnerParams::min_size()),
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
                                min_binding_size: Some(min_binding_size),
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
        self.buffers[buffer_index as usize]
            .as_ref()
            .and_then(|pb| pb.buffer())
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
