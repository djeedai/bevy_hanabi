use rand::Rng;
use std::{borrow::Cow, cmp::Ordering, num::NonZeroU64, ops::Range};

use crate::{asset::EffectAsset, render::Particle, ParticleEffect};

use bevy::{
    asset::{AssetEvent, Assets, Handle, HandleUntyped},
    core::{cast_slice, FloatOrd, Pod, Time, Zeroable},
    core_pipeline::Transparent3d,
    ecs::{
        prelude::*,
        system::{lifetimeless::*, SystemState},
    },
    log::{trace, warn},
    math::{const_vec3, Mat4, Vec2, Vec3, Vec4Swizzles},
    reflect::TypeUuid,
    render::{
        color::Color,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_phase::{Draw, DrawFunctions, RenderPhase, TrackedRenderPass},
        render_resource::{std140::AsStd140, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::{BevyDefault, Image},
        view::{ComputedVisibility, ExtractedView, ViewUniform, ViewUniformOffset, ViewUniforms},
        RenderWorld,
    },
    sprite::Rect,
    transform::components::GlobalTransform,
    utils::{HashMap, HashSet},
};
use bytemuck::cast_slice_mut;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

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

pub struct SliceRef {
    range: Range<u32>,
    item_size: u32,
}

impl SliceRef {
    pub fn len(&self) -> u32 {
        self.range.end - self.range.start
    }

    pub fn byte_size(&self) -> usize {
        (self.len() as usize) * (self.item_size as usize)
    }
}

pub struct EffectBuffer {
    /// GPU buffer holding all particles for the entire group of effects.
    particle_buffer: Buffer,
    /// GPU buffer holding the indirection indices for the entire group of effects.
    indirect_buffer: Buffer,
    /// Size of each particle, in bytes.
    item_size: u32,
    /// Total buffer capacity in bytes.
    capacity: u32,
    /// Used buffer size, either from allocated slices or from slices in the free list.
    used_size: u32,
    /// Collection of slices into the buffer, each slice being one effect instance.
    slices: Vec<EffectSlice>,
    /// Array of free ranges for new allocations.
    free_slices: Vec<Range<u32>>,
    /// Map of entities and slices.
    slice_from_entity: HashMap<Entity, usize>,
    /// Compute pipeline for the effect update pass.
    //pub compute_pipeline: ComputePipeline, // FIXME - ComputePipelineId, to avoid duplicating per instance!
    /// Handle of all effects common in this buffer. TODO - replace with compatible layout
    asset: Handle<EffectAsset>,
}

struct BestRange {
    range: Range<u32>,
    capacity: u32,
    index: usize,
}

impl EffectBuffer {
    /// Minimum buffer capacity to allocate, in number of particles.
    pub const MIN_CAPACITY: u32 = 65536; // at least 64k particles

    /// Create a new group and a GPU buffer to back it up.
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
        let indirect_label = if let Some(label) = label {
            format!("{}_indirect", label)
        } else {
            "vfx_indirect_buffer".to_owned()
        };
        let indirect_capacity_bytes: BufferAddress =
            capacity as u64 * std::mem::size_of::<u32>() as u64;
        let indirect_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some(&indirect_label),
            size: indirect_capacity_bytes,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        EffectBuffer {
            particle_buffer,
            indirect_buffer,
            item_size,
            capacity,
            used_size: 0,
            slices: vec![],
            free_slices: vec![],
            slice_from_entity: HashMap::default(),
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

    /// Return a binding for the entire buffer.
    pub fn max_binding(&self) -> BindingResource {
        let capacity_bytes = self.capacity as u64 * self.item_size as u64;
        BindingResource::Buffer(BufferBinding {
            buffer: &self.particle_buffer,
            offset: 0,
            size: Some(NonZeroU64::new(capacity_bytes).unwrap()),
        })
    }

    /// Return a binding of the buffer for a starting range of a given size (in bytes).
    pub fn binding(&self, size: u32) -> BindingResource {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.particle_buffer,
            offset: 0,
            size: Some(NonZeroU64::new(size as u64).unwrap()),
        })
    }

    /// Return a binding for the entire indirect buffer associated with the current effect buffer.
    pub fn indirect_max_binding(&self) -> BindingResource {
        let capacity_bytes = self.capacity as u64 * std::mem::size_of::<u32>() as u64;
        BindingResource::Buffer(BufferBinding {
            buffer: &self.indirect_buffer,
            offset: 0,
            size: Some(NonZeroU64::new(capacity_bytes).unwrap()),
        })
    }

    fn pop_free_slice(&mut self, size: u32) -> Option<Range<u32>> {
        if self.free_slices.is_empty() {
            return None;
        }
        let slice0 = &self.free_slices[0];
        let mut result = BestRange {
            range: slice0.clone(),
            capacity: (slice0.end - slice0.start),
            index: 0,
        };
        for (index, slice) in self.free_slices.iter().skip(1).enumerate() {
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
        self.free_slices.swap_remove(result.index);
        Some(result.range)
    }

    /// Allocate a new slice in the buffer to store the particles of a single effect.
    pub fn allocate_slice(&mut self, capacity: u32, item_size: u32) -> Option<SliceRef> {
        trace!(
            "EffectBuffer::allocate_slice: capacity={} item_size={}",
            capacity,
            item_size
        );

        let byte_size = capacity
            .checked_mul(item_size)
            .expect("Effect slice size overflow");

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
                    warn!("Cannot allocate slice of size {} ({} B) in effect cache buffer of capacity {}.", capacity, byte_size, self.capacity);
                }
                return None;
            }
        };

        Some(SliceRef { range, item_size })
    }

    // pub fn write_slice(&mut self, slice: &SliceRef, data: &[u8], queue: &RenderQueue) {
    //     assert!(data.len() <= slice.byte_size());
    //     let bytes: &[u8] = cast_slice(data);
    //     queue.write_buffer(buffer, slice.range.begin, &bytes[slice.range]);
    // }

    pub fn is_compatible(&self, handle: &Handle<EffectAsset>) -> bool {
        // TODO - replace with check particle layout is compatible to allow update in the same single dispatch call
        *handle == self.asset
    }
}

/// Identifier referencing an effect cached in an internal effect cache.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectCacheId(u64);

impl EffectCacheId {
    /// An invalid handle, corresponding to nothing.
    pub const INVALID: Self = EffectCacheId(u64::MAX);

    /// Generate a new valid effect cache identifier.
    pub fn new() -> EffectCacheId {
        static NEXT_EFFECT_CACHE_ID: AtomicU64 = AtomicU64::new(0);
        EffectCacheId(NEXT_EFFECT_CACHE_ID.fetch_add(1, AtomicOrdering::Relaxed))
    }
}

/// Cache for effect instances sharing common GPU data structures.
pub struct EffectCache {
    /// Render device the GPU resources (buffers) are allocated from.
    device: RenderDevice,
    /// Collection of effect buffers managed by this cache.
    buffers: Vec<EffectBuffer>,
    /// Map from an effect cache ID to the index of the buffer and the slice into that buffer.
    effects: HashMap<EffectCacheId, (usize, SliceRef)>,
}

impl EffectCache {
    pub fn new(device: RenderDevice) -> Self {
        EffectCache {
            device,
            buffers: vec![],
            effects: HashMap::default(),
        }
    }

    pub fn buffers(&self) -> &[EffectBuffer] {
        &self.buffers
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
                // The buffer must be compatible with the effect layout, to allow the update pass
                // to update all particles at once from all compatible effects in a single dispatch.
                if !buffer.is_compatible(&asset) {
                    return None;
                }

                // Try to allocate a slice into the buffer
                buffer
                    .allocate_slice(capacity, item_size)
                    .map(|slice| (buffer_index, slice))
            })
            .or_else(|| {
                // Cannot find any suitable buffer; allocate a new one
                let buffer_index = self.buffers.len();
                let byte_size = capacity.checked_mul(item_size).expect(&format!(
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
                self.buffers.push(EffectBuffer::new(
                    asset,
                    capacity,
                    item_size,
                    //pipeline,
                    &self.device,
                    Some(&format!("effect_buffer{}", self.buffers.len())),
                ));
                let buffer = self.buffers.last_mut().unwrap();
                Some((
                    buffer_index,
                    buffer.allocate_slice(capacity, item_size).unwrap(),
                ))
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
        return id;
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
}
