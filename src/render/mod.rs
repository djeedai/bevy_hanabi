use std::{
    borrow::Cow,
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
    num::{NonZeroU32, NonZeroU64},
    ops::{Deref, DerefMut, Range},
    time::Duration,
    vec,
};

#[cfg(feature = "2d")]
use bevy::core_pipeline::core_2d::{Transparent2d, CORE_2D_DEPTH_FORMAT};
#[cfg(feature = "2d")]
use bevy::math::FloatOrd;
#[cfg(feature = "3d")]
use bevy::{
    core_pipeline::{
        core_3d::{
            AlphaMask3d, Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey, Transparent3d,
            CORE_3D_DEPTH_FORMAT,
        },
        prepass::{OpaqueNoLightmap3dBatchSetKey, OpaqueNoLightmap3dBinKey},
    },
    render::render_phase::{BinnedPhaseItem, ViewBinnedRenderPhases},
};
use bevy::{
    ecs::{
        change_detection::Tick,
        prelude::*,
        system::{lifetimeless::*, SystemParam, SystemState},
    },
    log::trace,
    mesh::MeshVertexBufferLayoutRef,
    platform::collections::HashMap,
    prelude::*,
    render::{
        mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo},
        render_phase::{
            Draw, DrawError, DrawFunctions, PhaseItemExtraIndex, SortedPhaseItem,
            TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        sync_world::{MainEntity, RenderEntity, TemporaryRenderEntity},
        texture::GpuImage,
        view::{
            ExtractedView, RenderVisibleEntities, ViewTarget, ViewUniform, ViewUniformOffset,
            ViewUniforms,
        },
        Extract, MainWorld,
    },
};
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use effect_cache::{CachedEffect, EffectSlice, SlabState};
use event::{CachedChildInfo, CachedEffectEvents, CachedParentInfo, GpuChildInfo};
use fixedbitset::FixedBitSet;
use gpu_buffer::GpuBuffer;
use naga_oil::compose::{Composer, NagaModuleDescriptor};

use crate::{
    asset::{DefaultMesh, EffectAsset},
    calc_func_id,
    render::{
        batch::{BatchInput, EffectDrawBatch, EffectSorter, InitAndUpdatePipelineIds},
        effect_cache::{
            AnyDrawIndirectArgs, CachedDrawIndirectArgs, DispatchBufferIndices, SlabId,
        },
    },
    AlphaMode, Attribute, CompiledParticleEffect, EffectProperties, EffectShader, EffectSimulation,
    EffectSpawner, EffectVisibilityClass, ParticleLayout, PropertyLayout, SimulationCondition,
    TextureLayout,
};

mod aligned_buffer_vec;
mod batch;
mod buffer_table;
mod effect_cache;
mod event;
mod gpu_buffer;
mod property;
mod shader_cache;
mod sort;

use aligned_buffer_vec::AlignedBufferVec;
use batch::BatchSpawnInfo;
pub(crate) use batch::SortedEffectBatches;
use buffer_table::{BufferTable, BufferTableId};
pub(crate) use effect_cache::EffectCache;
pub(crate) use event::{allocate_events, on_remove_cached_effect_events, EventCache};
pub(crate) use property::{
    allocate_properties, on_remove_cached_properties, prepare_property_buffers, PropertyBindGroups,
    PropertyCache,
};
use property::{CachedEffectProperties, PropertyBindGroupKey};
pub use shader_cache::ShaderCache;
pub(crate) use sort::SortBindGroups;

use self::batch::EffectBatch;

// Size of an indirect index (including both parts of the ping-pong buffer) in
// bytes.
const INDIRECT_INDEX_SIZE: u32 = 12;

/// Helper to calculate a hash of a given hashable value.
fn calc_hash<H: Hash>(value: &H) -> u64 {
    let mut hasher = DefaultHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Source data (buffer and range inside the buffer) to create a buffer binding.
#[derive(Debug, Clone)]
pub(crate) struct BufferBindingSource {
    buffer: Buffer,
    offset: u32,
    size: NonZeroU32,
}

impl BufferBindingSource {
    /// Get a binding over the source data.
    pub fn as_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.buffer,
            offset: self.offset as u64 * 4,
            size: Some(self.size.into()),
        })
    }
}

impl PartialEq for BufferBindingSource {
    fn eq(&self, other: &Self) -> bool {
        self.buffer.id() == other.buffer.id()
            && self.offset == other.offset
            && self.size == other.size
    }
}

impl<'a> From<&'a BufferBindingSource> for BufferBinding<'a> {
    fn from(value: &'a BufferBindingSource) -> Self {
        BufferBinding {
            buffer: &value.buffer,
            offset: value.offset as u64,
            size: Some(value.size.into()),
        }
    }
}

/// Simulation parameters, available to all shaders of all effects.
#[derive(Debug, Default, Clone, Copy, Resource)]
pub(crate) struct SimParams {
    /// Current effect system simulation time since startup, in seconds.
    /// This is based on the [`Time<EffectSimulation>`](EffectSimulation) clock.
    time: f64,
    /// Delta time, in seconds, since last effect system update.
    delta_time: f32,

    /// Current virtual time since startup, in seconds.
    /// This is based on the [`Time<Virtual>`](Virtual) clock.
    virtual_time: f64,
    /// Virtual delta time, in seconds, since last effect system update.
    virtual_delta_time: f32,

    /// Current real time since startup, in seconds.
    /// This is based on the [`Time<Real>`](Real) clock.
    real_time: f64,
    /// Real delta time, in seconds, since last effect system update.
    real_delta_time: f32,
}

/// GPU representation of [`SimParams`], as well as additional per-frame
/// effect-independent values.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, ShaderType)]
struct GpuSimParams {
    /// Delta time, in seconds, since last effect system update.
    delta_time: f32,
    /// Current effect system simulation time since startup, in seconds.
    ///
    /// This is a lower-precision variant of [`SimParams::time`].
    time: f32,
    /// Virtual delta time, in seconds, since last effect system update.
    virtual_delta_time: f32,
    /// Current virtual time since startup, in seconds.
    ///
    /// This is a lower-precision variant of [`SimParams::time`].
    virtual_time: f32,
    /// Real delta time, in seconds, since last effect system update.
    real_delta_time: f32,
    /// Current real time since startup, in seconds.
    ///
    /// This is a lower-precision variant of [`SimParams::time`].
    real_time: f32,
    /// Total number of effects to update this frame. Used by the indirect
    /// compute pipeline to cap the compute thread to the actual number of
    /// effects to process.
    ///
    /// This is only used by the `vfx_indirect` compute shader.
    num_effects: u32,
}

impl Default for GpuSimParams {
    fn default() -> Self {
        Self {
            delta_time: 0.04,
            time: 0.0,
            virtual_delta_time: 0.04,
            virtual_time: 0.0,
            real_delta_time: 0.04,
            real_time: 0.0,
            num_effects: 0,
        }
    }
}

impl From<SimParams> for GpuSimParams {
    #[inline]
    fn from(src: SimParams) -> Self {
        Self::from(&src)
    }
}

impl From<&SimParams> for GpuSimParams {
    fn from(src: &SimParams) -> Self {
        Self {
            delta_time: src.delta_time,
            time: src.time as f32,
            virtual_delta_time: src.virtual_delta_time,
            virtual_time: src.virtual_time as f32,
            real_delta_time: src.real_delta_time,
            real_time: src.real_time as f32,
            ..default()
        }
    }
}

/// Compressed representation of a transform for GPU transfer.
///
/// The transform is stored as the three first rows of a transposed [`Mat4`],
/// assuming the last row is the unit row [`Vec4::W`]. The transposing ensures
/// that the three values are [`Vec4`] types which are naturally aligned and
/// without padding when used in WGSL. Without this, storing only the first
/// three components of each column would introduce padding, and would use the
/// same storage size on GPU as a full [`Mat4`].
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
pub(crate) struct GpuCompressedTransform {
    pub x_row: [f32; 4],
    pub y_row: [f32; 4],
    pub z_row: [f32; 4],
}

impl From<Mat4> for GpuCompressedTransform {
    fn from(value: Mat4) -> Self {
        let tr = value.transpose();
        #[cfg(test)]
        crate::test_utils::assert_approx_eq!(tr.w_axis, Vec4::W);
        Self {
            x_row: tr.x_axis.to_array(),
            y_row: tr.y_axis.to_array(),
            z_row: tr.z_axis.to_array(),
        }
    }
}

impl From<&Mat4> for GpuCompressedTransform {
    fn from(value: &Mat4) -> Self {
        let tr = value.transpose();
        #[cfg(test)]
        crate::test_utils::assert_approx_eq!(tr.w_axis, Vec4::W);
        Self {
            x_row: tr.x_axis.to_array(),
            y_row: tr.y_axis.to_array(),
            z_row: tr.z_axis.to_array(),
        }
    }
}

impl GpuCompressedTransform {
    /// Returns the translation as represented by this transform.
    #[allow(dead_code)]
    pub fn translation(&self) -> Vec3 {
        Vec3 {
            x: self.x_row[3],
            y: self.y_row[3],
            z: self.z_row[3],
        }
    }
}

/// Extension trait for shader types stored in a WGSL storage buffer.
pub(crate) trait StorageType {
    /// Get the aligned size, in bytes, of this type such that it aligns to the
    /// given alignment, in bytes.
    ///
    /// This is mainly used to align GPU types to device requirements.
    fn aligned_size(alignment: u32) -> NonZeroU64;

    /// Get the WGSL padding code to append to the GPU struct to align it.
    ///
    /// This is useful if the struct needs to be bound directly with a dynamic
    /// bind group offset, which requires the offset to be a multiple of a GPU
    /// device specific alignment value.
    fn padding_code(alignment: u32) -> String;
}

impl<T: ShaderType> StorageType for T {
    fn aligned_size(alignment: u32) -> NonZeroU64 {
        NonZeroU64::new(T::min_size().get().next_multiple_of(alignment as u64)).unwrap()
    }

    fn padding_code(alignment: u32) -> String {
        let aligned_size = T::aligned_size(alignment);
        trace!(
            "Aligning {} to {} bytes as device limits requires. Orignal size: {} bytes. Aligned size: {} bytes.",
            std::any::type_name::<T>(),
            alignment,
            T::min_size().get(),
            aligned_size
        );

        // We need to pad the Spawner WGSL struct based on the device padding so that we
        // can use it as an array element but also has a direct struct binding.
        if T::min_size() != aligned_size {
            let padding_size = aligned_size.get() - T::min_size().get();
            assert!(padding_size % 4 == 0);
            format!("padding: array<u32, {}>", padding_size / 4)
        } else {
            "".to_string()
        }
    }
}

/// GPU representation of spawner parameters.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
pub(crate) struct GpuSpawnerParams {
    /// Transform of the effect (origin of the emitter). This is either added to
    /// emitted particles at spawn time, if the effect simulated in world
    /// space, or to all simulated particles during rendering if the effect is
    /// simulated in local space.
    transform: GpuCompressedTransform,
    /// Inverse of [`transform`], stored with the same convention.
    ///
    /// [`transform`]: Self::transform
    inverse_transform: GpuCompressedTransform,
    /// Number of particles to spawn this frame.
    spawn: i32,
    /// Spawn seed, for randomized modifiers.
    seed: u32,
    /// Index of the pong (read) buffer for indirect indices, used by the render
    /// shader to fetch particles and render them. Only temporarily stored
    /// between indirect and render passes, and overwritten each frame by CPU
    /// upload. This is mostly a hack to transfer a value between those 2
    /// compute passes.
    render_pong: u32,
    /// Index of the [`GpuEffectMetadata`] for this effect.
    effect_metadata_index: u32,
    /// Index of the [`GpuDrawIndirect`] or [`GpuDrawIndexedIndirect`] for this
    /// effect.
    draw_indirect_index: u32,
    /// Start offset of the particles and indirect indices into the effect's
    /// slab, in number of particles (row index).
    slab_offset: u32,
    /// Start offset of the particles and indirect indices into the parent
    /// effect's slab (if the effect has a parent effect), in number of
    /// particles (row index). This is ignored if the effect has no parent.
    parent_slab_offset: u32,
}

/// GPU representation of an indirect compute dispatch input.
///
/// Note that unlike most other data structure, this doesn't need to be aligned
/// (except for the default 4-byte align for most GPU types) to any uniform or
/// storage buffer offset alignment, because the buffer storing this is only
/// ever used as input to indirect dispatch commands, and never bound as a
/// shader resource.
///
/// See https://docs.rs/wgpu/latest/wgpu/util/struct.DispatchIndirectArgs.html.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct GpuDispatchIndirectArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Default for GpuDispatchIndirectArgs {
    fn default() -> Self {
        Self { x: 0, y: 1, z: 1 }
    }
}

/// GPU representation of an indirect (non-indexed) render input.
///
/// Note that unlike most other data structure, this doesn't need to be aligned
/// (except for the default 4-byte align for most GPU types) to any uniform or
/// storage buffer offset alignment, because the buffer storing this is only
/// ever used as input to indirect render commands, and never bound as a shader
/// resource.
///
/// See https://docs.rs/wgpu/latest/wgpu/util/struct.DrawIndirectArgs.html.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable, ShaderType)]
pub struct GpuDrawIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

impl Default for GpuDrawIndirectArgs {
    fn default() -> Self {
        Self {
            vertex_count: 0,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        }
    }
}

/// GPU representation of an indirect indexed render input.
///
/// Note that unlike most other data structure, this doesn't need to be aligned
/// (except for the default 4-byte align for most GPU types) to any uniform or
/// storage buffer offset alignment, because the buffer storing this is only
/// ever used as input to indirect render commands, and never bound as a shader
/// resource.
///
/// See https://docs.rs/wgpu/latest/wgpu/util/struct.DrawIndexedIndirectArgs.html.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable, ShaderType)]
pub struct GpuDrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

impl Default for GpuDrawIndexedIndirectArgs {
    fn default() -> Self {
        Self {
            index_count: 0,
            instance_count: 1,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        }
    }
}

/// Stores metadata about each particle effect.
///
/// This is written by the CPU and read by the GPU.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Pod, Zeroable, ShaderType)]
pub struct GpuEffectMetadata {
    //
    // Some runtime variables modified on GPU only (capacity is constant)
    /// Effect capacity, in number of particles.
    pub capacity: u32,
    // Additional data not part of the required draw indirect args
    /// Number of alive particles.
    pub alive_count: u32,
    /// Cached value of `alive_count` to cap threads in update pass.
    pub max_update: u32,
    /// Cached value of `dead_count` to cap threads in init pass.
    pub max_spawn: u32,
    /// Index of the ping buffer for particle indices. Init and update compute
    /// passes always write into the ping buffer and read from the pong buffer.
    /// The buffers are swapped (ping = 1 - ping) during the indirect dispatch.
    pub indirect_write_index: u32,

    //
    // Some real metadata values depending on where the effect instance is allocated.
    /// Index of the [`GpuDispatchIndirect`] struct inside the global
    /// [`EffectsMeta::dispatch_indirect_buffer`].
    pub indirect_dispatch_index: u32,
    /// Index of the [`GpuDrawIndirect`] or [`GpuDrawIndexedIndirect`] struct
    /// inside the global [`EffectsMeta::draw_indirect_buffer`] or
    /// [`EffectsMeta::draw_indexed_indirect_buffer`]. The actual buffer depends
    /// on whether the mesh is indexed or not, which is stored in
    /// [`CachedMeshLocation`].
    pub indirect_draw_index: u32,
    /// Offset (in u32 count) of the init indirect dispatch struct inside its
    /// buffer. This avoids having to align those 16-byte structs to the GPU
    /// alignment (at least 32 bytes, even 256 bytes on some).
    pub init_indirect_dispatch_index: u32,
    /// Index of this effect into its parent's ChildInfo array
    /// ([`EffectChildren::effect_cache_ids`] and its associated GPU
    /// array). This starts at zero for the first child of each effect, and is
    /// only unique per parent, not globally. Only available if this effect is a
    /// child of another effect (i.e. if it has a parent).
    pub local_child_index: u32,
    /// For children, global index of the ChildInfo into the shared array.
    pub global_child_index: u32,
    /// For parents, base index of the their first ChildInfo into the shared
    /// array.
    pub base_child_index: u32,

    /// Particle stride, in number of u32.
    pub particle_stride: u32,
    /// Offset from the particle start to the first sort key, in number of u32.
    pub sort_key_offset: u32,
    /// Offset from the particle start to the second sort key, in number of u32.
    pub sort_key2_offset: u32,

    //
    // Again some runtime-only GPU-mutated data
    /// Atomic counter incremented each time a particle spawns. Useful for
    /// things like RIBBON_ID or any other use where a unique value is needed.
    /// The value loops back after some time, but unless some particle lives
    /// forever there's little chance of repetition.
    pub particle_counter: u32,
}

/// Single init fill dispatch item in an [`InitFillDispatchQueue`].
#[derive(Debug)]
pub(super) struct InitFillDispatchItem {
    /// Index of the source [`GpuChildInfo`] entry to read the event count from.
    pub global_child_index: u32,
    /// Index of the [`GpuDispatchIndirect`] entry to write the workgroup count
    /// to.
    pub dispatch_indirect_index: u32,
}

/// Queue of fill dispatch operations for the init indirect pass.
///
/// The queue stores the init fill dispatch operations for the current frame,
/// without the reference to the source and destination buffers, which may be
/// reallocated later in the frame. This allows enqueuing operations during the
/// prepare rendering phase, while deferring GPU buffer (re-)allocation to a
/// later stage.
#[derive(Debug, Default, Resource)]
pub(super) struct InitFillDispatchQueue {
    queue: Vec<InitFillDispatchItem>,
    submitted_queue_index: Option<u32>,
}

impl InitFillDispatchQueue {
    /// Clear the queue.
    #[inline]
    pub fn clear(&mut self) {
        self.queue.clear();
        self.submitted_queue_index = None;
    }

    /// Check if the queue is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Enqueue a new operation.
    #[inline]
    pub fn enqueue(&mut self, global_child_index: u32, dispatch_indirect_index: u32) {
        assert!(global_child_index != u32::MAX);
        self.queue.push(InitFillDispatchItem {
            global_child_index,
            dispatch_indirect_index,
        });
    }

    /// Submit pending operations for this frame.
    pub fn submit(
        &mut self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        gpu_buffer_operations: &mut GpuBufferOperations,
    ) {
        if self.queue.is_empty() {
            return;
        }

        // Sort by source. We can only batch if the destination is also contiguous, so
        // we can check with a linear walk if the source is already sorted.
        self.queue
            .sort_unstable_by_key(|item| item.global_child_index);

        let mut fill_queue = GpuBufferOperationQueue::new();

        // Batch and schedule all init indirect dispatch operations
        assert!(
            self.queue[0].global_child_index != u32::MAX,
            "Global child index not initialized"
        );
        let mut src_start = self.queue[0].global_child_index;
        let mut dst_start = self.queue[0].dispatch_indirect_index;
        let mut src_end = src_start + 1;
        let mut dst_end = dst_start + 1;
        let src_stride = GpuChildInfo::min_size().get() as u32 / 4;
        let dst_stride = GpuDispatchIndirectArgs::SHADER_SIZE.get() as u32 / 4;
        for i in 1..self.queue.len() {
            let InitFillDispatchItem {
                global_child_index: src,
                dispatch_indirect_index: dst,
            } = self.queue[i];
            if src != src_end || dst != dst_end {
                let count = src_end - src_start;
                debug_assert_eq!(count, dst_end - dst_start);
                let args = GpuBufferOperationArgs {
                    src_offset: src_start * src_stride + 1,
                    src_stride,
                    dst_offset: dst_start * dst_stride,
                    dst_stride,
                    count,
                };
                trace!(
                "enqueue_init_fill(): src:global_child_index={} dst:init_indirect_dispatch_index={} args={:?} src_buffer={:?} dst_buffer={:?}",
                src_start,
                dst_start,
                args,
                src_buffer.id(),
                dst_buffer.id(),
            );
                fill_queue.enqueue(
                    GpuBufferOperationType::FillDispatchArgs,
                    args,
                    src_buffer.clone(),
                    0,
                    None,
                    dst_buffer.clone(),
                    0,
                    None,
                );
                src_start = src;
                dst_start = dst;
            }
            src_end = src + 1;
            dst_end = dst + 1;
        }
        if src_start != src_end || dst_start != dst_end {
            let count = src_end - src_start;
            debug_assert_eq!(count, dst_end - dst_start);
            let args = GpuBufferOperationArgs {
                src_offset: src_start * src_stride + 1,
                src_stride,
                dst_offset: dst_start * dst_stride,
                dst_stride,
                count,
            };
            trace!(
            "IFDA::submit(): src:global_child_index={} dst:init_indirect_dispatch_index={} args={:?} src_buffer={:?} dst_buffer={:?}",
            src_start,
            dst_start,
            args,
            src_buffer.id(),
            dst_buffer.id(),
        );
            fill_queue.enqueue(
                GpuBufferOperationType::FillDispatchArgs,
                args,
                src_buffer.clone(),
                0,
                None,
                dst_buffer.clone(),
                0,
                None,
            );
        }

        debug_assert!(self.submitted_queue_index.is_none());
        if !fill_queue.operation_queue.is_empty() {
            self.submitted_queue_index = Some(gpu_buffer_operations.submit(fill_queue));
        }
    }
}

/// Compute pipeline to run the `vfx_indirect` dispatch workgroup calculation
/// shader.
#[derive(Resource)]
pub(crate) struct DispatchIndirectPipeline {
    /// Layout of bind group sim_params@0.
    sim_params_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// Layout of bind group effect_metadata@1.
    effect_metadata_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// Layout of bind group spawner@2.
    spawner_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// Layout of bind group child_infos@3.
    child_infos_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// Shader when no GPU events are used (no bind group @3).
    indirect_shader_noevent: Handle<Shader>,
    /// Shader when GPU events are used (bind group @3 present).
    indirect_shader_events: Handle<Shader>,
}

impl FromWorld for DispatchIndirectPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        // Copy the indirect pipeline shaders to self, because we can't access anything
        // else during pipeline specialization.
        let (indirect_shader_noevent, indirect_shader_events) = {
            let effects_meta = world.get_resource::<EffectsMeta>().unwrap();
            (
                effects_meta.indirect_shader_noevent.clone(),
                effects_meta.indirect_shader_events.clone(),
            )
        };

        let storage_alignment = render_device.limits().min_storage_buffer_offset_alignment;
        let effect_metadata_size = GpuEffectMetadata::aligned_size(storage_alignment);
        let spawner_min_binding_size = GpuSpawnerParams::aligned_size(storage_alignment);

        // @group(0) @binding(0) var<uniform> sim_params : SimParams;
        trace!("GpuSimParams: min_size={}", GpuSimParams::min_size());
        let sim_params_bind_group_layout = BindGroupLayoutDescriptor::new(
            "hanabi:bind_group_layout:dispatch_indirect:sim_params",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(GpuSimParams::min_size()),
                },
                count: None,
            }],
        );

        trace!(
            "GpuEffectMetadata: min_size={} padded_size={}",
            GpuEffectMetadata::min_size(),
            effect_metadata_size,
        );
        let effect_metadata_bind_group_layout = BindGroupLayoutDescriptor::new(
            "hanabi:bind_group_layout:dispatch_indirect:effect_metadata@1",
            &[
                // @group(0) @binding(0) var<storage, read_write> effect_metadata_buffer :
                // array<u32>;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(effect_metadata_size),
                    },
                    count: None,
                },
                // @group(0) @binding(1) var<storage, read_write> dispatch_indirect_buffer :
                // array<u32>;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZeroU64::new(INDIRECT_INDEX_SIZE as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
                // @group(0) @binding(2) var<storage, read_write> draw_indirect_buffer :
                // array<u32>;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuDrawIndexedIndirectArgs::SHADER_SIZE),
                    },
                    count: None,
                },
            ],
        );

        // @group(2) @binding(0) var<storage, read_write> spawner_buffer :
        // array<Spawner>;
        let spawner_bind_group_layout = BindGroupLayoutDescriptor::new(
            "hanabi:bind_group_layout:dispatch_indirect:spawner@2",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(spawner_min_binding_size),
                },
                count: None,
            }],
        );

        // @group(3) @binding(0) var<storage, read_write> child_info_buffer :
        // ChildInfoBuffer;
        let child_infos_bind_group_layout = BindGroupLayoutDescriptor::new(
            "hanabi:bind_group_layout:dispatch_indirect:child_infos",
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
            sim_params_bind_group_layout_desc: sim_params_bind_group_layout,
            effect_metadata_bind_group_layout_desc: effect_metadata_bind_group_layout,
            spawner_bind_group_layout_desc: spawner_bind_group_layout,
            child_infos_bind_group_layout_desc: child_infos_bind_group_layout,
            indirect_shader_noevent,
            indirect_shader_events,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct DispatchIndirectPipelineKey {
    /// True if any allocated effect uses GPU spawn events. In that case, the
    /// pipeline is specialized to clear all GPU events each frame after the
    /// indirect init pass consumed them to spawn particles, and before the
    /// update pass optionally produce more events.
    /// Key: HAS_GPU_SPAWN_EVENTS
    has_events: bool,
}

impl SpecializedComputePipeline for DispatchIndirectPipeline {
    type Key = DispatchIndirectPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        trace!(
            "Specializing indirect pipeline (has_events={})",
            key.has_events
        );

        let mut shader_defs = Vec::with_capacity(2);
        // Spawner struct needs to be defined with padding, because it's bound as an
        // array
        shader_defs.push("SPAWNER_PADDING".into());
        if key.has_events {
            shader_defs.push("HAS_GPU_SPAWN_EVENTS".into());
        }

        let mut layout = Vec::with_capacity(4);
        layout.push(self.sim_params_bind_group_layout_desc.clone());
        layout.push(self.effect_metadata_bind_group_layout_desc.clone());
        layout.push(self.spawner_bind_group_layout_desc.clone());
        if key.has_events {
            layout.push(self.child_infos_bind_group_layout_desc.clone());
        }

        let label = format!(
            "hanabi:compute_pipeline:dispatch_indirect{}",
            if key.has_events {
                "_events"
            } else {
                "_noevent"
            }
        );

        ComputePipelineDescriptor {
            label: Some(label.into()),
            layout,
            shader: if key.has_events {
                self.indirect_shader_events.clone()
            } else {
                self.indirect_shader_noevent.clone()
            },
            shader_defs,
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        }
    }
}

/// Type of GPU buffer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GpuBufferOperationType {
    /// Clear the destination buffer to zero.
    ///
    /// The source parameters [`src_offset`] and [`src_stride`] are ignored.
    ///
    /// [`src_offset`]: crate::GpuBufferOperationArgs::src_offset
    /// [`src_stride`]: crate::GpuBufferOperationArgs::src_stride
    #[allow(dead_code)]
    Zero,
    /// Copy a source buffer into a destination buffer.
    ///
    /// The source can have a stride between each `u32` copied. The destination
    /// is always a contiguous buffer.
    #[allow(dead_code)]
    Copy,
    /// Fill the arguments for a later indirect dispatch call.
    ///
    /// This is similar to a copy, but will round up the source value to the
    /// number of threads per workgroup (64) before writing it into the
    /// destination.
    FillDispatchArgs,
    /// Fill the arguments for a later indirect dispatch call.
    ///
    /// This is the same as [`FillDispatchArgs`], but the source element count
    /// is read from the fourth entry in the destination buffer directly,
    /// and the source buffer and source arguments are unused.
    #[allow(dead_code)]
    FillDispatchArgsSelf,
}

/// GPU representation of the arguments of a block operation on a buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Pod, Zeroable, ShaderType)]
pub(super) struct GpuBufferOperationArgs {
    /// Offset, as u32 count, where the operation starts in the source buffer.
    src_offset: u32,
    /// Stride, as u32 count, between elements in the source buffer.
    src_stride: u32,
    /// Offset, as u32 count, where the operation starts in the destination
    /// buffer.
    dst_offset: u32,
    /// Stride, as u32 count, between elements in the destination buffer.
    dst_stride: u32,
    /// Number of u32 elements to process for this operation.
    count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct QueuedOperationBindGroupKey {
    src_buffer: BufferId,
    src_binding_size: Option<NonZeroU32>,
    dst_buffer: BufferId,
    dst_binding_size: Option<NonZeroU32>,
}

#[derive(Debug, Clone)]
struct QueuedOperation {
    op: GpuBufferOperationType,
    args_index: u32,
    src_buffer: Buffer,
    src_binding_offset: u32,
    src_binding_size: Option<NonZeroU32>,
    dst_buffer: Buffer,
    dst_binding_offset: u32,
    dst_binding_size: Option<NonZeroU32>,
}

impl From<&QueuedOperation> for QueuedOperationBindGroupKey {
    fn from(value: &QueuedOperation) -> Self {
        Self {
            src_buffer: value.src_buffer.id(),
            src_binding_size: value.src_binding_size,
            dst_buffer: value.dst_buffer.id(),
            dst_binding_size: value.dst_binding_size,
        }
    }
}

/// Queue of GPU buffer operations.
///
/// The queue records a series of ordered operations on GPU buffers. It can be
/// submitted for this frame via [`GpuBufferOperations::submit()`], and
/// subsequently dispatched as a compute pass via
/// [`GpuBufferOperations::dispatch()`].
pub struct GpuBufferOperationQueue {
    /// Operation arguments.
    args: Vec<GpuBufferOperationArgs>,
    /// Queued operations.
    operation_queue: Vec<QueuedOperation>,
}

impl GpuBufferOperationQueue {
    /// Create a new empty queue.
    pub fn new() -> Self {
        Self {
            args: vec![],
            operation_queue: vec![],
        }
    }

    /// Enqueue a generic operation.
    pub fn enqueue(
        &mut self,
        op: GpuBufferOperationType,
        args: GpuBufferOperationArgs,
        src_buffer: Buffer,
        src_binding_offset: u32,
        src_binding_size: Option<NonZeroU32>,
        dst_buffer: Buffer,
        dst_binding_offset: u32,
        dst_binding_size: Option<NonZeroU32>,
    ) -> u32 {
        trace!(
            "Queue {:?} op: args={:?} src_buffer={:?} src_binding_offset={} src_binding_size={:?} dst_buffer={:?} dst_binding_offset={} dst_binding_size={:?}",
            op,
            args,
            src_buffer,
            src_binding_offset,
            src_binding_size,
            dst_buffer,
            dst_binding_offset,
            dst_binding_size,
        );
        let args_index = self.args.len() as u32;
        self.args.push(args);
        self.operation_queue.push(QueuedOperation {
            op,
            args_index,
            src_buffer,
            src_binding_offset,
            src_binding_size,
            dst_buffer,
            dst_binding_offset,
            dst_binding_size,
        });
        args_index
    }
}

/// GPU buffer operations for this frame.
///
/// This resource contains a list of submitted [`GpuBufferOperationQueue`] for
/// the current frame, and ensures the bind groups for those operations are up
/// to date.
#[derive(Resource)]
pub(super) struct GpuBufferOperations {
    /// Arguments for the buffer operations submitted this frame.
    args_buffer: AlignedBufferVec<GpuBufferOperationArgs>,

    /// Bind groups for the submitted operations.
    bind_groups: HashMap<QueuedOperationBindGroupKey, BindGroup>,

    /// Submitted queues for this frame.
    queues: Vec<Vec<QueuedOperation>>,
}

impl FromWorld for GpuBufferOperations {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let align = render_device.limits().min_uniform_buffer_offset_alignment;
        Self::new(align)
    }
}

impl GpuBufferOperations {
    pub fn new(align: u32) -> Self {
        let args_buffer = AlignedBufferVec::new(
            BufferUsages::UNIFORM,
            Some(NonZeroU64::new(align as u64).unwrap()),
            Some("hanabi:buffer:gpu_operation_args".to_string()),
        );
        Self {
            args_buffer,
            bind_groups: default(),
            queues: vec![],
        }
    }

    /// Clear the queue and begin recording operations for a new frame.
    pub fn begin_frame(&mut self) {
        self.args_buffer.clear();
        self.bind_groups.clear(); // for now; might consider caching frame-to-frame
        self.queues.clear();
    }

    /// Submit a recorded queue.
    ///
    /// # Panics
    ///
    /// Panics if the queue submitted is empty.
    pub fn submit(&mut self, mut queue: GpuBufferOperationQueue) -> u32 {
        assert!(!queue.operation_queue.is_empty());
        let queue_index = self.queues.len() as u32;
        for qop in &mut queue.operation_queue {
            qop.args_index = self.args_buffer.push(queue.args[qop.args_index as usize]) as u32;
        }
        self.queues.push(queue.operation_queue);
        queue_index
    }

    /// Finish recording operations for this frame, and schedule buffer writes
    /// to GPU.
    pub fn end_frame(&mut self, device: &RenderDevice, render_queue: &RenderQueue) {
        assert_eq!(
            self.args_buffer.len(),
            self.queues.iter().fold(0, |len, q| len + q.len())
        );

        // Upload to GPU buffer
        self.args_buffer.write_buffer(device, render_queue);
    }

    /// Create all necessary bind groups for all queued operations.
    pub fn create_bind_groups(
        &mut self,
        render_device: &RenderDevice,
        utils_pipeline: &UtilsPipeline,
    ) {
        trace!(
            "Creating bind groups for {} operation queues...",
            self.queues.len()
        );
        for queue in &self.queues {
            for qop in queue {
                let key: QueuedOperationBindGroupKey = qop.into();
                self.bind_groups.entry(key).or_insert_with(|| {
                    let src_id: NonZeroU32 = qop.src_buffer.id().into();
                    let dst_id: NonZeroU32 = qop.dst_buffer.id().into();
                    let label = format!("hanabi:bind_group:util_{}_{}", src_id.get(), dst_id.get());
                    let use_dynamic_offset = matches!(qop.op, GpuBufferOperationType::FillDispatchArgs);
                    let bind_group_layout =
                        utils_pipeline.bind_group_layout(qop.op, use_dynamic_offset);
                    let (src_offset, dst_offset) = if use_dynamic_offset {
                        (0, 0)
                    } else {
                        (qop.src_binding_offset as u64, qop.dst_binding_offset as u64)
                    };
                    trace!(
                        "-> Creating new bind group '{}': src#{} (@+{}B:{:?}B) dst#{} (@+{}B:{:?}B)",
                        label,
                        src_id,
                        src_offset,
                        qop.src_binding_size,
                        dst_id,
                        dst_offset,
                        qop.dst_binding_size,
                    );
                    render_device.create_bind_group(
                        Some(&label[..]),
                        bind_group_layout,
                        &[
                            BindGroupEntry {
                                binding: 0,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: self.args_buffer.buffer().unwrap(),
                                    offset: 0,
                                    // We always bind exactly 1 row of arguments
                                    size: Some(
                                        NonZeroU64::new(self.args_buffer.aligned_size() as u64)
                                            .unwrap(),
                                    ),
                                }),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: &qop.src_buffer,
                                    offset: src_offset,
                                    size: qop.src_binding_size.map(Into::into),
                                }),
                            },
                            BindGroupEntry {
                                binding: 2,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: &qop.dst_buffer,
                                    offset: dst_offset,
                                    size: qop.dst_binding_size.map(Into::into),
                                }),
                            },
                        ],
                    )
                });
            }
        }
    }

    /// Dispatch a submitted queue by index.
    ///
    /// This creates a new, optionally labelled, compute pass, and records to
    /// the render context a series of compute workgroup dispatch, one for each
    /// enqueued operation.
    ///
    /// The compute pipeline(s) used for each operation are fetched from the
    /// [`UtilsPipeline`], and the associated bind groups are used from a
    /// previous call to [`Self::create_bind_groups()`].
    pub fn dispatch(
        &self,
        index: u32,
        render_context: &mut RenderContext,
        utils_pipeline: &UtilsPipeline,
        compute_pass_label: Option<&str>,
    ) {
        let queue = &self.queues[index as usize];
        trace!(
            "Recording GPU commands for queue #{} ({} ops)...",
            index,
            queue.len(),
        );

        if queue.is_empty() {
            return;
        }

        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: compute_pass_label,
                    timestamp_writes: None,
                });

        let mut prev_op = None;
        for qop in queue {
            trace!("qop={:?}", qop);

            if Some(qop.op) != prev_op {
                compute_pass.set_pipeline(utils_pipeline.get_pipeline(qop.op));
                prev_op = Some(qop.op);
            }

            let key: QueuedOperationBindGroupKey = qop.into();
            if let Some(bind_group) = self.bind_groups.get(&key) {
                let args_offset = self.args_buffer.dynamic_offset(qop.args_index as usize);
                let use_dynamic_offset = matches!(qop.op, GpuBufferOperationType::FillDispatchArgs);
                let (src_offset, dst_offset) = if use_dynamic_offset {
                    (qop.src_binding_offset, qop.dst_binding_offset)
                } else {
                    (0, 0)
                };
                compute_pass.set_bind_group(0, bind_group, &[args_offset, src_offset, dst_offset]);
                trace!(
                    "set bind group with args_offset=+{}B src_offset=+{}B dst_offset=+{}B",
                    args_offset,
                    src_offset,
                    dst_offset
                );
            } else {
                error!("GPU fill dispatch buffer operation bind group not found for buffers src#{:?} dst#{:?}", qop.src_buffer.id(), qop.dst_buffer.id());
                continue;
            }

            // Dispatch the operations for this buffer
            const WORKGROUP_SIZE: u32 = 64;
            let num_ops = 1u32; // TODO - batching!
            let workgroup_count = num_ops.div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            trace!(
                "-> fill dispatch compute dispatched: num_ops={} workgroup_count={}",
                num_ops,
                workgroup_count
            );
        }
    }
}

/// Compute pipeline to run the `vfx_utils` shader.
#[derive(Resource)]
pub(crate) struct UtilsPipeline {
    #[allow(dead_code)]
    bind_group_layout: BindGroupLayout,
    bind_group_layout_dyn: BindGroupLayout,
    bind_group_layout_no_src: BindGroupLayout,
    pipelines: [ComputePipeline; 4],
}

impl FromWorld for UtilsPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let bind_group_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:utils",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuBufferOperationArgs::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                },
            ],
        );

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("hanabi:pipeline_layout:utils"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let bind_group_layout_dyn = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:utils_dyn",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(GpuBufferOperationArgs::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                },
            ],
        );

        let pipeline_layout_dyn = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("hanabi:pipeline_layout:utils_dyn"),
            bind_group_layouts: &[&bind_group_layout_dyn],
            push_constant_ranges: &[],
        });

        let bind_group_layout_no_src = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:utils_no_src",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuBufferOperationArgs::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                },
            ],
        );

        let pipeline_layout_no_src =
            render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("hanabi:pipeline_layout:utils_no_src"),
                bind_group_layouts: &[&bind_group_layout_no_src],
                push_constant_ranges: &[],
            });

        let shader_code = include_str!("vfx_utils.wgsl");

        // Resolve imports. Because we don't insert this shader into Bevy' pipeline
        // cache, we don't get that part "for free", so we have to do it manually here.
        let shader_source = {
            let mut composer = Composer::default();

            let shader_defs = default();

            match composer.make_naga_module(NagaModuleDescriptor {
                source: shader_code,
                file_path: "vfx_utils.wgsl",
                shader_defs,
                ..Default::default()
            }) {
                Ok(naga_module) => ShaderSource::Naga(Cow::Owned(naga_module)),
                Err(compose_error) => panic!(
                    "Failed to compose vfx_utils.wgsl, naga_oil returned: {}",
                    compose_error.emit_to_string(&composer)
                ),
            }
        };

        debug!("Create utils shader module:\n{}", shader_code);
        #[allow(unsafe_code)]
        let shader_module = unsafe {
            render_device.create_shader_module(ShaderModuleDescriptor {
                label: Some("hanabi:shader:utils"),
                source: shader_source,
            })
        };

        trace!("Create vfx_utils pipelines...");
        let zero_pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("hanabi:compute_pipeline:zero_buffer"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("zero_buffer"),
            compilation_options: PipelineCompilationOptions {
                constants: &[],
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        });
        let copy_pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("hanabi:compute_pipeline:copy_buffer"),
            layout: Some(&pipeline_layout_dyn),
            module: &shader_module,
            entry_point: Some("copy_buffer"),
            compilation_options: PipelineCompilationOptions {
                constants: &[],
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        });
        let fill_dispatch_args_pipeline =
            render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some("hanabi:compute_pipeline:fill_dispatch_args"),
                layout: Some(&pipeline_layout_dyn),
                module: &shader_module,
                entry_point: Some("fill_dispatch_args"),
                compilation_options: PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });
        let fill_dispatch_args_self_pipeline =
            render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some("hanabi:compute_pipeline:fill_dispatch_args_self"),
                layout: Some(&pipeline_layout_no_src),
                module: &shader_module,
                entry_point: Some("fill_dispatch_args_self"),
                compilation_options: PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });

        Self {
            bind_group_layout,
            bind_group_layout_dyn,
            bind_group_layout_no_src,
            pipelines: [
                zero_pipeline,
                copy_pipeline,
                fill_dispatch_args_pipeline,
                fill_dispatch_args_self_pipeline,
            ],
        }
    }
}

impl UtilsPipeline {
    fn get_pipeline(&self, op: GpuBufferOperationType) -> &ComputePipeline {
        match op {
            GpuBufferOperationType::Zero => &self.pipelines[0],
            GpuBufferOperationType::Copy => &self.pipelines[1],
            GpuBufferOperationType::FillDispatchArgs => &self.pipelines[2],
            GpuBufferOperationType::FillDispatchArgsSelf => &self.pipelines[3],
        }
    }

    fn bind_group_layout(
        &self,
        op: GpuBufferOperationType,
        with_dynamic_offsets: bool,
    ) -> &BindGroupLayout {
        if op == GpuBufferOperationType::FillDispatchArgsSelf {
            assert!(
                !with_dynamic_offsets,
                "FillDispatchArgsSelf op cannot use dynamic offset (not implemented)"
            );
            &self.bind_group_layout_no_src
        } else if with_dynamic_offsets {
            &self.bind_group_layout_dyn
        } else {
            &self.bind_group_layout
        }
    }
}

#[derive(Resource)]
pub(crate) struct ParticlesInitPipeline {
    sim_params_layout_desc: BindGroupLayoutDescriptor,
}

impl Default for ParticlesInitPipeline {
    fn default() -> Self {
        let sim_params_layout_desc = BindGroupLayoutDescriptor::new(
            "hanabi:bind_group_layout:vfx_init:sim_params@0",
            // @group(0) @binding(0) var<uniform> sim_params: SimParams;
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(GpuSimParams::min_size()),
                },
                count: None,
            }],
        );

        Self {
            sim_params_layout_desc,
        }
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ParticleInitPipelineKeyFlags: u8 {
        //const CLONE = (1u8 << 0); // DEPRECATED
        const ATTRIBUTE_PREV = (1u8 << 1);
        const ATTRIBUTE_NEXT = (1u8 << 2);
        const CONSUME_GPU_SPAWN_EVENTS = (1u8 << 3);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ParticleInitPipelineKey {
    /// Compute shader, with snippets applied, but not preprocessed yet.
    shader: Handle<Shader>,
    /// Minimum binding size in bytes for the particle layout buffer.
    particle_layout_min_binding_size: NonZeroU32,
    /// Minimum binding size in bytes for the particle layout buffer of the
    /// parent effect, if any.
    /// Key: READ_PARENT_PARTICLE
    parent_particle_layout_min_binding_size: Option<NonZeroU32>,
    /// Pipeline flags.
    flags: ParticleInitPipelineKeyFlags,
    /// Layout of the particle@1 bind group this pipeline was specialized with.
    particle_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// Layout of the spawner@2 bind group this pipeline was specialized with.
    spawner_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// Layout of the metadata@3 bind group this pipeline was specialized with.
    metadata_bind_group_layout_desc: BindGroupLayoutDescriptor,
}

impl SpecializedComputePipeline for ParticlesInitPipeline {
    type Key = ParticleInitPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        // We use the hash to correlate the key content with the GPU resource name
        let hash = calc_hash(&key);
        trace!("Specializing init pipeline {hash:016X} with key {key:?}");

        let mut shader_defs = Vec::with_capacity(4);
        if key
            .flags
            .contains(ParticleInitPipelineKeyFlags::ATTRIBUTE_PREV)
        {
            shader_defs.push("ATTRIBUTE_PREV".into());
        }
        if key
            .flags
            .contains(ParticleInitPipelineKeyFlags::ATTRIBUTE_NEXT)
        {
            shader_defs.push("ATTRIBUTE_NEXT".into());
        }
        let consume_gpu_spawn_events = key
            .flags
            .contains(ParticleInitPipelineKeyFlags::CONSUME_GPU_SPAWN_EVENTS);
        if consume_gpu_spawn_events {
            shader_defs.push("CONSUME_GPU_SPAWN_EVENTS".into());
        }
        // FIXME - for now this needs to keep in sync with consume_gpu_spawn_events
        if key.parent_particle_layout_min_binding_size.is_some() {
            assert!(consume_gpu_spawn_events);
            shader_defs.push("READ_PARENT_PARTICLE".into());
        } else {
            assert!(!consume_gpu_spawn_events);
        }

        let label = format!("hanabi:pipeline:init_{hash:016X}");
        trace!(
            "-> creating pipeline '{}' with shader defs:{}",
            label,
            shader_defs
                .iter()
                .fold(String::new(), |acc, x| acc + &format!(" {x:?}"))
        );

        ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![
                self.sim_params_layout_desc.clone(),
                key.particle_bind_group_layout_desc.clone(),
                key.spawner_bind_group_layout_desc.clone(),
                key.metadata_bind_group_layout_desc.clone(),
            ],
            shader: key.shader,
            shader_defs,
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        }
    }
}

#[derive(Resource)]
pub(crate) struct ParticlesUpdatePipeline {
    sim_params_layout_desc: BindGroupLayoutDescriptor,
}

impl Default for ParticlesUpdatePipeline {
    fn default() -> Self {
        trace!("GpuSimParams: min_size={}", GpuSimParams::min_size());
        let sim_params_layout_desc = BindGroupLayoutDescriptor::new(
            "hanabi:bind_group_layout:vfx_update:sim_params@0",
            &[
                // @group(0) @binding(0) var<uniform> sim_params : SimParams;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuSimParams::min_size()),
                    },
                    count: None,
                },
                // @group(0) @binding(1) var<storage, read_write> draw_indirect_buffer :
                // array<DrawIndexedIndirectArgs>;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuDrawIndexedIndirectArgs::SHADER_SIZE),
                    },
                    count: None,
                },
            ],
        );

        Self {
            sim_params_layout_desc,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct ParticleUpdatePipelineKey {
    /// Compute shader, with snippets applied, but not preprocessed yet.
    shader: Handle<Shader>,
    /// Particle layout.
    particle_layout: ParticleLayout,
    /// Minimum binding size in bytes for the particle layout buffer of the
    /// parent effect, if any.
    /// Key: READ_PARENT_PARTICLE
    parent_particle_layout_min_binding_size: Option<NonZeroU32>,
    /// Key: EMITS_GPU_SPAWN_EVENTS
    num_event_buffers: u32,
    /// Layout of the particle@1 bind group this pipeline was specialized with.
    particle_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// Layout of the spawner@2 bind group this pipeline was specialized with.
    spawner_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// Layout of the metadata@3 bind group this pipeline was specialized with.
    metadata_bind_group_layout_desc: BindGroupLayoutDescriptor,
}

impl SpecializedComputePipeline for ParticlesUpdatePipeline {
    type Key = ParticleUpdatePipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        // We use the hash to correlate the key content with the GPU resource name
        let hash = calc_hash(&key);
        trace!("Specializing update pipeline {hash:016X} with key {key:?}");

        let mut shader_defs = Vec::with_capacity(6);
        shader_defs.push("EM_MAX_SPAWN_ATOMIC".into());
        // ChildInfo needs atomic event_count because all threads append to the event
        // buffer(s) in parallel.
        shader_defs.push("CHILD_INFO_EVENT_COUNT_IS_ATOMIC".into());
        if key.particle_layout.contains(Attribute::PREV) {
            shader_defs.push("ATTRIBUTE_PREV".into());
        }
        if key.particle_layout.contains(Attribute::NEXT) {
            shader_defs.push("ATTRIBUTE_NEXT".into());
        }
        if key.parent_particle_layout_min_binding_size.is_some() {
            shader_defs.push("READ_PARENT_PARTICLE".into());
        }
        if key.num_event_buffers > 0 {
            shader_defs.push("EMITS_GPU_SPAWN_EVENTS".into());
        }

        let hash = calc_func_id(&key);
        let label = format!("hanabi:pipeline:update_{hash:016X}");
        trace!(
            "-> creating pipeline '{}' with shader defs:{}",
            label,
            shader_defs
                .iter()
                .fold(String::new(), |acc, x| acc + &format!(" {x:?}"))
        );

        ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![
                self.sim_params_layout_desc.clone(),
                key.particle_bind_group_layout_desc.clone(),
                key.spawner_bind_group_layout_desc.clone(),
                key.metadata_bind_group_layout_desc.clone(),
            ],
            shader: key.shader,
            shader_defs,
            entry_point: Some("main".into()),
            push_constant_ranges: Vec::new(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

#[derive(Resource)]
pub(crate) struct ParticlesRenderPipeline {
    render_device: RenderDevice,
    view_layout_desc: BindGroupLayoutDescriptor,
    material_layout_descs: HashMap<TextureLayout, BindGroupLayoutDescriptor>,
}

impl ParticlesRenderPipeline {
    /// Cache a material, creating its bind group layout based on the texture
    /// layout.
    pub fn cache_material(&mut self, layout: &TextureLayout) {
        if layout.layout.is_empty() {
            return;
        }

        // FIXME - no current stable API to insert an entry into a HashMap only if it
        // doesn't exist, and without having to build a key (as opposed to a reference).
        // So do 2 lookups instead, to avoid having to clone the layout if it's already
        // cached (which should be the common case).
        if self.material_layout_descs.contains_key(layout) {
            return;
        }

        let mut entries = Vec::with_capacity(layout.layout.len() * 2);
        let mut index = 0;
        for _slot in &layout.layout {
            entries.push(BindGroupLayoutEntry {
                binding: index,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            });
            entries.push(BindGroupLayoutEntry {
                binding: index + 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            });
            index += 2;
        }
        debug!(
            "Creating material bind group with {} entries [{:?}] for layout {:?}",
            entries.len(),
            entries,
            layout
        );
        let material_bind_group_layout_desc =
            BindGroupLayoutDescriptor::new("hanabi:material_layout_render", &entries[..]);
        self.material_layout_descs
            .insert(layout.clone(), material_bind_group_layout_desc);
    }

    /// Retrieve a bind group layout for a cached material.
    pub fn get_material(&self, layout: &TextureLayout) -> Option<&BindGroupLayoutDescriptor> {
        // Prevent a hash and lookup for the trivial case of an empty layout
        if layout.layout.is_empty() {
            return None;
        }

        self.material_layout_descs.get(layout)
    }
}

impl FromWorld for ParticlesRenderPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let view_layout_desc = BindGroupLayoutDescriptor::new(
            "hanabi:bind_group_layout:render:view@0",
            &[
                // @group(0) @binding(0) var<uniform> view: View;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(ViewUniform::min_size()),
                    },
                    count: None,
                },
                // @group(0) @binding(1) var<uniform> sim_params : SimParams;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuSimParams::min_size()),
                    },
                    count: None,
                },
            ],
        );

        Self {
            render_device: render_device.clone(),
            view_layout_desc,
            material_layout_descs: default(),
        }
    }
}

#[cfg(all(feature = "2d", feature = "3d"))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum PipelineMode {
    Camera2d,
    Camera3d,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct ParticleRenderPipelineKey {
    /// Render shader, with snippets applied, but not preprocessed yet.
    shader: Handle<Shader>,
    /// Particle layout.
    particle_layout: ParticleLayout,
    mesh_layout: Option<MeshVertexBufferLayoutRef>,
    /// Texture layout.
    texture_layout: TextureLayout,
    /// Key: LOCAL_SPACE_SIMULATION
    /// The effect is simulated in local space, and during rendering all
    /// particles are transformed by the effect's [`GlobalTransform`].
    local_space_simulation: bool,
    /// Key: USE_ALPHA_MASK, OPAQUE
    /// The particle's alpha masking behavior.
    alpha_mask: ParticleRenderAlphaMaskPipelineKey,
    /// The effect needs Alpha blend.
    alpha_mode: AlphaMode,
    /// Key: FLIPBOOK
    /// The effect is rendered with flipbook texture animation based on the
    /// sprite index of each particle.
    flipbook: bool,
    /// Key: NEEDS_UV
    /// The effect needs UVs.
    needs_uv: bool,
    /// Key: NEEDS_NORMAL
    /// The effect needs normals.
    needs_normal: bool,
    /// Key: NEEDS_PARTICLE_IN_FRAGMENT
    /// The effect needs access to the particle index and buffer in the fragment
    /// shader.
    needs_particle_fragment: bool,
    /// Key: RIBBONS
    /// The effect has ribbons.
    ribbons: bool,
    /// For dual-mode configurations only, the actual mode of the current render
    /// pipeline. Otherwise the mode is implicitly determined by the active
    /// feature.
    #[cfg(all(feature = "2d", feature = "3d"))]
    pipeline_mode: PipelineMode,
    /// MSAA sample count.
    msaa_samples: u32,
    /// Is the camera using an HDR render target?
    hdr: bool,
}

#[derive(Clone, Copy, Default, Hash, PartialEq, Eq, Debug)]
pub(crate) enum ParticleRenderAlphaMaskPipelineKey {
    #[default]
    Blend,
    /// Key: USE_ALPHA_MASK
    /// The effect is rendered with alpha masking.
    AlphaMask,
    /// Key: OPAQUE
    /// The effect is rendered fully-opaquely.
    Opaque,
}

impl Default for ParticleRenderPipelineKey {
    fn default() -> Self {
        Self {
            shader: Handle::default(),
            particle_layout: ParticleLayout::empty(),
            mesh_layout: None,
            texture_layout: default(),
            local_space_simulation: false,
            alpha_mask: default(),
            alpha_mode: AlphaMode::Blend,
            flipbook: false,
            needs_uv: false,
            needs_normal: false,
            needs_particle_fragment: false,
            ribbons: false,
            #[cfg(all(feature = "2d", feature = "3d"))]
            pipeline_mode: PipelineMode::Camera3d,
            msaa_samples: Msaa::default().samples(),
            hdr: false,
        }
    }
}

impl SpecializedRenderPipeline for ParticlesRenderPipeline {
    type Key = ParticleRenderPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        trace!("Specializing render pipeline for key: {key:?}");

        trace!("Creating layout for bind group particle@1 of render pass");
        let alignment = self
            .render_device
            .limits()
            .min_storage_buffer_offset_alignment;
        let spawner_min_binding_size = GpuSpawnerParams::aligned_size(alignment);
        let entries = [
            // @group(1) @binding(0) var<storage, read> particle_buffer : ParticleBuffer;
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(key.particle_layout.min_binding_size()),
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
                    min_binding_size: Some(spawner_min_binding_size),
                },
                count: None,
            },
        ];
        let particle_bind_group_layout_desc = BindGroupLayoutDescriptor::new(
            "hanabi:bind_group_layout:render:particle@1",
            &entries[..],
        );

        let mut layout = vec![
            self.view_layout_desc.clone(),
            particle_bind_group_layout_desc,
        ];
        let mut shader_defs = vec![];

        let vertex_buffer_layout = key.mesh_layout.as_ref().and_then(|mesh_layout| {
            mesh_layout
                .0
                .get_layout(&[
                    Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
                    Mesh::ATTRIBUTE_UV_0.at_shader_location(1),
                    Mesh::ATTRIBUTE_NORMAL.at_shader_location(2),
                ])
                .ok()
        });

        if let Some(material_bind_group_layout) = self.get_material(&key.texture_layout) {
            layout.push(material_bind_group_layout.clone());
        }

        // Key: LOCAL_SPACE_SIMULATION
        if key.local_space_simulation {
            shader_defs.push("LOCAL_SPACE_SIMULATION".into());
        }

        match key.alpha_mask {
            ParticleRenderAlphaMaskPipelineKey::Blend => {}
            ParticleRenderAlphaMaskPipelineKey::AlphaMask => {
                // Key: USE_ALPHA_MASK
                shader_defs.push("USE_ALPHA_MASK".into())
            }
            ParticleRenderAlphaMaskPipelineKey::Opaque => {
                // Key: OPAQUE
                shader_defs.push("OPAQUE".into())
            }
        }

        // Key: FLIPBOOK
        if key.flipbook {
            shader_defs.push("FLIPBOOK".into());
        }

        // Key: NEEDS_UV
        if key.needs_uv {
            shader_defs.push("NEEDS_UV".into());
        }

        // Key: NEEDS_NORMAL
        if key.needs_normal {
            shader_defs.push("NEEDS_NORMAL".into());
        }

        if key.needs_particle_fragment {
            shader_defs.push("NEEDS_PARTICLE_FRAGMENT".into());
        }

        // Key: RIBBONS
        if key.ribbons {
            shader_defs.push("RIBBONS".into());
        }

        #[cfg(feature = "2d")]
        let depth_stencil_2d = DepthStencilState {
            format: CORE_2D_DEPTH_FORMAT,
            // Use depth buffer with alpha-masked particles, not with transparent ones
            depth_write_enabled: false, // TODO - opaque/alphamask 2d
            // Bevy uses reverse-Z, so GreaterEqual really means closer
            depth_compare: CompareFunction::GreaterEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        };

        #[cfg(feature = "3d")]
        let depth_stencil_3d = DepthStencilState {
            format: CORE_3D_DEPTH_FORMAT,
            // Use depth buffer with alpha-masked or opaque particles, not
            // with transparent ones
            depth_write_enabled: matches!(
                key.alpha_mask,
                ParticleRenderAlphaMaskPipelineKey::AlphaMask
                    | ParticleRenderAlphaMaskPipelineKey::Opaque
            ),
            // Bevy uses reverse-Z, so GreaterEqual really means closer
            depth_compare: CompareFunction::GreaterEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        };

        #[cfg(all(feature = "2d", feature = "3d"))]
        assert_eq!(CORE_2D_DEPTH_FORMAT, CORE_3D_DEPTH_FORMAT);
        #[cfg(all(feature = "2d", feature = "3d"))]
        let depth_stencil = match key.pipeline_mode {
            PipelineMode::Camera2d => Some(depth_stencil_2d),
            PipelineMode::Camera3d => Some(depth_stencil_3d),
        };

        #[cfg(all(feature = "2d", not(feature = "3d")))]
        let depth_stencil = Some(depth_stencil_2d);

        #[cfg(all(feature = "3d", not(feature = "2d")))]
        let depth_stencil = Some(depth_stencil_3d);

        let format = if key.hdr {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };

        let hash = calc_func_id(&key);
        let label = format!("hanabi:pipeline:render_{hash:016X}");
        trace!(
            "-> creating pipeline '{}' with shader defs:{}",
            label,
            shader_defs
                .iter()
                .fold(String::new(), |acc, x| acc + &format!(" {x:?}"))
        );

        RenderPipelineDescriptor {
            label: Some(label.into()),
            vertex: VertexState {
                shader: key.shader.clone(),
                entry_point: Some("vertex".into()),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_buffer_layout.expect("Vertex buffer layout not present")],
            },
            fragment: Some(FragmentState {
                shader: key.shader,
                shader_defs,
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(key.alpha_mode.into()),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            layout,
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
            },
            depth_stencil,
            multisample: MultisampleState {
                count: key.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            push_constant_ranges: Vec::new(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

/// A single effect instance extracted from a [`ParticleEffect`] as a
/// render world item.
///
/// [`ParticleEffect`]: crate::ParticleEffect
#[derive(Debug, Clone, PartialEq, Component)]
#[require(CachedPipelines, CachedReadyState, CachedEffectMetadata)]
pub(crate) struct ExtractedEffect {
    /// Handle to the effect asset this instance is based on.
    /// The handle is weak to prevent refcount cycles and gracefully handle
    /// assets unloaded or destroyed after a draw call has been submitted.
    pub handle: Handle<EffectAsset>,
    /// Particle layout for the effect.
    pub particle_layout: ParticleLayout,
    /// Effect capacity, in number of particles.
    pub capacity: u32,
    /// Layout flags.
    pub layout_flags: LayoutFlags,
    /// Texture layout.
    pub texture_layout: TextureLayout,
    /// Textures.
    pub textures: Vec<Handle<Image>>,
    /// Alpha mode.
    pub alpha_mode: AlphaMode,
    /// Effect shaders.
    pub effect_shaders: EffectShader,
    /// Condition under which the effect is simulated.
    pub simulation_condition: SimulationCondition,
}

/// Extracted data for the [`GpuSpawnerParams`].
///
/// This contains all data which may change each frame during the regular usage
/// of the effect, but doesn't require any particular GPU resource update
/// (except re-uploading that new data to GPU, of course).
#[derive(Debug, Clone, PartialEq, Component)]
pub(crate) struct ExtractedSpawner {
    /// Number of particles to spawn this frame.
    ///
    /// This is ignored if the effect is a child effect consuming GPU spawn
    /// events.
    pub spawn_count: u32,
    /// PRNG seed.
    pub prng_seed: u32,
    /// Global transform of the effect origin.
    pub transform: GlobalTransform,
    /// Is the effect visible this frame?
    pub is_visible: bool,
}

/// Cache info for the metadata of the effect.
///
/// This manages the GPU allocation of the [`GpuEffectMetadata`] for this
/// effect.
#[derive(Debug, Default, Component)]
pub(crate) struct CachedEffectMetadata {
    /// Allocation ID.
    pub table_id: BufferTableId,
    /// Current metadata values, cached on CPU for change detection.
    pub metadata: GpuEffectMetadata,
}

/// Extracted parent information for a child effect.
///
/// This component is present on the [`RenderEntity`] of an extracted effect if
/// the effect has a parent effect. Otherwise, it's removed.
///
/// This components forms an ECS relationship with [`ChildrenEffects`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
#[relationship(relationship_target = ChildrenEffects)]
pub(crate) struct ChildEffectOf {
    /// Render entity of the parent.
    pub parent: Entity,
}

/// Extracted children information for a parent effect.
///
/// This component is present on the [`RenderEntity`] of an extracted effect if
/// the effect is a parent effect for one or more child effects. Otherwise, it's
/// removed.
///
/// This components forms an ECS relationship with [`ChildEffectOf`]. Note that
/// we don't use `linked_spawn` because:
/// 1. This would fight with the `SyncToRenderWorld` as the main world
///    parent-child hierarchy is by design not an ECS relationship (it's a lose
///    declarative coupling).
/// 2. The components on the render entity often store GPU resources or other
///    data we need to clean-up manually, and not all of them currently use
///    lifecycle hooks, so we want to manage despawning manually to prevent
///    leaks.
#[derive(Debug, Clone, PartialEq, Eq, Component)]
#[relationship_target(relationship = ChildEffectOf)]
pub(crate) struct ChildrenEffects(Vec<Entity>);

impl<'a> IntoIterator for &'a ChildrenEffects {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = std::slice::Iter<'a, Entity>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl Deref for ChildrenEffects {
    type Target = [Entity];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Extracted data for an effect's properties, if any.
///
/// This component is present on the [`RenderEntity`] of an extracted effect if
/// that effect has properties. It optionally contains new CPU data to
/// (re-)upload this frame. If the effect has no property, this component is
/// removed.
#[derive(Debug, Component)]
pub(crate) struct ExtractedProperties {
    /// Property layout for the effect.
    pub property_layout: PropertyLayout,
    /// Values of properties written in a binary blob according to
    /// [`property_layout`].
    ///
    /// This is `Some(blob)` if the data needs to be (re)uploaded to GPU, or
    /// `None` if nothing needs to be done for this frame.
    ///
    /// [`property_layout`]: crate::render::ExtractedEffect::property_layout
    pub property_data: Option<Vec<u8>>,
}

#[derive(Default, Resource)]
pub(crate) struct EffectAssetEvents {
    pub images: Vec<AssetEvent<Image>>,
}

/// System extracting all the asset events for the [`Image`] assets to enable
/// dynamic update of images bound to any effect.
///
/// This system runs in parallel of [`extract_effects`].
pub(crate) fn extract_effect_events(
    mut events: ResMut<EffectAssetEvents>,
    mut image_events: Extract<MessageReader<AssetEvent<Image>>>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("extract_effect_events").entered();
    trace!("extract_effect_events()");

    let EffectAssetEvents { ref mut images } = *events;
    *images = image_events.read().copied().collect();
}

/// Debugging settings.
///
/// Settings used to debug Hanabi. These have no effect on the actual behavior
/// of Hanabi, but may affect its performance.
///
/// # Example
///
/// ```
/// # use bevy::prelude::*;
/// # use bevy_hanabi::*;
/// fn startup(mut debug_settings: ResMut<DebugSettings>) {
///     // Each time a new effect is spawned, capture 2 frames
///     debug_settings.start_capture_on_new_effect = true;
///     debug_settings.capture_frame_count = 2;
/// }
/// ```
#[derive(Debug, Default, Clone, Copy, Resource)]
pub struct DebugSettings {
    /// Enable automatically starting a GPU debugger capture as soon as this
    /// frame starts rendering (extract phase).
    ///
    /// Enable this feature to automatically capture one or more GPU frames when
    /// the `extract_effects()` system runs next. This instructs any attached
    /// GPU debugger to start a capture; this has no effect if no debugger
    /// is attached.
    ///
    /// If a capture is already on-going this has no effect; the on-going
    /// capture needs to be terminated first. Note however that a capture can
    /// stop and another start in the same frame.
    ///
    /// This value is not reset automatically. If you set this to `true`, you
    /// should set it back to `false` on next frame to avoid capturing forever.
    pub start_capture_this_frame: bool,

    /// Enable automatically starting a GPU debugger capture when one or more
    /// effects are spawned.
    ///
    /// Enable this feature to automatically capture one or more GPU frames when
    /// a new effect is spawned (as detected by ECS change detection). This
    /// instructs any attached GPU debugger to start a capture; this has no
    /// effect if no debugger is attached.
    pub start_capture_on_new_effect: bool,

    /// Number of frames to capture with a GPU debugger.
    ///
    /// By default this value is zero, and a GPU debugger capture runs for a
    /// single frame. If a non-zero frame count is specified here, the capture
    /// will instead stop once the specified number of frames has been recorded.
    ///
    /// You should avoid setting this to a value too large, to prevent the
    /// capture size from getting out of control. A typical value is 1 to 3
    /// frames, or possibly more (up to 10) for exceptional contexts. Some GPU
    /// debuggers or graphics APIs might further limit this value on their own,
    /// so there's no guarantee the graphics API will honor this value.
    pub capture_frame_count: u32,
}

#[derive(Debug, Default, Clone, Copy, Resource)]
pub(crate) struct RenderDebugSettings {
    /// Is a GPU debugger capture on-going?
    is_capturing: bool,
    /// Start time of any on-going GPU debugger capture.
    capture_start: Duration,
    /// Number of frames captured so far for on-going GPU debugger capture.
    captured_frames: u32,
}

/// Manage GPU debug capture start/stop.
///
/// If any GPU debug capture is configured to start or stop in
/// [`DebugSettings`], they do so during this system's run. This ensures
/// that all GPU commands produced by Hanabi are recorded (but may miss some
/// from Bevy itself, if another Bevy system runs before this one).
///
/// We do this during extract to try and capture as close as possible to an
/// entire GPU frame.
pub(crate) fn start_stop_gpu_debug_capture(
    real_time: Extract<Res<Time<Real>>>,
    render_device: Res<RenderDevice>,
    debug_settings: Extract<Res<DebugSettings>>,
    mut render_debug_settings: ResMut<RenderDebugSettings>,
    q_added_effects: Extract<Query<(), Added<CompiledParticleEffect>>>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("start_stop_debug_capture").entered();
    trace!("start_stop_debug_capture()");

    // Stop any pending capture if needed
    if render_debug_settings.is_capturing {
        render_debug_settings.captured_frames += 1;

        if render_debug_settings.captured_frames >= debug_settings.capture_frame_count {
            #[expect(unsafe_code, reason = "Debugging only")]
            unsafe {
                render_device.wgpu_device().stop_graphics_debugger_capture();
            }
            render_debug_settings.is_capturing = false;
            warn!(
                "Stopped GPU debug capture after {} frames, at t={}s.",
                render_debug_settings.captured_frames,
                real_time.elapsed().as_secs_f64()
            );
        }
    }

    // If no pending capture, consider starting a new one
    if !render_debug_settings.is_capturing
        && (debug_settings.start_capture_this_frame
            || (debug_settings.start_capture_on_new_effect && !q_added_effects.is_empty()))
    {
        #[expect(unsafe_code, reason = "Debugging only")]
        unsafe {
            render_device
                .wgpu_device()
                .start_graphics_debugger_capture();
        }
        render_debug_settings.is_capturing = true;
        render_debug_settings.capture_start = real_time.elapsed();
        render_debug_settings.captured_frames = 0;
        warn!(
            "Started GPU debug capture of {} frames at t={}s.",
            debug_settings.capture_frame_count,
            render_debug_settings.capture_start.as_secs_f64()
        );
    }
}

/// Write the ready state of all render world effects back into their source
/// effect in the main world.
pub(crate) fn report_ready_state(
    mut main_world: ResMut<MainWorld>,
    q_ready_state: Query<&CachedReadyState>,
) {
    let mut q_effects = main_world.query::<(RenderEntity, &mut CompiledParticleEffect)>();
    for (render_entity, mut compiled_particle_effect) in q_effects.iter_mut(&mut main_world) {
        if let Ok(cached_ready_state) = q_ready_state.get(render_entity) {
            compiled_particle_effect.is_ready = cached_ready_state.is_ready();
        }
    }
}

/// System extracting data for rendering of all active [`ParticleEffect`]
/// components.
///
/// [`ParticleEffect`]: crate::ParticleEffect
pub(crate) fn extract_effects(
    mut commands: Commands,
    effects: Extract<Res<Assets<EffectAsset>>>,
    default_mesh: Extract<Res<DefaultMesh>>,
    // Main world effects to extract
    q_effects: Extract<
        Query<(
            Entity,
            RenderEntity,
            Option<&InheritedVisibility>,
            Option<&ViewVisibility>,
            &EffectSpawner,
            &CompiledParticleEffect,
            Option<Ref<EffectProperties>>,
            &GlobalTransform,
        )>,
    >,
    // Render world effects extracted from a previous frame, if any
    mut q_extracted_effects: Query<(
        &mut ExtractedEffect,
        Option<&mut ExtractedSpawner>,
        Option<&ChildEffectOf>, // immutable, because of relationship
        Option<&mut ExtractedEffectMesh>,
        Option<&mut ExtractedProperties>,
    )>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("extract_effects").entered();
    trace!("extract_effects()");

    // Loop over all existing effects to extract them
    trace!("Extracting {} effects...", q_effects.iter().len());
    for (
        main_entity,
        render_entity,
        maybe_inherited_visibility,
        maybe_view_visibility,
        effect_spawner,
        compiled_effect,
        maybe_properties,
        transform,
    ) in q_effects.iter()
    {
        // Check if shaders are configured
        let Some(effect_shaders) = compiled_effect.get_configured_shaders() else {
            trace!("Effect {:?}: no configured shader, skipped.", main_entity);
            continue;
        };

        // Check if asset is available, otherwise silently ignore
        let Some(asset) = effects.get(&compiled_effect.asset) else {
            trace!(
                "Effect {:?}: EffectAsset not ready, skipped. asset:{:?}",
                main_entity,
                compiled_effect.asset
            );
            continue;
        };

        let is_visible = maybe_inherited_visibility
            .map(|cv| cv.get())
            .unwrap_or(true)
            && maybe_view_visibility.map(|cv| cv.get()).unwrap_or(true);

        let mut cmd = commands.entity(render_entity);

        // Fetch the existing extraction compoennts, if any, which we need to update.
        // Because we use SyncToRenderWorld, there's always a render entity, but it may
        // miss all components. And because we can't query only optional components
        // (that would match all entities in the entire world), we force querying
        // ExtractedEffect, which means we get a miss if it's the first extraction and
        // it's not spawned yet. That's OK, we'll spawn it below.
        let (
            maybe_extracted_effect,
            maybe_extracted_spawner,
            maybe_child_of,
            maybe_extracted_mesh,
            maybe_extracted_properties,
        ) = q_extracted_effects
            .get_mut(render_entity)
            .map(|(extracted_effect, b, c, d, e)| (Some(extracted_effect), b, c, d, e))
            .unwrap_or((None, None, None, None, None));

        // Extract general effect data
        let texture_layout = asset.module().texture_layout();
        let layout_flags = compiled_effect.layout_flags;
        let alpha_mode = compiled_effect.alpha_mode;
        trace!(
            "Extracted instance of effect '{}' on entity {:?} (render entity {:?}): texture_layout_count={} texture_count={} layout_flags={:?}",
            asset.name,
            main_entity,
            render_entity,
            texture_layout.layout.len(),
            compiled_effect.textures.len(),
            layout_flags,
        );
        let new_extracted_effect = ExtractedEffect {
            handle: compiled_effect.asset.clone(),
            particle_layout: asset.particle_layout().clone(),
            capacity: asset.capacity(),
            layout_flags,
            texture_layout,
            textures: compiled_effect.textures.clone(),
            alpha_mode,
            effect_shaders: effect_shaders.clone(),
            simulation_condition: asset.simulation_condition,
        };
        if let Some(mut extracted_effect) = maybe_extracted_effect {
            extracted_effect.set_if_neq(new_extracted_effect);
        } else {
            trace!(
                "Inserting new ExtractedEffect component on {:?}",
                render_entity
            );
            cmd.insert(new_extracted_effect);
        }

        // Extract the spawner data
        let new_spawner = ExtractedSpawner {
            spawn_count: effect_spawner.spawn_count,
            prng_seed: compiled_effect.prng_seed,
            transform: *transform,
            is_visible,
        };
        trace!(
            "[Effect {}] spawn_count={} prng_seed={}",
            render_entity,
            new_spawner.spawn_count,
            new_spawner.prng_seed
        );
        if let Some(mut extracted_spawner) = maybe_extracted_spawner {
            extracted_spawner.set_if_neq(new_spawner);
        } else {
            trace!(
                "Inserting new ExtractedSpawner component on {}",
                render_entity
            );
            cmd.insert(new_spawner);
        }

        // Extract the effect mesh
        let mesh = compiled_effect
            .mesh
            .clone()
            .unwrap_or(default_mesh.0.clone());
        let new_mesh = ExtractedEffectMesh { mesh: mesh.id() };
        if let Some(mut extracted_mesh) = maybe_extracted_mesh {
            extracted_mesh.set_if_neq(new_mesh);
        } else {
            trace!(
                "Inserting new ExtractedEffectMesh component on {:?}",
                render_entity
            );
            cmd.insert(new_mesh);
        }

        // Extract the parent, if any, and resolve its render entity
        let parent_render_entity = if let Some(main_entity) = compiled_effect.parent {
            let Ok((_, render_entity, _, _, _, _, _, _)) = q_effects.get(main_entity) else {
                error!(
                    "Failed to resolve render entity of parent with main entity {:?}.",
                    main_entity
                );
                cmd.remove::<ChildEffectOf>();
                // TODO - prevent extraction altogether here, instead of just de-parenting?
                continue;
            };
            Some(render_entity)
        } else {
            None
        };
        if let Some(render_entity) = parent_render_entity {
            let new_child_of = ChildEffectOf {
                parent: render_entity,
            };
            // If there's already an ExtractedParent component, ensure we overwrite only if
            // different, to not trigger ECS change detection that we rely on.
            if let Some(child_effect_of) = maybe_child_of {
                // The relationship makes ChildEffectOf immutable, so re-insert to mutate
                if *child_effect_of != new_child_of {
                    cmd.insert(new_child_of);
                }
            } else {
                trace!(
                    "Inserting new ChildEffectOf component on {:?}",
                    render_entity
                );
                cmd.insert(new_child_of);
            }
        } else {
            cmd.remove::<ChildEffectOf>();
        }

        // Extract property data
        let property_layout = asset.property_layout();
        if property_layout.is_empty() {
            cmd.remove::<ExtractedProperties>();
        } else {
            // Re-extract CPU property data if any. Note that this data is not a "new value"
            // but instead a "value that must be uploaded this frame", and therefore is
            // empty when there's no change (as opposed to, having a constant value
            // frame-to-frame).
            let property_data = if let Some(properties) = maybe_properties {
                if properties.is_changed() {
                    trace!("Detected property change, re-serializing...");
                    Some(properties.serialize(&property_layout))
                } else {
                    None
                }
            } else {
                None
            };

            let new_properties = ExtractedProperties {
                property_layout,
                property_data,
            };
            trace!("new_properties = {new_properties:?}");

            if let Some(mut extracted_properties) = maybe_extracted_properties {
                // Always mutate if there's new CPU data to re-upload. Otherwise check for any
                // other change.
                if new_properties.property_data.is_some()
                    || (extracted_properties.property_layout != new_properties.property_layout)
                {
                    trace!(
                        "Updating existing ExtractedProperties (was: {:?})",
                        extracted_properties.as_ref()
                    );
                    *extracted_properties = new_properties;
                }
            } else {
                trace!(
                    "Inserting new ExtractedProperties component on {:?}",
                    render_entity
                );
                cmd.insert(new_properties);
            }
        }
    }
}

pub(crate) fn extract_sim_params(
    real_time: Extract<Res<Time<Real>>>,
    virtual_time: Extract<Res<Time<Virtual>>>,
    time: Extract<Res<Time<EffectSimulation>>>,
    mut sim_params: ResMut<SimParams>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("extract_sim_params").entered();
    trace!("extract_sim_params()");

    // Save simulation params into render world
    sim_params.time = time.elapsed_secs_f64();
    sim_params.delta_time = time.delta_secs();
    sim_params.virtual_time = virtual_time.elapsed_secs_f64();
    sim_params.virtual_delta_time = virtual_time.delta_secs();
    sim_params.real_time = real_time.elapsed_secs_f64();
    sim_params.real_delta_time = real_time.delta_secs();
    trace!(
        "SimParams: time={} delta_time={} vtime={} delta_vtime={} rtime={} delta_rtime={}",
        sim_params.time,
        sim_params.delta_time,
        sim_params.virtual_time,
        sim_params.virtual_delta_time,
        sim_params.real_time,
        sim_params.real_delta_time,
    );
}

/// Various GPU limits and aligned sizes computed once and cached.
struct GpuLimits {
    /// Value of [`WgpuLimits::min_storage_buffer_offset_alignment`].
    ///
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`]: bevy::render::settings::WgpuLimits::min_storage_buffer_offset_alignment
    storage_buffer_align: NonZeroU32,

    /// Size of [`GpuEffectMetadata`] aligned to the contraint of
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`].
    ///
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`]: bevy::render::settings::WgpuLimits::min_storage_buffer_offset_alignment
    effect_metadata_aligned_size: NonZeroU32,
}

impl GpuLimits {
    pub fn from_device(render_device: &RenderDevice) -> Self {
        let storage_buffer_align =
            render_device.limits().min_storage_buffer_offset_alignment as u64;

        let effect_metadata_aligned_size = NonZeroU32::new(
            GpuEffectMetadata::min_size()
                .get()
                .next_multiple_of(storage_buffer_align) as u32,
        )
        .unwrap();

        trace!(
            "GPU-aligned sizes (align: {} B):\n- GpuEffectMetadata: {} B -> {} B",
            storage_buffer_align,
            GpuEffectMetadata::min_size().get(),
            effect_metadata_aligned_size.get(),
        );

        Self {
            storage_buffer_align: NonZeroU32::new(storage_buffer_align as u32).unwrap(),
            effect_metadata_aligned_size,
        }
    }

    /// Byte alignment for any storage buffer binding.
    pub fn storage_buffer_align(&self) -> NonZeroU32 {
        self.storage_buffer_align
    }

    /// Byte offset of the [`GpuEffectMetadata`] of a given buffer.
    pub fn effect_metadata_offset(&self, buffer_index: u32) -> u64 {
        self.effect_metadata_aligned_size.get() as u64 * buffer_index as u64
    }
}

/// Global render world resource containing the GPU data to draw all the
/// particle effects in all views.
///
/// The resource is populated by [`prepare_effects()`] with all the effects to
/// render for the current frame, for all views in the frame, and consumed by
/// [`queue_effects()`] to actually enqueue the drawning commands to draw those
/// effects.
#[derive(Resource)]
pub struct EffectsMeta {
    /// Bind group for the camera view, containing the camera projection and
    /// other uniform values related to the camera.
    view_bind_group: Option<BindGroup>,
    /// Bind group #0 of the vfx_update shader, for the simulation parameters
    /// like the current time and frame delta time.
    update_sim_params_bind_group: Option<BindGroup>,
    /// Bind group #0 of the vfx_indirect shader, for the simulation parameters
    /// like the current time and frame delta time. This is shared with the
    /// vfx_init pass too.
    indirect_sim_params_bind_group: Option<BindGroup>,
    /// Bind group #1 of the vfx_indirect shader, containing both the indirect
    /// compute dispatch and render buffers.
    indirect_metadata_bind_group: Option<BindGroup>,
    /// Bind group #2 of the vfx_indirect shader, containing the spawners.
    indirect_spawner_bind_group: Option<BindGroup>,
    /// Global shared GPU uniform buffer storing the simulation parameters,
    /// uploaded each frame from CPU to GPU.
    sim_params_uniforms: UniformBuffer<GpuSimParams>,
    /// Global shared GPU buffer storing the various spawner parameter structs
    /// for the active effect instances.
    spawner_buffer: AlignedBufferVec<GpuSpawnerParams>,
    /// Global shared GPU buffer storing the various indirect dispatch structs
    /// for the indirect dispatch of the Update pass.
    dispatch_indirect_buffer: GpuBuffer<GpuDispatchIndirectArgs>,
    /// Global shared GPU buffer storing the various indirect draw structs
    /// for the indirect Render pass. Note that we use
    /// GpuDrawIndexedIndirectArgs as the largest of the two variants (the
    /// other being GpuDrawIndirectArgs). For non-indexed entries, we ignore
    /// the last `u32` value.
    draw_indirect_buffer: BufferTable<GpuDrawIndexedIndirectArgs>,
    /// Global shared GPU buffer storing the various `EffectMetadata`
    /// structs for the active effect instances.
    effect_metadata_buffer: BufferTable<GpuEffectMetadata>,
    /// Various GPU limits and aligned sizes lazily allocated and cached for
    /// convenience.
    gpu_limits: GpuLimits,
    indirect_shader_noevent: Handle<Shader>,
    indirect_shader_events: Handle<Shader>,
    /// Pipeline cache ID of the two indirect dispatch pass pipelines (the
    /// -noevent and -events variants).
    indirect_pipeline_ids: [CachedComputePipelineId; 2],
    /// Pipeline cache ID of the active indirect dispatch pass pipeline, which
    /// is either the -noevent or -events variant depending on whether there's
    /// any child effect with GPU events currently active.
    active_indirect_pipeline_id: CachedComputePipelineId,
}

impl EffectsMeta {
    pub fn new(
        device: RenderDevice,
        indirect_shader_noevent: Handle<Shader>,
        indirect_shader_events: Handle<Shader>,
    ) -> Self {
        let gpu_limits = GpuLimits::from_device(&device);

        // Ensure individual GpuSpawnerParams elements are properly aligned so they can
        // be addressed individually by the computer shaders.
        let item_align = gpu_limits.storage_buffer_align();
        trace!(
            "Aligning storage buffers to {} bytes as device limits requires.",
            item_align.get()
        );

        Self {
            view_bind_group: None,
            update_sim_params_bind_group: None,
            indirect_sim_params_bind_group: None,
            indirect_metadata_bind_group: None,
            indirect_spawner_bind_group: None,
            sim_params_uniforms: UniformBuffer::default(),
            spawner_buffer: AlignedBufferVec::new(
                BufferUsages::STORAGE,
                Some(item_align.into()),
                Some("hanabi:buffer:spawner".to_string()),
            ),
            dispatch_indirect_buffer: GpuBuffer::new(
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
                Some("hanabi:buffer:dispatch_indirect".to_string()),
            ),
            draw_indirect_buffer: BufferTable::new(
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
                Some(GpuDrawIndexedIndirectArgs::SHADER_SIZE),
                Some("hanabi:buffer:draw_indirect".to_string()),
            ),
            effect_metadata_buffer: BufferTable::new(
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
                Some(item_align.into()),
                Some("hanabi:buffer:effect_metadata".to_string()),
            ),
            gpu_limits,
            indirect_shader_noevent,
            indirect_shader_events,
            indirect_pipeline_ids: [
                CachedComputePipelineId::INVALID,
                CachedComputePipelineId::INVALID,
            ],
            active_indirect_pipeline_id: CachedComputePipelineId::INVALID,
        }
    }

    pub fn allocate_spawner(
        &mut self,
        global_transform: &GlobalTransform,
        spawn_count: u32,
        prng_seed: u32,
        slab_offset: u32,
        parent_slab_offset: Option<u32>,
        effect_metadata_buffer_table_id: BufferTableId,
        maybe_cached_draw_indirect_args: Option<&CachedDrawIndirectArgs>,
    ) -> u32 {
        let spawner_base = self.spawner_buffer.len() as u32;
        let transform = global_transform.to_matrix().into();
        let inverse_transform = Mat4::from(
            // Inverse the Affine3A first, then convert to Mat4. This is a lot more
            // efficient than inversing the Mat4.
            global_transform.affine().inverse(),
        )
        .into();
        let spawner_params = GpuSpawnerParams {
            transform,
            inverse_transform,
            spawn: spawn_count as i32,
            seed: prng_seed,
            effect_metadata_index: effect_metadata_buffer_table_id.0,
            draw_indirect_index: maybe_cached_draw_indirect_args
                .map(|cdia| cdia.get_row().0)
                .unwrap_or_default(),
            slab_offset,
            parent_slab_offset: parent_slab_offset.unwrap_or(u32::MAX),
            ..default()
        };
        trace!("spawner params = {:?}", spawner_params);
        self.spawner_buffer.push(spawner_params);
        spawner_base
    }

    pub fn allocate_draw_indirect(
        &mut self,
        draw_args: &AnyDrawIndirectArgs,
    ) -> CachedDrawIndirectArgs {
        let row = self
            .draw_indirect_buffer
            .insert(draw_args.bitcast_to_row_entry());
        CachedDrawIndirectArgs {
            row,
            args: *draw_args,
        }
    }

    pub fn update_draw_indirect(&mut self, row_index: &CachedDrawIndirectArgs) {
        self.draw_indirect_buffer
            .update(row_index.get_row(), row_index.args.bitcast_to_row_entry());
    }

    pub fn free_draw_indirect(&mut self, row_index: &CachedDrawIndirectArgs) {
        self.draw_indirect_buffer.remove(row_index.get_row());
    }
}

bitflags! {
    /// Effect flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct LayoutFlags: u32 {
        /// No flags.
        const NONE = 0;
        // DEPRECATED - The effect uses an image texture.
        //const PARTICLE_TEXTURE = (1 << 0);
        /// The effect is simulated in local space.
        const LOCAL_SPACE_SIMULATION = (1 << 2);
        /// The effect uses alpha masking instead of alpha blending. Only used for 3D.
        const USE_ALPHA_MASK = (1 << 3);
        /// The effect is rendered with flipbook texture animation based on the
        /// [`Attribute::SPRITE_INDEX`] of each particle.
        const FLIPBOOK = (1 << 4);
        /// The effect needs UVs.
        const NEEDS_UV = (1 << 5);
        /// The effect has ribbons.
        const RIBBONS = (1 << 6);
        /// The effects needs normals.
        const NEEDS_NORMAL = (1 << 7);
        /// The effect is fully-opaque.
        const OPAQUE = (1 << 8);
        /// The (update) shader emits GPU spawn events to instruct another effect to spawn particles.
        const EMIT_GPU_SPAWN_EVENTS = (1 << 9);
        /// The (init) shader spawns particles by consuming GPU spawn events, instead of
        /// a single CPU spawn count.
        const CONSUME_GPU_SPAWN_EVENTS = (1 << 10);
        /// The (init or update) shader needs access to its parent particle. This allows
        /// a particle init or update pass to read the data of a parent particle, for
        /// example to inherit some of the attributes.
        const READ_PARENT_PARTICLE = (1 << 11);
        /// The effect access to the particle data in the fragment shader.
        const NEEDS_PARTICLE_FRAGMENT = (1 << 12);
    }
}

impl Default for LayoutFlags {
    fn default() -> Self {
        Self::NONE
    }
}

/// Observer raised when the [`CachedEffect`] component is removed, which
/// indicates that the effect instance was despawned.
pub(crate) fn on_remove_cached_effect(
    trigger: On<Remove, CachedEffect>,
    query: Query<(
        Entity,
        &MainEntity,
        &CachedEffect,
        &DispatchBufferIndices,
        Option<&CachedEffectProperties>,
        Option<&CachedParentInfo>,
        Option<&CachedEffectEvents>,
    )>,
    mut effect_cache: ResMut<EffectCache>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    mut effects_meta: ResMut<EffectsMeta>,
    mut event_cache: ResMut<EventCache>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("on_remove_cached_effect").entered();

    // FIXME - review this Observer pattern; this triggers for each event one by
    // one, which could kill performance if many effects are removed.

    // Fecth the components of the effect being destroyed. Note that the despawn
    // command above is not yet applied, so this query should always succeed.
    let Ok((
        render_entity,
        main_entity,
        cached_effect,
        dispatch_buffer_indices,
        _opt_props,
        _opt_parent,
        opt_cached_effect_events,
    )) = query.get(trigger.event().entity)
    else {
        return;
    };

    // Dealllocate the effect slice in the event buffer, if any.
    if let Some(cached_effect_events) = opt_cached_effect_events {
        match event_cache.free(cached_effect_events) {
            Err(err) => {
                error!("Error while freeing effect event slice: {err:?}");
            }
            Ok(buffer_state) => {
                if buffer_state != SlabState::Used {
                    // Clear bind groups associated with the old buffer
                    effect_bind_groups.init_metadata_bind_groups.clear();
                    effect_bind_groups.update_metadata_bind_groups.clear();
                }
            }
        }
    }

    // Deallocate the effect slice in the GPU effect buffer, and if this was the
    // last slice, also deallocate the GPU buffer itself.
    trace!(
        "=> ParticleEffect on render entity {:?} associated with main entity {:?}, removing...",
        render_entity,
        main_entity,
    );
    let Ok(SlabState::Free) = effect_cache.remove(cached_effect) else {
        // Buffer was not affected, so all bind groups are still valid. Nothing else to
        // do.
        return;
    };

    // Clear bind groups associated with the removed buffer
    trace!(
        "=> GPU particle slab #{} gone, destroying its bind groups...",
        cached_effect.slab_id.index()
    );
    effect_bind_groups
        .particle_slabs
        .remove(&cached_effect.slab_id);
    effects_meta
        .dispatch_indirect_buffer
        .free(dispatch_buffer_indices.update_dispatch_indirect_buffer_row_index);
}

/// Observer raised when the [`CachedEffectMetadata`] component is removed, to
/// deallocate the GPU resources associated with the indirect draw args.
pub(crate) fn on_remove_cached_metadata(
    trigger: On<Remove, CachedEffectMetadata>,
    query: Query<&CachedEffectMetadata>,
    mut effects_meta: ResMut<EffectsMeta>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("on_remove_cached_metadata").entered();

    if let Ok(cached_metadata) = query.get(trigger.event().entity) {
        if cached_metadata.table_id.is_valid() {
            effects_meta
                .effect_metadata_buffer
                .remove(cached_metadata.table_id);
        }
    };
}

/// Observer raised when the [`CachedDrawIndirectArgs`] component is removed, to
/// deallocate the GPU resources associated with the indirect draw args.
pub(crate) fn on_remove_cached_draw_indirect_args(
    trigger: On<Remove, CachedDrawIndirectArgs>,
    query: Query<&CachedDrawIndirectArgs>,
    mut effects_meta: ResMut<EffectsMeta>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("on_remove_cached_draw_indirect_args").entered();

    if let Ok(cached_draw_args) = query.get(trigger.event().entity) {
        effects_meta.free_draw_indirect(cached_draw_args);
    };
}

/// Clear pending GPU resources left from previous frame.
///
/// Those generally are source buffers for buffer-to-buffer copies on capacity
/// growth, which need the source buffer to be alive until the copy is done,
/// then can be discarded here.
pub(crate) fn clear_previous_frame_resizes(
    mut effects_meta: ResMut<EffectsMeta>,
    mut sort_bind_groups: ResMut<SortBindGroups>,
    mut init_fill_dispatch_queue: ResMut<InitFillDispatchQueue>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("clear_previous_frame_resizes").entered();
    trace!("clear_previous_frame_resizes");

    init_fill_dispatch_queue.clear();

    // Clear last frame's buffer resizes which may have occured during last frame,
    // during `Node::run()` while the `BufferTable` could not be mutated. This is
    // the first point at which we can do that where we're not blocking the main
    // world (so, excluding the extract system).
    effects_meta
        .dispatch_indirect_buffer
        .clear_previous_frame_resizes();
    effects_meta
        .draw_indirect_buffer
        .clear_previous_frame_resizes();
    effects_meta
        .effect_metadata_buffer
        .clear_previous_frame_resizes();
    sort_bind_groups.clear_previous_frame_resizes();
}

// Fixup the [`CachedChildInfo::global_child_index`] once all child infos have
// been allocated.
pub fn fixup_parents(
    q_changed_parents: Query<(Entity, Ref<CachedParentInfo>)>,
    mut q_children: Query<&mut CachedChildInfo>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("fixup_parents").entered();
    trace!("fixup_parents");

    // Once all parents are (re-)allocated, fix up the global index of all
    // children if the parent base index changed.
    trace!(
        "Updating the global index of children of parent effects whose child list just changed..."
    );
    for (parent_entity, cached_parent_info) in q_changed_parents.iter() {
        let base_index =
            cached_parent_info.byte_range.start / GpuChildInfo::SHADER_SIZE.get() as u32;
        let parent_changed = cached_parent_info.is_changed();
        trace!(
            "Updating {} children of parent effect {:?} with base child index {} (parent_changed:{})...",
            cached_parent_info.children.len(),
            parent_entity,
            base_index,
            parent_changed
        );
        for (child_entity, _) in &cached_parent_info.children {
            let Ok(mut cached_child_info) = q_children.get_mut(*child_entity) else {
                error!(
                    "Cannot find child {:?} declared by parent {:?}",
                    *child_entity, parent_entity
                );
                continue;
            };
            if !cached_child_info.is_changed() && !parent_changed {
                continue;
            }
            cached_child_info.global_child_index = base_index + cached_child_info.local_child_index;
            trace!(
                "+ Updated global index for child ID {:?} of parent {:?}: local={}, global={}",
                child_entity,
                parent_entity,
                cached_child_info.local_child_index,
                cached_child_info.global_child_index
            );
        }
    }
}

/// Allocate the GPU resources for all extracted effects.
///
/// This adds the [`CachedEffect`] component as needed, and update it with the
/// allocation in the [`EffectCache`].
pub fn allocate_effects(
    mut commands: Commands,
    mut q_extracted_effects: Query<
        (
            Entity,
            &ExtractedEffect,
            Has<ChildEffectOf>,
            Option<&mut CachedEffect>,
            Has<DispatchBufferIndices>,
        ),
        Changed<ExtractedEffect>,
    >,
    mut effect_cache: ResMut<EffectCache>,
    mut effects_meta: ResMut<EffectsMeta>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("allocate_effects").entered();
    trace!("allocate_effects");

    for (entity, extracted_effect, has_parent, maybe_cached_effect, has_dispatch_buffer_indices) in
        &mut q_extracted_effects
    {
        // Insert or update the effect into the EffectCache
        if let Some(mut cached_effect) = maybe_cached_effect {
            trace!("Updating EffectCache entry for entity {entity:?}...");
            let _ = effect_cache.remove(cached_effect.as_ref());
            *cached_effect = effect_cache.insert(
                extracted_effect.handle.clone(),
                extracted_effect.capacity,
                &extracted_effect.particle_layout,
            );
        } else {
            trace!("Allocating new entry in EffectCache for entity {entity:?}...");
            let cached_effect = effect_cache.insert(
                extracted_effect.handle.clone(),
                extracted_effect.capacity,
                &extracted_effect.particle_layout,
            );
            commands.entity(entity).insert(cached_effect);
        }

        // Ensure the particle@1 bind group layout exists for the given configuration of
        // particle layout. We do this here only for effects without a parent; for those
        // with a parent, we'll do it after we resolved that parent.
        if !has_parent {
            let parent_min_binding_size = None;
            effect_cache.ensure_particle_bind_group_layout_desc(
                extracted_effect.particle_layout.min_binding_size32(),
                parent_min_binding_size,
            );
        }

        // Ensure the metadata@3 bind group layout exists for the init pass.
        {
            let consume_gpu_spawn_events = extracted_effect
                .layout_flags
                .contains(LayoutFlags::CONSUME_GPU_SPAWN_EVENTS);
            effect_cache.ensure_metadata_init_bind_group_layout_desc(consume_gpu_spawn_events);
        }

        // Allocate DispatchBufferIndices if not present yet
        if !has_dispatch_buffer_indices {
            let update_dispatch_indirect_buffer_row_index =
                effects_meta.dispatch_indirect_buffer.allocate();
            commands.entity(entity).insert(DispatchBufferIndices {
                update_dispatch_indirect_buffer_row_index,
            });
        }
    }
}

/// Update any cached mesh info based on any relocation done by Bevy itself.
///
/// Bevy will merge small meshes into larger GPU buffers automatically. When
/// this happens, the mesh location changes, and we need to update our
/// references to it in order to know how to issue the draw commands.
///
/// This system updates both the [`CachedMeshLocation`] and the
/// [`CachedIndirectDrawArgs`] components.
pub fn update_mesh_locations(
    mut commands: Commands,
    mut effects_meta: ResMut<EffectsMeta>,
    mesh_allocator: Res<MeshAllocator>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    mut q_cached_effects: Query<(
        Entity,
        &ExtractedEffectMesh,
        Option<&mut CachedMeshLocation>,
        Option<&mut CachedDrawIndirectArgs>,
    )>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("update_mesh_locations").entered();
    trace!("update_mesh_locations");

    for (entity, extracted_mesh, maybe_cached_mesh_location, maybe_cached_draw_indirect_args) in
        &mut q_cached_effects
    {
        let mut cmds = commands.entity(entity);

        // Resolve the render mesh
        let Some(render_mesh) = render_meshes.get(extracted_mesh.mesh) else {
            warn!(
                "Cannot find render mesh of particle effect instance on entity {:?}, despite applying default mesh. Invalid asset handle: {:?}",
                entity, extracted_mesh.mesh
            );
            cmds.remove::<CachedMeshLocation>();
            continue;
        };

        // Find the location where the render mesh was allocated. This is handled by
        // Bevy itself in the allocate_and_free_meshes() system. Bevy might
        // re-batch the vertex and optional index data of meshes together at any point,
        // so we need to confirm that the location data we may have cached is still
        // valid.
        let Some(mesh_vertex_buffer_slice) = mesh_allocator.mesh_vertex_slice(&extracted_mesh.mesh)
        else {
            trace!(
                "Effect main_entity {:?}: cannot find vertex slice of render mesh {:?}",
                entity,
                extracted_mesh.mesh
            );
            cmds.remove::<CachedMeshLocation>();
            continue;
        };
        let mesh_index_buffer_slice = mesh_allocator.mesh_index_slice(&extracted_mesh.mesh);
        let indexed =
            if let RenderMeshBufferInfo::Indexed { index_format, .. } = render_mesh.buffer_info {
                if let Some(ref slice) = mesh_index_buffer_slice {
                    Some(MeshIndexSlice {
                        format: index_format,
                        buffer: slice.buffer.clone(),
                        range: slice.range.clone(),
                    })
                } else {
                    trace!(
                        "Effect main_entity {:?}: cannot find index slice of render mesh {:?}",
                        entity,
                        extracted_mesh.mesh
                    );
                    cmds.remove::<CachedMeshLocation>();
                    continue;
                }
            } else {
                None
            };

        // Calculate the new draw args and mesh location based on Bevy's info
        let new_draw_args = AnyDrawIndirectArgs::from_slices(
            &mesh_vertex_buffer_slice,
            mesh_index_buffer_slice.as_ref(),
        );
        let new_mesh_location = match &mesh_index_buffer_slice {
            // Indexed mesh rendering
            Some(mesh_index_buffer_slice) => CachedMeshLocation {
                vertex_buffer: mesh_vertex_buffer_slice.buffer.id(),
                vertex_or_index_count: mesh_index_buffer_slice.range.len() as u32,
                first_index_or_vertex_offset: mesh_index_buffer_slice.range.start,
                vertex_offset_or_base_instance: mesh_vertex_buffer_slice.range.start as i32,
                indexed,
            },
            // Non-indexed mesh rendering
            None => CachedMeshLocation {
                vertex_buffer: mesh_vertex_buffer_slice.buffer.id(),
                vertex_or_index_count: mesh_vertex_buffer_slice.range.len() as u32,
                first_index_or_vertex_offset: mesh_vertex_buffer_slice.range.start,
                vertex_offset_or_base_instance: 0,
                indexed: None,
            },
        };

        // We don't allocate the draw indirect args ahead of time because we need to
        // select the indexed vs. non-indexed buffer. Now that we know whether the mesh
        // is indexed, we can allocate it (or reallocate it if indexing mode changed).
        if let Some(mut cached_draw_indirect) = maybe_cached_draw_indirect_args {
            assert!(cached_draw_indirect.row.is_valid());

            // If the GPU draw args changed, re-upload to GPU.
            if new_draw_args != cached_draw_indirect.args {
                debug!(
                    "Indirect draw args changed for asset {:?}\nold:{:?}\nnew:{:?}",
                    entity, cached_draw_indirect.args, new_draw_args
                );
                cached_draw_indirect.args = new_draw_args;
                effects_meta.update_draw_indirect(cached_draw_indirect.as_ref());
            }
        } else {
            cmds.insert(effects_meta.allocate_draw_indirect(&new_draw_args));
        }

        // Compare to any cached data and update if necessary, or insert if missing.
        // This will trigger change detection in the ECS, which will in turn trigger
        // GpuEffectMetadata re-upload.
        if let Some(mut old_mesh_location) = maybe_cached_mesh_location {
            if *old_mesh_location != new_mesh_location {
                debug!(
                    "Mesh location changed for asset {:?}\nold:{:?}\nnew:{:?}",
                    entity, old_mesh_location, new_mesh_location
                );
                *old_mesh_location = new_mesh_location;
            }
        } else {
            cmds.insert(new_mesh_location);
        }
    }
}

/// Allocate an entry in the GPU table for any [`CachedEffectMetadata`] missing
/// one.
///
/// This system does NOT take care of (re-)uploading recent CPU data to GPU.
/// This is done much later in the frame, after batching and once all data for
/// it is ready. But it's necessary to ensure the allocation is determined
/// already ahead of time, in order to do batching of contiguous metadata
/// blocks (TODO; not currently used, also may end up using binary search in
/// shader, in which case we won't need continguous-ness and can maybe remove
/// this system).
// TODO - consider using observer OnAdd instead?
pub fn allocate_metadata(
    mut effects_meta: ResMut<EffectsMeta>,
    mut q_metadata: Query<&mut CachedEffectMetadata>,
) {
    for mut metadata in &mut q_metadata {
        if !metadata.table_id.is_valid() {
            metadata.table_id = effects_meta
                .effect_metadata_buffer
                .insert(metadata.metadata);
        } else {
            // Unless this is the first time we allocate the GPU entry (above),
            // we should never reach the beginning of this frame
            // with a changed metadata which has not
            // been re-uploaded last frame.
            // NO! We can only detect the change *since last run of THIS system*
            // so wont' see that a latter system the data.
            // assert!(!metadata.is_changed());
        }
    }
}

/// Update the [`CachedParentInfo`] of parent effects and the
/// [`CachedChildInfo`] of child effects.
pub fn allocate_parent_child_infos(
    mut commands: Commands,
    mut effect_cache: ResMut<EffectCache>,
    mut event_cache: ResMut<EventCache>,
    // All extracted child effects. May or may not already have a CachedChildInfo. If not, this
    // will be spawned below.
    mut q_child_effects: Query<(
        Entity,
        &ExtractedEffect,
        &ChildEffectOf,
        &CachedEffectEvents,
        Option<&mut CachedChildInfo>,
    )>,
    // All parent effects from a previous frame (already have CachedParentInfo), which can be
    // updated in-place without spawning a new CachedParentInfo.
    mut q_parent_effects: Query<(
        Entity,
        &ExtractedEffect,
        &CachedEffect,
        &ChildrenEffects,
        Option<&mut CachedParentInfo>,
    )>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("allocate_child_infos").entered();
    trace!("allocate_child_infos");

    // Loop on all child effects and ensure their CachedChildInfo is up-to-date.
    for (child_entity, _, child_effect_of, cached_effect_events, maybe_cached_child_info) in
        &mut q_child_effects
    {
        // Fetch the parent effect
        let parent_entity = child_effect_of.parent;
        let Ok((_, _, parent_cached_effect, children_effects, _)) =
            q_parent_effects.get(parent_entity)
        else {
            warn!("Unknown parent #{parent_entity:?} on child entity {child_entity:?}, removing CachedChildInfo.");
            if maybe_cached_child_info.is_some() {
                commands.entity(child_entity).remove::<CachedChildInfo>();
            }
            continue;
        };

        // Find the index of this child entity in its parent's storage
        let Some(local_child_index) = children_effects.0.iter().position(|e| *e == child_entity)
        else {
            warn!("Cannot find child entity {child_entity:?} in the children collection of parent entity {parent_entity:?}. Relationship desync?");
            if maybe_cached_child_info.is_some() {
                commands.entity(child_entity).remove::<CachedChildInfo>();
            }
            continue;
        };
        let local_child_index = local_child_index as u32;

        // Fetch the effect buffer of the parent effect
        let Some(parent_buffer_binding_source) = effect_cache
            .get_slab(&parent_cached_effect.slab_id)
            .map(|effect_buffer| effect_buffer.max_binding_source())
        else {
            warn!(
                "Unknown parent slab #{} on parent entity {:?}, removing CachedChildInfo.",
                parent_cached_effect.slab_id.index(),
                parent_entity
            );
            if maybe_cached_child_info.is_some() {
                commands.entity(child_entity).remove::<CachedChildInfo>();
            }
            continue;
        };

        let new_cached_child_info = CachedChildInfo {
            parent_slab_id: parent_cached_effect.slab_id,
            parent_slab_offset: parent_cached_effect.slice.range().start,
            parent_particle_layout: parent_cached_effect.slice.particle_layout.clone(),
            parent_buffer_binding_source,
            local_child_index,
            global_child_index: u32::MAX, // fixed up later by fixup_parents()
            init_indirect_dispatch_index: cached_effect_events.init_indirect_dispatch_index,
        };
        if let Some(mut cached_child_info) = maybe_cached_child_info {
            if !cached_child_info.is_locally_equal(&new_cached_child_info) {
                *cached_child_info = new_cached_child_info;
            }
        } else {
            commands.entity(child_entity).insert(new_cached_child_info);
        }
    }

    // Loop on all parent effects and ensure their CachedParentInfo is up-to-date.
    for (parent_entity, parent_extracted_effect, _, children_effects, maybe_cached_parent_info) in
        &mut q_parent_effects
    {
        let parent_min_binding_size = parent_extracted_effect.particle_layout.min_binding_size32();

        // Loop over children and gather GpuChildInfo
        let mut new_children = Vec::with_capacity(children_effects.0.len());
        let mut new_child_infos = Vec::with_capacity(children_effects.0.len());
        for child_entity in children_effects.0.iter() {
            // Fetch the child's event buffer allocation info
            let Ok((_, child_extracted_effect, _, cached_effect_events, _)) =
                q_child_effects.get(*child_entity)
            else {
                warn!("Child entity {child_entity:?} from parent entity {parent_entity:?} didnt't resolve to a child instance. The parent effect cannot be processed.");
                if maybe_cached_parent_info.is_some() {
                    commands.entity(parent_entity).remove::<CachedParentInfo>();
                }
                break;
            };

            // Fetch the GPU event buffer of the child
            let Some(event_buffer) = event_cache.get_buffer(cached_effect_events.buffer_index)
            else {
                warn!("Child entity {child_entity:?} from parent entity {parent_entity:?} doesn't have an allocated GPU event buffer. The parent effect cannot be processed.");
                break;
            };

            let buffer_binding_source = BufferBindingSource {
                buffer: event_buffer.clone(),
                offset: cached_effect_events.range.start,
                size: NonZeroU32::new(cached_effect_events.range.len() as u32).unwrap(),
            };
            new_children.push((*child_entity, buffer_binding_source));

            new_child_infos.push(GpuChildInfo {
                event_count: 0,
                init_indirect_dispatch_index: cached_effect_events.init_indirect_dispatch_index,
            });

            // Ensure the particle@1 bind group layout exists for the given configuration of
            // particle layout. We do this here only for effects with a parent; for those
            // without a parent, we already did this in allocate_effects().
            effect_cache.ensure_particle_bind_group_layout_desc(
                child_extracted_effect.particle_layout.min_binding_size32(),
                Some(parent_min_binding_size),
            );
        }

        // If we don't have all children, just abort this effect. We don't try to have
        // partial relationships, this is too complex for shader bindings.
        debug_assert_eq!(new_children.len(), new_child_infos.len());
        if (new_children.len() < children_effects.len()) && maybe_cached_parent_info.is_some() {
            warn!("One or more child effect(s) on parent effect {parent_entity:?} failed to configure. The parent effect cannot be processed.");
            commands.entity(parent_entity).remove::<CachedParentInfo>();
            continue;
        }

        // Insert or update the CachedParentInfo component of the parent effect
        if let Some(mut cached_parent_info) = maybe_cached_parent_info {
            if cached_parent_info.children != new_children {
                // FIXME - missing way to just update in-place without changing the allocation
                // size!
                // if cached_parent_info.children.len() == new_children.len() {
                //} else {
                event_cache.reallocate_child_infos(
                    parent_entity,
                    new_children,
                    &new_child_infos[..],
                    cached_parent_info.as_mut(),
                );
                //}
            }
        } else {
            let cached_parent_info =
                event_cache.allocate_child_infos(parent_entity, new_children, &new_child_infos[..]);
            commands.entity(parent_entity).insert(cached_parent_info);
        }
    }
}

/// Prepare the init and update compute pipelines for an effect.
///
/// This caches the pipeline IDs once resolved, and their compiling state when
/// it changes, to determine when an effect is ready to be used.
///
/// Note that we do that proactively even if the effect will be skipped this
/// frame (for example because it's not visible). This ensures we queue pipeline
/// compilations ASAP, as they can take a long time (10+ frames). We also use
/// the pipeline compiling state, which we query here, to inform whether the
/// effect is ready for this frame. So in general if this is a new pipeline, it
/// won't be ready this frame.
pub fn prepare_init_update_pipelines(
    mut q_effects: Query<(
        Entity,
        &ExtractedEffect,
        &CachedEffect,
        Option<&CachedChildInfo>,
        Option<&CachedParentInfo>,
        Option<&CachedEffectProperties>,
        &mut CachedPipelines,
    )>,
    // FIXME - need mut for bind group layout creation; shouldn't be create there though
    mut effect_cache: ResMut<EffectCache>,
    pipeline_cache: Res<PipelineCache>,
    property_cache: ResMut<PropertyCache>,
    init_pipeline: Res<ParticlesInitPipeline>,
    update_pipeline: Res<ParticlesUpdatePipeline>,
    mut specialized_init_pipelines: ResMut<SpecializedComputePipelines<ParticlesInitPipeline>>,
    mut specialized_update_pipelines: ResMut<SpecializedComputePipelines<ParticlesUpdatePipeline>>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("prepare_init_update_pipelines").entered();
    trace!("prepare_init_update_pipelines");

    // Note: As of Bevy 0.16 we can't evict old pipelines from the cache. They're
    // inserted forever. https://github.com/bevyengine/bevy/issues/19925

    for (
        entity,
        extracted_effect,
        cached_effect,
        maybe_cached_child_info,
        maybe_cached_parent_info,
        maybe_cached_properties,
        mut cached_pipelines,
    ) in &mut q_effects
    {
        trace!(
            "Preparing pipelines for effect {:?}... (flags: {:?})",
            entity,
            cached_pipelines.flags
        );

        let particle_layout = &cached_effect.slice.particle_layout;
        let particle_layout_min_binding_size = particle_layout.min_binding_size32();
        let has_event_buffer = maybe_cached_child_info.is_some();
        let parent_particle_layout_min_binding_size = maybe_cached_child_info
            .as_ref()
            .map(|cci| cci.parent_particle_layout.min_binding_size32());

        let Some(particle_bind_group_layout_desc) = effect_cache.particle_bind_group_layout_desc(
            particle_layout_min_binding_size,
            parent_particle_layout_min_binding_size,
        ) else {
            error!("Failed to find particle sim bind group @1 for min_binding_size={} parent_min_binding_size={:?}",
                particle_layout_min_binding_size, parent_particle_layout_min_binding_size);
            continue;
        };
        let particle_bind_group_layout_desc = particle_bind_group_layout_desc.clone();

        // This should always exist by the time we reach this point, because we should
        // have inserted any property in the cache, which would have allocated the
        // proper bind group layout (or the default no-property one).
        let property_layout_min_binding_size =
            maybe_cached_properties.map(|cp| cp.property_layout.min_binding_size());
        let spawner_bind_group_layout_desc = property_cache
            .bind_group_layout_desc(property_layout_min_binding_size)
            .unwrap_or_else(|| {
                panic!(
                    "Failed to find spawner@2 bind group layout for property binding size {:?}",
                    property_layout_min_binding_size,
                )
            });
        trace!(
            "Retrieved spawner@2 bind group layout desc for property binding size {}:  {:?}.",
            property_layout_min_binding_size
                .as_ref()
                .map(|size| size.get())
                .unwrap_or(0),
            spawner_bind_group_layout_desc,
        );

        // Resolve the init pipeline
        let init_pipeline_id = if let Some(init_pipeline_id) = cached_pipelines.init.as_ref() {
            *init_pipeline_id
        } else {
            // Clear flag just in case, to ensure consistency.
            cached_pipelines
                .flags
                .remove(CachedPipelineFlags::INIT_PIPELINE_READY);

            // Fetch the metadata@3 bind group layout from the cache
            let metadata_bind_group_layout_desc = effect_cache
                .metadata_init_bind_group_layout_desc(has_event_buffer)
                .unwrap()
                .clone();

            let init_pipeline_key_flags = {
                let mut flags = ParticleInitPipelineKeyFlags::empty();
                flags.set(
                    ParticleInitPipelineKeyFlags::ATTRIBUTE_PREV,
                    particle_layout.contains(Attribute::PREV),
                );
                flags.set(
                    ParticleInitPipelineKeyFlags::ATTRIBUTE_NEXT,
                    particle_layout.contains(Attribute::NEXT),
                );
                flags.set(
                    ParticleInitPipelineKeyFlags::CONSUME_GPU_SPAWN_EVENTS,
                    has_event_buffer,
                );
                flags
            };

            let init_pipeline_id: CachedComputePipelineId = specialized_init_pipelines.specialize(
                pipeline_cache.as_ref(),
                &init_pipeline,
                ParticleInitPipelineKey {
                    shader: extracted_effect.effect_shaders.init.clone(),
                    particle_layout_min_binding_size,
                    parent_particle_layout_min_binding_size,
                    flags: init_pipeline_key_flags,
                    particle_bind_group_layout_desc: particle_bind_group_layout_desc.clone(),
                    spawner_bind_group_layout_desc: spawner_bind_group_layout_desc.clone(),
                    metadata_bind_group_layout_desc,
                },
            );
            trace!("Init pipeline specialized: id={:?}", init_pipeline_id);

            cached_pipelines.init = Some(init_pipeline_id);
            init_pipeline_id
        };

        // Resolve the update pipeline
        let update_pipeline_id = if let Some(update_pipeline_id) = cached_pipelines.update.as_ref()
        {
            *update_pipeline_id
        } else {
            // Clear flag just in case, to ensure consistency.
            cached_pipelines
                .flags
                .remove(CachedPipelineFlags::UPDATE_PIPELINE_READY);

            let num_event_buffers = maybe_cached_parent_info
                .as_ref()
                .map(|p| p.children.len() as u32)
                .unwrap_or_default();

            // FIXME: currently don't hava a way to determine when this is needed, because
            // we know the number of children per parent only after resolving
            // all parents, but by that point we forgot if this is a newly added
            // effect or not. So since we need to re-ensure for all effects, not
            // only new ones, might as well do here...
            effect_cache.ensure_metadata_update_bind_group_layout_desc(num_event_buffers);

            // Fetch the bind group layouts from the cache
            let metadata_bind_group_layout_desc = effect_cache
                .metadata_update_bind_group_layout_desc(num_event_buffers)
                .unwrap()
                .clone();

            let update_pipeline_id = specialized_update_pipelines.specialize(
                pipeline_cache.as_ref(),
                &update_pipeline,
                ParticleUpdatePipelineKey {
                    shader: extracted_effect.effect_shaders.update.clone(),
                    particle_layout: particle_layout.clone(),
                    parent_particle_layout_min_binding_size,
                    num_event_buffers,
                    particle_bind_group_layout_desc: particle_bind_group_layout_desc.clone(),
                    spawner_bind_group_layout_desc: spawner_bind_group_layout_desc.clone(),
                    metadata_bind_group_layout_desc,
                },
            );
            trace!("Update pipeline specialized: id={:?}", update_pipeline_id);

            cached_pipelines.update = Some(update_pipeline_id);
            update_pipeline_id
        };

        // Never batch an effect with a pipeline not available; this will prevent its
        // init/update pass from running, but the vfx_indirect pass will run
        // nonetheless, which causes desyncs and leads to bugs.
        if pipeline_cache
            .get_compute_pipeline(init_pipeline_id)
            .is_none()
        {
            trace!(
                "Skipping effect from render entity {:?} due to missing or not ready init pipeline (status: {:?})",
                entity,
                pipeline_cache.get_compute_pipeline_state(init_pipeline_id)
            );
            cached_pipelines
                .flags
                .remove(CachedPipelineFlags::INIT_PIPELINE_READY);
            continue;
        }

        // PipelineCache::get_compute_pipeline() only returns a value if the pipeline is
        // ready
        cached_pipelines
            .flags
            .insert(CachedPipelineFlags::INIT_PIPELINE_READY);
        trace!("[Effect {:?}] Init pipeline ready.", entity);

        // Never batch an effect with a pipeline not available; this will prevent its
        // init/update pass from running, but the vfx_indirect pass will run
        // nonetheless, which causes desyncs and leads to bugs.
        if pipeline_cache
            .get_compute_pipeline(update_pipeline_id)
            .is_none()
        {
            trace!(
                "Skipping effect from render entity {:?} due to missing or not ready update pipeline (status: {:?})",
                entity,
                pipeline_cache.get_compute_pipeline_state(update_pipeline_id)
            );
            cached_pipelines
                .flags
                .remove(CachedPipelineFlags::UPDATE_PIPELINE_READY);
            continue;
        }

        // PipelineCache::get_compute_pipeline() only returns a value if the pipeline is
        // ready
        cached_pipelines
            .flags
            .insert(CachedPipelineFlags::UPDATE_PIPELINE_READY);
        trace!("[Effect {:?}] Update pipeline ready.", entity);
    }
}

pub fn prepare_indirect_pipeline(
    event_cache: Res<EventCache>,
    mut effects_meta: ResMut<EffectsMeta>,
    pipeline_cache: Res<PipelineCache>,
    indirect_pipeline: Res<DispatchIndirectPipeline>,
    mut specialized_indirect_pipelines: ResMut<
        SpecializedComputePipelines<DispatchIndirectPipeline>,
    >,
) {
    // Ensure the 2 variants of the indirect pipelines are created.
    // TODO - move that elsewhere in some one-time setup?
    if effects_meta.indirect_pipeline_ids[0] == CachedComputePipelineId::INVALID {
        effects_meta.indirect_pipeline_ids[0] = specialized_indirect_pipelines.specialize(
            pipeline_cache.as_ref(),
            &indirect_pipeline,
            DispatchIndirectPipelineKey { has_events: false },
        );
    }
    if effects_meta.indirect_pipeline_ids[1] == CachedComputePipelineId::INVALID {
        effects_meta.indirect_pipeline_ids[1] = specialized_indirect_pipelines.specialize(
            pipeline_cache.as_ref(),
            &indirect_pipeline,
            DispatchIndirectPipelineKey { has_events: true },
        );
    }

    // Select the active one depending on whether there's any child info to consume
    let is_empty = event_cache.child_infos().is_empty();
    if effects_meta.active_indirect_pipeline_id == CachedComputePipelineId::INVALID {
        if is_empty {
            effects_meta.active_indirect_pipeline_id = effects_meta.indirect_pipeline_ids[0];
        } else {
            effects_meta.active_indirect_pipeline_id = effects_meta.indirect_pipeline_ids[1];
        }
    } else {
        // If this is the first time we insert an event buffer, we need to switch the
        // indirect pass from non-event to event mode. That is, we need to re-allocate
        // the pipeline with the child infos buffer binding. Conversely, if there's no
        // more effect using GPU spawn events, we can deallocate.
        let was_empty =
            effects_meta.active_indirect_pipeline_id == effects_meta.indirect_pipeline_ids[0];
        if was_empty && !is_empty {
            trace!("First event buffer inserted; switching indirect pass to event mode...");
            effects_meta.active_indirect_pipeline_id = effects_meta.indirect_pipeline_ids[1];
        } else if is_empty && !was_empty {
            trace!("Last event buffer removed; switching indirect pass to no-event mode...");
            effects_meta.active_indirect_pipeline_id = effects_meta.indirect_pipeline_ids[0];
        }
    }
}

// TEMP - Mark all cached effects as invalid for this frame until another system
// explicitly marks them as valid. Otherwise we early out in some parts, and
// reuse by mistake the previous frame's extraction.
pub fn clear_transient_batch_inputs(
    mut commands: Commands,
    mut q_cached_effects: Query<Entity, With<BatchInput>>,
) {
    for entity in &mut q_cached_effects {
        if let Ok(mut cmd) = commands.get_entity(entity) {
            cmd.remove::<BatchInput>();
        }
    }
}

/// Effect mesh extracted from the main world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub(crate) struct ExtractedEffectMesh {
    /// Asset of the effect mesh to draw.
    pub mesh: AssetId<Mesh>,
}

/// Indexed mesh metadata for [`CachedMesh`].
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct MeshIndexSlice {
    /// Index format.
    pub format: IndexFormat,
    /// GPU buffer containing the indices.
    pub buffer: Buffer,
    /// Range inside [`Self::buffer`] where the indices are.
    pub range: Range<u32>,
}

impl PartialEq for MeshIndexSlice {
    fn eq(&self, other: &Self) -> bool {
        self.format == other.format
            && self.buffer.id() == other.buffer.id()
            && self.range == other.range
    }
}

impl Eq for MeshIndexSlice {}

/// Cached info about a mesh location in a Bevy buffer. This information is
/// uploaded to GPU into [`GpuEffectMetadata`] for indirect rendering, but is
/// also kept CPU side in this component to detect when Bevy relocated a mesh,
/// so we can invalidate that GPU data.
#[derive(Debug, Clone, PartialEq, Eq, Component)]
pub(crate) struct CachedMeshLocation {
    /// Vertex buffer.
    pub vertex_buffer: BufferId,
    /// See [`GpuEffectMetadata::vertex_or_index_count`].
    pub vertex_or_index_count: u32,
    /// See [`GpuEffectMetadata::first_index_or_vertex_offset`].
    pub first_index_or_vertex_offset: u32,
    /// See [`GpuEffectMetadata::vertex_offset_or_base_instance`].
    pub vertex_offset_or_base_instance: i32,
    /// Indexed rendering metadata.
    pub indexed: Option<MeshIndexSlice>,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct CachedPipelineFlags: u8 {
        const NONE = 0;
        /// The init pipeline for this effect is ready for use. This means the compute pipeline is compiled and cached.
        const INIT_PIPELINE_READY = (1u8 << 0);
        /// The update pipeline for this effect is ready for use. This means the compute pipeline is compiled and cached.
        const UPDATE_PIPELINE_READY = (1u8 << 1);
    }
}

impl Default for CachedPipelineFlags {
    fn default() -> Self {
        Self::NONE
    }
}

/// Render world cached shader pipelines for a [`CachedEffect`].
///
/// This is updated with the IDs of the pipelines when they are queued for
/// compiling, and with the state of those pipelines to detect when the effect
/// is ready to be used.
///
/// This component is always auto-inserted alongside [`ExtractedEffect`] as soon
/// as a new effect instance is spawned, because it contains the readiness state
/// of those pipelines, which we want to query each frame. The pipelines are
/// also mandatory, so this component is always needed.
#[derive(Debug, Default, Component)]
pub(crate) struct CachedPipelines {
    /// Caching flags indicating the pipelines readiness.
    pub flags: CachedPipelineFlags,
    /// ID of the cached init pipeline. This is valid once the pipeline is
    /// queued for compilation, but this doesn't mean the pipeline is ready for
    /// use. Readiness is encoded in [`Self::flags`].
    pub init: Option<CachedComputePipelineId>,
    /// ID of the cached update pipeline. This is valid once the pipeline is
    /// queued for compilation, but this doesn't mean the pipeline is ready for
    /// use. Readiness is encoded in [`Self::flags`].
    pub update: Option<CachedComputePipelineId>,
}

impl CachedPipelines {
    /// Check if all pipelines for this effect are ready.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.flags.contains(
            CachedPipelineFlags::INIT_PIPELINE_READY | CachedPipelineFlags::UPDATE_PIPELINE_READY,
        )
    }
}

/// Ready state for this effect.
///
/// An effect is ready if:
/// - Its init and update pipelines are ready, as reported by
///   [`CachedPipelines::is_ready()`].
///
/// This components holds the calculated ready state propagated from all
/// ancestor effects, if any. That propagation is done by the
/// [`propagate_ready_state()`] system.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Component)]
pub(crate) struct CachedReadyState {
    is_ready: bool,
}

impl CachedReadyState {
    #[inline(always)]
    pub fn new(is_ready: bool) -> Self {
        Self { is_ready }
    }

    #[inline(always)]
    pub fn and(mut self, ancestors_ready: bool) -> Self {
        self.and_with(ancestors_ready);
        self
    }

    #[inline(always)]
    pub fn and_with(&mut self, ancestors_ready: bool) {
        self.is_ready = self.is_ready && ancestors_ready;
    }

    #[inline(always)]
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }
}

#[derive(SystemParam)]
pub struct PrepareEffectsReadOnlyParams<'w, 's> {
    sim_params: Res<'w, SimParams>,
    render_device: Res<'w, RenderDevice>,
    render_queue: Res<'w, RenderQueue>,
    marker: PhantomData<&'s usize>,
}

/// Update the ready state of all effects, and propagate recursively to
/// children.
pub(crate) fn propagate_ready_state(
    mut q_root_effects: Query<
        (
            Entity,
            Option<&ChildrenEffects>,
            Ref<CachedPipelines>,
            &mut CachedReadyState,
        ),
        Without<ChildEffectOf>,
    >,
    mut orphaned: RemovedComponents<ChildEffectOf>,
    q_ready_state: Query<
        (
            Ref<CachedPipelines>,
            &mut CachedReadyState,
            Option<&ChildrenEffects>,
        ),
        With<ChildEffectOf>,
    >,
    q_child_effects: Query<(Entity, Ref<ChildEffectOf>), With<CachedReadyState>>,
    mut orphaned_entities: Local<Vec<Entity>>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("propagate_ready_state").entered();
    trace!("propagate_ready_state");

    // Update orphaned list for this frame, and sort it so we can efficiently binary
    // search it
    orphaned_entities.clear();
    orphaned_entities.extend(orphaned.read());
    orphaned_entities.sort_unstable();

    // Iterate in parallel over all root effects (those without any parent). This is
    // the most common case, so should take care of the heavy lifting of propagating
    // to most effects. For child effects, we then descend recursively.
    q_root_effects.par_iter_mut().for_each(
        |(entity, maybe_children, cached_pipelines, mut cached_ready_state)| {
            // Update the ready state of this root effect
            let changed = cached_pipelines.is_changed() || cached_ready_state.is_added() || orphaned_entities.binary_search(&entity).is_ok();
            trace!("[Entity {}] changed={} cached_pipelines={} ready_state={}", entity, changed, cached_pipelines.is_ready(), cached_ready_state.is_ready);
            if changed {
                // Root effects by default are ready since they have no ancestors to check. After that we check the ready conditions for this effect alone.
                let new_ready_state = CachedReadyState::new(cached_pipelines.is_ready());
                if *cached_ready_state != new_ready_state {
                    debug!(
                        "[Entity {}] Changed ready to: {}",
                        entity,
                        new_ready_state.is_ready()
                    );
                    *cached_ready_state = new_ready_state;
                }
            }

            // Recursively update the ready state of its descendants
            if let Some(children) = maybe_children {
                for (child, child_of) in q_child_effects.iter_many(children) {
                    assert_eq!(
                        child_of.parent, entity,
                        "Malformed hierarchy. This probably means that your hierarchy has been improperly maintained, or contains a cycle"
                    );
                    // SAFETY:
                    // - `child` must have consistent parentage, or the above assertion would panic.
                    //   Since `child` is parented to a root entity, the entire hierarchy leading to it
                    //   is consistent.
                    // - We may operate as if all descendants are consistent, since
                    //   `propagate_ready_state_recursive` will panic before continuing to propagate if it
                    //   encounters an entity with inconsistent parentage.
                    // - Since each root entity is unique and the hierarchy is consistent and
                    //   forest-like, other root entities' `propagate_ready_state_recursive` calls will not conflict
                    //   with this one.
                    // - Since this is the only place where `transform_query` gets used, there will be
                    //   no conflicting fetches elsewhere.
                    #[expect(unsafe_code, reason = "`propagate_ready_state_recursive()` is unsafe due to its use of `Query::get_unchecked()`.")]
                    unsafe {
                        propagate_ready_state_recursive(
                            &cached_ready_state,
                            &q_ready_state,
                            &q_child_effects,
                            child,
                            changed || child_of.is_changed(),
                        );
                    }
                }
            }
        },
    );
}

#[expect(
    unsafe_code,
    reason = "This function uses `Query::get_unchecked()`, which can result in multiple mutable references if the preconditions are not met."
)]
unsafe fn propagate_ready_state_recursive(
    parent_state: &CachedReadyState,
    q_ready_state: &Query<
        (
            Ref<CachedPipelines>,
            &mut CachedReadyState,
            Option<&ChildrenEffects>,
        ),
        With<ChildEffectOf>,
    >,
    q_child_of: &Query<(Entity, Ref<ChildEffectOf>), With<CachedReadyState>>,
    entity: Entity,
    mut changed: bool,
) {
    // Update this effect in-place by checking its own state and the state of its
    // parent (which has already been propagated from all the parent's ancestors, so
    // is correct for this frame).
    let (cached_ready_state, maybe_children) = {
        let Ok((cached_pipelines, mut cached_ready_state, maybe_children)) =
        // SAFETY: Copied from Bevy's transform propagation, same reasoning
        (unsafe { q_ready_state.get_unchecked(entity) }) else {
            return;
        };

        changed |= cached_pipelines.is_changed() || cached_ready_state.is_added();
        if changed {
            let new_ready_state =
                CachedReadyState::new(parent_state.is_ready()).and(cached_pipelines.is_ready());
            // Ensure we don't trigger ECS change detection here if state didn't change, so
            // we can avoid this effect branch on next iteration.
            if *cached_ready_state != new_ready_state {
                debug!(
                    "[Entity {}] Changed ready to: {}",
                    entity,
                    new_ready_state.is_ready()
                );
                *cached_ready_state = new_ready_state;
            }
        }
        (cached_ready_state, maybe_children)
    };

    // Recurse into descendants
    let Some(children) = maybe_children else {
        return;
    };
    for (child, child_of) in q_child_of.iter_many(children) {
        assert_eq!(
        child_of.parent, entity,
        "Malformed hierarchy. This probably means that your hierarchy has been improperly maintained, or contains a cycle"
    );
        // SAFETY: The caller guarantees that `transform_query` will not be fetched for
        // any descendants of `entity`, so it is safe to call
        // `propagate_recursive` for each child.
        //
        // The above assertion ensures that each child has one and only one unique
        // parent throughout the entire hierarchy.
        unsafe {
            propagate_ready_state_recursive(
                cached_ready_state.as_ref(),
                q_ready_state,
                q_child_of,
                child,
                changed || child_of.is_changed(),
            );
        }
    }
}

/// Once all effects are extracted and all cached components are updated, it's
/// time to prepare for sorting and batching. Collect all relevant data and
/// insert/update the [`BatchInput`] for each effect.
pub(crate) fn prepare_batch_inputs(
    mut commands: Commands,
    read_only_params: PrepareEffectsReadOnlyParams,
    pipeline_cache: Res<PipelineCache>,
    mut effects_meta: ResMut<EffectsMeta>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    mut property_bind_groups: ResMut<PropertyBindGroups>,
    q_cached_effects: Query<(
        MainEntity,
        Entity,
        &ExtractedEffect,
        &ExtractedSpawner,
        &CachedEffect,
        &CachedEffectMetadata,
        &CachedReadyState,
        &CachedPipelines,
        Option<&CachedDrawIndirectArgs>,
        Option<&CachedParentInfo>,
        Option<&ChildEffectOf>,
        Option<&CachedChildInfo>,
        Option<&CachedEffectEvents>,
    )>,
    mut sort_bind_groups: ResMut<SortBindGroups>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("prepare_batch_inputs").entered();
    trace!("prepare_batch_inputs");

    // Workaround for too many params in system (TODO: refactor to split work?)
    let sim_params = read_only_params.sim_params.into_inner();
    let render_device = read_only_params.render_device.into_inner();
    let render_queue = read_only_params.render_queue.into_inner();

    // Clear per-instance buffers, which are filled below and re-uploaded each frame
    effects_meta.spawner_buffer.clear();

    // Build batcher inputs from extracted effects, updating all cached components
    // for each effect on the fly.
    let mut extracted_effect_count = 0;
    let mut prepared_effect_count = 0;
    for (
        main_entity,
        render_entity,
        extracted_effect,
        extracted_spawner,
        cached_effect,
        cached_effect_metadata,
        cached_ready_state,
        cached_pipelines,
        maybe_cached_draw_indirect_args,
        maybe_cached_parent_info,
        maybe_child_effect_of,
        maybe_cached_child_info,
        maybe_cached_effect_events,
    ) in &q_cached_effects
    {
        extracted_effect_count += 1;

        // Skip this effect if not ready
        if !cached_ready_state.is_ready() {
            trace!("Pipelines not ready for effect {}, skipped.", render_entity);
            continue;
        }

        // Skip this effect if not visible and not simulating when hidden
        if !extracted_spawner.is_visible
            && (extracted_effect.simulation_condition == SimulationCondition::WhenVisible)
        {
            trace!(
                "Effect {} not visible, and simulation condition is WhenVisible, so skipped.",
                render_entity
            );
            continue;
        }

        // Fetch the init and update pipelines.
        // SAFETY: If is_ready() returns true, this means the pipelines are cached and
        // ready, so the IDs must be valid.
        let init_and_update_pipeline_ids = InitAndUpdatePipelineIds {
            init: cached_pipelines.init.unwrap(),
            update: cached_pipelines.update.unwrap(),
        };

        let effect_slice = EffectSlice {
            slice: cached_effect.slice.range(),
            slab_id: cached_effect.slab_id,
            particle_layout: cached_effect.slice.particle_layout.clone(),
        };

        // Fetch the bind group layouts from the cache
        trace!("child_effect_of={:?}", maybe_child_effect_of);
        let parent_slab_id = if let Some(child_effect_of) = maybe_child_effect_of {
            let Ok((_, _, _, _, parent_cached_effect, _, _, _, _, _, _, _, _)) =
                q_cached_effects.get(child_effect_of.parent)
            else {
                // At this point we should have discarded invalid effects with a missing parent,
                // so if the parent is not found this is a bug.
                error!(
                    "Effect main_entity {:?}: parent render entity {:?} not found.",
                    main_entity, child_effect_of.parent
                );
                continue;
            };
            Some(parent_cached_effect.slab_id)
        } else {
            None
        };

        // For ribbons, we need the sorting pipeline to be ready to sort the ribbon's
        // particles by age in order to build a contiguous mesh.
        if extracted_effect.layout_flags.contains(LayoutFlags::RIBBONS) {
            // Ensure the bind group layout for sort-fill is ready. This will also ensure
            // the pipeline is created and queued if needed.
            if let Err(err) = sort_bind_groups.ensure_sort_fill_bind_group_layout_desc(
                &pipeline_cache,
                &extracted_effect.particle_layout,
            ) {
                error!(
                    "Failed to create bind group for ribbon effect sorting: {:?}",
                    err
                );
                continue;
            }

            // Check sort pipelines are ready, otherwise we might desync some buffers if
            // running only some of them but not all.
            if !sort_bind_groups
                .is_pipeline_ready(&extracted_effect.particle_layout, &pipeline_cache)
            {
                trace!(
                    "Sort pipeline not ready for effect on main entity {:?}; skipped.",
                    main_entity
                );
                continue;
            }
        }

        // Output some debug info
        trace!("init_shader = {:?}", extracted_effect.effect_shaders.init);
        trace!(
            "update_shader = {:?}",
            extracted_effect.effect_shaders.update
        );
        trace!(
            "render_shader = {:?}",
            extracted_effect.effect_shaders.render
        );
        trace!("layout_flags = {:?}", extracted_effect.layout_flags);
        trace!("particle_layout = {:?}", effect_slice.particle_layout);

        let parent_slab_offset = maybe_cached_child_info.map(|cci| cci.parent_slab_offset);

        assert!(cached_effect_metadata.table_id.is_valid());
        let spawner_index = effects_meta.allocate_spawner(
            &extracted_spawner.transform,
            extracted_spawner.spawn_count,
            extracted_spawner.prng_seed,
            cached_effect.slice.range().start,
            parent_slab_offset,
            cached_effect_metadata.table_id,
            maybe_cached_draw_indirect_args,
        );

        trace!("Updating cached effect at entity {render_entity:?}...");
        let mut cmd = commands.entity(render_entity);
        // Inserting the BatchInput component marks the effect as ready for this frame
        cmd.insert(BatchInput {
            effect_slice,
            init_and_update_pipeline_ids,
            parent_slab_id,
            event_buffer_index: maybe_cached_effect_events.map(|cee| cee.buffer_index),
            child_effects: maybe_cached_parent_info
                .as_ref()
                .map(|cp| cp.children.clone())
                .unwrap_or_default(),
            spawner_index,
            init_indirect_dispatch_index: maybe_cached_child_info
                .as_ref()
                .map(|cc| cc.init_indirect_dispatch_index),
        });

        prepared_effect_count += 1;
    }
    trace!("Prepared {prepared_effect_count}/{extracted_effect_count} extracted effect(s)");

    // Update simulation parameters, including the total effect count for this frame
    {
        let mut gpu_sim_params: GpuSimParams = sim_params.into();
        gpu_sim_params.num_effects = prepared_effect_count;
        trace!(
            "Simulation parameters: time={} delta_time={} virtual_time={} \
                virtual_delta_time={} real_time={} real_delta_time={} num_effects={}",
            gpu_sim_params.time,
            gpu_sim_params.delta_time,
            gpu_sim_params.virtual_time,
            gpu_sim_params.virtual_delta_time,
            gpu_sim_params.real_time,
            gpu_sim_params.real_delta_time,
            gpu_sim_params.num_effects,
        );
        effects_meta.sim_params_uniforms.set(gpu_sim_params);
    }

    // Write the entire spawner buffer for this frame, for all effects combined
    assert_eq!(
        prepared_effect_count,
        effects_meta.spawner_buffer.len() as u32
    );
    if effects_meta
        .spawner_buffer
        .write_buffer(render_device, render_queue)
    {
        // All property bind groups use the spawner buffer, which was reallocate
        effect_bind_groups.particle_slabs.clear();
        property_bind_groups.clear(true);
        effects_meta.indirect_spawner_bind_group = None;
    }
}

/// Batch compatible effects together into a single pass.
///
/// For all effects marked as ready for this frame (have a BatchInput
/// component), sort the effects by grouping compatible effects together, then
/// batch those groups together. Each batch can be updated and rendered with a
/// single compute dispatch or draw call.
pub(crate) fn batch_effects(
    mut commands: Commands,
    effects_meta: Res<EffectsMeta>,
    mut sort_bind_groups: ResMut<SortBindGroups>,
    mut q_cached_effects: Query<(
        Entity,
        &MainEntity,
        &ExtractedEffect,
        &ExtractedSpawner,
        &ExtractedEffectMesh,
        &CachedDrawIndirectArgs,
        &CachedEffectMetadata,
        Option<&CachedEffectEvents>,
        Option<&ChildEffectOf>,
        Option<&CachedChildInfo>,
        Option<&CachedEffectProperties>,
        &mut DispatchBufferIndices,
        // The presence of BatchInput ensure the effect is ready
        &mut BatchInput,
    )>,
    mut sorted_effect_batches: ResMut<SortedEffectBatches>,
    mut gpu_buffer_operations: ResMut<GpuBufferOperations>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("batch_effects").entered();
    trace!("batch_effects");

    // Sort effects in batching order, so that we can batch by simply doing a linear
    // scan of the effects in this order. Currently compatible effects mean:
    // - same effect slab (so we can bind the buffers once for all batched effects)
    // - in order of increasing sub-allocation inside those buffers (to make the
    //   sort stable)
    // - with parents before their children, to ensure ???? FIXME don't we need to
    //   opposite?!!!
    let mut effect_sorter = EffectSorter::new();
    for (entity, _, _, _, _, _, _, _, child_of, _, _, _, input) in &q_cached_effects {
        effect_sorter.insert(
            entity,
            input.effect_slice.slab_id,
            input.effect_slice.slice.start,
            child_of.map(|co| co.parent),
        );
    }
    effect_sorter.sort();

    // For now we re-create that buffer each frame. Since there's no CPU -> GPU
    // transfer, this is pretty cheap in practice.
    sort_bind_groups.clear_indirect_dispatch_buffer();

    let mut sort_queue = GpuBufferOperationQueue::new();

    // Loop on all extracted effects in sorted order, and try to batch them together
    // to reduce draw calls. -- currently does nothing, batching was broken and
    // never fixed, but at least we minimize the GPU state changes with the sorting!
    trace!("Batching {} effects...", q_cached_effects.iter().len());
    sorted_effect_batches.clear();
    for entity in effect_sorter.effects.iter().map(|e| e.entity) {
        let Ok((
            entity,
            main_entity,
            extracted_effect,
            extracted_spawner,
            extracted_effect_mesh,
            cached_draw_indirect_args,
            cached_effect_metadata,
            cached_effect_events,
            _,
            cached_child_info,
            cached_properties,
            dispatch_buffer_indices,
            mut input,
        )) = q_cached_effects.get_mut(entity)
        else {
            continue;
        };

        let translation = extracted_spawner.transform.translation();

        // Spawn one EffectBatch per instance (no batching; TODO). This contains
        // most of the data needed to drive rendering. However this doesn't drive
        // rendering; this is just storage.
        let mut effect_batch = EffectBatch::from_input(
            main_entity.id(),
            extracted_effect,
            extracted_spawner,
            extracted_effect_mesh,
            cached_effect_events,
            cached_child_info,
            &mut input,
            *dispatch_buffer_indices,
            cached_draw_indirect_args.row,
            cached_effect_metadata.table_id,
            cached_properties.map(|cp| PropertyBindGroupKey {
                buffer_index: cp.buffer_index,
                binding_size: cp.property_layout.min_binding_size().get() as u32,
            }),
            cached_properties.map(|cp| cp.range.start),
        );

        // If the batch has ribbons, we need to sort the particles by RIBBON_ID and AGE
        // for ribbon meshing, in order to avoid gaps when some particles in the middle
        // of the ribbon die (since we can't guarantee a linear lifetime through the
        // ribbon).
        if extracted_effect.layout_flags.contains(LayoutFlags::RIBBONS) {
            // This buffer is allocated in prepare_effects(), so should always be available
            let Some(effect_metadata_buffer) = effects_meta.effect_metadata_buffer.buffer() else {
                error!("Failed to find effect metadata buffer. This is a bug.");
                continue;
            };

            // Allocate a GpuDispatchIndirect entry
            let sort_fill_indirect_dispatch_index = sort_bind_groups.allocate_indirect_dispatch();
            effect_batch.sort_fill_indirect_dispatch_index =
                Some(sort_fill_indirect_dispatch_index);

            // Enqueue a fill dispatch operation which reads GpuEffectMetadata::alive_count,
            // compute a number of workgroups to dispatch based on that particle count, and
            // store the result into a GpuDispatchIndirect struct which will be used to
            // dispatch the fill-sort pass.
            {
                let src_buffer = effect_metadata_buffer.clone();
                let src_binding_offset = effects_meta
                    .effect_metadata_buffer
                    .dynamic_offset(effect_batch.metadata_table_id);
                let src_binding_size = effects_meta.gpu_limits.effect_metadata_aligned_size;
                let Some(dst_buffer) = sort_bind_groups.indirect_buffer() else {
                    error!("Missing indirect dispatch buffer for sorting, cannot schedule particle sort for ribbon. This is a bug.");
                    continue;
                };
                let dst_buffer = dst_buffer.clone();
                let dst_binding_offset = 0; // see dst_offset below
                                            //let dst_binding_size = NonZeroU32::new(12).unwrap();
                trace!(
                    "queue_fill_dispatch(): src#{:?}@+{}B ({}B) -> dst#{:?}@+{}B ({}B)",
                    src_buffer.id(),
                    src_binding_offset,
                    src_binding_size.get(),
                    dst_buffer.id(),
                    dst_binding_offset,
                    -1, //dst_binding_size.get(),
                );
                let src_offset = std::mem::offset_of!(GpuEffectMetadata, alive_count) as u32 / 4;
                debug_assert_eq!(
                    src_offset, 1,
                    "GpuEffectMetadata changed, update this assert."
                );
                // FIXME - This is a quick fix to get 0.15 out. The previous code used the
                // dynamic binding offset, but the indirect dispatch structs are only 12 bytes,
                // so are not aligned to min_storage_buffer_offset_alignment. The fix uses a
                // binding offset of 0 and binds the entire destination buffer,
                // then use the dst_offset value embedded inside the GpuBufferOperationArgs to
                // index the proper offset in the buffer. This requires of
                // course binding the entire buffer, or at least enough to index all operations
                // (hence the None below). This is not really a general solution, so should be
                // reviewed.
                let dst_offset = sort_bind_groups
                    .get_indirect_dispatch_byte_offset(sort_fill_indirect_dispatch_index)
                    / 4;
                sort_queue.enqueue(
                    GpuBufferOperationType::FillDispatchArgs,
                    GpuBufferOperationArgs {
                        src_offset,
                        src_stride: effects_meta.gpu_limits.effect_metadata_aligned_size.get() / 4,
                        dst_offset,
                        dst_stride: GpuDispatchIndirectArgs::SHADER_SIZE.get() as u32 / 4,
                        count: 1,
                    },
                    src_buffer,
                    src_binding_offset,
                    Some(src_binding_size),
                    dst_buffer,
                    dst_binding_offset,
                    None, //Some(dst_binding_size),
                );
            }
        }

        let effect_batch_index = sorted_effect_batches.push(effect_batch);
        trace!(
            "Spawned effect batch #{:?} from cached instance on entity {:?}.",
            effect_batch_index,
            entity,
        );

        // Spawn an EffectDrawBatch, to actually drive rendering.
        commands
            .spawn(EffectDrawBatch {
                effect_batch_index,
                translation,
                main_entity: *main_entity,
            })
            .insert(TemporaryRenderEntity);
    }

    gpu_buffer_operations.begin_frame();
    debug_assert!(sorted_effect_batches.dispatch_queue_index.is_none());
    if !sort_queue.operation_queue.is_empty() {
        sorted_effect_batches.dispatch_queue_index = Some(gpu_buffer_operations.submit(sort_queue));
    }
}

/// Per-buffer bind groups for a GPU effect buffer.
///
/// This contains all bind groups specific to a single [`EffectBuffer`].
///
/// [`EffectBuffer`]: crate::render::effect_cache::EffectBuffer
pub(crate) struct BufferBindGroups {
    /// Bind group for the render shader.
    ///
    /// ```wgsl
    /// @binding(0) var<storage, read> particle_buffer : ParticleBuffer;
    /// @binding(1) var<storage, read> indirect_buffer : IndirectBuffer;
    /// @binding(2) var<storage, read> spawner : Spawner;
    /// ```
    render: BindGroup,
    // /// Bind group for filling the indirect dispatch arguments of any child init
    // /// pass.
    // ///
    // /// This bind group is optional; it's only created if the current effect has
    // /// a GPU spawn event buffer, irrelevant of whether it has child effects
    // /// (although normally the event buffer is not created if there's no
    // /// children).
    // ///
    // /// The source buffer is always the current effect's event buffer. The
    // /// destination buffer is the global shared buffer for indirect fill args
    // /// operations owned by the [`EffectCache`]. The uniform buffer of operation
    // /// args contains the data to index the relevant part of the global shared
    // /// buffer for this effect buffer; it may contain multiple entries in case
    // /// multiple effects are batched inside the current effect buffer.
    // ///
    // /// ```wgsl
    // /// @group(0) @binding(0) var<uniform> args : BufferOperationArgs;
    // /// @group(0) @binding(1) var<storage, read> src_buffer : array<u32>;
    // /// @group(0) @binding(2) var<storage, read_write> dst_buffer : array<u32>;
    // /// ```
    // init_fill_dispatch: Option<BindGroup>,
}

/// Combination of a texture layout and the bound textures.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
struct Material {
    layout: TextureLayout,
    textures: Vec<AssetId<Image>>,
}

impl Material {
    /// Get the bind group entries to create a bind group.
    pub fn make_entries<'a>(
        &self,
        gpu_images: &'a RenderAssets<GpuImage>,
    ) -> Result<Vec<BindGroupEntry<'a>>, ()> {
        if self.textures.is_empty() {
            return Ok(vec![]);
        }

        let entries: Vec<BindGroupEntry<'a>> = self
            .textures
            .iter()
            .enumerate()
            .flat_map(|(index, id)| {
                let base_binding = index as u32 * 2;
                if let Some(gpu_image) = gpu_images.get(*id) {
                    vec![
                        BindGroupEntry {
                            binding: base_binding,
                            resource: BindingResource::TextureView(&gpu_image.texture_view),
                        },
                        BindGroupEntry {
                            binding: base_binding + 1,
                            resource: BindingResource::Sampler(&gpu_image.sampler),
                        },
                    ]
                } else {
                    vec![]
                }
            })
            .collect();
        if entries.len() == self.textures.len() * 2 {
            return Ok(entries);
        }
        Err(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BindingKey {
    pub buffer_id: BufferId,
    pub offset: u32,
    pub size: NonZeroU32,
}

impl<'a> From<BufferSlice<'a>> for BindingKey {
    fn from(value: BufferSlice<'a>) -> Self {
        Self {
            buffer_id: value.buffer.id(),
            offset: value.offset,
            size: value.size,
        }
    }
}

impl<'a> From<&BufferSlice<'a>> for BindingKey {
    fn from(value: &BufferSlice<'a>) -> Self {
        Self {
            buffer_id: value.buffer.id(),
            offset: value.offset,
            size: value.size,
        }
    }
}

impl From<&BufferBindingSource> for BindingKey {
    fn from(value: &BufferBindingSource) -> Self {
        Self {
            buffer_id: value.buffer.id(),
            offset: value.offset,
            size: value.size,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConsumeEventKey {
    child_infos_buffer_id: BufferId,
    events: BindingKey,
}

impl From<&ConsumeEventBuffers<'_>> for ConsumeEventKey {
    fn from(value: &ConsumeEventBuffers) -> Self {
        Self {
            child_infos_buffer_id: value.child_infos_buffer.id(),
            events: value.events.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct InitMetadataBindGroupKey {
    pub slab_id: SlabId,
    pub effect_metadata_buffer: BufferId,
    pub consume_event_key: Option<ConsumeEventKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct UpdateMetadataBindGroupKey {
    pub slab_id: SlabId,
    pub effect_metadata_buffer: BufferId,
    pub child_info_buffer_id: Option<BufferId>,
    pub event_buffers_keys: Vec<BindingKey>,
}

/// Bind group cached with an associated key.
///
/// The cached bind group is associated with the given key representing the
/// inputs that the bind group depends on. When those inputs change, the key
/// should change, indicating the bind group needs to be recreated.
///
/// This object manages a single bind group and its key.
struct CachedBindGroup<K: Eq> {
    /// Key the bind group was created from. Each time the key changes, the bind
    /// group should be re-created.
    key: K,
    /// Bind group created from the key.
    bind_group: BindGroup,
}

#[derive(Debug, Clone, Copy)]
struct BufferSlice<'a> {
    pub buffer: &'a Buffer,
    pub offset: u32,
    pub size: NonZeroU32,
}

impl<'a> From<BufferSlice<'a>> for BufferBinding<'a> {
    fn from(value: BufferSlice<'a>) -> Self {
        Self {
            buffer: value.buffer,
            offset: value.offset.into(),
            size: Some(value.size.into()),
        }
    }
}

impl<'a> From<&BufferSlice<'a>> for BufferBinding<'a> {
    fn from(value: &BufferSlice<'a>) -> Self {
        Self {
            buffer: value.buffer,
            offset: value.offset.into(),
            size: Some(value.size.into()),
        }
    }
}

impl<'a> From<&'a BufferBindingSource> for BufferSlice<'a> {
    fn from(value: &'a BufferBindingSource) -> Self {
        Self {
            buffer: &value.buffer,
            offset: value.offset,
            size: value.size,
        }
    }
}

/// Optional input to [`EffectBindGroups::get_or_create_init_metadata()`] when
/// the init pass consumes GPU events as a mechanism to spawn particles.
struct ConsumeEventBuffers<'a> {
    /// Entire buffer containing the [`GpuChildInfo`] entries for all effects.
    /// This is dynamically indexed inside the shader.
    child_infos_buffer: &'a Buffer,
    /// Slice of the [`EventBuffer`] where the GPU spawn events are stored.
    events: BufferSlice<'a>,
}

#[derive(Default, Resource)]
pub struct EffectBindGroups {
    /// Map from a slab ID to the bind groups shared among all effects that
    /// use that particle slab.
    particle_slabs: HashMap<SlabId, BufferBindGroups>,
    /// Map of bind groups for image assets used as particle textures.
    images: HashMap<AssetId<Image>, BindGroup>,
    /// Map from buffer index to its metadata bind group (group 3) for the init
    /// pass.
    // FIXME - doesn't work with batching; this should be the instance ID
    init_metadata_bind_groups: HashMap<SlabId, CachedBindGroup<InitMetadataBindGroupKey>>,
    /// Map from buffer index to its metadata bind group (group 3) for the
    /// update pass.
    // FIXME - doesn't work with batching; this should be the instance ID
    update_metadata_bind_groups: HashMap<SlabId, CachedBindGroup<UpdateMetadataBindGroupKey>>,
    /// Map from an effect material to its bind group.
    material_bind_groups: HashMap<Material, BindGroup>,
}

impl EffectBindGroups {
    pub fn particle_render(&self, slab_id: &SlabId) -> Option<&BindGroup> {
        self.particle_slabs.get(slab_id).map(|bg| &bg.render)
    }

    /// Retrieve the metadata@3 bind group for the init pass, creating it if
    /// needed.
    pub(self) fn get_or_create_init_metadata(
        &mut self,
        effect_batch: &EffectBatch,
        render_device: &RenderDevice,
        layout: &BindGroupLayout,
        effect_metadata_buffer: &Buffer,
        consume_event_buffers: Option<ConsumeEventBuffers>,
    ) -> Result<&BindGroup, ()> {
        assert!(effect_batch.metadata_table_id.is_valid());

        let key = InitMetadataBindGroupKey {
            slab_id: effect_batch.slab_id,
            effect_metadata_buffer: effect_metadata_buffer.id(),
            consume_event_key: consume_event_buffers.as_ref().map(Into::into),
        };

        let make_entry = || {
            let mut entries = Vec::with_capacity(3);
            entries.push(
                // @group(3) @binding(0) var<storage, read_write> effect_metadatas :
                // array<EffectMetadata>;
                BindGroupEntry {
                    binding: 0,
                    resource: effect_metadata_buffer.as_entire_binding(),
                },
            );
            if let Some(consume_event_buffers) = consume_event_buffers.as_ref() {
                entries.push(
                    // @group(3) @binding(1) var<storage, read> child_info_buffer :
                    // ChildInfoBuffer;
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: consume_event_buffers.child_infos_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                );
                entries.push(
                    // @group(3) @binding(2) var<storage, read> event_buffer : EventBuffer;
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(consume_event_buffers.events.into()),
                    },
                );
            }

            let bind_group = render_device.create_bind_group(
                "hanabi:bind_group:init:metadata@3",
                layout,
                &entries[..],
            );

            trace!(
                    "Created new metadata@3 bind group for init pass and buffer index {}: effect_metadata=#{}",
                    effect_batch.slab_id.index(),
                    effect_batch.metadata_table_id.0,
                );

            bind_group
        };

        Ok(&self
            .init_metadata_bind_groups
            .entry(effect_batch.slab_id)
            .and_modify(|cbg| {
                if cbg.key != key {
                    trace!(
                        "Bind group key changed for init metadata@3, re-creating bind group... old={:?} new={:?}",
                        cbg.key,
                        key
                    );
                    cbg.key = key;
                    cbg.bind_group = make_entry();
                }
            })
            .or_insert_with(|| {
                trace!("Inserting new bind group for init metadata@3 with key={:?}", key);
                CachedBindGroup {
                    key,
                    bind_group: make_entry(),
                }
            })
            .bind_group)
    }

    /// Retrieve the metadata@3 bind group for the update pass, creating it if
    /// needed.
    pub(self) fn get_or_create_update_metadata(
        &mut self,
        effect_batch: &EffectBatch,
        render_device: &RenderDevice,
        layout: &BindGroupLayout,
        effect_metadata_buffer: &Buffer,
        child_info_buffer: Option<&Buffer>,
        event_buffers: &[(Entity, BufferBindingSource)],
    ) -> Result<&BindGroup, ()> {
        assert!(effect_batch.metadata_table_id.is_valid());

        // Check arguments consistency
        assert_eq!(effect_batch.child_event_buffers.len(), event_buffers.len());
        let emits_gpu_spawn_events = !event_buffers.is_empty();
        let child_info_buffer_id = if emits_gpu_spawn_events {
            child_info_buffer.as_ref().map(|buffer| buffer.id())
        } else {
            // Note: child_info_buffer can be Some() if allocated, but we only consider it
            // if relevant, that is if the effect emits GPU spawn events.
            None
        };
        assert_eq!(emits_gpu_spawn_events, child_info_buffer_id.is_some());

        let event_buffers_keys = event_buffers
            .iter()
            .map(|(_, buffer_binding_source)| buffer_binding_source.into())
            .collect::<Vec<_>>();

        let key = UpdateMetadataBindGroupKey {
            slab_id: effect_batch.slab_id,
            effect_metadata_buffer: effect_metadata_buffer.id(),
            child_info_buffer_id,
            event_buffers_keys,
        };

        let make_entry = || {
            let mut entries = Vec::with_capacity(2 + event_buffers.len());
            // @group(3) @binding(0) var<storage, read_write> effect_metadatas :
            // array<EffectMetadata>;
            entries.push(BindGroupEntry {
                binding: 0,
                resource: effect_metadata_buffer.as_entire_binding(),
            });
            if emits_gpu_spawn_events {
                let child_info_buffer = child_info_buffer.unwrap();

                // @group(3) @binding(1) var<storage, read_write> child_info_buffer :
                // ChildInfoBuffer;
                entries.push(BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: child_info_buffer,
                        offset: 0,
                        size: None,
                    }),
                });

                for (index, (_, buffer_binding_source)) in event_buffers.iter().enumerate() {
                    // @group(3) @binding(2+N) var<storage, read_write> event_buffer_N :
                    // EventBuffer;
                    // FIXME - BufferBindingSource originally was for Events, counting in u32, but
                    // then moved to counting in bytes, so now need some conversion. Need to review
                    // all of this...
                    let mut buffer_binding: BufferBinding = buffer_binding_source.into();
                    buffer_binding.offset *= 4;
                    buffer_binding.size = buffer_binding
                        .size
                        .map(|sz| NonZeroU64::new(sz.get() * 4).unwrap());
                    entries.push(BindGroupEntry {
                        binding: 2 + index as u32,
                        resource: BindingResource::Buffer(buffer_binding),
                    });
                }
            }

            let bind_group = render_device.create_bind_group(
                "hanabi:bind_group:update:metadata@3",
                layout,
                &entries[..],
            );

            trace!(
                "Created new metadata@3 bind group for update pass and slab ID {}: effect_metadata={}",
                effect_batch.slab_id.index(),
                effect_batch.metadata_table_id.0,
            );

            bind_group
        };

        Ok(&self
            .update_metadata_bind_groups
            .entry(effect_batch.slab_id)
            .and_modify(|cbg| {
                if cbg.key != key {
                    trace!(
                        "Bind group key changed for update metadata@3, re-creating bind group... old={:?} new={:?}",
                        cbg.key,
                        key
                    );
                    cbg.key = key.clone();
                    cbg.bind_group = make_entry();
                }
            })
            .or_insert_with(|| {
                trace!(
                    "Inserting new bind group for update metadata@3 with key={:?}",
                    key
                );
                CachedBindGroup {
                    key: key.clone(),
                    bind_group: make_entry(),
                }
            })
            .bind_group)
    }
}

#[derive(SystemParam)]
pub struct QueueEffectsReadOnlyParams<'w, 's> {
    #[cfg(feature = "2d")]
    draw_functions_2d: Res<'w, DrawFunctions<Transparent2d>>,
    #[cfg(feature = "3d")]
    draw_functions_3d: Res<'w, DrawFunctions<Transparent3d>>,
    #[cfg(feature = "3d")]
    draw_functions_alpha_mask: Res<'w, DrawFunctions<AlphaMask3d>>,
    #[cfg(feature = "3d")]
    draw_functions_opaque: Res<'w, DrawFunctions<Opaque3d>>,
    marker: PhantomData<&'s usize>,
}

fn emit_sorted_draw<T, F>(
    views: &Query<(&RenderVisibleEntities, &ExtractedView, &Msaa)>,
    render_phases: &mut ResMut<ViewSortedRenderPhases<T>>,
    view_entities: &mut FixedBitSet,
    sorted_effect_batches: &SortedEffectBatches,
    effect_draw_batches: &Query<(Entity, &mut EffectDrawBatch)>,
    render_pipeline: &mut ParticlesRenderPipeline,
    mut specialized_render_pipelines: Mut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    render_meshes: &RenderAssets<RenderMesh>,
    pipeline_cache: &PipelineCache,
    make_phase_item: F,
    #[cfg(all(feature = "2d", feature = "3d"))] pipeline_mode: PipelineMode,
) where
    T: SortedPhaseItem,
    F: Fn(CachedRenderPipelineId, (Entity, MainEntity), &EffectDrawBatch, &ExtractedView) -> T,
{
    trace!("emit_sorted_draw() {} views", views.iter().len());

    for (visible_entities, view, msaa) in views.iter() {
        trace!(
            "Process new sorted view with {} visible particle effect entities",
            visible_entities.len::<CompiledParticleEffect>()
        );

        let Some(render_phase) = render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        {
            #[cfg(feature = "trace")]
            let _span = bevy::log::info_span!("collect_view_entities").entered();

            view_entities.clear();
            view_entities.extend(
                visible_entities
                    .iter::<EffectVisibilityClass>()
                    .map(|e| e.1.index_u32() as usize),
            );
        }

        // For each view, loop over all the effect batches to determine if the effect
        // needs to be rendered for that view, and enqueue a view-dependent
        // batch if so.
        for (draw_entity, draw_batch) in effect_draw_batches.iter() {
            #[cfg(feature = "trace")]
            let _span_draw = bevy::log::info_span!("draw_batch").entered();

            trace!(
                "Process draw batch: draw_entity={:?} effect_batch_index={:?}",
                draw_entity,
                draw_batch.effect_batch_index,
            );

            // Get the EffectBatches this EffectDrawBatch is part of.
            let Some(effect_batch) = sorted_effect_batches.get(draw_batch.effect_batch_index)
            else {
                continue;
            };

            trace!(
                "-> EffectBach: slab_id={} spawner_base={} layout_flags={:?}",
                effect_batch.slab_id.index(),
                effect_batch.spawner_base,
                effect_batch.layout_flags,
            );

            // AlphaMask is a binned draw, so no sorted draw can possibly use it
            if effect_batch
                .layout_flags
                .intersects(LayoutFlags::USE_ALPHA_MASK | LayoutFlags::OPAQUE)
            {
                trace!("Non-transparent batch. Skipped.");
                continue;
            }

            // Check if batch contains any entity visible in the current view. Otherwise we
            // can skip the entire batch. Note: This is O(n^2) but (unlike
            // the Sprite renderer this is inspired from) we don't expect more than
            // a handful of particle effect instances, so would rather not pay the memory
            // cost of a FixedBitSet for the sake of an arguable speed-up.
            // TODO - Profile to confirm.
            #[cfg(feature = "trace")]
            let _span_check_vis = bevy::log::info_span!("check_visibility").entered();
            let has_visible_entity = effect_batch
                .entities
                .iter()
                .any(|index| view_entities.contains(*index as usize));
            if !has_visible_entity {
                trace!("No visible entity for view, not emitting any draw call.");
                continue;
            }
            #[cfg(feature = "trace")]
            _span_check_vis.exit();

            // Create and cache the bind group layout for this texture layout
            render_pipeline.cache_material(&effect_batch.texture_layout);

            // FIXME - We draw the entire batch, but part of it may not be visible in this
            // view! We should re-batch for the current view specifically!

            let local_space_simulation = effect_batch
                .layout_flags
                .contains(LayoutFlags::LOCAL_SPACE_SIMULATION);
            let alpha_mask = ParticleRenderAlphaMaskPipelineKey::from(effect_batch.layout_flags);
            let flipbook = effect_batch.layout_flags.contains(LayoutFlags::FLIPBOOK);
            let needs_uv = effect_batch.layout_flags.contains(LayoutFlags::NEEDS_UV);
            let needs_normal = effect_batch
                .layout_flags
                .contains(LayoutFlags::NEEDS_NORMAL);
            let needs_particle_fragment = effect_batch
                .layout_flags
                .contains(LayoutFlags::NEEDS_PARTICLE_FRAGMENT);
            let ribbons = effect_batch.layout_flags.contains(LayoutFlags::RIBBONS);
            let image_count = effect_batch.texture_layout.layout.len() as u8;

            // FIXME - Maybe it's better to copy the mesh layout into the batch, instead of
            // re-querying here...?
            let Some(render_mesh) = render_meshes.get(effect_batch.mesh) else {
                trace!("Batch has no render mesh, skipped.");
                continue;
            };
            let mesh_layout = render_mesh.layout.clone();

            // Specialize the render pipeline based on the effect batch
            trace!(
                "Specializing render pipeline: render_shader={:?} image_count={} alpha_mask={:?} flipbook={:?} hdr={}",
                effect_batch.render_shader,
                image_count,
                alpha_mask,
                flipbook,
                view.hdr
            );

            // Add a draw pass for the effect batch
            trace!("Emitting individual draw for batch");

            let alpha_mode = effect_batch.alpha_mode;

            #[cfg(feature = "trace")]
            let _span_specialize = bevy::log::info_span!("specialize").entered();
            let render_pipeline_id = specialized_render_pipelines.specialize(
                pipeline_cache,
                render_pipeline,
                ParticleRenderPipelineKey {
                    shader: effect_batch.render_shader.clone(),
                    mesh_layout: Some(mesh_layout),
                    particle_layout: effect_batch.particle_layout.clone(),
                    texture_layout: effect_batch.texture_layout.clone(),
                    local_space_simulation,
                    alpha_mask,
                    alpha_mode,
                    flipbook,
                    needs_uv,
                    needs_normal,
                    needs_particle_fragment,
                    ribbons,
                    #[cfg(all(feature = "2d", feature = "3d"))]
                    pipeline_mode,
                    msaa_samples: msaa.samples(),
                    hdr: view.hdr,
                },
            );
            #[cfg(feature = "trace")]
            _span_specialize.exit();

            trace!("+ Render pipeline specialized: id={:?}", render_pipeline_id,);
            trace!(
                "+ Add Transparent for batch on draw_entity {:?}: slab_id={} \
                spawner_base={} handle={:?}",
                draw_entity,
                effect_batch.slab_id.index(),
                effect_batch.spawner_base,
                effect_batch.handle
            );
            render_phase.add(make_phase_item(
                render_pipeline_id,
                (draw_entity, MainEntity::from(Entity::PLACEHOLDER)),
                draw_batch,
                view,
            ));
        }
    }
}

#[cfg(feature = "3d")]
fn emit_binned_draw<T, F, G>(
    views: &Query<(&RenderVisibleEntities, &ExtractedView, &Msaa)>,
    render_phases: &mut ResMut<ViewBinnedRenderPhases<T>>,
    view_entities: &mut FixedBitSet,
    sorted_effect_batches: &SortedEffectBatches,
    effect_draw_batches: &Query<(Entity, &mut EffectDrawBatch)>,
    render_pipeline: &mut ParticlesRenderPipeline,
    mut specialized_render_pipelines: Mut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    pipeline_cache: &PipelineCache,
    render_meshes: &RenderAssets<RenderMesh>,
    make_batch_set_key: F,
    make_bin_key: G,
    #[cfg(all(feature = "2d", feature = "3d"))] pipeline_mode: PipelineMode,
    alpha_mask: ParticleRenderAlphaMaskPipelineKey,
    change_tick: &mut Tick,
) where
    T: BinnedPhaseItem,
    F: Fn(CachedRenderPipelineId, &EffectDrawBatch, &ExtractedView) -> T::BatchSetKey,
    G: Fn() -> T::BinKey,
{
    use bevy::render::render_phase::{BinnedRenderPhaseType, InputUniformIndex};

    trace!("emit_binned_draw() {} views", views.iter().len());

    for (visible_entities, view, msaa) in views.iter() {
        trace!("Process new binned view (alpha_mask={:?})", alpha_mask);

        let Some(render_phase) = render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        {
            #[cfg(feature = "trace")]
            let _span = bevy::log::info_span!("collect_view_entities").entered();

            view_entities.clear();
            view_entities.extend(
                visible_entities
                    .iter::<EffectVisibilityClass>()
                    .map(|e| e.1.index_u32() as usize),
            );
        }

        // For each view, loop over all the effect batches to determine if the effect
        // needs to be rendered for that view, and enqueue a view-dependent
        // batch if so.
        for (draw_entity, draw_batch) in effect_draw_batches.iter() {
            #[cfg(feature = "trace")]
            let _span_draw = bevy::log::info_span!("draw_batch").entered();

            trace!(
                "Process draw batch: draw_entity={:?} effect_batch_index={:?}",
                draw_entity,
                draw_batch.effect_batch_index,
            );

            // Get the EffectBatches this EffectDrawBatch is part of.
            let Some(effect_batch) = sorted_effect_batches.get(draw_batch.effect_batch_index)
            else {
                continue;
            };

            trace!(
                "-> EffectBaches: slab_id={} spawner_base={} layout_flags={:?}",
                effect_batch.slab_id.index(),
                effect_batch.spawner_base,
                effect_batch.layout_flags,
            );

            if ParticleRenderAlphaMaskPipelineKey::from(effect_batch.layout_flags) != alpha_mask {
                trace!(
                    "Mismatching alpha mask pipeline key (batches={:?}, expected={:?}). Skipped.",
                    effect_batch.layout_flags,
                    alpha_mask
                );
                continue;
            }

            // Check if batch contains any entity visible in the current view. Otherwise we
            // can skip the entire batch. Note: This is O(n^2) but (unlike
            // the Sprite renderer this is inspired from) we don't expect more than
            // a handful of particle effect instances, so would rather not pay the memory
            // cost of a FixedBitSet for the sake of an arguable speed-up.
            // TODO - Profile to confirm.
            #[cfg(feature = "trace")]
            let _span_check_vis = bevy::log::info_span!("check_visibility").entered();
            let has_visible_entity = effect_batch
                .entities
                .iter()
                .any(|index| view_entities.contains(*index as usize));
            if !has_visible_entity {
                trace!("No visible entity for view, not emitting any draw call.");
                continue;
            }
            #[cfg(feature = "trace")]
            _span_check_vis.exit();

            // Create and cache the bind group layout for this texture layout
            render_pipeline.cache_material(&effect_batch.texture_layout);

            // FIXME - We draw the entire batch, but part of it may not be visible in this
            // view! We should re-batch for the current view specifically!

            let local_space_simulation = effect_batch
                .layout_flags
                .contains(LayoutFlags::LOCAL_SPACE_SIMULATION);
            let alpha_mask = ParticleRenderAlphaMaskPipelineKey::from(effect_batch.layout_flags);
            let flipbook = effect_batch.layout_flags.contains(LayoutFlags::FLIPBOOK);
            let needs_uv = effect_batch.layout_flags.contains(LayoutFlags::NEEDS_UV);
            let needs_normal = effect_batch
                .layout_flags
                .contains(LayoutFlags::NEEDS_NORMAL);
            let needs_particle_fragment = effect_batch
                .layout_flags
                .contains(LayoutFlags::NEEDS_PARTICLE_FRAGMENT);
            let ribbons = effect_batch.layout_flags.contains(LayoutFlags::RIBBONS);
            let image_count = effect_batch.texture_layout.layout.len() as u8;
            let render_mesh = render_meshes.get(effect_batch.mesh);

            // Specialize the render pipeline based on the effect batch
            trace!(
                "Specializing render pipeline: render_shaders={:?} image_count={} alpha_mask={:?} flipbook={:?} hdr={}",
                effect_batch.render_shader,
                image_count,
                alpha_mask,
                flipbook,
                view.hdr
            );

            // Add a draw pass for the effect batch
            trace!("Emitting individual draw for batch");

            let alpha_mode = effect_batch.alpha_mode;

            let Some(mesh_layout) = render_mesh.map(|gpu_mesh| gpu_mesh.layout.clone()) else {
                trace!("Missing mesh vertex buffer layout. Skipped.");
                continue;
            };

            #[cfg(feature = "trace")]
            let _span_specialize = bevy::log::info_span!("specialize").entered();
            let render_pipeline_id = specialized_render_pipelines.specialize(
                pipeline_cache,
                render_pipeline,
                ParticleRenderPipelineKey {
                    shader: effect_batch.render_shader.clone(),
                    mesh_layout: Some(mesh_layout),
                    particle_layout: effect_batch.particle_layout.clone(),
                    texture_layout: effect_batch.texture_layout.clone(),
                    local_space_simulation,
                    alpha_mask,
                    alpha_mode,
                    flipbook,
                    needs_uv,
                    needs_normal,
                    needs_particle_fragment,
                    ribbons,
                    #[cfg(all(feature = "2d", feature = "3d"))]
                    pipeline_mode,
                    msaa_samples: msaa.samples(),
                    hdr: view.hdr,
                },
            );
            #[cfg(feature = "trace")]
            _span_specialize.exit();

            trace!("+ Render pipeline specialized: id={:?}", render_pipeline_id,);
            trace!(
                "+ Add Transparent for batch on draw_entity {:?}: slab_id={} \
                spawner_base={} handle={:?}",
                draw_entity,
                effect_batch.slab_id.index(),
                effect_batch.spawner_base,
                effect_batch.handle
            );
            render_phase.add(
                make_batch_set_key(render_pipeline_id, draw_batch, view),
                make_bin_key(),
                (draw_entity, draw_batch.main_entity),
                InputUniformIndex::default(),
                BinnedRenderPhaseType::NonMesh,
                *change_tick,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn queue_effects(
    views: Query<(&RenderVisibleEntities, &ExtractedView, &Msaa)>,
    effects_meta: Res<EffectsMeta>,
    mut render_pipeline: ResMut<ParticlesRenderPipeline>,
    mut specialized_render_pipelines: ResMut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    sorted_effect_batches: Res<SortedEffectBatches>,
    effect_draw_batches: Query<(Entity, &mut EffectDrawBatch)>,
    events: Res<EffectAssetEvents>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    read_params: QueueEffectsReadOnlyParams,
    mut view_entities: Local<FixedBitSet>,
    #[cfg(feature = "2d")] mut transparent_2d_render_phases: ResMut<
        ViewSortedRenderPhases<Transparent2d>,
    >,
    #[cfg(feature = "3d")] mut transparent_3d_render_phases: ResMut<
        ViewSortedRenderPhases<Transparent3d>,
    >,
    #[cfg(feature = "3d")] (mut opaque_3d_render_phases, mut alpha_mask_3d_render_phases): (
        ResMut<ViewBinnedRenderPhases<Opaque3d>>,
        ResMut<ViewBinnedRenderPhases<AlphaMask3d>>,
    ),
    mut change_tick: Local<Tick>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("hanabi:queue_effects").entered();

    trace!("queue_effects");

    // Bump the change tick so that Bevy is forced to rebuild the binned render
    // phase bins. We don't use the built-in caching so we don't want Bevy to
    // reuse stale data.
    let next_change_tick = change_tick.get() + 1;
    change_tick.set(next_change_tick);

    // If an image has changed, the GpuImage has (probably) changed
    for event in &events.images {
        match event {
            AssetEvent::Added { .. } => (),
            AssetEvent::LoadedWithDependencies { .. } => (),
            AssetEvent::Unused { .. } => (),
            AssetEvent::Modified { id } => {
                if effect_bind_groups.images.remove(id).is_some() {
                    trace!("Destroyed bind group of modified image asset {:?}", id);
                }
            }
            AssetEvent::Removed { id } => {
                if effect_bind_groups.images.remove(id).is_some() {
                    trace!("Destroyes bind group of removed image asset {:?}", id);
                }
            }
        };
    }

    if effects_meta.spawner_buffer.buffer().is_none() || effects_meta.spawner_buffer.is_empty() {
        // No spawners are active
        return;
    }

    // Loop over all 2D cameras/views that need to render effects
    #[cfg(feature = "2d")]
    {
        #[cfg(feature = "trace")]
        let _span_draw = bevy::log::info_span!("draw_2d").entered();

        let draw_effects_function_2d = read_params
            .draw_functions_2d
            .read()
            .get_id::<DrawEffects>()
            .unwrap();

        // Effects with full alpha blending
        if !views.is_empty() {
            trace!("Emit effect draw calls for alpha blended 2D views...");
            emit_sorted_draw(
                &views,
                &mut transparent_2d_render_phases,
                &mut view_entities,
                &sorted_effect_batches,
                &effect_draw_batches,
                &mut render_pipeline,
                specialized_render_pipelines.reborrow(),
                &render_meshes,
                &pipeline_cache,
                |id, entity, draw_batch, _view| Transparent2d {
                    sort_key: FloatOrd(draw_batch.translation.z),
                    entity,
                    pipeline: id,
                    draw_function: draw_effects_function_2d,
                    batch_range: 0..1,
                    extracted_index: 0, // ???
                    extra_index: PhaseItemExtraIndex::None,
                    indexed: true, // ???
                },
                #[cfg(feature = "3d")]
                PipelineMode::Camera2d,
            );
        }
    }

    // Loop over all 3D cameras/views that need to render effects
    #[cfg(feature = "3d")]
    {
        #[cfg(feature = "trace")]
        let _span_draw = bevy::log::info_span!("draw_3d").entered();

        // Effects with full alpha blending
        if !views.is_empty() {
            trace!("Emit effect draw calls for alpha blended 3D views...");

            let draw_effects_function_3d = read_params
                .draw_functions_3d
                .read()
                .get_id::<DrawEffects>()
                .unwrap();

            emit_sorted_draw(
                &views,
                &mut transparent_3d_render_phases,
                &mut view_entities,
                &sorted_effect_batches,
                &effect_draw_batches,
                &mut render_pipeline,
                specialized_render_pipelines.reborrow(),
                &render_meshes,
                &pipeline_cache,
                |id, entity, batch, view| Transparent3d {
                    distance: view.rangefinder3d().distance(&batch.translation),
                    pipeline: id,
                    entity,
                    draw_function: draw_effects_function_3d,
                    batch_range: 0..1,
                    extra_index: PhaseItemExtraIndex::None,
                    indexed: true, // ???
                },
                #[cfg(feature = "2d")]
                PipelineMode::Camera3d,
            );
        }

        // Effects with alpha mask
        if !views.is_empty() {
            #[cfg(feature = "trace")]
            let _span_draw = bevy::log::info_span!("draw_alphamask").entered();

            trace!("Emit effect draw calls for alpha masked 3D views...");

            let draw_effects_function_alpha_mask = read_params
                .draw_functions_alpha_mask
                .read()
                .get_id::<DrawEffects>()
                .unwrap();

            emit_binned_draw(
                &views,
                &mut alpha_mask_3d_render_phases,
                &mut view_entities,
                &sorted_effect_batches,
                &effect_draw_batches,
                &mut render_pipeline,
                specialized_render_pipelines.reborrow(),
                &pipeline_cache,
                &render_meshes,
                |id, _batch, _view| OpaqueNoLightmap3dBatchSetKey {
                    pipeline: id,
                    draw_function: draw_effects_function_alpha_mask,
                    material_bind_group_index: None,
                    vertex_slab: default(),
                    index_slab: None,
                },
                // Unused for now
                || OpaqueNoLightmap3dBinKey {
                    asset_id: AssetId::<Mesh>::invalid().untyped(),
                },
                #[cfg(feature = "2d")]
                PipelineMode::Camera3d,
                ParticleRenderAlphaMaskPipelineKey::AlphaMask,
                &mut change_tick,
            );
        }

        // Opaque particles
        if !views.is_empty() {
            #[cfg(feature = "trace")]
            let _span_draw = bevy::log::info_span!("draw_opaque").entered();

            trace!("Emit effect draw calls for opaque 3D views...");

            let draw_effects_function_opaque = read_params
                .draw_functions_opaque
                .read()
                .get_id::<DrawEffects>()
                .unwrap();

            emit_binned_draw(
                &views,
                &mut opaque_3d_render_phases,
                &mut view_entities,
                &sorted_effect_batches,
                &effect_draw_batches,
                &mut render_pipeline,
                specialized_render_pipelines.reborrow(),
                &pipeline_cache,
                &render_meshes,
                |id, _batch, _view| Opaque3dBatchSetKey {
                    pipeline: id,
                    draw_function: draw_effects_function_opaque,
                    material_bind_group_index: None,
                    vertex_slab: default(),
                    index_slab: None,
                    lightmap_slab: None,
                },
                // Unused for now
                || Opaque3dBinKey {
                    asset_id: AssetId::<Mesh>::invalid().untyped(),
                },
                #[cfg(feature = "2d")]
                PipelineMode::Camera3d,
                ParticleRenderAlphaMaskPipelineKey::Opaque,
                &mut change_tick,
            );
        }
    }
}

/// Once a child effect is batched, and therefore passed validations to be
/// updated and rendered this frame, dispatch a new GPU operation to fill the
/// indirect dispatch args of its init pass based on the number of GPU events
/// emitted in the previous frame and stored in its event buffer.
pub fn queue_init_indirect_workgroup_update(
    q_cached_effects: Query<(
        Entity,
        &CachedChildInfo,
        &CachedEffectEvents,
        &CachedReadyState,
    )>,
    mut init_fill_dispatch_queue: ResMut<InitFillDispatchQueue>,
) {
    debug_assert_eq!(
        GpuChildInfo::min_size().get() % 4,
        0,
        "Invalid GpuChildInfo alignment."
    );

    // Schedule some GPU buffer operation to update the number of workgroups to
    // dispatch during the indirect init pass of this effect based on the number of
    // GPU spawn events written in its buffer.
    for (entity, cached_child_info, cached_effect_events, cached_ready_state) in &q_cached_effects {
        if !cached_ready_state.is_ready() {
            trace!(
                "[Effect {:?}] Skipping init_fill_dispatch.enqueue() because effect is not ready.",
                entity
            );
            continue;
        }
        let init_indirect_dispatch_index = cached_effect_events.init_indirect_dispatch_index;
        let global_child_index = cached_child_info.global_child_index;
        trace!(
            "[Effect {:?}] init_fill_dispatch.enqueue(): src:global_child_index={} dst:init_indirect_dispatch_index={}",
            entity,
            global_child_index,
            init_indirect_dispatch_index,
        );
        assert!(global_child_index != u32::MAX);
        init_fill_dispatch_queue.enqueue(global_child_index, init_indirect_dispatch_index);
    }
}

/// Prepare GPU resources for effect rendering.
///
/// This system runs in the [`RenderSet::PrepareResources`] render set, after
/// Bevy has updated the [`ViewUniforms`], which need to be referenced to get
/// access to the current camera view.
pub(crate) fn prepare_gpu_resources(
    mut effects_meta: ResMut<EffectsMeta>,
    //mut effect_cache: ResMut<EffectCache>,
    mut event_cache: ResMut<EventCache>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    mut sort_bind_groups: ResMut<SortBindGroups>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    view_uniforms: Res<ViewUniforms>,
    render_pipeline: Res<ParticlesRenderPipeline>,
    pipeline_cache: Res<PipelineCache>,
) {
    // Get the binding for the ViewUniform, the uniform data structure containing
    // the Camera data for the current view. If not available, we cannot render
    // anything.
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };

    // Upload simulation parameters for this frame
    let prev_buffer_id = effects_meta.sim_params_uniforms.buffer().map(|b| b.id());
    effects_meta
        .sim_params_uniforms
        .write_buffer(&render_device, &render_queue);
    if prev_buffer_id != effects_meta.sim_params_uniforms.buffer().map(|b| b.id()) {
        // Buffer changed, invalidate bind groups
        effects_meta.update_sim_params_bind_group = None;
        effects_meta.indirect_sim_params_bind_group = None;
    }

    // Create the bind group for the camera/view parameters
    // FIXME - Not here!
    effects_meta.view_bind_group = Some(render_device.create_bind_group(
        "hanabi:bind_group_camera_view",
        &pipeline_cache.get_bind_group_layout(&render_pipeline.view_layout_desc),
        &[
            BindGroupEntry {
                binding: 0,
                resource: view_binding,
            },
            BindGroupEntry {
                binding: 1,
                resource: effects_meta.sim_params_uniforms.binding().unwrap(),
            },
        ],
    ));

    // Re-/allocate the draw indirect args buffer if needed
    if effects_meta
        .draw_indirect_buffer
        .allocate_gpu(&render_device, &render_queue)
    {
        // All those bind groups use the buffer so need to be re-created
        trace!("*** Draw indirect args buffer re-allocated; clearing all bind groups using it.");
        effects_meta.update_sim_params_bind_group = None;
        effects_meta.indirect_metadata_bind_group = None;
    }

    // Re-/allocate any GPU buffer if needed
    //effect_cache.prepare_buffers(&render_device, &render_queue, &mut
    // effect_bind_groups);
    event_cache.prepare_buffers(&render_device, &render_queue, &mut effect_bind_groups);
    sort_bind_groups.prepare_buffers(&render_device);
    if effects_meta
        .dispatch_indirect_buffer
        .prepare_buffers(&render_device)
    {
        // All those bind groups use the buffer so need to be re-created
        trace!("*** Dispatch indirect buffer for update pass re-allocated; clearing all bind groups using it.");
        effect_bind_groups.particle_slabs.clear();
    }
}

/// Update the [`GpuEffectMetadata`] of all the effects queued for update/render
/// this frame.
///
/// By this point, all effects should have a [`CachedEffectMetadata`] with a
/// valid allocation in the GPU table for a [`GpuEffectMetadata`] entry. This
/// system actually synchronize the CPU value with the GPU one in case of
/// change.
pub(crate) fn prepare_effect_metadata(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut q_effects: Query<(
        MainEntity,
        Ref<ExtractedEffect>,
        Ref<CachedEffect>,
        Ref<DispatchBufferIndices>,
        Option<Ref<CachedChildInfo>>,
        Option<Ref<CachedParentInfo>>,
        Option<Ref<CachedDrawIndirectArgs>>,
        Option<Ref<CachedEffectEvents>>,
        &mut CachedEffectMetadata,
    )>,
    mut effects_meta: ResMut<EffectsMeta>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("prepare_effect_metadata").entered();
    trace!("prepare_effect_metadata");

    for (
        main_entity,
        extracted_effect,
        cached_effect,
        dispatch_buffer_indices,
        maybe_cached_child_info,
        maybe_cached_parent_info,
        maybe_cached_draw_indirect_args,
        maybe_cached_effect_events,
        mut cached_effect_metadata,
    ) in &mut q_effects
    {
        // Check if anything relevant to GpuEffectMetadata changed this frame; otherwise
        // early out and skip this effect.
        let is_changed_ee = extracted_effect.is_changed();
        let is_changed_ce = cached_effect.is_changed();
        let is_changed_dbi = dispatch_buffer_indices.is_changed();
        let is_changed_cci = maybe_cached_child_info
            .as_ref()
            .map(|cci| cci.is_changed())
            .unwrap_or(false);
        let is_changed_cpi = maybe_cached_parent_info
            .as_ref()
            .map(|cpi| cpi.is_changed())
            .unwrap_or(false);
        let is_changed_cdia = maybe_cached_draw_indirect_args
            .as_ref()
            .map(|cdia| cdia.is_changed())
            .unwrap_or(false);
        let is_changed_cee = maybe_cached_effect_events
            .as_ref()
            .map(|cee| cee.is_changed())
            .unwrap_or(false);
        trace!(
            "Preparting GpuEffectMetadata for effect {:?}: is_changed[] = {} {} {} {} {} {} {}",
            main_entity,
            is_changed_ee,
            is_changed_ce,
            is_changed_dbi,
            is_changed_cci,
            is_changed_cpi,
            is_changed_cdia,
            is_changed_cee
        );
        if !is_changed_ee
            && !is_changed_ce
            && !is_changed_dbi
            && !is_changed_cci
            && !is_changed_cpi
            && !is_changed_cdia
            && !is_changed_cee
        {
            continue;
        }

        let capacity = cached_effect.slice.len();

        // Global and local indices of this effect as a child of another (parent) effect
        let (global_child_index, local_child_index) = maybe_cached_child_info
            .map(|cci| (cci.global_child_index, cci.local_child_index))
            .unwrap_or((u32::MAX, u32::MAX));

        // Base index of all children of this (parent) effect
        let base_child_index = maybe_cached_parent_info
            .map(|cpi| {
                debug_assert_eq!(
                    cpi.byte_range.start % GpuChildInfo::SHADER_SIZE.get() as u32,
                    0
                );
                cpi.byte_range.start / GpuChildInfo::SHADER_SIZE.get() as u32
            })
            .unwrap_or(u32::MAX);

        let particle_stride = extracted_effect.particle_layout.min_binding_size32().get() / 4;
        let sort_key_offset = extracted_effect
            .particle_layout
            .byte_offset(Attribute::RIBBON_ID)
            .map(|byte_offset| byte_offset / 4)
            .unwrap_or(u32::MAX);
        let sort_key2_offset = extracted_effect
            .particle_layout
            .byte_offset(Attribute::AGE)
            .map(|byte_offset| byte_offset / 4)
            .unwrap_or(u32::MAX);

        let gpu_effect_metadata = GpuEffectMetadata {
            capacity,
            alive_count: 0,
            max_update: 0,
            max_spawn: capacity,
            indirect_write_index: 0,
            indirect_dispatch_index: dispatch_buffer_indices
                .update_dispatch_indirect_buffer_row_index,
            indirect_draw_index: maybe_cached_draw_indirect_args
                .map(|cdia| cdia.get_row().0)
                .unwrap_or(u32::MAX),
            init_indirect_dispatch_index: maybe_cached_effect_events
                .map(|cee| cee.init_indirect_dispatch_index)
                .unwrap_or(u32::MAX),
            local_child_index,
            global_child_index,
            base_child_index,
            particle_stride,
            sort_key_offset,
            sort_key2_offset,
            ..default()
        };

        // Insert of update entry in GPU buffer table
        assert!(cached_effect_metadata.table_id.is_valid());
        if gpu_effect_metadata != cached_effect_metadata.metadata {
            effects_meta
                .effect_metadata_buffer
                .update(cached_effect_metadata.table_id, gpu_effect_metadata);

            cached_effect_metadata.metadata = gpu_effect_metadata;

            // This triggers on all new spawns and annoys everyone; silence until we can at
            // least warn only on non-first-spawn, and ideally split indirect data from that
            // struct so we don't overwrite it and solve the issue.
            debug!(
                "Updated metadata entry {} for effect {:?}, this will reset it.",
                cached_effect_metadata.table_id.0, main_entity
            );
        }
    }

    // Once all EffectMetadata values are written, schedule a GPU upload
    if effects_meta
        .effect_metadata_buffer
        .allocate_gpu(render_device.as_ref(), render_queue.as_ref())
    {
        // All those bind groups use the buffer so need to be re-created
        trace!("*** Effect metadata buffer re-allocated; clearing all bind groups using it.");
        effects_meta.indirect_metadata_bind_group = None;
        effect_bind_groups.init_metadata_bind_groups.clear();
        effect_bind_groups.update_metadata_bind_groups.clear();
    }
}

/// Read the queued init fill dispatch operations, batch them together by
/// contiguous source and destination entries in the buffers, and enqueue
/// corresponding GPU buffer fill dispatch operations for all batches.
///
/// This system runs after the GPU buffers have been (re-)allocated in
/// [`prepare_gpu_resources()`], so that it can read the new buffer IDs and
/// reference them from the generic [`GpuBufferOperationQueue`].
pub(crate) fn queue_init_fill_dispatch_ops(
    event_cache: Res<EventCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut init_fill_dispatch_queue: ResMut<InitFillDispatchQueue>,
    mut gpu_buffer_operations: ResMut<GpuBufferOperations>,
) {
    // Submit all queued init fill dispatch operations with the proper buffers
    if !init_fill_dispatch_queue.is_empty() {
        let src_buffer = event_cache.child_infos().buffer();
        let dst_buffer = event_cache.init_indirect_dispatch_buffer();
        if let (Some(src_buffer), Some(dst_buffer)) = (src_buffer, dst_buffer) {
            init_fill_dispatch_queue.submit(src_buffer, dst_buffer, &mut gpu_buffer_operations);
        } else {
            if src_buffer.is_none() {
                warn!("Event cache has no allocated GpuChildInfo buffer, but there's {} init fill dispatch operation(s) queued. Ignoring those operations. This will prevent child particles from spawning.", init_fill_dispatch_queue.queue.len());
            }
            if dst_buffer.is_none() {
                warn!("Event cache has no allocated GpuDispatchIndirect buffer, but there's {} init fill dispatch operation(s) queued. Ignoring those operations. This will prevent child particles from spawning.", init_fill_dispatch_queue.queue.len());
            }
        }
    }

    // Once all GPU operations for this frame are enqueued, upload them to GPU
    gpu_buffer_operations.end_frame(&render_device, &render_queue);
}

#[derive(SystemParam)]
pub struct PipelineParams<'w, 's> {
    dispatch_indirect_pipeline: Res<'w, DispatchIndirectPipeline>,
    utils_pipeline: Res<'w, UtilsPipeline>,
    init_pipeline: Res<'w, ParticlesInitPipeline>,
    update_pipeline: Res<'w, ParticlesUpdatePipeline>,
    render_pipeline: ResMut<'w, ParticlesRenderPipeline>,
    marker: PhantomData<&'s usize>,
}

pub(crate) fn prepare_bind_groups(
    mut effects_meta: ResMut<EffectsMeta>,
    mut effect_cache: ResMut<EffectCache>,
    mut event_cache: ResMut<EventCache>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    mut property_bind_groups: ResMut<PropertyBindGroups>,
    mut sort_bind_groups: ResMut<SortBindGroups>,
    property_cache: Res<PropertyCache>,
    sorted_effect_batched: Res<SortedEffectBatches>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: PipelineParams,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut gpu_buffer_operation_queue: ResMut<GpuBufferOperations>,
) {
    // We can't simulate nor render anything without at least the spawner buffer
    if effects_meta.spawner_buffer.is_empty() {
        return;
    }
    let Some(spawner_buffer) = effects_meta.spawner_buffer.buffer().cloned() else {
        return;
    };

    // Workaround for too many params in system (TODO: refactor to split work?)
    let dispatch_indirect_pipeline = pipelines.dispatch_indirect_pipeline.into_inner();
    let utils_pipeline = pipelines.utils_pipeline.into_inner();
    let init_pipeline = pipelines.init_pipeline.into_inner();
    let update_pipeline = pipelines.update_pipeline.into_inner();
    let render_pipeline = pipelines.render_pipeline.into_inner();

    // Ensure child_infos@3 bind group for the indirect pass is available if needed.
    // This returns `None` if the buffer is not ready, either because it's not
    // created yet or because it's not needed (no child effect).
    event_cache.ensure_indirect_child_info_buffer_bind_group(&render_device);

    {
        #[cfg(feature = "trace")]
        let _span = bevy::log::info_span!("shared_bind_groups").entered();

        // Make a copy of the buffer IDs before borrowing effects_meta mutably in the
        // loop below. Also allows earlying out before doing any work in case some
        // buffer is missing.
        let Some(spawner_buffer) = effects_meta.spawner_buffer.buffer().cloned() else {
            return;
        };

        // Create the sim_params@0 bind group for the global simulation parameters,
        // which is shared by the init and update passes.
        if effects_meta.update_sim_params_bind_group.is_none() {
            if let Some(draw_indirect_buffer) = effects_meta.draw_indirect_buffer.buffer() {
                effects_meta.update_sim_params_bind_group = Some(render_device.create_bind_group(
                    "hanabi:bind_group:vfx_update:sim_params@0",
                    &pipeline_cache.get_bind_group_layout(&update_pipeline.sim_params_layout_desc),
                    &[
                        // @group(0) @binding(0) var<uniform> sim_params : SimParams;
                        BindGroupEntry {
                            binding: 0,
                            resource: effects_meta.sim_params_uniforms.binding().unwrap(),
                        },
                        // @group(0) @binding(1) var<storage, read_write> draw_indirect_buffer :
                        // array<DrawIndexedIndirectArgs>;
                        BindGroupEntry {
                            binding: 1,
                            resource: draw_indirect_buffer.as_entire_binding(),
                        },
                    ],
                ));
            } else {
                debug!("Cannot allocate bind group for vfx_update:sim_params@0 - draw_indirect_buffer not ready");
            }
        }
        if effects_meta.indirect_sim_params_bind_group.is_none() {
            effects_meta.indirect_sim_params_bind_group = Some(render_device.create_bind_group(
                "hanabi:bind_group:vfx_indirect:sim_params@0",
                &pipeline_cache.get_bind_group_layout(&init_pipeline.sim_params_layout_desc), // FIXME - Shared with init
                &[
                    // @group(0) @binding(0) var<uniform> sim_params : SimParams;
                    BindGroupEntry {
                        binding: 0,
                        resource: effects_meta.sim_params_uniforms.binding().unwrap(),
                    },
                ],
            ));
        }

        // Create the @1 bind group for the indirect dispatch preparation pass of all
        // effects at once
        effects_meta.indirect_metadata_bind_group = match (
            effects_meta.effect_metadata_buffer.buffer(),
            effects_meta.dispatch_indirect_buffer.buffer(),
            effects_meta.draw_indirect_buffer.buffer(),
        ) {
            (
                Some(effect_metadata_buffer),
                Some(dispatch_indirect_buffer),
                Some(draw_indirect_buffer),
            ) => {
                // Base bind group for indirect pass
                Some(render_device.create_bind_group(
                    "hanabi:bind_group:vfx_indirect:metadata@1",
                    &pipeline_cache.get_bind_group_layout(
                        &dispatch_indirect_pipeline.effect_metadata_bind_group_layout_desc,
                    ),
                    &[
                        // @group(1) @binding(0) var<storage, read_write> effect_metadata_buffer :
                        // array<u32>;
                        BindGroupEntry {
                            binding: 0,
                            resource: effect_metadata_buffer.as_entire_binding(),
                        },
                        // @group(1) @binding(1) var<storage, read_write> dispatch_indirect_buffer
                        // : array<DispatchIndirectArgs>;
                        BindGroupEntry {
                            binding: 1,
                            resource: dispatch_indirect_buffer.as_entire_binding(),
                        },
                        // @group(1) @binding(2) var<storage, read_write> draw_indirect_buffer :
                        // array<u32>;
                        BindGroupEntry {
                            binding: 2,
                            resource: draw_indirect_buffer.as_entire_binding(),
                        },
                    ],
                ))
            }

            // Some buffer is not yet available, can't create the bind group
            _ => None,
        };

        // Create the @2 bind group for the indirect dispatch preparation pass of all
        // effects at once
        if effects_meta.indirect_spawner_bind_group.is_none() {
            let bind_group = render_device.create_bind_group(
                "hanabi:bind_group:vfx_indirect:spawner@2",
                &pipeline_cache.get_bind_group_layout(
                    &dispatch_indirect_pipeline.spawner_bind_group_layout_desc,
                ),
                &[
                    // @group(2) @binding(0) var<storage, read> spawner_buffer : array<Spawner>;
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &spawner_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            );

            effects_meta.indirect_spawner_bind_group = Some(bind_group);
        }
    }

    // Create the per-slab bind groups
    trace!("Create per-slab bind groups...");
    for (slab_index, particle_slab) in effect_cache.slabs().iter().enumerate() {
        #[cfg(feature = "trace")]
        let _span_buffer = bevy::log::info_span!("create_buffer_bind_groups").entered();

        let Some(particle_slab) = particle_slab else {
            trace!(
                "Particle slab index #{} has no allocated EffectBuffer, skipped.",
                slab_index
            );
            continue;
        };

        // Ensure all effects in this batch have a bind group for the entire buffer of
        // the group, since the update phase runs on an entire group/buffer at once,
        // with all the effect instances in it batched together.
        trace!("effect particle slab_index=#{}", slab_index);
        effect_bind_groups
            .particle_slabs
            .entry(SlabId::new(slab_index as u32))
            .or_insert_with(|| {
                // Bind group particle@1 for render pass
                trace!("Creating particle@1 bind group for buffer #{slab_index} in render pass");
                let spawner_min_binding_size = GpuSpawnerParams::aligned_size(
                    render_device.limits().min_storage_buffer_offset_alignment,
                );
                let entries = [
                    // @group(1) @binding(0) var<storage, read> particle_buffer : ParticleBuffer;
                    BindGroupEntry {
                        binding: 0,
                        resource: particle_slab.as_entire_binding_particle(),
                    },
                    // @group(1) @binding(1) var<storage, read> indirect_buffer : IndirectBuffer;
                    BindGroupEntry {
                        binding: 1,
                        resource: particle_slab.as_entire_binding_indirect(),
                    },
                    // @group(1) @binding(2) var<storage, read> spawner : Spawner;
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &spawner_buffer,
                            offset: 0,
                            size: Some(spawner_min_binding_size),
                        }),
                    },
                ];
                let render = render_device.create_bind_group(
                    &format!("hanabi:bind_group:render:particles@1:vfx{slab_index}")[..],
                    particle_slab.render_particles_buffer_layout(),
                    &entries[..],
                );

                BufferBindGroups { render }
            });
    }

    // Create bind groups for queued GPU buffer operations
    gpu_buffer_operation_queue.create_bind_groups(&render_device, utils_pipeline);

    // Create the per-effect bind groups
    let spawner_buffer_binding_size =
        NonZeroU64::new(effects_meta.spawner_buffer.aligned_size() as u64).unwrap();
    for effect_batch in sorted_effect_batched.iter() {
        #[cfg(feature = "trace")]
        let _span_buffer = bevy::log::info_span!("create_batch_bind_groups").entered();

        // Create the property bind group @2 if needed
        if let Some(property_key) = &effect_batch.property_key {
            if let Err(err) = property_bind_groups.ensure_exists(
                property_key,
                &property_cache,
                &spawner_buffer,
                spawner_buffer_binding_size,
                &render_device,
                &pipeline_cache,
            ) {
                error!("Failed to create property bind group for effect batch: {err:?}");
                continue;
            }
        } else if let Err(err) = property_bind_groups.ensure_exists_no_property(
            &property_cache,
            &spawner_buffer,
            spawner_buffer_binding_size,
            &render_device,
            &pipeline_cache,
        ) {
            error!("Failed to create property bind group for effect batch: {err:?}");
            continue;
        }

        // Bind group particle@1 for the simulate compute shaders (init and udpate) to
        // simulate particles.
        if effect_cache
            .create_particle_sim_bind_group(
                &effect_batch.slab_id,
                &render_device,
                effect_batch.particle_layout.min_binding_size32(),
                effect_batch.parent_min_binding_size,
                effect_batch.parent_binding_source.as_ref(),
                &pipeline_cache,
            )
            .is_err()
        {
            error!("No particle buffer allocated for effect batch.");
            continue;
        }

        // Bind group @3 of init pass
        // FIXME - this is instance-dependent, not buffer-dependent
        {
            let consume_gpu_spawn_events = effect_batch
                .layout_flags
                .contains(LayoutFlags::CONSUME_GPU_SPAWN_EVENTS);
            let consume_event_buffers = if let BatchSpawnInfo::GpuSpawner { .. } =
                effect_batch.spawn_info
            {
                assert!(consume_gpu_spawn_events);
                let cached_effect_events = effect_batch.cached_effect_events.as_ref().unwrap();
                Some(ConsumeEventBuffers {
                    child_infos_buffer: event_cache.child_infos_buffer().unwrap(),
                    events: BufferSlice {
                        buffer: event_cache
                            .get_buffer(cached_effect_events.buffer_index)
                            .unwrap(),
                        // Note: event range is in u32 count, not bytes
                        offset: cached_effect_events.range.start * 4,
                        size: NonZeroU32::new(cached_effect_events.range.len() as u32 * 4).unwrap(),
                    },
                })
            } else {
                assert!(!consume_gpu_spawn_events);
                None
            };
            let Some(init_metadata_layout_desc) =
                effect_cache.metadata_init_bind_group_layout_desc(consume_gpu_spawn_events)
            else {
                continue;
            };
            if effect_bind_groups
                .get_or_create_init_metadata(
                    effect_batch,
                    &render_device,
                    &pipeline_cache.get_bind_group_layout(init_metadata_layout_desc),
                    effects_meta.effect_metadata_buffer.buffer().unwrap(),
                    consume_event_buffers,
                )
                .is_err()
            {
                continue;
            }
        }

        // Bind group @3 of update pass
        // FIXME - this is instance-dependent, not buffer-dependent#
        {
            let num_event_buffers = effect_batch.child_event_buffers.len() as u32;

            let Some(update_metadata_layout_desc) =
                effect_cache.metadata_update_bind_group_layout_desc(num_event_buffers)
            else {
                continue;
            };
            if effect_bind_groups
                .get_or_create_update_metadata(
                    effect_batch,
                    &render_device,
                    &pipeline_cache.get_bind_group_layout(update_metadata_layout_desc),
                    effects_meta.effect_metadata_buffer.buffer().unwrap(),
                    event_cache.child_infos_buffer(),
                    &effect_batch.child_event_buffers[..],
                )
                .is_err()
            {
                continue;
            }
        }

        if effect_batch.layout_flags.contains(LayoutFlags::RIBBONS) {
            let effect_buffer = effect_cache.get_slab(&effect_batch.slab_id).unwrap();

            // Bind group @0 of sort-fill pass
            let particle_buffer = effect_buffer.particle_buffer();
            let indirect_index_buffer = effect_buffer.indirect_index_buffer();
            let effect_metadata_buffer = effects_meta.effect_metadata_buffer.buffer().unwrap();
            if let Err(err) = sort_bind_groups.ensure_sort_fill_bind_group(
                &effect_batch.particle_layout,
                particle_buffer,
                indirect_index_buffer,
                effect_metadata_buffer,
                &spawner_buffer,
                &pipeline_cache,
            ) {
                error!(
                    "Failed to create sort-fill bind group @0 for ribbon effect: {:?}",
                    err
                );
                continue;
            }

            // Bind group @0 of sort pass
            if let Err(err) = sort_bind_groups.ensure_sort_bind_group(&pipeline_cache) {
                error!(
                    "Failed to create sort bind group @0 for ribbon effect: {:?}",
                    err
                );
                continue;
            }

            // Bind group @0 of sort-copy pass
            let indirect_index_buffer = effect_buffer.indirect_index_buffer();
            if let Err(err) = sort_bind_groups.ensure_sort_copy_bind_group(
                indirect_index_buffer,
                effect_metadata_buffer,
                &spawner_buffer,
                &pipeline_cache,
            ) {
                error!(
                    "Failed to create sort-copy bind group @0 for ribbon effect: {:?}",
                    err
                );
                continue;
            }
        }

        // Ensure the particle texture(s) are available as GPU resources and that a bind
        // group for them exists
        // FIXME fix this insert+get below
        if !effect_batch.texture_layout.layout.is_empty() {
            // This should always be available, as this is cached into the render pipeline
            // just before we start specializing it.
            let Some(material_bind_group_layout_desc) =
                render_pipeline.get_material(&effect_batch.texture_layout)
            else {
                error!(
                    "Failed to find material bind group layout for particle slab #{}",
                    effect_batch.slab_id.index()
                );
                continue;
            };

            // TODO = move
            let material = Material {
                layout: effect_batch.texture_layout.clone(),
                textures: effect_batch.textures.iter().map(|h| h.id()).collect(),
            };
            assert_eq!(material.layout.layout.len(), material.textures.len());

            //let bind_group_entries = material.make_entries(&gpu_images).unwrap();
            let Ok(bind_group_entries) = material.make_entries(&gpu_images) else {
                trace!(
                    "Temporarily ignoring material {:?} due to missing image(s)",
                    material
                );
                continue;
            };

            effect_bind_groups
                .material_bind_groups
                .entry(material.clone())
                .or_insert_with(|| {
                    debug!("Creating material bind group for material {:?}", material);
                    render_device.create_bind_group(
                        &format!(
                            "hanabi:material_bind_group_{}",
                            material.layout.layout.len()
                        )[..],
                        &pipeline_cache.get_bind_group_layout(material_bind_group_layout_desc),
                        &bind_group_entries[..],
                    )
                });
        }
    }
}

type DrawEffectsSystemState = SystemState<(
    SRes<EffectsMeta>,
    SRes<EffectBindGroups>,
    SRes<PipelineCache>,
    SRes<RenderAssets<RenderMesh>>,
    SRes<MeshAllocator>,
    SQuery<Read<ViewUniformOffset>>,
    SRes<SortedEffectBatches>,
    SQuery<Read<EffectDrawBatch>>,
)>;

/// Draw function for rendering all active effects for the current frame.
///
/// Effects are rendered in the [`Transparent2d`] phase of the main 2D pass,
/// and the [`Transparent3d`] phase of the main 3D pass.
pub(crate) struct DrawEffects {
    params: DrawEffectsSystemState,
}

impl DrawEffects {
    pub fn new(world: &mut World) -> Self {
        Self {
            params: SystemState::new(world),
        }
    }
}

/// Draw all particles of a single effect in view, in 2D or 3D.
///
/// FIXME: use pipeline ID to look up which group index it is.
fn draw<'w>(
    world: &'w World,
    pass: &mut TrackedRenderPass<'w>,
    view: Entity,
    entity: (Entity, MainEntity),
    pipeline_id: CachedRenderPipelineId,
    params: &mut DrawEffectsSystemState,
) {
    let (
        effects_meta,
        effect_bind_groups,
        pipeline_cache,
        meshes,
        mesh_allocator,
        views,
        sorted_effect_batches,
        effect_draw_batches,
    ) = params.get(world);
    let view_uniform = views.get(view).unwrap();
    let effects_meta = effects_meta.into_inner();
    let effect_bind_groups = effect_bind_groups.into_inner();
    let meshes = meshes.into_inner();
    let mesh_allocator = mesh_allocator.into_inner();
    let effect_draw_batch = effect_draw_batches.get(entity.0).unwrap();
    let effect_batch = sorted_effect_batches
        .get(effect_draw_batch.effect_batch_index)
        .unwrap();

    let Some(pipeline) = pipeline_cache.into_inner().get_render_pipeline(pipeline_id) else {
        return;
    };

    trace!("render pass");

    pass.set_render_pipeline(pipeline);

    let Some(render_mesh): Option<&RenderMesh> = meshes.get(effect_batch.mesh) else {
        return;
    };
    let Some(vertex_buffer_slice) = mesh_allocator.mesh_vertex_slice(&effect_batch.mesh) else {
        return;
    };

    // Vertex buffer containing the particle model to draw. Generally a quad.
    // FIXME - need to upload "vertex_buffer_slice.range.start as i32" into
    // "base_vertex" in the indirect struct...
    pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));

    // View properties (camera matrix, etc.)
    pass.set_bind_group(
        0,
        effects_meta.view_bind_group.as_ref().unwrap(),
        &[view_uniform.offset],
    );

    // Particles buffer
    let spawner_base = effect_batch.spawner_base;
    let spawner_buffer_aligned = effects_meta.spawner_buffer.aligned_size();
    assert!(spawner_buffer_aligned >= GpuSpawnerParams::min_size().get() as usize);
    let spawner_offset = spawner_base * spawner_buffer_aligned as u32;
    pass.set_bind_group(
        1,
        effect_bind_groups
            .particle_render(&effect_batch.slab_id)
            .unwrap(),
        &[spawner_offset],
    );

    // Particle texture
    // TODO = move
    let material = Material {
        layout: effect_batch.texture_layout.clone(),
        textures: effect_batch.textures.iter().map(|h| h.id()).collect(),
    };
    if !effect_batch.texture_layout.layout.is_empty() {
        if let Some(bind_group) = effect_bind_groups.material_bind_groups.get(&material) {
            pass.set_bind_group(2, bind_group, &[]);
        } else {
            // Texture(s) not ready; skip this drawing for now
            trace!(
                "Particle material bind group not available for batch slab_id={}. Skipping draw call.",
                effect_batch.slab_id.index(),
            );
            return;
        }
    }

    let draw_indirect_index = effect_batch.draw_indirect_buffer_row_index.0;
    assert_eq!(GpuDrawIndexedIndirectArgs::SHADER_SIZE.get(), 20);
    let draw_indirect_offset =
        draw_indirect_index as u64 * GpuDrawIndexedIndirectArgs::SHADER_SIZE.get();
    trace!(
        "Draw up to {} particles with {} vertices per particle for batch from particle slab #{} \
            (effect_metadata_index={}, draw_indirect_offset={}B).",
        effect_batch.slice.len(),
        render_mesh.vertex_count,
        effect_batch.slab_id.index(),
        draw_indirect_index,
        draw_indirect_offset,
    );

    let Some(indirect_buffer) = effects_meta.draw_indirect_buffer.buffer() else {
        trace!(
            "The draw indirect buffer containing the indirect draw args is not ready for batch slab_id=#{}. Skipping draw call.",
            effect_batch.slab_id.index(),
        );
        return;
    };

    match render_mesh.buffer_info {
        RenderMeshBufferInfo::Indexed { index_format, .. } => {
            let Some(index_buffer_slice) = mesh_allocator.mesh_index_slice(&effect_batch.mesh)
            else {
                trace!(
                    "The index buffer for indexed rendering is not ready for batch slab_id=#{}. Skipping draw call.",
                    effect_batch.slab_id.index(),
                );
                return;
            };

            pass.set_index_buffer(index_buffer_slice.buffer.slice(..), index_format);
            pass.draw_indexed_indirect(indirect_buffer, draw_indirect_offset);
        }
        RenderMeshBufferInfo::NonIndexed => {
            pass.draw_indirect(indirect_buffer, draw_indirect_offset);
        }
    }
}

#[cfg(feature = "2d")]
impl Draw<Transparent2d> for DrawEffects {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        item: &Transparent2d,
    ) -> Result<(), DrawError> {
        trace!("Draw<Transparent2d>: view={:?}", view);
        draw(
            world,
            pass,
            view,
            item.entity,
            item.pipeline,
            &mut self.params,
        );
        Ok(())
    }
}

#[cfg(feature = "3d")]
impl Draw<Transparent3d> for DrawEffects {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        item: &Transparent3d,
    ) -> Result<(), DrawError> {
        trace!("Draw<Transparent3d>: view={:?}", view);
        draw(
            world,
            pass,
            view,
            item.entity,
            item.pipeline,
            &mut self.params,
        );
        Ok(())
    }
}

#[cfg(feature = "3d")]
impl Draw<AlphaMask3d> for DrawEffects {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        item: &AlphaMask3d,
    ) -> Result<(), DrawError> {
        trace!("Draw<AlphaMask3d>: view={:?}", view);
        draw(
            world,
            pass,
            view,
            item.representative_entity,
            item.batch_set_key.pipeline,
            &mut self.params,
        );
        Ok(())
    }
}

#[cfg(feature = "3d")]
impl Draw<Opaque3d> for DrawEffects {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        item: &Opaque3d,
    ) -> Result<(), DrawError> {
        trace!("Draw<Opaque3d>: view={:?}", view);
        draw(
            world,
            pass,
            view,
            item.representative_entity,
            item.batch_set_key.pipeline,
            &mut self.params,
        );
        Ok(())
    }
}

/// Render node to run the simulation sub-graph once per frame.
///
/// This node doesn't simulate anything by itself, but instead schedules the
/// simulation sub-graph, where other nodes like [`VfxSimulateNode`] do the
/// actual simulation.
///
/// The simulation sub-graph is scheduled to run before the [`CameraDriverNode`]
/// renders all the views, such that rendered views have access to the
/// just-simulated particles to render them.
///
/// [`CameraDriverNode`]: bevy::render::camera::CameraDriverNode
pub(crate) struct VfxSimulateDriverNode;

impl Node for VfxSimulateDriverNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        _render_context: &mut RenderContext,
        _world: &World,
    ) -> Result<(), NodeRunError> {
        graph.run_sub_graph(
            crate::plugin::simulate_graph::HanabiSimulateGraph,
            vec![],
            None,
            Some("hanabi".to_string()),
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HanabiPipelineId {
    Invalid,
    Cached(CachedComputePipelineId),
}

#[derive(Debug)]
pub(crate) enum ComputePipelineError {
    Queued,
    Creating,
    Error,
}

impl From<&CachedPipelineState> for ComputePipelineError {
    fn from(value: &CachedPipelineState) -> Self {
        match value {
            CachedPipelineState::Queued => Self::Queued,
            CachedPipelineState::Creating(_) => Self::Creating,
            CachedPipelineState::Err(_) => Self::Error,
            _ => panic!("Trying to convert Ok state to error."),
        }
    }
}

pub(crate) struct HanabiComputePass<'a> {
    /// Pipeline cache to fetch cached compute pipelines by ID.
    pipeline_cache: &'a PipelineCache,
    /// WGPU compute pass.
    compute_pass: ComputePass<'a>,
    /// Current pipeline (cached).
    pipeline_id: HanabiPipelineId,
}

impl<'a> Deref for HanabiComputePass<'a> {
    type Target = ComputePass<'a>;

    fn deref(&self) -> &Self::Target {
        &self.compute_pass
    }
}

impl DerefMut for HanabiComputePass<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.compute_pass
    }
}

impl<'a> HanabiComputePass<'a> {
    pub fn new(pipeline_cache: &'a PipelineCache, compute_pass: ComputePass<'a>) -> Self {
        Self {
            pipeline_cache,
            compute_pass,
            pipeline_id: HanabiPipelineId::Invalid,
        }
    }

    pub fn set_cached_compute_pipeline(
        &mut self,
        pipeline_id: CachedComputePipelineId,
    ) -> Result<(), ComputePipelineError> {
        if HanabiPipelineId::Cached(pipeline_id) == self.pipeline_id {
            trace!("set_cached_compute_pipeline() id={pipeline_id:?} -> already set; skipped");
            return Ok(());
        }
        trace!("set_cached_compute_pipeline() id={pipeline_id:?}");
        let Some(pipeline) = self.pipeline_cache.get_compute_pipeline(pipeline_id) else {
            let state = self.pipeline_cache.get_compute_pipeline_state(pipeline_id);
            if let CachedPipelineState::Err(err) = state {
                error!(
                    "Failed to find compute pipeline #{}: {:?}",
                    pipeline_id.id(),
                    err
                );
            } else {
                debug!("Compute pipeline not ready #{}", pipeline_id.id());
            }
            return Err(state.into());
        };
        self.compute_pass.set_pipeline(pipeline);
        self.pipeline_id = HanabiPipelineId::Cached(pipeline_id);
        Ok(())
    }
}

/// Render node to run the simulation of all effects once per frame.
///
/// Runs inside the simulation sub-graph, looping over all extracted effect
/// batches to simulate them.
pub(crate) struct VfxSimulateNode {}

impl VfxSimulateNode {
    /// Create a new node for simulating the effects of the given world.
    pub fn new(_world: &mut World) -> Self {
        Self {}
    }

    /// Begin a new compute pass and return a wrapper with extra
    /// functionalities.
    pub fn begin_compute_pass<'encoder>(
        &self,
        label: &str,
        pipeline_cache: &'encoder PipelineCache,
        render_context: &'encoder mut RenderContext,
    ) -> HanabiComputePass<'encoder> {
        let compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
        HanabiComputePass::new(pipeline_cache, compute_pass)
    }
}

impl Node for VfxSimulateNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![]
    }

    fn update(&mut self, _world: &mut World) {}

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        trace!("VfxSimulateNode::run()");

        let pipeline_cache = world.resource::<PipelineCache>();
        let effects_meta = world.resource::<EffectsMeta>();
        let effect_bind_groups = world.resource::<EffectBindGroups>();
        let property_bind_groups = world.resource::<PropertyBindGroups>();
        let sort_bind_groups = world.resource::<SortBindGroups>();
        let utils_pipeline = world.resource::<UtilsPipeline>();
        let effect_cache = world.resource::<EffectCache>();
        let event_cache = world.resource::<EventCache>();
        let gpu_buffer_operations = world.resource::<GpuBufferOperations>();
        let sorted_effect_batches = world.resource::<SortedEffectBatches>();
        let init_fill_dispatch_queue = world.resource::<InitFillDispatchQueue>();

        // Make sure to schedule any buffer copy before accessing their content later in
        // the GPU commands below.
        {
            let command_encoder = render_context.command_encoder();
            effects_meta
                .dispatch_indirect_buffer
                .write_buffers(command_encoder);
            effects_meta
                .draw_indirect_buffer
                .write_buffer(command_encoder);
            effects_meta
                .effect_metadata_buffer
                .write_buffer(command_encoder);
            event_cache.write_buffers(command_encoder);
            sort_bind_groups.write_buffers(command_encoder);
        }

        // Compute init fill dispatch pass - Fill the indirect dispatch structs for any
        // upcoming init pass of this frame, based on the GPU spawn events emitted by
        // the update pass of their parent effect during the previous frame.
        if let Some(queue_index) = init_fill_dispatch_queue.submitted_queue_index.as_ref() {
            gpu_buffer_operations.dispatch(
                *queue_index,
                render_context,
                utils_pipeline,
                Some("hanabi:init_indirect_fill_dispatch"),
            );
        }

        // If there's no batch, there's nothing more to do. Avoid continuing because
        // some GPU resources are missing, which is expected when there's no effect but
        // is an error (and will log warnings/errors) otherwise.
        if sorted_effect_batches.is_empty() {
            return Ok(());
        }

        // Compute init pass
        {
            trace!("init: loop over effect batches...");

            let mut compute_pass =
                self.begin_compute_pass("hanabi:init", pipeline_cache, render_context);

            // Bind group simparams@0 is common to everything, only set once per init pass
            compute_pass.set_bind_group(
                0,
                effects_meta
                    .indirect_sim_params_bind_group
                    .as_ref()
                    .unwrap(),
                &[],
            );

            // Dispatch init compute jobs for all batches
            for effect_batch in sorted_effect_batches.iter() {
                // Do not dispatch any init work if there's nothing to spawn this frame for the
                // batch. Note that this hopefully should have been skipped earlier.
                {
                    let use_indirect_dispatch = effect_batch
                        .layout_flags
                        .contains(LayoutFlags::CONSUME_GPU_SPAWN_EVENTS);
                    match effect_batch.spawn_info {
                        BatchSpawnInfo::CpuSpawner { total_spawn_count } => {
                            assert!(!use_indirect_dispatch);
                            if total_spawn_count == 0 {
                                continue;
                            }
                        }
                        BatchSpawnInfo::GpuSpawner { .. } => {
                            assert!(use_indirect_dispatch);
                        }
                    }
                }

                // Fetch bind group particle@1
                let Some(particle_bind_group) =
                    effect_cache.particle_sim_bind_group(&effect_batch.slab_id)
                else {
                    error!(
                        "Failed to find init particle@1 bind group for slab #{}",
                        effect_batch.slab_id.index()
                    );
                    continue;
                };

                // Fetch bind group metadata@3
                let Some(metadata_bind_group) = effect_bind_groups
                    .init_metadata_bind_groups
                    .get(&effect_batch.slab_id)
                else {
                    error!(
                        "Failed to find init metadata@3 bind group for slab #{}",
                        effect_batch.slab_id.index()
                    );
                    continue;
                };

                if compute_pass
                    .set_cached_compute_pipeline(effect_batch.init_and_update_pipeline_ids.init)
                    .is_err()
                {
                    continue;
                }

                // Compute dynamic offsets
                let spawner_base = effect_batch.spawner_base;
                let spawner_aligned_size = effects_meta.spawner_buffer.aligned_size();
                debug_assert!(spawner_aligned_size >= GpuSpawnerParams::min_size().get() as usize);
                let spawner_offset = spawner_base * spawner_aligned_size as u32;
                let property_offset = effect_batch.property_offset;

                // Setup init pass
                compute_pass.set_bind_group(1, particle_bind_group, &[]);
                let offsets = if let Some(property_offset) = property_offset {
                    vec![spawner_offset, property_offset]
                } else {
                    vec![spawner_offset]
                };
                compute_pass.set_bind_group(
                    2,
                    property_bind_groups
                        .get(effect_batch.property_key.as_ref())
                        .unwrap(),
                    &offsets[..],
                );
                compute_pass.set_bind_group(3, &metadata_bind_group.bind_group, &[]);

                // Dispatch init job
                match effect_batch.spawn_info {
                    // Indirect dispatch via GPU spawn events
                    BatchSpawnInfo::GpuSpawner {
                        init_indirect_dispatch_index,
                        ..
                    } => {
                        assert!(effect_batch
                            .layout_flags
                            .contains(LayoutFlags::CONSUME_GPU_SPAWN_EVENTS));

                        // Note: the indirect offset of a dispatch workgroup only needs
                        // 4-byte alignment
                        assert_eq!(GpuDispatchIndirectArgs::min_size().get(), 12);
                        let indirect_offset = init_indirect_dispatch_index as u64 * 12;

                        trace!(
                            "record commands for indirect init pipeline of effect {:?} \
                                init_indirect_dispatch_index={} \
                                indirect_offset={} \
                                spawner_base={} \
                                spawner_offset={} \
                                property_key={:?}...",
                            effect_batch.handle,
                            init_indirect_dispatch_index,
                            indirect_offset,
                            spawner_base,
                            spawner_offset,
                            effect_batch.property_key,
                        );

                        compute_pass.dispatch_workgroups_indirect(
                            event_cache.init_indirect_dispatch_buffer().unwrap(),
                            indirect_offset,
                        );
                    }

                    // Direct dispatch via CPU spawn count
                    BatchSpawnInfo::CpuSpawner {
                        total_spawn_count: spawn_count,
                    } => {
                        assert!(!effect_batch
                            .layout_flags
                            .contains(LayoutFlags::CONSUME_GPU_SPAWN_EVENTS));

                        const WORKGROUP_SIZE: u32 = 64;
                        let workgroup_count = spawn_count.div_ceil(WORKGROUP_SIZE);

                        trace!(
                            "record commands for init pipeline of effect {:?} \
                                (spawn {} particles => {} workgroups) spawner_base={} \
                                spawner_offset={} \
                                property_key={:?}...",
                            effect_batch.handle,
                            spawn_count,
                            workgroup_count,
                            spawner_base,
                            spawner_offset,
                            effect_batch.property_key,
                        );

                        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                    }
                }

                trace!("init compute dispatched");
            }
        }

        // Compute indirect dispatch pass
        if let (
            Some(_),
            true,
            Some(indirect_metadata_bind_group),
            Some(indirect_sim_params_bind_group),
            Some(indirect_spawner_bind_group),
        ) = (
            effects_meta.spawner_buffer.buffer(),
            !effects_meta.spawner_buffer.is_empty(),
            &effects_meta.indirect_metadata_bind_group,
            &effects_meta.indirect_sim_params_bind_group,
            &effects_meta.indirect_spawner_bind_group,
        ) {
            // Only start a compute pass if there's an effect; makes things clearer in
            // debugger.
            let mut compute_pass =
                self.begin_compute_pass("hanabi:indirect_dispatch", pipeline_cache, render_context);

            // Dispatch indirect dispatch compute job
            trace!("record commands for indirect dispatch pipeline...");

            let has_gpu_spawn_events = !event_cache.child_infos().is_empty();
            if has_gpu_spawn_events {
                if let Some(indirect_child_info_buffer_bind_group) =
                    event_cache.indirect_child_info_buffer_bind_group()
                {
                    assert!(has_gpu_spawn_events);
                    compute_pass.set_bind_group(3, indirect_child_info_buffer_bind_group, &[]);
                } else {
                    error!("Missing child_info_buffer@3 bind group for the vfx_indirect pass.");
                    // render_context
                    //     .command_encoder()
                    //     .insert_debug_marker("ERROR:MissingIndirectBindGroup3");
                    // FIXME - Bevy doesn't allow returning custom errors here...
                    return Ok(());
                }
            }

            if compute_pass
                .set_cached_compute_pipeline(effects_meta.active_indirect_pipeline_id)
                .is_err()
            {
                // FIXME - Bevy doesn't allow returning custom errors here...
                return Ok(());
            }

            //error!("FIXME - effect_metadata_buffer has gaps!!!! this won't work. len() is
            // the size exluding gaps!");
            const WORKGROUP_SIZE: u32 = 64;
            //let total_effect_count = effects_meta.effect_metadata_buffer.len();
            let total_effect_count = effects_meta.spawner_buffer.len() as u32;
            let workgroup_count = total_effect_count.div_ceil(WORKGROUP_SIZE);

            // Setup vfx_indirect pass
            compute_pass.set_bind_group(0, indirect_sim_params_bind_group, &[]);
            compute_pass.set_bind_group(1, indirect_metadata_bind_group, &[]);
            compute_pass.set_bind_group(2, indirect_spawner_bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            trace!(
                "indirect dispatch compute dispatched: total_effect_count={} workgroup_count={}",
                total_effect_count,
                workgroup_count
            );
        }

        // Compute update pass
        {
            let Some(indirect_buffer) = effects_meta.dispatch_indirect_buffer.buffer() else {
                warn!("Missing indirect buffer for update pass, cannot dispatch anything.");
                render_context
                    .command_encoder()
                    .insert_debug_marker("ERROR:MissingUpdateIndirectBuffer");
                // FIXME - Bevy doesn't allow returning custom errors here...
                return Ok(());
            };

            let mut compute_pass =
                self.begin_compute_pass("hanabi:update", pipeline_cache, render_context);

            // Bind group simparams@0 is common to everything, only set once per update pass
            compute_pass.set_bind_group(
                0,
                effects_meta.update_sim_params_bind_group.as_ref().unwrap(),
                &[],
            );

            // Dispatch update compute jobs
            for effect_batch in sorted_effect_batches.iter() {
                // Fetch bind group particle@1
                let Some(particle_bind_group) =
                    effect_cache.particle_sim_bind_group(&effect_batch.slab_id)
                else {
                    error!(
                        "Failed to find update particle@1 bind group for slab #{}",
                        effect_batch.slab_id.index()
                    );
                    compute_pass.insert_debug_marker("ERROR:MissingParticleSimBindGroup");
                    continue;
                };

                // Fetch bind group metadata@3
                let Some(metadata_bind_group) = effect_bind_groups
                    .update_metadata_bind_groups
                    .get(&effect_batch.slab_id)
                else {
                    error!(
                        "Failed to find update metadata@3 bind group for slab #{}",
                        effect_batch.slab_id.index()
                    );
                    compute_pass.insert_debug_marker("ERROR:MissingMetadataBindGroup");
                    continue;
                };

                // Fetch compute pipeline
                if let Err(err) = compute_pass
                    .set_cached_compute_pipeline(effect_batch.init_and_update_pipeline_ids.update)
                {
                    compute_pass.insert_debug_marker(&format!(
                        "ERROR:FailedToSetCachedUpdatePipeline:{:?}",
                        err
                    ));
                    continue;
                }

                // Compute dynamic offsets
                let spawner_base = effect_batch.spawner_base;
                let spawner_aligned_size = effects_meta.spawner_buffer.aligned_size();
                assert!(spawner_aligned_size >= GpuSpawnerParams::min_size().get() as usize);
                let spawner_offset = spawner_base * spawner_aligned_size as u32;
                let property_offset = effect_batch.property_offset;

                trace!(
                    "record commands for update pipeline of effect {:?} spawner_base={}",
                    effect_batch.handle,
                    spawner_base,
                );

                // Setup update pass
                compute_pass.set_bind_group(1, particle_bind_group, &[]);
                let offsets = if let Some(property_offset) = property_offset {
                    vec![spawner_offset, property_offset]
                } else {
                    vec![spawner_offset]
                };
                compute_pass.set_bind_group(
                    2,
                    property_bind_groups
                        .get(effect_batch.property_key.as_ref())
                        .unwrap(),
                    &offsets[..],
                );
                compute_pass.set_bind_group(3, &metadata_bind_group.bind_group, &[]);

                // Dispatch update job
                let dispatch_indirect_offset = effect_batch
                    .dispatch_buffer_indices
                    .update_dispatch_indirect_buffer_row_index
                    * 12;
                trace!(
                    "dispatch_workgroups_indirect: buffer={:?} offset={}B",
                    indirect_buffer,
                    dispatch_indirect_offset,
                );
                compute_pass
                    .dispatch_workgroups_indirect(indirect_buffer, dispatch_indirect_offset as u64);

                trace!("update compute dispatched");
            }
        }

        // Compute sort fill dispatch pass - Fill the indirect dispatch structs for any
        // batch of particles which needs sorting, based on the actual number of alive
        // particles in the batch after their update in the compute update pass. Since
        // particles may die during update, this may be different from the number of
        // particles updated.
        if let Some(queue_index) = sorted_effect_batches.dispatch_queue_index.as_ref() {
            gpu_buffer_operations.dispatch(
                *queue_index,
                render_context,
                utils_pipeline,
                Some("hanabi:sort_fill_dispatch"),
            );
        }

        // Compute sort pass
        {
            let mut compute_pass =
                self.begin_compute_pass("hanabi:sort", pipeline_cache, render_context);

            let effect_metadata_buffer = effects_meta.effect_metadata_buffer.buffer().unwrap();
            let indirect_buffer = sort_bind_groups.indirect_buffer().unwrap();

            // Loop on batches and find those which need sorting
            for effect_batch in sorted_effect_batches.iter() {
                trace!("Processing effect batch for sorting...");
                if !effect_batch.layout_flags.contains(LayoutFlags::RIBBONS) {
                    continue;
                }
                assert!(effect_batch.particle_layout.contains(Attribute::RIBBON_ID));
                assert!(effect_batch.particle_layout.contains(Attribute::AGE)); // or is that optional?

                let Some(effect_buffer) = effect_cache.get_slab(&effect_batch.slab_id) else {
                    warn!("Missing sort-fill effect buffer.");
                    // render_context
                    //     .command_encoder()
                    //     .insert_debug_marker("ERROR:MissingEffectBatchBuffer");
                    continue;
                };

                let indirect_dispatch_index = *effect_batch
                    .sort_fill_indirect_dispatch_index
                    .as_ref()
                    .unwrap();
                let indirect_offset =
                    sort_bind_groups.get_indirect_dispatch_byte_offset(indirect_dispatch_index);

                // Fill the sort buffer with the key-value pairs to sort
                {
                    compute_pass.push_debug_group("hanabi:sort_fill");

                    // Fetch compute pipeline
                    let Some(pipeline_id) =
                        sort_bind_groups.get_sort_fill_pipeline_id(&effect_batch.particle_layout)
                    else {
                        warn!("Missing sort-fill pipeline.");
                        compute_pass.insert_debug_marker("ERROR:MissingSortFillPipeline");
                        continue;
                    };
                    if compute_pass
                        .set_cached_compute_pipeline(pipeline_id)
                        .is_err()
                    {
                        compute_pass.insert_debug_marker("ERROR:FailedToSetSortFillPipeline");
                        compute_pass.pop_debug_group();
                        // FIXME - Bevy doesn't allow returning custom errors here...
                        return Ok(());
                    }

                    let spawner_base = effect_batch.spawner_base;
                    let spawner_aligned_size = effects_meta.spawner_buffer.aligned_size();
                    assert!(spawner_aligned_size >= GpuSpawnerParams::min_size().get() as usize);
                    let spawner_offset = spawner_base * spawner_aligned_size as u32;

                    // Bind group sort_fill@0
                    let particle_buffer = effect_buffer.particle_buffer();
                    let indirect_index_buffer = effect_buffer.indirect_index_buffer();
                    let Some(bind_group) = sort_bind_groups.sort_fill_bind_group(
                        particle_buffer.id(),
                        indirect_index_buffer.id(),
                        effect_metadata_buffer.id(),
                    ) else {
                        warn!("Missing sort-fill bind group.");
                        compute_pass.insert_debug_marker("ERROR:MissingSortFillBindGroup");
                        continue;
                    };
                    let effect_metadata_offset = effects_meta
                        .gpu_limits
                        .effect_metadata_offset(effect_batch.metadata_table_id.0)
                        as u32;
                    compute_pass.set_bind_group(
                        0,
                        bind_group,
                        &[effect_metadata_offset, spawner_offset],
                    );

                    compute_pass
                        .dispatch_workgroups_indirect(indirect_buffer, indirect_offset as u64);
                    trace!("Dispatched sort-fill with indirect offset +{indirect_offset}");

                    compute_pass.pop_debug_group();
                }

                // Do the actual sort
                {
                    compute_pass.push_debug_group("hanabi:sort");

                    if compute_pass
                        .set_cached_compute_pipeline(sort_bind_groups.sort_pipeline_id())
                        .is_err()
                    {
                        compute_pass.insert_debug_marker("ERROR:FailedToSetSortPipeline");
                        compute_pass.pop_debug_group();
                        // FIXME - Bevy doesn't allow returning custom errors here...
                        return Ok(());
                    }

                    let Some(bind_group) = sort_bind_groups.sort_bind_group() else {
                        warn!("Missing sort bind group.");
                        compute_pass.insert_debug_marker("ERROR:MissingSortBindGroup");
                        continue;
                    };
                    compute_pass.set_bind_group(0, bind_group, &[]);
                    compute_pass
                        .dispatch_workgroups_indirect(indirect_buffer, indirect_offset as u64);
                    trace!("Dispatched sort with indirect offset +{indirect_offset}");

                    compute_pass.pop_debug_group();
                }

                // Copy the sorted particle indices back into the indirect index buffer, where
                // the render pass will read them.
                {
                    compute_pass.push_debug_group("hanabi:copy_sorted_indices");

                    // Fetch compute pipeline
                    let pipeline_id = sort_bind_groups.get_sort_copy_pipeline_id();
                    if compute_pass
                        .set_cached_compute_pipeline(pipeline_id)
                        .is_err()
                    {
                        compute_pass.insert_debug_marker("ERROR:FailedToSetSortCopyPipeline");
                        compute_pass.pop_debug_group();
                        // FIXME - Bevy doesn't allow returning custom errors here...
                        return Ok(());
                    }

                    let spawner_base = effect_batch.spawner_base;
                    let spawner_aligned_size = effects_meta.spawner_buffer.aligned_size();
                    assert!(spawner_aligned_size >= GpuSpawnerParams::min_size().get() as usize);
                    let spawner_offset = spawner_base * spawner_aligned_size as u32;

                    // Bind group sort_copy@0
                    let indirect_index_buffer = effect_buffer.indirect_index_buffer();
                    let Some(bind_group) = sort_bind_groups.sort_copy_bind_group(
                        indirect_index_buffer.id(),
                        effect_metadata_buffer.id(),
                    ) else {
                        warn!("Missing sort-copy bind group.");
                        compute_pass.insert_debug_marker("ERROR:MissingSortCopyBindGroup");
                        continue;
                    };
                    let effect_metadata_offset = effects_meta
                        .effect_metadata_buffer
                        .dynamic_offset(effect_batch.metadata_table_id);
                    compute_pass.set_bind_group(
                        0,
                        bind_group,
                        &[effect_metadata_offset, spawner_offset],
                    );

                    compute_pass
                        .dispatch_workgroups_indirect(indirect_buffer, indirect_offset as u64);
                    trace!("Dispatched sort-copy with indirect offset +{indirect_offset}");

                    compute_pass.pop_debug_group();
                }
            }
        }

        Ok(())
    }
}

impl From<LayoutFlags> for ParticleRenderAlphaMaskPipelineKey {
    fn from(layout_flags: LayoutFlags) -> Self {
        if layout_flags.contains(LayoutFlags::USE_ALPHA_MASK) {
            ParticleRenderAlphaMaskPipelineKey::AlphaMask
        } else if layout_flags.contains(LayoutFlags::OPAQUE) {
            ParticleRenderAlphaMaskPipelineKey::Opaque
        } else {
            ParticleRenderAlphaMaskPipelineKey::Blend
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_flags() {
        let flags = LayoutFlags::default();
        assert_eq!(flags, LayoutFlags::NONE);
    }

    #[cfg(feature = "gpu_tests")]
    #[test]
    fn gpu_limits() {
        use crate::test_utils::MockRenderer;

        let renderer = MockRenderer::new();
        let device = renderer.device();
        let limits = GpuLimits::from_device(&device);

        // assert!(limits.storage_buffer_align().get() >= 1);
        assert!(limits.effect_metadata_offset(256) >= 256 * GpuEffectMetadata::min_size().get());
    }

    #[cfg(feature = "gpu_tests")]
    #[test]
    fn gpu_ops_ifda() {
        use crate::test_utils::MockRenderer;

        let renderer = MockRenderer::new();
        let device = renderer.device();
        let render_queue = renderer.queue();

        let mut world = World::new();
        world.insert_resource(device.clone());
        let mut buffer_ops = GpuBufferOperations::from_world(&mut world);

        let src_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 256,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let dst_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 256,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Two consecutive ops can be merged. This includes having contiguous slices
        // both in source and destination.
        buffer_ops.begin_frame();
        {
            let mut q = InitFillDispatchQueue::default();
            q.enqueue(0, 0);
            assert_eq!(q.queue.len(), 1);
            q.enqueue(1, 1);
            // Ops are not batched yet
            assert_eq!(q.queue.len(), 2);
            // On submit, the ops get batched together
            q.submit(&src_buffer, &dst_buffer, &mut buffer_ops);
            assert_eq!(buffer_ops.args_buffer.len(), 1);
        }
        buffer_ops.end_frame(&device, &render_queue);

        // Even if out of order, the init fill dispatch ops are batchable. Here the
        // offsets are enqueued inverted.
        buffer_ops.begin_frame();
        {
            let mut q = InitFillDispatchQueue::default();
            q.enqueue(1, 1);
            assert_eq!(q.queue.len(), 1);
            q.enqueue(0, 0);
            // Ops are not batched yet
            assert_eq!(q.queue.len(), 2);
            // On submit, the ops get batched together
            q.submit(&src_buffer, &dst_buffer, &mut buffer_ops);
            assert_eq!(buffer_ops.args_buffer.len(), 1);
        }
        buffer_ops.end_frame(&device, &render_queue);

        // However, both the source and destination need to be contiguous at the same
        // time. Here they are mixed so we can't batch.
        buffer_ops.begin_frame();
        {
            let mut q = InitFillDispatchQueue::default();
            q.enqueue(0, 1);
            assert_eq!(q.queue.len(), 1);
            q.enqueue(1, 0);
            // Ops are not batched yet
            assert_eq!(q.queue.len(), 2);
            // On submit, the ops cannot get batched together
            q.submit(&src_buffer, &dst_buffer, &mut buffer_ops);
            assert_eq!(buffer_ops.args_buffer.len(), 2);
        }
        buffer_ops.end_frame(&device, &render_queue);
    }
}
