use std::marker::PhantomData;
use std::{
    borrow::Cow,
    hash::{DefaultHasher, Hash, Hasher},
    num::{NonZeroU32, NonZeroU64},
    ops::{Deref, DerefMut, Range},
    time::Duration,
};

#[cfg(feature = "2d")]
use bevy::core_pipeline::core_2d::{Transparent2d, CORE_2D_DEPTH_FORMAT};
#[cfg(feature = "2d")]
use bevy::math::FloatOrd;
#[cfg(feature = "3d")]
use bevy::{
    core_pipeline::{
        core_3d::{AlphaMask3d, Opaque3d, Transparent3d, CORE_3D_DEPTH_FORMAT},
        prepass::OpaqueNoLightmap3dBinKey,
    },
    render::render_phase::{BinnedPhaseItem, ViewBinnedRenderPhases},
};
use bevy::{
    ecs::{
        prelude::*,
        system::{lifetimeless::*, SystemParam, SystemState},
    },
    log::trace,
    prelude::*,
    render::{
        mesh::{
            allocator::MeshAllocator, MeshVertexBufferLayoutRef, RenderMesh, RenderMeshBufferInfo,
        },
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
        Extract,
    },
    utils::{Entry, HashMap, HashSet},
};
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use effect_cache::{BufferState, CachedEffect, EffectSlice};
use event::{CachedChildInfo, CachedEffectEvents, CachedParentInfo, CachedParentRef, GpuChildInfo};
use fixedbitset::FixedBitSet;
use naga_oil::compose::{Composer, NagaModuleDescriptor};

use crate::{
    asset::{DefaultMesh, EffectAsset},
    calc_func_id,
    plugin::WithCompiledParticleEffect,
    render::{
        batch::{BatchInput, EffectDrawBatch, InitAndUpdatePipelineIds},
        effect_cache::DispatchBufferIndices,
    },
    AlphaMode, Attribute, CompiledParticleEffect, EffectProperties, EffectShader, EffectSimulation,
    EffectSpawner, ParticleLayout, PropertyLayout, SimulationCondition, TextureLayout,
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
pub(crate) use event::EventCache;
pub(crate) use property::{
    on_remove_cached_properties, prepare_property_buffers, PropertyBindGroups, PropertyCache,
};
use property::{CachedEffectProperties, PropertyBindGroupKey};
pub use shader_cache::ShaderCache;
pub(crate) use sort::SortBindGroups;

use self::batch::EffectBatch;

// Size of an indirect index (including both parts of the ping-pong buffer) in
// bytes.
const INDIRECT_INDEX_SIZE: u32 = 12;

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
    pub fn binding(&self) -> BindingResource {
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
pub struct GpuDispatchIndirect {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Default for GpuDispatchIndirect {
    fn default() -> Self {
        Self { x: 0, y: 1, z: 1 }
    }
}

/// Stores metadata about each particle effect.
///
/// This is written by the CPU and read by the GPU.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct GpuEffectMetadata {
    /// The number of vertices in the mesh, if non-indexed; if indexed, the
    /// number of indices in the mesh.
    pub vertex_or_index_count: u32,
    /// The number of instances to render.
    pub instance_count: u32,
    /// The first index to render, if the mesh is indexed; the offset of the
    /// first vertex, if the mesh is non-indexed.
    pub first_index_or_vertex_offset: u32,
    /// The offset of the first vertex, if the mesh is indexed; the first
    /// instance to render, if the mesh is non-indexed.
    pub vertex_offset_or_base_instance: i32,
    /// The first instance to render, if indexed; unused if non-indexed.
    pub base_instance: u32,

    // Additional data not part of the required draw indirect args
    /// Number of alive particles.
    pub alive_count: u32,
    /// Cached value of `alive_count` to cap threads in update pass.
    pub max_update: u32,
    /// Number of dead particles.
    pub dead_count: u32,
    /// Cached value of `dead_count` to cap threads in init pass.
    pub max_spawn: u32,
    /// Index of the ping buffer for particle indices. Init and update compute
    /// passes always write into the ping buffer and read from the pong buffer.
    /// The buffers are swapped (ping = 1 - ping) during the indirect dispatch.
    pub ping: u32,
    /// Unused. TODO remove.
    pub spawner_index: u32,
    /// Index of the [`GpuDispatchIndirect`] struct inside the global
    /// [`EffectsMeta::dispatch_indirect_buffer`].
    pub indirect_dispatch_index: u32,
    /// Index of the [`GpuRenderIndirect`] struct inside the global
    /// [`EffectsMeta::render_group_dispatch_buffer`].
    pub indirect_render_index: u32,
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

    /// Atomic counter incremented each time a particle spawns. Useful for
    /// things like RIBBON_ID or any other use where a unique value is needed.
    /// The value loops back after some time, but unless some particle lives
    /// forever there's little chance of repetition.
    pub particle_counter: u32,
}

/// Compute pipeline to run the `vfx_indirect` dispatch workgroup calculation
/// shader.
#[derive(Resource)]
pub(crate) struct DispatchIndirectPipeline {
    /// Layout of bind group sim_params@0.
    sim_params_bind_group_layout: BindGroupLayout,
    /// Layout of bind group effect_metadata@1.
    effect_metadata_bind_group_layout: BindGroupLayout,
    /// Layout of bind group spawner@2.
    spawner_bind_group_layout: BindGroupLayout,
    /// Layout of bind group child_infos@3.
    child_infos_bind_group_layout: BindGroupLayout,
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
        let render_effect_metadata_size = GpuEffectMetadata::aligned_size(storage_alignment);
        let spawner_min_binding_size = GpuSpawnerParams::aligned_size(storage_alignment);

        // @group(0) @binding(0) var<uniform> sim_params : SimParams;
        trace!("GpuSimParams: min_size={}", GpuSimParams::min_size());
        let sim_params_bind_group_layout = render_device.create_bind_group_layout(
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
            render_effect_metadata_size,
        );
        let effect_metadata_bind_group_layout = render_device.create_bind_group_layout(
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
                        min_binding_size: Some(render_effect_metadata_size),
                    },
                    count: None,
                },
                // @group(0) @binding(2) var<storage, read_write> dispatch_indirect_buffer :
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
            ],
        );

        // @group(2) @binding(0) var<storage, read_write> spawner_buffer :
        // array<Spawner>;
        let spawner_bind_group_layout = render_device.create_bind_group_layout(
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
        let child_infos_bind_group_layout = render_device.create_bind_group_layout(
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
            sim_params_bind_group_layout,
            effect_metadata_bind_group_layout,
            spawner_bind_group_layout,
            child_infos_bind_group_layout,
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
        layout.push(self.sim_params_bind_group_layout.clone());
        layout.push(self.effect_metadata_bind_group_layout.clone());
        layout.push(self.spawner_bind_group_layout.clone());
        if key.has_events {
            layout.push(self.child_infos_bind_group_layout.clone());
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
            entry_point: "main".into(),
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
    /// Same as [`FillDispatchArgs`], but with a specialization for the indirect
    /// init pass, where we read the destination offset from the source buffer.
    InitFillDispatchArgs,
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

#[derive(Debug, Clone)]
struct InitFillDispatchArgs {
    args_index: u32,
    event_buffer_index: u32,
    event_slice: std::ops::Range<u32>,
}

/// Queue of GPU buffer operations for this frame.
#[derive(Resource)]
pub(super) struct GpuBufferOperationQueue {
    /// Arguments for the buffer operations submitted this frame.
    args_buffer: AlignedBufferVec<GpuBufferOperationArgs>,

    /// Unsorted temporary storage for this-frame operations, which will be
    /// written to [`args_buffer`] at the end of the frame after being sorted.
    args_buffer_unsorted: Vec<GpuBufferOperationArgs>,

    /// Queued operations.
    operation_queue: Vec<QueuedOperation>,

    /// Queued INIT_FILL_DISPATCH operations.
    init_fill_dispatch_args: Vec<InitFillDispatchArgs>,

    /// Bind groups for the queued operations.
    bind_groups: HashMap<QueuedOperationBindGroupKey, BindGroup>,
}

impl FromWorld for GpuBufferOperationQueue {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let align = render_device.limits().min_uniform_buffer_offset_alignment;
        Self::new(align)
    }
}

impl GpuBufferOperationQueue {
    pub fn new(align: u32) -> Self {
        let args_buffer = AlignedBufferVec::new(
            BufferUsages::UNIFORM,
            Some(NonZeroU64::new(align as u64).unwrap()),
            Some("hanabi:buffer:gpu_operation_args".to_string()),
        );
        Self {
            args_buffer,
            args_buffer_unsorted: vec![],
            operation_queue: vec![],
            init_fill_dispatch_args: vec![],
            bind_groups: default(),
        }
    }

    /// Get a binding for all the entries of the arguments buffer associated
    /// with the given event buffer.
    pub fn init_args_buffer_binding(
        &self,
        event_buffer_index: u32,
    ) -> Option<(BindingResource, u32)> {
        // Find the slice corresponding to this event buffer. The entries are sorted by
        // event buffer index, so the list of entries is a contiguous slice inside the
        // overall buffer.
        let Some(start) = self
            .init_fill_dispatch_args
            .iter()
            .position(|ifda| ifda.event_buffer_index == event_buffer_index)
        else {
            trace!("Event buffer #{event_buffer_index} has no allocated operation.");
            return None;
        };
        let end = if let Some(end) = self
            .init_fill_dispatch_args
            .iter()
            .skip(start)
            .position(|ifda| ifda.event_buffer_index != event_buffer_index)
        {
            end
        } else {
            self.init_fill_dispatch_args.len()
        };
        assert!(start < end);
        let count = (end - start) as u32;
        trace!("Event buffer #{event_buffer_index} has {count} allocated operation(s).");

        self.args_buffer
            .lead_binding(count)
            .map(|binding| (binding, count))
    }

    /// Clear the queue and begin recording operations for a new frame.
    pub fn begin_frame(&mut self) {
        self.args_buffer.clear();
        self.args_buffer_unsorted.clear();
        self.operation_queue.clear();
        self.bind_groups.clear(); // for now; might consider caching frame-to-frame
        self.init_fill_dispatch_args.clear();
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
        assert_ne!(
            op,
            GpuBufferOperationType::InitFillDispatchArgs,
            "FIXME - InitFillDispatchArgs needs enqueue_init_fill() instead"
        );
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
        let args_index = self.args_buffer_unsorted.len() as u32;
        self.args_buffer_unsorted.push(args);
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

    /// Queue a new [`GpuBufferOperationType::InitFillDispatchArgs`] operation.
    pub fn enqueue_init_fill(
        &mut self,
        event_buffer_index: u32,
        event_slice: std::ops::Range<u32>,
        args: GpuBufferOperationArgs,
    ) {
        trace!(
            "Queue InitFillDispatchArgs op: ev_buffer#{} ev_slice={:?} args={:?}",
            event_buffer_index,
            event_slice,
            args
        );
        let args_index = self.args_buffer_unsorted.len() as u32;
        self.args_buffer_unsorted.push(args);
        self.init_fill_dispatch_args.push(InitFillDispatchArgs {
            event_buffer_index,
            args_index,
            event_slice,
        });
    }

    /// Finish recording operations for this frame, and schedule buffer writes
    /// to GPU.
    pub fn end_frame(&mut self, device: &RenderDevice, render_queue: &RenderQueue) {
        assert_eq!(
            self.args_buffer_unsorted.len(),
            self.operation_queue.len() + self.init_fill_dispatch_args.len()
        );
        assert!(self.args_buffer.is_empty());

        if self.operation_queue.is_empty() && self.init_fill_dispatch_args.is_empty() {
            self.args_buffer.set_content(vec![]);
        } else {
            let mut sorted_args =
                Vec::with_capacity(self.init_fill_dispatch_args.len() + self.operation_queue.len());

            // Sort the commands by buffer, so we can dispatch them in groups with a single
            // dispatch per buffer
            trace!(
                "Sorting {} InitFillDispatch ops...",
                self.init_fill_dispatch_args.len()
            );
            self.init_fill_dispatch_args
                .sort_unstable_by(|ifda1, ifda2| {
                    if ifda1.event_buffer_index != ifda2.event_buffer_index {
                        ifda1.event_buffer_index.cmp(&ifda2.event_buffer_index)
                    } else if ifda1.event_slice != ifda2.event_slice {
                        ifda1.event_slice.start.cmp(&ifda2.event_slice.start)
                    } else {
                        // Sort by source offset, which at this point contains the child_index
                        let arg1 = &self.args_buffer_unsorted[ifda1.args_index as usize];
                        let arg2 = &self.args_buffer_unsorted[ifda2.args_index as usize];
                        arg1.src_offset.cmp(&arg2.src_offset)
                    }
                });

            // Note: Do NOT sort queued operations; they migth depend on each other. It's
            // the caller's responsibility to ensure e.g. multiple copies can be batched
            // together.

            // Push entries into the final storage before GPU upload. It's a bit unfortunate
            // we have to make copies, but those arrays should be small.
            {
                let mut sorted_ifda = Vec::with_capacity(self.init_fill_dispatch_args.len());
                let mut prev_buffer = u32::MAX;
                for ifda in &self.init_fill_dispatch_args {
                    trace!("+ op: ifda={:?}", ifda);
                    if !sorted_args.is_empty() && (prev_buffer == ifda.event_buffer_index) {
                        let prev_idx = sorted_args.len() - 1;
                        let prev: &mut GpuBufferOperationArgs = &mut sorted_args[prev_idx];
                        let cur = &self.args_buffer_unsorted[ifda.args_index as usize];
                        if prev.src_stride == cur.src_stride
                    // at this point src_offset == child_index, and we want them to be contiguous in the source buffer so that we can increment by src_stride
                    && cur.src_offset == prev.src_offset + 1
                    && cur.dst_offset == prev.dst_offset + 1
                        {
                            prev.count += 1;
                            trace!("-> merged op with previous one {:?}", prev);
                            continue;
                        }
                    }
                    prev_buffer = ifda.event_buffer_index;
                    let sorted_args_index = sorted_args.len() as u32;
                    sorted_ifda.push(InitFillDispatchArgs {
                        event_buffer_index: ifda.event_buffer_index,
                        event_slice: ifda.event_slice.clone(),
                        args_index: sorted_args_index,
                    });
                    sorted_args.push(self.args_buffer_unsorted[ifda.args_index as usize]);
                }
                trace!("Final ops (sorted IFDAs): {:?}", sorted_ifda);
                self.init_fill_dispatch_args = sorted_ifda;
            }

            // Just copy this, we want to preserve order
            {
                for qop in &self.operation_queue {
                    let args_index = qop.args_index as usize;
                    // ensure the index returned by enqueue() is still valid for COPY ops
                    // FIXME - all this stuff is too brittle...
                    assert_eq!(args_index, sorted_args.len());
                    sorted_args.push(self.args_buffer_unsorted[args_index]);
                }
            }

            // Write CPU content for all arguments
            self.args_buffer.set_content(sorted_args);
        }

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
            "Creating bind groups for {} queued operations...",
            self.operation_queue.len()
        );
        for qop in &self.operation_queue {
            let key: QueuedOperationBindGroupKey = qop.into();
            self.bind_groups.entry(key).or_insert_with(|| {
                let src_id: NonZeroU32 = qop.src_buffer.id().into();
                let dst_id: NonZeroU32 = qop.dst_buffer.id().into();
                let label = format!("hanabi:bind_group:util_{}_{}", src_id.get(), dst_id.get());
                let bind_group_layout = match qop.op {
                    GpuBufferOperationType::FillDispatchArgs => {
                        utils_pipeline.bind_group_layout(qop.op, true)
                    }
                    _ => utils_pipeline.bind_group_layout(qop.op, false),
                };
                trace!(
                    "-> Creating new bind group '{}': src#{} ({:?}B) dst#{} ({:?}B)",
                    label,
                    src_id,
                    qop.src_binding_size,
                    dst_id,
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
                                offset: 0,
                                size: qop.src_binding_size.map(Into::into),
                            }),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: &qop.dst_buffer,
                                offset: 0,
                                size: qop.dst_binding_size.map(Into::into),
                            }),
                        },
                    ],
                )
            });
        }
    }

    /// Dispatch any pending [`GpuBufferOperationType::FillDispatchArgs`]
    /// operation.
    pub fn dispatch_fill(&self, render_context: &mut RenderContext, pipeline: &ComputePipeline) {
        trace!(
            "Recording GPU commands for fill dispatch operations using the {:?} pipeline...",
            pipeline
        );

        if self.operation_queue.is_empty() {
            return;
        }

        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("hanabi:fill_dispatch"),
                    timestamp_writes: None,
                });

        compute_pass.set_pipeline(pipeline);

        for qop in &self.operation_queue {
            trace!("qop={:?}", qop);
            if qop.op != GpuBufferOperationType::FillDispatchArgs {
                continue;
            }

            let key: QueuedOperationBindGroupKey = qop.into();
            if let Some(bind_group) = self.bind_groups.get(&key) {
                let args_offset = self.args_buffer.dynamic_offset(qop.args_index as usize);
                let src_offset = qop.src_binding_offset;
                let dst_offset = qop.dst_binding_offset;
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

    /// Dispatch any pending [`GpuBufferOperationType::InitFillDispatchArgs`]
    /// operation for indirect init passes.
    pub fn dispatch_init_fill(
        &self,
        render_context: &mut RenderContext,
        pipeline: &ComputePipeline,
        bind_groups: &EffectBindGroups,
    ) {
        if self.init_fill_dispatch_args.is_empty() {
            return;
        }

        trace!(
            "Recording GPU commands for the init fill dispatch pipeline... {:?}",
            pipeline
        );

        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("hanabi:init_fill_dispatch"),
                    timestamp_writes: None,
                });

        compute_pass.set_pipeline(pipeline);

        assert_eq!(
            self.init_fill_dispatch_args.len() + self.operation_queue.len(),
            self.args_buffer.content().len()
        );

        for (args_index, event_buffer_index) in self
            .init_fill_dispatch_args
            .iter()
            .enumerate()
            .map(|(args_index, ifda)| (args_index as u32, ifda.event_buffer_index))
        {
            trace!(
                "event_buffer_index={} args_index={:?}",
                event_buffer_index,
                args_index
            );
            if let Some(bind_group) = bind_groups.init_fill_dispatch(event_buffer_index) {
                let dst_offset = self.args_buffer.dynamic_offset(args_index as usize);
                compute_pass.set_bind_group(0, bind_group, &[]);
                trace!(
                    "found bind group for event buffer index #{} with dst_offset +{}B",
                    event_buffer_index,
                    dst_offset
                );
            } else {
                warn!(
                    "bind group not found for event buffer index #{}",
                    event_buffer_index
                );
                continue;
            }

            // Dispatch the operations for this buffer
            const WORKGROUP_SIZE: u32 = 64;
            let num_ops = 1u32;
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
    pipelines: [ComputePipeline; 5],
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
        let shader_module = render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("hanabi:shader:utils"),
            source: shader_source,
        });

        trace!("Create vfx_utils pipelines...");
        let dummy = std::collections::HashMap::<String, f64>::new();
        let zero_pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("hanabi:compute_pipeline:zero_buffer"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("zero_buffer"),
            compilation_options: PipelineCompilationOptions {
                constants: &dummy,
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
                constants: &dummy,
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
                    constants: &dummy,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });
        let init_fill_dispatch_args_pipeline =
            render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some("hanabi:compute_pipeline:init_fill_dispatch_args"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("init_fill_dispatch_args"),
                compilation_options: PipelineCompilationOptions {
                    constants: &dummy,
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
                    constants: &dummy,
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
                init_fill_dispatch_args_pipeline,
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
            GpuBufferOperationType::InitFillDispatchArgs => &self.pipelines[3],
            GpuBufferOperationType::FillDispatchArgsSelf => &self.pipelines[4],
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
    sim_params_layout: BindGroupLayout,

    // Temporary values passed to specialize()
    // https://github.com/bevyengine/bevy/issues/17132
    /// Layout of the particle@1 bind group this pipeline was specialized with.
    temp_particle_bind_group_layout: Option<BindGroupLayout>,
    /// Layout of the spawner@2 bind group this pipeline was specialized with.
    temp_spawner_bind_group_layout: Option<BindGroupLayout>,
    /// Layout of the metadata@3 bind group this pipeline was specialized with.
    temp_metadata_bind_group_layout: Option<BindGroupLayout>,
}

impl FromWorld for ParticlesInitPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let sim_params_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:update_sim_params",
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
            sim_params_layout,
            temp_particle_bind_group_layout: None,
            temp_spawner_bind_group_layout: None,
            temp_metadata_bind_group_layout: None,
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
    // Note: can't directly store BindGroupLayout because it's not Eq nor Hash
    particle_bind_group_layout_id: BindGroupLayoutId,
    /// Layout of the spawner@2 bind group this pipeline was specialized with.
    // Note: can't directly store BindGroupLayout because it's not Eq nor Hash
    spawner_bind_group_layout_id: BindGroupLayoutId,
    /// Layout of the metadata@3 bind group this pipeline was specialized with.
    // Note: can't directly store BindGroupLayout because it's not Eq nor Hash
    metadata_bind_group_layout_id: BindGroupLayoutId,
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

        // This should always be valid when specialize() is called, by design. This is
        // how we pass the value to specialize() to work around the lack of access to
        // external data.
        // https://github.com/bevyengine/bevy/issues/17132
        let particle_bind_group_layout = self.temp_particle_bind_group_layout.as_ref().unwrap();
        assert_eq!(
            particle_bind_group_layout.id(),
            key.particle_bind_group_layout_id
        );
        let spawner_bind_group_layout = self.temp_spawner_bind_group_layout.as_ref().unwrap();
        assert_eq!(
            spawner_bind_group_layout.id(),
            key.spawner_bind_group_layout_id
        );
        let metadata_bind_group_layout = self.temp_metadata_bind_group_layout.as_ref().unwrap();
        assert_eq!(
            metadata_bind_group_layout.id(),
            key.metadata_bind_group_layout_id
        );

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
                self.sim_params_layout.clone(),
                particle_bind_group_layout.clone(),
                spawner_bind_group_layout.clone(),
                metadata_bind_group_layout.clone(),
            ],
            shader: key.shader,
            shader_defs,
            entry_point: "main".into(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        }
    }
}

#[derive(Resource)]
pub(crate) struct ParticlesUpdatePipeline {
    sim_params_layout: BindGroupLayout,

    // Temporary values passed to specialize()
    // https://github.com/bevyengine/bevy/issues/17132
    /// Layout of the particle@1 bind group this pipeline was specialized with.
    temp_particle_bind_group_layout: Option<BindGroupLayout>,
    /// Layout of the spawner@2 bind group this pipeline was specialized with.
    temp_spawner_bind_group_layout: Option<BindGroupLayout>,
    /// Layout of the metadata@3 bind group this pipeline was specialized with.
    temp_metadata_bind_group_layout: Option<BindGroupLayout>,
}

impl FromWorld for ParticlesUpdatePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        trace!("GpuSimParams: min_size={}", GpuSimParams::min_size());
        let sim_params_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:update:particle",
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
            sim_params_layout,
            temp_particle_bind_group_layout: None,
            temp_spawner_bind_group_layout: None,
            temp_metadata_bind_group_layout: None,
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
    // Note: can't directly store BindGroupLayout because it's not Eq nor Hash
    particle_bind_group_layout_id: BindGroupLayoutId,
    /// Layout of the spawner@2 bind group this pipeline was specialized with.
    // Note: can't directly store BindGroupLayout because it's not Eq nor Hash
    spawner_bind_group_layout_id: BindGroupLayoutId,
    /// Layout of the metadata@3 bind group this pipeline was specialized with.
    // Note: can't directly store BindGroupLayout because it's not Eq nor Hash
    metadata_bind_group_layout_id: BindGroupLayoutId,
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
        shader_defs.push("CHILD_INFO_IS_ATOMIC".into());
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

        // This should always be valid when specialize() is called, by design. This is
        // how we pass the value to specialize() to work around the lack of access to
        // external data.
        // https://github.com/bevyengine/bevy/issues/17132
        let particle_bind_group_layout = self.temp_particle_bind_group_layout.as_ref().unwrap();
        assert_eq!(
            particle_bind_group_layout.id(),
            key.particle_bind_group_layout_id
        );
        let spawner_bind_group_layout = self.temp_spawner_bind_group_layout.as_ref().unwrap();
        assert_eq!(
            spawner_bind_group_layout.id(),
            key.spawner_bind_group_layout_id
        );
        let metadata_bind_group_layout = self.temp_metadata_bind_group_layout.as_ref().unwrap();
        assert_eq!(
            metadata_bind_group_layout.id(),
            key.metadata_bind_group_layout_id
        );

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
                self.sim_params_layout.clone(),
                particle_bind_group_layout.clone(),
                spawner_bind_group_layout.clone(),
                metadata_bind_group_layout.clone(),
            ],
            shader: key.shader,
            shader_defs,
            entry_point: "main".into(),
            push_constant_ranges: Vec::new(),
            zero_initialize_workgroup_memory: false,
        }
    }
}

#[derive(Resource)]
pub(crate) struct ParticlesRenderPipeline {
    render_device: RenderDevice,
    view_layout: BindGroupLayout,
    material_layouts: HashMap<TextureLayout, BindGroupLayout>,
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
        if self.material_layouts.contains_key(layout) {
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
        let material_bind_group_layout = self
            .render_device
            .create_bind_group_layout("hanabi:material_layout_render", &entries[..]);

        self.material_layouts
            .insert(layout.clone(), material_bind_group_layout);
    }

    /// Retrieve a bind group layout for a cached material.
    pub fn get_material(&self, layout: &TextureLayout) -> Option<&BindGroupLayout> {
        // Prevent a hash and lookup for the trivial case of an empty layout
        if layout.layout.is_empty() {
            return None;
        }

        self.material_layouts.get(layout)
    }
}

impl FromWorld for ParticlesRenderPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let view_layout = render_device.create_bind_group_layout(
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
            view_layout,
            material_layouts: default(),
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
                visibility: ShaderStages::VERTEX,
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
        let particle_bind_group_layout = self
            .render_device
            .create_bind_group_layout("hanabi:bind_group_layout:render:particle@1", &entries[..]);

        let mut layout = vec![self.view_layout.clone(), particle_bind_group_layout];
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
                entry_point: "vertex".into(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_buffer_layout.expect("Vertex buffer layout not present")],
            },
            fragment: Some(FragmentState {
                shader: key.shader,
                shader_defs,
                entry_point: "fragment".into(),
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
#[derive(Debug)]
pub(crate) struct ExtractedEffect {
    /// Main world entity owning the [`CompiledParticleEffect`] this effect was
    /// extracted from. Mainly used for visibility.
    pub main_entity: MainEntity,
    /// Render world entity, if any, where the [`CachedEffect`] component
    /// caching this extracted effect resides. If this component was never
    /// cached in the render world, this is `None`. In that case a new
    /// [`CachedEffect`] will be spawned automatically.
    pub render_entity: RenderEntity,
    /// Handle to the effect asset this instance is based on.
    /// The handle is weak to prevent refcount cycles and gracefully handle
    /// assets unloaded or destroyed after a draw call has been submitted.
    pub handle: Handle<EffectAsset>,
    /// Particle layout for the effect.
    #[allow(dead_code)]
    pub particle_layout: ParticleLayout,
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
    /// Number of particles to spawn this frame.
    ///
    /// This is ignored if the effect is a child effect consuming GPU spawn
    /// events.
    pub spawn_count: u32,
    /// PRNG seed.
    pub prng_seed: u32,
    /// Global transform of the effect origin.
    pub transform: GlobalTransform,
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
}

pub struct AddedEffectParent {
    pub entity: MainEntity,
    pub layout: ParticleLayout,
    /// GPU spawn event count to allocate for this effect.
    pub event_count: u32,
}

/// Extracted data for newly-added [`ParticleEffect`] component requiring a new
/// GPU allocation.
///
/// [`ParticleEffect`]: crate::ParticleEffect
pub struct AddedEffect {
    /// Entity with a newly-added [`ParticleEffect`] component.
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    pub entity: MainEntity,
    #[allow(dead_code)]
    pub render_entity: RenderEntity,
    /// Capacity, in number of particles, of the effect.
    pub capacity: u32,
    /// Resolved particle mesh, either the one provided by the user or the
    /// default one. This should always be valid.
    pub mesh: Handle<Mesh>,
    /// Parent effect, if any.
    pub parent: Option<AddedEffectParent>,
    /// Layout of particle attributes.
    pub particle_layout: ParticleLayout,
    /// Layout of properties for the effect, if properties are used at all, or
    /// an empty layout.
    pub property_layout: PropertyLayout,
    /// Effect flags.
    pub layout_flags: LayoutFlags,
    /// Handle of the effect asset.
    pub handle: Handle<EffectAsset>,
}

/// Collection of all extracted effects for this frame, inserted into the
/// render world as a render resource.
#[derive(Default, Resource)]
pub(crate) struct ExtractedEffects {
    /// Extracted effects this frame.
    pub effects: Vec<ExtractedEffect>,
    /// Newly added effects without a GPU allocation yet.
    pub added_effects: Vec<AddedEffect>,
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
    mut image_events: Extract<EventReader<AssetEvent<Image>>>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::utils::tracing::info_span!("extract_effect_events").entered();
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
    /// the [`extract_effects`] system runs next. This instructs any attached
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

/// System extracting data for rendering of all active [`ParticleEffect`]
/// components.
///
/// Extract rendering data for all [`ParticleEffect`] components in the world
/// which are visible ([`ComputedVisibility::is_visible`] is `true`), and wrap
/// the data into a new [`ExtractedEffect`] instance added to the
/// [`ExtractedEffects`] resource.
///
/// This system runs in parallel of [`extract_effect_events`].
///
/// If any GPU debug capture is configured to start or stop in
/// [`DebugSettings`], they do so at the beginning of this system. This ensures
/// that all GPU commands produced by Hanabi are recorded (but may miss some
/// from Bevy itself, if another Bevy system runs before this one).
///
/// [`ParticleEffect`]: crate::ParticleEffect
pub(crate) fn extract_effects(
    real_time: Extract<Res<Time<Real>>>,
    virtual_time: Extract<Res<Time<Virtual>>>,
    time: Extract<Res<Time<EffectSimulation>>>,
    effects: Extract<Res<Assets<EffectAsset>>>,
    q_added_effects: Extract<
        Query<
            (Entity, &RenderEntity, &CompiledParticleEffect),
            (Added<CompiledParticleEffect>, With<GlobalTransform>),
        >,
    >,
    q_effects: Extract<
        Query<(
            Entity,
            &RenderEntity,
            Option<&InheritedVisibility>,
            Option<&ViewVisibility>,
            &EffectSpawner,
            &CompiledParticleEffect,
            Option<Ref<EffectProperties>>,
            &GlobalTransform,
        )>,
    >,
    render_device: Res<RenderDevice>,
    debug_settings: Extract<Res<DebugSettings>>,
    default_mesh: Extract<Res<DefaultMesh>>,
    mut sim_params: ResMut<SimParams>,
    mut extracted_effects: ResMut<ExtractedEffects>,
    mut render_debug_settings: ResMut<RenderDebugSettings>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::utils::tracing::info_span!("extract_effects").entered();
    trace!("extract_effects()");

    // Manage GPU debug capture
    if render_debug_settings.is_capturing {
        render_debug_settings.captured_frames += 1;

        // Stop any pending capture if needed
        if render_debug_settings.captured_frames >= debug_settings.capture_frame_count {
            render_device.wgpu_device().stop_capture();
            render_debug_settings.is_capturing = false;
            warn!(
                "Stopped GPU debug capture after {} frames, at t={}s.",
                render_debug_settings.captured_frames,
                real_time.elapsed().as_secs_f64()
            );
        }
    }
    if !render_debug_settings.is_capturing {
        // If no pending capture, consider starting a new one
        if debug_settings.start_capture_this_frame
            || (debug_settings.start_capture_on_new_effect && !q_added_effects.is_empty())
        {
            render_device.wgpu_device().start_capture();
            render_debug_settings.is_capturing = true;
            render_debug_settings.capture_start = real_time.elapsed();
            render_debug_settings.captured_frames = 0;
            warn!(
                "Started GPU debug capture at t={}s.",
                render_debug_settings.capture_start.as_secs_f64()
            );
        }
    }

    // Save simulation params into render world
    sim_params.time = time.elapsed_secs_f64();
    sim_params.delta_time = time.delta_secs();
    sim_params.virtual_time = virtual_time.elapsed_secs_f64();
    sim_params.virtual_delta_time = virtual_time.delta_secs();
    sim_params.real_time = real_time.elapsed_secs_f64();
    sim_params.real_delta_time = real_time.delta_secs();

    // Collect added effects for later GPU data allocation
    extracted_effects.added_effects = q_added_effects
        .iter()
        .filter_map(|(entity, render_entity, compiled_effect)| {
            let handle = compiled_effect.asset.clone_weak();
            let asset = effects.get(&compiled_effect.asset)?;
            let particle_layout = asset.particle_layout();
            assert!(
                particle_layout.size() > 0,
                "Invalid empty particle layout for effect '{}' on entity {:?} (render entity {:?}). Did you forget to add some modifier to the asset?",
                asset.name,
                entity,
                render_entity.id(),
            );
            let property_layout = asset.property_layout();
            let mesh = compiled_effect
                .mesh
                .clone()
                .unwrap_or(default_mesh.0.clone());

            trace!(
                "Found new effect: entity {:?} | render entity {:?} | capacity {:?} | particle_layout {:?} | \
                 property_layout {:?} | layout_flags {:?} | mesh {:?}",
                 entity,
                 render_entity.id(),
                 asset.capacity(),
                 particle_layout,
                 property_layout,
                 compiled_effect.layout_flags,
                 mesh);

            // FIXME - fixed 256 events per child (per frame) for now... this neatly avoids any issue with alignment 32/256 byte storage buffer align for bind groups
            const FIXME_HARD_CODED_EVENT_COUNT: u32 = 256;
            let parent = compiled_effect.parent.map(|entity| AddedEffectParent {
                entity: entity.into(),
                layout: compiled_effect.parent_particle_layout.as_ref().unwrap().clone(),
                event_count: FIXME_HARD_CODED_EVENT_COUNT,
            });

            trace!("Found new effect: entity {:?} | capacity {:?} | particle_layout {:?} | property_layout {:?} | layout_flags {:?}", entity, asset.capacity(), particle_layout, property_layout, compiled_effect.layout_flags);
            Some(AddedEffect {
                entity: MainEntity::from(entity),
                render_entity: *render_entity,
                capacity: asset.capacity(),
                mesh,
                parent,
                particle_layout,
                property_layout,
                layout_flags: compiled_effect.layout_flags,
                handle,
            })
        })
        .collect();

    // Loop over all existing effects to extract them
    extracted_effects.effects.clear();
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
            continue;
        };

        // Check if hidden, unless always simulated
        if compiled_effect.simulation_condition == SimulationCondition::WhenVisible
            && !maybe_inherited_visibility
                .map(|cv| cv.get())
                .unwrap_or(true)
            && !maybe_view_visibility.map(|cv| cv.get()).unwrap_or(true)
        {
            continue;
        }

        // Check if asset is available, otherwise silently ignore
        let Some(asset) = effects.get(&compiled_effect.asset) else {
            trace!(
                "EffectAsset not ready; skipping ParticleEffect instance on entity {:?}.",
                main_entity
            );
            continue;
        };

        // Resolve the render entity of the parent, if any
        let _parent = if let Some(main_entity) = compiled_effect.parent {
            let Ok((_, render_entity, _, _, _, _, _, _)) = q_effects.get(main_entity) else {
                error!(
                    "Failed to resolve render entity of parent with main entity {:?}.",
                    main_entity
                );
                continue;
            };
            Some(*render_entity)
        } else {
            None
        };

        let property_layout = asset.property_layout();
        let property_data = if let Some(properties) = maybe_properties {
            // Note: must check that property layout is not empty, because the
            // EffectProperties component is marked as changed when added but contains an
            // empty Vec if there's no property, which would later raise an error if we
            // don't return None here.
            if properties.is_changed() && !property_layout.is_empty() {
                trace!("Detected property change, re-serializing...");
                Some(properties.serialize(&property_layout))
            } else {
                None
            }
        } else {
            None
        };

        let texture_layout = asset.module().texture_layout();
        let layout_flags = compiled_effect.layout_flags;
        // let mesh = compiled_effect
        //     .mesh
        //     .clone()
        //     .unwrap_or(default_mesh.0.clone());
        let alpha_mode = compiled_effect.alpha_mode;

        trace!(
            "Extracted instance of effect '{}' on entity {:?} (render entity {:?}): texture_layout_count={} texture_count={} layout_flags={:?}",
            asset.name,
            main_entity,
            render_entity.id(),
            texture_layout.layout.len(),
            compiled_effect.textures.len(),
            layout_flags,
        );

        extracted_effects.effects.push(ExtractedEffect {
            render_entity: *render_entity,
            main_entity: main_entity.into(),
            handle: compiled_effect.asset.clone_weak(),
            particle_layout: asset.particle_layout().clone(),
            property_layout,
            property_data,
            spawn_count: effect_spawner.spawn_count,
            prng_seed: compiled_effect.prng_seed,
            transform: *transform,
            layout_flags,
            texture_layout,
            textures: compiled_effect.textures.clone(),
            alpha_mode,
            effect_shaders: effect_shaders.clone(),
        });
    }
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

    /// Byte alignment for [`GpuEffectMetadata`].
    pub fn effect_metadata_size(&self) -> NonZeroU64 {
        NonZeroU64::new(self.effect_metadata_aligned_size.get() as u64).unwrap()
    }
}

/// Global resource containing the GPU data to draw all the particle effects in
/// all views.
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
    /// Bind group #0 of the vfx_indirect shader, for the simulation parameters
    /// like the current time and frame delta time.
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
    update_dispatch_indirect_buffer: BufferTable<GpuDispatchIndirect>,
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
        let item_align = gpu_limits.storage_buffer_align().get() as u64;
        trace!(
            "Aligning storage buffers to {} bytes as device limits requires.",
            item_align
        );

        Self {
            view_bind_group: None,
            indirect_sim_params_bind_group: None,
            indirect_metadata_bind_group: None,
            indirect_spawner_bind_group: None,
            sim_params_uniforms: UniformBuffer::default(),
            spawner_buffer: AlignedBufferVec::new(
                BufferUsages::STORAGE,
                NonZeroU64::new(item_align),
                Some("hanabi:buffer:spawner".to_string()),
            ),
            update_dispatch_indirect_buffer: BufferTable::new(
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
                // Indirect dispatch args don't need to be aligned
                None,
                Some("hanabi:buffer:update_dispatch_indirect".to_string()),
            ),
            effect_metadata_buffer: BufferTable::new(
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
                NonZeroU64::new(item_align),
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

    /// Allocate internal resources for newly spawned effects.
    ///
    /// After this system ran, all valid extracted effects from the main world
    /// have a corresponding entity with a [`CachedEffect`] component in the
    /// render world. An extracted effect is considered valid if it passed some
    /// basic checks, like having a valid mesh. Note however that the main
    /// world's entity might still be missing its [`RenderEntity`]
    /// reference, since we cannot yet write into the main world.
    pub fn add_effects(
        &mut self,
        mut commands: Commands,
        mut added_effects: Vec<AddedEffect>,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        mesh_allocator: &MeshAllocator,
        render_meshes: &RenderAssets<RenderMesh>,
        effect_bind_groups: &mut ResMut<EffectBindGroups>,
        effect_cache: &mut ResMut<EffectCache>,
        property_cache: &mut ResMut<PropertyCache>,
        event_cache: &mut ResMut<EventCache>,
    ) {
        // FIXME - We delete a buffer above, and have a chance to immediatly re-create
        // it below. We should keep the GPU buffer around until the end of this method.
        // On the other hand, we should also be careful that allocated buffers need to
        // be tightly packed because 'vfx_indirect.wgsl' index them by buffer index in
        // order, so doesn't support offset.

        trace!("Adding {} newly spawned effects", added_effects.len());
        for added_effect in added_effects.drain(..) {
            trace!("+ added effect: capacity={}", added_effect.capacity);

            // Allocate an indirect dispatch arguments struct for this instance
            let update_dispatch_indirect_buffer_table_id = self
                .update_dispatch_indirect_buffer
                .insert(GpuDispatchIndirect::default());

            // Allocate per-effect metadata. Note that we run after Bevy has allocated
            // meshes, so we already know the buffer and position of the particle mesh, and
            // can fill the indirect args with it.
            let (gpu_effect_metadata, cached_mesh) = {
                // FIXME - this is too soon because prepare_assets::<RenderMesh>() didn't
                // necessarily run. we should defer CachedMesh until later,
                // as we don't really need it here anyway. use Added<CachedEffect> to detect
                // newly added effects later in the render frame? note also that
                // we use cmd.get(entity).insert() so technically the CachedEffect _could_
                // already exist... maybe should only do the bare minimum here
                // (insert into caches) and not update components eagerly? not sure...

                let Some(render_mesh) = render_meshes.get(added_effect.mesh.id()) else {
                    warn!(
                        "Cannot find render mesh of particle effect instance on entity {:?}, despite applying default mesh. Invalid asset handle: {:?}",
                        added_effect.entity, added_effect.mesh
                    );
                    continue;
                };
                let Some(mesh_vertex_buffer_slice) =
                    mesh_allocator.mesh_vertex_slice(&added_effect.mesh.id())
                else {
                    trace!(
                        "Effect main_entity {:?}: cannot find vertex slice of render mesh {:?}",
                        added_effect.entity,
                        added_effect.mesh
                    );
                    continue;
                };
                let mesh_index_buffer_slice =
                    mesh_allocator.mesh_index_slice(&added_effect.mesh.id());
                let indexed = if let RenderMeshBufferInfo::Indexed { index_format, .. } =
                    render_mesh.buffer_info
                {
                    if let Some(ref slice) = mesh_index_buffer_slice {
                        Some(MeshIndexSlice {
                            format: index_format,
                            buffer: slice.buffer.clone(),
                            range: slice.range.clone(),
                        })
                    } else {
                        trace!(
                            "Effect main_entity {:?}: cannot find index slice of render mesh {:?}",
                            added_effect.entity,
                            added_effect.mesh
                        );
                        continue;
                    }
                } else {
                    None
                };

                (
                    match &mesh_index_buffer_slice {
                        // Indexed mesh rendering
                        Some(mesh_index_buffer_slice) => {
                            let ret = GpuEffectMetadata {
                                vertex_or_index_count: mesh_index_buffer_slice.range.len() as u32,
                                instance_count: 0,
                                first_index_or_vertex_offset: mesh_index_buffer_slice.range.start,
                                vertex_offset_or_base_instance: mesh_vertex_buffer_slice.range.start
                                    as i32,
                                base_instance: 0,
                                alive_count: 0,
                                max_update: 0,
                                dead_count: added_effect.capacity,
                                max_spawn: added_effect.capacity,
                                ..default()
                            };
                            trace!("+ Effect[indexed]: {:?}", ret);
                            ret
                        }
                        // Non-indexed mesh rendering
                        None => {
                            let ret = GpuEffectMetadata {
                                vertex_or_index_count: mesh_vertex_buffer_slice.range.len() as u32,
                                instance_count: 0,
                                first_index_or_vertex_offset: mesh_vertex_buffer_slice.range.start,
                                vertex_offset_or_base_instance: 0,
                                base_instance: 0,
                                alive_count: 0,
                                max_update: 0,
                                dead_count: added_effect.capacity,
                                max_spawn: added_effect.capacity,
                                ..default()
                            };
                            trace!("+ Effect[non-indexed]: {:?}", ret);
                            ret
                        }
                    },
                    CachedMesh {
                        mesh: added_effect.mesh.id(),
                        buffer: mesh_vertex_buffer_slice.buffer.clone(),
                        range: mesh_vertex_buffer_slice.range.clone(),
                        indexed,
                    },
                )
            };
            let effect_metadata_buffer_table_id =
                self.effect_metadata_buffer.insert(gpu_effect_metadata);
            let dispatch_buffer_indices = DispatchBufferIndices {
                update_dispatch_indirect_buffer_table_id,
                effect_metadata_buffer_table_id,
            };

            // Insert the effect into the cache. This will allocate all the necessary
            // mandatory GPU resources as needed.
            let cached_effect = effect_cache.insert(
                added_effect.handle,
                added_effect.capacity,
                &added_effect.particle_layout,
                added_effect.layout_flags,
            );
            let mut cmd = commands.entity(added_effect.render_entity.id());
            cmd.insert((
                added_effect.entity,
                cached_effect,
                dispatch_buffer_indices,
                cached_mesh,
            ));

            // Allocate storage for properties if needed
            if !added_effect.property_layout.is_empty() {
                let cached_effect_properties = property_cache.insert(&added_effect.property_layout);
                cmd.insert(cached_effect_properties);
            } else {
                cmd.remove::<CachedEffectProperties>();
            }

            // Allocate storage for the reference to the parent effect if needed. Note that
            // we cannot yet allocate the complete parent info (CachedChildInfo) because it
            // depends on the list of children, which we can't resolve until all
            // effects have been added/removed this frame. This will be done later in
            // resolve_parents().
            if let Some(parent) = added_effect.parent.as_ref() {
                let cached_parent: CachedParentRef = CachedParentRef {
                    entity: parent.entity,
                };
                cmd.insert(cached_parent);
                trace!("+ new effect declares parent entity {:?}", parent.entity);
            } else {
                cmd.remove::<CachedParentRef>();
                trace!("+ new effect declares no parent");
            }

            // Allocate storage for GPU spawn events if needed
            if let Some(parent) = added_effect.parent.as_ref() {
                let cached_events = event_cache.allocate(parent.event_count);
                cmd.insert(cached_events);
            } else {
                cmd.remove::<CachedEffectEvents>();
            }

            // Ensure the particle@1 bind group layout exists for the given configuration of
            // particle layout and (optionally) parent particle layout.
            {
                let parent_min_binding_size = added_effect
                    .parent
                    .map(|added_parent| added_parent.layout.min_binding_size32());
                effect_cache.ensure_particle_bind_group_layout(
                    added_effect.particle_layout.min_binding_size32(),
                    parent_min_binding_size,
                );
            }

            // Ensure the metadata@3 bind group layout exists for init pass.
            {
                let consume_gpu_spawn_events = added_effect
                    .layout_flags
                    .contains(LayoutFlags::CONSUME_GPU_SPAWN_EVENTS);
                effect_cache.ensure_metadata_init_bind_group_layout(consume_gpu_spawn_events);
            }

            // We cannot yet determine the layout of the metadata@3 bind group for the
            // update pass, because it depends on the number of children, and
            // this is encoded indirectly via the number of child effects
            // pointing to this parent, and only calculated later in
            // resolve_parents().

            trace!(
                "+ added effect entity {:?}: main_entity={:?} \
                first_update_group_dispatch_buffer_index={} \
                render_effect_dispatch_buffer_id={}",
                added_effect.render_entity,
                added_effect.entity,
                update_dispatch_indirect_buffer_table_id.0,
                effect_metadata_buffer_table_id.0
            );
        }

        // Once all changes are applied, immediately schedule any GPU buffer
        // (re)allocation based on the new buffer size. The actual GPU buffer content
        // will be written later.
        if self
            .update_dispatch_indirect_buffer
            .allocate_gpu(render_device, render_queue)
        {
            // All those bind groups use the buffer so need to be re-created
            trace!("*** Dispatch indirect buffer for update pass re-allocated; clearing all bind groups using it.");
            effect_bind_groups.particle_buffers.clear();
        }
    }

    pub fn allocate_spawner(
        &mut self,
        global_transform: &GlobalTransform,
        spawn_count: u32,
        prng_seed: u32,
        effect_metadata_buffer_table_id: BufferTableId,
    ) -> u32 {
        let spawner_base = self.spawner_buffer.len() as u32;
        let transform = global_transform.compute_matrix().into();
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
            ..default()
        };
        trace!("spawner params = {:?}", spawner_params);
        self.spawner_buffer.push(spawner_params);
        spawner_base
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
    trigger: Trigger<OnRemove, CachedEffect>,
    query: Query<(
        Entity,
        MainEntity,
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
    let _span = bevy::utils::tracing::info_span!("on_remove_cached_effect").entered();

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
    )) = query.get(trigger.entity())
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
                if buffer_state != BufferState::Used {
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
    let Ok(BufferState::Free) = effect_cache.remove(cached_effect) else {
        // Buffer was not affected, so all bind groups are still valid. Nothing else to
        // do.
        return;
    };

    // Clear bind groups associated with the removed buffer
    trace!(
        "=> GPU buffer #{} gone, destroying its bind groups...",
        cached_effect.buffer_index
    );
    effect_bind_groups
        .particle_buffers
        .remove(&cached_effect.buffer_index);
    effects_meta
        .update_dispatch_indirect_buffer
        .remove(dispatch_buffer_indices.update_dispatch_indirect_buffer_table_id);
    effects_meta
        .effect_metadata_buffer
        .remove(dispatch_buffer_indices.effect_metadata_buffer_table_id);
}

/// Update the [`CachedEffect`] component for any newly allocated effect.
///
/// After this system ran, and its commands are applied, all valid extracted
/// effects have a corresponding entity in the render world, with a
/// [`CachedEffect`] component. From there, we operate on those exclusively.
pub(crate) fn add_effects(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mesh_allocator: Res<MeshAllocator>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    commands: Commands,
    mut effects_meta: ResMut<EffectsMeta>,
    mut effect_cache: ResMut<EffectCache>,
    mut property_cache: ResMut<PropertyCache>,
    mut event_cache: ResMut<EventCache>,
    mut extracted_effects: ResMut<ExtractedEffects>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    mut sort_bind_groups: ResMut<SortBindGroups>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::utils::tracing::info_span!("add_effects").entered();
    trace!("add_effects");

    // Clear last frame's buffer resizes which may have occured during last frame,
    // during `Node::run()` while the `BufferTable` could not be mutated. This is
    // the first point at which we can do that where we're not blocking the main
    // world (so, excluding the extract system).
    effects_meta
        .update_dispatch_indirect_buffer
        .clear_previous_frame_resizes();
    effects_meta
        .effect_metadata_buffer
        .clear_previous_frame_resizes();
    sort_bind_groups.clear_previous_frame_resizes();

    // Allocate new effects
    effects_meta.add_effects(
        commands,
        std::mem::take(&mut extracted_effects.added_effects),
        &render_device,
        &render_queue,
        &mesh_allocator,
        &render_meshes,
        &mut effect_bind_groups,
        &mut effect_cache,
        &mut property_cache,
        &mut event_cache,
    );

    // Note: we don't need to explicitly allocate GPU buffers for effects,
    // because EffectBuffer already contains a reference to the
    // RenderDevice, so has done so internally. This is not ideal
    // design-wise, but works.
}

/// Check if two lists of entities are equal.
fn is_child_list_changed(
    parent_entity: Entity,
    old: impl ExactSizeIterator<Item = Entity>,
    new: impl ExactSizeIterator<Item = Entity>,
) -> bool {
    if old.len() != new.len() {
        trace!(
            "Child list changed for effect {:?}: old #{} != new #{}",
            parent_entity,
            old.len(),
            new.len()
        );
        return true;
    }

    // TODO - this value is arbitrary
    if old.len() >= 16 {
        // For large-ish lists, use a hash set.
        let old = HashSet::from_iter(old);
        let new = HashSet::from_iter(new);
        if old != new {
            trace!(
                "Child list changed for effect {parent_entity:?}: old [{old:?}] != new [{new:?}]"
            );
            true
        } else {
            false
        }
    } else {
        // For small lists, just use a linear array and sort it
        let mut old = old.collect::<Vec<_>>();
        let mut new = new.collect::<Vec<_>>();
        old.sort_unstable();
        new.sort_unstable();
        if old != new {
            trace!(
                "Child list changed for effect {parent_entity:?}: old [{old:?}] != new [{new:?}]"
            );
            true
        } else {
            false
        }
    }
}

/// Resolve parents and children, updating their [`CachedParent`] and
/// [`CachedChild`] components, as well as (re-)allocating any [`GpuChildInfo`]
/// slice for all children of each parent.
pub(crate) fn resolve_parents(
    mut commands: Commands,
    q_child_effects: Query<
        (
            Entity,
            &CachedParentRef,
            &CachedEffectEvents,
            Option<&CachedChildInfo>,
        ),
        With<CachedEffect>,
    >,
    q_cached_effects: Query<(Entity, MainEntity, &CachedEffect)>,
    effect_cache: Res<EffectCache>,
    mut q_parent_effects: Query<(Entity, &mut CachedParentInfo), With<CachedEffect>>,
    mut event_cache: ResMut<EventCache>,
    mut children_from_parent: Local<
        HashMap<Entity, (Vec<(Entity, BufferBindingSource)>, Vec<GpuChildInfo>)>,
    >,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::utils::tracing::info_span!("resolve_parents").entered();
    let num_parent_effects = q_parent_effects.iter().len();
    trace!("resolve_parents: num_parents={num_parent_effects}");

    // Build map of render entity from main entity for all cached effects.
    let render_from_main_entity = q_cached_effects
        .iter()
        .map(|(render_entity, main_entity, _)| (main_entity, render_entity))
        .collect::<HashMap<_, _>>();

    // Group child effects by parent, building a list of children for each parent,
    // solely based on the declaration each child makes of its parent. This doesn't
    // mean yet that the parent exists.
    if children_from_parent.capacity() < num_parent_effects {
        let extra = num_parent_effects - children_from_parent.capacity();
        children_from_parent.reserve(extra);
    }
    for (child_entity, cached_parent_ref, cached_effect_events, cached_child_info) in
        q_child_effects.iter()
    {
        // Resolve the parent reference into the render world
        let parent_main_entity = cached_parent_ref.entity;
        let Some(parent_entity) = render_from_main_entity.get(&parent_main_entity.id()) else {
            warn!(
                "Cannot resolve parent render entity for parent main entity {:?}, removing CachedChildInfo from child entity {:?}.",
                parent_main_entity, child_entity
            );
            commands.entity(child_entity).remove::<CachedChildInfo>();
            continue;
        };
        let parent_entity = *parent_entity;

        // Resolve the parent
        let Ok((_, _, parent_cached_effect)) = q_cached_effects.get(parent_entity) else {
            // Since we failed to resolve, remove this component so the next systems ignore
            // this effect.
            warn!(
                "Unknown parent render entity {:?}, removing CachedChildInfo from child entity {:?}.",
                parent_entity, child_entity
            );
            commands.entity(child_entity).remove::<CachedChildInfo>();
            continue;
        };
        let Some(parent_buffer_binding_source) = effect_cache
            .get_buffer(parent_cached_effect.buffer_index)
            .map(|effect_buffer| effect_buffer.max_binding_source())
        else {
            // Since we failed to resolve, remove this component so the next systems ignore
            // this effect.
            warn!(
                "Unknown parent buffer #{} on entity {:?}, removing CachedChildInfo.",
                parent_cached_effect.buffer_index, child_entity
            );
            commands.entity(child_entity).remove::<CachedChildInfo>();
            continue;
        };

        let Some(child_event_buffer) = event_cache.get_buffer(cached_effect_events.buffer_index)
        else {
            // Since we failed to resolve, remove this component so the next systems ignore
            // this effect.
            warn!(
                "Unknown child event buffer #{} on entity {:?}, removing CachedChildInfo.",
                cached_effect_events.buffer_index, child_entity
            );
            commands.entity(child_entity).remove::<CachedChildInfo>();
            continue;
        };
        let child_buffer_binding_source = BufferBindingSource {
            buffer: child_event_buffer.clone(),
            offset: cached_effect_events.range.start,
            size: NonZeroU32::new(cached_effect_events.range.len() as u32).unwrap(),
        };

        // Push the child entity into the children list
        let (child_vec, child_infos) = children_from_parent.entry(parent_entity).or_default();
        let local_child_index = child_vec.len() as u32;
        child_vec.push((child_entity, child_buffer_binding_source));
        child_infos.push(GpuChildInfo {
            event_count: 0,
            init_indirect_dispatch_index: cached_effect_events.init_indirect_dispatch_index,
        });

        // Check if child info changed. Avoid overwriting if no change.
        if let Some(old_cached_child_info) = cached_child_info {
            if parent_entity == old_cached_child_info.parent
                && parent_cached_effect.slice.particle_layout
                    == old_cached_child_info.parent_particle_layout
                && parent_buffer_binding_source
                    == old_cached_child_info.parent_buffer_binding_source
                // Note: if local child index didn't change, then keep global one too for now. Chances are the parent didn't change, but anyway we can't know for now without inspecting all its children.
                && local_child_index == old_cached_child_info.local_child_index
                && cached_effect_events.init_indirect_dispatch_index
                    == old_cached_child_info.init_indirect_dispatch_index
            {
                trace!(
                    "ChildInfo didn't change for child entity {:?}, skipping component write.",
                    child_entity
                );
                continue;
            }
        }

        // Allocate (or overwrite, if already existing) the child info, now that the
        // parent is resolved.
        let cached_child_info = CachedChildInfo {
            parent: parent_entity,
            parent_particle_layout: parent_cached_effect.slice.particle_layout.clone(),
            parent_buffer_binding_source,
            local_child_index,
            global_child_index: u32::MAX, // fixed up later by fixup_parents()
            init_indirect_dispatch_index: cached_effect_events.init_indirect_dispatch_index,
        };
        commands.entity(child_entity).insert(cached_child_info);
        trace!("Spawned CachedChildInfo on child entity {:?}", child_entity);
    }

    // Once all parents are resolved, diff all children of already-cached parents,
    // and re-allocate their GpuChildInfo if needed.
    for (parent_entity, mut cached_parent_info) in q_parent_effects.iter_mut() {
        // Fetch the newly extracted list of children
        let Some((_, (children, child_infos))) = children_from_parent.remove_entry(&parent_entity)
        else {
            trace!("Entity {parent_entity:?} is no more a parent, removing CachedParentInfo component...");
            commands.entity(parent_entity).remove::<CachedParentInfo>();
            continue;
        };

        // Check if any child changed compared to the existing CachedChildren component
        if !is_child_list_changed(
            parent_entity,
            cached_parent_info
                .children
                .iter()
                .map(|(entity, _)| *entity),
            children.iter().map(|(entity, _)| *entity),
        ) {
            continue;
        }

        event_cache.reallocate_child_infos(
            parent_entity,
            children,
            &child_infos[..],
            cached_parent_info.deref_mut(),
        );
    }

    // Once this is done, the children hash map contains all entries which don't
    // already have a CachedParentInfo component. That is, all entities which are
    // new parents.
    for (parent_entity, (children, child_infos)) in children_from_parent.drain() {
        let cached_parent_info =
            event_cache.allocate_child_infos(parent_entity, children, &child_infos[..]);
        commands.entity(parent_entity).insert(cached_parent_info);
    }

    // // Once all changes are applied, immediately schedule any GPU buffer
    // // (re)allocation based on the new buffer size. The actual GPU buffer
    // content // will be written later.
    // if event_cache
    //     .child_infos()
    //     .allocate_gpu(render_device, render_queue)
    // {
    //     // All those bind groups use the buffer so need to be re-created
    //     effect_bind_groups.particle_buffers.clear();
    // }
}

pub fn fixup_parents(
    q_changed_parents: Query<(Entity, &CachedParentInfo), Changed<CachedParentInfo>>,
    mut q_children: Query<&mut CachedChildInfo>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::utils::tracing::info_span!("fixup_parents").entered();
    trace!("fixup_parents");

    // Once all parents are (re-)allocated, fix up the global index of all
    // children if the parent base index changed.
    trace!(
        "Updating the global index of children of parent effects whose child list just changed..."
    );
    for (parent_entity, cached_parent_info) in q_changed_parents.iter() {
        let base_index =
            cached_parent_info.byte_range.start / GpuChildInfo::SHADER_SIZE.get() as u32;
        trace!(
            "Updating {} children of parent effect {:?} with base child index {}...",
            cached_parent_info.children.len(),
            parent_entity,
            base_index
        );
        for (child_entity, _) in &cached_parent_info.children {
            let Ok(mut cached_child_info) = q_children.get_mut(*child_entity) else {
                continue;
            };
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

// TEMP - Mark all cached effects as invalid for this frame until another system
// explicitly marks them as valid. Otherwise we early out in some parts, and
// reuse by mistake the previous frame's extraction.
pub fn clear_all_effects(
    mut commands: Commands,
    mut q_cached_effects: Query<Entity, With<BatchInput>>,
) {
    for entity in &mut q_cached_effects {
        if let Some(mut cmd) = commands.get_entity(entity) {
            cmd.remove::<BatchInput>();
        }
    }
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

/// Render world cached mesh infos for a single effect instance.
#[derive(Debug, Clone, Component)]
pub(crate) struct CachedMesh {
    /// Asset of the effect mesh to draw.
    pub mesh: AssetId<Mesh>,
    /// GPU buffer storing the [`mesh`] of the effect.
    pub buffer: Buffer,
    /// Range slice inside the GPU buffer for the effect mesh.
    pub range: Range<u32>,
    /// Indexed rendering metadata.
    #[allow(unused)]
    pub indexed: Option<MeshIndexSlice>,
}

/// Render world cached properties info for a single effect instance.
#[allow(unused)]
#[derive(Debug, Component)]
pub(crate) struct CachedProperties {
    /// Layout of the effect properties.
    pub layout: PropertyLayout,
    /// Index of the buffer in the [`EffectCache`].
    pub buffer_index: u32,
    /// Offset in bytes inside the buffer.
    pub offset: u32,
    /// Binding size in bytes of the property struct.
    pub binding_size: u32,
}

#[derive(SystemParam)]
pub struct PrepareEffectsReadOnlyParams<'w, 's> {
    sim_params: Res<'w, SimParams>,
    render_device: Res<'w, RenderDevice>,
    render_queue: Res<'w, RenderQueue>,
    #[system_param(ignore)]
    marker: PhantomData<&'s usize>,
}

#[derive(SystemParam)]
pub struct PipelineSystemParams<'w, 's> {
    pipeline_cache: Res<'w, PipelineCache>,
    init_pipeline: ResMut<'w, ParticlesInitPipeline>,
    indirect_pipeline: Res<'w, DispatchIndirectPipeline>,
    update_pipeline: ResMut<'w, ParticlesUpdatePipeline>,
    specialized_init_pipelines: ResMut<'w, SpecializedComputePipelines<ParticlesInitPipeline>>,
    specialized_update_pipelines: ResMut<'w, SpecializedComputePipelines<ParticlesUpdatePipeline>>,
    specialized_indirect_pipelines:
        ResMut<'w, SpecializedComputePipelines<DispatchIndirectPipeline>>,
    #[system_param(ignore)]
    marker: PhantomData<&'s usize>,
}

pub(crate) fn prepare_effects(
    mut commands: Commands,
    read_only_params: PrepareEffectsReadOnlyParams,
    mut pipelines: PipelineSystemParams,
    mut property_cache: ResMut<PropertyCache>,
    event_cache: Res<EventCache>,
    mut effect_cache: ResMut<EffectCache>,
    mut effects_meta: ResMut<EffectsMeta>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    mut extracted_effects: ResMut<ExtractedEffects>,
    mut property_bind_groups: ResMut<PropertyBindGroups>,
    q_cached_effects: Query<(
        MainEntity,
        &CachedEffect,
        Ref<CachedMesh>,
        &DispatchBufferIndices,
        Option<&CachedEffectProperties>,
        Option<&CachedParentInfo>,
        Option<&CachedChildInfo>,
        Option<&CachedEffectEvents>,
    )>,
    q_debug_all_entities: Query<MainEntity>,
    mut gpu_buffer_operation_queue: ResMut<GpuBufferOperationQueue>,
    mut sort_bind_groups: ResMut<SortBindGroups>,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::utils::tracing::info_span!("prepare_effects").entered();
    trace!("prepare_effects");

    // Workaround for too many params in system (TODO: refactor to split work?)
    let sim_params = read_only_params.sim_params.into_inner();
    let render_device = read_only_params.render_device.into_inner();
    let render_queue = read_only_params.render_queue.into_inner();
    let pipeline_cache = pipelines.pipeline_cache.into_inner();
    let specialized_init_pipelines = pipelines.specialized_init_pipelines.into_inner();
    let specialized_update_pipelines = pipelines.specialized_update_pipelines.into_inner();
    let specialized_indirect_pipelines = pipelines.specialized_indirect_pipelines.into_inner();

    // // sort first by z and then by handle. this ensures that, when possible,
    // batches span multiple z layers // batches won't span z-layers if there is
    // another batch between them extracted_effects.effects.sort_by(|a, b| {
    //     match FloatOrd(a.transform.w_axis[2]).cmp(&FloatOrd(b.transform.
    // w_axis[2])) {         Ordering::Equal => a.handle.cmp(&b.handle),
    //         other => other,
    //     }
    // });

    // Ensure the indirect pipelines are created
    if effects_meta.indirect_pipeline_ids[0] == CachedComputePipelineId::INVALID {
        effects_meta.indirect_pipeline_ids[0] = specialized_indirect_pipelines.specialize(
            pipeline_cache,
            &pipelines.indirect_pipeline,
            DispatchIndirectPipelineKey { has_events: false },
        );
    }
    if effects_meta.indirect_pipeline_ids[1] == CachedComputePipelineId::INVALID {
        effects_meta.indirect_pipeline_ids[1] = specialized_indirect_pipelines.specialize(
            pipeline_cache,
            &pipelines.indirect_pipeline,
            DispatchIndirectPipelineKey { has_events: true },
        );
    }
    if effects_meta.active_indirect_pipeline_id == CachedComputePipelineId::INVALID {
        effects_meta.active_indirect_pipeline_id = effects_meta.indirect_pipeline_ids[0];
    } else {
        // If this is the first time we insert an event buffer, we need to switch the
        // indirect pass from non-event to event mode. That is, we need to re-allocate
        // the pipeline with the child infos buffer binding. Conversely, if there's no
        // more effect using GPU spawn events, we can deallocate.
        let was_empty =
            effects_meta.active_indirect_pipeline_id == effects_meta.indirect_pipeline_ids[0];
        let is_empty = event_cache.child_infos().is_empty();
        if was_empty && !is_empty {
            trace!("First event buffer inserted; switching indirect pass to event mode...");
            effects_meta.active_indirect_pipeline_id = effects_meta.indirect_pipeline_ids[1];
        } else if is_empty && !was_empty {
            trace!("Last event buffer removed; switching indirect pass to no-event mode...");
            effects_meta.active_indirect_pipeline_id = effects_meta.indirect_pipeline_ids[0];
        }
    }

    gpu_buffer_operation_queue.begin_frame();

    // Clear per-instance buffers, which are filled below and re-uploaded each frame
    effects_meta.spawner_buffer.clear();

    // Build batcher inputs from extracted effects, updating all cached components
    // for each effect on the fly.
    let effects = std::mem::take(&mut extracted_effects.effects);
    let extracted_effect_count = effects.len();
    let mut prepared_effect_count = 0;
    for extracted_effect in effects.into_iter() {
        // Skip effects not cached. Since we're iterating over the extracted effects
        // instead of the cached ones, it might happen we didn't cache some effect on
        // purpose because they failed earlier validations.
        // FIXME - extract into ECS directly so we don't have to do that?
        let Ok((
            main_entity,
            cached_effect,
            cached_mesh,
            dispatch_buffer_indices,
            cached_effect_properties,
            cached_parent_info,
            cached_child_info,
            cached_effect_events,
        )) = q_cached_effects.get(extracted_effect.render_entity.id())
        else {
            warn!(
                "Unknown render entity {:?} for extracted effect.",
                extracted_effect.render_entity.id()
            );
            if let Ok(main_entity) = q_debug_all_entities.get(extracted_effect.render_entity.id()) {
                info!(
                    "Render entity {:?} exists with main entity {:?}, some component missing!",
                    extracted_effect.render_entity.id(),
                    main_entity
                );
            } else {
                info!(
                    "Render entity {:?} does not exists with a MainEntity.",
                    extracted_effect.render_entity.id()
                );
            }
            continue;
        };

        let effect_slice = EffectSlice {
            slice: cached_effect.slice.range(),
            buffer_index: cached_effect.buffer_index,
            particle_layout: cached_effect.slice.particle_layout.clone(),
        };

        let has_event_buffer = cached_child_info.is_some();
        // FIXME: decouple "consumes event" from "reads parent particle" (here, p.layout
        // should be Option<T>, not T)
        let property_layout_min_binding_size = if extracted_effect.property_layout.is_empty() {
            None
        } else {
            Some(extracted_effect.property_layout.min_binding_size())
        };

        // Schedule some GPU buffer operation to update the number of workgroups to
        // dispatch during the indirect init pass of this effect based on the number of
        // GPU spawn events written in its buffer.
        if let (Some(cached_effect_events), Some(cached_child_info)) =
            (cached_effect_events, cached_child_info)
        {
            debug_assert_eq!(
                GpuChildInfo::min_size().get() % 4,
                0,
                "Invalid GpuChildInfo alignment."
            );

            // Resolve parent entry
            let Ok((_, _, _, _, _, cached_parent_info, _, _)) =
                q_cached_effects.get(cached_child_info.parent)
            else {
                continue;
            };
            let Some(cached_parent_info) = cached_parent_info else {
                error!("Effect {:?} indicates its parent is {:?}, but that parent effect is missing a CachedParentInfo component. This is a bug.", extracted_effect.render_entity.id(), cached_child_info.parent);
                continue;
            };

            let init_indirect_dispatch_index = cached_effect_events.init_indirect_dispatch_index;
            let child_info_size_u32 = GpuChildInfo::min_size().get() as u32 / 4;
            assert_eq!(0, cached_parent_info.byte_range.start % 4);
            let global_child_index = cached_child_info.global_child_index;

            // Schedule a fill dispatch
            let event_buffer_index = cached_effect_events.buffer_index;
            let event_slice = cached_effect_events.range.clone();
            trace!(
                "queue_init_fill(): event_buffer_index={} event_slice={:?} src:global_child_index={} dst:init_indirect_dispatch_index={}",
                event_buffer_index,
                event_slice,
                global_child_index,
                init_indirect_dispatch_index,
            );
            gpu_buffer_operation_queue.enqueue_init_fill(
                event_buffer_index,
                event_slice,
                GpuBufferOperationArgs {
                    src_offset: global_child_index,
                    src_stride: child_info_size_u32,
                    dst_offset: init_indirect_dispatch_index,
                    dst_stride: GpuDispatchIndirect::SHADER_SIZE.get() as u32 / 4,
                    count: 1, // FIXME - should be a batch here!!
                },
            );
        }

        // Create init pipeline key flags.
        let init_pipeline_key_flags = {
            let mut flags = ParticleInitPipelineKeyFlags::empty();
            flags.set(
                ParticleInitPipelineKeyFlags::ATTRIBUTE_PREV,
                effect_slice.particle_layout.contains(Attribute::PREV),
            );
            flags.set(
                ParticleInitPipelineKeyFlags::ATTRIBUTE_NEXT,
                effect_slice.particle_layout.contains(Attribute::NEXT),
            );
            flags.set(
                ParticleInitPipelineKeyFlags::CONSUME_GPU_SPAWN_EVENTS,
                has_event_buffer,
            );
            flags
        };

        // This should always exist by the time we reach this point, because we should
        // have inserted any property in the cache, which would have allocated the
        // proper bind group layout (or the default no-property one).
        let spawner_bind_group_layout = property_cache
            .bind_group_layout(property_layout_min_binding_size)
            .unwrap_or_else(|| {
                panic!(
                    "Failed to find spawner@2 bind group layout for property binding size {:?}",
                    property_layout_min_binding_size,
                )
            });
        trace!(
            "Retrieved spawner@2 bind group layout {:?} for property binding size {:?}.",
            spawner_bind_group_layout.id(),
            property_layout_min_binding_size
        );

        // Fetch the bind group layouts from the cache
        trace!("cached_child_info={:?}", cached_child_info);
        let (parent_particle_layout_min_binding_size, parent_buffer_index) =
            if let Some(cached_child) = cached_child_info.as_ref() {
                let Ok((_, parent_cached_effect, _, _, _, _, _, _)) =
                    q_cached_effects.get(cached_child.parent)
                else {
                    // At this point we should have discarded invalid effects with a missing parent,
                    // so if the parent is not found this is a bug.
                    error!(
                        "Effect main_entity {:?}: parent render entity {:?} not found.",
                        main_entity, cached_child.parent
                    );
                    continue;
                };
                (
                    Some(
                        parent_cached_effect
                            .slice
                            .particle_layout
                            .min_binding_size32(),
                    ),
                    Some(parent_cached_effect.buffer_index),
                )
            } else {
                (None, None)
            };
        let Some(particle_bind_group_layout) = effect_cache.particle_bind_group_layout(
            effect_slice.particle_layout.min_binding_size32(),
            parent_particle_layout_min_binding_size,
        ) else {
            error!("Failed to find particle sim bind group @1 for min_binding_size={} parent_min_binding_size={:?}", 
            effect_slice.particle_layout.min_binding_size32(), parent_particle_layout_min_binding_size);
            continue;
        };
        let particle_bind_group_layout = particle_bind_group_layout.clone();
        trace!(
            "Retrieved particle@1 bind group layout {:?} for particle binding size {:?} and parent binding size {:?}.",
            particle_bind_group_layout.id(),
            effect_slice.particle_layout.min_binding_size32(),
            parent_particle_layout_min_binding_size,
        );

        let particle_layout_min_binding_size = effect_slice.particle_layout.min_binding_size32();
        let spawner_bind_group_layout = spawner_bind_group_layout.clone();

        // Specialize the init pipeline based on the effect.
        let init_pipeline_id = {
            let consume_gpu_spawn_events = init_pipeline_key_flags
                .contains(ParticleInitPipelineKeyFlags::CONSUME_GPU_SPAWN_EVENTS);

            // Fetch the metadata@3 bind group layout from the cache
            let metadata_bind_group_layout = effect_cache
                .metadata_init_bind_group_layout(consume_gpu_spawn_events)
                .unwrap()
                .clone();

            // https://github.com/bevyengine/bevy/issues/17132
            let particle_bind_group_layout_id = particle_bind_group_layout.id();
            let spawner_bind_group_layout_id = spawner_bind_group_layout.id();
            let metadata_bind_group_layout_id = metadata_bind_group_layout.id();
            pipelines.init_pipeline.temp_particle_bind_group_layout =
                Some(particle_bind_group_layout.clone());
            pipelines.init_pipeline.temp_spawner_bind_group_layout =
                Some(spawner_bind_group_layout.clone());
            pipelines.init_pipeline.temp_metadata_bind_group_layout =
                Some(metadata_bind_group_layout);
            let init_pipeline_id: CachedComputePipelineId = specialized_init_pipelines.specialize(
                pipeline_cache,
                &pipelines.init_pipeline,
                ParticleInitPipelineKey {
                    shader: extracted_effect.effect_shaders.init.clone(),
                    particle_layout_min_binding_size,
                    parent_particle_layout_min_binding_size,
                    flags: init_pipeline_key_flags,
                    particle_bind_group_layout_id,
                    spawner_bind_group_layout_id,
                    metadata_bind_group_layout_id,
                },
            );
            // keep things tidy; this is just a hack, should not persist
            pipelines.init_pipeline.temp_particle_bind_group_layout = None;
            pipelines.init_pipeline.temp_spawner_bind_group_layout = None;
            pipelines.init_pipeline.temp_metadata_bind_group_layout = None;
            trace!("Init pipeline specialized: id={:?}", init_pipeline_id);

            init_pipeline_id
        };

        let update_pipeline_id = {
            let num_event_buffers = cached_parent_info
                .map(|p| p.children.len() as u32)
                .unwrap_or_default();

            // FIXME: currently don't hava a way to determine when this is needed, because
            // we know the number of children per parent only after resolving
            // all parents, but by that point we forgot if this is a newly added
            // effect or not. So since we need to re-ensure for all effects, not
            // only new ones, might as well do here...
            effect_cache.ensure_metadata_update_bind_group_layout(num_event_buffers);

            // Fetch the bind group layouts from the cache
            let metadata_bind_group_layout = effect_cache
                .metadata_update_bind_group_layout(num_event_buffers)
                .unwrap()
                .clone();

            // https://github.com/bevyengine/bevy/issues/17132
            let particle_bind_group_layout_id = particle_bind_group_layout.id();
            let spawner_bind_group_layout_id = spawner_bind_group_layout.id();
            let metadata_bind_group_layout_id = metadata_bind_group_layout.id();
            pipelines.update_pipeline.temp_particle_bind_group_layout =
                Some(particle_bind_group_layout);
            pipelines.update_pipeline.temp_spawner_bind_group_layout =
                Some(spawner_bind_group_layout);
            pipelines.update_pipeline.temp_metadata_bind_group_layout =
                Some(metadata_bind_group_layout);
            let update_pipeline_id = specialized_update_pipelines.specialize(
                pipeline_cache,
                &pipelines.update_pipeline,
                ParticleUpdatePipelineKey {
                    shader: extracted_effect.effect_shaders.update.clone(),
                    particle_layout: effect_slice.particle_layout.clone(),
                    parent_particle_layout_min_binding_size,
                    num_event_buffers,
                    particle_bind_group_layout_id,
                    spawner_bind_group_layout_id,
                    metadata_bind_group_layout_id,
                },
            );
            // keep things tidy; this is just a hack, should not persist
            pipelines.update_pipeline.temp_particle_bind_group_layout = None;
            pipelines.update_pipeline.temp_spawner_bind_group_layout = None;
            pipelines.update_pipeline.temp_metadata_bind_group_layout = None;
            trace!("Update pipeline specialized: id={:?}", update_pipeline_id);

            update_pipeline_id
        };

        let init_and_update_pipeline_ids = InitAndUpdatePipelineIds {
            init: init_pipeline_id,
            update: update_pipeline_id,
        };

        // For ribbons, which need particle sorting, create a bind group layout for
        // sorting the effect, based on its particle layout.
        if extracted_effect.layout_flags.contains(LayoutFlags::RIBBONS) {
            if let Err(err) = sort_bind_groups.ensure_sort_fill_bind_group_layout(
                pipeline_cache,
                &extracted_effect.particle_layout,
            ) {
                error!(
                    "Failed to create bind group for ribbon effect sorting: {:?}",
                    err
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

        let spawner_index = effects_meta.allocate_spawner(
            &extracted_effect.transform,
            extracted_effect.spawn_count,
            extracted_effect.prng_seed,
            dispatch_buffer_indices.effect_metadata_buffer_table_id,
        );

        trace!(
            "Updating cached effect at entity {:?}...",
            extracted_effect.render_entity.id()
        );
        let mut cmd = commands.entity(extracted_effect.render_entity.id());
        cmd.insert(BatchInput {
            handle: extracted_effect.handle,
            entity: extracted_effect.render_entity.id(),
            main_entity: extracted_effect.main_entity,
            effect_slice,
            init_and_update_pipeline_ids,
            parent_buffer_index,
            event_buffer_index: cached_effect_events.map(|cee| cee.buffer_index),
            child_effects: cached_parent_info
                .map(|cp| cp.children.clone())
                .unwrap_or_default(),
            layout_flags: extracted_effect.layout_flags,
            texture_layout: extracted_effect.texture_layout.clone(),
            textures: extracted_effect.textures.clone(),
            alpha_mode: extracted_effect.alpha_mode,
            particle_layout: extracted_effect.particle_layout.clone(),
            shaders: extracted_effect.effect_shaders,
            spawner_base: spawner_index,
            spawn_count: extracted_effect.spawn_count,
            position: extracted_effect.transform.translation(),
            init_indirect_dispatch_index: cached_child_info
                .map(|cc| cc.init_indirect_dispatch_index),
        });

        // Update properties
        if let Some(cached_effect_properties) = cached_effect_properties {
            // Because the component is persisted, it may be there from a previous version
            // of the asset. And add_remove_effects() only add new instances or remove old
            // ones, but doesn't update existing ones. Check if it needs to be removed.
            // FIXME - Dedupe with add_remove_effect(), we shouldn't have 2 codepaths doing
            // the same thing at 2 different times.
            if extracted_effect.property_layout.is_empty() {
                trace!(
                    "Render entity {:?} had CachedEffectProperties component, but newly extracted property layout is empty. Removing component...",
                    extracted_effect.render_entity.id(),
                );
                cmd.remove::<CachedEffectProperties>();
                // Also remove the other one. FIXME - dedupe those two...
                cmd.remove::<CachedProperties>();

                if extracted_effect.property_data.is_some() {
                    warn!(
                        "Effect on entity {:?} doesn't declare any property in its Module, but some property values were provided. Those values will be discarded.",
                        extracted_effect.main_entity.id(),
                    );
                }
            } else {
                // Insert a new component or overwrite the existing one
                cmd.insert(CachedProperties {
                    layout: extracted_effect.property_layout.clone(),
                    buffer_index: cached_effect_properties.buffer_index,
                    offset: cached_effect_properties.range.start,
                    binding_size: cached_effect_properties.range.len() as u32,
                });

                // Write properties for this effect if they were modified.
                // FIXME - This doesn't work with batching!
                if let Some(property_data) = &extracted_effect.property_data {
                    trace!(
                    "Properties changed; (re-)uploading to GPU... New data: {} bytes. Capacity: {} bytes.",
                    property_data.len(),
                    cached_effect_properties.range.len(),
                );
                    if property_data.len() <= cached_effect_properties.range.len() {
                        let property_buffer = property_cache.buffers_mut()
                            [cached_effect_properties.buffer_index as usize]
                            .as_mut()
                            .unwrap();
                        property_buffer.write(cached_effect_properties.range.start, property_data);
                    } else {
                        error!(
                            "Cannot upload properties: existing property slice in property buffer #{} is too small ({} bytes) for the new data ({} bytes).",
                            cached_effect_properties.buffer_index,
                            cached_effect_properties.range.len(),
                            property_data.len()
                        );
                    }
                }
            }
        } else {
            // No property on the effect; remove the component
            trace!(
                "No CachedEffectProperties on render entity {:?}, remove any CachedProperties component too.",
                extracted_effect.render_entity.id()
            );
            cmd.remove::<CachedProperties>();
        }

        // Now that the effect is entirely prepared and all GPU resources are allocated,
        // update its GpuEffectMetadata with all those infos.
        // FIXME - should do this only when the below changes (not only the mesh), via
        // some invalidation mechanism and ECS change detection.
        if cached_mesh.is_changed() {
            let capacity = cached_effect.slice.len();

            // Global and local indices of this effect as a child of another (parent) effect
            let (global_child_index, local_child_index) = cached_child_info
                .map(|cci| (cci.global_child_index, cci.local_child_index))
                .unwrap_or_default();

            // Base index of all children of this (parent) effect
            let base_child_index = cached_parent_info
                .map(|cpi| {
                    debug_assert_eq!(
                        cpi.byte_range.start % GpuChildInfo::SHADER_SIZE.get() as u32,
                        0
                    );
                    cpi.byte_range.start / GpuChildInfo::SHADER_SIZE.get() as u32
                })
                .unwrap_or_default();

            let particle_stride = extracted_effect.particle_layout.min_binding_size32().get() / 4;
            let sort_key_offset = extracted_effect
                .particle_layout
                .offset(Attribute::RIBBON_ID)
                .unwrap_or(0)
                / 4;
            let sort_key2_offset = extracted_effect
                .particle_layout
                .offset(Attribute::AGE)
                .unwrap_or(0)
                / 4;

            let mut gpu_effect_metadata = GpuEffectMetadata {
                instance_count: 0,
                base_instance: 0,
                alive_count: 0,
                max_update: 0,
                dead_count: capacity,
                max_spawn: capacity,
                ping: 0,
                spawner_index: 0xDEADBEEF, // unused
                indirect_dispatch_index: dispatch_buffer_indices
                    .update_dispatch_indirect_buffer_table_id
                    .0,
                // Note: the indirect draw args are at the start of the GpuEffectMetadata struct
                indirect_render_index: dispatch_buffer_indices.effect_metadata_buffer_table_id.0,
                init_indirect_dispatch_index: cached_effect_events
                    .map(|cee| cee.init_indirect_dispatch_index)
                    .unwrap_or_default(),
                local_child_index,
                global_child_index,
                base_child_index,
                particle_stride,
                sort_key_offset,
                sort_key2_offset,
                ..default()
            };
            if let Some(indexed) = &cached_mesh.indexed {
                gpu_effect_metadata.vertex_or_index_count = indexed.range.len() as u32;
                gpu_effect_metadata.first_index_or_vertex_offset = indexed.range.start;
                gpu_effect_metadata.vertex_offset_or_base_instance = cached_mesh.range.start as i32;
            } else {
                gpu_effect_metadata.vertex_or_index_count = cached_mesh.range.len() as u32;
                gpu_effect_metadata.first_index_or_vertex_offset = cached_mesh.range.start;
                gpu_effect_metadata.vertex_offset_or_base_instance = 0;
            };
            assert!(dispatch_buffer_indices
                .effect_metadata_buffer_table_id
                .is_valid());
            effects_meta.effect_metadata_buffer.update(
                dispatch_buffer_indices.effect_metadata_buffer_table_id,
                gpu_effect_metadata,
            );

            warn!(
                "Updated metadata entry {} for effect {:?}, this will reset it.",
                dispatch_buffer_indices.effect_metadata_buffer_table_id.0, main_entity
            );
        }

        prepared_effect_count += 1;
    }
    trace!("Prepared {prepared_effect_count}/{extracted_effect_count} extracted effect(s)");

    // Once all EffectMetadata values are written, schedule a GPU upload
    if effects_meta
        .effect_metadata_buffer
        .allocate_gpu(render_device, render_queue)
    {
        // All those bind groups use the buffer so need to be re-created
        trace!("*** Effect metadata buffer re-allocated; clearing all bind groups using it.");
        effects_meta.indirect_metadata_bind_group = None;
        effect_bind_groups.init_metadata_bind_groups.clear();
        effect_bind_groups.update_metadata_bind_groups.clear();
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
        property_bind_groups.clear(true);
        effects_meta.indirect_spawner_bind_group = None;
    }

    // Update simulation parameters
    effects_meta.sim_params_uniforms.set(sim_params.into());
    {
        let gpu_sim_params = effects_meta.sim_params_uniforms.get_mut();
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
    }
    let prev_buffer_id = effects_meta.sim_params_uniforms.buffer().map(|b| b.id());
    effects_meta
        .sim_params_uniforms
        .write_buffer(render_device, render_queue);
    if prev_buffer_id != effects_meta.sim_params_uniforms.buffer().map(|b| b.id()) {
        // Buffer changed, invalidate bind groups
        effects_meta.indirect_sim_params_bind_group = None;
    }
}

pub(crate) fn batch_effects(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    effects_meta: Res<EffectsMeta>,
    mut sort_bind_groups: ResMut<SortBindGroups>,
    mut q_cached_effects: Query<(
        Entity,
        &CachedMesh,
        Option<&CachedEffectEvents>,
        Option<&CachedChildInfo>,
        Option<&CachedProperties>,
        &mut DispatchBufferIndices,
        &mut BatchInput,
    )>,
    mut sorted_effect_batches: ResMut<SortedEffectBatches>,
    mut gpu_buffer_operation_queue: ResMut<GpuBufferOperationQueue>,
) {
    trace!("batch_effects");

    // Sort first by effect buffer index, then by slice range (see EffectSlice)
    // inside that buffer. This is critical for batching to work, because
    // batching effects is based on compatible items, which implies same GPU
    // buffer and continuous slice ranges (the next slice start must be equal to
    // the previous start end, without gap). EffectSlice already contains both
    // information, and the proper ordering implementation.
    // effect_entity_list.sort_by_key(|a| a.effect_slice.clone());

    // For now we re-create that buffer each frame. Since there's no CPU -> GPU
    // transfer, this is pretty cheap in practice.
    sort_bind_groups.clear_indirect_dispatch_buffer();

    // Loop on all extracted effects in order, and try to batch them together to
    // reduce draw calls. -- currently does nothing, batching was broken and never
    // fixed.
    // FIXME - This is in ECS order, if we re-add the sorting above we need a
    // different order here!
    trace!("Batching {} effects...", q_cached_effects.iter().len());
    sorted_effect_batches.clear();
    for (
        entity,
        cached_mesh,
        cached_effect_events,
        cached_child_info,
        cached_properties,
        dispatch_buffer_indices,
        mut input,
    ) in &mut q_cached_effects
    {
        // Detect if this cached effect was not updated this frame by a new extracted
        // effect. This happens when e.g. the effect is invisible and not simulated, or
        // some error prevented it from being extracted. We use the pipeline IDs vector
        // as a marker, because each frame we move it out of the CachedGroup
        // component during batching, so if empty this means a new one was not created
        // this frame.
        // if input.init_and_update_pipeline_ids.is_empty() {
        //     trace!(
        //         "Skipped cached effect on render entity {:?}: not extracted this
        // frame.",         entity
        //     );
        //     continue;
        // }

        let translation = input.position;

        // Spawn one EffectBatch per instance (no batching; TODO). This contains
        // most of the data needed to drive rendering. However this doesn't drive
        // rendering; this is just storage.
        let mut effect_batch = EffectBatch::from_input(
            cached_mesh,
            cached_effect_events,
            cached_child_info,
            &mut input,
            *dispatch_buffer_indices.as_ref(),
            cached_properties.map(|cp| PropertyBindGroupKey {
                buffer_index: cp.buffer_index,
                binding_size: cp.binding_size,
            }),
            cached_properties.map(|cp| cp.offset),
        );

        // If the batch has ribbons, we need to sort the particles by RIBBON_ID and AGE
        // for ribbon meshing, in order to avoid gaps when some particles in the middle
        // of the ribbon die (since we can't guarantee a linear lifetime through the
        // ribbon).
        if input.layout_flags.contains(LayoutFlags::RIBBONS) {
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
                let src_binding_offset = effects_meta.effect_metadata_buffer.dynamic_offset(
                    effect_batch
                        .dispatch_buffer_indices
                        .effect_metadata_buffer_table_id,
                );
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
                    src_offset, 5,
                    "GpuEffectMetadata changed, update this assert."
                );
                // FIXME - This is a quick fix to get 0.15 out. The previous code used the
                // dynamic binding offset, but the indirect dispatch structs are only 12 bytes,
                // os are not aligned to min_storage_buffer_offset_alignment. The fix uses a
                // binding offset of 0 and binds the entire destination buffer,
                // then use the dst_offset value embedded inside the GpuBufferOperationArgs to
                // index the proper offset in the buffer. This requires of
                // course binding the entire buffer, or at least enough to index all operations
                // (hence the None below). This is not really a general solution, so should be
                // reviewed.
                let dst_offset = sort_bind_groups
                    .get_indirect_dispatch_byte_offset(sort_fill_indirect_dispatch_index)
                    / 4;
                gpu_buffer_operation_queue.enqueue(
                    GpuBufferOperationType::FillDispatchArgs,
                    GpuBufferOperationArgs {
                        src_offset,
                        src_stride: effects_meta.gpu_limits.effect_metadata_aligned_size.get() / 4,
                        dst_offset,
                        dst_stride: GpuDispatchIndirect::SHADER_SIZE.get() as u32 / 4,
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
            })
            .insert(TemporaryRenderEntity);
    }

    // Once all GPU operations for this frame are enqueued, upload them to GPU
    gpu_buffer_operation_queue.end_frame(&render_device, &render_queue);

    sorted_effect_batches.sort();
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
    pub buffer_index: u32,
    pub effect_metadata_buffer: BufferId,
    pub effect_metadata_offset: u32,
    pub consume_event_key: Option<ConsumeEventKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct UpdateMetadataBindGroupKey {
    pub buffer_index: u32,
    pub effect_metadata_buffer: BufferId,
    pub effect_metadata_offset: u32,
    pub child_info_buffer_id: Option<BufferId>,
    pub event_buffers_keys: Vec<BindingKey>,
}

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
    /// Map from buffer index to the bind groups shared among all effects that
    /// use that buffer.
    particle_buffers: HashMap<u32, BufferBindGroups>,
    /// Map of bind groups for image assets used as particle textures.
    images: HashMap<AssetId<Image>, BindGroup>,
    /// Map from buffer index to its metadata bind group (group 3) for the init
    /// pass.
    // FIXME - doesn't work with batching; this should be the instance ID
    init_metadata_bind_groups: HashMap<u32, CachedBindGroup<InitMetadataBindGroupKey>>,
    /// Map from buffer index to its metadata bind group (group 3) for the
    /// update pass.
    // FIXME - doesn't work with batching; this should be the instance ID
    update_metadata_bind_groups: HashMap<u32, CachedBindGroup<UpdateMetadataBindGroupKey>>,
    /// Map from an effect material to its bind group.
    material_bind_groups: HashMap<Material, BindGroup>,
    /// Map from an event buffer index to the bind group @0 for the init fill
    /// pass in charge of filling all its init dispatches.
    init_fill_dispatch: HashMap<u32, BindGroup>,
}

impl EffectBindGroups {
    pub fn particle_render(&self, buffer_index: u32) -> Option<&BindGroup> {
        self.particle_buffers
            .get(&buffer_index)
            .map(|bg| &bg.render)
    }

    /// Retrieve the metadata@3 bind group for the init pass, creating it if
    /// needed.
    pub(self) fn get_or_create_init_metadata(
        &mut self,
        effect_batch: &EffectBatch,
        gpu_limits: &GpuLimits,
        render_device: &RenderDevice,
        layout: &BindGroupLayout,
        effect_metadata_buffer: &Buffer,
        consume_event_buffers: Option<ConsumeEventBuffers>,
    ) -> Result<&BindGroup, ()> {
        let DispatchBufferIndices {
            effect_metadata_buffer_table_id,
            ..
        } = &effect_batch.dispatch_buffer_indices;

        let effect_metadata_offset =
            gpu_limits.effect_metadata_offset(effect_metadata_buffer_table_id.0) as u32;
        let key = InitMetadataBindGroupKey {
            buffer_index: effect_batch.buffer_index,
            effect_metadata_buffer: effect_metadata_buffer.id(),
            effect_metadata_offset,
            consume_event_key: consume_event_buffers.as_ref().map(Into::into),
        };

        let make_entry = || {
            let mut entries = Vec::with_capacity(3);
            entries.push(
                // @group(3) @binding(0) var<storage, read_write> effect_metadata : EffectMetadata;
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: effect_metadata_buffer,
                        offset: key.effect_metadata_offset as u64,
                        size: Some(gpu_limits.effect_metadata_size()),
                    }),
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
                    effect_batch.buffer_index,
                    effect_metadata_buffer_table_id.0,
                );

            bind_group
        };

        Ok(&self
            .init_metadata_bind_groups
            .entry(effect_batch.buffer_index)
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
        gpu_limits: &GpuLimits,
        render_device: &RenderDevice,
        layout: &BindGroupLayout,
        effect_metadata_buffer: &Buffer,
        child_info_buffer: Option<&Buffer>,
        event_buffers: &[(Entity, BufferBindingSource)],
    ) -> Result<&BindGroup, ()> {
        let DispatchBufferIndices {
            effect_metadata_buffer_table_id,
            ..
        } = &effect_batch.dispatch_buffer_indices;

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
            buffer_index: effect_batch.buffer_index,
            effect_metadata_buffer: effect_metadata_buffer.id(),
            effect_metadata_offset: gpu_limits
                .effect_metadata_offset(effect_metadata_buffer_table_id.0)
                as u32,
            child_info_buffer_id,
            event_buffers_keys,
        };

        let make_entry = || {
            let mut entries = Vec::with_capacity(2 + event_buffers.len());
            // @group(3) @binding(0) var<storage, read_write> effect_metadata :
            // EffectMetadata;
            entries.push(BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: effect_metadata_buffer,
                    offset: key.effect_metadata_offset as u64,
                    size: Some(gpu_limits.effect_metadata_aligned_size.into()),
                }),
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
                "Created new metadata@3 bind group for update pass and buffer index {}: effect_metadata={}",
                effect_batch.buffer_index,
                effect_metadata_buffer_table_id.0,
            );

            bind_group
        };

        Ok(&self
            .update_metadata_bind_groups
            .entry(effect_batch.buffer_index)
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

    pub fn init_fill_dispatch(&self, event_buffer_index: u32) -> Option<&BindGroup> {
        self.init_fill_dispatch.get(&event_buffer_index)
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
    #[system_param(ignore)]
    marker: PhantomData<&'s usize>,
}

fn emit_sorted_draw<T, F>(
    views: &Query<(Entity, &RenderVisibleEntities, &ExtractedView, &Msaa)>,
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

    for (view_entity, visible_entities, view, msaa) in views.iter() {
        trace!(
            "Process new sorted view with {} visible particle effect entities",
            visible_entities.len::<WithCompiledParticleEffect>()
        );

        let Some(render_phase) = render_phases.get_mut(&view_entity) else {
            continue;
        };

        {
            #[cfg(feature = "trace")]
            let _span = bevy::utils::tracing::info_span!("collect_view_entities").entered();

            view_entities.clear();
            view_entities.extend(
                visible_entities
                    .iter::<WithCompiledParticleEffect>()
                    .map(|e| e.1.index() as usize),
            );
        }

        // For each view, loop over all the effect batches to determine if the effect
        // needs to be rendered for that view, and enqueue a view-dependent
        // batch if so.
        for (draw_entity, draw_batch) in effect_draw_batches.iter() {
            #[cfg(feature = "trace")]
            let _span_draw = bevy::utils::tracing::info_span!("draw_batch").entered();

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
                "-> EffectBach: buffer_index={} spawner_base={} layout_flags={:?}",
                effect_batch.buffer_index,
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
            let _span_check_vis = bevy::utils::tracing::info_span!("check_visibility").entered();
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
            let _span_specialize = bevy::utils::tracing::info_span!("specialize").entered();
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
                "+ Add Transparent for batch on draw_entity {:?}: buffer_index={} \
                spawner_base={} handle={:?}",
                draw_entity,
                effect_batch.buffer_index,
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
fn emit_binned_draw<T, F>(
    views: &Query<(Entity, &RenderVisibleEntities, &ExtractedView, &Msaa)>,
    render_phases: &mut ResMut<ViewBinnedRenderPhases<T>>,
    view_entities: &mut FixedBitSet,
    sorted_effect_batches: &SortedEffectBatches,
    effect_draw_batches: &Query<(Entity, &mut EffectDrawBatch)>,
    render_pipeline: &mut ParticlesRenderPipeline,
    mut specialized_render_pipelines: Mut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    pipeline_cache: &PipelineCache,
    render_meshes: &RenderAssets<RenderMesh>,
    make_bin_key: F,
    #[cfg(all(feature = "2d", feature = "3d"))] pipeline_mode: PipelineMode,
    alpha_mask: ParticleRenderAlphaMaskPipelineKey,
) where
    T: BinnedPhaseItem,
    F: Fn(CachedRenderPipelineId, &EffectDrawBatch, &ExtractedView) -> T::BinKey,
{
    use bevy::render::render_phase::BinnedRenderPhaseType;

    trace!("emit_binned_draw() {} views", views.iter().len());

    for (view_entity, visible_entities, view, msaa) in views.iter() {
        trace!("Process new binned view (alpha_mask={:?})", alpha_mask);

        let Some(render_phase) = render_phases.get_mut(&view_entity) else {
            continue;
        };

        {
            #[cfg(feature = "trace")]
            let _span = bevy::utils::tracing::info_span!("collect_view_entities").entered();

            view_entities.clear();
            view_entities.extend(
                visible_entities
                    .iter::<WithCompiledParticleEffect>()
                    .map(|e| e.1.index() as usize),
            );
        }

        // For each view, loop over all the effect batches to determine if the effect
        // needs to be rendered for that view, and enqueue a view-dependent
        // batch if so.
        for (draw_entity, draw_batch) in effect_draw_batches.iter() {
            #[cfg(feature = "trace")]
            let _span_draw = bevy::utils::tracing::info_span!("draw_batch").entered();

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
                "-> EffectBaches: buffer_index={} spawner_base={} layout_flags={:?}",
                effect_batch.buffer_index,
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
            let _span_check_vis = bevy::utils::tracing::info_span!("check_visibility").entered();
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
            let _span_specialize = bevy::utils::tracing::info_span!("specialize").entered();
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
                "+ Add Transparent for batch on draw_entity {:?}: buffer_index={} \
                spawner_base={} handle={:?}",
                draw_entity,
                effect_batch.buffer_index,
                effect_batch.spawner_base,
                effect_batch.handle
            );
            render_phase.add(
                make_bin_key(render_pipeline_id, draw_batch, view),
                (draw_entity, MainEntity::from(Entity::PLACEHOLDER)),
                BinnedRenderPhaseType::NonMesh,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn queue_effects(
    views: Query<(Entity, &RenderVisibleEntities, &ExtractedView, &Msaa)>,
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
    #[cfg(feature = "3d")] mut alpha_mask_3d_render_phases: ResMut<
        ViewBinnedRenderPhases<AlphaMask3d>,
    >,
) {
    #[cfg(feature = "trace")]
    let _span = bevy::utils::tracing::info_span!("hanabi:queue_effects").entered();

    trace!("queue_effects");

    // If an image has changed, the GpuImage has (probably) changed
    for event in &events.images {
        match event {
            AssetEvent::Added { .. } => None,
            AssetEvent::LoadedWithDependencies { .. } => None,
            AssetEvent::Unused { .. } => None,
            AssetEvent::Modified { id } => {
                trace!("Destroy bind group of modified image asset {:?}", id);
                effect_bind_groups.images.remove(id)
            }
            AssetEvent::Removed { id } => {
                trace!("Destroy bind group of removed image asset {:?}", id);
                effect_bind_groups.images.remove(id)
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
        let _span_draw = bevy::utils::tracing::info_span!("draw_2d").entered();

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
                    extra_index: PhaseItemExtraIndex::NONE,
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
        let _span_draw = bevy::utils::tracing::info_span!("draw_3d").entered();

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
                    draw_function: draw_effects_function_3d,
                    pipeline: id,
                    entity,
                    distance: view
                        .rangefinder3d()
                        .distance_translation(&batch.translation),
                    batch_range: 0..1,
                    extra_index: PhaseItemExtraIndex::NONE,
                },
                #[cfg(feature = "2d")]
                PipelineMode::Camera3d,
            );
        }

        // Effects with alpha mask
        if !views.is_empty() {
            #[cfg(feature = "trace")]
            let _span_draw = bevy::utils::tracing::info_span!("draw_alphamask").entered();

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
                |id, _batch, _view| OpaqueNoLightmap3dBinKey {
                    pipeline: id,
                    draw_function: draw_effects_function_alpha_mask,
                    asset_id: AssetId::<Image>::default().untyped(),
                    material_bind_group_id: None,
                    // },
                    // distance: view
                    //     .rangefinder3d()
                    //     .distance_translation(&batch.translation_3d),
                    // batch_range: 0..1,
                    // extra_index: PhaseItemExtraIndex::NONE,
                },
                #[cfg(feature = "2d")]
                PipelineMode::Camera3d,
                ParticleRenderAlphaMaskPipelineKey::AlphaMask,
            );
        }

        // Opaque particles
        if !views.is_empty() {
            #[cfg(feature = "trace")]
            let _span_draw = bevy::utils::tracing::info_span!("draw_opaque").entered();

            trace!("Emit effect draw calls for opaque 3D views...");

            let draw_effects_function_opaque = read_params
                .draw_functions_opaque
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
                |id, _batch, _view| OpaqueNoLightmap3dBinKey {
                    pipeline: id,
                    draw_function: draw_effects_function_opaque,
                    asset_id: AssetId::<Image>::default().untyped(),
                    material_bind_group_id: None,
                    // },
                    // distance: view
                    //     .rangefinder3d()
                    //     .distance_translation(&batch.translation_3d),
                    // batch_range: 0..1,
                    // extra_index: PhaseItemExtraIndex::NONE,
                },
                #[cfg(feature = "2d")]
                PipelineMode::Camera3d,
                ParticleRenderAlphaMaskPipelineKey::Opaque,
            );
        }
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
) {
    // Get the binding for the ViewUniform, the uniform data structure containing
    // the Camera data for the current view. If not available, we cannot render
    // anything.
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };

    // Create the bind group for the camera/view parameters
    // FIXME - Not here!
    effects_meta.view_bind_group = Some(render_device.create_bind_group(
        "hanabi:bind_group_camera_view",
        &render_pipeline.view_layout,
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

    // Re-/allocate any GPU buffer if needed
    //effect_cache.prepare_buffers(&render_device, &render_queue, &mut
    // effect_bind_groups);
    event_cache.prepare_buffers(&render_device, &render_queue, &mut effect_bind_groups);
    sort_bind_groups.prepare_buffers(&render_device);
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
    dispatch_indirect_pipeline: Res<DispatchIndirectPipeline>,
    utils_pipeline: Res<UtilsPipeline>,
    update_pipeline: Res<ParticlesUpdatePipeline>,
    render_pipeline: ResMut<ParticlesRenderPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut gpu_buffer_operation_queue: ResMut<GpuBufferOperationQueue>,
) {
    // We can't simulate nor render anything without at least the spawner buffer
    if effects_meta.spawner_buffer.is_empty() {
        return;
    }
    let Some(spawner_buffer) = effects_meta.spawner_buffer.buffer().cloned() else {
        return;
    };

    // Ensure child_infos@3 bind group for the indirect pass is available if needed.
    // This returns `None` if the buffer is not ready, either because it's not
    // created yet or because it's not needed (no child effect).
    event_cache.ensure_indirect_child_info_buffer_bind_group(&render_device);

    {
        #[cfg(feature = "trace")]
        let _span = bevy::utils::tracing::info_span!("shared_bind_groups").entered();

        // Make a copy of the buffer IDs before borrowing effects_meta mutably in the
        // loop below. Also allows earlying out before doing any work in case some
        // buffer is missing.
        let Some(spawner_buffer) = effects_meta.spawner_buffer.buffer().cloned() else {
            return;
        };

        // Create the sim_params@0 bind group for the global simulation parameters,
        // which is shared by the init and update passes.
        if effects_meta.indirect_sim_params_bind_group.is_none() {
            effects_meta.indirect_sim_params_bind_group = Some(render_device.create_bind_group(
                "hanabi:bind_group:vfx_indirect:sim_params@0",
                &update_pipeline.sim_params_layout, // FIXME - Shared with init
                &[BindGroupEntry {
                    binding: 0,
                    resource: effects_meta.sim_params_uniforms.binding().unwrap(),
                }],
            ));
        }

        // Create the @1 bind group for the indirect dispatch preparation pass of all
        // effects at once
        effects_meta.indirect_metadata_bind_group = match (
            effects_meta.effect_metadata_buffer.buffer(),
            effects_meta.update_dispatch_indirect_buffer.buffer(),
        ) {
            (Some(effect_metadata_buffer), Some(dispatch_indirect_buffer)) => {
                // Base bind group for indirect pass
                Some(render_device.create_bind_group(
                    "hanabi:bind_group:vfx_indirect:metadata@1",
                    &dispatch_indirect_pipeline.effect_metadata_bind_group_layout,
                    &[
                        // @group(1) @binding(0) var<storage, read_write> effect_metadata_buffer : array<u32>;
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: effect_metadata_buffer,
                                offset: 0,
                                size: None, //NonZeroU64::new(256), // Some(GpuEffectMetadata::min_size()),
                            }),
                        },
                        // @group(1) @binding(1) var<storage, read_write> dispatch_indirect_buffer : array<u32>;
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: dispatch_indirect_buffer,
                                offset: 0,
                                size: None, //NonZeroU64::new(256), // Some(GpuDispatchIndirect::min_size()),
                            }),
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
                &dispatch_indirect_pipeline.spawner_bind_group_layout,
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

    // Create the per-buffer bind groups
    trace!("Create per-buffer bind groups...");
    for (buffer_index, effect_buffer) in effect_cache.buffers().iter().enumerate() {
        #[cfg(feature = "trace")]
        let _span_buffer = bevy::utils::tracing::info_span!("create_buffer_bind_groups").entered();

        let Some(effect_buffer) = effect_buffer else {
            trace!(
                "Effect buffer index #{} has no allocated EffectBuffer, skipped.",
                buffer_index
            );
            continue;
        };

        // Ensure all effects in this batch have a bind group for the entire buffer of
        // the group, since the update phase runs on an entire group/buffer at once,
        // with all the effect instances in it batched together.
        trace!("effect particle buffer_index=#{}", buffer_index);
        effect_bind_groups
            .particle_buffers
            .entry(buffer_index as u32)
            .or_insert_with(|| {
                // Bind group particle@1 for render pass
                trace!("Creating particle@1 bind group for buffer #{buffer_index} in render pass");
                let spawner_min_binding_size = GpuSpawnerParams::aligned_size(
                    render_device.limits().min_storage_buffer_offset_alignment,
                );
                let entries = [
                    // @group(1) @binding(0) var<storage, read> particle_buffer : ParticleBuffer;
                    BindGroupEntry {
                        binding: 0,
                        resource: effect_buffer.max_binding(),
                    },
                    // @group(1) @binding(1) var<storage, read> indirect_buffer : IndirectBuffer;
                    BindGroupEntry {
                        binding: 1,
                        resource: effect_buffer.indirect_index_max_binding(),
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
                    &format!("hanabi:bind_group:render:particles@1:vfx{buffer_index}")[..],
                    effect_buffer.render_particles_buffer_layout(),
                    &entries[..],
                );

                BufferBindGroups { render }
            });
    }

    // Create bind groups for queued GPU buffer operations
    gpu_buffer_operation_queue.create_bind_groups(&render_device, &utils_pipeline);

    // Create the per-event-buffer bind groups
    for (event_buffer_index, event_buffer) in event_cache.buffers().iter().enumerate() {
        if event_buffer.is_none() {
            trace!(
                "Event buffer index #{event_buffer_index} has no allocated EventBuffer, skipped.",
            );
            continue;
        }
        let event_buffer_index = event_buffer_index as u32;

        // Check if the entry is missing
        let entry = effect_bind_groups
            .init_fill_dispatch
            .entry(event_buffer_index);
        if matches!(entry, Entry::Vacant(_)) {
            trace!(
                "Event buffer #{} missing a bind group @0 for init fill args. Trying to create now...",
                event_buffer_index
            );

            // Check if the binding is available to create the bind group and fill the entry
            let Some((args_binding, args_count)) =
                gpu_buffer_operation_queue.init_args_buffer_binding(event_buffer_index)
            else {
                continue;
            };

            let Some(source_binding_resource) = event_cache.child_infos().max_binding() else {
                warn!("Event buffer #{event_buffer_index} has {args_count} operations pending, but the effect cache has no child_infos binding for the source buffer. Discarding event operations for this frame. This will result in particles not spawning.");
                continue;
            };

            let Some(target_binding_resource) =
                event_cache.init_indirect_dispatch_binding_resource()
            else {
                warn!("Event buffer #{event_buffer_index} has {args_count} operations pending, but the effect cache has no init_indirect_dispatch_binding_resource for the target buffer. Discarding event operations for this frame. This will result in particles not spawning.");
                continue;
            };

            // Actually create the new bind group entry
            entry.insert(render_device.create_bind_group(
                &format!("hanabi:bind_group:init_fill_dispatch@0:event{event_buffer_index}")[..],
                &utils_pipeline.bind_group_layout,
                &[
                    // @group(0) @binding(0) var<uniform> args : BufferOperationArgs
                    BindGroupEntry {
                        binding: 0,
                        resource: args_binding,
                    },
                    // @group(0) @binding(1) var<storage, read> src_buffer : array<u32>
                    BindGroupEntry {
                        binding: 1,
                        resource: source_binding_resource,
                    },
                    // @group(0) @binding(2) var<storage, read_write> dst_buffer :
                    // array<u32>
                    BindGroupEntry {
                        binding: 2,
                        resource: target_binding_resource,
                    },
                ],
            ));
            trace!(
                "Created new bind group for init fill args of event buffer #{}",
                event_buffer_index
            );
        }
    }

    // Create the per-effect bind groups
    let spawner_buffer_binding_size =
        NonZeroU64::new(effects_meta.spawner_buffer.aligned_size() as u64).unwrap();
    for effect_batch in sorted_effect_batched.iter() {
        #[cfg(feature = "trace")]
        let _span_buffer = bevy::utils::tracing::info_span!("create_batch_bind_groups").entered();

        // Create the property bind group @2 if needed
        if let Some(property_key) = &effect_batch.property_key {
            if let Err(err) = property_bind_groups.ensure_exists(
                property_key,
                &property_cache,
                &spawner_buffer,
                spawner_buffer_binding_size,
                &render_device,
            ) {
                error!("Failed to create property bind group for effect batch: {err:?}");
                continue;
            }
        } else if let Err(err) = property_bind_groups.ensure_exists_no_property(
            &property_cache,
            &spawner_buffer,
            spawner_buffer_binding_size,
            &render_device,
        ) {
            error!("Failed to create property bind group for effect batch: {err:?}");
            continue;
        }

        // Bind group particle@1 for the simulate compute shaders (init and udpate) to
        // simulate particles.
        if effect_cache
            .create_particle_sim_bind_group(
                effect_batch.buffer_index,
                &render_device,
                effect_batch.particle_layout.min_binding_size32(),
                effect_batch.parent_min_binding_size,
                effect_batch.parent_binding_source.as_ref(),
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
            let Some(init_metadata_layout) =
                effect_cache.metadata_init_bind_group_layout(consume_gpu_spawn_events)
            else {
                continue;
            };
            if effect_bind_groups
                .get_or_create_init_metadata(
                    effect_batch,
                    &effects_meta.gpu_limits,
                    &render_device,
                    init_metadata_layout,
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

            let Some(update_metadata_layout) =
                effect_cache.metadata_update_bind_group_layout(num_event_buffers)
            else {
                continue;
            };
            if effect_bind_groups
                .get_or_create_update_metadata(
                    effect_batch,
                    &effects_meta.gpu_limits,
                    &render_device,
                    update_metadata_layout,
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
            let effect_buffer = effect_cache.get_buffer(effect_batch.buffer_index).unwrap();

            // Bind group @0 of sort-fill pass
            let particle_buffer = effect_buffer.particle_buffer();
            let indirect_index_buffer = effect_buffer.indirect_index_buffer();
            let effect_metadata_buffer = effects_meta.effect_metadata_buffer.buffer().unwrap();
            if let Err(err) = sort_bind_groups.ensure_sort_fill_bind_group(
                &effect_batch.particle_layout,
                particle_buffer,
                indirect_index_buffer,
                effect_metadata_buffer,
            ) {
                error!(
                    "Failed to create sort-fill bind group @0 for ribbon effect: {:?}",
                    err
                );
                continue;
            }

            // Bind group @0 of sort-copy pass
            let indirect_index_buffer = effect_buffer.indirect_index_buffer();
            if let Err(err) = sort_bind_groups
                .ensure_sort_copy_bind_group(indirect_index_buffer, effect_metadata_buffer)
            {
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
            let Some(material_bind_group_layout) =
                render_pipeline.get_material(&effect_batch.texture_layout)
            else {
                error!(
                    "Failed to find material bind group layout for buffer #{}",
                    effect_batch.buffer_index
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
                        material_bind_group_layout,
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

    let gpu_limits = &effects_meta.gpu_limits;

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
    assert_eq!(effect_batch.mesh_buffer_id, vertex_buffer_slice.buffer.id());
    assert_eq!(effect_batch.mesh_slice, vertex_buffer_slice.range);
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
            .particle_render(effect_batch.buffer_index)
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
                "Particle material bind group not available for batch buf={}. Skipping draw call.",
                effect_batch.buffer_index,
            );
            return;
        }
    }

    let effect_metadata_index = effect_batch
        .dispatch_buffer_indices
        .effect_metadata_buffer_table_id
        .0;
    let effect_metadata_offset =
        effect_metadata_index as u64 * gpu_limits.effect_metadata_aligned_size.get() as u64;
    trace!(
        "Draw up to {} particles with {} vertices per particle for batch from buffer #{} \
            (effect_metadata_index={}, offset={}B).",
        effect_batch.slice.len(),
        render_mesh.vertex_count,
        effect_batch.buffer_index,
        effect_metadata_index,
        effect_metadata_offset,
    );

    // Note: the indirect draw args are the first few fields of GpuEffectMetadata
    let Some(indirect_buffer) = effects_meta.effect_metadata_buffer.buffer() else {
        trace!(
            "The metadata buffer containing the indirect draw args is not ready for batch buf=#{}. Skipping draw call.",
            effect_batch.buffer_index,
        );
        return;
    };

    match render_mesh.buffer_info {
        RenderMeshBufferInfo::Indexed { index_format, .. } => {
            let Some(index_buffer_slice) = mesh_allocator.mesh_index_slice(&effect_batch.mesh)
            else {
                return;
            };

            pass.set_index_buffer(index_buffer_slice.buffer.slice(..), 0, index_format);
            pass.draw_indexed_indirect(indirect_buffer, effect_metadata_offset);
        }
        RenderMeshBufferInfo::NonIndexed => {
            pass.draw_indirect(indirect_buffer, effect_metadata_offset);
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
            item.key.pipeline,
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
            item.key.pipeline,
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
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HanabiPipelineId {
    Invalid,
    Cached(CachedComputePipelineId),
}

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
        trace!("set_cached_compute_pipeline() id={pipeline_id:?}");
        if HanabiPipelineId::Cached(pipeline_id) == self.pipeline_id {
            trace!("-> already set; skipped");
            return Ok(());
        }
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
        let gpu_buffer_operation_queue = world.resource::<GpuBufferOperationQueue>();
        let sorted_effect_batches = world.resource::<SortedEffectBatches>();

        // Make sure to schedule any buffer copy before accessing their content later in
        // the GPU commands below.
        {
            let command_encoder = render_context.command_encoder();
            effects_meta
                .update_dispatch_indirect_buffer
                .write_buffer(command_encoder);
            effects_meta
                .effect_metadata_buffer
                .write_buffer(command_encoder);
            sort_bind_groups.write_buffers(command_encoder);
        }

        // Compute init fill dispatch pass - Fill the indirect dispatch structs for any
        // upcoming init pass of this frame, based on the GPU spawn events emitted by
        // the update pass of their parent effect during the previous frame.
        gpu_buffer_operation_queue.dispatch_init_fill(
            render_context,
            utils_pipeline.get_pipeline(GpuBufferOperationType::InitFillDispatchArgs),
            effect_bind_groups,
        );

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
                    effect_cache.particle_sim_bind_group(effect_batch.buffer_index)
                else {
                    error!(
                        "Failed to find init particle@1 bind group for buffer index {}",
                        effect_batch.buffer_index
                    );
                    continue;
                };

                // Fetch bind group metadata@3
                let Some(metadata_bind_group) = effect_bind_groups
                    .init_metadata_bind_groups
                    .get(&effect_batch.buffer_index)
                else {
                    error!(
                        "Failed to find init metadata@3 bind group for buffer index {}",
                        effect_batch.buffer_index
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
                let spawner_index = effect_batch.spawner_base;
                let spawner_aligned_size = effects_meta.spawner_buffer.aligned_size();
                assert!(spawner_aligned_size >= GpuSpawnerParams::min_size().get() as usize);
                let spawner_offset = spawner_index * spawner_aligned_size as u32;
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
                        assert_eq!(GpuDispatchIndirect::min_size().get(), 12);
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
                            spawner_index,
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
                            spawner_index,
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
        if effects_meta.spawner_buffer.buffer().is_some()
            && !effects_meta.spawner_buffer.is_empty()
            && effects_meta.indirect_metadata_bind_group.is_some()
            && effects_meta.indirect_sim_params_bind_group.is_some()
        {
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
                    render_context
                        .command_encoder()
                        .insert_debug_marker("ERROR:MissingIndirectBindGroup3");
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
            compute_pass.set_bind_group(
                0,
                effects_meta
                    .indirect_sim_params_bind_group
                    .as_ref()
                    .unwrap(),
                &[],
            );
            compute_pass.set_bind_group(
                1,
                // FIXME - got some unwrap() panic here, investigate... possibly race
                // condition!
                effects_meta.indirect_metadata_bind_group.as_ref().unwrap(),
                &[],
            );
            compute_pass.set_bind_group(
                2,
                effects_meta.indirect_spawner_bind_group.as_ref().unwrap(),
                &[],
            );
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            trace!(
                "indirect dispatch compute dispatched: total_effect_count={} workgroup_count={}",
                total_effect_count,
                workgroup_count
            );
        }

        // Compute update pass
        {
            let Some(indirect_buffer) = effects_meta.update_dispatch_indirect_buffer.buffer()
            else {
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
                effects_meta
                    .indirect_sim_params_bind_group
                    .as_ref()
                    .unwrap(),
                &[],
            );

            // Dispatch update compute jobs
            for effect_batch in sorted_effect_batches.iter() {
                // Fetch bind group particle@1
                let Some(particle_bind_group) =
                    effect_cache.particle_sim_bind_group(effect_batch.buffer_index)
                else {
                    error!(
                        "Failed to find update particle@1 bind group for buffer index {}",
                        effect_batch.buffer_index
                    );
                    continue;
                };

                // Fetch bind group metadata@3
                let Some(metadata_bind_group) = effect_bind_groups
                    .update_metadata_bind_groups
                    .get(&effect_batch.buffer_index)
                else {
                    error!(
                        "Failed to find update metadata@3 bind group for buffer index {}",
                        effect_batch.buffer_index
                    );
                    continue;
                };

                // Fetch compute pipeline
                if compute_pass
                    .set_cached_compute_pipeline(effect_batch.init_and_update_pipeline_ids.update)
                    .is_err()
                {
                    continue;
                }

                // Compute dynamic offsets
                let spawner_index = effect_batch.spawner_base;
                let spawner_aligned_size = effects_meta.spawner_buffer.aligned_size();
                assert!(spawner_aligned_size >= GpuSpawnerParams::min_size().get() as usize);
                let spawner_offset = spawner_index * spawner_aligned_size as u32;
                let property_offset = effect_batch.property_offset;

                trace!(
                    "record commands for update pipeline of effect {:?} spawner_base={}",
                    effect_batch.handle,
                    spawner_index,
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
                let dispatch_indirect_buffer_table_id = effect_batch
                    .dispatch_buffer_indices
                    .update_dispatch_indirect_buffer_table_id;
                let dispatch_indirect_offset = dispatch_indirect_buffer_table_id.0 * 12;
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
        gpu_buffer_operation_queue.dispatch_fill(
            render_context,
            utils_pipeline.get_pipeline(GpuBufferOperationType::FillDispatchArgs),
        );

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

                let Some(effect_buffer) = effect_cache.get_buffer(effect_batch.buffer_index) else {
                    warn!("Missing sort-fill effect buffer.");
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
                        continue;
                    };
                    if compute_pass
                        .set_cached_compute_pipeline(pipeline_id)
                        .is_err()
                    {
                        compute_pass.pop_debug_group();
                        // FIXME - Bevy doesn't allow returning custom errors here...
                        return Ok(());
                    }

                    // Bind group sort_fill@0
                    let particle_buffer = effect_buffer.particle_buffer();
                    let indirect_index_buffer = effect_buffer.indirect_index_buffer();
                    let Some(bind_group) = sort_bind_groups.sort_fill_bind_group(
                        particle_buffer.id(),
                        indirect_index_buffer.id(),
                        effect_metadata_buffer.id(),
                    ) else {
                        warn!("Missing sort-fill bind group.");
                        continue;
                    };
                    let particle_offset = effect_buffer.particle_offset(effect_batch.slice.start);
                    let indirect_index_offset =
                        effect_buffer.indirect_index_offset(effect_batch.slice.start);
                    let effect_metadata_offset = effects_meta.gpu_limits.effect_metadata_offset(
                        effect_batch
                            .dispatch_buffer_indices
                            .effect_metadata_buffer_table_id
                            .0,
                    ) as u32;
                    compute_pass.set_bind_group(
                        0,
                        bind_group,
                        &[
                            particle_offset,
                            indirect_index_offset,
                            effect_metadata_offset,
                        ],
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
                        compute_pass.pop_debug_group();
                        // FIXME - Bevy doesn't allow returning custom errors here...
                        return Ok(());
                    }

                    compute_pass.set_bind_group(0, sort_bind_groups.sort_bind_group(), &[]);
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
                        compute_pass.pop_debug_group();
                        // FIXME - Bevy doesn't allow returning custom errors here...
                        return Ok(());
                    }

                    // Bind group sort_copy@0
                    let indirect_index_buffer = effect_buffer.indirect_index_buffer();
                    let Some(bind_group) = sort_bind_groups.sort_copy_bind_group(
                        indirect_index_buffer.id(),
                        effect_metadata_buffer.id(),
                    ) else {
                        warn!("Missing sort-copy bind group.");
                        continue;
                    };
                    let indirect_index_offset = effect_batch.slice.start;
                    let effect_metadata_offset =
                        effects_meta.effect_metadata_buffer.dynamic_offset(
                            effect_batch
                                .dispatch_buffer_indices
                                .effect_metadata_buffer_table_id,
                        );
                    compute_pass.set_bind_group(
                        0,
                        bind_group,
                        &[indirect_index_offset, effect_metadata_offset],
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
    fn gpu_ops_queue() {
        use crate::test_utils::MockRenderer;

        let renderer = MockRenderer::new();
        let device = renderer.device();
        let render_queue = renderer.queue();

        let mut world = World::new();
        world.insert_resource(device.clone());
        let mut queue = GpuBufferOperationQueue::from_world(&mut world);

        // Two consecutive ops can be merged if in order. This includes having
        // contiguous slices both in source and destination.
        queue.begin_frame();
        queue.enqueue_init_fill(
            0,
            0..200,
            GpuBufferOperationArgs {
                src_offset: 0,
                src_stride: 2,
                dst_offset: 0,
                dst_stride: 0,
                count: 1,
            },
        );
        queue.enqueue_init_fill(
            0,
            200..300,
            GpuBufferOperationArgs {
                src_offset: 1,
                src_stride: 2,
                dst_offset: 1,
                dst_stride: 0,
                count: 1,
            },
        );
        queue.end_frame(&device, &render_queue);
        assert_eq!(queue.init_fill_dispatch_args.len(), 1);
        assert_eq!(queue.args_buffer.content().len(), 1);

        // However if out of order, they remain distinct. Here the source offsets are
        // inverted.
        queue.begin_frame();
        queue.enqueue_init_fill(
            0,
            0..200,
            GpuBufferOperationArgs {
                src_offset: 1,
                src_stride: 2,
                dst_offset: 0,
                dst_stride: 0,
                count: 1,
            },
        );
        queue.enqueue_init_fill(
            0,
            200..300,
            GpuBufferOperationArgs {
                src_offset: 0,
                src_stride: 2,
                dst_offset: 1,
                dst_stride: 0,
                count: 1,
            },
        );
        queue.end_frame(&device, &render_queue);
        assert_eq!(queue.init_fill_dispatch_args.len(), 2);
        assert_eq!(queue.args_buffer.content().len(), 2);
    }
}
