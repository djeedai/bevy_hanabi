use std::{
    borrow::Cow,
    hash::{DefaultHasher, Hash, Hasher},
    num::{NonZero, NonZeroU32, NonZeroU64},
    ops::Deref,
    time::Duration,
    u32,
};
use std::{iter, marker::PhantomData};

use batch::InitAndUpdatePipelineIds;
#[cfg(feature = "2d")]
use bevy::core_pipeline::core_2d::Transparent2d;
#[cfg(feature = "2d")]
use bevy::math::FloatOrd;
#[cfg(feature = "3d")]
use bevy::{
    core_pipeline::{
        core_3d::{AlphaMask3d, Transparent3d},
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
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo},
        render_phase::{
            Draw, DrawFunctions, PhaseItemExtraIndex, SortedPhaseItem, TrackedRenderPass,
            ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::{BevyDefault, GpuImage},
        view::{
            ExtractedView, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms,
            VisibleEntities,
        },
        Extract,
    },
    utils::{hashbrown::hash_map::Entry, HashMap},
};
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use effect_cache::{EffectBuffer, TrailDispatchBufferIndices};
use fixedbitset::FixedBitSet;
use naga_oil::compose::{Composer, NagaModuleDescriptor};

use crate::{
    asset::EffectAsset,
    next_multiple_of,
    plugin::WithCompiledParticleEffect,
    render::{
        batch::{BatchesInput, EffectDrawBatch},
        effect_cache::DispatchBufferIndices,
    },
    spawn::{EffectCloner, EffectInitializer, EffectInitializers, Initializer},
    AlphaMode, Attribute, CompiledParticleEffect, EffectProperties, EffectShader, EffectSimulation,
    HanabiPlugin, ParticleLayout, PropertyLayout, RemovedEffectsEvent, SimulationCondition,
    TextureLayout, ToWgslString,
};

mod aligned_buffer_vec;
mod batch;
mod buffer_table;
mod effect_cache;
mod gpu_buffer;
mod shader_cache;

use aligned_buffer_vec::AlignedBufferVec;
use buffer_table::{BufferTable, BufferTableId};
pub(crate) use effect_cache::{EffectCache, EffectCacheId};
pub use shader_cache::ShaderCache;

use self::batch::EffectBatches;

// Size of an indirect index (including both parts of the ping-pong buffer) in
// bytes.
const INDIRECT_INDEX_SIZE: u32 = 12;

fn calc_hash<H: Hash>(value: &H) -> u64 {
    let mut hasher = DefaultHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
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
    /// Total number of groups to simulate this frame. Used by the indirect
    /// compute pipeline to cap the compute thread to the actual number of
    /// groups to process.
    ///
    /// This is only used by the `vfx_indirect` compute shader.
    num_groups: u32,
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
            num_groups: 0,
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
    pub x_row: Vec4,
    pub y_row: Vec4,
    pub z_row: Vec4,
}

impl From<Mat4> for GpuCompressedTransform {
    fn from(value: Mat4) -> Self {
        let tr = value.transpose();
        #[cfg(test)]
        crate::test_utils::assert_approx_eq!(tr.w_axis, Vec4::W);
        Self {
            x_row: tr.x_axis,
            y_row: tr.y_axis,
            z_row: tr.z_axis,
        }
    }
}

impl From<&Mat4> for GpuCompressedTransform {
    fn from(value: &Mat4) -> Self {
        let tr = value.transpose();
        #[cfg(test)]
        crate::test_utils::assert_approx_eq!(tr.w_axis, Vec4::W);
        Self {
            x_row: tr.x_axis,
            y_row: tr.y_axis,
            z_row: tr.z_axis,
        }
    }
}

impl GpuCompressedTransform {
    /// Returns the translation as represented by this transform.
    #[allow(dead_code)]
    pub fn translation(&self) -> Vec3 {
        Vec3 {
            x: self.x_row.w,
            y: self.y_row.w,
            z: self.z_row.w,
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
        NonZeroU64::new(next_multiple_of(T::min_size().get() as usize, alignment as usize) as u64)
            .unwrap()
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
    /// space, or to all simulated particles if the effect is simulated in
    /// local space.
    transform: GpuCompressedTransform,
    /// Inverse of [`transform`], stored with the same convention.
    ///
    /// [`transform`]: crate::render::GpuSpawnerParams::transform
    inverse_transform: GpuCompressedTransform,
    /// Number of particles to spawn this frame.
    spawn: i32,
    /// Spawn seed, for randomized modifiers.
    seed: u32,
    /// Current number of used particles.
    count: i32,
    /// Index of the effect in the indirect dispatch and render buffers.
    effect_index: u32,
    /// The time in seconds that the cloned particles live, if this is a cloner.
    ///
    /// If this is a spawner, this value is zero.
    lifetime: f32,
    /// Padding.
    pad: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct GpuDispatchIndirect {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub pong: u32,
}

impl Default for GpuDispatchIndirect {
    fn default() -> Self {
        Self {
            x: 0,
            y: 1,
            z: 1,
            pong: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct GpuInitDispatchIndirect {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Default for GpuInitDispatchIndirect {
    fn default() -> Self {
        Self { x: 0, y: 1, z: 1 }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct GpuChildInfo {
    pub init_indirect_dispatch_index: u32,
    pub event_count: i32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct GpuRenderEffectMetadata {
    pub ping: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct GpuRenderGroupIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub vertex_offset: i32,
    pub base_instance: u32,
    //
    pub alive_count: u32,
    pub max_update: u32,
    pub dead_count: u32,
    pub max_spawn: u32,
}

/// Stores metadata about each particle group.
///
/// This is written by the CPU and read by the GPU.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct GpuParticleGroup {
    /// The index of this particle group in the global particle group buffer.
    pub global_group_index: u32,
    /// The global index of the entire particle effect.
    pub effect_index: u32,
    /// The index of this effect in the group.
    ///
    /// For example, the first group in an effect has index 0, the second has
    /// index 1, etc.
    pub group_index_in_effect: u32,
    /// The index of the first particle in this group in the indirect index
    /// buffer.
    pub indirect_index: u32,
    /// The capacity of this group in number of particles.
    pub capacity: u32,
    /// The index of the first particle in this effect in the particle and
    /// indirect buffers.
    pub effect_particle_offset: u32,
    /// Offset (in u32 count) of the event indirect dispatch struct inside its
    /// buffer. This avoids having to align those 16-byte structs to the GPU
    /// alignment (at least 32 bytes, even 256 bytes on some).
    pub init_indirect_dispatch_index: u32,
    /// Index of this effect as a child of its parent. This indexes into the
    /// parent's array of [`GpuChildInfo`]s.
    pub child_index: u32,
}

/// Compute pipeline to run the `vfx_indirect` dispatch workgroup calculation
/// shader.
#[derive(Resource)]
pub(crate) struct DispatchIndirectPipeline {
    dispatch_indirect_layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl FromWorld for DispatchIndirectPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let storage_alignment = render_device.limits().min_storage_buffer_offset_alignment;
        let render_effect_indirect_size = GpuRenderEffectMetadata::aligned_size(storage_alignment);
        let render_group_indirect_size = GpuRenderGroupIndirect::aligned_size(storage_alignment);
        let dispatch_indirect_size = GpuDispatchIndirect::aligned_size(storage_alignment);
        let particle_group_size = GpuParticleGroup::aligned_size(storage_alignment);

        trace!(
            "GpuRenderEffectMetadata: min_size={} padded_size={} | GpuRenderGroupIndirect: min_size={} padded_size={} | \
            GpuDispatchIndirect: min_size={} padded_size={} | GpuParticleGroup: min_size={} padded_size={}",
            GpuRenderEffectMetadata::min_size(),
            render_effect_indirect_size,
            GpuRenderGroupIndirect::min_size(),
            render_group_indirect_size,
            GpuDispatchIndirect::min_size(),
            dispatch_indirect_size,
            GpuParticleGroup::min_size(),
            particle_group_size
        );
        let dispatch_indirect_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:dispatch_indirect_dispatch_indirect",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(render_effect_indirect_size),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(render_group_indirect_size),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(dispatch_indirect_size),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(particle_group_size),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuSpawnerParams::min_size()),
                    },
                    count: None,
                },
            ],
        );

        trace!("GpuSimParams: min_size={}", GpuSimParams::min_size());
        let sim_params_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:dispatch_indirect_sim_params",
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

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("hanabi:pipeline_layout:dispatch_indirect"),
            bind_group_layouts: &[&dispatch_indirect_layout, &sim_params_layout],
            push_constant_ranges: &[],
        });

        let render_effect_indirect_stride_code =
            (render_effect_indirect_size.get() as u32).to_wgsl_string();
        let render_group_indirect_stride_code =
            (render_group_indirect_size.get() as u32).to_wgsl_string();
        let indirect_code = include_str!("vfx_indirect.wgsl")
            .replace(
                "{{RENDER_EFFECT_INDIRECT_STRIDE}}",
                &render_effect_indirect_stride_code,
            )
            .replace(
                "{{RENDER_GROUP_INDIRECT_STRIDE}}",
                &render_group_indirect_stride_code,
            );

        // Resolve imports. Because we don't insert this shader into Bevy' pipeline
        // cache, we don't get that part "for free", so we have to do it manually here.
        let indirect_naga_module = {
            let mut composer = Composer::default();

            // Import bevy_hanabi::vfx_common
            {
                let common_shader = HanabiPlugin::make_common_shader(
                    render_device.limits().min_storage_buffer_offset_alignment,
                );
                let mut desc: naga_oil::compose::ComposableModuleDescriptor<'_> =
                    (&common_shader).into();
                desc.shader_defs.insert(
                    "SPAWNER_PADDING".to_string(),
                    naga_oil::compose::ShaderDefValue::Bool(true),
                );
                let res = composer.add_composable_module(desc);
                assert!(res.is_ok());
            }

            let shader_defs = default();

            match composer.make_naga_module(NagaModuleDescriptor {
                source: &indirect_code,
                file_path: "vfx_indirect.wgsl",
                shader_defs,
                ..Default::default()
            }) {
                Ok(naga_module) => ShaderSource::Naga(Cow::Owned(naga_module)),
                Err(compose_error) => panic!(
                    "Failed to compose vfx_indirect.wgsl, naga_oil returned: {}",
                    compose_error.emit_to_string(&composer)
                ),
            }
        };

        debug!("Create indirect dispatch shader:\n{}", indirect_code);

        let shader_module = render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("hanabi:vfx_indirect_shader"),
            source: indirect_naga_module,
        });

        let pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("hanabi:compute_pipeline:dispatch_indirect"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: default(),
        });

        Self {
            dispatch_indirect_layout,
            pipeline,
        }
    }
}

/// Type of GPU buffer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuBufferOperationType {
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
    /// Number of u32 elements to process for this operation.
    count: u32,
}

struct InitFillDispatchArgs {
    event_buffer_index: u32,
    event_slice: std::ops::Range<u32>,
    args_index: u32,
}

struct InitFillDispatchArgsSlice {
    event_buffer_index: u32,
    args_offset: u32,
    args_count: u32,
}

struct InitFillDispatchArgsSliceIter<'a> {
    args: &'a [InitFillDispatchArgs],
    event_buffer_index: u32,
    args_start: u32,
}

impl<'a> InitFillDispatchArgsSliceIter<'a> {
    pub fn new(args: &'a [InitFillDispatchArgs]) -> Self {
        Self {
            args,
            event_buffer_index: 0,
            args_start: 0,
        }
    }
}

impl<'a> Iterator for InitFillDispatchArgsSliceIter<'a> {
    type Item = InitFillDispatchArgsSlice;

    fn next(&mut self) -> Option<Self::Item> {
        let len = self.args.len() as u32;
        let mut idx = self.args_start;
        while idx < len {
            let cur_buffer_index = self.args[idx as usize].event_buffer_index;

            // Check if next item is for a new slice
            if idx == 0 {
                self.event_buffer_index = cur_buffer_index;
            } else if cur_buffer_index != self.event_buffer_index {
                let item = Self::Item {
                    event_buffer_index: self.event_buffer_index,
                    args_offset: self.args_start,
                    args_count: idx - self.args_start,
                };

                // Update the iterator for the next slice
                self.event_buffer_index = cur_buffer_index;
                self.args_start = idx;

                return Some(item);
            }

            // Move to next args item
            idx += 1;
        }

        // Last slice, if not empty
        if self.args_start < idx {
            let item = Self::Item {
                event_buffer_index: self.event_buffer_index,
                args_offset: self.args_start,
                args_count: idx - self.args_start,
            };

            // Update the iterator for the next slice
            self.args_start = idx;

            return Some(item);
        }

        None
    }
}

/// Queue of GPU buffer operations for this frame.
#[derive(Resource)]
pub(super) struct GpuBufferOperationQueue {
    /// Arguments for the buffer operations submitted this frame.
    args_buffer: AlignedBufferVec<GpuBufferOperationArgs>,

    /// Unsorted temporary storage for this-frame operations, which will be
    /// written to [`args_buffer`] at the end of the frame after being sorted.
    args_buffer_unsorted: Vec<GpuBufferOperationArgs>,

    /// Indices into the arguments buffer of the queued dispatch fill args
    /// operations for the indirect init pass of next frame. Those
    /// operations are dispatched at the end of the current frame.
    init_fill_dispatch: Vec<InitFillDispatchArgs>,
}

impl FromWorld for GpuBufferOperationQueue {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let align = render_device.limits().min_uniform_buffer_offset_alignment;
        let args_buffer = AlignedBufferVec::new(
            BufferUsages::UNIFORM,
            Some(NonZeroU64::new(align as u64).unwrap()),
            Some("hanabi:buffer:gpu_operation_args".to_string()),
        );
        Self {
            args_buffer,
            args_buffer_unsorted: vec![],
            init_fill_dispatch: vec![],
        }
    }
}

impl GpuBufferOperationQueue {
    /// Get a binding for all the entries of the arguments buffer associated
    /// with the given event buffer.
    pub fn args_buffer_binding(&self, event_buffer_index: u32) -> Option<(BindingResource, u32)> {
        // Find the slice corresponding to this event buffer. The entries are sorted by
        // event buffer index, so the list of entries is a contiguous slice inside the
        // overall buffer.
        let Some(start) = self
            .init_fill_dispatch
            .iter()
            .position(|ifda| ifda.event_buffer_index == event_buffer_index)
        else {
            trace!("Event buffer #{event_buffer_index} has no allocated operation.");
            return None;
        };
        let end = if let Some(end) = self
            .init_fill_dispatch
            .iter()
            .skip(start)
            .position(|ifda| ifda.event_buffer_index != event_buffer_index)
        {
            end
        } else {
            self.init_fill_dispatch.len()
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
        self.init_fill_dispatch.clear();
    }

    /// Queue a new [`GpuBufferOperationType::FillDispatchArgs`] operation.
    pub fn queue_init_fill(
        &mut self,
        event_buffer_index: u32,
        event_slice: std::ops::Range<u32>,
        args: GpuBufferOperationArgs,
    ) {
        trace!(
            "Queue FillDispatchArgs op: ev_buffer#{} ev_slice={:?} args={:?}",
            event_buffer_index,
            event_slice,
            args
        );
        let args_index = self.args_buffer_unsorted.len() as u32;
        self.args_buffer_unsorted.push(args);
        self.init_fill_dispatch.push(InitFillDispatchArgs {
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
            self.init_fill_dispatch.len()
        );
        assert!(self.args_buffer.is_empty());

        if self.init_fill_dispatch.is_empty() {
            self.args_buffer.set_content(vec![]);
        } else {
            // Sort the commands by buffer, so we can dispatch them in groups with a single
            // dispatch per buffer
            trace!(
                "Sorting {} InitFillDispatch ops...",
                self.init_fill_dispatch.len()
            );
            self.init_fill_dispatch.sort_unstable_by(|ifda1, ifda2| {
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

            // Push entries into the final storage before GPU upload. It's a bit unfortunate
            // we have to make copies, but those arrays should be small.
            let mut sorted_args = Vec::with_capacity(self.init_fill_dispatch.len());
            let mut prev_buffer = u32::MAX;
            for ifda in &self.init_fill_dispatch {
                if !sorted_args.is_empty() && (prev_buffer == ifda.event_buffer_index) {
                    prev_buffer = ifda.event_buffer_index;
                    let prev_idx = sorted_args.len() - 1;
                    let prev: &mut GpuBufferOperationArgs = &mut sorted_args[prev_idx];
                    let cur = &self.args_buffer_unsorted[ifda.args_index as usize];
                    if prev.src_stride == cur.src_stride
                    // at this point src_offset == child_index, and we want them to be contiguous in the source buffer so that we can increment by src_stride
                    && cur.src_offset == prev.src_offset + 1
                    {
                        prev.count += 1;
                        continue;
                    }
                }
                prev_buffer = ifda.event_buffer_index;
                sorted_args.push(self.args_buffer_unsorted[ifda.args_index as usize]);
            }

            self.args_buffer.set_content(sorted_args);
        }

        // Write entries to GPU
        self.args_buffer.write_buffer(device, render_queue);
    }

    /// Returns (event_buffer_index, args_offset, args_count) for each slice of
    /// operations on each buffer.
    fn init_fill_dispatch_slices<'a>(&'a self) -> InitFillDispatchArgsSliceIter<'a> {
        // Assume this is called after end_frame() so we're sorted already
        InitFillDispatchArgsSliceIter::new(&self.init_fill_dispatch[..])
    }

    /// Dispatch any pending [`GpuBufferOperationType::FillDispatchArgs`]
    /// operation for indirect init passes, and clear those pending operations.
    pub fn dispatch_init_fill(
        &self,
        render_context: &mut RenderContext,
        pipeline: &ComputePipeline,
        bind_groups: &EffectBindGroups,
    ) {
        if self.init_fill_dispatch.is_empty() {
            return;
        }

        trace!(
            "Recording GPU commands for the fill dispatch pipeline... {:?}",
            pipeline
        );

        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("hanabi:fill_dispatch"),
                    timestamp_writes: None,
                });

        compute_pass.set_pipeline(pipeline);

        for ifda_slice in self.init_fill_dispatch_slices() {
            trace!(
                "event_buffer_index={} args_offset={:?} args_count={}",
                ifda_slice.event_buffer_index,
                ifda_slice.args_offset,
                ifda_slice.args_count
            );
            if let Some(bind_group) = bind_groups.init_fill_dispatch(ifda_slice.event_buffer_index)
            {
                let dst_offset = self
                    .args_buffer
                    .dynamic_offset(ifda_slice.args_offset as usize);
                compute_pass.set_bind_group(0, bind_group, &[]);
                trace!(
                    "found bind group for event buffer index #{} with dst_offset +{}B",
                    ifda_slice.event_buffer_index,
                    dst_offset
                );
            } else {
                warn!(
                    "bind group not found for event buffer index #{}",
                    ifda_slice.event_buffer_index
                );
                continue;
            }

            // Dispatch the operations for this buffer
            const WORKGROUP_SIZE: u32 = 64;
            let num_ops = 1;
            let workgroup_count = (num_ops + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
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
                source: &shader_code,
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
        let zero_pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("hanabi:compute_pipeline:zero_buffer"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "zero_buffer",
            compilation_options: default(),
        });
        let copy_pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("hanabi:compute_pipeline:copy_buffer"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "copy_buffer",
            compilation_options: default(),
        });
        let fill_dispatch_args_pipeline =
            render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some("hanabi:compute_pipeline:fill_dispatch_args"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "fill_dispatch_args",
                compilation_options: default(),
            });
        let fill_dispatch_args_self_pipeline =
            render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some("hanabi:compute_pipeline:fill_dispatch_args_self"),
                layout: Some(&pipeline_layout_no_src),
                module: &shader_module,
                entry_point: "fill_dispatch_args_self",
                compilation_options: default(),
            });

        Self {
            bind_group_layout,
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
}

#[derive(Resource)]
pub(crate) struct ParticlesInitPipeline {
    render_device: RenderDevice,
    sim_params_layout: BindGroupLayout,
    spawner_buffer_layout: BindGroupLayout,
    render_indirect_layout: BindGroupLayout,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ParticleInitPipelineKeyFlags: u8 {
        const CLONE = 0x1;
        const ATTRIBUTE_PREV = 0x2;
        const ATTRIBUTE_NEXT = 0x4;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ParticleInitPipelineKey {
    /// Compute shader, with snippets applied, but not preprocessed yet.
    shader: Handle<Shader>,
    /// Minimum binding size in bytes for the particle layout buffer.
    particle_layout_min_binding_size: NonZeroU64,
    /// Minimum binding size in bytes for the property layout buffer, if the
    /// effect has any property. Otherwise this is `None`.
    property_layout_min_binding_size: Option<NonZeroU64>,
    flags: ParticleInitPipelineKeyFlags,
    /// Minimum binding size in bytes for the particle layout buffer of the
    /// parent effect, if any.
    parent_particle_layout_min_binding_size: Option<NonZeroU64>,
}

impl FromWorld for ParticlesInitPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let sim_params_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:update_sim_params",
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

        let spawner_buffer_layout = render_device.create_bind_group_layout(
            "hanabi:buffer_layout:init_spawner",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: true,
                    min_binding_size: Some(GpuSpawnerParams::min_size()),
                },
                count: None,
            }],
        );

        let render_indirect_layout = create_init_render_indirect_bind_group_layout(
            render_device,
            "hanabi:bind_group_layout:init_render_indirect",
            true,
        );

        Self {
            render_device: render_device.clone(),
            sim_params_layout,
            spawner_buffer_layout,
            render_indirect_layout,
        }
    }
}

impl SpecializedComputePipeline for ParticlesInitPipeline {
    type Key = ParticleInitPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        // We use the hash to correlate the key content with the GPU resource name
        let hash = calc_hash(&key);
        trace!("Specializing init pipeline {:08X} with key {:?}", hash, key);

        // FIXME - This duplicates the layout created in EffectBuffer::new()!!!
        // Likely this one should go away, because we can't cache from inside
        // specialize() (non-mut access)
        let has_event_buffer = key.parent_particle_layout_min_binding_size.is_some();
        // create_init_particles_bind_group_layout()
        // "hanabi:init_particles_buffer_layout",
        let particles_buffer_layout = EffectBuffer::make_init_layout(
            &self.render_device,
            key.particle_layout_min_binding_size,
            key.property_layout_min_binding_size,
            key.parent_particle_layout_min_binding_size,
            has_event_buffer,
        );

        let mut shader_defs = vec![];
        if key.flags.contains(ParticleInitPipelineKeyFlags::CLONE) {
            shader_defs.push(ShaderDefVal::Bool("CLONE".to_string(), true));
        }
        if key
            .flags
            .contains(ParticleInitPipelineKeyFlags::ATTRIBUTE_PREV)
        {
            shader_defs.push(ShaderDefVal::Bool("ATTRIBUTE_PREV".to_string(), true));
        }
        if key
            .flags
            .contains(ParticleInitPipelineKeyFlags::ATTRIBUTE_NEXT)
        {
            shader_defs.push(ShaderDefVal::Bool("ATTRIBUTE_NEXT".to_string(), true));
        }
        // Key: USE_GPU_SPAWN_EVENTS
        if key.parent_particle_layout_min_binding_size.is_some() {
            shader_defs.push("USE_GPU_SPAWN_EVENTS".into());
        }

        let label = format!("hanabi:pipeline_init_compute_{:08X}", hash);

        ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![
                self.sim_params_layout.clone(),
                particles_buffer_layout,
                self.spawner_buffer_layout.clone(),
                self.render_indirect_layout.clone(),
            ],
            shader: key.shader,
            shader_defs,
            entry_point: "main".into(),
            push_constant_ranges: vec![],
        }
    }
}

#[derive(Resource)]
pub(crate) struct ParticlesUpdatePipeline {
    render_device: RenderDevice,
    sim_params_layout: BindGroupLayout,
    spawner_buffer_layout: BindGroupLayout,
    render_indirect_layout: BindGroupLayout,
}

impl FromWorld for ParticlesUpdatePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        trace!("GpuSimParams: min_size={}", GpuSimParams::min_size());
        let sim_params_layout = render_device.create_bind_group_layout(
            "hanabi:update_sim_params_layout",
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
            "GpuSpawnerParams: min_size={}",
            GpuSpawnerParams::min_size()
        );
        let spawner_buffer_layout = render_device.create_bind_group_layout(
            "hanabi:update_spawner_buffer_layout",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: true,
                    min_binding_size: Some(GpuSpawnerParams::min_size()),
                },
                count: None,
            }],
        );

        let storage_alignment = render_device.limits().min_storage_buffer_offset_alignment;
        let render_effect_indirect_size = GpuRenderEffectMetadata::aligned_size(storage_alignment);
        let render_group_indirect_size = GpuRenderGroupIndirect::aligned_size(storage_alignment);
        trace!("GpuRenderEffectMetadata: min_size={} padded_size={} | GpuRenderGroupIndirect: min_size={} padded_size={}",
            GpuRenderEffectMetadata::min_size(),
            render_effect_indirect_size.get(),
            GpuRenderGroupIndirect::min_size(),
            render_group_indirect_size.get());
        let render_indirect_layout = render_device.create_bind_group_layout(
            "hanabi:update_render_indirect_layout",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(render_effect_indirect_size),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        // Array; needs padded size
                        min_binding_size: Some(render_group_indirect_size),
                    },
                    count: None,
                },
            ],
        );

        Self {
            render_device: render_device.clone(),
            sim_params_layout,
            spawner_buffer_layout,
            render_indirect_layout,
        }
    }
}

#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub(crate) struct ParticleUpdatePipelineKey {
    /// Compute shader, with snippets applied, but not preprocessed yet.
    shader: Handle<Shader>,
    /// Particle layout.
    particle_layout: ParticleLayout,
    /// Property layout.
    property_layout: PropertyLayout,
    /// Key: EMITS_GPU_SPAWN_EVENTS
    num_event_buffers: u32,
    is_trail: bool,
}

impl SpecializedComputePipeline for ParticlesUpdatePipeline {
    type Key = ParticleUpdatePipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        // We use the hash to correlate the key content with the GPU resource name
        let hash = calc_hash(&key);
        trace!(
            "Specializing update pipeline {:08X} with key {:?}",
            hash,
            key
        );

        // create_update_bind_group_layout()
        //      "hanabi:update_particles_buffer_layout",
        let update_particles_buffer_layout = EffectBuffer::make_update_layout(
            &self.render_device,
            key.particle_layout.min_binding_size(),
            if key.property_layout.is_empty() {
                None
            } else {
                Some(key.property_layout.min_binding_size())
            },
            key.num_event_buffers,
        );

        let mut shader_defs = vec!["REM_MAX_SPAWN_ATOMIC".into()];
        if key.particle_layout.contains(Attribute::PREV) {
            shader_defs.push("ATTRIBUTE_PREV".into());
        }
        if key.particle_layout.contains(Attribute::NEXT) {
            shader_defs.push("ATTRIBUTE_NEXT".into());
        }
        if key.is_trail {
            shader_defs.push("TRAIL".into());
        }
        // Key: EMITS_GPU_SPAWN_EVENTS
        if key.num_event_buffers > 0 {
            shader_defs.push("EMITS_GPU_SPAWN_EVENTS".into());
        }

        let label = format!("hanabi:pipeline_update_compute_{:08X}", hash);

        ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![
                self.sim_params_layout.clone(),
                update_particles_buffer_layout,
                self.spawner_buffer_layout.clone(),
                self.render_indirect_layout.clone(),
            ],
            shader: key.shader,
            shader_defs,
            entry_point: "main".into(),
            push_constant_ranges: Vec::new(),
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
            "hanabi:view_layout_render",
            &[
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
    /// Texture layout.
    texture_layout: TextureLayout,
    /// Key: LOCAL_SPACE_SIMULATION
    /// The effect is simulated in local space, and during rendering all
    /// particles are transformed by the effect's [`GlobalTransform`].
    local_space_simulation: bool,
    /// Key: USE_ALPHA_MASK
    /// The effect is rendered with alpha masking.
    use_alpha_mask: bool,
    /// The effect needs Alpha blend.
    alpha_mode: AlphaMode,
    /// Key: FLIPBOOK
    /// The effect is rendered with flipbook texture animation based on the
    /// sprite index of each particle.
    flipbook: bool,
    /// Key: NEEDS_UV
    /// The effect needs UVs.
    needs_uv: bool,
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

impl Default for ParticleRenderPipelineKey {
    fn default() -> Self {
        Self {
            shader: Handle::default(),
            particle_layout: ParticleLayout::empty(),
            texture_layout: default(),
            local_space_simulation: false,
            use_alpha_mask: false,
            alpha_mode: AlphaMode::Blend,
            flipbook: false,
            needs_uv: false,
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
        trace!("Specializing render pipeline for key: {:?}", key);

        // Base mandatory part of vertex buffer layout
        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: 20,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                //  @location(0) vertex_position: vec3<f32>
                VertexAttribute {
                    format: VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                //  @location(1) vertex_uv: vec2<f32>
                VertexAttribute {
                    format: VertexFormat::Float32x2,
                    offset: 12,
                    shader_location: 1,
                },
                //  @location(1) vertex_color: u32
                // VertexAttribute {
                //     format: VertexFormat::Uint32,
                //     offset: 12,
                //     shader_location: 1,
                // },
                //  @location(2) vertex_velocity: vec3<f32>
                // VertexAttribute {
                //     format: VertexFormat::Float32x3,
                //     offset: 12,
                //     shader_location: 1,
                // },
                //  @location(3) vertex_uv: vec2<f32>
                // VertexAttribute {
                //     format: VertexFormat::Float32x2,
                //     offset: 28,
                //     shader_location: 3,
                // },
            ],
        };

        let dispatch_indirect_size = GpuDispatchIndirect::aligned_size(
            self.render_device
                .limits()
                .min_storage_buffer_offset_alignment,
        );
        let mut entries = vec![
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
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(4u64),
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
        if key.local_space_simulation {
            entries.push(BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: Some(GpuSpawnerParams::min_size()),
                },
                count: None,
            });
        }

        trace!(
            "GpuParticle: layout.min_binding_size={}",
            key.particle_layout.min_binding_size()
        );
        trace!(
            "Creating render bind group layout with {} entries",
            entries.len()
        );
        let particles_buffer_layout = self
            .render_device
            .create_bind_group_layout("hanabi:buffer_layout_render", &entries);

        let mut layout = vec![self.view_layout.clone(), particles_buffer_layout];
        let mut shader_defs = vec!["SPAWNER_READONLY".into()];

        if let Some(material_bind_group_layout) = self.get_material(&key.texture_layout) {
            layout.push(material_bind_group_layout.clone());
            // //  @location(1) vertex_uv: vec2<f32>
            // vertex_buffer_layout.attributes.push(VertexAttribute {
            //     format: VertexFormat::Float32x2,
            //     offset: 12,
            //     shader_location: 1,
            // });
            // vertex_buffer_layout.array_stride += 8;
        }

        // Key: LOCAL_SPACE_SIMULATION
        if key.local_space_simulation {
            shader_defs.push("LOCAL_SPACE_SIMULATION".into());
            shader_defs.push("RENDER_NEEDS_SPAWNER".into());
        }

        // Key: USE_ALPHA_MASK
        if key.use_alpha_mask {
            shader_defs.push("USE_ALPHA_MASK".into());
        }

        // Key: FLIPBOOK
        if key.flipbook {
            shader_defs.push("FLIPBOOK".into());
        }

        // Key: NEEDS_UV
        if key.needs_uv {
            shader_defs.push("NEEDS_UV".into());
        }

        // Key: RIBBONS
        if key.ribbons {
            shader_defs.push("RIBBONS".into());
        }

        #[cfg(all(feature = "2d", feature = "3d"))]
        let depth_stencil = match key.pipeline_mode {
            // Bevy's Transparent2d render phase doesn't support a depth-stencil buffer.
            PipelineMode::Camera2d => None,
            PipelineMode::Camera3d => Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                // Use depth buffer with alpha-masked particles, not with transparent ones
                depth_write_enabled: key.use_alpha_mask,
                // Bevy uses reverse-Z, so Greater really means closer
                depth_compare: CompareFunction::Greater,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
        };

        #[cfg(all(feature = "2d", not(feature = "3d")))]
        let depth_stencil: Option<DepthStencilState> = None;

        #[cfg(all(feature = "3d", not(feature = "2d")))]
        let depth_stencil = Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            // Use depth buffer with alpha-masked particles, not with transparent ones
            depth_write_enabled: key.use_alpha_mask,
            // Bevy uses reverse-Z, so Greater really means closer
            depth_compare: CompareFunction::Greater,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        });

        let format = if key.hdr {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: key.shader.clone(),
                entry_point: "vertex".into(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_buffer_layout],
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
            label: Some("hanabi:pipeline_render".into()),
            push_constant_ranges: Vec::new(),
        }
    }
}

/// A single effect instance extracted from a [`ParticleEffect`] as a
/// render world item.
///
/// [`ParticleEffect`]: crate::ParticleEffect
#[derive(Debug, Component)]
pub(crate) struct ExtractedEffect {
    /// Handle to the effect asset this instance is based on.
    /// The handle is weak to prevent refcount cycles and gracefully handle
    /// assets unloaded or destroyed after a draw call has been submitted.
    pub handle: Handle<EffectAsset>,
    /// Parent effect, if any.
    pub parent: Option<Entity>,
    /// Particle layout for the effect.
    #[allow(dead_code)]
    pub particle_layout: ParticleLayout,
    /// Property layout for the effect.
    pub property_layout: PropertyLayout,
    /// Particle layout of the parent effect, if any.
    pub parent_particle_layout: Option<ParticleLayout>,
    /// Values of properties written in a binary blob according to
    /// [`property_layout`].
    ///
    /// This is `Some(blob)` if the data needs to be (re)uploaded to GPU, or
    /// `None` if nothing needs to be done for this frame.
    ///
    /// [`property_layout`]: crate::render::ExtractedEffect::property_layout
    pub property_data: Option<Vec<u8>>,
    /// Maps a group number to the runtime initializer for that group.
    ///
    /// Obtained from calling [`EffectSpawner::tick()`] on the source effect
    /// instance.
    ///
    /// [`EffectSpawner::tick()`]: crate::EffectSpawner::tick
    pub initializers: Vec<EffectInitializer>,
    /// Global transform of the effect origin, extracted from the
    /// [`GlobalTransform`].
    pub transform: Mat4,
    /// Inverse global transform of the effect origin, extracted from the
    /// [`GlobalTransform`].
    pub inverse_transform: Mat4,
    /// Layout flags.
    pub layout_flags: LayoutFlags,
    /// Texture layout.
    pub texture_layout: TextureLayout,
    /// Textures.
    pub textures: Vec<Handle<Image>>,
    /// Alpha mode.
    pub alpha_mode: AlphaMode,
    /// Effect shaders.
    pub effect_shaders: Vec<EffectShader>,
    /// For 2D rendering, the Z coordinate used as the sort key. Ignored for 3D
    /// rendering.
    #[cfg(feature = "2d")]
    pub z_sort_key_2d: FloatOrd,
}

/// Extracted data for newly-added [`ParticleEffect`] component requiring a new
/// GPU allocation.
///
/// [`ParticleEffect`]: crate::ParticleEffect
pub struct AddedEffect {
    /// Entity with a newly-added [`ParticleEffect`] component.
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    pub entity: Entity,
    /// Entity of the parent effect, if any.
    pub parent_entity: Option<Entity>,
    /// GPU spawn event count to allocate for this effect. This is zero if the
    /// effect uses CPU spawning (has no parent).
    pub event_count: u32,
    pub groups: Vec<AddedEffectGroup>,
    /// Layout of particle attributes.
    pub particle_layout: ParticleLayout,
    /// Layout of particle attributes of the parent effect, if any.
    pub parent_particle_layout: Option<ParticleLayout>,
    /// Layout of properties for the effect, if properties are used at all, or
    /// an empty layout.
    pub property_layout: PropertyLayout,
    pub layout_flags: LayoutFlags,
    /// Handle of the effect asset.
    pub handle: Handle<EffectAsset>,
    /// The order in which we evaluate groups.
    pub group_order: Vec<u32>,
}

pub struct AddedEffectGroup {
    pub capacity: u32,
    pub src_group_index_if_trail: Option<u32>,
}

/// Collection of all extracted effects for this frame, inserted into the
/// render world as a render resource.
#[derive(Default, Resource)]
pub(crate) struct ExtractedEffects {
    /// Map of extracted effects from the entity the source [`ParticleEffect`]
    /// is on.
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    pub effects: HashMap<Entity, ExtractedEffect>,
    /// Entites which had their [`ParticleEffect`] component removed.
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    pub removed_effect_entities: Vec<Entity>,
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
    trace!("====== extract_effect_events()");

    let EffectAssetEvents { ref mut images } = *events;
    *images = image_events.read().copied().collect();
}

/// Debugging settings.
///
/// Settings used to debug Hanabi. These have no effect on the actual behavior
/// of Hanabi, but may affect its performance.
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
    /// This value always resets each frame, even if a capture was not started
    /// (generally because one is already on-going).
    pub start_capture_this_frame: bool,

    /// Enable automatically starting a GPU debugger capture when one or more
    /// effects are spawned.
    ///
    /// Enable this feature to automatically capture one or more GPU frames when
    /// a new effect is spawned (as detected by ECS change detection). This
    /// instructs any attached GPU debugger to start a capture; this has no
    /// effect if no debugger is attached.
    pub start_capture_on_new_effect: bool,

    /// Duration of a GPU debugger capture.
    ///
    /// By default this value is zero, and a GPU debugger capture runs for a
    /// single frame. If a non-zero duration is specified here, the capture
    /// will instead stop once the specified duration is reached. All times
    /// are measured using the `Time<Virtual>` clock.
    ///
    /// You should avoid setting this to a value too large, to prevent the
    /// capture size from getting out of control. A value of 50 ms or so, when
    /// running at 60 FPS (16.6 ms per frame) is generally enough to get a
    /// few frames worth of debugging. This can be increased in case of lag
    /// (frames taking longer). Note that only entire frames are
    /// captured, so the actual capture duration is generally longer than
    /// this limit.
    pub capture_duration: Duration,

    ///
    pub capture_frame_count: u32,
}

#[derive(Debug, Default, Clone, Copy, Resource)]
pub(crate) struct RenderDebugSettings {
    is_capturing: bool,
    capture_start: Duration,
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
    mut query: Extract<
        ParamSet<(
            // All existing ParticleEffect components
            Query<(
                Entity,
                Option<&InheritedVisibility>,
                Option<&ViewVisibility>,
                &EffectInitializers,
                &CompiledParticleEffect,
                Option<Ref<EffectProperties>>,
                &GlobalTransform,
            )>,
            // Newly added ParticleEffect components
            Query<
                (Entity, &CompiledParticleEffect),
                (Added<CompiledParticleEffect>, With<GlobalTransform>),
            >,
        )>,
    >,
    mut removed_effects_event_reader: Extract<EventReader<RemovedEffectsEvent>>,
    mut sim_params: ResMut<SimParams>,
    mut extracted_effects: ResMut<ExtractedEffects>,
    render_device: Res<RenderDevice>,
    debug_settings: Extract<Res<DebugSettings>>,
    mut render_debug_settings: ResMut<RenderDebugSettings>,
) {
    trace!("====== extract_effects()");

    // Manage GPU debug capture
    if render_debug_settings.is_capturing {
        render_debug_settings.captured_frames += 1;
    }
    if debug_settings.start_capture_this_frame || debug_settings.start_capture_on_new_effect {
        // Stop any pending capture if needed
        if render_debug_settings.is_capturing
            && real_time.elapsed() - render_debug_settings.capture_start
                > debug_settings.capture_duration
            && render_debug_settings.captured_frames >= debug_settings.capture_frame_count
        {
            render_device.wgpu_device().stop_capture();
            render_debug_settings.is_capturing = false;
            warn!(
                "Stopped GPU debug capture after {} frames, at t={}s.",
                render_debug_settings.captured_frames,
                real_time.elapsed().as_secs_f64()
            );
        }

        // If no pending capture, consider starting a new one
        if !render_debug_settings.is_capturing {
            // Check if we can start a capture
            if debug_settings.start_capture_this_frame
                || (debug_settings.start_capture_on_new_effect && !query.p1().is_empty())
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

            // Always reset, even if capture didn't start, so that a "this
            // frame" flag doesn't trigger a capture on a different
            // frame. debug_settings.start_capture_this_frame =
            // false;
        }
    }

    // Save simulation params into render world
    sim_params.time = time.elapsed_seconds_f64();
    sim_params.delta_time = time.delta_seconds();
    sim_params.virtual_time = virtual_time.elapsed_seconds_f64();
    sim_params.virtual_delta_time = virtual_time.delta_seconds();
    sim_params.real_time = real_time.elapsed_seconds_f64();
    sim_params.real_delta_time = real_time.delta_seconds();

    // Collect removed effects for later GPU data purge
    extracted_effects.removed_effect_entities =
        removed_effects_event_reader
            .read()
            .fold(vec![], |mut acc, ev| {
                // FIXME - Need to clone because we can't consume the event, we only have
                // read-only access to the main world
                acc.append(&mut ev.entities.clone());
                acc
            });
    trace!(
        "Found {} removed effect(s).",
        extracted_effects.removed_effect_entities.len()
    );

    // Collect added effects for later GPU data allocation
    extracted_effects.added_effects = query
        .p1()
        .iter()
        .filter_map(|(entity, compiled_effect)| {
            let handle = compiled_effect.asset.clone_weak();
            let asset = effects.get(&compiled_effect.asset)?;
            let particle_layout = asset.particle_layout();
            assert!(
                particle_layout.size() > 0,
                "Invalid empty particle layout for effect '{}' on entity {:?}. Did you forget to add some modifier to the asset?",
                asset.name,
                entity
            );
            let property_layout = asset.property_layout();
            let group_order = asset.calculate_group_order();

            trace!(
                "Found new effect: entity {:?} | capacities {:?} | particle_layout {:?} | \
                 property_layout {:?} | layout_flags {:?}",
                 entity,
                 asset.capacities(),
                 particle_layout,
                 property_layout,
                 compiled_effect.layout_flags);

            trace!("Found new effect: entity {:?} | capacities {:?} | particle_layout {:?} | property_layout {:?} | layout_flags {:?}", entity, asset.capacities(), particle_layout, property_layout, compiled_effect.layout_flags);
            Some(AddedEffect {
                entity,
                parent_entity: compiled_effect.parent,
                // FIXME - fixed 400 events per child (per frame) for now...
                event_count: if compiled_effect.parent.is_some() { 400 } else { 0 },
                groups: asset.capacities().iter().zip(asset.init.iter()).map(|(&capacity, init)| {
                    AddedEffectGroup {
                        capacity,
                        src_group_index_if_trail: match init {
                            Initializer::Spawner(_) => None,
                            Initializer::Cloner(cloner) => Some(cloner.src_group_index),
                        }
                    }
                }).collect(),
                particle_layout,
                parent_particle_layout: compiled_effect.parent_particle_layout.clone(),
                property_layout,
                group_order,
                layout_flags: compiled_effect.layout_flags,
                handle,
            })
        })
        .collect();

    // Loop over all existing effects to extract them
    extracted_effects.effects.clear();
    for (
        entity,
        maybe_inherited_visibility,
        maybe_view_visibility,
        initializers,
        compiled_effect,
        maybe_properties,
        transform,
    ) in query.p0().iter_mut()
    {
        // Check if shaders are configured
        let effect_shaders = compiled_effect.get_configured_shaders();
        if effect_shaders.is_empty() {
            continue;
        }

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
                entity
            );
            continue;
        };

        #[cfg(feature = "2d")]
        let z_sort_key_2d = compiled_effect.z_layer_2d;

        let property_layout = asset.property_layout();
        let texture_layout = asset.module().texture_layout();

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

        let layout_flags = compiled_effect.layout_flags;
        let alpha_mode = compiled_effect.alpha_mode;

        trace!(
            "Extracted instance of effect '{}' on entity {:?}: texture_layout_count={} texture_count={} layout_flags={:?}",
            asset.name,
            entity,
            texture_layout.layout.len(),
            compiled_effect.textures.len(),
            layout_flags,
        );

        extracted_effects.effects.insert(
            entity,
            ExtractedEffect {
                handle: compiled_effect.asset.clone_weak(),
                parent: compiled_effect.parent.clone(),
                particle_layout: asset.particle_layout().clone(),
                property_layout,
                parent_particle_layout: compiled_effect.parent_particle_layout.clone(),
                property_data,
                initializers: initializers.0.clone(),
                transform: transform.compute_matrix(),
                // TODO - more efficient/correct way than inverse()?
                inverse_transform: transform.compute_matrix().inverse(),
                layout_flags,
                texture_layout,
                textures: compiled_effect.textures.clone(),
                alpha_mode,
                effect_shaders: effect_shaders.to_vec(),
                #[cfg(feature = "2d")]
                z_sort_key_2d,
            },
        );
    }
}

/// GPU representation of a single vertex of a particle mesh stored in a GPU
/// buffer.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, ShaderType)]
struct GpuParticleVertex {
    /// Vertex position.
    pub position: [f32; 3],
    /// UV coordinates of vertex.
    pub uv: [f32; 2],
}

/// Various GPU limits and aligned sizes computed once and cached.
struct GpuLimits {
    /// Value of [`WgpuLimits::min_storage_buffer_offset_alignment`].
    ///
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`]: bevy::render::settings::WgpuLimits::min_storage_buffer_offset_alignment
    storage_buffer_align: NonZeroU32,

    /// Size of [`GpuDispatchIndirect`] aligned to the contraint of
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`].
    ///
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`]: bevy::render::settings::WgpuLimits::min_storage_buffer_offset_alignment
    dispatch_indirect_aligned_size: NonZeroU32,

    /// Size of [`GpuRenderEffectMetadata`] aligned to the contraint of
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`].
    ///
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`]: bevy::render::settings::WgpuLimits::min_storage_buffer_offset_alignment
    render_effect_indirect_aligned_size: NonZeroU32,

    /// Size of [`GpuRenderGroupIndirect`] aligned to the contraint of
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`].
    ///
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`]: bevy::render::settings::WgpuLimits::min_storage_buffer_offset_alignment
    render_group_indirect_aligned_size: NonZeroU32,

    /// Size of [`GpuParticleGroup`] aligned to the contraint of
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`].
    ///
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`]: bevy::render::settings::WgpuLimits::min_storage_buffer_offset_alignment
    particle_group_aligned_size: NonZeroU32,
}

impl GpuLimits {
    pub fn from_device(render_device: &RenderDevice) -> Self {
        let storage_buffer_align = render_device.limits().min_storage_buffer_offset_alignment;

        let dispatch_indirect_aligned_size = NonZeroU32::new(next_multiple_of(
            GpuDispatchIndirect::min_size().get() as usize,
            storage_buffer_align as usize,
        ) as u32)
        .unwrap();

        let render_effect_indirect_aligned_size = NonZeroU32::new(next_multiple_of(
            GpuRenderEffectMetadata::min_size().get() as usize,
            storage_buffer_align as usize,
        ) as u32)
        .unwrap();

        let render_group_indirect_aligned_size = NonZeroU32::new(next_multiple_of(
            GpuRenderGroupIndirect::min_size().get() as usize,
            storage_buffer_align as usize,
        ) as u32)
        .unwrap();

        let particle_group_aligned_size = NonZeroU32::new(next_multiple_of(
            GpuParticleGroup::min_size().get() as usize,
            storage_buffer_align as usize,
        ) as u32)
        .unwrap();

        trace!(
            "GPU-aligned sizes (align: {} B):\n- GpuRenderEffectMetadata: {} B -> {} B\n\
            - GpuRenderGroupIndirect: {} B -> {} B\n- GpuParticleGroup: {} B -> {} B",
            storage_buffer_align,
            GpuRenderEffectMetadata::min_size().get(),
            dispatch_indirect_aligned_size.get(),
            GpuRenderGroupIndirect::min_size().get(),
            render_effect_indirect_aligned_size.get(),
            GpuParticleGroup::min_size().get(),
            particle_group_aligned_size.get()
        );

        Self {
            storage_buffer_align: NonZeroU32::new(storage_buffer_align).unwrap(),
            dispatch_indirect_aligned_size,
            render_effect_indirect_aligned_size,
            render_group_indirect_aligned_size,
            particle_group_aligned_size,
        }
    }

    /// Byte alignment for any storage buffer binding.
    pub fn storage_buffer_align(&self) -> NonZeroU32 {
        self.storage_buffer_align
    }

    /// Byte alignment for [`GpuDispatchIndirect`].
    pub fn dispatch_indirect_offset(&self, buffer_index: u32) -> u32 {
        self.dispatch_indirect_aligned_size.get() * buffer_index
    }

    /// Byte offset of the [`GpuRenderEffectMetadata`] of a given buffer.
    pub fn render_effect_indirect_offset(&self, buffer_index: u32) -> u64 {
        self.render_effect_indirect_aligned_size.get() as u64 * buffer_index as u64
    }

    /// Byte alignment for [`GpuRenderEffectMetadata`].
    pub fn render_effect_indirect_size(&self) -> NonZeroU64 {
        NonZeroU64::new(self.render_effect_indirect_aligned_size.get() as u64).unwrap()
    }

    /// Byte offset for the [`GpuRenderGroupIndirect`] of a given buffer.
    pub fn render_group_indirect_offset(&self, buffer_index: u32) -> u64 {
        self.render_group_indirect_aligned_size.get() as u64 * buffer_index as u64
    }

    /// Byte alignment for [`GpuRenderGroupIndirect`].
    pub fn render_group_indirect_size(&self) -> NonZeroU64 {
        NonZeroU64::new(self.render_group_indirect_aligned_size.get() as u64).unwrap()
    }

    /// Byte offset for the [`GpuParticleGroup`] of a given buffer.
    pub fn particle_group_offset(&self, buffer_index: u32) -> u32 {
        self.particle_group_aligned_size.get() * buffer_index
    }
}

struct CacheEntry {
    cache_id: EffectCacheId,
    parent_entity: Option<Entity>,
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
    /// Map from an entity of the main world with a [`ParticleEffect`] component
    /// attached to it, to the associated effect slice allocated in the
    /// [`EffectCache`].
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    entity_map: HashMap<Entity, CacheEntry>,
    /// Bind group for the camera view, containing the camera projection and
    /// other uniform values related to the camera.
    view_bind_group: Option<BindGroup>,
    /// Bind group for the simulation parameters, like the current time and
    /// frame delta time.
    sim_params_bind_group: Option<BindGroup>,
    /// Bind group for the spawning parameters (number of particles to spawn
    /// this frame, ...).
    spawner_bind_group: Option<BindGroup>,
    /// Bind group #0 of the vfx_indirect shader, containing both the indirect
    /// compute dispatch and render buffers.
    dr_indirect_bind_group: Option<BindGroup>,
    /// Bind group #3 of the vfx_init shader, containing the indirect render
    /// buffer.
    init_render_indirect_bind_group: Option<BindGroup>,
    /// Global shared GPU uniform buffer storing the simulation parameters,
    /// uploaded each frame from CPU to GPU.
    sim_params_uniforms: UniformBuffer<GpuSimParams>,
    /// Global shared GPU buffer storing the various spawner parameter structs
    /// for the active effect instances.
    spawner_buffer: AlignedBufferVec<GpuSpawnerParams>,
    /// Global shared GPU buffer storing the various indirect dispatch structs
    /// for the indirect dispatch of the Update pass.
    dispatch_indirect_buffer: BufferTable<GpuDispatchIndirect>,
    /// Global shared GPU buffer storing the various `RenderEffectMetadata`
    /// structs for the active effect instances.
    render_effect_dispatch_buffer: BufferTable<GpuRenderEffectMetadata>,
    /// Stores the GPU `RenderGroupIndirect` structures, which describe mutable
    /// data specific to a particle group.
    ///
    /// These structures also store the data needed for indirect dispatch of
    /// drawcalls.
    render_group_dispatch_buffer: BufferTable<GpuRenderGroupIndirect>,
    /// Stores the GPU `ParticleGroup` structures, which are metadata describing
    /// each particle group that's populated by the CPU and read (only read) by
    /// the GPU.
    particle_group_buffer: AlignedBufferVec<GpuParticleGroup>,
    /// Unscaled vertices of the mesh of a single particle, generally a quad.
    /// The mesh is later scaled during rendering by the "particle size".
    // FIXME - This is a per-effect thing, unless we merge all meshes into a single buffer (makes
    // sense) but in that case we need a vertex slice too to know which mesh to draw per effect.
    vertices: BufferVec<GpuParticleVertex>,
    /// Various GPU limits and aligned sizes lazily allocated and cached for
    /// convenience.
    gpu_limits: GpuLimits,
}

impl EffectsMeta {
    pub fn new(device: RenderDevice) -> Self {
        let mut vertices = BufferVec::new(BufferUsages::VERTEX);
        for v in QUAD_VERTEX_POSITIONS {
            let uv = v.truncate() + 0.5;
            let v = *v * Vec3::new(1.0, 1.0, 1.0);
            vertices.push(GpuParticleVertex {
                position: v.into(),
                uv: uv.into(),
            });
        }

        let gpu_limits = GpuLimits::from_device(&device);

        // Ensure individual GpuSpawnerParams elements are properly aligned so they can
        // be addressed individually by the computer shaders.
        let item_align = gpu_limits.storage_buffer_align().get() as u64;
        trace!(
            "Aligning storage buffers to {} bytes as device limits requires.",
            item_align
        );

        Self {
            entity_map: HashMap::default(),
            view_bind_group: None,
            sim_params_bind_group: None,
            spawner_bind_group: None,
            dr_indirect_bind_group: None,
            init_render_indirect_bind_group: None,
            sim_params_uniforms: UniformBuffer::default(),
            spawner_buffer: AlignedBufferVec::new(
                BufferUsages::STORAGE,
                NonZeroU64::new(item_align),
                Some("hanabi:buffer:spawner".to_string()),
            ),
            dispatch_indirect_buffer: BufferTable::new(
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
                // NOTE: Technically we're using an offset in dispatch_workgroups_indirect(), but
                // `min_storage_buffer_offset_alignment` is documented as being for the offset in
                // BufferBinding and the dynamic offset in set_bind_group(), so either the
                // documentation is lacking or we don't need to align here.
                NonZeroU64::new(item_align),
                Some("hanabi:buffer:dispatch_indirect".to_string()),
            ),
            render_effect_dispatch_buffer: BufferTable::new(
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
                NonZeroU64::new(item_align),
                Some("hanabi:buffer:render_effect_dispatch".to_string()),
            ),
            render_group_dispatch_buffer: BufferTable::new(
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
                NonZeroU64::new(item_align),
                Some("hanabi:buffer:render_group_dispatch".to_string()),
            ),
            particle_group_buffer: AlignedBufferVec::new(
                BufferUsages::STORAGE,
                NonZeroU64::new(item_align),
                Some("hanabi:buffer:particle_group".to_string()),
            ),
            vertices,
            gpu_limits,
        }
    }

    /// Allocate internal resources for newly spawned effects, and deallocate
    /// them for just-removed ones.
    pub fn add_remove_effects(
        &mut self,
        mut added_effects: Vec<AddedEffect>,
        removed_effect_entities: Vec<Entity>,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        effect_bind_groups: &mut ResMut<EffectBindGroups>,
        effect_cache: &mut ResMut<EffectCache>,
    ) {
        // Deallocate GPU data for destroyed effect instances. This will automatically
        // drop any group where there is no more effect slice.
        trace!(
            "Removing {} despawned effects",
            removed_effect_entities.len()
        );
        for entity in &removed_effect_entities {
            trace!("Removing ParticleEffect on entity {:?}", entity);
            if let Some(entry) = self.entity_map.remove(entity) {
                trace!(
                    "=> ParticleEffect on entity {:?} had cache ID {:?}, removing...",
                    entity,
                    entry.cache_id
                );
                if let Some(cached_effect_indices) = effect_cache.remove(entry.cache_id) {
                    // Clear bind groups associated with the removed buffer
                    trace!(
                        "=> GPU buffer #{} gone, destroying its bind groups...",
                        cached_effect_indices.buffer_index
                    );
                    effect_bind_groups
                        .particle_buffers
                        .remove(&cached_effect_indices.buffer_index);

                    let slices_ref = &cached_effect_indices.slices;
                    debug_assert!(slices_ref.ranges.len() >= 2);
                    let group_count = (slices_ref.ranges.len() - 1) as u32;

                    let first_row = slices_ref
                        .dispatch_buffer_indices
                        .first_update_group_dispatch_buffer_index
                        .0;
                    for table_id in first_row..(first_row + group_count) {
                        self.dispatch_indirect_buffer
                            .remove(BufferTableId(table_id));
                    }
                    self.render_effect_dispatch_buffer.remove(
                        slices_ref
                            .dispatch_buffer_indices
                            .render_effect_metadata_buffer_index,
                    );
                    let first_row = slices_ref
                        .dispatch_buffer_indices
                        .first_render_group_dispatch_buffer_index
                        .0;
                    for table_id in first_row..(first_row + group_count) {
                        self.render_group_dispatch_buffer
                            .remove(BufferTableId(table_id));
                    }
                }
            }
        }

        // FIXME - We delete a buffer above, and have a chance to immediatly re-create
        // it below. We should keep the GPU buffer around until the end of this method.
        // On the other hand, we should also be careful that allocated buffers need to
        // be tightly packed because 'vfx_indirect.wgsl' index them by buffer index in
        // order, so doesn't support offset.

        trace!("Adding {} newly spawned effects", added_effects.len());
        for added_effect in added_effects.drain(..) {
            trace!(
                "+ added effect: total_capacity={}",
                added_effect.groups.iter().map(|g| g.capacity).sum::<u32>()
            );

            let first_update_group_dispatch_buffer_index = allocate_sequential_buffers(
                &mut self.dispatch_indirect_buffer,
                iter::repeat(GpuDispatchIndirect::default()).take(added_effect.groups.len()),
            );

            let render_effect_dispatch_buffer_id = self
                .render_effect_dispatch_buffer
                .insert(GpuRenderEffectMetadata::default());

            let mut current_base_instance = 0;
            let first_render_group_dispatch_buffer_index = allocate_sequential_buffers(
                &mut self.render_group_dispatch_buffer,
                added_effect.groups.iter().map(|group| {
                    let indirect_dispatch = GpuRenderGroupIndirect {
                        vertex_count: 6, // TODO - Flexible vertex count and mesh particles
                        dead_count: group.capacity,
                        base_instance: current_base_instance,
                        max_spawn: group.capacity,
                        ..default()
                    };
                    current_base_instance += group.capacity;
                    indirect_dispatch
                }),
            );

            let mut trail_dispatch_buffer_indices = HashMap::new();
            for (dest_group_index, group) in added_effect.groups.iter().enumerate() {
                let Some(src_group_index) = group.src_group_index_if_trail else {
                    continue;
                };
                trail_dispatch_buffer_indices.insert(
                    dest_group_index as u32,
                    TrailDispatchBufferIndices {
                        dest: first_render_group_dispatch_buffer_index
                            .offset(dest_group_index as u32),
                        src: first_render_group_dispatch_buffer_index.offset(src_group_index),
                    },
                );
            }

            let dispatch_buffer_indices = DispatchBufferIndices {
                first_update_group_dispatch_buffer_index,
                render_effect_metadata_buffer_index: render_effect_dispatch_buffer_id,
                first_render_group_dispatch_buffer_index,
                trail_dispatch_buffer_indices,
            };

            // Insert the effect into the cache. This will allocate all the necessary GPU
            // resources as needed.
            let cache_id = effect_cache.insert(
                added_effect.entity,
                added_effect.handle,
                added_effect
                    .groups
                    .iter()
                    .map(|group| group.capacity)
                    .collect(),
                &added_effect.particle_layout,
                added_effect.parent_particle_layout.as_ref(),
                &added_effect.property_layout,
                added_effect.layout_flags,
                added_effect.event_count,
                dispatch_buffer_indices,
                added_effect.group_order,
            );

            let entity = added_effect.entity;
            self.entity_map.insert(
                entity,
                CacheEntry {
                    cache_id,
                    parent_entity: added_effect.parent_entity,
                },
            );

            // Note: those effects are already in extracted_effects.effects
            // because they were gathered by the same query as
            // previously existing ones, during extraction.

            // let index = self.effect_cache.buffer_index(cache_id).unwrap();
            //
            // let table_id = self
            // .dispatch_indirect_buffer
            // .insert(GpuDispatchIndirect::default());
            // assert_eq!(
            // table_id.0, index,
            // "Broken table invariant: buffer={} row={}",
            // index, table_id.0
            // );
        }

        // Resolve parents
        effect_cache.resolve_parents(self.entity_map.iter().filter_map(|(_entity, entry)| {
            if let Some(parent_entity) = entry.parent_entity {
                Some((entry.cache_id, parent_entity))
            } else {
                None
            }
        }));

        // Once all changes are applied, immediately schedule any GPU buffer
        // (re)allocation based on the new buffer size. The actual GPU buffer content
        // will be written later.
        if self
            .dispatch_indirect_buffer
            .allocate_gpu(render_device, render_queue)
        {
            // All those bind groups use the buffer so need to be re-created
            effect_bind_groups.particle_buffers.clear();
        }
        if self
            .render_effect_dispatch_buffer
            .allocate_gpu(render_device, render_queue)
        {
            // All those bind groups use the buffer so need to be re-created
            self.dr_indirect_bind_group = None;
            self.init_render_indirect_bind_group = None;
            effect_bind_groups
                .update_render_indirect_bind_groups
                .clear();
        }
        if self
            .render_group_dispatch_buffer
            .allocate_gpu(render_device, render_queue)
        {
            // All those bind groups use the buffer so need to be re-created
            self.dr_indirect_bind_group = None;
            self.init_render_indirect_bind_group = None;
            effect_bind_groups
                .update_render_indirect_bind_groups
                .clear();
        }
    }
}

const QUAD_VERTEX_POSITIONS: &[Vec3] = &[
    Vec3::from_array([-0.5, -0.5, 0.0]),
    Vec3::from_array([0.5, 0.5, 0.0]),
    Vec3::from_array([-0.5, 0.5, 0.0]),
    Vec3::from_array([-0.5, -0.5, 0.0]),
    Vec3::from_array([0.5, -0.5, 0.0]),
    Vec3::from_array([0.5, 0.5, 0.0]),
];

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
        /// The effect is rendered with flipbook texture animation based on the [`Attribute::SPRITE_INDEX`] of each particle.
        const FLIPBOOK = (1 << 4);
        /// The effect needs UVs.
        const NEEDS_UV = (1 << 5);
        /// The effect has ribbons.
        const RIBBONS = (1 << 6);
        /// The (update) shader emits GPU spawn events.
        const EMIT_GPU_SPAWN_EVENTS = (1 << 7);
        /// The (init) shader spawns particles by consuming GPU spawn events, instead of a single CPU spawn count.
        const CONSUME_GPU_SPAWN_EVENTS = (1 << 8);
    }
}

impl Default for LayoutFlags {
    fn default() -> Self {
        Self::NONE
    }
}

pub(crate) fn prepare_effects(
    mut commands: Commands,
    sim_params: Res<SimParams>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    init_pipeline: Res<ParticlesInitPipeline>,
    update_pipeline: Res<ParticlesUpdatePipeline>,
    mut specialized_init_pipelines: ResMut<SpecializedComputePipelines<ParticlesInitPipeline>>,
    mut specialized_update_pipelines: ResMut<SpecializedComputePipelines<ParticlesUpdatePipeline>>,
    mut effects_meta: ResMut<EffectsMeta>,
    mut effect_cache: ResMut<EffectCache>,
    mut extracted_effects: ResMut<ExtractedEffects>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    mut gpu_buffer_operation_queue: ResMut<GpuBufferOperationQueue>,
) {
    trace!("====== prepare_effects()");

    // Write vertices (TODO - lazily once only)
    effects_meta
        .vertices
        .write_buffer(&render_device, &render_queue);

    // Clear last frame's buffer resizes which may have occured during last frame,
    // during `Node::run()` while the `BufferTable` could not be mutated.
    effects_meta
        .dispatch_indirect_buffer
        .clear_previous_frame_resizes();
    effects_meta
        .render_effect_dispatch_buffer
        .clear_previous_frame_resizes();
    effects_meta
        .render_group_dispatch_buffer
        .clear_previous_frame_resizes();

    gpu_buffer_operation_queue.begin_frame();

    // Allocate new effects, deallocate removed ones, resolve parent-child
    // relationships
    let removed_effect_entities = std::mem::take(&mut extracted_effects.removed_effect_entities);
    for entity in &removed_effect_entities {
        extracted_effects.effects.remove(entity);
    }
    effects_meta.add_remove_effects(
        std::mem::take(&mut extracted_effects.added_effects),
        removed_effect_entities,
        &render_device,
        &render_queue,
        &mut effect_bind_groups,
        &mut effect_cache,
    );

    // // sort first by z and then by handle. this ensures that, when possible,
    // batches span multiple z layers // batches won't span z-layers if there is
    // another batch between them extracted_effects.effects.sort_by(|a, b| {
    //     match FloatOrd(a.transform.w_axis[2]).cmp(&FloatOrd(b.transform.
    // w_axis[2])) {         Ordering::Equal => a.handle.cmp(&b.handle),
    //         other => other,
    //     }
    // });

    let effects = std::mem::take(&mut extracted_effects.effects);

    // Resolve parent and child buffers
    let mut children = HashMap::with_capacity(16);
    let parents = effects
        .iter()
        .filter_map(|(entity, extracted_effect)| {
            // Get the parent buffer index, if any
            let parent_entity = extracted_effect.parent?;
            let parent_cache_id = effects_meta.entity_map.get(&parent_entity)?.cache_id;
            let parent_effect_slices = effect_cache.get_slices(parent_cache_id);
            let parent_buffer_index = parent_effect_slices.buffer_index;

            // Add this buffer as a child of the parent entity
            let id: EffectCacheId = effects_meta.entity_map.get(entity)?.cache_id;
            children.entry(parent_entity).or_insert(vec![]).push(id);

            Some((*entity, parent_buffer_index))
        })
        .collect::<HashMap<_, _>>();

    // Build batcher inputs from extracted effects
    let effect_entity_list = effects
        .into_iter()
        .map(|(entity, extracted_effect)| {
            // FIXME - way too many look-ups here again and again with the same id...
            let id = effects_meta.entity_map.get(&entity).unwrap().cache_id;
            let property_buffer = effect_cache.get_property_buffer(id).cloned(); // clone handle for lifetime
            let effect_slices = effect_cache.get_slices(id);
            let init_indirect_dispatch_index = effect_cache.get_init_indirect_dispatch_index(id);
            let parent_buffer_index = parents.get(&entity).cloned();
            let child_effects = if let Some(children) = children.get_mut(&entity) {
                std::mem::take(children)
            } else {
                vec![]
            };
            let group_order = effect_cache.get_group_order(id);

            BatchesInput {
                handle: extracted_effect.handle,
                entity,
                effect_slices,
                property_layout: extracted_effect.property_layout.clone(),
                parent_particle_layout: extracted_effect.parent_particle_layout.clone(),
                parent_buffer_index,
                child_effects,
                effect_shaders: extracted_effect.effect_shaders.clone(),
                layout_flags: extracted_effect.layout_flags,
                texture_layout: extracted_effect.texture_layout.clone(),
                textures: extracted_effect.textures.clone(),
                alpha_mode: extracted_effect.alpha_mode,
                transform: extracted_effect.transform.into(),
                inverse_transform: extracted_effect.inverse_transform.into(),
                particle_layout: extracted_effect.particle_layout.clone(),
                property_buffer,
                group_order: group_order.to_vec(),
                property_data: extracted_effect.property_data,
                init_indirect_dispatch_index,
                initializers: extracted_effect.initializers,
                #[cfg(feature = "2d")]
                z_sort_key_2d: extracted_effect.z_sort_key_2d,
            }
        })
        .collect::<Vec<_>>();
    trace!("Collected {} extracted effects", effect_entity_list.len());

    // Sort first by effect buffer index, then by slice range (see EffectSlice)
    // inside that buffer. This is critical for batching to work, because
    // batching effects is based on compatible items, which implies same GPU
    // buffer and continuous slice ranges (the next slice start must be equal to
    // the previous start end, without gap). EffectSlice already contains both
    // information, and the proper ordering implementation.
    // effect_entity_list.sort_by_key(|a| a.effect_slice.clone());

    // Loop on all extracted effects in order, and try to batch them together to
    // reduce draw calls.
    effects_meta.spawner_buffer.clear();
    effects_meta.particle_group_buffer.clear();
    let mut total_group_count = 0;
    for (batch_index, input) in effect_entity_list.into_iter().enumerate() {
        let effect_cache_id = effects_meta.entity_map.get(&input.entity).unwrap().cache_id;
        let buffer_index = effect_cache.get_slices(effect_cache_id).buffer_index;
        let event_buffer_ref = effect_cache.get_event_slice(effect_cache_id);
        let child_index = effect_cache.get_child_index(effect_cache_id);
        let particle_layout_min_binding_size =
            input.effect_slices.particle_layout.min_binding_size();
        let property_layout_min_binding_size = if input.property_layout.is_empty() {
            None
        } else {
            Some(input.property_layout.min_binding_size())
        };

        // Get the index of the indirect struct, if using indirect dispatch for the init
        // pass. Only effects consuming GPU spawn events use indirect dispatch for their
        // particle init; others are CPU driven and use normal dispatch.
        // FIXME - unused arg
        if let (Some(event_buffer_ref), Some(init_indirect_dispatch_index), Some(child_index)) = (
            event_buffer_ref,
            input.init_indirect_dispatch_index,
            child_index,
        ) {
            debug_assert_eq!(
                GpuChildInfo::min_size().get() % 4,
                0,
                "Invalid GpuChildInfo alignment."
            );

            // Schedule a fill dispatch
            let event_buffer_index = event_buffer_ref.buffer_index;
            let event_slice = event_buffer_ref.slice.clone();
            gpu_buffer_operation_queue.queue_init_fill(
                event_buffer_index,
                event_slice,
                GpuBufferOperationArgs {
                    src_offset: child_index,
                    src_stride: GpuChildInfo::min_size().get() as u32 / 4,
                    dst_offset: init_indirect_dispatch_index,
                    count: 1, // FIXME - should be a batch here!!
                },
            );
        }

        // Create init pipeline key flags.
        let mut init_pipeline_key_flags = ParticleInitPipelineKeyFlags::empty();
        init_pipeline_key_flags.set(
            ParticleInitPipelineKeyFlags::ATTRIBUTE_PREV,
            input.particle_layout.contains(Attribute::PREV),
        );
        init_pipeline_key_flags.set(
            ParticleInitPipelineKeyFlags::ATTRIBUTE_NEXT,
            input.particle_layout.contains(Attribute::NEXT),
        );

        // Specialize the init and update pipelines based on the effect.
        trace!(
            "Specializing pipelines: particle_layout={:?}",
            input.effect_slices.particle_layout
        );
        let init_and_update_pipeline_ids = input
            .effect_shaders
            .iter()
            .enumerate()
            .map(|(group_index, shader)| {
                let mut flags = init_pipeline_key_flags;

                // If this is a cloner, add the appropriate flag.
                match input.initializers[group_index] {
                    EffectInitializer::Spawner(_) => {}
                    EffectInitializer::Cloner(_) => {
                        flags.insert(ParticleInitPipelineKeyFlags::CLONE);
                    }
                }

                let init_pipeline_id = specialized_init_pipelines.specialize(
                    &pipeline_cache,
                    &init_pipeline,
                    ParticleInitPipelineKey {
                        shader: shader.init.clone(),
                        particle_layout_min_binding_size,
                        property_layout_min_binding_size,
                        flags,
                        parent_particle_layout_min_binding_size: input
                            .parent_particle_layout
                            .as_ref()
                            .map(|layout| layout.min_binding_size()),
                    },
                );
                trace!("Init pipeline specialized: id={:?}", init_pipeline_id);

                // Ensure the bind group layout for the update phase exists for the particular
                // config of this effect, before the update pipeline specialization needs it to
                // create the pipeline layout.
                let effect_buffer = effect_cache.buffers_mut()[buffer_index as usize]
                    .as_mut()
                    .unwrap();
                effect_buffer.ensure_particle_update_bind_group_layout(
                    input.child_effects.len() as u32,
                    &render_device,
                    input.effect_slices.particle_layout.min_binding_size(),
                    if input.property_layout.is_empty() {
                        None
                    } else {
                        Some(input.property_layout.min_binding_size())
                    },
                );

                let update_pipeline_id = specialized_update_pipelines.specialize(
                    &pipeline_cache,
                    &update_pipeline,
                    ParticleUpdatePipelineKey {
                        shader: shader.update.clone(),
                        particle_layout: input.effect_slices.particle_layout.clone(),
                        property_layout: input.property_layout.clone(),
                        is_trail: matches!(
                            input.initializers[group_index],
                            EffectInitializer::Cloner(_)
                        ),
                        num_event_buffers: input.child_effects.len() as u32,
                    },
                );
                trace!("Update pipeline specialized: id={:?}", update_pipeline_id);

                InitAndUpdatePipelineIds {
                    init: init_pipeline_id,
                    update: update_pipeline_id,
                }
            })
            .collect();

        let init_shaders: Vec<_> = input
            .effect_shaders
            .iter()
            .map(|shaders| shaders.init.clone())
            .collect();
        trace!("init_shader(s) = {:?}", init_shaders);

        let update_shaders: Vec<_> = input
            .effect_shaders
            .iter()
            .map(|shaders| shaders.update.clone())
            .collect();
        trace!("update_shader(s) = {:?}", update_shaders);

        let render_shaders: Vec<_> = input
            .effect_shaders
            .iter()
            .map(|shaders| shaders.render.clone())
            .collect();
        trace!("render_shader(s) = {:?}", render_shaders);

        let layout_flags = input.layout_flags;
        trace!("layout_flags = {:?}", layout_flags);

        trace!(
            "particle_layout = {:?}",
            input.effect_slices.particle_layout
        );

        #[cfg(feature = "2d")]
        {
            trace!("z_sort_key_2d = {:?}", input.z_sort_key_2d);
        }

        // This callback is raised when creating a new batch from a single item, so the
        // base index for spawners is the current buffer size. Per-effect spawner values
        // will be pushed in order into the array.
        let spawner_base = effects_meta.spawner_buffer.len() as u32;

        for initializer in input.initializers.iter() {
            match initializer {
                EffectInitializer::Spawner(effect_spawner) => {
                    let spawner_params = GpuSpawnerParams {
                        transform: input.transform,
                        inverse_transform: input.inverse_transform,
                        spawn: effect_spawner.spawn_count as i32,
                        // FIXME - Probably bad to re-seed each time there's a change
                        seed: 0, // random::<u32>(),
                        count: 0,
                        // FIXME: the effect_index is global inside the global spawner buffer,
                        // but the group_index is the index of the particle buffer, which can
                        // in theory (with batching) contain > 1 effect per buffer.
                        effect_index: input.effect_slices.buffer_index,
                        lifetime: 0.0,
                        pad: Default::default(),
                    };
                    trace!("spawner params = {:?}", spawner_params);
                    effects_meta.spawner_buffer.push(spawner_params);
                }

                EffectInitializer::Cloner(effect_cloner) => {
                    let spawner_params = GpuSpawnerParams {
                        transform: input.transform,
                        inverse_transform: input.inverse_transform,
                        spawn: 0,
                        // FIXME - Probably bad to re-seed each time there's a change
                        seed: 0, // random::<u32>(),
                        count: 0,
                        // FIXME: the effect_index is global inside the global spawner buffer,
                        // but the group_index is the index of the particle buffer, which can
                        // in theory (with batching) contain > 1 effect per buffer.
                        effect_index: input.effect_slices.buffer_index,
                        lifetime: effect_cloner.cloner.lifetime,
                        pad: Default::default(),
                    };
                    trace!("cloner params = {:?}", spawner_params);
                    effects_meta.spawner_buffer.push(spawner_params);
                }
            }
        }

        // Create the particle group buffer entries.
        let mut first_particle_group_buffer_index = None;
        let mut local_group_count = 0;
        for (group_index, range) in input.effect_slices.slices.windows(2).enumerate() {
            let particle_group_buffer_index =
                effects_meta.particle_group_buffer.push(GpuParticleGroup {
                    global_group_index: total_group_count,
                    // FIXME - Batching: multiple effects in a same batch should have different
                    // indices
                    effect_index: batch_index as u32,
                    group_index_in_effect: group_index as u32,
                    indirect_index: range[0],
                    capacity: range[1] - range[0],
                    effect_particle_offset: input.effect_slices.slices[0],
                    // We can safely unwrap_or() here, if we don't do indirect dispatch then this
                    // value is not read anyway.
                    init_indirect_dispatch_index: input.init_indirect_dispatch_index.unwrap_or(0),
                    // We can safely unwrap_or() here, if we're not a child then this is unused.
                    child_index: child_index.unwrap_or(0),
                });
            if group_index == 0 {
                first_particle_group_buffer_index = Some(particle_group_buffer_index as u32);
            }
            total_group_count += 1;
            local_group_count += 1;
        }

        let dispatch_buffer_indices = effect_cache
            .get_dispatch_buffer_indices(effect_cache_id)
            .clone();

        // Write properties for this effect if they were modified.
        // FIXME - This doesn't work with batching!
        if let Some(property_data) = &input.property_data {
            trace!("Properties changed, need to (re-)upload to GPU");
            if let Some(property_buffer) = input.property_buffer.as_ref() {
                trace!("Scheduled property upload to GPU");
                render_queue.write_buffer(property_buffer, 0, property_data);
            } else {
                error!("Cannot upload properties to GPU, no property buffer!");
            }
        }

        #[cfg(feature = "2d")]
        let z_sort_key_2d = input.z_sort_key_2d;

        #[cfg(feature = "3d")]
        let translation_3d = input.transform.translation();

        // Spawn one shared EffectBatches for all groups of this effect. This contains
        // most of the data needed to drive rendering, except the per-group data.
        // However this doesn't drive rendering; this is just storage.
        let batches = EffectBatches::from_input(
            input,
            spawner_base,
            effect_cache_id,
            init_and_update_pipeline_ids,
            dispatch_buffer_indices,
            first_particle_group_buffer_index.unwrap_or_default(),
        );
        let batches_entity = commands.spawn(batches).id();

        // Spawn one EffectDrawBatch per group, to actually drive rendering. Each group
        // renders with a different indirect call. These are the entities that the
        // render phase items will receive.
        for group_index in 0..local_group_count {
            commands.spawn(EffectDrawBatch {
                batches_entity,
                group_index,
                #[cfg(feature = "2d")]
                z_sort_key_2d,
                #[cfg(feature = "3d")]
                translation_3d,
            });
        }
    }

    // Once all operations are enqueued, upload to GPU
    gpu_buffer_operation_queue.end_frame(&render_device, &render_queue);

    // Write the entire spawner buffer for this frame, for all effects combined
    effects_meta
        .spawner_buffer
        .write_buffer(&render_device, &render_queue);

    // Write the entire particle group buffer for this frame
    if effects_meta
        .particle_group_buffer
        .write_buffer(&render_device, &render_queue)
    {
        // The buffer changed; invalidate all bind groups for all effects.
    }

    // Update simulation parameters
    effects_meta
        .sim_params_uniforms
        .set(sim_params.deref().into());
    {
        let gpu_sim_params = effects_meta.sim_params_uniforms.get_mut();
        gpu_sim_params.num_groups = total_group_count;

        trace!(
            "Simulation parameters: time={} delta_time={} virtual_time={} \
                virtual_delta_time={} real_time={} real_delta_time={} num_groups={}",
            gpu_sim_params.time,
            gpu_sim_params.delta_time,
            gpu_sim_params.virtual_time,
            gpu_sim_params.virtual_delta_time,
            gpu_sim_params.real_time,
            gpu_sim_params.real_delta_time,
            gpu_sim_params.num_groups,
        );
    }
    let prev_buffer_id = effects_meta.sim_params_uniforms.buffer().map(|b| b.id());
    effects_meta
        .sim_params_uniforms
        .write_buffer(&render_device, &render_queue);
    if prev_buffer_id != effects_meta.sim_params_uniforms.buffer().map(|b| b.id()) {
        // Buffer changed, invalidate bind groups
        effects_meta.sim_params_bind_group = None;
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
    /// The optional binding behind the RENDER_NEEDS_SPAWNER conditional exists
    /// only when local space simulation is used, to get access to the emitter
    /// transform.
    ///
    /// ```wgsl
    /// @binding(0) var<storage, read> particle_buffer : ParticleBuffer;
    /// @binding(1) var<storage, read> indirect_buffer : IndirectBuffer;
    /// @binding(2) var<storage, read> dispatch_indirect : UpdateIndirectDispatch;
    /// #ifdef RENDER_NEEDS_SPAWNER
    /// @binding(3) var<storage, read> spawner : Spawner;
    /// #endif
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
    ) -> Vec<BindGroupEntry<'a>> {
        self.textures
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
            .collect()
    }
}

#[derive(Default, Resource)]
pub struct EffectBindGroups {
    /// Map from buffer index to the bind groups shared among all effects that
    /// use that buffer.
    particle_buffers: HashMap<u32, BufferBindGroups>,
    /// Map of bind groups for image assets used as particle textures.
    images: HashMap<AssetId<Image>, BindGroup>,
    /// Map from effect index to its indirect init bind group. Only present if
    /// the effect is a child effect driven by GPU spawn events.
    // init_indirect_bind_groups: HashMap<EffectCacheId, BindGroup>,
    /// Map from effect index to its update render indirect bind group (group
    /// 3).
    update_render_indirect_bind_groups: HashMap<EffectCacheId, BindGroup>,
    /// Map from an effect material to its bind group.
    material_bind_groups: HashMap<Material, BindGroup>,
    /// Map from an event buffer index to the bind group for the init fill pass
    /// in charge of filling all its init dispatches.
    init_fill_dispatch: HashMap<u32, BindGroup>,
}

impl EffectBindGroups {
    pub fn particle_render(&self, buffer_index: u32) -> Option<&BindGroup> {
        self.particle_buffers
            .get(&buffer_index)
            .map(|bg| &bg.render)
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
    #[system_param(ignore)]
    marker: PhantomData<&'s usize>,
}

fn emit_sorted_draw<T, F>(
    views: &Query<(Entity, &VisibleEntities, &ExtractedView)>,
    render_phases: &mut ResMut<ViewSortedRenderPhases<T>>,
    view_entities: &mut FixedBitSet,
    effect_batches: &Query<(Entity, &mut EffectBatches)>,
    effect_draw_batches: &Query<(Entity, &mut EffectDrawBatch)>,
    render_pipeline: &mut ParticlesRenderPipeline,
    mut specialized_render_pipelines: Mut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    pipeline_cache: &PipelineCache,
    msaa_samples: u32,
    make_phase_item: F,
    #[cfg(all(feature = "2d", feature = "3d"))] pipeline_mode: PipelineMode,
) where
    T: SortedPhaseItem,
    F: Fn(CachedRenderPipelineId, Entity, &EffectDrawBatch, u32, &ExtractedView) -> T,
{
    trace!("emit_sorted_draw() {} views", views.iter().len());

    for (view_entity, visible_entities, view) in views.iter() {
        trace!("Process new sorted view");

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
                    .map(|e| e.index() as usize),
            );
        }

        // For each view, loop over all the effect batches to determine if the effect
        // needs to be rendered for that view, and enqueue a view-dependent
        // batch if so.
        for (draw_entity, draw_batch) in effect_draw_batches.iter() {
            #[cfg(feature = "trace")]
            let _span_draw = bevy::utils::tracing::info_span!("draw_batch").entered();

            trace!(
                "Process draw batch: draw_entity={:?} group_index={} batches_entity={:?}",
                draw_entity,
                draw_batch.group_index,
                draw_batch.batches_entity,
            );

            // Get the EffectBatches this EffectDrawBatch is part of.
            let Ok((batches_entity, batches)) = effect_batches.get(draw_batch.batches_entity)
            else {
                continue;
            };

            trace!(
                "-> EffectBaches: entity={:?} buffer_index={} spawner_base={} layout_flags={:?}",
                batches_entity,
                batches.buffer_index,
                batches.spawner_base,
                batches.layout_flags,
            );

            // AlphaMask is a binned draw, so no sorted draw can possibly use it
            if batches.layout_flags.contains(LayoutFlags::USE_ALPHA_MASK) {
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
            let has_visible_entity = batches
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
            render_pipeline.cache_material(&batches.texture_layout);

            // FIXME - We draw the entire batch, but part of it may not be visible in this
            // view! We should re-batch for the current view specifically!

            let local_space_simulation = batches
                .layout_flags
                .contains(LayoutFlags::LOCAL_SPACE_SIMULATION);
            let use_alpha_mask = batches.layout_flags.contains(LayoutFlags::USE_ALPHA_MASK);
            let flipbook = batches.layout_flags.contains(LayoutFlags::FLIPBOOK);
            let needs_uv = batches.layout_flags.contains(LayoutFlags::NEEDS_UV);
            let ribbons = batches.layout_flags.contains(LayoutFlags::RIBBONS);
            let image_count = batches.texture_layout.layout.len() as u8;

            // Specialize the render pipeline based on the effect batch
            trace!(
                "Specializing render pipeline: render_shaders={:?} image_count={} use_alpha_mask={:?} flipbook={:?} hdr={}",
                batches.render_shaders,
                image_count,
                use_alpha_mask,
                flipbook,
                view.hdr
            );

            // Add a draw pass for the effect batch
            trace!("Emitting individual draws for batches and groups: group_batches.len()={} batches.render_shaders.len()={}", batches.group_batches.len(), batches.render_shaders.len());
            let render_shader_source = &batches.render_shaders[draw_batch.group_index as usize];
            trace!("Emit for group index #{}", draw_batch.group_index);

            let alpha_mode = batches.alpha_mode;

            #[cfg(feature = "trace")]
            let _span_specialize = bevy::utils::tracing::info_span!("specialize").entered();
            let render_pipeline_id = specialized_render_pipelines.specialize(
                pipeline_cache,
                render_pipeline,
                ParticleRenderPipelineKey {
                    shader: render_shader_source.clone(),
                    particle_layout: batches.particle_layout.clone(),
                    texture_layout: batches.texture_layout.clone(),
                    local_space_simulation,
                    use_alpha_mask,
                    alpha_mode,
                    flipbook,
                    needs_uv,
                    ribbons,
                    #[cfg(all(feature = "2d", feature = "3d"))]
                    pipeline_mode,
                    msaa_samples,
                    hdr: view.hdr,
                },
            );
            #[cfg(feature = "trace")]
            _span_specialize.exit();

            trace!(
                "+ Render pipeline specialized: id={:?} -> group_index={}",
                render_pipeline_id,
                draw_batch.group_index
            );
            trace!(
                "+ Add Transparent for batch on draw_entity {:?}: buffer_index={} \
                group_index={} spawner_base={} handle={:?}",
                draw_entity,
                batches.buffer_index,
                draw_batch.group_index,
                batches.spawner_base,
                batches.handle
            );
            render_phase.add(make_phase_item(
                render_pipeline_id,
                draw_entity,
                draw_batch,
                draw_batch.group_index,
                view,
            ));
        }
    }
}

#[cfg(feature = "3d")]
fn emit_binned_draw<T, F>(
    views: &Query<(Entity, &VisibleEntities, &ExtractedView)>,
    render_phases: &mut ResMut<ViewBinnedRenderPhases<T>>,
    view_entities: &mut FixedBitSet,
    effect_batches: &Query<(Entity, &mut EffectBatches)>,
    effect_draw_batches: &Query<(Entity, &mut EffectDrawBatch)>,
    render_pipeline: &mut ParticlesRenderPipeline,
    mut specialized_render_pipelines: Mut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    pipeline_cache: &PipelineCache,
    msaa_samples: u32,
    make_bin_key: F,
    #[cfg(all(feature = "2d", feature = "3d"))] pipeline_mode: PipelineMode,
    use_alpha_mask: bool,
) where
    T: BinnedPhaseItem,
    F: Fn(CachedRenderPipelineId, &EffectDrawBatch, u32, &ExtractedView) -> T::BinKey,
{
    use bevy::render::render_phase::BinnedRenderPhaseType;

    trace!("emit_binned_draw() {} views", views.iter().len());

    for (view_entity, visible_entities, view) in views.iter() {
        trace!(
            "Process new binned view (use_alpha_mask={})",
            use_alpha_mask
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
                    .map(|e| e.index() as usize),
            );
        }

        // For each view, loop over all the effect batches to determine if the effect
        // needs to be rendered for that view, and enqueue a view-dependent
        // batch if so.
        for (draw_entity, draw_batch) in effect_draw_batches.iter() {
            #[cfg(feature = "trace")]
            let _span_draw = bevy::utils::tracing::info_span!("draw_batch").entered();

            trace!(
                "Process draw batch: draw_entity={:?} group_index={} batches_entity={:?}",
                draw_entity,
                draw_batch.group_index,
                draw_batch.batches_entity,
            );

            // Get the EffectBatches this EffectDrawBatch is part of.
            let Ok((batches_entity, batches)) = effect_batches.get(draw_batch.batches_entity)
            else {
                continue;
            };

            trace!(
                "-> EffectBaches: entity={:?} buffer_index={} spawner_base={} layout_flags={:?}",
                batches_entity,
                batches.buffer_index,
                batches.spawner_base,
                batches.layout_flags,
            );

            if use_alpha_mask != batches.layout_flags.contains(LayoutFlags::USE_ALPHA_MASK) {
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
            let has_visible_entity = batches
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
            render_pipeline.cache_material(&batches.texture_layout);

            // FIXME - We draw the entire batch, but part of it may not be visible in this
            // view! We should re-batch for the current view specifically!

            let local_space_simulation = batches
                .layout_flags
                .contains(LayoutFlags::LOCAL_SPACE_SIMULATION);
            let use_alpha_mask = batches.layout_flags.contains(LayoutFlags::USE_ALPHA_MASK);
            let flipbook = batches.layout_flags.contains(LayoutFlags::FLIPBOOK);
            let needs_uv = batches.layout_flags.contains(LayoutFlags::NEEDS_UV);
            let ribbons = batches.layout_flags.contains(LayoutFlags::RIBBONS);
            let image_count = batches.texture_layout.layout.len() as u8;

            // Specialize the render pipeline based on the effect batch
            trace!(
                "Specializing render pipeline: render_shaders={:?} image_count={} use_alpha_mask={:?} flipbook={:?} hdr={}",
                batches.render_shaders,
                image_count,
                use_alpha_mask,
                flipbook,
                view.hdr
            );

            // Add a draw pass for the effect batch
            trace!("Emitting individual draws for batches and groups: group_batches.len()={} batches.render_shaders.len()={}", batches.group_batches.len(), batches.render_shaders.len());
            let render_shader_source = &batches.render_shaders[draw_batch.group_index as usize];
            trace!("Emit for group index #{}", draw_batch.group_index);

            let alpha_mode = batches.alpha_mode;

            #[cfg(feature = "trace")]
            let _span_specialize = bevy::utils::tracing::info_span!("specialize").entered();
            let render_pipeline_id = specialized_render_pipelines.specialize(
                pipeline_cache,
                render_pipeline,
                ParticleRenderPipelineKey {
                    shader: render_shader_source.clone(),
                    particle_layout: batches.particle_layout.clone(),
                    texture_layout: batches.texture_layout.clone(),
                    local_space_simulation,
                    use_alpha_mask,
                    alpha_mode,
                    flipbook,
                    needs_uv,
                    ribbons,
                    #[cfg(all(feature = "2d", feature = "3d"))]
                    pipeline_mode,
                    msaa_samples,
                    hdr: view.hdr,
                },
            );
            #[cfg(feature = "trace")]
            _span_specialize.exit();

            trace!(
                "+ Render pipeline specialized: id={:?} -> group_index={}",
                render_pipeline_id,
                draw_batch.group_index
            );
            trace!(
                "+ Add Transparent for batch on draw_entity {:?}: buffer_index={} \
                group_index={} spawner_base={} handle={:?}",
                draw_entity,
                batches.buffer_index,
                draw_batch.group_index,
                batches.spawner_base,
                batches.handle
            );
            render_phase.add(
                make_bin_key(render_pipeline_id, draw_batch, draw_batch.group_index, view),
                draw_entity,
                BinnedRenderPhaseType::NonMesh,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn queue_effects(
    views: Query<(Entity, &VisibleEntities, &ExtractedView)>,
    effects_meta: Res<EffectsMeta>,
    mut render_pipeline: ResMut<ParticlesRenderPipeline>,
    mut specialized_render_pipelines: ResMut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    effect_batches: Query<(Entity, &mut EffectBatches)>,
    effect_draw_batches: Query<(Entity, &mut EffectDrawBatch)>,
    events: Res<EffectAssetEvents>,
    read_params: QueueEffectsReadOnlyParams,
    msaa: Res<Msaa>,
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
                &effect_batches,
                &effect_draw_batches,
                &mut render_pipeline,
                specialized_render_pipelines.reborrow(),
                &pipeline_cache,
                msaa.samples(),
                |id, entity, draw_batch, _group, _view| Transparent2d {
                    draw_function: draw_effects_function_2d,
                    pipeline: id,
                    entity,
                    sort_key: draw_batch.z_sort_key_2d,
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
                &effect_batches,
                &effect_draw_batches,
                &mut render_pipeline,
                specialized_render_pipelines.reborrow(),
                &pipeline_cache,
                msaa.samples(),
                |id, entity, batch, _group, view| Transparent3d {
                    draw_function: draw_effects_function_3d,
                    pipeline: id,
                    entity,
                    distance: view
                        .rangefinder3d()
                        .distance_translation(&batch.translation_3d),
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
                &effect_batches,
                &effect_draw_batches,
                &mut render_pipeline,
                specialized_render_pipelines.reborrow(),
                &pipeline_cache,
                msaa.samples(),
                |id, _batch, _group, _view| OpaqueNoLightmap3dBinKey {
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
                true,
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
    mut effect_cache: ResMut<EffectCache>,
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

    // Re-/allocate any GPU buffer needed by the effect cache
    effect_cache.prepare_buffers(&render_device, &render_queue);
}

pub(crate) fn prepare_bind_groups(
    mut effects_meta: ResMut<EffectsMeta>,
    mut effect_cache: ResMut<EffectCache>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    effect_batches: Query<(Entity, &mut EffectBatches)>,
    render_device: Res<RenderDevice>,
    dispatch_indirect_pipeline: Res<DispatchIndirectPipeline>,
    utils_pipeline: Res<UtilsPipeline>,
    init_pipeline: Res<ParticlesInitPipeline>,
    update_pipeline: Res<ParticlesUpdatePipeline>,
    render_pipeline: ResMut<ParticlesRenderPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    gpu_buffer_operation_queue: Res<GpuBufferOperationQueue>,
) {
    // If there's no spawner buffer, there can't be any effect allocated, so skip
    // this system entirely.
    if effects_meta.spawner_buffer.is_empty() || effects_meta.spawner_buffer.buffer().is_none() {
        return;
    }

    {
        #[cfg(feature = "trace")]
        let _span = bevy::utils::tracing::info_span!("shared_bind_groups").entered();

        // Create the bind group for the global simulation parameters
        if effects_meta.sim_params_bind_group.is_none() {
            effects_meta.sim_params_bind_group = Some(render_device.create_bind_group(
                "hanabi:bind_group_sim_params",
                &update_pipeline.sim_params_layout, /* FIXME - Shared with vfx_update, is
                                                     * that OK? */
                &[BindGroupEntry {
                    binding: 0,
                    resource: effects_meta.sim_params_uniforms.binding().unwrap(),
                }],
            ));
        }

        // Create the bind group for the spawner parameters
        // FIXME - This is shared by init and update; should move
        // "update_pipeline.spawner_buffer_layout" out of "update_pipeline"
        trace!(
            "Spawner buffer bind group: size={} aligned_size={}",
            GpuSpawnerParams::min_size().get(),
            effects_meta.spawner_buffer.aligned_size()
        );
        assert!(
            effects_meta.spawner_buffer.aligned_size()
                >= GpuSpawnerParams::min_size().get() as usize
        );
        // Note: we clear effects_meta.spawner_buffer each frame in prepare_effects(),
        // so this bind group is always invalid at the minute and always needs
        // re-creation.
        effects_meta.spawner_bind_group = effects_meta.spawner_buffer.buffer().map(|buffer| {
            render_device.create_bind_group(
                "hanabi:bind_group_spawner_buffer",
                &update_pipeline.spawner_buffer_layout, // FIXME - Shared with init,is that OK?
                &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer,
                        offset: 0,
                        size: Some(
                            NonZeroU64::new(effects_meta.spawner_buffer.aligned_size() as u64)
                                .unwrap(),
                        ),
                    }),
                }],
            )
        });

        // Create the bind group for the indirect dispatch of all effects
        effects_meta.dr_indirect_bind_group = match (
            effects_meta.render_effect_dispatch_buffer.buffer(),
            effects_meta.render_group_dispatch_buffer.buffer(),
            effects_meta.dispatch_indirect_buffer.buffer(),
            effects_meta.particle_group_buffer.buffer(),
            effects_meta.spawner_buffer.buffer(),
        ) {
            (
                Some(render_effect_dispatch_buffer),
                Some(render_group_dispatch_buffer),
                Some(dispatch_indirect_buffer),
                Some(particle_group_buffer),
                Some(spawner_buffer),
            ) => {
                Some(render_device.create_bind_group(
                    "hanabi:bind_group_vfx_indirect_dr_indirect",
                    &dispatch_indirect_pipeline.dispatch_indirect_layout,
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: render_effect_dispatch_buffer,
                                offset: 0,
                                size: None, //NonZeroU64::new(256), // Some(GpuRenderIndirect::min_size()),
                            }),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: render_group_dispatch_buffer,
                                offset: 0,
                                size: None, //NonZeroU64::new(256), // Some(GpuRenderIndirect::min_size()),
                            }),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: dispatch_indirect_buffer,
                                offset: 0,
                                size: None, //NonZeroU64::new(256), // Some(GpuDispatchIndirect::min_size()),
                            }),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: particle_group_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: spawner_buffer,
                                offset: 0,
                                size: None,
                            }),
                        },
                    ],
                ))
            }
            _ => None,
        };

        // Create the bind group for the indirect render buffer use in the init shader
        let init_render_indirect_bind_group = match (
            effects_meta.render_effect_dispatch_buffer.buffer(),
            effects_meta.render_group_dispatch_buffer.buffer(),
        ) {
            (Some(render_effect_dispatch_buffer), Some(render_group_dispatch_buffer)) => {
                Some(render_device.create_bind_group(
                    "hanabi:bind_group_init_render_dispatch",
                    &init_pipeline.render_indirect_layout,
                    &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: render_effect_dispatch_buffer,
                                offset: 0,
                                size: Some(effects_meta.gpu_limits.render_effect_indirect_size()),
                            }),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: render_group_dispatch_buffer,
                                offset: 0,
                                size: Some(effects_meta.gpu_limits.render_group_indirect_size()),
                            }),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: render_group_dispatch_buffer,
                                offset: 0,
                                size: Some(effects_meta.gpu_limits.render_group_indirect_size()),
                            }),
                        },
                    ],
                ))
            }

            (_, _) => None,
        };
        effects_meta.init_render_indirect_bind_group = init_render_indirect_bind_group;
    }

    // Make a copy of the buffer ID before borrowing effects_meta mutably in the
    // loop below
    let Some(indirect_buffer) = effects_meta.dispatch_indirect_buffer.buffer().cloned() else {
        return;
    };
    let Some(spawner_buffer) = effects_meta.spawner_buffer.buffer().cloned() else {
        return;
    };

    // Create the per-buffer bind groups
    trace!("Create per-buffer bind groups...");
    for (buffer_index, buffer) in effect_cache.buffers().iter().enumerate() {
        #[cfg(feature = "trace")]
        let _span_buffer = bevy::utils::tracing::info_span!("create_buffer_bind_groups").entered();

        let Some(buffer) = buffer else {
            trace!(
                "Effect buffer index #{} has no allocated EffectBuffer, skipped.",
                buffer_index
            );
            continue;
        };

        // Ensure all effects in this batch have a bind group for the entire buffer of
        // the group, since the update phase runs on an entire group/buffer at
        // once, with all the effect instances in it batched together.
        trace!("effect particle buffer_index=#{}", buffer_index);
        effect_bind_groups
            .particle_buffers
            .entry(buffer_index as u32)
            .or_insert_with(|| {
                trace!(
                    "Create new particle bind groups for buffer_index={} | property_layout {:?}",
                    buffer_index,
                    buffer.property_layout(),
                );

                let dispatch_indirect_size = GpuDispatchIndirect::aligned_size(
                    render_device.limits().min_storage_buffer_offset_alignment,
                );
                let mut entries = vec![
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
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &indirect_buffer,
                            offset: 0,
                            size: Some(dispatch_indirect_size),
                        }),
                    },
                ];
                if buffer
                    .layout_flags()
                    .contains(LayoutFlags::LOCAL_SPACE_SIMULATION)
                {
                    entries.push(BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &spawner_buffer,
                            offset: 0,
                            size: Some(GpuSpawnerParams::min_size()),
                        }),
                    });
                }
                trace!(
                    "Creating render bind group with {} entries (layout flags: {:?})",
                    entries.len(),
                    buffer.layout_flags()
                );
                let render = render_device.create_bind_group(
                    &format!("hanabi:bind_group:render_vfx{buffer_index}_particles")[..],
                    buffer.particle_layout_bind_group_with_dispatch(),
                    &entries,
                );

                BufferBindGroups { render }
            });
    }

    // Create the per-event-buffer bind groups
    for (event_buffer_index, event_buffer) in effect_cache.event_buffers() {
        // Check if the entry is missing
        let entry = effect_bind_groups
            .init_fill_dispatch
            .entry(event_buffer_index);
        if matches!(entry, Entry::Vacant(_)) {
            trace!(
                "Event buffer #{} missing a bind group for init fill args. Trying to create now...",
                event_buffer_index
            );

            // Check if the binding is available to create the bind group and fill the entry
            let Some((args_binding, args_count)) =
                gpu_buffer_operation_queue.args_buffer_binding(event_buffer_index)
            else {
                continue;
            };

            let child_id = event_buffer.effect_cache_id();
            let Some(source_binding) = effect_cache.get_child_info_binding(child_id) else {
                warn!("Event buffer #{event_buffer_index} for effect #{child_id:?} has {args_count} operations pending, but the effect cache has no get_child_info_binding for the source buffer. Discarding event operations for this frame. This will result in particles not spawning.");
                continue;
            };

            let Some(target_binding) = effect_cache.init_indirect_dispatch_buffer_binding() else {
                warn!("Event buffer #{event_buffer_index} for effect #{child_id:?} has {args_count} operations pending, but the effect cache has no init_indirect_dispatch_buffer_binding for the target buffer. Discarding event operations for this frame. This will result in particles not spawning.");
                continue;
            };

            // Actually create the new bind group entry
            entry.insert(render_device.create_bind_group(
                &format!("hanabi:bind_group:event{event_buffer_index}_init_fill_dispatch")[..],
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
                        resource: BindingResource::Buffer(source_binding),
                    },
                    // @group(0) @binding(2) var<storage, read_write> dst_buffer :
                    // array<u32>
                    BindGroupEntry {
                        binding: 2,
                        resource: target_binding,
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
    for (_entity, effect_batches) in effect_batches.iter() {
        #[cfg(feature = "trace")]
        let _span_buffer = bevy::utils::tracing::info_span!("create_batch_bind_groups").entered();

        let effect_cache_id = effect_batches.effect_cache_id;

        // Convert indirect buffer offsets from indices to bytes.
        let first_effect_particle_group_buffer_offset = effects_meta
            .gpu_limits
            .particle_group_offset(effect_batches.first_particle_group_buffer_index)
            as u64;
        let effect_particle_groups_buffer_size = NonZeroU64::try_from(
            u32::from(effects_meta.gpu_limits.particle_group_aligned_size) as u64
                * effect_batches.group_batches.len() as u64,
        )
        .unwrap();
        let group_binding = BufferBinding {
            buffer: effects_meta.particle_group_buffer.buffer().unwrap(),
            offset: first_effect_particle_group_buffer_offset,
            size: Some(effect_particle_groups_buffer_size),
        };

        // Bind group for the init compute shader to simulate particles.
        let effect_buffer_binding =
            effect_cache.get_event_buffer_binding(effect_batches.effect_cache_id);
        if !effect_cache.ensure_init_bind_group(
            effect_batches.effect_cache_id,
            effect_batches.buffer_index,
            effect_batches.parent_buffer_index,
            group_binding.clone(),
            effect_buffer_binding.as_ref().map(|ebb| ebb.binding()),
        ) {
            continue;
        }

        // Bind group for the update compute shader to simulate particles.
        if !effect_cache.ensure_update_bind_group(
            effect_batches.effect_cache_id,
            effect_batches.buffer_index,
            group_binding,
            &effect_batches.child_effects[..],
        ) {
            continue;
        }

        if effect_bind_groups
            .update_render_indirect_bind_groups
            .get(&effect_cache_id)
            .is_none()
        {
            let particles_buffer_layout_update_render_indirect = render_device.create_bind_group(
                "hanabi:bind_group_update_render_group_dispatch",
                &update_pipeline.render_indirect_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: effects_meta.render_effect_dispatch_buffer.buffer().unwrap(),
                            offset: 0,
                            size: None,
                        }),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: effects_meta.render_group_dispatch_buffer.buffer().unwrap(),
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            );

            effect_bind_groups
                .update_render_indirect_bind_groups
                .insert(
                    effect_cache_id,
                    particles_buffer_layout_update_render_indirect,
                );
        }

        // Ensure the particle texture(s) are available as GPU resources and that a bind
        // group for them exists FIXME fix this insert+get below
        if !effect_batches.texture_layout.layout.is_empty() {
            // This should always be available, as this is cached into the render pipeline
            // just before we start specializing it.
            let Some(material_bind_group_layout) =
                render_pipeline.get_material(&effect_batches.texture_layout)
            else {
                error!(
                    "Failed to find material bind group layout for buffer #{}",
                    effect_batches.buffer_index
                );
                continue;
            };

            // TODO = move
            let material = Material {
                layout: effect_batches.texture_layout.clone(),
                textures: effect_batches.textures.iter().map(|h| h.id()).collect(),
            };
            assert_eq!(material.layout.layout.len(), material.textures.len());

            let bind_group_entries = material.make_entries(&gpu_images);

            effect_bind_groups
                .material_bind_groups
                .entry(material.clone())
                .or_insert_with(|| {
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
    SQuery<Read<ViewUniformOffset>>,
    SQuery<Read<EffectBatches>>,
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
    entity: Entity,
    pipeline_id: CachedRenderPipelineId,
    params: &mut DrawEffectsSystemState,
) {
    let (effects_meta, effect_bind_groups, pipeline_cache, views, effects, effect_draw_batches) =
        params.get(world);
    let view_uniform = views.get(view).unwrap();
    let effects_meta = effects_meta.into_inner();
    let effect_bind_groups = effect_bind_groups.into_inner();
    let effect_draw_batch = effect_draw_batches.get(entity).unwrap();
    let effect_batches = effects.get(effect_draw_batch.batches_entity).unwrap();

    let gpu_limits = &effects_meta.gpu_limits;

    let Some(pipeline) = pipeline_cache.into_inner().get_render_pipeline(pipeline_id) else {
        return;
    };

    trace!("render pass");

    pass.set_render_pipeline(pipeline);

    // Vertex buffer containing the particle model to draw. Generally a quad.
    pass.set_vertex_buffer(0, effects_meta.vertices.buffer().unwrap().slice(..));

    // View properties (camera matrix, etc.)
    pass.set_bind_group(
        0,
        effects_meta.view_bind_group.as_ref().unwrap(),
        &[view_uniform.offset],
    );

    // Particles buffer
    let dispatch_indirect_offset = gpu_limits.dispatch_indirect_offset(effect_batches.buffer_index);
    trace!(
        "set_bind_group(1): dispatch_indirect_offset={}",
        dispatch_indirect_offset
    );
    let spawner_base = effect_batches.spawner_base;
    let spawner_buffer_aligned = effects_meta.spawner_buffer.aligned_size();
    assert!(spawner_buffer_aligned >= GpuSpawnerParams::min_size().get() as usize);
    let spawner_offset = spawner_base * spawner_buffer_aligned as u32;
    let dyn_uniform_indices: [u32; 2] = [dispatch_indirect_offset, spawner_offset];
    let dyn_uniform_indices = if effect_batches
        .layout_flags
        .contains(LayoutFlags::LOCAL_SPACE_SIMULATION)
    {
        &dyn_uniform_indices
    } else {
        &dyn_uniform_indices[..1]
    };
    pass.set_bind_group(
        1,
        effect_bind_groups
            .particle_render(effect_batches.buffer_index)
            .unwrap(),
        dyn_uniform_indices,
    );

    // Particle texture
    // TODO = move
    let material = Material {
        layout: effect_batches.texture_layout.clone(),
        textures: effect_batches.textures.iter().map(|h| h.id()).collect(),
    };
    if !effect_batches.texture_layout.layout.is_empty() {
        if let Some(bind_group) = effect_bind_groups.material_bind_groups.get(&material) {
            pass.set_bind_group(2, bind_group, &[]);
        } else {
            // Texture(s) not ready; skip this drawing for now
            trace!(
                "Particle material bind group not available for batch buf={}. Skipping draw call.",
                effect_batches.buffer_index,
            );
            return; // continue;
        }
    }

    let render_indirect_buffer = effects_meta.render_group_dispatch_buffer.buffer().unwrap();
    let group_index = effect_draw_batch.group_index;
    let effect_batch = &effect_batches.group_batches[group_index as usize];

    let render_group_dispatch_indirect_index = effect_batches
        .dispatch_buffer_indices
        .first_render_group_dispatch_buffer_index
        .0
        + group_index;

    trace!(
        "Draw up to {} particles with {} vertices per particle for batch from buffer #{} \
            (render_group_dispatch_indirect_index={:?}, group_index={}).",
        effect_batch.slice.len(),
        effects_meta.vertices.len(),
        effect_batches.buffer_index,
        render_group_dispatch_indirect_index,
        group_index,
    );

    pass.draw_indirect(
        render_indirect_buffer,
        render_group_dispatch_indirect_index as u64
            * u32::from(gpu_limits.render_group_indirect_aligned_size) as u64,
    );
}

#[cfg(feature = "2d")]
impl Draw<Transparent2d> for DrawEffects {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        item: &Transparent2d,
    ) {
        trace!("Draw<Transparent2d>: view={:?}", view);
        draw(
            world,
            pass,
            view,
            item.entity,
            item.pipeline,
            &mut self.params,
        );
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
    ) {
        trace!("Draw<Transparent3d>: view={:?}", view);
        draw(
            world,
            pass,
            view,
            item.entity,
            item.pipeline,
            &mut self.params,
        );
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
    ) {
        trace!("Draw<AlphaMask3d>: view={:?}", view);
        draw(
            world,
            pass,
            view,
            item.representative_entity,
            item.key.pipeline,
            &mut self.params,
        );
    }
}

fn create_init_particles_bind_group_layout(
    render_device: &RenderDevice,
    label: &str,
    particle_layout_min_binding_size: NonZero<u64>,
    property_layout_min_binding_size: Option<NonZero<u64>>,
) -> BindGroupLayout {
    let mut entries = Vec::with_capacity(3);
    // (1,0) ParticleBuffer
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
    // (1,1) IndirectBuffer
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
    // (1,2) array<ParticleGroup>
    let particle_group_size =
        GpuParticleGroup::aligned_size(render_device.limits().min_storage_buffer_offset_alignment);
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
    if let Some(min_binding_size) = property_layout_min_binding_size {
        // (1,3) Properties
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

    trace!(
        "Creating particle bind group layout '{}' for init pass with {} entries.",
        label,
        entries.len()
    );
    render_device.create_bind_group_layout(label, &entries)
}

fn create_init_render_indirect_bind_group_layout(
    render_device: &RenderDevice,
    label: &str,
    clone: bool,
) -> BindGroupLayout {
    let storage_alignment = render_device.limits().min_storage_buffer_offset_alignment;
    let render_effect_indirect_size = GpuRenderEffectMetadata::aligned_size(storage_alignment);
    let render_group_indirect_size = GpuRenderGroupIndirect::aligned_size(storage_alignment);

    let mut entries = vec![
        // @binding(0) var<storage, read_write> render_effect_indirect :
        // RenderEffectMetadata
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: true,
                min_binding_size: Some(render_effect_indirect_size),
            },
            count: None,
        },
        // @binding(1) var<storage, read_write> dest_render_group_indirect : RenderGroupIndirect
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: true,
                min_binding_size: Some(render_group_indirect_size),
            },
            count: None,
        },
    ];

    if clone {
        // @binding(2) var<storage, read_write> src_render_group_indirect :
        // RenderGroupIndirect
        entries.push(BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: true,
                min_binding_size: Some(render_group_indirect_size),
            },
            count: None,
        });
    }

    render_device.create_bind_group_layout(label, &entries)
}

fn create_update_bind_group_layout(
    render_device: &RenderDevice,
    label: &str,
    particle_layout_min_binding_size: NonZero<u64>,
    property_layout_min_binding_size: Option<NonZero<u64>>,
) -> BindGroupLayout {
    let particle_group_size =
        GpuParticleGroup::aligned_size(render_device.limits().min_storage_buffer_offset_alignment);
    let mut entries = vec![
        // @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: Some(particle_layout_min_binding_size),
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
                min_binding_size: BufferSize::new(INDIRECT_INDEX_SIZE as _),
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
                min_binding_size: Some(particle_group_size),
            },
            count: None,
        },
    ];
    if let Some(property_layout_min_binding_size) = property_layout_min_binding_size {
        // @binding(3) var<storage, read> properties : Properties
        entries.push(BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false, // TODO
                min_binding_size: Some(property_layout_min_binding_size),
            },
            count: None,
        });
    }

    trace!(
        "Creating particle bind group layout '{}' for update pass with {} entries.",
        label,
        entries.len()
    );
    render_device.create_bind_group_layout(label, &entries)
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

/// Render node to run the simulation of all effects once per frame.
///
/// Runs inside the simulation sub-graph, looping over all extracted effect
/// batches to simulate them.
pub(crate) struct VfxSimulateNode {
    /// Query to retrieve the batches of effects to simulate and render.
    effect_query: QueryState<(Entity, Read<EffectBatches>)>,
}

impl VfxSimulateNode {
    /// Output particle buffer for that view. TODO - how to handle multiple
    /// buffers?! Should use Entity instead??
    // pub const OUT_PARTICLE_BUFFER: &'static str = "particle_buffer";

    /// Create a new node for simulating the effects of the given world.
    pub fn new(world: &mut World) -> Self {
        Self {
            effect_query: QueryState::new(world),
        }
    }
}

impl Node for VfxSimulateNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![]
    }

    fn update(&mut self, world: &mut World) {
        trace!("VfxSimulateNode::update()");
        self.effect_query.update_archetypes(world);
    }

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
        let dispatch_indirect_pipeline = world.resource::<DispatchIndirectPipeline>();
        let utils_pipeline = world.resource::<UtilsPipeline>();
        let effect_cache = world.resource::<EffectCache>();

        // Make sure to schedule any buffer copy from changed effects before accessing
        // them later in the GPU execution below.
        {
            let command_encoder = render_context.command_encoder();
            effects_meta
                .dispatch_indirect_buffer
                .write_buffer(command_encoder);
            effects_meta
                .render_effect_dispatch_buffer
                .write_buffer(command_encoder);
            effects_meta
                .render_group_dispatch_buffer
                .write_buffer(command_encoder);
        }

        // Compute init fill dispatch pass - Fill the indirect dispatch structs for any
        // upcoming init pass of this frame, based on the GPU spawn events emitted by
        // the update pass of their parent effect during the previous frame.
        {
            let gpu_buffer_operation_queue = world.resource::<GpuBufferOperationQueue>();
            gpu_buffer_operation_queue.dispatch_init_fill(
                render_context,
                utils_pipeline.get_pipeline(GpuBufferOperationType::FillDispatchArgs),
                effect_bind_groups,
            );
        }

        // Compute init pass
        // let mut total_group_count = 0;
        {
            {
                trace!("init: loop over effect batches...");

                // Dispatch init compute jobs for all effects
                for (entity, batches) in self.effect_query.iter_manual(world) {
                    // For each group in this effect, dispatch its init pass
                    for &dest_group_index in batches.group_order.iter() {
                        let initializer = &batches.initializers[dest_group_index as usize];

                        let dest_render_group_dispatch_buffer_index = BufferTableId(
                            batches
                                .dispatch_buffer_indices
                                .first_render_group_dispatch_buffer_index
                                .0
                                + dest_group_index,
                        );

                        // Destination group spawners are packed one after one another.
                        let spawner_base = batches.spawner_base + dest_group_index;
                        let spawner_buffer_aligned = effects_meta.spawner_buffer.aligned_size();
                        assert!(
                            spawner_buffer_aligned >= GpuSpawnerParams::min_size().get() as usize
                        );
                        let spawner_offset = spawner_base * spawner_buffer_aligned as u32;

                        match initializer {
                            EffectInitializer::Spawner(effect_spawner) => {
                                let mut compute_pass = render_context
                                    .command_encoder()
                                    .begin_compute_pass(&ComputePassDescriptor {
                                        label: Some("hanabi:init"),
                                        timestamp_writes: None,
                                    });

                                let render_effect_dispatch_buffer_index = batches
                                    .dispatch_buffer_indices
                                    .render_effect_metadata_buffer_index;

                                // FIXME - Currently we unconditionally count
                                // all groups because the dispatch pass always
                                // runs on all groups. We should consider if
                                // it's worth skipping e.g. dormant or finished
                                // effects at the cost of extra complexity.
                                // total_group_count += batches.group_batches.len() as u32;

                                let pipeline_id = batches.init_and_update_pipeline_ids
                                    [dest_group_index as usize]
                                    .init;
                                let Some(init_pipeline) =
                                    pipeline_cache.get_compute_pipeline(pipeline_id)
                                else {
                                    // if let CachedPipelineState::Err(err) = pipeline_cache
                                    //     .get_compute_pipeline_state(
                                    //         batches.init_and_update_pipeline_ids
                                    //             [dest_group_index as usize]
                                    //             .init,
                                    //     )
                                    // {
                                    //     error!(
                                    //         "Failed to find init pipeline #{} for effect {:?}: \
                                    //         {:?}",
                                    //         batches.init_and_update_pipeline_ids
                                    //             [dest_group_index as usize]
                                    //             .init
                                    //             .id(),
                                    //         entity,
                                    //         err
                                    //     );
                                    // }
                                    continue;
                                };

                                let indirect_dispatch = batches
                                    .layout_flags
                                    .contains(LayoutFlags::CONSUME_GPU_SPAWN_EVENTS);

                                // Do not dispatch any init work if there's nothing to spawn this
                                // frame
                                let spawn_count = effect_spawner.spawn_count;
                                if spawn_count == 0 && !indirect_dispatch {
                                    trace!("-> init batch empty (spawn_count={spawn_count}, indirect_dispatch={indirect_dispatch})");
                                    continue;
                                }

                                const WORKGROUP_SIZE: u32 = 64;
                                let workgroup_count =
                                    (spawn_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

                                let effect_cache_id = batches.effect_cache_id;

                                // for (effect_entity, effect_slice) in
                                // effects_meta.entity_map.iter()
                                // Retrieve the ExtractedEffect from the entity
                                // trace!("effect_entity={:?} effect_slice={:?}", effect_entity,
                                // effect_slice); let effect =
                                // self.effect_query.get_manual(world, *effect_entity).unwrap();

                                // Get the slice to init
                                // let effect_slice = effects_meta.get(&effect_entity);
                                // let effect_group =
                                //     &effects_meta.effect_cache.buffers()[batch.buffer_index as
                                // usize];
                                let Some(particles_init_bind_group) =
                                    effect_cache.init_bind_group(effect_cache_id)
                                else {
                                    error!(
                                        "Failed to find init particle buffer bind group for \
                                        entity {:?}",
                                        entity
                                    );
                                    continue;
                                };

                                let render_effect_indirect_offset =
                                    effects_meta.gpu_limits.render_effect_indirect_offset(
                                        render_effect_dispatch_buffer_index.0,
                                    );

                                let render_group_indirect_offset =
                                    effects_meta.gpu_limits.render_group_indirect_offset(
                                        dest_render_group_dispatch_buffer_index.0,
                                    );

                                trace!(
                                    "record commands for init pipeline of effect {:?} \
                                        (spawn {} = {} workgroups) spawner_base={} \
                                        spawner_offset={} \
                                        render_effect_indirect_offset={} \
                                        first_render_group_indirect_offset={}...",
                                    batches.handle,
                                    spawn_count,
                                    workgroup_count,
                                    spawner_base,
                                    spawner_offset,
                                    render_effect_indirect_offset,
                                    render_group_indirect_offset,
                                );
                                // Setup compute pass
                                compute_pass.set_pipeline(init_pipeline);
                                compute_pass.set_bind_group(
                                    0,
                                    effects_meta.sim_params_bind_group.as_ref().unwrap(),
                                    &[],
                                );
                                compute_pass.set_bind_group(1, particles_init_bind_group, &[]);
                                compute_pass.set_bind_group(
                                    2,
                                    effects_meta.spawner_bind_group.as_ref().unwrap(),
                                    &[spawner_offset],
                                );
                                compute_pass.set_bind_group(
                                    3,
                                    effects_meta
                                        .init_render_indirect_bind_group
                                        .as_ref()
                                        .unwrap(),
                                    &[
                                        render_effect_indirect_offset as u32,
                                        render_group_indirect_offset as u32,
                                        render_group_indirect_offset as u32,
                                    ],
                                );
                                if indirect_dispatch {
                                    // Note: the indirect offset of a dispatch workgroup only needs
                                    // 4-byte alignment
                                    assert_eq!(GpuInitDispatchIndirect::min_size().get(), 12);
                                    let indirect_offset =
                                        batches.init_indirect_dispatch_index.unwrap() as u64 * 12;
                                    trace!("init indirect dispatch offset: {}", indirect_offset);
                                    compute_pass.dispatch_workgroups_indirect(
                                        effect_cache.init_indirect_dispatch_buffer().unwrap(),
                                        indirect_offset,
                                    );
                                } else {
                                    compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                                }
                                trace!("init compute dispatched (indirect:{})", indirect_dispatch);
                            }

                            EffectInitializer::Cloner(EffectCloner {
                                clone_this_frame: spawn_this_frame,
                                ..
                            }) => {
                                if !spawn_this_frame {
                                    continue;
                                }

                                let mut compute_pass = render_context
                                    .command_encoder()
                                    .begin_compute_pass(&ComputePassDescriptor {
                                        label: Some("hanabi:clone"),
                                        timestamp_writes: None,
                                    });

                                let clone_pipeline_id = batches.init_and_update_pipeline_ids
                                    [dest_group_index as usize]
                                    .init;

                                let effect_cache_id = batches.effect_cache_id;

                                let Some(clone_pipeline) =
                                    pipeline_cache.get_compute_pipeline(clone_pipeline_id)
                                else {
                                    if let CachedPipelineState::Err(err) =
                                        pipeline_cache.get_compute_pipeline_state(clone_pipeline_id)
                                    {
                                        error!(
                                            "Failed to find clone pipeline #{} for effect \
                                                    {:?}: {:?}",
                                            clone_pipeline_id.id(),
                                            entity,
                                            err
                                        );
                                    }
                                    continue;
                                };

                                let Some(particles_init_bind_group) =
                                    effect_cache.init_bind_group(effect_cache_id)
                                else {
                                    error!(
                                        "Failed to find clone particle buffer bind group \
                                                 for entity {:?}, effect cache ID {:?}",
                                        entity, effect_cache_id
                                    );
                                    continue;
                                };

                                let render_effect_dispatch_buffer_index = batches
                                    .dispatch_buffer_indices
                                    .render_effect_metadata_buffer_index;
                                let clone_dest_render_group_dispatch_buffer_index = batches
                                    .dispatch_buffer_indices
                                    .trail_dispatch_buffer_indices[&dest_group_index]
                                    .dest;
                                let clone_src_render_group_dispatch_buffer_index = batches
                                    .dispatch_buffer_indices
                                    .trail_dispatch_buffer_indices[&dest_group_index]
                                    .src;

                                let render_effect_indirect_offset =
                                    effects_meta.gpu_limits.render_effect_indirect_offset(
                                        render_effect_dispatch_buffer_index.0,
                                    );

                                let clone_dest_render_group_indirect_offset =
                                    effects_meta.gpu_limits.render_group_indirect_offset(
                                        clone_dest_render_group_dispatch_buffer_index.0,
                                    );
                                let clone_src_render_group_indirect_offset =
                                    effects_meta.gpu_limits.render_group_indirect_offset(
                                        clone_src_render_group_dispatch_buffer_index.0,
                                    );

                                compute_pass.set_pipeline(clone_pipeline);
                                compute_pass.set_bind_group(
                                    0,
                                    effects_meta.sim_params_bind_group.as_ref().unwrap(),
                                    &[],
                                );
                                compute_pass.set_bind_group(1, particles_init_bind_group, &[]);
                                compute_pass.set_bind_group(
                                    2,
                                    effects_meta.spawner_bind_group.as_ref().unwrap(),
                                    &[spawner_offset],
                                );
                                compute_pass.set_bind_group(
                                    3,
                                    effects_meta
                                        .init_render_indirect_bind_group
                                        .as_ref()
                                        .unwrap(),
                                    &[
                                        render_effect_indirect_offset as u32,
                                        clone_dest_render_group_indirect_offset as u32,
                                        clone_src_render_group_indirect_offset as u32,
                                    ],
                                );

                                if let Some(dispatch_indirect_buffer) =
                                    effects_meta.dispatch_indirect_buffer.buffer()
                                {
                                    compute_pass.dispatch_workgroups_indirect(
                                        dispatch_indirect_buffer,
                                        clone_src_render_group_indirect_offset,
                                    );
                                }
                                trace!("clone compute dispatched");
                            }
                        }
                    }
                }
            }
        }

        // Compute indirect dispatch pass
        if effects_meta.spawner_buffer.buffer().is_some()
            && !effects_meta.spawner_buffer.is_empty()
            && effects_meta.dr_indirect_bind_group.is_some()
            && effects_meta.sim_params_bind_group.is_some()
        {
            // Only start a compute pass if there's an effect; makes things clearer in
            // debugger.
            let mut compute_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hanabi:indirect_dispatch"),
                        timestamp_writes: None,
                    });

            // Dispatch indirect dispatch compute job
            trace!("record commands for indirect dispatch pipeline...");

            // FIXME - The `vfx_indirect` shader assumes a contiguous array of ParticleGroup
            // structures. So we need to pass the full array size, and we
            // just update the unused groups for nothing. Otherwise we might
            // update some unused group and miss some used ones, if there's any gap
            // in the array.
            const WORKGROUP_SIZE: u32 = 64;
            let total_group_count = effects_meta.particle_group_buffer.len() as u32;
            let workgroup_count = (total_group_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

            // Setup compute pass
            compute_pass.set_pipeline(&dispatch_indirect_pipeline.pipeline);
            compute_pass.set_bind_group(
                0,
                // FIXME - got some unwrap() panic here, investigate... possibly race
                // condition!
                effects_meta.dr_indirect_bind_group.as_ref().unwrap(),
                &[],
            );
            compute_pass.set_bind_group(
                1,
                effects_meta.sim_params_bind_group.as_ref().unwrap(),
                &[],
            );
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
            trace!(
                "indirect dispatch compute dispatched: num_batches={} workgroup_count={}",
                total_group_count,
                workgroup_count
            );
        }

        // Once all init dispatch indirect are consumed for this frame, schedule any GPU
        // buffer resize if needed. There's no need to copy any data from the CPU. We
        // need to do that here after the init pass possibly used the values in the
        // buffer from the previous frame's update, and before the current frame's
        // update might fill it with new values for next frame.
        // {
        //     render_context
        //         .command_encoder()
        //         .push_debug_group("hanabi:alloc_init_indirect_dispatch");

        //     effect_cache.allocate_init_indirect_dispatch(render_context.
        // render_device());

        //     render_context.command_encoder().pop_debug_group();
        // }

        // Compute update pass
        {
            let mut compute_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hanabi:update"),
                        timestamp_writes: None,
                    });

            // Dispatch update compute jobs
            for (entity, batches) in self.effect_query.iter_manual(world) {
                let effect_cache_id = batches.effect_cache_id;

                let Some(particles_update_bind_group) =
                    effect_cache.update_bind_group(effect_cache_id, &batches.child_effects[..])
                else {
                    error!(
                        "Failed to find update particle buffer bind group for entity {:?}, effect cache ID {:?}",
                        entity, effect_cache_id
                    );
                    continue;
                };

                let first_update_group_dispatch_buffer_index = batches
                    .dispatch_buffer_indices
                    .first_update_group_dispatch_buffer_index;

                let Some(update_render_indirect_bind_group) = &effect_bind_groups
                    .update_render_indirect_bind_groups
                    .get(&effect_cache_id)
                else {
                    error!(
                        "Failed to find update render indirect bind group for effect cache ID: {:?}, IDs present: {:?}",
                        effect_cache_id,
                        effect_bind_groups
                            .update_render_indirect_bind_groups
                            .keys()
                            .collect::<Vec<_>>()
                    );
                    continue;
                };

                for &group_index in batches.group_order.iter() {
                    let init_and_update_pipeline_id =
                        &batches.init_and_update_pipeline_ids[group_index as usize];
                    let Some(update_pipeline) =
                        pipeline_cache.get_compute_pipeline(init_and_update_pipeline_id.update)
                    else {
                        if let CachedPipelineState::Err(err) = pipeline_cache
                            .get_compute_pipeline_state(init_and_update_pipeline_id.update)
                        {
                            error!(
                                "Failed to find update pipeline #{} for effect {:?}, group {}: {:?}",
                                init_and_update_pipeline_id.update.id(),
                                entity,
                                group_index,
                                err
                            );
                        }
                        continue;
                    };

                    let update_group_dispatch_buffer_offset =
                        effects_meta.gpu_limits.dispatch_indirect_offset(
                            first_update_group_dispatch_buffer_index.0 + group_index,
                        );

                    // Destination group spawners are packed one after one another.
                    let spawner_base = batches.spawner_base + group_index;
                    let spawner_buffer_aligned = effects_meta.spawner_buffer.aligned_size();
                    assert!(spawner_buffer_aligned >= GpuSpawnerParams::min_size().get() as usize);
                    let spawner_offset = spawner_base * spawner_buffer_aligned as u32;

                    // for (effect_entity, effect_slice) in effects_meta.entity_map.iter()
                    // Retrieve the ExtractedEffect from the entity
                    // trace!("effect_entity={:?} effect_slice={:?}", effect_entity,
                    // effect_slice); let effect =
                    // self.effect_query.get_manual(world, *effect_entity).unwrap();

                    // Get the slice to update
                    // let effect_slice = effects_meta.get(&effect_entity);
                    // let effect_group =
                    //     &effects_meta.effect_cache.buffers()[batch.buffer_index as usize];

                    trace!(
                        "record commands for update pipeline of effect {:?} \
                        spawner_base={} update_group_dispatch_buffer_offset={}…",
                        batches.handle,
                        spawner_base,
                        update_group_dispatch_buffer_offset,
                    );

                    // Setup compute pass
                    // compute_pass.set_pipeline(&effect_group.update_pipeline);
                    compute_pass.set_pipeline(update_pipeline);
                    compute_pass.set_bind_group(
                        0,
                        effects_meta.sim_params_bind_group.as_ref().unwrap(),
                        &[],
                    );
                    compute_pass.set_bind_group(1, particles_update_bind_group, &[]);
                    compute_pass.set_bind_group(
                        2,
                        effects_meta.spawner_bind_group.as_ref().unwrap(),
                        &[spawner_offset],
                    );
                    compute_pass.set_bind_group(3, update_render_indirect_bind_group, &[]);

                    if let Some(buffer) = effects_meta.dispatch_indirect_buffer.buffer() {
                        trace!(
                            "dispatch_workgroups_indirect: buffer={:?} offset={}",
                            buffer,
                            update_group_dispatch_buffer_offset,
                        );
                        compute_pass.dispatch_workgroups_indirect(
                            buffer,
                            update_group_dispatch_buffer_offset as u64,
                        );
                        // TODO - offset
                    }

                    trace!("update compute dispatched");
                }
            }
        }

        Ok(())
    }
}

// FIXME - Remove this, handle it properly with a BufferTable::insert_many() or
// so...
fn allocate_sequential_buffers<T, I>(
    buffer_table: &mut BufferTable<T>,
    iterator: I,
) -> BufferTableId
where
    T: Pod + ShaderSize,
    I: Iterator<Item = T>,
{
    let mut first_buffer = None;
    for (object_index, object) in iterator.enumerate() {
        let buffer = buffer_table.insert(object);
        match first_buffer {
            None => first_buffer = Some(buffer),
            Some(ref first_buffer) => {
                if first_buffer.0 + object_index as u32 != buffer.0 {
                    error!(
                        "Allocator didn't allocate sequential indices (expected {:?}, got {:?}). \
                        Expect trouble!",
                        first_buffer.0 + object_index as u32,
                        buffer.0
                    );
                }
            }
        }
    }

    first_buffer.expect("No buffers allocated")
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
        assert!(
            limits.render_effect_indirect_offset(256)
                >= 256 * GpuRenderEffectMetadata::min_size().get()
        );
        assert!(
            limits.render_group_indirect_offset(256)
                >= 256 * GpuRenderGroupIndirect::min_size().get()
        );
        assert!(
            limits.dispatch_indirect_offset(256) as u64
                >= 256 * GpuDispatchIndirect::min_size().get()
        );
    }
}
