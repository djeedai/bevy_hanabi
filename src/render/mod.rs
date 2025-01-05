use std::{
    borrow::Cow,
    num::{NonZero, NonZeroU32, NonZeroU64},
    ops::{Deref, Range},
};
use std::{iter, marker::PhantomData};

use batch::{CachedGroups, InitAndUpdatePipelineIds};
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
    utils::HashMap,
};
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use effect_cache::{
    BufferState, CachedEffect, EffectSlices, RenderGroupDispatchIndices, TrailDispatchBufferIndices,
};
use fixedbitset::FixedBitSet;
use naga_oil::compose::{Composer, NagaModuleDescriptor};
use rand::random;

use crate::{
    asset::EffectAsset,
    calc_func_id,
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
mod property;
mod shader_cache;

use aligned_buffer_vec::AlignedBufferVec;
use buffer_table::{BufferTable, BufferTableId};
pub(crate) use effect_cache::EffectCache;
pub(crate) use property::PropertyCache;
use property::{CachedEffectProperties, CachedPropertiesError};
pub use shader_cache::ShaderCache;

use self::batch::EffectBatches;

// Size of an indirect index (including both parts of the ping-pong buffer) in
// bytes.
const INDIRECT_INDEX_SIZE: u32 = 12;

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
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct GpuRenderEffectMetadata {
    pub ping: u32,
}

/// Indirect draw parameters, with some data of our own tacked on to the end.
///
/// A few fields of this differ depending on whether the mesh is indexed or
/// non-indexed.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
pub struct GpuRenderGroupIndirect {
    /// The number of vertices in the mesh, if non-indexed; if indexed, the
    /// number of indices in the mesh.
    pub vertex_count: u32,
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
    /// The absolute index of this particle group in the global particle group
    /// buffer, which includes all effects.
    pub global_group_index: u32,
    /// The global index of the particle effect.
    pub effect_index: u32,
    /// The relative index of this particle group in the effect.
    ///
    /// For example, the first group in an effect has index 0, the second has
    /// index 1, etc. This is always 0 when not using groups.
    pub group_index_in_effect: u32,
    /// The index of the first particle in this group in the indirect index
    /// buffer.
    pub indirect_index: u32,
    /// The capacity of this group, in number of particles.
    pub capacity: u32,
    /// The index of the first particle in the particle and indirect buffers of
    /// this effect.
    pub effect_particle_offset: u32,
    /// Index of the [`GpuDispatchIndirect`] struct inside the global
    /// [`EffectsMeta::dispatch_indirect_buffer`].
    pub indirect_dispatch_index: u32,
    /// Index of the [`GpuRenderGroupIndirect`] struct inside the global
    /// [`EffectsMeta::render_group_dispatch_buffer`].
    pub indirect_render_index: u32,
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
        let dispatch_indirect_stride_code = (dispatch_indirect_size.get() as u32).to_wgsl_string();
        let indirect_code = include_str!("vfx_indirect.wgsl")
            .replace(
                "{{RENDER_EFFECT_INDIRECT_STRIDE}}",
                &render_effect_indirect_stride_code,
            )
            .replace(
                "{{RENDER_GROUP_INDIRECT_STRIDE}}",
                &render_group_indirect_stride_code,
            )
            .replace(
                "{{DISPATCH_INDIRECT_STRIDE}}",
                &dispatch_indirect_stride_code,
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
            entry_point: Some("main"),
            compilation_options: default(),
            cache: None,
        });

        Self {
            dispatch_indirect_layout,
            pipeline,
        }
    }
}

#[derive(Resource)]
pub(crate) struct ParticlesInitPipeline {
    render_device: RenderDevice,
    sim_params_layout: BindGroupLayout,
    render_indirect_spawn_layout: BindGroupLayout,
    render_indirect_clone_layout: BindGroupLayout,
    // Workaround for specialize() not being able to access external data: we temporarily store the
    // bind group layout just before calling specialize, and remove it after.
    // https://github.com/bevyengine/bevy/issues/17132
    temp_spawner_buffer_layout: Option<BindGroupLayout>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ParticleInitPipelineKey {
    shader: Handle<Shader>,
    particle_layout_min_binding_size: NonZero<u64>,
    flags: ParticleInitPipelineKeyFlags,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ParticleInitPipelineKeyFlags: u8 {
        const CLONE = 0x1;
        const ATTRIBUTE_PREV = 0x2;
        const ATTRIBUTE_NEXT = 0x4;
    }
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

        let render_indirect_spawn_layout = create_init_render_indirect_bind_group_layout(
            render_device,
            "hanabi:bind_group_layout:init_render_indirect_spawn",
            false,
        );
        let render_indirect_clone_layout = create_init_render_indirect_bind_group_layout(
            render_device,
            "hanabi:bind_group_layout:init_render_indirect_clone",
            true,
        );

        Self {
            render_device: render_device.clone(),
            sim_params_layout,
            render_indirect_spawn_layout,
            render_indirect_clone_layout,
            temp_spawner_buffer_layout: None,
        }
    }
}

impl SpecializedComputePipeline for ParticlesInitPipeline {
    type Key = ParticleInitPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        trace!("Specializing init pipeline for key: {key:?}");

        // FIXME: Remove from here, and use instead the one created in
        // EffectBuffer::new()!
        let particles_buffer_layout = create_sim_bind_group_layout(
            &self.render_device,
            "hanabi:init_particles_buffer_layout",
            key.particle_layout_min_binding_size,
        );

        let mut shader_defs = Vec::with_capacity(3);
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

        let render_indirect_layout = if key.flags.contains(ParticleInitPipelineKeyFlags::CLONE) {
            self.render_indirect_clone_layout.clone()
        } else {
            self.render_indirect_spawn_layout.clone()
        };

        // This should always be valid when specialize() is called, by design. This is
        // how we pass the value to specialize() to work around the lack of access to
        // external data.
        // https://github.com/bevyengine/bevy/issues/17132
        let spawner_buffer_layout = self.temp_spawner_buffer_layout.as_ref().unwrap();

        let hash = calc_func_id(&key);
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
                particles_buffer_layout,
                spawner_buffer_layout.clone(),
                render_indirect_layout,
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
    render_device: RenderDevice,
    sim_params_layout: BindGroupLayout,
    render_indirect_layout: BindGroupLayout,
    // Workaround for specialize() not being able to access external data: we temporarily store the
    // bind group layout just before calling specialize, and remove it after.
    // https://github.com/bevyengine/bevy/issues/17132
    temp_spawner_buffer_layout: Option<BindGroupLayout>,
}

impl FromWorld for ParticlesUpdatePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let limits = render_device.limits();
        bevy::log::info!(
            "GPU limits:\n- max_compute_invocations_per_workgroup={}\n- max_compute_workgroup_size_x={}\n- max_compute_workgroup_size_y={}\n- max_compute_workgroup_size_z={}\n- max_compute_workgroups_per_dimension={}\n- min_storage_buffer_offset_alignment={}",
            limits.max_compute_invocations_per_workgroup, limits.max_compute_workgroup_size_x, limits.max_compute_workgroup_size_y, limits.max_compute_workgroup_size_z,
            limits.max_compute_workgroups_per_dimension, limits.min_storage_buffer_offset_alignment
        );

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
            render_indirect_layout,
            temp_spawner_buffer_layout: None,
        }
    }
}

#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub(crate) struct ParticleUpdatePipelineKey {
    /// Compute shader, with snippets applied, but not preprocessed yet.
    shader: Handle<Shader>,
    /// Particle layout.
    particle_layout: ParticleLayout,
    is_trail: bool,
}

impl SpecializedComputePipeline for ParticlesUpdatePipeline {
    type Key = ParticleUpdatePipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        trace!("Specializing update pipeline for key: {key:?}");

        // FIXME: Remove from here, and use instead the one created in
        // EffectBuffer::new()!
        let update_particles_buffer_layout = create_sim_bind_group_layout(
            &self.render_device,
            "hanabi:update_particles_buffer_layout",
            key.particle_layout.min_binding_size(),
        );

        let mut shader_defs = Vec::with_capacity(4);
        shader_defs.push("REM_MAX_SPAWN_ATOMIC".into());
        if key.particle_layout.contains(Attribute::PREV) {
            shader_defs.push("ATTRIBUTE_PREV".into());
        }
        if key.particle_layout.contains(Attribute::NEXT) {
            shader_defs.push("ATTRIBUTE_NEXT".into());
        }
        if key.is_trail {
            shader_defs.push("TRAIL".into());
        }

        // This should always be valid when specialize() is called, by design. This is
        // how we pass the value to specialize() to work around the lack of access to
        // external data.
        // https://github.com/bevyengine/bevy/issues/17132
        let spawner_buffer_layout = self.temp_spawner_buffer_layout.as_ref().unwrap();

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
                update_particles_buffer_layout,
                spawner_buffer_layout.clone(),
                self.render_indirect_layout.clone(),
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
#[derive(Debug, Component)]
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
    /// Maps a group number to the runtime initializer for that group.
    ///
    /// Obtained from calling [`EffectSpawner::tick()`] on the source effect
    /// instance.
    ///
    /// [`EffectSpawner::tick()`]: crate::EffectSpawner::tick
    pub initializers: Vec<EffectInitializer>,
    /// Global transform of the effect origin.
    pub transform: GlobalTransform,
    /// Layout flags.
    pub layout_flags: LayoutFlags,
    pub mesh: Handle<Mesh>,
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
    pub entity: MainEntity,
    #[allow(dead_code)]
    pub render_entity: RenderEntity,
    pub groups: Vec<AddedEffectGroup>,
    /// Layout of particle attributes.
    pub particle_layout: ParticleLayout,
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
    /// Extracted effects this frame.
    pub effects: Vec<ExtractedEffect>,
    /// Entities which had their [`ParticleEffect`] component removed.
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    pub removed_effect_entities: Vec<RenderEntity>,
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
    trace!("extract_effect_events");

    let EffectAssetEvents { ref mut images } = *events;
    *images = image_events.read().copied().collect();
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
                &RenderEntity,
                Option<&InheritedVisibility>,
                Option<&ViewVisibility>,
                &EffectInitializers,
                &CompiledParticleEffect,
                Option<Ref<EffectProperties>>,
                &GlobalTransform,
            )>,
            // Newly added ParticleEffect components
            Query<
                (Entity, &RenderEntity, &CompiledParticleEffect),
                (Added<CompiledParticleEffect>, With<GlobalTransform>),
            >,
        )>,
    >,
    mut removed_effects_event_reader: Extract<EventReader<RemovedEffectsEvent>>,
    mut sim_params: ResMut<SimParams>,
    mut extracted_effects: ResMut<ExtractedEffects>,
    effects_meta: Res<EffectsMeta>,
) {
    trace!("extract_effects");

    // Save simulation params into render world
    sim_params.time = time.elapsed_secs_f64();
    sim_params.delta_time = time.delta_secs();
    sim_params.virtual_time = virtual_time.elapsed_secs_f64();
    sim_params.virtual_delta_time = virtual_time.delta_secs();
    sim_params.real_time = real_time.elapsed_secs_f64();
    sim_params.real_delta_time = real_time.delta_secs();

    // Collect removed effects for later GPU data purge
    extracted_effects.removed_effect_entities =
        removed_effects_event_reader
            .read()
            .fold(vec![], |mut acc, ev| {
                acc.extend(
                    ev.entities
                        .iter()
                        .filter_map(|(_, render_entity)| *render_entity),
                );
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
            let group_order = asset.calculate_group_order();

            trace!(
                "Found new effect: entity {:?} | render entity {:?} | capacities {:?} | particle_layout {:?} | \
                 property_layout {:?} | layout_flags {:?}",
                 entity,
                 render_entity.id(),
                 asset.capacities(),
                 particle_layout,
                 property_layout,
                 compiled_effect.layout_flags);

            Some(AddedEffect {
                entity: MainEntity::from(entity),
                render_entity: *render_entity,
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
        main_entity,
        render_entity,
        maybe_inherited_visibility,
        maybe_view_visibility,
        initializers,
        effect,
        maybe_properties,
        transform,
    ) in query.p0().iter_mut()
    {
        // Check if shaders are configured
        let effect_shaders = effect.get_configured_shaders();
        if effect_shaders.is_empty() {
            continue;
        }

        // Check if hidden, unless always simulated
        if effect.simulation_condition == SimulationCondition::WhenVisible
            && !maybe_inherited_visibility
                .map(|cv| cv.get())
                .unwrap_or(true)
            && !maybe_view_visibility.map(|cv| cv.get()).unwrap_or(true)
        {
            continue;
        }

        // Check if asset is available, otherwise silently ignore
        let Some(asset) = effects.get(&effect.asset) else {
            trace!(
                "EffectAsset not ready; skipping ParticleEffect instance on entity {:?}.",
                main_entity
            );
            continue;
        };

        #[cfg(feature = "2d")]
        let z_sort_key_2d = effect.z_layer_2d;

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
        let layout_flags = effect.layout_flags;
        let mesh = match effect.mesh {
            None => effects_meta.default_mesh.clone(),
            Some(ref mesh) => (*mesh).clone(),
        };
        let alpha_mode = effect.alpha_mode;

        trace!(
            "Extracted instance of effect '{}' on entity {:?} (render entity {:?}): texture_layout_count={} texture_count={} layout_flags={:?}",
            asset.name,
            main_entity,
            render_entity.id(),
            texture_layout.layout.len(),
            effect.textures.len(),
            layout_flags,
        );

        extracted_effects.effects.push(ExtractedEffect {
            render_entity: *render_entity,
            main_entity: main_entity.into(),
            handle: effect.asset.clone_weak(),
            particle_layout: asset.particle_layout().clone(),
            property_layout,
            property_data,
            initializers: initializers.0.clone(),
            transform: *transform,
            layout_flags,
            mesh,
            texture_layout,
            textures: effect.textures.clone(),
            alpha_mode,
            effect_shaders: effect_shaders.to_vec(),
            #[cfg(feature = "2d")]
            z_sort_key_2d,
        });
    }
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
        let storage_buffer_align =
            render_device.limits().min_storage_buffer_offset_alignment as u64;

        let dispatch_indirect_aligned_size = NonZeroU32::new(
            GpuDispatchIndirect::min_size()
                .get()
                .next_multiple_of(storage_buffer_align) as u32,
        )
        .unwrap();

        let render_effect_indirect_aligned_size = NonZeroU32::new(
            GpuRenderEffectMetadata::min_size()
                .get()
                .next_multiple_of(storage_buffer_align) as u32,
        )
        .unwrap();

        let render_group_indirect_aligned_size = NonZeroU32::new(
            GpuRenderGroupIndirect::min_size()
                .get()
                .next_multiple_of(storage_buffer_align) as u32,
        )
        .unwrap();

        let particle_group_aligned_size = NonZeroU32::new(
            GpuParticleGroup::min_size()
                .get()
                .next_multiple_of(storage_buffer_align) as u32,
        )
        .unwrap();

        trace!(
            "GpuLimits: storage_buffer_align={} gpu_dispatch_indirect_aligned_size={} \
            gpu_render_effect_indirect_aligned_size={} gpu_render_group_indirect_aligned_size={}",
            storage_buffer_align,
            dispatch_indirect_aligned_size.get(),
            render_effect_indirect_aligned_size.get(),
            render_group_indirect_aligned_size.get()
        );

        Self {
            storage_buffer_align: NonZeroU32::new(storage_buffer_align as u32).unwrap(),
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
    /// Bind group for the simulation parameters, like the current time and
    /// frame delta time.
    sim_params_bind_group: Option<BindGroup>,
    /// Bind group #0 of the vfx_indirect shader, containing both the indirect
    /// compute dispatch and render buffers.
    dr_indirect_bind_group: Option<BindGroup>,
    /// Bind group #3 of the vfx_init shader, containing the indirect render
    /// buffer, in the case of a spawner with no source buffer.
    init_render_indirect_spawn_bind_group: Option<BindGroup>,
    /// Bind group #3 of the vfx_init shader, containing the indirect render
    /// buffer, in the case of a cloner with a source buffer.
    init_render_indirect_clone_bind_group: Option<BindGroup>,
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
    /// The mesh used when particle effects don't specify one (i.e. a quad).
    default_mesh: Handle<Mesh>,
    /// Various GPU limits and aligned sizes lazily allocated and cached for
    /// convenience.
    gpu_limits: GpuLimits,
}

impl EffectsMeta {
    pub fn new(device: RenderDevice, mesh_assets: &mut Assets<Mesh>) -> Self {
        let default_mesh = mesh_assets.add(Plane3d::new(Vec3::Z, Vec2::splat(0.5)));

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
            sim_params_bind_group: None,
            dr_indirect_bind_group: None,
            init_render_indirect_spawn_bind_group: None,
            init_render_indirect_clone_bind_group: None,
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
            default_mesh,
            gpu_limits,
        }
    }

    /// Allocate internal resources for newly spawned effects, and deallocate
    /// them for just-removed ones.
    ///
    /// After this system ran, all valid extracted effects from the main world
    /// have a corresponding entity with a [`CachedEffect`] component in the
    /// render world. An extracted effect is considered valid if it passed some
    /// basic checks, like having a valid mesh. Note however that the main
    /// world's entity might still be missing its [`RenderEntity`]
    /// reference, since we cannot yet write into the main world.
    pub fn add_remove_effects(
        &mut self,
        mut commands: Commands,
        q_cached_effects: &Query<(
            MainEntity,
            &CachedEffect,
            &DispatchBufferIndices,
            Option<&CachedEffectProperties>,
        )>,
        mut added_effects: Vec<AddedEffect>,
        removed_effect_entities: Vec<RenderEntity>,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        effect_bind_groups: &mut ResMut<EffectBindGroups>,
        effect_cache: &mut ResMut<EffectCache>,
        property_cache: &mut ResMut<PropertyCache>,
    ) {
        // Deallocate GPU data for destroyed effect instances. This will automatically
        // drop any group where there is no more effect slice.
        trace!(
            "Removing {} despawned effects",
            removed_effect_entities.len()
        );
        for render_entity in &removed_effect_entities {
            trace!(
                "Removing ParticleEffect on render entity {:?}",
                render_entity
            );

            // Schedule a despawn of the render entity for the end of this system
            if let Some(cmd) = commands.get_entity(render_entity.id()) {
                cmd.despawn_recursive();
            } else {
                warn!("Unknown render entity {:?}, cannot despawn", render_entity);
                continue;
            }

            // Fecth the components of the effect being destroyed. Note that the despawn
            // command above is not yet applied, so this query should always succeed.
            let Ok((main_entity, cached_effect, dispatch_buffer_indices, cached_effect_properties)) =
                q_cached_effects.get(render_entity.id())
            else {
                error!(
                    "Render entity {:?} exists, but failed to query its components.",
                    render_entity
                );
                continue;
            };

            // Deallocate properties if any
            if let Some(cached_effect_properties) = cached_effect_properties {
                match property_cache.remove_properties(cached_effect_properties) {
                    Err(err) => match err {
                        CachedPropertiesError::InvalidBufferIndex(buffer_index)
                            => error!("Failed to remove cached properties of render entity {render_entity:?} from buffer #{buffer_index}: the index is invalid."),
                        CachedPropertiesError::BufferDeallocated(buffer_index)
                            => error!("Failed to remove cached properties of render entity {render_entity:?} from buffer #{buffer_index}: the buffer is not allocated."),
                    }
                    Ok(buffer_state) => if buffer_state == BufferState::Free {
                        // The entire buffer was deallocated; destroy all bind groups referencing it
                        let key = cached_effect_properties.to_key();
                        trace!("Destroying property bind group for key {key:?} due to property buffer deallocated.");
                        effect_bind_groups.property_bind_groups.remove(&key);
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
            if let Ok(BufferState::Free) = effect_cache.remove(cached_effect) {
                // Clear bind groups associated with the removed buffer
                trace!(
                    "=> GPU buffer #{} gone, destroying its bind groups...",
                    cached_effect.buffer_index
                );
                effect_bind_groups
                    .particle_buffers
                    .remove(&cached_effect.buffer_index);

                let group_count = cached_effect.slices.group_count();

                let first_row = dispatch_buffer_indices
                    .first_update_group_dispatch_buffer_index
                    .0;
                for table_id in first_row..(first_row + group_count) {
                    self.dispatch_indirect_buffer
                        .remove(BufferTableId(table_id));
                }

                self.render_effect_dispatch_buffer
                    .remove(dispatch_buffer_indices.render_effect_metadata_buffer_index);

                if let RenderGroupDispatchIndices::Allocated {
                    first_render_group_dispatch_buffer_index,
                    ..
                } = &dispatch_buffer_indices.render_group_dispatch_indices
                {
                    for row_index in (first_render_group_dispatch_buffer_index.0)
                        ..(first_render_group_dispatch_buffer_index.0 + group_count)
                    {
                        self.render_group_dispatch_buffer
                            .remove(BufferTableId(row_index));
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
            // Allocate per-group update dispatch indirect
            let first_update_group_dispatch_buffer_index = allocate_sequential_buffers(
                &mut self.dispatch_indirect_buffer,
                iter::repeat(GpuDispatchIndirect::default()).take(added_effect.groups.len()),
            );

            // Allocate per-effect metadata
            let render_effect_dispatch_buffer_id = self
                .render_effect_dispatch_buffer
                .insert(GpuRenderEffectMetadata::default());

            let dispatch_buffer_indices = DispatchBufferIndices {
                first_update_group_dispatch_buffer_index,
                render_effect_metadata_buffer_index: render_effect_dispatch_buffer_id,
                render_group_dispatch_indices: RenderGroupDispatchIndices::Pending {
                    groups: added_effect.groups.iter().map(Into::into).collect(),
                },
            };

            // Insert the effect into the cache. This will allocate all the necessary GPU
            // resources as needed.
            let cached_effect = effect_cache.insert(
                added_effect.handle,
                added_effect
                    .groups
                    .iter()
                    .map(|group| group.capacity)
                    .collect(),
                &added_effect.particle_layout,
                added_effect.layout_flags,
                added_effect.group_order,
            );
            let mut cmd = commands.entity(added_effect.render_entity.id());
            cmd.insert((added_effect.entity, cached_effect, dispatch_buffer_indices));

            // Allocate storage for properties if needed
            if !added_effect.property_layout.is_empty() {
                let cached_effect_properties = property_cache.insert(&added_effect.property_layout);
                cmd.insert(cached_effect_properties);
            }

            trace!(
                "+ added effect entity {:?}: main_entity={:?} \
                first_update_group_dispatch_buffer_index={} \
                render_effect_dispatch_buffer_id={}",
                added_effect.render_entity,
                added_effect.entity,
                first_update_group_dispatch_buffer_index.0,
                render_effect_dispatch_buffer_id.0
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
            self.init_render_indirect_spawn_bind_group = None;
            self.init_render_indirect_clone_bind_group = None;
            effect_bind_groups
                .update_render_indirect_bind_groups
                .clear();
        }
    }

    pub fn allocate_gpu(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
        effect_bind_groups: &mut ResMut<EffectBindGroups>,
    ) {
        if self
            .render_group_dispatch_buffer
            .allocate_gpu(render_device, render_queue)
        {
            // All those bind groups use the buffer so need to be re-created
            self.dr_indirect_bind_group = None;
            self.init_render_indirect_spawn_bind_group = None;
            self.init_render_indirect_clone_bind_group = None;
            effect_bind_groups
                .update_render_indirect_bind_groups
                .clear();
        }
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
        /// The effect is rendered with flipbook texture animation based on the [`Attribute::SPRITE_INDEX`] of each particle.
        const FLIPBOOK = (1 << 4);
        /// The effect needs UVs.
        const NEEDS_UV = (1 << 5);
        /// The effect has ribbons.
        const RIBBONS = (1 << 6);
        /// The effects needs normals.
        const NEEDS_NORMAL = (1 << 7);
        /// The effect is fully-opaque.
        const OPAQUE = (1 << 8);
    }
}

impl Default for LayoutFlags {
    fn default() -> Self {
        Self::NONE
    }
}

/// Spawn a new render entity with a [`CachedEffect`] component for any newly
/// allocated effect, and conversely despawn any entity with such component for
/// removed effects.
///
/// After this system ran, and its commands are applied, all valid extracted
/// effects have a corresponding entity in the render world, with a
/// [`CachedEffect`] component. From there, we operate on those exclusively.
pub(crate) fn add_remove_effects(
    q_cached_effects: Query<(
        MainEntity,
        &CachedEffect,
        &DispatchBufferIndices,
        Option<&CachedEffectProperties>,
    )>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    commands: Commands,
    mut effects_meta: ResMut<EffectsMeta>,
    mut effect_cache: ResMut<EffectCache>,
    mut property_cache: ResMut<PropertyCache>,
    mut extracted_effects: ResMut<ExtractedEffects>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
) {
    trace!("add_remove_effects");

    // Clear last frame's buffer resizes which may have occured during last frame,
    // during `Node::run()` while the `BufferTable` could not be mutated. This is
    // the first point at which we can do that where we're not blocking the main
    // world (so, excluding the extract system).
    effects_meta
        .dispatch_indirect_buffer
        .clear_previous_frame_resizes();
    effects_meta
        .render_effect_dispatch_buffer
        .clear_previous_frame_resizes();
    effects_meta
        .render_group_dispatch_buffer
        .clear_previous_frame_resizes();

    // Allocate new effects, deallocate removed ones
    effects_meta.add_remove_effects(
        commands,
        &q_cached_effects,
        std::mem::take(&mut extracted_effects.added_effects),
        std::mem::take(&mut extracted_effects.removed_effect_entities),
        &render_device,
        &render_queue,
        &mut effect_bind_groups,
        &mut effect_cache,
        &mut property_cache,
    );

    // Note: we don't need to explicitly allocate GPU buffers for effects, because
    // EffectBuffer already contains a reference to the RenderDevice, so has done so
    // internally. This is not ideal design-wise, but works.

    // Allocate all the property buffer(s) as needed, before we move to the next
    // step which will need those buffers to schedule data copies from CPU.
    for buffer in property_cache.buffers_mut().iter_mut().flatten() {
        let _changed = buffer.write_buffer(&render_device, &render_queue);
        // FIXME - invalidate bind groups!
    }
}

/// Indexed mesh metadata for [`CachedMesh`].
#[derive(Debug)]
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
#[derive(Debug, Component)]
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
    /// GPU buffer containing the property data.
    pub buffer: Buffer,
    /// Index of the buffer in the [`EffectCache`].
    pub buffer_index: u32,
    /// Offset in bytes inside the buffer.
    pub offset: u32,
    /// Binding size in bytes of the property struct.
    pub binding_size: u32,
}

pub(crate) fn prepare_effects(
    mut commands: Commands,
    sim_params: Res<SimParams>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut init_pipeline: ResMut<ParticlesInitPipeline>,
    mut update_pipeline: ResMut<ParticlesUpdatePipeline>,
    mesh_allocator: Res<MeshAllocator>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    mut property_cache: ResMut<PropertyCache>,
    mut specialized_init_pipelines: ResMut<SpecializedComputePipelines<ParticlesInitPipeline>>,
    mut specialized_update_pipelines: ResMut<SpecializedComputePipelines<ParticlesUpdatePipeline>>,
    mut effects_meta: ResMut<EffectsMeta>,
    mut extracted_effects: ResMut<ExtractedEffects>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    mut q_cached_effects: Query<(
        MainEntity,
        &CachedEffect,
        &mut DispatchBufferIndices,
        Option<&CachedEffectProperties>,
    )>,
) {
    trace!("prepare_effects");

    // // sort first by z and then by handle. this ensures that, when possible,
    // batches span multiple z layers // batches won't span z-layers if there is
    // another batch between them extracted_effects.effects.sort_by(|a, b| {
    //     match FloatOrd(a.transform.w_axis[2]).cmp(&FloatOrd(b.transform.
    // w_axis[2])) {         Ordering::Equal => a.handle.cmp(&b.handle),
    //         other => other,
    //     }
    // });

    // Clear per-instance buffers, which are filled below and re-uploaded each frame
    effects_meta.spawner_buffer.clear();
    effects_meta.particle_group_buffer.clear();

    // Build batcher inputs from extracted effects
    let mut total_effect_count = 0;
    let mut total_group_count = 0;
    let effects = std::mem::take(&mut extracted_effects.effects);
    for extracted_effect in effects.into_iter() {
        // Skip effects not cached. Since we're iterating over the extracted effects
        // instead of the cached ones, it might happen we didn't cache some effect on
        // purpose because they failed earlier validations.
        let Ok((main_entity, cached_effect, mut dispatch_buffer_indices, cached_effect_properties)) =
            q_cached_effects.get_mut(extracted_effect.render_entity.id())
        else {
            trace!(
                "Unknown render entity {:?} for extracted effect.",
                extracted_effect.render_entity.id()
            );
            continue;
        };

        // If the mesh is not available, skip this effect
        let Some(render_mesh) = render_meshes.get(extracted_effect.mesh.id()) else {
            trace!(
                "Effect main_entity {:?}: missing render mesh {:?}",
                main_entity,
                extracted_effect.mesh
            );
            continue;
        };
        let Some(mesh_vertex_buffer_slice) =
            mesh_allocator.mesh_vertex_slice(&extracted_effect.mesh.id())
        else {
            trace!(
                "Effect main_entity {:?}: missing render mesh vertex slice",
                main_entity
            );
            continue;
        };
        let mesh_index_buffer_slice = mesh_allocator.mesh_index_slice(&extracted_effect.mesh.id());
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
                        "Effect main_entity {:?}: missing render mesh index slice",
                        main_entity
                    );
                    continue;
                }
            } else {
                None
            };

        // Now that the mesh has been processed by Bevy itself, we know where it's
        // allocated inside the GPU buffer, so we can extract its base vertex and index
        // values, and allocate our indirect structs.
        if let RenderGroupDispatchIndices::Pending { groups } =
            &dispatch_buffer_indices.render_group_dispatch_indices
        {
            trace!("Effect render_entity {:?}: allocating render group indirect dispatch entries for {} groups...", extracted_effect.render_entity, groups.len());
            let mut current_base_instance = 0;
            let first_render_group_dispatch_buffer_index = allocate_sequential_buffers(
                &mut effects_meta.render_group_dispatch_buffer,
                groups.iter().map(|group| {
                    let indirect_dispatch = match &mesh_index_buffer_slice {
                        // Indexed mesh rendering
                        Some(mesh_index_buffer_slice) => {
                            let ret = GpuRenderGroupIndirect {
                                vertex_count: mesh_index_buffer_slice.range.len() as u32,
                                instance_count: 0,
                                first_index_or_vertex_offset: mesh_index_buffer_slice.range.start,
                                vertex_offset_or_base_instance: mesh_vertex_buffer_slice.range.start
                                    as i32,
                                base_instance: current_base_instance as u32,
                                alive_count: 0,
                                max_update: 0,
                                dead_count: group.capacity,
                                max_spawn: group.capacity,
                            };
                            trace!("+ Group[indexed]: {:?}", ret);
                            ret
                        }
                        // Non-indexed mesh rendering
                        None => {
                            let ret = GpuRenderGroupIndirect {
                                vertex_count: mesh_vertex_buffer_slice.range.len() as u32,
                                instance_count: 0,
                                first_index_or_vertex_offset: mesh_vertex_buffer_slice.range.start,
                                vertex_offset_or_base_instance: current_base_instance,
                                base_instance: current_base_instance as u32,
                                alive_count: 0,
                                max_update: 0,
                                dead_count: group.capacity,
                                max_spawn: group.capacity,
                            };
                            trace!("+ Group[non-indexed]: {:?}", ret);
                            ret
                        }
                    };
                    current_base_instance += group.capacity as i32;
                    indirect_dispatch
                }),
            );

            let mut trail_dispatch_buffer_indices = HashMap::new();
            for (dest_group_index, group) in groups.iter().enumerate() {
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

            trace!(
                    "-> Allocated {} render group dispatch indirect entries at offset +{}. Trails: {:?}",
                    groups.len(),
                    first_render_group_dispatch_buffer_index.0,
                    trail_dispatch_buffer_indices
                );
            dispatch_buffer_indices.render_group_dispatch_indices =
                RenderGroupDispatchIndices::Allocated {
                    first_render_group_dispatch_buffer_index,
                    trail_dispatch_buffer_indices,
                };
        }

        let effect_slices = EffectSlices {
            slices: cached_effect.slices.ranges.clone(),
            buffer_index: cached_effect.buffer_index,
            particle_layout: cached_effect.slices.particle_layout().clone(),
        };

        let particle_layout_min_binding_size = effect_slices.particle_layout.min_binding_size();
        let property_layout_min_binding_size = if extracted_effect.property_layout.is_empty() {
            None
        } else {
            Some(extracted_effect.property_layout.min_binding_size())
        };

        // Create init pipeline key flags.
        let init_pipeline_key_flags = {
            let mut flags = ParticleInitPipelineKeyFlags::empty();
            flags.set(
                ParticleInitPipelineKeyFlags::ATTRIBUTE_PREV,
                effect_slices.particle_layout.contains(Attribute::PREV),
            );
            flags.set(
                ParticleInitPipelineKeyFlags::ATTRIBUTE_NEXT,
                effect_slices.particle_layout.contains(Attribute::NEXT),
            );
            flags
        };

        // This should always exist by the time we reach this point, because we should
        // have inserted any property in the cache, which would have allocated the
        // proper layout (or the default no-property one).
        let spawner_buffer_layout = property_cache
            .bind_group_layout(property_layout_min_binding_size)
            .unwrap_or_else(|| {
                panic!(
                    "Failed to find bind group layout for property binding size {:?}",
                    property_layout_min_binding_size,
                )
            });
        trace!(
            "Retrieved property bind group layout {:?} for binding size {:?}...",
            spawner_buffer_layout.id(),
            property_layout_min_binding_size
        );

        // Specialize the init pipeline based on the effect.
        let init_and_update_pipeline_ids: Vec<InitAndUpdatePipelineIds> = extracted_effect
            .effect_shaders
            .iter()
            .enumerate()
            .map(|(group_index, shader)| {
                let mut flags = init_pipeline_key_flags;

                // If this is a cloner, add the appropriate flag.
                match extracted_effect.initializers[group_index] {
                    EffectInitializer::Spawner(_) => {}
                    EffectInitializer::Cloner(_) => {
                        flags.insert(ParticleInitPipelineKeyFlags::CLONE);
                    }
                }

                // https://github.com/bevyengine/bevy/issues/17132
                init_pipeline.temp_spawner_buffer_layout = Some(spawner_buffer_layout.clone());
                let init_pipeline_id: CachedComputePipelineId = specialized_init_pipelines
                    .specialize(
                        &pipeline_cache,
                        &init_pipeline,
                        ParticleInitPipelineKey {
                            shader: shader.init.clone(),
                            particle_layout_min_binding_size,
                            flags,
                        },
                    );
                init_pipeline.temp_spawner_buffer_layout = None; // keep things tidy; this is just a hack, should not persist
                trace!("Init pipeline specialized: id={:?}", init_pipeline_id);

                // https://github.com/bevyengine/bevy/issues/17132
                update_pipeline.temp_spawner_buffer_layout = Some(spawner_buffer_layout.clone());
                let update_pipeline_id = specialized_update_pipelines.specialize(
                    &pipeline_cache,
                    &update_pipeline,
                    ParticleUpdatePipelineKey {
                        shader: shader.update.clone(),
                        particle_layout: effect_slices.particle_layout.clone(),
                        is_trail: matches!(
                            extracted_effect.initializers[group_index],
                            EffectInitializer::Cloner(_)
                        ),
                    },
                );
                update_pipeline.temp_spawner_buffer_layout = None; // keep things tidy; this is just a hack, should not persist
                trace!("Update pipeline specialized: id={:?}", update_pipeline_id);

                InitAndUpdatePipelineIds {
                    init: init_pipeline_id,
                    update: update_pipeline_id,
                }
            })
            .collect();

        // Output some debug info
        trace!(
            "init_shader(s) = {:?}",
            extracted_effect
                .effect_shaders
                .iter()
                .map(|shaders| shaders.init.clone())
        );
        trace!(
            "update_shader(s) = {:?}",
            extracted_effect
                .effect_shaders
                .iter()
                .map(|shaders| shaders.update.clone())
        );
        trace!(
            "render_shader(s) = {:?}",
            extracted_effect
                .effect_shaders
                .iter()
                .map(|shaders| shaders.render.clone())
        );
        trace!("layout_flags = {:?}", extracted_effect.layout_flags);
        trace!("particle_layout = {:?}", effect_slices.particle_layout);
        #[cfg(feature = "2d")]
        {
            trace!("z_sort_key_2d = {:?}", extracted_effect.z_sort_key_2d);
        }

        // Allocate the spawner entry for the effect instance
        let spawner_base = effects_meta.spawner_buffer.len() as u32;
        for initializer in extracted_effect.initializers.iter() {
            let transform = extracted_effect.transform.compute_matrix().into();
            let inverse_transform = Mat4::from(
                // Inverse the Affine3A first, then convert to Mat4.This is a lot more
                // efficient than inversing the Mat4.
                extracted_effect.transform.affine().inverse(),
            )
            .into();
            // FIXME - Probably bad to re-seed each time there's a change
            let seed = random::<u32>();
            // FIXME: the effect_index is global inside the global spawner buffer,
            // but the group_index is the index of the particle buffer, which can
            // in theory (with batching) contain > 1 effect per buffer.
            let effect_index = effect_slices.buffer_index;
            let spawner_params = match initializer {
                EffectInitializer::Spawner(effect_spawner) => {
                    let spawner_params = GpuSpawnerParams {
                        transform,
                        inverse_transform,
                        spawn: effect_spawner.spawn_count as i32,
                        seed,
                        effect_index,
                        ..default()
                    };
                    trace!("spawner params = {:?}", spawner_params);
                    spawner_params
                }
                EffectInitializer::Cloner(effect_cloner) => {
                    let spawner_params = GpuSpawnerParams {
                        transform,
                        inverse_transform,
                        seed,
                        effect_index,
                        lifetime: effect_cloner.cloner.lifetime,
                        ..default()
                    };
                    trace!("cloner params = {:?}", spawner_params);
                    spawner_params
                }
            };
            effects_meta.spawner_buffer.push(spawner_params);
        }

        // Allocate the particle group buffer entries for the effect instance
        let mut first_particle_group_buffer_index = None;
        let mut local_group_count = 0;
        for (group_index, range) in effect_slices.slices.windows(2).enumerate() {
            let indirect_render_index = match &dispatch_buffer_indices.render_group_dispatch_indices
            {
                RenderGroupDispatchIndices::Allocated {
                    first_render_group_dispatch_buffer_index,
                    ..
                } => first_render_group_dispatch_buffer_index.0 + group_index as u32,
                _ => u32::MAX, /* should never happen, as lazily allocated above (unless
                                * something went wrong) */
            };
            let particle_group_buffer_index =
                effects_meta.particle_group_buffer.push(GpuParticleGroup {
                    global_group_index: total_group_count,
                    effect_index: dispatch_buffer_indices
                        .render_effect_metadata_buffer_index
                        .0,
                    group_index_in_effect: group_index as u32,
                    indirect_index: range[0],
                    capacity: range[1] - range[0],
                    effect_particle_offset: effect_slices.slices[0],
                    indirect_dispatch_index: dispatch_buffer_indices
                        .first_update_group_dispatch_buffer_index
                        .0
                        + group_index as u32,
                    indirect_render_index,
                });
            if group_index == 0 {
                first_particle_group_buffer_index = Some(particle_group_buffer_index as u32);
            }
            total_group_count += 1;
            local_group_count += 1;
        }
        assert_eq!(local_group_count, init_and_update_pipeline_ids.len());

        trace!(
            "Updating cached effect at entity {:?} ({} groups)...",
            extracted_effect.render_entity.id(),
            init_and_update_pipeline_ids.len(),
        );
        let mut cmd = commands.entity(extracted_effect.render_entity.id());
        cmd.insert((
            BatchesInput {
                handle: extracted_effect.handle,
                entity: extracted_effect.render_entity.id(),
                main_entity: extracted_effect.main_entity,
                effect_slices,
                effect_shaders: extracted_effect.effect_shaders.clone(),
                layout_flags: extracted_effect.layout_flags,
                texture_layout: extracted_effect.texture_layout.clone(),
                textures: extracted_effect.textures.clone(),
                alpha_mode: extracted_effect.alpha_mode,
                particle_layout: extracted_effect.particle_layout.clone(),
                group_order: cached_effect.group_order.clone(),
                initializers: extracted_effect.initializers,
                #[cfg(feature = "2d")]
                z_sort_key_2d: extracted_effect.z_sort_key_2d,
                #[cfg(feature = "3d")]
                position: extracted_effect.transform.translation(),
            },
            CachedMesh {
                mesh: extracted_effect.mesh.id(),
                buffer: mesh_vertex_buffer_slice.buffer.clone(),
                range: mesh_vertex_buffer_slice.range.clone(),
                indexed,
            },
            CachedGroups {
                spawner_base,
                first_particle_group_buffer_index,
                init_and_update_pipeline_ids,
            },
        ));

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

                match property_cache.remove_properties(cached_effect_properties) {
                    Err(err) => {
                        let render_entity = extracted_effect.render_entity.id();
                        match err {
                            CachedPropertiesError::InvalidBufferIndex(buffer_index)
                                => error!("Failed to remove cached properties of render entity {render_entity:?} from buffer #{buffer_index}: the index is invalid."),
                            CachedPropertiesError::BufferDeallocated(buffer_index)
                                => error!("Failed to remove cached properties of render entity {render_entity:?} from buffer #{buffer_index}: the buffer is not allocated."),
                        }
                    }
                    Ok(buffer_state) => {
                        if buffer_state == BufferState::Free {
                            // The entire buffer was deallocated; destroy all bind groups
                            // referencing it
                            let key = cached_effect_properties.to_key();
                            trace!("Destroying property bind group for key {key:?} due to property buffer deallocated.");
                            effect_bind_groups.property_bind_groups.remove(&key);
                        }
                    }
                }

                if extracted_effect.property_data.is_some() {
                    warn!(
                        "Effect on entity {:?} doesn't declare any property in its Module, but some property values were provided. Those values will be discarded.",
                        extracted_effect.main_entity.id(),
                    );
                }
            } else {
                // Effect has properties, needs a component. However it can only get one if
                // there's a buffer allocated.
                if let Some(buffer) =
                    property_cache.get_buffer(cached_effect_properties.buffer_index)
                {
                    // Insert a new component or overwrite the existing one
                    cmd.insert(CachedProperties {
                        layout: extracted_effect.property_layout.clone(),
                        buffer: buffer.clone(),
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
                            render_queue.write_buffer(
                                buffer,
                                cached_effect_properties.range.start as u64,
                                property_data,
                            );
                        } else {
                            error!(
                            "Cannot upload properties: existing property slice in property buffer #{} is too small ({} bytes) for the new data ({} bytes).",
                            cached_effect_properties.buffer_index,
                            cached_effect_properties.range.len(),
                            property_data.len()
                        );
                        }
                    }
                } else {
                    error!(
                    "Render entity {:?} has a CachedEffectProperties component referencing the property buffer #{}, but that buffer was not found.",
                    extracted_effect.render_entity.id(),
                    cached_effect_properties.buffer_index
                );
                    cmd.remove::<CachedProperties>();
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

        total_effect_count += 1;
    }
    trace!(
        "Collected {} extracted effect(s) and {} groups",
        total_effect_count,
        total_group_count
    );

    // Perform any GPU allocation if we (lazily) allocated some rows into the render
    // group dispatch indirect buffer.
    effects_meta.allocate_gpu(&render_device, &render_queue, &mut effect_bind_groups);

    // Write the entire spawner buffer for this frame, for all effects combined
    if effects_meta
        .spawner_buffer
        .write_buffer(&render_device, &render_queue)
    {
        // All property bind groups use the spawner buffer, which was reallocate
        effect_bind_groups.property_bind_groups.clear();
        effect_bind_groups.no_property_bind_group = None;
    }

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

pub(crate) fn batch_effects(
    mut commands: Commands,
    mut q_cached_effects: Query<(
        Entity,
        &CachedMesh,
        Option<&CachedProperties>,
        &mut CachedGroups,
        &mut DispatchBufferIndices,
        &mut BatchesInput,
    )>,
) {
    trace!("batch_effects");

    // Sort first by effect buffer index, then by slice range (see EffectSlice)
    // inside that buffer. This is critical for batching to work, because
    // batching effects is based on compatible items, which implies same GPU
    // buffer and continuous slice ranges (the next slice start must be equal to
    // the previous start end, without gap). EffectSlice already contains both
    // information, and the proper ordering implementation.
    // effect_entity_list.sort_by_key(|a| a.effect_slice.clone());

    // Loop on all extracted effects in order, and try to batch them together to
    // reduce draw calls. -- currently does nothing, batching was broken and never
    // fixed.
    // FIXME - This is in ECS order, if we re-add the sorting above we need a
    // different order here!
    trace!("Batching {} effects...", q_cached_effects.iter().len());
    for (
        entity,
        cached_mesh,
        cached_properties,
        mut cached_groups,
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
        if cached_groups.init_and_update_pipeline_ids.is_empty() {
            trace!(
                "Skipped cached effect on render entity {:?}: not extracted this frame.",
                entity
            );
            continue;
        }

        #[cfg(feature = "2d")]
        let z_sort_key_2d = input.z_sort_key_2d;

        #[cfg(feature = "3d")]
        let translation_3d = input.position;

        let local_group_count = cached_groups.init_and_update_pipeline_ids.len() as u32;

        // Spawn one shared EffectBatches for all groups of this effect. This contains
        // most of the data needed to drive rendering, except the per-group data.
        // However this doesn't drive rendering; this is just storage.
        let batches = EffectBatches::from_input(
            cached_mesh,
            &mut input,
            cached_groups.spawner_base,
            std::mem::take(&mut cached_groups.init_and_update_pipeline_ids),
            dispatch_buffer_indices.clone(),
            cached_groups
                .first_particle_group_buffer_index
                .unwrap_or_default(),
            cached_properties.map(|cp| PropertyBindGroupKey {
                buffer_index: cp.buffer_index,
                binding_size: cp.binding_size,
            }),
            cached_properties.map(|cp| cp.offset),
        );
        let batches_entity = commands.spawn(batches).insert(TemporaryRenderEntity).id();
        trace!(
            "Spawned effect batch with {} groups at entity {:?} from cached mesh of entity {:?}.",
            local_group_count,
            batches_entity,
            entity,
        );

        // Spawn one EffectDrawBatch per group, to actually drive rendering. Each group
        // renders with a different indirect call. These are the entities that the
        // render phase items will receive.
        for group_index in 0..local_group_count {
            commands
                .spawn(EffectDrawBatch {
                    batches_entity,
                    group_index,
                    #[cfg(feature = "2d")]
                    z_sort_key_2d,
                    #[cfg(feature = "3d")]
                    translation_3d,
                })
                .insert(TemporaryRenderEntity);
        }
    }
}

/// Per-buffer bind groups for a GPU effect buffer.
///
/// This contains all bind groups specific to a single [`EffectBuffer`].
///
/// [`EffectBuffer`]: crate::render::effect_cache::EffectBuffer
pub(crate) struct BufferBindGroups {
    /// Bind group for the render graphic shader.
    ///
    /// ```wgsl
    /// @binding(0) var<storage, read> particle_buffer : ParticleBuffer;
    /// @binding(1) var<storage, read> indirect_buffer : IndirectBuffer;
    /// @binding(2) var<storage, read> dispatch_indirect : DispatchIndirect;
    /// #ifdef RENDER_NEEDS_SPAWNER
    /// @binding(3) var<storage, read> spawner : Spawner;
    /// #endif
    /// ```
    render: BindGroup,
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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PropertyBindGroupKey {
    pub buffer_index: u32,
    pub binding_size: u32,
}

#[derive(Default, Resource)]
pub struct EffectBindGroups {
    /// Map from buffer index to the bind groups shared among all effects that
    /// use that buffer.
    particle_buffers: HashMap<u32, BufferBindGroups>,
    /// Map of bind groups for image assets used as particle textures.
    images: HashMap<AssetId<Image>, BindGroup>,
    /// Map from buffer index to its update render indirect bind group (group
    /// 3).
    // FIXME - possibly doesn't work with batch if different effects in same buffer need different
    // bindings, not sure...
    update_render_indirect_bind_groups: HashMap<u32, BindGroup>,
    /// Map from an effect material to its bind group.
    material_bind_groups: HashMap<Material, BindGroup>,
    /// Map from a [`PropertyBuffer`] index and a binding size to the
    /// corresponding bind group.
    property_bind_groups: HashMap<PropertyBindGroupKey, BindGroup>,
    /// Bind group for the variant without any property.
    no_property_bind_group: Option<BindGroup>,
}

impl EffectBindGroups {
    pub fn particle_render(&self, buffer_index: u32) -> Option<&BindGroup> {
        self.particle_buffers
            .get(&buffer_index)
            .map(|bg| &bg.render)
    }

    pub fn property(&self, key: Option<&PropertyBindGroupKey>) -> Option<&BindGroup> {
        if let Some(key) = key {
            self.property_bind_groups.get(key)
        } else {
            self.no_property_bind_group.as_ref()
        }
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
    effect_batches: &Query<(Entity, &mut EffectBatches)>,
    effect_draw_batches: &Query<(Entity, &mut EffectDrawBatch)>,
    render_pipeline: &mut ParticlesRenderPipeline,
    mut specialized_render_pipelines: Mut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    render_meshes: &RenderAssets<RenderMesh>,
    pipeline_cache: &PipelineCache,
    make_phase_item: F,
    #[cfg(all(feature = "2d", feature = "3d"))] pipeline_mode: PipelineMode,
) where
    T: SortedPhaseItem,
    F: Fn(CachedRenderPipelineId, (Entity, MainEntity), &EffectDrawBatch, u32, &ExtractedView) -> T,
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
            if batches
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
            let alpha_mask = ParticleRenderAlphaMaskPipelineKey::from(batches.layout_flags);
            let flipbook = batches.layout_flags.contains(LayoutFlags::FLIPBOOK);
            let needs_uv = batches.layout_flags.contains(LayoutFlags::NEEDS_UV);
            let needs_normal = batches.layout_flags.contains(LayoutFlags::NEEDS_NORMAL);
            let ribbons = batches.layout_flags.contains(LayoutFlags::RIBBONS);
            let image_count = batches.texture_layout.layout.len() as u8;

            // FIXME - Maybe it's better to copy the mesh layout into the batch, instead of
            // re-querying here...?
            let Some(render_mesh) = render_meshes.get(batches.mesh) else {
                trace!("Batch has no render mesh, skipped.");
                continue;
            };
            let mesh_layout = render_mesh.layout.clone();

            // Specialize the render pipeline based on the effect batch
            trace!(
                "Specializing render pipeline: render_shaders={:?} image_count={} alpha_mask={:?} flipbook={:?} hdr={}",
                batches.render_shaders,
                image_count,
                alpha_mask,
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
                    mesh_layout: Some(mesh_layout),
                    particle_layout: batches.particle_layout.clone(),
                    texture_layout: batches.texture_layout.clone(),
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
                (draw_entity, MainEntity::from(Entity::PLACEHOLDER)),
                draw_batch,
                draw_batch.group_index,
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
    effect_batches: &Query<(Entity, &mut EffectBatches)>,
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
    F: Fn(CachedRenderPipelineId, &EffectDrawBatch, u32, &ExtractedView) -> T::BinKey,
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

            if ParticleRenderAlphaMaskPipelineKey::from(batches.layout_flags) != alpha_mask {
                trace!(
                    "Mismatching alpha mask pipeline key (batches={:?}, expected={:?}). Skipped.",
                    batches.layout_flags,
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
            let alpha_mask = ParticleRenderAlphaMaskPipelineKey::from(batches.layout_flags);
            let flipbook = batches.layout_flags.contains(LayoutFlags::FLIPBOOK);
            let needs_uv = batches.layout_flags.contains(LayoutFlags::NEEDS_UV);
            let needs_normal = batches.layout_flags.contains(LayoutFlags::NEEDS_NORMAL);
            let ribbons = batches.layout_flags.contains(LayoutFlags::RIBBONS);
            let image_count = batches.texture_layout.layout.len() as u8;
            let render_mesh = render_meshes.get(batches.mesh);

            // Specialize the render pipeline based on the effect batch
            trace!(
                "Specializing render pipeline: render_shaders={:?} image_count={} alpha_mask={:?} flipbook={:?} hdr={}",
                batches.render_shaders,
                image_count,
                alpha_mask,
                flipbook,
                view.hdr
            );

            // Add a draw pass for the effect batch
            trace!("Emitting individual draws for batches and groups: group_batches.len()={} batches.render_shaders.len()={}", batches.group_batches.len(), batches.render_shaders.len());
            let render_shader_source = &batches.render_shaders[draw_batch.group_index as usize];
            trace!("Emit for group index #{}", draw_batch.group_index);

            let alpha_mode = batches.alpha_mode;

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
                    shader: render_shader_source.clone(),
                    mesh_layout: Some(mesh_layout),
                    particle_layout: batches.particle_layout.clone(),
                    texture_layout: batches.texture_layout.clone(),
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
    effect_batches: Query<(Entity, &mut EffectBatches)>,
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
                &effect_batches,
                &effect_draw_batches,
                &mut render_pipeline,
                specialized_render_pipelines.reborrow(),
                &render_meshes,
                &pipeline_cache,
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
                &render_meshes,
                &pipeline_cache,
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
                &render_meshes,
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
                &effect_batches,
                &effect_draw_batches,
                &mut render_pipeline,
                specialized_render_pipelines.reborrow(),
                &pipeline_cache,
                &render_meshes,
                |id, _batch, _group, _view| OpaqueNoLightmap3dBinKey {
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
/// This system runs in the [`RenderSet::Prepare`] render set, after Bevy has
/// updated the [`ViewUniforms`], which need to be referenced to get access to
/// the current camera view.
pub(crate) fn prepare_gpu_resources(
    mut effects_meta: ResMut<EffectsMeta>,
    render_device: Res<RenderDevice>,
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
}

pub(crate) fn prepare_bind_groups(
    mut effects_meta: ResMut<EffectsMeta>,
    mut effect_cache: ResMut<EffectCache>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    property_cache: Res<PropertyCache>,
    effect_batches: Query<(Entity, &mut EffectBatches)>,
    render_device: Res<RenderDevice>,
    dispatch_indirect_pipeline: Res<DispatchIndirectPipeline>,
    init_pipeline: Res<ParticlesInitPipeline>,
    update_pipeline: Res<ParticlesUpdatePipeline>,
    render_pipeline: ResMut<ParticlesRenderPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    // We can't simulate nor render anything without at least the spawner buffer
    if effects_meta.spawner_buffer.is_empty() {
        return;
    }
    let Some(spawner_buffer) = effects_meta.spawner_buffer.buffer().cloned() else {
        return;
    };

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

        // Create the bind group for the spawner parameters in the no-property variant.
        // For the other variants with properties, we need the binding size, so we'll
        // lazily create them below when looping on effect instances.
        trace!(
            "Spawner buffer bind group: size={} aligned_size={}",
            GpuSpawnerParams::min_size().get(),
            effects_meta.spawner_buffer.aligned_size()
        );
        assert!(
            effects_meta.spawner_buffer.aligned_size()
                >= GpuSpawnerParams::min_size().get() as usize
        );
        if effect_bind_groups.no_property_bind_group.is_none() {
            // If all effects use properties, there won't be a layout for the no-property
            // variant. This is fine, we don't need it.
            if let Some(layout) = property_cache.bind_group_layout(None) {
                trace!("Creating new bind group for no-property variant.");
                effect_bind_groups.no_property_bind_group = Some(
                    render_device.create_bind_group(
                        "hanabi:bind_group:no_property{}",
                        layout,
                        &[BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: &spawner_buffer,
                                offset: 0,
                                size: Some(
                                    NonZeroU64::new(
                                        effects_meta.spawner_buffer.aligned_size() as u64
                                    )
                                    .unwrap(),
                                ),
                            }),
                        }],
                    ),
                );
            }
        }

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

        let (init_render_indirect_spawn_bind_group, init_render_indirect_clone_bind_group) = match (
            effects_meta.render_effect_dispatch_buffer.buffer(),
            effects_meta.render_group_dispatch_buffer.buffer(),
        ) {
            (Some(render_effect_dispatch_buffer), Some(render_group_dispatch_buffer)) => (
                Some(render_device.create_bind_group(
                    "hanabi:bind_group_init_render_dispatch_spawn",
                    &init_pipeline.render_indirect_spawn_layout,
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
                    ],
                )),
                Some(render_device.create_bind_group(
                    "hanabi:bind_group_init_render_dispatch_clone",
                    &init_pipeline.render_indirect_clone_layout,
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
                )),
            ),

            (_, _) => (None, None),
        };

        // Create the bind group for the indirect render buffer use in the init shader
        effects_meta.init_render_indirect_spawn_bind_group = init_render_indirect_spawn_bind_group;
        effects_meta.init_render_indirect_clone_bind_group = init_render_indirect_clone_bind_group;
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
                    "Create new particle bind groups for buffer_index={} | particle_layout {:?}",
                    buffer_index,
                    buffer.particle_layout(),
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

    // Create the per-effect bind groups.
    for (entity, effect_batches) in effect_batches.iter() {
        #[cfg(feature = "trace")]
        let _span_buffer = bevy::utils::tracing::info_span!("create_batch_bind_groups").entered();

        // Create the property layout if needed
        if let Some(property_key) = &effect_batches.property_key {
            let Some(property_buffer) = property_cache.get_buffer(property_key.buffer_index) else {
                error!(
                    "Missing property buffer #{}, referenced by effect batch on render entity {:?}.",
                    property_key.buffer_index, entity
                );
                continue;
            };

            // This should always be non-zero if the property key is Some().
            let binding_size = NonZeroU64::new(property_key.binding_size as u64).unwrap();
            let Some(layout) = property_cache.bind_group_layout(Some(binding_size)) else {
                error!(
                    "Missing property bind group layout for binding size {}, referenced by effect batch on render entity {:?}.",
                    binding_size.get(), entity
                );
                continue;
            };

            effect_bind_groups
                .property_bind_groups
                .entry(*property_key)
                .or_insert_with(|| {
                    trace!(
                        "Creating new bind group for property buffer #{} and binding size {}",
                        property_key.buffer_index,
                        property_key.binding_size
                    );
                    render_device.create_bind_group(
                        Some(
                            &format!(
                                "hanabi:bind_group:property{}_size{}",
                                property_key.buffer_index, property_key.binding_size
                            )[..],
                        ),
                        layout,
                        &[
                            BindGroupEntry {
                                binding: 0,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: &spawner_buffer,
                                    offset: 0,
                                    size: Some(
                                        NonZeroU64::new(
                                            effects_meta.spawner_buffer.aligned_size() as u64
                                        )
                                        .unwrap(),
                                    ),
                                }),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: property_buffer,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                        ],
                    )
                });
        }

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
        // TODO - move this creation in RenderSet::PrepareBindGroups
        if effect_cache
            .create_sim_bind_group(effect_batches.buffer_index, &render_device, group_binding)
            .is_err()
        {
            error!("No particle buffer allocated for entity {:?}", entity);
            continue;
        }

        if effect_bind_groups
            .update_render_indirect_bind_groups
            .get(&effect_batches.buffer_index)
            .is_none()
        {
            let DispatchBufferIndices {
                render_effect_metadata_buffer_index: render_effect_dispatch_buffer_index,
                render_group_dispatch_indices,
                ..
            } = &effect_batches.dispatch_buffer_indices;
            let RenderGroupDispatchIndices::Allocated {
                first_render_group_dispatch_buffer_index,
                ..
            } = render_group_dispatch_indices
            else {
                continue;
            };

            let storage_alignment = effects_meta.gpu_limits.storage_buffer_align.get();
            let render_effect_indirect_size =
                GpuRenderEffectMetadata::aligned_size(storage_alignment);
            let total_render_group_indirect_size = NonZeroU64::new(
                GpuRenderGroupIndirect::aligned_size(storage_alignment).get()
                    * effect_batches.group_batches.len() as u64,
            )
            .unwrap();
            let particles_buffer_layout_update_render_indirect = render_device.create_bind_group(
                "hanabi:bind_group_update_render_group_dispatch",
                &update_pipeline.render_indirect_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: effects_meta.render_effect_dispatch_buffer.buffer().unwrap(),
                            offset: effects_meta.gpu_limits.render_effect_indirect_offset(
                                render_effect_dispatch_buffer_index.0,
                            ),
                            size: Some(render_effect_indirect_size),
                        }),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: effects_meta.render_group_dispatch_buffer.buffer().unwrap(),
                            offset: effects_meta.gpu_limits.render_group_indirect_offset(
                                first_render_group_dispatch_buffer_index.0,
                            ),
                            size: Some(total_render_group_indirect_size),
                        }),
                    },
                ],
            );

            trace!(
                "Created new update render indirect bind group for buffer index {}: \
                render_effect={} \
                render_group={} group_count={}",
                effect_batches.buffer_index,
                render_effect_dispatch_buffer_index.0,
                first_render_group_dispatch_buffer_index.0,
                effect_batches.group_batches.len()
            );

            effect_bind_groups
                .update_render_indirect_bind_groups
                .insert(
                    effect_batches.buffer_index,
                    particles_buffer_layout_update_render_indirect,
                );
        }

        // Ensure the particle texture(s) are available as GPU resources and that a bind
        // group for them exists
        // FIXME fix this insert+get below
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
        effects,
        effect_draw_batches,
    ) = params.get(world);
    let view_uniform = views.get(view).unwrap();
    let effects_meta = effects_meta.into_inner();
    let effect_bind_groups = effect_bind_groups.into_inner();
    let meshes = meshes.into_inner();
    let mesh_allocator = mesh_allocator.into_inner();
    let effect_draw_batch = effect_draw_batches.get(entity.0).unwrap();
    let effect_batches = effects.get(effect_draw_batch.batches_entity).unwrap();

    let gpu_limits = &effects_meta.gpu_limits;

    let Some(pipeline) = pipeline_cache.into_inner().get_render_pipeline(pipeline_id) else {
        return;
    };

    trace!("render pass");

    pass.set_render_pipeline(pipeline);

    let Some(render_mesh): Option<&RenderMesh> = meshes.get(effect_batches.mesh) else {
        return;
    };
    let Some(vertex_buffer_slice) = mesh_allocator.mesh_vertex_slice(&effect_batches.mesh) else {
        return;
    };

    let RenderGroupDispatchIndices::Allocated {
        first_render_group_dispatch_buffer_index,
        ..
    } = &effect_batches
        .dispatch_buffer_indices
        .render_group_dispatch_indices
    else {
        return;
    };

    // Vertex buffer containing the particle model to draw. Generally a quad.
    // FIXME - need to upload "vertex_buffer_slice.range.start as i32" into
    // "base_vertex" in the indirect struct...
    assert_eq!(
        effect_batches.mesh_buffer.id(),
        vertex_buffer_slice.buffer.id()
    );
    assert_eq!(effect_batches.mesh_slice, vertex_buffer_slice.range);
    pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));

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

    let render_group_dispatch_indirect_index =
        first_render_group_dispatch_buffer_index.offset(group_index);

    trace!(
        "Draw up to {} particles with {} vertices per particle for batch from buffer #{} \
            (render_group_dispatch_indirect_index={:?}, group_index={}).",
        effect_batch.slice.len(),
        render_mesh.vertex_count,
        effect_batches.buffer_index,
        render_group_dispatch_indirect_index,
        group_index,
    );

    match render_mesh.buffer_info {
        RenderMeshBufferInfo::Indexed {
            count: _,
            index_format,
        } => {
            let Some(index_buffer_slice) = mesh_allocator.mesh_index_slice(&effect_batches.mesh)
            else {
                return;
            };

            pass.set_index_buffer(index_buffer_slice.buffer.slice(..), 0, index_format);

            pass.draw_indexed_indirect(
                render_indirect_buffer,
                render_group_dispatch_indirect_index.0 as u64
                    * u32::from(gpu_limits.render_group_indirect_aligned_size) as u64,
            );
        }
        RenderMeshBufferInfo::NonIndexed => {
            pass.draw_indirect(
                render_indirect_buffer,
                render_group_dispatch_indirect_index.0 as u64
                    * u32::from(gpu_limits.render_group_indirect_aligned_size) as u64,
            );
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

fn create_sim_bind_group_layout(
    render_device: &RenderDevice,
    label: &str,
    particle_layout_min_binding_size: NonZero<u64>,
) -> BindGroupLayout {
    let particle_group_size =
        GpuParticleGroup::aligned_size(render_device.limits().min_storage_buffer_offset_alignment);
    let entries = [
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

    trace!(
        "Creating particle bind group layout '{}' for simulation passes with {} entries.",
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

        // Get the Entity containing the ViewEffectsEntity component used as container
        // for the input data for this node.
        // let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        let pipeline_cache = world.resource::<PipelineCache>();

        let effects_meta = world.resource::<EffectsMeta>();
        let effect_cache = world.resource::<EffectCache>();
        let effect_bind_groups = world.resource::<EffectBindGroups>();
        let dispatch_indirect_pipeline = world.resource::<DispatchIndirectPipeline>();
        // let render_queue = world.resource::<RenderQueue>();

        // Make sure to schedule any buffer copy from changed effects before accessing
        // them
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

        // Compute init pass
        // let mut total_group_count = 0;
        {
            {
                trace!("init: loop over effect batches...");

                // Dispatch init compute jobs
                for (entity, batches) in self.effect_query.iter_manual(world) {
                    let RenderGroupDispatchIndices::Allocated {
                        first_render_group_dispatch_buffer_index,
                        trail_dispatch_buffer_indices,
                    } = &batches
                        .dispatch_buffer_indices
                        .render_group_dispatch_indices
                    else {
                        continue;
                    };

                    for &dest_group_index in batches.group_order.iter() {
                        let initializer = &batches.initializers[dest_group_index as usize];
                        let dest_render_group_dispatch_buffer_index =
                            first_render_group_dispatch_buffer_index.offset(dest_group_index);

                        // Destination group spawners are packed one after one another.
                        let spawner_index = batches.spawner_base + dest_group_index;
                        let spawner_aligned_size = effects_meta.spawner_buffer.aligned_size();
                        assert!(
                            spawner_aligned_size >= GpuSpawnerParams::min_size().get() as usize
                        );
                        let spawner_offset = spawner_index * spawner_aligned_size as u32;

                        let property_offset = batches.property_offset;

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

                                let Some(init_pipeline) = pipeline_cache.get_compute_pipeline(
                                    batches.init_and_update_pipeline_ids[dest_group_index as usize]
                                        .init,
                                ) else {
                                    if let CachedPipelineState::Err(err) = pipeline_cache
                                        .get_compute_pipeline_state(
                                            batches.init_and_update_pipeline_ids
                                                [dest_group_index as usize]
                                                .init,
                                        )
                                    {
                                        error!(
                                            "Failed to find init pipeline #{} for effect {:?}: \
                                             {:?}",
                                            batches.init_and_update_pipeline_ids
                                                [dest_group_index as usize]
                                                .init
                                                .id(),
                                            entity,
                                            err
                                        );
                                    }
                                    continue;
                                };

                                // Do not dispatch any init work if there's nothing to spawn this
                                // frame
                                let spawn_count = effect_spawner.spawn_count;
                                if spawn_count == 0 {
                                    continue;
                                }

                                const WORKGROUP_SIZE: u32 = 64;
                                let workgroup_count = spawn_count.div_ceil(WORKGROUP_SIZE);

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
                                    effect_cache.init_bind_group(batches.buffer_index)
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
                                    spawner_index,
                                    spawner_offset,
                                    render_effect_indirect_offset,
                                    render_group_indirect_offset,
                                );

                                // Dispatch init pass
                                compute_pass.set_pipeline(init_pipeline);
                                compute_pass.set_bind_group(
                                    0,
                                    effects_meta.sim_params_bind_group.as_ref().unwrap(),
                                    &[],
                                );
                                compute_pass.set_bind_group(1, particles_init_bind_group, &[]);
                                let offsets = if let Some(property_offset) = property_offset {
                                    vec![spawner_offset, property_offset]
                                } else {
                                    vec![spawner_offset]
                                };
                                compute_pass.set_bind_group(
                                    2,
                                    effect_bind_groups
                                        .property(batches.property_key.as_ref())
                                        .unwrap(),
                                    &offsets[..],
                                );
                                compute_pass.set_bind_group(
                                    3,
                                    effects_meta
                                        .init_render_indirect_spawn_bind_group
                                        .as_ref()
                                        .unwrap(),
                                    &[
                                        render_effect_indirect_offset as u32,
                                        render_group_indirect_offset as u32,
                                    ],
                                );
                                compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                                trace!("init compute dispatched");
                            }

                            EffectInitializer::Cloner(EffectCloner {
                                cloner,
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
                                    effect_cache.init_bind_group(batches.buffer_index)
                                else {
                                    error!(
                                        "Failed to find clone particle buffer bind group \
                                                 for entity {:?}, buffer index {}",
                                        entity, batches.buffer_index
                                    );
                                    continue;
                                };

                                let render_effect_dispatch_buffer_index = batches
                                    .dispatch_buffer_indices
                                    .render_effect_metadata_buffer_index;
                                let clone_dest_render_group_dispatch_buffer_index =
                                    trail_dispatch_buffer_indices[&dest_group_index].dest;
                                let clone_src_render_group_dispatch_buffer_index =
                                    trail_dispatch_buffer_indices[&dest_group_index].src;

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

                                let first_update_group_dispatch_buffer_index = batches
                                    .dispatch_buffer_indices
                                    .first_update_group_dispatch_buffer_index;

                                let src_group_index = cloner.src_group_index;
                                let update_src_group_dispatch_buffer_offset =
                                    effects_meta.gpu_limits.dispatch_indirect_offset(
                                        first_update_group_dispatch_buffer_index.0
                                            + src_group_index,
                                    );

                                compute_pass.set_pipeline(clone_pipeline);
                                compute_pass.set_bind_group(
                                    0,
                                    effects_meta.sim_params_bind_group.as_ref().unwrap(),
                                    &[],
                                );
                                compute_pass.set_bind_group(1, particles_init_bind_group, &[]);
                                let offsets = if let Some(property_offset) = property_offset {
                                    vec![spawner_offset, property_offset]
                                } else {
                                    vec![spawner_offset]
                                };
                                compute_pass.set_bind_group(
                                    2,
                                    effect_bind_groups
                                        .property(batches.property_key.as_ref())
                                        .unwrap(),
                                    &offsets[..],
                                );
                                compute_pass.set_bind_group(
                                    3,
                                    effects_meta
                                        .init_render_indirect_clone_bind_group
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
                                    trace!(
                                        "record commands for init clone pipeline of effect {:?} \
                                            first_update_group_dispatch_buffer_index={} \
                                            src_group_index={} \
                                            update_src_group_dispatch_buffer_offset={}...",
                                        batches.handle,
                                        first_update_group_dispatch_buffer_index.0,
                                        src_group_index,
                                        update_src_group_dispatch_buffer_offset,
                                    );

                                    compute_pass.dispatch_workgroups_indirect(
                                        dispatch_indirect_buffer,
                                        update_src_group_dispatch_buffer_offset as u64,
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
            let workgroup_count = total_group_count.div_ceil(WORKGROUP_SIZE);

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
                let Some(particles_update_bind_group) =
                    effect_cache.update_bind_group(batches.buffer_index)
                else {
                    error!(
                        "Failed to find update particle buffer bind group for entity {:?}, buffer index {}",
                        entity, batches.buffer_index
                    );
                    continue;
                };

                let first_update_group_dispatch_buffer_index = batches
                    .dispatch_buffer_indices
                    .first_update_group_dispatch_buffer_index;

                let Some(update_render_indirect_bind_group) = effect_bind_groups
                    .update_render_indirect_bind_groups
                    .get(&batches.buffer_index)
                else {
                    error!(
                        "Failed to find update render indirect bind group for buffer index: {}, IDs present: {:?}",
                        batches.buffer_index,
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
                    let spawner_index = batches.spawner_base + group_index;
                    let spawner_aligned_size = effects_meta.spawner_buffer.aligned_size();
                    assert!(spawner_aligned_size >= GpuSpawnerParams::min_size().get() as usize);
                    let spawner_offset = spawner_index * spawner_aligned_size as u32;

                    let property_offset = batches.property_offset;

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
                        spawner_index,
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
                    let offsets = if let Some(property_offset) = property_offset {
                        vec![spawner_offset, property_offset]
                    } else {
                        vec![spawner_offset]
                    };
                    compute_pass.set_bind_group(
                        2,
                        effect_bind_groups
                            .property(batches.property_key.as_ref())
                            .unwrap(),
                        &offsets[..],
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
