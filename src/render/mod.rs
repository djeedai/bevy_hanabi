#[cfg(feature = "2d")]
use bevy::utils::FloatOrd;
use bevy::{
    asset::{AssetEvent, AssetId, Assets, Handle},
    core::{Pod, Zeroable},
    ecs::{
        prelude::*,
        system::{lifetimeless::*, SystemParam, SystemState},
    },
    log::trace,
    math::{Mat4, Vec3},
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo},
        render_phase::{Draw, DrawFunctions, PhaseItem, RenderPhase, TrackedRenderPass},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::{BevyDefault, Image},
        view::{
            ExtractedView, InheritedVisibility, Msaa, ViewTarget, ViewUniform, ViewUniformOffset,
            ViewUniforms, ViewVisibility, VisibleEntities,
        },
        Extract,
    },
    time::Time,
    transform::components::GlobalTransform,
    utils::HashMap,
};
use bitflags::bitflags;
use rand::random;
use std::marker::PhantomData;
use std::{
    borrow::Cow,
    num::{NonZeroU32, NonZeroU64},
};

#[cfg(feature = "2d")]
use bevy::core_pipeline::core_2d::Transparent2d;
#[cfg(feature = "3d")]
use bevy::core_pipeline::core_3d::{AlphaMask3d, Transparent3d};

use crate::{
    asset::EffectAsset,
    modifier::ForceFieldSource,
    next_multiple_of,
    render::batch::{BatchInput, BatchState, Batcher, EffectBatch},
    spawn::EffectSpawner,
    CompiledParticleEffect, EffectProperties, EffectShader, ParticleLayout, PropertyLayout,
    RemovedEffectsEvent, SimulationCondition, SimulationSpace,
};

mod aligned_buffer_vec;
mod batch;
mod buffer_table;
mod effect_cache;
mod shader_cache;

use aligned_buffer_vec::AlignedBufferVec;
use buffer_table::{BufferTable, BufferTableId};
pub(crate) use effect_cache::{EffectCache, EffectCacheId};

pub use effect_cache::{EffectBuffer, EffectSlice};
pub use shader_cache::ShaderCache;

/// Labels for the Hanabi systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, SystemSet)]
pub enum EffectSystems {
    /// Tick all effect instances to generate particle spawn counts.
    ///
    /// This system runs during the [`PostUpdate`] schedule. Any system which
    /// modifies an effect spawner should run before this set to ensure the
    /// spawner takes into account the newly set values during its ticking.
    TickSpawners,

    /// Compile the effect instances, updating the [`CompiledParticleEffect`]
    /// components.
    ///
    /// This system runs during the [`PostUpdate`] schedule. This is largely an
    /// internal task which can be ignored by most users.
    CompileEffects,

    /// Update the properties of the effect instance based on the declared
    /// properties in the [`EffectAsset`], updating the associated
    /// [`EffectProperties`] component.
    ///
    /// This system runs during Bevy's own [`UpdateAssets`] schedule, after the
    /// assets have been updated. Any system which modifies an
    /// [`EffectAsset`]'s declared properties should run before [`UpdateAssets`]
    /// in order for changes to be taken into account in the same frame.
    ///
    /// [`UpdateAssets`]: bevy::asset::UpdateAssets
    UpdatePropertiesFromAsset,

    /// Gather all removed [`ParticleEffect`] components during the
    /// [`PostUpdate`] set, to be able to clean-up GPU resources.
    ///
    /// Systems deleting entities with a [`ParticleEffect`] component should run
    /// before this set if they want the particle effect is cleaned-up during
    /// the same frame.
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    GatherRemovedEffects,

    /// Prepare effect assets for the extracted effects.
    PrepareEffectAssets,

    /// Queue the GPU commands for the extracted effects.
    QueueEffects,

    /// Prepare GPU data for the queued effects.
    PrepareEffectGpuResources,
}

/// Simulation parameters, available to all shaders of all effects.
#[derive(Debug, Default, Clone, Copy, Resource)]
pub(crate) struct SimParams {
    /// Current effect system simulation time since startup, in seconds.
    time: f64,
    /// Delta time, in seconds, since last effect system update.
    delta_time: f32,
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
    /// Total number of effects to simulate this frame. Used by the indirect
    /// compute pipeline to cap the compute thread to the actual number of
    /// effect to process.
    ///
    /// This is only used by the `vfx_indirect` compute shader.
    num_effects: u32,
    /// Stride in bytes of a render indirect block, used to index the effect's
    /// block based on its index.
    ///
    /// This is only used by the `vfx_indirect` compute shader.
    render_stride: u32,
    /// Stride in bytes of a dispatch indirect block, used to index the effect's
    /// block based on its index.
    ///
    /// This is only used by the `vfx_indirect` compute shader.
    dispatch_stride: u32,
}

impl Default for GpuSimParams {
    fn default() -> Self {
        Self {
            delta_time: 0.04,
            time: 0.0,
            num_effects: 0,
            render_stride: 0,   // invalid
            dispatch_stride: 0, // invalid
        }
    }
}

impl From<SimParams> for GpuSimParams {
    fn from(src: SimParams) -> Self {
        Self {
            delta_time: src.delta_time,
            time: src.time as f32,
            ..default()
        }
    }
}

/// GPU representation of a [`ForceFieldSource`].
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
struct GpuForceFieldSource {
    pub position_or_direction: Vec3,
    pub max_radius: f32,
    pub min_radius: f32,
    pub mass: f32,
    pub force_exponent: f32,
    pub conform_to_sphere: f32,
}

impl From<ForceFieldSource> for GpuForceFieldSource {
    fn from(source: ForceFieldSource) -> Self {
        Self {
            position_or_direction: source.position,
            max_radius: source.max_radius,
            min_radius: source.min_radius,
            mass: source.mass,
            force_exponent: source.force_exponent,
            conform_to_sphere: if source.conform_to_sphere { 1.0 } else { 0.0 },
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

/// GPU representation of spawner parameters.
///
/// This structure contains the fixed-size part of the parameters. Inside the
/// GPU buffer, it is followed by an array of [`GpuForceFieldSource`], which
/// together form the spawner parameter buffer.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
struct GpuSpawnerParams {
    /// Transform of the effect, as a Mat4 without the last row (which is always
    /// (0,0,0,1) for an affine transform), stored transposed as a mat3x4 to
    /// avoid padding in WGSL. This is either added to emitted particles at
    /// spawn time, if the effect simulated in world space, or to all
    /// simulated particles if the effect is simulated in local space.
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
    /// Index of the effect into the indirect dispatch and render buffers.
    effect_index: u32,
    /// Force field components. One GpuForceFieldSource takes up 32 bytes.
    force_field: [GpuForceFieldSource; ForceFieldSource::MAX_SOURCES],
}

// FIXME - min_storage_buffer_offset_alignment
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
pub struct GpuRenderIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub base_index: u32,
    pub vertex_offset: i32,
    pub base_instance: u32,
    //
    pub alive_count: u32,
    pub dead_count: u32,
    pub max_spawn: u32,
    //
    pub ping: u32,
    pub max_update: u32,
    pub __pad1: u32,
    pub __pad2: u32,
    // FIXME - min_storage_buffer_offset_alignment
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
        let world = world.cell();
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        // The GpuSpawnerParams is bound as an array element or as a standalone struct,
        // so needs the proper align. Because WGSL removed the @stride attribute, we pad
        // the WGSL type manually, so need to enforce min_binding_size everywhere.
        let item_align = render_device.limits().min_storage_buffer_offset_alignment as usize;
        let spawner_aligned_size =
            next_multiple_of(GpuSpawnerParams::min_size().get() as usize, item_align);
        trace!(
            "Aligning spawner params to {} bytes as device limits requires. Size: {} bytes.",
            item_align,
            spawner_aligned_size
        );

        trace!(
            "GpuRenderIndirect: min_size={} | GpuDispatchIndirect: min_size={}",
            GpuRenderIndirect::min_size(),
            GpuDispatchIndirect::min_size()
        );
        let dispatch_indirect_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(GpuRenderIndirect::min_size()),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(GpuDispatchIndirect::min_size()),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                NonZeroU64::new(spawner_aligned_size as u64).unwrap(),
                            ),
                        },
                        count: None,
                    },
                ],
                label: Some("hanabi:bind_group_layout:dispatch_indirect_dispatch_indirect"),
            });

        trace!("GpuSimParams: min_size={}", GpuSimParams::min_size());
        let sim_params_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuSimParams::min_size()),
                    },
                    count: None,
                }],
                label: Some("hanabi:bind_group_layout:dispatch_indirect_sim_params"),
            });

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("hanabi:pipeline_layout:dispatch_indirect"),
            bind_group_layouts: &[&dispatch_indirect_layout, &sim_params_layout],
            push_constant_ranges: &[],
        });

        // We need to pad the Spawner WGSL struct based on the device padding so that we
        // can use it as an array element but also has a direct struct binding.
        let spawner_padding_code = if GpuSpawnerParams::min_size().get() as usize
            != spawner_aligned_size
        {
            let padding_size = spawner_aligned_size - GpuSpawnerParams::min_size().get() as usize;
            assert!(padding_size % 4 == 0);
            format!("padding: array<u32, {}>", padding_size / 4)
        } else {
            "".to_string()
        };
        let indirect_code = include_str!("vfx_indirect.wgsl")
            .to_string()
            .replace("{{SPAWNER_PADDING}}", &spawner_padding_code);
        debug!("Create indirect dispatch shader:\n{}", indirect_code);
        let shader_module = render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("hanabi:vfx_indirect_shader"),
            source: ShaderSource::Wgsl(Cow::Owned(indirect_code)),
        });

        let pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("hanabi:compute_pipeline:dispatch_indirect"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });

        Self {
            dispatch_indirect_layout,
            pipeline,
        }
    }
}

#[derive(Resource)]
pub(crate) struct ParticlesInitPipeline {
    /// Render device the pipeline is attached to.
    render_device: RenderDevice,
    sim_params_layout: BindGroupLayout,
    spawner_buffer_layout: BindGroupLayout,
    render_indirect_layout: BindGroupLayout,
}

impl FromWorld for ParticlesInitPipeline {
    fn from_world(world: &mut World) -> Self {
        let world = world.cell();
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let limits = render_device.limits();
        bevy::log::info!(
            "GPU limits:\n- max_compute_invocations_per_workgroup={}\n- max_compute_workgroup_size_x={}\n- max_compute_workgroup_size_y={}\n- max_compute_workgroup_size_z={}\n- max_compute_workgroups_per_dimension={}\n- min_storage_buffer_offset_alignment={}\n- max_storage_buffers_per_shader_stage={}\n- max_bind_groups={}",
            limits.max_compute_invocations_per_workgroup, limits.max_compute_workgroup_size_x, limits.max_compute_workgroup_size_y, limits.max_compute_workgroup_size_z,
            limits.max_compute_workgroups_per_dimension, limits.min_storage_buffer_offset_alignment, limits.max_storage_buffers_per_shader_stage, limits.max_bind_groups
        );

        trace!("GpuSimParams: min_size={}", GpuSimParams::min_size());
        let sim_params_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuSimParams::min_size()),
                    },
                    count: None,
                }],
                label: Some("hanabi:bind_group_layout:update_sim_params"),
            });

        trace!(
            "GpuSpawnerParams: min_size={}",
            GpuSpawnerParams::min_size()
        );
        let spawner_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: Some(GpuSpawnerParams::min_size()),
                    },
                    count: None,
                }],
                label: Some("hanabi:buffer_layout:init_spawner"),
            });

        trace!(
            "GpuRenderIndirect: min_size={}",
            GpuRenderIndirect::min_size()
        );
        let render_indirect_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: Some(GpuRenderIndirect::min_size()),
                    },
                    count: None,
                }],
                label: Some("hanabi:bind_group_layout:init_render_indirect"),
            });

        Self {
            render_device: render_device.clone(),
            sim_params_layout,
            spawner_buffer_layout,
            render_indirect_layout,
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub(crate) struct ParticleInitPipelineKey {
    /// Compute shader, with snippets applied, but not preprocessed yet.
    shader: Handle<Shader>,
    /// Minimum binding size in bytes for the particle layout buffer.
    particle_layout_min_binding_size: NonZeroU64,
    /// Minimum binding size in bytes for the property layout buffer, if the
    /// effect has any property. Otherwise this is `None`.
    property_layout_min_binding_size: Option<NonZeroU64>,
}

impl SpecializedComputePipeline for ParticlesInitPipeline {
    type Key = ParticleInitPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        trace!(
            "GpuParticle: attributes.min_binding_size={} properties.min_binding_size={}",
            key.particle_layout_min_binding_size.get(),
            key.property_layout_min_binding_size
                .map(|sz| sz.get())
                .unwrap_or(0),
        );

        let mut entries = Vec::with_capacity(3);
        // (1,0) ParticleBuffer
        entries.push(BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: true,
                min_binding_size: Some(key.particle_layout_min_binding_size),
            },
            count: None,
        });
        // (1,1) IndirectBuffer
        entries.push(BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: true,
                min_binding_size: BufferSize::new(12),
            },
            count: None,
        });
        if let Some(min_binding_size) = key.property_layout_min_binding_size {
            // (1,2) Properties
            entries.push(BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, // TODO
                    min_binding_size: Some(min_binding_size),
                },
                count: None,
            });
        }

        let label = "hanabi:init_particles_buffer_layout";
        trace!(
            "Creating particle bind group layout '{}' for init pass with {} entries.",
            label,
            entries.len()
        );
        let particles_buffer_layout =
            self.render_device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    entries: &entries,
                    label: Some(label),
                });

        ComputePipelineDescriptor {
            label: Some("hanabi:pipeline_init_compute".into()),
            layout: vec![
                self.sim_params_layout.clone(),
                particles_buffer_layout,
                self.spawner_buffer_layout.clone(),
                self.render_indirect_layout.clone(),
            ],
            shader: key.shader,
            shader_defs: vec![],
            entry_point: "main".into(),
            push_constant_ranges: Vec::new(),
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
        let world = world.cell();
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let limits = render_device.limits();
        bevy::log::info!(
            "GPU limits:\n- max_compute_invocations_per_workgroup={}\n- max_compute_workgroup_size_x={}\n- max_compute_workgroup_size_y={}\n- max_compute_workgroup_size_z={}\n- max_compute_workgroups_per_dimension={}\n- min_storage_buffer_offset_alignment={}",
            limits.max_compute_invocations_per_workgroup, limits.max_compute_workgroup_size_x, limits.max_compute_workgroup_size_y, limits.max_compute_workgroup_size_z,
            limits.max_compute_workgroups_per_dimension, limits.min_storage_buffer_offset_alignment
        );

        trace!("GpuSimParams: min_size={}", GpuSimParams::min_size());
        let sim_params_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuSimParams::min_size()),
                    },
                    count: None,
                }],
                label: Some("hanabi:update_sim_params_layout"),
            });

        trace!(
            "GpuSpawnerParams: min_size={}",
            GpuSpawnerParams::min_size()
        );
        let spawner_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: Some(GpuSpawnerParams::min_size()),
                    },
                    count: None,
                }],
                label: Some("hanabi:update_spawner_buffer_layout"),
            });

        trace!(
            "GpuRenderIndirect: min_size={}",
            GpuRenderIndirect::min_size()
        );
        let render_indirect_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: Some(GpuRenderIndirect::min_size()),
                    },
                    count: None,
                }],
                label: Some("hanabi:update_render_indirect_layout"),
            });

        Self {
            render_device: render_device.clone(),
            sim_params_layout,
            spawner_buffer_layout,
            render_indirect_layout,
        }
    }
}

#[derive(Default, Clone, Hash, PartialEq, Eq)]
pub(crate) struct ParticleUpdatePipelineKey {
    /// Compute shader, with snippets applied, but not preprocessed yet.
    shader: Handle<Shader>,
    /// Particle layout.
    particle_layout: ParticleLayout,
    /// Property layout.
    property_layout: PropertyLayout,
}

impl SpecializedComputePipeline for ParticlesUpdatePipeline {
    type Key = ParticleUpdatePipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        trace!(
            "GpuParticle: attributes.min_binding_size={} properties.min_binding_size={}",
            key.particle_layout.min_binding_size().get(),
            if key.property_layout.is_empty() {
                0
            } else {
                key.property_layout.min_binding_size().get()
            },
        );

        let mut entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: true,
                    min_binding_size: Some(key.particle_layout.min_binding_size()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: true,
                    min_binding_size: BufferSize::new(12),
                },
                count: None,
            },
        ];
        if !key.property_layout.is_empty() {
            entries.push(BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, // TODO
                    min_binding_size: Some(key.property_layout.min_binding_size()),
                },
                count: None,
            });
        }

        let label = "hanabi:update_particles_buffer_layout";
        trace!(
            "Creating particle bind group layout '{}' for update pass with {} entries.",
            label,
            entries.len()
        );
        let particles_buffer_layout =
            self.render_device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    entries: &entries,
                    label: Some(label),
                });

        ComputePipelineDescriptor {
            label: Some("hanabi:pipeline_update_compute".into()),
            layout: vec![
                self.sim_params_layout.clone(),
                particles_buffer_layout,
                self.spawner_buffer_layout.clone(),
                self.render_indirect_layout.clone(),
            ],
            shader: key.shader,
            shader_defs: vec![],
            entry_point: "main".into(),
            push_constant_ranges: Vec::new(),
        }
    }
}

#[derive(Resource)]
pub(crate) struct ParticlesRenderPipeline {
    render_device: RenderDevice,
    view_layout: BindGroupLayout,
    material_layout: BindGroupLayout,
}

impl FromWorld for ParticlesRenderPipeline {
    fn from_world(world: &mut World) -> Self {
        let world = world.cell();
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
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
            label: Some("hanabi:view_layout_render"),
        });

        let material_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("hanabi:material_layout_render"),
        });

        Self {
            render_device: render_device.clone(),
            view_layout,
            material_layout,
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
    /// Key: PARTICLE_TEXTURE
    /// Define a texture sampled to modulate the particle color.
    /// This key requires the presence of UV coordinates on the particle
    /// vertices.
    has_image: bool,
    /// Key: PARTICLE_SCREEN_SPACE_SIZE
    /// The particle size is expressed in screen space. The particle has a
    /// constant size on screen (in logical pixels) which is not influenced by
    /// the camera projection (and so, not influenced by the distance to the
    /// camera).
    screen_space_size: bool,
    /// Key: LOCAL_SPACE_SIMULATION
    /// The effect is simulated in local space, and during rendering all
    /// particles are transformed by the effect's [`GlobalTransform`].
    local_space_simulation: bool,
    /// Key: USE_ALPHA_MASK
    /// The effect is rendered with alpha masking.
    use_alpha_mask: bool,
    /// Key: FLIPBOOK
    /// The effect is rendered with flipbook texture animation based on the
    /// sprite index of each particle.
    flipbook: bool,
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
            has_image: false,
            screen_space_size: false,
            local_space_simulation: false,
            use_alpha_mask: false,
            flipbook: false,
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
                    min_binding_size: Some(GpuDispatchIndirect::min_size()),
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
        let particles_buffer_layout =
            self.render_device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    entries: &entries,
                    label: Some("hanabi:buffer_layout_render"),
                });

        let mut layout = vec![self.view_layout.clone(), particles_buffer_layout];
        let mut shader_defs = vec![];

        // Key: PARTICLE_TEXTURE
        if key.has_image {
            layout.push(self.material_layout.clone());
            shader_defs.push("PARTICLE_TEXTURE".into());
            // //  @location(1) vertex_uv: vec2<f32>
            // vertex_buffer_layout.attributes.push(VertexAttribute {
            //     format: VertexFormat::Float32x2,
            //     offset: 12,
            //     shader_location: 1,
            // });
            // vertex_buffer_layout.array_stride += 8;
        }

        // Key: PARTICLE_SCREEN_SPACE_SIZE
        if key.screen_space_size {
            shader_defs.push("PARTICLE_SCREEN_SPACE_SIZE".into());
        }

        // Key: LOCAL_SPACE_SIMULATION
        if key.local_space_simulation {
            shader_defs.push("LOCAL_SPACE_SIMULATION".into());
        }

        // Key: USE_ALPHA_MASK
        if key.use_alpha_mask {
            shader_defs.push("USE_ALPHA_MASK".into());
        }

        // Key: FLIPBOOK
        if key.flipbook {
            shader_defs.push("FLIPBOOK".into());
        }

        #[cfg(all(feature = "2d", feature = "3d"))]
        let depth_stencil = match key.pipeline_mode {
            // Bevy's Transparent2d render phase doesn't support a depth-stencil buffer.
            PipelineMode::Camera2d => None,
            PipelineMode::Camera3d => Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
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
            depth_write_enabled: false,
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
                    blend: Some(BlendState::ALPHA_BLENDING),
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
    /// Number of particles to spawn this frame for the effect.
    /// Obtained from calling [`EffectSpawner::tick()`] on the source effect
    /// instance.
    ///
    /// [`EffectSpawner::tick()`]: crate::EffectSpawner::tick
    pub spawn_count: u32,
    /// Global transform of the effect origin, extracted from the
    /// [`GlobalTransform`].
    pub transform: Mat4,
    /// Inverse global transform of the effect origin, extracted from the
    /// [`GlobalTransform`].
    pub inverse_transform: Mat4,
    /// Force field applied to all particles in the "update" phase.
    // FIXME - Remove from here, this should be using properties. Only left here for back-compat
    // until we have a proper graph solution to replace it.
    force_field: [ForceFieldSource; ForceFieldSource::MAX_SOURCES],
    /// Layout flags.
    pub layout_flags: LayoutFlags,
    /// Texture to modulate the particle color.
    pub image_handle: Handle<Image>,
    /// Effect shader.
    pub effect_shader: EffectShader,
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
    /// Capacity of the effect (and therefore, the particle buffer), in number
    /// of particles.
    pub capacity: u32,
    /// Layout of particle attributes.
    pub particle_layout: ParticleLayout,
    /// Layout of properties for the effect, if properties are used at all, or
    /// an empty layout.
    pub property_layout: PropertyLayout,
    pub layout_flags: LayoutFlags,
    /// Handle of the effect asset.
    pub handle: Handle<EffectAsset>,
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
    time: Extract<Res<Time>>,
    effects: Extract<Res<Assets<EffectAsset>>>,
    _images: Extract<Res<Assets<Image>>>,
    mut query: Extract<
        ParamSet<(
            // All existing ParticleEffect components
            Query<(
                Entity,
                Option<&InheritedVisibility>,
                Option<&ViewVisibility>,
                &EffectSpawner,
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
) {
    trace!("extract_effects");

    // Save simulation params into render world
    let dt = time.delta_seconds();
    sim_params.time = time.elapsed_seconds_f64();
    sim_params.delta_time = dt;

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
        "Found {} removed entities.",
        extracted_effects.removed_effect_entities.len()
    );

    // Collect added effects for later GPU data allocation
    extracted_effects.added_effects = query
        .p1()
        .iter()
        .map(|(entity, effect)| {
            let handle = effect.asset.clone_weak();
            let asset = effects.get(&effect.asset).unwrap();
            let particle_layout = asset.particle_layout();
            assert!(
                particle_layout.size() > 0,
                "Invalid empty particle layout for effect '{}' on entity {:?}. Did you forget to add some modifier to the asset?",
                asset.name,
                entity
            );
            let property_layout = asset.property_layout();

            let mut layout_flags = LayoutFlags::NONE;
            if asset.simulation_space == SimulationSpace::Local {
                layout_flags |= LayoutFlags::LOCAL_SPACE_SIMULATION;
            }
            if let crate::AlphaMode::Mask(_) = &asset.alpha_mode {
                layout_flags |= LayoutFlags::USE_ALPHA_MASK;
            }
            // TODO - should we init the other flags here? (they're currently not used)

            trace!("Found new effect: entity {:?} | capacity {} | particle_layout {:?} | property_layout {:?}", entity, asset.capacity(), particle_layout, property_layout);
            AddedEffect {
                entity,
                capacity: asset.capacity(),
                particle_layout,
                property_layout,
                layout_flags,
                handle,
            }
        })
        .collect();

    // Loop over all existing effects to update them
    extracted_effects.effects.clear();
    for (
        entity,
        maybe_inherited_visibility,
        maybe_view_visibility,
        spawner,
        effect,
        maybe_properties,
        transform,
    ) in query.p0().iter_mut()
    {
        // Check if shaders are configured
        let Some(effect_shader) = effect.get_configured_shader() else {
            continue;
        };

        // Check if hidden, unless always simulated
        if effect.simulation_condition == SimulationCondition::WhenVisible
            && !maybe_inherited_visibility
                .map(|cv| cv.get())
                .unwrap_or(true)
            && !maybe_view_visibility.map(|cv| cv.get()).unwrap_or(true)
        {
            continue;
        }

        // Retrieve other values from the compiled effect
        let spawn_count = spawner.spawn_count();
        let force_field = effect.force_field; // TEMP

        // Check if asset is available, otherwise silently ignore
        let Some(asset) = effects.get(&effect.asset) else {
            trace!(
                "EffectAsset not ready; skipping ParticleEffect instance on entity {:?}.",
                entity
            );
            continue;
        };

        #[cfg(feature = "2d")]
        let z_sort_key_2d = effect.z_layer_2d;

        let image_handle = effect
            .particle_texture
            .as_ref()
            .map(|handle| handle.clone_weak())
            .unwrap_or_default();

        let property_layout = asset.property_layout();

        let property_data = if let Some(properties) = maybe_properties {
            if properties.is_changed() {
                Some(properties.serialize(&property_layout))
            } else {
                None
            }
        } else {
            None
        };

        let mut layout_flags = effect.layout_flags;
        if effect.particle_texture.is_some() {
            layout_flags |= LayoutFlags::PARTICLE_TEXTURE;
        }

        trace!(
            "Extracted instance of effect '{}' on entity {:?}: image_handle={:?} has_image={} layout_flags={:?}",
            asset.name,
            entity,
            image_handle,
            effect.particle_texture.is_some(),
            layout_flags,
        );

        extracted_effects.effects.insert(
            entity,
            ExtractedEffect {
                handle: effect.asset.clone_weak(),
                particle_layout: asset.particle_layout().clone(),
                property_layout,
                property_data,
                spawn_count,
                transform: transform.compute_matrix(),
                // TODO - more efficient/correct way than inverse()?
                inverse_transform: transform.compute_matrix().inverse(),
                force_field,
                layout_flags,
                image_handle,
                effect_shader,
                #[cfg(feature = "2d")]
                z_sort_key_2d,
            },
        );
    }
}

/// GPU representation of a single vertex of a particle mesh stored in a GPU
/// buffer.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
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
    /// Size of [`GpuRenderIndirect`] aligned to the contraint of
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`].
    ///
    /// [`WgpuLimits::min_storage_buffer_offset_alignment`]: bevy::render::settings::WgpuLimits::min_storage_buffer_offset_alignment
    render_indirect_aligned_size: NonZeroU32,
}

impl GpuLimits {
    pub fn from_device(render_device: &RenderDevice) -> Self {
        let storage_buffer_align = render_device.limits().min_storage_buffer_offset_alignment;

        let dispatch_indirect_aligned_size = NonZeroU32::new(next_multiple_of(
            GpuDispatchIndirect::min_size().get() as usize,
            storage_buffer_align as usize,
        ) as u32)
        .unwrap();

        let render_indirect_aligned_size = NonZeroU32::new(next_multiple_of(
            GpuRenderIndirect::min_size().get() as usize,
            storage_buffer_align as usize,
        ) as u32)
        .unwrap();

        trace!(
            "GpuLimits: storage_buffer_align={} gpu_dispatch_indirect_aligned_size={} gpu_render_indirect_aligned_size={}",
            storage_buffer_align,
            dispatch_indirect_aligned_size.get(),
            render_indirect_aligned_size.get()
        );

        Self {
            storage_buffer_align: NonZeroU32::new(storage_buffer_align).unwrap(),
            dispatch_indirect_aligned_size,
            render_indirect_aligned_size,
        }
    }

    pub fn storage_buffer_align(&self) -> NonZeroU32 {
        self.storage_buffer_align
    }

    pub fn dispatch_indirect_offset(&self, buffer_index: u32) -> u32 {
        self.dispatch_indirect_aligned_size.get() * buffer_index
    }

    pub fn render_indirect_offset(&self, buffer_index: u32) -> u64 {
        self.render_indirect_aligned_size.get() as u64 * buffer_index as u64
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
    /// Map from an entity with a [`ParticleEffect`] component attached to it,
    /// to the associated effect slice allocated in an [`EffectCache`].
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    entity_map: HashMap<Entity, EffectCacheId>,
    /// Global effect cache for all effects in use.
    effect_cache: EffectCache,
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
    /// Bind group #3 of the vfx_update shader, containing the indirect render
    /// buffer.
    update_render_indirect_bind_group: Option<BindGroup>,

    sim_params_uniforms: UniformBuffer<GpuSimParams>,
    spawner_buffer: AlignedBufferVec<GpuSpawnerParams>,
    dispatch_indirect_buffer: BufferTable<GpuDispatchIndirect>,
    render_dispatch_buffer: BufferTable<GpuRenderIndirect>,
    /// Unscaled vertices of the mesh of a single particle, generally a quad.
    /// The mesh is later scaled during rendering by the "particle size".
    // FIXME - This is a per-effect thing, unless we merge all meshes into a single buffer (makes
    // sense) but in that case we need a vertex slice too to know which mesh to draw per effect.
    vertices: BufferVec<GpuParticleVertex>,
    ///
    indirect_dispatch_pipeline: Option<ComputePipeline>,
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
            effect_cache: EffectCache::new(device),
            view_bind_group: None,
            sim_params_bind_group: None,
            spawner_bind_group: None,
            dr_indirect_bind_group: None,
            init_render_indirect_bind_group: None,
            update_render_indirect_bind_group: None,
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
            render_dispatch_buffer: BufferTable::new(
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
                NonZeroU64::new(item_align),
                Some("hanabi:buffer:render_dispatch".to_string()),
            ),
            vertices,
            indirect_dispatch_pipeline: None,
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
    ) {
        // Deallocate GPU data for destroyed effect instances. This will automatically
        // drop any group where there is no more effect slice.
        trace!(
            "Removing {} despawned effects",
            removed_effect_entities.len()
        );
        for entity in &removed_effect_entities {
            trace!("Removing ParticleEffect on entity {:?}", entity);
            if let Some(id) = self.entity_map.remove(entity) {
                trace!(
                    "=> ParticleEffect on entity {:?} had cache ID {:?}, removing...",
                    entity,
                    id
                );
                if let Some(buffer_index) = self.effect_cache.remove(id) {
                    // Clear bind groups associated with the removed buffer
                    trace!(
                        "=> GPU buffer #{} gone, destroying its bind groups...",
                        buffer_index
                    );
                    effect_bind_groups.particle_buffers.remove(&buffer_index);

                    // NOTE: by convention (see assert below) the cache ID is also the table ID, as
                    // those 3 data structures stay in sync.
                    let table_id = BufferTableId(buffer_index);
                    self.dispatch_indirect_buffer.remove(table_id);
                    self.render_dispatch_buffer.remove(table_id);
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
            let cache_id = self.effect_cache.insert(
                added_effect.handle,
                added_effect.capacity,
                &added_effect.particle_layout,
                &added_effect.property_layout,
                added_effect.layout_flags,
                // update_pipeline.pipeline.clone(),
                render_queue,
            );

            let entity = added_effect.entity;
            self.entity_map.insert(entity, cache_id);

            // Note: those effects are already in extracted_effects.effects because
            // they were gathered by the same query as previously existing
            // ones, during extraction.

            // FIXME - Kind of brittle since the EffectCache doesn't know about those
            let index = self.effect_cache.buffer_index(cache_id).unwrap();

            let table_id = self
                .dispatch_indirect_buffer
                .insert(GpuDispatchIndirect::default());
            // FIXME - Should have a single index and table bookeeping data structure, used
            // by multiple buffers
            assert_eq!(
                table_id.0 as usize, index,
                "Broken table invariant: buffer={} row={}",
                index, table_id.0
            );

            let table_id = self.render_dispatch_buffer.insert(GpuRenderIndirect {
                vertex_count: 6, // TODO - Flexible vertex count and mesh particles
                dead_count: added_effect.capacity,
                max_spawn: added_effect.capacity,
                ..default()
            });
            // FIXME - Should have a single index and table bookeeping data structure, used
            // by multiple buffers
            assert_eq!(
                table_id.0 as usize, index,
                "Broken table invariant: buffer={} row={}",
                index, table_id.0
            );
        }

        // Once all changes are applied, immediately schedule any GPU buffer
        // (re)allocation based on the new buffer size. The actual GPU buffer content
        // will be written later.
        if self
            .dispatch_indirect_buffer
            .allocate_gpu(render_device, render_queue)
        {
            // All those bind groups use the indirect buffer so need to be re-created.
            effect_bind_groups.particle_buffers.clear();
        }
        if self
            .render_dispatch_buffer
            .allocate_gpu(render_device, render_queue)
        {
            // Currently we always re-create each frame any bind group that
            // binds this buffer, so there's nothing to do here.
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
        /// The effect uses an image texture.
        const PARTICLE_TEXTURE = (1 << 0);
        /// The effect's particles have a size specified in screen space.
        const SCREEN_SPACE_SIZE = (1 << 1);
        /// The effect is simulated in local space.
        const LOCAL_SPACE_SIMULATION = (1 << 2);
        /// The effect uses alpha masking instead of alpha blending. Only used for 3D.
        const USE_ALPHA_MASK = (1 << 3);
        /// The effect is rendered with flipbook texture animation based on the [`Attribute::SPRITE_INDEX`] of each particle.
        const FLIPBOOK = (1 << 4);
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
    dispatch_indirect_pipeline: Res<DispatchIndirectPipeline>,
    init_pipeline: Res<ParticlesInitPipeline>,
    update_pipeline: Res<ParticlesUpdatePipeline>,
    mut specialized_init_pipelines: ResMut<SpecializedComputePipelines<ParticlesInitPipeline>>,
    mut specialized_update_pipelines: ResMut<SpecializedComputePipelines<ParticlesUpdatePipeline>>,
    // update_pipeline: Res<ParticlesUpdatePipeline>, // TODO move update_pipeline.pipeline to
    // EffectsMeta
    mut effects_meta: ResMut<EffectsMeta>,
    mut extracted_effects: ResMut<ExtractedEffects>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
) {
    trace!("prepare_effects");

    // Allocate spawner buffer if needed
    // if effects_meta.spawner_buffer.is_empty() {
    //    effects_meta.spawner_buffer.push(GpuSpawnerParams::default());
    //}

    // Write vertices (TODO - lazily once only)
    effects_meta
        .vertices
        .write_buffer(&render_device, &render_queue);

    effects_meta.indirect_dispatch_pipeline = Some(dispatch_indirect_pipeline.pipeline.clone());

    // Clear last frame's buffer resizes which may have occured during last frame,
    // during `Node::run()` while the `BufferTable` could not be mutated.
    effects_meta
        .dispatch_indirect_buffer
        .clear_previous_frame_resizes();
    effects_meta
        .render_dispatch_buffer
        .clear_previous_frame_resizes();

    // Allocate new effects, deallocate removed ones
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
    );

    // // sort first by z and then by handle. this ensures that, when possible,
    // batches span multiple z layers // batches won't span z-layers if there is
    // another batch between them extracted_effects.effects.sort_by(|a, b| {
    //     match FloatOrd(a.transform.w_axis[2]).cmp(&FloatOrd(b.transform.
    // w_axis[2])) {         Ordering::Equal => a.handle.cmp(&b.handle),
    //         other => other,
    //     }
    // });

    // Build batcher inputs from extracted effects
    let effects = std::mem::take(&mut extracted_effects.effects);
    let mut effect_entity_list = effects
        .into_iter()
        .map(|(entity, extracted_effect)| {
            let id = *effects_meta.entity_map.get(&entity).unwrap();
            let property_buffer = effects_meta.effect_cache.get_property_buffer(id).cloned(); // clone handle for lifetime
            let effect_slice = effects_meta.effect_cache.get_slice(id);

            BatchInput {
                handle: extracted_effect.handle,
                entity_index: entity.index(),
                effect_slice,
                property_layout: extracted_effect.property_layout.clone(),
                effect_shader: extracted_effect.effect_shader.clone(),
                layout_flags: extracted_effect.layout_flags,
                image_handle: extracted_effect.image_handle,
                force_field: extracted_effect.force_field,
                spawn_count: extracted_effect.spawn_count,
                transform: extracted_effect.transform.into(),
                inverse_transform: extracted_effect.inverse_transform.into(),
                property_buffer,
                property_data: extracted_effect.property_data,
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
    effect_entity_list.sort_by_key(|a| a.effect_slice.clone());

    // Loop on all extracted effects in order and try to batch them together to
    // reduce draw calls
    effects_meta.spawner_buffer.clear();
    let mut num_emitted = 0;
    {
        let mut batcher = Batcher::<BatchState, EffectBatch, BatchInput>::new(
            |mut input: BatchInput| -> (BatchState, EffectBatch) {
                trace!("Creating new batch from incompatible extracted effect");

                // Specialize the init pipeline based on the effect
                trace!(
                    "Specializing compute pipeline: init_shader={:?} particle_layout={:?}",
                    input.effect_shader.init,
                    input.effect_slice.particle_layout
                );
                let init_pipeline_id = specialized_init_pipelines.specialize(
                    &pipeline_cache,
                    &init_pipeline,
                    ParticleInitPipelineKey {
                        shader: input.effect_shader.init.clone(),
                        particle_layout_min_binding_size: input
                            .effect_slice
                            .particle_layout
                            .min_binding_size(),
                        property_layout_min_binding_size: if input.property_layout.is_empty() {
                            None
                        } else {
                            Some(input.property_layout.min_binding_size())
                        },
                    },
                );
                trace!("Init pipeline specialized: id={:?}", init_pipeline_id);

                // Specialize the update pipeline based on the effect
                trace!(
                    "Specializing update pipeline: update_shader={:?} particle_layout={:?} property_layout={:?}",
                    input.effect_shader.update,
                    input.effect_slice.particle_layout,
                    input.property_layout,
                );
                let update_pipeline_id = specialized_update_pipelines.specialize(
                    &pipeline_cache,
                    &update_pipeline,
                    ParticleUpdatePipelineKey {
                        shader: input.effect_shader.update.clone(),
                        particle_layout: input.effect_slice.particle_layout.clone(),
                        property_layout: input.property_layout.clone(),
                    },
                );
                trace!("Update pipeline specialized: id={:?}", update_pipeline_id);

                let init_shader = input.effect_shader.init.clone();
                trace!("init_shader = {:?}", init_shader);

                let update_shader = input.effect_shader.update.clone();
                trace!("update_shader = {:?}", update_shader);

                let render_shader = input.effect_shader.render.clone();
                trace!("render_shader = {:?}", render_shader);

                trace!("image_handle = {:?}", input.image_handle);

                let layout_flags = input.layout_flags;
                trace!("layout_flags = {:?}", layout_flags);

                trace!("particle_layout = {:?}", input.effect_slice.particle_layout);

                #[cfg(feature = "2d")]
                {
                    trace!("z_sort_key_2d = {:?}", input.z_sort_key_2d);
                }

                // This callback is raised when creating a new batch from a single item, so the
                // base index for spawners is the current buffer size. Per-effect spawner values
                // will be pushed in order into the array.
                let spawner_base = effects_meta.spawner_buffer.len() as u32;

                // FIXME - This overwrites the value of the previous effect if > 1 are batched
                // together!
                let spawn_count = input.spawn_count;
                trace!("spawn_count = {}", spawn_count);

                // Prepare the spawner block for the current slice
                // FIXME - This is once per EFFECT/SLICE, not once per BATCH, so indeed this is
                // spawner_BASE, and need an array of them in the compute shader!!!!!!!!!!!!!!
                let spawner_params = GpuSpawnerParams {
                    transform: input.transform,
                    inverse_transform: input.inverse_transform,
                    spawn: input.spawn_count as i32,
                    seed: random::<u32>(), /* FIXME - Probably bad to re-seed each time there's a
                                            * change */
                    count: 0,
                    effect_index: input.effect_slice.group_index,
                    force_field: input.force_field.map(Into::into),
                };
                trace!("spawner_params = {:?}", spawner_params);
                effects_meta.spawner_buffer.push(spawner_params);

                // Write properties for this effect if they were modified.
                // FIXME - This doesn't work with batching!
                if let Some(property_data) = &input.property_data {
                    if let Some(property_buffer) = input.property_buffer.as_ref() {
                        render_queue.write_buffer(property_buffer, 0, property_data);
                    }
                }

                let state = BatchState::from_input(&mut input);

                let batch = EffectBatch::from_input(
                    input,
                    spawner_base,
                    init_pipeline_id,
                    update_pipeline_id,
                );

                (state, batch)
            },
            |batch: EffectBatch| {
                // assert_ne!(asset, Handle::<EffectAsset>::default());
                assert!(batch.particle_layout.size() > 0);
                trace!(
                        "Emit batch: buffer #{} | spawner_base {} | spawn_count {} | slice {:?} | particle_layout {:?} | render_shader {:?} | entities {}",
                        batch.buffer_index,
                        batch.spawner_base,
                        batch.spawn_count,
                        batch.slice,
                        batch.particle_layout,
                        batch.render_shader,
                        batch.entities.len(),
                    );
                commands.spawn(batch);
                num_emitted += 1;
            },
        );

        batcher.batch(effect_entity_list);
    }

    // Write the entire spawner buffer for this frame, for all effects combined
    effects_meta
        .spawner_buffer
        .write_buffer(&render_device, &render_queue);

    // Allocate simulation uniform if needed
    // if effects_meta.sim_params_uniforms.is_empty() {
    effects_meta
        .sim_params_uniforms
        .set(GpuSimParams::default());
    //}

    // Update simulation parameters
    {
        let storage_align = effects_meta.gpu_limits.storage_buffer_align().get() as usize;

        let gpu_sim_params = effects_meta.sim_params_uniforms.get_mut();
        let sim_params = *sim_params;
        *gpu_sim_params = sim_params.into();

        gpu_sim_params.num_effects = num_emitted;

        gpu_sim_params.render_stride =
            next_multiple_of(GpuRenderIndirect::min_size().get() as usize, storage_align) as u32;
        gpu_sim_params.dispatch_stride = next_multiple_of(
            GpuDispatchIndirect::min_size().get() as usize,
            storage_align,
        ) as u32;

        trace!(
                "Simulation parameters: time={} delta_time={} num_effects={} render_stride={} dispatch_stride={}",
                gpu_sim_params.time,
                gpu_sim_params.delta_time,
                gpu_sim_params.num_effects,
                gpu_sim_params.render_stride,
                gpu_sim_params.dispatch_stride
            );
    }
    effects_meta
        .sim_params_uniforms
        .write_buffer(&render_device, &render_queue);
}

/// Per-buffer bind groups for the GPU particle buffer.
pub(crate) struct BufferBindGroups {
    /// Bind group for the init and update compute shaders.
    ///
    /// ```wgsl
    /// @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer;
    /// @binding(1) var<storage, read_write> indirect_buffer : IndirectBuffer;
    /// @binding(2) var<storage, read> properties : Properties; // optional
    /// ```
    simulate: BindGroup,
    /// Bind group for the render graphic shader.
    ///
    /// ```wgsl
    /// @binding(0) var<storage, read> particle_buffer : ParticlesBuffer;
    /// @binding(1) var<storage, read> indirect_buffer : IndirectBuffer;
    /// @binding(2) var<storage, read> dispatch_indirect : DispatchIndirect;
    /// ```
    render: BindGroup,
}

#[derive(Default, Resource)]
pub struct EffectBindGroups {
    /// Map from buffer index to its bind groups.
    particle_buffers: HashMap<u32, BufferBindGroups>,
    /// Map of bind groups for image assets used as particle textures.
    images: HashMap<AssetId<Image>, BindGroup>,
}

impl EffectBindGroups {
    pub fn particle_simulate(&self, buffer_index: u32) -> Option<&BindGroup> {
        self.particle_buffers
            .get(&buffer_index)
            .map(|bg| &bg.simulate)
    }

    pub fn particle_render(&self, buffer_index: u32) -> Option<&BindGroup> {
        self.particle_buffers
            .get(&buffer_index)
            .map(|bg| &bg.render)
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
    dispatch_indirect_pipeline: Res<'w, DispatchIndirectPipeline>,
    init_pipeline: Res<'w, ParticlesInitPipeline>,
    update_pipeline: Res<'w, ParticlesUpdatePipeline>,
    render_pipeline: Res<'w, ParticlesRenderPipeline>,
    #[system_param(ignore)]
    marker: PhantomData<&'s usize>,
}

fn emit_draw<T, F>(
    views: &mut Query<(&mut RenderPhase<T>, &VisibleEntities, &ExtractedView)>,
    effect_batches: &Query<(Entity, &mut EffectBatch)>,
    mut effect_bind_groups: Mut<EffectBindGroups>,
    gpu_images: &RenderAssets<Image>,
    render_device: RenderDevice,
    read_params: &QueueEffectsReadOnlyParams,
    mut specialized_render_pipelines: Mut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    pipeline_cache: &PipelineCache,
    msaa_samples: u32,
    make_phase_item: F,
    #[cfg(all(feature = "2d", feature = "3d"))] pipeline_mode: PipelineMode,
    use_alpha_mask: bool,
) where
    T: PhaseItem,
    F: Fn(CachedRenderPipelineId, Entity, &EffectBatch) -> T,
{
    for (mut render_phase, visible_entities, view) in views.iter_mut() {
        trace!("Process new view (use_alpha_mask={})", use_alpha_mask);

        let view_entities: Vec<u32> = visible_entities
            .entities
            .iter()
            .map(|e| e.index())
            .collect();

        // For each view, loop over all the effect batches to determine if the effect
        // needs to be rendered for that view, and enqueue a view-dependent
        // batch if so.
        for (entity, batch) in effect_batches.iter() {
            trace!(
                "Process batch entity={:?} buffer_index={} spawner_base={} slice={:?} layout_flags={:?}",
                entity,
                batch.buffer_index,
                batch.spawner_base,
                batch.slice,
                batch.layout_flags,
            );

            if use_alpha_mask != batch.layout_flags.contains(LayoutFlags::USE_ALPHA_MASK) {
                continue;
            }

            // Check if batch contains any entity visible in the current view. Otherwise we
            // can skip the entire batch. Note: This is O(n^2) but (unlike
            // the Sprite renderer this is inspired from) we don't expect more than
            // a handful of particle effect instances, so would rather not pay the memory
            // cost of a FixedBitSet for the sake of an arguable speed-up.
            // TODO - Profile to confirm.
            let has_visible_entity = view_entities
                .iter()
                .any(|index| batch.entities.contains(index));
            if !has_visible_entity {
                continue;
            }

            // FIXME - We draw the entire batch, but part of it may not be visible in this
            // view! We should re-batch for the current view specifically!

            // Ensure the particle texture is available as a GPU resource and create a bind
            // group for it
            let has_image = batch.layout_flags.contains(LayoutFlags::PARTICLE_TEXTURE);
            if has_image {
                if effect_bind_groups
                    .images
                    .get(&batch.image_handle.id())
                    .is_none()
                {
                    trace!(
                        "Batch buffer #{} slice={:?} has missing GPU image bind group, creating...",
                        batch.buffer_index,
                        batch.slice
                    );
                    // If texture doesn't have a bind group yet from another instance of the
                    // same effect, then try to create one now
                    if let Some(gpu_image) = gpu_images.get(&batch.image_handle) {
                        let bind_group = render_device.create_bind_group(
                            "hanabi:material_bind_group",
                            &read_params.render_pipeline.material_layout,
                            &[
                                BindGroupEntry {
                                    binding: 0,
                                    resource: BindingResource::TextureView(&gpu_image.texture_view),
                                },
                                BindGroupEntry {
                                    binding: 1,
                                    resource: BindingResource::Sampler(&gpu_image.sampler),
                                },
                            ],
                        );
                        effect_bind_groups
                            .images
                            .insert(batch.image_handle.id(), bind_group);
                    } else {
                        // Texture is not ready; skip for now...
                        trace!("GPU image not yet available; skipping batch for now.");
                        continue;
                    }
                } else {
                    trace!(
                        "Image {:?} already has bind group {:?}.",
                        batch.image_handle,
                        effect_bind_groups
                            .images
                            .get(&batch.image_handle.id())
                            .unwrap()
                    );
                }
            }

            let screen_space_size = batch.layout_flags.contains(LayoutFlags::SCREEN_SPACE_SIZE);
            let local_space_simulation = batch
                .layout_flags
                .contains(LayoutFlags::LOCAL_SPACE_SIMULATION);
            let use_alpha_mask = batch.layout_flags.contains(LayoutFlags::USE_ALPHA_MASK);
            let flipbook = batch.layout_flags.contains(LayoutFlags::FLIPBOOK);

            // Specialize the render pipeline based on the effect batch
            trace!(
                "Specializing render pipeline: render_shader={:?} has_image={:?} screen_space_size={:?} use_alpha_mask={:?} flipbook={:?} hdr={}",
                batch.render_shader,
                has_image,
                screen_space_size,
                use_alpha_mask,
                flipbook,
                view.hdr
            );
            let render_pipeline_id = specialized_render_pipelines.specialize(
                pipeline_cache,
                &read_params.render_pipeline,
                ParticleRenderPipelineKey {
                    shader: batch.render_shader.clone(),
                    particle_layout: batch.particle_layout.clone(),
                    has_image,
                    screen_space_size,
                    local_space_simulation,
                    use_alpha_mask,
                    flipbook,
                    #[cfg(all(feature = "2d", feature = "3d"))]
                    pipeline_mode,
                    msaa_samples,
                    hdr: view.hdr,
                },
            );
            trace!("Render pipeline specialized: id={:?}", render_pipeline_id);

            // Add a draw pass for the effect batch
            trace!("Add Transparent for batch on entity {:?}: buffer_index={} spawner_base={} slice={:?} handle={:?}", entity, batch.buffer_index, batch.spawner_base, batch.slice, batch.handle);
            render_phase.add(make_phase_item(render_pipeline_id, entity, batch));
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn queue_effects(
    #[cfg(feature = "2d")] mut views_2d: Query<(
        &mut RenderPhase<Transparent2d>,
        &VisibleEntities,
        &ExtractedView,
    )>,
    #[cfg(feature = "3d")] mut views_3d: Query<(
        &mut RenderPhase<Transparent3d>,
        &VisibleEntities,
        &ExtractedView,
    )>,
    #[cfg(feature = "3d")] mut views_alpha_mask: Query<(
        &mut RenderPhase<AlphaMask3d>,
        &VisibleEntities,
        &ExtractedView,
    )>,
    mut effects_meta: ResMut<EffectsMeta>,
    render_device: Res<RenderDevice>,
    mut specialized_render_pipelines: ResMut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    gpu_images: Res<RenderAssets<Image>>,
    effect_batches: Query<(Entity, &mut EffectBatch)>,
    events: Res<EffectAssetEvents>,
    read_params: QueueEffectsReadOnlyParams,
    msaa: Res<Msaa>,
) {
    trace!("queue_effects");

    // If an image has changed, the GpuImage has (probably) changed
    for event in &events.images {
        match event {
            AssetEvent::Added { .. } => None,
            AssetEvent::LoadedWithDependencies { .. } => None,
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

    if effects_meta.spawner_buffer.buffer().is_none() {
        // No spawners are active
        return;
    }

    // Create the bind group for the global simulation parameters
    effects_meta.sim_params_bind_group = Some(render_device.create_bind_group(
        "hanabi:bind_group_sim_params",
        &read_params.update_pipeline.sim_params_layout, /* FIXME - Shared with vfx_update, is
                                                         * that OK? */
        &[BindGroupEntry {
            binding: 0,
            resource: effects_meta.sim_params_uniforms.binding().unwrap(),
        }],
    ));

    // Create the bind group for the spawner parameters
    // FIXME - This is shared by init and update; should move
    // "update_pipeline.spawner_buffer_layout" out of "update_pipeline"
    trace!(
        "Spawner buffer bind group: size={} aligned_size={}",
        GpuSpawnerParams::min_size().get(),
        effects_meta.spawner_buffer.aligned_size()
    );
    assert!(
        effects_meta.spawner_buffer.aligned_size() >= GpuSpawnerParams::min_size().get() as usize
    );
    effects_meta.spawner_bind_group = Some(render_device.create_bind_group(
        "hanabi:bind_group_spawner_buffer",
        &read_params.update_pipeline.spawner_buffer_layout, // FIXME - Shared with init,is that OK?
        &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: effects_meta.spawner_buffer.buffer().unwrap(),
                offset: 0,
                size: Some(
                    NonZeroU64::new(effects_meta.spawner_buffer.aligned_size() as u64).unwrap(),
                ),
            }),
        }],
    ));

    // Create the bind group for the indirect dispatch of all effects
    effects_meta.dr_indirect_bind_group = Some(
        render_device.create_bind_group(
            "hanabi:bind_group_vfx_indirect_dr_indirect",
            &read_params
                .dispatch_indirect_pipeline
                .dispatch_indirect_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: effects_meta.render_dispatch_buffer.buffer().unwrap(),
                        offset: 0,
                        size: None, //NonZeroU64::new(256), // Some(GpuRenderIndirect::min_size()),
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: effects_meta.dispatch_indirect_buffer.buffer().unwrap(),
                        offset: 0,
                        size: None, //NonZeroU64::new(256), // Some(GpuDispatchIndirect::min_size()),
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: effects_meta.spawner_buffer.buffer().unwrap(),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        ),
    );

    // Create the bind group for the indirect render buffer use in the init shader
    effects_meta.init_render_indirect_bind_group = Some(render_device.create_bind_group(
        "hanabi:bind_group_init_render_dispatch",
        &read_params.init_pipeline.render_indirect_layout,
        &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: effects_meta.render_dispatch_buffer.buffer().unwrap(),
                offset: 0,
                size: Some(GpuRenderIndirect::min_size()),
            }),
        }],
    ));

    // Create the bind group for the indirect render buffer use in the update shader
    effects_meta.update_render_indirect_bind_group = Some(render_device.create_bind_group(
        "hanabi:bind_group_update_render_dispatch",
        &read_params.update_pipeline.render_indirect_layout,
        &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: effects_meta.render_dispatch_buffer.buffer().unwrap(),
                offset: 0,
                size: Some(GpuRenderIndirect::min_size()),
            }),
        }],
    ));

    // Make a copy of the buffer ID before borrowing effects_meta mutably in the
    // loop below
    let indirect_buffer = effects_meta
        .dispatch_indirect_buffer
        .buffer()
        .cloned()
        .unwrap();
    let spawner_buffer = effects_meta.spawner_buffer.buffer().cloned().unwrap();

    // Create the per-effect bind groups
    trace!("Create per-effect bind groups...");
    for (buffer_index, buffer) in effects_meta.effect_cache.buffers().iter().enumerate() {
        let Some(buffer) = buffer else {
            trace!(
                "Effect buffer index #{} has no allocated EffectBuffer, skipped.",
                buffer_index
            );
            continue;
        };

        // Ensure all effect groups have a bind group for the entire buffer of the
        // group, since the update phase runs on an entire group/buffer at once,
        // with all the effect instances in it batched together.
        trace!("effect particle buffer_index=#{}", buffer_index);
        effect_bind_groups
            .particle_buffers
            .entry(buffer_index as u32)
            .or_insert_with(|| {
                trace!(
                    "Create new particle bind groups for buffer_index={} | particle_layout {:?} | property_layout {:?}",
                    buffer_index,
                    buffer.particle_layout(),
                    buffer.property_layout(),
                );

                // Bind group shared by the init and update compute shaders to simulate particles.
                let layout = buffer.particle_layout_bind_group_simulate();
                let label = format!("hanabi:bind_group_simulate_vfx{}_particles", buffer_index);
                let simulate = if let Some(property_binding) = buffer.properties_max_binding() {
                    let entries = [
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
                            resource: property_binding,
                        },
                    ];
                    trace!("=> create update bind group '{}' with 3 entries", label);
                    render_device.create_bind_group(Some(&label[..]), layout, &entries)
                } else {
                    let entries = [
                        BindGroupEntry {
                            binding: 0,
                            resource: buffer.max_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.indirect_max_binding(),
                        },
                    ];
                    trace!("=> create update bind group '{}' with 2 entries", label);
                    render_device.create_bind_group(Some(&label[..]), layout, &entries)
                };

                // 
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
                            size: Some(GpuDispatchIndirect::min_size()),
                        }),
                    },
                ];
                if buffer.layout_flags().contains(LayoutFlags::LOCAL_SPACE_SIMULATION) {
                    entries.push(BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &spawner_buffer,
                            offset: 0,
                            size: Some(GpuSpawnerParams::min_size()),
                        }),
                    });
                }
                trace!("Creating render bind group with {} entries", entries.len());
                let render = render_device.create_bind_group(
                    &format!("hanabi:bind_group_render_vfx{buffer_index}_particles")[..],
                     buffer.particle_layout_bind_group_with_dispatch(),
                     &entries,
                );

                BufferBindGroups {
                    simulate,
                    render,
                }
            });
    }

    // Loop over all 2D cameras/views that need to render effects
    #[cfg(feature = "2d")]
    {
        let draw_effects_function_2d = read_params
            .draw_functions_2d
            .read()
            .get_id::<DrawEffects>()
            .unwrap();

        // Effects with full alpha blending
        if !views_2d.is_empty() {
            trace!("Emit effect draw calls for alpha blended 2D views...");
            emit_draw(
                &mut views_2d,
                &effect_batches,
                effect_bind_groups.reborrow(),
                &gpu_images,
                render_device.clone(),
                &read_params,
                specialized_render_pipelines.reborrow(),
                &pipeline_cache,
                msaa.samples(),
                |id, entity, batch| Transparent2d {
                    draw_function: draw_effects_function_2d,
                    pipeline: id,
                    entity,
                    sort_key: batch.z_sort_key_2d,
                    batch_range: 0..1,
                    dynamic_offset: None,
                },
                #[cfg(feature = "3d")]
                PipelineMode::Camera2d,
                false,
            );
        }
    }

    // Loop over all 3D cameras/views that need to render effects
    #[cfg(feature = "3d")]
    {
        // Effects with full alpha blending
        if !views_3d.is_empty() {
            trace!("Emit effect draw calls for alpha blended 3D views...");

            let draw_effects_function_3d = read_params
                .draw_functions_3d
                .read()
                .get_id::<DrawEffects>()
                .unwrap();

            emit_draw(
                &mut views_3d,
                &effect_batches,
                effect_bind_groups.reborrow(),
                &gpu_images,
                render_device.clone(),
                &read_params,
                specialized_render_pipelines.reborrow(),
                &pipeline_cache,
                msaa.samples(),
                |id, entity, _batch| Transparent3d {
                    draw_function: draw_effects_function_3d,
                    pipeline: id,
                    entity,
                    distance: 0.0, // TODO
                    batch_range: 0..1,
                    dynamic_offset: None,
                },
                #[cfg(feature = "2d")]
                PipelineMode::Camera3d,
                false,
            );
        }

        // Effects with alpha mask
        if !views_alpha_mask.is_empty() {
            trace!("Emit effect draw calls for alpha masked 3D views...");

            let draw_effects_function_alpha_mask = read_params
                .draw_functions_alpha_mask
                .read()
                .get_id::<DrawEffects>()
                .unwrap();

            emit_draw(
                &mut views_alpha_mask,
                &effect_batches,
                effect_bind_groups.reborrow(),
                &gpu_images,
                render_device.clone(),
                &read_params,
                specialized_render_pipelines.reborrow(),
                &pipeline_cache,
                msaa.samples(),
                |id, entity, _batch| AlphaMask3d {
                    draw_function: draw_effects_function_alpha_mask,
                    pipeline: id,
                    entity,
                    distance: 0.0, // TODO
                    batch_range: 0..1,
                    dynamic_offset: None,
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
/// This system runs in the [`Prepare`] render set, after Bevy has updated the
/// [`ViewUniforms`], which need to be referenced to get access to the current
/// camera view.
pub(crate) fn prepare_resources(
    mut effects_meta: ResMut<EffectsMeta>,
    render_device: Res<RenderDevice>,
    view_uniforms: Res<ViewUniforms>,
    read_params: QueueEffectsReadOnlyParams,
) {
    // Get the binding for the ViewUniform, the uniform data structure containing
    // the Camera data for the current view.
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };

    // Create the bind group for the camera/view parameters
    effects_meta.view_bind_group = Some(render_device.create_bind_group(
        "hanabi:bind_group_camera_view",
        &read_params.render_pipeline.view_layout,
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

type DrawEffectsSystemState = SystemState<(
    SRes<EffectsMeta>,
    SRes<EffectBindGroups>,
    SRes<PipelineCache>,
    SQuery<Read<ViewUniformOffset>>,
    SQuery<Read<EffectBatch>>,
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

/// Draw all particles of all effects in view, in 2D or 3D.
fn draw<'w>(
    world: &'w World,
    pass: &mut TrackedRenderPass<'w>,
    view: Entity,
    entity: Entity,
    pipeline_id: CachedRenderPipelineId,
    params: &mut DrawEffectsSystemState,
) {
    let (effects_meta, effect_bind_groups, pipeline_cache, views, effects) = params.get(world);
    let view_uniform = views.get(view).unwrap();
    let effects_meta = effects_meta.into_inner();
    let effect_bind_groups = effect_bind_groups.into_inner();
    let effect_batch = effects.get(entity).unwrap();

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
    let dispatch_indirect_offset = gpu_limits.dispatch_indirect_offset(effect_batch.buffer_index);
    trace!(
        "set_bind_group(1): dispatch_indirect_offset={}",
        dispatch_indirect_offset
    );
    let spawner_base = effect_batch.spawner_base;
    let spawner_buffer_aligned = effects_meta.spawner_buffer.aligned_size();
    assert!(spawner_buffer_aligned >= GpuSpawnerParams::min_size().get() as usize);
    let spawner_offset = spawner_base * spawner_buffer_aligned as u32;
    let dyn_uniform_indices: [u32; 2] = [dispatch_indirect_offset, spawner_offset];
    let dyn_uniform_indices = if effect_batch
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
            .particle_render(effect_batch.buffer_index)
            .unwrap(),
        dyn_uniform_indices,
    );

    // Particle texture
    if effect_batch
        .layout_flags
        .contains(LayoutFlags::PARTICLE_TEXTURE)
    {
        if let Some(bind_group) = effect_bind_groups
            .images
            .get(&effect_batch.image_handle.id())
        {
            pass.set_bind_group(2, bind_group, &[]);
        } else {
            // Texture not ready; skip this drawing for now
            trace!(
                "Particle texture bind group not available for batch buf={} slice={:?}. Skipping draw call.",
                effect_batch.buffer_index,
                effect_batch.slice
            );
            return; // continue;
        }
    }

    let render_indirect_buffer = effects_meta.render_dispatch_buffer.buffer().unwrap();

    let render_indirect_offset = gpu_limits.render_indirect_offset(effect_batch.buffer_index);
    trace!(
        "Draw {} particles with {} vertices per particle for batch from buffer #{} (render_indirect_offset={}).",
        effect_batch.slice.len(),
        effects_meta.vertices.len(),
        effect_batch.buffer_index,
        render_indirect_offset
    );
    pass.draw_indirect(render_indirect_buffer, render_indirect_offset);
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
            item.entity,
            item.pipeline,
            &mut self.params,
        );
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
        graph.run_sub_graph(crate::plugin::simulate_graph::NAME, vec![], None)?;
        Ok(())
    }
}

/// Render node to run the simulation of all effects once per frame.
///
/// Runs inside the simulation sub-graph, looping over all extracted effect
/// batches to simulate them.
pub(crate) struct VfxSimulateNode {
    /// Query to retrieve the batches of effects to simulate and render.
    effect_query: QueryState<&'static EffectBatch>,
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

        // Get the Entity containing the ViewEffectsEntity component used as container
        // for the input data for this node.
        // let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        let pipeline_cache = world.resource::<PipelineCache>();

        let effects_meta = world.resource::<EffectsMeta>();
        let effect_bind_groups = world.resource::<EffectBindGroups>();
        // let render_queue = world.resource::<RenderQueue>();

        // Make sure to schedule any buffer copy from changed effects before accessing
        // them
        effects_meta
            .dispatch_indirect_buffer
            .write_buffer(render_context.command_encoder());
        effects_meta
            .render_dispatch_buffer
            .write_buffer(render_context.command_encoder());

        // Compute init pass
        {
            let mut compute_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hanabi:init"),
                    });

            {
                trace!("loop over effect batches...");

                // Dispatch init compute jobs
                for batch in self.effect_query.iter_manual(world) {
                    if let Some(init_pipeline) =
                        pipeline_cache.get_compute_pipeline(batch.init_pipeline_id)
                    {
                        // Do not dispatch any init work if there's nothing to spawn this frame
                        let spawn_count = batch.spawn_count;
                        if spawn_count == 0 {
                            continue;
                        }

                        const WORKGROUP_SIZE: u32 = 64;
                        let workgroup_count = (spawn_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

                        // for (effect_entity, effect_slice) in effects_meta.entity_map.iter() {
                        // Retrieve the ExtractedEffect from the entity
                        // trace!("effect_entity={:?} effect_slice={:?}", effect_entity,
                        // effect_slice); let effect =
                        // self.effect_query.get_manual(world, *effect_entity).unwrap();

                        // Get the slice to init
                        // let effect_slice = effects_meta.get(&effect_entity);
                        // let effect_group =
                        //     &effects_meta.effect_cache.buffers()[batch.buffer_index as usize];
                        let Some(particles_bind_group) =
                            effect_bind_groups.particle_simulate(batch.buffer_index)
                        else {
                            continue;
                        };

                        let item_size = batch.particle_layout.min_binding_size();
                        let item_count = batch.slice.end - batch.slice.start;

                        let spawner_base = batch.spawner_base;
                        let buffer_offset = batch.slice.start;

                        let spawner_buffer_aligned = effects_meta.spawner_buffer.aligned_size();
                        assert!(
                            spawner_buffer_aligned >= GpuSpawnerParams::min_size().get() as usize
                        );
                        let spawner_offset = spawner_base * spawner_buffer_aligned as u32;

                        let render_indirect_offset = batch.buffer_index
                            * effects_meta.render_dispatch_buffer.aligned_size() as u32;

                        trace!(
                            "record commands for init pipeline of effect {:?} ({} items / {}B/item) (spawn {} = {} workgroups) spawner_base={} spawner_offset={} buffer_offset={} render_indirect_offset={}...",
                            batch.handle,
                            item_count,
                            item_size,
                            spawn_count,
                            workgroup_count,
                            spawner_base,
                            spawner_offset,
                            buffer_offset,
                            render_indirect_offset,
                        );

                        // Setup compute pass
                        // compute_pass.set_pipeline(&effect_group.init_pipeline);
                        compute_pass.set_pipeline(init_pipeline);
                        compute_pass.set_bind_group(
                            0,
                            effects_meta.sim_params_bind_group.as_ref().unwrap(),
                            &[],
                        );
                        compute_pass.set_bind_group(
                            1,
                            particles_bind_group,
                            &[buffer_offset, buffer_offset], /* FIXME: probably in bytes, so
                                                              * probably wrong! */
                        );
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
                            &[render_indirect_offset],
                        );
                        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                        trace!("init compute dispatched");
                    }
                }
            }
        }

        // Compute indirect dispatch pass
        if effects_meta.spawner_buffer.buffer().is_some() {
            // Only if there's an effect
            let mut compute_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hanabi:indirect_dispatch"),
                    });

            // Dispatch indirect dispatch compute job
            if let Some(indirect_dispatch_pipeline) = &effects_meta.indirect_dispatch_pipeline {
                trace!("record commands for indirect dispatch pipeline...");

                let num_batches = self.effect_query.iter_manual(world).count() as u32;

                const WORKGROUP_SIZE: u32 = 64;
                let workgroup_count = (num_batches + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

                // Setup compute pass
                compute_pass.set_pipeline(indirect_dispatch_pipeline);
                compute_pass.set_bind_group(
                    0,
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
                    num_batches,
                    workgroup_count
                );
            }
        }

        // Compute update pass
        {
            let mut compute_pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hanabi:update"),
                    });

            // Dispatch update compute jobs
            for batch in self.effect_query.iter_manual(world) {
                if let Some(update_pipeline) =
                    pipeline_cache.get_compute_pipeline(batch.update_pipeline_id)
                {
                    // for (effect_entity, effect_slice) in effects_meta.entity_map.iter() {
                    // Retrieve the ExtractedEffect from the entity
                    // trace!("effect_entity={:?} effect_slice={:?}", effect_entity,
                    // effect_slice); let effect =
                    // self.effect_query.get_manual(world, *effect_entity).unwrap();

                    // Get the slice to update
                    // let effect_slice = effects_meta.get(&effect_entity);
                    // let effect_group =
                    //     &effects_meta.effect_cache.buffers()[batch.buffer_index as usize];
                    let Some(particles_bind_group) =
                        effect_bind_groups.particle_simulate(batch.buffer_index)
                    else {
                        continue;
                    };

                    let item_size = batch.particle_layout.size();
                    let item_count = batch.slice.end - batch.slice.start;

                    let spawner_base = batch.spawner_base;
                    let buffer_offset = batch.slice.start;

                    let spawner_buffer_aligned = effects_meta.spawner_buffer.aligned_size();
                    assert!(spawner_buffer_aligned >= GpuSpawnerParams::min_size().get() as usize);

                    let dispatch_indirect_offset = batch.buffer_index as u64
                        * effects_meta.dispatch_indirect_buffer.aligned_size() as u64;

                    let render_indirect_offset = batch.buffer_index
                        * effects_meta.render_dispatch_buffer.aligned_size() as u32;

                    trace!(
                        "record commands for update pipeline of effect {:?} ({} items / {}B/item) spawner_base={} buffer_offset={} dispatch_indirect_offset={} render_indirect_offset={}...",
                        batch.handle,
                        item_count,
                        item_size,
                        spawner_base,
                        buffer_offset,
                        dispatch_indirect_offset,
                        render_indirect_offset
                    );

                    // Setup compute pass
                    // compute_pass.set_pipeline(&effect_group.update_pipeline);
                    compute_pass.set_pipeline(update_pipeline);
                    compute_pass.set_bind_group(
                        0,
                        effects_meta.sim_params_bind_group.as_ref().unwrap(),
                        &[],
                    );
                    compute_pass.set_bind_group(
                        1,
                        particles_bind_group,
                        &[buffer_offset, buffer_offset], /* FIXME: probably in bytes, so
                                                          * probably wrong! */
                    );
                    compute_pass.set_bind_group(
                        2,
                        effects_meta.spawner_bind_group.as_ref().unwrap(),
                        &[spawner_base * spawner_buffer_aligned as u32],
                    );
                    compute_pass.set_bind_group(
                        3,
                        effects_meta
                            .update_render_indirect_bind_group
                            .as_ref()
                            .unwrap(),
                        &[render_indirect_offset],
                    );

                    if let Some(buffer) = effects_meta.dispatch_indirect_buffer.buffer() {
                        trace!(
                            "dispatch_workgroups_indirect: buffer={:?} offset={}",
                            buffer,
                            dispatch_indirect_offset
                        );
                        compute_pass.dispatch_workgroups_indirect(buffer, dispatch_indirect_offset);
                        // TODO - offset
                    }

                    trace!("update compute dispatched");
                }
            }
        }

        Ok(())
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
        assert!(limits.render_indirect_offset(256) >= 256 * GpuRenderIndirect::min_size().get());
        assert!(
            limits.dispatch_indirect_offset(256) as u64
                >= 256 * GpuDispatchIndirect::min_size().get()
        );
    }
}
