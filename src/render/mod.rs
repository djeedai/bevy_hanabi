#![allow(unused_imports)] // TEMP

use bevy::{
    asset::{AssetEvent, Assets, Handle, HandleId, HandleUntyped},
    core::{cast_slice, Pod, Zeroable},
    ecs::{
        prelude::*,
        system::{lifetimeless::*, SystemState},
    },
    log::trace,
    math::{Mat4, Quat, Vec2, Vec3, Vec4Swizzles},
    reflect::TypeUuid,
    render::{
        color::Color,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_phase::{Draw, DrawFunctions, RenderPhase, TrackedRenderPass},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::{BevyDefault, Image},
        view::{
            ComputedVisibility, ExtractedView, Msaa, ViewUniform, ViewUniformOffset, ViewUniforms,
        },
        Extract,
    },
    time::Time,
    transform::components::GlobalTransform,
    utils::{FloatOrd, HashMap, HashSet},
};
use bitflags::bitflags;
use bytemuck::cast_slice_mut;
use rand::random;
use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::{borrow::Cow, cmp::Ordering, num::NonZeroU64, ops::Range};

#[cfg(feature = "2d")]
use bevy::core_pipeline::core_2d::Transparent2d;
#[cfg(feature = "3d")]
use bevy::core_pipeline::core_3d::Transparent3d;

use crate::{
    asset::EffectAsset,
    modifier::update::ForceFieldSource,
    spawn::{new_rng, Random},
    Gradient, ParticleEffect, RemovedEffectsEvent, ToWgslString,
};

mod aligned_buffer_vec;
mod effect_cache;
mod pipeline_template;

use aligned_buffer_vec::AlignedBufferVec;
pub(crate) use effect_cache::{EffectCache, EffectCacheId};

pub use effect_cache::{EffectBuffer, EffectSlice};
pub use pipeline_template::PipelineRegistry;

pub(crate) const PARTICLES_UPDATE_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 2763343953151597126);

pub(crate) const PARTICLES_RENDER_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 2763343953151597145);

/// Labels for the Hanabi systems.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum EffectSystems {
    /// Tick all effect instances to generate spawner counts, and configure
    /// shaders based on modifiers. This system runs during the
    /// [`CoreStage::PostUpdate`] stage.
    ///
    /// [`CoreStage::PostUpdate`]: bevy::app::CoreStage::PostUpdate
    TickSpawners,
    /// Gather all removed [`ParticleEffect`] components during the
    /// [`CoreStage::PostUpdate`] stage, to be able to clean-up GPU
    /// resources.
    ///
    /// [`CoreStage::PostUpdate`]: bevy::app::CoreStage::PostUpdate
    GatherRemovedEffects,
    /// Extract the effects to render this frame.
    ExtractEffects,
    /// Extract the effect events to process this frame.
    ExtractEffectEvents,
    /// Prepare GPU data for the extracted effects.
    PrepareEffects,
    /// Queue the GPU commands for the extracted effects.
    QueueEffects,
}

/// Reimplementing of bevy::sprite::Rect to avoid the dependency.
/// See https://github.com/bevyengine/bevy/issues/5575
#[derive(Debug, Default, Clone, Copy, PartialEq, Resource)]
pub(crate) struct MinMaxRect {
    pub min: Vec2,
    pub max: Vec2,
}

/// Simulation parameters.
#[derive(Debug, Default, Clone, Copy, Resource)]
pub(crate) struct SimParams {
    /// Current simulation time.
    time: f64,
    /// Frame timestep.
    dt: f32,
}

/// GPU representation of [`SimParams`].
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, ShaderType)]
struct SimParamsUniform {
    dt: f32,
    time: f32,
}

impl Default for SimParamsUniform {
    fn default() -> SimParamsUniform {
        SimParamsUniform {
            dt: 0.04,
            time: 0.0,
        }
    }
}

impl From<SimParams> for SimParamsUniform {
    fn from(src: SimParams) -> Self {
        SimParamsUniform {
            dt: src.dt,
            time: src.time as f32,
        }
    }
}

/// GPU representation of a [`ForceFieldSource`].
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Resource, Zeroable, ShaderType)]
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
        GpuForceFieldSource {
            position_or_direction: source.position,
            max_radius: source.max_radius,
            min_radius: source.min_radius,
            mass: source.mass,
            force_exponent: source.force_exponent,
            conform_to_sphere: if source.conform_to_sphere { 1.0 } else { 0.0 },
        }
    }
}

/// GPU representation of spawner parameters.
///
/// This structure contains the fixed-size part of the parameters. Inside the
/// GPU buffer, it is followed by an array of [`GpuForceFieldSource`], which
/// together form the spawner parameter buffer.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Resource, Zeroable, ShaderType)]
struct GpuSpawnerParams {
    /// Transform of the effect, as a Mat4 without the last row (which is always
    /// (0,0,0,1) for an affine transform), stored transposed as a mat3x4 to
    /// avoid padding in WGSL. This is either added to emitted particles at
    /// spawn time, if the effect simulated in world space, or to all
    /// simulated particles if the effect is simulated in local space.
    transform: [f32; 12],
    /// Global acceleration applied to all particles each frame.
    /// TODO - This is NOT a spawner/emitter thing, but is a per-effect one.
    /// Rename GpuSpawnerParams?
    accel: Vec3,
    /// Number of particles to spawn this frame.
    spawn: i32,
    /// Force field components. One PullingForceFieldParam takes up 32 bytes.
    force_field: [GpuForceFieldSource; ForceFieldSource::MAX_SOURCES],
    /// Spawn seed, for randomized modifiers.
    seed: u32,
    /// Current number of used particles.
    count: i32,
}

#[derive(Resource)]
pub(crate) struct ParticlesUpdatePipeline {
    sim_params_layout: BindGroupLayout,
    particles_buffer_layout: BindGroupLayout,
    spawner_buffer_layout: BindGroupLayout,
    pipeline_layout: PipelineLayout,
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

        trace!(
            "SimParamsUniform: min_size={}",
            SimParamsUniform::min_size()
        );
        let sim_params_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(SimParamsUniform::min_size()),
                    },
                    count: None,
                }],
                label: Some("hanabi:update_sim_params_layout"),
            });

        trace!("Particle: min_size={}", Particle::min_size());
        let particles_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: true,
                            min_binding_size: Some(Particle::min_size()),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: true,
                            min_binding_size: BufferSize::new(std::mem::size_of::<u32>() as u64),
                        },
                        count: None,
                    },
                ],
                label: Some("hanabi:update_particles_buffer_layout"),
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

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("hanabi:update_pipeline_layout"),
            bind_group_layouts: &[
                &sim_params_layout,
                &particles_buffer_layout,
                &spawner_buffer_layout,
            ],
            push_constant_ranges: &[],
        });

        ParticlesUpdatePipeline {
            sim_params_layout,
            particles_buffer_layout,
            spawner_buffer_layout,
            pipeline_layout,
        }
    }
}

#[derive(Default, Clone, Hash, PartialEq, Eq, Resource)]
pub(crate) struct ParticleUpdatePipelineKey {
    /// Compute shader, with snippets applied, but not preprocessed yet.
    shader: Handle<Shader>,
}

impl SpecializedComputePipeline for ParticlesUpdatePipeline {
    type Key = ParticleUpdatePipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("hanabi:update_compute_pipeline".into()),
            layout: Some(vec![
                self.sim_params_layout.clone(),
                self.particles_buffer_layout.clone(),
                self.spawner_buffer_layout.clone(),
            ]),
            shader: key.shader,
            shader_defs: vec![],
            entry_point: "main".into(),
        }
    }
}
#[derive(Resource)]
pub(crate) struct ParticlesRenderPipeline {
    view_layout: BindGroupLayout,
    particles_buffer_layout: BindGroupLayout,
    material_layout: BindGroupLayout,
}

impl FromWorld for ParticlesRenderPipeline {
    fn from_world(world: &mut World) -> Self {
        let world = world.cell();
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(ViewUniform::min_size()),
                },
                count: None,
            }],
            label: Some("hanabi:view_layout_render"),
        });

        let particles_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(Particle::min_size()),
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
                ],
                label: Some("hanabi:buffer_layout_render"),
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

        ParticlesRenderPipeline {
            view_layout,
            particles_buffer_layout,
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

#[derive(Clone, Hash, PartialEq, Eq)]
pub(crate) struct ParticleRenderPipelineKey {
    /// Render shader, with snippets applied, but not preprocessed yet.
    shader: Handle<Shader>,
    /// Key: PARTICLE_TEXTURE
    /// Define a texture sampled to modulate the particle color.
    /// This key requires the presence of UV coordinates on the particle
    /// vertices.
    particle_texture: Option<Handle<Image>>,
    /// For dual-mode configurations only, the actual mode of the current render
    /// pipeline. Otherwise the mode is implicitly determined by the active
    /// feature.
    #[cfg(all(feature = "2d", feature = "3d"))]
    pipeline_mode: PipelineMode,
    /// MSAA sample count.
    msaa_samples: u32,
}

impl Default for ParticleRenderPipelineKey {
    fn default() -> Self {
        ParticleRenderPipelineKey {
            shader: PARTICLES_RENDER_SHADER_HANDLE.typed::<Shader>(),
            particle_texture: None,
            #[cfg(all(feature = "2d", feature = "3d"))]
            pipeline_mode: PipelineMode::Camera3d,
            msaa_samples: Msaa::default().samples,
        }
    }
}

impl SpecializedRenderPipeline for ParticlesRenderPipeline {
    type Key = ParticleRenderPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
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

        let mut layout = vec![
            self.view_layout.clone(),
            self.particles_buffer_layout.clone(),
        ];
        let mut shader_defs = vec![];

        // Key: PARTICLE_TEXTURE
        if key.particle_texture.is_some() {
            layout.push(self.material_layout.clone());
            shader_defs.push("PARTICLE_TEXTURE".to_string());
            // //  @location(1) vertex_uv: vec2<f32>
            // vertex_buffer_layout.attributes.push(VertexAttribute {
            //     format: VertexFormat::Float32x2,
            //     offset: 12,
            //     shader_location: 1,
            // });
            // vertex_buffer_layout.array_stride += 8;
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
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            layout: Some(layout),
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
            label: Some("hanabi:render_pipeline".into()),
        }
    }
}

/// A single effect instance extracted from a [`ParticleEffect`] as a
/// [`RenderWorld`] item.
#[derive(Component)]
pub(crate) struct ExtractedEffect {
    /// Handle to the effect asset this instance is based on.
    /// The handle is weak to prevent refcount cycles and gracefully handle
    /// assets unloaded or destroyed after a draw call has been submitted.
    pub handle: Handle<EffectAsset>,
    /// Number of particles to spawn this frame for the effect.
    /// Obtained from calling [`Spawner::tick()`] on the source effect instance.
    pub spawn_count: u32,
    /// Global transform of the effect origin, extracted from the
    /// [`GlobalTransform`].
    pub transform: Mat4,
    /// Constant acceleration applied to all particles.
    pub accel: Vec3,
    /// Force field applied to all particles in the "update" phase.
    force_field: [ForceFieldSource; ForceFieldSource::MAX_SOURCES],
    /// Particles tint to modulate with the texture image.
    pub color: Color,
    pub rect: MinMaxRect,
    // Texture to use for the sprites of the particles of this effect.
    //pub image: Handle<Image>,
    pub has_image: bool, // TODO -> use flags
    /// Texture to modulate the particle color.
    pub image_handle_id: HandleId,
    /// Compute shader.
    pub compute_shader: Handle<Shader>,
    /// Render shader.
    pub render_shader: Handle<Shader>,
    /// Update position code.
    pub position_code: String,
    /// Update force field code.
    pub force_field_code: String,
    /// Update lifetime code.
    pub lifetime_code: String,
    /// For 2D rendering, the Z coordinate used as the sort key. Ignored for 3D
    /// rendering.
    pub z_sort_key_2d: FloatOrd,
}

/// Extracted data for newly-added [`ParticleEffect`] component requiring a new
/// GPU allocation.
pub(crate) struct AddedEffect {
    /// Entity with a newly-added [`ParticleEffect`] component.
    pub entity: Entity,
    /// Capacity of the effect (and therefore, the particle buffer), in number
    /// of particles.
    pub capacity: u32,
    /// Size in bytes of each particle.
    pub item_size: u32,
    /// Handle of the effect asset.
    pub handle: Handle<EffectAsset>,
}

/// Collection of all extracted effects for this frame, inserted into the
/// [`RenderWorld`] as a render resource.
#[derive(Default, Resource)]
pub(crate) struct ExtractedEffects {
    /// Map of extracted effects from the entity the source [`ParticleEffect`]
    /// is on.
    pub effects: HashMap<Entity, ExtractedEffect>,
    /// Entites which had their [`ParticleEffect`] component removed.
    pub removed_effect_entities: Vec<Entity>,
    /// Newly added effects without a GPU allocation yet.
    pub added_effects: Vec<AddedEffect>,
}

#[derive(Default, Resource)]
pub(crate) struct EffectAssetEvents {
    pub images: Vec<AssetEvent<Image>>,
}

pub(crate) fn extract_effect_events(
    mut events: ResMut<EffectAssetEvents>,
    mut image_events: Extract<EventReader<AssetEvent<Image>>>,
) {
    trace!("extract_effect_events");

    let EffectAssetEvents { ref mut images } = *events;
    images.clear();

    for image in image_events.iter() {
        // AssetEvent: !Clone
        images.push(match image {
            AssetEvent::Created { handle } => AssetEvent::Created {
                handle: handle.clone_weak(),
            },
            AssetEvent::Modified { handle } => AssetEvent::Modified {
                handle: handle.clone_weak(),
            },
            AssetEvent::Removed { handle } => AssetEvent::Removed {
                handle: handle.clone_weak(),
            },
        });
    }
}

/// System extracting data for rendering of all active [`ParticleEffect`]
/// components.
///
/// Extract rendering data for all [`ParticleEffect`] components in the world
/// which are visible ([`ComputedVisibility::is_visible`] is `true`), and wrap
/// the data into a new [`ExtractedEffect`] instance added to the
/// [`ExtractedEffects`] resource.
pub(crate) fn extract_effects(
    time: Extract<Res<Time>>,
    effects: Extract<Res<Assets<EffectAsset>>>,
    _images: Extract<Res<Assets<Image>>>,
    mut query: Extract<
        ParamSet<(
            // All existing ParticleEffect components
            Query<(
                Entity,
                &ComputedVisibility,
                &ParticleEffect, /* TODO - Split EffectAsset::Spawner (desc) and
                                  * ParticleEffect::SpawnerData (runtime data), and init the
                                  * latter on component add without a need for the former */
                &GlobalTransform,
            )>,
            // Newly added ParticleEffect components
            Query<
                (Entity, &ParticleEffect),
                (
                    Added<ParticleEffect>,
                    With<ComputedVisibility>,
                    With<GlobalTransform>,
                ),
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
    sim_params.time = time.seconds_since_startup();
    sim_params.dt = dt;

    // Collect removed effects for later GPU data purge
    extracted_effects.removed_effect_entities =
        removed_effects_event_reader
            .iter()
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
            let handle = effect.handle.clone_weak();
            let asset = effects.get(&effect.handle).unwrap();
            AddedEffect {
                entity,
                capacity: asset.capacity,
                item_size: Particle::min_size().get() as u32, // effect.item_size(),
                handle,
            }
        })
        .collect();

    // Loop over all existing effects to update them
    for (entity, computed_visibility, effect, transform) in query.p0().iter_mut() {
        // Check if visible
        if !computed_visibility.is_visible() {
            continue;
        }

        // Check if shaders are configured
        let compute_shader = if let Some(compute_shader) = &effect.configured_compute_shader {
            compute_shader
        } else {
            continue;
        };
        let render_shader = if let Some(render_shader) = &effect.configured_render_shader {
            render_shader
        } else {
            continue;
        };

        // TEMP - see tick_spawners()
        let spawn_count = effect.spawn_count;
        let accel = effect.accel;
        let force_field = effect.force_field;
        let position_code = effect.position_code.clone();
        let force_field_code = effect.force_field_code.clone();
        let lifetime_code = effect.lifetime_code.clone();

        // Check if asset is available, otherwise silently ignore
        if let Some(asset) = effects.get(&effect.handle) {
            let z_sort_key_2d = if let Some(z_layer_2d) = effect.z_layer_2d {
                FloatOrd(z_layer_2d)
            } else {
                FloatOrd(asset.z_layer_2d)
            };

            extracted_effects.effects.insert(
                entity,
                ExtractedEffect {
                    handle: effect.handle.clone_weak(),
                    spawn_count,
                    color: Color::RED, //effect.color,
                    transform: transform.compute_matrix(),
                    accel,
                    force_field,
                    rect: MinMaxRect {
                        min: Vec2::splat(-0.1),
                        max: Vec2::splat(0.1), // effect
                                               //.custom_size
                                               //.unwrap_or_else(|| Vec2::new(size.width as f32, size.height as f32)),
                    },
                    has_image: asset.render_layout.particle_texture.is_some(),
                    image_handle_id: asset
                        .render_layout
                        .particle_texture
                        .clone()
                        .map_or(HandleId::default::<Image>(), |handle| handle.id()),
                    compute_shader: compute_shader.clone(),
                    render_shader: render_shader.clone(),
                    position_code,
                    force_field_code,
                    lifetime_code,
                    z_sort_key_2d,
                },
            );
        }
    }
}

/// A single particle as stored in a GPU buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, ShaderType)]
struct Particle {
    /// Particle position in effect space (local or world).
    pub position: [f32; 3],
    /// Current particle age in \[0:`lifetime`\].
    pub age: f32,
    /// Particle velocity in effect space (local or world).
    pub velocity: [f32; 3],
    /// Total particle lifetime.
    pub lifetime: f32,
}

/// A single vertex of a particle mesh as stored in a GPU buffer.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ParticleVertex {
    /// Vertex position.
    pub position: [f32; 3],
    /// UV coordinates of vertex.
    pub uv: [f32; 2],
}

/// Global resource containing the GPU data to draw all the particle effects in
/// all views.
///
/// The resource is populated by [`prepare_effects()`] with all the effects to
/// render for the current frame, for all views in the frame, and consumed by
/// [`queue_effects()`] to actually enqueue the drawning commands to draw those
/// effects.
#[derive(Resource)]
pub(crate) struct EffectsMeta {
    /// Map from an entity with a [`ParticleEffect`] component attached to it,
    /// to the associated effect slice allocated in an [`EffectCache`].
    entity_map: HashMap<Entity, EffectCacheId>,
    /// Global effect cache for all effects in use.
    effect_cache: EffectCache,
    /// Bind group for the camera view, containing the camera projection and
    /// other uniform values related to the camera.
    view_bind_group: Option<BindGroup>,
    /// Bind group for the simulation parameters, like the current time and
    /// frame delta time.
    sim_params_bind_group: Option<BindGroup>,
    /// Bind group for the particles buffer itself.
    particles_bind_group: Option<BindGroup>,
    /// Bind group for the spawning parameters (number of particles to spawn
    /// this frame, ...).
    spawner_bind_group: Option<BindGroup>,
    sim_params_uniforms: UniformBuffer<SimParamsUniform>,
    spawner_buffer: AlignedBufferVec<GpuSpawnerParams>,
    /// Unscaled vertices of the mesh of a single particle, generally a quad.
    /// The mesh is later scaled during rendering by the "particle size".
    // FIXME - This is a per-effect thing, unless we merge all meshes into a single buffer (makes
    // sense) but in that case we need a vertex slice too to know which mesh to draw per effect.
    vertices: BufferVec<ParticleVertex>,
}

impl EffectsMeta {
    pub fn new(device: RenderDevice) -> Self {
        let mut vertices = BufferVec::new(BufferUsages::VERTEX);
        for v in QUAD_VERTEX_POSITIONS {
            let uv = v.truncate() + 0.5;
            let v = *v * Vec3::new(1.0, 1.0, 1.0);
            vertices.push(ParticleVertex {
                position: v.into(),
                uv: uv.into(),
            });
        }

        let item_align = device.limits().min_storage_buffer_offset_alignment as u64;

        Self {
            entity_map: HashMap::default(),
            effect_cache: EffectCache::new(device),
            view_bind_group: None,
            sim_params_bind_group: None,
            particles_bind_group: None,
            spawner_bind_group: None,
            sim_params_uniforms: UniformBuffer::default(),
            spawner_buffer: AlignedBufferVec::new(
                BufferUsages::STORAGE,
                NonZeroU64::new(item_align),
                Some("hanabi:spawner_buffer".to_string()),
            ),
            vertices,
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
    struct LayoutFlags: u32 {
        const NONE = 0;
        const PARTICLE_TEXTURE = 0b00000001;
    }
}

impl Default for LayoutFlags {
    fn default() -> Self {
        LayoutFlags::NONE
    }
}

/// A batch of multiple instances of the same effect, rendered all together to
/// reduce GPU shader permutations and draw call overhead.
#[derive(Component)]
pub(crate) struct EffectBatch {
    /// Index of the GPU effect buffer effects in this batch are contained in.
    buffer_index: u32,
    /// Index of the first Spawner of the effects in the batch.
    spawner_base: u32,
    /// Size of a single particle.
    item_size: u32,
    /// Slice of particles in the GPU effect buffer for the entire batch.
    slice: Range<u32>,
    /// Handle of the underlying effect asset describing the effect.
    handle: Handle<EffectAsset>,
    /// Flags describing the render layout.
    layout_flags: LayoutFlags,
    /// Texture to modulate the particle color.
    image_handle_id: HandleId,
    /// Configured shader used for the particle updating of this batch.
    compute_shader: Handle<Shader>,
    /// Configured shader used for the particle rendering of this batch.
    render_shader: Handle<Shader>,
    /// Update position code.
    position_code: String,
    /// Update force field code.
    force_field_code: String,
    /// Update lifetime code.
    lifetime_code: String,
    /// Compute pipeline specialized for this batch.
    compute_pipeline_id: CachedComputePipelineId,
    /// For 2D rendering, the Z coordinate used as the sort key. Ignored for 3D
    /// rendering.
    z_sort_key_2d: FloatOrd,
}

pub(crate) fn prepare_effects(
    mut commands: Commands,
    sim_params: Res<SimParams>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut pipeline_cache: ResMut<PipelineCache>,
    update_pipeline: Res<ParticlesUpdatePipeline>,
    mut specialized_compute_pipelines: ResMut<SpecializedComputePipelines<ParticlesUpdatePipeline>>,
    //update_pipeline: Res<ParticlesUpdatePipeline>, // TODO move update_pipeline.pipeline to
    // EffectsMeta
    mut effects_meta: ResMut<EffectsMeta>,
    mut extracted_effects: ResMut<ExtractedEffects>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
) {
    trace!("prepare_effects");

    // Allocate simulation uniform if needed
    //if effects_meta.sim_params_uniforms.is_empty() {
    effects_meta
        .sim_params_uniforms
        .set(SimParamsUniform::default());
    //}

    // Update simulation parameters
    {
        let sim_params_uni = effects_meta.sim_params_uniforms.get_mut();
        let sim_params = *sim_params;
        *sim_params_uni = sim_params.into();
    }
    trace!(
        "Simulation parameters: time={} dt={}",
        sim_params.time,
        sim_params.dt
    );
    effects_meta
        .sim_params_uniforms
        .write_buffer(&render_device, &render_queue);

    // Allocate spawner buffer if needed
    //if effects_meta.spawner_buffer.is_empty() {
    //    effects_meta.spawner_buffer.push(GpuSpawnerParams::default());
    //}

    // Write vertices (TODO - lazily once only)
    effects_meta
        .vertices
        .write_buffer(&render_device, &render_queue);

    // Allocate GPU data for newly created effect instances. Do this first to ensure
    // a group is not left unused and dropped due to the last effect being
    // removed but a new compatible one added not being inserted yet. By
    // inserting first, we ensure the group is not dropped in this case.
    trace!(
        "Adding {} newly spawned effects",
        extracted_effects.added_effects.len()
    );
    for added_effect in extracted_effects.added_effects.drain(..) {
        let entity = added_effect.entity;
        let id = effects_meta.effect_cache.insert(
            added_effect.handle,
            added_effect.capacity,
            added_effect.item_size,
            //update_pipeline.pipeline.clone(),
            &render_queue,
        );
        effects_meta.entity_map.insert(entity, id);
        // Note: those effects are already in extracted_effects.effects because
        // they were gathered by the same query as previously existing
        // ones, during extraction.
    }

    // Deallocate GPU data for destroyed effect instances. This will automatically
    // drop any group where there is no more effect slice.
    let removed_entities = std::mem::take(&mut extracted_effects.removed_effect_entities);
    trace!("Removing {} despawned effects", removed_entities.len());
    for entity in &removed_entities {
        trace!("Removing ParticleEffect on entity {:?}", entity);
        if let Some(id) = effects_meta.entity_map.remove(entity) {
            trace!(
                "=> ParticleEffect on entity {:?} had cache ID {:?}, removing...",
                entity,
                id
            );
            if let Some(buffer_index) = effects_meta.effect_cache.remove(id) {
                // Clear bind groups associated with the removed buffer
                effect_bind_groups
                    .update_particle_buffers
                    .remove(&buffer_index);
                effect_bind_groups
                    .render_particle_buffers
                    .remove(&buffer_index);
            }
        }
        extracted_effects.effects.remove(entity);
    }

    // // sort first by z and then by handle. this ensures that, when possible,
    // batches span multiple z layers // batches won't span z-layers if there is
    // another batch between them extracted_effects.effects.sort_by(|a, b| {
    //     match FloatOrd(a.transform.w_axis[2]).cmp(&FloatOrd(b.transform.
    // w_axis[2])) {         Ordering::Equal => a.handle.cmp(&b.handle),
    //         other => other,
    //     }
    // });

    // Get the effect-entity mapping
    let mut effect_entity_list = extracted_effects
        .effects
        .iter()
        .map(|(entity, extracted_effect)| {
            let id = *effects_meta.entity_map.get(entity).unwrap();
            let slice = effects_meta.effect_cache.get_slice(id);
            (slice, extracted_effect)
        })
        .collect::<Vec<_>>();
    trace!("Collected {} extracted effects", effect_entity_list.len());

    // Sort first by effect buffer, then by slice range (see EffectSlice)
    effect_entity_list.sort_by(|a, b| a.0.cmp(&b.0));

    // Loop on all extracted effects in order
    effects_meta.spawner_buffer.clear();
    let mut spawner_base = 0;
    let mut item_size = 0;
    let mut current_buffer_index = u32::MAX;
    let mut asset: Handle<EffectAsset> = Default::default();
    let mut layout_flags = LayoutFlags::NONE;
    let mut image_handle_id: HandleId = HandleId::default::<Image>();
    let mut compute_shader: Handle<Shader> = Default::default();
    let mut render_shader: Handle<Shader> = Default::default();
    let mut start = 0;
    let mut end = 0;
    let mut num_emitted = 0;
    let mut position_code = String::default();
    let mut force_field_code = String::default();
    let mut lifetime_code = String::default();
    let mut compute_pipeline_id = CachedComputePipelineId::INVALID;
    let mut z_sort_key_2d = FloatOrd(f32::NAN);

    for (slice, extracted_effect) in effect_entity_list {
        let buffer_index = slice.group_index;
        let range = slice.slice;
        layout_flags = if extracted_effect.has_image {
            LayoutFlags::PARTICLE_TEXTURE
        } else {
            LayoutFlags::NONE
        };
        image_handle_id = extracted_effect.image_handle_id;
        trace!(
            "Effect: buffer #{} | range {:?} | z_sort_key_2d {:?}",
            buffer_index,
            range,
            extracted_effect.z_sort_key_2d
        );

        // Check the buffer the effect is in
        assert!(buffer_index >= current_buffer_index || current_buffer_index == u32::MAX);
        // FIXME - This breaks batches in 3D even though the Z sort key is only for 2D.
        // Do we need separate batches for 2D and 3D? :'(
        if current_buffer_index != buffer_index || z_sort_key_2d != extracted_effect.z_sort_key_2d {
            if current_buffer_index != buffer_index {
                trace!(
                    "+ New buffer! ({} -> {})",
                    current_buffer_index,
                    buffer_index
                );
            } else {
                trace!(
                    "+ New Z sort key! ({:?} -> {:?})",
                    z_sort_key_2d,
                    extracted_effect.z_sort_key_2d
                );
            }
            // Commit previous buffer if any
            if current_buffer_index != u32::MAX {
                // Record open batch if any
                trace!("+ Prev: {} - {}", start, end);
                if end > start {
                    assert_ne!(asset, Handle::<EffectAsset>::default());
                    assert!(item_size > 0);
                    trace!(
                        "Emit batch: buffer #{} | spawner_base {} | slice {:?} | item_size {} | compute_shader {:?} | render_shader {:?} | z_sort_key_2d {:?}",
                        current_buffer_index,
                        spawner_base,
                        start..end,
                        item_size,
                        compute_shader,
                        render_shader,
                        z_sort_key_2d,
                    );
                    commands.spawn((EffectBatch {
                        buffer_index: current_buffer_index,
                        spawner_base: spawner_base as u32,
                        slice: start..end,
                        item_size,
                        handle: asset.clone_weak(),
                        layout_flags,
                        image_handle_id,
                        compute_shader: compute_shader.clone(),
                        render_shader: render_shader.clone(),
                        position_code: position_code.clone(),
                        force_field_code: force_field_code.clone(),
                        lifetime_code: lifetime_code.clone(),
                        compute_pipeline_id,
                        z_sort_key_2d,
                    },));
                    num_emitted += 1;
                }
            }

            // Move to next buffer
            current_buffer_index = buffer_index;
            start = 0;
            end = 0;
            spawner_base = effects_meta.spawner_buffer.len();
            trace!("+ New spawner_base = {}", spawner_base);
            // Each effect buffer contains effect instances with a compatible layout
            // FIXME - Currently this means same effect asset, so things are easier...
            asset = extracted_effect.handle.clone_weak();
            item_size = slice.item_size;
            z_sort_key_2d = extracted_effect.z_sort_key_2d;
        }

        assert_ne!(asset, Handle::<EffectAsset>::default());

        // Specialize the compute pipeline based on the effect
        trace!(
            "Specializing compute pipeline: compute_shader={:?}",
            extracted_effect.compute_shader
        );
        compute_pipeline_id = specialized_compute_pipelines.specialize(
            &mut pipeline_cache,
            &update_pipeline,
            ParticleUpdatePipelineKey {
                shader: extracted_effect.compute_shader.clone(),
            },
        );
        trace!("Compute pipeline specialized: id={:?}", compute_pipeline_id);

        compute_shader = extracted_effect.compute_shader.clone();
        trace!("compute_shader = {:?}", compute_shader);

        render_shader = extracted_effect.render_shader.clone();
        trace!("render_shader = {:?}", render_shader);

        trace!("item_size = {}B", slice.item_size);

        position_code = extracted_effect.position_code.clone();
        trace!("position_code = {}", position_code);

        force_field_code = extracted_effect.force_field_code.clone();
        trace!("force_field_code = {}", force_field_code);

        lifetime_code = extracted_effect.lifetime_code.clone();
        trace!("lifetime_code = {}", lifetime_code);

        trace!("z_sort_key_2d = {:?}", z_sort_key_2d);

        // extract the force field and turn it into a struct that is compliant with
        // GPU use, namely GpuForceFieldSource
        let mut extracted_force_field =
            [GpuForceFieldSource::default(); ForceFieldSource::MAX_SOURCES];
        for (i, ff) in extracted_effect.force_field.iter().enumerate() {
            extracted_force_field[i] = (*ff).into();
        }

        // Prepare the spawner block for the current slice
        // FIXME - This is once per EFFECT/SLICE, not once per BATCH, so indeed this is
        // spawner_BASE, and need an array of them in the compute shader!!!!!!!!!!!!!!
        let tr = extracted_effect.transform.transpose().to_cols_array();
        let spawner_params = GpuSpawnerParams {
            spawn: extracted_effect.spawn_count as i32,
            count: 0,
            transform: [
                tr[0], tr[1], tr[2], tr[3], tr[4], tr[5], tr[6], tr[7], tr[8], tr[9], tr[10],
                tr[11],
            ],
            accel: extracted_effect.accel,
            force_field: extracted_force_field, // extracted_effect.force_field,
            seed: random::<u32>(),
        };
        trace!("spawner_params = {:?}", spawner_params);
        effects_meta.spawner_buffer.push(spawner_params);

        trace!("slice = {}-{} | prev end = {}", range.start, range.end, end);
        if (range.start > end) || (item_size != slice.item_size) {
            // Discontinuous slices; create a new batch
            if end > start {
                // Record the previous batch
                assert_ne!(asset, Handle::<EffectAsset>::default());
                assert!(item_size > 0);
                trace!(
                    "Emit batch: buffer #{} | spawner_base {} | slice {:?} | item_size {} | compute_shader {:?} | render_shader {:?}",
                    buffer_index,
                    spawner_base,
                    start..end,
                    item_size,
                    compute_shader,
                    render_shader
                );
                commands.spawn((EffectBatch {
                    buffer_index,
                    spawner_base: spawner_base as u32,
                    slice: start..end,
                    item_size,
                    handle: asset.clone_weak(),
                    layout_flags,
                    image_handle_id,
                    compute_shader: compute_shader.clone(),
                    render_shader: render_shader.clone(),
                    position_code: position_code.clone(),
                    force_field_code: force_field_code.clone(),
                    lifetime_code: lifetime_code.clone(),
                    compute_pipeline_id,
                    z_sort_key_2d,
                },));
                num_emitted += 1;
            }
            start = range.start;
            item_size = slice.item_size;
        }
        end = range.end;
    }

    // Record last open batch if any
    if end > start {
        assert_ne!(asset, Handle::<EffectAsset>::default());
        assert!(item_size > 0);
        trace!(
            "Emit LAST batch: buffer #{} | spawner_base {} | slice {:?} | item_size {} | compute_shader {:?} | render_shader {:?}",
            current_buffer_index,
            spawner_base,
            start..end,
            item_size,
            compute_shader,
            render_shader
        );
        commands.spawn((EffectBatch {
            buffer_index: current_buffer_index,
            spawner_base: spawner_base as u32,
            slice: start..end,
            item_size,
            handle: asset.clone_weak(),
            layout_flags,
            image_handle_id,
            compute_shader,
            render_shader,
            position_code,
            force_field_code,
            lifetime_code,
            compute_pipeline_id,
            z_sort_key_2d,
        },));
        num_emitted += 1;
    }
    trace!(
        "Emitted {} buffers, spawner_buffer len = {}",
        num_emitted,
        effects_meta.spawner_buffer.len()
    );

    // Write the entire spawner buffer for this frame, for all effects combined
    effects_meta
        .spawner_buffer
        .write_buffer(&render_device, &render_queue);
}

#[derive(Default, Resource)]
pub(crate) struct EffectBindGroups {
    /// Bind groups for each group index for compute shader.
    update_particle_buffers: HashMap<u32, BindGroup>,
    /// Same for render shader.
    render_particle_buffers: HashMap<u32, BindGroup>,
    ///
    images: HashMap<Handle<Image>, BindGroup>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn queue_effects(
    #[cfg(feature = "2d")] draw_functions_2d: Res<DrawFunctions<Transparent2d>>,
    #[cfg(feature = "3d")] draw_functions_3d: Res<DrawFunctions<Transparent3d>>,
    render_device: Res<RenderDevice>,
    mut effects_meta: ResMut<EffectsMeta>,
    view_uniforms: Res<ViewUniforms>,
    update_pipeline: Res<ParticlesUpdatePipeline>,
    render_pipeline: Res<ParticlesRenderPipeline>,
    mut specialized_render_pipelines: ResMut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    gpu_images: Res<RenderAssets<Image>>,
    effect_batches: Query<(Entity, &mut EffectBatch)>,
    #[cfg(feature = "2d")] mut views_2d: Query<&mut RenderPhase<Transparent2d>>,
    #[cfg(feature = "3d")] mut views_3d: Query<&mut RenderPhase<Transparent3d>>,
    events: Res<EffectAssetEvents>,
    msaa: Res<Msaa>,
) {
    trace!("queue_effects");

    // If an image has changed, the GpuImage has (probably) changed
    for event in &events.images {
        match event {
            AssetEvent::Created { .. } => None,
            AssetEvent::Modified { handle } => effect_bind_groups.images.remove(handle),
            AssetEvent::Removed { handle } => effect_bind_groups.images.remove(handle),
        };
    }

    // Get the binding for the ViewUniform, the uniform data structure containing
    // the Camera data for the current view.
    let view_binding = match view_uniforms.uniforms.binding() {
        Some(view_binding) => view_binding,
        None => {
            return;
        }
    };

    if effects_meta.spawner_buffer.buffer().is_none() {
        // No spawners are active
        return;
    }

    // Create the bind group for the camera/view parameters
    effects_meta.view_bind_group = Some(render_device.create_bind_group(&BindGroupDescriptor {
        entries: &[BindGroupEntry {
            binding: 0,
            resource: view_binding,
        }],
        label: Some("hanabi:view_bind_group"),
        layout: &render_pipeline.view_layout,
    }));

    // Create the bind group for the global simulation parameters
    effects_meta.sim_params_bind_group =
        Some(render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[BindGroupEntry {
                binding: 0,
                resource: effects_meta.sim_params_uniforms.binding().unwrap(),
            }],
            label: Some("hanabi:sim_params_bind_group"),
            layout: &update_pipeline.sim_params_layout,
        }));

    // Create the bind group for the spawner parameters
    trace!(
        "GpuSpawnerParams::min_size() = {}",
        GpuSpawnerParams::min_size()
    );
    effects_meta.spawner_bind_group = Some(render_device.create_bind_group(&BindGroupDescriptor {
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: effects_meta.spawner_buffer.buffer().unwrap(),
                offset: 0,
                size: Some(GpuSpawnerParams::min_size()),
            }),
        }],
        label: Some("hanabi:spawner_bind_group"),
        layout: &update_pipeline.spawner_buffer_layout,
    }));

    // Queue the update compute
    trace!("queue effects from cache...");
    for (buffer_index, buffer) in effects_meta
        .effect_cache
        .buffers_mut()
        .iter_mut()
        .enumerate()
    {
        let buffer = if let Some(buffer) = buffer {
            buffer
        } else {
            continue;
        };

        // Ensure all effect groups have a bind group for the entire buffer of the
        // group, since the update phase runs on an entire group/buffer at once,
        // with all the effect instances in it batched together.
        trace!("effect buffer_index=#{}", buffer_index);
        effect_bind_groups
            .update_particle_buffers
            .entry(buffer_index as u32)
            .or_insert_with(|| {
                trace!(
                    "Create new particle update bind group for buffer_index={}",
                    buffer_index
                );
                render_device.create_bind_group(&BindGroupDescriptor {
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: buffer.max_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.indirect_max_binding(),
                        },
                    ],
                    label: Some(&format!(
                        "hanabi:vfx_particles_bind_group_update{}",
                        buffer_index
                    )),
                    layout: &update_pipeline.particles_buffer_layout,
                })
            });

        // Same for the render pipeline, ensure all buffers have a bind group.
        effect_bind_groups
            .render_particle_buffers
            .entry(buffer_index as u32)
            .or_insert_with(|| {
                trace!(
                    "Create new particle render bind group for buffer_index={}",
                    buffer_index
                );
                render_device.create_bind_group(&BindGroupDescriptor {
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: buffer.max_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: buffer.indirect_max_binding(),
                        },
                    ],
                    label: Some(&format!(
                        "hanabi:vfx_particles_bind_group_render{}",
                        buffer_index
                    )),
                    layout: &render_pipeline.particles_buffer_layout,
                })
            });
    }

    // Loop over all 2D cameras/views that need to render effects
    #[cfg(feature = "2d")]
    {
        let draw_effects_function_2d = draw_functions_2d.read().get_id::<DrawEffects>().unwrap();
        for mut transparent_phase_2d in views_2d.iter_mut() {
            trace!("Process new Transparent2d view");
            // For each view, loop over all the effect batches to determine if the effect
            // needs to be rendered for that view, and enqueue a view-dependent
            // batch if so.
            for (entity, batch) in effect_batches.iter() {
                trace!(
                    "Process batch entity={:?} buffer_index={} spawner_base={} slice={:?}",
                    entity,
                    batch.buffer_index,
                    batch.spawner_base,
                    batch.slice
                );
                // Ensure the particle texture is available as a GPU resource and create a bind
                // group for it
                let particle_texture = if batch.layout_flags.contains(LayoutFlags::PARTICLE_TEXTURE)
                {
                    let image_handle = Handle::weak(batch.image_handle_id);
                    if effect_bind_groups.images.get(&image_handle).is_none() {
                        trace!(
                            "Batch buffer #{} slice={:?} has missing GPU image bind group, creating...",
                            batch.buffer_index,
                            batch.slice
                        );
                        // If texture doesn't have a bind group yet from another instance of the
                        // same effect, then try to create one now
                        if let Some(gpu_image) = gpu_images.get(&image_handle) {
                            let bind_group =
                                render_device.create_bind_group(&BindGroupDescriptor {
                                    entries: &[
                                        BindGroupEntry {
                                            binding: 0,
                                            resource: BindingResource::TextureView(
                                                &gpu_image.texture_view,
                                            ),
                                        },
                                        BindGroupEntry {
                                            binding: 1,
                                            resource: BindingResource::Sampler(&gpu_image.sampler),
                                        },
                                    ],
                                    label: Some("hanabi:material_bind_group"),
                                    layout: &render_pipeline.material_layout,
                                });
                            effect_bind_groups
                                .images
                                .insert(image_handle.clone(), bind_group);
                            Some(image_handle)
                        } else {
                            // Texture is not ready; skip for now...
                            trace!("GPU image not yet available; skipping batch for now.");
                            None
                        }
                    } else {
                        // Bind group already exists, meaning texture is ready
                        Some(image_handle)
                    }
                } else {
                    // Batch doesn't use particle texture
                    None
                };

                // Specialize the render pipeline based on the effect batch
                trace!(
                    "Specializing render pipeline: render_shader={:?} particle_texture={:?}",
                    batch.render_shader,
                    particle_texture
                );
                let render_pipeline_id = specialized_render_pipelines.specialize(
                    &mut pipeline_cache,
                    &render_pipeline,
                    ParticleRenderPipelineKey {
                        particle_texture,
                        shader: batch.render_shader.clone(),
                        #[cfg(feature = "3d")]
                        pipeline_mode: PipelineMode::Camera2d,
                        msaa_samples: msaa.samples,
                    },
                );
                trace!("Render pipeline specialized: id={:?}", render_pipeline_id);

                // Add a draw pass for the effect batch
                trace!("Add Transparent for batch on entity {:?}: buffer_index={} spawner_base={} slice={:?} handle={:?}", entity, batch.buffer_index, batch.spawner_base, batch.slice, batch.handle);
                transparent_phase_2d.add(Transparent2d {
                    draw_function: draw_effects_function_2d,
                    pipeline: render_pipeline_id,
                    entity,
                    sort_key: batch.z_sort_key_2d,
                    batch_range: None,
                });
            }
        }
    }

    // Loop over all 3D cameras/views that need to render effects
    #[cfg(feature = "3d")]
    {
        let draw_effects_function_3d = draw_functions_3d.read().get_id::<DrawEffects>().unwrap();
        for mut transparent_phase_3d in views_3d.iter_mut() {
            trace!("Process new Transparent3d view");
            // For each view, loop over all the effect batches to determine if the effect
            // needs to be rendered for that view, and enqueue a view-dependent
            // batch if so.
            for (entity, batch) in effect_batches.iter() {
                trace!(
                    "Process batch entity={:?} buffer_index={} spawner_base={} slice={:?}",
                    entity,
                    batch.buffer_index,
                    batch.spawner_base,
                    batch.slice
                );
                // Ensure the particle texture is available as a GPU resource and create a bind
                // group for it
                let particle_texture = if batch.layout_flags.contains(LayoutFlags::PARTICLE_TEXTURE)
                {
                    let image_handle = Handle::weak(batch.image_handle_id);
                    if effect_bind_groups.images.get(&image_handle).is_none() {
                        trace!(
                            "Batch buffer #{} slice={:?} has missing GPU image bind group, creating...",
                            batch.buffer_index,
                            batch.slice
                        );
                        // If texture doesn't have a bind group yet from another instance of the
                        // same effect, then try to create one now
                        if let Some(gpu_image) = gpu_images.get(&image_handle) {
                            let bind_group =
                                render_device.create_bind_group(&BindGroupDescriptor {
                                    entries: &[
                                        BindGroupEntry {
                                            binding: 0,
                                            resource: BindingResource::TextureView(
                                                &gpu_image.texture_view,
                                            ),
                                        },
                                        BindGroupEntry {
                                            binding: 1,
                                            resource: BindingResource::Sampler(&gpu_image.sampler),
                                        },
                                    ],
                                    label: Some("hanabi:material_bind_group"),
                                    layout: &render_pipeline.material_layout,
                                });
                            effect_bind_groups
                                .images
                                .insert(image_handle.clone(), bind_group);
                            Some(image_handle)
                        } else {
                            // Texture is not ready; skip for now...
                            trace!("GPU image not yet available; skipping batch for now.");
                            None
                        }
                    } else {
                        // Bind group already exists, meaning texture is ready
                        Some(image_handle)
                    }
                } else {
                    // Batch doesn't use particle texture
                    None
                };

                // Specialize the render pipeline based on the effect batch
                trace!(
                    "Specializing render pipeline: render_shader={:?} particle_texture={:?}",
                    batch.render_shader,
                    particle_texture
                );
                let render_pipeline_id = specialized_render_pipelines.specialize(
                    &mut pipeline_cache,
                    &render_pipeline,
                    ParticleRenderPipelineKey {
                        particle_texture,
                        shader: batch.render_shader.clone(),
                        #[cfg(feature = "2d")]
                        pipeline_mode: PipelineMode::Camera3d,
                        msaa_samples: msaa.samples,
                    },
                );
                trace!("Render pipeline specialized: id={:?}", render_pipeline_id);

                // Add a draw pass for the effect batch
                trace!("Add Transparent for batch on entity {:?}: buffer_index={} spawner_base={} slice={:?} handle={:?}", entity, batch.buffer_index, batch.spawner_base, batch.slice, batch.handle);
                transparent_phase_3d.add(Transparent3d {
                    draw_function: draw_effects_function_3d,
                    pipeline: render_pipeline_id,
                    entity,
                    distance: 0.0, // TODO ??????
                });
            }
        }
    }
}

/// Component to hold all the entities with a [`ExtractedEffect`] component on
/// them that need to be updated this frame with a compute pass. This is
/// view-independent because the update phase itself is also view-independent
/// (effects like camera facing are applied in the render phase, which runs once
/// per view).
#[derive(Component)]
struct ExtractedEffectEntities {
    pub entities: Vec<Entity>,
}

/// Draw function for rendering all active effects for the current frame.
///
/// Effects are rendered in the [`Transparent2d`] phase of the main 2D pass,
/// and the [`Transparent3d`] phase of the main 3D pass.
pub(crate) struct DrawEffects {
    params: SystemState<(
        SRes<EffectsMeta>,
        SRes<EffectBindGroups>,
        SRes<PipelineCache>,
        SQuery<Read<ViewUniformOffset>>,
        SQuery<Read<EffectBatch>>,
    )>,
}

impl DrawEffects {
    pub fn new(world: &mut World) -> Self {
        Self {
            params: SystemState::new(world),
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
    ) {
        trace!("Draw<Transparent2d>: view={:?}", view);
        let (effects_meta, effect_bind_groups, pipeline_cache, views, effects) =
            self.params.get(world);
        let view_uniform = views.get(view).unwrap();
        let effects_meta = effects_meta.into_inner();
        let effect_bind_groups = effect_bind_groups.into_inner();
        let effect_batch = effects.get(item.entity).unwrap();
        if let Some(pipeline) = pipeline_cache
            .into_inner()
            .get_render_pipeline(item.pipeline)
        {
            trace!("render pass");
            //let effect_group = &effects_meta.effect_cache.buffers()[0]; // TODO

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
            pass.set_bind_group(
                1,
                effect_bind_groups
                    .render_particle_buffers
                    .get(&effect_batch.buffer_index)
                    .unwrap(),
                &[],
            );

            // Particle texture
            if effect_batch
                .layout_flags
                .contains(LayoutFlags::PARTICLE_TEXTURE)
            {
                let image_handle = Handle::weak(effect_batch.image_handle_id);
                if let Some(bind_group) = effect_bind_groups.images.get(&image_handle) {
                    pass.set_bind_group(2, bind_group, &[]);
                } else {
                    // Texture not ready; skip this drawing for now
                    trace!(
                        "Particle texture bind group not available for batch buf={} slice={:?}. Skipping draw call.",
                        effect_batch.buffer_index,
                        effect_batch.slice
                    );
                    return; //continue;
                }
            }

            let vertex_count = effects_meta.vertices.len() as u32;
            let particle_count = effect_batch.slice.end - effect_batch.slice.start;

            trace!(
                "Draw {} particles with {} vertices per particle for batch from buffer #{}.",
                particle_count,
                vertex_count,
                effect_batch.buffer_index
            );
            pass.draw(0..vertex_count, 0..particle_count);
        }
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
        let (effects_meta, effect_bind_groups, pipeline_cache, views, effects) =
            self.params.get(world);
        let view_uniform = views.get(view).unwrap();
        let effects_meta = effects_meta.into_inner();
        let effect_bind_groups = effect_bind_groups.into_inner();
        let effect_batch = effects.get(item.entity).unwrap();
        if let Some(pipeline) = pipeline_cache
            .into_inner()
            .get_render_pipeline(item.pipeline)
        {
            trace!("render pass");
            //let effect_group = &effects_meta.effect_cache.buffers()[0]; // TODO

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
            pass.set_bind_group(
                1,
                effect_bind_groups
                    .render_particle_buffers
                    .get(&effect_batch.buffer_index)
                    .unwrap(),
                &[],
            );

            // Particle texture
            if effect_batch
                .layout_flags
                .contains(LayoutFlags::PARTICLE_TEXTURE)
            {
                let image_handle = Handle::weak(effect_batch.image_handle_id);
                if let Some(bind_group) = effect_bind_groups.images.get(&image_handle) {
                    pass.set_bind_group(2, bind_group, &[]);
                } else {
                    // Texture not ready; skip this drawing for now
                    trace!(
                        "Particle texture bind group not available for batch buf={} slice={:?}. Skipping draw call.",
                        effect_batch.buffer_index,
                        effect_batch.slice
                    );
                    return; //continue;
                }
            }

            let vertex_count = effects_meta.vertices.len() as u32;
            let particle_count = effect_batch.slice.end - effect_batch.slice.start;

            trace!(
                "Draw {} particles with {} vertices per particle for batch from buffer #{}.",
                particle_count,
                vertex_count,
                effect_batch.buffer_index
            );
            pass.draw(0..vertex_count, 0..particle_count);
        }
    }
}

/// A render node to update the particles of all particle efects.
pub(crate) struct ParticleUpdateNode {
    /// Query to retrieve the list of entities holding an extracted particle
    /// effect to update.
    entity_query: QueryState<&'static ExtractedEffectEntities>,
    /// Query to retrieve the
    effect_query: QueryState<&'static EffectBatch>,
}

impl ParticleUpdateNode {
    /// Input entity marking the view.
    pub const IN_VIEW: &'static str = "view";

    /// Output particle buffer for that view. TODO - how to handle multiple
    /// buffers?! Should use Entity instead??
    //pub const OUT_PARTICLE_BUFFER: &'static str = "particle_buffer";

    pub fn new(world: &mut World) -> Self {
        Self {
            entity_query: QueryState::new(world),
            effect_query: QueryState::new(world),
        }
    }
}

impl Node for ParticleUpdateNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(ParticleUpdateNode::IN_VIEW, SlotType::Entity)]
    }

    // fn output(&self) -> Vec<SlotInfo> {
    //     vec![SlotInfo::new(
    //         ParticleUpdateNode::OUT_PARTICLE_BUFFER,
    //         SlotType::Buffer,
    //     )]
    // }

    fn update(&mut self, world: &mut World) {
        trace!("ParticleUpdateNode::update()");
        self.entity_query.update_archetypes(world);
        self.effect_query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        trace!("ParticleUpdateNode::run()");

        // Get the Entity containing the ViewEffectsEntity component used as container
        // for the input data for this node.
        //let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        let pipeline_cache = world.resource::<PipelineCache>();

        // Begin encoder
        // trace!(
        //     "begin compute update pass... (world={:?} ents={:?} comps={:?})",
        //     world,
        //     world.entities(),
        //     world.components()
        // );
        trace!("begin compute update pass...");
        render_context
            .command_encoder
            .push_debug_group("hanabi_update");

        // Compute update pass
        {
            let mut compute_pass =
                render_context
                    .command_encoder
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hanabi:update_compute_pass"),
                    });

            let effects_meta = world.resource::<EffectsMeta>();
            let effect_bind_groups = world.resource::<EffectBindGroups>();

            // Retrieve the ExtractedEffectEntities component itself
            //if let Ok(extracted_effect_entities) = self.entity_query.get_manual(world,
            // view_entity) if let Ok(effect_batches) =
            // self.effect_query.get_manual(world, )
            {
                // Loop on all entities recorded inside the ExtractedEffectEntities input
                trace!("loop over effect batches...");
                //for effect_entity in extracted_effect_entities.entities.iter().copied() {

                for batch in self.effect_query.iter_manual(world) {
                    if let Some(compute_pipeline) =
                        pipeline_cache.get_compute_pipeline(batch.compute_pipeline_id)
                    {
                        //for (effect_entity, effect_slice) in effects_meta.entity_map.iter() {
                        // Retrieve the ExtractedEffect from the entity
                        //trace!("effect_entity={:?} effect_slice={:?}", effect_entity,
                        // effect_slice); let effect =
                        // self.effect_query.get_manual(world, *effect_entity).unwrap();

                        // Get the slice to update
                        //let effect_slice = effects_meta.get(&effect_entity);
                        // let effect_group =
                        //     &effects_meta.effect_cache.buffers()[batch.buffer_index as usize];
                        let particles_bind_group = effect_bind_groups
                            .update_particle_buffers
                            .get(&batch.buffer_index)
                            .unwrap();

                        let item_size = batch.item_size;
                        let item_count = batch.slice.end - batch.slice.start;
                        let workgroup_count = (item_count + 63) / 64;

                        let spawner_base = batch.spawner_base;
                        let buffer_offset = batch.slice.start;

                        let spawner_buffer_aligned = effects_meta.spawner_buffer.aligned_size();
                        assert!(
                            spawner_buffer_aligned >= GpuSpawnerParams::min_size().get() as usize
                        );

                        trace!(
                            "record commands for pipeline of effect {:?} ({} items / {}B/item = {} workgroups) spawner_base={} buffer_offset={}...",
                            batch.handle,
                            item_count,
                            item_size,
                            workgroup_count,
                            spawner_base,
                            buffer_offset,
                        );

                        // Setup compute pass
                        //compute_pass.set_pipeline(&effect_group.compute_pipeline);
                        compute_pass.set_pipeline(compute_pipeline);
                        compute_pass.set_bind_group(
                            0,
                            effects_meta.sim_params_bind_group.as_ref().unwrap(),
                            &[],
                        );
                        compute_pass.set_bind_group(
                            1,
                            particles_bind_group,
                            &[buffer_offset, buffer_offset],
                        );
                        compute_pass.set_bind_group(
                            2,
                            effects_meta.spawner_bind_group.as_ref().unwrap(),
                            &[spawner_base * spawner_buffer_aligned as u32],
                        );
                        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                        trace!("compute dispatched");
                    }
                }
            }
        }

        // End encoder
        render_context.command_encoder.pop_debug_group();
        trace!("compute update pass done");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec4;

    #[test]
    fn layout_flags() {
        let flags = LayoutFlags::default();
        assert_eq!(flags, LayoutFlags::NONE);
    }
}
