#![allow(unused_imports)] // TEMP

use rand::Rng;
use std::{borrow::Cow, cmp::Ordering, num::NonZeroU64, ops::Range};

use crate::{asset::EffectAsset, ParticleEffect};

use bevy::{
    asset::{AssetEvent, Assets, Handle, HandleId, HandleUntyped},
    core::{cast_slice, FloatOrd, Pod, Time, Zeroable},
    core_pipeline::Transparent3d,
    ecs::{
        prelude::*,
        system::{lifetimeless::*, SystemState},
    },
    log::trace,
    math::{const_vec3, Mat4, Vec2, Vec3, Vec4Swizzles},
    reflect::TypeUuid,
    render::{
        color::Color,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_phase::{Draw, DrawFunctions, RenderPhase, TrackedRenderPass},
        render_resource::{std140::AsStd140, std430::AsStd430, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::{BevyDefault, Image},
        view::{ComputedVisibility, ExtractedView, ViewUniform, ViewUniformOffset, ViewUniforms},
        RenderWorld,
    },
    sprite::Rect,
    transform::components::GlobalTransform,
    utils::{HashMap, HashSet},
};
use bitflags::bitflags;
use bytemuck::cast_slice_mut;
use rand::random;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use crate::Gradient;

mod effect_cache;
mod pipeline_template;

pub use effect_cache::{EffectBuffer, EffectCache, EffectCacheId, EffectSlice};
pub use pipeline_template::PipelineRegistry;

pub const PARTICLES_UPDATE_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 2763343953151597126);

pub const PARTICLES_RENDER_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 2763343953151597145);

const PARTICLES_RENDER_SHADER_TEMPLATE: &'static str = include_str!("particles_render.wgsl");

/// Labels for the Hanabi systems.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum EffectSystems {
    /// Extract the effects to render this frame.
    ExtractEffects,
    /// Extract the effect events to process this frame.
    ExtractEffectEvents,
    /// Prepare GPU data for the extracted effects.
    PrepareEffects,
    /// Queue the GPU commands for the extracted effects.
    QueueEffects,
}

trait ToWgslFloat {
    fn to_float_string(&self) -> String;
}

impl ToWgslFloat for f32 {
    fn to_float_string(&self) -> String {
        let s = format!("{:.6}", self);
        s.trim_end_matches("0").to_string()
    }
}

trait ShaderCode {
    fn to_shader_code(&self) -> String;
}

impl ShaderCode for Gradient {
    fn to_shader_code(&self) -> String {
        if self.keys().len() == 0 {
            return String::new();
        }
        let mut s: String = self
            .keys()
            .iter()
            .enumerate()
            .map(|(index, key)| {
                format!(
                    "let t{0} = {1};\nlet c{0} = vec4<f32>({2}, {3}, {4}, {5});",
                    index,
                    key.ratio.to_float_string(),
                    key.color.x.to_float_string(),
                    key.color.y.to_float_string(),
                    key.color.z.to_float_string(),
                    key.color.w.to_float_string()
                )
            })
            .fold("// Gradient\n".into(), |s, key| s + &key + "\n");
        if self.keys().len() == 1 {
            s + "out.color = c0;\n"
        } else {
            s += "let life = particle.age / particle.lifetime;\nif (life <= t0) { out.color = c0; }\n";
            let mut s = self
                .keys()
                .iter()
                .skip(1)
                .enumerate()
                .map(|(index, _key)| {
                    format!(
                        "else if (life <= t{1}) {{ out.color = mix(c{0}, c{1}, (life - t{0}) / (t{1} - t{0})); }}\n",
                        index,
                        index + 1
                    )
                })
                .fold(s, |s, key| s + &key);
            s += &format!("else {{ out.color = c{}; }}\n", self.keys().len() - 1);
            s
        }
    }
}

/// Simulation parameters.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SimParams {
    /// Current simulation time.
    time: f64,
    /// Frame timestep.
    dt: f32,
}

/// GPU representation of [`SimParams`].
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, AsStd140)]
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

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable, AsStd430)]
struct SpawnerParams {
    /// Origin of the effect. This is either added to emitted particles at spawn time, if the effect simulated
    /// in world space, or to all simulated particles if the effect is simulated in local space.
    origin: Vec3,
    /// Number of particles to spawn this frame.
    spawn: i32,
    /// Global acceleration applied to all particles each frame.
    /// TODO - This is NOT a spawner/emitter thing, but is a per-effect one. Rename SpawnerParams?
    accel: Vec3,
    /// Current number of used particles.
    count: i32,
    ///
    __pad0: Vec3,
    /// Spawn seed, for randomized modifiers.
    seed: u32,
    ///
    __pad1: Vec3,
    ///
    __pad2: f32,
}

pub struct ParticlesUpdatePipeline {
    pipeline: ComputePipeline,
    sim_params_layout: BindGroupLayout,
    particles_buffer_layout: BindGroupLayout,
    spawner_buffer_layout: BindGroupLayout,
}

impl FromWorld for ParticlesUpdatePipeline {
    fn from_world(world: &mut World) -> Self {
        let world = world.cell();
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        trace!(
            "SimParamsUniform: min_binding_size = {}",
            SimParamsUniform::std140_size_static()
        );
        let sim_params_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(
                            SimParamsUniform::std140_size_static() as u64
                        ),
                    },
                    count: None,
                }],
                label: Some("particles_update_sim_params_layout"),
            });

        trace!(
            "Particle: min_binding_size = {}",
            Particle::std430_size_static()
        );
        let particles_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(Particle::std430_size_static() as u64),
                    },
                    count: None,
                }],
                label: Some("particles_update_particles_buffer_layout"),
            });

        trace!(
            "SpawnerParams: min_binding_size = {}",
            SpawnerParams::std430_size_static()
        );
        let spawner_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(
                            SpawnerParams::std430_size_static() as u64
                        ),
                    },
                    count: None,
                }],
                label: Some("particles_update_spawner_buffer_layout"),
            });

        let shader_module = render_device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("particles_update.wgsl"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("particles_update.wgsl"))),
        });

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("particles_update_pipeline_layout"),
            bind_group_layouts: &[
                &sim_params_layout,
                &particles_buffer_layout,
                &spawner_buffer_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = render_device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("particles_update_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });

        ParticlesUpdatePipeline {
            pipeline,
            sim_params_layout,
            particles_buffer_layout,
            spawner_buffer_layout,
        }
    }
}

pub struct ParticlesRenderPipeline {
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
                    min_binding_size: BufferSize::new(ViewUniform::std140_size_static() as u64),
                },
                count: None,
            }],
            label: Some("particles_view_layout_render"),
        });

        let particles_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(Particle::std430_size_static() as u64),
                    },
                    count: None,
                }],
                label: Some("particles_buffer_layout_render"),
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
            label: Some("particles_material_layout_render"),
        });

        ParticlesRenderPipeline {
            view_layout,
            particles_buffer_layout,
            material_layout,
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ParticlesPipelineKey {
    /// Render shader, with template applied, but not preprocessed yet.
    shader: Handle<Shader>,
    /// Key: PARTICLE_TEXTURE
    /// Define a texture sampled to modulate the particle color.
    /// This key requires the presence of UV coordinates on the particle vertices.
    particle_texture: Option<Handle<Image>>,
}

impl Default for ParticlesPipelineKey {
    fn default() -> Self {
        ParticlesPipelineKey {
            shader: PARTICLES_RENDER_SHADER_HANDLE.typed::<Shader>(),
            particle_texture: None,
        }
    }
}

impl SpecializedPipeline for ParticlesRenderPipeline {
    type Key = ParticlesPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // Base mandatory part of vertex buffer layout
        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: 20,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                // [[location(0)]] vertex_position: vec3<f32>
                VertexAttribute {
                    format: VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                // [[location(1)]] vertex_uv: vec2<f32>
                VertexAttribute {
                    format: VertexFormat::Float32x2,
                    offset: 12,
                    shader_location: 1,
                },
                // [[location(1)]] vertex_color: u32
                // VertexAttribute {
                //     format: VertexFormat::Uint32,
                //     offset: 12,
                //     shader_location: 1,
                // },
                // [[location(2)]] vertex_velocity: vec3<f32>
                // VertexAttribute {
                //     format: VertexFormat::Float32x3,
                //     offset: 12,
                //     shader_location: 1,
                // },
                // [[location(3)]] vertex_uv: vec2<f32>
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
            // // [[location(1)]] vertex_uv: vec2<f32>
            // vertex_buffer_layout.attributes.push(VertexAttribute {
            //     format: VertexFormat::Float32x2,
            //     offset: 12,
            //     shader_location: 1,
            // });
            // vertex_buffer_layout.array_stride += 8;
        }

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
                targets: vec![ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                }],
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
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: CompareFunction::Always,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: 4, // TODO: Res<Msaa>.samples
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            label: Some("particle_render_pipeline".into()),
        }
    }
}

/// A single effect instance extracted from a [`ParticleEffect`] as a [`RenderWorld`] item.
#[derive(Component)]
pub struct ExtractedEffect {
    /// Handle to the effect asset this instance is based on.
    /// The handle is weak to prevent refcount cycles and gracefully handle assets unloaded
    /// or destroyed after a draw call has been submitted.
    pub handle: Handle<EffectAsset>,
    /// Number of particles to spawn this frame for the effect.
    /// Obtained from calling [`Spawner::tick()`] on the source effect instance.
    pub spawn_count: u32,
    /// Global transform of the effect origin.
    pub transform: Mat4,
    /// Constant acceleration applied to all particles.
    pub accel: Vec3,
    /// Particles tint to modulate with the texture image.
    pub color: Color,
    pub rect: Rect,
    // Texture to use for the sprites of the particles of this effect.
    //pub image: Handle<Image>,
    pub has_image: bool, // TODO -> use flags
    /// Texture to modulate the particle color.
    pub image_handle_id: HandleId,
    /// Render shader.
    pub shader: Handle<Shader>,
}

/// Extracted data for newly-added ParticleEffect component requiring a new GPU allocation.
pub struct AddedEffect {
    /// Entity with a newly-added ParticleEffect component.
    pub entity: Entity,
    /// Capacity of the effect (and therefore, the particle buffer), in number of particles.
    pub capacity: u32,
    /// Size in bytes of each particle.
    pub item_size: u32,
    /// Handle of the effect asset.
    pub handle: Handle<EffectAsset>,
}

/// Collection of all extracted effects for this frame, inserted into the
/// [`RenderWorld`] as a render resource.
#[derive(Default)]
pub struct ExtractedEffects {
    /// Map of extracted effects from the entity the source [`ParticleEffect`] is on.
    pub effects: HashMap<Entity, ExtractedEffect>,
    /// Entites which had their ParticleEffect component removed.
    pub removed_effect_entities: Vec<Entity>,
    /// Newly added effects without a GPU allocation yet.
    pub added_effects: Vec<AddedEffect>,
}

#[derive(Default)]
pub struct EffectAssetEvents {
    pub images: Vec<AssetEvent<Image>>,
}

pub fn extract_effect_events(
    mut render_world: ResMut<RenderWorld>,
    mut image_events: EventReader<AssetEvent<Image>>,
) {
    trace!("extract_effect_events");
    let mut events = render_world
        .get_resource_mut::<EffectAssetEvents>()
        .unwrap();
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

/// System extracting data for rendering of all active [`ParticleEffect`] components.
///
/// Extract rendering data for all [`ParticleEffect`] components in the world which are
/// visible ([`ComputedVisibility::is_visible`] is `true`), and wrap the data into a new
/// [`ExtractedEffect`] instance added to the [`ExtractedEffects`] resource.
pub(crate) fn extract_effects(
    mut render_world: ResMut<RenderWorld>,
    time: Res<Time>,
    effects: Res<Assets<EffectAsset>>,
    _images: Res<Assets<Image>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut pipeline_registry: ResMut<PipelineRegistry>,
    mut query: QuerySet<(
        // All existing ParticleEffect components
        QueryState<(
            Entity,
            &ComputedVisibility,
            &mut ParticleEffect, //TODO - Split EffectAsset::Spawner (desc) and ParticleEffect::SpawnerData (runtime data), and init the latter on component add without a need for the former
            &GlobalTransform,
        )>,
        // Newly added ParticleEffect components
        QueryState<
            (Entity, &mut ParticleEffect),
            (
                Added<ParticleEffect>,
                With<ComputedVisibility>,
                With<GlobalTransform>,
            ),
        >,
    )>,
    removed_effects: RemovedComponents<ParticleEffect>,
) {
    trace!("extract_effects");

    // Save simulation params into render world
    let mut sim_params = render_world.get_resource_mut::<SimParams>().unwrap();
    let dt = time.delta_seconds();
    sim_params.time = time.seconds_since_startup();
    sim_params.dt = dt;

    let mut extracted_effects = render_world.get_resource_mut::<ExtractedEffects>().unwrap();

    // Collect removed effects for later GPU data purge
    extracted_effects.removed_effect_entities = removed_effects.iter().collect();

    // Collect added effects for later GPU data allocation
    extracted_effects.added_effects = query
        .q1()
        .iter()
        .map(|(entity, effect)| {
            let handle = effect.handle.clone_weak();
            let asset = effects.get(&effect.handle).unwrap();
            AddedEffect {
                entity,
                capacity: asset.capacity,
                item_size: Particle::std430_size_static() as u32, // effect.item_size(),
                handle,
            }
        })
        .collect();

    // Loop over all existing effects to update them
    for (entity, computed_visibility, mut effect, transform) in query.q0().iter_mut() {
        // Check if visible
        if !computed_visibility.is_visible {
            continue;
        }

        // Check if asset is available, otherwise silently ignore
        if let Some(asset) = effects.get(&effect.handle) {
            //let size = image.texture_descriptor.size;

            // Tick the effect's spawner to determine the spawn count for this frame
            let spawner = effect.spawner(&asset.spawner);
            let spawn_count = spawner.tick(dt);

            // Extract the acceleration
            let accel = asset.update_layout.accel;

            // Generate the shader code for the color over lifetime gradient.
            // TODO - Move that to a pre-pass, not each frame!
            let vertex_modifiers = if let Some(grad) = &asset.render_layout.lifetime_color_gradient
            {
                grad.to_shader_code()
            } else {
                String::new()
            };

            // Configure the shader template, and make sure a corresponding shader asset exists
            let shader_source =
                PARTICLES_RENDER_SHADER_TEMPLATE.replace("{{VERTEX_MODIFIERS}}", &vertex_modifiers);
            let shader = pipeline_registry.configure(&shader_source, &mut shaders);

            trace!(
                "extracted: handle={:?} shader={:?} has_image={}",
                effect.handle,
                shader,
                if asset.render_layout.particle_texture.is_some() {
                    "Y"
                } else {
                    "N"
                }
            );

            extracted_effects.effects.insert(
                entity,
                ExtractedEffect {
                    handle: effect.handle.clone_weak(),
                    spawn_count,
                    color: Color::RED, //effect.color,
                    transform: transform.compute_matrix(),
                    accel,
                    rect: Rect {
                        min: Vec2::ZERO,
                        max: Vec2::new(0.2, 0.2), // effect
                                                  //.custom_size
                                                  //.unwrap_or_else(|| Vec2::new(size.width as f32, size.height as f32)),
                    },
                    has_image: asset.render_layout.particle_texture.is_some(),
                    image_handle_id: asset
                        .render_layout
                        .particle_texture
                        .clone()
                        .map_or(HandleId::default::<Image>(), |handle| handle.id),
                    shader,
                },
            );
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, AsStd430)]
struct Particle {
    pub position: [f32; 3],
    pub age: f32,
    pub velocity: [f32; 3],
    pub pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ParticleVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

/// Global resource containing the GPU data to draw all the particle effects.
///
/// The resource is populated by [`prepare_effects()`] with all the effects to render
/// for the current frame, and consumed by [`queue_effects()`] to actually enqueue the
/// drawning commands to draw those effects.
pub(crate) struct EffectsMeta {
    entity_map: HashMap<Entity, EffectSlice>,
    effect_cache: EffectCache,
    /// Bind group for the camera view, containing the camera projection and other uniform
    /// values related to the camera.
    view_bind_group: Option<BindGroup>,
    sim_params_bind_group: Option<BindGroup>,
    particles_bind_group: Option<BindGroup>,
    spawner_bind_group: Option<BindGroup>,
    sim_params_uniforms: UniformVec<SimParamsUniform>,
    spawner_buffer: BufferVec<SpawnerParams>,
    /// TEMP
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

        Self {
            entity_map: HashMap::default(),
            effect_cache: EffectCache::new(device),
            view_bind_group: None,
            sim_params_bind_group: None,
            particles_bind_group: None,
            spawner_bind_group: None,
            sim_params_uniforms: UniformVec::default(),
            spawner_buffer: BufferVec::new(BufferUsages::STORAGE),
            vertices,
        }
    }
}

const QUAD_VERTEX_POSITIONS: &[Vec3] = &[
    const_vec3!([-0.5, -0.5, 0.0]),
    const_vec3!([0.5, 0.5, 0.0]),
    const_vec3!([-0.5, 0.5, 0.0]),
    const_vec3!([-0.5, -0.5, 0.0]),
    const_vec3!([0.5, -0.5, 0.0]),
    const_vec3!([0.5, 0.5, 0.0]),
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

/// A batch of multiple instances of the same effect, rendered all together to reduce GPU shader
/// permutations and draw call overhead.
#[derive(Component)]
pub struct EffectBatch {
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
    /// Render shader.
    shader: Handle<Shader>,
}

pub(crate) fn prepare_effects(
    mut commands: Commands,
    sim_params: Res<SimParams>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    update_pipeline: Res<ParticlesUpdatePipeline>, // TODO move update_pipeline.pipeline to EffectsMeta
    mut effects_meta: ResMut<EffectsMeta>,
    mut extracted_effects: ResMut<ExtractedEffects>,
) {
    trace!("prepare_effects");

    // Allocate simulation uniform if needed
    if effects_meta.sim_params_uniforms.is_empty() {
        effects_meta
            .sim_params_uniforms
            .push(SimParamsUniform::default());
    }

    // Update simulation parameters
    {
        let sim_params_uni = effects_meta.sim_params_uniforms.get_mut(0);
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
    //    effects_meta.spawner_buffer.push(SpawnerParams::default());
    //}

    // Write vertices (TODO - lazily once only)
    effects_meta
        .vertices
        .write_buffer(&render_device, &render_queue);

    // Allocate GPU data for newly created effect instances. Do this first to ensure a group is not left
    // unused and dropped due to the last effect being removed but a new compatible one added not being
    // inserted yet. By inserting first, we ensure the group is not dropped in this case.
    for added_effect in extracted_effects.added_effects.drain(..) {
        let entity = added_effect.entity;
        let id = effects_meta.effect_cache.insert(
            added_effect.handle,
            added_effect.capacity,
            added_effect.item_size,
            update_pipeline.pipeline.clone(),
            &render_queue,
        );
        let slice = effects_meta.effect_cache.get_slice(id);
        effects_meta.entity_map.insert(entity, slice);
    }

    // Upload modified groups to GPU (either completely new group/buffer, or existing one with one or more
    // newly added slices into it).
    effects_meta.effect_cache.flush(&render_queue);

    // Deallocate GPU data for destroyed effect instances. This will automatically drop any group where
    // there is no more effect slice.
    for _entity in extracted_effects.removed_effect_entities.iter() {
        unimplemented!("Remove particle effect.");
        //effects_meta.remove(&*entity);
    }

    // // sort first by z and then by handle. this ensures that, when possible, batches span multiple z layers
    // // batches won't span z-layers if there is another batch between them
    // extracted_effects.effects.sort_by(|a, b| {
    //     match FloatOrd(a.transform.w_axis[2]).cmp(&FloatOrd(b.transform.w_axis[2])) {
    //         Ordering::Equal => a.handle.cmp(&b.handle),
    //         other => other,
    //     }
    // });

    // Get the effect-entity mapping
    let mut effect_entity_list = extracted_effects
        .effects
        .iter()
        .map(|(entity, extracted_effect)| {
            let slice = effects_meta.entity_map.get(&entity).unwrap().clone();
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
    let mut shader: Handle<Shader> = Default::default();
    let mut start = 0;
    let mut end = 0;
    let mut num_emitted = 0;
    for (slice, extracted_effect) in effect_entity_list {
        let buffer_index = slice.group_index;
        let range = slice.slice;
        layout_flags = if extracted_effect.has_image {
            LayoutFlags::PARTICLE_TEXTURE
        } else {
            LayoutFlags::NONE
        };
        image_handle_id = extracted_effect.image_handle_id;
        trace!("Effect: buffer #{} | range {:?}", buffer_index, range);

        // Check the buffer the effect is in
        assert!(buffer_index >= current_buffer_index || current_buffer_index == u32::MAX);
        if current_buffer_index != buffer_index {
            trace!(
                "+ New buffer! ({} -> {})",
                current_buffer_index,
                buffer_index
            );
            // Commit previous buffer if any
            if current_buffer_index != u32::MAX {
                // Record open batch if any
                trace!("+ Prev: {} - {}", start, end);
                if end > start {
                    assert_ne!(asset, Handle::<EffectAsset>::default());
                    assert!(item_size > 0);
                    trace!(
                        "Emit batch: buffer #{} | spawner_base {} | slice {:?} | item_size {} | shader {:?}",
                        current_buffer_index,
                        spawner_base,
                        start..end,
                        item_size,
                        shader
                    );
                    commands.spawn_bundle((EffectBatch {
                        buffer_index: current_buffer_index,
                        spawner_base: spawner_base as u32,
                        slice: start..end,
                        item_size,
                        handle: asset.clone_weak(),
                        layout_flags,
                        image_handle_id,
                        shader: shader.clone(),
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
        }

        assert_ne!(asset, Handle::<EffectAsset>::default());

        shader = extracted_effect.shader.clone();
        trace!("shader = {:?}", shader);

        trace!("item_size = {}B", slice.item_size);

        // Prepare the spawner block for the current slice
        // FIXME - This is once per EFFECT/SLICE, not once per BATCH, so indeed this is spawner_BASE, and need an array of them in the compute shader!!!!!!!!!!!!!!
        let spawner_params = SpawnerParams {
            spawn: extracted_effect.spawn_count as i32,
            count: 0,
            origin: extracted_effect.transform.col(3).truncate(),
            accel: extracted_effect.accel,
            seed: random::<u32>(),
            ..Default::default()
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
                    "Emit batch: buffer #{} | spawner_base {} | slice {:?} | item_size {} | shader {:?}",
                    buffer_index,
                    spawner_base,
                    start..end,
                    item_size,
                    shader
                );
                commands.spawn_bundle((EffectBatch {
                    buffer_index,
                    spawner_base: spawner_base as u32,
                    slice: start..end,
                    item_size,
                    handle: asset.clone_weak(),
                    layout_flags,
                    image_handle_id,
                    shader: shader.clone(),
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
            "Emit LAST batch: buffer #{} | spawner_base {} | slice {:?} | item_size {} | shader {:?}",
            current_buffer_index,
            spawner_base,
            start..end,
            item_size,
            shader
        );
        commands.spawn_bundle((EffectBatch {
            buffer_index: current_buffer_index,
            spawner_base: spawner_base as u32,
            slice: start..end,
            item_size,
            handle: asset.clone_weak(),
            layout_flags,
            image_handle_id,
            shader: shader.clone(),
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

#[derive(Default)]
pub struct ImageBindGroups {
    values: HashMap<Handle<Image>, BindGroup>,
}

#[derive(Default)]
pub struct EffectBindGroups {
    /// Bind groups for each group index for compute shader.
    values: HashMap<u32, BindGroup>,
    /// Same for render shader.
    render_values: HashMap<u32, BindGroup>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn queue_effects(
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    render_device: Res<RenderDevice>,
    mut effects_meta: ResMut<EffectsMeta>,
    view_uniforms: Res<ViewUniforms>,
    update_pipeline: Res<ParticlesUpdatePipeline>,
    render_pipeline: Res<ParticlesRenderPipeline>,
    mut specialized_render_pipelines: ResMut<SpecializedPipelines<ParticlesRenderPipeline>>,
    mut render_pipeline_cache: ResMut<RenderPipelineCache>,
    mut image_bind_groups: ResMut<ImageBindGroups>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    gpu_images: Res<RenderAssets<Image>>,
    effect_batches: Query<(Entity, &EffectBatch)>,
    mut views: Query<&mut RenderPhase<Transparent3d>>,
    events: Res<EffectAssetEvents>,
) {
    trace!("queue_effects");

    // If an image has changed, the GpuImage has (probably) changed
    for event in &events.images {
        match event {
            AssetEvent::Created { .. } => None,
            AssetEvent::Modified { handle } => image_bind_groups.values.remove(handle),
            AssetEvent::Removed { handle } => image_bind_groups.values.remove(handle),
        };
    }

    // Get the binding for the ViewUniform, the uniform data structure containing the Camera data
    // for the current view.
    let view_binding = match view_uniforms.uniforms.binding() {
        Some(view_binding) => view_binding,
        None => {
            return;
        }
    };

    // Create the bind group for global the simulation parameters
    effects_meta.sim_params_bind_group =
        Some(render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[BindGroupEntry {
                binding: 0,
                resource: effects_meta.sim_params_uniforms.binding().unwrap(),
            }],
            label: Some("particles_sim_params_bind_group"),
            layout: &update_pipeline.sim_params_layout,
        }));

    // Create the bind group for the spawner
    effects_meta.spawner_bind_group = Some(render_device.create_bind_group(&BindGroupDescriptor {
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: effects_meta.spawner_buffer.buffer().unwrap(),
                offset: 0,
                size: Some(NonZeroU64::new(SpawnerParams::std430_size_static() as u64).unwrap()),
            }),
        }],
        label: Some("particles_spawner_bind_group"),
        layout: &update_pipeline.spawner_buffer_layout,
    }));

    // Queue the update compute
    trace!("queue effects from cache...");
    for (buffer_index, buffer) in effects_meta.effect_cache.buffers().iter().enumerate() {
        // Ensure all effect groups have a binding for the entire buffer of the group,
        // since the update phase runs on an entire group/buffer at once, with all the
        // effect instances in it batched together.
        trace!("effect #{}", buffer_index);
        effect_bind_groups
            .values
            .entry(buffer_index as u32)
            .or_insert_with(|| {
                trace!(
                    "Create new update bind group for buffer_index={}",
                    buffer_index
                );
                render_device.create_bind_group(&BindGroupDescriptor {
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buffer.binding(32768 * 32),
                    }],
                    label: Some(&format!(
                        "particles_particles_bind_group_compute{}",
                        buffer_index
                    )),
                    layout: &update_pipeline.particles_buffer_layout,
                })
            });

        effect_bind_groups
            .render_values
            .entry(buffer_index as u32)
            .or_insert_with(|| {
                trace!(
                    "Create new render bind group for buffer_index={}",
                    buffer_index
                );
                render_device.create_bind_group(&BindGroupDescriptor {
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buffer.binding(32768 * 32),
                    }],
                    label: Some(&format!(
                        "particles_particles_bind_group_render{}",
                        buffer_index
                    )),
                    layout: &render_pipeline.particles_buffer_layout,
                })
            });
    }

    // Queue the rendering
    effects_meta.view_bind_group = Some(render_device.create_bind_group(&BindGroupDescriptor {
        entries: &[BindGroupEntry {
            binding: 0,
            resource: view_binding,
        }],
        label: Some("particles_view_bind_group"),
        layout: &render_pipeline.view_layout,
    }));
    let draw_effects_function = draw_functions.read().get_id::<DrawEffects>().unwrap();
    for mut transparent_phase in views.iter_mut() {
        for (entity, batch) in effect_batches.iter() {
            trace!(
                "Process batch entity={:?} buffer_index={} spawner_base={} slice={:?}",
                entity,
                batch.buffer_index,
                batch.spawner_base,
                batch.slice
            );
            // Ensure the particle texture is available as a GPU resource and create a bind group for it
            let particle_texture = if batch.layout_flags.contains(LayoutFlags::PARTICLE_TEXTURE) {
                let image_handle = Handle::weak(batch.image_handle_id);
                if image_bind_groups.values.get(&image_handle).is_none() {
                    trace!(
                        "Batch buffer #{} slice={:?} has missing GPU image bind group, creating...",
                        batch.buffer_index,
                        batch.slice
                    );
                    // If texture doesn't have a bind group yet from another instance of the same effect,
                    // then try to create one now
                    if let Some(gpu_image) = gpu_images.get(&image_handle) {
                        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
                            entries: &[
                                BindGroupEntry {
                                    binding: 0,
                                    resource: BindingResource::TextureView(&gpu_image.texture_view),
                                },
                                BindGroupEntry {
                                    binding: 1,
                                    resource: BindingResource::Sampler(&gpu_image.sampler),
                                },
                            ],
                            label: Some("particles_material_bind_group"),
                            layout: &render_pipeline.material_layout,
                        });
                        image_bind_groups
                            .values
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
                "Specializing render pipeline: shader={:?} particle_texture={:?}",
                batch.shader,
                particle_texture
            );
            let render_pipeline_id = specialized_render_pipelines.specialize(
                &mut render_pipeline_cache,
                &render_pipeline,
                ParticlesPipelineKey {
                    particle_texture,
                    shader: batch.shader.clone(),
                },
            );
            trace!("Render pipeline specialized: id={:?}", render_pipeline_id);

            // Add a draw pass for the effect batch
            trace!("Add Transparent3d for batch on entity {:?}: buffer_index={} spawner_base={} slice={:?} handle={:?}", entity, batch.buffer_index, batch.spawner_base, batch.slice, batch.handle);
            transparent_phase.add(Transparent3d {
                draw_function: draw_effects_function,
                pipeline: render_pipeline_id,
                entity,
                distance: 0.0, // TODO ??????
            });
        }
        // for (_buffer_index, _buffer) in effects_meta.effect_cache.buffers().iter().enumerate() {
        //     let entity = *effects_meta.entity_map.keys().next().unwrap(); // TODO

        //     //for (entity, batch) in effect_batches.iter_mut() {
        //     // image_bind_groups
        //     //     .values
        //     //     .entry(batch.image.clone_weak())
        //     //     .or_insert_with(|| {
        //     //         let gpu_image = gpu_images.get(&batch.image).unwrap();
        //     //         render_device.create_bind_group(&BindGroupDescriptor {
        //     //             entries: &[
        //     //                 BindGroupEntry {
        //     //                     binding: 0,
        //     //                     resource: BindingResource::TextureView(&gpu_image.texture_view),
        //     //                 },
        //     //                 BindGroupEntry {
        //     //                     binding: 1,
        //     //                     resource: BindingResource::Sampler(&gpu_image.sampler),
        //     //                 },
        //     //             ],
        //     //             label: Some("particles_material_bind_group"),
        //     //             layout: &render_pipeline.material_layout,
        //     //         })
        //     //     });

        //     transparent_phase.add(Transparent3d {
        //         draw_function: draw_effects_function,
        //         pipeline: render_pipeline_id,
        //         entity,
        //         distance: 0.0, // TODO ??????
        //     });
        // }
    }
}

/// Draw function for rendering all active effects for the current frame.
///
/// Effects are rendered in the [`Transparent3d`] phase of the main 3D pass.
pub struct DrawEffects {
    params: SystemState<(
        SRes<EffectsMeta>,
        SRes<ImageBindGroups>,
        SRes<EffectBindGroups>,
        SRes<RenderPipelineCache>,
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

/// Component to hold all the entities with a [`ExtractedEffect`] component on them
/// that need to be updated this frame with a compute pass. This is view-independent
/// because the update phase itself is also view-independent (effects like camera
/// facing are applied in the render phase, which runs once per view).
#[derive(Component)]
pub struct ExtractedEffectEntities {
    pub entities: Vec<Entity>,
}

impl Draw<Transparent3d> for DrawEffects {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        item: &Transparent3d,
    ) {
        trace!("Draw<Transparent3d>: view={:?}", view);
        let (
            effects_meta,
            image_bind_groups,
            effect_bind_groups,
            specialized_render_pipelines,
            views,
            effects,
        ) = self.params.get(world);
        let view_uniform = views.get(view).unwrap();
        let effects_meta = effects_meta.into_inner();
        let image_bind_groups = image_bind_groups.into_inner();
        let effect_bind_groups = effect_bind_groups.into_inner();
        let effect_batch = effects.get(item.entity).unwrap();
        if let Some(pipeline) = specialized_render_pipelines.into_inner().get(item.pipeline) {
            trace!("render pass");
            //let effect_group = &effects_meta.effect_cache.buffers()[0]; // TODO

            pass.set_render_pipeline(pipeline);
            pass.set_vertex_buffer(0, effects_meta.vertices.buffer().unwrap().slice(..));
            pass.set_bind_group(
                0,
                effects_meta.view_bind_group.as_ref().unwrap(),
                &[view_uniform.offset],
            );
            pass.set_bind_group(
                1,
                effect_bind_groups
                    .render_values
                    .get(&effect_batch.buffer_index)
                    .unwrap(),
                &[],
            );
            if effect_batch
                .layout_flags
                .contains(LayoutFlags::PARTICLE_TEXTURE)
            {
                let image_handle = Handle::weak(effect_batch.image_handle_id);
                if let Some(bind_group) = image_bind_groups.values.get(&image_handle) {
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

            let count = effect_batch.slice.end - effect_batch.slice.start;

            trace!(
                "Draw {} particles for batch from buffer #{}.",
                count,
                effect_batch.buffer_index
            );
            pass.draw(0..6, 0..count); // TODO vertex count
        }
    }
}

/// A render node to update the particles of all particle efects.
pub struct ParticleUpdateNode {
    /// Query to retrieve the list of entities holding an extracted particle effect to update.
    entity_query: QueryState<&'static ExtractedEffectEntities>,
    /// Query to retrieve the
    effect_query: QueryState<&'static EffectBatch>,
}

impl ParticleUpdateNode {
    /// Input entity marking the view.
    pub const IN_VIEW: &'static str = "view";
    /// Output particle buffer for that view. TODO - how to handle multiple buffers?! Should use Entity instead??
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
                        label: Some("update_compute_pass"),
                    });

            let effects_meta = world.get_resource::<EffectsMeta>().unwrap();
            let effect_bind_groups = world.get_resource::<EffectBindGroups>().unwrap();

            // Retrieve the ExtractedEffectEntities component itself
            //if let Ok(extracted_effect_entities) = self.entity_query.get_manual(world, view_entity)
            //if let Ok(effect_batches) = self.effect_query.get_manual(world, )
            {
                // Loop on all entities recorded inside the ExtractedEffectEntities input
                trace!("loop over effect batches...");
                //for effect_entity in extracted_effect_entities.entities.iter().copied() {

                for batch in self.effect_query.iter_manual(world) {
                    //for (effect_entity, effect_slice) in effects_meta.entity_map.iter() {
                    // Retrieve the ExtractedEffect from the entity
                    //trace!("effect_entity={:?} effect_slice={:?}", effect_entity, effect_slice);
                    //let effect = self.effect_query.get_manual(world, *effect_entity).unwrap();

                    // Get the slice to update
                    //let effect_slice = effects_meta.get(&effect_entity);
                    let effect_group =
                        &effects_meta.effect_cache.buffers()[batch.buffer_index as usize];
                    let particles_bind_group =
                        effect_bind_groups.values.get(&batch.buffer_index).unwrap();

                    let item_size = batch.item_size;
                    let item_count = (batch.slice.end - batch.slice.start) / item_size;
                    let workgroup_count = item_count / 64;

                    let spawner_base = batch.spawner_base;
                    let buffer_offset = batch.slice.start / item_size;

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
                    compute_pass.set_pipeline(&effect_group.compute_pipeline);
                    compute_pass.set_bind_group(
                        0,
                        effects_meta.sim_params_bind_group.as_ref().unwrap(),
                        &[],
                    );
                    compute_pass.set_bind_group(1, particles_bind_group, &[buffer_offset]);
                    compute_pass.set_bind_group(
                        2,
                        effects_meta.spawner_bind_group.as_ref().unwrap(),
                        &[spawner_base * SpawnerParams::std430_size_static() as u32],
                    );
                    compute_pass.dispatch(workgroup_count, 1, 1);
                    trace!("compute dispatched");
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

    #[test]
    fn to_shader_code() {
        let mut grad = Gradient::new();
        assert_eq!("", grad.to_shader_code());

        grad.add_key(0.0, Vec4::splat(0.0));
        assert_eq!(
            "let t0 = 0;\nlet c0 = vec4<f32>(0, 0, 0, 0);\nout.color = c0;\n",
            grad.to_shader_code()
        );

        grad.add_key(1.0, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(
            r#"let t0 = 0;
let c0 = vec4<f32>(0, 0, 0, 0);
let t1 = 1;
let c1 = vec4<f32>(1, 0, 0, 1);
let life = particle.age / particle.lifetime;
if (life <= t0) { out.color = c0; }
else if (life <= t1) { out.color = mix(c0, c1, (life - t0) / (t1 - t0)); }
else { out.color = c1; }
"#,
            grad.to_shader_code()
        );
    }
}
