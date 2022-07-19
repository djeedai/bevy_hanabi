#![allow(unused_imports)] // TEMP

use bevy::{
    asset::{AssetEvent, Assets, Handle, HandleId, HandleUntyped},
    core::{cast_slice, FloatOrd, Pod, Time, Zeroable},
    ecs::{
        prelude::*,
        system::{lifetimeless::*, SystemState},
    },
    log::trace,
    math::{const_vec3, Mat4, Rect, Vec2, Vec3, Vec4, Vec4Swizzles},
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
    transform::components::GlobalTransform,
    utils::{HashMap, HashSet},
};
use bitflags::bitflags;
use bytemuck::cast_slice_mut;
use rand::random;
use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::{borrow::Cow, cmp::Ordering, num::NonZeroU64, ops::Range};

#[cfg(feature = "2d")]
use bevy::core_pipeline::Transparent2d;
#[cfg(feature = "3d")]
use bevy::core_pipeline::Transparent3d;

use crate::{
    asset::EffectAsset,
    modifiers::{ForceFieldParam, FFNUM},
    spawn::{new_rng, Random},
    Gradient, ParticleEffect, ToWgslString,
};

mod aligned_buffer_vec;
mod compute_cache;
mod effect_cache;
mod pipeline_template;

use aligned_buffer_vec::AlignedBufferVec;

pub use compute_cache::{ComputeCache, SpecializedComputePipeline};
pub use effect_cache::{EffectBuffer, EffectCache, EffectCacheId, EffectSlice};
pub use pipeline_template::PipelineRegistry;

pub const PARTICLES_UPDATE_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 2763343953151597126);

pub const PARTICLES_RENDER_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 2763343953151597145);

const PARTICLES_UPDATE_SHADER_TEMPLATE: &str = include_str!("particles_update.wgsl");
const PARTICLES_RENDER_SHADER_TEMPLATE: &str = include_str!("particles_render.wgsl");

const DEFAULT_POSITION_CODE: &str = r##"
    ret.pos = vec3<f32>(0., 0., 0.);
    var dir = rand3() * 2. - 1.;
    dir = normalize(dir);
    var speed = 2.;
    ret.vel = dir * speed;
"##;

const DEFAULT_LIFETIME_CODE: &str = r##"
ret = 5.0;
"##;

const DEFAULT_FORCE_FIELD_CODE: &str = r##"
    vVel = vVel + (spawner.accel * sim_params.dt);
    vPos = vPos + vVel * sim_params.dt;
"##;

const FORCE_FIELD_CODE: &str = include_str!("force_field_code.wgsl");

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

/// Trait to convert any data structure to its equivalent shader code.
trait ShaderCode {
    /// Generate the shader code for the current state of the object.
    fn to_shader_code(&self) -> String;
}

impl ShaderCode for Gradient<Vec2> {
    fn to_shader_code(&self) -> String {
        if self.keys().is_empty() {
            return String::new();
        }
        let mut s: String = self
            .keys()
            .iter()
            .enumerate()
            .map(|(index, key)| {
                format!(
                    "let t{0} = {1};\nlet v{0} = {2};",
                    index,
                    key.ratio().to_wgsl_string(),
                    key.value.to_wgsl_string()
                )
            })
            .fold("// Gradient\n".into(), |s, key| s + &key + "\n");
        if self.keys().len() == 1 {
            s + "size = v0;\n"
        } else {
            // FIXME - particle.age and particle.lifetime are unrelated to Gradient<Vec4>
            s += "let life = particle.age / particle.lifetime;\nif (life <= t0) { size = v0; }\n";
            let mut s = self
                .keys()
                .iter()
                .skip(1)
                .enumerate()
                .map(|(index, _key)| {
                    format!(
                        "else if (life <= t{1}) {{ size = mix(v{0}, v{1}, (life - t{0}) / (t{1} - t{0})); }}\n",
                        index,
                        index + 1
                    )
                })
                .fold(s, |s, key| s + &key);
            s += &format!("else {{ size = v{}; }}\n", self.keys().len() - 1);
            s
        }
    }
}

impl ShaderCode for Gradient<Vec4> {
    fn to_shader_code(&self) -> String {
        if self.keys().is_empty() {
            return String::new();
        }
        let mut s: String = self
            .keys()
            .iter()
            .enumerate()
            .map(|(index, key)| {
                format!(
                    "let t{0} = {1};\nlet c{0} = {2};",
                    index,
                    key.ratio().to_wgsl_string(),
                    key.value.to_wgsl_string()
                )
            })
            .fold("// Gradient\n".into(), |s, key| s + &key + "\n");
        if self.keys().len() == 1 {
            s + "out.color = c0;\n"
        } else {
            // FIXME - particle.age and particle.lifetime are unrelated to Gradient<Vec4>
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
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, AsStd430)]
pub struct ForceFieldStd430 {
    pub position_or_direction: Vec3,
    pub max_radius: f32,
    pub min_radius: f32,
    pub mass: f32,
    pub force_exponent: f32,
    pub conform_to_sphere: f32,
}

impl From<ForceFieldParam> for ForceFieldStd430 {
    fn from(param: ForceFieldParam) -> Self {
        ForceFieldStd430 {
            position_or_direction: param.position,
            max_radius: param.max_radius,
            min_radius: param.min_radius,
            mass: param.mass,
            force_exponent: param.force_exponent,
            conform_to_sphere: if param.conform_to_sphere { 1.0 } else { 0.0 },
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

    /// Force field components. One PullingForceFieldParam takes up 32 bytes.
    force_field: [ForceFieldStd430; FFNUM],
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
    sim_params_layout: BindGroupLayout,
    particles_buffer_layout: BindGroupLayout,
    spawner_buffer_layout: BindGroupLayout,
    indirect_buffer_layout: BindGroupLayout,
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
            "SimParamsUniform: std140_size_static = {}",
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
            "Particle: std430_size_static = {}",
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
            "SpawnerParams: std430_size_static = {}",
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

        let indirect_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(std::mem::size_of::<u32>() as u64),
                    },
                    count: None,
                }],
                label: Some("particles_update_indirect_buffer_layout"),
            });

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("particles_update_pipeline_layout"),
            bind_group_layouts: &[
                &sim_params_layout,
                &particles_buffer_layout,
                &spawner_buffer_layout,
                &indirect_buffer_layout,
            ],
            push_constant_ranges: &[],
        });

        ParticlesUpdatePipeline {
            sim_params_layout,
            particles_buffer_layout,
            spawner_buffer_layout,
            indirect_buffer_layout,
            pipeline_layout,
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

#[derive(Default, Clone, Hash, PartialEq, Eq)]
pub struct ParticleUpdatePipelineKey {
    /// Code for the position initialization of newly emitted particles.
    position_code: String,
    force_field_code: String,
    lifetime_code: String,
}

impl SpecializedComputePipeline for ParticlesUpdatePipeline {
    type Key = ParticleUpdatePipelineKey;

    fn specialize(&self, key: Self::Key, render_device: &RenderDevice) -> ComputePipeline {
        let mut source =
            PARTICLES_UPDATE_SHADER_TEMPLATE.replace("{{INIT_POS_VEL}}", &key.position_code);

        source = source.replace("{{FORCE_FIELD_CODE}}", &key.force_field_code);

        source = source.replace("{{INIT_LIFETIME}}", &key.lifetime_code);

        //trace!("Specialized compute pipeline:\n{}", source);

        let shader_module = render_device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("particles_update.wgsl"),
            source: ShaderSource::Wgsl(Cow::Owned(source)),
        });

        render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("particles_update_compute_pipeline"),
            layout: Some(&self.pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        })
    }
}

#[cfg(all(feature = "2d", feature = "3d"))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum PipelineMode {
    Camera2d,
    Camera3d,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ParticleRenderPipelineKey {
    /// Render shader, with template applied, but not preprocessed yet.
    shader: Handle<Shader>,
    /// Key: PARTICLE_TEXTURE
    /// Define a texture sampled to modulate the particle color.
    /// This key requires the presence of UV coordinates on the particle vertices.
    particle_texture: Option<Handle<Image>>,
    /// For dual-mode configurations only, the actual mode of the current render
    /// pipeline. Otherwise the mode is implicitly determined by the active feature.
    #[cfg(all(feature = "2d", feature = "3d"))]
    pipeline_mode: PipelineMode,
}

impl Default for ParticleRenderPipelineKey {
    fn default() -> Self {
        ParticleRenderPipelineKey {
            shader: PARTICLES_RENDER_SHADER_HANDLE.typed::<Shader>(),
            particle_texture: None,
            #[cfg(all(feature = "2d", feature = "3d"))]
            pipeline_mode: PipelineMode::Camera3d,
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
            depth_stencil,
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
    /// Force field applied to all particles in the "update" phase.
    force_field: [ForceFieldParam; FFNUM],
    /// Particles tint to modulate with the texture image.
    pub color: Color,
    pub rect: Rect<f32>,
    // Texture to use for the sprites of the particles of this effect.
    //pub image: Handle<Image>,
    pub has_image: bool, // TODO -> use flags
    /// Texture to modulate the particle color.
    pub image_handle_id: HandleId,
    /// Render shader.
    pub shader: Handle<Shader>,
    /// Update position code.
    pub position_code: String,
    /// Update force field code.
    pub force_field_code: String,
    /// Update lifetime code.
    pub lifetime_code: String,
}

/// Extracted data for newly-added [`ParticleEffect`] component requiring a new GPU allocation.
pub struct AddedEffect {
    /// Entity with a newly-added [`ParticleEffect`] component.
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
    /// Entites which had their [`ParticleEffect`] component removed.
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
    mut rng: ResMut<Random>,
    mut query: ParamSet<(
        // All existing ParticleEffect components
        Query<(
            Entity,
            &ComputedVisibility,
            &mut ParticleEffect, //TODO - Split EffectAsset::Spawner (desc) and ParticleEffect::SpawnerData (runtime data), and init the latter on component add without a need for the former
            &GlobalTransform,
        )>,
        // Newly added ParticleEffect components
        Query<
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
        .p1()
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
    for (entity, computed_visibility, mut effect, transform) in query.p0().iter_mut() {
        // Check if visible
        if !computed_visibility.is_visible {
            continue;
        }

        // Check if asset is available, otherwise silently ignore
        if let Some(asset) = effects.get(&effect.handle) {
            //let size = image.texture_descriptor.size;

            // Tick the effect's spawner to determine the spawn count for this frame
            let spawner = effect.spawner(&asset.spawner);

            let spawn_count = spawner.tick(dt, &mut rng.0);

            // Extract the acceleration
            let accel = asset.update_layout.accel;
            let force_field = asset.update_layout.force_field;

            // Generate the shader code for the position initializing of newly emitted particles
            // TODO - Move that to a pre-pass, not each frame!
            let position_code = &asset.init_layout.position_code;
            let position_code = if position_code.is_empty() {
                DEFAULT_POSITION_CODE.to_owned()
            } else {
                position_code.clone()
            };

            // Generate the shader code for the lifetime initializing of newly emitted particles
            // TODO - Move that to a pre-pass, not each frame!
            let lifetime_code = &asset.init_layout.lifetime_code;
            let lifetime_code = if lifetime_code.is_empty() {
                DEFAULT_LIFETIME_CODE.to_owned()
            } else {
                lifetime_code.clone()
            };

            // Generate the shader code for the force field of newly emitted particles
            // TODO - Move that to a pre-pass, not each frame!
            // let force_field_code = &asset.init_layout.force_field_code;
            // let force_field_code = if force_field_code.is_empty() {
            let force_field_code = if 0.0 == asset.update_layout.force_field[0].force_exponent {
                DEFAULT_FORCE_FIELD_CODE.to_owned()
            } else {
                FORCE_FIELD_CODE.to_owned()
            };

            // Generate the shader code for the color over lifetime gradient.
            // TODO - Move that to a pre-pass, not each frame!
            let mut vertex_modifiers =
                if let Some(grad) = &asset.render_layout.lifetime_color_gradient {
                    grad.to_shader_code()
                } else {
                    String::new()
                };
            if let Some(grad) = &asset.render_layout.size_color_gradient {
                vertex_modifiers += &grad.to_shader_code();
            }
            trace!("vertex_modifiers={}", vertex_modifiers);

            // Configure the shader template, and make sure a corresponding shader asset exists
            let shader_source =
                PARTICLES_RENDER_SHADER_TEMPLATE.replace("{{VERTEX_MODIFIERS}}", &vertex_modifiers);
            let shader = pipeline_registry.configure(&shader_source, &mut shaders);

            trace!(
                "extracted: handle={:?} shader={:?} has_image={} position_code={} force_field_code={} lifetime_code={}",
                effect.handle,
                shader,
                if asset.render_layout.particle_texture.is_some() {
                    "Y"
                } else {
                    "N"
                },
                position_code,
                force_field_code,
                lifetime_code,
            );

            extracted_effects.effects.insert(
                entity,
                ExtractedEffect {
                    handle: effect.handle.clone_weak(),
                    spawn_count,
                    color: Color::RED, //effect.color,
                    transform: transform.compute_matrix(),
                    accel,
                    force_field,
                    rect: Rect {
                        left: -0.1,
                        top: -0.1,
                        right: 0.1,
                        bottom: 0.1, // effect
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
                    position_code,
                    force_field_code,
                    lifetime_code,
                },
            );
        }
    }
}

/// A single particle as stored in a GPU buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, AsStd430)]
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

/// Global resource containing the GPU data to draw all the particle effects in all views.
///
/// The resource is populated by [`prepare_effects()`] with all the effects to render
/// for the current frame, for all views in the frame, and consumed by [`queue_effects()`]
/// to actually enqueue the drawning commands to draw those effects.
pub(crate) struct EffectsMeta {
    /// Map from an entity with a [`ParticleEffect`] component attached to it, to the associated
    /// effect slice allocated in an [`EffectCache`].
    entity_map: HashMap<Entity, EffectSlice>,
    /// Global effect cache for all effects in use.
    effect_cache: EffectCache,
    /// Bind group for the camera view, containing the camera projection and other uniform
    /// values related to the camera.
    view_bind_group: Option<BindGroup>,
    /// Bind group for the simulation parameters, like the current time and frame delta time.
    sim_params_bind_group: Option<BindGroup>,
    /// Bind group for the particles buffer itself.
    particles_bind_group: Option<BindGroup>,
    /// Bind group for the spawning parameters (number of particles to spawn this frame, ...).
    spawner_bind_group: Option<BindGroup>,
    /// Bind group for the indirect buffer.
    indirect_buffer_bind_group: Option<BindGroup>,
    sim_params_uniforms: UniformVec<SimParamsUniform>,
    spawner_buffer: AlignedBufferVec<SpawnerParams>,
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

        let item_align = device.limits().min_storage_buffer_offset_alignment as usize;

        Self {
            entity_map: HashMap::default(),
            effect_cache: EffectCache::new(device),
            view_bind_group: None,
            sim_params_bind_group: None,
            particles_bind_group: None,
            spawner_bind_group: None,
            indirect_buffer_bind_group: None,
            sim_params_uniforms: UniformVec::default(),
            spawner_buffer: AlignedBufferVec::new(
                BufferUsages::STORAGE,
                item_align,
                Some("spawner_buffer".to_string()),
            ),
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
    /// Update position code.
    position_code: String,
    /// Update force field code.
    force_field_code: String,
    /// Update lifetime code.
    lifetime_code: String,
    /// Compute pipeline specialized for this batch.
    compute_pipeline: Option<ComputePipeline>,
}

pub(crate) fn prepare_effects(
    mut commands: Commands,
    sim_params: Res<SimParams>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    //update_pipeline: Res<ParticlesUpdatePipeline>, // TODO move update_pipeline.pipeline to EffectsMeta
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
            //update_pipeline.pipeline.clone(),
            &render_queue,
        );
        let slice = effects_meta.effect_cache.get_slice(id);
        effects_meta.entity_map.insert(entity, slice);
    }

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
            let slice = effects_meta.entity_map.get(entity).unwrap().clone();
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
    let mut position_code = String::default();
    let mut force_field_code = String::default();
    let mut lifetime_code = String::default();

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
                        position_code: position_code.clone(),
                        force_field_code: force_field_code.clone(),
                        lifetime_code: lifetime_code.clone(),
                        compute_pipeline: None,
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

        position_code = extracted_effect.position_code.clone();
        trace!("position_code = {}", position_code);

        force_field_code = extracted_effect.force_field_code.clone();
        trace!("force_field_code = {}", force_field_code);

        lifetime_code = extracted_effect.lifetime_code.clone();
        trace!("lifetime_code = {}", lifetime_code);

        // extract the force field and turn it into a struct that is compliant with Std430,
        // namely ForceFieldStd430
        let mut extracted_force_field = [ForceFieldStd430::default(); FFNUM];
        for (i, ff) in extracted_effect.force_field.iter().enumerate() {
            extracted_force_field[i] = (*ff).into();
        }

        // Prepare the spawner block for the current slice
        // FIXME - This is once per EFFECT/SLICE, not once per BATCH, so indeed this is spawner_BASE, and need an array of them in the compute shader!!!!!!!!!!!!!!
        let spawner_params = SpawnerParams {
            spawn: extracted_effect.spawn_count as i32,
            count: 0,
            origin: extracted_effect.transform.col(3).truncate(),
            accel: extracted_effect.accel,
            force_field: extracted_force_field, // extracted_effect.force_field,
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
                    position_code: position_code.clone(),
                    force_field_code: force_field_code.clone(),
                    lifetime_code: lifetime_code.clone(),
                    compute_pipeline: None,
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
            shader,
            position_code,
            force_field_code,
            lifetime_code,
            compute_pipeline: None,
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
pub struct EffectBindGroups {
    /// Bind groups for each group index for compute shader.
    update_particle_buffers: HashMap<u32, BindGroup>,
    /// Same for render shader.
    render_particle_buffers: HashMap<u32, BindGroup>,
    /// Bind groups for each indirect buffer associated with each particle buffer (update stage).
    update_indirect_buffers: HashMap<u32, BindGroup>,
    /// Bind groups for each indirect buffer associated with each particle buffer (render stage).
    render_indirect_buffers: HashMap<u32, BindGroup>,
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
    mut compute_cache: ResMut<ComputeCache<ParticlesUpdatePipeline>>,
    render_pipeline: Res<ParticlesRenderPipeline>,
    mut specialized_render_pipelines: ResMut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    mut render_pipeline_cache: ResMut<PipelineCache>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    gpu_images: Res<RenderAssets<Image>>,
    mut effect_batches: Query<(Entity, &mut EffectBatch)>,
    #[cfg(feature = "2d")] mut views_2d: Query<&mut RenderPhase<Transparent2d>>,
    #[cfg(feature = "3d")] mut views_3d: Query<&mut RenderPhase<Transparent3d>>,
    events: Res<EffectAssetEvents>,
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

    // Get the binding for the ViewUniform, the uniform data structure containing the Camera data
    // for the current view.
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
        label: Some("particles_view_bind_group"),
        layout: &render_pipeline.view_layout,
    }));

    // Create the bind group for the global simulation parameters
    effects_meta.sim_params_bind_group =
        Some(render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[BindGroupEntry {
                binding: 0,
                resource: effects_meta.sim_params_uniforms.binding().unwrap(),
            }],
            label: Some("particles_sim_params_bind_group"),
            layout: &update_pipeline.sim_params_layout,
        }));

    // Create the bind group for the spawner parameters
    trace!(
        "SpawnerParams::std430_size_static() = {}",
        SpawnerParams::std430_size_static()
    );
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
        // Ensure all effect groups have a bind group for the entire buffer of the group,
        // since the update phase runs on an entire group/buffer at once, with all the
        // effect instances in it batched together.
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
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buffer.max_binding(),
                    }],
                    label: Some(&format!("vfx_particles_bind_group_update{}", buffer_index)),
                    layout: &update_pipeline.particles_buffer_layout,
                })
            });

        effect_bind_groups
            .update_indirect_buffers
            .entry(buffer_index as u32)
            .or_insert_with(|| {
                trace!(
                    "Create new indirect update buffer bind group for buffer_index={}",
                    buffer_index
                );
                render_device.create_bind_group(&BindGroupDescriptor {
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buffer.indirect_max_binding(),
                    }],
                    label: Some(&format!(
                        "vfx_indirect_buffer_bind_group_update{}",
                        buffer_index
                    )),
                    layout: &update_pipeline.indirect_buffer_layout,
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
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buffer.max_binding(),
                    }],
                    label: Some(&format!("vfx_particles_bind_group_render{}", buffer_index)),
                    layout: &render_pipeline.particles_buffer_layout,
                })
            });

        effect_bind_groups
            .render_indirect_buffers
            .entry(buffer_index as u32)
            .or_insert_with(|| {
                trace!(
                    "Create new indirect render buffer bind group for buffer_index={}",
                    buffer_index
                );
                render_device.create_bind_group(&BindGroupDescriptor {
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buffer.indirect_max_binding(),
                    }],
                    label: Some(&format!(
                        "vfx_indirect_buffer_bind_group_render{}",
                        buffer_index
                    )),
                    layout: &update_pipeline.indirect_buffer_layout,
                })
            });
    }

    // Queue the update
    // TODO - Move to prepare(), there's no view-dependent thing here!
    for (_entity, mut batch) in effect_batches.iter_mut() {
        // Specialize the update pipeline based on the effect batch
        trace!(
            "Specializing update pipeline: position_code={:?}",
            batch.position_code,
        );
        let compute_pipeline = compute_cache.specialize(
            &update_pipeline,
            ParticleUpdatePipelineKey {
                position_code: batch.position_code.clone(),
                force_field_code: batch.force_field_code.clone(),
                lifetime_code: batch.lifetime_code.clone(),
            },
            &render_device,
        );
        trace!("Update pipeline specialized: {:?}", compute_pipeline);

        batch.compute_pipeline = Some(compute_pipeline.clone());
    }

    // Loop over all 2D cameras/views that need to render effects
    #[cfg(feature = "2d")]
    {
        let draw_effects_function_2d = draw_functions_2d.read().get_id::<DrawEffects>().unwrap();
        for mut transparent_phase_2d in views_2d.iter_mut() {
            trace!("Process new Transparent2d view");
            // For each view, loop over all the effect batches to determine if the effect needs to be rendered
            // for that view, and enqueue a view-dependent batch if so.
            for (entity, batch) in effect_batches.iter() {
                trace!(
                    "Process batch entity={:?} buffer_index={} spawner_base={} slice={:?}",
                    entity,
                    batch.buffer_index,
                    batch.spawner_base,
                    batch.slice
                );
                // Ensure the particle texture is available as a GPU resource and create a bind group for it
                let particle_texture = if batch.layout_flags.contains(LayoutFlags::PARTICLE_TEXTURE)
                {
                    let image_handle = Handle::weak(batch.image_handle_id);
                    if effect_bind_groups.images.get(&image_handle).is_none() {
                        trace!(
                            "Batch buffer #{} slice={:?} has missing GPU image bind group, creating...",
                            batch.buffer_index,
                            batch.slice
                        );
                        // If texture doesn't have a bind group yet from another instance of the same effect,
                        // then try to create one now
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
                                    label: Some("particles_material_bind_group"),
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
                    "Specializing render pipeline: shader={:?} particle_texture={:?}",
                    batch.shader,
                    particle_texture
                );
                let render_pipeline_id = specialized_render_pipelines.specialize(
                    &mut render_pipeline_cache,
                    &render_pipeline,
                    ParticleRenderPipelineKey {
                        particle_texture,
                        shader: batch.shader.clone(),
                        #[cfg(feature = "3d")]
                        pipeline_mode: PipelineMode::Camera2d,
                    },
                );
                trace!("Render pipeline specialized: id={:?}", render_pipeline_id);

                // Add a draw pass for the effect batch
                trace!("Add Transparent for batch on entity {:?}: buffer_index={} spawner_base={} slice={:?} handle={:?}", entity, batch.buffer_index, batch.spawner_base, batch.slice, batch.handle);
                transparent_phase_2d.add(Transparent2d {
                    draw_function: draw_effects_function_2d,
                    pipeline: render_pipeline_id,
                    entity,
                    sort_key: FloatOrd(0.0),
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
            // For each view, loop over all the effect batches to determine if the effect needs to be rendered
            // for that view, and enqueue a view-dependent batch if so.
            for (entity, batch) in effect_batches.iter() {
                trace!(
                    "Process batch entity={:?} buffer_index={} spawner_base={} slice={:?}",
                    entity,
                    batch.buffer_index,
                    batch.spawner_base,
                    batch.slice
                );
                // Ensure the particle texture is available as a GPU resource and create a bind group for it
                let particle_texture = if batch.layout_flags.contains(LayoutFlags::PARTICLE_TEXTURE)
                {
                    let image_handle = Handle::weak(batch.image_handle_id);
                    if effect_bind_groups.images.get(&image_handle).is_none() {
                        trace!(
                            "Batch buffer #{} slice={:?} has missing GPU image bind group, creating...",
                            batch.buffer_index,
                            batch.slice
                        );
                        // If texture doesn't have a bind group yet from another instance of the same effect,
                        // then try to create one now
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
                                    label: Some("particles_material_bind_group"),
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
                    "Specializing render pipeline: shader={:?} particle_texture={:?}",
                    batch.shader,
                    particle_texture
                );
                let render_pipeline_id = specialized_render_pipelines.specialize(
                    &mut render_pipeline_cache,
                    &render_pipeline,
                    ParticleRenderPipelineKey {
                        particle_texture,
                        shader: batch.shader.clone(),
                        #[cfg(feature = "2d")]
                        pipeline_mode: PipelineMode::Camera3d,
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

/// Component to hold all the entities with a [`ExtractedEffect`] component on them
/// that need to be updated this frame with a compute pass. This is view-independent
/// because the update phase itself is also view-independent (effects like camera
/// facing are applied in the render phase, which runs once per view).
#[derive(Component)]
pub struct ExtractedEffectEntities {
    pub entities: Vec<Entity>,
}

/// Draw function for rendering all active effects for the current frame.
///
/// Effects are rendered in the [`Transparent2d`] phase of the main 2D pass,
/// and the [`Transparent3d`] phase of the main 3D pass.
pub struct DrawEffects {
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
        let (effects_meta, effect_bind_groups, specialized_render_pipelines, views, effects) =
            self.params.get(world);
        let view_uniform = views.get(view).unwrap();
        let effects_meta = effects_meta.into_inner();
        let effect_bind_groups = effect_bind_groups.into_inner();
        let effect_batch = effects.get(item.entity).unwrap();
        if let Some(pipeline) = specialized_render_pipelines
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
        let (effects_meta, effect_bind_groups, specialized_render_pipelines, views, effects) =
            self.params.get(world);
        let view_uniform = views.get(view).unwrap();
        let effects_meta = effects_meta.into_inner();
        let effect_bind_groups = effect_bind_groups.into_inner();
        let effect_batch = effects.get(item.entity).unwrap();
        if let Some(pipeline) = specialized_render_pipelines
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
                    if let Some(compute_pipeline) = &batch.compute_pipeline {
                        //for (effect_entity, effect_slice) in effects_meta.entity_map.iter() {
                        // Retrieve the ExtractedEffect from the entity
                        //trace!("effect_entity={:?} effect_slice={:?}", effect_entity, effect_slice);
                        //let effect = self.effect_query.get_manual(world, *effect_entity).unwrap();

                        // Get the slice to update
                        //let effect_slice = effects_meta.get(&effect_entity);
                        // let effect_group =
                        //     &effects_meta.effect_cache.buffers()[batch.buffer_index as usize];
                        let particles_bind_group = effect_bind_groups
                            .update_particle_buffers
                            .get(&batch.buffer_index)
                            .unwrap();

                        let indirect_bind_group = effect_bind_groups
                            .update_indirect_buffers
                            .get(&batch.buffer_index)
                            .unwrap();

                        let item_size = batch.item_size;
                        let item_count = batch.slice.end - batch.slice.start;
                        let workgroup_count = (item_count + 63) / 64;

                        let spawner_base = batch.spawner_base;
                        let buffer_offset = batch.slice.start;

                        let spawner_buffer_aligned = effects_meta.spawner_buffer.aligned_size();
                        assert!(spawner_buffer_aligned >= SpawnerParams::std430_size_static());

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
                        compute_pass.set_bind_group(1, particles_bind_group, &[buffer_offset]);
                        compute_pass.set_bind_group(
                            2,
                            effects_meta.spawner_bind_group.as_ref().unwrap(),
                            &[spawner_base * spawner_buffer_aligned as u32],
                        );
                        compute_pass.set_bind_group(3, indirect_bind_group, &[buffer_offset]);
                        compute_pass.dispatch(workgroup_count, 1, 1);
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

    #[test]
    fn to_shader_code() {
        let mut grad = Gradient::new();
        assert_eq!("", grad.to_shader_code());

        grad.add_key(0.0, Vec4::splat(0.0));
        assert_eq!(
            "// Gradient\nlet t0 = 0.;\nlet c0 = vec4<f32>(0., 0., 0., 0.);\nout.color = c0;\n",
            grad.to_shader_code()
        );

        grad.add_key(1.0, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(
            r#"// Gradient
let t0 = 0.;
let c0 = vec4<f32>(0., 0., 0., 0.);
let t1 = 1.;
let c1 = vec4<f32>(1., 0., 0., 1.);
let life = particle.age / particle.lifetime;
if (life <= t0) { out.color = c0; }
else if (life <= t1) { out.color = mix(c0, c1, (life - t0) / (t1 - t0)); }
else { out.color = c1; }
"#,
            grad.to_shader_code()
        );
    }
}
