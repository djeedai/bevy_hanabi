#[cfg(feature = "2d")]
use bevy::core_pipeline::core_2d::Transparent2d;
#[cfg(feature = "3d")]
use bevy::core_pipeline::core_3d::{AlphaMask3d, Opaque3d, Transparent3d};
use bevy::{
    prelude::*,
    render::{
        mesh::allocator::allocate_and_free_meshes,
        render_asset::prepare_assets,
        render_graph::RenderGraph,
        render_phase::DrawFunctions,
        render_resource::{SpecializedComputePipelines, SpecializedRenderPipelines},
        renderer::{RenderAdapterInfo, RenderDevice},
        texture::GpuImage,
        view::{check_visibility, prepare_view_uniforms, visibility::VisibilitySystems},
        Render, RenderApp, RenderSet,
    },
    time::{time_system, TimeSystem},
};

#[cfg(feature = "serde")]
use crate::asset::EffectAssetLoader;
use crate::{
    asset::EffectAsset,
    compile_effects,
    properties::EffectProperties,
    render::{
        add_effects, batch_effects, extract_effect_events, extract_effects, fixup_parents,
        on_remove_cached_effect, on_remove_cached_properties, prepare_bind_groups, prepare_effects,
        prepare_gpu_resources, prepare_property_buffers, queue_effects, resolve_parents,
        DebugSettings, DispatchIndirectPipeline, DrawEffects, EffectAssetEvents, EffectBindGroups,
        EffectCache, EffectsMeta, EventCache, ExtractedEffects, GpuBufferOperationQueue,
        GpuDispatchIndirect, GpuParticleGroup, GpuRenderEffectMetadata, GpuRenderGroupIndirect,
        GpuSpawnerParams, ParticlesInitPipeline, ParticlesRenderPipeline, ParticlesUpdatePipeline,
        PropertyBindGroups, PropertyCache, RenderDebugSettings, ShaderCache, SimParams,
        StorageType as _, UtilsPipeline, VfxSimulateDriverNode, VfxSimulateNode,
    },
    spawn::{self, Random},
    tick_initializers,
    time::effect_simulation_time_system,
    update_properties_from_asset, CompiledParticleEffect, EffectSimulation, ParticleEffect,
    Spawner, ToWgslString,
};

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
    ///
    /// [`CompiledParticleEffect`]: crate::CompiledParticleEffect
    CompileEffects,

    /// Update the properties of the effect instance based on the declared
    /// properties in the [`EffectAsset`], updating the associated
    /// [`EffectProperties`] component.
    ///
    /// This system runs during the [`PostUpdate`] schedule, after the assets
    /// have been updated. Any system which modifies an [`EffectAsset`]'s
    /// declared properties should run before this set in order for changes to
    /// be taken into account in the same frame.
    UpdatePropertiesFromAsset,

    /// Prepare effect assets for the extracted effects.
    ///
    /// Part of Bevy's own [`RenderSet::PrepareAssets`].
    PrepareEffectAssets,

    /// Queue the GPU commands for the extracted effects.
    ///
    /// Part of Bevy's own [`RenderSet::Queue`].
    QueueEffects,

    /// Prepare GPU data for the queued effects.
    ///
    /// Part of Bevy's own [`RenderSet::PrepareResources`].
    PrepareEffectGpuResources,

    /// Prepare the GPU bind groups once all buffers have been (re-)allocated
    /// and won't change this frame.
    ///
    /// Part of Bevy's own [`RenderSet::PrepareBindGroups`].
    PrepareBindGroups,
}

pub mod main_graph {
    pub mod node {
        use bevy::render::render_graph::RenderLabel;

        /// Label for the simulation driver node running the simulation graph.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, RenderLabel)]
        pub struct HanabiDriverNode;
    }
}

pub mod simulate_graph {
    use bevy::render::render_graph::RenderSubGraph;

    /// Name of the simulation sub-graph.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, RenderSubGraph)]
    pub struct HanabiSimulateGraph;

    pub mod node {
        use bevy::render::render_graph::RenderLabel;

        /// Label for the simulation node (init and update compute passes;
        /// view-independent).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, RenderLabel)]
        pub struct HanabiSimulateNode;
    }
}

// {626E7AD3-4E54-487E-B796-9A90E34CC1EC}
const HANABI_COMMON_TEMPLATE_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(0x626E7AD34E54487EB7969A90E34CC1ECu128);

/// Plugin to add systems related to Hanabi.
#[derive(Debug, Clone, Copy)]
pub struct HanabiPlugin;

impl HanabiPlugin {
    /// Create the `vfx_common.wgsl` shader with proper alignment.
    ///
    /// This creates a new [`Shader`] from the `vfx_common.wgsl` template file,
    /// by applying the given alignment for storage buffers. This produces a
    /// shader ready for the specific GPU device associated with that
    /// alignment.
    pub(crate) fn make_common_shader(min_storage_buffer_offset_alignment: u32) -> Shader {
        let spawner_padding_code =
            GpuSpawnerParams::padding_code(min_storage_buffer_offset_alignment);
        let dispatch_indirect_padding_code =
            GpuDispatchIndirect::padding_code(min_storage_buffer_offset_alignment);
        let dispatch_indirect_stride_code =
            (GpuDispatchIndirect::aligned_size(min_storage_buffer_offset_alignment).get() as u32)
                .to_wgsl_string();
        let render_effect_indirect_padding_code =
            GpuRenderEffectMetadata::padding_code(min_storage_buffer_offset_alignment);
        let render_group_indirect_padding_code =
            GpuRenderGroupIndirect::padding_code(min_storage_buffer_offset_alignment);
        let particle_group_padding_code =
            GpuParticleGroup::padding_code(min_storage_buffer_offset_alignment);
        let render_effect_indirect_size =
            GpuRenderEffectMetadata::aligned_size(min_storage_buffer_offset_alignment);
        let render_effect_indirect_stride_code =
            (render_effect_indirect_size.get() as u32).to_wgsl_string();
        let render_group_indirect_size =
            GpuRenderGroupIndirect::aligned_size(min_storage_buffer_offset_alignment);
        let render_group_indirect_stride_code =
            (render_group_indirect_size.get() as u32).to_wgsl_string();
        let common_code = include_str!("render/vfx_common.wgsl")
            .replace("{{SPAWNER_PADDING}}", &spawner_padding_code)
            .replace(
                "{{DISPATCH_INDIRECT_PADDING}}",
                &dispatch_indirect_padding_code,
            )
            .replace(
                "{{DISPATCH_INDIRECT_STRIDE}}",
                &dispatch_indirect_stride_code,
            )
            .replace(
                "{{RENDER_EFFECT_INDIRECT_PADDING}}",
                &render_effect_indirect_padding_code,
            )
            .replace(
                "{{RENDER_EFFECT_INDIRECT_STRIDE}}",
                &render_effect_indirect_stride_code,
            )
            .replace(
                "{{RENDER_GROUP_INDIRECT_PADDING}}",
                &render_group_indirect_padding_code,
            )
            .replace(
                "{{RENDER_GROUP_INDIRECT_STRIDE}}",
                &render_group_indirect_stride_code,
            )
            .replace("{{PARTICLE_GROUP_PADDING}}", &particle_group_padding_code);
        Shader::from_wgsl(
            common_code,
            std::path::Path::new(file!())
                .parent()
                .unwrap()
                .join(format!(
                    "render/vfx_common_{}.wgsl",
                    min_storage_buffer_offset_alignment
                ))
                .to_string_lossy(),
        )
    }

    /// Create the `vfx_indirect.wgsl` shader with proper alignment.
    ///
    /// This creates a new [`Shader`] from the `vfx_indirect.wgsl` template
    /// file, by applying the given alignment for storage buffers. This
    /// produces a shader ready for the specific GPU device associated with
    /// that alignment.
    pub(crate) fn make_indirect_shader(
        min_storage_buffer_offset_alignment: u32,
        has_events: bool,
    ) -> Shader {
        let render_effect_indirect_size =
            GpuRenderEffectMetadata::aligned_size(min_storage_buffer_offset_alignment);
        let render_effect_indirect_stride_code =
            (render_effect_indirect_size.get() as u32).to_wgsl_string();
        let render_group_indirect_size =
            GpuRenderGroupIndirect::aligned_size(min_storage_buffer_offset_alignment);
        let render_group_indirect_stride_code =
            (render_group_indirect_size.get() as u32).to_wgsl_string();
        let indirect_code = include_str!("render/vfx_indirect.wgsl")
            .replace(
                "{{RENDER_EFFECT_INDIRECT_STRIDE}}",
                &render_effect_indirect_stride_code,
            )
            .replace(
                "{{RENDER_GROUP_INDIRECT_STRIDE}}",
                &render_group_indirect_stride_code,
            );
        Shader::from_wgsl(
            indirect_code,
            std::path::Path::new(file!())
                .parent()
                .unwrap()
                .join(format!(
                    "render/vfx_indirect_{}_{}.wgsl",
                    min_storage_buffer_offset_alignment,
                    if has_events { "events" } else { "noevent" },
                ))
                .to_string_lossy(),
        )
    }
}

/// A convenient alias for `With<CompiledParticleEffect>`, for use with
/// [`bevy_render::view::VisibleEntities`].
pub type WithCompiledParticleEffect = With<CompiledParticleEffect>;

impl Plugin for HanabiPlugin {
    fn build(&self, app: &mut App) {
        // Register asset
        app.init_asset::<EffectAsset>()
            .insert_resource(Random(spawn::new_rng()))
            .init_resource::<ShaderCache>()
            .init_resource::<DebugSettings>()
            .init_resource::<Time<EffectSimulation>>()
            .configure_sets(
                PostUpdate,
                (
                    EffectSystems::TickSpawners
                        // This checks the visibility to skip work, so needs to run after
                        // ComputedVisibility was updated.
                        .after(VisibilitySystems::VisibilityPropagate),
                    EffectSystems::CompileEffects,
                ),
            )
            .configure_sets(
                PreUpdate,
                EffectSystems::UpdatePropertiesFromAsset.after(bevy::asset::TrackAssets),
            )
            .add_systems(
                First,
                effect_simulation_time_system
                    .after(time_system)
                    .in_set(TimeSystem),
            )
            .add_systems(
                PostUpdate,
                (
                    tick_initializers.in_set(EffectSystems::TickSpawners),
                    compile_effects.in_set(EffectSystems::CompileEffects),
                    update_properties_from_asset.in_set(EffectSystems::UpdatePropertiesFromAsset),
                    check_visibility::<WithCompiledParticleEffect>
                        .in_set(VisibilitySystems::CheckVisibility),
                ),
            );

        #[cfg(feature = "serde")]
        app.init_asset_loader::<EffectAssetLoader>();

        // Register types with reflection
        app.register_type::<EffectAsset>()
            .register_type::<ParticleEffect>()
            .register_type::<EffectProperties>()
            .register_type::<Spawner>()
            .register_type::<Time<EffectSimulation>>();
    }

    fn finish(&self, app: &mut App) {
        let render_device = app
            .sub_app(RenderApp)
            .world()
            .resource::<RenderDevice>()
            .clone();

        let adapter_name = app
            .world()
            .get_resource::<RenderAdapterInfo>()
            .map(|ai| &ai.name[..])
            .unwrap_or("<unknown>");

        // Check device limits
        let limits = render_device.limits();
        if limits.max_bind_groups < 4 {
            error!("Hanabi requires a GPU device supporting at least 4 bind groups (Limits::max_bind_groups).\n  Current adapter: {}\n  Supported bind groups: {}", adapter_name, limits.max_bind_groups);
            return;
        } else {
            info!("Initializing Hanabi for GPU adapter {}", adapter_name);
        }

        // Insert the properly aligned `vfx_common.wgsl` shader into Assets<Shader>, so
        // that the automated Bevy shader processing finds it as an import. This is used
        // for init/update/render shaders (but not the indirect one).
        {
            let common_shader = HanabiPlugin::make_common_shader(
                render_device.limits().min_storage_buffer_offset_alignment,
            );
            let mut assets = app.world_mut().resource_mut::<Assets<Shader>>();
            assets.insert(&HANABI_COMMON_TEMPLATE_HANDLE, common_shader);
        }

        // Insert the two variants of the properly aligned `vfx_indirect.wgsl` shaders
        // into Assets<Shader>.
        let (indirect_shader_noevent, indirect_shader_events) = {
            let align = render_device.limits().min_storage_buffer_offset_alignment;
            let indirect_shader_noevent = HanabiPlugin::make_indirect_shader(align, false);
            let indirect_shader_events = HanabiPlugin::make_indirect_shader(align, true);

            let mut assets = app.world_mut().resource_mut::<Assets<Shader>>();
            let indirect_shader_noevent = assets.add(indirect_shader_noevent);
            let indirect_shader_events = assets.add(indirect_shader_events);

            (indirect_shader_noevent, indirect_shader_events)
        };

        let effects_meta = {
            let mut meshes = app.world_mut().resource_mut::<Assets<Mesh>>();
            EffectsMeta::new(
                render_device.clone(),
                &mut meshes,
                indirect_shader_noevent,
                indirect_shader_events,
            )
        };

        let effect_cache = EffectCache::new(render_device.clone());
        let property_cache = PropertyCache::new(render_device.clone());
        let event_cache = EventCache::new(render_device);

        // Register the custom render pipeline
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(effects_meta)
            .insert_resource(effect_cache)
            .insert_resource(property_cache)
            .insert_resource(event_cache)
            .init_resource::<RenderDebugSettings>()
            .init_resource::<EffectBindGroups>()
            .init_resource::<PropertyBindGroups>()
            .init_resource::<GpuBufferOperationQueue>()
            .init_resource::<UtilsPipeline>()
            .init_resource::<DispatchIndirectPipeline>()
            .init_resource::<SpecializedComputePipelines<DispatchIndirectPipeline>>()
            .init_resource::<ParticlesInitPipeline>()
            .init_resource::<SpecializedComputePipelines<ParticlesInitPipeline>>()
            .init_resource::<ParticlesInitPipeline>()
            .init_resource::<SpecializedComputePipelines<ParticlesInitPipeline>>()
            .init_resource::<ParticlesUpdatePipeline>()
            .init_resource::<SpecializedComputePipelines<ParticlesUpdatePipeline>>()
            .init_resource::<ParticlesRenderPipeline>()
            .init_resource::<SpecializedRenderPipelines<ParticlesRenderPipeline>>()
            .init_resource::<ExtractedEffects>()
            .init_resource::<EffectAssetEvents>()
            .init_resource::<SimParams>()
            .configure_sets(
                Render,
                (
                    EffectSystems::PrepareEffectAssets.in_set(RenderSet::PrepareAssets),
                    EffectSystems::QueueEffects.in_set(RenderSet::Queue),
                    EffectSystems::PrepareEffectGpuResources.in_set(RenderSet::PrepareResources),
                    EffectSystems::PrepareBindGroups.in_set(RenderSet::PrepareBindGroups),
                ),
            )
            .edit_schedule(ExtractSchedule, |schedule| {
                schedule.add_systems((extract_effects, extract_effect_events));
            })
            .add_systems(
                Render,
                (
                    (
                        add_effects,
                        resolve_parents,
                        fixup_parents,
                        prepare_effects,
                        batch_effects,
                    )
                        .chain()
                        .in_set(EffectSystems::PrepareEffectAssets)
                        // Ensure we run after Bevy prepared the render Mesh
                        .after(allocate_and_free_meshes),
                    queue_effects
                        .in_set(EffectSystems::QueueEffects)
                        .after(batch_effects),
                    prepare_gpu_resources
                        .in_set(EffectSystems::PrepareEffectGpuResources)
                        .after(prepare_view_uniforms)
                        .before(prepare_bind_groups),
                    prepare_property_buffers
                        .in_set(EffectSystems::PrepareEffectGpuResources)
                        .after(add_effects)
                        .before(prepare_bind_groups),
                    prepare_bind_groups
                        .in_set(EffectSystems::PrepareBindGroups)
                        .after(queue_effects)
                        .after(prepare_assets::<GpuImage>),
                ),
            );
        render_app.world_mut().add_observer(on_remove_cached_effect);
        render_app
            .world_mut()
            .add_observer(on_remove_cached_properties);

        // Register the draw function for drawing the particles. This will be called
        // during the main 2D/3D pass, at the Transparent2d/3d phase, after the
        // opaque objects have been rendered (or, rather, commands for those
        // have been recorded).
        #[cfg(feature = "2d")]
        {
            let draw_particles = DrawEffects::new(render_app.world_mut());
            render_app
                .world()
                .get_resource::<DrawFunctions<Transparent2d>>()
                .unwrap()
                .write()
                .add(draw_particles);
        }
        #[cfg(feature = "3d")]
        {
            let draw_particles = DrawEffects::new(render_app.world_mut());
            render_app
                .world()
                .get_resource::<DrawFunctions<Transparent3d>>()
                .unwrap()
                .write()
                .add(draw_particles);

            let draw_particles = DrawEffects::new(render_app.world_mut());
            render_app
                .world()
                .get_resource::<DrawFunctions<AlphaMask3d>>()
                .unwrap()
                .write()
                .add(draw_particles);

            let draw_particles = DrawEffects::new(render_app.world_mut());
            render_app
                .world()
                .get_resource::<DrawFunctions<Opaque3d>>()
                .unwrap()
                .write()
                .add(draw_particles);
        }

        // Add the simulation sub-graph. This render graph runs once per frame no matter
        // how many cameras/views are active (view-independent).
        let mut simulate_graph = RenderGraph::default();
        let simulate_node = VfxSimulateNode::new(render_app.world_mut());
        simulate_graph.add_node(simulate_graph::node::HanabiSimulateNode, simulate_node);
        let mut graph = render_app
            .world_mut()
            .get_resource_mut::<RenderGraph>()
            .unwrap();
        graph.add_sub_graph(simulate_graph::HanabiSimulateGraph, simulate_graph);

        // Add the simulation driver node which executes the simulation sub-graph. It
        // runs before the camera driver, since rendering needs to access simulated
        // particles.
        graph.add_node(main_graph::node::HanabiDriverNode, VfxSimulateDriverNode {});
        graph.add_node_edge(
            main_graph::node::HanabiDriverNode,
            bevy::render::graph::CameraDriverLabel,
        );
    }
}
