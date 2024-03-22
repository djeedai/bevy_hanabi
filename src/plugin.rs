#[cfg(feature = "2d")]
use bevy::core_pipeline::core_2d::Transparent2d;
#[cfg(feature = "3d")]
use bevy::core_pipeline::core_3d::{AlphaMask3d, Transparent3d};
use bevy::{
    prelude::*,
    render::{
        render_graph::RenderGraph,
        render_phase::DrawFunctions,
        render_resource::{SpecializedComputePipelines, SpecializedRenderPipelines},
        renderer::{RenderAdapterInfo, RenderDevice},
        view::{prepare_view_uniforms, visibility::VisibilitySystems},
        Render, RenderApp, RenderSet,
    },
    time::{virtual_time_system, TimeSystem},
};

use crate::{
    asset::{EffectAsset, EffectAssetLoader},
    compile_effects, gather_removed_effects,
    properties::EffectProperties,
    render::{
        extract_effect_events, extract_effects, prepare_bind_groups, prepare_effects,
        prepare_resources, queue_effects, DispatchIndirectPipeline, DrawEffects, EffectAssetEvents,
        EffectBindGroups, EffectCache, EffectSystems, EffectsMeta, ExtractedEffects,
        GpuSpawnerParams, ParticlesInitPipeline, ParticlesRenderPipeline, ParticlesUpdatePipeline,
        ShaderCache, SimParams, VfxSimulateDriverNode, VfxSimulateNode,
    },
    spawn::{self, Random},
    tick_spawners,
    time::effect_simulation_time_system,
    update_properties_from_asset, EffectSimulation, ParticleEffect, RemovedEffectsEvent, Spawner,
};

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
    /// This creates a new [`Shader`] from the `vfx_common.wgsl` code, by
    /// applying the given alignment for storage buffers. This produces a shader
    /// ready for the specific GPU device associated with that alignment.
    pub(crate) fn make_common_shader(min_storage_buffer_offset_alignment: usize) -> Shader {
        let spawner_padding_code =
            GpuSpawnerParams::padding_code(min_storage_buffer_offset_alignment);
        let common_code = include_str!("render/vfx_common.wgsl")
            .replace("{{SPAWNER_PADDING}}", &spawner_padding_code);
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
}

impl Plugin for HanabiPlugin {
    fn build(&self, app: &mut App) {
        // Register asset
        app.init_asset::<EffectAsset>()
            .add_event::<RemovedEffectsEvent>()
            .insert_resource(Random(spawn::new_rng()))
            .init_resource::<ShaderCache>()
            .init_asset_loader::<EffectAssetLoader>()
            .init_resource::<Time<EffectSimulation>>()
            .configure_sets(
                PostUpdate,
                (
                    EffectSystems::TickSpawners
                        // This checks the visibility to skip work, so needs to run after
                        // ComputedVisibility was updated.
                        .after(VisibilitySystems::VisibilityPropagate),
                    EffectSystems::CompileEffects,
                    EffectSystems::GatherRemovedEffects,
                ),
            )
            .configure_sets(
                bevy::asset::UpdateAssets,
                EffectSystems::UpdatePropertiesFromAsset.after(bevy::asset::TrackAssets),
            )
            .add_systems(
                First,
                effect_simulation_time_system
                    .after(virtual_time_system)
                    .in_set(TimeSystem),
            )
            .add_systems(
                PostUpdate,
                (
                    tick_spawners.in_set(EffectSystems::TickSpawners),
                    compile_effects.in_set(EffectSystems::CompileEffects),
                    update_properties_from_asset.in_set(EffectSystems::UpdatePropertiesFromAsset),
                    gather_removed_effects.in_set(EffectSystems::GatherRemovedEffects),
                ),
            );

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
            .world
            .resource::<RenderDevice>()
            .clone();

        let adapter_name = app
            .world
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
                render_device.limits().min_storage_buffer_offset_alignment as usize,
            );
            let mut assets = app.world.resource_mut::<Assets<Shader>>();
            assets.insert(HANABI_COMMON_TEMPLATE_HANDLE, common_shader);
        }

        let effects_meta = EffectsMeta::new(render_device.clone());
        let effect_cache = EffectCache::new(render_device);

        // Register the custom render pipeline
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(effects_meta)
            .insert_resource(effect_cache)
            .init_resource::<EffectBindGroups>()
            .init_resource::<DispatchIndirectPipeline>()
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
                    EffectSystems::PrepareEffectGpuResources.in_set(RenderSet::Prepare),
                    EffectSystems::PrepareBindGroups.in_set(RenderSet::PrepareBindGroups),
                ),
            )
            .edit_schedule(ExtractSchedule, |schedule| {
                schedule.add_systems((extract_effects, extract_effect_events));
            })
            .add_systems(
                Render,
                (
                    prepare_effects.in_set(EffectSystems::PrepareEffectAssets),
                    queue_effects
                        .in_set(EffectSystems::QueueEffects)
                        .after(prepare_effects),
                    prepare_resources
                        .in_set(EffectSystems::PrepareEffectGpuResources)
                        .after(prepare_view_uniforms),
                    prepare_bind_groups
                        .in_set(EffectSystems::PrepareBindGroups)
                        .after(queue_effects),
                ),
            );

        // Register the draw function for drawing the particles. This will be called
        // during the main 2D/3D pass, at the Transparent2d/3d phase, after the
        // opaque objects have been rendered (or, rather, commands for those
        // have been recorded).
        #[cfg(feature = "2d")]
        {
            let draw_particles = DrawEffects::new(&mut render_app.world);
            render_app
                .world
                .get_resource::<DrawFunctions<Transparent2d>>()
                .unwrap()
                .write()
                .add(draw_particles);
        }
        #[cfg(feature = "3d")]
        {
            let draw_particles = DrawEffects::new(&mut render_app.world);
            render_app
                .world
                .get_resource::<DrawFunctions<Transparent3d>>()
                .unwrap()
                .write()
                .add(draw_particles);

            let draw_particles = DrawEffects::new(&mut render_app.world);
            render_app
                .world
                .get_resource::<DrawFunctions<AlphaMask3d>>()
                .unwrap()
                .write()
                .add(draw_particles);
        }

        // Add the simulation sub-graph. This render graph runs once per frame no matter
        // how many cameras/views are active (view-independent).
        let mut simulate_graph = RenderGraph::default();
        let simulate_node = VfxSimulateNode::new(&mut render_app.world);
        simulate_graph.add_node(simulate_graph::node::HanabiSimulateNode, simulate_node);
        let mut graph = render_app.world.get_resource_mut::<RenderGraph>().unwrap();
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
