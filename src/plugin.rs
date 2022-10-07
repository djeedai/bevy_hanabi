#[cfg(feature = "2d")]
use bevy::core_pipeline::core_2d::Transparent2d;
#[cfg(feature = "3d")]
use bevy::core_pipeline::core_3d::Transparent3d;
use bevy::{
    prelude::*,
    render::{
        render_graph::RenderGraph,
        render_phase::DrawFunctions,
        render_resource::{
            SpecializedComputePipelines, SpecializedRenderPipelines, WgpuAdapterInfo,
        },
        renderer::RenderDevice,
        view::visibility::VisibilitySystems,
        RenderApp, RenderStage,
    },
};

use crate::{
    asset::{EffectAsset, EffectAssetLoader},
    gather_removed_effects,
    render::{
        extract_effect_events, extract_effects, prepare_effects, queue_effects,
        DispatchIndirectPipeline, DrawEffects, EffectAssetEvents, EffectBindGroups, EffectSystems,
        EffectsMeta, ExtractedEffects, ParticleUpdateNode, ParticlesInitPipeline,
        ParticlesRenderPipeline, ParticlesUpdatePipeline, PipelineRegistry, SimParams,
        VFX_INDIRECT_SHADER_HANDLE,
    },
    spawn::{self, Random},
    tick_spawners, ParticleEffect, RemovedEffectsEvent, Spawner,
};

pub mod draw_graph {
    pub mod node {
        /// Label for the particle update compute node.
        pub const PARTICLE_UPDATE_PASS: &str = "particle_update_pass";
    }
}

/// Plugin to add systems related to Hanabi.
#[derive(Debug, Clone, Copy)]
pub struct HanabiPlugin;

impl Plugin for HanabiPlugin {
    fn build(&self, app: &mut App) {
        let render_device = app.world.get_resource::<RenderDevice>().unwrap().clone();

        // Check device limits
        let limits = render_device.limits();
        if limits.max_bind_groups < 4 {
            let adapter_name = app
                .world
                .get_resource::<WgpuAdapterInfo>()
                .map(|ai| &ai.name[..])
                .unwrap_or("<unknown>");
            error!("Hanabi requires a GPU device supporting at least 4 bind groups (Limits::max_bind_groups).\n  Current adapter: {}\n  Supported bind groups: {}", adapter_name, limits.max_bind_groups);
            return;
        }

        // Register asset
        app.add_asset::<EffectAsset>()
            .add_event::<RemovedEffectsEvent>()
            .insert_resource(Random(spawn::new_rng()))
            .init_resource::<PipelineRegistry>()
            .init_asset_loader::<EffectAssetLoader>()
            .add_system_to_stage(
                CoreStage::PostUpdate,
                tick_spawners
                    .label(EffectSystems::TickSpawners)
                    // This checks the visibility to skip shader work, so needs to run
                    // after ComputedVisibility was updated.
                    .after(VisibilitySystems::CheckVisibility),
            )
            .add_system_to_stage(
                CoreStage::PostUpdate,
                gather_removed_effects.label(EffectSystems::GatherRemovedEffects),
            );

        // Register the built-in shaders
        let mut shaders = app.world.get_resource_mut::<Assets<Shader>>().unwrap();
        let indirect_shader = Shader::from_wgsl(include_str!("render/vfx_indirect.wgsl"));
        shaders.set_untracked(VFX_INDIRECT_SHADER_HANDLE, indirect_shader);

        // Register the component reflection
        app.register_type::<EffectAsset>();
        app.register_type::<ParticleEffect>();
        app.register_type::<Spawner>();

        let effects_meta = EffectsMeta::new(render_device.clone());

        // Register the custom render pipeline
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(effects_meta)
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
            .add_system_to_stage(
                RenderStage::Extract,
                extract_effects.label(EffectSystems::ExtractEffects),
            )
            .add_system_to_stage(
                RenderStage::Extract,
                extract_effect_events.label(EffectSystems::ExtractEffectEvents),
            )
            .add_system_to_stage(
                RenderStage::Prepare,
                prepare_effects.label(EffectSystems::PrepareEffects),
            )
            .add_system_to_stage(
                RenderStage::Queue,
                queue_effects.label(EffectSystems::QueueEffects),
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
        }

        // Register the update node before the 2D/3D main pass, where the particles are
        // drawn. This ensures the update compute pipelines for all the active
        // particle effects are executed before the 2D/3D main pass starts,
        // which consumes the result of the updated particles to render them.
        #[cfg(feature = "2d")]
        use bevy::core_pipeline::core_2d::graph as draw_2d_graph;
        #[cfg(feature = "3d")]
        use bevy::core_pipeline::core_3d::graph as draw_3d_graph;

        #[cfg(feature = "2d")]
        let update_node_2d = ParticleUpdateNode::new(&mut render_app.world);

        #[cfg(feature = "3d")]
        let update_node_3d = ParticleUpdateNode::new(&mut render_app.world);

        let mut graph = render_app.world.get_resource_mut::<RenderGraph>().unwrap();

        #[cfg(feature = "2d")]
        {
            let draw_graph = graph.get_sub_graph_mut(draw_2d_graph::NAME).unwrap();
            draw_graph.add_node(draw_graph::node::PARTICLE_UPDATE_PASS, update_node_2d);
            draw_graph
                .add_node_edge(
                    draw_graph::node::PARTICLE_UPDATE_PASS,
                    draw_2d_graph::node::MAIN_PASS,
                )
                .unwrap();
            draw_graph
                .add_slot_edge(
                    draw_graph.input_node().unwrap().id,
                    draw_2d_graph::input::VIEW_ENTITY,
                    draw_graph::node::PARTICLE_UPDATE_PASS,
                    ParticleUpdateNode::IN_VIEW,
                )
                .unwrap();
        }

        #[cfg(feature = "3d")]
        {
            let draw_graph = graph.get_sub_graph_mut(draw_3d_graph::NAME).unwrap();
            draw_graph.add_node(draw_graph::node::PARTICLE_UPDATE_PASS, update_node_3d);
            draw_graph
                .add_node_edge(
                    draw_graph::node::PARTICLE_UPDATE_PASS,
                    draw_3d_graph::node::MAIN_PASS,
                )
                .unwrap();
            draw_graph
                .add_slot_edge(
                    draw_graph.input_node().unwrap().id,
                    draw_3d_graph::input::VIEW_ENTITY,
                    draw_graph::node::PARTICLE_UPDATE_PASS,
                    ParticleUpdateNode::IN_VIEW,
                )
                .unwrap();
        }
    }
}
