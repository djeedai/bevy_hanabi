#[cfg(feature="2d")]
use bevy::core_pipeline::Transparent2d as Transparent;
#[cfg(feature="3d")]
use bevy::core_pipeline::Transparent3d as Transparent;
use bevy::{
    prelude::*,
    render::{
        render_graph::RenderGraph, render_phase::DrawFunctions,
        render_resource::SpecializedPipelines, renderer::RenderDevice, RenderApp, RenderStage,
    },
};

use crate::{
    asset::{EffectAsset, EffectAssetLoader},
    render::{
        extract_effect_events, extract_effects, prepare_effects, queue_effects, ComputeCache,
        DrawEffects, EffectAssetEvents, EffectBindGroups, EffectSystems, EffectsMeta,
        ExtractedEffects, ImageBindGroups, ParticleUpdateNode, ParticlesRenderPipeline,
        ParticlesUpdatePipeline, PipelineRegistry, SimParams, PARTICLES_RENDER_SHADER_HANDLE,
        PARTICLES_UPDATE_SHADER_HANDLE,
    },
    spawn::{self, Random},
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
        // Register asset
        app.add_asset::<EffectAsset>()
            .insert_resource(Random(spawn::new_rng()))
            .init_resource::<PipelineRegistry>()
            .init_asset_loader::<EffectAssetLoader>();

        // Register the spawn and update systems
        // app.add_system(hanabi_spawn.system())
        //     .add_system(hanabi_update.system());

        // Register the particles shaders
        let mut shaders = app.world.get_resource_mut::<Assets<Shader>>().unwrap();
        let update_shader = Shader::from_wgsl(include_str!("render/particles_update.wgsl"));
        shaders.set_untracked(PARTICLES_UPDATE_SHADER_HANDLE, update_shader);
        let render_shader = Shader::from_wgsl(include_str!("render/particles_render.wgsl"));
        shaders.set_untracked(PARTICLES_RENDER_SHADER_HANDLE, render_shader);

        // Register the component reflection
        //app.register_type::<ParticleEffect>();

        let render_device = app.world.get_resource::<RenderDevice>().unwrap();
        let effects_meta = EffectsMeta::new(render_device.clone());

        // Register the custom render pipeline
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(effects_meta)
            .init_resource::<ImageBindGroups>()
            .init_resource::<EffectBindGroups>()
            .init_resource::<ParticlesUpdatePipeline>()
            .init_resource::<ComputeCache<ParticlesUpdatePipeline>>()
            .init_resource::<ParticlesRenderPipeline>()
            .init_resource::<SpecializedPipelines<ParticlesRenderPipeline>>()
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

        // Register the draw function for drawing the particles. This will be called during
        // the main 2D/3D pass, at the Transparent2d/3d phase, after the opaque objects have been
        // rendered (or, rather, commands for those have been recorded).
        {
            let draw_particles = DrawEffects::new(&mut render_app.world);
            render_app
                .world
                .get_resource::<DrawFunctions<Transparent>>()
                .unwrap()
                .write()
                .add(draw_particles);
        }

        // Register the update node before the 2D/3D main pass, where the particles are drawn.
        // This ensures the update compute pipelines for all the active particle effects are
        // executed before the 2D/3D main pass starts, which consumes the result of the updated
        // particles to render them.
        #[cfg(feature="2d")]
        use bevy::core_pipeline::draw_2d_graph as bevy_draw_graph;
        #[cfg(feature="3d")]
        use bevy::core_pipeline::draw_3d_graph as bevy_draw_graph;

        let update_node = ParticleUpdateNode::new(&mut render_app.world);
        let mut graph = render_app.world.get_resource_mut::<RenderGraph>().unwrap();
        let draw_graph = graph
            .get_sub_graph_mut(bevy_draw_graph::NAME)
            .unwrap();
        draw_graph.add_node(draw_graph::node::PARTICLE_UPDATE_PASS, update_node);
        draw_graph
            .add_node_edge(
                draw_graph::node::PARTICLE_UPDATE_PASS,
                bevy_draw_graph::node::MAIN_PASS,
            )
            .unwrap();
        draw_graph
            .add_slot_edge(
                draw_graph.input_node().unwrap().id,
                bevy_draw_graph::input::VIEW_ENTITY,
                draw_graph::node::PARTICLE_UPDATE_PASS,
                ParticleUpdateNode::IN_VIEW,
            )
            .unwrap();
    }
}

// pub fn hanabi_spawn(
//     time: Res<Time>,
//     mut query: Query<(&mut ParticleEffect, &mut SpawnState, &mut UpdateState)>,
// ) {
//     for (ref mut effect, ref mut spawn_state, ref mut state) in query.iter_mut() {
//         effect
//             .spawner
//             .spawn(spawn_state, state, time.delta_seconds());
//     }
// }

// pub fn hanabi_update(time: Res<Time>, mut query: Query<(&mut ParticleEffect, &mut UpdateState)>) {
//     for (ref mut effect, ref mut motion) in query.iter_mut() {
//         effect.updater.update(motion, time.delta_seconds());
//     }
// }
