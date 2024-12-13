//! Puffs
//!
//! This example creates cartoony smoke puffs out of spherical meshes.

use std::{error::Error, f32::consts::FRAC_PI_2};

use bevy::{
    color::palettes::css::FOREST_GREEN,
    core_pipeline::tonemapping::Tonemapping,
    math::vec3,
    prelude::*,
    render::mesh::{SphereKind, SphereMeshBuilder},
};
use bevy_hanabi::prelude::*;
use serde::{Deserialize, Serialize};

use crate::utils::*;

mod utils;

// A simple custom modifier that lights the meshes with Lambertian lighting.
// Other lighting models are possible, up to and including PBR.
#[derive(Clone, Copy, Reflect, Serialize, Deserialize)]
struct LambertianLightingModifier {
    // The direction that light is coming from, in particle system space.
    light_direction: Vec3,
    // The brightness of the ambient light (which is assumed to be white in
    // this example).
    ambient: f32,
}

// The position of the light in the scene.
static LIGHT_POSITION: Vec3 = vec3(-20.0, 40.0, 5.0);

fn main() -> Result<(), Box<dyn Error>> {
    let app_exit = make_test_app("puffs")
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 500.0,
        })
        .add_systems(Startup, setup)
        .add_systems(Update, setup_scene_once_loaded)
        .run();
    app_exit.into_result()
}

// Performs initialization of the scene.
fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Spawn the camera.
    commands.spawn((
        Transform::from_xyz(25.0, 15.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
        Camera {
            hdr: true,
            clear_color: Color::BLACK.into(),
            ..default()
        },
        Camera3d::default(),
        Tonemapping::None,
    ));

    // Spawn the fox.
    commands.spawn((
        SceneRoot(asset_server.load("Fox.glb#Scene0")),
        Transform::from_scale(Vec3::splat(0.1)),
    ));

    // Spawn the circular base.
    commands.spawn((
        Transform::from_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
        Mesh3d(meshes.add(Circle::new(15.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: FOREST_GREEN.into(),
            ..default()
        })),
    ));

    // Spawn a light.
    commands.spawn((
        Transform::from_translation(LIGHT_POSITION).looking_at(Vec3::ZERO, Vec3::Y),
        DirectionalLight {
            color: Color::WHITE,
            illuminance: 2000.0,
            shadows_enabled: true,
            ..default()
        },
    ));

    // Create the mesh.
    let mesh = meshes.add(SphereMeshBuilder::new(0.5, SphereKind::Ico { subdivisions: 4 }).build());

    // Create the effect asset.
    let effect = create_effect(mesh, &mut effects);

    // Spawn the effect.
    commands.spawn((
        Name::new("cartoon explosion"),
        ParticleEffectBundle {
            effect: ParticleEffect::new(effect),
            ..default()
        },
    ));
}

// Builds the smoke puffs.
fn create_effect(mesh: Handle<Mesh>, effects: &mut Assets<EffectAsset>) -> Handle<EffectAsset> {
    let writer = ExprWriter::new();

    // Position the particle laterally within a small radius.
    let init_xz_pos = SetPositionCircleModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        axis: writer.lit(Vec3::Z).expr(),
        radius: writer.lit(1.0).expr(),
        dimension: ShapeDimension::Volume,
    };

    // Position the particle vertically. Jiggle it a little bit for variety's
    // sake.
    let init_y_pos = SetAttributeModifier::new(
        Attribute::POSITION,
        writer
            .attr(Attribute::POSITION)
            .add(writer.rand(VectorType::VEC3F) * writer.lit(vec3(0.0, 1.0, 0.0)))
            .expr(),
    );

    // Set up the age and lifetime.
    let init_age = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.0).expr());
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, writer.lit(3.0).expr());

    // Vary the size a bit.
    let init_size = SetAttributeModifier::new(
        Attribute::F32_0,
        (writer.rand(ScalarType::Float) * writer.lit(2.0) + writer.lit(0.5)).expr(),
    );

    // Make the particles move backwards at a constant speed.
    let init_velocity = SetAttributeModifier::new(
        Attribute::VELOCITY,
        writer.lit(vec3(0.0, 0.0, -20.0)).expr(),
    );

    // Make the particles shrink over time.
    let update_size = SetAttributeModifier::new(
        Attribute::SIZE,
        writer
            .attr(Attribute::F32_0)
            .mul(
                writer
                    .lit(1.0)
                    .sub((writer.attr(Attribute::AGE)).mul(writer.lit(0.75)))
                    .max(writer.lit(0.0)),
            )
            .expr(),
    );

    // Add some nice shading to the particles.
    let render_lambertian =
        LambertianLightingModifier::new(LIGHT_POSITION.normalize_or_zero(), 0.7);

    let module = writer.finish();

    // Add the effect.
    effects.add(
        EffectAsset::new(256, Spawner::burst(16.0.into(), 0.45.into()), module)
            .with_name("cartoon explosion")
            .init(init_xz_pos)
            .init(init_y_pos)
            .init(init_age)
            .init(init_lifetime)
            .init(init_size)
            .init(init_velocity)
            .update(update_size)
            .render(render_lambertian)
            .mesh(mesh),
    )
}

// A system that plays the running animation once the fox loads.
fn setup_scene_once_loaded(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut animation_graphs: ResMut<Assets<AnimationGraph>>,
    mut players: Query<(Entity, &mut AnimationPlayer), Added<AnimationPlayer>>,
) {
    for (entity, mut animation_player) in players.iter_mut() {
        let (animation_graph, animation_graph_node) =
            AnimationGraph::from_clip(asset_server.load("Fox.glb#Animation2"));
        let animation_graph = AnimationGraphHandle(animation_graphs.add(animation_graph));
        animation_player.play(animation_graph_node).repeat();
        commands.entity(entity).insert(animation_graph.clone());
    }
}

impl LambertianLightingModifier {
    fn new(light_direction: Vec3, ambient: f32) -> LambertianLightingModifier {
        LambertianLightingModifier {
            light_direction,
            ambient,
        }
    }
}

// Boilerplate implementation of `Modifier` for our lighting modifier.
#[cfg_attr(feature = "serde", typetag::serde)]
impl Modifier for LambertianLightingModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Render
    }

    fn as_render(&self) -> Option<&dyn RenderModifier> {
        Some(self)
    }

    fn as_render_mut(&mut self) -> Option<&mut dyn RenderModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[Attribute] {
        &[]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn apply(&self, _: &mut Module, _: &mut ShaderWriter) -> Result<(), ExprError> {
        Err(ExprError::TypeError("Wrong modifier context".to_string()))
    }
}

// The implementation of Lambertian lighting.
#[cfg_attr(feature = "serde", typetag::serde)]
impl RenderModifier for LambertianLightingModifier {
    fn apply_render(&self, _: &mut Module, context: &mut RenderContext) -> Result<(), ExprError> {
        // We need the vertex normals to light the mesh.
        context.set_needs_normal();

        // Shade each fragment.
        context.fragment_code += &format!(
            "color = vec4(color.rgb * mix({}, 1.0, dot(normal, {})), color.a);",
            self.ambient.to_wgsl_string(),
            self.light_direction.to_wgsl_string()
        );

        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(*self)
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}
