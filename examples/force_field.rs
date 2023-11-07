//! Force field example.
//!
//! This example demonstrates how to use the `ForceFieldModifier` to simulate
//! attraction and repulsion forces. The example is interactif; left clicking
//! spawns particles that are repulsed by one point and attracted by another.
//! The attractor also conforms the particles that are close to a sphere around
//! it.
//!
//! The example also demonstrates the `KillAabbModifier` and
//! `KillSphereModifier`: a green "allow" box to which particles are confined,
//! and a red "forbid" sphere killing all particles entering it.
//!
//! Note: Some particles may _appear_ to penetrate the red "forbid" sphere due
//! to the projection on screen; however those particles are actually at a
//! different depth, in front or behind the sphere.

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    log::LogPlugin,
    prelude::*,
    render::{
        camera::Projection, render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin,
    },
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

use bevy_hanabi::prelude::*;

// use smooth_bevy_cameras::{
//     controllers::orbit::{OrbitCameraBundle, OrbitCameraController,
// OrbitCameraPlugin},     LookTransformPlugin,
// };

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut wgpu_settings = WgpuSettings::default();
    wgpu_settings
        .features
        .set(WgpuFeatures::VERTEX_WRITABLE_STORAGE, true);

    App::default()
        .insert_resource(ClearColor(Color::DARK_GRAY))
        .add_plugins(
            DefaultPlugins
                .set(LogPlugin {
                    level: bevy::log::Level::WARN,
                    filter: "bevy_hanabi=warn,force_field=trace".to_string(),
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings.into(),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "🎆 Hanabi — force field".to_string(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        //.add_plugins(LookTransformPlugin)
        //.add_plugins(OrbitCameraPlugin::default())
        .add_plugins(HanabiPlugin)
        .add_plugins(WorldInspectorPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (bevy::window::close_on_esc, update))
        .run();

    Ok(())
}

const BALL_RADIUS: f32 = 0.05;

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = Camera3dBundle {
        tonemapping: Tonemapping::None,
        ..default()
    };
    let mut projection = OrthographicProjection::default();
    projection.scaling_mode = bevy::render::camera::ScalingMode::FixedVertical(5.);
    camera.transform.translation.z = projection.far / 2.0;
    camera.projection = Projection::Orthographic(projection);
    commands.spawn(camera);

    let attractor1_position = Vec3::new(0.01, 0.0, 0.0);
    let attractor2_position = Vec3::new(1.0, 0.5, 0.0);

    commands.spawn(PointLightBundle {
        transform: Transform::from_xyz(4.0, 5.0, 4.0),
        ..Default::default()
    });
    commands.spawn(PointLightBundle {
        transform: Transform::from_xyz(4.0, -5.0, -4.0),
        ..Default::default()
    });

    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::UVSphere {
            sectors: 128,
            stacks: 4,
            radius: BALL_RADIUS * 2.0,
        })),
        material: materials.add(StandardMaterial {
            base_color: Color::YELLOW,
            unlit: false,
            ..Default::default()
        }),
        transform: Transform::from_translation(attractor1_position),
        ..Default::default()
    });

    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::UVSphere {
            sectors: 128,
            stacks: 4,
            radius: BALL_RADIUS * 1.0,
        })),
        material: materials.add(StandardMaterial {
            base_color: Color::PURPLE,
            unlit: false,
            ..Default::default()
        }),
        transform: Transform::from_translation(attractor2_position),
        ..Default::default()
    });

    // "allow" box
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Box::new(6., 4., 6.))),
        material: materials.add(StandardMaterial {
            base_color: Color::rgba(0., 0.7, 0., 0.3),
            unlit: true,
            alpha_mode: bevy::pbr::AlphaMode::Blend,
            ..Default::default()
        }),
        ..Default::default()
    });

    // "forbid" sphere
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::UVSphere {
            radius: 0.6,
            sectors: 32,
            stacks: 8,
        })),
        material: materials.add(StandardMaterial {
            base_color: Color::rgba(0.7, 0., 0., 0.3),
            unlit: true,
            alpha_mode: bevy::pbr::AlphaMode::Blend,
            ..Default::default()
        }),
        transform: Transform::from_translation(Vec3::new(-2., -1., 0.1)),
        ..Default::default()
    });

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(0.0, 1.0, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::new(0.0, 1.0, 1.0, 0.0));

    // Prevent the spawner from immediately spawning on activation, and instead
    // require a manual reset() call.
    let spawn_immediately = false;

    let spawner = Spawner::once(30.0.into(), spawn_immediately);

    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(5.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let center = writer.lit(Vec3::ZERO).expr();
    let half_size = writer.lit(Vec3::new(3., 2., 3.)).expr();
    let allow_zone = KillAabbModifier::new(center, half_size);

    let center = writer.lit(Vec3::new(-2., -1., 0.)).expr();
    let radius = writer.lit(0.6);
    let sqr_radius = (radius.clone() * radius).expr();
    let deny_zone = KillSphereModifier::new(center, sqr_radius).with_kill_inside(true);

    let init_pos = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(BALL_RADIUS).expr(),
        dimension: ShapeDimension::Surface,
    };

    let init_vel = SetVelocitySphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        speed: (writer.rand(ScalarType::Float) * writer.lit(0.2) + writer.lit(0.1)).expr(),
    };

    // Force field effects
    let effect = effects.add(
        EffectAsset::new(32768, spawner, writer.finish())
            .with_name("force_field")
            .init(init_pos)
            .init(init_vel)
            .init(init_age)
            .init(init_lifetime)
            .update(ForceFieldModifier::new(vec![
                ForceFieldSource {
                    position: attractor2_position,
                    max_radius: 1000000.0,
                    min_radius: BALL_RADIUS * 6.0,
                    // a negative mass produces a repulsive force instead of an attractive one
                    mass: -1.5,
                    // linear force: proportional to 1 / distance
                    force_exponent: 1.0,
                    conform_to_sphere: true,
                },
                ForceFieldSource {
                    position: attractor1_position,
                    max_radius: 1000000.0,
                    min_radius: BALL_RADIUS * 6.0,
                    mass: 3.0,
                    // quadratic force: proportional to 1 / distance^2
                    force_exponent: 2.0,
                    conform_to_sphere: true,
                },
            ]))
            .update(allow_zone)
            .update(deny_zone)
            .render(SizeOverLifetimeModifier {
                gradient: Gradient::constant(Vec2::splat(0.05)),
                screen_space_size: false,
            })
            .render(ColorOverLifetimeModifier { gradient }),
    );

    commands.spawn(ParticleEffectBundle::new(effect).with_spawner(spawner));
}

fn update(
    mut q_effect: Query<(&mut EffectSpawner, &mut Transform), Without<Projection>>,
    mouse_button_input: Res<Input<MouseButton>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Projection>>,
    window: Query<&Window, With<bevy::window::PrimaryWindow>>,
) {
    // Note: On first frame where the effect spawns, EffectSpawner is spawned during
    // CoreSet::PostUpdate, so will not be available yet. Ignore for a frame if
    // so.
    let Ok((mut spawner, mut effect_transform)) = q_effect.get_single_mut() else {
        return;
    };

    let (camera, camera_transform) = camera_query.single();

    if let Ok(window) = window.get_single() {
        if let Some(mouse_pos) = window.cursor_position() {
            if mouse_button_input.just_pressed(MouseButton::Left) {
                let ray = camera
                    .viewport_to_world(camera_transform, mouse_pos)
                    .unwrap();
                let spawning_pos = Vec3::new(ray.origin.x, ray.origin.y, 0.);

                effect_transform.translation = spawning_pos;

                // Spawn a single burst of particles
                spawner.reset();
            }
        }
    }
}
