//! Force field example.
//!
//! This example demonstrates how to use the `ForceFieldModifier` to simulate
//! attraction and repulsion forces. The example is interactif; left clicking
//! spawns particles that are repulsed by one point and attracted by another.
//! The attractor also conforms the particles that are close to a sphere around
//! it.
//!
//! The example also demonstrates the `AabbKillModifier` through two boxes: a
//! green "allow" box to which particles are confined, and a red "forbid" box
//! killing all particles entering it.

use bevy::{
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
                    filter: "bevy_hanabi=warn,spawn=trace".to_string(),
                })
                .set(RenderPlugin { wgpu_settings }),
        )
        //.add_plugin(LookTransformPlugin)
        //.add_plugin(OrbitCameraPlugin::default())
        .add_plugin(HanabiPlugin)
        // Have to wait for update.
        // .add_plugin(WorldInspectorPlugin::default())
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
    let mut camera = Camera3dBundle::default();
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
            alpha_mode: AlphaMode::Blend,
            ..Default::default()
        }),
        ..Default::default()
    });

    // "forbid" box
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Box::new(0.8, 0.4, 6.))),
        material: materials.add(StandardMaterial {
            base_color: Color::rgba(0.7, 0., 0., 0.3),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            ..Default::default()
        }),
        transform: Transform::from_translation(Vec3::new(-2., -1., 0.)),
        ..Default::default()
    });

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(0.0, 1.0, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::new(0.0, 1.0, 1.0, 0.0));

    // Prevent the spawner from immediately spawning on activation, and instead
    // require a manual reset() call.
    let spawn_immediately = false;

    let spawner = Spawner::once(30.0.into(), spawn_immediately);

    // Force field effects
    let effect = effects.add(
        EffectAsset {
            name: "Impact".into(),
            capacity: 32768,
            spawner,
            ..Default::default()
        }
        .init(InitPositionSphereModifier {
            center: Vec3::ZERO,
            radius: BALL_RADIUS,
            dimension: ShapeDimension::Surface,
        })
        .init(InitVelocitySphereModifier {
            center: Vec3::ZERO,
            speed: Value::Uniform((0.1, 0.3)),
        })
        .init(InitLifetimeModifier {
            lifetime: 5_f32.into(),
        })
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
        .update(AabbKillModifier {
            min: Vec3::new(-3., -2., -3.),
            max: Vec3::new(3., 2., 3.),
            kill_inside: false,
        })
        .update(AabbKillModifier {
            min: Vec3::new(-2.4, -1.2, -3.),
            max: Vec3::new(-1.6, -0.8, 3.),
            kill_inside: true,
        })
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::constant(Vec2::splat(0.05)),
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
    let Ok((mut spawner, mut effect_transform)) = q_effect.get_single_mut() else { return; };

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
