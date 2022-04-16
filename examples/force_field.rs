//! Left clicking spawns particles that are repulsed by one point and attracted by another.
//! The attractor also conforms the particles that are close to a sphere around it.
//! Left Control + Mouse movement orbits the camera.
//! Mouse scroll wheel zooms the camera.
use bevy::{
    prelude::*,
    render::{render_resource::WgpuFeatures, settings::WgpuSettings},
};
//use bevy_inspector_egui::WorldInspectorPlugin;

use bevy_hanabi::*;
// use smooth_bevy_cameras::{
//     controllers::orbit::{OrbitCameraBundle, OrbitCameraController, OrbitCameraPlugin},
//     LookTransformPlugin,
// };

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = WgpuSettings::default();
    options
        .features
        .set(WgpuFeatures::VERTEX_WRITABLE_STORAGE, true);
    App::default()
        .insert_resource(options)
        .insert_resource(bevy::log::LogSettings {
            level: bevy::log::Level::WARN,
            filter: "bevy_hanabi=error,spawn=trace".to_string(),
        })
        .add_plugins(DefaultPlugins)
        .add_system(bevy::input::system::exit_on_esc_system)
        //.add_plugin(LookTransformPlugin)
        //.add_plugin(OrbitCameraPlugin::default())
        .add_plugin(HanabiPlugin)
        //.add_plugin(WorldInspectorPlugin::new())
        .add_startup_system(setup)
        .add_system(update)
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
    // let orbit_controller = OrbitCameraController {
    //     mouse_translate_sensitivity: Vec2::ZERO,
    //     ..Default::default()
    // };
    // commands.spawn_bundle(OrbitCameraBundle::new(
    //     orbit_controller,
    //     PerspectiveCameraBundle::default(),
    //     Vec3::new(0.0, 0.0, 6.0), // eye of the camera
    //     Vec3::new(0., 0., 0.),
    // ));
    let mut bundle = PerspectiveCameraBundle::new_3d();
    bundle.transform.translation = Vec3::new(0.0, 0.0, 6.0);
    commands.spawn_bundle(bundle);

    let attractor1_position = Vec3::new(0.01, 0.0, 0.0);
    let attractor2_position = Vec3::new(1.0, 0.5, 0.0);

    commands.spawn_bundle(PointLightBundle {
        transform: Transform::from_xyz(4.0, 5.0, 4.0),
        ..Default::default()
    });
    commands.spawn_bundle(PointLightBundle {
        transform: Transform::from_xyz(4.0, -5.0, -4.0),
        ..Default::default()
    });

    commands.spawn_bundle(PbrBundle {
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

    commands.spawn_bundle(PbrBundle {
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

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(0.0, 1.0, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::new(0.0, 1.0, 1.0, 0.0));

    let spawner = Spawner::once(30.0.into(), false);

    // Force field effects
    let effect = effects.add(
        EffectAsset {
            name: "Impact".into(),
            capacity: 32768,
            spawner,
            ..Default::default()
        }
        .init(PositionSphereModifier {
            radius: BALL_RADIUS,
            speed: Value::Uniform((0.1, 0.3)),
            dimension: ShapeDimension::Surface,
            ..Default::default()
        })
        .update(bevy_hanabi::ForceFieldModifier::new(vec![
            ForceFieldParam {
                position: attractor2_position,
                max_radius: 1000000.0,
                min_radius: BALL_RADIUS * 6.0,
                // a negative mass produces a repulsive force instead of an attractive one
                mass: -1.5,
                // linear force: proportional to 1 / distance
                force_exponent: 1.0,
                conform_to_sphere: true,
            },
            ForceFieldParam {
                position: attractor1_position,
                max_radius: 1000000.0,
                min_radius: BALL_RADIUS * 6.0,
                mass: 3.0,
                // quadratic force: proportional to 1 / distance^2
                force_exponent: 2.0,
                conform_to_sphere: true,
            },
        ]))
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::constant(Vec2::splat(0.05)),
        })
        .render(ColorOverLifetimeModifier { gradient }),
    );

    commands.spawn_bundle(ParticleEffectBundle::new(effect).with_spawner(spawner));
}

fn update(
    mut effect: Query<(&mut ParticleEffect, &mut Transform), Without<PerspectiveProjection>>,
    mouse_button_input: Res<Input<MouseButton>>,
    camera_query: Query<&Transform, With<PerspectiveProjection>>,
    windows: Res<Windows>,
) {
    let (mut effect, mut effect_transform) = effect.single_mut();
    let camera_transform = camera_query.single();

    let up = camera_transform.up();
    let right = camera_transform.right();

    let window = windows.get_primary().unwrap();

    if let Some(mouse_pos) = window.cursor_position() {
        if mouse_button_input.just_pressed(MouseButton::Left) {
            let screen_mouse_pos = (mouse_pos - Vec2::new(window.width(), window.height()) / 2.0)
                * camera_transform.translation.length()
                / 870.0; // investigate: why 870?

            // converts the mouse position to a position on the view plane centered at the origin.
            let spawning_pos = screen_mouse_pos.x * right + screen_mouse_pos.y * up;

            effect_transform.translation = spawning_pos;

            // Spawn the particles
            effect.maybe_spawner().unwrap().reset();
        }
    }
}
