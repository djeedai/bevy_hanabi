//! Clicking spawns particles that gravitate around two points.
//!
use bevy::{
    prelude::*,
    render::{options::WgpuOptions, render_resource::WgpuFeatures},
};
use bevy_inspector_egui::WorldInspectorPlugin;

use bevy_hanabi::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = WgpuOptions::default();
    options
        .features
        .set(WgpuFeatures::VERTEX_WRITABLE_STORAGE, true);
    App::default()
        .insert_resource(options)
        .insert_resource(MousePosition::default())
        .insert_resource(bevy::log::LogSettings {
            level: bevy::log::Level::WARN,
            filter: "bevy_hanabi=error,spawn=trace".to_string(),
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(HanabiPlugin)
        .add_plugin(WorldInspectorPlugin::new())
        .add_startup_system(setup)
        .add_system(update)
        .add_system(record_mouse_events_system)
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
    // let mut camera = OrthographicCameraBundle::new_3d();
    let mut camera = PerspectiveCameraBundle::new_3d();
    // camera.orthographic_projection.scale = 1.2;
    camera.transform.translation.z = 6.0;
    commands.spawn_bundle(camera);

    let attractor1_position = Vec3::new(0.01, 0.0, 0.0);
    let attractor2_position = Vec3::new(1.0, 0.5, 0.0);

    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::UVSphere {
            sectors: 32,
            stacks: 2,
            radius: BALL_RADIUS * 2.0,
        })),
        material: materials.add(StandardMaterial {
            base_color: Color::YELLOW,
            unlit: true,
            ..Default::default()
        }),
        transform: Transform::from_translation(attractor1_position),
        ..Default::default()
    });

    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::UVSphere {
            sectors: 32,
            stacks: 2,
            radius: BALL_RADIUS * 1.0,
        })),
        material: materials.add(StandardMaterial {
            base_color: Color::YELLOW,
            unlit: true,
            ..Default::default()
        }),
        transform: Transform::from_translation(attractor2_position),
        ..Default::default()
    });

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(0.0, 1.0, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::new(0.0, 1.0, 1.0, 0.0));

    let spawner = Spawner::once(30.0.into(), false);
    let effect = effects.add(
        EffectAsset {
            name: "Impact".into(),
            capacity: 32768,
            spawner,
            ..Default::default()
        }
        .init(PositionSphereModifier {
            radius: BALL_RADIUS,
            speed: 0.2,
            dimension: ShapeDimension::Surface,
            ..Default::default()
        })
        .update(bevy_hanabi::PullingForceFieldModifier::new(vec![
            ForceFieldParam {
                position_or_direction: attractor2_position,
                max_radius: 1000000.0,
                min_radius: BALL_RADIUS * 6.0,
                mass: 3.0,
                force_type: ForceType::Linear,
                conform_to_sphere: true,
            },
            ForceFieldParam {
                position_or_direction: attractor1_position,
                max_radius: 1000000.0,
                min_radius: BALL_RADIUS * 6.0,
                mass: -0.5,
                force_type: ForceType::Quadratic,
                conform_to_sphere: false,
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
    mut effect: Query<(&mut ParticleEffect, &mut Transform)>,
    mouse_button_input: Res<Input<MouseButton>>,
    mouse_position: Res<MousePosition>,
) {
    let (mut effect, mut effect_transform) = effect.single_mut();

    if mouse_button_input.just_pressed(MouseButton::Left) {
        effect_transform.translation = mouse_position.position.extend(0.0);

        // Spawn the particles
        effect.maybe_spawner().unwrap().reset();
    }
}

#[derive(Default, Debug)]
pub struct MousePosition {
    pub position: Vec2,
}

fn record_mouse_events_system(
    mut cursor_moved_events: EventReader<CursorMoved>,
    mut mouse_position: ResMut<MousePosition>,
    mut windows: ResMut<Windows>,
    cam_transform_query: Query<&Transform, With<PerspectiveProjection>>,
) {
    for event in cursor_moved_events.iter() {
        let cursor_in_pixels = event.position;
        let window_size = Vec2::new(
            windows.get_primary_mut().unwrap().width(),
            windows.get_primary_mut().unwrap().height(),
        );

        let screen_position = cursor_in_pixels - window_size / 2.0;

        let cam_transform = cam_transform_query.iter().next().unwrap();

        // TODO: use bevy_mod_picking instead
        let cursor_vec4: Vec4 =
            cam_transform.compute_matrix() * screen_position.extend(0.0).extend(1.0) / 145.0;

        let cursor_pos = Vec2::new(cursor_vec4.x, cursor_vec4.y);
        mouse_position.position = cursor_pos;
    }
}
