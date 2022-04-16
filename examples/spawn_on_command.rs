//! A circle bounces around in a box and spawns particles
//! when it hits the wall.
//!
use bevy::{
    math::Vec3Swizzles,
    prelude::*,
    render::{render_resource::WgpuFeatures, settings::WgpuSettings},
};
//use bevy_inspector_egui::WorldInspectorPlugin;

use bevy_hanabi::*;

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
        .add_plugin(HanabiPlugin)
        //.add_plugin(WorldInspectorPlugin::new())
        .add_startup_system(setup)
        .add_system(update)
        .run();

    Ok(())
}

#[derive(Component)]
struct Ball {
    velocity: Vec2,
}

const BOX_SIZE: f32 = 2.0;
const BALL_RADIUS: f32 = 0.05;

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = OrthographicCameraBundle::new_3d();
    camera.orthographic_projection.scale = 1.2;
    camera.transform.translation.z = camera.orthographic_projection.far / 2.0;
    commands.spawn_bundle(camera);

    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Quad {
                size: Vec2::splat(BOX_SIZE),
                ..Default::default()
            })),
            material: materials.add(StandardMaterial {
                base_color: Color::BLACK,
                unlit: true,
                ..Default::default()
            }),
            ..Default::default()
        })
        .insert(Name::new("box"));

    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::UVSphere {
                sectors: 32,
                stacks: 2,
                radius: BALL_RADIUS,
            })),
            material: materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                ..Default::default()
            }),
            ..Default::default()
        })
        .insert(Ball {
            velocity: Vec2::new(1.0, 2f32.sqrt()),
        })
        .insert(Name::new("ball"));

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
            speed: 0.2.into(),
            dimension: ShapeDimension::Surface,
            ..Default::default()
        })
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::constant(Vec2::splat(0.05)),
        })
        .render(ColorOverLifetimeModifier { gradient }),
    );

    commands
        .spawn_bundle(ParticleEffectBundle::new(effect).with_spawner(spawner))
        .insert(Name::new("effect"));
}

fn update(
    mut balls: Query<(&mut Ball, &mut Transform)>,
    mut effect: Query<(&mut ParticleEffect, &mut Transform), Without<Ball>>,
    time: Res<Time>,
) {
    const HALF_SIZE: f32 = BOX_SIZE / 2.0 - BALL_RADIUS;

    let (mut effect, mut effect_transform) = effect.single_mut();

    for (mut ball, mut transform) in balls.iter_mut() {
        let mut pos = transform.translation.xy() + ball.velocity * time.delta_seconds();
        let mut collision = false;

        for (coord, vel_coord) in pos.as_mut().iter_mut().zip(ball.velocity.as_mut()) {
            while *coord < -HALF_SIZE || *coord > HALF_SIZE {
                if *coord < -HALF_SIZE {
                    *coord = 2.0 * -HALF_SIZE - *coord;
                } else if *coord > HALF_SIZE {
                    *coord = 2.0 * HALF_SIZE - *coord;
                }
                *vel_coord *= -1.0;
                collision = true;
            }
        }

        transform.translation = pos.extend(transform.translation.z);

        if collision {
            // This isn't the most accurate place to spawn the particle effect,
            // but this is just for demonstration, so whatever.
            effect_transform.translation = transform.translation;
            // Spawn the particles
            effect.maybe_spawner().unwrap().reset();
        }
    }
}
