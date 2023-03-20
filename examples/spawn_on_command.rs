//! A circle bounces around in a box and spawns particles
//! when it hits the wall.
use bevy::{
    log::LogPlugin,
    math::Vec3Swizzles,
    prelude::*,
    render::{
        camera::{Projection, ScalingMode},
        render_resource::WgpuFeatures,
        settings::WgpuSettings,
        RenderPlugin,
    },
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

use bevy_hanabi::prelude::*;

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
        .add_system(bevy::window::close_on_esc)
        .add_plugin(HanabiPlugin)
        .add_plugin(WorldInspectorPlugin::default())
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
    let mut camera = Camera3dBundle::default();
    let mut projection = OrthographicProjection::default();
    projection.scaling_mode = ScalingMode::FixedVertical(2.);
    projection.scale = 1.2;
    camera.transform.translation.z = projection.far / 2.0;
    camera.projection = Projection::Orthographic(projection);
    commands.spawn(camera);

    commands
        .spawn(PbrBundle {
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
        .spawn(PbrBundle {
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

    let spawner = Spawner::once(30.0.into(), false);
    let effect = effects.add(
        EffectAsset {
            name: "Impact".into(),
            capacity: 32768,
            spawner,
            ..Default::default()
        }
        .with_property("my_color", graph::Value::Uint(0xFFFFFFFF))
        .init(InitPositionSphereModifier {
            center: Vec3::ZERO,
            radius: BALL_RADIUS,
            dimension: ShapeDimension::Surface,
        })
        .init(InitVelocitySphereModifier {
            center: Vec3::ZERO,
            speed: 0.2.into(),
        })
        .init(InitLifetimeModifier {
            lifetime: 5_f32.into(),
        })
        .init(InitAttributeModifier {
            attribute: Attribute::COLOR,
            value: "my_color".into(),
        })
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::constant(Vec2::splat(0.05)),
        }),
    );

    commands
        .spawn(ParticleEffectBundle::new(effect).with_spawner(spawner))
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

            // Pick a random particle color
            let r = rand::random::<u8>();
            let g = rand::random::<u8>();
            let b = rand::random::<u8>();
            let color = 0xFF000000u32 | (b as u32) << 16 | (g as u32) << 8 | (r as u32);
            effect.set_property("my_color", color.into());

            // Spawn the particles
            effect.maybe_spawner().unwrap().reset();
        }
    }
}
