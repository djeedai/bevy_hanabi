//! A circle bobs up and down in the water, spawning square bubbles when in the
//! water.
//!
//! This example demonstrates the use of [`Spawner::set_active()`] to enable or
//! disable particle spawning, under the control of the application. This is
//! similar to the `spawn_on_command.rs` example, where [`Spawner::reset()`] is
//! used instead to spawn a single burst of particles.

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    log::LogPlugin,
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
                    filter: "bevy_hanabi=warn,activate=trace".to_string(),
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings.into(),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "🎆 Hanabi — activate".to_string(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(HanabiPlugin)
        .add_plugins(WorldInspectorPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (bevy::window::close_on_esc, update))
        .run();

    Ok(())
}

#[derive(Component)]
struct Ball {
    velocity_y: f32,
}

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
    projection.scaling_mode = ScalingMode::FixedVertical(2.);
    projection.scale = 1.0;
    camera.transform.translation.z = projection.far / 2.0;
    camera.projection = Projection::Orthographic(projection);
    commands.spawn(camera);

    commands
        .spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Quad {
                size: Vec2::splat(4.0),
                ..Default::default()
            })),
            material: materials.add(StandardMaterial {
                base_color: Color::BLUE,
                unlit: true,
                ..Default::default()
            }),
            transform: Transform::from_xyz(0.0, -2.0, 0.0),
            ..Default::default()
        })
        .insert(Name::new("water"));

    let mut ball = commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::UVSphere {
            sectors: 32,
            stacks: 2,
            radius: 0.05,
        })),
        material: materials.add(StandardMaterial {
            base_color: Color::WHITE,
            unlit: true,
            ..Default::default()
        }),
        ..Default::default()
    });
    ball.insert(Ball { velocity_y: 1.0 })
        .insert(Name::new("ball"));

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(0.5, 0.5, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::new(0.5, 0.5, 1.0, 0.0));

    let spawner = Spawner::rate(30.0.into()).with_starts_active(false);

    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(5.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let init_pos = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(0.05).expr(),
        dimension: ShapeDimension::Surface,
    };

    let init_vel = SetVelocitySphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        speed: writer.lit(0.1).expr(),
    };

    let effect = effects.add(
        EffectAsset::new(32768, spawner, writer.finish())
            .with_name("activate")
            .init(init_pos)
            .init(init_vel)
            .init(init_age)
            .init(init_lifetime)
            .render(SizeOverLifetimeModifier {
                gradient: Gradient::constant(Vec2::splat(0.02)),
                screen_space_size: false,
            })
            .render(ColorOverLifetimeModifier { gradient }),
    );

    ball.with_children(|node| {
        node.spawn(ParticleEffectBundle::new(effect).with_spawner(spawner))
            .insert(Name::new("effect"));
    });
}

fn update(
    mut q_balls: Query<(&mut Ball, &mut Transform, &Children)>,
    mut q_spawner: Query<&mut EffectSpawner>,
    time: Res<Time>,
) {
    const ACCELERATION: f32 = 1.0;
    for (mut ball, mut transform, children) in q_balls.iter_mut() {
        let accel = if transform.translation.y >= 0.0 {
            -ACCELERATION
        } else {
            ACCELERATION
        };
        ball.velocity_y += accel * time.delta_seconds();
        transform.translation.y += ball.velocity_y * time.delta_seconds();

        // Note: On first frame where the effect spawns, EffectSpawner is spawned during
        // CoreSet::PostUpdate, so will not be available yet. Ignore for a frame
        // if so.
        if let Ok(mut spawner) = q_spawner.get_mut(children[0]) {
            spawner.set_active(transform.translation.y < 0.0);
        }
    }
}
