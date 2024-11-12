//! A circle bobs up and down in the water, spawning bubbles when in the water.
//!
//! This example demonstrates the use of [`Spawner::set_active()`] to enable or
//! disable particle spawning, under the control of the application. This is
//! similar to the `spawn_on_command.rs` example, where [`Spawner::reset()`] is
//! used instead to spawn a single burst of particles.
//!
//! A small vertical acceleration simulate a pseudo-buoyancy making the bubbles
//! slowly move upward toward the surface. The example uses a
//! [`KillAabbModifier`] to ensure the bubble particles never escape water, and
//! are despawned when reaching the surface.

use bevy::{core_pipeline::tonemapping::Tonemapping, prelude::*, render::camera::ScalingMode};
use bevy_hanabi::prelude::*;

mod utils;
use bevy_text::TextFont;
use utils::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::make_test_app("activate")
        .add_systems(Startup, setup)
        .add_systems(Update, update)
        .run();
    app_exit.into_result()
}

#[derive(Component)]
struct StatusText;

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
    let mut projection = OrthographicProjection::default_3d();
    projection.scaling_mode = ScalingMode::FixedVertical {
        viewport_height: 2.,
    };
    projection.scale = 1.0;
    commands.spawn((
        Transform::from_translation(Vec3::Z),
        Projection::Orthographic(projection),
        Camera3d::default(),
        Tonemapping::None,
    ));

    // Blue rectangle mesh for "water"
    commands.spawn((
        Transform::from_xyz(0.0, -2.0, 0.0),
        Mesh3d(meshes.add(Rectangle {
            half_size: Vec2::splat(2.0),
        })),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: utils::COLOR_BLUE,
            unlit: true,
            ..Default::default()
        })),
        Name::new("water"),
    ));

    let mut ball = commands.spawn((
        Mesh3d(meshes.add(Sphere { radius: 0.05 })),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            unlit: true,
            ..Default::default()
        })),
        Ball { velocity_y: 1.0 },
        Name::new("ball"),
    ));

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

    let buoyancy = writer.lit(Vec3::Y * 0.2).expr();
    let update_buoyancy = AccelModifier::new(buoyancy);

    // Kill particles getting out of water
    let center = writer.lit(Vec3::Y * -2.02).expr();
    let half_size = writer.lit(Vec3::splat(2.0)).expr();
    let allow_zone = KillAabbModifier::new(center, half_size);

    let mut module = writer.finish();

    let round = RoundModifier::constant(&mut module, 1.0);

    let effect = effects.add(
        EffectAsset::new(32768, spawner, module)
            .with_name("activate")
            .init(init_pos)
            .init(init_vel)
            .init(init_age)
            .init(init_lifetime)
            .update(update_buoyancy)
            .update(allow_zone)
            .render(SetSizeModifier {
                size: Vec3::splat(0.02).into(),
            })
            .render(ColorOverLifetimeModifier { gradient })
            .render(round),
    );

    ball.with_children(|node| {
        node.spawn(ParticleEffectBundle::new(effect))
            .insert(Name::new("effect"));
    });

    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(30.),
            right: Val::Px(30.),
            ..default()
        },
        Text::new("Active"),
        TextFont {
            font_size: 60.0,
            ..default()
        },
        TextColor(Color::WHITE.into()),
        StatusText,
    ));
}

fn update(
    mut q_balls: Query<(&mut Ball, &mut Transform, &Children)>,
    mut q_spawner: Query<&mut EffectInitializers>,
    mut q_text: Query<&mut Text, With<StatusText>>,
    time: Res<Time>,
) {
    const ACCELERATION: f32 = 1.0;
    for (mut ball, mut transform, children) in q_balls.iter_mut() {
        let accel = if transform.translation.y >= 0.0 {
            -ACCELERATION
        } else {
            ACCELERATION
        };
        ball.velocity_y += accel * time.delta_secs();
        transform.translation.y += ball.velocity_y * time.delta_secs();

        // Note: On first frame where the effect spawns, EffectSpawner is spawned during
        // CoreSet::PostUpdate, so will not be available yet. Ignore for a frame
        // if so.
        let is_active = transform.translation.y < 0.0;
        if let Ok(mut spawner) = q_spawner.get_mut(children[0]) {
            spawner.set_active(is_active);
        }

        let mut text = q_text.single_mut();
        text.0 = (if is_active { "Active" } else { "Inactive" }).to_string();
    }
}
