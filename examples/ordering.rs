//! Ordering
//!
//! This example demonstrates occluding particle effects behind partially transparent objects.
use bevy::{
    core_pipeline::{bloom::BloomSettings, tonemapping::Tonemapping},
    log::LogPlugin,
    prelude::*,
};

use bevy_hanabi::prelude::*;
#[cfg(feature = "examples_world_inspector")]
use bevy_inspector_egui::quick::WorldInspectorPlugin;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut app = App::default();
    app.add_plugins(
        DefaultPlugins
            .set(LogPlugin {
                level: bevy::log::Level::WARN,
                filter: "bevy_hanabi=warn,firework=trace".to_string(),
                update_subscriber: None,
            })
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "🎆 Hanabi — ordering".to_string(),
                    ..default()
                }),
                ..default()
            }),
    )
    .add_systems(Update, bevy::window::close_on_esc)
    .add_plugins(HanabiPlugin);

    #[cfg(feature = "examples_world_inspector")]
    app.add_plugins(WorldInspectorPlugin::default());

    app.add_systems(Startup, setup).run();

    Ok(())
}

fn make_firework() -> EffectAsset {
    let mut color_gradient1 = Gradient::new();
    color_gradient1.add_key(0.0, Vec4::new(4.0, 4.0, 4.0, 1.0));
    color_gradient1.add_key(0.1, Vec4::new(4.0, 4.0, 0.0, 1.0));
    color_gradient1.add_key(0.9, Vec4::new(4.0, 0.0, 0.0, 1.0));
    color_gradient1.add_key(1.0, Vec4::new(4.0, 0.0, 0.0, 0.0));

    let mut size_gradient1 = Gradient::new();
    size_gradient1.add_key(0.0, Vec2::splat(0.1));
    size_gradient1.add_key(0.3, Vec2::splat(0.1));
    size_gradient1.add_key(1.0, Vec2::splat(0.0));

    let writer = ExprWriter::new();

    // Give a bit of variation by randomizing the age per particle. This will
    // control the starting color and starting size of particles.
    let age = writer.lit(0.).uniform(writer.lit(0.2)).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    // Give a bit of variation by randomizing the lifetime per particle
    let lifetime = writer.lit(0.8).uniform(writer.lit(1.2)).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    // Add constant downward acceleration to simulate gravity
    let accel = writer.lit(Vec3::Y * -8.).expr();
    let update_accel = AccelModifier::new(accel);

    // Add drag to make particles slow down a bit after the initial explosion
    let drag = writer.lit(5.).expr();
    let update_drag = LinearDragModifier::new(drag);

    let init_pos = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(2.).expr(),
        dimension: ShapeDimension::Volume,
    };

    // Give a bit of variation by randomizing the initial speed
    let init_vel = SetVelocitySphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        speed: (writer.rand(ScalarType::Float) * writer.lit(20.) + writer.lit(60.)).expr(),
    };

    EffectAsset::new(
        // 2k lead particles, with 32 trail particles each
        vec![2048, 2048 * 32],
        Spawner::burst(2048.0.into(), 2.0.into()),
        writer.finish(),
    )
    .with_name("firework")
    .init(init_pos)
    .init(init_vel)
    .init(init_age)
    .init(init_lifetime)
    .update(update_drag)
    .update(update_accel)
    .render(ColorOverLifetimeModifier {
        gradient: color_gradient1,
    })
    .render(SizeOverLifetimeModifier {
        gradient: size_gradient1,
        screen_space_size: false,
    })
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0., 0., 50.)),
            camera: Camera {
                hdr: true,
                clear_color: Color::BLACK.into(),
                ..default()
            },
            tonemapping: Tonemapping::None,
            ..default()
        },
        BloomSettings::default(),
    ));

    let effect1 = effects.add(make_firework());

    commands.spawn((
        Name::new("firework"),
        ParticleEffectBundle {
            effect: ParticleEffect::new(effect1),
            transform: Transform {
                translation: Vec3::Z,
                ..default()
            },
            ..default()
        },
    ));

    // Background square at origin.
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(Rectangle {
            half_size: Vec2 { x: 0.5, y: 0.5 },
        })),
        material: materials.add(StandardMaterial {
            base_color: Color::RED,
            alpha_mode: bevy::pbr::AlphaMode::Blend,
            ..default()
        }),
        transform: Transform {
            scale: Vec3::splat(10.),
            ..default()
        },
        ..default()
    });

    // Blue square in front of particles with AlphaMode::Blend.
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(Rectangle {
            half_size: Vec2 { x: 0.5, y: 0.5 },
        })),
        material: materials.add(StandardMaterial {
            base_color: Color::BLUE,
            alpha_mode: bevy::pbr::AlphaMode::Blend,
            ..default()
        }),
        transform: Transform {
            translation: Vec3::Y * 5. + Vec3::Z * 25.,
            scale: Vec3::splat(5.),
            ..default()
        },
        ..default()
    });

    // Green square in front of particles with AlphaMode::Opaque.
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(Rectangle {
            half_size: Vec2 { x: 0.5, y: 0.5 },
        })),
        material: materials.add(StandardMaterial {
            base_color: Color::GREEN,
            alpha_mode: bevy::pbr::AlphaMode::Opaque,
            ..default()
        }),
        transform: Transform {
            translation: Vec3::Y * -5. + Vec3::Z * 25.,
            scale: Vec3::splat(5.),
            ..default()
        },
        ..default()
    });
}
