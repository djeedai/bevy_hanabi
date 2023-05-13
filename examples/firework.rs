//! Firework
//!
//! This example demonstrate the use of the [`LinearDragModifier`] to slow down
//! particles over time. Combined with an HDR camera with Bloom, the example
//! renders a firework explosion.
//!
//! The firework effect is composed of several key elements:
//! - An HDR camera with [`BloomSettings`] to ensure the particles "glow".
//! - Use of a [`ColorOverLifetimeModifier`] with a [`Gradient`] made of colors
//!   outside the \[0:1\] range, to ensure bloom has an effect.
//! - [`InitVelocitySphereModifier`] with a reasonably large initial speed for
//!   particles, and [`LinearDragModifier`] to quickly slow them down. This is
//!   the core of the "explosion" effect.
//! - An [`AccelModifier`] to pull particles down once they slow down, for
//!   increased realism. This is a subtle effect, but of importance.

use bevy::{
    core_pipeline::{bloom::BloomSettings, clear_color::ClearColorConfig},
    log::LogPlugin,
    prelude::*,
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

use bevy_hanabi::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    App::default()
        .add_plugins(DefaultPlugins.set(LogPlugin {
            level: bevy::log::Level::WARN,
            filter: "bevy_hanabi=warn,firework=trace".to_string(),
        }))
        .add_system(bevy::window::close_on_esc)
        .add_plugin(HanabiPlugin)
        .add_plugin(WorldInspectorPlugin::default())
        .add_startup_system(setup)
        .run();

    Ok(())
}

fn setup(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0., 0., 50.)),
            camera: Camera {
                hdr: true,
                ..default()
            },
            camera_3d: Camera3d {
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            ..default()
        },
        BloomSettings::default(),
    ));

    let mut color_gradient1 = Gradient::new();
    color_gradient1.add_key(0.0, Vec4::new(4.0, 4.0, 4.0, 1.0));
    color_gradient1.add_key(0.1, Vec4::new(4.0, 4.0, 0.0, 1.0));
    color_gradient1.add_key(0.9, Vec4::new(4.0, 0.0, 0.0, 1.0));
    color_gradient1.add_key(1.0, Vec4::new(4.0, 0.0, 0.0, 0.0));

    let mut size_gradient1 = Gradient::new();
    size_gradient1.add_key(0.0, Vec2::splat(0.1));
    size_gradient1.add_key(0.3, Vec2::splat(0.1));
    size_gradient1.add_key(1.0, Vec2::splat(0.0));

    let effect1 = effects.add(
        EffectAsset {
            name: "firework".to_string(),
            capacity: 32768,
            spawner: Spawner::burst(2500.0.into(), 2.0.into()),
            ..Default::default()
        }
        .init(InitPositionSphereModifier {
            center: Vec3::ZERO,
            radius: 2.,
            dimension: ShapeDimension::Volume,
        })
        .init(InitVelocitySphereModifier {
            center: Vec3::ZERO,
            // Give a bit of variation by randomizing the initial speed
            speed: Value::Uniform((65., 75.)),
        })
        .init(InitLifetimeModifier {
            // Give a bit of variation by randomizing the lifetime per particle
            lifetime: Value::Uniform((0.8, 1.2)),
        })
        .init(InitAgeModifier {
            // Give a bit of variation by randomizing the age per particle. This will control the
            // starting color and starting size of particles.
            age: Value::Uniform((0.0, 0.2)),
        })
        .update(LinearDragModifier::constant(5.))
        .update(AccelModifier::constant(Vec3::new(0., -8., 0.)))
        .render(ColorOverLifetimeModifier {
            gradient: color_gradient1,
        })
        .render(SizeOverLifetimeModifier {
            gradient: size_gradient1,
        }),
    );

    commands.spawn((
        Name::new("firework"),
        ParticleEffectBundle {
            effect: ParticleEffect::new(effect1),
            transform: Transform::IDENTITY,
            ..Default::default()
        },
    ));
}
