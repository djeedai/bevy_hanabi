//! Portal
//!
//! An example demonstrating the use of the `InitVelocityTangentModifier` to
//! create a kind of portal effect where particles turn around a circle and
//! appear to be ejected from it. The `OrientAlongVelocityModifier` paired with
//! an elongated particle size gives the appearance of sparks.
//!
//! The addition of some gravity and drag, combined with a careful choice of
//! lifetime, give a subtle effect of particles appearing to fall down right
//! before they disappear, like sparkles fading away.

use bevy::{
    core_pipeline::{
        bloom::BloomSettings, clear_color::ClearColorConfig, tonemapping::Tonemapping,
    },
    log::LogPlugin,
    prelude::*,
};
// use bevy_inspector_egui::quick::WorldInspectorPlugin;

use bevy_hanabi::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    App::default()
        .add_plugins(DefaultPlugins.set(LogPlugin {
            level: bevy::log::Level::WARN,
            filter: "bevy_hanabi=warn,portal=trace".to_string(),
        }))
        .add_systems(Update, bevy::window::close_on_esc)
        .add_plugins(HanabiPlugin)
        // Have to wait for update.
        // .add_plugins(WorldInspectorPlugin::default())
        .add_systems(Startup, setup)
        .run();

    Ok(())
}

fn setup(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0., 0., 25.)),
            camera: Camera {
                hdr: true,
                ..default()
            },
            camera_3d: Camera3d {
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            tonemapping: Tonemapping::None,
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
    size_gradient1.add_key(0.3, Vec2::new(0.2, 0.02));
    size_gradient1.add_key(1.0, Vec2::splat(0.0));

    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = InitAttributeModifier::new(Attribute::AGE, age);

    // Give a bit of variation by randomizing the lifetime per particle
    let lifetime = writer.lit(0.6).uniform(writer.lit(1.3)).expr();
    let init_lifetime = InitAttributeModifier::new(Attribute::LIFETIME, lifetime);

    // Add drag to make particles slow down a bit after the initial acceleration
    let drag = writer.lit(2.).expr();
    let update_drag = LinearDragModifier::new(drag);

    let effect1 = effects.add(
        EffectAsset::new(32768, Spawner::rate(5000.0.into()), writer.finish())
            .with_name("portal")
            .init(InitPositionCircleModifier {
                center: Vec3::ZERO,
                axis: Vec3::Z,
                radius: 4.,
                dimension: ShapeDimension::Surface,
            })
            .init(init_age)
            .init(init_lifetime)
            .update(update_drag)
            .update(RadialAccelModifier::constant(Vec3::ZERO, -6.0))
            .update(TangentAccelModifier::constant(Vec3::ZERO, Vec3::Z, 30.))
            .render(ColorOverLifetimeModifier {
                gradient: color_gradient1,
            })
            .render(SizeOverLifetimeModifier {
                gradient: size_gradient1,
                screen_space_size: false,
            })
            .render(OrientAlongVelocityModifier),
    );

    commands.spawn((
        Name::new("portal"),
        ParticleEffectBundle {
            effect: ParticleEffect::new(effect1),
            transform: Transform::IDENTITY,
            ..Default::default()
        },
    ));
}
