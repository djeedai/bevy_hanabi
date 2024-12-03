//! Portal
//!
//! An example demonstrating the use of the `TangentAccelModifier` to create a
//! kind of portal effect where particles turn around a circle and appear to be
//! ejected from it.
//!
//! The `OrientMode::AlongVelocity` of the `OrientModifier` paired with an
//! elongated particle size gives the appearance of sparks.
//!
//! The addition of some gravity and drag, combined with a careful choice of
//! lifetime, give a subtle effect of particles appearing to fall down right
//! before they disappear, like sparkles fading away.

use bevy::{
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
    prelude::*,
};
use bevy_hanabi::prelude::*;

mod utils;
use utils::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::make_test_app("portal")
        .add_systems(Startup, setup)
        .run();
    app_exit.into_result()
}

fn setup(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
    commands.spawn((
        Transform::from_translation(Vec3::new(0., 0., 25.)),
        Camera {
            hdr: true,
            clear_color: Color::BLACK.into(),
            ..default()
        },
        Camera3d::default(),
        Tonemapping::None,
        Bloom::default(),
    ));

    let mut color_gradient1 = Gradient::new();
    color_gradient1.add_key(0.0, Vec4::new(4.0, 4.0, 4.0, 1.0));
    color_gradient1.add_key(0.1, Vec4::new(4.0, 4.0, 0.0, 1.0));
    color_gradient1.add_key(0.9, Vec4::new(4.0, 0.0, 0.0, 1.0));
    color_gradient1.add_key(1.0, Vec4::new(4.0, 0.0, 0.0, 0.0));

    let mut size_gradient1 = Gradient::new();
    size_gradient1.add_key(0.3, Vec3::new(0.2, 0.02, 1.0));
    size_gradient1.add_key(1.0, Vec3::splat(0.0));

    let writer = ExprWriter::new();

    let init_pos = SetPositionCircleModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        axis: writer.lit(Vec3::Z).expr(),
        radius: writer.lit(4.).expr(),
        dimension: ShapeDimension::Surface,
    };

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    // Give a bit of variation by randomizing the lifetime per particle
    let lifetime = writer.lit(0.6).uniform(writer.lit(1.3)).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    // Add drag to make particles slow down a bit after the initial acceleration
    let drag = writer.lit(2.).expr();
    let update_drag = LinearDragModifier::new(drag);

    let mut module = writer.finish();

    let tangent_accel = TangentAccelModifier::constant(&mut module, Vec3::ZERO, Vec3::Z, 30.);

    let effect1 = effects.add(
        EffectAsset::new(16384, Spawner::rate(5000.0.into()), module)
            .with_name("portal")
            .init(init_pos)
            .init(init_age)
            .init(init_lifetime)
            .update(update_drag)
            .update(tangent_accel)
            .render(ColorOverLifetimeModifier {
                gradient: color_gradient1,
            })
            .render(SizeOverLifetimeModifier {
                gradient: size_gradient1,
                screen_space_size: false,
            })
            .render(OrientModifier::new(OrientMode::AlongVelocity)),
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
