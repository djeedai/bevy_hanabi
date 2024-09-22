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
//! - [`SetVelocitySphereModifier`] with a reasonably large initial speed for
//!   particles, and [`LinearDragModifier`] to quickly slow them down. This is
//!   the core of the "explosion" effect.
//! - An [`AccelModifier`] to pull particles down once they slow down, for
//!   increased realism. This is a subtle effect, but of importance.
//!
//! The particles also have a trail, created with the [`CloneModifier`]. The
//! trail particles are stitched together to form an arc using the
//! [`RibbonModifier`].

use bevy::{
    core_pipeline::{bloom::BloomSettings, tonemapping::Tonemapping},
    prelude::*,
};
use bevy_hanabi::prelude::*;

mod utils;
use utils::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::make_test_app("firework")
        .add_systems(Startup, setup)
        .run();
    app_exit.into_result()
}

/// Create the effect for the rocket itself, which spawns infrequently and
/// rapidly raises until it explodes (dies).
fn create_rocket_effect() -> EffectAsset {
    let writer = ExprWriter::new();

    // Always start from the same launch point
    let init_pos = SetPositionCircleModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        axis: writer.lit(Vec3::Y).expr(),
        radius: writer.lit(30.).expr(),
        dimension: ShapeDimension::Volume,
    };

    // Give a bit of variation by randomizing the initial speed and direction
    let zero = writer.lit(0.);
    let y = writer.lit(140.).uniform(writer.lit(160.));
    let v = zero.clone().vec3(y, zero);
    let init_vel = SetAttributeModifier::new(Attribute::VELOCITY, v.expr());

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    // Give a bit of variation by randomizing the lifetime per particle
    let lifetime = writer.lit(0.8).uniform(writer.lit(1.2)).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    // Add constant downward acceleration to simulate gravity
    let accel = writer.lit(Vec3::Y * -16.).expr();
    let update_accel = AccelModifier::new(accel);

    // Add drag to make particles slow down as they ascend
    let drag = writer.lit(4.).expr();
    let update_drag = LinearDragModifier::new(drag);

    // As the rocket particle rises in the air, it leaves behind a trail of
    // sparkles. To achieve this, the particle emits spawn events for its child
    // effect.
    let update_spawn_trail = EmitSpawnEventModifier {
        condition: EventEmitCondition::Always,
        count: 5,
        // We use channel #0 for those sparkle trail events; see EffectParent
        channel_index: 0,
    };

    // When the rocket particle dies, it "explodes" and spawns the actual firework
    // particles. To achieve this, when a rocket particle dies, it emits spawn
    // events for its child(ren) effects.
    let update_spawn_on_die = EmitSpawnEventModifier {
        condition: EventEmitCondition::OnDie,
        count: 100,
        // We use channel #1 for the explosion itself; see EffectParent
        channel_index: 1,
    };

    EffectAsset::new(vec![32], Spawner::rate(1.0.into()), writer.finish())
        .with_name("rocket")
        .init(init_pos)
        .init(init_vel)
        .init(init_age)
        .init(init_lifetime)
        .update(update_drag)
        .update(update_accel)
        .update(update_spawn_trail)
        .update(update_spawn_on_die)
        .render(ColorOverLifetimeModifier {
            gradient: Gradient::constant(Vec4::ONE),
        })
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::constant(Vec2::ONE * 0.1),
            screen_space_size: false,
        })
}

/// Create the effect for the sparkle trail coming out of the rocket as it
/// raises in the air.
fn create_sparkle_trail_effect() -> EffectAsset {
    let writer = ExprWriter::new();

    // Inherit the start position from the parent effect (the rocket particle)
    let init_pos = InheritAttributeModifier::new(Attribute::POSITION);

    // The velocity is random in any direction
    let vel = writer.rand(VectorType::VEC3F).normalized();
    let speed = writer.lit(1.).uniform(writer.lit(4.));
    let vel = (vel * speed).expr();
    let init_vel = SetAttributeModifier::new(Attribute::VELOCITY, vel);

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    // Give a bit of variation by randomizing the lifetime per particle
    let lifetime = writer.lit(0.2).uniform(writer.lit(0.4)).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    // Add constant downward acceleration to simulate gravity
    let accel = writer.lit(Vec3::Y * -16.).expr();
    let update_accel = AccelModifier::new(accel);

    // Add drag to make particles slow down as they ascend
    let drag = writer.lit(4.).expr();
    let update_drag = LinearDragModifier::new(drag);

    // The (CPU) spawner is unused
    let spawner = Spawner::default();

    let mut color_gradient = Gradient::new();
    color_gradient.add_key(0.0, Vec4::new(4.0, 4.0, 4.0, 1.0));
    color_gradient.add_key(0.8, Vec4::new(4.0, 4.0, 0.0, 1.0));
    color_gradient.add_key(1.0, Vec4::new(4.0, 0.0, 0.0, 0.0));

    EffectAsset::new(vec![1000], spawner, writer.finish())
        .with_name("sparkle_trail")
        .init(init_pos)
        .init(init_vel)
        .init(init_age)
        .init(init_lifetime)
        .update(update_drag)
        .update(update_accel)
        .render(ColorOverLifetimeModifier {
            gradient: color_gradient,
        })
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::constant(Vec2::ONE * 0.02),
            screen_space_size: false,
        })
}

/// Create the effect for the trails coming out of the rocket explosion. They
/// spawn in burst each time a rocket particle dies (= "explodes").
fn create_trails_effect() -> EffectAsset {
    let writer = ExprWriter::new();

    // Inherit the start position from the parent effect (the rocket particle)
    let init_pos = InheritAttributeModifier::new(Attribute::POSITION);

    // The velocity is random in any direction
    let center = writer.attr(Attribute::POSITION).expr();
    let speed = writer.lit(300.).uniform(writer.lit(400.)).expr();
    let init_vel = SetVelocitySphereModifier { center, speed };

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    // Give a bit of variation by randomizing the lifetime per particle
    let lifetime = writer.lit(0.8).uniform(writer.lit(1.2)).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    // Add constant downward acceleration to simulate gravity
    let accel = writer.lit(Vec3::Y * -16.).expr();
    let update_accel = AccelModifier::new(accel);

    // Add drag to make particles slow down as they ascend
    let drag = writer.lit(4.).expr();
    let update_drag = LinearDragModifier::new(drag);

    // The (CPU) spawner is unused
    let spawner = Spawner::default();

    let mut color_gradient = Gradient::new();
    color_gradient.add_key(0.0, Vec4::new(4.0, 4.0, 4.0, 1.0));
    color_gradient.add_key(0.1, Vec4::new(0.0, 4.0, 4.0, 1.0));
    color_gradient.add_key(0.6, Vec4::new(0.0, 0.0, 4.0, 1.0));
    color_gradient.add_key(1.0, Vec4::new(0.0, 0.0, 4.0, 0.0));

    EffectAsset::new(vec![10000], spawner, writer.finish())
        .with_name("trail")
        .init(init_pos)
        .init(init_vel)
        .init(init_age)
        .init(init_lifetime)
        .update(update_drag)
        .update(update_accel)
        .render(ColorOverLifetimeModifier {
            gradient: color_gradient,
        })
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::constant(Vec2::ONE * 0.3),
            screen_space_size: false,
        })
}

fn setup(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0., 20., 50.)),
            camera: Camera {
                hdr: true,
                clear_color: Color::BLACK.into(),
                ..default()
            },
            tonemapping: Tonemapping::None,
            ..default()
        },
        BloomSettings {
            intensity: 0.2,
            ..default()
        },
    ));

    // Rocket
    let rocket_effect = effects.add(create_rocket_effect());
    let rocket_entity = commands
        .spawn((
            Name::new("rocket"),
            ParticleEffectBundle {
                effect: ParticleEffect::new(rocket_effect),
                transform: Transform::IDENTITY,
                ..Default::default()
            },
        ))
        .id();

    // Sparkle trail
    let sparkle_trail_effect = effects.add(create_sparkle_trail_effect());
    commands.spawn((
        Name::new("sparkle_trail"),
        ParticleEffectBundle {
            effect: ParticleEffect::new(sparkle_trail_effect),
            ..Default::default()
        },
        // Set the rocket effect as parent. This gives access to the rocket effect's particles,
        // which in turns allows inheriting their position (and other attributes if
        // needed).
        EffectParent {
            entity: rocket_entity,
            channel_index: 0,
        },
    ));

    // Trails
    let trails_effect = effects.add(create_trails_effect());
    commands.spawn((
        Name::new("trails"),
        ParticleEffectBundle {
            effect: ParticleEffect::new(trails_effect),
            ..Default::default()
        },
        // Set the rocket effect as parent. This gives access to the rocket effect's particles,
        // which in turns allows inheriting their position (and other attributes if needed).
        EffectParent {
            entity: rocket_entity,
            channel_index: 1,
        },
    ));
}
