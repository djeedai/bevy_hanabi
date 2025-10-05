//! Firework
//!
//! This example demonstrates the use of the [`LinearDragModifier`] to slow down
//! particles over time. Combined with an HDR camera with Bloom, the example
//! renders a firework explosion.
//!
//! The firework effect is composed of several key elements:
//! - An HDR camera with [`Bloom`] to ensure the particles "glow".
//! - Use of a [`ColorOverLifetimeModifier`] with a [`Gradient`] made of colors
//!   outside the \[0:1\] range, to ensure bloom has an effect.
//! - [`SetVelocitySphereModifier`] with a reasonably large initial speed for
//!   particles, and [`LinearDragModifier`] to quickly slow them down. This is
//!   the core of the "explosion" effect.
//! - An [`AccelModifier`] to pull particles down once they slow down, for
//!   increased realism. This is a subtle effect, but of importance.
//!
//! The particles also have a trail. The trail particles are stitched together
//! to form an arc using [`EffectAsset::with_ribbons`].

use bevy::{
    core_pipeline::tonemapping::Tonemapping, post_process::bloom::Bloom, prelude::*,
    render::view::Hdr,
};
use bevy_hanabi::prelude::*;

mod utils;
use utils::*;

const DEMO_DESC: &str = include_str!("firework.txt");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::DemoApp::new("firework")
        .with_desc(DEMO_DESC)
        .build()
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

    // Store a random color per particle, which will be inherited by the spark ones
    // on explosion. We don't store it in Attribute::COLOR otherwise it's going to
    // affect the color of the rocket particle itself.
    let rgb = writer.rand(VectorType::VEC3F) * writer.lit(0.9) + writer.lit(0.1);
    let color = rgb.vec4_xyz_w(writer.lit(1.)).pack4x8unorm();
    let init_trails_color = SetAttributeModifier::new(Attribute::U32_0, color.expr());

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
        count: writer.lit(5u32).expr(),
        // We use channel #0 for those sparkle trail events; see EffectParent
        child_index: 0,
    };

    // When the rocket particle dies, it "explodes" and spawns the actual firework
    // particles. To achieve this, when a rocket particle dies, it emits spawn
    // events for its child(ren) effects.
    let update_spawn_on_die = EmitSpawnEventModifier {
        condition: EventEmitCondition::OnDie,
        count: writer.lit(1000u32).expr(),
        // We use channel #1 for the explosion itself; see EffectParent
        child_index: 1,
    };

    let spawner = SpawnerSettings::rate((1., 3.).into());

    EffectAsset::new(32, spawner, writer.finish())
        .with_name("rocket")
        .init(init_pos)
        .init(init_vel)
        .init(init_age)
        .init(init_lifetime)
        .init(init_trails_color)
        .update(update_drag)
        .update(update_accel)
        .update(update_spawn_trail)
        .update(update_spawn_on_die)
        .render(ColorOverLifetimeModifier {
            gradient: bevy_hanabi::Gradient::constant(Vec4::ONE),
            blend: ColorBlendMode::Overwrite,
            mask: ColorBlendMask::RGBA,
        })
        .render(SizeOverLifetimeModifier {
            gradient: bevy_hanabi::Gradient::constant(Vec3::ONE * 0.1),
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
    let vel = writer.rand(VectorType::VEC3F);
    let vel = vel * writer.lit(2.) - writer.lit(1.); // remap [0:1] to [-1:1]
    let vel = vel.normalized();
    let speed = writer.lit(1.); //.uniform(writer.lit(4.));
    let vel = (vel * speed).expr();
    let init_vel = SetAttributeModifier::new(Attribute::VELOCITY, vel);

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    // Give a bit of variation by randomizing the lifetime per particle
    let lifetime = writer.lit(0.2).expr(); //.uniform(writer.lit(0.4)).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    // Add constant downward acceleration to simulate gravity
    let accel = writer.lit(Vec3::Y * -16.).expr();
    let update_accel = AccelModifier::new(accel);

    // Add drag to make particles slow down as they ascend
    let drag = writer.lit(4.).expr();
    let update_drag = LinearDragModifier::new(drag);

    // The (CPU) spawner is unused
    let spawner = SpawnerSettings::default();

    let mut color_gradient = bevy_hanabi::Gradient::new();
    color_gradient.add_key(0.0, Vec4::new(4.0, 4.0, 4.0, 1.0));
    color_gradient.add_key(0.8, Vec4::new(4.0, 4.0, 4.0, 1.0));
    color_gradient.add_key(1.0, Vec4::new(4.0, 4.0, 4.0, 0.0));

    EffectAsset::new(1000, spawner, writer.finish())
        .with_name("sparkle_trail")
        .init(init_pos)
        .init(init_vel)
        .init(init_age)
        .init(init_lifetime)
        .update(update_drag)
        .update(update_accel)
        .render(ColorOverLifetimeModifier {
            gradient: color_gradient,
            blend: ColorBlendMode::Modulate,
            mask: ColorBlendMask::RGBA,
        })
        .render(SizeOverLifetimeModifier {
            gradient: bevy_hanabi::Gradient::constant(Vec3::ONE * 0.02),
            screen_space_size: false,
        })
}

/// Create the effect for the trails coming out of the rocket explosion. They
/// spawn in burst each time a rocket particle dies (= "explodes").
fn create_trails_effect() -> EffectAsset {
    let writer = ExprWriter::new();

    // Inherit the start position from the parent effect (the rocket particle)
    let init_pos = InheritAttributeModifier::new(Attribute::POSITION);

    // Pull the color from the parent's Attribute::U32_0.
    let init_color = SetAttributeModifier::new(
        Attribute::COLOR,
        writer.parent_attr(Attribute::U32_0).expr(),
    );

    // The velocity is random in any direction
    let center = writer.attr(Attribute::POSITION);
    let speed = writer.lit(40.).uniform(writer.lit(60.));
    let dir = writer
        .rand(VectorType::VEC3F)
        .mul(writer.lit(2.0))
        .sub(writer.lit(1.0))
        .normalized();
    let init_vel = SetAttributeModifier::new(Attribute::VELOCITY, (center + dir * speed).expr());

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

    // Orient particle toward its velocity to create a cheap 1-particle trail
    let orient = OrientModifier::new(OrientMode::AlongVelocity);

    // The (CPU) spawner is unused
    let spawner = SpawnerSettings::default();

    let mut color_gradient = bevy_hanabi::Gradient::new();
    color_gradient.add_key(0.0, Vec4::new(4.0, 4.0, 4.0, 1.0));
    color_gradient.add_key(0.6, Vec4::new(4.0, 4.0, 4.0, 1.0));
    color_gradient.add_key(1.0, Vec4::new(4.0, 4.0, 4.0, 0.0));

    EffectAsset::new(10000, spawner, writer.finish())
        .with_name("trail")
        .init(init_pos)
        .init(init_vel)
        .init(init_age)
        .init(init_lifetime)
        .init(init_color)
        .update(update_drag)
        .update(update_accel)
        .render(ColorOverLifetimeModifier {
            gradient: color_gradient,
            blend: ColorBlendMode::Modulate,
            mask: ColorBlendMask::RGBA,
        })
        .render(SizeOverLifetimeModifier {
            gradient: bevy_hanabi::Gradient::constant(Vec3::new(0.2, 0.05, 0.05)),
            screen_space_size: false,
        })
        .render(orient)
}

fn setup(mut commands: Commands, effects: ResMut<Assets<EffectAsset>>) {
    // Camera
    commands.spawn((
        Transform::from_translation(Vec3::new(0., 20., 50.)),
        Camera3d::default(),
        Camera {
            clear_color: Color::BLACK.into(),
            ..default()
        },
        Hdr,
        Tonemapping::None,
        Bloom {
            intensity: 0.5,
            ..default()
        },
    ));

    create_effect(commands, effects);
}

fn create_effect(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
    // Rocket
    let rocket_effect = effects.add(create_rocket_effect());
    let rocket_entity = commands
        .spawn((Name::new("rocket"), ParticleEffect::new(rocket_effect)))
        .id();

    // Sparkle trail
    let sparkle_trail_effect = effects.add(create_sparkle_trail_effect());
    commands.spawn((
        Name::new("sparkle_trail"),
        ParticleEffect::new(sparkle_trail_effect),
        // Set the rocket effect as parent. This gives access to the rocket effect's particles,
        // which in turns allows inheriting their position (and other attributes if needed).
        EffectParent::new(rocket_entity),
    ));

    // Trails
    let trails_effect = effects.add(create_trails_effect());
    commands.spawn((
        Name::new("trails"),
        ParticleEffect::new(trails_effect),
        // Set the rocket effect as parent. This gives access to the rocket effect's particles,
        // which in turns allows inheriting their position (and other attributes if needed).
        EffectParent::new(rocket_entity),
    ));
}
