//! Worms
//!
//! Demonstrates a simple use of particle trails.

use std::f32::consts::FRAC_PI_2;

use bevy::{
    core_pipeline::{bloom::BloomSettings, tonemapping::Tonemapping},
    math::{vec3, vec4},
    prelude::*,
};
use bevy_hanabi::prelude::*;

mod utils;
use utils::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::make_test_app("worms")
        .add_systems(Startup, setup)
        .run();
    app_exit.into_result()
}

fn setup(
    mut commands: Commands,
    asset_server: ResMut<AssetServer>,
    mut effects: ResMut<Assets<EffectAsset>>,
) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0., 0., 25.)),
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

    let circle: Handle<Image> = asset_server.load("circle.png");

    let writer = ExprWriter::new();

    // Init modifiers

    // Spawn the particles within a reasonably large box.
    let set_initial_position_modifier = SetAttributeModifier::new(
        Attribute::POSITION,
        ((writer.rand(ValueType::Vector(VectorType::VEC3F)) + writer.lit(vec3(-0.5, -0.5, 0.0)))
            * writer.lit(vec3(16.0, 16.0, 0.0)))
        .expr(),
    );

    // Randomize the initial angle of the particle, storing it in the `F32_0`
    // scratch attribute.`
    let set_initial_angle_modifier = SetAttributeModifier::new(
        Attribute::F32_0,
        writer.lit(0.0).normal(writer.lit(1.0)).expr(),
    );

    // Give each particle a random opaque color.
    let set_color_modifier = SetAttributeModifier::new(
        Attribute::COLOR,
        (writer.rand(ValueType::Vector(VectorType::VEC4F)) * writer.lit(vec4(1.0, 1.0, 1.0, 0.0))
            + writer.lit(Vec4::W))
        .pack4x8unorm()
        .expr(),
    );

    // Give the particles a long lifetime.
    let set_age_modifier = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.0).expr());
    let set_lifetime_modifier =
        SetAttributeModifier::new(Attribute::LIFETIME, writer.lit(10.0).expr());

    // Update modifiers

    // Make the particle wiggle, following a sine wave.
    let main_set_velocity_modifier = SetAttributeModifier::new(
        Attribute::VELOCITY,
        WriterExpr::sin(
            writer.lit(vec3(1.0, 1.0, 0.0))
                * (writer.attr(Attribute::F32_0)
                    + (writer.time() * writer.lit(5.0)).sin() * writer.lit(1.0))
                + writer.lit(vec3(0.0, FRAC_PI_2, 0.0)),
        )
        .mul(writer.lit(5.0))
        .expr(),
    );
    let trail_set_velocity_modifier =
        SetAttributeModifier::new(Attribute::VELOCITY, writer.lit(Vec3::ZERO).expr());

    // Render modifiers

    // Set the particle size.
    let main_set_size_modifier = SetSizeModifier {
        size: Vec2::splat(0.4).into(),
    };
    let trail_set_size_modifier = SetSizeModifier {
        size: Vec2::splat(0.3).into(),
    };

    // Make each particle round.
    let particle_texture_modifier = ParticleTextureModifier {
        texture_slot: writer.lit(0u32).expr(),
        sample_mapping: ImageSampleMapping::Modulate,
    };

    let mut module = writer.finish();
    module.add_texture("shape");

    // Allocate room for 32,768 trail particles. Give each particle a 5-particle
    // trail, and spawn a new trail particle every ⅛ of a second.
    let effect = effects.add(
        EffectAsset::new(5000, Spawner::rate(CpuValue::Single(5.0)), module)
            .with_trails(5000 * 5, 1.0 / 8.0, 1.0, 0)
            .with_name("worms")
            .init_groups(set_initial_position_modifier, ParticleGroupSet::single(0))
            .init_groups(set_initial_angle_modifier, ParticleGroupSet::single(0))
            .init_groups(set_age_modifier, ParticleGroupSet::single(0))
            .init_groups(set_lifetime_modifier, ParticleGroupSet::single(0))
            .init_groups(set_color_modifier, ParticleGroupSet::single(0))
            .update_groups(main_set_velocity_modifier, ParticleGroupSet::single(0))
            .update_groups(trail_set_velocity_modifier, ParticleGroupSet::single(1))
            .render_groups(main_set_size_modifier, ParticleGroupSet::single(0))
            .render_groups(trail_set_size_modifier, ParticleGroupSet::single(1))
            .render(particle_texture_modifier),
    );

    commands.spawn((
        Name::new("worms"),
        ParticleEffectBundle {
            effect: ParticleEffect::new(effect),
            transform: Transform::IDENTITY,
            ..default()
        },
        EffectMaterial {
            images: vec![circle],
        },
    ));
}
