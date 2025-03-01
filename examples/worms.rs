//! Worms
//!
//! Demonstrates the combined use of particle trails and child effects / GPU
//! spawn events. A first effect spawns worm "head" particles, which move
//! forward wiggling in a sine wave. Each such "head particle" emits at constant
//! rate a GPU spawn event for a second child effect. That child effect is a
//! trail following the "head particle", and forms the body of the worms.

use std::f32::consts::FRAC_PI_2;

use bevy::{
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
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

fn create_head_effect() -> EffectAsset {
    let writer = ExprWriter::new();

    // Init modifiers

    // Spawn the particles within a reasonably large box.
    let init_position_modifier = SetAttributeModifier::new(
        Attribute::POSITION,
        ((writer.rand(ValueType::Vector(VectorType::VEC3F)) + writer.lit(vec3(-0.5, -0.5, 0.0)))
            * writer.lit(vec3(16.0, 16.0, 0.0)))
        .expr(),
    );

    // Randomize the initial angle of the particle, storing it in the `F32_0`
    // scratch attribute. Each particle gets a unique angle.
    let init_angle_modifier = SetAttributeModifier::new(
        Attribute::F32_0,
        writer.lit(0.0).normal(writer.lit(1.0)).expr(),
    );

    // Give each particle a random opaque color.
    let init_color_modifier = SetAttributeModifier::new(
        Attribute::COLOR,
        (writer.rand(ValueType::Vector(VectorType::VEC4F)) * writer.lit(vec4(1.0, 1.0, 1.0, 0.0))
            + writer.lit(Vec4::W))
        .pack4x8unorm()
        .expr(),
    );

    // Give the particles a long lifetime.
    let init_age_modifier = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.0).expr());
    let init_lifetime_modifier =
        SetAttributeModifier::new(Attribute::LIFETIME, writer.lit(3.0).expr());

    // Update modifiers

    // Make the particle wiggle, following a sine wave.
    let set_velocity_modifier = SetAttributeModifier::new(
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

    // Spawn a trail of child body particles into the other effect
    let update_spawn_trail = EmitSpawnEventModifier {
        condition: EventEmitCondition::Always,
        count: 5,
        // We use channel #0; see EffectParent
        child_index: 0,
    };

    // Render modifiers

    // Set the particle size.
    let set_size_modifier = SetSizeModifier {
        size: Vec3::splat(0.4).into(),
    };

    // Make each particle round.
    let particle_texture_modifier = ParticleTextureModifier {
        texture_slot: writer.lit(0u32).expr(),
        sample_mapping: ImageSampleMapping::Modulate,
    };

    let mut module = writer.finish();
    module.add_texture_slot("shape");

    // Allocate room for 100 "head" particles (100 worms)
    EffectAsset::new(100, Spawner::rate(2.0.into()), module)
        .with_name("worms_heads")
        .init(init_position_modifier)
        .init(init_angle_modifier)
        .init(init_age_modifier)
        .init(init_lifetime_modifier)
        .init(init_color_modifier)
        .update(set_velocity_modifier)
        .update(update_spawn_trail)
        .render(set_size_modifier)
        .render(particle_texture_modifier)
}

fn create_body_effect() -> EffectAsset {
    let writer = ExprWriter::new();

    // Init modifiers

    // Particles inherit the position of their parent (the head particle of the
    // worm, from the other effect)
    let inherit_position_modifier = InheritAttributeModifier::new(Attribute::POSITION);

    // Particles use their parent's ID as ribbon ID. This "attaches" each trail
    // particle of this effect to the head particle of the other effect which
    // spawned it.
    let init_ribbon_id_modifier = SetAttributeModifier::new(
        Attribute::RIBBON_ID,
        writer.parent_attr(Attribute::ID).expr(),
    );

    // When using ribbons, particles need the AGE attribute.
    let init_age_modifier = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.0).expr());

    // Set a lifetime for the trail
    let init_lifetime_modifier =
        SetAttributeModifier::new(Attribute::LIFETIME, writer.lit(1.5).expr());

    // Render modifiers

    // Set the particle size.
    let set_size_modifier = SetSizeModifier {
        size: Vec3::splat(0.3).into(),
    };

    let module = writer.finish();

    // Allocate room for 500 trail particles
    EffectAsset::new(5000, Spawner::rate(0.5.into()), module)
        .with_name("worms_bodies")
        // Body particles don't move. No need to integrate anything (particles don't have any
        // VELOCITY attribute anyway, so that would generate a warning).
        .with_motion_integration(MotionIntegration::None)
        .init(inherit_position_modifier)
        .init(init_ribbon_id_modifier)
        .init(init_age_modifier)
        .init(init_lifetime_modifier)
        .render(set_size_modifier)
}

fn setup(
    mut commands: Commands,
    asset_server: ResMut<AssetServer>,
    mut effects: ResMut<Assets<EffectAsset>>,
) {
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

    let circle: Handle<Image> = asset_server.load("circle.png");

    let head_effect = create_head_effect();
    let head_effect = effects.add(head_effect);

    let head_entity = commands
        .spawn((
            Name::new("worms_heads"),
            ParticleEffectBundle {
                effect: ParticleEffect::new(head_effect),
                ..default()
            },
            EffectMaterial {
                images: vec![circle.clone()],
            },
        ))
        .id();

    let body_effect = create_body_effect();
    let body_effect = effects.add(body_effect);

    commands.spawn((
        Name::new("worms_bodies"),
        ParticleEffectBundle {
            effect: ParticleEffect::new(body_effect),
            ..default()
        },
        EffectParent::new(head_entity),
        // EffectMaterial {
        //     images: vec![circle],
        // },
    ));
}
