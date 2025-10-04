//! Worms
//!
//! Demonstrates the combined use of particle trails and child effects / GPU
//! spawn events. A first effect spawns worm "head" particles, which move
//! forward wiggling in a sine wave. Each such "head particle" emits at constant
//! rate a GPU spawn event for a second child effect. That child effect is a
//! trail following the "head particle", and forms the body of the worms.

use std::f32::consts::FRAC_PI_2;

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    math::{vec3, vec4},
    post_process::bloom::Bloom,
    prelude::*,
    render::view::Hdr,
};
use bevy_hanabi::prelude::*;

mod utils;
use utils::*;

const DEMO_DESC: &str = include_str!("worms.txt");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::DemoApp::new("worms")
        .with_desc(DEMO_DESC)
        .build()
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

    // Store a unique value per head particle. This will be used as the ribbon ID of
    // the trail.
    let init_ribbon_id = SetAttributeModifier::new(
        Attribute::U32_0,
        writer.attr(Attribute::PARTICLE_COUNTER).expr(),
    );

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
        count: writer.lit(5u32).expr(),
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
    EffectAsset::new(100, SpawnerSettings::rate(2.0.into()), module)
        .with_name("worms_heads")
        .init(init_position_modifier)
        .init(init_angle_modifier)
        .init(init_age_modifier)
        .init(init_lifetime_modifier)
        .init(init_color_modifier)
        .init(init_ribbon_id)
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

    // We need to figure out a shared RIBBON_ID for all body particles "attached" to
    // a given head one from the other effect. The obvious choice is the parent's
    // ID, which is a pseudo-attribute unique to each parent particle. This almost
    // works, but breaks when the parent particle dies and gets recycled in the same
    // frame (with the same ID). So instead of that ID we use the same kind of idea,
    // reading a unique value from the parent, but we use one which is truely
    // unique, stored per parent particle in U32_0.
    let init_ribbon_id_modifier = SetAttributeModifier::new(
        Attribute::RIBBON_ID,
        writer.parent_attr(Attribute::U32_0).expr(),
    );

    // When using ribbons, particles need the AGE attribute.
    let init_age_modifier = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.0).expr());

    // Set a lifetime for the trail
    let init_lifetime_modifier =
        SetAttributeModifier::new(Attribute::LIFETIME, writer.lit(1.5).expr());

    // The trail inherits the color of its parent head
    let init_color_modifier = SetAttributeModifier::new(
        Attribute::COLOR,
        writer.parent_attr(Attribute::COLOR).expr(),
    );

    // Render modifiers

    // Set the particle size.
    let set_size_modifier = SetSizeModifier {
        size: Vec3::splat(0.3).into(),
    };

    let module = writer.finish();

    // Allocate room for 500 trail particles
    EffectAsset::new(5000, SpawnerSettings::rate(0.5.into()), module)
        .with_name("worms_bodies")
        // Body particles don't move. No need to integrate anything (particles don't have any
        // VELOCITY attribute anyway, so that would generate a warning).
        .with_motion_integration(MotionIntegration::None)
        .init(inherit_position_modifier)
        .init(init_ribbon_id_modifier)
        .init(init_age_modifier)
        .init(init_lifetime_modifier)
        .init(init_color_modifier)
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
            clear_color: Color::BLACK.into(),
            ..default()
        },
        Camera3d::default(),
        Hdr,
        Tonemapping::None,
        Bloom::default(),
    ));

    let circle: Handle<Image> = asset_server.load("circle.png");

    let head_effect = create_head_effect();
    let head_effect = effects.add(head_effect);

    let head_entity = commands
        .spawn((
            Name::new("worms_heads"),
            ParticleEffect::new(head_effect),
            EffectMaterial {
                images: vec![circle.clone()],
            },
        ))
        .id();

    let body_effect = create_body_effect();
    let body_effect = effects.add(body_effect);

    commands.spawn((
        Name::new("worms_bodies"),
        ParticleEffect::new(body_effect),
        EffectParent::new(head_entity),
        // EffectMaterial {
        //     images: vec![circle],
        // },
    ));
}
