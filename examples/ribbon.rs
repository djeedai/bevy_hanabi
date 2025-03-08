//! Draw a "tracer" or a "trail" following an Entity.
//!
//! The emitter associated with the Entity is moved by the [move_head] system,
//! which simply assigns its [`Transform`]. Each frame some particle is spawned
//! at that new position, in global space. By decreasing the particle size over
//! its lifetime, and using a constant lifetime for all particles, the particles
//! appear to create a trail following the emitter's Transform. This is just an
//! illusion though; the particles don't move.
//!
//! To complete the illusion, all particles are assigned a same
//! [`Attribute::RIBBON_ID`], which causes them to be rendered as a single
//! continuous ribbon of quads, each linking the current particle to the
//! previous one (the first particle is skipped in ribbon mode).

use bevy::math::vec4;
use bevy::prelude::*;
use bevy::{
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
    math::vec3,
};
use bevy_hanabi::prelude::*;

mod utils;
use utils::*;

// These determine the shape of the Spirograph:
// https://en.wikipedia.org/wiki/Spirograph#Mathematical_basis
const K: f32 = 0.64;
const L: f32 = 0.384;

const TIME_SCALE: f32 = 6.5;
const SHAPE_SCALE: f32 = 25.0;
// Note: because Hanabi doesn't currently support position interpolation between
// frames, spawning more than 1 particle per frame in a ribbon is a pure waste;
// all particles spawned in a same frame spawn at the same position. So we
// assume up to 60 FPS here.
const RIBBON_SPAWN_RATE: f32 = 60.0;
const RIBBON_LIFETIME: f32 = 1.5;
// 60 particles / second * 1.5 seconds = up to 90 particles alive at once.
// Allocate a tiny bit more just to have some wiggle room.
const PARTICLE_CAPACITY: u32 = 100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::make_test_app("ribbon")
        .add_systems(Startup, setup)
        .add_systems(Update, move_head)
        .run();
    app_exit.into_result()
}

fn setup(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
    commands.spawn((
        Transform::from_translation(Vec3::new(0., 0., 50.)),
        Camera {
            hdr: true,
            clear_color: Color::BLACK.into(),
            ..default()
        },
        Camera3d::default(),
        Tonemapping::None,
        Bloom::default(),
    ));

    let writer = ExprWriter::new();

    let init_position_attr = SetAttributeModifier {
        attribute: Attribute::POSITION,
        value: writer.lit(Vec3::ZERO).expr(),
    };

    let init_age_attr = SetAttributeModifier {
        attribute: Attribute::AGE,
        value: writer.lit(0.0).expr(),
    };

    // For ribbons we generally want the lifetime to be a constant, so that
    // particles dying are at the end of the ribbon and particles spawning at
    // the beginning. Otherwise if a particle dies in the middle of the ribbon this
    // will generate some reordering and visually will look odd.
    let init_lifetime_attr = SetAttributeModifier {
        attribute: Attribute::LIFETIME,
        value: writer.lit(RIBBON_LIFETIME).expr(),
    };

    let init_size_attr = SetAttributeModifier {
        attribute: Attribute::SIZE,
        value: writer.lit(0.5).expr(),
    };

    // In this example the entire effect is a single ribbon/trail, so all its
    // particle are connected. This means they all share the same RIBBON_ID, which
    // can be any value.
    let init_ribbon_id = SetAttributeModifier {
        attribute: Attribute::RIBBON_ID,
        value: writer.lit(0u32).expr(),
    };

    let render_color = ColorOverLifetimeModifier {
        gradient: Gradient::linear(vec4(3.0, 0.0, 0.0, 1.0), vec4(3.0, 0.0, 0.0, 0.0)),
    };

    let spawner = Spawner::rate(RIBBON_SPAWN_RATE.into());

    let effect = EffectAsset::new(PARTICLE_CAPACITY, spawner, writer.finish())
        .with_name("ribbon")
        // Disable motion integration; particles stay where spawned, the illusion of movement is
        // created by the ribbon and the particles spawning and dying, as well as the fact the
        // emitter itself moves (we set its Transform in move_head() below).
        .with_motion_integration(MotionIntegration::None)
        // Detach particles from the emitter, they should keep a fixed position in global space even
        // when we move the emitter.
        .with_simulation_space(SimulationSpace::Global)
        .init(init_position_attr)
        .init(init_age_attr)
        .init(init_lifetime_attr)
        .init(init_size_attr)
        .init(init_ribbon_id)
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::linear(Vec3::ONE, Vec3::ZERO),
            ..default()
        })
        .render(render_color);

    let effect = effects.add(effect);

    commands.spawn((ParticleEffect::new(effect), Name::new("ribbon")));
}

fn move_head(mut query: Query<&mut Transform, With<ParticleEffect>>, timer: Res<Time>) {
    for mut transform in query.iter_mut() {
        let time = timer.elapsed_secs() * TIME_SCALE;
        let pos = vec3(
            (1.0 - K) * (time.clone().cos()) + (L * K) * (((1.0 - K) / K) * time.clone()).cos(),
            (1.0 - K) * (time.clone().sin()) - (L * K) * (((1.0 - K) / K) * time.clone()).sin(),
            0.0,
        ) * SHAPE_SCALE;
        transform.translation = pos;
    }
}
