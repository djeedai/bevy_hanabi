//! Draw "tracers" or "trails", each following an Entity.
//!
//! The emitter associated with an Entity is moved by the [move_head] system,
//! which simply assigns its [`Transform`]. Each frame some particle is spawned
//! at that new position, in global space. By decreasing the particle size over
//! its lifetime, and using a constant lifetime for all particles, the particles
//! appear to create a trail following the emitter's Transform. This is just an
//! illusion though; the particles don't move.
//!
//! To complete the illusion, all particles of a same trail are assigned a same
//! [`Attribute::RIBBON_ID`], which causes them to be rendered as a single
//! continuous ribbon of quads, each linking the current particle to the
//! previous one (the first particle is skipped in ribbon mode).
//!
//! This example also demonstrates managing several ribbon effects at runtime:
//! ribbons are spawned and despawned continuously over time (see
//! [spawn_ribbons] / [recycle_ribbons]) rather than once on startup, so several
//! trails are alive at the same time.

use bevy::camera::Hdr;
use bevy::math::vec4;
use bevy::prelude::*;
use bevy::{core_pipeline::tonemapping::Tonemapping, math::vec3, post_process::bloom::Bloom};
use bevy_hanabi::prelude::*;

mod utils;
use utils::*;

// These determine the shape of the Spirograph:
// https://en.wikipedia.org/wiki/Spirograph#Mathematical_basis
const K: f32 = 0.64;
const L: f32 = 0.384;

#[derive(Clone)]
enum ShapeConfig {
    Spirograph { k: f32, l: f32 },
    Lissajou { a: f32, b: f32 },
}

#[derive(Component, Clone)]
struct Shape {
    pub config: ShapeConfig,
    pub time_scale: f32,
    pub shape_scale: Vec3,
    /// Per-instance offset along the curve, so each spawned ribbon starts at a
    /// different position instead of on top of the previous one.
    pub phase: f32,
}

impl Shape {
    pub fn tick(&mut self, time: f32) -> Vec3 {
        let time = (time + self.phase) * self.time_scale;
        let pos = match self.config {
            ShapeConfig::Spirograph { k, l } => vec3(
                (1.0 - k) * (time.cos()) + (l * k) * (((1.0 - k) / k) * time).cos(),
                (1.0 - k) * (time.sin()) - (l * k) * (((1.0 - k) / k) * time).sin(),
                0.0,
            ),
            ShapeConfig::Lissajou { a, b } => vec3((a * time).cos(), (b * time).sin(), 0.0),
        };
        pos * self.shape_scale
    }
}

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

const DEMO_DESC: &str = include_str!("ribbon.txt");

const RIBBON_SPAWN_INTERVAL: f32 = 2.0;
const RIBBON_DESPAWN_AFTER: f32 = 10.0;
// Curve offset added per spawned ribbon, in seconds (before time_scale), so
// successive ribbons appear at distinct positions instead of overlapping.
const RIBBON_PHASE_STEP: f32 = 0.7;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::DemoApp::new("ribbon")
        .with_desc(DEMO_DESC)
        .build()
        .add_systems(Startup, setup)
        .add_systems(Update, (spawn_ribbons, recycle_ribbons, move_head))
        .run();
    app_exit.into_result()
}

/// Drives the continuous ribbon spawn/despawn cycle.
#[derive(Resource)]
struct RibbonSpawner {
    effect: Handle<EffectAsset>,
    shapes: Vec<Shape>,
    timer: Timer,
    next: usize,
}

#[derive(Component)]
struct DespawnAfter(Timer);

fn setup(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
    commands.spawn((
        Transform::from_translation(Vec3::new(0., 0., 50.)),
        Camera {
            clear_color: Color::BLACK.into(),
            ..default()
        },
        Camera3d::default(),
        Hdr,
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

    // Each effect is a single ribbon/trail, so all its particles are connected.
    // This means they all share the same RIBBON_ID, which can be any value.
    let init_ribbon_id = SetAttributeModifier {
        attribute: Attribute::RIBBON_ID,
        value: writer.lit(0u32).expr(),
    };

    let render_color = ColorOverLifetimeModifier::new(bevy_hanabi::Gradient::linear(
        vec4(3.0, 0.0, 0.0, 1.0),
        vec4(0.0, 0.0, 3.0, 0.0),
    ));

    let spawner = SpawnerSettings::rate(RIBBON_SPAWN_RATE.into());

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
            gradient: bevy_hanabi::Gradient::linear(Vec3::ONE, Vec3::ZERO),
            ..default()
        })
        .render(render_color);

    // The ribbons share a single effect asset; each spawned instance is an
    // independent effect, only the immutable asset is reused.
    commands.insert_resource(RibbonSpawner {
        effect: effects.add(effect),
        timer: Timer::from_seconds(RIBBON_SPAWN_INTERVAL, TimerMode::Repeating),
        next: 0,
        shapes: vec![
            Shape {
                config: ShapeConfig::Spirograph { k: K, l: L },
                time_scale: TIME_SCALE,
                shape_scale: Vec3::ONE * SHAPE_SCALE,
                phase: 0.0, // set per-instance in spawn_ribbons()
            },
            Shape {
                config: ShapeConfig::Lissajou { a: 3., b: 4. },
                time_scale: 2.,
                shape_scale: Vec3::ONE * 18.,
                phase: 0.0, // set per-instance in spawn_ribbons()
            },
        ],
    });
}

/// Spawn one ribbon every [`RIBBON_SPAWN_INTERVAL`], cycling through the
/// registered shapes. Each ribbon is despawned a few seconds later by
/// [`recycle_ribbons`], keeping a rolling set of ribbons alive.
fn spawn_ribbons(time: Res<Time>, mut commands: Commands, mut spawner: ResMut<RibbonSpawner>) {
    if !spawner.timer.tick(time.delta()).just_finished() {
        return;
    }

    let mut shape = spawner.shapes[spawner.next % spawner.shapes.len()].clone();
    // Offset each spawn along the curve so successive ribbons don't land on top
    // of the previous one (move_head() drives all ribbons from the same clock).
    shape.phase = spawner.next as f32 * RIBBON_PHASE_STEP;
    spawner.next = spawner.next.wrapping_add(1);

    commands.spawn((
        ParticleEffect::new(spawner.effect.clone()),
        shape,
        DespawnAfter(Timer::from_seconds(RIBBON_DESPAWN_AFTER, TimerMode::Once)),
    ));
}

/// Recycle ribbons at the end of their [`DespawnAfter`] lifetime: stop emitting
/// for the final [`RIBBON_LIFETIME`] so the existing particles age out and fade
/// the tail away (via the size/color-over-lifetime modifiers), then despawn the
/// now-empty entity.
fn recycle_ribbons(
    time: Res<Time>,
    mut commands: Commands,
    mut query: Query<(Entity, &mut DespawnAfter, Option<&mut EffectSpawner>)>,
) {
    for (entity, mut despawn_after, spawner) in query.iter_mut() {
        despawn_after.0.tick(time.delta());
        if despawn_after.0.just_finished() {
            commands.entity(entity).despawn();
        } else if despawn_after.0.remaining_secs() <= RIBBON_LIFETIME {
            // Stop spawning new particles; the live ones keep aging and fade out.
            if let Some(mut spawner) = spawner {
                spawner.active = false;
            }
        }
    }
}

fn move_head(
    mut query: Query<(&mut Shape, &mut Transform), With<ParticleEffect>>,
    timer: Res<Time>,
) {
    for (mut shape, mut transform) in query.iter_mut() {
        let time = timer.elapsed_secs();
        let pos = shape.tick(time);
        transform.translation = pos;
    }
}
