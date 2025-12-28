//! Vertical lightning bolt striking from sky to ground with impact sparks.
//!
//! This example demonstrates procedural geometry using [`ExprWriter`] to create
//! a jagged lightning bolt as a ribbon. Each particle's position is computed
//! from a deterministic hash based on its index and a random seed, creating
//! sharp angular segments that look like electricity.
//!
//! The bolt uses [`Attribute::RIBBON_ID`] to connect particles into a
//! continuous strip. A parabolic weight tapers the jitter at both endpoints,
//! ensuring the bolt always starts and ends at fixed positions.
//!
//! An impact burst effect spawns radial sparks when the lightning strikes.

use bevy::{
    core_pipeline::tonemapping::Tonemapping, post_process::bloom::Bloom, prelude::*,
    render::view::Hdr,
};
use bevy_hanabi::prelude::*;

mod utils;
use utils::AppExitIntoResult;

const PARTICLES_PER_BOLT: u32 = 40;
const BOLT_LIFETIME: f32 = 0.3;
const BOLT_LENGTH: f32 = 30.0;
/// Maximum horizontal deviation from the central axis, in world units.
const MAX_SPREAD: f32 = 1.5;
const BURST_INTERVAL: f32 = 1.5;
const GROUND_Y: f32 = -15.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::DemoApp::new("lightning")
        .with_desc(include_str!("lightning.txt"))
        .build()
        .add_systems(Startup, setup)
        .add_systems(Update, update_lightning_timers)
        .run()
        .into_result()
}

#[derive(Component)]
struct LightningTimer {
    timer: Timer,
}

#[derive(Component)]
struct ImpactEffect;

fn setup(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
    commands.spawn((
        Transform::from_translation(Vec3::new(0., 0., 60.)).looking_at(Vec3::ZERO, Vec3::Y),
        Camera3d::default(),
        Camera {
            clear_color: Color::srgb(0.02, 0.02, 0.04).into(),
            ..default()
        },
        Hdr,
        Tonemapping::None,
        Bloom {
            intensity: 0.3,
            ..default()
        },
    ));

    let bolt_handle = effects.add(create_bolt_effect());
    let impact_handle = effects.add(create_impact_effect());

    // Start timer near completion so first strike happens quickly
    let mut timer = Timer::from_seconds(BURST_INTERVAL, TimerMode::Repeating);
    timer.set_elapsed(std::time::Duration::from_secs_f32(BURST_INTERVAL - 0.1));

    commands.spawn((
        ParticleEffect::new(bolt_handle),
        EffectProperties::default(),
        LightningTimer { timer },
        Name::new("lightning_bolt"),
    ));

    commands.spawn((
        Transform::from_translation(Vec3::new(0., GROUND_Y, 0.)),
        ParticleEffect::new(impact_handle),
        EffectProperties::default(),
        ImpactEffect,
        Name::new("impact_burst"),
    ));
}

fn create_bolt_effect() -> EffectAsset {
    let writer = ExprWriter::new();

    // Seed randomizes the bolt shape each strike
    let wave_seed = writer.add_property("wave_seed", 0.0.into());
    let wave_seed_int =
        ((writer.prop(wave_seed) + writer.lit(100.0)) * writer.lit(1000.0)).cast(ScalarType::Uint);

    let particle_index = writer.attr(Attribute::PARTICLE_COUNTER) % writer.lit(PARTICLES_PER_BOLT);
    let cell_size = writer.lit(4u32);
    let cell_id = particle_index.clone() / cell_size.clone();

    // Hash function to generate deterministic jitter per grid cell
    let get_control_point = |id: WriterExpr| {
        let hash = |mult: u32, modulus: u32| {
            (id.clone() * writer.lit(mult) + wave_seed_int.clone() * writer.lit(67891u32))
                % writer.lit(modulus)
        };
        let jitter = hash(12345, 10111) % writer.lit(3u32);
        let x_rnd =
            hash(54321, 10111).cast(ScalarType::Float) / writer.lit(5055.5) - writer.lit(1.0);
        let z_rnd =
            hash(98765, 10111).cast(ScalarType::Float) / writer.lit(5055.5) - writer.lit(1.0);
        (jitter, x_rnd, z_rnd)
    };

    // Catmull-Rom style interpolation between 3 control points
    let id0 = cell_id.clone().max(writer.lit(1u32)) - writer.lit(1u32);
    let id1 = cell_id.clone();
    let id2 = cell_id.clone() + writer.lit(1u32);

    let (j0, xr0, zr0) = get_control_point(id0.clone());
    let (j1, xr1, zr1) = get_control_point(id1.clone());
    let (j2, xr2, zr2) = get_control_point(id2.clone());

    let p0 = id0 * cell_size.clone() + j0;
    let p1 = id1 * cell_size.clone() + j1;
    let p2 = id2 * cell_size.clone() + j2;

    let is_after_p1 = p1
        .clone()
        .cast(ScalarType::Float)
        .step(particle_index.clone().cast(ScalarType::Float));

    let start_p = p0
        .clone()
        .cast(ScalarType::Float)
        .mix(p1.clone().cast(ScalarType::Float), is_after_p1.clone());
    let end_p = p1
        .cast(ScalarType::Float)
        .mix(p2.cast(ScalarType::Float), is_after_p1.clone());

    let start_xr = xr0.mix(xr1.clone(), is_after_p1.clone());
    let end_xr = xr1.mix(xr2, is_after_p1.clone());

    let start_zr = zr0.mix(zr1.clone(), is_after_p1.clone());
    let end_zr = zr1.mix(zr2, is_after_p1.clone());

    let dist = (end_p.clone() - start_p.clone()).max(writer.lit(1.0));
    let progress = (particle_index.clone().cast(ScalarType::Float) - start_p) / dist;

    let x_jitter = start_xr.mix(end_xr, progress.clone()) * writer.lit(MAX_SPREAD);
    let z_jitter = start_zr.mix(end_zr, progress.clone()) * writer.lit(MAX_SPREAD * 0.5);

    let total_progress = particle_index.clone().cast(ScalarType::Float)
        / writer.lit((PARTICLES_PER_BOLT - 1) as f32);
    let y_top = writer.lit(GROUND_Y + BOLT_LENGTH);
    let y_pos = y_top - total_progress.clone() * writer.lit(BOLT_LENGTH);

    // Parabolic falloff: max jitter at center, zero at endpoints
    let weight =
        writer.lit(4.0) * total_progress.clone() * (writer.lit(1.0) - total_progress.clone());

    let position = (x_jitter * weight.clone()).vec3(y_pos, z_jitter * weight.clone());

    // Tiny age offset ensures stable ribbon ordering (particles spawned same frame)
    let init_age = particle_index.clone().cast(ScalarType::Float) * writer.lit(0.0001);
    let init_ribbon_id =
        (writer.attr(Attribute::PARTICLE_COUNTER) / writer.lit(PARTICLES_PER_BOLT)).expr();
    let init_lifetime = writer.lit(BOLT_LIFETIME).expr();
    let init_size = (writer.lit(0.08) * (weight + writer.lit(0.1))).expr();

    EffectAsset::new(
        1024,
        SpawnerSettings::once((PARTICLES_PER_BOLT as f32).into()),
        writer.finish(),
    )
    .with_name("lightning_bolt")
    .with_motion_integration(MotionIntegration::None)
    .init(SetAttributeModifier::new(
        Attribute::POSITION,
        position.expr(),
    ))
    .init(SetAttributeModifier::new(Attribute::AGE, init_age.expr()))
    .init(SetAttributeModifier::new(
        Attribute::LIFETIME,
        init_lifetime,
    ))
    .init(SetAttributeModifier::new(
        Attribute::RIBBON_ID,
        init_ribbon_id,
    ))
    .init(SetAttributeModifier::new(Attribute::SIZE, init_size))
    .render(ColorOverLifetimeModifier {
        gradient: {
            let mut g = bevy_hanabi::Gradient::new();
            g.add_key(0.0, Vec4::new(20.0, 22.0, 30.0, 1.0));
            g.add_key(0.2, Vec4::new(12.0, 16.0, 28.0, 1.0));
            g.add_key(0.6, Vec4::new(6.0, 10.0, 22.0, 0.8));
            g.add_key(1.0, Vec4::new(0.0, 0.0, 10.0, 0.0));
            g
        },
        blend: ColorBlendMode::Overwrite,
        mask: ColorBlendMask::RGBA,
    })
}

fn create_impact_effect() -> EffectAsset {
    let writer = ExprWriter::new();

    let angle = writer.rand(ScalarType::Float) * writer.lit(std::f32::consts::TAU);
    let speed = writer.lit(15.0) + writer.rand(ScalarType::Float) * writer.lit(25.0);
    let velocity = (angle.clone().cos() * speed.clone()).vec3(
        writer.lit(5.0) + writer.rand(ScalarType::Float) * writer.lit(15.0),
        angle.sin() * speed,
    );

    let init_pos = writer.lit(Vec3::ZERO).expr();
    let init_vel = velocity.expr();
    let init_lifetime = (writer.lit(0.3) + writer.rand(ScalarType::Float) * writer.lit(0.4)).expr();
    let init_size = (writer.lit(0.1) + writer.rand(ScalarType::Float) * writer.lit(0.15)).expr();
    let drag = writer.lit(2.0).expr();
    let gravity = writer.lit(Vec3::new(0., -40., 0.)).expr();

    EffectAsset::new(512, SpawnerSettings::once(80.0.into()), writer.finish())
        .with_name("impact_burst")
        .init(SetAttributeModifier::new(Attribute::POSITION, init_pos))
        .init(SetAttributeModifier::new(Attribute::VELOCITY, init_vel))
        .init(SetAttributeModifier::new(
            Attribute::LIFETIME,
            init_lifetime,
        ))
        .init(SetAttributeModifier::new(Attribute::SIZE, init_size))
        .update(LinearDragModifier::new(drag))
        .update(AccelModifier::new(gravity))
        .render(ColorOverLifetimeModifier {
            gradient: {
                let mut g = bevy_hanabi::Gradient::new();
                g.add_key(0.0, Vec4::new(25.0, 28.0, 35.0, 1.0));
                g.add_key(0.15, Vec4::new(18.0, 20.0, 30.0, 1.0));
                g.add_key(0.4, Vec4::new(8.0, 12.0, 25.0, 0.8));
                g.add_key(0.7, Vec4::new(3.0, 5.0, 18.0, 0.4));
                g.add_key(1.0, Vec4::new(0.0, 0.0, 8.0, 0.0));
                g
            },
            blend: ColorBlendMode::Overwrite,
            mask: ColorBlendMask::RGBA,
        })
        .render(SizeOverLifetimeModifier {
            gradient: {
                let mut g = bevy_hanabi::Gradient::new();
                g.add_key(0.0, Vec3::splat(1.0));
                g.add_key(0.3, Vec3::splat(0.8));
                g.add_key(1.0, Vec3::splat(0.0));
                g
            },
            screen_space_size: false,
        })
}

fn update_lightning_timers(
    time: Res<Time>,
    mut bolt_query: Query<
        (
            &mut LightningTimer,
            &mut EffectSpawner,
            &mut EffectProperties,
        ),
        Without<ImpactEffect>,
    >,
    mut impact_query: Query<&mut EffectSpawner, With<ImpactEffect>>,
) {
    for (mut lt, mut spawner, mut properties) in &mut bolt_query {
        lt.timer.tick(time.delta());
        if lt.timer.just_finished() {
            properties.set("wave_seed", (rand::random::<f32>() * 100.0 - 50.0).into());
            spawner.reset();

            for mut impact_spawner in &mut impact_query {
                impact_spawner.reset();
            }
        }
    }
}
