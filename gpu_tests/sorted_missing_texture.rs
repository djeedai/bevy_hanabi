//! Test that a sorted effect whose texture never loads doesn't stop the app
//! while another sorted effect is drawing.
//!
//! The effect missing its texture never gets its bind groups, but the sort
//! pass still runs for the frame because the other effect needs it, and walks
//! every sorted batch. The other effect is untextured, so that the two don't
//! share a property bind group.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use bevy::{
    app::AppExit,
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    log::LogPlugin,
    prelude::*,
};
use bevy_hanabi::prelude::*;

/// Long enough for the effects' pipelines to compile and both to draw.
const RUN_SECS: f32 = 5.;

// Read after the app exits, which takes its world with it.
static FRAMES: AtomicU32 = AtomicU32::new(0);
static FINISHED: AtomicBool = AtomicBool::new(false);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let exit = App::default()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins(DefaultPlugins.set(LogPlugin {
            level: bevy::log::Level::INFO,
            filter: "bevy_hanabi=debug".to_string(),
            ..default()
        }))
        .add_plugins(HanabiPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, timeout)
        .run();

    if !exit.is_success() || !FINISHED.load(Ordering::Relaxed) {
        let frames = FRAMES.load(Ordering::Relaxed);
        return Err(format!("app exited with {exit:?} after {frames} frames").into());
    }
    Ok(())
}

/// Ribbons are sorted on the GPU every frame.
fn ribbon(name: &str, textured: bool) -> EffectAsset {
    let writer = ExprWriter::new();
    let init_pos = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(1.).expr(),
        dimension: ShapeDimension::Volume,
    };
    let init_age = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.).expr());
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, writer.lit(1.).expr());
    let init_ribbon_id = SetAttributeModifier::new(Attribute::RIBBON_ID, writer.lit(0u32).expr());
    let mut module = writer.finish();
    if textured {
        module.add_texture_slot("color", SlotDimension::D2);
    }
    let effect = EffectAsset::new(256, SpawnerSettings::rate(60.0.into()), module)
        .with_name(name)
        .with_motion_integration(MotionIntegration::None)
        .init(init_pos)
        .init(init_age)
        .init(init_lifetime)
        .init(init_ribbon_id);
    if textured {
        effect.render(ParticleTextureModifier {
            texture_slot: 0,
            sample_mapping: ImageSampleMapping::Modulate,
        })
    } else {
        effect
    }
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut effects: ResMut<Assets<EffectAsset>>,
) {
    commands.spawn((
        Transform::from_xyz(0., 0., 5.),
        Camera3d::default(),
        Tonemapping::None,
        DebandDither::Disabled,
    ));

    commands.spawn(ParticleEffect::new(
        effects.add(ribbon("untextured", false)),
    ));

    let missing: Handle<Image> = asset_server.load("this/texture/does/not/exist.png");
    commands.spawn((
        ParticleEffect::new(effects.add(ribbon("missing", true))),
        EffectMaterial {
            images: vec![missing],
        },
    ));
}

fn timeout(time: Res<Time>, mut ev_app_exit: MessageWriter<AppExit>) {
    FRAMES.fetch_add(1, Ordering::Relaxed);
    if time.elapsed_secs() >= RUN_SECS {
        FINISHED.store(true, Ordering::Relaxed);
        info!("SUCCESS!");
        ev_app_exit.write(AppExit::Success);
    }
}
