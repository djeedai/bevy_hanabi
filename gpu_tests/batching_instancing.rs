//! Regression test for batched effect instances.
//!
//! Spawns several instances of the same effect asset and runs the app headless.
//! The test fails on any panic in the batched init/update/render paths.

use bevy::{app::AppExit, core_pipeline::tonemapping::Tonemapping, log::LogPlugin, prelude::*};
use bevy_hanabi::prelude::*;

#[derive(Default, Resource)]
struct Frame(u32);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut app = App::default();
    app.insert_resource(ClearColor(Color::BLACK))
        .add_plugins(DefaultPlugins.set(LogPlugin {
            level: bevy::log::Level::INFO,
            filter: "bevy_hanabi=trace,wgpu_hal=warn".to_string(),
            ..default()
        }))
        .add_plugins(HanabiPlugin)
        .init_resource::<Frame>()
        .add_systems(Startup, setup)
        .add_systems(Update, timeout)
        .run();

    Ok(())
}

fn setup(mut commands: Commands, mut assets: ResMut<Assets<EffectAsset>>) {
    commands.spawn((
        Transform::from_translation(Vec3::new(0., 0., 80.)),
        Camera3d::default(),
        Camera {
            clear_color: Color::BLACK.into(),
            ..default()
        },
        Tonemapping::None,
    ));

    let mut module = Module::default();
    let pos = module.lit(Vec3::ZERO);
    let vel = module.lit(Vec3::Y * 0.5);
    let lifetime = module.lit(4.0);
    let size = module.lit(Vec3::splat(2.0));
    let mut asset = EffectAsset::new(2048, SpawnerSettings::rate(300.0.into()), module)
        .init(SetAttributeModifier::new(Attribute::POSITION, pos))
        .init(SetAttributeModifier::new(Attribute::VELOCITY, vel))
        .init(SetAttributeModifier::new(Attribute::LIFETIME, lifetime))
        .init(SetAttributeModifier::new(Attribute::SIZE3, size));
    asset.name = "batching_instancing_asset".to_string();
    let handle = assets.add(asset);

    // Multiple instances sharing the same effect asset should exercise
    // cross-instance batching.
    for i in 0..8 {
        commands.spawn((
            Transform::from_xyz((i as f32 - 3.5) * 6.0, 0.0, 0.0),
            ParticleEffect::new(handle.clone()),
        ));
    }
}

fn timeout(mut frame: ResMut<Frame>, mut ev_app_exit: MessageWriter<AppExit>) {
    frame.0 += 1;
    if frame.0 >= 240 {
        info!("SUCCESS!");
        ev_app_exit.write(AppExit::Success);
    }
}
