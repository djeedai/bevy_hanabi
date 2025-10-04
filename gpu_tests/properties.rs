//! Test for effect properties.
//!
//! - Add an effect with a property.
//! - Start running.
//! - Replace the effect with a new one having no property.
//!
//! The backend should automatically de-allocate the property storage, and
//! gracefully re-create any GPU bind group layout and bind group without any
//! property reference.

use bevy::{app::AppExit, core_pipeline::tonemapping::Tonemapping, log::LogPlugin, prelude::*};
use bevy_hanabi::prelude::*;

#[derive(Default, Resource)]
struct Frame(pub u32);

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
    let mut module = Module::default();
    module.add_property("my_property", VectorValue::new_vec3(Vec3::ZERO).into());
    let pos = module.lit(Vec3::ZERO);
    let mut asset = EffectAsset::new(128, SpawnerSettings::rate(1.0.into()), module)
        .init(SetAttributeModifier::new(Attribute::POSITION, pos));
    asset.name = "test_asset".to_string();
    let handle = assets.add(asset);

    commands.spawn((Camera3d::default(), Tonemapping::None));
    commands.spawn(ParticleEffect::new(handle));
}

fn timeout(
    mut commands: Commands,
    mut assets: ResMut<Assets<EffectAsset>>,
    mut frame: ResMut<Frame>,
    mut query: Query<(Entity, &mut ParticleEffect)>,
    mut ev_app_exit: MessageWriter<AppExit>,
) {
    frame.0 += 1;

    if frame.0 == 10 {
        let (_, mut effect) = query.single_mut().unwrap();

        // New effect without any property
        let mut module = Module::default();
        let pos = module.lit(Vec3::ZERO);
        let asset = EffectAsset::new(128, SpawnerSettings::rate(1.0.into()), module)
            .init(SetAttributeModifier::new(Attribute::POSITION, pos));
        let handle = assets.add(asset);

        effect.handle = handle;
    }

    if frame.0 == 15 {
        let (entity, _) = query.single().unwrap();
        commands.entity(entity).despawn();
    }

    if frame.0 >= 20 {
        info!("SUCCESS!");
        ev_app_exit.write(AppExit::Success);
    }
}
