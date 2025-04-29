//! Test that an effect with a single particle renders it.

use bevy::{app::AppExit, core_pipeline::tonemapping::Tonemapping, log::LogPlugin, prelude::*};
use bevy_hanabi::prelude::*;

#[derive(Default, Resource)]
struct Frame(pub u32);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut app = App::default();
    app.insert_resource(ClearColor(Color::BLACK))
        .add_plugins(DefaultPlugins.set(LogPlugin {
            level: bevy::log::Level::INFO,
            filter: "bevy_hanabi=trace".to_string(),
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
        Transform::from_translation(Vec3::new(0., 0., 50.)),
        Camera3d::default(),
        Camera {
            hdr: false,
            clear_color: Color::BLACK.into(),
            ..default()
        },
        Tonemapping::None,
    ));

    let mut module = Module::default();
    let pos = module.lit(Vec3::new(0.1, 0.2, 0.3));
    let size = module.lit(Vec3::ONE * 10.);
    let mut asset = EffectAsset::new(16, SpawnerSettings::rate(1000.0.into()), module)
        .init(SetAttributeModifier::new(Attribute::POSITION, pos))
        .init(SetAttributeModifier::new(Attribute::SIZE3, size));
    asset.name = "test_asset".to_string();
    let handle = assets.add(asset);
    commands.spawn(ParticleEffect::new(handle));
}

fn timeout(mut frame: ResMut<Frame>, mut ev_app_exit: EventWriter<AppExit>) {
    frame.0 += 1;
    if frame.0 >= 1000 {
        info!("SUCCESS!");
        ev_app_exit.write(AppExit::Success);
    }
}
