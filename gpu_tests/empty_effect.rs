//! Test that an empty (invalid) bundle doesn't produce any error.

use bevy::{app::AppExit, core_pipeline::tonemapping::Tonemapping, log::LogPlugin, prelude::*};
use bevy_hanabi::prelude::*;

#[derive(Default, Resource)]
struct Frame(pub u32);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut app = App::default();
    app.insert_resource(ClearColor(Color::BLACK))
        .add_plugins(DefaultPlugins.set(LogPlugin {
            level: bevy::log::Level::INFO,
            filter: "bevy_hanabi=debug".to_string(),
            ..default()
        }))
        .add_plugins(HanabiPlugin)
        .init_resource::<Frame>()
        .add_systems(Startup, setup)
        .add_systems(Update, timeout)
        .run();

    Ok(())
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera3dBundle {
        tonemapping: Tonemapping::None,
        ..default()
    });
    commands.spawn(ParticleEffectBundle::default());
}

fn timeout(mut frame: ResMut<Frame>, mut ev_app_exit: EventWriter<AppExit>) {
    frame.0 += 1;
    if frame.0 >= 10 {
        info!("SUCCESS!");
        ev_app_exit.send(AppExit::Success);
    }
}
