#![allow(unused)]

use std::{fmt::Display, num::NonZeroU8};

use crate::prelude::*;
use bevy::{
    log::LogPlugin,
    prelude::*,
    render::{settings::WgpuSettings, RenderPlugin},
};

#[cfg(feature = "examples_world_inspector")]
use bevy_inspector_egui::quick::WorldInspectorPlugin;

/// Helper system to enable closing the example application by pressing the
/// escape key (ESC).
pub fn close_on_esc(mut ev_app_exit: EventWriter<AppExit>, input: Res<ButtonInput<KeyCode>>) {
    if input.just_pressed(KeyCode::Escape) {
        ev_app_exit.send(AppExit::Success);
    }
}

/// Calculate a log filter for the LogPlugin based on the example app name.
pub fn get_log_filters(example_name: &str) -> String {
    [
        // The example app itself is at trace level so we can see everything
        &format!("{}=trace", example_name),
        // Default Hanabi to warn, probably don't need more
        "bevy_hanabi=warn",
        // Prevent HAL from dumping all naga-generated shader code in logs
        "wgpu_hal::dx12::device=warn",
    ]
    .join(",")
}

/// Create a test app for an example.
pub fn make_test_app(example_name: &str) -> App {
    make_test_app_with_settings(example_name, WgpuSettings::default())
}

/// Create a test app for an example, with explicit WGPU settings.
pub fn make_test_app_with_settings(example_name: &str, wgpu_settings: WgpuSettings) -> App {
    let mut app = App::default();
    app.insert_resource(ClearColor(Color::BLACK))
        .add_plugins(
            DefaultPlugins
                .set(LogPlugin {
                    level: bevy::log::Level::INFO,
                    filter: get_log_filters(example_name),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings.into(),
                    synchronous_pipeline_compilation: false,
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: format!("🎆 Hanabi — {}", example_name),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(HanabiPlugin)
        .add_systems(Update, close_on_esc);

    #[cfg(feature = "examples_world_inspector")]
    app.add_plugins(WorldInspectorPlugin::default());

    app
}

/// Error struct wrapping an app error code.
#[derive(Debug)]
pub struct ExampleFailedError(pub NonZeroU8);

impl Display for ExampleFailedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "App terminated with error code {}", self.0.get())
    }
}

impl std::error::Error for ExampleFailedError {}

/// Convert an [`AppExit`] into a `Result`, for error code propagation to the
/// OS.
pub trait AppExitIntoResult {
    fn into_result(&self) -> Result<(), Box<dyn std::error::Error>>;
}

impl AppExitIntoResult for AppExit {
    fn into_result(&self) -> Result<(), Box<dyn std::error::Error>> {
        match *self {
            AppExit::Success => Ok(()),
            AppExit::Error(code) => Err(Box::new(ExampleFailedError(code))),
        }
    }
}

pub const COLOR_RED: Color = Color::linear_rgb(1., 0., 0.);
pub const COLOR_GREEN: Color = Color::linear_rgb(0., 1., 0.);
pub const COLOR_BLUE: Color = Color::linear_rgb(0., 0., 1.);
pub const COLOR_YELLOW: Color = Color::linear_rgb(1., 1., 0.);
pub const COLOR_CYAN: Color = Color::linear_rgb(0., 1., 1.);
pub const COLOR_OLIVE: Color = Color::linear_rgb(0.5, 0.5, 0.);
pub const COLOR_PURPLE: Color = Color::linear_rgb(0.5, 0., 0.5);
