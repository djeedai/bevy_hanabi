#![allow(unused)]

use std::num::NonZeroU8;

use bevy::{
    camera::{visibility::RenderLayers, CameraOutputMode},
    log::LogPlugin,
    prelude::*,
    render::{settings::WgpuSettings, RenderDebugFlags, RenderPlugin},
    text::{TextColor, TextFont},
    ui::{
        widget::Text, BackgroundColor, BorderColor, BorderRadius, Display, Node, Overflow,
        PositionType, UiRect, Val, ZIndex,
    },
};
use wgpu::BlendState;

use crate::prelude::*;

/// Helper system to enable closing the example application by pressing the
/// escape key (ESC).
pub fn close_on_esc(mut ev_app_exit: MessageWriter<AppExit>, input: Res<ButtonInput<KeyCode>>) {
    if input.just_pressed(KeyCode::Escape) {
        ev_app_exit.write(AppExit::Success);
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
        // Tune down the verbose Vulkan driver output
        "wgpu_hal::vulkan::instance=warn",
    ]
    .join(",")
}

#[derive(Default, Clone, Copy)]
pub enum DescPosition {
    #[default]
    LeftColumn,
    BottomRow,
}

#[derive(Default)]
pub struct DemoApp {
    name: String,
    desc: String,
    wgpu_settings: WgpuSettings,
    desc_position: DescPosition,
}

impl DemoApp {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..default()
        }
    }

    pub fn with_desc(mut self, desc: &str) -> Self {
        self.desc = desc.to_string();
        self
    }

    pub fn with_desc_position(mut self, desc_position: DescPosition) -> Self {
        self.desc_position = desc_position;
        self
    }

    pub fn with_wgpu_settings(mut self, wgpu_settings: WgpuSettings) -> Self {
        self.wgpu_settings = wgpu_settings;
        self
    }

    pub fn build(self) -> App {
        let mut app = App::default();
        app.insert_resource(ClearColor(Color::BLACK))
            .add_plugins(
                DefaultPlugins
                    .set(LogPlugin {
                        level: bevy::log::Level::INFO,
                        filter: get_log_filters(&self.name),
                        ..default()
                    })
                    .set(RenderPlugin {
                        render_creation: self.wgpu_settings.into(),
                        synchronous_pipeline_compilation: false,
                        debug_flags: RenderDebugFlags::empty(),
                    })
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: format!("🎆 Hanabi — {}", self.name),
                            ..default()
                        }),
                        ..default()
                    }),
            )
            .add_plugins(HanabiPlugin)
            .insert_resource(Demo {
                name: self.name,
                desc: self.desc,
                desc_position: self.desc_position,
            })
            .add_systems(Startup, spawn_demo_ui)
            .add_systems(Update, close_on_esc);

        app
    }
}

#[derive(Resource)]
pub struct Demo {
    pub name: String,
    pub desc: String,
    pub desc_position: DescPosition,
}

/// Error struct wrapping an app error code.
#[derive(Debug)]
pub struct ExampleFailedError(pub NonZeroU8);

impl std::fmt::Display for ExampleFailedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "App terminated with error code {}", self.0.get())
    }
}

impl std::error::Error for ExampleFailedError {}

/// Convert an [`AppExit`] into a `Result`, for error code propagation to the
/// OS.
pub trait AppExitIntoResult {
    fn into_result(self) -> Result<(), Box<dyn std::error::Error>>;
}

impl AppExitIntoResult for AppExit {
    fn into_result(self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
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

fn spawn_demo_ui(mut cmd: Commands, demo: Res<Demo>) {
    debug!("Spawning UI for demo {}", demo.name);

    // Camera
    let ui_camera = cmd
        .spawn((
            Camera2d,
            Projection::Orthographic(OrthographicProjection::default_2d()),
            Camera {
                order: 1000, // render UI above everything
                clear_color: ClearColorConfig::None,
                output_mode: CameraOutputMode::Write {
                    blend_state: Some(BlendState::ALPHA_BLENDING),
                    clear_color: ClearColorConfig::None,
                },
                ..default()
            },
            Name::new("UI camera"),
            RenderLayers::layer(63),
        ))
        .id();

    // Description UI panel
    let (left, top, right, bottom, width) = match demo.desc_position {
        DescPosition::LeftColumn => (Val::Vw(5.), Val::Vw(5.), Val::Auto, Val::Auto, Val::Vw(30.)),
        DescPosition::BottomRow => (Val::Vw(5.), Val::Auto, Val::Vw(5.), Val::Vw(5.), Val::Auto),
    };
    cmd.spawn((
        Node {
            display: Display::Block,
            position_type: PositionType::Absolute,
            overflow: Overflow::clip(),
            left,
            top,
            right,
            bottom,
            min_width: width,
            width,
            border: UiRect::all(Val::Px(1.)),
            border_radius: BorderRadius::all(Val::Px(8.)),
            ..default()
        },
        BackgroundColor(Color::linear_rgba(0., 0., 0., 0.8)),
        BorderColor::all(Color::linear_rgb(0.8, 0.8, 0.8)),
        ZIndex(3000),
        children![
            (
                Node {
                    padding: UiRect::all(Val::Px(3.)),
                    margin: UiRect::all(Val::Px(8.)),
                    ..default()
                },
                Text::new(demo.name.clone()),
                TextColor(Color::linear_rgb(1., 1., 1.)),
                TextFont::from_font_size(18.),
            ),
            (
                Node {
                    padding: UiRect::all(Val::Px(3.)),
                    margin: UiRect::all(Val::Px(8.)),
                    ..default()
                },
                Text::new(demo.desc.clone()),
                TextColor(Color::linear_rgb(0.8, 0.8, 0.8)),
                TextFont::from_font_size(12.),
            )
        ],
        UiTargetCamera(ui_camera),
    ));
}
