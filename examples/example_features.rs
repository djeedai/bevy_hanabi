use bevy::{
    log::LogPlugin,
    prelude::*,
    render::{render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin},
};

#[cfg(feature = "world_inspector")]
use bevy_inspector_egui::quick::WorldInspectorPlugin;

#[derive(Default)]
pub struct ExampleFeaturesPlugin {
    pub window_title: String,
    pub wgpu_settings: WgpuSettings,
}

impl Plugin for ExampleFeaturesPlugin {
    fn build(&self, app: &mut App) {
        let mut wgpu_settings = self.wgpu_settings.clone();
        wgpu_settings
            .features
            .set(WgpuFeatures::VERTEX_WRITABLE_STORAGE, true);

        app.add_plugins(
            DefaultPlugins
                .set(LogPlugin {
                    level: bevy::log::Level::WARN,
                    filter: "bevy_hanabi=warn,gradient=trace".to_string(),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings.into(),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: self.window_title.clone(),
                        ..default()
                    }),
                    ..default()
                }),
        );
        #[cfg(feature = "world_inspector")]
        app.add_plugins(WorldInspectorPlugin::default());
    }
}
