//! This example demonstrates saving and loading an EffectAsset.

use bevy::{
    log::LogPlugin,
    prelude::*,
    render::{render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin},
};
use bevy_inspector_egui::{bevy_egui, egui, quick::WorldInspectorPlugin};

use bevy_hanabi::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut wgpu_settings = WgpuSettings::default();
    wgpu_settings
        .features
        .set(WgpuFeatures::VERTEX_WRITABLE_STORAGE, true);

    App::default()
        .insert_resource(ClearColor(Color::DARK_GRAY))
        .add_plugins(
            DefaultPlugins
                .set(LogPlugin {
                    level: bevy::log::Level::WARN,
                    filter: "bevy_hanabi=warn,spawn=trace".to_string(),
                })
                .set(RenderPlugin { wgpu_settings }),
        )
        .add_system(bevy::window::close_on_esc)
        .add_plugin(HanabiPlugin)
        .add_plugin(WorldInspectorPlugin::default())
        .add_startup_system(setup)
        .add_system(respawn)
        .add_system(load_save_ui)
        .run();

    Ok(())
}

const COLOR: Vec4 = Vec4::new(0.7, 0.7, 1.0, 1.0);
const PATH: &str = "disk.effect";

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut effects: ResMut<Assets<EffectAsset>>,
) {
    // Spawn camera.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 3.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // Try to load the asset first. If it doesn't exist yet, create it.
    let effect = match asset_server
        .asset_io()
        .get_metadata(std::path::Path::new(PATH))
    {
        Ok(metadata) => {
            assert!(metadata.is_file());
            asset_server.load(PATH)
        }
        Err(_) => effects.add(
            EffectAsset {
                name: "💾".to_owned(),
                capacity: 32768,
                spawner: Spawner::rate(48.0.into()),
                ..Default::default()
            }
            .init(InitPositionSphereModifier {
                center: Vec3::ZERO,
                radius: 1.,
                dimension: ShapeDimension::Volume,
            })
            .init(InitAttributeModifier {
                attribute: Attribute::VELOCITY,
                value: ValueOrProperty::Value((Vec3::Y * 2.).into()),
            })
            .init(InitLifetimeModifier {
                lifetime: 0.9.into(),
            })
            .render(ParticleTextureModifier {
                // Need to supply a handle and a path in order to save it later.
                texture: AssetHandle::new(asset_server.load("cloud.png"), "cloud.png"),
            })
            .render(SetColorModifier {
                color: COLOR.into(),
            })
            .render(SizeOverLifetimeModifier {
                gradient: {
                    let mut gradient = Gradient::new();
                    gradient.add_key(0.0, Vec2::splat(0.1));
                    gradient.add_key(0.1, Vec2::splat(1.0));
                    gradient.add_key(1.0, Vec2::splat(0.01));
                    gradient
                },
            }),
        ),
    };

    spawn_effect(&mut commands, effect);
}

fn spawn_effect(commands: &mut Commands, effect: Handle<EffectAsset>) -> Entity {
    commands
        .spawn((
            Name::new("💾"),
            ParticleEffectBundle {
                effect: ParticleEffect::new(effect),
                ..Default::default()
            },
        ))
        .id()
}

// Respawn effects when the asset changes.
fn respawn(
    mut commands: Commands,
    mut effect_events: EventReader<AssetEvent<EffectAsset>>,
    effects: Res<Assets<EffectAsset>>,
    query: Query<Entity, With<ParticleEffect>>,
) {
    for event in effect_events.iter() {
        match event {
            AssetEvent::Created { handle } | AssetEvent::Modified { handle } => {
                for entity in query.iter() {
                    commands.entity(entity).despawn();
                    let mut handle = handle.clone();
                    handle.make_strong(&effects);
                    spawn_effect(&mut commands, handle);
                }
                return;
            }
            _ => (),
        }
    }
}

fn load_save_ui(
    asset_server: Res<AssetServer>,
    mut contexts: bevy_egui::EguiContexts,
    effects: ResMut<Assets<EffectAsset>>,
) {
    use std::io::Write;

    egui::Window::new("💾").show(contexts.ctx_mut(), |ui| {
        // You can edit the asset on disk and click load to see changes.
        let load = ui.button("Load");
        if load.clicked() {
            // Reload the asset.
            asset_server.reload_asset(PATH);
        }

        // Save effect to PATH.
        let save = ui.button("Save");
        if save.clicked() {
            let (_handle, effect) = effects.iter().next().unwrap();
            let ron = ron::ser::to_string_pretty(&effect, Default::default()).unwrap();
            let mut file = std::fs::File::create(format!(
                "{}/{}/{}",
                std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_owned()),
                "assets",
                PATH
            ))
            .unwrap();
            file.write_all(ron.as_bytes()).unwrap();
        }
    });
}
