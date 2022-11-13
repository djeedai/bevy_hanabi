//! A particle system with a 2D camera.
//!
//! The particle effect instance override its `z_layer_2d` field, which can be
//! tweaked at runtime via the egui inspector to move the 2D rendering layer of
//! particle above or below the reference square.

use bevy::{
    log::LogPlugin,
    prelude::*,
    render::{camera::ScalingMode, render_resource::WgpuFeatures, settings::WgpuSettings},
    sprite::MaterialMesh2dBundle,
};
use bevy_inspector_egui::WorldInspectorPlugin;

use bevy_hanabi::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = WgpuSettings::default();
    options
        .features
        .set(WgpuFeatures::VERTEX_WRITABLE_STORAGE, true);
    App::default()
        .insert_resource(ClearColor(Color::DARK_GRAY))
        .insert_resource(options)
        .add_plugins(DefaultPlugins.set(LogPlugin {
            level: bevy::log::Level::WARN,
            filter: "bevy_hanabi=warn,spawn=trace".to_string(),
        }))
        .add_system(bevy::window::close_on_esc)
        .add_plugin(HanabiPlugin)
        .add_plugin(WorldInspectorPlugin::new())
        .add_startup_system(setup)
        .run();

    Ok(())
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    // Spawn a 2D camera
    let mut camera = Camera2dBundle::default();
    camera.projection.scale = 1.0;
    camera.projection.scaling_mode = ScalingMode::FixedVertical(1.);
    commands.spawn(camera);

    // Spawn a reference white square in the center of the screen at Z=0
    commands
        .spawn(MaterialMesh2dBundle {
            mesh: meshes
                .add(Mesh::from(shape::Quad {
                    size: Vec2::splat(0.2),
                    ..Default::default()
                }))
                .into(),
            material: materials.add(ColorMaterial {
                color: Color::WHITE,
                ..Default::default()
            }),
            ..Default::default()
        })
        .insert(Name::new("square"));

    // Create a color gradient for the particles
    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(0.5, 0.5, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::new(0.5, 0.5, 1.0, 0.0));

    // Create a new effect asset spawning 30 particles per second from a circle
    // and slowly fading from blue-ish to transparent over their lifetime.
    // By default the asset spawns the particles at Z=0.
    let spawner = Spawner::rate(30.0.into());
    let effect = effects.add(
        EffectAsset {
            name: "Effect".into(),
            capacity: 4096,
            spawner,
            ..Default::default()
        }
        .init(PositionCircleModifier {
            radius: 0.05,
            speed: 0.1.into(),
            dimension: ShapeDimension::Surface,
            ..Default::default()
        })
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::constant(Vec2::splat(0.02)),
        })
        .render(ColorOverLifetimeModifier { gradient }),
    );

    // Spawn an instance of the particle effect, and override its Z layer to
    // be above the reference white square previously spawned.
    commands
        .spawn(ParticleEffectBundle {
            // Assign the Z layer so it appears in the egui inspector and can be modified at runtime
            effect: ParticleEffect::new(effect).with_z_layer_2d(Some(0.1)),
            ..default()
        })
        .insert(Name::new("effect:2d"));
}
