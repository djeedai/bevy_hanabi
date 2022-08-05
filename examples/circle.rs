//! Example of using the circle spawner with random velocity.
//! A sphere spawns dust in a circle.

use bevy::{
    prelude::*,
    render::{render_resource::WgpuFeatures, settings::WgpuSettings},
};

use bevy_hanabi::*;
use bevy_inspector_egui::WorldInspectorPlugin;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = WgpuSettings::default();
    options
        .features
        .set(WgpuFeatures::VERTEX_WRITABLE_STORAGE, true);
    // options
    //     .features
    //     .set(WgpuFeatures::MAPPABLE_PRIMARY_BUFFERS, false);
    // println!("wgpu options: {:?}", options.features);
    App::default()
        .insert_resource(options)
        .insert_resource(bevy::log::LogSettings {
            level: bevy::log::Level::WARN,
            filter: "bevy_hanabi=error,circle=trace".to_string(),
        })
        .add_plugins(DefaultPlugins)
        .add_system(bevy::window::close_on_esc)
        .add_plugin(HanabiPlugin)
        .add_plugin(WorldInspectorPlugin::new())
        .add_startup_system(setup)
        .run();

    Ok(())
}

fn setup(
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = Camera3dBundle::default();
    camera.transform =
        Transform::from_xyz(3.0, 3.0, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    commands.spawn_bundle(camera);

    let texture_handle: Handle<Image> = asset_server.load("cloud.png");

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::splat(1.0));
    gradient.add_key(0.5, Vec4::splat(1.0));
    gradient.add_key(1.0, Vec4::new(1.0, 1.0, 1.0, 0.0));

    let effect = effects.add(
        EffectAsset {
            name: "Gradient".to_string(),
            // TODO: Figure out why no particle spawns if this is 1
            capacity: 32768,
            spawner: Spawner::once(32.0.into(), true),
            ..Default::default()
        }
        .init(PositionCircleModifier {
            center: Vec3::Y * 0.1,
            axis: Vec3::Y,
            radius: 0.4,
            speed: Value::Uniform((1.0, 1.5)),
            dimension: ShapeDimension::Surface,
        })
        .render(ParticleTextureModifier {
            texture: texture_handle.clone(),
        })
        .render(ColorOverLifetimeModifier { gradient })
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::constant([0.2; 2].into()),
        }),
    );

    // The ground
    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Plane { size: 4.0 })),
            material: materials.add(Color::BLUE.into()),
            ..Default::default()
        })
        .insert(Name::new("ground"));

    // The sphere
    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::UVSphere {
                radius: 1.0,
                sectors: 32,
                stacks: 16,
            })),
            material: materials.add(Color::CYAN.into()),
            transform: Transform::from_translation(Vec3::Y),
            ..Default::default()
        })
        .insert(Name::new("sphere"));

    commands
        .spawn_bundle(ParticleEffectBundle::new(effect))
        .insert(Name::new("effect"));
}
