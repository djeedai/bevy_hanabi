//! Example of using the circle spawner.
//! A sphere bounces on the ground and spawns dust in a circle.

use bevy::{
    prelude::*,
    render::{options::WgpuOptions, render_resource::WgpuFeatures},
};

use bevy_hanabi::*;
use bevy_inspector_egui::WorldInspectorPlugin;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = WgpuOptions::default();
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
            filter: "bevy_hanabi=error,spawn=trace".to_string(),
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(HanabiPlugin)
        .add_plugin(WorldInspectorPlugin::new())
        .add_startup_system(setup)
        .add_system(update)
        .run();

    Ok(())
}

/// Labels the sphere that bounces
#[derive(Clone, Component)]
struct Bounce {
    bounce_velocity: f32,
    velocity: f32,
    dust: Handle<EffectAsset>,
}

impl Bounce {
    fn new(bounce_velocity: f32, dust: Handle<EffectAsset>) -> Self {
        Self {
            bounce_velocity,
            velocity: bounce_velocity,
            dust,
        }
    }
}

fn setup(
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = PerspectiveCameraBundle::new_3d();
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
            capacity: 32768,
            spawner: Spawner::new(SpawnMode::Once(SpawnCount::Single(32.0))),
            ..Default::default()
        }
        .init(PositionCircleModifier {
            center: Vec3::Y * 0.1,
            axis: Vec3::Y,
            radius: 0.4,
            speed: 1.0,
            dimension: ShapeDimension::Surface,
        })
        .render(ParticleTextureModifier {
            texture: texture_handle.clone(),
        })
        .render(ColorOverLifetimeModifier { gradient })
        .render(SizeOverLifetimeModifier { gradient: Gradient::constant(2.0) }),
    );

    // The ground
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Plane { size: 4.0 })),
        material: materials.add(Color::BLUE.into()),
        ..Default::default()
    });

    // The bouncing sphere
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
        .insert(Bounce::new(2.0, effect));
}

fn update(time: Res<Time>, mut query: Query<&mut Transform, With<ParticleEffect>>) {}
