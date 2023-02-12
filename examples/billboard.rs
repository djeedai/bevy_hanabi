//! An example using the [`BillboardModifier`] to force
//! particles to always render facing the camera.

use bevy::{
    log::LogPlugin,
    prelude::*,
    render::{camera::Projection, render_resource::WgpuFeatures, settings::WgpuSettings},
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

use bevy_hanabi::prelude::*;

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
        .add_plugins(DefaultPlugins.set(LogPlugin {
            level: bevy::log::Level::WARN,
            filter: "bevy_hanabi=warn,circle=trace".to_string(),
        }))
        .add_system(bevy::window::close_on_esc)
        .add_plugin(HanabiPlugin)
        .add_plugin(WorldInspectorPlugin)
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
    let camera = Camera3dBundle {
        transform: Transform::from_xyz(3.0, 3.0, 3.0).looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        projection: Projection::Perspective(PerspectiveProjection {
            fov: 120.0,
            ..Default::default()
        }),
        ..Default::default()
    };

    commands.spawn(camera);

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
            spawner: Spawner::rate(64.0.into()),
            ..Default::default()
        }
        .init(InitPositionCircleModifier {
            center: Vec3::Y * 0.1,
            axis: Vec3::Y,
            radius: 1.0,
            dimension: ShapeDimension::Volume,
        })
        .init(InitVelocityCircleModifier {
            center: Vec3::ZERO,
            axis: Vec3::Y,
            speed: Value::Uniform((0.7, 0.5)),
        })
        .init(InitLifetimeModifier {
            lifetime: 5_f32.into(),
        })
        .render(ParticleTextureModifier {
            texture: texture_handle,
        })
        .render(BillboardModifier {})
        .render(ColorOverLifetimeModifier { gradient })
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::constant([0.2; 2].into()),
        }),
    );

    // The ground
    commands
        .spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Plane { size: 4.0 })),
            material: materials.add(Color::BLUE.into()),
            transform: Transform::from_xyz(0.0, -0.5, 0.0),
            ..Default::default()
        })
        .insert(Name::new("ground"));

    commands
        .spawn(ParticleEffectBundle::new(effect))
        .insert(Name::new("effect"));
}
