use bevy::{
    prelude::*,
    render::{mesh::shape::Cube, options::WgpuOptions, render_resource::WgpuFeatures},
};
use bevy_inspector_egui::WorldInspectorPlugin;

use bevy_hanabi::*;

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
        .run();

    Ok(())
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = PerspectiveCameraBundle::new_3d();
    camera.transform.translation = Vec3::new(0.0, 0.0, 100.0);
    commands.spawn_bundle(camera);

    commands.spawn_bundle(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::WHITE,
            // Crank the illuminance way (too) high to make the reference cube clearly visible
            illuminance: 100000.,
            shadows_enabled: false,
            ..Default::default()
        },
        ..Default::default()
    });

    let cube = meshes.add(Mesh::from(Cube { size: 1.0 }));
    let mat = materials.add(Color::PURPLE.into());

    let mut gradient1 = Gradient::new();
    gradient1.add_key(0.0, Vec4::splat(1.0));
    gradient1.add_key(0.1, Vec4::new(1.0, 1.0, 0.0, 1.0));
    gradient1.add_key(0.4, Vec4::new(1.0, 0.0, 0.0, 1.0));
    gradient1.add_key(1.0, Vec4::splat(0.0));

    let effect1 = effects.add(
        EffectAsset {
            name: "emit:rate".to_string(),
            capacity: 32768,
            spawner: Spawner::new(SpawnMode::rate(5.)),
        }
        .with(ColorOverLifetimeModifier {
            gradient: gradient1,
        }),
    );

    commands
        .spawn()
        .insert(Name::new("emit:rate"))
        .insert_bundle(ParticleEffectBundle {
            effect: ParticleEffect::new(effect1),
            transform: Transform::from_translation(Vec3::new(-30., 0., 0.)),
            ..Default::default()
        })
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn().insert_bundle(PbrBundle {
                mesh: cube.clone(),
                material: mat.clone(),
                ..Default::default()
            });
        });

    let mut gradient2 = Gradient::new();
    gradient2.add_key(0.0, Vec4::new(0.0, 0.0, 1.0, 1.0));
    gradient2.add_key(1.0, Vec4::splat(0.0));

    let effect2 = effects.add(
        EffectAsset {
            name: "emit:once".to_string(),
            capacity: 32768,
            spawner: Spawner::new(SpawnMode::once(1000.)),
        }
        .with(ColorOverLifetimeModifier {
            gradient: gradient2,
        }),
    );

    commands
        .spawn()
        .insert(Name::new("emit:once"))
        .insert_bundle(ParticleEffectBundle {
            effect: ParticleEffect::new(effect2),
            transform: Transform::from_translation(Vec3::new(0., 0., 0.)),
            ..Default::default()
        })
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn().insert_bundle(PbrBundle {
                mesh: cube.clone(),
                material: mat.clone(),
                ..Default::default()
            });
        });

    let mut gradient3 = Gradient::new();
    gradient3.add_key(0.0, Vec4::new(0.0, 0.0, 1.0, 1.0));
    gradient3.add_key(1.0, Vec4::splat(0.0));

    let effect3 = effects.add(
        EffectAsset {
            name: "emit:burst".to_string(),
            capacity: 32768,
            spawner: Spawner::new(SpawnMode::burst(40., 3.)),
        }
        .with(ColorOverLifetimeModifier {
            gradient: gradient3,
        }),
    );

    commands
        .spawn()
        .insert(Name::new("emit:burst"))
        .insert_bundle(ParticleEffectBundle {
            effect: ParticleEffect::new(effect3),
            transform: Transform::from_translation(Vec3::new(30., 0., 0.)),
            ..Default::default()
        })
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn().insert_bundle(PbrBundle {
                mesh: cube.clone(),
                material: mat.clone(),
                ..Default::default()
            });
        });
}
