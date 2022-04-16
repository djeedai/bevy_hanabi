//! Example of using random spawner params.
//! Spawns a random number of particles at random times.

use bevy::{
    prelude::*,
    render::{mesh::shape::Cube, render_resource::WgpuFeatures, settings::WgpuSettings},
};
//use bevy_inspector_egui::WorldInspectorPlugin;

use bevy_hanabi::*;

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
            filter: "bevy_hanabi=error,spawn=trace".to_string(),
        })
        .add_plugins(DefaultPlugins)
        .add_system(bevy::input::system::exit_on_esc_system)
        .add_plugin(HanabiPlugin)
        //.add_plugin(WorldInspectorPlugin::new())
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

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(0.0, 0.0, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::new(0.0, 0.0, 1.0, 0.0));

    let effect = effects.add(
        EffectAsset {
            name: "emit:burst".to_string(),
            capacity: 32768,
            spawner: Spawner::burst(Value::Uniform((1., 100.)), Value::Uniform((1., 4.))),
            ..Default::default()
        }
        .init(PositionSphereModifier {
            center: Vec3::ZERO,
            radius: 5.,
            dimension: ShapeDimension::Volume,
            speed: 2.0.into(),
        })
        .update(AccelModifier {
            accel: Vec3::new(0., 5., 0.),
        })
        .render(ColorOverLifetimeModifier { gradient }),
    );

    commands
        .spawn()
        .insert(Name::new("emit:random"))
        .insert_bundle(ParticleEffectBundle {
            effect: ParticleEffect::new(effect),
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
}
