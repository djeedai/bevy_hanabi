use bevy::{prelude::*, render::mesh::shape::Cube};
use bevy_inspector_egui::WorldInspectorPlugin;

use bevy_hanabi::prelude::*;

fn main() {
    App::default()
        .insert_resource(bevy::log::LogSettings {
            level: bevy::log::Level::WARN,
            filter: "bevy_hanabi=warn,instancing=trace".to_string(),
        })
        .add_plugins(DefaultPlugins)
        .add_system(bevy::window::close_on_esc)
        .add_plugin(HanabiPlugin)
        .add_plugin(WorldInspectorPlugin::new())
        .add_startup_system(setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = Camera3dBundle::default();
    camera.transform.translation = Vec3::new(0.0, 0.0, 180.0);
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
    gradient.add_key(1.0, Vec4::splat(0.0));

    let effect = effects.add(
        EffectAsset {
            name: "effect".to_string(),
            capacity: 512,
            spawner: Spawner::rate(50.0.into()),
            ..Default::default()
        }
        .init(PositionSphereModifier {
            center: Vec3::ZERO,
            radius: 1.,
            dimension: ShapeDimension::Volume,
            speed: 2.0.into(),
        })
        .init(ParticleLifetimeModifier { lifetime: 12.0 })
        .render(ColorOverLifetimeModifier { gradient }),
    );

    for j in -4..=4 {
        for i in -5..=5 {
            commands
                .spawn()
                .insert(Name::new(format!("({},{})", i, j)))
                .insert_bundle(ParticleEffectBundle {
                    effect: ParticleEffect::new(effect.clone()),
                    transform: Transform::from_translation(Vec3::new(
                        i as f32 * 10.,
                        j as f32 * 10.,
                        0.,
                    )),
                    ..Default::default()
                })
                .with_children(|p| {
                    // Reference cube to visualize the emit origin
                    p.spawn()
                        .insert_bundle(PbrBundle {
                            mesh: cube.clone(),
                            material: mat.clone(),
                            ..Default::default()
                        })
                        .insert(Name::new("source"));
                });
        }
    }
}
