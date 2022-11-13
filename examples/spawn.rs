use bevy::{
    log::LogPlugin,
    prelude::*,
    render::{
        mesh::shape::Cube,
        settings::{WgpuLimits, WgpuSettings},
    },
};
use bevy_inspector_egui::WorldInspectorPlugin;

use bevy_hanabi::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Optional; test that a stronger constraint is handled correctly.
    // For example, on macOS the alignment for storage buffer offsets is commonly
    // 256 bytes, whereas on Desktop GPUs it can be much smaller, like 16 bytes
    // only. Force the downlevel limits here, and as an example of how
    // to force a particular limit, and to show Hanabi works with those settings.
    let mut options = WgpuSettings::default();
    let limits = WgpuLimits::downlevel_defaults();
    options.constrained_limits = Some(limits);

    App::default()
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
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = Camera3dBundle::default();
    camera.transform.translation = Vec3::new(0.0, 0.0, 100.0);
    commands.spawn(camera);

    commands.spawn(DirectionalLightBundle {
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

    let mut color_gradient1 = Gradient::new();
    color_gradient1.add_key(0.0, Vec4::splat(1.0));
    color_gradient1.add_key(0.1, Vec4::new(1.0, 1.0, 0.0, 1.0));
    color_gradient1.add_key(0.4, Vec4::new(1.0, 0.0, 0.0, 1.0));
    color_gradient1.add_key(1.0, Vec4::splat(0.0));

    let mut size_gradient1 = Gradient::new();
    size_gradient1.add_key(0.0, Vec2::splat(1.0));
    size_gradient1.add_key(0.5, Vec2::splat(5.0));
    size_gradient1.add_key(0.8, Vec2::splat(0.8));
    size_gradient1.add_key(1.0, Vec2::splat(0.0));

    let effect1 = effects.add(
        EffectAsset {
            name: "emit:rate".to_string(),
            capacity: 32768,
            spawner: Spawner::rate(5.0.into()),
            ..Default::default()
        }
        .init(PositionSphereModifier {
            center: Vec3::ZERO,
            radius: 2.,
            dimension: ShapeDimension::Surface,
            speed: 6.0.into(),
        })
        .update(AccelModifier {
            accel: Vec3::new(0., -3., 0.),
        })
        .render(ColorOverLifetimeModifier {
            gradient: color_gradient1,
        })
        .render(SizeOverLifetimeModifier {
            gradient: size_gradient1,
        }),
    );

    commands
        .spawn((
            Name::new("emit:rate"),
            ParticleEffectBundle {
                effect: ParticleEffect::new(effect1),
                transform: Transform::from_translation(Vec3::new(-30., 0., 0.)),
                ..Default::default()
            },
        ))
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn((
                PbrBundle {
                    mesh: cube.clone(),
                    material: mat.clone(),
                    ..Default::default()
                },
                Name::new("source"),
            ));
        });

    let mut gradient2 = Gradient::new();
    gradient2.add_key(0.0, Vec4::new(0.0, 0.0, 1.0, 1.0));
    gradient2.add_key(1.0, Vec4::splat(0.0));

    let effect2 = effects.add(
        EffectAsset {
            name: "emit:once".to_string(),
            capacity: 32768,
            spawner: Spawner::once(1000.0.into(), true),
            ..Default::default()
        }
        .render(ColorOverLifetimeModifier {
            gradient: gradient2,
        }),
    );

    commands
        .spawn((
            Name::new("emit:once"),
            ParticleEffectBundle {
                effect: ParticleEffect::new(effect2),
                transform: Transform::from_translation(Vec3::new(0., 0., 0.)),
                ..Default::default()
            },
        ))
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn((
                PbrBundle {
                    mesh: cube.clone(),
                    material: mat.clone(),
                    ..Default::default()
                },
                Name::new("source"),
            ));
        });

    // Note: same as gradient2, will yield shared render shader between effects #2
    // and #3
    let mut gradient3 = Gradient::new();
    gradient3.add_key(0.0, Vec4::new(0.0, 0.0, 1.0, 1.0));
    gradient3.add_key(1.0, Vec4::splat(0.0));

    let effect3 = effects.add(
        EffectAsset {
            name: "emit:burst".to_string(),
            capacity: 32768,
            spawner: Spawner::burst(400.0.into(), 3.0.into()),
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
        .render(ColorOverLifetimeModifier {
            gradient: gradient3,
        }),
    );

    commands
        .spawn((
            Name::new("emit:burst"),
            ParticleEffectBundle {
                effect: ParticleEffect::new(effect3),
                transform: Transform::from_translation(Vec3::new(30., 0., 0.)),
                ..Default::default()
            },
        ))
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn((
                PbrBundle {
                    mesh: cube.clone(),
                    material: mat.clone(),
                    ..Default::default()
                },
                Name::new("source"),
            ));
        });
}
