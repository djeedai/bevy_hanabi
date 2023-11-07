//! Lifetime
//!
//! An example demonstrating the effect of the particle's lifetime. The example
//! spawns 3 effect instances, which emit a burst of particles every 3 seconds.
//! Each effect has a different particle lifetime:
//! - The left effect has a lifetime of 12 seconds, much longer than the spawn
//!   rate. Multiple bursts of particles are alive at the same time.
//! - The center effect has a lifetime of 3 seconds, exactly the spawn rate. As
//!   soon as particles die, a new burst spawns some more.
//! - The right effect has a lifetime of 0.75 seconds. Particle die very
//!   quickly, and during 2.25 seconds there's no particle, until the next burst
//!   spawns some more.

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    log::LogPlugin,
    prelude::*,
    render::{
        mesh::shape::Cube, render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin,
    },
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

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
                    filter: "bevy_hanabi=warn,lifetime=trace".to_string(),
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings.into(),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "🎆 Hanabi — lifetime".to_string(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_systems(Update, bevy::window::close_on_esc)
        .add_plugins(HanabiPlugin)
        .add_plugins(WorldInspectorPlugin::default())
        .add_systems(Startup, setup)
        .run();

    Ok(())
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = Camera3dBundle {
        tonemapping: Tonemapping::None,
        ..default()
    };
    camera.transform.translation = Vec3::new(0.0, 0.0, 180.0);
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

    let lifetime1 = 12.;
    let lifetime2 = 3.;
    let lifetime3 = 0.75;
    let period = 3.;

    let mut gradient1 = Gradient::new();
    gradient1.add_key(0.0, Vec4::new(1.0, 0.0, 0.0, 1.0));
    gradient1.add_key(0.25, Vec4::new(1.0, 1.0, 0.0, 1.0));
    gradient1.add_key(0.5, Vec4::new(0.0, 1.0, 0.0, 1.0));
    gradient1.add_key(0.75, Vec4::new(0.0, 1.0, 1.0, 1.0));
    gradient1.add_key(1.0, Vec4::ONE);

    let writer1 = ExprWriter::new();
    let age1 = writer1.lit(0.).expr();
    let init_age1 = SetAttributeModifier::new(Attribute::AGE, age1);
    let lifetime1 = writer1.lit(lifetime1).expr();
    let init_lifetime1 = SetAttributeModifier::new(Attribute::LIFETIME, lifetime1);
    let init_pos1 = SetPositionSphereModifier {
        center: writer1.lit(Vec3::ZERO).expr(),
        radius: writer1.lit(5.).expr(),
        dimension: ShapeDimension::Volume,
    };
    let init_vel1 = SetVelocitySphereModifier {
        center: writer1.lit(Vec3::ZERO).expr(),
        speed: writer1.lit(2.).expr(),
    };
    let effect1 = effects.add(
        EffectAsset::new(
            512,
            Spawner::burst(50.0.into(), period.into()),
            writer1.finish(),
        )
        .with_name("emit:burst")
        .init(init_pos1)
        .init(init_vel1)
        .init(init_age1)
        .init(init_lifetime1)
        .render(ColorOverLifetimeModifier {
            gradient: gradient1,
        }),
    );

    commands
        .spawn((
            Name::new("burst 12s"),
            ParticleEffectBundle {
                effect: ParticleEffect::new(effect1),
                transform: Transform::from_translation(Vec3::new(-50., 0., 0.)),
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
    gradient2.add_key(0.0, Vec4::new(1.0, 0.0, 0.0, 1.0));
    gradient2.add_key(1.0, Vec4::new(1.0, 1.0, 0.0, 1.0));

    let writer2 = ExprWriter::new();
    let age2 = writer2.lit(0.).expr();
    let init_age2 = SetAttributeModifier::new(Attribute::AGE, age2);
    let lifetime2 = writer2.lit(lifetime2).expr();
    let init_lifetime2 = SetAttributeModifier::new(Attribute::LIFETIME, lifetime2);
    let init_pos2 = SetPositionSphereModifier {
        center: writer2.lit(Vec3::ZERO).expr(),
        radius: writer2.lit(5.).expr(),
        dimension: ShapeDimension::Volume,
    };
    let init_vel2 = SetVelocitySphereModifier {
        center: writer2.lit(Vec3::ZERO).expr(),
        speed: writer2.lit(2.).expr(),
    };
    let effect2 = effects.add(
        EffectAsset::new(
            512,
            Spawner::burst(50.0.into(), period.into()),
            writer2.finish(),
        )
        .with_name("emit:burst")
        .init(init_pos2)
        .init(init_vel2)
        .init(init_age2)
        .init(init_lifetime2)
        .render(ColorOverLifetimeModifier {
            gradient: gradient2,
        }),
    );

    commands
        .spawn((
            Name::new("burst 3s"),
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

    let mut gradient3 = Gradient::new();
    gradient3.add_key(0.0, Vec4::new(1.0, 0.0, 0.0, 1.0));
    gradient3.add_key(1.0, Vec4::new(0.75, 0.25, 0.0, 1.0));

    let writer3 = ExprWriter::new();
    let age3 = writer3.lit(0.).expr();
    let init_age3 = SetAttributeModifier::new(Attribute::AGE, age3);
    let lifetime3 = writer3.lit(lifetime3).expr();
    let init_lifetime3 = SetAttributeModifier::new(Attribute::LIFETIME, lifetime3);
    let init_pos3 = SetPositionSphereModifier {
        center: writer3.lit(Vec3::ZERO).expr(),
        radius: writer3.lit(5.).expr(),
        dimension: ShapeDimension::Volume,
    };
    let init_vel3 = SetVelocitySphereModifier {
        center: writer3.lit(Vec3::ZERO).expr(),
        speed: writer3.lit(2.).expr(),
    };
    let effect3 = effects.add(
        EffectAsset::new(
            512,
            Spawner::burst(50.0.into(), period.into()),
            writer3.finish(),
        )
        .with_name("emit:burst")
        .init(init_pos3)
        .init(init_vel3)
        .init(init_age3)
        .init(init_lifetime3)
        .render(ColorOverLifetimeModifier {
            gradient: gradient3,
        }),
    );

    commands
        .spawn((
            Name::new("burst 0.75s"),
            ParticleEffectBundle {
                effect: ParticleEffect::new(effect3),
                transform: Transform::from_translation(Vec3::new(50., 0., 0.)),
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
