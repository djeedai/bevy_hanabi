//! Example showing the effect of [`SimulationCondition`] to continue simulating
//! or not when the entity is invisible.
//!
//! This example spawns two effects:
//! - The top one is only simulated when visible
//!   ([`SimulationCondition::WhenVisible`]; default behavior).
//! - The bottom one is always simulated, even when invisible
//!   ([`SimulationCondition::Always`]).
//!
//! A system updates the visibility of the effects, toggling it ON and OFF. We
//! can observe that the top effect continue to be simulated while hidden.

use std::time::Duration;

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
                    filter: "bevy_hanabi=warn,visibility=trace".to_string(),
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings.into(),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "🎆 Hanabi — visibility".to_string(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(HanabiPlugin)
        .add_plugins(WorldInspectorPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (bevy::window::close_on_esc, update))
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

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(1.0, 0.0, 0.0, 1.0));
    gradient.add_key(0.25, Vec4::new(0.0, 1.0, 0.0, 1.0));
    gradient.add_key(0.5, Vec4::new(0.0, 0.0, 1.0, 1.0));
    gradient.add_key(0.75, Vec4::new(0.0, 1.0, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::new(1.0, 1.0, 1.0, 1.0));

    let writer = ExprWriter::new();

    // Set the same constant velocity to all particles so they stay grouped
    // together. This is not really representative of a real world visual effect,
    // but is useful for the purpose of this example.
    let velocity = writer.lit(Vec3::X * 3.).expr();
    let init_velocity = SetAttributeModifier::new(Attribute::VELOCITY, velocity);

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(15.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let init_pos = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(5.).expr(),
        dimension: ShapeDimension::Volume,
    };

    let mut asset = EffectAsset::new(
        4096,
        Spawner::burst(50.0.into(), 15.0.into()),
        writer.finish(),
    )
    .with_simulation_condition(SimulationCondition::WhenVisible)
    .init(init_pos)
    .init(init_velocity)
    .init(init_age)
    .init(init_lifetime)
    //.update(AccelModifier::constant(Vec3::new(0., 2., 0.)))
    .render(ColorOverLifetimeModifier { gradient });
    let effect1 = effects.add(asset.clone());

    // Reference cube to visualize the emit origin
    commands
        .spawn(PbrBundle {
            mesh: cube.clone(),
            material: mat.clone(),
            transform: Transform::from_translation(Vec3::new(-30., -20., 0.)),
            ..Default::default()
        })
        .with_children(|p| {
            p.spawn((
                Name::new("WhenVisible"),
                ParticleEffectBundle {
                    effect: ParticleEffect::new(effect1),
                    ..Default::default()
                },
            ));
        });

    asset.simulation_condition = SimulationCondition::Always;
    let effect2 = effects.add(asset);

    // Reference cube to visualize the emit origin
    commands
        .spawn(PbrBundle {
            mesh: cube.clone(),
            material: mat.clone(),
            transform: Transform::from_translation(Vec3::new(-30., 20., 0.)),
            ..Default::default()
        })
        .with_children(|p| {
            p.spawn((
                Name::new("Always"),
                ParticleEffectBundle {
                    effect: ParticleEffect::new(effect2),
                    ..Default::default()
                },
            ));
        });
}

fn update(
    time: Res<Time>,
    mut last_time: Local<u64>,
    mut query: Query<&mut Visibility, With<ParticleEffect>>,
) {
    // Every half second, toggle the visibility. For the left effect (WhenVisible)
    // this will effectively halve the simulation time compared to the real
    // wall-clock time. For the right effect (Always) nothing will change because it
    // continues to simulate when hidden.
    // warn!(
    //     "t={} l={} d={}",
    //     time.elapsed().as_millis(),
    //     *last_time,
    //     (time.elapsed() - Duration::from_millis(*last_time)).as_millis()
    // );
    if time.elapsed() - Duration::from_millis(*last_time) >= Duration::from_millis(1500) {
        // warn!("TOGGLE: ");
        *last_time = time.elapsed().as_millis() as u64;
        for mut visibility in query.iter_mut() {
            *visibility = if *visibility == Visibility::Visible {
                Visibility::Hidden
            } else {
                Visibility::Visible
            };
        }
    }
}
