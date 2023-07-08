//! This example demonstrates the various position initializing modifiers.
//!
//! The example spawns a single burst of particles according to several position
//! modifiers, with a near-infinite lifetime (1 hour) and without any velocity
//! nor acceleration. This allows visualizing the distribution of particles on
//! spawn.

use std::f32::consts::PI;

use bevy::{
    log::LogPlugin,
    prelude::*,
    render::{
        mesh::shape::Cube, render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin,
    },
};
// use bevy_inspector_egui::quick::WorldInspectorPlugin;

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
                    filter: "bevy_hanabi=warn,spawn=trace".to_string(),
                })
                .set(RenderPlugin { wgpu_settings }),
        )
        .add_plugins(HanabiPlugin)
        // Have to wait for update.
        // .add_plugins(WorldInspectorPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (bevy::window::close_on_esc, rotate_camera))
        .run();

    Ok(())
}

const COLOR: Vec4 = Vec4::new(0.7, 0.7, 1.0, 1.0);
const SIZE: Vec2 = Vec2::splat(0.1);

fn base_effect(name: impl Into<String>) -> EffectAsset {
    let writer = ExprWriter::new();

    let lifetime = writer.lit(3600.).expr();
    let init_lifetime = InitAttributeModifier::new(Attribute::LIFETIME, lifetime);

    EffectAsset::new(32768, Spawner::once(COUNT.into(), true), writer.finish())
        .with_name(name)
        .init(init_lifetime)
        .render(BillboardModifier)
        .render(SetColorModifier {
            color: COLOR.into(),
        })
        .render(SetSizeModifier {
            size: SIZE.into(),
            screen_space_size: false,
        })
}

fn spawn_effect(
    commands: &mut Commands,
    name: String,
    transform: Transform,
    effect: Handle<EffectAsset>,
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
) {
    commands
        .spawn((
            Name::new(name),
            ParticleEffectBundle {
                effect: ParticleEffect::new(effect),
                transform,
                ..Default::default()
            },
        ))
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn(PbrBundle {
                mesh,
                material,
                ..Default::default()
            });
        });
}

const COUNT: f32 = 500_f32;

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = Camera3dBundle::default();
    camera.transform.translation = Vec3::new(0.0, 0.0, 50.0);
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

    let cube = meshes.add(Mesh::from(Cube { size: 0.1 }));
    let mat = materials.add(Color::PURPLE.into());

    spawn_effect(
        &mut commands,
        "InitPositionCircleModifier".to_string(),
        Transform::from_translation(Vec3::new(-20., 0., 0.)),
        effects.add(
            base_effect("InitPositionCircleModifier").init(InitPositionCircleModifier {
                center: Vec3::ZERO,
                axis: Vec3::Z,
                radius: 5.,
                dimension: ShapeDimension::Volume,
            }),
        ),
        cube.clone(),
        mat.clone(),
    );

    spawn_effect(
        &mut commands,
        "InitPositionSphereModifier".to_string(),
        Transform::from_translation(Vec3::new(0., 0., 0.)),
        effects.add(
            base_effect("InitPositionSphereModifier").init(InitPositionSphereModifier {
                center: Vec3::ZERO,
                radius: 5.,
                dimension: ShapeDimension::Volume,
            }),
        ),
        cube.clone(),
        mat.clone(),
    );

    spawn_effect(
        &mut commands,
        "InitPositionCone3dModifier".to_string(),
        Transform::from_translation(Vec3::new(20., -5., 0.))
            .with_rotation(Quat::from_rotation_z(1.)),
        effects.add(
            base_effect("InitPositionCone3dModifier").init(InitPositionCone3dModifier {
                height: 10.,
                base_radius: 1.,
                top_radius: 4.,
                dimension: ShapeDimension::Volume,
            }),
        ),
        cube.clone(),
        mat.clone(),
    );
}

fn rotate_camera(time: Res<Time>, mut query: Query<&mut Transform, With<Camera>>) {
    let mut transform = query.single_mut();
    let radius_xz = 50.;
    let a = ((time.elapsed_seconds() * 0.3).sin() + 1.) * PI / 2.;
    let (s, c) = a.sin_cos();
    *transform =
        Transform::from_xyz(c * radius_xz, 3.0, s * radius_xz).looking_at(Vec3::ZERO, Vec3::Y)
}
