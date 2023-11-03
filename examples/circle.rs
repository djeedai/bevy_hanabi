//! Example of using the circle spawner with random velocity.
//!
//! A sphere spawns dust in a circle.

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    log::LogPlugin,
    prelude::*,
    render::{render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin},
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
                    filter: "bevy_hanabi=warn,circle=trace".to_string(),
                })
                .set(RenderPlugin { wgpu_settings })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "🎆 Hanabi — circle".to_string(),
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
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = Camera3dBundle {
        tonemapping: Tonemapping::None,
        ..default()
    };
    camera.transform =
        Transform::from_xyz(3.0, 3.0, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    commands.spawn(camera);

    let texture_handle: Handle<Image> = asset_server.load("cloud.png");

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::splat(0.5));
    gradient.add_key(0.5, Vec4::splat(0.5));
    gradient.add_key(1.0, Vec4::new(0.5, 0.5, 0.5, 0.0));

    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(5.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let init_pos = SetPositionCircleModifier {
        center: writer.lit(Vec3::Y * 0.1).expr(),
        axis: writer.lit(Vec3::Y).expr(),
        radius: writer.lit(0.4).expr(),
        dimension: ShapeDimension::Surface,
    };

    let init_vel = SetVelocityCircleModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        axis: writer.lit(Vec3::Y).expr(),
        speed: (writer.lit(1.) + writer.lit(0.5) * writer.rand(ScalarType::Float)).expr(),
    };

    let effect = effects.add(
        EffectAsset::new(32768, Spawner::once(32.0.into(), true), writer.finish())
            .with_name("circle")
            .init(init_pos)
            .init(init_vel)
            .init(init_age)
            .init(init_lifetime)
            .render(ParticleTextureModifier {
                texture: texture_handle.clone(),
                sample_mapping: ImageSampleMapping::ModulateOpacityFromR,
            })
            .render(ColorOverLifetimeModifier { gradient })
            .render(SizeOverLifetimeModifier {
                gradient: Gradient::constant([0.2; 2].into()),
                screen_space_size: false,
            }),
    );

    // The ground
    commands
        .spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Plane {
                size: 4.0,
                ..default()
            })),
            material: materials.add(Color::BLUE.into()),
            ..Default::default()
        })
        .insert(Name::new("ground"));

    // The sphere
    commands
        .spawn(PbrBundle {
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
        .spawn(ParticleEffectBundle::new(effect))
        .insert(Name::new("effect"));
}
