//! Example of additive blend mode.
//!
//! This example demonstrate how to change the blend mode for the particle renderer.

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    log::LogPlugin,
    prelude::*,
    render::{
        camera::Projection, render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin,
    },
};
#[cfg(feature = "examples_world_inspector")]
use bevy_inspector_egui::quick::WorldInspectorPlugin;

use bevy_hanabi::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut wgpu_settings = WgpuSettings::default();
    wgpu_settings
        .features
        .set(WgpuFeatures::VERTEX_WRITABLE_STORAGE, true);

    let mut app = App::default();
    app.insert_resource(ClearColor(Color::DARK_GRAY))
        .add_plugins(
            DefaultPlugins
                .set(LogPlugin {
                    level: bevy::log::Level::WARN,
                    filter: "bevy_hanabi=warn,additive=trace".to_string(),
                    update_subscriber: None,
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings.into(),
                    synchronous_pipeline_compilation: false,
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "🎆 Hanabi — additive blending".to_string(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(HanabiPlugin);

    #[cfg(feature = "examples_world_inspector")]
    app.add_plugins(WorldInspectorPlugin::default());

    app.add_systems(Startup, setup)
        .add_systems(Update, bevy::window::close_on_esc)
        .run();

    Ok(())
}

fn setup(
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
) {
    let camera = Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        projection: Projection::Perspective(PerspectiveProjection {
            fov: 90.0,
            ..Default::default()
        }),
        tonemapping: Tonemapping::None,
        ..Default::default()
    };

    commands.spawn(camera);

    let texture_handle: Handle<Image> = asset_server.load("orange_circle.png");

    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(5.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let init_pos = SetPositionCircleModifier {
        center: writer.lit(0.).expr(),
        axis: writer.lit(Vec3::Z).expr(),
        radius: writer.lit(0.2).expr(),
        dimension: ShapeDimension::Volume,
    };

    let init_vel = SetVelocityCircleModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        axis: writer.lit(Vec3::Z).expr(),
        speed: (writer.lit(0.2) + writer.lit(0.2) * writer.rand(ScalarType::Float)).expr(),
    };

    // Use the F32_0 attribute as a per-particle rotation value, initialized on
    // spawn and constant after. The rotation angle is in radians, here randomly
    // selected in [0:2*PI].
    let rotation = (writer.rand(ScalarType::Float) * writer.lit(std::f32::consts::TAU)).expr();
    let init_rotation = SetAttributeModifier::new(Attribute::F32_0, rotation);

    let size = Vec2::splat(0.8);
    let effect = effects.add(
        EffectAsset::new(vec![32768], Spawner::rate(5.0.into()), writer.finish())
            .with_name("additive")
            .with_blending_mode(BlendingMode::Additive)
            .init(init_pos)
            .init(init_vel)
            .init(init_age)
            .init(init_lifetime)
            .init(init_rotation)
            .render(ParticleTextureModifier {
                texture: texture_handle,
                sample_mapping: ImageSampleMapping::Modulate,
            })
            .render(SetSizeModifier { size: size.into() }),
    );

    commands
        .spawn(ParticleEffectBundle::new(effect))
        .insert(Name::new("effect"));
}
