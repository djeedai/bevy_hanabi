//! An example using the [`OrientModifier`] to force particles to always render
//! facing the camera, even when the view moves. This is particularly useful
//! with flat particles, to prevent stretching and maintain the illusion of 3D.
//!
//! This example also demonstrates the use of [`AlphaMode::Mask`] to render
//! particles with an alpha mask threshold. This feature is generally useful to
//! obtain non-square "cutout" opaque shapes. The alpha cutoff value is animated
//! over time with an expression, to show how the value affects the shape of the
//! particle.
//!
//! To obtain some visual diversity, each particle is spawned with its own
//! random color and random in-plane rotation.
//! - The color is stored into the per-particle [`Attribute::COLOR`], and
//!   automatically used in the render pass as the base color of the particle.
//!   The random value is a 4-component floating-point vector `vec4<f32>`
//!   obtained with the `rand()` expression, and is converted to a
//!   low-resolution `u32` RGBA color with the `pack4x8unorm()` expression.
//! - There's no built-in attribute to store the rotation angle, so the example
//!   makes use of the [`Attribute::F32_0`] attribute, one of the generic
//!   attibutes which you can use to store any per-particle floating-point
//!   value. The attribute is used here to storethe in-plane rotation angle, in
//!   radians, which will be passed to the [`OrientModifier`] to rotate the
//!   particles around their normal.
//!
//! Note: Particles can sometimes flicker. This is a current limitation of
//! Hanabi, which doesn't yet have any particle sorting feature, so the
//! rendering order may vary frame to frame. This is tracked on GitHub as issue
//! #183.

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    log::LogPlugin,
    prelude::*,
    render::{
        camera::Projection, render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin,
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
                    filter: "bevy_hanabi=warn,billboard=trace".to_string(),
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings.into(),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "🎆 Hanabi — billboard".to_string(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(HanabiPlugin)
        .add_plugins(WorldInspectorPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (bevy::window::close_on_esc, rotate_camera))
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
        transform: Transform::from_xyz(3.0, 3.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        projection: Projection::Perspective(PerspectiveProjection {
            fov: 120.0,
            ..Default::default()
        }),
        tonemapping: Tonemapping::None,
        ..Default::default()
    };

    commands.spawn(camera);

    let texture_handle: Handle<Image> = asset_server.load("cloud.png");

    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(5.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let init_pos = SetPositionCircleModifier {
        center: writer.lit(Vec3::Y * 0.1).expr(),
        axis: writer.lit(Vec3::Y).expr(),
        radius: writer.lit(1.).expr(),
        dimension: ShapeDimension::Volume,
    };

    let init_vel = SetVelocityCircleModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        axis: writer.lit(Vec3::Y).expr(),
        speed: (writer.lit(0.5) + writer.lit(0.2) * writer.rand(ScalarType::Float)).expr(),
    };

    // To give some visual diversity, we initialize each spawned particle with a
    // random per-particle color. The COLOR attribute is read back in the vertex
    // shader to initialize the particle's base color, which is later modulated
    // in this example with the texture of the ParticleTextureModifier.
    // Note that the ParticleTextureModifier uses
    // ImageSampleMapping::ModulateOpacityFromR so it will override
    // the alpha component of the color. Therefore we don't need to care about
    // rand() assigning a transparent value and making the particle invisible.
    let color = writer.rand(VectorType::VEC4F).pack4x8unorm();
    let init_color = SetAttributeModifier::new(Attribute::COLOR, color.expr());

    // Use the F32_0 attribute as a per-particle rotation value, initialized on
    // spawn and constant after. The rotation angle is in radians, here randomly
    // selected in [0:2*PI].
    let rotation = (writer.rand(ScalarType::Float) * writer.lit(std::f32::consts::TAU)).expr();
    let init_rotation = SetAttributeModifier::new(Attribute::F32_0, rotation);

    // Bounce the alpha cutoff value between 0 and 1, to show its effect on the
    // alpha masking
    let alpha_cutoff =
        ((writer.time() * writer.lit(2.)).sin() * writer.lit(0.3) + writer.lit(0.4)).expr();

    // The rotation of the OrientModifier is read from the F32_0 attribute (our
    // per-particle rotation)
    let rotation_attr = writer.attr(Attribute::F32_0).expr();

    let effect = effects.add(
        EffectAsset::new(32768, Spawner::rate(64.0.into()), writer.finish())
            .with_name("billboard")
            .with_alpha_mode(bevy_hanabi::AlphaMode::Mask(alpha_cutoff))
            .init(init_pos)
            .init(init_vel)
            .init(init_age)
            .init(init_lifetime)
            .init(init_rotation)
            .init(init_color)
            .render(ParticleTextureModifier {
                texture: texture_handle,
                sample_mapping: ImageSampleMapping::ModulateOpacityFromR,
            })
            .render(OrientModifier {
                mode: OrientMode::FaceCameraPosition,
                rotation: Some(rotation_attr),
            })
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
            transform: Transform::from_xyz(0.0, -0.5, 0.0),
            ..Default::default()
        })
        .insert(Name::new("ground"));

    commands
        .spawn(ParticleEffectBundle::new(effect))
        .insert(Name::new("effect"));
}

fn rotate_camera(time: Res<Time>, mut query: Query<&mut Transform, With<Camera>>) {
    let mut transform = query.single_mut();
    let radius_xz = 18_f32.sqrt();
    let a = (time.elapsed_seconds() * 0.3).sin();
    let (s, c) = a.sin_cos();
    *transform =
        Transform::from_xyz(c * radius_xz, 3.0, s * radius_xz).looking_at(Vec3::ZERO, Vec3::Y)
}
