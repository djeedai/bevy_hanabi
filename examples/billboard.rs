//! An example using the [`BillboardModifier`] to force
//! particles to always render facing the camera.

use bevy::{
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
                    filter: "bevy_hanabi=warn,spawn=trace".to_string(),
                })
                .set(RenderPlugin { wgpu_settings }),
        )
        .add_system(bevy::window::close_on_esc)
        .add_plugin(HanabiPlugin)
        .add_plugin(WorldInspectorPlugin::default())
        .add_startup_system(setup)
        .add_system(rotate_camera)
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
        ..Default::default()
    };

    commands.spawn(camera);

    let texture_handle: Handle<Image> = asset_server.load("cloud.png");

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::ONE);
    gradient.add_key(0.5, Vec4::ONE);
    gradient.add_key(1.0, Vec4::new(1.0, 1.0, 1.0, 0.0));

    let writer = ExprWriter::new();

    let lifetime = writer.lit(5.).expr();
    let init_lifetime = InitAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let effect = effects.add(
        EffectAsset::new(32768, Spawner::rate(64.0.into()), writer.finish())
            .with_name("billboard")
            .init(InitPositionCircleModifier {
                center: Vec3::Y * 0.1,
                axis: Vec3::Y,
                radius: 1.0,
                dimension: ShapeDimension::Volume,
            })
            .init(InitVelocityCircleModifier {
                center: Vec3::ZERO,
                axis: Vec3::Y,
                speed: Value::Uniform((0.7, 0.5)),
            })
            .init(init_lifetime)
            .render(ParticleTextureModifier {
                texture: texture_handle,
            })
            .render(BillboardModifier {})
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
