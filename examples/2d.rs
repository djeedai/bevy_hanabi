//! A particle system with a 2D camera.
//!
//! The particle effect instance override its `z_layer_2d` field, which can be
//! tweaked at runtime via the egui inspector to move the 2D rendering layer of
//! particle above or below the reference square.

use bevy::{
    log::LogPlugin,
    prelude::*,
    render::{
        camera::ScalingMode, render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin,
    },
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
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
                    filter: "bevy_hanabi=warn,2d=trace".to_string(),
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings.into(),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "🎆 Hanabi — 2d".to_string(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(HanabiPlugin)
        .add_plugins(WorldInspectorPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (bevy::window::close_on_esc, update_plane))
        .run();

    Ok(())
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    // Spawn a 2D camera
    let mut camera = Camera2dBundle::default();
    camera.projection.scale = 1.0;
    camera.projection.scaling_mode = ScalingMode::FixedVertical(1.);
    commands.spawn(camera);

    // Spawn a reference white square in the center of the screen at Z=0
    commands
        .spawn(MaterialMesh2dBundle {
            mesh: meshes
                .add(Mesh::from(shape::Quad {
                    size: Vec2::splat(0.2),
                    ..Default::default()
                }))
                .into(),
            material: materials.add(ColorMaterial {
                color: Color::WHITE,
                ..Default::default()
            }),
            ..Default::default()
        })
        .insert(Name::new("square"));

    // Create a color gradient for the particles
    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(0.5, 0.5, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::new(0.5, 0.5, 1.0, 0.0));

    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(5.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let init_pos = SetPositionCircleModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        axis: writer.lit(Vec3::Z).expr(),
        radius: writer.lit(0.05).expr(),
        dimension: ShapeDimension::Surface,
    };

    let init_vel = SetVelocityCircleModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        axis: writer.lit(Vec3::Z).expr(),
        speed: writer.lit(0.1).expr(),
    };

    // Create a new effect asset spawning 30 particles per second from a circle
    // and slowly fading from blue-ish to transparent over their lifetime.
    // By default the asset spawns the particles at Z=0.
    let spawner = Spawner::rate(30.0.into());
    let effect = effects.add(
        EffectAsset::new(4096, spawner, writer.finish())
            .with_name("2d")
            .init(init_pos)
            .init(init_vel)
            .init(init_age)
            .init(init_lifetime)
            .render(SizeOverLifetimeModifier {
                gradient: Gradient::constant(Vec2::splat(0.02)),
                screen_space_size: false,
            })
            .render(ColorOverLifetimeModifier { gradient }),
    );

    // Spawn an instance of the particle effect, and override its Z layer to
    // be above the reference white square previously spawned.
    commands
        .spawn(ParticleEffectBundle {
            // Assign the Z layer so it appears in the egui inspector and can be modified at runtime
            effect: ParticleEffect::new(effect).with_z_layer_2d(Some(0.1)),
            ..default()
        })
        .insert(Name::new("effect:2d"));
}

fn update_plane(time: Res<Time>, mut query: Query<&mut Transform, With<Mesh2dHandle>>) {
    let mut transform = query.single_mut();
    // Move the plane back and forth to show particles ordering relative to it
    transform.translation.z = (time.elapsed_seconds() * 2.5).sin() * 0.045 + 0.1;
}
