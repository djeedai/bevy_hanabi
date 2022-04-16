//! A particle system with a 2D camera.

use bevy::{
    prelude::*,
    render::{camera::ScalingMode, render_resource::WgpuFeatures, settings::WgpuSettings},
    sprite::MaterialMesh2dBundle,
};
//use bevy_inspector_egui::WorldInspectorPlugin;

use bevy_hanabi::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = WgpuSettings::default();
    options
        .features
        .set(WgpuFeatures::VERTEX_WRITABLE_STORAGE, true);
    App::default()
        .insert_resource(ClearColor(Color::DARK_GRAY))
        .insert_resource(options)
        .insert_resource(bevy::log::LogSettings {
            level: bevy::log::Level::WARN,
            filter: "bevy_hanabi=error,spawn=trace".to_string(),
        })
        .add_plugins(DefaultPlugins)
        .add_system(bevy::input::system::exit_on_esc_system)
        .add_plugin(HanabiPlugin)
        //.add_plugin(WorldInspectorPlugin::new())
        .add_startup_system(setup)
        .run();

    Ok(())
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mut camera = OrthographicCameraBundle::new_2d();
    camera.orthographic_projection.scale = 1.0;
    camera.orthographic_projection.scaling_mode = ScalingMode::FixedVertical;
    //camera.transform.translation.z = camera.orthographic_projection.far / 2.0;
    commands.spawn_bundle(camera);

    let mut ball = commands.spawn_bundle(MaterialMesh2dBundle {
        mesh: meshes
            .add(Mesh::from(shape::Quad {
                size: Vec2::splat(0.1),
                ..Default::default()
            }))
            .into(),
        material: materials.add(ColorMaterial {
            color: Color::WHITE,
            ..Default::default()
        }),
        ..Default::default()
    });
    ball.insert(Name::new("ball"));

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(0.5, 0.5, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::new(0.5, 0.5, 1.0, 0.0));

    let spawner = Spawner::rate(30.0.into());
    let effect = effects.add(
        EffectAsset {
            name: "Effect".into(),
            capacity: 32768,
            spawner,
            ..Default::default()
        }
        .init(PositionCircleModifier {
            radius: 0.05,
            speed: 0.1.into(),
            dimension: ShapeDimension::Surface,
            ..Default::default()
        })
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::constant(Vec2::splat(0.02)),
        })
        .render(ColorOverLifetimeModifier { gradient }),
    );

    ball.with_children(|node| {
        node.spawn_bundle(ParticleEffectBundle::new(effect).with_spawner(spawner))
            .insert(Name::new("effect:2d"));
    });
}
