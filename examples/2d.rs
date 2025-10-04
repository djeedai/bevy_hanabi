//! A particle system with a 2D camera.

use bevy::{camera::ScalingMode, prelude::*};
use bevy_hanabi::prelude::*;

mod utils;
use utils::*;

const DEMO_DESC: &str = include_str!("2d.txt");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::DemoApp::new("2d")
        .with_desc(DEMO_DESC)
        .with_desc_position(DescPosition::BottomRow)
        .build()
        .add_systems(Startup, setup)
        .add_systems(Update, update_plane)
        .run();
    app_exit.into_result()
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    // Spawn a 2D camera
    let mut proj = OrthographicProjection::default_2d();
    proj.scale = 1.0;
    proj.scaling_mode = ScalingMode::FixedVertical {
        viewport_height: 1.,
    };
    commands.spawn((Camera2d, Projection::Orthographic(proj)));

    // Spawn a reference white square in the center of the screen at Z=0
    commands.spawn((
        Mesh2d(meshes.add(Rectangle {
            half_size: Vec2::splat(0.1),
        })),
        MeshMaterial2d(materials.add(ColorMaterial {
            color: Color::WHITE,
            ..Default::default()
        })),
        Name::new("square"),
    ));

    // Create a color gradient for the particles
    let mut gradient = bevy_hanabi::Gradient::new();
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

    let mut module = writer.finish();

    let round = RoundModifier::constant(&mut module, 2.0 / 3.0);

    // Create a new effect asset spawning 30 particles per second from a circle
    // and slowly fading from blue-ish to transparent over their lifetime.
    // By default the asset spawns the particles at Z=0.
    let spawner = SpawnerSettings::rate(30.0.into());
    let effect = effects.add(
        EffectAsset::new(4096, spawner, module)
            .with_name("2d")
            .init(init_pos)
            .init(init_vel)
            .init(init_age)
            .init(init_lifetime)
            .render(SizeOverLifetimeModifier {
                gradient: bevy_hanabi::Gradient::constant(Vec3::splat(0.02)),
                screen_space_size: false,
            })
            .render(ColorOverLifetimeModifier::new(gradient))
            .render(round),
    );

    // Spawn an instance of the particle effect, and override its Z layer to
    // be above the reference white square previously spawned.
    commands.spawn((ParticleEffect::new(effect), Name::new("effect:2d")));
}

fn update_plane(time: Res<Time>, mut query: Query<&mut Transform, With<Mesh2d>>) {
    if let Ok(mut transform) = query.single_mut() {
        // Move the plane back and forth to show particles ordering relative to it
        transform.translation.z = (time.elapsed_secs() * 2.5).sin() * 0.045;
    }
}
