//! Example of using random spawner params.
//! Spawns a random number of particles at random times.

use bevy::{core_pipeline::tonemapping::Tonemapping, prelude::*};
use bevy_hanabi::prelude::*;

mod utils;
use utils::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::make_test_app("random")
        .add_systems(Startup, setup)
        .run();
    app_exit.into_result()
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Transform::from_translation(Vec3::Z * 100.),
        Camera3d::default(),
        Tonemapping::None,
    ));

    commands.spawn(DirectionalLight {
        color: Color::WHITE,
        // Crank the illuminance way (too) high to make the reference cube clearly visible
        illuminance: 100000.,
        shadows_enabled: false,
        ..Default::default()
    });

    let cube = meshes.add(Cuboid {
        half_size: Vec3::splat(0.5),
    });
    let mat = materials.add(utils::COLOR_PURPLE);

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(0.0, 0.0, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::new(0.0, 0.0, 1.0, 0.0));

    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(5.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let accel = writer.lit(Vec3::Y * 5.).expr();
    let update_accel = AccelModifier::new(accel);

    let init_pos = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(5.).expr(),
        dimension: ShapeDimension::Volume,
    };

    let init_vel = SetVelocitySphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        speed: writer.lit(2.).expr(),
    };

    let effect = effects.add(
        EffectAsset::new(
            32768,
            Spawner::burst(CpuValue::Uniform((1., 100.)), CpuValue::Uniform((1., 4.))),
            writer.finish(),
        )
        .with_name("emit:burst")
        .init(init_pos)
        .init(init_vel)
        .init(init_age)
        .init(init_lifetime)
        .update(update_accel)
        .render(ColorOverLifetimeModifier { gradient }),
    );

    commands
        .spawn((
            Name::new("emit:random"),
            ParticleEffectBundle {
                effect: ParticleEffect::new(effect),
                transform: Transform::from_translation(Vec3::new(0., 0., 0.)),
                ..Default::default()
            },
        ))
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn((Mesh3d(cube.clone()), MeshMaterial3d(mat.clone())));
        });
}
