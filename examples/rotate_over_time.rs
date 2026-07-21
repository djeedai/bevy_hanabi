//! An example using the [`RotateOverTimeModifier`].

use std::f32::consts::FRAC_PI_2;

use bevy::{core_pipeline::tonemapping::Tonemapping, prelude::*};
use bevy_hanabi::{prelude::*, Gradient};

mod utils;
use utils::*;

const DEMO_DESC: &str = include_str!("rotate_over_time.txt");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::DemoApp::new("box")
        .with_desc(DEMO_DESC)
        .build()
        .add_systems(Startup, setup)
        .add_systems(Update, rotate_camera)
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
        Transform::from_xyz(3.0, 3.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 120.0,
            ..default()
        }),
        Tonemapping::None,
    ));

    // The ground
    commands.spawn((
        Transform::from_xyz(0.0, -0.5, 0.0)
            * Transform::from_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
        Mesh3d(meshes.add(Rectangle {
            half_size: Vec2::splat(2.0),
        })),
        MeshMaterial3d(materials.add(utils::COLOR_BLUE)),
        Name::new("ground"),
    ));

    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(3.).expr();
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

    let init_axis_x = SetAttributeModifier::new(Attribute::AXIS_X, writer.lit(Vec3::X).expr());
    let init_axis_y = SetAttributeModifier::new(Attribute::AXIS_Y, writer.lit(Vec3::Y).expr());
    let init_axis_z = SetAttributeModifier::new(Attribute::AXIS_Z, writer.lit(Vec3::Z).expr());

    // Particle will complete 3/4 of a rotation around Y axis per second
    let rotate_over_time = RotateOverTimeModifier {
        rotation: writer
            .lit(MatrixValue::from(
                Transform::from_rotation(Quat::from_euler(
                    EulerRot::XYZ,
                    0.,
                    270.0f32.to_radians(),
                    0.,
                ))
                .to_matrix(),
            ))
            .expr(),
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

    let module = writer.finish();

    let effect = effects.add(
        EffectAsset::new(32768, SpawnerSettings::rate(64.0.into()), module)
            .with_name("box")
            .init(init_pos)
            .init(init_vel)
            .init(init_axis_x)
            .init(init_axis_y)
            .init(init_axis_z)
            .init(init_age)
            .init(init_lifetime)
            .init(init_color)
            .update(rotate_over_time)
            .render(SizeOverLifetimeModifier {
                gradient: Gradient::constant(Vec3::splat(0.2)),
                screen_space_size: false,
            }),
    );

    commands.spawn((
        Transform::from_translation(Vec3::new(0., 1., 0.)),
        ParticleEffect::new(effect),
        Name::new("Rotate Over Time"),
    ));
}

fn rotate_camera(time: Res<Time>, mut camera_transform: Single<&mut Transform, With<Camera3d>>) {
    let radius_xz = 18_f32.sqrt();
    let a = (time.elapsed_secs() * 0.3).sin();
    let (s, c) = a.sin_cos();
    **camera_transform =
        Transform::from_xyz(c * radius_xz, 3.0, s * radius_xz).looking_at(Vec3::ZERO, Vec3::Y)
}
