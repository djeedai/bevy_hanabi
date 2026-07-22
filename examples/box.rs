//! An example using the [`SetPositionBoxModifier`].

use std::f32::consts::FRAC_PI_2;

use bevy::{core_pipeline::tonemapping::Tonemapping, prelude::*};
use bevy_hanabi::{prelude::*, Gradient};

mod utils;
use utils::*;

const DEMO_DESC: &str = include_str!("box.txt");

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
            half_size: Vec2::splat(7.5),
        })),
        MeshMaterial3d(materials.add(utils::COLOR_BLUE)),
        Name::new("ground"),
    ));

    commands.spawn(build_particle_system(
        Vec3::new(-3., 1., 0.),
        ShapeDimension::Surface,
        &mut effects,
    ));
    commands.spawn(build_particle_system(
        Vec3::new(3., 1., 0.),
        ShapeDimension::Volume,
        &mut effects,
    ));
}

fn rotate_camera(time: Res<Time>, mut camera_transform: Single<&mut Transform, With<Camera3d>>) {
    let radius_xz = 421_f32.sqrt();
    let a = (time.elapsed_secs() * 0.3).sin();
    let (s, c) = a.sin_cos();
    **camera_transform =
        Transform::from_xyz(c * radius_xz, 7.5, s * radius_xz).looking_at(Vec3::ZERO, Vec3::Y)
}

fn build_particle_system(
    translation: Vec3,
    dimension: ShapeDimension,
    effects: &mut Assets<EffectAsset>,
) -> impl Bundle {
    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(0.5).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let init_pos = SetPositionBoxModifier {
        scale: writer.lit(Vec3::new(2., 2., 2.)).expr(),
        dimension,
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

    // The rotation of the OrientModifier is read from the F32_0 attribute (our
    // per-particle rotation)
    let rotation_attr = writer.attr(Attribute::F32_0).expr();

    let module = writer.finish();

    let effect = effects.add(
        EffectAsset::new(32768, SpawnerSettings::rate(64.0.into()), module)
            .with_name("box")
            .init(init_pos)
            .init(init_age)
            .init(init_lifetime)
            .init(init_rotation)
            .init(init_color)
            .render(OrientModifier {
                mode: OrientMode::FaceCameraPosition,
                rotation: Some(rotation_attr),
            })
            .render(SizeOverLifetimeModifier {
                gradient: Gradient::constant(Vec3::splat(0.2)),
                screen_space_size: false,
            }),
    );

    (
        Transform::from_translation(translation),
        ParticleEffect::new(effect),
        Name::new(format!("Box {:?}", dimension)),
    )
}
