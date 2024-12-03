//! Ordering
//!
//! This example demonstrates occluding particle effects behind opaque and
//! partially transparent objects. The occlusion is based on the built-in Bevy
//! ordering of transparent objects, which are sorted by their distance to the
//! camera, and therefore only works for non-overlapping objects. For Hanabi
//! effects, the origin of the emitter (the Entity with the ParticleEffect
//! component) is used as the origin of the object, and therefore the point from
//! which the distance to the camera is calculated. In this example, we
//! therefore ensure that the rectangles in front and behind the particle effect
//! do not overlap the bounding box of the effect itself.
use bevy::{
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
    prelude::*,
};
use bevy_hanabi::prelude::*;

mod utils;
use utils::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::make_test_app("ordering")
        .add_systems(Startup, setup)
        .run();
    app_exit.into_result()
}

/// Create the firework particle effect which will be rendered in-between other
/// PBR objects.
fn make_firework() -> EffectAsset {
    // Set the particles bright white (HDR; value=4.) so we can see the effect of
    // any colored object covering them.
    let mut color_gradient1 = Gradient::new();
    color_gradient1.add_key(0.0, Vec4::new(4.0, 4.0, 4.0, 1.0));
    color_gradient1.add_key(1.0, Vec4::new(4.0, 4.0, 4.0, 0.0));

    // Keep the size large so we can more visibly see the particles for longer, and
    // see the effect of alpha blending.
    let mut size_gradient1 = Gradient::new();
    size_gradient1.add_key(0.0, Vec3::ONE);
    size_gradient1.add_key(0.1, Vec3::ONE);
    size_gradient1.add_key(1.0, Vec3::ZERO);

    let writer = ExprWriter::new();

    // Give a bit of variation by randomizing the age per particle. This will
    // control the starting color and starting size of particles.
    let age = writer.lit(0.).uniform(writer.lit(0.2)).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    // Give a bit of variation by randomizing the lifetime per particle
    let lifetime = writer.lit(2.).uniform(writer.lit(3.)).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    // Add constant downward acceleration to simulate gravity
    let accel = writer.lit(Vec3::Y * -8.).expr();
    let update_accel = AccelModifier::new(accel);

    // Add drag to make particles slow down a bit after the initial explosion
    let drag = writer.lit(5.).expr();
    let update_drag = LinearDragModifier::new(drag);

    let init_pos = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(2.).expr(),
        dimension: ShapeDimension::Volume,
    };

    // Give a bit of variation by randomizing the initial speed
    let init_vel = SetVelocitySphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        speed: (writer.rand(ScalarType::Float) * writer.lit(20.) + writer.lit(60.)).expr(),
    };

    EffectAsset::new(2048, Spawner::rate(128.0.into()), writer.finish())
        .with_name("firework")
        .init(init_pos)
        .init(init_vel)
        .init(init_age)
        .init(init_lifetime)
        .update(update_drag)
        .update(update_accel)
        // Note: we (ab)use the ColorOverLifetimeModifier to set a fixed color hard-coded in the
        // render shader, without having to store a per-particle color. This is an optimization.
        .render(ColorOverLifetimeModifier {
            gradient: color_gradient1,
        })
        .render(SizeOverLifetimeModifier {
            gradient: size_gradient1,
            screen_space_size: false,
        })
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Transform::from_translation(Vec3::new(0., 0., 50.)),
        Camera {
            hdr: true,
            clear_color: Color::BLACK.into(),
            ..default()
        },
        Camera3d::default(),
        Tonemapping::None,
        Bloom::default(),
    ));

    let effect1 = effects.add(make_firework());

    commands.spawn((
        Name::new("firework"),
        ParticleEffectBundle {
            effect: ParticleEffect::new(effect1),
            transform: Transform {
                translation: Vec3::Z,
                ..default()
            },
            ..default()
        },
    ));

    // Red background at origin, with alpha blending
    commands.spawn((
        Transform {
            scale: Vec3::splat(50.),
            ..default()
        },
        Mesh3d(meshes.add(Mesh::from(Rectangle {
            half_size: Vec2 { x: 0.5, y: 0.5 },
        }))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::linear_rgba(1., 0., 0., 0.5),
            alpha_mode: bevy::prelude::AlphaMode::Blend,
            ..default()
        })),
    ));

    // Blue rectangle in front of particles, with alpha blending
    commands.spawn((
        Transform {
            translation: Vec3::Y * 6. + Vec3::Z * 40.,
            scale: Vec3::splat(10.),
            ..default()
        },
        Mesh3d(meshes.add(Mesh::from(Rectangle {
            half_size: Vec2 { x: 0.5, y: 0.5 },
        }))),
        MeshMaterial3d(materials.add(StandardMaterial {
            // Keep the alpha quite high, because the particles are very bright (HDR, value=4.)
            // so otherwise we can't see the attenuation of the blue box over the white particles.
            base_color: Color::linear_rgba(0., 0., 1., 0.95),
            alpha_mode: bevy::prelude::AlphaMode::Blend,
            ..default()
        })),
    ));

    // Green square in front of particles, without alpha blending
    commands.spawn((
        Transform {
            translation: Vec3::Y * -6. + Vec3::Z * 40.,
            scale: Vec3::splat(10.),
            ..default()
        },
        Mesh3d(meshes.add(Mesh::from(Rectangle {
            half_size: Vec2 { x: 0.5, y: 0.5 },
        }))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: utils::COLOR_GREEN,
            alpha_mode: bevy::prelude::AlphaMode::Opaque,
            ..default()
        })),
    ));
}
