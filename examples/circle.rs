//! Example of using the circle spawner with random velocity.
//!
//! A sphere spawns dust in a circle. Each dust particle is animated with a
//! [`FlipbookModifier`], from a procedurally generated sprite sheet.

use std::f32::consts::FRAC_PI_2;

use bevy::{core_pipeline::tonemapping::Tonemapping, prelude::*};
use bevy_hanabi::prelude::*;

mod texutils;
mod utils;

use texutils::make_anim_img;
use utils::*;

const DEMO_DESC: &str = include_str!("circle.txt");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::DemoApp::new("circle")
        .with_desc(DEMO_DESC)
        .build()
        .add_systems(Startup, setup)
        .run();
    app_exit.into_result()
}

fn setup(
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Transform::from_xyz(3.0, 3.0, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        Camera3d::default(),
        Tonemapping::None,
    ));

    // Procedurally create a sprite sheet representing an animated texture
    let sprite_size = UVec2::new(64, 64);
    let sprite_grid_size = UVec2::new(8, 8);
    let anim_img = make_anim_img(sprite_size, sprite_grid_size, Vec3::new(0.1, 0.1, 0.1));
    let texture_handle = images.add(anim_img);

    // Also use a serialized asset
    let texture_handle2 = asset_server.load("ramp.png");

    // The sprites form a grid, with a total animation frame count equal to the
    // number of sprites.
    let frame_count = sprite_grid_size.x * sprite_grid_size.y;

    let mut gradient = bevy_hanabi::Gradient::new();
    gradient.add_key(0.0, Vec4::ONE);
    gradient.add_key(0.5, Vec4::ONE);
    gradient.add_key(1.0, Vec3::ONE.extend(0.));

    let writer = ExprWriter::new();

    // Initialize the AGE to a random [0:1] value to ensure not all particles start
    // their animation at the same frame. Otherwise they all animate in sync.
    let age = writer.rand(ScalarType::Float).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    // All particles stay alive until their AGE is 5 seconds. Note that this doesn't
    // mean they live for 5 seconds; if the AGE is initialized to a non-zero value
    // at spawn, the total particle lifetime is (LIFETIME - AGE).
    let lifetime = writer.lit(5.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let init_pos = SetPositionCircleModifier {
        center: writer.lit(Vec3::Y * 0.1).expr(),
        axis: writer.lit(Vec3::Y).expr(),
        radius: writer.lit(0.4).expr(),
        dimension: ShapeDimension::Surface,
    };

    let init_vel = SetVelocityCircleModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        axis: writer.lit(Vec3::Y).expr(),
        speed: (writer.lit(1.) + writer.lit(0.5) * writer.rand(ScalarType::Float)).expr(),
    };

    // Animate the SPRITE_INDEX attribute of each particle based on its age.
    // We want to animate back and forth the index in [0:N-1] where N is the total
    // number of sprites in the sprite sheet.
    // - For the back and forth, we build a linear ramp z 1 -> 0 -> 1 with abs(x)
    //   and y linear in [-1:1]
    // - To get that linear cyclic y variable in [-1:1], we build a linear cyclic x
    //   variable in [0:1]
    // - To get that linear cyclic x variable in [0:1], we take the fractional part
    //   of the age
    // - Because we want to have one full cycle every couple of seconds, we need to
    //   scale down the age value (0.02)
    // - Finally the linear ramp z is scaled to the [0:N-1] range
    // Putting it together we get:
    //   sprite_index = i32(
    //       abs(fract(particle.age * 0.02) * 2. - 1.) * frame_count
    //     ) % frame_count;
    let sprite_index = writer
        .attr(Attribute::AGE)
        .mul(writer.lit(0.1))
        .fract()
        .mul(writer.lit(2.))
        .sub(writer.lit(1.))
        .abs()
        .mul(writer.lit(frame_count as f32))
        .cast(ScalarType::Int)
        .rem(writer.lit(frame_count as i32))
        .expr();
    let update_sprite_index = SetAttributeModifier::new(Attribute::SPRITE_INDEX, sprite_index);

    let texture_slot = writer.lit(0u32).expr();
    let texture_slot2 = writer.lit(1u32).expr();

    let mut module = writer.finish();
    module.add_texture_slot("color");
    module.add_texture_slot("shape");

    let effect = effects.add(
        EffectAsset::new(
            32768,
            SpawnerSettings::burst(32.0.into(), 8.0.into()),
            module,
        )
        .with_name("circle")
        .init(init_pos)
        .init(init_vel)
        .init(init_age)
        .init(init_lifetime)
        .update(update_sprite_index)
        .render(ParticleTextureModifier {
            texture_slot,
            sample_mapping: ImageSampleMapping::ModulateOpacityFromR,
        })
        .render(ParticleTextureModifier {
            texture_slot: texture_slot2,
            sample_mapping: ImageSampleMapping::ModulateRGB,
        })
        .render(FlipbookModifier { sprite_grid_size })
        .render(ColorOverLifetimeModifier::new(gradient))
        .render(SizeOverLifetimeModifier {
            gradient: bevy_hanabi::Gradient::constant([0.5; 3].into()),
            screen_space_size: false,
        }),
    );

    // The ground
    commands.spawn((
        Transform::from_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
        Mesh3d(meshes.add(Rectangle {
            half_size: Vec2::splat(2.0),
        })),
        MeshMaterial3d(materials.add(utils::COLOR_BLUE)),
        Name::new("ground"),
    ));

    // The sphere
    commands.spawn((
        Transform::from_translation(Vec3::Y),
        Mesh3d(meshes.add(Sphere { radius: 1.0 })),
        MeshMaterial3d(materials.add(utils::COLOR_CYAN)),
        Name::new("sphere"),
    ));

    commands.spawn((
        ParticleEffect::new(effect),
        EffectMaterial {
            images: vec![texture_handle, texture_handle2],
        },
        Name::new("effect"),
    ));
}
