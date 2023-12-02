//! Example of using the circle spawner with random velocity.
//!
//! A sphere spawns dust in a circle. Each dust particle is animated with a
//! [`FlipbookModifier`], from a procedurally generated sprite sheet.

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    log::LogPlugin,
    prelude::*,
    render::{render_resource::WgpuFeatures, settings::WgpuSettings, RenderPlugin},
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

use bevy_hanabi::prelude::*;

mod texutils;

use texutils::make_anim_img;

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
                    filter: "bevy_hanabi=warn,circle=trace".to_string(),
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings.into(),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "🎆 Hanabi — circle".to_string(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_systems(Update, bevy::window::close_on_esc)
        .add_plugins(HanabiPlugin)
        .add_plugins(WorldInspectorPlugin::default())
        .add_systems(Startup, setup)
        .run();

    Ok(())
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = Camera3dBundle {
        tonemapping: Tonemapping::None,
        ..default()
    };
    camera.transform =
        Transform::from_xyz(3.0, 3.0, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    commands.spawn(camera);

    // Procedurally create a sprite sheet representing an animated texture
    let sprite_size = UVec2::new(64, 64);
    let sprite_grid_size = UVec2::new(8, 8);
    let anim_img = make_anim_img(sprite_size, sprite_grid_size, Vec3::new(0.1, 0.1, 0.1));
    let texture_handle = images.add(anim_img);

    // The sprites form a grid, with a total animation frame count equal to the
    // number of sprites.
    let frame_count = sprite_grid_size.x * sprite_grid_size.y;

    let mut gradient = Gradient::new();
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

    let effect = effects.add(
        EffectAsset::new(
            32768,
            Spawner::burst(32.0.into(), 8.0.into()),
            writer.finish(),
        )
        .with_name("circle")
        .init(init_pos)
        .init(init_vel)
        .init(init_age)
        .init(init_lifetime)
        .update(update_sprite_index)
        .render(ParticleTextureModifier {
            texture: texture_handle.clone(),
            sample_mapping: ImageSampleMapping::ModulateOpacityFromR,
        })
        .render(FlipbookModifier { sprite_grid_size })
        .render(ColorOverLifetimeModifier { gradient })
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::constant([0.5; 2].into()),
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
            ..Default::default()
        })
        .insert(Name::new("ground"));

    // The sphere
    commands
        .spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::UVSphere {
                radius: 1.0,
                sectors: 32,
                stacks: 16,
            })),
            material: materials.add(Color::CYAN.into()),
            transform: Transform::from_translation(Vec3::Y),
            ..Default::default()
        })
        .insert(Name::new("sphere"));

    commands
        .spawn(ParticleEffectBundle::new(effect))
        .insert(Name::new("effect"));
}
