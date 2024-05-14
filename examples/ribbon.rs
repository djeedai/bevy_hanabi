//! Draws a trail and connects the trails using a ribbon.

use bevy::math::vec4;
use bevy::prelude::*;
use bevy::{
    core_pipeline::{bloom::BloomSettings, tonemapping::Tonemapping},
    math::vec3,
};
use bevy_hanabi::prelude::*;

#[cfg(feature = "examples_world_inspector")]
use bevy_inspector_egui::quick::WorldInspectorPlugin;

// These determine the shape of the Spirograph:
// https://en.wikipedia.org/wiki/Spirograph#Mathematical_basis
const K: f32 = 0.64;
const L: f32 = 0.384;

const TIME_SCALE: f32 = 10.0;
const SHAPE_SCALE: f32 = 25.0;
const LIFETIME: f32 = 2.5;
const TRAIL_SPAWN_RATE: f32 = 256.0;

fn main() {
    let mut app = App::default();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "🎆 Hanabi — ribbon".to_string(),
            ..default()
        }),
        ..default()
    }))
    .add_plugins(HanabiPlugin)
    .add_systems(Update, bevy::window::close_on_esc)
    .add_systems(Startup, setup)
    .add_systems(Update, move_particle_effect);

    #[cfg(feature = "examples_world_inspector")]
    app.add_plugins(WorldInspectorPlugin::default());

    app.run();
}

fn setup(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0., 0., 50.)),
            camera: Camera {
                hdr: true,
                clear_color: Color::BLACK.into(),
                ..default()
            },
            tonemapping: Tonemapping::None,
            ..default()
        },
        BloomSettings::default(),
    ));

    let writer = ExprWriter::new();

    let init_position_attr = SetAttributeModifier {
        attribute: Attribute::POSITION,
        value: writer.lit(Vec3::ZERO).expr(),
    };

    let init_velocity_attr = SetAttributeModifier {
        attribute: Attribute::VELOCITY,
        value: writer.lit(Vec3::ZERO).expr(),
    };

    let init_age_attr = SetAttributeModifier {
        attribute: Attribute::AGE,
        value: writer.lit(0.0).expr(),
    };

    let init_lifetime_attr = SetAttributeModifier {
        attribute: Attribute::LIFETIME,
        value: writer.lit(999999.0).expr(),
    };

    let init_size_attr = SetAttributeModifier {
        attribute: Attribute::SIZE,
        value: writer.lit(0.5).expr(),
    };

    let clone_modifier = CloneModifier::new(1.0 / TRAIL_SPAWN_RATE, 1);

    let time = writer.time().mul(writer.lit(TIME_SCALE));

    let move_modifier = SetAttributeModifier {
        attribute: Attribute::POSITION,
        value: (WriterExpr::vec3(
            writer.lit(1.0 - K).mul(time.clone().cos())
                + writer.lit(L * K) * (writer.lit((1.0 - K) / K) * time.clone()).cos(),
            writer.lit(1.0 - K).mul(time.clone().sin())
                - writer.lit(L * K) * (writer.lit((1.0 - K) / K) * time.clone()).sin(),
            writer.lit(0.0),
        ) * writer.lit(SHAPE_SCALE))
        .expr(),
    };

    let update_lifetime_attr = SetAttributeModifier {
        attribute: Attribute::LIFETIME,
        value: writer.lit(LIFETIME).expr(),
    };

    let render_color = ColorOverLifetimeModifier {
        gradient: Gradient::linear(vec4(3.0, 0.0, 0.0, 1.0), vec4(3.0, 0.0, 0.0, 0.0)),
    };

    let effect = EffectAsset::new(
        vec![256, 32768],
        Spawner::once(1.0.into(), true),
        writer.finish(),
    )
    .with_name("ribbon")
    .with_simulation_space(SimulationSpace::Global)
    .init(init_position_attr)
    .init(init_velocity_attr)
    .init(init_age_attr)
    .init(init_lifetime_attr)
    .init(init_size_attr)
    .update_groups(move_modifier, ParticleGroupSet::single(0))
    .update_groups(clone_modifier, ParticleGroupSet::single(0))
    .update_groups(update_lifetime_attr, ParticleGroupSet::single(1))
    .render(RibbonModifier)
    .render_groups(render_color, ParticleGroupSet::single(1));

    let effect = effects.add(effect);

    commands
        .spawn(ParticleEffectBundle {
            effect: ParticleEffect::new(effect),
            transform: Transform::IDENTITY,
            ..default()
        })
        .insert(Name::new("ribbon"));
}

fn move_particle_effect(mut query: Query<&mut Transform, With<ParticleEffect>>, timer: Res<Time>) {
    let theta = timer.elapsed_seconds() * 1.0;
    for mut transform in query.iter_mut() {
        transform.translation = vec3(f32::cos(theta), f32::sin(theta), 0.0) * 5.0;
    }
}
