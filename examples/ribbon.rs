//! Uses with_ribbons to draw a "tracer" or a "trail" following an Entity.
//! The movement of the head particle is achieved by linking the particle
//! position to a CPU position using a [Property] in [move_head].

use bevy::math::vec4;
use bevy::prelude::*;
use bevy::{
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
    math::vec3,
};
use bevy_hanabi::prelude::*;

mod utils;
use utils::*;

// These determine the shape of the Spirograph:
// https://en.wikipedia.org/wiki/Spirograph#Mathematical_basis
const K: f32 = 0.64;
const L: f32 = 0.384;

const TIME_SCALE: f32 = 6.5;
const SHAPE_SCALE: f32 = 25.0;
const LIFETIME: f32 = 1.5;
const TRAIL_SPAWN_RATE: f32 = 256.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::make_test_app("ribbon")
        .add_systems(Startup, setup)
        .add_systems(Update, move_head)
        .run();
    app_exit.into_result()
}

fn setup(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
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

    let pos = writer.add_property("head_pos", Vec3::ZERO.into());
    let pos = writer.prop(pos);

    let move_modifier = SetAttributeModifier {
        attribute: Attribute::POSITION,
        value: pos.expr(),
    };

    let render_color = ColorOverLifetimeModifier {
        gradient: Gradient::linear(vec4(3.0, 0.0, 0.0, 1.0), vec4(3.0, 0.0, 0.0, 0.0)),
    };

    let effect = EffectAsset::new(256, Spawner::once(1.0.into(), true), writer.finish())
        .with_ribbons(32768, 1.0 / TRAIL_SPAWN_RATE, LIFETIME, 0)
        .with_simulation_space(SimulationSpace::Global)
        .init_groups(init_position_attr, ParticleGroupSet::single(0))
        .init_groups(init_velocity_attr, ParticleGroupSet::single(0))
        .init_groups(init_age_attr, ParticleGroupSet::single(0))
        .init_groups(init_lifetime_attr, ParticleGroupSet::single(0))
        .init_groups(init_size_attr, ParticleGroupSet::single(0))
        .update_groups(move_modifier, ParticleGroupSet::single(0))
        .render(SizeOverLifetimeModifier {
            gradient: Gradient::linear(Vec3::ONE, Vec3::ZERO),
            ..default()
        })
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

fn move_head(
    mut query: Query<&mut Transform, With<ParticleEffect>>,
    mut effect: Query<&mut EffectProperties>,
    timer: Res<Time>,
) {
    let Ok(mut properties) = effect.get_single_mut() else {
        return;
    };
    for mut transform in query.iter_mut() {
        let time = timer.elapsed_secs() * TIME_SCALE;
        let pos = vec3(
            (1.0 - K) * (time.clone().cos()) + (L * K) * (((1.0 - K) / K) * time.clone()).cos(),
            (1.0 - K) * (time.clone().sin()) - (L * K) * (((1.0 - K) / K) * time.clone()).sin(),
            0.0,
        ) * SHAPE_SCALE;

        properties.set("head_pos", (pos).into());
        transform.translation = pos;
    }
}
