//! A circle bounces around in a box and spawns particles when it hits the wall.
//!
//! This example demonstrates the use of effect properties to control some
//! particle properties like the spawn velocity direction and initial particle
//! color. Particles are spawned "manually" with [`Spawner::reset()`], providing
//! total control to the application.

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    log::LogPlugin,
    math::Vec3Swizzles,
    prelude::*,
    render::{
        camera::{Projection, ScalingMode},
        render_resource::WgpuFeatures,
        settings::WgpuSettings,
        RenderPlugin,
    },
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

use bevy_hanabi::prelude::*;

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
                    filter: "bevy_hanabi=warn,spawn_on_command=trace".to_string(),
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings.into(),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "🎆 Hanabi — spawn on command".to_string(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(HanabiPlugin)
        .add_plugins(WorldInspectorPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (bevy::window::close_on_esc, update))
        .run();

    Ok(())
}

#[derive(Component)]
struct Ball {
    velocity: Vec2,
}

const BOX_SIZE: f32 = 2.0;
const BALL_RADIUS: f32 = 0.05;

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = Camera3dBundle {
        tonemapping: Tonemapping::None,
        ..default()
    };
    let mut projection = OrthographicProjection::default();
    projection.scaling_mode = ScalingMode::FixedVertical(2.);
    projection.scale = 1.2;
    camera.transform.translation.z = projection.far / 2.0;
    camera.projection = Projection::Orthographic(projection);
    commands.spawn(camera);

    commands
        .spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Quad {
                size: Vec2::splat(BOX_SIZE),
                ..Default::default()
            })),
            material: materials.add(StandardMaterial {
                base_color: Color::BLACK,
                unlit: true,
                ..Default::default()
            }),
            ..Default::default()
        })
        .insert(Name::new("box"));

    commands
        .spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::UVSphere {
                sectors: 32,
                stacks: 2,
                radius: BALL_RADIUS,
            })),
            material: materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                ..Default::default()
            }),
            ..Default::default()
        })
        .insert(Ball {
            velocity: Vec2::new(1.0, 2f32.sqrt()),
        })
        .insert(Name::new("ball"));

    // Set `spawn_immediately` to false to spawn on command with Spawner::reset()
    let spawner = Spawner::once(100.0.into(), false);

    let writer = ExprWriter::new();

    // Init the age of particles to 0, and their lifetime to 1.5 second.
    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);
    let lifetime = writer.lit(1.5).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    // Add a bit of linear drag to slow down particles after the inital spawning.
    // This keeps the particle around the spawn point, making it easier to visualize
    // the different groups of particles.
    let drag = writer.lit(2.).expr();
    let update_drag = LinearDragModifier::new(drag);

    // Bind the initial particle color to the value of the 'spawn_color' property
    // when the particle spawns. The particle will keep that color afterward,
    // even if the property changes, because the color will be saved
    // per-particle (due to the Attribute::COLOR).
    let color = writer.prop("spawn_color").expr();
    let init_color = SetAttributeModifier::new(Attribute::COLOR, color);

    let normal = writer.prop("normal");

    // Set the position to be the collision point, which in this example is always
    // the emitter position (0,0,0) at the ball center, minus the ball radius
    // alongside the collision normal. Also raise particle to Z=0.2 so they appear
    // above the black background box.
    //   pos = -normal * BALL_RADIUS + Z * 0.2;
    let pos = normal.clone() * writer.lit(-BALL_RADIUS) + writer.lit(Vec3::Z * 0.2);
    let init_pos = SetAttributeModifier::new(Attribute::POSITION, pos.expr());

    // Set the velocity to be a random direction mostly along the collision normal,
    // but with some spread. This cheaply ensures that we spawn only particles
    // inside the black background box (or almost; we ignore the edge case around
    // the corners). An alternative would be to use something
    // like a KillAabbModifier, but that would spawn particles and kill them
    // immediately, wasting compute resources and GPU memory.
    //   tangent = cross(Z, normal);
    //   spread = frand() * 2. - 1.;  // in [-1:1]
    //   speed = frand() * 0.2;
    //   velocity = normalize(normal + tangent * spread * 5.) * speed;
    let tangent = writer.lit(Vec3::Z).cross(normal.clone());
    let spread = writer.rand(ScalarType::Float) * writer.lit(2.) - writer.lit(1.);
    let speed = writer.rand(ScalarType::Float) * writer.lit(0.2);
    let velocity = (normal + tangent * spread * writer.lit(5.0)).normalized() * speed;
    let init_vel = SetAttributeModifier::new(Attribute::VELOCITY, velocity.expr());

    let effect = effects.add(
        EffectAsset::new(32768, spawner, writer.finish())
            .with_name("spawn_on_command")
            .with_property("spawn_color", 0xFFFFFFFFu32.into())
            .with_property("normal", Vec3::ZERO.into())
            .init(init_pos)
            .init(init_vel)
            .init(init_age)
            .init(init_lifetime)
            .init(init_color)
            .update(update_drag)
            // Set a size of 3 (logical) pixels, constant in screen space, independent of projection
            .render(SetSizeModifier {
                size: Vec2::splat(3.).into(),
                screen_space_size: true,
            }),
    );

    commands
        .spawn((
            ParticleEffectBundle::new(effect).with_spawner(spawner),
            EffectProperties::default(),
        ))
        .insert(Name::new("effect"));
}

fn update(
    mut balls: Query<(&mut Ball, &mut Transform)>,
    mut effect: Query<(&mut EffectProperties, &mut EffectSpawner, &mut Transform), Without<Ball>>,
    time: Res<Time>,
) {
    const HALF_SIZE: f32 = BOX_SIZE / 2.0 - BALL_RADIUS;

    // Note: On first frame where the effect spawns, EffectSpawner is spawned during
    // PostUpdate, so will not be available yet. Ignore for a frame if so.
    let Ok((mut properties, mut spawner, mut effect_transform)) = effect.get_single_mut() else {
        return;
    };

    for (mut ball, mut transform) in balls.iter_mut() {
        let mut pos = transform.translation.xy() + ball.velocity * time.delta_seconds();
        let mut collision = false;

        let mut normal = Vec2::ZERO;
        for ((coord, vel_coord), normal) in pos
            .as_mut()
            .iter_mut()
            .zip(ball.velocity.as_mut())
            .zip(normal.as_mut())
        {
            while *coord < -HALF_SIZE || *coord > HALF_SIZE {
                if *coord < -HALF_SIZE {
                    *coord = 2.0 * -HALF_SIZE - *coord;
                    *normal = 1.;
                } else if *coord > HALF_SIZE {
                    *coord = 2.0 * HALF_SIZE - *coord;
                    *normal = -1.;
                }
                *vel_coord *= -1.0;
                collision = true;
            }
        }

        transform.translation = pos.extend(transform.translation.z);

        if collision {
            // This isn't the most accurate place to spawn the particle effect,
            // but this is just for demonstration, so whatever.
            effect_transform.translation = transform.translation;

            // Pick a random particle color
            let r = rand::random::<u8>();
            let g = rand::random::<u8>();
            let b = rand::random::<u8>();
            let color = 0xFF000000u32 | (b as u32) << 16 | (g as u32) << 8 | (r as u32);
            properties.set("spawn_color", color.into());

            // Set the collision normal
            let normal = normal.normalize();
            info!("Collision: n={:?}", normal);
            properties.set("normal", normal.extend(0.).into());

            // Spawn the particles
            spawner.reset();
        }
    }
}
