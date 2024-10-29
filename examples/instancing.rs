//! Instancing
//!
//! An example to demonstrate instancing a single effect asset multiple times.
//! The example defines a single [`EffectAsset`] then creates many
//! [`ParticleEffect`]s from that same asset, disposed in a grid pattern.
//!
//! Use the SPACE key to add more effect instances, or the DELETE key to remove
//! an existing instance.

#![allow(dead_code)]

use bevy::{core_pipeline::tonemapping::Tonemapping, prelude::*};
use bevy_hanabi::prelude::*;
use rand::Rng;

mod utils;
use utils::*;

#[derive(Default, Resource)]
struct InstanceManager {
    effect: Handle<EffectAsset>,
    alt_effect: Handle<EffectAsset>,
    texture: Handle<Image>,
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    instances: Vec<Option<Entity>>,
    grid_size: IVec2,
    count: usize,
    frame: u64,
}

impl InstanceManager {
    pub fn new(half_width: i32, half_height: i32) -> Self {
        let grid_size = IVec2::new(half_width * 2 + 1, half_height * 2 + 1);
        let count = grid_size.x as usize * grid_size.y as usize;
        let mut instances = Vec::with_capacity(count);
        instances.resize(count, None);
        Self {
            effect: default(),
            alt_effect: default(),
            texture: default(),
            mesh: default(),
            material: default(),
            instances,
            grid_size,
            count: 0,
            frame: 0,
        }
    }

    /// Get the origin of the grid in the 2D camera space. This is the offset to
    /// apply to a particle effect to transform it from the grid space to the
    /// camera space.
    pub fn origin(&self) -> IVec2 {
        IVec2::new(-(self.grid_size.x - 1) / 2, -(self.grid_size.y - 1) / 2)
    }

    /// Spawn a particle effect at the given index in the grid. The index
    /// determines both the position in the global effect array and the
    /// associated 2D grid position. If a particle effect already exists at this
    /// index / grid position, the call is ignored.
    pub fn spawn_index(&mut self, index: i32, commands: &mut Commands, alt: bool) {
        if self.count >= self.instances.len() {
            return;
        }

        let origin = self.origin();

        let entry = &mut self.instances[index as usize];
        if entry.is_some() {
            return;
        }

        let pos = origin
            + IVec2::new(
                index as i32 % self.grid_size.x,
                index as i32 / self.grid_size.x,
            );

        *entry = Some(
            commands
                .spawn((
                    Name::new(format!("{:?}", pos)),
                    ParticleEffectBundle {
                        effect: ParticleEffect::new(if alt {
                            self.alt_effect.clone()
                        } else {
                            self.effect.clone()
                        }),
                        transform: Transform::from_translation(Vec3::new(
                            pos.x as f32 * 10.,
                            pos.y as f32 * 10.,
                            0.,
                        )),
                        ..Default::default()
                    },
                    // Only used if alt_effect, but just simpler to add all the time for this
                    // example only.
                    EffectMaterial {
                        images: vec![self.texture.clone()],
                    },
                ))
                .with_children(|p| {
                    // Reference cube to visualize the emit origin
                    p.spawn((
                        PbrBundle {
                            mesh: self.mesh.clone(),
                            material: self.material.clone(),
                            ..Default::default()
                        },
                        Name::new("source"),
                    ));
                })
                .id(),
        );

        self.count += 1;
    }

    /// Spawn a particle effect at a random free position in the grid. The
    /// effect is always spawned, unless the grid is full.
    pub fn spawn_random(&mut self, commands: &mut Commands, alt: bool) {
        if self.count >= self.instances.len() {
            return;
        }
        let free_count = self.instances.len() - self.count;

        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..free_count);
        let (index, _) = self
            .instances
            .iter_mut()
            .enumerate()
            .filter(|(_, entity)| entity.is_none())
            .nth(index)
            .unwrap();
        self.spawn_index(index as i32, commands, alt);
    }

    /// Despawn the n-th existing particle effect.
    pub fn despawn_nth(&mut self, commands: &mut Commands, n: usize) {
        let entry = self
            .instances
            .iter_mut()
            .filter(|entity| entity.is_some())
            .nth(n)
            .unwrap();
        let entity = entry.take().unwrap();
        if let Some(entity_commands) = commands.get_entity(entity) {
            entity_commands.despawn_recursive();
        }
        self.count -= 1;
    }

    /// Despawn the last particle effect spawned.
    pub fn despawn_last(&mut self, commands: &mut Commands) {
        if self.count > 0 {
            self.despawn_nth(commands, self.count - 1);
        }
    }

    /// Randomly despawn one of the existing particle effects, if any.
    pub fn despawn_random(&mut self, commands: &mut Commands) {
        if self.count > 0 {
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..self.count);
            self.despawn_nth(commands, index);
        }
    }

    /// Despawn all existing particle effects.
    pub fn despawn_all(&mut self, commands: &mut Commands) {
        for entity in &mut self.instances {
            if let Some(entity) = entity.take() {
                if let Some(entity_commands) = commands.get_entity(entity) {
                    entity_commands.despawn_recursive();
                }
            }
        }
        self.count = 0;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::make_test_app("instancing")
        .insert_resource(InstanceManager::new(5, 4))
        .add_systems(Startup, setup)
        .add_systems(Update, keyboard_input_system)
        //.add_system(stress_test.after(keyboard_input_system))
        .run();
    app_exit.into_result()
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut my_effect: ResMut<InstanceManager>,
    asset_server: Res<AssetServer>,
) {
    info!("Usage: Press the SPACE key to spawn more instances, and the DELETE key to remove an existing instance.");

    let mut camera = Camera3dBundle {
        tonemapping: Tonemapping::None,
        ..default()
    };
    camera.transform.translation = Vec3::new(0.0, 0.0, 180.0);
    commands.spawn(camera);

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::WHITE,
            // Crank the illuminance way (too) high to make the reference cube clearly visible
            illuminance: 100000.,
            shadows_enabled: false,
            ..Default::default()
        },
        ..Default::default()
    });

    let mesh = meshes.add(Cuboid {
        half_size: Vec3::splat(0.5),
    });
    let mat = materials.add(utils::COLOR_PURPLE);

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(0.0, 0.0, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::splat(0.0));

    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(12.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let init_pos = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(1.).expr(),
        dimension: ShapeDimension::Volume,
    };

    let init_vel = SetVelocitySphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        speed: writer.lit(2.).expr(),
    };

    let effect = effects.add(
        EffectAsset::new(512, Spawner::rate(50.0.into()), writer.finish())
            .with_name("instancing")
            .init(init_pos)
            .init(init_vel)
            .init(init_age)
            .init(init_lifetime)
            .render(ColorOverLifetimeModifier { gradient }),
    );

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(1., 0., 0., 0.));
    gradient.add_key(0.1, Vec4::new(1., 0., 0., 1.));
    gradient.add_key(1.0, Vec4::new(1., 0., 0., 0.));

    let writer = ExprWriter::new();

    let lifetime = writer.lit(5.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let init_pos = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(7.).expr(),
        dimension: ShapeDimension::Volume,
    };

    let init_vel = SetVelocityTangentModifier {
        origin: writer.lit(Vec3::ZERO).expr(),
        axis: writer.lit(Vec3::Z).expr(),
        speed: writer.lit(4.).expr(),
    };

    let radial_accel =
        RadialAccelModifier::new(writer.lit(Vec3::ZERO).expr(), writer.lit(-3).expr());

    let texture_slot = writer.lit(0u32).expr();

    let mut module = writer.finish();
    module.add_texture("color");

    let alt_effect = effects.add(
        EffectAsset::new(512, Spawner::rate(102.0.into()), module)
            .with_simulation_space(SimulationSpace::Local)
            .with_name("alternate instancing")
            .init(init_pos)
            .init(init_vel)
            .init(init_lifetime)
            .update(radial_accel)
            .render(ParticleTextureModifier {
                texture_slot,
                sample_mapping: ImageSampleMapping::Modulate,
            })
            .render(ColorOverLifetimeModifier { gradient }),
    );

    // Store the effects for later reference
    my_effect.effect = effect;
    my_effect.alt_effect = alt_effect;
    my_effect.texture = asset_server.load("circle.png");
    my_effect.mesh = mesh;
    my_effect.material = mat;

    // Spawn a few effects as example; others can be added/removed with keyboard
    for i in 0..45 {
        my_effect.spawn_random(&mut commands, (i % 15) == 14);
    }
}

fn keyboard_input_system(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut my_effect: ResMut<InstanceManager>,
) {
    my_effect.frame += 1;

    if keyboard_input.just_pressed(KeyCode::Space) {
        my_effect.spawn_random(&mut commands, keyboard_input.pressed(KeyCode::ShiftLeft));
    } else if keyboard_input.just_pressed(KeyCode::Delete)
        || keyboard_input.just_pressed(KeyCode::Backspace)
    {
        my_effect.despawn_random(&mut commands);
    }

    // #123 - Hanabi 0.5.2 Causes Panic on Unwrap
    // if my_effect.frame == 5 {
    //     my_effect.despawn_nth(&mut commands, 3);
    //     my_effect.despawn_nth(&mut commands, 2);
    //     my_effect.spawn_random(&mut commands);
    // }
}

fn stress_test(mut commands: Commands, mut my_effect: ResMut<InstanceManager>) {
    let mut rng = rand::thread_rng();
    let r = rng.gen_range(0_f32..1_f32);
    if r < 0.45 {
        let spawn_count = (r * 10.) as i32 + 1;
        for _ in 0..spawn_count {
            my_effect.spawn_random(&mut commands, false);
        }
    } else if r < 0.9 {
        let despawn_count = ((r - 0.45) * 10.) as i32 + 1;
        for _ in 0..despawn_count {
            my_effect.despawn_random(&mut commands);
        }
    } else if r < 0.95 {
        my_effect.despawn_all(&mut commands);
    }
}
