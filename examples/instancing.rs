#![allow(dead_code)]

use bevy::{log::LogPlugin, prelude::*, render::mesh::shape::Cube};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use rand::Rng;

use bevy_hanabi::prelude::*;

#[derive(Default, Resource)]
struct InstanceManager {
    effect: Handle<EffectAsset>,
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
            mesh: default(),
            material: default(),
            instances,
            grid_size,
            count: 0,
            frame: 0,
        }
    }

    pub fn origin(&self) -> IVec2 {
        IVec2::new(-(self.grid_size.x - 1) / 2, -(self.grid_size.y - 1) / 2)
    }

    pub fn spawn_random(&mut self, commands: &mut Commands) {
        if self.count >= self.instances.len() {
            return;
        }
        let free_count = self.instances.len() - self.count;

        let pos = self.origin();

        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..free_count);
        let (index, entry) = self
            .instances
            .iter_mut()
            .enumerate()
            .filter(|(_, entity)| entity.is_none())
            .nth(index)
            .unwrap();
        let pos = pos
            + IVec2::new(
                index as i32 % self.grid_size.x,
                index as i32 / self.grid_size.x,
            );

        *entry = Some(
            commands
                .spawn((
                    Name::new(format!("{:?}", pos)),
                    ParticleEffectBundle {
                        effect: ParticleEffect::new(self.effect.clone()),
                        transform: Transform::from_translation(Vec3::new(
                            pos.x as f32 * 10.,
                            pos.y as f32 * 10.,
                            0.,
                        )),
                        ..Default::default()
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

    pub fn despawn_index(&mut self, commands: &mut Commands, index: usize) {
        let entry = self
            .instances
            .iter_mut()
            .filter(|entity| entity.is_some())
            .nth(index)
            .unwrap();
        let entity = entry.take().unwrap();
        if let Some(entity_commands) = commands.get_entity(entity) {
            entity_commands.despawn_recursive();
        }
        self.count -= 1;
    }

    pub fn despawn_last(&mut self, commands: &mut Commands) {
        if self.count > 0 {
            self.despawn_index(commands, self.count - 1);
        }
    }

    pub fn despawn_random(&mut self, commands: &mut Commands) {
        if self.count > 0 {
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..self.count);
            self.despawn_index(commands, index);
        }
    }

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

fn main() {
    App::default()
        .add_plugins(DefaultPlugins.set(LogPlugin {
            level: bevy::log::Level::WARN,
            filter: "bevy_hanabi=warn,instancing=trace".to_string(),
        }))
        .add_system(bevy::window::close_on_esc)
        .add_plugin(HanabiPlugin)
        .add_plugin(WorldInspectorPlugin)
        .insert_resource(InstanceManager::new(5, 4))
        .add_startup_system(setup)
        .add_system(keyboard_input_system)
        //.add_system(stress_test.after(keyboard_input_system))
        .run();
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut my_effect: ResMut<InstanceManager>,
) {
    info!("Usage: Press the SPACE key to spawn more instances, and the DELETE key to remove an existing instance.");

    let mut camera = Camera3dBundle::default();
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

    let mesh = meshes.add(Mesh::from(Cube { size: 1.0 }));
    let mat = materials.add(Color::PURPLE.into());

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(0.0, 0.0, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::splat(0.0));

    let effect = effects.add(
        EffectAsset {
            name: "effect".to_string(),
            capacity: 512,
            spawner: Spawner::rate(50.0.into()),
            ..Default::default()
        }
        .init(PositionSphereModifier {
            center: Vec3::ZERO,
            radius: 1.,
            dimension: ShapeDimension::Volume,
            speed: 2.0.into(),
        })
        .init(ParticleLifetimeModifier { lifetime: 12.0 })
        .render(ColorOverLifetimeModifier { gradient }),
    );

    // Store the effect for later reference
    my_effect.effect = effect.clone();
    my_effect.mesh = mesh.clone();
    my_effect.material = mat.clone();

    // Spawn a few effects as example; others can be added/removed with keyboard
    for _ in 0..45 {
        my_effect.spawn_random(&mut commands);
    }
}

fn keyboard_input_system(
    keyboard_input: Res<Input<KeyCode>>,
    mut commands: Commands,
    mut my_effect: ResMut<InstanceManager>,
) {
    my_effect.frame += 1;

    if keyboard_input.just_pressed(KeyCode::Space) {
        my_effect.spawn_random(&mut commands);
    } else if keyboard_input.just_pressed(KeyCode::Delete) {
        my_effect.despawn_random(&mut commands);
    }

    // #123 - Hanabi 0.5.2 Causes Panic on Unwrap
    // if my_effect.frame == 5 {
    //     my_effect.despawn_index(&mut commands, 3);
    //     my_effect.despawn_index(&mut commands, 2);
    //     my_effect.spawn_random(&mut commands);
    // }
}

fn stress_test(mut commands: Commands, mut my_effect: ResMut<InstanceManager>) {
    let mut rng = rand::thread_rng();
    let r = rng.gen_range(0_f32..1_f32);
    if r < 0.45 {
        let spawn_count = (r * 10.) as i32 + 1;
        for _ in 0..spawn_count {
            my_effect.spawn_random(&mut commands);
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
