use bevy::{log::LogPlugin, prelude::*, render::mesh::shape::Cube};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

use bevy_hanabi::prelude::*;

#[derive(Default, Resource)]
struct MyEffect {
    effect: Handle<EffectAsset>,
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    next_pos: IVec2,
    instances: Vec<Entity>,
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
        .init_resource::<MyEffect>()
        .add_startup_system(setup)
        .add_system(keyboard_input_system)
        .run();
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut my_effect: ResMut<MyEffect>,
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
    my_effect.next_pos = IVec2::new(-5, -4);
    my_effect.mesh = mesh.clone();
    my_effect.material = mat.clone();

    // Spawn a few effects as example; others can be added/removed with keyboard
    for _ in 0..45 {
        let (id, next_pos) = spawn_instance(
            &mut commands,
            my_effect.next_pos,
            my_effect.effect.clone(),
            my_effect.mesh.clone(),
            my_effect.material.clone(),
        );
        my_effect.instances.push(id);
        my_effect.next_pos = next_pos;
    }
}

fn spawn_instance(
    commands: &mut Commands,
    pos: IVec2,
    effect: Handle<EffectAsset>,
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
) -> (Entity, IVec2) {
    let mut next_pos = pos;
    next_pos.x += 1;
    if next_pos.x > 5 {
        next_pos.x = -5;
        next_pos.y += 1;
    }

    let id = commands
        .spawn((
            Name::new(format!("{:?}", pos)),
            ParticleEffectBundle {
                effect: ParticleEffect::new(effect),
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
                    mesh,
                    material,
                    ..Default::default()
                },
                Name::new("source"),
            ));
        })
        .id();

    (id, next_pos)
}

fn keyboard_input_system(
    keyboard_input: Res<Input<KeyCode>>,
    mut commands: Commands,
    mut my_effect: ResMut<MyEffect>,
) {
    if keyboard_input.just_pressed(KeyCode::Space) {
        // Spawn a new instance
        let (id, next_pos) = spawn_instance(
            &mut commands,
            my_effect.next_pos,
            my_effect.effect.clone(),
            my_effect.mesh.clone(),
            my_effect.material.clone(),
        );
        my_effect.instances.push(id);
        my_effect.next_pos = next_pos;
    } else if keyboard_input.just_pressed(KeyCode::Delete) {
        // Delete an existing instance
        if let Some(entity) = my_effect.instances.pop() {
            if let Some(entity_commands) = commands.get_entity(entity) {
                entity_commands.despawn_recursive();
            }
            my_effect.next_pos.x -= 1;
            if my_effect.next_pos.x < -5 {
                my_effect.next_pos.x = 5;
                my_effect.next_pos.y -= 1;
            }
        }
    }
}
