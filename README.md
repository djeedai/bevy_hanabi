# 🎆 Bevy Hanabi

[![License: MIT or Apache 2.0](https://img.shields.io/badge/License-MIT%20or%20Apache2-blue.svg)](./LICENSE) [![Doc](https://docs.rs/bevy_hanabi/badge.svg)](https://docs.rs/bevy_hanabi) [![Crate](https://img.shields.io/crates/v/bevy_hanabi.svg)](https://crates.io/crates/bevy_hanabi)
[![Bevy tracking](https://img.shields.io/badge/Bevy%20tracking-v0.6-lightblue)](https://github.com/bevyengine/bevy/blob/main/docs/plugins_guidelines.md#main-branch-tracking)

Hanabi — a particle system plugin for the Bevy game engine.

## Usage

This plugin is only compatible with Bevy v0.6.

### System setup

Add the Hanabi plugin to your app:

```rust
App::default()
    .add_plugins(DefaultPlugins)
    .add_plugin(HanabiPlugin)
    .run();
```

### Create a particle effect

Create an `EffectAsset` describing a visual effect:

```rust
fn setup(mut effects: ResMut<Assets<EffectAsset>>) {
    // Define a color gradient from red to transparent black
    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(1., 0., 0., 1.));
    gradient.add_key(1.0, Vec4::splat(0.)

    // Create the effect asset
    let effect = effects.add(EffectAsset {
            name: "MyEffect".to_string(),
            // Maximum number of particles alive at a time
            capacity: 32768,
            // Spawn at a rate of 5 particles per second
            spawner: Spawner::new(SpawnMode::rate(5.)),
            ..Default::default()
        }
        // On spawn, randomly initialize the position and velocity
        // of the particle over a sphere of radius 2 units, with a
        // radial initial velocity of 6 units/sec away from the
        // sphere center.
        .init(PositionSphereModifier {
            center: Vec3::ZERO,
            radius: 2.,
            dimension: ShapeDimension::Surface,
            speed: 6.,
        })
        // Every frame, add a gravity-like acceleration downward
        .update(AccelModifier {
            accel: Vec3::new(0., -3., 0.),
        })
        // Render the particles with a color gradient over their
        // lifetime.
        .render(ColorOverLifetimeModifier { gradient })
    );
}
```

### Add a particle effect

Use a `ParticleEffectBundle` to create an effect instance from an existing asset:

```rust
commands
    .spawn()
    .insert(Name::new("MyEffectInstance"))
    .insert_bundle(ParticleEffectBundle {
        effect: ParticleEffect::new(effect),
        transform: Transform::from_translation(Vec3::new(0., 1., 0.)),
        ..Default::default()
    });
```

## Examples

See the [`examples/`](https://github.com/djeedai/bevy_hanabi/examples) folder.

![gradient](https://raw.githubusercontent.com/djeedai/bevy_hanabi/main/examples/gradient.gif)

## Feature List

- Spawn
  - [x] Constant rate
  - [x] One-time burst
  - [x] Repeated burst
- Initialize
  - [ ] Constant position
  - [x] Position over shape
    - [ ] cube
    - [x] sphere
    - [ ] cone
    - [ ] plane
    - [ ] generic mesh / point cloud (?)
  - [ ] Random position offset
  - [x] Constant velocity
  - [ ] Random velocity
  - [ ] Constant color
  - [ ] Random color
- Update
  - [x] Motion integration
  - [x] Apply forces
    - [x] Constant (gravity)
    - [ ] Force field
  - [ ] Collision
    - [ ] Shape
      - [ ] plane
      - [ ] cube
      - [ ] sphere
    - [ ] Depth buffer
  - [x] Lifetime
  - [ ] Size change over lifetime
  - [x] Color change over lifetime
  - [ ] Face camera
  - [ ] Face constant direction
- Render
  - [x] Quad (sprite)
    - [x] Textured
  - [ ] Generic 3D mesh
  - [ ] Deformation
    - [ ] Velocity (trail)
- Debug
  - [x] GPU debug labels / groups
  - [ ] Debug visualization
    - [ ] Position magnitude
    - [ ] Velocity magnitude
    - [ ] Age / lifetime

## Compatible Bevy versions

The `main` branch is compatible with the latest Bevy release.

Compatibility of `bevy_hanabi` versions:

| `bevy_hanabi` | `bevy` |
| :--           | :--    |
| `0.1`         | `0.6`  |
