# 🎆 Bevy Hanabi

[![License: MIT or Apache 2.0](https://img.shields.io/badge/License-MIT%20or%20Apache2-blue.svg)](./LICENSE)
[![Doc](https://docs.rs/bevy_hanabi/badge.svg)](https://docs.rs/bevy_hanabi)
[![Crate](https://img.shields.io/crates/v/bevy_hanabi.svg)](https://crates.io/crates/bevy_hanabi)
[![Build Status](https://github.com/djeedai/bevy_hanabi/actions/workflows/ci.yaml/badge.svg)](https://github.com/djeedai/bevy_hanabi/actions/workflows/ci.yaml)
[![Coverage Status](https://coveralls.io/repos/github/djeedai/bevy_hanabi/badge.svg?branch=main)](https://coveralls.io/github/djeedai/bevy_hanabi?branch=main)
[![Bevy tracking](https://img.shields.io/badge/Bevy%20tracking-v0.6-lightblue)](https://github.com/bevyengine/bevy/blob/main/docs/plugins_guidelines.md#main-branch-tracking)

Hanabi — a GPU particle system for the Bevy game engine.

## Overview

The Hanabi particle system is a modern GPU-based particle system for the Bevy game engine. It focuses on scale to produce stunning visual effects (VFX) in real time, offloading most of the work to the GPU, with minimal CPU intervention. The design is inspired by modern particle systems found in other industry-leading game engines.

🚧 _This project is under heavy development, and is currently lacking both features and performance / usability polish. However, for moderate-size effects, it can already be used in your project. Feedback and contributions on both design and features are very much welcome._

## Usage

The 🎆 Bevy Hanabi plugin is only compatible with Bevy v0.6.

### System setup

Add the `HanabiPlugin` to your app:

```rust
use bevy_hanabi::*;

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
            spawner: Spawner::rate(5.0.into()),
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
            speed: 6.0.into(),
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

### Gradient

Animate an emitter by moving its `Transform` component, and emit textured quad particles with a `ColorOverLifetimeModifier`.

```shell
cargo run --example gradient --features="bevy/bevy_winit bevy/png"
```

![gradient](https://raw.githubusercontent.com/djeedai/bevy_hanabi/471669f735f202d3877969e25c488e5d74fc3393/examples/gradient.gif)

### Force Field

This example demonstrates the force field modifier `ForceFieldModifier`, which allows creating some attraction and repulsion sources affecting the motion of the particles.

```shell
cargo run --example force_field --features="bevy/bevy_winit"
```

![force_field](https://raw.githubusercontent.com/djeedai/bevy_hanabi/471669f735f202d3877969e25c488e5d74fc3393/examples/force_field.gif)

### Activate

This example demonstrates manual activation and deactivation of a spawner, from code (CPU). The circle bobs up and down in the water, spawning square bubbles when in the water only.

```shell
cargo run --example activate --features="bevy/bevy_winit"
```

![activate](https://raw.githubusercontent.com/djeedai/bevy_hanabi/471669f735f202d3877969e25c488e5d74fc3393/examples/activate.gif)

### Spawn

This example demonstrates three spawn modes:

- **Left:** Continuous emission with a fixed rate (particles/second).
- **Center:** One-shot burst emission of a fixed count of particles.
- **Right:** Continuous bursts of particles, an hybrid between the previous two.

It also shows the applying of constant force (downward gravity-like, or upward smoke-style).

```shell
cargo run --example spawn --features="bevy/bevy_winit"
```

![spawn](https://raw.githubusercontent.com/djeedai/bevy_hanabi/471669f735f202d3877969e25c488e5d74fc3393/examples/spawn.gif)

### Spawn on Command

This example demonstrates how to emit a burst of particles when an event occurs. This gives total control of the spawning to the user code.

```shell
cargo run --example spawn_on_command --features="bevy/bevy_winit"
```

![spawn](https://raw.githubusercontent.com/djeedai/bevy_hanabi/471669f735f202d3877969e25c488e5d74fc3393/examples/spawn_on_command.gif)

### Circle

This example demonstrates the `circle` spawner type, which emits particles along a circle perimeter or a disk surface. This allows for example simulating a dust ring around an object colliding with the ground.

```shell
cargo run --example circle --features="bevy/bevy_winit bevy/png"
```

![circle](https://raw.githubusercontent.com/djeedai/bevy_hanabi/471669f735f202d3877969e25c488e5d74fc3393/examples/circle.gif)

### Random

This example spawns particles with randomized parameters.

```shell
cargo run --example random --features="bevy/bevy_winit"
```

![spawn](https://raw.githubusercontent.com/djeedai/bevy_hanabi/471669f735f202d3877969e25c488e5d74fc3393/examples/random.gif)

## Feature List

- Spawn
  - [x] Constant rate
  - [x] One-time burst
  - [x] Repeated burst
  - [x] Spawner resetting
  - [x] Spawner activation/deactivation
  - [x] Randomized spawning parameters
- Initialize
  - [ ] Constant position
  - [x] Position over shape
    - [ ] cube
    - [x] circle
    - [x] sphere
    - [ ] cone
    - [ ] plane
    - [ ] generic mesh / point cloud (?)
  - [ ] Random position offset
  - [x] Constant velocity
  - [x] Random velocity
  - [ ] Constant color
  - [ ] Random color
- Update
  - [x] Motion integration
  - [x] Apply forces
    - [x] Constant (gravity)
    - [x] Force field
  - [ ] Collision
    - [ ] Shape
      - [ ] plane
      - [ ] cube
      - [ ] sphere
    - [ ] Depth buffer
  - [x] Lifetime
  - [x] Size change over lifetime
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
