# 🎆 Bevy Hanabi

[![License: MIT or Apache 2.0](https://img.shields.io/badge/License-MIT%20or%20Apache2-blue.svg)](./LICENSE)
[![Doc](https://docs.rs/bevy_hanabi/badge.svg)](https://docs.rs/bevy_hanabi)
[![Crate](https://img.shields.io/crates/v/bevy_hanabi.svg)](https://crates.io/crates/bevy_hanabi)
[![Build Status](https://github.com/djeedai/bevy_hanabi/actions/workflows/ci.yaml/badge.svg)](https://github.com/djeedai/bevy_hanabi/actions/workflows/ci.yaml)
[![Coverage Status](https://coveralls.io/repos/github/djeedai/bevy_hanabi/badge.svg?branch=main)](https://coveralls.io/github/djeedai/bevy_hanabi?branch=main)
[![Bevy tracking](https://img.shields.io/badge/Bevy%20tracking-v0.10-lightblue)](https://github.com/bevyengine/bevy/blob/main/docs/plugins_guidelines.md#main-branch-tracking)

🎆 Hanabi — a GPU particle system for the Bevy game engine.

## Overview

The Hanabi particle system is a modern GPU-based particle system for the Bevy game engine. It focuses on scale to produce stunning visual effects (VFX) in real time, offloading most of the work to the GPU, with minimal CPU intervention. The design is inspired by modern particle systems found in other industry-leading game engines.

🚧 _This project is under heavy development, and is currently lacking both features and performance / usability polish. However, for moderate-size effects, it can already be used in your project. Feedback and contributions on both design and features are very much welcome._

🎆 Hanabi makes heavy use of compute shaders to offload work to the GPU in a performant way, and therefore is not available for the `wasm` target (WebAssembly). This is a limitation of how Bevy itself uses `wgpu` as of latest stable release (0.10). The next Bevy release (0.11) will have WebGPU support, making it possible to add WebAssembly support to 🎆 Hanabi. See [#41](https://github.com/djeedai/bevy_hanabi/issues/41) for details.

## Usage

The 🎆 Bevy Hanabi plugin is compatible with Bevy versions >= 0.6; see [Compatible Bevy versions](#compatible-bevy-versions).

### Add the dependency

Add the `bevy_hanabi` dependency to `Cargo.toml`:

```toml
[dependencies]
bevy_hanabi = "0.6"
```

See also [Features](#features) below for the list of supported features.

### System setup

Add the `HanabiPlugin` to your app:

```rust
use bevy_hanabi::prelude::*;

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
    gradient.add_key(0.0, Vec4::new(1., 0., 0., 1.)); // Red
    gradient.add_key(1.0, Vec4::ZERO); // Transparent black

    // Create the effect asset
    let effect = effects.add(EffectAsset {
            name: "MyEffect".to_string(),
            // Maximum number of particles alive at a time
            capacity: 32768,
            // Spawn at a rate of 5 particles per second
            spawner: Spawner::rate(5.0.into()),
            ..Default::default()
        }
        // On spawn, randomly initialize the position of the particle
        // to be over the surface of a sphere of radius 2 units.
        .init(InitPositionSphereModifier {
            center: Vec3::ZERO,
            radius: 2.,
            dimension: ShapeDimension::Surface,
        })
        // Also initialize a radial initial velocity to 6 units/sec
        // away from the (same) sphere center.
        .init(InitVelocitySphereModifier {
            center: Vec3::ZERO,
            speed: 6.0.into(),
        })
        // Also initialize the total lifetime of the particle, that is
        // the time for which it's simulated and rendered. This modifier
        // is mandatory, otherwise the particles won't show up.
        .init(InitLifetimeModifier { lifetime: 10_f32.into() })
        // Every frame, add a gravity-like acceleration downward
        .update(AccelModifier::constant(Vec3::new(0., -3., 0.)))
        // Render the particles with a color gradient over their
        // lifetime. This maps the gradient key 0 to the particle spawn
        // time, and the gradient key 1 to the particle death (here, 10s).
        .render(ColorOverLifetimeModifier { gradient })
    );
}
```

### Add a particle effect

Use a `ParticleEffectBundle` to create an effect instance from an existing asset:

```rust
commands
    .spawn(ParticleEffectBundle {
        effect: ParticleEffect::new(effect),
        transform: Transform::from_translation(Vec3::new(0., 1., 0.)),
        ..Default::default()
    });
```

## Examples

See the [`examples/`](https://github.com/djeedai/bevy_hanabi/tree/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples) folder.

Note for Linux users: The examples build with the `bevy/x11` feature by default to enable support for the X11 display server. If you want to use the Wayland display server instead, add the `bevy/wayland` feature.

### Firework

Combine the `InitPositionSphereModifier` for spawning and `LinearDragModifier` to slow down particles, to create a firework effect. This example makes use of an HDR camera with Bloom. See the example file for more details about how the effect is designed.

```shell
cargo run --example firework --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
```

![firework](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/firework.gif)

### Portal

Combine the `InitVelocityTangentModifier` for tangential rotation of particles around a circle and the `OrientAlongVelocityModifier` to create elongated sparks, to produce a kind of "magic portal" effect. This example makes use of an HDR camera with Bloom. See the example file for more details about how the effect is designed.

```shell
cargo run --example portal --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
```

![portal](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/portal.gif)

### Gradient

Animate an emitter by moving its `Transform` component, and emit textured quad particles with a `ColorOverLifetimeModifier`.

```shell
cargo run --example gradient --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
```

![gradient](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/gradient.gif)

### Force Field

This example demonstrates the force field modifier `ForceFieldModifier`, which allows creating some attraction and repulsion sources affecting the motion of the particles. It also demonstrates the use of the `AabbKillModifier` to either kill the particles exiting an "allowed" space (green box) or entering a "forbidden" space (red box).

```shell
cargo run --example force_field --features="bevy/bevy_winit bevy/bevy_pbr 3d"
```

![force_field](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/force_field.gif)

### 2D

This example shows how to use 🎆 Hanabi with a 2D camera.

The white square mesh is moving forward and backward along the camera depth. The 2D effect itself remains at a constant position. When the square mesh moves behind the effect, the particles are rendered in front of it, and conversely when it moves forward the particles are rendered behind it.

```shell
cargo run --example 2d --features="bevy/bevy_winit bevy/bevy_sprite 2d"
```

![2d](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/2d.gif)

### Multi-camera

The example demonstrates the use of multiple cameras and render layers to selectively render effects. Each camera uses a different combination of layers, and each effect is assigned a different layer.

```shell
cargo run --example multicam --features="bevy/bevy_winit bevy/bevy_pbr 3d"
```

![multicam](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/multicam.gif)

### Activate

This example demonstrates manual activation and deactivation of a spawner, from code (CPU). The circle bobs up and down in the water, spawning square bubbles when in the water only.

```shell
cargo run --example activate --features="bevy/bevy_winit bevy/bevy_pbr 3d"
```

![activate](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/activate.gif)

### Spawn

This example demonstrates three spawn modes:

- **Left:** Continuous emission with a fixed rate (particles/second).
- **Center:** One-shot burst emission of a fixed count of particles.
- **Right:** Continuous bursts of particles, an hybrid between the previous two. This effect also uses a property to change over time the direction of the acceleration applied to all particles.

It also shows the applying of constant acceleration to all particles. The right spawner's acceleration (gravity) is controlled by a custom property, which is slowly rotated by a Bevy system (CPU side).

```shell
cargo run --example spawn --features="bevy/bevy_winit bevy/bevy_pbr 3d"
```

![spawn](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/spawn.gif)

### Spawn on command

This example demonstrates how to emit a burst of particles when an event occurs. A property is also used to modify the color of the particles spawned. This gives total control of the spawning to the user code.

```shell
cargo run --example spawn_on_command --features="bevy/bevy_winit bevy/bevy_pbr 3d"
```

![spawn_on_command](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/spawn_on_command.gif)

### Circle

This example demonstrates the `circle` spawner type, which emits particles along a circle perimeter or a disk surface. This allows for example simulating a dust ring around an object colliding with the ground.

```shell
cargo run --example circle --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
```

![circle](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/circle.gif)

### Visibility

This example demonstrates the difference between the default `SimulationCondition::WhenVisible` which simulates an effect when it's visible only, and `SimulationCondition::Always` which always simulates an effect even if the entity is hidden.

- The **top** effect uses `SimulationCondition::Always`, continuing to simulate even when hidden, moving to the right.
- The **bottom** effect uses `SimulationCondition::WhenVisible`, pausing simulation while hidden, and resuming its motion once visible again from the position where it was last visible.

```shell
cargo run --example visibility --features="bevy/bevy_winit bevy/bevy_pbr 3d"
```

![circle](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/visibility.gif)

### Random

This example spawns particles with randomized parameters.

```shell
cargo run --example random --features="bevy/bevy_winit bevy/bevy_pbr 3d"
```

![random](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/random.gif)

### Lifetime

This example demonstrates particle effects with different lifetimes. Each effect emits particles every 3 seconds, with a particle lifetime of:

- **Left:** 12 seconds, longer than the emit rate, so multiple bursts accumulate.
- **Center:** 3 seconds, like the emit rate, so particles die when new ones are emitted.
- **Right:** 0.75 second, shorter than the emit rate, so particles die much earlier than the next burst.

```shell
cargo run --example lifetime --features="bevy/bevy_winit bevy/bevy_pbr 3d"
```

![lifetime](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/lifetime.gif)

### Billboard

This example demonstrates particles with the billboard render modifier, making them always face the camera.

```shell
cargo run --example billboard --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
```

The image on the left has the `BillboardModifier` enabled.
![billboard](https://raw.githubusercontent.com/djeedai/bevy_hanabi/3ad8f9e34daf3db3e2d821e2f9bac3023cdf0af4/examples/billboard.png)

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
    - [x] cone / truncated cone (3D)
    - [ ] plane
    - [ ] generic mesh / point cloud (?)
  - [ ] Random position offset
  - [x] Velocity over shape (with random speed)
    - [x] circle
    - [x] sphere
    - [x] tangent
  - [x] Constant/random per-particle color
  - [x] Constant/random per-particle size
  - [x] Constant/random par-particle age and lifetime
- Update
  - [x] Simulation condition
    - [x] Always, even when hidden
    - [x] Only when visible
  - [x] Motion integration (Euler)
  - [x] Apply forces and accelerations
    - [x] Constant acceleration (gravity)
    - [x] Radial acceleration
    - [x] Tangent acceleration
    - [x] Force field
    - [x] Linear drag
  - [ ] Collision
    - [ ] Shape
      - [ ] plane
      - [ ] cube
      - [ ] sphere
    - [ ] Depth buffer
  - [x] Allow/deny despawn box
  - [x] Lifetime
  - [x] Size change over lifetime
  - [x] Color change over lifetime
- Render
  - [x] Quad
    - [x] Textured
  - [ ] Generic 3D mesh
  - [ ] Deformation
    - [ ] Velocity (trail)
  - [x] Camera support
    - [x] Render layers
    - [x] 2D cameras ([`Camera2dBundle`](https://docs.rs/bevy/0.10.0/bevy/core_pipeline/core_2d/struct.Camera2dBundle.html)) only
    - [x] 3D cameras ([`Camera3dBundle`](https://docs.rs/bevy/0.10.0/bevy/core_pipeline/core_3d/struct.Camera3dBundle.html)) only
    - [x] Simultaneous dual 2D/3D cameras
    - [x] Multiple viewports (split screen)
    - [x] HDR camera and bloom
  - [ ] Orient particles
    - [x] Face camera (Billboard)
    - [ ] Face constant direction
    - [x] Orient alongside velocity
- Debug
  - [x] GPU debug labels / groups
  - [ ] Debug visualization
    - [ ] Position magnitude
    - [ ] Velocity magnitude
    - [ ] Age / lifetime

## Features

🎆 Bevy Hanabi supports the following cargo features:

| Feature | Default | Description |
|---|:-:|---|
| `2d` | ✔ | Enable rendering through 2D cameras ([`Camera2dBundle`](https://docs.rs/bevy/0.10.0/bevy/core_pipeline/core_2d/struct.Camera2dBundle.html)) |
| `3d` | ✔ | Enable rendering through 3D cameras ([`Camera3dBundle`](https://docs.rs/bevy/0.10.0/bevy/core_pipeline/core_3d/struct.Camera3dBundle.html)) |

For optimization purpose, users of a single type of camera can disable the other type by skipping default features in their `Cargo.toml`. For example to use only the 3D mode:

```toml
bevy_hanabi = { version = "0.6", default-features = false, features = [ "3d" ] }
```

## Compatible Bevy versions

The `main` branch is compatible with the latest Bevy release.

Compatibility of `bevy_hanabi` versions:

| `bevy_hanabi` | `bevy` |
| :--           | :--    |
| `0.6`         | `0.10` |
| `0.5`         | `0.9`  |
| `0.3`-`0.4`   | `0.8`  |
| `0.2`         | `0.7`  |
| `0.1`         | `0.6`  |
