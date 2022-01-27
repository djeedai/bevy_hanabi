# Bevy Hanabi

[![License: MIT or Apache 2.0](https://img.shields.io/badge/License-MIT%20or%20Apache2-yellow.svg)](./LICENSE) [![Doc](https://docs.rs/bevy_hanabi/badge.svg)](https://docs.rs/bevy_hanabi) [![Crate](https://img.shields.io/crates/v/bevy_hanabi.svg)](https://crates.io/crates/bevy_hanabi)

Hanabi — a particle system plugin for the Bevy game engine.

## Usage

This plugin is only compatible with the `main` branch of Bevy (post-`0.5` version, with new renderer).

### System setup

Add the Hanabi plugin to your app:

```rust
App::default()
    .add_default_plugins()
    .add_plugin(HanabiPlugin)
    .run();
```

### Add a particles effect

TODO; this library is under development...

## Examples

See the [`examples/`](https://github.com/djeedai/bevy_extra/tree/main/bevy_hanabi/examples) folder.

## Feature List

- Spawn
  - [x] Constant rate
  - [ ] Burst
  - [ ] Repeated burst
- Initialize
  - [ ] Constant position
  - [ ] Position over shape
    - [ ] cube
    - [ ] sphere
    - [ ] cone
    - [ ] plane
    - [ ] generic mesh / point cloud (?)
  - [ ] Random position offset
  - [ ] Constant velocity
  - [ ] Random velocity
  - [ ] Constant color
  - [ ] Random color
- Update
  - [ ] Verlet motion integration
  - [ ] Apply forces
    - [ ] Constant
    - [ ] Bounded (AABB, etc.)
  - [ ] Lifetime
  - [ ] Size change over lifetime
  - [ ] Color change over lifetime
  - [ ] Face camera
  - [ ] Face constant direction
- Render
  - [ ] Quad (sprite)
  - [ ] Generic 3D mesh