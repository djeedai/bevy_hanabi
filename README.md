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

### Add a particles effect

🚧 TODO; this library is under development... 🚧

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
  - [x] Motion integration
  - [ ] Apply forces
    - [ ] Constant
    - [ ] Bounded (AABB, etc.)
  - [x] Lifetime
  - [ ] Size change over lifetime
  - [x] Color change over lifetime
  - [ ] Face camera
  - [ ] Face constant direction
- Render
  - [x] Quad (sprite)
    - [x] Textured
  - [ ] Generic 3D mesh

## Compatible Bevy versions

The `main` branch is compatible with the latest Bevy release.

Compatibility of `bevy_hanabi` versions:

| `bevy_hanabi` | `bevy` |
| :--           | :--    |
| `0.1`         | `0.6`  |
