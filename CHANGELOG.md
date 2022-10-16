# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Respect user-defined MSAA setting by reading the value of `Msaa::samples` when building the render pipeline. (#59)
- Fixed a bug in the effect cache causing a panic sometimes when effects are removed. (#60)

## [0.4.0] 2022-10-11

### Added

- Added `PositionCone3dModifier` to spawn particles inside a truncated 3D cone.
- All GPU profiling markers are now prefixed with `hanabi:` to make it easier to find Hanabi-related GPU resources.

### Changed

- Moved all modifiers into a top-level `modifier` module, and further into some `init`, `update`, and `render` sub-modules.
- Added a `bevy_hanabi::prelude` containing most public types, to be used preferably over `use bevy_hanabi::*`.
- Renamed `ForceFieldParam` to `ForceFieldSource`.
- Renamed the `FFNUM` constant to `ForceFieldSource::MAX_SOURCES`.
- Renamed `ForceFieldModifier::force_field` to `ForceFieldModifier::sources`.

### Fixed

- The orientation of the `Entity` of the `ParticleEffect` is now taken into account for spawning. (#42)
- Ensure all GPU resources are deallocated when a `ParticleEffect` component is despawned. (#45)

### Removed

- `EffectCacheId` is now private. It was exposed publicly by error, and cannot be used for anything in the public API anyway.

## [0.3.1] 2022-08-19

### Added

- Added `EffectAsset::z_layer_2d` and `ParticleEffect::z_layer_2d` to control the Z layer at which particles are rendered in 2D mode. Note that effects with different Z values cannot be batched together, which may negatively affect performance.
- Added `BillboardModifier` to force the particles to face the camera.

## [0.3.0] 2022-08-06

### Changed

- Switch to Bevy v0.8.
- Update spawners in a separate system `tick_spawners()` (label: `EffectSystems::TickSpawners`) which runs in the `CoreStage::PostUpdate` stage after the visibility system updated all `ComputedVisibility`, to allow skipping effect instances which are not visible. Spawners were previously ticked in the render extract phase.

## [0.2.0] 2022-04-17

### Added

- Added test-only feature `gpu_tests` active by default to enable tests requiring a working graphic adapter (GPU). This is disabled in most CI tests, except on Linux where the CPU-based Vulkan emulator `lavapipe` is used.

### Changed

- Switch to Bevy v0.7.
- Changed features `2d` and `3d` to be purely additive. They are now both active by default, allowing to render through both 2D and 3D cameras at the same time. Users can optionally select either of those exclusively via the `--no-default-features --features='2d'` options (or similar for 3D), as an optimization for applications using only one of the two codepaths.
- Tighter set of dependencies, removing the general `bevy/render` and instead depending on `bevy/bevy_core_pipeline` and `bevy/bevy_render` only.

### Fixed

- Fix missing `derive` feature in `bytemuck` dependency occasionally causing build errors.
- Fix a bug in spawner parameters alignment making the library crash on some GPUs. The spawner parameters are now properly aligned according to the device-dependent constraints queried at runtime. (#26)

## [0.1.2] 2022-04-07

### Added

- Add `SizeOverLifetimeModifier`.
- Add `PositionCircleModifier` to allow spawning from a circle or disc.
- Revamped spawning system:
  - `SpawnMode` is gone; `Spawner`s are constructed with associated functions `new`, `once`, `rate`, and `burst`.
  - Spawners can be reset with `Spawner::reset`. This gives control over when to spawn a burst of particles.
  - Spawners can be activated or deactivated with `Spawner::set_active`.
  - `ParticleEffectBundle`s can be initialized with a spawner with `ParticleEffectBundle::with_spawner`.
- Implemented `ToWgslFloat` for `Vec2` / `Vec3` / `Vec4`.
- Implemented `ToWgslFloat` for `Value<f32>`.
- Derive-implemented `PartialEq` for `Value<T>` and `Spawner`.
- Implemented randomization for randomized spawning parameters
- New force field effect:
  - Add `ForceFieldModifier` to allow attraction or repulsion from point sources.
  - Add `ForceFieldParam` in both the modifiers and the particle update shader.
  - Add `force_field` example showcasing a repulsor, an attractor and the conforming to sphere functionality.
- Add rendering with a 2D camera.

### Changed

- Renamed the `ToWgslFloat` trait into `ToWgslString`, and its `to_float_string()` method into `to_wgsl_string()`. Also made the trait public.
- Position modifiers now use `Value<f32>` for velocity to allow for random velocity.
- Either the "3d" feature or the "2d" feature must be enabled.

### Fixed

- Fixed depth sorting of particles relative to opaque objects. Particles are now correctly hidden when behind opaque objects.
- Fixed truncation in compute workgroup count preventing update of some particles, and in degenerate cases (`capacity < 64`) completely disabling update.
- Made the `GradientKey<T>::ratio` field private to avoid any modification via `Gradient<T>::keys_mut()` which would corrupt the internal sorting of keys.
- Fixed a bug where adding the `HanabiPlugin` to an app without spawning any `ParticleEffect` would crash at  runtime. (#9).

## [0.1.1] 2022-02-15

### Fixed

- Fix homepage link in `Cargo.toml`
- Bevy 0.6.1 fixed build on nightly, thereby fixing docs.rs builds

## [0.1.0] 2022-02-11

Initial alpha version. Lots of things missing, but the barebone functionality is there.
See the README.md for the list of planned and implemented features.
