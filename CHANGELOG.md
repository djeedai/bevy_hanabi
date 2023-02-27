# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a `SimulationSpace` enum with a single value `Global`, in preparation of future support for local-space particle simulation.
- Added `InitAgeModifier` to initialize the age of particles to a specific (possibly random) value instead of the default `0.0`.
- Added _properties_, named quantities associated with an `EffectAsset` and modifiable at runtime and assignable to values of modifiers. This enables dynamically modifying at runtime any property-based modifier value. Other non-property-based values are assumed to be constant, and are optimized by hard-coding them into the various shaders. Use `EffectAsset::with_property()` and `EffectAsset::add_property()` to define a new property before binding it to a modifier's value. The `spawn.rs` example demonstrates this with the acceleration of the `AccelModifier`.
- Added particle `Attribute`s, individual items holding a single per-particle scalar or vector value. Composed together, the attributes form the particle layout, which defines at runtime the set of per-particle values used in the simulation. This change marks a transition from the previous design where the particle attributes were hard-coded to be the position, velocity, age, and lifetime of the particle. With this change, a collection of various attributes is available for modifiers to manipulate. This ensures flexibility in customizing while retaining a minimal per-particle memory footprint for a given effect. Note that [`Attribute`]s are all built-in; a custom [`Attribute`] cannot be used.
- Added a new `AabbKillModifier` which kills all particles entering or exiting an AABB. The `force_field.rs` example shows an example of each variant.
- Added a new `OrientAlongVelocityModifier` which orients the local X axis of particles alongside their velocity, and builds a local Z axis perpendicular to the camera view's plane. The local Y axis is derived to form an orthonormal frame.

### Changed

- The `ParticleLifetimeModifier` was renamed to `InitLifetimeModifer` for clarity, and its `lifetime` field is now a `Value<f32>` to allow randomizing per particle.
- Effects no longer have a default particle lifetime of 5 seconds. Instead an explicit lifetime must be set with the `InitLifetimeModifer`. Failure to set the lifetime will trigger a warning at runtime, and particles will default to a lifetime of zero and instantly die.
- The `PositionSphereModifier` was previously initializing both the position and velocity of particles. It has been split into an `InitPositionSphereModifier` to initialize the position, and `InitVelocitySphereModifier` to initialize the velocity.
- The `PositionCircleModifier` was previously initializing both the position and velocity of particles. It has been split into an `InitPositionCircleModifier` to initialize the position, and `InitVelocityCircleModifier` to initialize the velocity.
- Effects no longer have a default initial position similar to the ex-`PositionSphereModifier`. Add an `InitPositionSphereModifier` and an `InitVelocitySphereModifier` explicitly to an effect to restore the previous behavior, with a center of `Vec3::ZERO`, a radius of `1.`, a dimension of `ShapeDimension::Volume`, and a speed of `2.`.
- The acceleration of the `AccelModifier` is now property-based. To dynamically change the acceleration at runtime, assign its value to an effect property with `AccelModifier::via_property()`. Otherwise a constant value built with `AccelModifier::constant()` can be used, and will be optimized in the update shader. See the `spawn.rs` example.
- All modifiers are now fully reflected (derive both `Reflect` and `FromReflect`) and de/serializable. They're serialized as enums, using the `typetag` crate.
- `ShapeDimension` now derives `Debug` and is fully reflected (derives both `Reflect` and `FromReflect`) and de/serializable.
- `Value<T>` now requires `T: FromReflect`, and itself derives both `Reflect` and `FromReflect`.
- Consequence of `Value<T>` being fully reflected, several fields on `Spawner` are now fully reflected too and not ignored anymore.
- The conforming to sphere feature of `ForceFieldModifier` is now applied before the Euler integration updating the particle position. This may result is tiny deviations from the previous behavior, as the particle position will not strictly conform to the sphere at the end of the step. However the delta should be very small, and no visible difference is expected in practice.
- Changed the `instancing` example to allow removing particles with the BACKSPACE key in addition of the DELETE one, mainly for useability on macOS.
- Changed the `lifetime` example to render particles with a colored gradient, to make the lifetime effect more clear.
- The library builds with `#[deny(dead_code)]`.

### Removed

- Deleted `InitLayout`, `UpdateLayout`, and `RenderLayout`. Init, update, and render modifiers are now directly applying themselves to the `InitContext`, `UpdateContext`, and `RenderContext`, respectively.

### Fixed

- Fixed the `spawn` example failing to start on several devices due to the monitor resolution being larger than the maximum resolution imposed by the downlevel settings of WGPU. The downlevel settings are now disabled by default, and can be manually re-added for testing.

## [0.5.3] 2023-02-07

### Fixed

- Fix a panic on `unwrap()` after despawning N > 1 effects and re-spawning M < N effects. (#123)

### Changed

- Changed the `instance.rs` example to spawn effects in random positions. Also added a (disabled) stress test which randomly spawns and despawns effects quickly to uncover bugs more easily.

## [0.5.2] 2023-01-20

### Added

- Made `Gradient<T>` reflected and serializable by implementing `Reflect`, `FromReflect`, `Serialize`, and `Deserialize`.

### Fixed

- Fix (most common cases of) a bug where effects spawned after another effect was despawned will not work. This is a partial workaround; the bug can still trigger but under more rare conditions. (#106)
- Fix simulate compute jobs running once per view instead of once per frame. (#102)
- Fix 2D rendering not using indirect (GPU-driven) rendering.
- Fix particles being reset after another effect is despawned. (#117)

### Removed

- Removed `MinMaxRect` in favor of Bevy's own `Rect` type.
- Removed `Resource` derive from `EffectAsset`, which made little sense on an asset.

## [0.5.1] 2022-12-03

### Added

- Add support for HDR cameras (`Camera::hdr == true`).
- Add support for render layers (`RenderLayers`), allowing to select which camera(s) renders the `ParticleEffect`s.
- Add support for linear drag force via the `LinearDragModifier`. This enables slowing down the particles over time.

### Fixed

- Fix a panic when running the plugin without any effect.
- Fix a bug in the way `BillboardModifier` was projecting the particle vertices onto the camera plane, producing some partial or total clipping of particles.

## [0.5.0] 2022-11-14

### Changed

- Switch to Bevy v0.9.
- Disabled broken effect batching until #73 is fixed, to prevent triggering batching which breaks rendering.
- Switched to a new internal architecture, splitting the initializing of newly spawned particles from the updating of all alive particles, to achieve more consistent workload on the update compute. Added GPU-driven compute dispatch and rendering, which slighly improves performance and reduces CPU dependency/synchronization. This is mostly an internal change, but with the potential to unblock or facilitate several other issues. (#19)
- Removed `ParticleEffect::spawner()` from the public API, which was intended for internal use and is a bit confusing.
- Renamed the oddly-named `RenderLayout::size_color_gradient` into the more understandable `RenderLayout::lifetime_size_gradient`.
- Renamed `PipelineRegistry` into `ShaderCache`, and its `configure()` method into `get_or_insert()`, for clarity.

### Fixed

- Prevent `ShaderCache::get_or_insert()` from unnecessarily triggering change detection on `Assets<Shader>` when the item is already in the cache.

## [0.4.1] 2022-10-28

### Fixed

- Respect user-defined MSAA setting by reading the value of `Msaa::samples` when building the render pipeline. (#59)
- Fixed a bug in the effect cache causing a panic sometimes when effects are removed. (#60)
- Fixed a bug where an effect instance would be allocated overwriting another existing instance.
- Fixed a bug in the calculation of some GPU buffer binding causing a panic under some combination of effect capacity and spawn count. (#68)

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
