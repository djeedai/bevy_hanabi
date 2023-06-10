# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] 2023-06-10

### Added

- Added `SetColorModifier` to set a per-particle color on spawning, which doesn't vary during the particle's lifetime.
- Added `SetSizeModifier` to set a per-particle size on spawning, which doesn't vary during the particle's lifetime.
- Added `EffectAsset::motion_integration` to configure the type of motion integration of the particles of a system. Using this field, the user can now completely disable motion integration, or perform it _before_ the modifiers are applied. The default behavior remains to perform the motion integration _after_ all modifiers have been applied (`MotionIntegration::PostUpdate`).
- Added `InitAttributeModifier` to initialize any attribute of a particle to a given hard-coded value or CPU property value.
- `Attribute` is now fully reflected (`Reflect` and `FromReflect`) as a struct with a "name" field containing its unique identifier, and a "default_value" field containing its default `Value`. However note that attributes are immutable, so all mutating methods of the `Reflect` and `FromReflect` traits are no-op, and will not produce the expected result. The `FromReflect` implementation expects a `String` value, and uses `Attribute::from_name()` to recover the corresponding built-in instance of the attribute with that name.
- `Attribute` is now serializable. Attributes are serialized as a string containing their name, since the list of valid attributes is hard-coded and cannot be modified at runtime, and new custom attributes are not supported. The default value is not serialized.
- Init modifiers now also have access to the effect's properties.
- Added `InitAttributeModifier` to initialize any attribute to a value or bind it to a property. This is especially useful with properties currently used implicitly like `Attribute::COLOR` and `Attribute::HDR_COLOR`; now you can set the color of particles without the need to (ab)use a `ColorOverLifetimeModifier` with a uniform gradient. The binding with properties also allows dynamically changing the spawning color; see the updated `spawn_on_command.rs` example.
- Added 2 new components:
  - `CompiledParticleEffect` caches the runtime data for a compiled `ParticleEffect`. This component is automatically managed by the library. Users can interact with it to manage the values of the properties for that particular effect instance.
  - `EffectSpawner` holds the runtime data for the spawner. Several fields and methods have been transfered from [`EffectAsset::spawner`] and [`ParticleEffect::spawer`] into this new component. The component is automatically spawned by the library. Users can interact with it to _e.g._ spawn a single burst of particle.
- Added a new system set `EffectSystems::CompileEffects` running the new `compile_effects()` system in parallel of the `tick_spawners()` system, during the `CoreSet::PostUpdate` set.
  - `compile_effects()` updates the `CompiledParticleEffect` of an `Entity` based on the state of its `ParticleEffect`.
  - `tick_spawners()` spawns if it doesn't exist then ticks the `EffectSpawner` component located again on that same `Entity`.

  Neither of those systems mutate the `ParticleEffect` component anymore (the change detection mechanism of Bevy components will not be triggered).

- Added `ParticleEffect::with_properties()` to define a set of properties from an iterator. Note however that the `set_property()` method moved to the `CompiledParticleEffect`.
- Added a `SimulationCondition` enum and an `EffectAsset::simulation_condition` field allowing to control whether the effect is simulated while hidden (`Visibility::Hidden`). (#166)
- Added a new `RadialAccelModifier` update modifier applying a per-particle acceleration in the radial direction defined as the direction from the modifier's specified origin to the particle current position.
- Added a new `TangentAccelModifier` update modifier applying a per-particle acceleration in the tangent direction defined as the cross product of the the modifier's specified rotation plane axis with the radial direction of the particle (calculated like `RadialAccelModifier`).

### Changed

- `Attribute` is now a wrapper around the private type `&'static AttributeInner`, and should be considered a "handle" which references an existing attribute and can be cheaply copied and compared. All mentions of `&'static Attribute` from previous versions should be replaced with `Attribute` (by value) to migrate to the new definition.
- The `Attribute::AGE` and `Attribute::LIFETIME` are not mandatory anymore, and are now only added to the particle layout of an effect if a modifier requires them.
  - Particles without an age are (obviously) not aged anymore.
  - Particles without a lifetime are not reaped and therefore do not die from aging. They continue to age though (their `Attribute::AGE` is updated each frame).
- The `Attribute::POSITION` and `Attribute::VELOCITY` are not mandatory anymore. They are required if `EffectAsset::motion_integration` is set to something other than `MotionIntegration::None`, but are not added automatically to all effects like they used to be, and instead require a modifier to explicitly insert them into the particle layout. Effects with a non-`None` motion integration but missing either of those two attributes will emit a warning at runtime. Add a position or velocity initializing modifier to fix it.
- The documentation for all modifiers has been updated to state which attribute(s) they require, if any. Modifiers insert the attributes they require into the particle layout of the effect the modifier is attached to.
- `Spawner` now contains the user-provided spawner configuration, which is serialized with the `EffectAsset`. All runtime fields and related methods, which are not serialized, have been moved to the new `EffectSpawner` components. Users should replace the following calls previously made on `ParticleEffect::maybe_spawner().unwrap()` to the new `EffectSpawner`: `set_active()`, `with_active()`, `is_active()`, `reset()`. See _e.g._ the `spawn_on_command.rs` example.
- The former `Spawner::tick()`, now moved to `EffectSpawner::tick()`, is now a public method. The method is still automatically called by the `tick_spawners()` system. It's publicly exposed for testing and in case users want more control.
- Moved `ParticleEffect::set_property()` to `CompiledParticleEffect` to prevent triggering change detection on `ParticleEffect` which invalidates the cache. (#162)
- The `Spawner` methods `with_active()`, `set_active()`, and `is_active()`, have been respectively renamed to `with_starts_active()`, `set_starts_active()`, and `starts_active()`. This highlights the fact the "active" state manipulated by those methods only refers to the initial state of the spawner. The current runtime active state is available from the `EffectSpawner` once it's spawned by `tick_spawners()` (after the first udpate).

### Removed

- Deleted the following unused types: `EffectMaterial`, `EffectMaterialUniformData`, `EffectMaterialPlugin`, `GpuEffectMaterial`.

### Fixed

- Fixed a bug where using `ParticleEffect::with_spawner()` would prevent properties from initializing correctly.
- Fixed a bug where the effect texture of the previously batched effect was incorrectly selected instead of the texture of the current effect. (#167)
- Fixed a bug on some GPUs (most notably, on macOS) where incorrect data padding was breaking simulation of all but the first effect. (#165)
- Fixed calls to `ParticleEffect::set_property()` being ignored if made before the particle effect has been updated once, due to properties not being resolved into the `EffectAsset` until the effect is effectively compiled. The `set_property()` method has now moved to the new `CompiledParticleEffect`, so cannot by design be made anymore before the effect is first updated.
- Fixed `ParticleEffect::set_property()` invalidating the shader cache of the particle effect and causing a full shader recompile, which was impacting performance and defeating the point of using properties in the first place. The method has been moved to `CompiledParticleEffect`. (#162)
- Fixed a bug where hidden (`Visibility::Hidden`) but still active effects were simulated in the background despite the documentation stating they were not. The newly-added `SimulationCondition::Always` allows explicitly opting in to this behavior in case you were relying on it, but this fix otherwise prevent those effects from being simulated. This _possibly_ relates to #67.

## [0.6.1] 2023-03-13

### Added

- Added an example `init.rs` showing the various kinds of position initializing modifiers.

### Changed

- Renamed `PositionCone3dModifier` into `InitPositionCone3dModifier`, and removed the velocity initializing. To recover the older behavior, add an extra `InitVelocitySphereModifier` to initialize the velocity like `PositionCone3dModifier` used to do.

### Fixed

- Fixed a bug in `PositionCone3dModifier` where the translation of the emitter is applied twice. (#152)

## [0.6.0] 2023-03-10

### Added

- Added a `SimulationSpace` enum with a single value `Global`, in preparation of future support for local-space particle simulation.
- Added `InitAgeModifier` to initialize the age of particles to a specific (possibly random) value instead of the default `0.0`.
- Added _properties_, named quantities associated with an `EffectAsset` and modifiable at runtime and assignable to values of modifiers. This enables dynamically modifying at runtime any property-based modifier value. Other non-property-based values are assumed to be constant, and are optimized by hard-coding them into the various shaders. Use `EffectAsset::with_property()` and `EffectAsset::add_property()` to define a new property before binding it to a modifier's value. The `spawn.rs` example demonstrates this with the acceleration of the `AccelModifier`.
- Added particle `Attribute`s, individual items holding a single per-particle scalar or vector value. Composed together, the attributes form the particle layout, which defines at runtime the set of per-particle values used in the simulation. This change marks a transition from the previous design where the particle attributes were hard-coded to be the position, velocity, age, and lifetime of the particle. With this change, a collection of various attributes is available for modifiers to manipulate. This ensures flexibility in customizing while retaining a minimal per-particle memory footprint for a given effect. Note that [`Attribute`]s are all built-in; a custom [`Attribute`] cannot be used.
- Added a new `AabbKillModifier` which kills all particles entering or exiting an AABB. The `force_field.rs` example shows an example of each variant.
- Added a new `OrientAlongVelocityModifier` which orients the local X axis of particles alongside their velocity, and builds a local Z axis perpendicular to the camera view's plane. The local Y axis is derived to form an orthonormal frame.

### Changed

- Switch to Bevy v0.10.
- The `ParticleLifetimeModifier` was renamed to `InitLifetimeModifier` for clarity, and its `lifetime` field is now a `Value<f32>` to allow randomizing per particle.
- Effects no longer have a default particle lifetime of 5 seconds. Instead an explicit lifetime must be set with the `InitLifetimeModifier`. Failure to set the lifetime will trigger a warning at runtime, and particles will default to a lifetime of zero and instantly die.
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

- Fixed a bug breaking effect simulation after some effects are despawned and others are subsequently spawned. (#106)
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
