# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a new `serde` feature, and moved all `Serialization` and `Deserialization` derives under that feature.
  This allows disabling support on wasm, where `typetag` is not available.
  The feature is enabled by default, so the behavior doesn't change on non-wasm targets.
- Added support for WebAssembly (`wasm`). This requires disabling the `serde` feature. (#41)
- Added support for sampling textures in the Expression API via the `TextureSampleExpr` expression.
  Currently only supports color textures (no depth texture) and only sampling via WGSL `textureSample()`. (#355)
- Added a new `EffectMaterial` component holding the actual textures to bind to the various slots of a `Module`.
- Added a new `Module::add_texture()` function to declare a new texture slot in a module.
- Particle trails and ribbons can now be initialized with init modifiers.

### Changed

- The `ParticleTextureModifier` doesn't directly hold the texture handle to use anymore.
  Instead, it holds an expression handle to the texture slot defined in the `Module` of the effect.
  This allows dynamically changing the texture sampled.
- `RenderModifier::apply_render()` now returns a `Result<String, ExprError>`.
- Several API changes occurred around the cloning of particles for trails and ribbons. See also the [migration guide (v0.12 -> v0.13)](./docs/migration-v0.12-to-v0.13.md).
  - `EffectSpawner` is now wrapped into a new `EffectInitializers`. Each effect group has an `EffectInitializer`, which can either be an `EffectSpawner` (CPU spawning) or an `EffectCloner` (GPU particle cloning).
  - `EffectAsset::new()` takes again a single capacity argument for the default first group. Other groups are incrementally added with `with_trails()` and `with_ribbons()`, specifying their respective capacity via those functions.
  - The age (`Attribute::AGE`) and lifetime (`Attribute::LIFETIME`) of cloned particles can no longer be assigned manually; instead it's set via an argument to `EffectAsset::with_trails()` and `EffectAsset::with_ribbons()`, and cannot be modified anymore with expressions (and properties in particular).

### Removed

- The `CloneModifier` was removed. Cloning is now controlled by the `EffectCloner`.
- The `RibbonModifier` was removed. Use `EffectAsset::with_ribbons()` to create ribbons.

### Fixed

- Fixed a bug where the generated render shader was declaring a binding as `storage<read>` (read-only)
  but the `Spawner` struct contained an `atomic<i32>`, which requires write access.
  The `Spawner` struct is now conditionally defining that field as `i32` instead.
- Fixed a race condition in ribbons leading to visual artifacts (particles linked to other unrelated particles). (#376)

## [0.12.2] 2024-08-05

### Fixed

- Fix regression building with `rustc 1.79`, which is the Bevy MSRV. (#360)

## [0.12.1] 2024-07-28

### Fixed

- Fix new `rustc 1.80` linter error breaking the crate build. (#351)

## [0.12.0] 2024-07-09

### Added

- Added `BuiltInOperator::IsAlive` exposing the current state of the `is_alive` particle flag,
  which represents whether a particle is still alive or will be deallocated at the end of the update pass.

### Changed

- Compatible with Bevy 0.14.

### Fixed

- Prevented an error from being emitted when a GPU pipeline is still being created. (#341)
- Fixed a panic when a particle effect as an invalid asset handle. (#343)

## [0.11.0] 2024-05-29

### Added

- Added a new `ScreenSpaceSizeModifier` which negates the effect of perspective projection, and makes the particle's size a pixel size in screen space, instead of a Bevy world unit size. This replaces the hard-coded behavior previously available on the `SetSizeModifier`.
- Added a new `ConformToSphereModifier` acting as an attractor applying a force toward a point (sphere center) to all particles in range, and making particles conform ("stick") to the sphere surface.
- Added `vec2` and `vec3` functions that allow construction of vectors from dynamic parts.
- Added missing `VectorValue::as_uvecX()` and `impl From<UVecX> for VectorValue` for all `X=2,3,4`.
- Added a new `ShaderWriter` helper, common to the init and update modifier contexts, and which replaces the `InitContext` and `UpdateContext`.
- Added a simple `impl Display for ModifierContext`.
- Added `Modifier::apply()` which replaces the now deleted `InitModifier::apply_init()` and `UdpdateModifier::apply_update()`.
- Added `RenderModifier::as_modifier()` to upcast a `RenderModifier` to a `Modifier`.
- Added `RenderModifier::boxed_render_clone()` to clone a `RenderModifier` into a boxed self (instead of a `BoxedModifier`, as we can't upcast from `Box<dyn RenderModifier>` to `Box<dyn Modifier>`). Added `impl Clone for Box<dyn RenderModifier>` based on this.
- Added `EffectAsset::add_modifier()` to add a pre-boxed `Modifier` (so, a `BoxedModifier`) to the init or update context, and added `EffectAsset::add_render_modifier()` to add a pre-boxed `RenderModifier` to the render context.
- Added `sqrt` and `inverseSqrt` expressions to the Expression API.
- Added `Hash` derive for `Property`.
- Added `Module::add_property()`, `Module::get_property()`, and `Module::properties()`.
- Added a new `PropertyHandle`, which is to `Property` what the existing `ExprHandle` is to `Expr`: a unique handle into a owning `Module`.
- Added a new `RoundModifier` that allows round particles to be constructed without having to create a texture.
- Added a new `trace` feature enabling some `span_info!()` profiling annotations.
- Added the ability to make particle trails, via the `CloneModifier` and the addition of separate particle groups. See the `worms.rs` example for usage.
- Added the ability to stitch trail particles together to form ribbons with the `RibbonModifier`.

### Changed

- `ExprHandle` is now `#[repr(transparent)]`, which guarantees that `Option<ExprHandle>` has the same size as `ExprHandle` itself (4 bytes).
- `EffectProperties::set_if_changed()` now returns the `Mut` variable it takes as input, to allow subsequent calls.
- `VectorValue::new_uvecX()` now take a `UVecX` instead of individual components, like for all other scalar types.
- Merged the `InitModifier` and `UpdateModifier` traits into the `Modifier` subtrait; see other changelog entries for details. This helps manage modifiers in a unified way, and generally simplifies writing and maintain modifiers compatible with both the init and update contexts.
- `EffectAsset::init()` and `EffectAsset::update()` now take a `Modifier`-bound type, and validate its `ModifierContext` is compatible (and panics if not).
- `EffectAsset::render()` now panics if the modifier is not compatible with the `ModifierContext::Render`. Note that this indicates a malformed render modifier, because all objects implementing `RenderModifier` must include `ModifierContext::Render` in their `Modifier::context()`.
- Improved the serialization format to reduce verbosity, by making the following types `#[serde(transparent)]`: `ExprHandle`, `LiteralExpr`, `PropertyExpr`.
- Moved properties from `EffectAsset` to `Module`.
- `Module::prop()` and `PropertyExpr::new()` now take a `PropertyHandle` instead of a property name (string).
- The following modifiers' `via_property()` function now takes a `PropertyHandle` instead of a property name: `AccelModifier`, `RadialAccelModifier`, `TangentAccelModifier`.
- `Expr` is now `Copy`, making it even more lightweight.
- `ExprWriter::prop()` now panics if the property doesn't exist. This ensures the created `WriterExpr` is well-formed, which was impossible to validate at expression write time previously.
- Effects rendered with `AlphaMode::Mask` now write to the depth buffer. Other effects continue to not write to it. This fixes particle flickering for alpha masked effects only.
- Bind groups for effect rendering are now created in a separate system in the `EffectSystems::PrepareBindGroups` set, itself part of Bevy's `RenderSet::PrepareBindGroups`. They're also cached, which increases the performance of rendering many effects.
- Merged the init and update pass bind groups for the particle buffer and associated resources in `EffectBfufer`. The new unified resources use the `_sim` (simulation) suffix.
- `CompiledParticleEffect` now holds a strong handle to the same `EffectAsset` as the `ParticleEffect` it's compiled from. This ensures the asset is not unloaded while in use during the frame. To allow an `EffectAsset` to unload, clear the handle of the `ParticleEffect`, then allow the `CompiledParticleEffect` to observe the change and clear its own handle too.
- The `EffectProperties` component is now mandatory, and has been added to the `ParticleEffectBundle`. (#309)

### Removed

- Removed the `screen_space_size` field from the `SetSizeModifier`. Use the new `ScreenSpaceSizeModifier` to use a screen-space size.
- Removed the built-in `ForceFieldSource` and associated `ForceFieldModifier`. Use the new `ConformToSphereModifer` instead. The behavior might change a bit as the conforming code is not strictly identical; use the `force_field.rs` example with the `examples_world_inspector` feature to tweak the parameters in real time and observe how they work and change the effect.
- Removed `InitContext` and `UpdateContext`; they're replaced with `ShaderWriter`.
- Removed `InitModifer` and `UpdateModifer`. Modifiers for the init and update contexts now only need to implement the base `Modifier` trait.
- Removed downcast methods from `Modifier` (`as_init()`, `as_init_mut()`, `as_update()`, `as_update_mut()`).
- Removed the various helper macros `impl_mod_xxx!()` to implement modifier traits; simply implement the trait by hand instead.
- Removed `EffectAsset::with_property()` and `EffectAsset::add_property()`; use `Module::add_property()` instead.
- `PropertyExpr` doesn't implement anymore `ToWgslString` nor `From<String>`. To create a `PropertyExpr`, you need a valid `PropertyHandle`, which would have been created via `Module::add_property()` with a default value enforcing a type. This ensures property expressions are well-formed on creation, and cannot reference non-existent properties.
- Removed `ParticleEffect::spawner`, which was used as a per-instance override of `EffectAsset::spawner`. Manual spawn an `EffectSpawner` component with overridden values to achieve the same feature.

### Fixed

- Fixed a panic in rendering randomly occurring when no effect is present.
- Fixed invalid WGSL being generated for large `u32` values.
- Fixed particle flickering, only for particles rendered through the `AlphaMask3d` render pass (`EffectAsset::alpha_mode == AlphaMode::Mask`). (#183)
- Fixed invalid layout of all `mat3xR<f32>` returned by `MatrixValue::as_bytes()`, which was missing padding. (#310)
- Fixed a regression where declaring properties but not adding an `EffectProperties` component would prevent properties from being uploaded to GPU. The `EffectProperties` component is now mandatory, even if the effect doesn't use any property. However there's still no GPU resource allocated if no property is used. (#309)
- Fixed the missing PRNG seeding per particle effect instance in the update pass. (#333)

## [0.10.0] 2024-02-24

### Changed

- Compatible with Bevy 0.13
- Moved most shared WGSL code into an import module `vfx_common.wgsl`. This requires using `naga_oil` for import resolution, which in turns means `naga` and `naga_oil` are now dependencies of `bevy_hanabi` itself.

### Fixed

- Fixed a bug where particle attributes used in the context of a function would emit invalid code. (#275)
- Fixed a bug where the screen-space size of particles was not correctly computed, leading to small variations in size. The new code correctly sizes particles based on a screen pixel size. This may increase the actual size of particles, if a larger size had been previously used to compensate the error introduced by this bug. (#269)
- Fixed a bug where screen-space size ignored the particle's local orientation. (#269)

## [0.9.0] 2023-12-26

### Added

- Added a new `EffectProperties` component holding the runtime values for all properties of a single `ParticleEffect` instance. This component can be added manually to the same `Entity` holding the `ParticleEffect` if you want to set initial values different from the default ones declared in the `EffectAsset`. Otherwise Hanabi will add the component automatically.
- Added a new `EffectSystems::UpdatePropertiesFromAsset` set running in the `PostUpdate` schedule. During this set, Hanabi automatically updates all `EffectProperties` if the properties declared in the underlying `EffectAsset` changed.
- Added `OrientModifier::rotation`, an optional expression which allows rotating the particle within its oriented plane. The actual meaning depends on the `OrientMode` used. (#258)
- Added custom float attributes, which have no predefined meaning, but instead can be used to store any per-particle value.
  - `Attribute::F32_0` to `Attribute::F32_3` for scalar `f32` attributes.
  - `Attribute::F32X2_0` to `Attribute::F32X2_3` for vector `vec2<f32>` attributes.
  - `Attribute::F32X3_0` to `Attribute::F32X3_3` for vector `vec3<f32>` attributes.
  - `Attribute::F32X4_0` to `Attribute::F32X4_3` for vector `vec4<f32>` attributes.
- Added 4 new expressions for packing and unpacking a `vec4<f32>` into a `u32`: `pack4x8snorm`, `pack4x8unorm`, `unpack4x8snorm`, `unpack4x8unorm`. This is particularly useful to convert between `COLOR` (`u32`) and `HDR_COLOR` (`vec4<f32>`). See the `billboard.rs` example for a use case. (#259)

### Changed

- Properties of an effect have been moved from `CompiledParticleEffect` to a new `EffectProperties` component. This splits the semantic of the `CompiledParticleEffect`, which is purely an internal optimization, from the list of properties stored in `EffectProperties`, which is commonly accessed by the user to assign new values to properties.
- Thanks to the split of properties into `EffectProperties`, change detection now works on properties, and uploads to GPU will only occur when change detection triggered on the component. Previously properties were re-uploaded each frame to the GPU even if unchanged.
- Effect properties are now reflected (via the new `EffectProperties` component).
- `Attribute::ALL` is now private; use `Attribute::all()` instead.
- Exposed publicly the `attributes` and `properties` modules, and documented them.

## [0.8.0] 2023-11-08

### Added

- Added a new `OrientModifier` and its `OrientMode`, allowing various modes of particle orienting during the rendering phase.
- Added `SimulationSpace::Local` to simulate particles in local effect space, before rendering them with the `GlobalTransform` of the effect's entity.
- Add access to `ModifierContext` and `ParticleLayout` from the `EvalContext` when evaluating modifiers.
- Added `SimulationSpace::eval()` to evaluate a context-specific expression allowing to transform the particles to the proper simulation space.
- Added a few more functions to `Gradient<T>`: `is_empty(` and `len()` which do as implied, `from_keys()` which creates a new gradient from a key point iterator, and `with_key()` and `with_keys()` which append one or more keys to an existing gradient.
- Added `AlphaMode` and the ability to render particles with alpha masking instead of alpha blending. This is controlled by `EffectAsset::alpha_mode` and the new `EffectAsset::with_alpha_mode()` helper.
- Added a new `BuiltInOperator::AlphaCutoff` value and associated expression, which represent the alpha cutoff threshold when rendering an effect with alpha masking. The `billboard` example has been updated to show how to use that value, and even dynamically change it with an expression.
- Added `PropertyLayout::properties()` to iterate over the layout.
- Added `From` implementations for the most common matrix types.
- Added many more expressions to `Expr`.
- Added `Expr::has_side_effect()` to determine if an expression has a side effect and therefore needs to be stored into a temporary local variable to avoid being evaluated more than once.
- Added `EvalContext::make_local_var()` to generate a unique name for a variable local to an `EvalContext` (generally, inside a function).
- Added `EvalContext::push_stmt()` to emit a single statement prepended to the currently evaluating expression. This is useful to define temporary local variables for storing expressions with a side-effect.
- Added `EvalContext::make_fn()` to create a function with a dedicated `EvalContext`, allowing to properly scope local variables and stored expression side effects.
- Added `Module::try_get()`, similar to `Module::get()` but returning a `Result<&Expr, ExprError>` instead for convenience.
- Added implementations of `ToWgslString` for the missing vector types (`UVec2/3/4`, `IVec2/3/4`, `BVec2/3/4`).
- Added new `CastExpr` expression to cast an operand expression to another `ValueType`. This adds a new variant `Expr::Cast` too.
- Added new `BinaryOperator::Remainder` to calculate the remainder (`%` operator) of two expressions.
- Added the `ImageSampleMapping` enum to determine how samples of the image of a `ParticleTextureModifier` are mapped to and modulated with the particle's base color. The new default behavior is `ImageSampleMapping::Modulate`, corresponding to a full modulate of all RGBA components. To restore the previous behavior, and use the Red channel of the texture as an opacity mask, set `ParticleTextureModifier::sample_mapping` to `ImageSampleMapping::ModulateOpacityFromR`.
- Added new `FlipbookModifier` to treat the image of a `ParticleTextureModifier` as a grid sprite sheet, and allow rendering a sprite from that sheet. By animating the selected sprite, this creates a flipbook animation for the particle.
- Added new `Attribute::SPRITE_INDEX` holding the `i32` index of a sprite inside a sprite sheet texture. This is used with the `FlipbookModifier` to render sprite-based animated particles.

### Changed

- Compatible with Bevy 0.12
- `InitContext::new()` and `UpdateContext::new()` now take an additional reference to the `ParticleLayout` of the effect.
- `RenderContext` now implements `EvalContext` like the init and update contexts, and like them reference the module, particle layout, and property layout of the effect.
- `Gradient<T>::new()`, `Gradient<T>::constant()`, and `Gradient<T>::linear()` do not require the `T: Default` trait bound anymore. The bound had been added by mistake, and is not necessary.
- `Gradient<T>::new()` is now a `const fn`.
- `Gradient<T>::constant()` and `Gradient<T>::linear()` do not attempt to perform linear searches anymore; instead they directly create the `Gradient<T>` object from scratch. This should not have any real consequence in practice though.
- Changed `CompiledParticleEffect` to store a `LayoutFlags` instead of individual boolean values, for convenience and consistency with the internal representation.
- Changed `RenderContext` to implement `EvalContext`. This allows render modifiers to use the expression API.
- `PropertyLayout::generate_code()` has no more extra empty line at the end of the struct in the generated code.
- `EvalContext::eval()` now caches the evaluation of an `ExprHandle` and guarantees that the evaluation is only ever performed once. This ensures that cloned `ExprHandle` making a same expression used in multiple places all reference the same evaluation, which is stored inside a local variable. This fixes an unexpected behavior where expressions with side effect like `rand()` where emitted multiple times, leading to different values, even though a single expression was used (via cloned handles). To restore the old behavior, simply generating separate expressions from a `Module` or an `ExprWriter` instead of cloning and reusing a same `ExprHandle`.
- The default texture sampling mode for `ParticleTextureModifier` is now a full RGBA modulate. See `ImageSampleMapping` for details. Use `ImageSampleMapping::ModulateOpacityFromR` to restore the previous behavior.

### Removed

- Removed the `BillboardModifier`; this is superseded by the `OrientModifier { mode: OrientMode::ParallelCameraDepthPlane }`.
- Removed the `OrientAlongVelocityModifier`; this is superseded by the `OrientModifier { mode: OrientMode::AlongVelocity }`.
- Removed `module()` and `expr()` from `EvalContext`; the current module is now passed explicitly alongside the `EvalContext` in functions such as `EvalContext::eval()`.

### Fixed

- Render modifiers can now access simulation parameters (time, delta time) like in any other context.
- Fixed a panic in Debug builds when a `ParticleEffect` was marked as changed (for example, via `Mut`) but the asset handle remained the same. (#228)
- Fixed a bug in the `to_wgsl_string` impl of `MatrixType` that caused the first element to be added twice.
- Fixed missing parentheses leading to incorrect operator order in the following modifiers depending on the expression(s) used:
  - `SetPositionCircleModifier`
  - `SetPositionSphereModifier`
  - `SetVelocityCircleModifier`
  - `SetVelocitySphereModifier`
  - `SetVelocityTangentModifier`

## [0.7.0] 2023-07-17

### Added

- Added `Gradient::linear()` helper method to produce a linear gradient between two values at keys `0.` and `1.`.
- `EffectAsset` now owns a `Module` field containing all the `Expr` used by the effect's modifiers.
- Added `ScalarType`, `VectorType`, and `MatrixType` to reprensent a scalar, vector, or matrix type, respectively.
- Added `ValueType::is_numeric()` as well as query methods to determine the kind of value type `is_scalar()` / `is_vector()` / `is_matrix()`.
- Added new Expression API: `Expr`, `ExprHandle`, `Module`, `ExprWriter`, `WriterExpr`.
- Added new `EvalContext` trait representing the evaluation context of an expression, and giving access to the underlying expression `Module`
  and the property layout of the efect. The trait is implemented by `InitContext` and `UpdateContext`.
- Added convenience method `PropertyLayout::contains()` to determine if a layout contains a property by name.
- Added `SetSizeModifier::screen_space_size` and `SizeOverLifetimeModifier::screen_space_size` boolean fields which change the behavior of the particle size to be expressed in screen-space logical pixels, independently of the camera projection. This enables creating particle effect with constant pixel size. Set `screen_space_size = false` to get the previous behavior.
- Added a new utility trait `FloatHash` to allow implementing `std::cmp::Eq` and `std::hash::Hash` on floating-point variants of `CpuValue` (ex-`spawn::Value`), making it possible to derive `std::hash::Hash` for any type using `CpuValue`.
- `SetColorModifier` and `SetSizeModifier` now implement `std::cmp::Eq`, thanks to `CpuValue` itself implementing that trait for all floating point types (see `FloatHash`).
- Added a new `KillSphereModifier`, similar to `KillAabbModifier` but with a sphere shape.

### Changed

- Renamed `spawn::Value` to `spawn::CpuValue` to prevent confusion with `graph::Value`. The former is a legacy construct which should eventually be replaced by the latter (but cannot be yet).
- `ValueType` is now one of `ScalarType` / `VectorType` / `MatrixType`, allowing to represent a wider range of types, including booleans and matrices.
- `graph::Value` is now one of `ScalarValue` / `VectorValue` / `MatrixValue`, for consistency with `ValueType`.
- `SimParams::dt` was renamed to `SimParams::delta_time` for readability. Inside shaders, `sim_params.dt` was also renamed to `sim_params.delta_time`.
- `InitContext` and `UpdateContext` now hold a mutable reference to the underlying `Module` to allow modifiers to create new `Expr`,
  and a read-only reference to the property layout of the effect.
- `ModifierContext` is now a bitfield (flags), allowing modifiers to be used in multiple contexts. A prime example is the renamed `SetAttributeModifier` which can be used both to initialize a particle attribute on spawn (if used in the `Init` context), or update it during the simulation (if used in the `Update` context). Adding the modifier to the `EffectAsset` with `init()` or `update()` determines in which context it's used.
- `InitModifier::apply()` was renamed to `InitModifier::apply_init()` to avoid a conflict with the other modifier traits in case a modifier implements multiple of them.
- `UpdateModifier::apply()` was renamed to `UpdateModifier::apply_update()` to avoid a conflict with the other modifier traits in case a modifier implements multiple of them.
- `RenderModifier::apply()` was renamed to `RenderModifier::apply_render()` to avoid a conflict with the other modifier traits in case a modifier implements multiple of them.
- `InitModifier::apply_init()` and `UpdateModifier::apply_update()` now return a `Result<(), ExprError>`.
- The following modifiers have been renamed, changing their `Init` prefix into `Set` to reflect the fact they can now be used both for particle init and update:
  - `InitAttributeModifier` -> `SetAttributeModifier`
  - `InitPositionCircleModifier` -> `SetPositionCircleModifier`
  - `InitPositionSphereModifier` -> `SetPositionSphereModifier`
  - `InitPositionCone3dModifier` -> `SetPositionCone3dModifier`
  - `InitVelocityCircleModifier` -> `SetVelocityCircleModifier`
  - `InitVelocitySphereModifier` -> `SetVelocitySphereModifier`
  - `InitVelocityTangentModifier` -> `SetVelocityTangentModifier`
- All modifiers were changed to leverage the new Expression API. This means most fields are now of type `ExprHandle` instead of their previous numeric type (like `Vec3` or `f32`). Refer to the various examples and the migration guide to understand how to migrate those modifiers.
- `Property::new()` takes a `default_value` argument as `impl Into<Value>` instead of `Value`. This should make it easier to call, without requiring any change to existing code.
- `PropertyLayout::new()` takes an `iter` argument as `impl IntoIterator` instead of `impl Iterator`. This should make it easier to call, without requiring any change to existing code.
- All `ParticleEffect`s are now compiled into a `CompiledParticleEffect` as soon as Hanabi detects they were spawned (generally, same frame), irrespective of whether they are visible or not. Previously only effects with `Visibility::Visible` where compiled, causing inconsistencies and panics when the effect was made visible later.
- Shaders are now named (as required by Bevy 0.11), allowing better error reporting. Because Hanabi shaders are generated, the naming pattern used is `hanabi/<effect_name>_<hash>.wgsl`, where `<effect_name>` is the value of `EffectAsset::name`, and `<hash>` a unique hash depending on the content of the effect (modifiers and their values).
- The content of the `modifier` module has been re-organized to group modifiers into submodules based on the part they modify. This means the `init`, `update`, and `render`, sub-modules are gone, replaced with others. Because all modifiers are re-exported (flattened hierarchy), this generally should not cause any breaking change, but can occasionally create a breakage if some old code was qualifying them with their full module path.

### Removed

- The `InitAgeModifier`, `InitLifetimeModifier`, and `InitSizeModifier`, were deleted. They're replaced with the more generic `SetAttributeModifier` (ex-`InitAttributeModifier`) which can set any attribute of a particle.
- `Modifier::resolve_properties()` was a temporary workaround which has now been entirely made obsolete, and as a result has been deleted.
- Deleted `DimValue` which was only used by `InitSizeModifier`, and is more generally covered by the Expression API.
- Deleted `ValueOrProperty` which is now unused, and is more generally covered by the Expression API.

### Fixed

- Fixed a bug where a `ParticleEffect` spawned hidden (with `Visibility::Hidden`) would make Hanabi panic when made visible. Effects are now always compiled as soon as spawned. (#182)
- Fixed the implementation of `std::hash::Hash` for `SetColorModifier` and `SetSizeModifier`, which were manually implemented and were not hashing the variant type of the value. This increased the risk of hash collision. The new implementation is derived thanks to `CpuValue` itself now implementing `std::hash::Hash`, and therefore likely hashes to a different value than in the previous release.

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
