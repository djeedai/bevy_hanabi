# Migration Guide v0.6 -> v0.7

🎆 Hanabi v0.7 marks an important milestone for the project: the addition of a generic expression-based API to enable complete customizing of the visual effects by the user.

The _Expression API_ departs radically from the previous version where the behavior of a particle was exclusively hard-coded into modifiers without any possibility to customize that behavior. Now, modifiers can leverage the Expression API to allow the user to provide a generic _expressions_ for any of their input value, rendering those modifiers completely under the control of the user.

This guide helps the user migrate from the pre-Expression v0.6 to the new Expression-based v0.7 of 🎆 Hanabi. Users are encouraged to also read the [`CHANGELOG`](../CHANGELOG.md) for an exhaustive list of all changes.

## Understanding Expressions

An _expression_ is an object generating some WGSL shader code to produce a value at runtime when the shader is executed on the GPU. For example, `5.` is a literal (constant) expression, whose value is hard-coded and will never change. `max(x, y)` on the other hand produces a value equal to the maximum of two other values. Those two other values are themselves expressions; expressions combine together to form more complex expressions.

An expression is represented by the `Expr` enum. Expressions are stored into a `Module`, and referenced by an `ExprHandle`, a sort of index into the module's internal storage. Each `EffectAsset` owns a `Module` which stores all the `Expr` used by the modifiers associated with that `EffectAsset`.

Because `Module` and `Expr` are focusing on storage (serialization) and runtime execution, manipulating expressions directly can be verbose. The `ExprWriter` utility simplifies writing expressions by providing a concise syntax, operator overloading (_e.g._ `impl std::ops::Add`), and other helper methods which make writing new expressions shorter and clearer.

The typical workflow for the end user is as follow:

1. Call `ExprWriter::new()` to create a new expression writer. The new writer internally allocates a `Module`, that it'll use to store expressions as they're produced.
1. Build new expressions with the `ExprWriter` methods and the use of the associated `WriterExpr` type. A `WriterExpr` is very similar to an `ExprHandle`, but is more heavyweight at the benefit of allowing a more concise syntax and operator overloading. `WriterExpr` represents an intermediate expression, and should not be stored long term.
1. Finalize any expression with `WriterExpr::expr()`, writing the corresponding `Expr` into the underlying `Module` and recovering the associated `ExprHandle`.
1. Create a new modifier and assign the `ExprHandle` to one of its field.
1. Repeat until all modifiers are ready for a given `EffectAsset`.
1. Finish using the writer with `ExprWriter::finish()`, recovering the finalized `Module` containing all written `Expr`.
1. Create a new effect with `EffectAsset::new()`, passing that `Module` as argument.

That last point is critical; expressions are owned by a `Module`, and assigning an `ExprHandle` from a different module produces undefined behaviors.

## Migrating `EffectAsset`

The `EffectAsset` struct has a `new()` associated function serving as the new entry point to create a new `EffectAsset` instance. This function takes the effect capacity, which is immutable after creation, the spawner, and the `Module` storing the effect expressions.

Commonly, the `Module` passed to `EffectAsset::new()` will be the one retrieved from the effect writer with `EffectWriter::finish()`.

// OLD v0.6

```rust
let asset = EffectAsset {
    capacity: 256,
    spawner: Spawner::rate(256.0.into()),
     ..Default::default()
};
```

// NEW v0.7

```rust
let writer = ExprWriter::new();
// [...] (use writer to setup modifiers)
let module = writer.finish();
let asset = EffectAsset::new(256, Spawner::rate(256.0.into()), module);
```

More rarely if no `Expr` is used in any modifier, just a default module can be used. As most modifiers use expressions, passing an empty `Module` is really anecdotal.

```rust
let asset = EffectAsset::new(256, Spawner::rate(256.0.into()), Module::default());
```

## Migrating modifiers

To migrate your modifiers from v0.6 to v0.7, you need to follow the steps above to prepare a `Module` with all the `Expr` describing the modifier inputs, then assign that module to the `EffectAsset` when you create it. Most examples have been migrated this way, and can be referenced for further details.

First, replace the following modifiers:

| Old (v0.6) | New (v0.7) | Comment |
|---|---|---|
| `InitAttributeModifier` | `SetAttributeModifier` | Modifier now useable in both `Init` and `Update` contexts. |
| `InitAgeModifier` | `SetAttributeModifer(Attribute::AGE)` | Old modifier deleted. |
| `InitLifetimeModifier` | `SetAttributeModifer(Attribute::LIFETIME)` | Old modifier deleted. |
| `InitSizeModifier` | `SetAttributeModifer(Attribute::SIZE)` | Old modifier deleted. |
| `InitPositionCircleModifier` | `SetPositionCircleModifier` | Modifier now useable in both `Init` and `Update` contexts. |
| `InitPositionSphereModifier` | `SetPositionSphereModifier` | Modifier now useable in both `Init` and `Update` contexts. |
| `InitPositionCone3dModifier` | `SetPositionCone3dModifier` | Modifier now useable in both `Init` and `Update` contexts. |
| `InitVelocityCircleModifier` | `SetVelocityCircleModifier` | Modifier now useable in both `Init` and `Update` contexts. |
| `InitVelocitySphereModifier` | `SetVelocitySphereModifier` | Modifier now useable in both `Init` and `Update` contexts. |
| `InitVelocityTangentModifier` | `SetVelocityTangentModifier` | Modifier now useable in both `Init` and `Update` contexts. |

Then, follow the above steps to build expressions for all the modifier fields which were migrated to use `ExprHandle`. For example, an `AccelModifier::accel` field previously initialized with a constant:

// OLD v0.6

```rust
let asset = EffectAsset {
    // [...]
    ..Default::default()
}
.update(AccelModifier::constant(Vec3::Y * -3.));
```

now requires building a literal expression instead:

// NEW v0.7

```rust
// Create an ExprWriter for convenience
let w = ExprWriter::new();

// Build the `accel` literal expression
let accel = w.lit(Vec3::Y * -3.);

// Write it down into the Module, and get back the ExprHandle
let accel_expr = accel.expr();

// Repeat for other modifiers...
// [...]

// Finish using the ExprWriter and recover the Module
let module = w.finish();

// Finally, create the EffectAsset with the modifiers
let asset = EffectAsset::new(capacity, spawner, module)
    .update(AccelModifier::new(accel_expr));
```

Note that previously a common pattern was to create modifiers inline while building the `EffectAsset`. This is not possible anymore for all modifiers using an `ExprHandle` (almost all of them), because the expression `Module` need to be finalized and assigned to the `EffectAsset` before the modifiers can be added to it. Otherwise the modifiers when they attach to the effect will fail their consistency check and panic.

## Other migration items

- Rename `spawn::Value` to `spawn::CpuValue`. This prevents confusion with `graph::Value` and allow importing both types at once. Going forward `CpuValue` is only used with the `Spawner`; all modifiers use `ExprHandle` instead.

- The `std::hash::Hash` implementation for `SetColorModifier` and `SetSizeModifier` changed. If you previously stored some hash values, they likely will be different between v0.6 and v0.7. You can check the old manual implementation in v0.6 if you need to write some conversion code.

- All typed values like `graph::Value::Float3` need to be replaced by their scalar/vector counterpart:
  - `graph::Value::Float(f)` becomes `graph::Value::Scalar(ScalarValue::Float(f)))`
  - `graph::Value::Float3(v)` becomes `graph::Value::Vector(VectorValue::new_vec3(v))`
  - _etc._
  
  Some conversions are provided via `From<>` /  `Into<>`, which can make the syntax shorter in some cases.

  // OLD v0.6

  ```rust
  let x = Value::Float(3.5);
  let v = Value::Float3(Vec3::ONE);
  ```

  // NEW v0.7

  ```rust
  let x = Value::Scalar(3.5.into());
  let v = Value::Vector(Vec3::ONE.into());
  // -OR-
  let x = Value::Scalar(ScalarValue::Float(3.5));
  let v = Value::Vector(VectorValue::new_vec3(Vec3::ONE));
  ```

- Same kind of conversions for `graph::ValueType`.

- Rename `SimParams::dt` into `SimParams::delta_time`, and any shader use of `dt` into `delta_time`.

- Add an extra `screen_space_size = false` field to the `SetSizeModifier` and `SizeOverLifetimeModifier`.

- `DimValue` was deleted. It was only used in the now deleted `InitSizeModifier`. There's no direct equivalent if you were using this in your code.
