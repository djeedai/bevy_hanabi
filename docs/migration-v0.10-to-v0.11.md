# Migration Guide v0.10 -> v0.11

🎆 Hanabi v0.11 contains several major changes.

This guide helps the user migrate from v0.10 to v0.11 of 🎆 Hanabi.
Users are encouraged to also read the [`CHANGELOG`](../CHANGELOG.md)
for an exhaustive list of all changes.

## Properties refactor

Effect _properties_ are shader variables modifiable from CPU.
They give great control over the simulation of particle effects,
at the small expense of a CPU to GPU upload when they change.

In 🎆 Hanabi v0.11, properties moved from the `EffectAsset`
to the `Module` where they're used.
The first reason for this change is user expectations;
properties are only ever used as part of expressions of a `Module`,
so it made little sense to have them declared as part of an `EffectAsset`.
The second, more fundamental reason is that this makes their default value
available at the point they're used in an expression.
This opens the door to writer-time validation of expressions as they're being built,
This also enables in the future automated cast based on type deduction,
which was not possible when the properties were defined on the `EffectAsset`,
after the `Module` has been fully written.

The properties usage changed as follow:

- Properties are now declared in 🎆 Hanabi v0.11 via `Module::add_property()` or `ExprWriter::add_property()`,
  which return a `PropertyHandle` similar to the `ExprHandle` for expressions.
- To create a property expression, `ExprWriter::prop()` now takes a `PropertyHandle` instead of a property name.
- Functions of various modifiers taking a property now take a `PropertyHandle` too.

To migrate, move the declaration of all properties from `EffectAsset` to the underlying `Module`.

// OLD v0.10

```rust
let writer = ExprWriter::new();
let p = writer.prop("my_prop");
// [...]
let asset = EffectAsset {
    // [...]
}.with_property("my_prop", Value::Scalar(3.0.into()));
```

// NEW v0.11

```rust
let writer = ExprWriter::new();
let my_prop = writer.add_property("my_prop", Value::Scalar(3.0.into()));
let p = writer.prop(my_prop);
```

Related to this, the `EffectProperties` component is now mandatory,
even if the effect doesn't use properties.
The component was added to the `ParticleEffectBundle`.
If you do not use the bundle, you must add that component manually
to the same `Entity` as your `ParticleEffect`.

## Leaner serialization format

The serialziation format was improved to make it less verbose.
A number of types have been marked as `#[serde(transparent)]`,
which removes a level of indirection.
In practice this generally means there's one less level of parentheses.

If you were relying on a particular serialization format,
or had asset serialized with a version of 🎆 Hanabi prior to v0.11,
unfortunately you will need to convert the assets one way or the other.
One approach is to manually edit them,
as the changes are generally mechanical and trivial.
In case you have a large pool of serialized assets,
a conversion tool might be needed,
which can be written by importing `bevy_hanabi` twice with different versions,
and using the Rust in-memory representation as an intermediate storage
before re-serializing with the new version.

More breaking changes on the serialization format are expected
in the next few 🎆 Hanabi releases,
as we attempt to quickly stabilize on a format
to eventually prevent this kind of breaking changes in the future.

// OLD v0.10

```ron
module: (
    expressions: [
        Literal((
            value: Vector((
                vector_type: (
                    elem_type: Float,
                    count: 3,
                ),
                storage: (1067030938, 3227307213, 1118770935, 0),
            )),
        )),
        Literal((
            value: Vector((
                vector_type: (
                    elem_type: Bool,
                    count: 2,
                ),
                storage: (0, 4294967295, 0, 0),
            )),
        )),
        Binary(
            op: Add,
            left: (
                index: 2,
            ),
            right: (
                index: 1,
            ),
        ),
    ],
),
```

// NEW v0.11

```ron
module: (
    expressions: [
        Literal(Vector(Vec3((1.2, -3.45, 87.54485)))),
        Literal(Vector(BVec2((false, true)))),
        Binary(
            op: Add,
            left: 2,
            right: 1,
        ),
    ],
    properties: [
        (
            name: "my_prop",
            default_value: Vector(Vec3((1.2, -2.3, 55.32))),
        ),  
    ]
)
```

## Strong handle for `CompiledParticleEffect`

The `CompiledParticleEffect` component used to store a weak handle
to the related `EffectAsset`.
This was designed to prevent keeping the asset loaded forever,
in case the user did want to unload it (unused asset).
However, this also means that nothing was keeping the asset loaded
while the various 🎆 Hanabi systems were using it,
leading to rare issues were the asset was unloaded,
but 🎆 Hanabi did not detect it fast enough and enqueued a draw for it,
which later panicked.

The `CompiledParticleEffect` now stores a strong handle to the asset,
meaning it owns one reference to it, like the `ParticleEffect` does.
However, because `CompiledParticleEffect` is kept automatically
kept in sync with the associated `ParticleEffect`,
changing the handle of the latter also changes the handle of the former.
So there should be not issue unloading the asset,
except for a potential one frame delay.

If your application depended on this weak reference somehow,
you might need to revisit your asset reference counting logic.

## Depth write for alpha masked effect

Particle effects using the alpha mask pass (`EffectAsset::alpha_mode == AlphaMode::Mask`)
now always render to the depth buffer. This is not configurable.
This should be the best option for everyone,
since alpha masked primitives are essentially opaque.
This also prevents particles from flickering,
as the rendering order now makes use of the depth buffer,
so became deterministic.
If this causes an issue for your use case, please open a bug on GitHub
describing your use case.

Note that effects using the transparent pass still don't write to the depth buffer,
like all transparent primitives in Bevy.
This means they still suffer from flickering,
since particles are not sorted by distance to camera when rendered.
Particle sorting is a planned feature.
