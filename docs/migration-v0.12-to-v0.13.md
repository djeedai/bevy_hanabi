# Migration Guide v0.12 -> v0.13

🎆 Hanabi v0.13 contains a few major API breaking changes.

This guide helps the user migrate from v0.12 to v0.13 of 🎆 Hanabi.
Users are encouraged to also read the [`CHANGELOG`](../CHANGELOG.md)
for an exhaustive list of all changes.

## Optional serialization and deserialization

Serialization and deserialization have been made optional
under a new feature flag `serde`.
This makes the `typetag` dependency also optional.

If your project relies on serialization and deserialization of 🎆 Hanabi types,
it should work by default out of the box,
because the new `serde` feature flag is active by default.

However, **if you disabled default features,
simply add the `serde` flag to re-enable serialization and deserialization**
of 🎆 Hanabi types.

```diff
- bevy_hanabi = { version = "0.12", default-features = false, features = [...] }
+ bevy_hanabi = { version = "0.13", default-features = false, features = ["serde", ...] }
```

## Trails and ribbons API change

The API for trails and ribbons changed,
primarily to fix a race condition occurring by design of the old API.
This also makes the new API somewhat more understandable.

In the previous version, the API for trails and ribbons
required the user to manually add the `CloneModifier` to spawn particles,
and the `RibbonModifier` to tie them together in a single ribbon.
However, the API had many implicit limitations:

- The particles forming the trail or ribbon were cloned in the Update pass,
  unlike regular particles spawned in the Init pass.
- Cloned particles were exact copies of their source particle,
  with no way for the user to initializer them
  (we didn't run any init modifier on them).
- The lifetime of trail and ribbon particles was inherited from the cloned particle,
  which is not always desirable.
- The age of trail and ribbon particles was always initialized to zero.

The new API in v0.13 fixes those issues,
by moving the particle cloning step into a separate pass similar to the Init one,
and running like it before the Update pass.

We introduce the concept of _initializer_,
to unify the existing _spawner_ (spawn from CPU),
and the new _cloner_ (clone existing GPU particle).
To that end, **the `EffectSpawner` component was wrapped into a new `EffectInitializers`**,
which takes a `Vec<EffectInitializer>` to define how each group initializes its particles.
A **new `EffectCloner`** holds the cloner configuration,
similar to how `EffectSpawner` holds the spawner configuration.

// OLD v0.12

```rust
// Spawn from CPU:
let spawner = EffectSpawner::new(asset);
```

// NEW v0.13

```rust
// Spawn from CPU:
let initializer: EffectInitializer = EffectSpawner::new(asset).into();

// Clone from existing GPU particle:
let initializer: EffectInitializer = EffectCloner::new(0, 3.0).into();
```

To simplify things further, `EffectAsset` adds **2 new helper methods**:

- `with_trails()` adds a new group which clones from the given source group
  at regular fixed interval.
- `with_ribbons()` does the same, but also ties together cloned particles
  to form a continuous chain ("ribbon").

The group creation relies heavily on those new API functions,
and moves away from indirectly declaring upfront an array of groups
via the array of capacities.
Instead, **`EffectAsset::new()` reverts to taking a single capacity for the default group #0**.

// OLD v0.12

```rust
let asset = EffectAsset::new(vec![256, RIBBONS_CAPACITY], ...)
    .update_groups(clone_modifier, ParticleGroupSet::single(0))
    .render(RibbonModifier);
```

// NEW v0.13

```rust
let asset = EffectAsset::new(256, ...)
    .with_ribbons(RIBBONS_CAPACITY, 1.0 / RIBBONS_SPAWN_RATE, RIBBONS_LIFETIME, 0);
```

The lifetime of trails and ribbons also changes semantic.
Previously, the age of cloned particles was set to zero,
and the lifetime inherited from the cloned particle.
This was both inflexible, as you couldn't set a lifetime easily,
and caused issues with particles in the middle of a ribbon dying
before the particles following them, causing gaps and rendering artifacts.
To fix this, **the new API enforces a per-ribbon lifetime**,
specified via the `with_ribbons()` helper.
Any lifetime set with modifiers is ignored and overwritten.
This new approach is more restrictive than previously,
as you cannot set a per-ribbon lifetime, only a global one for the entire effect.
But it guarantees a last cloned / first dead (LIFO) order for ribbon particles,
which solves the previously mentioned render artifacts and gaps.

Because of all the above, **the `CloneModifier` has been removed**.
To create trails or ribbons, use the helper functions mentioned above.
Similarly, **the `RibbonModifier` was also removed**. Use `with_ribbons()` instead.
Note that you can only create a single ribbon group per effect,
because the ribbon particles are chained via a linked list,
and there's only one set of attributes per particle to do so.
This was already the case in the previous version, but is made more explicit now.

As a result of those changes, init modifiers can now run on trail and ribbon particles.
This makes it cleaner and easier to initialize cloned particles,
in the same way particles spawned from GPU are initialized.
All modifiers are supported, except those touching the age, lifetime,
and previous and next pointer attributes (for ribbon linked list).
