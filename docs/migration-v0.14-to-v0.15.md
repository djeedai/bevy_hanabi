# Migration Guide v0.14 -> v0.15

🎆 Hanabi v0.15 contains a few major API breaking changes.

This guide helps the user migrate from v0.14 to v0.15 of 🎆 Hanabi.
Users are encouraged to also read the [`CHANGELOG`](../CHANGELOG.md)
for an exhaustive list of all changes.

## Hierarchical effects

🎆 Hanabi v0.15 supports a new feature called _hierarchical effects_,
whereby an effect can be parented to another effect.
Parenting an effect unlocks new features for effect authoring:

- A child effect has read-only access to the particles of its parent,
  and can therefore inherit _e.g._ their position via the new `InheritAttributeModifier`.
- A child effect's particles can be spawned dynamically on GPU by its parent,
  allowing to "chain" effects.
  For example, the parent can spawn a particle in a child effect
  when one of its own particles dies.

The effect parent/child hierarchy forms a tree (no cycles).
An effect can declare its parent with the `EffectParent` component,
and can have a single parent only.
A parent effect can have multiple children,
and can itself be a child of a third effect.

## GPU spawn events

Parent effects (see [Hierarchical effects](#hierarchical-effects)) can emit _GPU spawn events_,
which are GPU-side "events" to spawn particles into one of their child effects.
With GPU spawn events, child events can be entirely GPU driven,
and react to the behavior of their parent.
For example, a GPU spawn event can be emitted when a particle dies.
This allows _e.g._ a child event to emit an explosion of particles,
which visually looks in direct relation with the death of the parent's particle.

Particles spawned via GPU spawn events often inherit one or more attribute from their parent,
via the new `InheritAttributeModifier`.
This is made possible by the fact that a GPU spawn event contains the ID of the parent particle,
which allows reading its attributes from the child effect's init pass.

To declare a parent effect emitting GPU spawn events, use:

```rust
let parent_effect = EffectAsset::new(32, spawner, module)
    .update(EmitSpawnEventModifier {
        condition: EventEmitCondition::OnDie,
        count: 45,
        child_index: 0,
    });
let parent_handle = effects.add(parent_effect);
let parent_entity = commands.spawn(ParticleEffect::new(parent_handle)).id();
```

The child index determines which child effect will consume those emitted events,
since a parent effect can have multiple children.

For the child effect, which consumes those GPU spawn events, use:

```rust
let child_effect = EffectAsset::new(250, unused_spawner, module)
    // On spawn, copy the POSITION of the particle which emitted the GPU event
    .init(InheritAttributeModifier::new(Attribute::POSITION));
let child_handle = effects.add(child_effect);
commands.spawn((
    ParticleEffect::new(child_handle),
    EffectParent(parent_entity),
));
```

See the updated `firework.rs` example for a full-featured demo.

## New group-less ribbon and trail implementation

Previously in 🎆 Hanabi v0.14, ribbons and trails made use of _groups_,
which allowed partitioning a particle buffer into sub-buffers, one per group,
and spawn particles from one group into the other.

If this sounds familiar, this is because that feature is nearly identical to GPU spawn events,
at least conceptually.
The API and implementation however were extremely confusing for the user,
with things like initializing a group particle from the Update pass instead of the Init one,
and the inability to use init shaders for those particles.

The new ribbon and trail implementation gets rid of all those restrictions,
and restores the common patterns established for all effects:

- particles are initialized on spawn in the init pass, via init modifiers.
- particles are updated every frame while alive in the update pass, via update modifiers.

The entire group feature has been removed.
Instead, a ribbon or trail is now defined by the `Attribute::RIBBON_ID` assigned to each particle.
All particles with a same ribbon ID are part of the same ribbon.
There's no other meaning to that value, so you can use any value that makes sense.
At runtime, after the update pass, all particles are sorted by their `RIBBON_ID` to group them into ribbons,
and inside a given ribbon (same ribbon ID) the particles are sorted by their age.
This sorting not only solves the issue of grouping particles without complex buffer management,
but also gets rid of the annoying edge cases where a particle in a middle of a ribbon would die,
leaving a gap in it, with forced the previous implementation to constraint to lifetime of particles
via an external mechanism instead of using the `Attribute::LIFETIME`.

If you're familiar with trails and ribbons in 🎆 Hanabi v0.14 and earlier,
you may remember about the `Attribute::PREV` and `Attribute::NEXT`.
Those attributes were used to chain together particles into trails and ribbons,
forming a linked list.
Not only did they use a lot of storage space (twice as much as `RIBBON_ID`),
the operations on the linked list were difficult to perform atomically,
and have led to several bugs in the past.
With 🎆 Hanabi v0.15, those attributes are not used anymore, and are soft-deprected.
You can continue to use them for any other purpose,
but they do not have anymore a built-in effect.

To create a trail or ribbon, simply assign the `Attribute::RIBBON_ID`:

```rust
// Example: single trail/ribbon effect

let init_ribbon_id = SetAttributeModifier {
    attribute: Attribute::RIBBON_ID,
    // We use a constant '0' for all particles; any value works.
    value: writer.lit(0u32).expr(),
};
```

When using multi-trail / multi-ribbon, each trail/ribbon needs a unique ID.
You can calculate that value from the parent effect, store it,
and read it back in the child effect.

```rust
// Example: multi-trail/multi-ribbon

// In the parent effect:
let parent_init_ribbon_id = SetAttributeModifier::new(
    Attribute::U32_0,
    // Store a unique value per parent particle, used as ribbon ID in children
    writer.attr(Attribute::PARTICLE_COUNTER).expr(),
);

// In the child effect:
let child_init_ribbon_id = SetAttributeModifier {
    attribute: Attribute::RIBBON_ID,
    // Read back the unique value from the parent particle
    value: writer.parent_attr(Attribute::U32_0).expr(),
};
```

See the updated `ribbon.rs` example for a full-featured demo of a single ribbon,
and the `worms.rs` and `firework.rs` examples
for an example of combining hierarchical effects and ribbons,
and use multiple ribbons in the same effect.

## Bundle removal

Following Bevy's own deprecation of the bundle mechanism, `ParticleEffectBundle` has been removed.
Use `ParticleEffect` directly instead, which now supports the ECS `#[require()]` mechanism,
and will automatically add the mandatory components `CompiledParticleEffect`, `Visibility`, and `Transform`.

## Deterministic randomness

Effects using GPU expressions with randomness, like the built-in expression obtained from `ExprWriter::rand()`,
now use an explicit PRNG seed set stored in `EffectAsset::prng_seed`.
This value is `0` by default, meaning the effect will produce the same result each application run.
This ensures the effect authored is played back deterministically, which gives artistic control,
and is also useful to generate deterministic repro examples of bugs for debugging.

If you want true randomness (old behavior), simply assign this new field to a random value yourself,
for example:

```rust
EffectAsset { 
    prng_seed: rand::random::<u32>(),
    // [...]
}
```
