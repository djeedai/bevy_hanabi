# Authoring

Hanabi offers multiple effect authoring paths:

- Developers can create particle effect assets in code, and instanciate them into the ECS. This includes authoring effects at runtime while the application is running. This is the lower level path, which is more powerful but less user friendly.
- Artists can author complex VFX effects visually through a node graph, and preview them in realtime, using [Hanabi Workshop](https://github.com/djeedai/hanabi-workshop/), the official Editor for Hanabi. Those effects are authored ahead of time, and serialized to disk. They are generally not edited at runtime (although Hanabi supports some minor tweakings if needed). This is the recommended approach for most use cases.

Both paths are based on the same concepts detailed below, and sometimes exposed differently to the user.

## Effects and emitters

Effects are composed of multiple parts:

1. The _spawner_ is responsible for the logic to spawn new particles. Hanabi supports both CPU spawners, controlling the logic from Rust code, and GPU spawners, which emit new particles based on events emitted by other particles.
2. The _modifiers_ are building blocks to control the behavior of particles. Modifiers can be used in 3 different contexts:
   - During the _init pass_, when the spawner has instructed to spawn one or more new particles, modifiers are used to initialize those new particles.
   - During the _update pass_, modifiers control the behavior of particles each frame.
   - During the _render pass_, modifiers are used to turn particles into pixels on screen.
3. _Expressions_ form a micro-language used to customize modifiers (inputs), and therefore the behavior of particles. Hanabi, like Bevy, uses WGSL as its GPU shader language, and most Hanabi expressions map 1:1 to WGSL functions or features. Some extra Hanabi specific concepts are also modeled through expressions, like the simulation delta time since last frame.

The spawner and the set of modifiers, together with a few other settings, form a single _emitter_. A typical effect has a single emitter. However, Hanabi also supports more complex effects involving multiple emitters, often organized in parent-child relationship. A typical use case is a primary emitter producing a fast moving trail, which on death triggers a second emitter to model an explosion at the position where the primary emitter's particle died. This is how a typical firework effect is authored.

> [!WARNING]
> **Emitter vs. Effect** : As of version 0.19, Hanabi doesn't have any _serializable_ concept of a full effect with multiple emitters and parent-child relationships. Those advanced effects can be edited in code, but cannot be serialized; only single-emitter effects are serializable. And semantically, the term _effect_ is used interchangeably in code and docs to denote a single effect asset (single emitter), or a full multi-emitter VFX composition with parent-child relationship. The latter is the preferred semantic for _effect_, while the former is best referred to as an _emitter_. In this book, we use that strict semantic to avoid any confusion.

Within a single emitter definition, particles all share a common structure, composed of one or more _attributes_. Hanabi provides many attributes to compose a particle, like a position, a velocity, some color, and many more. Attributes are stored per particle; a particle nothing but an ordered set of attributes, in the same way that a Bevy entity is nothing more than a set of components. Each available attribute can be present inside a particle at most once. In general, the necessary attributes are automatically derived from the modifiers and other emitter settings, and their layout (order) automatically computed to minimize padding and avoid GPU memory waste.

The most common particle attributes are:

| Attribute | Description |
|-----------|-------------|
| POSITION  | The 3D position of the particle, either in global world frame or in the emitter's local frame. |
| VELOCITY  | The velocity (speed vector) of the particle. |
| AGE       | The particle age. Automatically incremented each frame during the Update pass. Used in conjunction with LIFETIME. |
| LIFETIME  | The lifetime (maximum age) of the particle. Particles whose AGE >= LIFETIME are automatically marked as dead, and recycled. |

For optimal performance, each emitter defines upfront a _capacity_, which is the maximum number of particles which can concurrently be _alive_ (actively simulated). This allows pre-allocating a fixed-size GPU buffer and other data structures, and is an ubiquitous design present in most particle systems. Authors should always take care to define an emitter's capacity to balance two opposing contraints:

- A **larger** capacity consumes more GPU memory, even when particles are not visible on screen. It can also lead to degraded performance, although Hanabi tries hard to minimize the performance cost of unused (_dead_) particles, notably by using indirect GPU dispatch based on the exact number of alive particles for each pass, and not the total capacity.
- A **smaller** capacity increases the chance that the spawner cannot allocate any new particle because all buffer entries are already in use, which typically "breaks" the effect and results in visual artifacts ("missing particles").

The recommended capacity is a small over-allocation over the maximum expected number of particles concurrently alive.

## Particle lifecycle

A particle follows the lifecycle:

Spawn
  : A new particle is allocated from the emitter's particle buffer, if the buffer's capacity is not reached yet.

Init
  : Hanabi runs a custom compute shader generated from the init modifiers assembled by the author. The init modifiers is in charge of initializing the particle's attributes. For example, set an initial position or velocity. This shader runs once per spawned particle only.

Update
  : Each frame, while the particle is _alive_, Hanabi runs a custom compute shader generated from the update modifiers and other settings (like the simulation space or conditions). This shader updates the particle; most commonly it integrates its motion to make it move. If the particle _dies_ (is not _alive_ anymore at the end of the update pass), the shader also recycles it, to allow more particles to spawn.

Render
  : After update, if the particle is still alive, it's converted to a mesh and rendered on screen. Hanabi uses a custom graphics shader, generated from render modifiers and other settings. Both quads (billboards) and full 3D meshes are supported.

A particle becomes _alive_ when allocated by Hanabi, based on the logic of the spawner. This is always automatic; the user cannot manually allocate particles. Conversely, a particle becomes _dead_ when marked as such inside the Update pass. In that case, both automated mechanisms (like particle age-based reaping) and manual ones (a modifier explicitly marking the particle as dead) are available. A typical example of manual death is a modifier killing all particles which leave a given shape (_e.g._, an AABB).

## Simulation

Particles are simulated during the Update pass, by dispatching a compute shader on GPU. Inside that shader, both fixed-function actions (_e.g._ motion integration) and custom user actions (from modifiers) are executed.

### Simulation Time

The simulation time is controlled from CPU by [`Time<EffectSimulation>`]. That time counter derives from Bevy's own [`Time<VirtualTime>`], which is the default time most ECS systems use. This means that by default the Hanabi simulation is affected by any pause or speed change of the Bevy time. This also means that simulation is tied to the overall Bevy frame time, which can vary frame-to-frame. This is general acceptable, if the Bevy frame time is roughly constant (stable FPS), and the effects are not too sensitive to small time differences (FPS remains high enough). However, this delta time can under extreme circumstances --- notably at app startup --- greatly increase, and is generally uncapped. VFX authors using Hanabi should consider whether the game on its whole benefits from using a fixed timestep, or instead stay in sync with the Bevy renderer's time.

The various times provided by Bevy and Hanabi (real, virtual, Hanabi's simulation) are accessible read-only through built-in expressions, to allow effects to customize their behavior based on those.

### Motion Integration

The simulation delta time `dt` since last simulation frame is used for _motion integration_, that is the calculation of the new particle position based on its previous position and its current velocity:

```txt
new_pos = old_pos + velocity * dt
```

Motion integration can be configured to run either before or after all modifiers. By default it runs before.

### Ageing and Reaping

By default, if an emitter defines a particle type containing the `AGE` and `LIFETIME` attributes, then Hanabi automatically activates _ageing_ and _reaping_.

Ageing
  : Each frame, during the Update pass, Hanabi increments each particle's `AGE` attribute value by the simulation delta time `dt`.

Reaping
  : Each frame, at the end of the Update pass, Hanabi compares each particle's `AGE` and `LIFETIME` attributes; if the age is greater or equal to the particle lifetime, the particle is automatically marked as dead (it's deallocated).

> [!NOTE]
> Currently, Hanabi does **NOT** automatically add the `AGE` or `LIFETIME` attribute to a particle layout that contains only the other attribute. This means that Reaping is only active if the effect author has added _both_ attributes explicitly. The simplest way to do so is to add a [`SetAttributeModifier`](https://docs.rs/bevy_hanabi/latest/bevy_hanabi/modifier/attr/struct.SetAttributeModifier.html) to the Init pass of the emitter.

## Modifiers

Modifiers are building blocks influencing the behavior of particles. Within each context (Init, Update, Render), an ordered list of modifiers is applied to modify the attributes of the particle being processed. Some low-level modifiers like [`SetAttributeModifier`](https://docs.rs/bevy_hanabi/latest/bevy_hanabi/modifier/attr/struct.SetAttributeModifier.html) simply assign a value to one attribute. Other, higher-level ones offer more sophisticated attribute mutations; for example the [`SetPositionSphereModifier`](https://docs.rs/bevy_hanabi/latest/bevy_hanabi/modifier/position/struct.SetPositionSphereModifier.html) randomly position the particle on or inside a sphere of given center and radius.

There are several categories of modifiers. Some are generic enough to be useable in all contexts, while others are specific to a given context.

### Acceleration

Acceleration modifiers change the `VELOCITY` of a particle to simulate an acceleration.

#### `AccelModifier`

Similar to motion integration, the `AccelModifier` updates the particle's `VELOCITY` to simulate acceleration.

```txt
particle.velocity += acceleration * dt;
```

It can be used to simulate gravity, by assigning to its acceleration input an expressions resolving to a constant value oriented downward.

```rust
let accel = AccelModifier::new(module.lit(Vec3::Y * -9.81));
```

#### `RadialAccelModifier`

The `RadialAccelModifier` produces a "radial" acceleration oriented alongside the radius of a sphere centered at the modifier's `origin` expression and passing through the current particle position. Said otherwise, the radial acceleration is oriented alongside the line going from the particle to the `origin` position. If the `origin` is constant, this effectively "pulls" the particles toward that point, or push them away if the `acceleration` expression is negative.

#### `TangentAccelModifier`

The `TangentAccelModifier` produces a "tangent" acceleration oriented alongside the tangent to a sphere centered at the modifier's `origin` expression and passing through the current particle position. Said otherwise, the tangent acceleration is oriented perpendicular the line going from the particle to the `origin` position. If the `origin` is constant, this effectively makes the particles rotate around the `origin`. The tangent direction is calculated as the cross product of the rotation plane `axis` expression, and the direction from the modifier `origin` to the particle position.

### Attribute

#### `SetAttributeModifier`

The `SetAttributeModifier` is the most fundamental modifier of Hanabi. It simply sets the value of a given attribute. This is the 

#### `InheritAttributeModifier`

## Properties

Because Hanabi effects run entirely on GPU, they typically cannot be controlled from CPU (Rust code), aside from the logic of CPU spawners. Often, games need their VFX to react to the player actions or other events. To provide some form of runtime control during the GPU simulation, you can use _properties_.

Properties are variables shared by the CPU and GPU. They can be assigned on CPU from the game, and read back from GPU inside the various lifecycle passes. For example, you can use a property to make particles move toward a point of interest, or change the colors of all newly spawned particles based on the state of the game. Each time a property changes value, it's re-uploaded from CPU to GPU; therefore, there's a non-zero cost --- in GPU bandwidth notably --- to using properties, and you should avoid using them to design/tweak an effect during authoring only. They're fundamentally a runtime-tweaking feature.

Each emitter can define zero or more properties. Properties are global to the emitter, and not associated with any specific particule. To store a per-particle value, use an _attribute_ instead. Modifiers can read the value of a property as an expression, and use it to customize the input of a modifier. For example, to change the color of a newly spawned particle based on the game state, create a new color property, update it from the game whenever necessary, and in the Init pass read it back to initialize a per-particle color attribute. Note that in this example, once the per-particle attribute is initialized, the color is stored inside the particle itself (in the attribute), and is not linked to the value of the property anymore. So subsequent property value changes do NOT modify the particle color after init. This is useful if you have for example a single effect emitting particles based on the surface material of a collision point each time an in-game collision is reported by the physics system.

In code, properties are controlled via the [`EffectProperties`](https://docs.rs/bevy_hanabi/latest/bevy_hanabi/properties/struct.EffectProperties.html) component.
