# Optimizing

This section details tips and tricks to optimize Hanabi effects for performance and GPU memory.

## Selecting a Capacity

The emitter capacity is the maximum number of concurrently alive particles. This directly determines the size of various GPU buffers, and notably the particle buffer. Minimizing the capacity directly saves GPU memory.

Because most VFX have some randomness component, in general it's not possible to calculate the exact maximum concurrent particle count. Therefore, the emitter capacity has to be estimated. In most cases, the particle LIFETIME if present directly determines the particle's lifetime duration. Combined with the spawner's frequency, you can calculate a good estimate of the capacity.

```txt
C = (spawn rate) * (particle lifetime)
```

For example, a spawn rate of 50 particles/second with a random LIFETIME in [1.0 : 2.0] seconds produces a worse case capacity C of 100 particles. However, because there can be rounding errors and variations in simulation time (if not using a fixed timestep), it's generally recommended to add some 5%~10% margin. Here, a capacity of 110 particles is probably enough.

## Despawn vs. Deactivate

_See also [Prewarming](#prewarming)._

Games commonly make use of VFX for specific events. In-between those events, the effect might be entirely inactive (no particle simulated) for a long time. There are 2 strategies to handle this, both with pros and cons:

Despawn
  : Despawning the [`ParticleEffect`](https://docs.rs/bevy_hanabi/latest/bevy_hanabi/struct.ParticleEffect.html) component destroys the emitter instance and frees up all CPU and GPU resources, including buffers and shaders. This helps with memory pressure for large effects. The downside is that if the custom shaders for that emitter are not otherwise used, they might be freed by Bevy, and would require a new, slow GPU compile. Same goes for shader pipelines and other resources.

Deactivate
  : Deactivating the emitter consists in setting [`EffectSpawner::active`](https://docs.rs/bevy_hanabi/latest/bevy_hanabi/struct.EffectSpawner.html#structfield.active) to `false`. This prevents the spawner from emitting new particles (but doesn't kill existing ones). While inactive, all GPU and CPU resources remain allocated; this means that re-activating the emitter is very cheap.

In general, emitters that need to be available on short notice, like those associated with gameplay events, are best left allocated and deactivated. On the other hand, emitters only used in a particular level can be despawned when the player moves to a different level, because it's generally accepted that changing level provides some time to re-allocated resources (_e.g._ loading screen).

Note that Hanabi always attempts to share resources, and notably shaders and pipelines, between emitter instances. This means that keeping at least one instance of a given [`EffectAsset`](https://docs.rs/bevy_hanabi/latest/bevy_hanabi/struct.EffectAsset.html) instantiated in the form of a (possibly hidden) [`ParticleEffect`](https://docs.rs/bevy_hanabi/latest/bevy_hanabi/struct.ParticleEffect.html) component is enough to ensure shared resources for that emitter are kept alive. This doesn't solve the allocation cost of GPU buffers for _e.g._ particles, which are per-instance, but mitigates at least the (often heavy) cost of GPU shader/pipeline compiling, and therefore can form a mixed startegy striking a good balance between GPU memory usage and respawn cost.

## Prewarming

_See also [Despawn vs. Deactive](#despawn-vs-deactivate)._

Spawning a new emitter instance for the first time generally involves some slow process to compile GPU shaders and pipelines, and allocate other GPU resources. If done just-in-time by the game, this can cause delays in the particle effect appearing on screen. To ensure a newly spawned emitter instance has no delay, you can prewarm the compiling of GPU shaders and pipelines by instanciating at least one [`ParticleEffect`](https://docs.rs/bevy_hanabi/latest/bevy_hanabi/struct.ParticleEffect.html) component, and deactivating it if you don't need it right away. Any new other instance of the _same_ emitter (same [`EffectAsset`](https://docs.rs/bevy_hanabi/latest/bevy_hanabi/struct.EffectAsset.html)) will share the same GPU shaders and pipelines, which are already built, resulting in a much faster emitter creation with no delay.

> [!NOTE]
> In previous versions of Hanabi, this prewarming technique was also used to prevent issues with a spawner starting to emit particles before the GPU resources were ready. This is no longer the case; currently Hanabi waits for the emitter instance to be ready, including all GPU resources, before it starts to tick spawners. See [`CompiledParticleEffect::is_ready()`](https://docs.rs/bevy_hanabi/latest/bevy_hanabi/struct.CompiledParticleEffect.html#method.is_ready).
