# 🎆 Hanabi rendering

## Overview

🎆 Hanabi uses compute shaders to simulate the particle effects and drive their rendering. The GPU work is divided into 4 passes, in order:

1. Init
2. Dispatch
3. Update
4. Render

### Init pass

For each effect instance, an optional init pass (compute shader) spawn new particles and initializes the value of their attributes. This pass only runs if the effect needs to spawn one or more particles this frame. Because the compute shader init code depends on the effect asset,

### Dispatch pass

The dispatch pass (compute shader) primarily handles book-keeping to prepare for the indirect compute dispatch of the update pass and the indirect rendering, as well as other once-per-instance work. Ideally this pass runs once for all effect instances at the same time, with 1 compute thread per effect instance.

### Update pass

For each effect instance, the update pass (compute shader) simulates the particles, updating their attributes based on the shader code generated from the effect asset the instance was created from. For each particle still alive after the simulation update, it increments the alive counter / render instance count.

## Render pass

The render pass contains an indirect draw call per effect instance. A single unit quad mesh is rendered with an indirect instanced draw call, where the number of instance is equal to the number of particles alive, and is used to index into the particle buffer.

## GPU buffers

There's 7 GPU buffers used, defined in `vfx_common.wgsl`:

1. The _particle buffer_ (storage) contains an array of particles. The exact layout of each particle depends on which attributes the particle uses, as defined in the effect asset.
2. The _indirect buffer_ (storage) is a triple buffer of indices. The first two indices are ping-pong indices whose role is swapped each frame. They are used as indirection into the particle buffer, to define which particles are still alive (particle buffer entry used). The last entry contains a free list of dead particles.
3. The _particle group buffer_ (storage) contains the definition of particle groups for each effect instance. A particle group defines a sub-effect inside an effect instance. In general only one group (sub-effect) is present per effect instance, unless particle trails are used (which then form a second group).
4. The _spawner buffer_ (storage) contains all the CPU data uploaded each frame for each effect instance, including the transform of the effect emitter, the number of particles to spawn this frame, or a per-effect-instance seed for the GPU PRNG.
5. The _dispatch indirect buffer_ (storage) contains for each effect instance a dispatch indirect structure (x, y=1, z=1, pong). The first three items are the compute threads to spawn, with y=z=1 always. The last index `pong` is a cache of the same value usually stored in the render indirect buffer, which can't be accessed during indirect rendering because the buffer is already bound for the draw call itself.
6. The _render indirect buffer_, or _render group indirect buffer_, (storage) similarly contains for each effect instance a render indirect structure used to drive the indirect draw call for that instance.
7. The _render effect metadata buffer_ (storage) contains some per-effect-instance data for the first group of each effect, which is currently threated differently from the other group as the init pass only spawns particles into the first group (particles can only be copied from group #0 into another group during simulation, but cannot currently be spawned as a fresh new particle with different attributes).
