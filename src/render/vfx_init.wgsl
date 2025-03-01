#import bevy_hanabi::vfx_common::{
    ChildInfo, ChildInfoBuffer, EventBuffer, IndirectDispatch, IndirectBuffer,
    EffectMetadata, RenderGroupIndirect, SimParams, Spawner,
    seed, tau, pcg_hash, to_float01, frand, frand2, frand3, frand4,
    rand_uniform_f, rand_uniform_vec2, rand_uniform_vec3, rand_uniform_vec4,
    rand_normal_f, rand_normal_vec2, rand_normal_vec3, rand_normal_vec4, proj
}

struct Particle {
{{ATTRIBUTES}}
}

struct ParticleBuffer {
    particles: array<Particle>,
}

#ifdef READ_PARENT_PARTICLE

struct ParentParticle {
    {{PARENT_ATTRIBUTES}}
}

struct ParentParticleBuffer {
    particles: array<ParentParticle>,
}

#endif

{{PROPERTIES}}

@group(0) @binding(0) var<uniform> sim_params : SimParams;

// "particle" group @1
@group(1) @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer;
@group(1) @binding(1) var<storage, read_write> indirect_buffer : IndirectBuffer;
#ifdef READ_PARENT_PARTICLE
@group(1) @binding(2) var<storage, read> parent_particle_buffer : ParentParticleBuffer;
#endif

// "spawner" group @2
@group(2) @binding(0) var<storage, read> spawner : Spawner;
{{PROPERTIES_BINDING}}

// "metadata" group @3
@group(3) @binding(0) var<storage, read_write> effect_metadata : EffectMetadata;
#ifdef CONSUME_GPU_SPAWN_EVENTS
@group(3) @binding(1) var<storage, read> child_info_buffer : ChildInfoBuffer;
@group(3) @binding(2) var<storage, read> event_buffer : EventBuffer;
#endif

{{INIT_EXTRA}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;

    // Cap to max number of dead particles, copied from dead_count at the end of the
    // previous iteration, and constant during this pass (unlike dead_count).
    let max_spawn = atomicLoad(&effect_metadata.max_spawn);
    if (thread_index >= max_spawn) {
        return;
    }

    // Cap to the actual number of spawning requested by CPU or GPU, since compute shaders run
    // in workgroup_size(64) so more threads than needed are launched (rounded up to 64).
#ifdef CONSUME_GPU_SPAWN_EVENTS
    let event_index = thread_index;
    let global_child_index = effect_metadata.global_child_index;
    let event_count = child_info_buffer.rows[global_child_index].event_count;
    if (event_index >= u32(event_count)) {
        return;
    }
#else
    // Cap to the actual number of spawning requested by CPU (in the case of
    // spawners) or the number of particles present in the source group (in the
    // case of cloners).
    let spawn_count: u32 = u32(spawner.spawn);
    if (thread_index >= spawn_count) {
        return;
    }
#endif

    // Always write into ping, read from pong
    let write_index = effect_metadata.ping;
    let read_index = 1u - write_index;

    // Recycle a dead particle from the destination group
    let dead_index = atomicSub(&effect_metadata.dead_count, 1u) - 1u;
    let particle_index = indirect_buffer.indices[3u * dead_index + 2u];

    let particle_counter = atomicAdd(&effect_metadata.particle_counter, 1u);

    // Initialize the PRNG seed
    seed = pcg_hash(particle_index ^ spawner.seed);

    // Spawner transform
    let transform = transpose(
        mat4x4(
            spawner.transform[0],
            spawner.transform[1],
            spawner.transform[2],
            vec4<f32>(0.0, 0.0, 0.0, 1.0)
        )
    );

#ifdef READ_PARENT_PARTICLE
    // Fetch parent particle which triggered this spawn
    let parent_particle_index = event_buffer.spawn_events[event_index].particle_index;
    let parent_particle = parent_particle_buffer.particles[parent_particle_index];
#endif

    // Initialize new particle
    var particle = Particle();
    {{INIT_CODE}}
#ifdef ATTRIBUTE_PREV
    particle.prev = 0xffffffffu;
#endif  // ATTRIBUTE_PREV
#ifdef ATTRIBUTE_NEXT
    particle.next = 0xffffffffu;
#endif  // ATTRIBUTE_NEXT

    // Only add emitter's transform to CPU-spawned particles. GPU-spawned particles
    // don't have a CPU spawner to read from (they can read the parent particle
    // transform, which is equivalent).
#ifndef CONSUME_GPU_SPAWN_EVENTS
    {{SIMULATION_SPACE_TRANSFORM_PARTICLE}}
#endif

    // Count as alive
    atomicAdd(&effect_metadata.alive_count, 1u);

    // Add to alive list
    let instance_index = atomicAdd(&effect_metadata.instance_count, 1u);
    indirect_buffer.indices[3u * instance_index + write_index] = particle_index;

    // Write back new particle
    particle_buffer.particles[particle_index] = particle;
}
