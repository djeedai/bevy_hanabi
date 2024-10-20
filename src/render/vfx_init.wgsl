#import bevy_hanabi::vfx_common::{
    EventBuffer, InitIndirectDispatch, IndirectBuffer, ParticleGroup, RenderEffectMetadata, RenderGroupIndirect, SimParams, Spawner,
    seed, tau, pcg_hash, to_float01, frand, frand2, frand3, frand4,
    rand_uniform_f, rand_uniform_vec2, rand_uniform_vec3, rand_uniform_vec4,
    rand_normal_f, rand_normal_vec2, rand_normal_vec3, rand_normal_vec4, proj
}

struct Particle {
{{PARTICLE_ATTRIBUTES}}
}

struct ParticleBuffer {
    particles: array<Particle>,
}

{{PARENT_PARTICLES}}

{{PROPERTIES}}

@group(0) @binding(0) var<uniform> sim_params : SimParams;
@group(1) @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer;
@group(1) @binding(1) var<storage, read_write> indirect_buffer : IndirectBuffer;
@group(1) @binding(2) var<storage, read> particle_groups : array<ParticleGroup>;
{{PROPERTIES_BINDING}}
#ifdef USE_GPU_SPAWN_EVENTS
@group(1) @binding(4) var<storage, read_write> event_buffer : EventBuffer;
#endif
{{PARENT_PARTICLE_BINDING}}
@group(2) @binding(0) var<storage, read_write> spawner : Spawner; // NOTE - same group as update  // FIXME - this should be read-only
@group(3) @binding(0) var<storage, read_write> render_effect_indirect : RenderEffectMetadata;
@group(3) @binding(1) var<storage, read_write> render_group_indirect : RenderGroupIndirect;

{{INIT_EXTRA}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;

    // Cap to max number of dead particles, copied from dead_count at the end of the
    // previous iteration, and constant during this pass (unlike dead_count).
    if (thread_index >= render_effect_indirect.max_spawn) {
        return;
    }

    // Cap to the actual number of spawning requested by CPU + GPU, since compute shaders run
    // in workgroup_size(64) so more threads than needed are launched (rounded up to 64).
#ifdef USE_GPU_SPAWN_EVENTS
    let event_index = thread_index;
    if (event_index >= render_effect_indirect.event_count) {
        return;
    }
#else
    let spawn_count = u32(spawner.spawn);
    if (thread_index >= spawn_count) {
        return;
    }
#endif

    // Recycle a dead particle from the first group
    let base_index = particle_groups[0].effect_particle_offset;
    let dead_index = atomicSub(&render_group_indirect.dead_count, 1u) - 1u;
    let index = indirect_buffer.indices[3u * (base_index + dead_index) + 2u];

    // Update PRNG seed
    seed = pcg_hash(index ^ spawner.seed);

#ifndef USE_GPU_SPAWN_EVENTS
    // Spawner transform
    let transform = transpose(
        mat4x4(
            spawner.transform[0],
            spawner.transform[1],
            spawner.transform[2],
            vec4<f32>(0.0, 0.0, 0.0, 1.0)
        )
    );
#endif

    // Initialize new particle
    var particle = Particle();
#ifdef USE_GPU_SPAWN_EVENTS
    let parent_index = event_buffer.spawn_events[event_index].particle_index;
    let parent_particle = parent_particle_buffer.particles[parent_index];
#endif
    {{INIT_CODE}}

    // Only add emitter's transform to CPU-spawned particles. GPU-spawned particles
    // don't have a CPU spawner to read from.
#ifndef USE_GPU_SPAWN_EVENTS
    {{SIMULATION_SPACE_TRANSFORM_PARTICLE}}
#endif

    // Count as alive
    atomicAdd(&render_group_indirect.alive_count, 1u);

    // Always write into ping, read from pong
    let ping = render_effect_indirect.ping;

    // Add to alive list
    let indirect_index = atomicAdd(&render_group_indirect.instance_count, 1u);
    indirect_buffer.indices[3u * (base_index + indirect_index) + ping] = index;

    // Write back spawned particle
    particle_buffer.particles[index] = particle;
}
