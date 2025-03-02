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
#ifdef EMITS_GPU_SPAWN_EVENTS
{{EMIT_EVENT_BUFFER_BINDINGS}}
#endif

{{UPDATE_EXTRA}}

#ifdef EMITS_GPU_SPAWN_EVENTS
{{EMIT_EVENT_BUFFER_APPEND_FUNCS}}
#endif

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;

    // Cap at maximum number of alive particles.
    if (thread_index >= effect_metadata.max_update) {
        return;
    }

    // Always write into ping, read from pong
    let write_index = effect_metadata.ping;
    let read_index = 1u - write_index;

    let particle_index = indirect_buffer.indices[3u * thread_index + read_index];

    // Initialize the PRNG seed
    seed = pcg_hash(particle_index ^ spawner.seed);

    var particle: Particle = particle_buffer.particles[particle_index];
    {{AGE_CODE}}
    {{REAP_CODE}}
    {{UPDATE_CODE}}

    {{WRITEBACK_CODE}}

    // Check if alive
    if (!is_alive) {
        // Save dead index
        let dead_index = atomicAdd(&effect_metadata.dead_count, 1u);
        indirect_buffer.indices[3u * dead_index + 2u] = particle_index;

        // Also increment copy of dead count, which was updated in dispatch indirect
        // pass just before, and need to remain correct after this pass
        atomicAdd(&effect_metadata.max_spawn, 1u);
        atomicSub(&effect_metadata.alive_count, 1u);
    } else {
        // Increment alive particle count and write indirection index for later rendering
        let indirect_index = atomicAdd(&effect_metadata.instance_count, 1u);
        indirect_buffer.indices[3u * indirect_index + write_index] = particle_index;
    }
}
