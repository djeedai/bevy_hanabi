#import bevy_hanabi::vfx_common::{
    ChildInfo, EventBuffer, InitIndirectDispatch, IndirectBuffer, ParticleGroup,
    RenderEffectMetadata, RenderGroupIndirect, SimParams, Spawner,
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

{{PROPERTIES}}

@group(0) @binding(0) var<uniform> sim_params : SimParams;

@group(1) @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer;
@group(1) @binding(1) var<storage, read_write> indirect_buffer : IndirectBuffer;
@group(1) @binding(2) var<storage, read> particle_groups : array<ParticleGroup>;
{{PROPERTIES_BINDING}}
#ifdef EMITS_GPU_SPAWN_EVENTS
{{CHILDREN_EVENT_BUFFER_BINDINGS}}
#endif

@group(2) @binding(0) var<storage, read_write> spawner : Spawner; // NOTE - same group as init

@group(3) @binding(0) var<storage, read_write> render_effect_indirect : array<RenderEffectMetadata>;
@group(3) @binding(1) var<storage, read_write> render_group_indirect : array<RenderGroupIndirect>;

{{UPDATE_EXTRA}}

#ifdef EMITS_GPU_SPAWN_EVENTS
{{CHILDREN_EVENT_BUFFER_APPEND}}
#endif

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;

    // Cap at maximum number of alive particles.
    if (thread_index >= render_group_indirect[{{GROUP_INDEX}}].max_update) {
        return;
    }

    let effect_index = spawner.effect_index;

    // Always write into ping, read from pong
    let ping = render_effect_indirect[effect_index].ping;
    let pong = 1u - ping;

    let effect_particle_offset = particle_groups[{{GROUP_INDEX}}].effect_particle_offset;
    let base_index = effect_particle_offset + particle_groups[{{GROUP_INDEX}}].indirect_index;
    let index = indirect_buffer.indices[3u * (base_index + thread_index) + pong];

    // Initialize the PRNG seed
    seed = pcg_hash(index ^ spawner.seed);

    var particle: Particle = particle_buffer.particles[index];

    {{AGE_CODE}}
    let was_alive = is_alive;
    {{REAP_CODE}}
    {{UPDATE_CODE}}

    {{WRITEBACK_CODE}}

    // Check if alive
    if (!is_alive) {
        // Unlink dead particle from trail linked list
#ifdef ATTRIBUTE_PREV
#ifdef ATTRIBUTE_NEXT
#ifdef TRAIL
        // We know that no particles behind us (including our previous) are going to be
        // alive after this, due to the strict LIFO lifetime of trail particles.
        // So we can just set the next particle's prev pointer to null.
        let next = particle_buffer.particles[index].next;
        if (next != 0xffffffffu) {
            particle_buffer.particles[next].prev = 0xffffffffu;
        }
#else   // TRAIL
        // Head particle; there's no worry about races here, because the trail particles
        // are all in a different group, which is simulated in a different dispatch.
        let prev = particle_buffer.particles[index].prev;
        if (prev != 0xffffffffu) {
            particle_buffer.particles[prev].next = 0xffffffffu;
        }
#endif  // TRAIL
#endif  // ATTRIBUTE_NEXT
#endif  // ATTRIBUTE_PREV

        // Save dead index
        let dead_index = atomicAdd(&render_group_indirect[{{GROUP_INDEX}}].dead_count, 1u);
        indirect_buffer.indices[3u * (base_index + dead_index) + 2u] = index;

        // Also increment copy of dead count, which was updated in dispatch indirect
        // pass just before, and need to remain correct after this pass
        atomicAdd(&render_group_indirect[{{GROUP_INDEX}}].max_spawn, 1u);
        atomicSub(&render_group_indirect[{{GROUP_INDEX}}].alive_count, 1u);
    } else {
        // Increment alive particle count and write indirection index for later rendering
        let indirect_index = atomicAdd(&render_group_indirect[{{GROUP_INDEX}}].instance_count, 1u);
        indirect_buffer.indices[3u * (base_index + indirect_index) + ping] = index;
    }
}
