#import bevy_hanabi::vfx_common::{
    ChildInfo, ChildInfoBuffer, EventBuffer, DispatchIndirectArgs, IndirectBuffer,
    EffectMetadata, RenderGroupIndirect, SimParams, Spawner, DrawIndexedIndirectArgs,
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
@group(0) @binding(1) var<storage, read_write> draw_indirect_buffer : array<DrawIndexedIndirectArgs>;

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
@group(3) @binding(0) var<storage, read_write> effect_metadatas : array<EffectMetadata>;
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
    let effect_metadata = &effect_metadatas[spawner.effect_metadata_index];
    if (thread_index >= (*effect_metadata).max_update) {
        return;
    }

    let base_particle = spawner.slab_offset;
#ifdef READ_PARENT_PARTICLE
    let parent_base_particle = spawner.parent_slab_offset;
#endif

    let write_index = effect_metadata.indirect_write_index;
    let read_index = 1u - write_index;

    let particle_index = indirect_buffer
        .rows[base_particle + thread_index]
        .particle_index[read_index];

    // Initialize the PRNG seed
    seed = pcg_hash(particle_index ^ spawner.seed);

    var particle: Particle = particle_buffer.particles[base_particle + particle_index];
    {{AGE_CODE}}
    {{REAP_CODE}}
    {{UPDATE_CODE}}

    {{WRITEBACK_CODE}}

    // Check if alive
    if (!is_alive) {
        // Save dead index
        let alive_index = atomicSub(&((*effect_metadata).alive_count), 1u) - 1u;
        indirect_buffer.rows[base_particle + alive_index].dead_index = base_particle + particle_index;

        // DEBUG
        //indirect_buffer.rows[base_particle + alive_index].particle_index[0] = 0xFFFFFFFFu;
        //indirect_buffer.rows[base_particle + alive_index].particle_index[1] = 0xFFFFFFFFu;

        // Also increment copy of dead count, which was updated in dispatch indirect
        // pass just before, and need to remain correct after this pass. We wouldn't have
        // to do that here if we had a per-effect pass between update and the next init.
        atomicAdd(&((*effect_metadata).max_spawn), 1u);
    } else {
        // Increment visible particle count (in the absence of any GPU culling), and write
        // the indirection index for later rendering.
        let indirect_index = atomicAdd(&draw_indirect_buffer[(*effect_metadata).indirect_render_index].instance_count, 1u);
        indirect_buffer.rows[base_particle + indirect_index].particle_index[write_index] = particle_index;
    }
}
