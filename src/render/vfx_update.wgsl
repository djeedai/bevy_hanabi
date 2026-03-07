#import bevy_hanabi::vfx_common::{
    ChildInfo, ChildInfoBuffer, EventBuffer, DispatchIndirectArgs, IndirectBuffer,
    EffectMetadata, RenderGroupIndirect, SimParams, Spawner, DrawIndexedIndirectArgs, BatchInfo,
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

/// Location of an effect in a slab.
struct EffectLocation {
    /// Index of the effect in the global list of effects.
    effect_index: u32,
    /// Base particle index, that is index in the slab of the first particle for this instance.
    base_particle: u32,
    /// Index of this particle relative to its effect. Note that if there's an indirection
    /// buffer then this is the linear index in [0:N[ of the particle to update, before the indirection.
    update_index: u32,
}

/// Find the index of an effect from the index of a particle.
///
/// This uses a binary search on the slab_offset field of the spawners array, which
/// represents a prefix sum of the particle count per effect (for previous effects;
/// the value is actually the base particle so the first entry is always 0).
///
/// Requirements:
/// - var<storage, read> batch_info : BatchInfo
/// - var<storage, read> prefix_sum : array<u32>
fn find_location_from_particle(slab_particle_index: u32) -> EffectLocation {
    var lo = batch_info.prefix_sum_offset;
    var hi = lo + batch_info.prefix_sum_count;
    var num_iter = 0;  // avoid deadlocking the GPU by capping the iteration count
    while (lo < hi) {
        let mid = (hi + lo) >> 1u;
        let base_particle = prefix_sum[mid];
        if (slab_particle_index >= base_particle) {
            lo = mid + 1u;
        } else if (slab_particle_index < base_particle) {
            hi = mid;
        }
        num_iter += 1;
        if (num_iter >= 100) {
            return EffectLocation(0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu);
        }
    }
    let base_particle = batch_info.base_particle + prefix_sum[lo - 1u];
    let effect_index = lo - 1u - batch_info.prefix_sum_offset;
    let update_index = slab_particle_index - base_particle;
    return EffectLocation(effect_index, base_particle, update_index);
}

@group(0) @binding(0) var<uniform> sim_params : SimParams;
@group(0) @binding(1) var<storage, read_write> draw_indirect_buffer : array<DrawIndexedIndirectArgs>;

// "particle" group @1
@group(1) @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer;
@group(1) @binding(1) var<storage, read_write> indirect_buffer : IndirectBuffer;
#ifdef READ_PARENT_PARTICLE
@group(1) @binding(2) var<storage, read> parent_particle_buffer : ParentParticleBuffer;
#endif

// "spawner" group @2
@group(2) @binding(0) var<storage, read> spawners : array<Spawner>;
@group(2) @binding(1) var<storage, read> prefix_sum : array<u32>;
@group(2) @binding(2) var<storage, read> batch_info : BatchInfo;
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

var<private> effect_metadata_index: u32;
var<private> properties_offset: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // Global particle index into the slab, including those particles from other
    // effect instances in the same batch, as well as possibly from other batches.
    // This is rarely useful on its own.
    let slab_particle_index = batch_info.base_particle + global_invocation_id.x;

    // Find the index of the effect this particle is part of.
    let location = find_location_from_particle(slab_particle_index);
    let spawner = &spawners[batch_info.base_effect + location.effect_index];
    effect_metadata_index = (*spawner).effect_metadata_index;
    let base_particle = location.base_particle;

    // Cap at maximum number of alive particles for the current effect
    let effect_metadata = &effect_metadatas[effect_metadata_index];
    if (location.update_index >= (*effect_metadata).max_update) {
        return;
    }
    properties_offset = (*effect_metadata).properties_offset;

    let write_index = effect_metadata.indirect_write_index;
    let read_index = 1u - write_index;

    // This is the actual particle index, from the indirection buffer, addressing an
    // actually alive particle.
    let particle_index = indirect_buffer
        .rows[slab_particle_index]
        .particle_index[read_index];

#ifdef READ_PARENT_PARTICLE
    let parent_base_particle = (*spawner).parent_slab_offset;
#endif

    // Initialize the PRNG seed
    seed = pcg_hash(particle_index ^ (*spawner).seed);

    var particle: Particle = particle_buffer.particles[base_particle + particle_index];
    {{AGE_CODE}}
    {{REAP_CODE}}
    {{UPDATE_CODE}}

    {{WRITEBACK_CODE}}

    // Check if alive
    if (!is_alive) {
        // Save dead index. Note that dead_index is a global slab index.
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
