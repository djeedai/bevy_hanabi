#import bevy_hanabi::vfx_common::{
    ChildInfo, ChildInfoBuffer, EventBuffer, DispatchIndirectArgs, IndirectBuffer,
    EffectMetadata, RenderGroupIndirect, SimParams, Spawner, BatchInfo,
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
    let base_particle = prefix_sum[lo - 1u];
    let effect_index = lo - 1u - batch_info.prefix_sum_offset;
    let update_index = slab_particle_index - base_particle;
    return EffectLocation(effect_index, base_particle, update_index);
}

@group(0) @binding(0) var<uniform> sim_params : SimParams;

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
#ifdef CONSUME_GPU_SPAWN_EVENTS
@group(3) @binding(1) var<storage, read> child_info_buffer : ChildInfoBuffer;
@group(3) @binding(2) var<storage, read> event_buffer : EventBuffer;
#endif

{{INIT_EXTRA}}

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

    // Cap to max number of dead particles, copied from (capacity - alive_count) at the end
    // of the previous iteration, and constant during this pass (unlike alive_count).
    let effect_metadata = &effect_metadatas[(*spawner).effect_metadata_index];
    let max_spawn = atomicLoad(&(*effect_metadata).max_spawn);
    if (location.update_index >= max_spawn) {
        return;
    }
    properties_offset = (*effect_metadata).properties_offset;

    // Cap to the actual number of spawning requested by CPU or GPU, since compute shaders run
    // in workgroup_size(64) so more threads than needed are launched (rounded up to 64).
#ifdef CONSUME_GPU_SPAWN_EVENTS
    let event_index = location.update_index;
    let global_child_index = (*effect_metadata).global_child_index;
    let event_count = child_info_buffer.rows[global_child_index].event_count;
    if (event_index >= u32(event_count)) {
        return;
    }
#else
    // Cap to the actual number of spawning requested by CPU (in the case of
    // spawners) or the number of particles present in the source group (in the
    // case of cloners).
    let spawn_count: u32 = u32((*spawner).spawn);
    if (location.update_index >= spawn_count) {
        return;
    }
#endif

    // Count as alive, and recycle a dead particle slot to store the newly spawned particle
    let alive_index = atomicAdd(&(*effect_metadata).alive_count, 1u);
    let slab_particle_dead_index = indirect_buffer.rows[base_particle + alive_index].dead_index;
    let particle_index = slab_particle_dead_index - base_particle;

    // DEBUG
    //indirect_buffer.rows[base_particle + alive_index].dead_index = 0xFFFFFFFFu;

    // Bump the particle counter each time we allocate a particle. This generates a unique
    // particle ID used for various purposes (but not directly by the simulation). We still
    // store it in a variable, because the INIT_CODE might access it.
    let particle_counter = atomicAdd(&(*effect_metadata).particle_counter, 1u);

    // Initialize the PRNG seed
    seed = pcg_hash(particle_index ^ (*spawner).seed);

    // Spawner transform
    let transform = transpose(
        mat4x4(
            (*spawner).transform[0],
            (*spawner).transform[1],
            (*spawner).transform[2],
            vec4<f32>(0.0, 0.0, 0.0, 1.0)
        )
    );

#ifdef READ_PARENT_PARTICLE
    // Fetch parent particle which triggered this spawn
    let parent_base_particle = (*spawner).parent_slab_offset;
    let parent_particle_index = event_buffer.spawn_events[event_index].particle_index;
    let parent_particle = parent_particle_buffer.particles[parent_base_particle + parent_particle_index];
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

    // Append to alive list of indirect buffer.
    let write_index = (*effect_metadata).indirect_write_index;
    indirect_buffer.rows[base_particle + alive_index].particle_index[write_index] = particle_index;

    // Write back new particle
    particle_buffer.particles[base_particle + particle_index] = particle;
}
