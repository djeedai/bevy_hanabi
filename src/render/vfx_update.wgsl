#import bevy_hanabi::vfx_common::{
    IndirectBuffer, ParticleGroup, RenderEffectMetadata, RenderGroupIndirect, SimParams, Spawner,
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

{{PROPERTIES}}

@group(0) @binding(0) var<uniform> sim_params : SimParams;
@group(1) @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer;
@group(1) @binding(1) var<storage, read_write> indirect_buffer : IndirectBuffer;
@group(1) @binding(2) var<storage, read> particle_groups : array<ParticleGroup>;
{{PROPERTIES_BINDING}}
@group(2) @binding(0) var<storage, read_write> spawner : Spawner; // NOTE - same group as init
@group(3) @binding(0) var<storage, read_write> render_effect_indirect : RenderEffectMetadata;
@group(3) @binding(1) var<storage, read_write> render_group_indirect : array<RenderGroupIndirect>;

{{UPDATE_EXTRA}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;

    // Cap at maximum number of particles.
    // FIXME - This is probably useless given below cap
    let max_particles : u32 = particle_groups[{{GROUP_INDEX}}].capacity;
    if (thread_index >= max_particles) {
        return;
    }

    // Cap at maximum number of alive particles.
    if (thread_index >= render_group_indirect[{{GROUP_INDEX}}].max_update) {
        return;
    }

    // Always write into ping, read from pong
    let ping = render_effect_indirect.ping;
    let pong = 1u - ping;

    let effect_particle_offset = particle_groups[{{GROUP_INDEX}}].effect_particle_offset;
    let base_index = effect_particle_offset + particle_groups[{{GROUP_INDEX}}].indirect_index;
    let index = indirect_buffer.indices[3u * (base_index + thread_index) + pong];

    var particle: Particle = particle_buffer.particles[index];

    // Update PRNG seed
    seed = pcg_hash(index ^ spawner.seed);

    {{AGE_CODE}}
    {{UPDATE_CODE}}
    {{REAP_CODE}}

    particle_buffer.particles[index] = particle;

    // Check if alive
    if (!is_alive) {
        // Save dead index
        let dead_index = atomicAdd(&render_group_indirect[{{GROUP_INDEX}}].dead_count, 1u);
        indirect_buffer.indices[3u * (base_index + dead_index) + 2u] = index;
        // Also increment copy of dead count, which was updated in dispatch indirect
        // pass just before, and need to remain correct after this pass
        atomicAdd(&render_effect_indirect.max_spawn, 1u);
        atomicSub(&render_group_indirect[{{GROUP_INDEX}}].alive_count, 1u);
    } else {
        // Increment alive particle count and write indirection index for later rendering
        let indirect_index = atomicAdd(&render_group_indirect[{{GROUP_INDEX}}].instance_count, 1u);
        indirect_buffer.indices[3u * (base_index + indirect_index) + ping] = index;
    }
}
