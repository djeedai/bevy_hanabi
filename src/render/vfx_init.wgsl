#import bevy_hanabi::vfx_common::{
    IndirectBuffer, ParticleGroup, RenderEffectMetadata, RenderGroupIndirect, SimParams, Spawner,
    seed, tau, pcg_hash, to_float01, frand, frand2, frand3, frand4,
    rand_uniform_f, rand_uniform_vec2, rand_uniform_vec3, rand_uniform_vec4, proj
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
@group(2) @binding(0) var<storage, read_write> spawner : Spawner; // NOTE - same group as update
@group(3) @binding(0) var<storage, read_write> render_effect_indirect : RenderEffectMetadata;
@group(3) @binding(1) var<storage, read_write> render_group_indirect : RenderGroupIndirect;

{{INIT_EXTRA}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    var index = global_invocation_id.x;

    // Cap to max number of dead particles, copied from dead_count at the end of the
    // previous iteration, and constant during this pass (unlike dead_count).
    if (index >= render_effect_indirect.max_spawn) {
        return;
    }

    // Cap to the actual number of spawning requested by CPU, since compute shaders run
    // in workgroup_size(64) so more threads than needed are launched (rounded up to 64).
    let spawn_count : u32 = u32(spawner.spawn);
    if (index >= spawn_count) {
        return;
    }

    // Recycle a dead particle from the first group
    let base_index = particle_groups[0].effect_particle_offset;
    let dead_index = atomicSub(&render_group_indirect.dead_count, 1u) - 1u;
    index = indirect_buffer.indices[3u * (base_index + dead_index) + 2u];

    // Update PRNG seed
    seed = pcg_hash(index ^ spawner.seed);

    // Spawner transform
    let transform = transpose(
        mat4x4(
            spawner.transform[0],
            spawner.transform[1],
            spawner.transform[2],
            vec4<f32>(0.0, 0.0, 0.0, 1.0)
        )
    );

    // Initialize new particle
    var particle = Particle();
    {{INIT_CODE}}

    {{SIMULATION_SPACE_TRANSFORM_PARTICLE}}

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
