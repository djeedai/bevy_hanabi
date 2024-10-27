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

@group(0) @binding(0) var<uniform> sim_params: SimParams;
@group(1) @binding(0) var<storage, read_write> particle_buffer: ParticleBuffer;
@group(1) @binding(1) var<storage, read_write> indirect_buffer: IndirectBuffer;
@group(1) @binding(2) var<storage, read> particle_groups: array<ParticleGroup>;
{{PROPERTIES_BINDING}}

@group(2) @binding(0) var<storage, read_write> spawner: Spawner; // NOTE - same group as update
@group(3) @binding(0) var<storage, read_write> render_effect_indirect: RenderEffectMetadata;
@group(3) @binding(1) var<storage, read_write> dest_render_group_indirect: RenderGroupIndirect;
@group(3) @binding(2) var<storage, read_write> src_render_group_indirect: RenderGroupIndirect;

{{INIT_EXTRA}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;

    // Cap to max number of dead particles, copied from dead_count at the end of the
    // previous iteration, and constant during this pass (unlike dead_count).
    let max_spawn = atomicLoad(&dest_render_group_indirect.max_spawn);
    if (thread_index >= max_spawn) {
        return;
    }

    // Cap to the actual number of spawning requested by CPU (in the case of
    // spawners) or the number of particles present in the source group (in the
    // case of cloners), since compute shaders run in workgroup_size(64) so more
    // threads than needed are launched (rounded up to 64).
#ifdef CLONE
    // FIXME: This doesn't actually need to be atomic.
    let spawn_count: u32 = atomicLoad(&src_render_group_indirect.alive_count);
#else   // CLONE
    let spawn_count: u32 = u32(spawner.spawn);
#endif  // CLONE
    if (thread_index >= spawn_count) {
        return;
    }

    // Always write into ping, read from pong
    let ping = render_effect_indirect.ping;
    let pong = 1u - ping;

#ifdef CLONE
    let src_base_index = particle_groups[{{SRC_GROUP_INDEX}}].effect_particle_offset +
        particle_groups[{{SRC_GROUP_INDEX}}].indirect_index;
    let src_index = indirect_buffer.indices[3u * (src_base_index + thread_index) + ping];
#endif  // CLONE

    // Recycle a dead particle from the destination group
    var dest_base_index = particle_groups[{{DEST_GROUP_INDEX}}].effect_particle_offset +
        particle_groups[{{DEST_GROUP_INDEX}}].indirect_index;
    let dest_dead_index = atomicSub(&dest_render_group_indirect.dead_count, 1u) - 1u;
    let dest_index = indirect_buffer.indices[3u * (dest_base_index + dest_dead_index) + 2u];

    seed = pcg_hash(dest_index ^ spawner.seed);

#ifdef CLONE
    var particle: Particle = particle_buffer.particles[src_index];
    {{INIT_CODE}}

    // For trails and ribbons, age and lifetime are managed automatically.
    particle.age = 0.0;
    particle.lifetime = spawner.lifetime;
#ifdef ATTRIBUTE_PREV
#ifdef ATTRIBUTE_NEXT
    let prev = particle.prev;
    let next = src_index;
    particle_buffer.particles[next].prev = dest_index;
    if (prev != 0xffffffffu) {
        particle_buffer.particles[prev].next = dest_index;
    }
    particle.next = src_index;
    particle.prev = prev;
#endif  // ATTRIBUTE_NEXT
#endif  // ATTRIBUTE_PREV
#else   // CLONE
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

#ifdef ATTRIBUTE_NEXT
    particle.next = 0xffffffffu;
#endif  // ATTRIBUTE_NEXT
#ifdef ATTRIBUTE_PREV
    particle.prev = 0xffffffffu;
#endif  // ATTRIBUTE_PREV

    {{SIMULATION_SPACE_TRANSFORM_PARTICLE}}
#endif  // CLONE

    // Count as alive
    atomicAdd(&dest_render_group_indirect.alive_count, 1u);

    // Add to alive list
    let dest_indirect_index = atomicAdd(&dest_render_group_indirect.instance_count, 1u);
    indirect_buffer.indices[3u * (dest_base_index + dest_indirect_index) + ping] = dest_index;

    // Write back new particle
    particle_buffer.particles[dest_index] = particle;
}
