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

{{PARENT_PARTICLES}}

{{PROPERTIES}}

@group(0) @binding(0) var<uniform> sim_params : SimParams;
@group(1) @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer;
@group(1) @binding(1) var<storage, read_write> indirect_buffer : IndirectBuffer;
@group(1) @binding(2) var<storage, read> particle_groups : array<ParticleGroup>;
{{PROPERTIES_BINDING}}
#ifdef USE_GPU_SPAWN_EVENTS
@group(1) @binding(4) var<storage, read> child_info : array<ChildInfo>;
@group(1) @binding(5) var<storage, read_write> event_buffer : EventBuffer;
#endif
{{PARENT_PARTICLE_BINDING}}
@group(2) @binding(0) var<storage, read_write> spawner : Spawner; // NOTE - same group as update  // FIXME - this should be read-only
@group(3) @binding(0) var<storage, read_write> render_effect_indirect : RenderEffectMetadata;
@group(3) @binding(1) var<storage, read_write> dest_render_group_indirect : RenderGroupIndirect;
#ifdef CLONE
@group(3) @binding(2) var<storage, read_write> src_render_group_indirect : RenderGroupIndirect;
#endif

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

    // Cap to the actual number of spawning requested by CPU or GPU, since compute shaders run
    // in workgroup_size(64) so more threads than needed are launched (rounded up to 64).
#ifdef USE_GPU_SPAWN_EVENTS
    let event_index = thread_index;
    let child_index = particle_groups[0].child_index; //< FIXME - Probably wrong...
    let event_count = atomicLoad(&child_info[child_index].event_count);
    if (event_index >= u32(event_count)) {
        return;
    }
#else
    // Cap to the actual number of spawning requested by CPU (in the case of
    // spawners) or the number of particles present in the source group (in the
    // case of cloners).
#ifdef CLONE
    // FIXME: This doesn't actually need to be atomic.
    let spawn_count: u32 = atomicLoad(&src_render_group_indirect.alive_count);
#else   // CLONE
    let spawn_count: u32 = u32(spawner.spawn);
#endif  // CLONE
    if (thread_index >= spawn_count) {
        return;
    }
#endif

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

    // Initialize the PRNG seed
    seed = pcg_hash(dest_index ^ spawner.seed);

#ifdef CLONE

    // Start from a pure clone of the head particle in the source group
    var particle: Particle = particle_buffer.particles[src_index];

    {{INIT_CODE}}

    // For trails and ribbons, age and lifetime are managed automatically.
    particle.age = 0.0;
    particle.lifetime = spawner.lifetime;

    // Insert the cloned particle into the linked list
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

#else  // CLONE

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

#ifdef USE_GPU_SPAWN_EVENTS
    // Fetch parent particle which triggered this spawn
    let parent_index = event_buffer.spawn_events[event_index].particle_index;
    let parent_particle = parent_particle_buffer.particles[parent_index];
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
    // don't have a CPU spawner to read from.
#ifndef USE_GPU_SPAWN_EVENTS
    {{SIMULATION_SPACE_TRANSFORM_PARTICLE}}
#endif

#endif  // CLONE

    // Count as alive
    atomicAdd(&dest_render_group_indirect.alive_count, 1u);

    // Add to alive list
    let dest_indirect_index = atomicAdd(&dest_render_group_indirect.instance_count, 1u);
    indirect_buffer.indices[3u * (dest_base_index + dest_indirect_index) + ping] = dest_index;

    // Write back new particle
    particle_buffer.particles[dest_index] = particle;
}
