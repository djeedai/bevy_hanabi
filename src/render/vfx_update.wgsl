#import bevy_hanabi::vfx_common::{
    IndirectBuffer, RenderIndirect, TrailRenderIndirect, SimParams, Spawner,
    seed, tau, pcg_hash, to_float01, frand, frand2, frand3, frand4,
    rand_uniform, proj
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
{{PROPERTIES_BINDING}}
{{TRAIL_BINDING}}
@group(2) @binding(0) var<storage, read_write> spawner : Spawner; // NOTE - same group as init
@group(3) @binding(0) var<storage, read_write> render_indirect : RenderIndirect;
{{TRAIL_RENDER_INDIRECT_BINDING}}

{{UPDATE_EXTRA}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;

    // Cap at maximum number of particles.
    // FIXME - This is probably useless given below cap
    let max_particles : u32 = arrayLength(&particle_buffer.particles);
    if (thread_index >= max_particles) {
        return;
    }

    // Cap at maximum number of alive particles.
    if (thread_index >= render_indirect.max_update) {
        return;
    }

    // Always write into ping, read from pong
    let ping = render_indirect.ping;
    let pong = 1u - ping;

    let index = indirect_buffer.indices[3u * thread_index + pong];

    var particle: Particle = particle_buffer.particles[index];

    {{AGE_CODE}}
    {{UPDATE_CODE}}
    {{REAP_CODE}}

    particle_buffer.particles[index] = particle;

    // Check if alive
    if (!is_alive) {
        // Save dead index
        let dead_index = atomicAdd(&render_indirect.dead_count, 1u);
        indirect_buffer.indices[3u * dead_index + 2u] = index;
        // Also increment copy of dead count, which was updated in dispatch indirect
        // pass just before, and need to remain correct after this pass
        atomicAdd(&render_indirect.max_spawn, 1u);
        atomicSub(&render_indirect.alive_count, 1u);
    } else {
        // Increment alive particle count and write indirection index for later rendering
        let indirect_index = atomicAdd(&render_indirect.instance_count, 1u);
        indirect_buffer.indices[3u * indirect_index + ping] = index;

#ifdef TRAILS
        // If we're alive and due to spawn a trail particle, do that now.
        //
        // Note that we only adjust the instance count here, to reduce the
        // number of atomic operations we need to do. *Next* frame,
        // `vfx_indirect.wgsl` will take care of computing the value to write
        // into the chunk buffer by examining the final instance count from
        // *this* frame.
        if (spawner.spawn_trail_particle != 0) {
            let dest_index = trail_render_indirect.base_instance +
                atomicAdd(&trail_render_indirect.instance_count, 1u);
            trail_buffer.particles[dest_index % spawner.trail_capacity] = particle;
        }
#endif
    }
}
