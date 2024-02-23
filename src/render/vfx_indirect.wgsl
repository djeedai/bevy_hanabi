#import bevy_hanabi::vfx_common::{
    SimParams, Spawner,
    DI_OFFSET_X, DI_OFFSET_PONG,
    RI_OFFSET_ALIVE_COUNT, RI_OFFSET_MAX_UPDATE, RI_OFFSET_DEAD_COUNT,
    RI_OFFSET_MAX_SPAWN, RI_OFFSET_INSTANCE_COUNT, RI_OFFSET_PING,
    TRI_OFFSET_BASE_INSTANCE, TRI_OFFSET_INSTANCE_COUNT, TRI_OFFSET_TRAIL_INDICES
}

struct SpawnerBuffer {
    spawners: array<Spawner>,
}

@group(0) @binding(0) var<storage, read_write> render_indirect_buffer : array<u32>;
@group(0) @binding(1) var<storage, read_write> dispatch_indirect_buffer : array<u32>;
@group(0) @binding(2) var<storage, read> spawner_buffer : SpawnerBuffer;
#ifdef TRAILS
@group(0) @binding(3) var<storage, read_write> trail_render_indirect_buffer : array<u32>;
@group(0) @binding(4) var<storage, read_write> trail_chunk_buffer : array<u32>;
#endif
@group(1) @binding(0) var<uniform> sim_params : SimParams;

/// Calculate the indirect workgroups counts based on the number of particles alive.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {

    // Cap at maximum number of effect to process
    let index = global_invocation_id.x;
    if (index >= sim_params.num_effects) {
        return;
    }

    // Cap at spawner array size, just for safety
    if (index >= arrayLength(&spawner_buffer.spawners)) {
        return;
    }

    // Retrieve the effect index from the spawner table
    let effect_index = spawner_buffer.spawners[index].effect_index;

    // Calculate the base offset (in number of u32 items) into the render indirect and
    // dispatch indirect arrays.
    let ri_base = sim_params.render_stride * effect_index / 4u;
    let di_base = sim_params.dispatch_stride * effect_index / 4u;
    let tri_base = sim_params.trail_render_stride * effect_index / 4u;

    // Calculate the number of thread groups to dispatch for the update pass, which is
    // the number of alive particles rounded up to 64 (workgroup_size).
    let alive_count = render_indirect_buffer[ri_base + RI_OFFSET_ALIVE_COUNT];
    dispatch_indirect_buffer[di_base + DI_OFFSET_X] = (alive_count + 63u) >> 6u;

    // Update max_update from current value of alive_count, so that the update pass
    // coming next can cap its threads to this value, while also atomically modifying
    // alive_count itself for next frame.
    render_indirect_buffer[ri_base + RI_OFFSET_MAX_UPDATE] = alive_count;

    // Copy the number of dead particles to a constant location, so that the init pass
    // on next frame can atomically modify dead_count in parallel yet still read its
    // initial value at the beginning of the init pass, and limit the number of particles
    // spawned to the number of dead particles to recycle.
    let dead_count = render_indirect_buffer[ri_base + RI_OFFSET_DEAD_COUNT];
    render_indirect_buffer[ri_base + RI_OFFSET_MAX_SPAWN] = dead_count;

    // Clear the rendering instance count, which will be upgraded by the update pass
    // with the particles actually alive at the end of their update (after aged).
    render_indirect_buffer[ri_base + RI_OFFSET_INSTANCE_COUNT] = 0u;

#ifdef TRAILS
    // If needed, spawn a trail particle.
    if (spawner_buffer.spawners[index].spawn_trail_particle != 0) {
        // Get the trail head and tail chunks.
        let head_chunk_index = spawner_buffer.spawners[index].trail_head_chunk;
        let tail_chunk_index = spawner_buffer.spawners[index].trail_tail_chunk;

        // The previous tick's final instance becomes the current head.
        let last_base_instance = trail_render_indirect_buffer[tri_base + TRI_OFFSET_BASE_INSTANCE];
        let last_instance_count = trail_render_indirect_buffer[tri_base + TRI_OFFSET_INSTANCE_COUNT];
        let last_instance_index = last_base_instance + last_instance_count;
        trail_chunk_buffer[head_chunk_index] = last_instance_index;

        // Calculate the new base count and instance index.
        let tail_trail_index = trail_chunk_buffer[tail_chunk_index];
        trail_render_indirect_buffer[tri_base + TRI_OFFSET_BASE_INSTANCE] = u32(tail_trail_index);
        trail_render_indirect_buffer[tri_base + TRI_OFFSET_INSTANCE_COUNT] =
            last_instance_index - u32(tail_trail_index);
    }
#endif

    // Swap ping/pong buffers
    let ping = render_indirect_buffer[ri_base + RI_OFFSET_PING];
    let pong = 1u - ping;
    render_indirect_buffer[ri_base + RI_OFFSET_PING] = pong;

    // Copy the new pong into the dispatch buffer, which will be used during rendering
    // to determine where to read particle indices.
    dispatch_indirect_buffer[di_base + DI_OFFSET_PONG] = pong;
}
