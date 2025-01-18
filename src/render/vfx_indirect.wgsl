#import bevy_hanabi::vfx_common::{
    ChildInfo, ParticleGroup, SimParams,
    DI_OFFSET_X, DI_OFFSET_PONG,
    RGI_OFFSET_ALIVE_COUNT, RGI_OFFSET_MAX_UPDATE, RGI_OFFSET_DEAD_COUNT,
    REM_OFFSET_MAX_SPAWN, RGI_OFFSET_INSTANCE_COUNT, RGI_OFFSET_MAX_SPAWN,
    REM_OFFSET_EVENT_COUNT, REM_OFFSET_PING, DISPATCH_INDIRECT_STRIDE,
    RENDER_EFFECT_INDIRECT_STRIDE, RENDER_GROUP_INDIRECT_STRIDE
}

@group(0) @binding(0) var<storage, read_write> render_effect_indirect_buffer : array<u32>;
@group(0) @binding(1) var<storage, read_write> render_group_indirect_buffer : array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch_indirect_buffer : array<u32>;
@group(0) @binding(3) var<storage, read> group_buffer : array<ParticleGroup>;
@group(1) @binding(0) var<uniform> sim_params : SimParams;
#ifdef HAS_GPU_SPAWN_EVENTS
@group(2) @binding(0) var<storage, read_write> child_info : array<ChildInfo>;
#endif

/// Calculate the indirect workgroups counts based on the number of particles alive.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {

    // Cap at maximum number of groups to process
    let index = global_invocation_id.x;

    // FIXME - the group_buffer array has gaps, we can't limit to the number of effects in use
    // otherwise we'll miss some and process the unused gaps instead of some active ones.
    // Since all writes below are idempotent, except the ping/pong one, there's no harm updating
    // unused effect rows.
    // if (index >= sim_params.num_groups) {
    //     return;
    // }

#ifdef HAS_GPU_SPAWN_EVENTS
    // Clear any GPU event. The indexing is safe because there are always less child effects
    // than there are effects in total, so 'index' will always cover the entire child info array.
    if (index < arrayLength(&child_info)) {
        // FIXME - doesn't need to be atomic
        atomicStore(&child_info[index].event_count, 0);
    }
#endif

    // Cap at group array size, just for safety
    if (index >= arrayLength(&group_buffer)) {
        return;
    }

    // Retrieve the effect index from the spawner table
    let group_index = group_buffer[index].group_index;
    let effect_index = group_buffer[index].effect_index;

    let rem_base = RENDER_EFFECT_INDIRECT_STRIDE * effect_index;
    let group_index_in_effect = group_buffer[index].index_in_effect;
    let is_first_group = group_index_in_effect == 0u;

    // Clear the rendering instance count, which will be upgraded by the update pass
    // with the particles actually alive at the end of their update (after aged).
    let indirect_render_index = group_buffer[index].indirect_render_index;
    let rgi_base = RENDER_GROUP_INDIRECT_STRIDE * indirect_render_index;
    render_group_indirect_buffer[rgi_base + RGI_OFFSET_INSTANCE_COUNT] = 0u;

    let alive_count = render_group_indirect_buffer[rgi_base + RGI_OFFSET_ALIVE_COUNT];
    let dead_count = render_group_indirect_buffer[rgi_base + RGI_OFFSET_DEAD_COUNT];

    // Calculate the number of thread groups to dispatch for the update
    // pass, which is the number of alive particles rounded up to 64
    // (workgroup_size).
    let indirect_dispatch_index = group_buffer[index].indirect_dispatch_index;
    let di_base = DISPATCH_INDIRECT_STRIDE * indirect_dispatch_index;
    dispatch_indirect_buffer[di_base + DI_OFFSET_X] = (alive_count + 63u) >> 6u;

    // Update max_update from current value of alive_count, so that the
    // update pass coming next can cap its threads to this value, while also
    // atomically modifying alive_count itself for next frame.
    render_group_indirect_buffer[rgi_base + RGI_OFFSET_MAX_UPDATE] = alive_count;

    // Always clear the number of GPU events before the next update increments it again
    //render_effect_indirect_buffer[rem_base + REM_OFFSET_EVENT_COUNT] = 0u;

    // Copy the number of dead particles to a constant location, so that the
    // init pass on next frame can atomically modify dead_count in parallel
    // yet still read its initial value at the beginning of the init pass,
    // and limit the number of particles spawned to the number of dead
    // particles to recycle.
    render_group_indirect_buffer[rgi_base + RGI_OFFSET_MAX_SPAWN] = dead_count;
    
    if (is_first_group) {
        // Swap ping/pong buffers
        let ping = render_effect_indirect_buffer[rem_base + REM_OFFSET_PING];
        let pong = 1u - ping;
        render_effect_indirect_buffer[rem_base + REM_OFFSET_PING] = pong;

        // Copy the new pong into the dispatch buffer, which will be used during rendering
        // to determine where to read particle indices.
        dispatch_indirect_buffer[di_base + DI_OFFSET_PONG] = pong;
    }
}
