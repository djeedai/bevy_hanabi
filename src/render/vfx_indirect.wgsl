#import bevy_hanabi::vfx_common::{
    ParticleGroup, SimParams, Spawner,
    DI_OFFSET_X, DI_OFFSET_PONG,
    RGI_OFFSET_ALIVE_COUNT, RGI_OFFSET_MAX_UPDATE, RGI_OFFSET_DEAD_COUNT,
    REM_OFFSET_MAX_SPAWN, RGI_OFFSET_INSTANCE_COUNT, REM_OFFSET_PING
}

@group(0) @binding(0) var<storage, read_write> render_effect_indirect_buffer : array<u32>;
@group(0) @binding(1) var<storage, read_write> render_group_indirect_buffer : array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch_indirect_buffer : array<u32>;
@group(0) @binding(3) var<storage, read> group_buffer : array<ParticleGroup>;
@group(1) @binding(0) var<uniform> sim_params : SimParams;

/// Calculate the indirect workgroups counts based on the number of particles alive.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {

    // Cap at maximum number of effect to process
    let index = global_invocation_id.x;
    if (index >= sim_params.num_groups) {
        return;
    }

    // Cap at spawner array size, just for safety
    if (index >= arrayLength(&group_buffer)) {
        return;
    }

    // Retrieve the effect index from the spawner table
    let group_index = group_buffer[index].group_index;
    let effect_index = group_buffer[index].effect_index;
    let is_first_group = group_buffer[index].index_in_effect == 0u;

    // Calculate the base offset (in number of u32 items) into the render indirect and
    // dispatch indirect arrays.
    let rgi_base = sim_params.render_group_stride * group_index / 4u;
    let di_base = sim_params.dispatch_stride * group_index / 4u;

    // Clear the rendering instance count, which will be upgraded by the update pass
    // with the particles actually alive at the end of their update (after aged).
    render_group_indirect_buffer[rgi_base + RGI_OFFSET_INSTANCE_COUNT] = 0u;

    // Calculate the number of thread groups to dispatch for the update
    // pass, which is the number of alive particles rounded up to 64
    // (workgroup_size).
    let alive_count = render_group_indirect_buffer[rgi_base + RGI_OFFSET_ALIVE_COUNT];
    dispatch_indirect_buffer[di_base + DI_OFFSET_X] = (alive_count + 63u) >> 6u;

    // Update max_update from current value of alive_count, so that the
    // update pass coming next can cap its threads to this value, while also
    // atomically modifying alive_count itself for next frame.
    render_group_indirect_buffer[rgi_base + RGI_OFFSET_MAX_UPDATE] = alive_count;

    if (is_first_group) {
        let rem_base = sim_params.render_effect_stride * effect_index / 4u;

        // Copy the number of dead particles to a constant location, so that the
        // init pass on next frame can atomically modify dead_count in parallel
        // yet still read its initial value at the beginning of the init pass,
        // and limit the number of particles spawned to the number of dead
        // particles to recycle.
        let dead_count = render_group_indirect_buffer[rgi_base + RGI_OFFSET_DEAD_COUNT];
        render_effect_indirect_buffer[rem_base + REM_OFFSET_MAX_SPAWN] = dead_count;

        // Swap ping/pong buffers
        let ping = render_effect_indirect_buffer[rem_base + REM_OFFSET_PING];
        let pong = 1u - ping;
        render_effect_indirect_buffer[rem_base + REM_OFFSET_PING] = pong;

        // Copy the new pong into the dispatch buffer, which will be used during rendering
        // to determine where to read particle indices.
        dispatch_indirect_buffer[di_base + DI_OFFSET_PONG] = pong;
    }
}
