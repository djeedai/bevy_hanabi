#import bevy_hanabi::vfx_common::{
    ChildInfo, ChildInfoBuffer, DispatchIndirectArgs, SimParams, Spawner,
    EM_OFFSET_ALIVE_COUNT, EM_OFFSET_MAX_UPDATE, EM_OFFSET_CAPACITY,
    EM_OFFSET_MAX_SPAWN, EM_OFFSET_INDIRECT_DISPATCH_INDEX, DRAW_INDEXED_INDIRECT_STRIDE,
    EM_OFFSET_INDIRECT_WRITE_INDEX, EFFECT_METADATA_STRIDE
}

@group(0) @binding(0) var<uniform> sim_params : SimParams;

// Tightly packed array of EffectMetadata[], accessed as u32 array.
@group(1) @binding(0) var<storage, read_write> effect_metadata_buffer : array<u32>;
@group(1) @binding(1) var<storage, read_write> dispatch_indirect_buffer : array<DispatchIndirectArgs>;
// Tightly packed array of DrawIndexedIndirectArgs[], accessed as u32 array. This can contain
// some DrawIndirectArgs[] instead, but in that case the stride is adjusted so all rows have
// the same size. Since we access the instance_count, which is at the same position in both,
// we ignore their size difference (the non-indexed one is padded).
@group(1) @binding(2) var<storage, read_write> draw_indirect_buffer : array<u32>;

@group(2) @binding(0) var<storage, read_write> spawner_buffer : array<Spawner>;

#ifdef HAS_GPU_SPAWN_EVENTS
@group(3) @binding(0) var<storage, read_write> child_info_buffer : ChildInfoBuffer;
#endif

/// Calculate the indirect workgroups counts based on the number of particles alive.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;

#ifdef HAS_GPU_SPAWN_EVENTS
    // Clear any GPU event. The indexing is safe because there are always less child effects
    // than there are effects in total, so 'index' will always cover the entire child info array.
    if (thread_index < arrayLength(&child_info_buffer.rows)) {
        child_info_buffer.rows[thread_index].event_count = 0;
    }
#endif

    // Cap at maximum number of effects
    let effect_index = thread_index;
    if (effect_index >= sim_params.num_effects) {
        return;
    }

    let effect_metadata_index = spawner_buffer[effect_index].effect_metadata_index;
    let em_base = EFFECT_METADATA_STRIDE * effect_metadata_index;

    // Clear the rendering instance count, which will be upgraded by the update pass
    // with the particles actually alive at the end of their update (after aged).
    let draw_indirect_index = spawner_buffer[effect_index].draw_indirect_index;
    let dri_base = DRAW_INDEXED_INDIRECT_STRIDE * draw_indirect_index;
    draw_indirect_buffer[dri_base + 1u] = 0u;

    let capacity = effect_metadata_buffer[em_base + EM_OFFSET_CAPACITY];
    let alive_count = effect_metadata_buffer[em_base + EM_OFFSET_ALIVE_COUNT];
    let dead_count = capacity - alive_count;

    // Update max_update from current value of alive_count, so that the
    // update pass coming next can cap its threads to this value, while also
    // atomically modifying alive_count itself for next frame.
    effect_metadata_buffer[em_base + EM_OFFSET_MAX_UPDATE] = alive_count;

    // Copy the number of dead particles to a constant location, so that the
    // init pass on next frame can atomically modify alive_count in parallel
    // yet still read its initial value at the beginning of the init pass,
    // and limit the number of particles spawned to the number of dead
    // particles to recycle.
    effect_metadata_buffer[em_base + EM_OFFSET_MAX_SPAWN] = dead_count;

    // Calculate the number of workgroups (thread groups) to dispatch for the update
    // pass, which is the number of alive particles rounded up to 64 (workgroup_size).
    let indirect_dispatch_index = effect_metadata_buffer[em_base + EM_OFFSET_INDIRECT_DISPATCH_INDEX];
    dispatch_indirect_buffer[indirect_dispatch_index].x = (alive_count + 63u) >> 6u;
    dispatch_indirect_buffer[indirect_dispatch_index].y = 1u;
    dispatch_indirect_buffer[indirect_dispatch_index].z = 1u;

    // Swap ping/pong buffers. The update pass always writes into ping, and both the update
    // pass and the render pass always read from pong.
    let ping = effect_metadata_buffer[em_base + EM_OFFSET_INDIRECT_WRITE_INDEX];
    let pong = 1u - ping;
    effect_metadata_buffer[em_base + EM_OFFSET_INDIRECT_WRITE_INDEX] = pong;

    // Copy the new pong into the spawner buffer, which will be used during rendering
    // to determine where to read particle indices.
    spawner_buffer[effect_index].render_indirect_read_index = pong;
}
