
struct RenderIndirect {
    @align(256) vertex_count: u32,
    instance_count: u32,
    base_index: u32,
    vertex_offset: i32,
    base_instance: u32,
    alive_count: u32,
    dead_count: u32,
    max_spawn: u32,
    ping: u32,
};

struct DispatchIndirect {
    @align(256) x: u32,
    y: u32,
    z: u32,
};

@group(0) @binding(0) var<storage, read_write> render_indirect_buffer : array<RenderIndirect>;
@group(0) @binding(1) var<storage, read_write> dispatch_indirect : array<DispatchIndirect>;

/// Calculate the indirect workgroups counts based on the number of particles alive.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let num_effects = 3u; // FIXME

    let effect_index = global_invocation_id.x;
    if (effect_index >= num_effects) {
        return;
    }

    let render_indirect = &render_indirect_buffer[effect_index];

    // Calculate the number of thread groups to dispatch for the update pass, which is
    // the number of alive particles rounded up to 64 (workgroup_size).
    let alive_count = (*render_indirect).alive_count;
    dispatch_indirect[effect_index].x = (alive_count + 63u) / 64u;

    // Copy the number of dead particles to a constant location, so that the init pass
    // on next frame can atomically modify dead_count in parallel yet still read its
    // initial value at the beginning of the init pass, and limit the number of particles
    // spawned to the number of dead particles to recycle.
    let dead_count = (*render_indirect).dead_count;
    (*render_indirect).max_spawn = dead_count;

    // Clear the rendering instance count, which will be upgraded by the update pass
    // with the particles actually alive at the end of their update (after aged).
    (*render_indirect).instance_count = 0u;

    (*render_indirect).ping = 1u - (*render_indirect).ping;
}
