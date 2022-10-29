
// struct RenderIndirect {
//     vertex_count: u32,
//     instance_count: u32,
//     base_index: u32,
//     vertex_offset: i32,
//     base_instance: u32,
//     alive_count: u32,
//     dead_count: u32,
//     max_spawn: u32,
//     ping: u32,
//     max_update: u32,
// };

// struct DispatchIndirect {
//     x: u32,
//     y: u32,
//     z: u32,
// };

struct SimParams {
    dt: f32,
    time: f32,
    num_effects: u32,
    render_stride: u32,
    dispatch_stride: u32,
};

// naga doesn't support 'const' yet
// https://github.com/gfx-rs/naga/issues/1829

// const OFFSET_INSTANCE_COUNT: u32 = 1u;
// const OFFSET_ALIVE_COUNT: u32 = 5u;
// const OFFSET_DEAD_COUNT: u32 = 6u;
// const OFFSET_MAX_SPAWN: u32 = 7u
// const OFFSET_PING: u32 = 8u;
// const OFFSET_MAX_UPDATE: u32 = 9u;

// const OFFSET_X: u32 = 0u;

@group(0) @binding(0) var<storage, read_write> render_indirect_buffer : array<u32>;
@group(0) @binding(1) var<storage, read_write> dispatch_indirect : array<u32>;
@group(1) @binding(0) var<uniform> sim_params : SimParams;

/// Calculate the indirect workgroups counts based on the number of particles alive.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {

    // Cap at maximum number of effect to process
    let effect_index = global_invocation_id.x;
    if (effect_index >= sim_params.num_effects) {
        return;
    }
    
    let ri_base = sim_params.render_stride * effect_index / 4u;
    let di_base = sim_params.dispatch_stride * effect_index / 4u;

    // Calculate the number of thread groups to dispatch for the update pass, which is
    // the number of alive particles rounded up to 64 (workgroup_size).
    let alive_count = render_indirect_buffer[ri_base + 5u];
    dispatch_indirect[di_base + 0u] = (alive_count + 63u) / 64u;

    // Update max_update from current value of alive_count, so that the update pass
    // coming next can cap its threads to this value, while also atomically modifying
    // alive_count itself for next frame.
    render_indirect_buffer[ri_base + 9u] = alive_count;

    // Copy the number of dead particles to a constant location, so that the init pass
    // on next frame can atomically modify dead_count in parallel yet still read its
    // initial value at the beginning of the init pass, and limit the number of particles
    // spawned to the number of dead particles to recycle.
    let dead_count = render_indirect_buffer[ri_base + 6u];
    render_indirect_buffer[ri_base + 7u] = dead_count;

    // Clear the rendering instance count, which will be upgraded by the update pass
    // with the particles actually alive at the end of their update (after aged).
    render_indirect_buffer[ri_base + 1u] = 0u;

    render_indirect_buffer[ri_base + 8u] = 1u - render_indirect_buffer[ri_base + 8u];
}
