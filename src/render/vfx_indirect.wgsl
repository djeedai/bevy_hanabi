
struct SimParams {
    delta_time: f32,
    time: f32,
    num_effects: u32,
    render_stride: u32,
    dispatch_stride: u32,
};

struct ForceFieldSource {
    position: vec3<f32>,
    max_radius: f32,
    min_radius: f32,
    mass: f32,
    force_exponent: f32,
    conform_to_sphere: f32,
};

struct Spawner {
    transform: mat3x4<f32>, // transposed (row-major)
    inverse_transform: mat3x4<f32>, // transposed (row-major)
    spawn: i32,
    seed: u32,
    count: i32,
    effect_index: u32,
    force_field: array<ForceFieldSource, 16>,
    {{SPAWNER_PADDING}}
};

struct SpawnerBuffer {
    spawners: array<Spawner>,
}

// naga doesn't support 'const' yet
// https://github.com/gfx-rs/naga/issues/1829

// const OFFSET_INSTANCE_COUNT: u32 = 1u;
// const OFFSET_ALIVE_COUNT: u32 = 5u;
// const OFFSET_DEAD_COUNT: u32 = 6u;
// const OFFSET_MAX_SPAWN: u32 = 7u
// const OFFSET_PING: u32 = 8u;
// const OFFSET_MAX_UPDATE: u32 = 9u;

// const OFFSET_X: u32 = 0u;
// const OFFSET_Y: u32 = 1u;
// const OFFSET_Z: u32 = 2u;
// const OFFSET_PONG: u32 = 3u;

@group(0) @binding(0) var<storage, read_write> render_indirect_buffer : array<u32>;
@group(0) @binding(1) var<storage, read_write> dispatch_indirect : array<u32>;
@group(0) @binding(2) var<storage, read> spawner_buffer : SpawnerBuffer;
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

    // Swap ping/pong buffers
    let ping = render_indirect_buffer[ri_base + 8u];
    let pong = 1u - ping;
    render_indirect_buffer[ri_base + 8u] = pong;

    // Copy the new pong into the dispatch buffer, which will be used during rendering
    // to determine where to read particle indices.
    dispatch_indirect[di_base + 3u] = pong;
}
