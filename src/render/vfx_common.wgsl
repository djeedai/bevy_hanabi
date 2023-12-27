#define_import_path bevy_hanabi::vfx_common

struct SimParams {
    /// Delta time in seconds since last simulation tick.
    delta_time: f32,
    /// Time in seconds since the start of simulation.
    time: f32,
//#ifdef SIM_PARAMS_INDIRECT_DATA
    /// Number of effects batched together.
    num_effects: u32,
    /// Stride in bytes of the RenderIndirect struct. Used to calculate
    /// the position of each effect's data into the buffer of a batch.
    render_stride: u32,
    /// Stride in bytes of the DispatchIndirect struct. Used to calculate
    /// the position of each effect's data into the buffer of a batch.
    dispatch_stride: u32,
//#endif
}

struct ForceFieldSource {
    position: vec3<f32>,
    max_radius: f32,
    min_radius: f32,
    mass: f32,
    force_exponent: f32,
    conform_to_sphere: f32,
}

struct Spawner {
    transform: mat3x4<f32>, // transposed (row-major)
    inverse_transform: mat3x4<f32>, // transposed (row-major)
    spawn: i32,
    seed: u32,
    count: atomic<i32>,
    effect_index: u32,
    force_field: array<ForceFieldSource, 16>,
#ifdef SPAWNER_PADDING
    {{SPAWNER_PADDING}}
#endif
}

struct IndirectBuffer {
    indices: array<u32>,
}

// Dispatch indirect array offsets. Used when accessing an array of DispatchIndirect
// as a raw array<u32>, so that we can avoid WGSL struct padding and keep data
// more compact in the render indirect buffer. Each offset corresponds to a field
// in the DispatchIndirect struct.
const DI_OFFSET_X: u32 = 0u;
const DI_OFFSET_Y: u32 = 1u;
const DI_OFFSET_Z: u32 = 2u;
const DI_OFFSET_PONG: u32 = 3u;

/// Dispatch indirect parameters for GPU driven update compute.
struct DispatchIndirect {
    x: u32,
    y: u32,
    z: u32,
    /// Index of the ping-pong buffer of particle indices to read particles from
    /// during rendering. Cached from RenderIndirect::ping after it's swapped
    /// in the indirect dispatch, because the RenderIndirect struct is used by GPU
    /// as an indirect draw source so cannot also be bound as regular storage
    /// buffer for reading.
    pong: u32,
}

// Render indirect array offsets. Used when accessing an array of RenderIndirect
// as a raw array<u32>, so that we can avoid WGSL struct padding and keep data
// more compact in the render indirect buffer. Each offset corresponds to a field
// in the RenderIndirect struct.
const RI_OFFSET_VERTEX_COUNT: u32 = 0u;
const RI_OFFSET_INSTANCE_COUNT: u32 = 1u;
const RI_OFFSET_BASE_INDEX: u32 = 2u;
const RI_OFFSET_VERTEX_OFFSET: u32 = 3u;
const RI_OFFSET_BASE_INSTANCE: u32 = 4u;
const RI_OFFSET_ALIVE_COUNT: u32 = 5u;
const RI_OFFSET_DEAD_COUNT: u32 = 6u;
const RI_OFFSET_MAX_SPAWN: u32 = 7u;
const RI_OFFSET_PING: u32 = 8u;
const RI_OFFSET_MAX_UPDATE: u32 = 9u;

/// Render indirect parameters for GPU driven rendering.
struct RenderIndirect {
    /// Number of vertices in the particle mesh. Currently always 4 (quad mesh).
    vertex_count: u32,
    /// Number of mesh instances, equal to the number of particles.
    instance_count: atomic<u32>,
    /// Base index (always zero).
    base_index: u32,
    /// Vertex offset (always zero).
    vertex_offset: i32,
    /// Base instance (always zero).
    base_instance: u32,
    /// Number of particles alive after the init pass, used to calculate the number
    /// of compute threads to spawn for the update pass and to cap those threads
    /// via `max_update`.
    alive_count: atomic<u32>,
    /// Number of dead particles, decremented during the init pass as new particles
    /// are spawned, and incremented during the update pass as existing particles die.
    dead_count: atomic<u32>,
    /// Maxmimum number of init threads to run on next frame. This is cached from
    /// `dead_count` during the indirect dispatch of the previous frame, so that the
    /// init compute pass can cap its thread count while also decrementing the actual
    /// `dead_count` as particles are spawned.
#ifdef RI_MAX_SPAWN_ATOMIC
    max_spawn: atomic<u32>,
#else
    max_spawn: u32,
#endif
    /// Index of the ping buffer for particle indices. Init and update compute passes
    /// always write into the ping buffer and read from the pong buffer. The buffers
    /// are swapped during the indirect dispatch.
    ping: u32,
    /// Maximum number of update threads to run. This is cached from `alive_count`
    /// during the indirect dispatch, so that the update compute pass can cap its
    /// thread count while also modifying the actual `alive_count` if some particle
    /// dies during the update pass.
    max_update: u32,
}
