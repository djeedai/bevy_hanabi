#define_import_path bevy_hanabi::vfx_common

struct SimParams {
    /// Delta time in seconds since last simulation tick.
    delta_time: f32,
    /// Time in seconds since the start of simulation.
    time: f32,
    /// Virtual delta time in seconds since last simulation tick.
    virtual_delta_time: f32,
    /// Virtual time in seconds since the start of simulation.
    virtual_time: f32,
    /// Real delta time in seconds since last simulation tick.
    real_delta_time: f32,
    /// Real time in seconds since the start of simulation.
    real_time: f32,
    /// Number of groups batched together.
    num_groups: u32,
}

struct Spawner {
    transform: mat3x4<f32>, // transposed (row-major)
    inverse_transform: mat3x4<f32>, // transposed (row-major)
    spawn: i32,
    seed: u32,
    // Can't use storage<read> with atomics
#ifdef SPAWNER_READONLY
    count: i32,
#else
    count: atomic<i32>,
#endif
    effect_index: u32,
    // The lifetime to initialize particles with. This is only used for cloners
    // (i.e. trails or ribbons).
    lifetime: f32,
#ifdef SPAWNER_PADDING
    {{SPAWNER_PADDING}}
#endif
}

// Per-group data for a single particle effect group inside an effect.
struct ParticleGroup {
    // Index of the group, generally zero unless there are trails.
    group_index: u32,
    effect_index: u32,
    // The index relative to the effect: e.g. 0 if this is the first group in
    // the effect.
    index_in_effect: u32,
    // Index of the first element for this group in the indirect index buffer.
    indirect_index: u32,
    // The capacity of this group.
    capacity: u32,
    // The index of the first particle in this effect in the particle and
    // indirect buffers.
    effect_particle_offset: u32,
    {{PARTICLE_GROUP_PADDING}}
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
    {{DISPATCH_INDIRECT_PADDING}}
}

// Render indirect array offsets. Used when accessing an array of RenderIndirect
// as a raw array<u32>, so that we can avoid WGSL struct padding and keep data
// more compact in the render indirect buffer. Each offset corresponds to a field
// in the RenderIndirect struct.
const REM_OFFSET_PING: u32 = 0u;

const RGI_OFFSET_VERTEX_COUNT: u32 = 0u;
const RGI_OFFSET_INSTANCE_COUNT: u32 = 1u;
const RGI_OFFSET_VERTEX_OFFSET: u32 = 2u;
const RGI_OFFSET_BASE_INSTANCE: u32 = 3u;
const RGI_OFFSET_ALIVE_COUNT: u32 = 4u;
const RGI_OFFSET_MAX_UPDATE: u32 = 5u;
const RGI_OFFSET_DEAD_COUNT: u32 = 6u;
const RGI_OFFSET_MAX_SPAWN: u32 = 7u;

struct RenderEffectMetadata {
    /// Index of the ping buffer for particle indices. Init and update compute passes
    /// always write into the ping buffer and read from the pong buffer. The buffers
    /// are swapped during the indirect dispatch.
    ping: u32,
    {{RENDER_EFFECT_INDIRECT_PADDING}}
}

/// Render indirect parameters for GPU driven rendering.
struct RenderGroupIndirect {
    /// Number of vertices in the particle mesh. Currently always 4 (quad mesh).
    vertex_count: u32,
    /// Number of mesh instances, equal to the number of particles.
    instance_count: atomic<u32>,
    /// Vertex offset (always zero).
    vertex_offset: i32,
    /// Base instance.
    base_instance: u32,
    /// Number of particles alive after the init pass, used to calculate the number
    /// of compute threads to spawn for the update pass and to cap those threads
    /// via `max_update`.
    alive_count: atomic<u32>,
    /// Maximum number of update threads to run. This is cached from `alive_count`
    /// during the indirect dispatch, so that the update compute pass can cap its
    /// thread count while also modifying the actual `alive_count` if some particle
    /// dies during the update pass.
    max_update: u32,
    /// Number of dead particles, decremented during the init pass as new particles
    /// are spawned, and incremented during the update pass as existing particles die.
    dead_count: atomic<u32>,
    max_spawn: atomic<u32>,
    {{RENDER_GROUP_INDIRECT_PADDING}}
}

var<private> seed : u32 = 0u;

const tau: f32 = 6.283185307179586476925286766559;

// Rand: PCG
// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
fn pcg_hash(input: u32) -> u32 {
    var state: u32 = input * 747796405u + 2891336453u;
    var word: u32 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn to_float01(u: u32) -> f32 {
    // Note: could generate only 24 bits of randomness
    return bitcast<f32>((u & 0x007fffffu) | 0x3f800000u) - 1.;
}

// Random floating-point number in [0:1]
fn frand() -> f32 {
    seed = pcg_hash(seed);
    return to_float01(pcg_hash(seed));
}

// Random floating-point number in [0:1]^2
fn frand2() -> vec2<f32> {
    seed = pcg_hash(seed);
    var x = to_float01(seed);
    seed = pcg_hash(seed);
    var y = to_float01(seed);
    return vec2<f32>(x, y);
}

// Random floating-point number in [0:1]^3
fn frand3() -> vec3<f32> {
    seed = pcg_hash(seed);
    var x = to_float01(seed);
    seed = pcg_hash(seed);
    var y = to_float01(seed);
    seed = pcg_hash(seed);
    var z = to_float01(seed);
    return vec3<f32>(x, y, z);
}

// Random floating-point number in [0:1]^4
fn frand4() -> vec4<f32> {
    // Each rand() produces 32 bits, and we need 24 bits per component,
    // so can get away with only 3 calls.
    var r0 = pcg_hash(seed);
    var r1 = pcg_hash(r0);
    var r2 = pcg_hash(r1);
    seed = r2;
    var x = to_float01(r0);
    var r01 = (r0 & 0xff000000u) >> 8u | (r1 & 0x0000ffffu);
    var y = to_float01(r01);
    var r12 = (r1 & 0xffff0000u) >> 8u | (r2 & 0x000000ffu);
    var z = to_float01(r12);
    var r22 = r2 >> 8u;
    var w = to_float01(r22);
    return vec4<f32>(x, y, z, w);
}

fn rand_uniform_f(a: f32, b: f32) -> f32 {
    return a + frand() * (b - a);
}

fn rand_uniform_vec2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + frand2() * (b - a);
}

fn rand_uniform_vec3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return a + frand3() * (b - a);
}

fn rand_uniform_vec4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return a + frand4() * (b - a);
}

// Normal distribution computed using Box-Muller transform
fn rand_normal_f(mean: f32, std_dev: f32) -> f32 {
    var u = frand();
    var v = frand();
    var r = sqrt(-2.0 * log(u));
    return mean + std_dev * r * cos(tau * v);
}

fn rand_normal_vec2(mean: vec2f, std_dev: vec2f) -> vec2f {
    var u = frand();
    var v = frand2();
    var r = sqrt(-2.0 * log(u));
    return mean + std_dev * r * cos(tau * v);
}

fn rand_normal_vec3(mean: vec3f, std_dev: vec3f) -> vec3f {
    var u = frand();
    var v = frand3();
    var r = sqrt(-2.0 * log(u));
    return mean + std_dev * r * cos(tau * v);
}

fn rand_normal_vec4(mean: vec4f, std_dev: vec4f) -> vec4f {
    var u = frand();
    var v = frand4();
    var r = sqrt(-2.0 * log(u));
    return mean + std_dev * r * cos(tau * v);
}

fn proj(u: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return dot(v, u) / dot(u,u) * u;
}
