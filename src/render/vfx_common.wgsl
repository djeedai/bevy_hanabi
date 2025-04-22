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
    /// Total number of effects to update this frame. Used by the indirect
    /// compute pipeline to cap the compute thread to the actual number of
    /// effects to process.
    num_effects: u32,
}

struct Spawner {
    /// Compressed transform of the emitter.
    transform: mat3x4<f32>, // transposed (row-major)
    /// Inverse compressed transform of the emitter.
    inverse_transform: mat3x4<f32>, // transposed (row-major)
    /// Number of particles to spawn this frame, as calculated by the CPU Spawner.
    ///
    /// This is only used if the effect is not a child effect (driven by GPU events).
    spawn: i32,
    /// PRNG seed for this effect instance. Currently this can change each time the
    /// effect is recompiled, and cannot be set deterministically (TODO).
    seed: u32,
    /// Index of the ping-pong buffer of particle indices to read particles from
    /// during rendering. Cached from RenderIndirect::ping after it's swapped
    /// in the indirect dispatch, because the RenderIndirect struct is used by GPU
    /// as an indirect draw source so cannot also be bound as regular storage
    /// buffer for reading.
    render_pong: u32,
    /// Index of the [`GpuEffectMetadata`] for this effect.
    effect_metadata_index: u32,
#ifdef SPAWNER_PADDING
    {{SPAWNER_PADDING}}
#endif
}

const SPAWNER_OFFSET_PONG: u32 = 27u;

struct IndirectBuffer {
    indices: array<u32>,
}

/// A event emitted by another effect during its update pass, to trigger the spawning
/// of one or more particle in this effect.
struct SpawnEvent {
    /// The particle index in the parent effect buffer of the source particle which
    /// triggered the event. This is used to inherit attributes like position or velocity.
    particle_index: u32,
}

/// Append buffer populated during the Update pass of the previous frame by a parent effect,
/// and read back by its child effect(s) during the Init pass of the next frame.
struct EventBuffer {
    /// The spawn events themselves.
    spawn_events: array<SpawnEvent>,
}

/// Info about a single child of a parent effect.
struct ChildInfo {
    /// Index of the effect's IndirectDispatch entry in the global init indirect dispatch array.
    init_indirect_dispatch_index: u32,
    /// Number of events in the associated event buffer.
#ifdef CHILD_INFO_EVENT_COUNT_IS_ATOMIC
    event_count: atomic<i32>,
#else
    event_count: i32,
#endif
}

/// Buffer storing all the ChildInfo structs for several effects.
struct ChildInfoBuffer {
    /// The child info structs themselves.
    rows: array<ChildInfo>,
}

/// Indirect compute dispatch struct for GPU-driven passes. The layout of this struct is dictated by WGSL.
struct IndirectDispatch {
    /// Number of workgroups. Each workgroup has exactly 64 threads.
    x: u32,
    /// Unused; always 1.
    y: u32,
    /// Unused; always 1.
    z: u32,
}

/// Stride in u32 count (4 bytes) of the IndirectDispatch struct.
const DISPATCH_INDIRECT_STRIDE: u32 = 3u;

// Effect metadata offsets. Used when accessing a tightly packed array of EffectMetadata
// as a raw array<u32>, so that we can avoid WGSL struct padding and keep data more compact
// in the GPU buffer. Each offset corresponds to a field in the EffectMetadata struct.
const EM_OFFSET_VERTEX_COUNT: u32 = 0u;
const EM_OFFSET_INSTANCE_COUNT: u32 = 1u;
const EM_OFFSET_FIRST_INDEX_OR_VERTEX_OFFSET: u32 = 2u;
const EM_OFFSET_VERTEX_OFFSET_OR_BASE_INSTANCE: u32 = 3u;
const EM_OFFSET_BASE_INSTANCE: u32 = 4u;
const EM_OFFSET_ALIVE_COUNT: u32 = 5u;
const EM_OFFSET_MAX_UPDATE: u32 = 6u;
const EM_OFFSET_DEAD_COUNT: u32 = 7u;
const EM_OFFSET_MAX_SPAWN: u32 = 8u;
const EM_OFFSET_PING: u32 = 9u;
const EM_OFFSET_INDIRECT_DISPATCH_INDEX: u32 = 10u;

/// Draw indirect parameters for GPU-driven rendering, and additional effect data.
struct EffectMetadata {
    /// Number of vertices in the particle mesh.
    vertex_count: u32,
    /// Number of mesh instances, equal to the number of particles.
    instance_count: atomic<u32>,
    /// First index (if indexed) or vertex offset (if non-indexed).
    first_index_or_vertex_offset: u32,
    /// Vertex offset (if indexed) or base instance (if non-indexed).
    vertex_offset_or_base_instance: i32,
    /// Base instance (if indexed).
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
    /// Maxmimum number of init threads to run on next frame. This is cached from
    /// `dead_count` during the indirect dispatch of the previous frame, so that the
    /// init compute pass can cap its thread count while also decrementing the actual
    /// `dead_count` as particles are spawned.
    max_spawn: atomic<u32>,
    /// Index of the ping buffer for particle indices. Init and update compute passes
    /// always write into the ping buffer and read from the pong buffer. The buffers
    /// are swapped (ping = 1 - ping) during the indirect dispatch.
    ping: u32,
    /// Index of the [`GpuDispatchIndirect`] struct inside the global
    /// [`EffectsMeta::dispatch_indirect_buffer`].
    indirect_dispatch_index: u32,
    /// Index of the [`GpuRenderIndirect`] struct inside the global
    /// [`EffectsMeta::render_group_dispatch_buffer`].
    indirect_render_index: u32,
    /// Offset (in u32 count) of the init indirect dispatch struct inside its
    /// buffer. This avoids having to align those 16-byte structs to the GPU
    /// alignment (at least 32 bytes, even 256 bytes on some).
    init_indirect_dispatch_index: u32,
    /// Index of this effect into its parent's ChildInfo array
    /// ([`EffectChildren::effect_cache_ids`] and its associated GPU
    /// array). This starts at zero for the first child of each effect, and is
    /// only unique per parent, not globally. Only available if this effect is a
    /// child of another effect (i.e. if it has a parent).
    local_child_index: u32,
    /// For children, global index of the ChildInfo into the shared array.
    global_child_index: u32,
    /// For parents, base index of the their first ChildInfo into the shared array.
    base_child_index: u32,

    /// Particle stride, in number of u32.
    particle_stride: u32,
    /// Offset from the particle start to the first sort key, in number of u32.
    sort_key_offset: u32,
    /// Offset from the particle start to the second sort key, in number of u32.
    sort_key2_offset: u32,

    /// Atomic counter incremented each time a particle spawns. Useful for
    /// things like RIBBON_ID or any other use where a unique value is needed.
    /// The value loops back after some time, but unless some particle lives
    /// forever there's little chance of repetition.
    particle_counter: atomic<u32>,

    /// Padding for storage buffer alignment. This struct is sometimes bound as part
    /// of an array, or sometimes individually as a single unit. In the later case,
    /// we need it to be aligned to the GPU limits of the device. That limit is only
    /// known at runtime when initializing the WebGPU device.
    // FIXME - not anymore, but would be again with proper batching, so keep for now
    {{EFFECT_METADATA_PADDING}}
}

/// Stride, in u32 count, between elements of an array<EffectMetadata>.
const EFFECT_METADATA_STRIDE: u32 = {{EFFECT_METADATA_STRIDE}} / 4u;

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
