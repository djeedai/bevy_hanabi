struct Particle {
{{ATTRIBUTES}}
}

struct ParticleBuffer {
    particles: array<Particle>,
}

struct SimParams {
    delta_time: f32,
    time: f32,
}

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
    count: atomic<i32>,
    effect_index: u32,
    force_field: array<ForceFieldSource, 16>,
}

struct IndirectBuffer {
    indices: array<u32>,
}

struct RenderIndirectBuffer {
    vertex_count: u32,
    instance_count: atomic<u32>,
    base_index: u32,
    vertex_offset: i32,
    base_instance: u32,
    alive_count: atomic<u32>,
    dead_count: atomic<u32>,
    max_spawn: u32,
    ping: u32,
}

{{PROPERTIES}}

@group(0) @binding(0) var<uniform> sim_params : SimParams;
@group(1) @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer;
@group(1) @binding(1) var<storage, read_write> indirect_buffer : IndirectBuffer;
{{PROPERTIES_BINDING}}
@group(2) @binding(0) var<storage, read_write> spawner : Spawner; // NOTE - same group as update
@group(3) @binding(0) var<storage, read_write> render_indirect : RenderIndirectBuffer;

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

fn rand_uniform(a: f32, b: f32) -> f32 {
    return a + frand() * (b - a);
}

fn proj(u: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return dot(v, u) / dot(u,u) * u;
}

{{INIT_EXTRA}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    var index = global_invocation_id.x;

    // Cap to max number of dead particles, copied from dead_count at the end of the
    // previous iteration, and constant during this pass (unlike dead_count).
    if (index >= render_indirect.max_spawn) {
        return;
    }

    // Cap to the actual number of spawning requested by CPU, since compute shaders run
    // in workgroup_size(64) so more threads than needed are launched (rounded up to 64).
    let spawn_count : u32 = u32(spawner.spawn);
    if (index >= spawn_count) {
        return;
    }

    // Recycle a dead particle
    let dead_index = atomicSub(&render_indirect.dead_count, 1u) - 1u;
    index = indirect_buffer.indices[3u * dead_index + 2u];

    // Update PRNG seed
    seed = pcg_hash(index ^ spawner.seed);

    // Spawner transform
    let transform = transpose(
        mat4x4(
            spawner.transform[0],
            spawner.transform[1],
            spawner.transform[2],
            vec4<f32>(0.0, 0.0, 0.0, 1.0)
        )
    );

    // Initialize new particle
    var particle = Particle();
    {{INIT_CODE}}

    {{SIMULATION_SPACE_TRANSFORM_PARTICLE}}

    // Count as alive
    atomicAdd(&render_indirect.alive_count, 1u);

    // Always write into ping, read from pong
    let ping = render_indirect.ping;

    // Add to alive list
    let indirect_index = atomicAdd(&render_indirect.instance_count, 1u);
    indirect_buffer.indices[3u * indirect_index + ping] = index;

    // Write back spawned particle
    particle_buffer.particles[index] = particle;
}
