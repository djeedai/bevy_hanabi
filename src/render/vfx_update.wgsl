#import bevy_hanabi::vfx_common::{ForceFieldSource, IndirectBuffer, RenderIndirect, SimParams, Spawner}

struct Particle {
{{ATTRIBUTES}}
}

struct ParticleBuffer {
    particles: array<Particle>,
}

{{PROPERTIES}}

@group(0) @binding(0) var<uniform> sim_params : SimParams;
@group(1) @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer;
@group(1) @binding(1) var<storage, read_write> indirect_buffer : IndirectBuffer;
{{PROPERTIES_BINDING}}
@group(2) @binding(0) var<storage, read_write> spawner : Spawner; // NOTE - same group as init
@group(3) @binding(0) var<storage, read_write> render_indirect : RenderIndirect;

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

{{UPDATE_EXTRA}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;

    // Cap at maximum number of particles.
    // FIXME - This is probably useless given below cap
    let max_particles : u32 = arrayLength(&particle_buffer.particles);
    if (thread_index >= max_particles) {
        return;
    }

    // Cap at maximum number of alive particles.
    if (thread_index >= render_indirect.max_update) {
        return;
    }

    // Always write into ping, read from pong
    let ping = render_indirect.ping;
    let pong = 1u - ping;

    let index = indirect_buffer.indices[3u * thread_index + pong];

    var particle: Particle = particle_buffer.particles[index];

    {{AGE_CODE}}
    {{UPDATE_CODE}}
    {{REAP_CODE}}

    particle_buffer.particles[index] = particle;

    // Check if alive
    if (!is_alive) {
        // Save dead index
        let dead_index = atomicAdd(&render_indirect.dead_count, 1u);
        indirect_buffer.indices[3u * dead_index + 2u] = index;
        // Also increment copy of dead count, which was updated in dispatch indirect
        // pass just before, and need to remain correct after this pass
        atomicAdd(&render_indirect.max_spawn, 1u);
        atomicSub(&render_indirect.alive_count, 1u);
    } else {
        // Increment alive particle count and write indirection index for later rendering
        let indirect_index = atomicAdd(&render_indirect.instance_count, 1u);
        indirect_buffer.indices[3u * indirect_index + ping] = index;
    }
}
