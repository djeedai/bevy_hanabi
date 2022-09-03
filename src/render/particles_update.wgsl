struct Particle {
    pos: vec3<f32>,
    age: f32,
    vel: vec3<f32>,
    lifetime: f32,
};

struct ParticleBuffer {
    particles: array<Particle>,
};

struct SimParams {
    dt: f32,
    time: f32,
};

struct ForceFieldParam {
    position: vec3<f32>,
    max_radius: f32,
    min_radius: f32,
    mass: f32,
    force_exponent: f32,
    conform_to_sphere: f32,
};

struct Spawner {
    transform: mat3x4<f32>, // transposed (row-major)
    accel: vec3<f32>,
    spawn: atomic<i32>,
    force_field: array<ForceFieldParam, 16>,
    seed: u32,
    count: atomic<i32>,
};

struct IndirectBuffer {
    indices: array<u32>,
};

@group(0) @binding(0) var<uniform> sim_params : SimParams;
@group(1) @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer;
@group(2) @binding(0) var<storage, read_write> spawner : Spawner;
@group(3) @binding(0) var<storage, read_write> indirect_buffer : IndirectBuffer;

var<private> seed : u32 = 0u;

let tau: f32 = 6.283185307179586476925286766559;

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
fn rand() -> f32 {
    seed = pcg_hash(seed);
    return to_float01(pcg_hash(seed));
}

// Random floating-point number in [0:1]^2
fn rand2() -> vec2<f32> {
    seed = pcg_hash(seed);
    var x = to_float01(seed);
    seed = pcg_hash(seed);
    var y = to_float01(seed);
    return vec2<f32>(x, y);
}

// Random floating-point number in [0:1]^3
fn rand3() -> vec3<f32> {
    seed = pcg_hash(seed);
    var x = to_float01(seed);
    seed = pcg_hash(seed);
    var y = to_float01(seed);
    seed = pcg_hash(seed);
    var z = to_float01(seed);
    return vec3<f32>(x, y, z);
}

// Random floating-point number in [0:1]^4
fn rand4(input: u32) -> vec4<f32> {
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

struct PosVel {
    pos: vec3<f32>,
    vel: vec3<f32>,
};

fn init_pos_vel(index: u32, transform: mat4x4<f32>) -> PosVel {
    var ret : PosVel;
{{INIT_POS_VEL}}
    return ret;
}

fn init_lifetime() -> f32 {
    var ret : f32;
{{INIT_LIFETIME}}
    return ret;
}

fn proj(u: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return dot(v, u) / dot(u,u) * u;
}


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let max_particles : u32 = arrayLength(&particle_buffer.particles);
    let index = global_invocation_id.x;
    if (index >= max_particles) {
        return;
    }

    var vPos : vec3<f32> = particle_buffer.particles[index].pos;
    var vVel : vec3<f32> = particle_buffer.particles[index].vel;
    var vAge : f32 = particle_buffer.particles[index].age;
    var vLifetime : f32 = particle_buffer.particles[index].lifetime;

    // Age the particle
    vAge = vAge + sim_params.dt;
    if (vAge >= vLifetime) {
        // Particle dead; try to recycle into newly-spawned one
        if (atomicSub(&spawner.spawn, 1) > 0) {
            // Update PRNG seed
            seed = pcg_hash(index ^ spawner.seed);

            let transform = transpose(
                mat4x4(
                    spawner.transform[0],
                    spawner.transform[1],
                    spawner.transform[2],
                    vec4<f32>(0.0, 0.0, 0.0, 1.0)
                )
            );

            // Initialize new particle
            var posVel = init_pos_vel(index, transform);
            vPos = posVel.pos;
            vVel = posVel.vel;
            vAge = 0.0;
            vLifetime = init_lifetime();
        } else {
            // Nothing to spawn; simply return without writing any update
            return;
        }
    }

{{FORCE_FIELD_CODE}}

    // Increment alive particle count and write indirection index
    let indirect_index = atomicAdd(&spawner.count, 1);
    indirect_buffer.indices[indirect_index] = index;

    // Write back particle itself
    particle_buffer.particles[index].pos = vPos;
    particle_buffer.particles[index].vel = vVel;
    particle_buffer.particles[index].age = vAge;
    particle_buffer.particles[index].lifetime = vLifetime;
}