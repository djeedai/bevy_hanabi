struct Particle {
    pos: vec3<f32>;
    age: f32;
    vel: vec3<f32>;
    lifetime: f32;
};

struct Particles {
    particles: [[stride(32)]] array<Particle>;
};

struct SimParams {
    dt: f32;
    time: f32;
};

struct Spawner {
    origin: vec3<f32>;
    spawn: atomic<i32>;
    accel: vec3<f32>;
    count: atomic<i32>;
    __pad0: vec3<f32>;
    seed: u32;
    __pad1: vec4<f32>;
};

[[group(0), binding(0)]] var<uniform> sim_params : SimParams;
[[group(1), binding(0)]] var<storage, read_write> particles : Particles;
[[group(2), binding(0)]] var<storage, read_write> spawner : Spawner;

var<private> seed : u32 = 0u;

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
    pos: vec3<f32>;
    vel: vec3<f32>;
};

fn init_pos_vel(index: u32) -> PosVel {
    var ret : PosVel;
{{INIT_POS_VEL}}
    return ret;
}

fn init_lifetime() -> f32 {
    return 5.0;
}

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    let total : u32 = arrayLength(&particles.particles);
    let index = global_invocation_id.x;
    if (index >= total) {
        return;
    }

    var vPos : vec3<f32> = particles.particles[index].pos;
    var vVel : vec3<f32> = particles.particles[index].vel;
    var vAge : f32 = particles.particles[index].age;
    var vLifetime : f32 = particles.particles[index].lifetime;

    vAge = vAge + sim_params.dt;
    if (vAge >= vLifetime) {
        // Particle dead; try to recycle into newly-spawned one
        if (atomicSub(&spawner.spawn, 1) > 0) {
            seed = pcg_hash(index ^ spawner.seed);
            var posVel = init_pos_vel(index);
            vPos = posVel.pos + spawner.origin;
            vVel = posVel.vel;
            vAge = 0.0;
            vLifetime = init_lifetime();
        } else {
            return;
        }
    }

    // integration
    vVel = vVel + (spawner.accel * sim_params.dt);

    // kinematic update
    vPos = vPos + (vVel * sim_params.dt);

    // Write back
    atomicAdd(&spawner.count, 1);
    particles.particles[index].pos = vPos;
    particles.particles[index].vel = vVel;
    particles.particles[index].age = vAge;
    particles.particles[index].lifetime = vLifetime;
}