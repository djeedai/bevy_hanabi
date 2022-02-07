// Adapted from https://github.com/gfx-rs/wgpu/blob/master/wgpu/examples/boids/compute.wgsl

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
};

[[group(0), binding(0)]] var<uniform> sim_params : SimParams;
[[group(1), binding(0)]] var<storage, read_write> particles : Particles;
[[group(2), binding(0)]] var<storage, read_write> spawner : Spawner;

var<private> seed : f32 = 0.0;

// Simplistic pseudo-RNG
fn rand(index: u32) -> f32 {
    seed = sin(sim_params.time * 1.75489 + seed + f32(index) * 94.54) * 4646.0548;
    seed = sin(sim_params.time * 0.7548789 + seed * 0.115487 + f32(index) * 0.65457);
    return seed;
}

struct PosVel {
    pos: vec3<f32>;
    vel: vec3<f32>;
};

fn init_pos_vel(index: u32) -> PosVel {
    var ret : PosVel;
    // Sphere radius
    var r = 2.0;
    // Radial speed
    var speed = 2.0;
    // Spawn randomly along the sphere surface
    var dir = vec3<f32>(rand(index), rand(index), rand(index));
    dir = normalize(dir);
    ret.pos = dir * r;
    // Radial speed away from sphere center
    ret.vel = dir * speed;
    return ret;
}

fn init_lifetime() -> f32 {
    return 5.0;
}

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
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