struct Particle {
    pos: vec3<f32>;
    age: f32;
    vel: vec3<f32>;
    lifetime: f32;
};

struct ParticleBuffer {
    particles: [[stride(32)]] array<Particle>;
};

struct SimParams {
    dt: f32;
    time: f32;
};

struct PullingForceFieldParam {
    position_or_direction: vec3<f32>;
    max_radius: f32;
    min_radius: f32;
    mass: f32;
    force_type: i32;
    conform_to_sphere: f32;
};

struct Spawner {
    origin: vec3<f32>;
    spawn: atomic<i32>;
    accel: vec3<f32>;
    count: atomic<i32>;
    force_field: array<PullingForceFieldParam, 16>;
    __pad0: vec3<f32>;
    seed: u32;
    __pad1: vec4<f32>;
};

struct IndirectBuffer {
    indices: [[stride(4)]] array<u32>;
};

[[group(0), binding(0)]] var<uniform> sim_params : SimParams;
[[group(1), binding(0)]] var<storage, read_write> particle_buffer : ParticleBuffer;
[[group(2), binding(0)]] var<storage, read_write> spawner : Spawner;
[[group(3), binding(0)]] var<storage, read_write> indirect_buffer : IndirectBuffer;

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

fn proj(u: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return dot(v, u) / dot(u,u) * u;
}


[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
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

            // Initialize new particle
            var posVel = init_pos_vel(index);
            vPos = posVel.pos + spawner.origin;
            vVel = posVel.vel;
            vAge = 0.0;
            vLifetime = init_lifetime();
        } else {
            // Nothing to spawn; simply return without writing any update
            return;
        }
    }

{{FORCE_FIELD_CODE}}

//     ///////////// Start of force field computation /////////////

//     // force field acceleration: note that the particles do not have a mass as of yet,
//     // or we could say that the particles all have a mass of one, which means F = 1 * a.
//     var ff_acceleration: vec3<f32> = vec3<f32>(0.0); 
//     var not_conformed_to_sphere: f32 = 1.0;

//     var unit_p2p_conformed: vec3<f32> = vec3<f32>(0.0);
//     var conforming_source: vec3<f32> = vec3<f32>(0.0);
//     var conforming_radius: f32 = 0.0;

//     for (var kk: i32 = 0; kk < 16; kk=kk+1) {
//         // As soon as a field component has a null mass, skip it and all subsequent ones.
//         // Is this better than not having the if statement in the first place?
//         // Likely answer:
//         // The if statement is probably good in this case because all the particles will encounter 
//         // the same number of field components.
//         if (spawner.force_field[kk].mass == 0.0) {
//             break;
//         }

//         let particle_to_point_source = vPos - spawner.force_field[kk].position_or_direction;
//         let distance = length(particle_to_point_source);
//         let unit_p2p = normalize(particle_to_point_source) ;

//         let min_dist_check = step(spawner.force_field[kk].min_radius, distance);
//         let max_dist_check = 1.0 - step(spawner.force_field[kk].max_radius, distance);
//         let force_type_check = 1.0 - step(f32(spawner.force_field[kk].force_type), 0.5); // 1.0 when constant field

//         // this turns into 0 when the field is an attractor and the particle is inside the min_radius and the source
//         // is an attractor.
//         if (spawner.force_field[kk].conform_to_sphere > 0.5) {
//             not_conformed_to_sphere = not_conformed_to_sphere 
//                 * max(min_dist_check, -(sign(spawner.force_field[kk].mass) - 1.0) / 2.0);

//             unit_p2p_conformed = 
//                 unit_p2p_conformed 
//                 + (1.0 - not_conformed_to_sphere) 
//                 * unit_p2p 
//                 * (1.0 - min_dist_check);

//             conforming_source = 
//                 conforming_source 
//                 + (1.0 - not_conformed_to_sphere) 
//                 * spawner.force_field[kk].position_or_direction
//                 * (1.0 - min_dist_check);

//             conforming_radius = conforming_radius 
//                 + (1.0 - not_conformed_to_sphere) 
//                 * spawner.force_field[kk].min_radius / 1.2
//                 * (1.0 - min_dist_check);
//         }

//         let constant_field = (1.0 - force_type_check) * normalize(spawner.force_field[kk].position_or_direction);
        
//         let point_source_force =             
//             - force_type_check * unit_p2p
//             * min_dist_check * max_dist_check
//             * spawner.force_field[kk].mass / 
//                 (0.0000001 + pow(distance, f32(spawner.force_field[kk].force_type)));


//         let force_component = constant_field + point_source_force;
        
//         // if the particle is within the min_radius of a source, then forget about
//         // the other sources and only use the conformed field, thus the "* min_dist_check"
//         ff_acceleration =  ff_acceleration * min_dist_check + force_component;
//     }

//     // conform to a sphere of radius min_radius/2 by projecting the velocity vector
//     // onto a plane that is tangent to the sphere.
//     let eps = vec3<f32>(0.000001);
//     let projected_on_sphere = vVel - proj(unit_p2p_conformed + eps, vVel + eps);
//     let conformed_field = 
//         (1.0 - not_conformed_to_sphere) * normalize(projected_on_sphere) * length(vVel);

//     ///////////// End of force field computation /////////////


//     // // Euler integration
//     vVel = (vVel + (spawner.accel * sim_params.dt)  + (ff_acceleration * sim_params.dt)) 
//         * not_conformed_to_sphere + conformed_field;


//     // let temp_vPos = vPos;
//     vPos = (vPos + (vVel * sim_params.dt));
    

//     // project on the sphere if within conforming distance
//     let pos_to_source = conforming_source - vPos ;
//     let difference = length(pos_to_source) - conforming_radius;
//     vPos = vPos  + difference * normalize(pos_to_source ) * (1.0 - not_conformed_to_sphere) ;

//     // // commented because of the potential bug where dt could be zero, although the simulation
//     // // works anyways, needs investigation
//     // vVel = (vPos - temp_vPos) / sim_params.dt;

    // Increment alive particle count and write indirection index
    let indirect_index = atomicAdd(&spawner.count, 1);
    indirect_buffer.indices[indirect_index] = index;

    // Write back particle itself
    particle_buffer.particles[index].pos = vPos;
    particle_buffer.particles[index].vel = vVel;
    particle_buffer.particles[index].age = vAge;
    particle_buffer.particles[index].lifetime = vLifetime;
}