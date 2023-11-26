#import bevy_render::view::View

struct Particle {
{{ATTRIBUTES}}
}

struct ParticlesBuffer {
    particles: array<Particle>,
}

struct SimParams {
    delta_time: f32,
    time: f32,
};

struct IndirectBuffer {
    indices: array<u32>,
}

struct DispatchIndirect {
    x: u32,
    y: u32,
    z: u32,
    pong: u32,
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
    count: i32,
    effect_index: u32,
    force_field: array<ForceFieldSource, 16>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
#ifdef PARTICLE_TEXTURE
    @location(1) uv: vec2<f32>,
#endif
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> sim_params : SimParams;
@group(1) @binding(0) var<storage, read> particle_buffer : ParticlesBuffer;
@group(1) @binding(1) var<storage, read> indirect_buffer : IndirectBuffer;
@group(1) @binding(2) var<storage, read> dispatch_indirect : DispatchIndirect;
#ifdef LOCAL_SPACE_SIMULATION
@group(1) @binding(3) var<storage, read> spawner : Spawner; // NOTE - same group as update
#endif
#ifdef PARTICLE_TEXTURE
@group(2) @binding(0) var particle_texture: texture_2d<f32>;
@group(2) @binding(1) var particle_sampler: sampler;
#endif
// #ifdef PARTICLE_GRADIENTS
// @group(3) @binding(0) var gradient_texture: texture_2d<f32>;
// @group(3) @binding(1) var gradient_sampler: sampler;
// #endif

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

fn get_camera_position_effect_space() -> vec3<f32> {
    let view_pos = view.view[3].xyz;
#ifdef LOCAL_SPACE_SIMULATION
    let inverse_transform = transpose(
        mat3x3(
            spawner.inverse_transform[0].xyz,
            spawner.inverse_transform[1].xyz,
            spawner.inverse_transform[2].xyz,
        )
    );
    return inverse_transform * view_pos;
#else
    return view_pos;
#endif
}

fn get_camera_rotation_effect_space() -> mat3x3<f32> {
    let view_rot = mat3x3(view.view[0].xyz, view.view[1].xyz, view.view[2].xyz);
#ifdef LOCAL_SPACE_SIMULATION
    let inverse_transform = transpose(
        mat3x3(
            spawner.inverse_transform[0].xyz,
            spawner.inverse_transform[1].xyz,
            spawner.inverse_transform[2].xyz,
        )
    );
    return inverse_transform * view_rot;
#else
    return view_rot;
#endif
}

{{RENDER_EXTRA}}

@vertex
fn vertex(
    @builtin(instance_index) instance_index: u32,
    @location(0) vertex_position: vec3<f32>,
#ifdef PARTICLE_TEXTURE
    @location(1) vertex_uv: vec2<f32>,
#endif
    // @location(1) vertex_color: u32,
    // @location(1) vertex_velocity: vec3<f32>,
) -> VertexOutput {
    let pong = dispatch_indirect.pong;
    let index = indirect_buffer.indices[3u * instance_index + pong];
    var particle = particle_buffer.particles[index];
    var out: VertexOutput;
#ifdef PARTICLE_TEXTURE
    var uv = vertex_uv;
#ifdef FLIPBOOK
    let row_count = {{FLIPBOOK_ROW_COUNT}};
    let ij = vec2<f32>(f32(particle.sprite_index % row_count), f32(particle.sprite_index / row_count));
    uv = (ij + uv) * {{FLIPBOOK_SCALE}};
#endif
    out.uv = uv;
#endif

{{INPUTS}}

{{VERTEX_MODIFIERS}}

#ifdef LOCAL_SPACE_SIMULATION
    let transform = transpose(
        mat4x4(
            spawner.transform[0],
            spawner.transform[1],
            spawner.transform[2],
            vec4<f32>(0.0, 0.0, 0.0, 1.0)
        )
    );
#endif

#ifdef PARTICLE_SCREEN_SPACE_SIZE
    let half_screen = view.viewport.zw / 2.;
    let vpos = vertex_position * vec3<f32>(size.x / half_screen.x, size.y / half_screen.y, 1.0);
    let local_position = particle.position;
    let world_position = {{SIMULATION_SPACE_TRANSFORM_PARTICLE}};
    out.position = view.view_proj * world_position + vec4<f32>(vpos, 0.0);
#else
    let vpos = vertex_position * vec3<f32>(size.x, size.y, 1.0);
    let local_position = particle.position
        + axis_x * vpos.x
        + axis_y * vpos.y;
    let world_position = {{SIMULATION_SPACE_TRANSFORM_PARTICLE}};
    out.position = view.view_proj * world_position;
#endif

    out.color = color;

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {

#ifdef USE_ALPHA_MASK
    var alpha_cutoff: f32 = {{ALPHA_CUTOFF}};
#endif

{{FRAGMENT_MODIFIERS}}

    var color = in.color;

#ifdef PARTICLE_TEXTURE
    var texColor = textureSample(particle_texture, particle_sampler, in.uv);
    {{PARTICLE_TEXTURE_SAMPLE_MAPPING}}
#endif

#ifdef USE_ALPHA_MASK
    if color.a >= alpha_cutoff {
        color.a = 1.0;
    } else {
        discard;
    }
#endif

    return color;
}