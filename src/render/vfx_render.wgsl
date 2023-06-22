struct View {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    world_position: vec3<f32>,
    width: f32,
    height: f32,
};

struct Particle {
{{ATTRIBUTES}}
};

struct ParticlesBuffer {
    particles: array<Particle>,
};

struct IndirectBuffer {
    indices: array<u32>,
};

struct DispatchIndirect {
    x: u32,
    y: u32,
    z: u32,
    pong: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
#ifdef PARTICLE_TEXTURE
    @location(1) uv: vec2<f32>,
#endif
};

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var<storage, read> particle_buffer : ParticlesBuffer;
@group(1) @binding(1) var<storage, read> indirect_buffer : IndirectBuffer;
@group(1) @binding(2) var<storage, read> dispatch_indirect : DispatchIndirect;
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
fn rand4() -> vec4<f32> {
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
    out.uv = vertex_uv;
#endif

{{INPUTS}}

{{VERTEX_MODIFIERS}}

    let vpos = vertex_position * vec3<f32>(size.x, size.y, 1.0);
    let world_position = particle.position
        + axis_x * vpos.x
        + axis_y * vpos.y;
    out.position = view.view_proj * vec4<f32>(world_position, 1.0);
    out.color = color;

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {

{{FRAGMENT_MODIFIERS}}

#ifdef PARTICLE_TEXTURE
    var color = textureSample(particle_texture, particle_sampler, in.uv);
    color = vec4<f32>(1.0, 1.0, 1.0, color.r); // FIXME - grayscale modulate
    color = in.color * color;
#else
    var color = in.color;
#endif
    return color;
}