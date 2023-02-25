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