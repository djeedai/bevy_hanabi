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

// fn color_over_lifetime(life: f32) -> vec4<f32> {
//     let c0 = vec4<f32>(1.0, 1.0, 1.0, 1.0);
//     let t1 = 0.1;
//     let c1 = vec4<f32>(1.0, 1.0, 0.0, 1.0);
//     let t2 = 0.4;
//     let c2 = vec4<f32>(1.0, 0.0, 0.0, 1.0);
//     let c3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
//     if (life <= t1) {
//         return mix(c0, c1, life / t1);
//     } else if (life <= t2) {
//         return mix(c1, c2, (life - t1) / (t2 - t1));
//     } else {
//         return mix(c2, c3, (life - t2) / (1.0 - t2));
//     }
// }

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

    var size = vec2<f32>(1.0, 1.0);

{{VERTEX_MODIFIERS}}

    out.position = view.view_proj * world_position;

    //out.color = vec4<f32>((vec4<u32>(vertex_color) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(255u)) / 255.0;
    //out.color = color_over_lifetime(particle.age / particle.lifetime);
    // out.color[3] = 1.0;
    // if (particle.age < 0.0) {
    //     out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    // }
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
#ifdef PARTICLE_TEXTURE
    var color = textureSample(particle_texture, particle_sampler, in.uv);
    color = vec4<f32>(1.0, 1.0, 1.0, color.r); // FIXME - grayscale modulate
    color = in.color * color;
#else
    var color = in.color;
#endif
    return color;
}