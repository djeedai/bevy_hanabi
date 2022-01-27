struct View {
    view_proj: mat4x4<f32>;
};

struct Particle {
    pos: vec3<f32>;
    age: f32;
    vel: vec3<f32>;
    lifetime: f32;
};

struct Particles {
    particles: [[stride(32)]] array<Particle>;
};

struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] color: vec4<f32>;
#ifdef PARTICLE_TEXTURE
    [[location(1)]] uv: vec2<f32>;
#endif
};

[[group(0), binding(0)]] var<uniform> view: View;
[[group(1), binding(0)]] var<storage, read> particles : Particles;
#ifdef PARTICLE_TEXTURE
[[group(2), binding(0)]] var particle_texture: texture_2d<f32>;
[[group(2), binding(1)]] var particle_sampler: sampler;
#endif
// #ifdef PARTICLE_GRADIENTS
// [[group(3), binding(0)]] var gradient_texture: texture_2d<f32>;
// [[group(3), binding(1)]] var gradient_sampler: sampler;
// #endif

fn color_over_lifetime(life: f32) -> vec4<f32> {
    let c0 = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    let t1 = 0.1;
    let c1 = vec4<f32>(1.0, 1.0, 0.0, 1.0);
    let t2 = 0.4;
    let c2 = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    let c3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (life <= t1) {
        return mix(c0, c1, life / t1);
    } else if (life <= t2) {
        return mix(c1, c2, (life - t1) / (t2 - t1));
    } else {
        return mix(c2, c3, (life - t2) / (1.0 - t2));
    }
}

[[stage(vertex)]]
fn vertex(
    [[builtin(instance_index)]] particle_index: u32,
    [[location(0)]] vertex_position: vec3<f32>,
#ifdef PARTICLE_TEXTURE
    [[location(1)]] vertex_uv: vec2<f32>,
#endif
    //[[location(1)]] vertex_color: u32,
    //[[location(1)]] vertex_velocity: vec3<f32>,
) -> VertexOutput {
    var particle = particles.particles[particle_index];
    var out: VertexOutput;
#ifdef PARTICLE_TEXTURE
    out.uv = vertex_uv;
#endif

{{VERTEX_MODIFIERS}}

    out.position = view.view_proj * vec4<f32>(particle.pos + vertex_position, 1.0);
    //out.color = vec4<f32>((vec4<u32>(vertex_color) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(255u)) / 255.0;
    out.color = color_over_lifetime(particle.age / particle.lifetime);
    // out.color[3] = 1.0;
    // if (particle.age < 0.0) {
    //     out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    // }
    return out;
}

[[stage(fragment)]]
fn fragment(in: VertexOutput) -> [[location(0)]] vec4<f32> {
#ifdef PARTICLE_TEXTURE
    var color = textureSample(particle_texture, particle_sampler, in.uv);
    color = vec4<f32>(1.0, 1.0, 1.0, color.r); // FIXME - grayscale modulate
    color = in.color * color;
#else
    var color = in.color;
#endif
    return color;
}