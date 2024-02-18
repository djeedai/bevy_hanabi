#import bevy_render::view::View
#import bevy_hanabi::vfx_common::{
    DispatchIndirect, ForceFieldSource, IndirectBuffer, SimParams, Spawner,
    seed, tau, pcg_hash, to_float01, frand, frand2, frand3, frand4,
    rand_uniform, proj
}

struct Particle {
{{ATTRIBUTES}}
}

struct ParticleBuffer {
    particles: array<Particle>,
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
@group(1) @binding(0) var<storage, read> particle_buffer : ParticleBuffer;
@group(1) @binding(1) var<storage, read> indirect_buffer : IndirectBuffer;
@group(1) @binding(2) var<storage, read> dispatch_indirect : DispatchIndirect;
#ifdef RENDER_NEEDS_SPAWNER
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

/// Unpack a compressed transform stored in transposed row-major form.
fn unpack_compressed_transform(compressed_transform: mat3x4<f32>) -> mat4x4<f32> {
    return transpose(
        mat4x4(
            compressed_transform[0],
            compressed_transform[1],
            compressed_transform[2],
            vec4<f32>(0.0, 0.0, 0.0, 1.0)
        )
    );
}

/// Transform a simulation space position into a world space position.
///
/// The simulation space depends on the effect's SimulationSpace value, and is either
/// the effect space (SimulationSpace::Local) or the world space (SimulationSpace::Global).
fn transform_position_simulation_to_world(sim_position: vec3<f32>) -> vec4<f32> {
#ifdef LOCAL_SPACE_SIMULATION
    let transform = unpack_compressed_transform(spawner.transform);
    return transform * vec4<f32>(sim_position, 1.0);
#else
    return vec4<f32>(sim_position, 1.0);
#endif
}

/// Transform a simulation space position into a clip space position.
///
/// The simulation space depends on the effect's SimulationSpace value, and is either
/// the effect space (SimulationSpace::Local) or the world space (SimulationSpace::Global).
/// The clip space is the final [-1:1]^3 space output from the vertex shader, before
/// perspective divide and viewport transform are applied.
fn transform_position_simulation_to_clip(sim_position: vec3<f32>) -> vec4<f32> {
    return view.view_proj * transform_position_simulation_to_world(sim_position);
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

#ifdef PARTICLE_SCREEN_SPACE_SIZE
    // Get perspective divide factor from clip space position. This is the "average" factor for the entire
    // particle, taken at its position (mesh origin), and applied uniformly for all vertices.
    let w_cs = transform_position_simulation_to_clip(particle.position).w;
    // Scale size by w_cs to negate the perspective divide which will happen later after the vertex shader.
    // The 2.0 factor is because clip space is in [-1:1] so we need to divide by the half screen size only.
    let screen_size_pixels = view.viewport.zw;
    let projection_scale = vec2<f32>(view.projection[0][0], view.projection[1][1]);
    size = (size * w_cs * 2.0) / min(screen_size_pixels.x * projection_scale.x, screen_size_pixels.y * projection_scale.y);
#endif

    // Expand particle mesh vertex based on particle position ("origin"), and local
    // orientation and size of the particle mesh (currently: only quad).
    let vpos = vertex_position * vec3<f32>(size.x, size.y, 1.0);
    let sim_position = particle.position
        + axis_x * vpos.x
        + axis_y * vpos.y;
    out.position = transform_position_simulation_to_clip(sim_position);

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