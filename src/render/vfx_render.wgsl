#import bevy_render::view::View
#import bevy_hanabi::vfx_common::{
    DispatchIndirect, IndirectBuffer, SimParams, Spawner,
    seed, tau, pcg_hash, to_float01, frand, frand2, frand3, frand4,
    rand_uniform_f, rand_uniform_vec2, rand_uniform_vec3, rand_uniform_vec4,
    rand_normal_f, rand_normal_vec2, rand_normal_vec3, rand_normal_vec4, proj
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
#ifdef NEEDS_UV
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
{{MATERIAL_BINDINGS}}

fn get_camera_position_effect_space() -> vec3<f32> {
    let view_pos = view.world_from_view[3].xyz;
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
    let view_rot = mat3x3(view.world_from_view[0].xyz, view.world_from_view[1].xyz, view.world_from_view[2].xyz);
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
    return view.clip_from_world * transform_position_simulation_to_world(sim_position);
}

{{RENDER_EXTRA}}

@vertex
fn vertex(
    @builtin(instance_index) instance_index: u32,
    @location(0) vertex_position: vec3<f32>,
#ifdef NEEDS_UV
    @location(1) vertex_uv: vec2<f32>,
#endif
    // @location(1) vertex_color: u32,
    // @location(1) vertex_velocity: vec3<f32>,
) -> VertexOutput {
    let pong = dispatch_indirect.pong;
    let index = indirect_buffer.indices[3u * instance_index + pong];
    var particle = particle_buffer.particles[index];
    var out: VertexOutput;
#ifdef NEEDS_UV
    var uv = vertex_uv;
#ifdef FLIPBOOK
    let row_count = {{FLIPBOOK_ROW_COUNT}};
    let ij = vec2<f32>(f32(particle.sprite_index % row_count), f32(particle.sprite_index / row_count));
    uv = (ij + uv) * {{FLIPBOOK_SCALE}};
#endif
    out.uv = uv;
#endif  // NEEDS_UV

{{INPUTS}}

{{VERTEX_MODIFIERS}}

#ifdef RIBBONS
    let next_index = particle.next;
    if (next_index >= arrayLength(&particle_buffer.particles)) {
        out.position = vec4(0.0);
        return out;
    }

    let next_particle = particle_buffer.particles[next_index];
    var delta = next_particle.position - particle.position;

    axis_x = normalize(delta);
    axis_y = normalize(cross(axis_x, axis_z));
    axis_z = cross(axis_x, axis_y);

    position = mix(next_particle.position, particle.position, 0.5);
    size = vec2(length(delta), size.y);
#endif  // RIBBONS

    // Expand particle mesh vertex based on particle position ("origin"), and local
    // orientation and size of the particle mesh (currently: only quad).
    let vpos = vertex_position * vec3<f32>(size.x, size.y, 1.0);
    let sim_position = position + axis_x * vpos.x + axis_y * vpos.y;
    out.position = transform_position_simulation_to_clip(sim_position);

    out.color = color;

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {

#ifdef USE_ALPHA_MASK
    var alpha_cutoff: f32 = {{ALPHA_CUTOFF}};
#endif
    var color = in.color;
#ifdef NEEDS_UV
    var uv = in.uv;
#endif

{{FRAGMENT_MODIFIERS}}

#ifdef USE_ALPHA_MASK
    if color.a >= alpha_cutoff {
        color.a = 1.0;
    } else {
        discard;
    }
#endif

    return color;
}