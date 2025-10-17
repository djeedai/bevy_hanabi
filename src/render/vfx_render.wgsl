#import bevy_render::view::View
#import bevy_hanabi::vfx_common::{
    IndirectBuffer, SimParams, Spawner,
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
#ifdef NEEDS_NORMAL
    @location(2) normal: vec3<f32>,
#endif
#ifdef NEEDS_PARTICLE_FRAGMENT
    @location(3) slab_particle_index: u32,
#endif
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> sim_params : SimParams;

@group(1) @binding(0) var<storage, read> particle_buffer : ParticleBuffer;
@group(1) @binding(1) var<storage, read> indirect_buffer : IndirectBuffer;
@group(1) @binding(2) var<storage, read> spawner : Spawner;

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

// Unpacks a compressed transform and transposes is.
fn unpack_compressed_transform_3x3_transpose(compressed_transform: mat3x4<f32>) -> mat3x3<f32> {
    return mat3x3(
        compressed_transform[0].xyz,
        compressed_transform[1].xyz,
        compressed_transform[2].xyz,
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

fn transform_normal_simulation_to_world(sim_normal: vec3<f32>) -> vec3<f32> {
#ifdef LOCAL_SPACE_SIMULATION
    // We use the inverse transpose transform to transform normals.
    // The inverse transpose is the same as the transposed inverse, so we can
    // safely use the inverse transform.
    let transform = unpack_compressed_transform_3x3_transpose(spawner.inverse_transform);
    return transform * sim_normal;
#else
    return sim_normal;
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

fn inverse_transpose_mat3(m: mat3x3<f32>) -> mat3x3<f32> {
    let tmp0 = cross(m[1], m[2]);
    let tmp1 = cross(m[2], m[0]);
    let tmp2 = cross(m[0], m[1]);
    let inv_det = 1.0 / dot(m[2], tmp2);
    return mat3x3<f32>(tmp0 * inv_det, tmp1 * inv_det, tmp2 * inv_det);
}

{{RENDER_EXTRA}}

@vertex
fn vertex(
    @builtin(instance_index) instance_index: u32,
    @location(0) vertex_position: vec3<f32>,
#ifdef NEEDS_UV
    @location(1) vertex_uv: vec2<f32>,
#endif
#ifdef NEEDS_NORMAL
    @location(2) vertex_normal: vec3<f32>,
#endif
    // @location(1) vertex_color: u32,
    // @location(1) vertex_velocity: vec3<f32>,
) -> VertexOutput {
    let base_particle = spawner.slab_offset;

    // Fetch particle
    let indirect_read_index = spawner.render_indirect_read_index;
    let particle_index = indirect_buffer.rows[base_particle + instance_index].particle_index[indirect_read_index];
    var particle = particle_buffer.particles[base_particle + particle_index];

    var out: VertexOutput;

#ifdef NEEDS_PARTICLE_FRAGMENT
    out.slab_particle_index = base_particle + particle_index;
#endif // NEEDS_PARTICLE_FRAGMENT

#ifdef RIBBONS
    // Discard first instance; we draw from second one, and link to previous one
    if (instance_index == 0) {
        out.position = vec4(0.0);
        return out;
    }

    // Fetch previous particle
    let prev_index = indirect_buffer.rows[base_particle + instance_index - 1u].particle_index[indirect_read_index];
    let prev_particle = particle_buffer.particles[base_particle + prev_index];

    // Discard this instance if previous one is from a different ribbon. Again,
    // we draw from second one of each ribbon.
    if (prev_particle.ribbon_id != particle.ribbon_id) {
        out.position = vec4(0.0);
        return out;
    }
#endif  // RIBBONS

#ifdef NEEDS_UV
    // Compute UVs
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
    var delta = particle.position - prev_particle.position;

    axis_x = normalize(delta);
    axis_y = normalize(cross(axis_x, axis_z));
    axis_z = cross(axis_x, axis_y);

    position = mix(particle.position, prev_particle.position, 0.5);
    size = vec3(length(delta), size.y, 1.0);
#endif  // RIBBONS

    // Expand particle mesh vertex based on particle position ("origin"), and local
    // orientation and size of the particle mesh.
    let vpos = vertex_position * size;
    let sim_position = position + axis_x * vpos.x + axis_y * vpos.y + axis_z * vpos.z;
    out.position = transform_position_simulation_to_clip(sim_position);

    out.color = color;

#ifdef NEEDS_NORMAL
    let normal = inverse_transpose_mat3(mat3x3(axis_x, axis_y, axis_z)) * vertex_normal;
    out.normal = transform_normal_simulation_to_world(normal);
#endif  // NEEDS_NORMAL

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Read fragment inputs
#ifdef USE_ALPHA_MASK
    var alpha_cutoff: f32 = {{ALPHA_CUTOFF}};
#endif
    var color = in.color;
#ifdef NEEDS_UV
    var uv = in.uv;
#endif
#ifdef NEEDS_NORMAL
    var normal = in.normal;
#endif
#ifdef NEEDS_PARTICLE_FRAGMENT
    var particle = particle_buffer.particles[in.slab_particle_index];
#endif // NEEDS_PARTICLE_FRAGMENT

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