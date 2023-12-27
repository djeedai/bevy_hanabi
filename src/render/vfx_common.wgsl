#define_import_path bevy_hanabi::vfx_common

struct SimParams {
    delta_time: f32,
    time: f32,
//#ifdef SIM_PARAMS_INDIRECT_DATA
    num_effects: u32,
    render_stride: u32,
    dispatch_stride: u32,
//#endif
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
    count: atomic<i32>,
    effect_index: u32,
    force_field: array<ForceFieldSource, 16>,
#ifdef SPAWNER_PADDING
    {{SPAWNER_PADDING}}
#endif
}