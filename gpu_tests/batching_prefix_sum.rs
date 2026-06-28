//! Headless GPU regression tests for batching dataflow contracts.

use std::borrow::Cow;

use bevy::prelude::Vec3;
use bevy_hanabi::prelude::{
    Attribute, EffectAsset, EffectShaderSources, ExprWriter, SetAttributeModifier, SpawnerSettings,
};
use bytemuck::{cast_slice, Pod, Zeroable};
use futures::channel::oneshot;
use futures::executor::block_on;
use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct BatchInfo {
    total_spawn_count: u32,
    total_update_count: u32,
    base_effect: u32,
    base_particle: u32,
    prefix_sum_offset: u32,
    prefix_sum_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SimParamsGpu {
    delta_time: f32,
    time: f32,
    virtual_delta_time: f32,
    virtual_time: f32,
    real_delta_time: f32,
    real_time: f32,
    num_effects: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DrawIndexedIndirectArgsGpu {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SpawnerGpu {
    transform: [[f32; 4]; 3],
    inverse_transform: [[f32; 4]; 3],
    spawn: i32,
    seed: u32,
    render_indirect_read_index: u32,
    effect_metadata_index: u32,
    draw_indirect_index: u32,
    slab_offset: u32,
    parent_slab_offset: u32,
    unused: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct EffectMetadataGpu {
    capacity: u32,
    alive_count: u32,
    max_update: u32,
    max_spawn: u32,
    indirect_write_index: u32,
    indirect_render_index: u32,
    init_indirect_dispatch_index: u32,
    properties_array_index: u32,
    local_child_index: u32,
    global_child_index: u32,
    base_child_index: u32,
    particle_stride: u32,
    sort_key_offset: u32,
    sort_key2_offset: u32,
    particle_counter: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ParticleGpu {
    position: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct IndirectEntryGpu {
    particle_index: [u32; 2],
    dead_index: u32,
}

/// Local copy of test_utils::MockRenderer::new() adapted for this headless
/// test.
struct MockRenderer {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl MockRenderer {
    /// Create a new mock renderer.
    ///
    /// If the `gpu_tests` feature is enabled, this creates a real GPU adapter
    /// and device. Otherwise this can create any kind of adapter and device.
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        #[cfg(debug_assertions)]
        let flags = wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION;
        #[cfg(not(debug_assertions))]
        let flags = wgpu::InstanceFlags::empty();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags,
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });

        #[cfg(feature = "gpu_tests")]
        eprintln!("Requesting real GPU adapter.");
        #[cfg(not(feature = "gpu_tests"))]
        eprintln!("Requesting headless adapter.");

        #[cfg(feature = "gpu_tests")]
        let request_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        };
        #[cfg(not(feature = "gpu_tests"))]
        let request_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface: None,
        };

        let adapter = instance.request_adapter(&request_options).await?;

        #[cfg(feature = "gpu_tests")]
        {
            let adapter_info = adapter.get_info();
            if matches!(
                adapter_info.device_type,
                wgpu::DeviceType::Cpu | wgpu::DeviceType::Other
            ) {
                return Err(format!(
                    "gpu_tests requires a real GPU adapter, got {:?} ({})",
                    adapter_info.device_type, adapter_info.name
                )
                .into());
            }
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                label: Some("hanabi_test_device"),
                trace: wgpu::Trace::Off,
            })
            .await?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

/// Submit the given command buffer and wait for completion.
async fn submit_and_wait(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    command_buffer: wgpu::CommandBuffer,
) {
    queue.submit([command_buffer]);
    let (tx, rx) = oneshot::channel();
    queue.on_submitted_work_done(move || {
        let _ = tx.send(());
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let _ = rx.await;
}

async fn readback_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    bytes: u64,
) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
    submit_and_wait(device, queue, encoder.finish()).await;

    let slice = staging.slice(..);
    let (tx, rx) = oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.await.unwrap().unwrap();
    let data = slice.get_mapped_range();
    let out = cast_slice::<u8, u32>(&data).to_vec();
    drop(data);
    staging.unmap();
    out
}

/// Create a WGPU shader module from the given source code.
fn create_shader_module(
    device: &wgpu::Device,
    shader_source: &str,
    file_path: &str,
) -> Result<wgpu::ShaderModule, Box<dyn std::error::Error>> {
    // vfx_common is templated in the crate and normally materialized by the plugin
    // with runtime alignment-dependent padding. For this headless test we only need
    // symbols imported by vfx_prefix_sum, so make the template parseable directly.
    let common_code = include_str!("../src/render/vfx_common.wgsl")
        .replace("{{SPAWNER_PADDING}}", "")
        .replace("{{BATCH_INFO_PADDING}}", "")
        .replace("{{EFFECT_METADATA_PADDING}}", "")
        .replace("{{EFFECT_METADATA_STRIDE}}", "64");

    let mut composer = Composer::default();
    composer.add_composable_module(ComposableModuleDescriptor {
        source: &common_code,
        file_path: "bevy_hanabi::vfx_common",
        ..Default::default()
    })?;
    let module = composer.make_naga_module(NagaModuleDescriptor {
        source: shader_source,
        file_path,
        shader_defs: Default::default(),
        ..Default::default()
    })?;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(file_path),
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });
    Ok(shader)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    block_on(async move {
        let renderer = match MockRenderer::new().await {
            Ok(renderer) => renderer,
            Err(err) => {
                #[cfg(feature = "gpu_tests")]
                return Err(err);
                #[cfg(not(feature = "gpu_tests"))]
                {
                    eprintln!("No adapter available; skipping headless batching GPU test.");
                    let _ = err;
                    return Ok(());
                }
            }
        };
        let device = renderer.device;
        let queue = renderer.queue;

        // ------------------------------------------------------------------
        // Test 1: Prefix sum pass dataflow
        // ------------------------------------------------------------------
        let prefix_shader = create_shader_module(
            &device,
            include_str!("../src/render/vfx_prefix_sum.wgsl"),
            "bevy_hanabi::vfx_prefix_sum",
        )?;

        let batches = vec![
            BatchInfo {
                total_spawn_count: 0,
                total_update_count: 0,
                base_effect: 0,
                base_particle: 100,
                prefix_sum_offset: 0,
                prefix_sum_count: 3,
            },
            BatchInfo {
                total_spawn_count: 0,
                total_update_count: 0,
                base_effect: 3,
                base_particle: 500,
                prefix_sum_offset: 3,
                prefix_sum_count: 1,
            },
        ];
        let prefix_init: Vec<u32> = vec![10, 5, 8, 6];
        let dispatch_init = vec![DispatchIndirectArgs { x: 0, y: 0, z: 0 }; 2];

        let batch_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("batch_buffer"),
            contents: cast_slice(&batches),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let prefix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prefix_buffer"),
            contents: cast_slice(&prefix_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let dispatch_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dispatch_buffer"),
            contents: cast_slice(&dispatch_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefix_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: batch_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prefix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dispatch_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prefix_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prefix_pipeline"),
            layout: Some(&pipeline_layout),
            module: &prefix_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("prefix_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        submit_and_wait(&device, &queue, encoder.finish()).await;

        let prefix_out = readback_u32(
            &device,
            &queue,
            &prefix_buffer,
            (prefix_init.len() * 4) as u64,
        )
        .await;
        assert_eq!(prefix_out, vec![0, 10, 15, 0], "prefix sum output mismatch");

        let dispatch_out = readback_u32(
            &device,
            &queue,
            &dispatch_buffer,
            (dispatch_init.len() * std::mem::size_of::<DispatchIndirectArgs>()) as u64,
        )
        .await;
        assert_eq!(dispatch_out[0], 1);
        assert_eq!(dispatch_out[3], 1);

        // ------------------------------------------------------------------
        // Test 2: Location mapping contract (base_particle + prefix_sum[mid])
        // ------------------------------------------------------------------
        let locate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("location_shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct BatchInfo {
    base_particle: u32,
    prefix_sum_offset: u32,
    prefix_sum_count: u32,
    _pad0: u32,
}
@group(0) @binding(0) var<storage, read> batch_info : BatchInfo;
@group(0) @binding(1) var<storage, read> prefix_sum : array<u32>;
@group(0) @binding(2) var<storage, read> slab_indices : array<u32>;
@group(0) @binding(3) var<storage, read_write> out_effect_index : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&slab_indices)) { return; }
    let slab_particle_index = slab_indices[i];
    var lo = batch_info.prefix_sum_offset;
    var hi = lo + batch_info.prefix_sum_count;
    while (lo < hi) {
        let mid = (hi + lo) >> 1u;
        let base_particle = batch_info.base_particle + prefix_sum[mid];
        if (slab_particle_index >= base_particle) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let effect_index = lo - 1u - batch_info.prefix_sum_offset;
    out_effect_index[i] = effect_index;
}
"#
                .into(),
            ),
        });

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct LocateBatch {
            base_particle: u32,
            prefix_sum_offset: u32,
            prefix_sum_count: u32,
            _pad0: u32,
        }
        let locate_batch = LocateBatch {
            base_particle: 100,
            prefix_sum_offset: 0,
            prefix_sum_count: 3,
            _pad0: 0,
        };
        let locate_prefix: Vec<u32> = vec![0, 10, 15];
        let slab_indices: Vec<u32> = vec![100, 109, 110, 114, 115, 122];
        let out_zero: Vec<u32> = vec![0; slab_indices.len()];

        let locate_batch_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("locate_batch"),
            contents: cast_slice(&[locate_batch]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let locate_prefix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("locate_prefix"),
            contents: cast_slice(&locate_prefix),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let slab_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("slab_indices"),
            contents: cast_slice(&slab_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("out_effect"),
            contents: cast_slice(&out_zero),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let locate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("locate_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let locate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("locate_bg"),
            layout: &locate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: locate_batch_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: locate_prefix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: slab_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });
        let locate_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("locate_pl"),
            bind_group_layouts: &[Some(&locate_bgl)],
            immediate_size: 0,
        });
        let locate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("locate_pipeline"),
            layout: Some(&locate_pl),
            module: &locate_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("locate_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("locate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&locate_pipeline);
            pass.set_bind_group(0, &locate_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        submit_and_wait(&device, &queue, encoder.finish()).await;
        let out_effect = readback_u32(
            &device,
            &queue,
            &out_buffer,
            (slab_indices.len() * 4) as u64,
        )
        .await;
        assert_eq!(
            out_effect,
            vec![0, 0, 1, 1, 2, 2],
            "effect index mapping mismatch"
        );

        // ------------------------------------------------------------------
        // Test 3: indirect + prefix-sum + update-style routing across instances
        // ------------------------------------------------------------------
        let indirect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("indirect_stage_shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct Spawner {
    effect_metadata_index: u32,
    draw_indirect_index: u32,
}
struct EffectMetadata {
    alive_count: u32,
    max_update: u32,
    indirect_dispatch_index: u32,
}
struct DispatchIndirectArgs {
    x: u32, y: u32, z: u32,
}
@group(0) @binding(0) var<storage, read_write> spawner_buffer : array<Spawner>;
@group(0) @binding(1) var<storage, read_write> effect_metadata_buffer : array<EffectMetadata>;
@group(0) @binding(2) var<storage, read_write> prefix_sum : array<u32>;
@group(0) @binding(3) var<storage, read_write> dispatch_buffer : array<DispatchIndirectArgs>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&spawner_buffer)) { return; }
    let sp = spawner_buffer[i];
    let em = effect_metadata_buffer[sp.effect_metadata_index];
    prefix_sum[i] = em.alive_count;
    effect_metadata_buffer[sp.effect_metadata_index].max_update = em.alive_count;
    let ddi = em.indirect_dispatch_index;
    dispatch_buffer[ddi].x = (em.alive_count + 63u) >> 6u;
    dispatch_buffer[ddi].y = 1u;
    dispatch_buffer[ddi].z = 1u;
}
"#
                .into(),
            ),
        });
        let update_route_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("update_route_shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct BatchInfo {
    total_update_count: u32,
    base_effect: u32,
    base_particle: u32,
    prefix_sum_offset: u32,
    prefix_sum_count: u32,
}
@group(0) @binding(0) var<storage, read> prefix_sum : array<u32>;
@group(0) @binding(1) var<storage, read> batch_info : BatchInfo;
@group(0) @binding(2) var<storage, read_write> routed_count : array<atomic<u32>>;

fn find_effect(slab_particle_index: u32) -> u32 {
    var lo = batch_info.prefix_sum_offset;
    var hi = lo + batch_info.prefix_sum_count;
    while (lo < hi) {
        let mid = (hi + lo) >> 1u;
        let base_particle = batch_info.base_particle + prefix_sum[mid];
        if (slab_particle_index >= base_particle) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo - 1u - batch_info.prefix_sum_offset;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= batch_info.total_update_count) { return; }
    let slab_particle_index = batch_info.base_particle + i;
    let effect_index = find_effect(slab_particle_index);
    atomicAdd(&routed_count[effect_index], 1u);
}
"#
                .into(),
            ),
        });

        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct SpawnerLite {
            effect_metadata_index: u32,
            draw_indirect_index: u32,
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct EffectMetadataLite {
            alive_count: u32,
            max_update: u32,
            indirect_dispatch_index: u32,
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct BatchInfoLite {
            total_update_count: u32,
            base_effect: u32,
            base_particle: u32,
            prefix_sum_offset: u32,
            prefix_sum_count: u32,
        }

        let spawners = vec![
            SpawnerLite {
                effect_metadata_index: 0,
                draw_indirect_index: 0,
            },
            SpawnerLite {
                effect_metadata_index: 1,
                draw_indirect_index: 1,
            },
        ];
        let em = vec![
            EffectMetadataLite {
                alive_count: 7,
                max_update: 0,
                indirect_dispatch_index: 0,
            },
            EffectMetadataLite {
                alive_count: 5,
                max_update: 0,
                indirect_dispatch_index: 1,
            },
        ];
        let mut prefix = vec![0_u32; 2];
        let ddi_zero = vec![DispatchIndirectArgs { x: 0, y: 0, z: 0 }; 2];

        let sp_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spawners_lite"),
            contents: cast_slice(&spawners),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let em_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("effect_meta_lite"),
            contents: cast_slice(&em),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let prefix_buffer2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prefix_lite"),
            contents: cast_slice(&prefix),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let ddi_buffer2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ddi_lite"),
            contents: cast_slice(&ddi_zero),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bgl_i = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("indirect_lite_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bg_i = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("indirect_lite_bg"),
            layout: &bgl_i,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: em_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: prefix_buffer2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ddi_buffer2.as_entire_binding(),
                },
            ],
        });
        let pl_i = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("indirect_lite_pl"),
            bind_group_layouts: &[Some(&bgl_i)],
            immediate_size: 0,
        });
        let pipe_i = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("indirect_lite_pipe"),
            layout: Some(&pl_i),
            module: &indirect_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("indirect_lite_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("indirect_lite_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipe_i);
            pass.set_bind_group(0, &bg_i, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        submit_and_wait(&device, &queue, encoder.finish()).await;

        prefix = readback_u32(&device, &queue, &prefix_buffer2, 8).await;
        assert_eq!(
            prefix,
            vec![7, 5],
            "indirect stage did not write per-instance alive counts"
        );

        // CPU-side emulation of vfx_prefix_sum for this single batch: [7,5] -> [0,7]
        let prefix_after = vec![0_u32, 7_u32];
        queue.write_buffer(&prefix_buffer2, 0, cast_slice(&prefix_after));
        let batch_lite = BatchInfoLite {
            total_update_count: 12,
            base_effect: 0,
            base_particle: 100,
            prefix_sum_offset: 0,
            prefix_sum_count: 2,
        };
        let batch_lite_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("batch_lite"),
            contents: cast_slice(&[batch_lite]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let routed_zero = vec![0_u32; 2];
        let routed_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("routed"),
            contents: cast_slice(&routed_zero),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bgl_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("update_route_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_route_bg"),
            layout: &bgl_u,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prefix_buffer2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: batch_lite_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: routed_buffer.as_entire_binding(),
                },
            ],
        });
        let pl_u = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("update_route_pl"),
            bind_group_layouts: &[Some(&bgl_u)],
            immediate_size: 0,
        });
        let pipe_u = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_route_pipe"),
            layout: Some(&pl_u),
            module: &update_route_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("update_route_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update_route_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipe_u);
            pass.set_bind_group(0, &bg_u, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        submit_and_wait(&device, &queue, encoder.finish()).await;

        let routed = readback_u32(&device, &queue, &routed_buffer, 8).await;
        assert_eq!(
            routed,
            vec![7, 5],
            "update routing collapsed instances unexpectedly"
        );

        // ------------------------------------------------------------------
        // Test 4: real generated vfx_update shader routes and counts instances
        // ------------------------------------------------------------------
        let writer = ExprWriter::new();
        let init_pos =
            SetAttributeModifier::new(Attribute::POSITION, writer.lit(Vec3::ZERO).expr());
        let module = writer.finish();
        let asset = EffectAsset::new(8, SpawnerSettings::rate(0.0.into()), module).init(init_pos);
        let sources = EffectShaderSources::generate(&asset, None, 0)?;
        let update_shader = create_shader_module(
            &device,
            &sources.update_shader_source,
            "bevy_hanabi::generated_vfx_update",
        )?;

        let sim_params = SimParamsGpu {
            delta_time: 1.0,
            time: 0.0,
            virtual_delta_time: 1.0,
            virtual_time: 0.0,
            real_delta_time: 1.0,
            real_time: 0.0,
            num_effects: 2,
            _pad0: 0,
        };
        let draw_indirect_init = vec![
            DrawIndexedIndirectArgsGpu {
                index_count: 0,
                instance_count: 0,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            },
            DrawIndexedIndirectArgsGpu {
                index_count: 0,
                instance_count: 0,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            },
        ];
        let particles = vec![ParticleGpu::zeroed(); 8];
        let mut indirect_rows = vec![IndirectEntryGpu::zeroed(); 8];
        indirect_rows[0].particle_index = [0, 0];
        indirect_rows[1].particle_index = [0, 1];
        indirect_rows[4].particle_index = [0, 0];
        let spawners = vec![
            SpawnerGpu {
                transform: [[0.0; 4]; 3],
                inverse_transform: [[0.0; 4]; 3],
                spawn: 0,
                seed: 1,
                render_indirect_read_index: 0,
                effect_metadata_index: 0,
                draw_indirect_index: 0,
                slab_offset: 0,
                parent_slab_offset: u32::MAX,
                unused: 0,
            },
            SpawnerGpu {
                transform: [[0.0; 4]; 3],
                inverse_transform: [[0.0; 4]; 3],
                spawn: 0,
                seed: 2,
                render_indirect_read_index: 0,
                effect_metadata_index: 1,
                draw_indirect_index: 1,
                slab_offset: 4,
                parent_slab_offset: u32::MAX,
                unused: 0,
            },
        ];
        let prefix_sum_update = vec![0_u32, 2_u32];
        let batch_info_update = BatchInfo {
            total_spawn_count: 0,
            total_update_count: 3,
            base_effect: 0,
            base_particle: 0,
            prefix_sum_offset: 0,
            prefix_sum_count: 2,
        };
        let effect_metadata = vec![
            EffectMetadataGpu {
                capacity: 8,
                alive_count: 2,
                max_update: 2,
                max_spawn: 0,
                indirect_write_index: 0,
                indirect_render_index: 0,
                init_indirect_dispatch_index: 0,
                properties_array_index: 0,
                local_child_index: 0,
                global_child_index: 0,
                base_child_index: 0,
                particle_stride: 4,
                sort_key_offset: 0,
                sort_key2_offset: 0,
                particle_counter: 0,
            },
            EffectMetadataGpu {
                capacity: 8,
                alive_count: 1,
                max_update: 1,
                max_spawn: 0,
                indirect_write_index: 0,
                indirect_render_index: 1,
                init_indirect_dispatch_index: 0,
                properties_array_index: 0,
                local_child_index: 0,
                global_child_index: 0,
                base_child_index: 0,
                particle_stride: 4,
                sort_key_offset: 0,
                sort_key2_offset: 0,
                particle_counter: 0,
            },
        ];

        let sim_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim_params_update"),
            contents: cast_slice(&[sim_params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let draw_indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("draw_indirect_update"),
            contents: cast_slice(&draw_indirect_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle_update"),
            contents: cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("indirect_update"),
            contents: cast_slice(&indirect_rows),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let spawner_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spawner_update"),
            contents: cast_slice(&spawners),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let prefix_buffer_update = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prefix_update"),
            contents: cast_slice(&prefix_sum_update),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let batch_info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("batch_info_update"),
            contents: cast_slice(&[batch_info_update]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("effect_meta_update"),
            contents: cast_slice(&effect_metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("update_bgl0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("update_bgl1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bgl2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("update_bgl2"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bgl3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("update_bgl3"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_bg0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: draw_indirect_buffer.as_entire_binding(),
                },
            ],
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_bg1"),
            layout: &bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_buffer.as_entire_binding(),
                },
            ],
        });
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_bg2"),
            layout: &bgl2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spawner_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prefix_buffer_update.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: batch_info_buffer.as_entire_binding(),
                },
            ],
        });
        let bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update_bg3"),
            layout: &bgl3,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: metadata_buffer.as_entire_binding(),
            }],
        });

        let update_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("real_update_pl"),
                bind_group_layouts: &[Some(&bgl0), Some(&bgl1), Some(&bgl2), Some(&bgl3)],
                immediate_size: 0,
            });
        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("real_update_pipe"),
            layout: Some(&update_pipeline_layout),
            module: &update_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("real_update_enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("real_update_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&update_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.set_bind_group(2, &bg2, &[]);
            pass.set_bind_group(3, &bg3, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        submit_and_wait(&device, &queue, encoder.finish()).await;

        let draw_out = readback_u32(
            &device,
            &queue,
            &draw_indirect_buffer,
            (draw_indirect_init.len() * std::mem::size_of::<DrawIndexedIndirectArgsGpu>()) as u64,
        )
        .await;
        assert_eq!(
            draw_out[1], 2,
            "real vfx_update did not accumulate instance_count for effect 0"
        );
        assert_eq!(
            draw_out[6], 1,
            "real vfx_update did not accumulate instance_count for effect 1"
        );

        let indirect_out = readback_u32(
            &device,
            &queue,
            &indirect_buffer,
            (indirect_rows.len() * std::mem::size_of::<IndirectEntryGpu>()) as u64,
        )
        .await;
        assert_eq!(
            indirect_out[0], 0,
            "effect 0 particle[0] write index mismatch"
        );
        assert_eq!(
            indirect_out[3], 1,
            "effect 0 particle[1] write index mismatch"
        );
        assert_eq!(
            indirect_out[12], 0,
            "effect 1 particle[0] write index mismatch"
        );

        Ok::<(), Box<dyn std::error::Error>>(())
    })?;

    println!("SUCCESS!");
    Ok(())
}

use wgpu::util::DeviceExt;
