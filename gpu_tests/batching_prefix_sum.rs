//! Headless GPU regression tests for batching dataflow contracts.

use bytemuck::{Pod, Zeroable, cast_slice};
use futures::channel::oneshot;
use futures::executor::block_on;

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

fn submit_and_wait(device: &wgpu::Device, queue: &wgpu::Queue, command_buffer: wgpu::CommandBuffer) {
    queue.submit([command_buffer]);
    let (tx, rx) = oneshot::channel();
    queue.on_submitted_work_done(move || {
        let _ = tx.send(());
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let _ = block_on(rx);
}

fn readback_u32(
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
    submit_and_wait(device, queue, encoder.finish());

    let slice = staging.slice(..);
    let (tx, rx) = oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    block_on(rx).unwrap().unwrap();
    let data = slice.get_mapped_range();
    let out = cast_slice::<u8, u32>(&data).to_vec();
    drop(data);
    staging.unmap();
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    block_on(async move {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(_) => {
                eprintln!("No adapter available; skipping headless batching GPU test.");
                return Ok(());
            }
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                label: Some("batching_prefix_sum_device"),
                trace: wgpu::Trace::Off,
            })
            .await?;

        // ------------------------------------------------------------------
        // Test 1: Prefix sum pass dataflow
        // ------------------------------------------------------------------
        let prefix_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batching_prefix_sum_shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct BatchInfo {
    total_spawn_count: u32,
    total_update_count: u32,
    base_effect: u32,
    base_particle: u32,
    prefix_sum_offset: u32,
    prefix_sum_count: u32,
}
struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}
@group(0) @binding(0) var<storage, read_write> batch_infos : array<BatchInfo>;
@group(0) @binding(1) var<storage, read_write> prefix_sum : array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch_indirect_buffer : array<DispatchIndirectArgs>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_index = gid.x;
    if (batch_index >= arrayLength(&batch_infos)) { return; }
    let offset = batch_infos[batch_index].prefix_sum_offset;
    let count = batch_infos[batch_index].prefix_sum_count;
    let end = offset + count;
    var sum = 0u;
    for (var i = offset; i < end; i += 1u) {
        let c = prefix_sum[i];
        prefix_sum[i] = sum;
        sum = sum + c;
    }
    batch_infos[batch_index].total_update_count = sum;
    dispatch_indirect_buffer[batch_index].x = (sum + 63u) >> 6u;
    dispatch_indirect_buffer[batch_index].y = 1u;
    dispatch_indirect_buffer[batch_index].z = 1u;
}
"#
                    .into(),
            ),
        });

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
        submit_and_wait(&device, &queue, encoder.finish());

        let prefix_out = readback_u32(&device, &queue, &prefix_buffer, (prefix_init.len() * 4) as u64);
        assert_eq!(prefix_out, vec![0, 10, 15, 0], "prefix sum output mismatch");

        let dispatch_out = readback_u32(
            &device,
            &queue,
            &dispatch_buffer,
            (dispatch_init.len() * std::mem::size_of::<DispatchIndirectArgs>()) as u64,
        );
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
        submit_and_wait(&device, &queue, encoder.finish());
        let out_effect = readback_u32(&device, &queue, &out_buffer, (slab_indices.len() * 4) as u64);
        assert_eq!(out_effect, vec![0, 0, 1, 1, 2, 2], "effect index mapping mismatch");

        Ok::<(), Box<dyn std::error::Error>>(())
    })?;

    println!("SUCCESS!");
    Ok(())
}

use wgpu::util::DeviceExt;
