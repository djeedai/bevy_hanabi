use std::{borrow::Cow, num::NonZeroU64};

use bevy::{
    prelude::Vec3,
    render::{render_resource::*, renderer::RenderQueue},
};
use bytemuck::{cast_slice, Pod, Zeroable};
use futures::channel::oneshot;
use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor};
use wgpu::util::DeviceExt;

use super::*;
use crate::{
    plugin::VFX_SORT_WGSL, test_utils::MockRenderer, Attribute, EffectAsset, EffectShaderSources,
    ExprWriter, SetAttributeModifier, SpawnerSettings,
};

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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LocateBatch {
    base_particle: u32,
    prefix_sum_offset: u32,
    prefix_sum_count: u32,
    _pad0: u32,
}

fn submit_and_wait(
    device: &RenderDevice,
    queue: &RenderQueue,
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
    let _ = futures::executor::block_on(rx);
}

fn readback_vec<T: Pod>(
    device: &RenderDevice,
    queue: &RenderQueue,
    src: &wgpu::Buffer,
    bytes: u64,
) -> Vec<T> {
    assert!(bytes.is_multiple_of(size_of::<T>() as u64));
    let wgpu_device = device.wgpu_device();
    let staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hanabi:test:staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hanabi:test:readback_encoder"),
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
    futures::executor::block_on(rx).unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let out = cast_slice::<u8, T>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    out
}

fn create_composed_shader_module(
    device: &RenderDevice,
    storage_alignment: u32,
    shader_source: &str,
    file_path: &str,
) -> Result<wgpu::ShaderModule, Box<dyn std::error::Error>> {
    let spawner_padding_code = GpuSpawnerParams::padding_code(storage_alignment);
    let batch_info_padding_code = GpuBatchInfo::padding_code(storage_alignment);
    let common_code = include_str!("vfx_common.wgsl")
        .replace("{{SPAWNER_PADDING}}", &spawner_padding_code)
        .replace("{{BATCH_INFO_PADDING}}", &batch_info_padding_code);

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
    Ok(device
        .wgpu_device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(file_path),
            source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
        }))
}

fn write_aligned_spawners(
    device: &RenderDevice,
    queue: &RenderQueue,
    storage_alignment: u32,
    rows: &[GpuSpawnerParams],
) -> AlignedBufferVec<GpuSpawnerParams> {
    let mut buffer = AlignedBufferVec::<GpuSpawnerParams>::new(
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        NonZeroU64::new(storage_alignment as u64),
        None,
    );
    for row in rows {
        buffer.push(*row);
    }
    assert!(buffer.write_buffer(device, queue));
    submit_and_wait(
        device,
        queue,
        device
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hanabi:test:flush_spawner_upload"),
            })
            .finish(),
    );
    buffer
}

/// Create an array containing the input slice content padded to the given
/// alignment.
fn padded_slice_content<T: ShaderType + Pod>(arr: &[T], align: u32) -> Vec<u8> {
    let aligned_size = (T::min_size().get() as usize).next_multiple_of(align as usize);
    let total_size = arr.len() * aligned_size;
    let mut data = vec![0u8; total_size];
    let cpu_size = size_of::<T>();
    for (index, item) in arr.iter().enumerate() {
        let offset = index * aligned_size;
        let item_bytes = cast_slice(std::slice::from_ref(item));
        data[offset..offset + cpu_size].copy_from_slice(item_bytes);
    }
    data
}

fn create_sort_pipeline(
    wgpu_device: &wgpu::Device,
    sort_buffer: &wgpu::Buffer,
) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
    let bgl = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:sort:bgl"),
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
    let bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:sort:bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: sort_buffer.as_entire_binding(),
        }],
    });
    let pl = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hanabi:test:sort:pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let source = VFX_SORT_WGSL
        .replace("#ifdef HAS_DUAL_KEY", "")
        .replace("#ifdef TEST", "")
        .replace("#endif", "");
    let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hanabi:test:vfx_sort"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hanabi:test:sort:pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });
    (pipeline, bg)
}

#[test]
fn real_ribbon_sort_chain_isolated_per_instance() -> Result<(), Box<dyn std::error::Error>> {
    let renderer = MockRenderer::new();
    let device = renderer.device();
    let queue = renderer.queue();
    let wgpu_device = device.wgpu_device();
    let storage_alignment = device.limits().min_storage_buffer_offset_alignment;

    let fill_shader = create_composed_shader_module(
        &device,
        storage_alignment,
        include_str!("vfx_sort_fill.wgsl"),
        "bevy_hanabi::vfx_sort_fill",
    )?;
    let copy_shader = create_composed_shader_module(
        &device,
        storage_alignment,
        include_str!("vfx_sort_copy.wgsl"),
        "bevy_hanabi::vfx_sort_copy",
    )?;

    let metadatas = [
        GpuEffectMetadata {
            capacity: 4,
            alive_count: 4,
            indirect_write_index: 0,
            particle_stride: 2,
            sort_key_offset: 0,
            sort_key2_offset: 1,
            ..default()
        },
        GpuEffectMetadata {
            capacity: 4,
            alive_count: 4,
            indirect_write_index: 0,
            particle_stride: 2,
            sort_key_offset: 0,
            sort_key2_offset: 1,
            ..default()
        },
    ];
    let spawners = [
        GpuSpawnerParams {
            effect_metadata_index: 0,
            slab_offset: 0,
            ..default()
        },
        GpuSpawnerParams {
            effect_metadata_index: 1,
            slab_offset: 4,
            ..default()
        },
    ];
    let particles = [
        1_u32,
        3.0_f32.to_bits(),
        0,
        1.0_f32.to_bits(),
        1,
        2.0_f32.to_bits(),
        0,
        4.0_f32.to_bits(), // Instance 0
        2,
        1.0_f32.to_bits(),
        1,
        4.0_f32.to_bits(),
        2,
        3.0_f32.to_bits(),
        1,
        2.0_f32.to_bits(), // Instance 1
    ];
    let indirect_indices = [
        3_u32,
        0,
        u32::MAX,
        2,
        0,
        u32::MAX,
        1,
        0,
        u32::MAX,
        0,
        0,
        u32::MAX,
        2,
        0,
        u32::MAX,
        0,
        0,
        u32::MAX,
        3,
        0,
        u32::MAX,
        1,
        0,
        u32::MAX,
    ];
    let particle_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:ribbon:particles"),
        contents: cast_slice(&particles),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let indirect_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:ribbon:indirect"),
        contents: cast_slice(&indirect_indices),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let metadata_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:ribbon:metadata"),
        contents: cast_slice(&metadatas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let spawner_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:ribbon:spawners"),
        contents: &padded_slice_content(&spawners, storage_alignment),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let sort_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hanabi:test:ribbon:sort"),
        size: 4 + 4 * 12,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dispatch_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:ribbon:dispatch"),
        contents: cast_slice(&[
            GpuDispatchIndirectArgs { x: 1, y: 1, z: 1 },
            GpuDispatchIndirectArgs { x: 1, y: 1, z: 1 },
        ]),
        usage: wgpu::BufferUsages::INDIRECT,
    });

    let storage = |binding, read_only, dynamic| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    };
    let fill_bgl = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:ribbon:fill:bgl"),
        entries: &[
            storage(0, false, false),
            storage(1, true, false),
            storage(2, true, false),
            storage(3, false, false),
            storage(4, true, true),
        ],
    });
    let copy_bgl = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:ribbon:copy:bgl"),
        entries: &[
            storage(0, false, false),
            storage(1, true, false),
            storage(2, false, false),
            storage(3, true, true),
        ],
    });
    let fill_bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:ribbon:fill:bg"),
        layout: &fill_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sort_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: particle_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: indirect_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: metadata_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &spawner_buffer,
                    offset: 0,
                    size: NonZeroU64::new(storage_alignment as u64),
                }),
            },
        ],
    });
    let copy_bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:ribbon:copy:bg"),
        layout: &copy_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: indirect_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sort_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: metadata_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &spawner_buffer,
                    offset: 0,
                    size: NonZeroU64::new(storage_alignment as u64),
                }),
            },
        ],
    });
    let fill_pl = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hanabi:test:ribbon:fill:pl"),
        bind_group_layouts: &[Some(&fill_bgl)],
        immediate_size: 0,
    });
    let copy_pl = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hanabi:test:ribbon:copy:pl"),
        bind_group_layouts: &[Some(&copy_bgl)],
        immediate_size: 0,
    });
    let fill_pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hanabi:test:ribbon:fill"),
        layout: Some(&fill_pl),
        module: &fill_shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });
    let copy_pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hanabi:test:ribbon:copy"),
        layout: Some(&copy_pl),
        module: &copy_shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });
    let (sort_pipeline, sort_bg) = create_sort_pipeline(wgpu_device, &sort_buffer);

    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hanabi:test:ribbon:chain"),
    });
    for instance in 0..2 {
        encoder.clear_buffer(&sort_buffer, 0, Some(4));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hanabi:test:ribbon:fill"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&fill_pipeline);
            pass.set_bind_group(
                0,
                &fill_bg,
                &[(instance * storage_alignment as usize) as u32],
            );
            pass.dispatch_workgroups_indirect(
                &dispatch_buffer,
                instance as u64 * std::mem::size_of::<GpuDispatchIndirectArgs>() as u64,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hanabi:test:ribbon:sort"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&sort_pipeline);
            pass.set_bind_group(0, &sort_bg, &[]);
            pass.dispatch_workgroups_indirect(
                &dispatch_buffer,
                instance as u64 * std::mem::size_of::<GpuDispatchIndirectArgs>() as u64,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hanabi:test:ribbon:copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&copy_pipeline);
            pass.set_bind_group(
                0,
                &copy_bg,
                &[(instance * storage_alignment as usize) as u32],
            );
            pass.dispatch_workgroups_indirect(
                &dispatch_buffer,
                instance as u64 * std::mem::size_of::<GpuDispatchIndirectArgs>() as u64,
            );
        }
    }
    submit_and_wait(&device, &queue, encoder.finish());

    let indirect_out = readback_vec::<u32>(
        &device,
        &queue,
        &indirect_buffer,
        (indirect_indices.len() * std::mem::size_of::<u32>()) as u64,
    );
    assert_eq!(
        &indirect_out[0..12],
        &[
            1,
            0,
            u32::MAX,
            3,
            0,
            u32::MAX,
            2,
            0,
            u32::MAX,
            0,
            0,
            u32::MAX
        ]
    );
    assert_eq!(
        &indirect_out[12..24],
        &[
            3,
            0,
            u32::MAX,
            1,
            0,
            u32::MAX,
            0,
            0,
            u32::MAX,
            2,
            0,
            u32::MAX
        ]
    );
    Ok(())
}

#[test]
fn real_vfx_prefix_sum_contracts() -> Result<(), Box<dyn std::error::Error>> {
    let renderer = MockRenderer::new();
    let device = renderer.device();
    let queue = renderer.queue();
    let wgpu_device = device.wgpu_device();
    let storage_alignment = device.limits().min_storage_buffer_offset_alignment;

    let shader = create_composed_shader_module(
        &device,
        storage_alignment,
        include_str!("vfx_prefix_sum.wgsl"),
        "bevy_hanabi::vfx_prefix_sum",
    )?;

    let batches = [
        GpuBatchInfo {
            base_particle: 100,
            prefix_sum_offset: 0,
            prefix_sum_count: 3,
            ..default()
        },
        GpuBatchInfo {
            spawner_base: 3,
            base_particle: 500,
            prefix_sum_offset: 3,
            prefix_sum_count: 1,
            ..default()
        },
    ];
    let prefix_init = [10_u32, 5, 8, 6];
    let dispatch_init = [
        GpuDispatchIndirectArgs::default(),
        GpuDispatchIndirectArgs::default(),
    ];

    let batch_content = padded_slice_content(
        &batches,
        wgpu_device.limits().min_storage_buffer_offset_alignment,
    );
    let batch_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:prefix:batch"),
        contents: &batch_content,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let prefix_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:prefix:sum"),
        contents: cast_slice(&prefix_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let dispatch_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:prefix:dispatch"),
        contents: cast_slice(&dispatch_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let bgl = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:prefix:bgl"),
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
    let bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:prefix:bg"),
        layout: &bgl,
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
    let pl = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hanabi:test:prefix:pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hanabi:test:prefix:pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hanabi:test:prefix:enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hanabi:test:prefix:pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    submit_and_wait(&device, &queue, encoder.finish());

    let prefix_out = readback_vec::<u32>(
        &device,
        &queue,
        &prefix_buffer,
        (prefix_init.len() * 4) as u64,
    );
    assert_eq!(prefix_out, vec![0, 10, 15, 0]);

    let dispatch_out = readback_vec::<GpuDispatchIndirectArgs>(
        &device,
        &queue,
        &dispatch_buffer,
        (std::mem::size_of::<GpuDispatchIndirectArgs>() * dispatch_init.len()) as u64,
    );
    assert_eq!(dispatch_out[0].x, 1);
    assert_eq!(dispatch_out[0].y, 1);
    assert_eq!(dispatch_out[0].z, 1);
    assert_eq!(dispatch_out[1].x, 1);
    assert_eq!(dispatch_out[1].y, 1);
    assert_eq!(dispatch_out[1].z, 1);

    Ok(())
}

#[test]
fn location_mapping_contracts() {
    let renderer = MockRenderer::new();
    let device = renderer.device();
    let queue = renderer.queue();
    let wgpu_device = device.wgpu_device();

    let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hanabi:test:location:shader"),
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
    out_effect_index[i] = lo - 1u - batch_info.prefix_sum_offset;
}
"#
            .into(),
        ),
    });

    let locate_batch = LocateBatch {
        base_particle: 100,
        prefix_sum_offset: 0,
        prefix_sum_count: 3,
        _pad0: 0,
    };
    let locate_prefix = [0_u32, 10, 15];
    let slab_indices = [100_u32, 109, 110, 114, 115, 122];
    let out_zero = [0_u32; 6];

    let batch_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:location:batch"),
        contents: cast_slice(&[locate_batch]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let prefix_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:location:prefix"),
        contents: cast_slice(&locate_prefix),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let slab_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:location:slab"),
        contents: cast_slice(&slab_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:location:out"),
        contents: cast_slice(&out_zero),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let bgl = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:location:bgl"),
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
    let bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:location:bg"),
        layout: &bgl,
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
                resource: slab_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buffer.as_entire_binding(),
            },
        ],
    });
    let pl = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hanabi:test:location:pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hanabi:test:location:pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hanabi:test:location:enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hanabi:test:location:pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    submit_and_wait(&device, &queue, encoder.finish());
    let out = readback_vec::<u32>(
        &device,
        &queue,
        &out_buffer,
        (slab_indices.len() * 4) as u64,
    );
    assert_eq!(out, vec![0, 0, 1, 1, 2, 2]);
}

#[test]
fn indirect_plus_update_routing_contracts() {
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
        spawner_base: u32,
        base_particle: u32,
        prefix_sum_offset: u32,
        prefix_sum_count: u32,
    }

    let renderer = MockRenderer::new();
    let device = renderer.device();
    let queue = renderer.queue();
    let wgpu_device = device.wgpu_device();

    let indirect_shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hanabi:test:indirect-lite"),
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
    let update_route_shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hanabi:test:update-route"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
struct BatchInfo {
    total_update_count: u32,
    spawner_base: u32,
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

    let spawners = [
        SpawnerLite {
            effect_metadata_index: 0,
            draw_indirect_index: 0,
        },
        SpawnerLite {
            effect_metadata_index: 1,
            draw_indirect_index: 1,
        },
    ];
    let em = [
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
    let prefix_init = [0_u32, 0];
    let ddi_init = [
        GpuDispatchIndirectArgs::default(),
        GpuDispatchIndirectArgs::default(),
    ];

    let sp_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:routing:spawner"),
        contents: cast_slice(&spawners),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let em_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:routing:metadata"),
        contents: cast_slice(&em),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let prefix_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:routing:prefix"),
        contents: cast_slice(&prefix_init),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let ddi_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:routing:ddi"),
        contents: cast_slice(&ddi_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let bgl_i = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:routing:indirect:bgl"),
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
    let bg_i = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:routing:indirect:bg"),
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
                resource: prefix_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ddi_buffer.as_entire_binding(),
            },
        ],
    });
    let pl_i = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hanabi:test:routing:indirect:pl"),
        bind_group_layouts: &[Some(&bgl_i)],
        immediate_size: 0,
    });
    let pipe_i = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hanabi:test:routing:indirect:pipe"),
        layout: Some(&pl_i),
        module: &indirect_shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });
    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hanabi:test:routing:indirect:enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hanabi:test:routing:indirect:pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe_i);
        pass.set_bind_group(0, &bg_i, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    submit_and_wait(&device, &queue, encoder.finish());

    let prefix = readback_vec::<u32>(&device, &queue, &prefix_buffer, 8);
    assert_eq!(prefix, vec![7, 5]);

    let prefix_after = [0_u32, 7_u32];
    queue.write_buffer(&prefix_buffer, 0, cast_slice(&prefix_after));
    let batch_lite = BatchInfoLite {
        total_update_count: 12,
        spawner_base: 0,
        base_particle: 100,
        prefix_sum_offset: 0,
        prefix_sum_count: 2,
    };
    let batch_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:routing:batch"),
        contents: cast_slice(&[batch_lite]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let routed_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:routing:routed"),
        contents: cast_slice(&[0_u32, 0_u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let bgl_u = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:routing:update:bgl"),
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
    let bg_u = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:routing:update:bg"),
        layout: &bgl_u,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: prefix_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: batch_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: routed_buffer.as_entire_binding(),
            },
        ],
    });
    let pl_u = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hanabi:test:routing:update:pl"),
        bind_group_layouts: &[Some(&bgl_u)],
        immediate_size: 0,
    });
    let pipe_u = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hanabi:test:routing:update:pipe"),
        layout: Some(&pl_u),
        module: &update_route_shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });
    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hanabi:test:routing:update:enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hanabi:test:routing:update:pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe_u);
        pass.set_bind_group(0, &bg_u, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    submit_and_wait(&device, &queue, encoder.finish());
    let routed = readback_vec::<u32>(&device, &queue, &routed_buffer, 8);
    assert_eq!(routed, vec![7, 5]);
}

#[test]
fn real_vfx_update_contracts() -> Result<(), Box<dyn std::error::Error>> {
    let renderer = MockRenderer::new();
    let device = renderer.device();
    let queue = renderer.queue();
    let wgpu_device = device.wgpu_device();
    let storage_alignment = device.limits().min_storage_buffer_offset_alignment;

    // #[allow(unsafe_code)]
    // unsafe {
    //     wgpu_device.start_graphics_debugger_capture();
    // }

    // POSITION-only particle effect (see ParticleGpu)
    let writer = ExprWriter::new();
    let init_pos = SetAttributeModifier::new(Attribute::POSITION, writer.lit(Vec3::ZERO).expr());
    let module = writer.finish();
    let asset = EffectAsset::new(8, SpawnerSettings::rate(0.0.into()), module).init(init_pos);
    let sources = EffectShaderSources::generate(&asset, None, 0)?;
    let shader = create_composed_shader_module(
        &device,
        storage_alignment,
        &sources.update_shader_source,
        "bevy_hanabi::generated_vfx_update",
    )?;

    let sim_params = GpuSimParams {
        delta_time: 1.0,
        time: 0.0,
        virtual_delta_time: 1.0,
        virtual_time: 0.0,
        real_delta_time: 1.0,
        real_time: 0.0,
        num_effects: 2,
    };
    let draw_placeholder = GpuDrawIndexedIndirectArgs {
        index_count: u32::MAX,
        instance_count: 0, // atomically incremented, need a valid 0 start value
        first_index: u32::MAX,
        base_vertex: i32::MAX,
        first_instance: u32::MAX,
    };
    let draw_init = [draw_placeholder; 2];
    let particle_placeholder = ParticleGpu {
        position: [f32::INFINITY; 4],
    };
    let mut particles = [particle_placeholder; 8];
    particles[0].position = [1.0, 2.0, 3.0, f32::MAX];
    particles[1].position = [4.0, 5.0, 6.0, f32::MAX];
    particles[4].position = [7.0, 8.0, 9.0, f32::MAX];
    let indirect_placeholder = IndirectEntryGpu {
        particle_index: [u32::MAX; 2],
        dead_index: u32::MAX,
    };
    let mut indirect_rows = [indirect_placeholder; 8];
    indirect_rows[0].particle_index = [0, 0];
    indirect_rows[1].particle_index = [0, 1];
    indirect_rows[4].particle_index = [0, 0];

    let spawner_rows = [
        GpuSpawnerParams {
            seed: 1,
            render_pong: 0,
            effect_metadata_index: 0,
            draw_indirect_index: 0,
            slab_offset: 0,
            parent_slab_offset: u32::MAX,
            ..default()
        },
        GpuSpawnerParams {
            seed: 2,
            render_pong: 0,
            effect_metadata_index: 1,
            draw_indirect_index: 1,
            slab_offset: 4,
            parent_slab_offset: u32::MAX,
            ..default()
        },
    ];
    let spawners = write_aligned_spawners(&device, &queue, storage_alignment, &spawner_rows);
    let spawner_buffer = spawners.buffer().unwrap();

    let metadata_rows = [
        GpuEffectMetadata {
            capacity: 8,
            alive_count: 2,
            max_update: 2,
            indirect_draw_index: 0,
            particle_stride: 4,
            ..default()
        },
        GpuEffectMetadata {
            capacity: 8,
            alive_count: 1,
            max_update: 1,
            indirect_draw_index: 1,
            particle_stride: 4,
            ..default()
        },
    ];

    let sim_params_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:update:sim_params"),
        contents: cast_slice(&[sim_params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let draw_indirect_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:update:draw"),
        contents: cast_slice(&draw_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let particle_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:update:particles"),
        contents: cast_slice(&particles),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let indirect_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:update:indirect"),
        contents: cast_slice(&indirect_rows),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let prefix_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:update:prefix"),
        contents: cast_slice(&[0_u32, 2_u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let batch_content = padded_slice_content(
        &[GpuBatchInfo {
            total_update_count: 3,
            prefix_sum_count: 2,
            ..default()
        }],
        wgpu_device.limits().min_storage_buffer_offset_alignment,
    );
    let batch_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:update:batch"),
        contents: &batch_content,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let metadata_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:update:metadata"),
        contents: cast_slice(&metadata_rows),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let bgl0 = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:update:bgl0"),
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
    let bgl1 = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:update:bgl1"),
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
    let bgl2 = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:update:bgl2"),
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
    let bgl3 = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:update:bgl3"),
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

    let bg0 = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:update:bg0"),
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
    let bg1 = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:update:bg1"),
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
    let bg2 = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:update:bg2"),
        layout: &bgl2,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: spawner_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: prefix_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: batch_buffer.as_entire_binding(),
            },
        ],
    });
    let bg3 = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:update:bg3"),
        layout: &bgl3,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: metadata_buffer.as_entire_binding(),
        }],
    });

    let pl = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hanabi:test:update:pipeline_layout"),
        bind_group_layouts: &[Some(&bgl0), Some(&bgl1), Some(&bgl2), Some(&bgl3)],
        immediate_size: 0,
    });
    let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hanabi:test:update:pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hanabi:test:update:enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hanabi:test:update:pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.set_bind_group(2, &bg2, &[]);
        pass.set_bind_group(3, &bg3, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    submit_and_wait(&device, &queue, encoder.finish());

    // #[allow(unsafe_code)]
    // unsafe {
    //     wgpu_device.stop_graphics_debugger_capture();
    // }

    let particle_out = readback_vec::<f32>(
        &device,
        &queue,
        &particle_buffer,
        size_of::<ParticleGpu>() as u64 * 8,
    );
    assert_ne!(particle_out[0], f32::INFINITY); // particles[0].position.x

    let indirect_out = readback_vec::<u32>(
        &device,
        &queue,
        &indirect_buffer,
        (std::mem::size_of::<IndirectEntryGpu>() * indirect_rows.len()) as u64,
    );
    // Write order inside an effect is non-deterministic, as GPU threads race on
    // atomic increment.
    if indirect_out[0] == 0 {
        assert_eq!(indirect_out[3], 1);
    } else {
        assert_eq!(indirect_out[0], 1);
        assert_eq!(indirect_out[3], 0);
    }
    // Inside the second effect though there's a single particle alive.
    assert_eq!(indirect_out[12], 0);

    let draw_out = readback_vec::<u32>(
        &device,
        &queue,
        &draw_indirect_buffer,
        (std::mem::size_of::<GpuDrawIndexedIndirectArgs>() * draw_init.len()) as u64,
    );
    assert_eq!(draw_out[1], 2); // args[0].instance_count
    assert_eq!(draw_out[6], 1); // args[1].instance_count

    Ok(())
}

#[test]
fn real_vfx_indirect_contracts() -> Result<(), Box<dyn std::error::Error>> {
    let renderer = MockRenderer::new();
    let device = renderer.device();
    let queue = renderer.queue();
    let wgpu_device = device.wgpu_device();
    let storage_alignment = device.limits().min_storage_buffer_offset_alignment;

    let shader = create_composed_shader_module(
        &device,
        storage_alignment,
        include_str!("vfx_indirect.wgsl"),
        "bevy_hanabi::vfx_indirect",
    )?;

    let sim_params = GpuSimParams {
        delta_time: 1.0,
        time: 0.0,
        virtual_delta_time: 1.0,
        virtual_time: 0.0,
        real_delta_time: 1.0,
        real_time: 0.0,
        num_effects: 2,
    };
    let sim_params_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:indirect:sim"),
        contents: cast_slice(&[sim_params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let metadata_placeholder = GpuEffectMetadata {
        max_update: u32::MAX,
        max_spawn: u32::MAX,
        init_indirect_dispatch_index: u32::MAX,
        properties_array_index: u32::MAX,
        local_child_index: u32::MAX,
        global_child_index: u32::MAX,
        base_child_index: u32::MAX,
        particle_stride: u32::MAX,
        sort_key_offset: u32::MAX,
        sort_key2_offset: u32::MAX,
        particle_counter: u32::MAX,
        ..default()
    };
    let mut metadata_content = [metadata_placeholder; 2];
    metadata_content[0].capacity = 200;
    metadata_content[0].alive_count = 130;
    metadata_content[0].indirect_write_index = 0;
    metadata_content[0].indirect_draw_index = 0;
    metadata_content[1].capacity = 5;
    metadata_content[1].alive_count = 1;
    metadata_content[1].indirect_write_index = 1;
    metadata_content[1].indirect_draw_index = 1;
    let metadata_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:indirect:metadata"),
        contents: cast_slice(&metadata_content),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let mut draw_init = [GpuDrawIndexedIndirectArgs::default(); 2];
    draw_init[0].instance_count = 9;
    draw_init[1].instance_count = 4;
    let draw_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:indirect:draw"),
        contents: cast_slice(&draw_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let spawner_rows = [
        GpuSpawnerParams {
            seed: 111,
            render_pong: 0,
            effect_metadata_index: 0,
            draw_indirect_index: 0,
            slab_offset: 0,
            parent_slab_offset: u32::MAX,
            ..default()
        },
        GpuSpawnerParams {
            seed: 222,
            render_pong: 1,
            effect_metadata_index: 1,
            draw_indirect_index: 1,
            slab_offset: 64,
            parent_slab_offset: u32::MAX,
            ..default()
        },
    ];
    let spawners = write_aligned_spawners(&device, &queue, storage_alignment, &spawner_rows);
    let spawner_buffer = spawners.buffer().unwrap();

    let prefix_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:indirect:prefix"),
        contents: cast_slice(&[0_u32, 0_u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let bgl0 = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:indirect:bgl0"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bgl1 = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:indirect:bgl1"),
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
    let bgl2 = wgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hanabi:test:indirect:bgl2"),
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
    let bg0 = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:indirect:bg0"),
        layout: &bgl0,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: sim_params_buffer.as_entire_binding(),
        }],
    });
    let bg1 = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:indirect:bg1"),
        layout: &bgl1,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: metadata_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: draw_buffer.as_entire_binding(),
            },
        ],
    });
    let bg2 = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hanabi:test:indirect:bg2"),
        layout: &bgl2,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: spawner_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: prefix_buffer.as_entire_binding(),
            },
        ],
    });

    let pl = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hanabi:test:indirect:pl"),
        bind_group_layouts: &[Some(&bgl0), Some(&bgl1), Some(&bgl2)],
        immediate_size: 0,
    });
    let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hanabi:test:indirect:pipe"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hanabi:test:indirect:enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hanabi:test:indirect:pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.set_bind_group(2, &bg2, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    submit_and_wait(&device, &queue, encoder.finish());

    let prefix_out = readback_vec::<u32>(&device, &queue, &prefix_buffer, 8);
    assert_eq!(prefix_out, vec![130, 1]);

    let metadata_out = readback_vec::<GpuEffectMetadata>(
        &device,
        &queue,
        &metadata_buffer,
        (metadata_content.len() * size_of::<GpuEffectMetadata>()) as u64,
    );
    assert_eq!(metadata_out[0].max_update, 130); // copied from alive_count
    assert_eq!(metadata_out[0].max_spawn, 70); // copied from dead_count = capacity - alive_count
    assert_eq!(metadata_out[1].max_update, 1);
    assert_eq!(metadata_out[1].max_spawn, 4);

    let draw_out = readback_vec::<GpuDrawIndexedIndirectArgs>(
        &device,
        &queue,
        &draw_buffer,
        (std::mem::size_of::<GpuDrawIndexedIndirectArgs>() * 2) as u64,
    );
    assert_eq!(draw_out[0].instance_count, 0);
    assert_eq!(draw_out[1].instance_count, 0);

    let spawner_out = readback_vec::<u32>(
        &device,
        &queue,
        spawner_buffer,
        (spawners.aligned_size() * 2) as u64,
    );
    let spawner_stride_u32 = spawners.aligned_size() / 4;
    let render_pong_offset_u32 = std::mem::offset_of!(GpuSpawnerParams, render_pong) / 4;
    assert_eq!(spawner_out[render_pong_offset_u32], 1);
    assert_eq!(spawner_out[spawner_stride_u32 + render_pong_offset_u32], 0);

    Ok(())
}
