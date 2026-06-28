use std::{borrow::Cow, num::NonZeroU64};

use bevy::render::{render_resource::*, renderer::RenderQueue};
use bytemuck::{cast_slice, Pod};
use futures::channel::oneshot;
use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor};
use wgpu::util::DeviceExt;

use super::*;
use crate::test_utils::MockRenderer;

fn submit_and_wait(device: &RenderDevice, queue: &RenderQueue, command_buffer: wgpu::CommandBuffer) {
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

fn create_real_indirect_shader_module(
    device: &RenderDevice,
    storage_alignment: u32,
) -> Result<wgpu::ShaderModule, Box<dyn std::error::Error>> {
    let spawner_padding_code = GpuSpawnerParams::padding_code(storage_alignment);
    let batch_info_padding_code = GpuBatchInfo::padding_code(storage_alignment);
    let effect_metadata_padding_code = GpuEffectMetadata::padding_code(storage_alignment);
    let effect_metadata_stride_code = GpuEffectMetadata::aligned_size(storage_alignment)
        .get()
        .to_string();
    let common_code = include_str!("vfx_common.wgsl")
        .replace("{{SPAWNER_PADDING}}", &spawner_padding_code)
        .replace("{{BATCH_INFO_PADDING}}", &batch_info_padding_code)
        .replace("{{EFFECT_METADATA_PADDING}}", &effect_metadata_padding_code)
        .replace("{{EFFECT_METADATA_STRIDE}}", &effect_metadata_stride_code);

    let mut composer = Composer::default();
    composer.add_composable_module(ComposableModuleDescriptor {
        source: &common_code,
        file_path: "bevy_hanabi::vfx_common",
        ..Default::default()
    })?;
    let module = composer.make_naga_module(NagaModuleDescriptor {
        source: include_str!("vfx_indirect.wgsl"),
        file_path: "bevy_hanabi::vfx_indirect",
        shader_defs: Default::default(),
        ..Default::default()
    })?;
    Ok(device.wgpu_device().create_shader_module(
        wgpu::ShaderModuleDescriptor {
            label: Some("bevy_hanabi::vfx_indirect"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
        },
    ))
}

#[test]
fn real_vfx_indirect_contracts() -> Result<(), Box<dyn std::error::Error>> {
    const EM_OFFSET_CAPACITY: usize = 0;
    const EM_OFFSET_ALIVE_COUNT: usize = 1;
    const EM_OFFSET_MAX_UPDATE: usize = 2;
    const EM_OFFSET_MAX_SPAWN: usize = 3;
    const EM_OFFSET_INDIRECT_WRITE_INDEX: usize = 4;
    const EM_OFFSET_INDIRECT_DISPATCH_INDEX: usize = 5;

    let renderer = MockRenderer::new();
    let device = renderer.device();
    let queue = renderer.queue();
    let wgpu_device = device.wgpu_device();
    let storage_alignment = device.limits().min_storage_buffer_offset_alignment;

    let shader = create_real_indirect_shader_module(&device, storage_alignment)?;

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
        label: Some("hanabi:test:sim_params_indirect"),
        contents: cast_slice(&[sim_params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let effect_stride_u32 = (GpuEffectMetadata::aligned_size(storage_alignment).get() / 4) as usize;
    let mut metadata_u32 = vec![0_u32; 2 * effect_stride_u32];
    metadata_u32[EM_OFFSET_CAPACITY] = 200;
    metadata_u32[EM_OFFSET_ALIVE_COUNT] = 130;
    metadata_u32[EM_OFFSET_INDIRECT_WRITE_INDEX] = 0;
    metadata_u32[EM_OFFSET_INDIRECT_DISPATCH_INDEX] = 0;
    let em1 = effect_stride_u32;
    metadata_u32[em1 + EM_OFFSET_CAPACITY] = 5;
    metadata_u32[em1 + EM_OFFSET_ALIVE_COUNT] = 1;
    metadata_u32[em1 + EM_OFFSET_INDIRECT_WRITE_INDEX] = 1;
    metadata_u32[em1 + EM_OFFSET_INDIRECT_DISPATCH_INDEX] = 1;
    let metadata_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:effect_metadata"),
        contents: cast_slice(&metadata_u32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let dispatch_init = [GpuDispatchIndirectArgs::default(), GpuDispatchIndirectArgs::default()];
    let dispatch_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:dispatch_indirect"),
        contents: cast_slice(&dispatch_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let mut draw_indirect = [GpuDrawIndexedIndirectArgs::default(); 2];
    draw_indirect[0].instance_count = 9;
    draw_indirect[1].instance_count = 4;
    let draw_indirect_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:draw_indirect"),
        contents: cast_slice(&draw_indirect),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let mut spawners = AlignedBufferVec::<GpuSpawnerParams>::new(
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        NonZeroU64::new(storage_alignment as u64),
        None,
    );
    spawners.push(GpuSpawnerParams {
        seed: 111,
        render_pong: 0,
        effect_metadata_index: 0,
        draw_indirect_index: 0,
        slab_offset: 0,
        parent_slab_offset: u32::MAX,
        ..default()
    });
    spawners.push(GpuSpawnerParams {
        seed: 222,
        render_pong: 1,
        effect_metadata_index: 1,
        draw_indirect_index: 1,
        slab_offset: 64,
        parent_slab_offset: u32::MAX,
        ..default()
    });
    spawners.write_buffer(&device, &queue);
    submit_and_wait(
        &device,
        &queue,
        wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hanabi:test:flush_spawners"),
            })
            .finish(),
    );
    let spawner_buffer = spawners.buffer().unwrap();

    let prefix_sum_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:prefix_sum"),
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
                resource: dispatch_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: draw_indirect_buffer.as_entire_binding(),
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
                resource: prefix_sum_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hanabi:test:indirect:pl"),
        bind_group_layouts: &[Some(&bgl0), Some(&bgl1), Some(&bgl2)],
        immediate_size: 0,
    });
    let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hanabi:test:indirect:pipe"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hanabi:test:indirect:encoder"),
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

    let prefix_out = readback_vec::<u32>(&device, &queue, &prefix_sum_buffer, 8);
    assert_eq!(prefix_out, vec![130, 1]);

    let metadata_out =
        readback_vec::<u32>(&device, &queue, &metadata_buffer, (metadata_u32.len() * 4) as u64);
    assert_eq!(metadata_out[EM_OFFSET_MAX_UPDATE], 130);
    assert_eq!(metadata_out[EM_OFFSET_MAX_SPAWN], 70);
    assert_eq!(metadata_out[em1 + EM_OFFSET_MAX_UPDATE], 1);
    assert_eq!(metadata_out[em1 + EM_OFFSET_MAX_SPAWN], 4);

    let dispatch_out = readback_vec::<GpuDispatchIndirectArgs>(
        &device,
        &queue,
        &dispatch_buffer,
        (std::mem::size_of::<GpuDispatchIndirectArgs>() * 2) as u64,
    );
    assert_eq!(dispatch_out[0].x, 3);
    assert_eq!(dispatch_out[0].y, 1);
    assert_eq!(dispatch_out[0].z, 1);
    assert_eq!(dispatch_out[1].x, 1);
    assert_eq!(dispatch_out[1].y, 1);
    assert_eq!(dispatch_out[1].z, 1);

    let draw_out = readback_vec::<GpuDrawIndexedIndirectArgs>(
        &device,
        &queue,
        &draw_indirect_buffer,
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
    assert_eq!(
        spawner_out[spawner_stride_u32 + render_pong_offset_u32],
        0
    );

    Ok(())
}
