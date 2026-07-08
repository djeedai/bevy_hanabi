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
    test_utils::MockRenderer, Attribute, EffectAsset, EffectShaderSources, ExprWriter,
    SetAttributeModifier, SpawnerSettings,
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

fn pack_effect_metadata_rows(rows: &[GpuEffectMetadata], storage_alignment: u32) -> Vec<u32> {
    let stride_u32 = (GpuEffectMetadata::aligned_size(storage_alignment).get() / 4) as usize;
    let row_u32 = std::mem::size_of::<GpuEffectMetadata>() / 4;
    let mut packed = vec![0_u32; rows.len() * stride_u32];
    for (i, row) in rows.iter().enumerate() {
        let src: &[u32] = cast_slice(std::slice::from_ref(row));
        let dst = &mut packed[i * stride_u32..i * stride_u32 + row_u32];
        dst.copy_from_slice(src);
    }
    packed
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
    buffer.write_buffer(device, queue);
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

/// Create an array containing the input slice content padded to the given alignment.
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
    let draw_init = [
        GpuDrawIndexedIndirectArgs {
            index_count: 0,
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        },
        GpuDrawIndexedIndirectArgs {
            index_count: 0,
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        },
    ];
    let particles = [ParticleGpu::zeroed(); 8];
    let mut indirect_rows = [IndirectEntryGpu::zeroed(); 8];
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
    let metadata_packed = pack_effect_metadata_rows(&metadata_rows, storage_alignment);

    let sim_params_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:update:sim_params"),
        contents: cast_slice(&[sim_params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let draw_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:update:draw"),
        contents: cast_slice(&draw_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let particle_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:update:particles"),
        contents: cast_slice(&particles),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let indirect_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:update:indirect"),
        contents: cast_slice(&indirect_rows),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let prefix_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:update:prefix"),
        contents: cast_slice(&[0_u32, 2_u32]),
        usage: wgpu::BufferUsages::STORAGE,
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
        contents: cast_slice(&metadata_packed),
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
                resource: draw_buffer.as_entire_binding(),
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
        label: Some("hanabi:test:update:pl"),
        bind_group_layouts: &[Some(&bgl0), Some(&bgl1), Some(&bgl2), Some(&bgl3)],
        immediate_size: 0,
    });
    let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hanabi:test:update:pipe"),
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

    let draw_out = readback_vec::<u32>(
        &device,
        &queue,
        &draw_buffer,
        (std::mem::size_of::<GpuDrawIndexedIndirectArgs>() * draw_init.len()) as u64,
    );
    assert_eq!(draw_out[1], 2);
    assert_eq!(draw_out[6], 1);

    let indirect_out = readback_vec::<u32>(
        &device,
        &queue,
        &indirect_buffer,
        (std::mem::size_of::<IndirectEntryGpu>() * indirect_rows.len()) as u64,
    );
    assert_eq!(indirect_out[0], 0);
    assert_eq!(indirect_out[3], 1);
    assert_eq!(indirect_out[12], 0);

    Ok(())
}

#[test]
fn real_vfx_indirect_contracts() -> Result<(), Box<dyn std::error::Error>> {
    const EM_OFFSET_CAPACITY: usize = 0;
    const EM_OFFSET_ALIVE_COUNT: usize = 1;
    const EM_OFFSET_MAX_UPDATE: usize = 2;
    const EM_OFFSET_MAX_SPAWN: usize = 3;
    const EM_OFFSET_INDIRECT_WRITE_INDEX: usize = 4;
    const EM_OFFSET_INDIRECT_DRAW_INDEX: usize = 5;

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

    let effect_stride_u32 = (GpuEffectMetadata::aligned_size(storage_alignment).get() / 4) as usize;
    let mut metadata_u32 = vec![0_u32; 2 * effect_stride_u32];
    metadata_u32[EM_OFFSET_CAPACITY] = 200;
    metadata_u32[EM_OFFSET_ALIVE_COUNT] = 130;
    metadata_u32[EM_OFFSET_INDIRECT_WRITE_INDEX] = 0;
    metadata_u32[EM_OFFSET_INDIRECT_DRAW_INDEX] = 0;
    let em1 = effect_stride_u32;
    metadata_u32[em1 + EM_OFFSET_CAPACITY] = 5;
    metadata_u32[em1 + EM_OFFSET_ALIVE_COUNT] = 1;
    metadata_u32[em1 + EM_OFFSET_INDIRECT_WRITE_INDEX] = 1;
    metadata_u32[em1 + EM_OFFSET_INDIRECT_DRAW_INDEX] = 1;
    let metadata_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hanabi:test:indirect:metadata"),
        contents: cast_slice(&metadata_u32),
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

    let metadata_out = readback_vec::<u32>(
        &device,
        &queue,
        &metadata_buffer,
        (metadata_u32.len() * 4) as u64,
    );
    assert_eq!(metadata_out[EM_OFFSET_MAX_UPDATE], 130);
    assert_eq!(metadata_out[EM_OFFSET_MAX_SPAWN], 70);
    assert_eq!(metadata_out[em1 + EM_OFFSET_MAX_UPDATE], 1);
    assert_eq!(metadata_out[em1 + EM_OFFSET_MAX_SPAWN], 4);

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
