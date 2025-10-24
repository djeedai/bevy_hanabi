use std::num::{NonZeroU32, NonZeroU64};

use bevy::{
    asset::Handle,
    ecs::{resource::Resource, world::World},
    platform::collections::{hash_map::Entry, HashMap},
    render::{
        render_resource::{
            BindGroup, BindGroupLayout, Buffer, BufferId, CachedComputePipelineId,
            CachedPipelineState, ComputePipelineDescriptor, PipelineCache,
        },
        renderer::RenderDevice,
    },
    shader::Shader,
    utils::default,
};
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType, BufferBinding,
    BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoder, ShaderStages,
};

use super::{gpu_buffer::GpuBuffer, GpuDispatchIndirectArgs, GpuEffectMetadata, StorageType};
use crate::{render::GpuSpawnerParams, Attribute, ParticleLayout};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SortFillBindGroupLayoutKey {
    particle_min_binding_size: NonZeroU32,
    particle_ribbon_id_offset: u32,
    particle_age_offset: u32,
}

impl SortFillBindGroupLayoutKey {
    pub fn from_particle_layout(particle_layout: &ParticleLayout) -> Result<Self, ()> {
        let particle_ribbon_id_offset = particle_layout
            .byte_offset(Attribute::RIBBON_ID)
            .ok_or(())?;
        let particle_age_offset = particle_layout.byte_offset(Attribute::AGE).ok_or(())?;
        let key = SortFillBindGroupLayoutKey {
            particle_min_binding_size: particle_layout.min_binding_size32(),
            particle_ribbon_id_offset,
            particle_age_offset,
        };
        Ok(key)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SortFillBindGroupKey {
    particle: BufferId,
    indirect_index: BufferId,
    effect_metadata: BufferId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SortCopyBindGroupKey {
    indirect_index: BufferId,
    sort: BufferId,
    effect_metadata: BufferId,
}

#[derive(Resource)]
pub struct SortBindGroups {
    /// Render device.
    render_device: RenderDevice,
    /// Sort-fill pass compute shader.
    sort_fill_shader: Handle<Shader>,
    /// GPU buffer of key-value pairs to sort.
    sort_buffer: Buffer,
    /// GPU buffer containing the [`GpuDispatchIndirect`] structs for the
    /// sort-fill and sort passes.
    indirect_buffer: GpuBuffer<GpuDispatchIndirectArgs>,
    /// Bind group layouts for group #0 of the sort-fill compute pass.
    sort_fill_bind_group_layouts:
        HashMap<SortFillBindGroupLayoutKey, (BindGroupLayout, CachedComputePipelineId)>,
    /// Bind groups for group #0 of the sort-fill compute pass.
    sort_fill_bind_groups: HashMap<SortFillBindGroupKey, BindGroup>,
    /// Bind group for group #0 of the sort compute pass.
    sort_bind_group: BindGroup,
    sort_copy_bind_group_layout: BindGroupLayout,
    /// Pipeline for sort pass.
    sort_pipeline_id: CachedComputePipelineId,
    /// Pipeline for sort-copy pass.
    sort_copy_pipeline_id: CachedComputePipelineId,
    /// Bind groups for group #0 of the sort-copy compute pass.
    sort_copy_bind_groups: HashMap<SortCopyBindGroupKey, BindGroup>,
}

impl SortBindGroups {
    pub fn new(
        world: &mut World,
        sort_fill_shader: Handle<Shader>,
        sort_shader: Handle<Shader>,
        sort_copy_shader: Handle<Shader>,
    ) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let sort_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("hanabi:buffer:sort:pairs"),
            size: 3 * 1024 * 1024,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let indirect_buffer_size = 3 * 1024;
        let indirect_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("hanabi:buffer:sort:indirect"),
            size: indirect_buffer_size,
            usage: BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST
                | BufferUsages::STORAGE
                | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let indirect_buffer = GpuBuffer::new_allocated(
            indirect_buffer,
            indirect_buffer_size as u32,
            Some("hanabi:buffer:sort:indirect".to_string()),
        );

        let sort_bind_group_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:sort",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(16).unwrap()), // count + dual kv pair
                },
                count: None,
            }],
        );

        let sort_bind_group = render_device.create_bind_group(
            "hanabi:bind_group:sort",
            &sort_bind_group_layout,
            &[
                // @group(0) @binding(0) var<storage, read_write> pairs : array<KeyValuePair>;
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &sort_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        );

        let sort_pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("hanabi:pipeline:sort".into()),
            layout: vec![sort_bind_group_layout],
            shader: sort_shader,
            shader_defs: vec!["HAS_DUAL_KEY".into()],
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let alignment = render_device.limits().min_storage_buffer_offset_alignment;
        let sort_copy_bind_group_layout = render_device.create_bind_group_layout(
            "hanabi:bind_group_layout:sort_copy",
            &[
                // @group(0) @binding(0) var<storage, read_write> indirect_index_buffer :
                // IndirectIndexBuffer;
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(12).unwrap()), // ping/pong+dead
                    },
                    count: None,
                },
                // @group(0) @binding(1) var<storage, read> sort_buffer : SortBuffer;
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(16).unwrap()), /* count + dual kv
                                                                               * pair */
                    },
                    count: None,
                },
                // @group(0) @binding(2) var<storage, read_write> effect_metadata : EffectMetadata;
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: Some(GpuEffectMetadata::aligned_size(alignment)),
                    },
                    count: None,
                },
                // @group(0) @binding(3) var<storage, read> spawner : Spawner;
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: Some(GpuSpawnerParams::aligned_size(alignment)),
                    },
                    count: None,
                },
            ],
        );

        let sort_copy_pipeline_id =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("hanabi:pipeline:sort_copy".into()),
                layout: vec![sort_copy_bind_group_layout.clone()],
                shader: sort_copy_shader,
                shader_defs: vec![],
                entry_point: Some("main".into()),
                push_constant_ranges: vec![],
                zero_initialize_workgroup_memory: false,
            });

        Self {
            render_device: render_device.clone(),
            sort_fill_shader,
            sort_buffer,
            indirect_buffer,
            sort_fill_bind_group_layouts: default(),
            sort_fill_bind_groups: default(),
            sort_bind_group,
            sort_copy_bind_group_layout,
            sort_pipeline_id,
            sort_copy_pipeline_id,
            sort_copy_bind_groups: default(),
        }
    }

    #[inline]
    pub fn clear_indirect_dispatch_buffer(&mut self) {
        self.indirect_buffer.clear();
    }

    #[inline]
    pub fn allocate_indirect_dispatch(&mut self) -> u32 {
        self.indirect_buffer.allocate()
    }

    #[inline]
    pub fn get_indirect_dispatch_byte_offset(&self, index: u32) -> u32 {
        self.indirect_buffer.item_size() as u32 * index
    }

    #[inline]
    #[allow(dead_code)]
    pub fn sort_buffer(&self) -> &Buffer {
        &self.sort_buffer
    }

    #[inline]
    pub fn indirect_buffer(&self) -> Option<&Buffer> {
        self.indirect_buffer.buffer()
    }

    #[inline]
    pub fn sort_bind_group(&self) -> &BindGroup {
        &self.sort_bind_group
    }

    #[inline]
    pub fn sort_pipeline_id(&self) -> CachedComputePipelineId {
        self.sort_pipeline_id
    }

    /// Check if the sort pipeline is ready to run for the given effect
    /// instance.
    ///
    /// This ensures all compute pipelines are compiled and ready to be used
    /// this frame.
    pub fn is_pipeline_ready(
        &self,
        particle_layout: &ParticleLayout,
        pipeline_cache: &PipelineCache,
    ) -> bool {
        // Validate the sort-fill pipeline. It was created and queued for compile by
        // ensure_sort_fill_bind_group_layout(), which normally is called just before
        // is_pipeline_ready().
        let Some(pipeline_id) = self.get_sort_fill_pipeline_id(particle_layout) else {
            return false;
        };
        if !matches!(
            pipeline_cache.get_compute_pipeline_state(pipeline_id),
            CachedPipelineState::Ok(_)
        ) {
            return false;
        }

        // The 2 pipelines below are created and queued for compile in new(), so are
        // almost always ready.
        // FIXME - they could be checked once a frame only, not once per effect...

        // Validate the sort pipeline
        if !matches!(
            pipeline_cache.get_compute_pipeline_state(self.sort_pipeline_id()),
            CachedPipelineState::Ok(_)
        ) {
            return false;
        }

        // Validate the sort-copy pipeline
        if !matches!(
            pipeline_cache.get_compute_pipeline_state(self.get_sort_copy_pipeline_id()),
            CachedPipelineState::Ok(_)
        ) {
            return false;
        }

        true
    }

    #[inline]
    pub fn prepare_buffers(&mut self, render_device: &RenderDevice) {
        self.indirect_buffer.prepare_buffers(render_device);
    }

    #[inline]
    pub fn write_buffers(&self, command_encoder: &mut CommandEncoder) {
        self.indirect_buffer.write_buffers(command_encoder);
    }

    #[inline]
    pub fn clear_previous_frame_resizes(&mut self) {
        self.indirect_buffer.clear_previous_frame_resizes();
    }

    pub fn ensure_sort_fill_bind_group_layout(
        &mut self,
        pipeline_cache: &PipelineCache,
        particle_layout: &ParticleLayout,
    ) -> Result<&BindGroupLayout, ()> {
        let key = SortFillBindGroupLayoutKey::from_particle_layout(particle_layout)?;
        let (layout, _) = self
            .sort_fill_bind_group_layouts
            .entry(key)
            .or_insert_with(|| {
                let alignment = self
                    .render_device
                    .limits()
                    .min_storage_buffer_offset_alignment;
                let bind_group_layout = self.render_device.create_bind_group_layout(
                    "hanabi:bind_group_layout:sort_fill",
                    &[
                        // @group(0) @binding(0) var<storage, read_write> sort_buffer : SortBuffer;
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: Some(NonZeroU64::new(16).unwrap()), // count + dual kv pair
                            },
                            count: None,
                        },
                        // @group(0) @binding(1) var<storage, read> particle_buffer : RawParticleBuffer;
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: Some(key.particle_min_binding_size.into()),
                            },
                            count: None,
                        },
                        // @group(0) @binding(2) var<storage, read> indirect_index_buffer : array<u32>;
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: Some(NonZeroU64::new(12).unwrap()), // ping/pong+dead
                            },
                            count: None,
                        },
                        // @group(0) @binding(3) var<storage, read_write> effect_metadata : EffectMetadata;
                        BindGroupLayoutEntry {
                            binding: 3,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: true,
                                min_binding_size: Some(GpuEffectMetadata::aligned_size(alignment)),
                            },
                            count: None,
                        },
                        // @group(0) @binding(4) var<storage, read> spawner : Spawner;
                        BindGroupLayoutEntry {
                            binding: 4,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: true,
                                min_binding_size: Some(GpuSpawnerParams::aligned_size(alignment)),
                            },
                            count: None,
                        },
                    ],
                );
                let pipeline_id =
                    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                        label: Some("hanabi:pipeline:sort_fill".into()),
                        layout: vec![bind_group_layout.clone()],
                        shader: self.sort_fill_shader.clone(),
                        shader_defs: vec!["HAS_DUAL_KEY".into()],
                        entry_point: Some("main".into()),
                        push_constant_ranges: vec![],
                        zero_initialize_workgroup_memory: false,
                    });
                (bind_group_layout, pipeline_id)
            });
        Ok(layout)
    }

    // We currently only use the bind group layout internally in
    // ensure_sort_fill_bind_group()
    #[allow(dead_code)]
    pub fn get_sort_fill_bind_group_layout(
        &self,
        particle_layout: &ParticleLayout,
    ) -> Option<&BindGroupLayout> {
        let key = SortFillBindGroupLayoutKey::from_particle_layout(particle_layout).ok()?;
        self.sort_fill_bind_group_layouts
            .get(&key)
            .map(|(layout, _)| layout)
    }

    pub fn get_sort_fill_pipeline_id(
        &self,
        particle_layout: &ParticleLayout,
    ) -> Option<CachedComputePipelineId> {
        let key = SortFillBindGroupLayoutKey::from_particle_layout(particle_layout).ok()?;
        self.sort_fill_bind_group_layouts
            .get(&key)
            .map(|(_, pipeline_id)| *pipeline_id)
    }

    pub fn get_sort_copy_pipeline_id(&self) -> CachedComputePipelineId {
        self.sort_copy_pipeline_id
    }

    pub fn ensure_sort_fill_bind_group(
        &mut self,
        particle_layout: &ParticleLayout,
        particle: &Buffer,
        indirect_index: &Buffer,
        effect_metadata: &Buffer,
        spawner_buffer: &Buffer,
    ) -> Result<&BindGroup, ()> {
        let key = SortFillBindGroupKey {
            particle: particle.id(),
            indirect_index: indirect_index.id(),
            effect_metadata: effect_metadata.id(),
        };
        let entry = self.sort_fill_bind_groups.entry(key);
        let bind_group = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                // Note: can't use get_bind_group_layout() because the function call mixes the
                // lifetimes of the two hash maps and complains the bind group one is already
                // borrowed. Doing a manual access to the layout one instead makes the compiler
                // happy.
                let key = SortFillBindGroupLayoutKey::from_particle_layout(particle_layout)?;
                let layout = &self.sort_fill_bind_group_layouts.get(&key).ok_or(())?.0;
                entry.insert(
                    self.render_device.create_bind_group(
                        "hanabi:bind_group:sort_fill",
                        layout,
                        &[
                            // @group(0) @binding(0) var<storage, read_write> pairs:
                            // array<KeyValuePair>;
                            BindGroupEntry {
                                binding: 0,
                                resource: self.sort_buffer.as_entire_binding(),
                            },
                            // @group(0) @binding(1) var<storage, read> particle_buffer:
                            // ParticleBuffer;
                            BindGroupEntry {
                                binding: 1,
                                resource: particle.as_entire_binding(),
                            },
                            // @group(0) @binding(2) var<storage, read> indirect_index_buffer :
                            // array<u32>;
                            BindGroupEntry {
                                binding: 2,
                                resource: indirect_index.as_entire_binding(),
                            },
                            // @group(0) @binding(3) var<storage, read> effect_metadata :
                            // EffectMetadata;
                            BindGroupEntry {
                                binding: 3,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: effect_metadata,
                                    offset: 0,
                                    size: Some(GpuEffectMetadata::aligned_size(
                                        self.render_device
                                            .limits()
                                            .min_storage_buffer_offset_alignment,
                                    )),
                                }),
                            },
                            // @group(0) @binding(4) var<storage, read> spawner : Spawner;
                            BindGroupEntry {
                                binding: 4,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: spawner_buffer,
                                    offset: 0,
                                    size: Some(GpuSpawnerParams::aligned_size(
                                        self.render_device
                                            .limits()
                                            .min_storage_buffer_offset_alignment,
                                    )),
                                }),
                            },
                        ],
                    ),
                )
            }
        };
        Ok(bind_group)
    }

    pub fn sort_fill_bind_group(
        &self,
        particle: BufferId,
        indirect_index: BufferId,
        effect_metadata: BufferId,
    ) -> Option<&BindGroup> {
        let key = SortFillBindGroupKey {
            particle,
            indirect_index,
            effect_metadata,
        };
        self.sort_fill_bind_groups.get(&key)
    }

    pub fn ensure_sort_copy_bind_group(
        &mut self,
        indirect_index_buffer: &Buffer,
        effect_metadata_buffer: &Buffer,
        spawner_buffer: &Buffer,
    ) -> Result<&BindGroup, ()> {
        let key = SortCopyBindGroupKey {
            indirect_index: indirect_index_buffer.id(),
            sort: self.sort_buffer.id(),
            effect_metadata: effect_metadata_buffer.id(),
        };
        let entry = self.sort_copy_bind_groups.entry(key);
        let bind_group = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                entry.insert(
                    self.render_device.create_bind_group(
                        "hanabi:bind_group:sort_copy",
                        &self.sort_copy_bind_group_layout,
                        &[
                            // @group(0) @binding(0) var<storage, read_write> indirect_index_buffer
                            // : IndirectIndexBuffer;
                            BindGroupEntry {
                                binding: 0,
                                resource: indirect_index_buffer.as_entire_binding(),
                            },
                            // @group(0) @binding(1) var<storage, read> sort_buffer : SortBuffer;
                            BindGroupEntry {
                                binding: 1,
                                resource: self.sort_buffer.as_entire_binding(),
                            },
                            // @group(0) @binding(2) var<storage, read> effect_metadata :
                            // EffectMetadata;
                            BindGroupEntry {
                                binding: 2,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: effect_metadata_buffer,
                                    offset: 0,
                                    size: Some(GpuEffectMetadata::aligned_size(
                                        self.render_device
                                            .limits()
                                            .min_storage_buffer_offset_alignment,
                                    )),
                                }),
                            },
                            // @group(0) @binding(3) var<storage, read> spawner : Spawner;
                            BindGroupEntry {
                                binding: 3,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: spawner_buffer,
                                    offset: 0,
                                    size: Some(GpuSpawnerParams::aligned_size(
                                        self.render_device
                                            .limits()
                                            .min_storage_buffer_offset_alignment,
                                    )),
                                }),
                            },
                        ],
                    ),
                )
            }
        };
        Ok(bind_group)
    }

    pub fn sort_copy_bind_group(
        &self,
        indirect_index: BufferId,
        effect_metadata: BufferId,
    ) -> Option<&BindGroup> {
        let key = SortCopyBindGroupKey {
            indirect_index,
            sort: self.sort_buffer.id(),
            effect_metadata,
        };
        self.sort_copy_bind_groups.get(&key)
    }
}

#[cfg(all(test, feature = "gpu_tests"))]
mod gpu_tests {
    use bevy::{
        math::FloatOrd,
        render::render_resource::{
            binding_types::storage_buffer_sized, BindGroupEntries, BindGroupLayoutEntries,
            ShaderSize, ShaderType,
        },
    };
    #[allow(unused_imports)]
    use bytemuck::{cast_slice, cast_slice_mut, Pod, Zeroable};
    use rand::Rng;
    use wgpu::{
        BufferDescriptor, BufferUsages, ComputePassDescriptor, ComputePipelineDescriptor,
        PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
        ShaderStages,
    };

    use crate::{plugin::VFX_SORT_WGSL, test_utils::*};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable, ShaderType)]
    #[repr(C)]
    struct KeyValuePair {
        pub key: u32,
        pub value: u32,
    }

    impl std::cmp::PartialOrd for KeyValuePair {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(&other))
        }
    }

    impl std::cmp::Ord for KeyValuePair {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.key.cmp(&other.key)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable, ShaderType)]
    #[repr(C)]
    struct DualKeyValuePair {
        pub key: u32,
        pub key2: f32,
        pub value: u32,
    }

    // Ignore weirdnesses with f32 NaN etc. here, we should never have a key with
    // such values.
    impl std::cmp::Eq for DualKeyValuePair {}

    impl std::cmp::PartialOrd for DualKeyValuePair {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            if self.key != other.key {
                return Some(self.key.cmp(&other.key));
            }
            self.key2.partial_cmp(&other.key2)
        }
    }

    impl std::cmp::Ord for DualKeyValuePair {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            if self.key != other.key {
                return self.key.cmp(&other.key);
            }
            FloatOrd(self.key2).cmp(&FloatOrd(other.key2))
        }
    }

    /// Workgroup-level sort with Batcher's odd-even mergesort.
    /// - spawn 64 threads (one workgroup)
    /// - each thread sorts 1024 / 64 = 16 elements
    #[test]
    fn test_batcher_odd_even_mergesort() {
        let renderer = MockRenderer::new();
        let device = renderer.device();
        let queue = renderer.queue();

        println!(
            "max_compute_workgroup_storage_size = {}",
            device.limits().max_compute_workgroup_storage_size
        );

        // SAFETY : for debugging only
        #[allow(unsafe_code)]
        unsafe {
            device.wgpu_device().start_graphics_debugger_capture()
        };

        // Clamp max block size to the device's reported storage
        let max_block_size = device.limits().max_compute_workgroup_storage_size
            / DualKeyValuePair::SHADER_SIZE.get() as u32;
        println!("max_block_size = {}", max_block_size);
        let num_kv = 1024.min(max_block_size);

        let byte_size = 4 + num_kv as u64 * DualKeyValuePair::SHADER_SIZE.get();
        let sort_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sort_buffer"),
            size: byte_size,
            usage: BufferUsages::STORAGE | BufferUsages::MAP_READ,
            mapped_at_creation: true,
        });
        {
            // Scope get_mapped_range_mut() to force a drop before unmap()
            {
                let slice: &mut [u8] = &mut sort_buffer.slice(..).get_mapped_range_mut();
                let header: &mut [u32] = cast_slice_mut(slice);
                header[0] = num_kv;
                let values: &mut [DualKeyValuePair] = cast_slice_mut(&mut slice[4..]);
                for (i, kv) in values.iter_mut().enumerate() {
                    kv.key = byte_size as u32 - i as u32;
                    kv.key2 = i as f32 * 0.1;
                    kv.value = i as u32;
                    //println!("[#{}] k={} k2={} v={}", i, kv.key, kv.key2,
                    // kv.value);
                }
            }
            sort_buffer.unmap();
        }

        // Create GPU resources
        let bind_group_layout = device.create_bind_group_layout(
            "bind_group_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::COMPUTE,
                storage_buffer_sized(false, None),
            ),
        );
        let bind_group = device.create_bind_group(
            None,
            &bind_group_layout,
            &BindGroupEntries::single(sort_buffer.as_entire_binding()),
        );
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let src = VFX_SORT_WGSL
            .replace("#ifdef HAS_DUAL_KEY", "")
            .replace("#ifdef TEST", "")
            .replace("#endif", "");
        let shader_module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("vfx_sort"),
            source: ShaderSource::Wgsl(src.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("test_batcher_odd_even_mergesort"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("test_batcher_odd_even_mergesort"),
            compilation_options: PipelineCompilationOptions {
                constants: &[("blockSize", max_block_size as f64)],
                // Ensure the shader behaves even if memory is not zero-initialized
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        });

        // Dispatch test
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("test_batcher_odd_even_mergesort"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        // Submit command queue and wait for execution
        println!("Executing pipeline...");
        let command_buffer = encoder.finish();
        queue.submit([command_buffer]);
        let (tx, rx) = futures::channel::oneshot::channel();
        queue.on_submitted_work_done(move || {
            tx.send(()).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait);
        let _ = futures::executor::block_on(rx);
        println!("Pipeline executed");

        // SAFETY : for debugging only
        #[allow(unsafe_code)]
        unsafe {
            device.wgpu_device().stop_graphics_debugger_capture()
        };

        // Read back (GPU -> CPU)
        println!("Downloading result buffer from GPU to CPU...");
        let buffer_slice = sort_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait);
        let _result = futures::executor::block_on(rx);
        let view = buffer_slice.get_mapped_range();
        println!("Result buffer downloaded to CPU");

        // Validate content
        assert_eq!(view.len(), byte_size as usize);
        let view_slice: &[DualKeyValuePair] = cast_slice(&view[4..]);
        for i in 1..view_slice.len() {
            //println!("[#{}] k={} k2={} v={}", i, kv.key, kv.key2, kv.value);

            // Ordered within one block of 16 elements (1024 items / 64 threads = 16
            // items/thread)
            if (i - 1) / 16 == i / 16 {
                assert!(view_slice[i] >= view_slice[i - 1])
            }
        }
    }

    /// Calculate the merge path for a parallel merge.
    #[test]
    fn test_calc_merge_path() {
        let renderer = MockRenderer::new();
        let device = renderer.device();
        let queue = renderer.queue();

        println!(
            "max_compute_workgroup_storage_size = {}",
            device.limits().max_compute_workgroup_storage_size
        );

        // SAFETY : for debugging only
        #[allow(unsafe_code)]
        unsafe {
            device.wgpu_device().start_graphics_debugger_capture()
        };

        // Clamp max block size to the device's reported storage. We need 2 workgroup
        // arrays, so we need to halve the limit for each.
        let max_block_size = device.limits().max_compute_workgroup_storage_size
            / DualKeyValuePair::SHADER_SIZE.get() as u32;
        println!("max_block_size = {}", max_block_size);
        let block_size = 1024.min(max_block_size); // total input buffer size
        let block_size = block_size / 2; // actual block size in shader
        let num_kv = block_size * 2; // ensure we don't have an odd size
        println!("block_size = {} (x2)", block_size);
        let list_len = 1024.min(block_size);
        println!("list_len = {}", list_len);

        let byte_size = 4 + num_kv as u64 * DualKeyValuePair::SHADER_SIZE.get();
        let sort_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sort_buffer"),
            size: byte_size,
            usage: BufferUsages::STORAGE | BufferUsages::MAP_READ,
            mapped_at_creation: true,
        });
        let mut expected = Vec::with_capacity(num_kv as usize);
        {
            // Scope get_mapped_range_mut() to force a drop before unmap()
            {
                let slice: &mut [u8] = &mut sort_buffer.slice(..).get_mapped_range_mut();
                let header: &mut [u32] = cast_slice_mut(slice);
                header[0] = num_kv;
                let values: &mut [DualKeyValuePair] = cast_slice_mut(&mut slice[4..]);

                assert_eq!(num_kv % 2, 0);
                assert_eq!(num_kv / 2, list_len);
                let (values_a, values_b) = values.split_at_mut(list_len as usize);
                let mut thread_rng = rand::rng();
                let max_value = list_len - 30; // limit the range to force duplicate values
                for i in 0..list_len {
                    values_a[i as usize].key = thread_rng.random_range(0..max_value);
                    values_a[i as usize].key2 = i as f32 * 0.1;
                    values_a[i as usize].value = i;

                    values_b[i as usize].key = thread_rng.random_range(0..max_value);
                    values_b[i as usize].key2 = i as f32 * 0.1;
                    values_b[i as usize].value = list_len + i;

                    expected.push(values_a[i as usize]);
                    expected.push(values_b[i as usize]);
                }
                values_a.sort();
                values_b.sort();
                // for (i, kv) in values_a.iter().enumerate() {
                //     println!("A [#{}] k={} k2={} v={}", i, kv.key, kv.key2,
                // kv.value); }
                // for (i, kv) in values_b.iter().enumerate() {
                //     println!("B [#{}] k={} k2={} v={}", i, kv.key, kv.key2,
                // kv.value); }
            }
            sort_buffer.unmap();
        }

        // Create GPU resources
        let bind_group_layout = device.create_bind_group_layout(
            "bind_group_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::COMPUTE,
                storage_buffer_sized(false, None),
            ),
        );
        let bind_group = device.create_bind_group(
            None,
            &bind_group_layout,
            &BindGroupEntries::single(sort_buffer.as_entire_binding()),
        );
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let src = VFX_SORT_WGSL
            .replace("#ifdef HAS_DUAL_KEY", "")
            .replace("#ifdef TEST", "")
            .replace("#endif", "");
        let shader_module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("vfx_sort"),
            source: ShaderSource::Wgsl(src.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("test_calc_merge_path"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("test_calc_merge_path"),
            compilation_options: PipelineCompilationOptions {
                constants: &[("blockSize", block_size as f64)],
                // Ensure the shader behaves even if memory is not zero-initialized
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        });

        // Dispatch test
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("test_calc_merge_path"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        // Submit command queue and wait for execution
        println!("Executing pipeline...");
        let command_buffer = encoder.finish();
        queue.submit([command_buffer]);
        let (tx, rx) = futures::channel::oneshot::channel();
        queue.on_submitted_work_done(move || {
            tx.send(()).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait);
        let _ = futures::executor::block_on(rx);
        println!("Pipeline executed");

        // SAFETY : for debugging only
        #[allow(unsafe_code)]
        unsafe {
            device.wgpu_device().stop_graphics_debugger_capture()
        };

        // Read back (GPU -> CPU)
        println!("Downloading result buffer from GPU to CPU...");
        let buffer_slice = sort_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait);
        let _result = futures::executor::block_on(rx);
        let view = buffer_slice.get_mapped_range();
        println!("Result buffer downloaded to CPU");

        // Calculate the expected result by doing the actual merge sort, and deriving
        // all the merge paths from it.
        expected.sort();
        let path_len = list_len.div_ceil(64);
        let num_paths = list_len.div_ceil(path_len);
        println!("path_len = {path_len}");
        println!("num_paths = {num_paths}");
        let mut paths: Vec<_> = (0..num_paths)
            .map(|ipath| {
                let start = (ipath * path_len) as usize;
                let end = (start + path_len as usize).min(list_len as usize);
                let slice = &expected[start..end];
                let mut ia = 0;
                for kv in slice {
                    if kv.value < list_len {
                        ia += 1;
                    }
                }
                ia
            })
            .collect();
        for i in 1..paths.len() {
            // The path at index #i is actually the cumulated path from start. Above we
            // counted only the i_a within a cross-diagonal interval, but the GPU calculates
            // the total i_a for the path starting at the beginning of the block up to the
            // diagonal.
            paths[i] += paths[i - 1];
        }
        //println!("EXPECTED:");
        // for (i, exp_val) in paths.iter().enumerate() {
        //     let diag = (i as u32 + 1) * path_len;
        //     let i_a = *exp_val;
        //     let i_b = diag - i_a;
        //     println!("+ PATH #{i} : diag={diag} i_a={i_a} i_b={i_b}");
        //     let s: String = (0..path_len)
        //         .map(|idx| expected[idx as usize + (i * path_len as usize)])
        //         .fold("".to_string(), |acc, x| {
        //             format!(
        //                 "{} {}[{},{},{}]",
        //                 acc,
        //                 if x.value < list_len { 'A' } else { 'B' },
        //                 x.key,
        //                 x.key2,
        //                 x.value
        //             )
        //         });
        //     println!("  {s}");
        // }

        // Validate content
        assert_eq!(view.len(), byte_size as usize);
        let view_slice: &[DualKeyValuePair] = cast_slice(&view[4..]);
        assert_eq!(view_slice.len(), list_len as usize * 2);
        assert!(view_slice.len() >= paths.len());
        for (i, exp_val) in paths.iter().enumerate() {
            // The test shader stores the resulting i_a[] into the 'value' of the first N
            // entries of the sort buffer, and the diagonal value of the path into the 'key'
            // field.

            let calc_diag = view_slice[i].key;
            let exp_diag = (i as u32 + 1) * path_len;
            assert_eq!(calc_diag, exp_diag);

            let calc_ia = view_slice[i].value;
            let exp_ia = *exp_val;
            assert_eq!(calc_ia, exp_ia);

            // println!(
            //     "-> [{}] diag={} a_i={} b_i={} [EXPECTED:{}]",
            //     i,
            //     view_slice[i].key,
            //     view_slice[i].value,
            //     view_slice[i].key - view_slice[i].value,
            //     *exp_val
            // );
        }
    }

    fn binary_search_prefix_sum(particle_index: u32, sums: &[u32]) -> u32 {
        let mut lo = 0;
        let mut hi = sums.len() as u32 - 1;
        while lo <= hi {
            let mid = (hi + lo) >> 1;
            if particle_index >= sums[mid as usize] {
                lo = mid + 1;
            } else if particle_index < sums[mid as usize] {
                hi = mid;
            }
        }
        return lo - 1;
    }

    #[test]
    fn test_cpu_binary_search_prefix_sum() {
        let sums = [0, 10, 20, 30];
        for i in 30..120 {
            let block = binary_search_prefix_sum(i, &sums[..]);
            assert_eq!(block, (i / 10).min(3), "Failed at i={i}");
        }
    }

    /// Calculate the effect index from a particle index using a binary search of the base particle prefix sum.
    #[test]
    fn test_binary_search_prefix_sum() {
        let renderer = MockRenderer::new();
        let device = renderer.device();
        let queue = renderer.queue();

        println!(
            "max_compute_workgroup_storage_size = {}",
            device.limits().max_compute_workgroup_storage_size
        );

        // SAFETY : for debugging only
        #[allow(unsafe_code)]
        unsafe {
            device.wgpu_device().start_graphics_debugger_capture()
        };

        // Clamp max block size to the device's reported storage
        let max_block_size = device.limits().max_compute_workgroup_storage_size
            / DualKeyValuePair::SHADER_SIZE.get() as u32;
        println!("max_block_size = {}", max_block_size);
        let max_block_size = 1024;
        let num_particle = 1024.min(max_block_size);

        let byte_size = 4 + num_particle as u64 * DualKeyValuePair::SHADER_SIZE.get();
        let sort_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sort_buffer"),
            size: byte_size,
            usage: BufferUsages::STORAGE | BufferUsages::MAP_READ,
            mapped_at_creation: true,
        });
        let effects = [0, 35, 399, 1000];
        assert!((effects.len() as u32) < num_particle);
        {
            // Scope get_mapped_range_mut() to force a drop before unmap()
            {
                let slice: &mut [u8] = &mut sort_buffer.slice(..).get_mapped_range_mut();
                let header: &mut [u32] = cast_slice_mut(slice);
                header[0] = effects.len() as u32;
                let values: &mut [DualKeyValuePair] = cast_slice_mut(&mut slice[4..]);
                for (i, kv) in values.iter_mut().enumerate() {
                    if i < effects.len() {
                        kv.key = effects[i];
                    } else {
                        kv.key = 0xDEAD0000;
                    }
                    kv.key2 = i as f32 * 0.1;
                    kv.value = i as u32;
                    //println!("[#{}] k={} k2={} v={}", i, kv.key, kv.key2,
                    // kv.value);
                }
            }
            sort_buffer.unmap();
        }

        // Create GPU resources
        let bind_group_layout = device.create_bind_group_layout(
            "bind_group_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::COMPUTE,
                storage_buffer_sized(false, None),
            ),
        );
        let bind_group = device.create_bind_group(
            None,
            &bind_group_layout,
            &BindGroupEntries::single(sort_buffer.as_entire_binding()),
        );
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let src = VFX_SORT_WGSL
            .replace("#ifdef HAS_DUAL_KEY", "")
            .replace("#ifdef TEST", "")
            .replace("#endif", "");
        let shader_module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("vfx_sort"),
            source: ShaderSource::Wgsl(src.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("test_binary_search_prefix_sum"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("test_find_effect_from_particle"),
            compilation_options: PipelineCompilationOptions {
                constants: &[("blockSize", max_block_size as f64)],
                // Ensure the shader behaves even if memory is not zero-initialized
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        });

        // Dispatch test
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("test_binary_search_prefix_sum"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        // Submit command queue and wait for execution
        println!("Executing pipeline...");
        let command_buffer = encoder.finish();
        queue.submit([command_buffer]);
        let (tx, rx) = futures::channel::oneshot::channel();
        queue.on_submitted_work_done(move || {
            tx.send(()).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait);
        let _ = futures::executor::block_on(rx);
        println!("Pipeline executed");

        // SAFETY : for debugging only
        #[allow(unsafe_code)]
        unsafe {
            device.wgpu_device().stop_graphics_debugger_capture()
        };

        // Read back (GPU -> CPU)
        println!("Downloading result buffer from GPU to CPU...");
        let buffer_slice = sort_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait);
        let _result = futures::executor::block_on(rx);
        let view = buffer_slice.get_mapped_range();
        println!("Result buffer downloaded to CPU");

        // Validate content
        assert_eq!(view.len(), byte_size as usize);
        let view_slice: &[DualKeyValuePair] = cast_slice(&view[4..]);
        for (i, kv) in view_slice.iter().enumerate() {
            //println!("[#{}] k={} k2={} v={}", i, kv.key, kv.key2, kv.value);

            let mut effect_index = usize::MAX;
            for (idx, base_particle) in effects.iter().enumerate().rev() {
                if i as u32 >= *base_particle {
                    effect_index = idx;
                    break;
                }
            }
            assert!(effect_index <= effects.len());
            assert_eq!(
                effect_index as u32, kv.value,
                "Test failed for particle {} : expected effect index {}, got {}",
                i, effect_index, kv.value
            );
        }
    }
}
