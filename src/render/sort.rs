use std::num::{NonZeroU32, NonZeroU64};

use bevy::{
    asset::Handle,
    ecs::{resource::Resource, world::World},
    platform::collections::{hash_map::Entry, HashMap},
    render::{
        render_resource::{
            BindGroup, BindGroupLayout, Buffer, BufferId, CachedComputePipelineId,
            ComputePipelineDescriptor, PipelineCache, Shader,
        },
        renderer::RenderDevice,
    },
    utils::default,
};
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType, BufferBinding,
    BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoder, ShaderStages,
};

use super::{gpu_buffer::GpuBuffer, GpuDispatchIndirect, GpuEffectMetadata, StorageType};
use crate::{Attribute, ParticleLayout};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SortFillBindGroupLayoutKey {
    particle_min_binding_size: NonZeroU32,
    particle_ribbon_id_offset: u32,
    particle_age_offset: u32,
}

impl SortFillBindGroupLayoutKey {
    pub fn from_particle_layout(particle_layout: &ParticleLayout) -> Result<Self, ()> {
        let particle_ribbon_id_offset = particle_layout.offset(Attribute::RIBBON_ID).ok_or(())?;
        let particle_age_offset = particle_layout.offset(Attribute::AGE).ok_or(())?;
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
    indirect_buffer: GpuBuffer<GpuDispatchIndirect>,
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
            entry_point: "main".into(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let effect_metadata_min_binding_size = GpuEffectMetadata::aligned_size(
            render_device.limits().min_storage_buffer_offset_alignment,
        );
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
                        has_dynamic_offset: true,
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
                        min_binding_size: Some(effect_metadata_min_binding_size),
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
                entry_point: "main".into(),
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
                        // @group(0) @binding(0) var<storage, read_write> pairs: array<KeyValuePair>;
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
                        // @group(0) @binding(1) var<storage, read> particle_buffer: ParticleBuffer;
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: true,
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
                                has_dynamic_offset: true,
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
                    ],
                );
                let pipeline_id =
                    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                        label: Some("hanabi:pipeline:sort_fill".into()),
                        layout: vec![bind_group_layout.clone()],
                        shader: self.sort_fill_shader.clone(),
                        shader_defs: vec!["HAS_DUAL_KEY".into()],
                        entry_point: "main".into(),
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
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: &self.sort_buffer,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            // @group(0) @binding(1) var<storage, read> particle_buffer:
                            // ParticleBuffer;
                            BindGroupEntry {
                                binding: 1,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: particle,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            // @group(0) @binding(2) var<storage, read> indirect_index_buffer :
                            // array<u32>;
                            BindGroupEntry {
                                binding: 2,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: indirect_index,
                                    offset: 0,
                                    size: None,
                                }),
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
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: indirect_index_buffer,
                                    offset: 0,
                                    size: None,
                                }),
                            },
                            // @group(0) @binding(1) var<storage, read> sort_buffer : SortBuffer;
                            BindGroupEntry {
                                binding: 1,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: &self.sort_buffer,
                                    offset: 0,
                                    size: None,
                                }),
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
