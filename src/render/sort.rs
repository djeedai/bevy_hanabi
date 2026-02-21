use std::num::{NonZeroU32, NonZeroU64};

use bevy::{
    asset::Handle,
    ecs::{resource::Resource, world::World},
    platform::collections::{hash_map::Entry, HashMap},
    render::{
        render_resource::{
            binding_types::{
                storage_buffer, storage_buffer_read_only, storage_buffer_read_only_sized,
                storage_buffer_sized,
            },
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            BufferId, CachedComputePipelineId, CachedPipelineState, ComputePipelineDescriptor,
            PipelineCache, ShaderType,
        },
        renderer::RenderDevice,
    },
    shader::Shader,
    utils::default,
};
use bytemuck::{Pod, Zeroable};
use wgpu::{BufferBinding, BufferDescriptor, BufferUsages, CommandEncoder, ShaderStages};

use super::{gpu_buffer::GpuBuffer, GpuDispatchIndirectArgs, GpuEffectMetadata, StorageType};
use crate::{
    render::{GpuIndirectIndex, GpuSpawnerParams},
    Attribute, ParticleLayout,
};

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

/// GPU representation of a single dual-key value pair, with the added buffer
/// count as prefix. This is mainly used for shorcuts in bindings, not directly
/// as a type.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, ShaderType)]
struct GpuSortBufferSingleEntry {
    /// Number of key-value pairs to sort. This is the first element of the
    /// entire buffer.
    pub count: u32,
    /// Key for the first entry.
    pub key: u32,
    /// Secondary key for the first entry.
    pub key2: u32,
    /// Value for the first entry.
    pub value: u32,
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
    sort_fill_bind_group_layout_descs:
        HashMap<SortFillBindGroupLayoutKey, (BindGroupLayoutDescriptor, CachedComputePipelineId)>,
    /// Bind groups for group #0 of the sort-fill compute pass.
    sort_fill_bind_groups: HashMap<SortFillBindGroupKey, BindGroup>,
    /// Bind group layout descriptor for group #0 of the sort compute pass.
    sort_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// Bind group for group #0 of the sort compute pass.
    sort_bind_group: Option<BindGroup>,
    sort_copy_bind_group_layout_desc: BindGroupLayoutDescriptor,
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

        let sort_bind_group_layout_desc = BindGroupLayoutDescriptor::new(
            "hanabi:bgl:sort",
            &BindGroupLayoutEntries::single(
                ShaderStages::COMPUTE,
                storage_buffer_sized(false, Some(NonZeroU64::new(16).unwrap())), /* count + dual
                                                                                  * kv pair */
            ),
        );

        let sort_pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("hanabi:pipeline:sort".into()),
            layout: vec![sort_bind_group_layout_desc.clone()],
            shader: sort_shader,
            shader_defs: vec!["HAS_DUAL_KEY".into()],
            entry_point: Some("main".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let alignment = render_device.limits().min_storage_buffer_offset_alignment;
        let sort_copy_bind_group_layout_desc = BindGroupLayoutDescriptor::new(
            "hanabi:bgl:sort_copy",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // @group(0) @binding(0) var<storage, read_write> indirect_index_buffer :
                    // IndirectIndexBuffer;
                    storage_buffer::<GpuIndirectIndex>(false),
                    // @group(0) @binding(1) var<storage, read> sort_buffer : SortBuffer;
                    storage_buffer_read_only::<GpuSortBufferSingleEntry>(false),
                    // @group(0) @binding(2) var<storage, read_write> effect_metadata :
                    // EffectMetadata;
                    storage_buffer_sized(true, Some(GpuEffectMetadata::aligned_size(alignment))),
                    // @group(0) @binding(3) var<storage, read> spawner : Spawner;
                    storage_buffer_read_only_sized(
                        true,
                        Some(GpuSpawnerParams::aligned_size(alignment)),
                    ),
                ),
            ),
        );

        let sort_copy_pipeline_id =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("hanabi:pipeline:sort_copy".into()),
                layout: vec![sort_copy_bind_group_layout_desc.clone()],
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
            sort_fill_bind_group_layout_descs: default(),
            sort_fill_bind_groups: default(),
            sort_bind_group_layout_desc,
            // This bind group is created later, once the pipeline and its bind group layouts are
            // created by the pipeline cache. Technically we could create the bind group layout
            // immediately because the PipelineCache pretends to but actually creates them
            // on-the-fly in get_bind_group_layout(), but this is brittle as any behavior change
            // would break Hanabi. Instead we create this bind group alongside all others, which is
            // more consistent too.
            sort_bind_group: None,
            sort_copy_bind_group_layout_desc,
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

    pub fn ensure_sort_fill_bind_group_layout_desc(
        &mut self,
        pipeline_cache: &PipelineCache,
        particle_layout: &ParticleLayout,
    ) -> Result<&BindGroupLayoutDescriptor, ()> {
        let key = SortFillBindGroupLayoutKey::from_particle_layout(particle_layout)?;
        let (layout, _) = self
            .sort_fill_bind_group_layout_descs
            .entry(key)
            .or_insert_with(|| {
                let alignment = self
                    .render_device
                    .limits()
                    .min_storage_buffer_offset_alignment;
                let bind_group_layout_desc = BindGroupLayoutDescriptor::new(
                    "hanabi:bgl:sort_fill",
                    &BindGroupLayoutEntries::sequential(
                        ShaderStages::COMPUTE,
                        (
                            // @group(0) @binding(0) var<storage, read_write> sort_buffer :
                            // SortBuffer;
                            storage_buffer::<GpuSortBufferSingleEntry>(false),
                            // @group(0) @binding(1) var<storage, read> particle_buffer :
                            // RawParticleBuffer;
                            storage_buffer_read_only_sized(
                                false,
                                Some(key.particle_min_binding_size.into()),
                            ),
                            // @group(0) @binding(2) var<storage, read> indirect_index_buffer :
                            // array<u32>;
                            storage_buffer_read_only::<GpuIndirectIndex>(false),
                            // @group(0) @binding(3) var<storage, read_write> effect_metadata :
                            // EffectMetadata;
                            storage_buffer_sized(
                                true,
                                Some(GpuEffectMetadata::aligned_size(alignment)),
                            ),
                            // @group(0) @binding(4) var<storage, read> spawner : Spawner;
                            storage_buffer_read_only_sized(
                                true,
                                Some(GpuSpawnerParams::aligned_size(alignment)),
                            ),
                        ),
                    ),
                );
                let pipeline_id =
                    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                        label: Some("hanabi:pipeline:sort_fill".into()),
                        layout: vec![bind_group_layout_desc.clone()],
                        shader: self.sort_fill_shader.clone(),
                        shader_defs: vec!["HAS_DUAL_KEY".into()],
                        entry_point: Some("main".into()),
                        push_constant_ranges: vec![],
                        zero_initialize_workgroup_memory: false,
                    });
                (bind_group_layout_desc, pipeline_id)
            });
        Ok(layout)
    }

    // We currently only use the bind group layout internally in
    // ensure_sort_fill_bind_group()
    #[allow(dead_code)]
    pub fn get_sort_fill_bind_group_layout_desc(
        &self,
        particle_layout: &ParticleLayout,
    ) -> Option<&BindGroupLayoutDescriptor> {
        let key = SortFillBindGroupLayoutKey::from_particle_layout(particle_layout).ok()?;
        self.sort_fill_bind_group_layout_descs
            .get(&key)
            .map(|(layout, _)| layout)
    }

    pub fn get_sort_fill_pipeline_id(
        &self,
        particle_layout: &ParticleLayout,
    ) -> Option<CachedComputePipelineId> {
        let key = SortFillBindGroupLayoutKey::from_particle_layout(particle_layout).ok()?;
        self.sort_fill_bind_group_layout_descs
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
        pipeline_cache: &PipelineCache,
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
                let layout_desc = &self
                    .sort_fill_bind_group_layout_descs
                    .get(&key)
                    .ok_or(())?
                    .0;
                entry.insert(
                    self.render_device.create_bind_group(
                        "hanabi:bg:sort_fill",
                        &pipeline_cache.get_bind_group_layout(layout_desc),
                        &BindGroupEntries::sequential((
                            // @group(0) @binding(0) var<storage, read_write> pairs:
                            // array<KeyValuePair>;
                            self.sort_buffer.as_entire_binding(),
                            // @group(0) @binding(1) var<storage, read> particle_buffer:
                            // ParticleBuffer;
                            particle.as_entire_binding(),
                            // @group(0) @binding(2) var<storage, read> indirect_index_buffer :
                            // array<u32>;
                            indirect_index.as_entire_binding(),
                            // @group(0) @binding(3) var<storage, read> effect_metadata :
                            // EffectMetadata;
                            BufferBinding {
                                buffer: effect_metadata,
                                offset: 0,
                                size: Some(GpuEffectMetadata::aligned_size(
                                    self.render_device
                                        .limits()
                                        .min_storage_buffer_offset_alignment,
                                )),
                            },
                            // @group(0) @binding(4) var<storage, read> spawner : Spawner;
                            BufferBinding {
                                buffer: spawner_buffer,
                                offset: 0,
                                size: Some(GpuSpawnerParams::aligned_size(
                                    self.render_device
                                        .limits()
                                        .min_storage_buffer_offset_alignment,
                                )),
                            },
                        )),
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

    /// Ensure the bind group for the sort pass is created.
    pub fn ensure_sort_bind_group(
        &mut self,
        pipeline_cache: &PipelineCache,
    ) -> Result<&BindGroup, ()> {
        if self.sort_bind_group.is_none() {
            let sort_bind_group = self.render_device.create_bind_group(
                "hanabi:bg:sort",
                &pipeline_cache.get_bind_group_layout(&self.sort_bind_group_layout_desc),
                // @group(0) @binding(0) var<storage, read_write> pairs : array<KeyValuePair>;
                &BindGroupEntries::single(self.sort_buffer.as_entire_binding()),
            );
            self.sort_bind_group = Some(sort_bind_group);
        }
        Ok(self.sort_bind_group.as_ref().unwrap())
    }

    #[inline]
    pub fn sort_bind_group(&self) -> Option<&BindGroup> {
        self.sort_bind_group.as_ref()
    }

    pub fn ensure_sort_copy_bind_group(
        &mut self,
        indirect_index_buffer: &Buffer,
        effect_metadata_buffer: &Buffer,
        spawner_buffer: &Buffer,
        pipeline_cache: &PipelineCache,
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
                        "hanabi:bg:sort_copy",
                        &pipeline_cache
                            .get_bind_group_layout(&self.sort_copy_bind_group_layout_desc),
                        &BindGroupEntries::sequential((
                            // @group(0) @binding(0) var<storage, read_write> indirect_index_buffer
                            // : IndirectIndexBuffer;
                            indirect_index_buffer.as_entire_binding(),
                            // @group(0) @binding(1) var<storage, read> sort_buffer : SortBuffer;
                            self.sort_buffer.as_entire_binding(),
                            // @group(0) @binding(2) var<storage, read> effect_metadata :
                            // EffectMetadata;
                            BufferBinding {
                                buffer: effect_metadata_buffer,
                                offset: 0,
                                size: Some(GpuEffectMetadata::aligned_size(
                                    self.render_device
                                        .limits()
                                        .min_storage_buffer_offset_alignment,
                                )),
                            },
                            // @group(0) @binding(3) var<storage, read> spawner : Spawner;
                            BufferBinding {
                                buffer: spawner_buffer,
                                offset: 0,
                                size: Some(GpuSpawnerParams::aligned_size(
                                    self.render_device
                                        .limits()
                                        .min_storage_buffer_offset_alignment,
                                )),
                            },
                        )),
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
