use std::marker::PhantomData;

use bevy::{
    log::trace,
    render::{
        render_resource::{
            BindingResource, Buffer, BufferAddress, BufferBinding, BufferDescriptor, BufferUsages,
            ShaderSize, ShaderType,
        },
        renderer::RenderDevice,
    },
};
use bytemuck::Pod;
use wgpu::CommandEncoder;

struct BufferAndSize {
    pub buffer: Buffer,
    pub size: usize,
}

/// GPU-only buffer without CPU-side storage.
///
/// This is a rather specialized helper to allocate an array on the GPU and
/// manage its buffer, depending on the device constraints and the WGSL rules
/// for data alignment, and allowing to resize the buffer without losing its
/// content (so, scheduling a buffer-to-buffer copy on GPU after reallocatin).
///
/// The element type `T` needs to implement the following traits:
/// - [`Pod`] to prevent user error. This is not strictly necessary, as there's
///   no copy from or to CPU, but if the placeholder type is not POD this might
///   indicate some user error.
/// - [`ShaderSize`] to ensure a fixed footprint, to allow packing multiple
///   instances inside a single buffer. This therefore excludes any
///   runtime-sized array (T being the element type here; it will itself be part
///   of an array).
pub struct GpuBuffer<T: Pod + ShaderSize> {
    /// GPU buffer if already allocated, or `None` otherwise.
    buffer: Option<BufferAndSize>,
    /// GPU buffer usages.
    buffer_usage: BufferUsages,
    /// Optional GPU buffer name, for debugging.
    label: Option<String>,
    _phantom: PhantomData<T>,
}

impl<T: Pod + ShaderType + ShaderSize> Default for GpuBuffer<T> {
    fn default() -> Self {
        Self {
            buffer: None,
            buffer_usage: BufferUsages::all(),
            label: None,
            _phantom: PhantomData,
        }
    }
}

impl<T: Pod + ShaderType + ShaderSize> GpuBuffer<T> {
    /// Create a new collection.
    ///
    /// The buffer usage is always augmented by [`BufferUsages::COPY_SRC`] and
    /// [`BufferUsages::COPY_DST`] in order to allow buffer-to-buffer copy when
    /// reallocating, to preserve old content.
    ///
    /// # Panics
    ///
    /// Panics if `buffer_usage` contains [`BufferUsages::UNIFORM`] and the
    /// layout of the element type `T` does not meet the requirements of the
    /// uniform address space, as tested by
    /// [`ShaderType::assert_uniform_compat()`].
    ///
    /// [`BufferUsages::UNIFORM`]: bevy::render::render_resource::BufferUsages::UNIFORM
    pub fn new(buffer_usage: BufferUsages, label: Option<String>) -> Self {
        // GPU-aligned item size, compatible with WGSL rules
        let item_size = <T as ShaderSize>::SHADER_SIZE.get() as usize;
        trace!("GpuBuffer: item_size={}", item_size);
        if buffer_usage.contains(BufferUsages::UNIFORM) {
            <T as ShaderType>::assert_uniform_compat();
        }
        Self {
            // We need both COPY_SRC and COPY_DST for copy_buffer_to_buffer() on realloc
            buffer_usage: buffer_usage | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            label,
            ..Default::default()
        }
    }

    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref().map(|b| &b.buffer)
    }

    /// Get a binding for the entire buffer.
    #[inline]
    pub fn binding(&self) -> Option<BindingResource> {
        let buffer = self.buffer()?;
        Some(BindingResource::Buffer(BufferBinding {
            buffer,
            offset: 0,
            size: None, // entire buffer
        }))
    }

    #[inline]
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.buffer.as_ref().map(|b| b.size).unwrap_or(0)
    }

    /// Size in bytes of a single item in the buffer.
    #[inline]
    pub fn item_size(&self) -> usize {
        <T as ShaderSize>::SHADER_SIZE.get() as usize
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_none()
    }

    /// Reserve some capacity into the buffer.
    ///
    /// If the buffer was re-allocated, schedule a buffer-to-buffer copy via the
    /// given command encoder in order to preserve the content of the GPU
    /// buffer.
    ///
    /// # Returns
    ///
    /// `true` if the buffer was (re)allocated, or `false` if an existing buffer
    /// was reused which already had enough capacity.
    pub fn reserve(
        &mut self,
        capacity: usize,
        device: &RenderDevice,
        command_encoder: &mut CommandEncoder,
    ) -> bool {
        if capacity == 0 {
            return false;
        }
        let old_capacity = self.capacity();
        if capacity > old_capacity {
            let size = self.item_size() * capacity;
            trace!(
                "reserve: increase capacity from {} to {} elements, new size {} bytes",
                old_capacity,
                capacity,
                size
            );
            let buffer = device.create_buffer(&BufferDescriptor {
                label: self.label.as_ref().map(|s| &s[..]),
                size: size as BufferAddress,
                usage: BufferUsages::COPY_DST | self.buffer_usage,
                mapped_at_creation: false,
            });
            if let Some(old_buffer) = self.buffer.take() {
                command_encoder.copy_buffer_to_buffer(
                    &old_buffer.buffer,
                    0,
                    &buffer,
                    0,
                    old_capacity as u64,
                );
            }
            self.buffer = Some(BufferAndSize {
                buffer,
                size: capacity,
            });
            true
        } else {
            false
        }
    }
}
