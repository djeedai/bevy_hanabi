use std::marker::PhantomData;

use bevy::{
    log::trace,
    render::{
        render_resource::{
            BindingResource, Buffer, BufferAddress, BufferDescriptor, BufferUsages, ShaderSize,
            ShaderType,
        },
        renderer::RenderDevice,
    },
};
use bytemuck::Pod;
use wgpu::CommandEncoder;

struct BufferAndSize {
    /// Allocate GPU buffer.
    pub buffer: Buffer,
    /// Size of the buffer, in number of elements.
    pub size: u32,
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
    /// Previous GPU buffer, pending copy.
    old_buffer: Option<BufferAndSize>,
    /// GPU buffer usages.
    buffer_usage: BufferUsages,
    /// Optional GPU buffer name, for debugging.
    label: Option<String>,
    /// Used size, in element count. Elements past this are all free. Elements
    /// with a lower index are either allocated or in the free list.
    used_size: u32,
    /// Free list.
    free_list: Vec<u32>,
    _phantom: PhantomData<T>,
}

impl<T: Pod + ShaderType + ShaderSize> Default for GpuBuffer<T> {
    fn default() -> Self {
        Self {
            buffer: None,
            old_buffer: None,
            buffer_usage: BufferUsages::all(),
            label: None,
            used_size: 0,
            free_list: vec![],
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
    #[allow(dead_code)]
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

    /// Create a new collection from an allocated buffer.
    ///
    /// The buffer usage must contain [`BufferUsages::COPY_SRC`] and
    /// [`BufferUsages::COPY_DST`] in order to allow buffer-to-buffer copy when
    /// reallocating, to preserve old content.
    ///
    /// # Panics
    ///
    /// Panics if `buffer_usage` doesn't contain [`BufferUsages::COPY_SRC`] or
    /// [`BufferUsages::COPY_DST`].
    ///
    /// Panics if `buffer_usage` contains [`BufferUsages::UNIFORM`] and the
    /// layout of the element type `T` does not meet the requirements of the
    /// uniform address space, as tested by
    /// [`ShaderType::assert_uniform_compat()`].
    ///
    /// [`BufferUsages::UNIFORM`]: bevy::render::render_resource::BufferUsages::UNIFORM
    pub fn new_allocated(buffer: Buffer, size: u32, label: Option<String>) -> Self {
        // GPU-aligned item size, compatible with WGSL rules
        let item_size = <T as ShaderSize>::SHADER_SIZE.get() as u32;
        let buffer_usage = buffer.usage();
        assert!(
            buffer_usage.contains(BufferUsages::COPY_SRC | BufferUsages::COPY_DST),
            "GpuBuffer requires COPY_SRC and COPY_DST buffer usages to allow copy on reallocation."
        );
        if buffer_usage.contains(BufferUsages::UNIFORM) {
            <T as ShaderType>::assert_uniform_compat();
        }
        trace!("GpuBuffer: item_size={}", item_size);
        Self {
            buffer: Some(BufferAndSize { buffer, size }),
            buffer_usage,
            label,
            ..Default::default()
        }
    }

    /// Clear the buffer.
    ///
    /// This doesn't de-allocate any GPU buffer.
    pub fn clear(&mut self) {
        self.free_list.clear();
        self.used_size = 0;
    }

    /// Allocate a new entry in the buffer.
    ///
    /// If the GPU buffer has not enough storage, or is not allocated yet, this
    /// schedules a (re-)allocation, which must be applied by calling
    /// [`allocate_gpu()`] once a frame after all [`allocate()`] calls were made
    /// for that frame.
    ///
    /// # Returns
    ///
    /// The index of the allocated entry.
    ///
    /// [`allocate_gpu()`]: Self::allocate_gpu
    /// [`allocate()`]: Self::allocate
    pub fn allocate(&mut self) -> u32 {
        if let Some(index) = self.free_list.pop() {
            index
        } else {
            // Note: we may return an index past the buffer capacity. This will instruct
            // allocate_gpu() to re-allocate the buffer.
            let index = self.used_size;
            self.used_size += 1;
            index
        }
    }

    /// Free an existing entry.
    ///
    /// # Panics
    ///
    /// In debug only, panics if the entry is not allocated (double-free). In
    /// non-debug, the behavior is undefined and will generally lead to bugs.
    // Currently we use GpuBuffer in sorting, and re-allocate everything each frame.
    #[allow(dead_code)]
    pub fn free(&mut self, index: u32) {
        if index < self.used_size {
            debug_assert!(
                !self.free_list.contains(&index),
                "Double-free in GpuBuffer at index #{}",
                index
            );
            self.free_list.push(index);
        }
    }

    /// Get the current GPU buffer, if allocated.
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref().map(|b| &b.buffer)
    }

    /// Get a binding for the entire GPU buffer, if allocated.
    #[inline]
    #[allow(dead_code)]
    pub fn as_entire_binding(&self) -> Option<BindingResource<'_>> {
        let buffer = self.buffer()?;
        Some(buffer.as_entire_binding())
    }

    /// Get the current buffer capacity, in element count.
    ///
    /// This is the CPU view of allocations, which counts the number of
    /// [`allocate()`] and [`free()`] calls.
    ///
    /// [`allocate()`]: Self::allocate
    /// [`free()`]: Self::allocate_gpu
    #[inline]
    #[allow(dead_code)]
    pub fn capacity(&self) -> u32 {
        debug_assert!(self.used_size >= self.free_list.len() as u32);
        self.used_size - self.free_list.len() as u32
    }

    /// Get the current GPU buffer capacity, in element count.
    ///
    /// Note that it is possible for [`allocate()`] to return an index greater
    /// than or equal to the value returned by [`capacity()`], at least
    /// temporarily until [`allocate_gpu()`] is called.
    ///
    /// [`allocate()`]: Self::allocate
    /// [`gpu_capacity()`]: Self::gpu_capacity
    /// [`allocate_gpu()`]: Self::allocate_gpu
    #[inline]
    pub fn gpu_capacity(&self) -> u32 {
        self.buffer.as_ref().map(|b| b.size).unwrap_or(0)
    }

    /// Size in bytes of a single item in the buffer.
    ///
    /// This is equal to [`ShaderSize::SHADER_SIZE`] for the buffer element `T`.
    #[inline]
    pub fn item_size(&self) -> usize {
        <T as ShaderSize>::SHADER_SIZE.get() as usize
    }

    /// Check if the buffer is empty.
    ///
    /// The check is based on the CPU representation of the buffer, that is the
    /// number of calls to [`allocate()`]. The buffer is considered empty if no
    /// [`allocate()`] call was made, or they all have been followed by a
    /// corresponding [`free()`] call. This makes no assumption about the GPU
    /// buffer.
    ///
    /// [`allocate()`]: Self::allocate
    /// [`free()`]: Self::free
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.used_size == 0
    }

    /// Allocate or reallocate the GPU buffer if needed.
    ///
    /// This allocates or reallocates a GPU buffer to ensure storage for all
    /// previous calls to [`allocate()`]. This is a no-op if a GPU buffer is
    /// already allocated and has sufficient storage.
    ///
    /// This should be called once a frame after any new [`allocate()`] in that
    /// frame. After this call, [`buffer()`] is guaranteed to return `Some(..)`.
    ///
    /// # Returns
    ///
    /// `true` if the buffer was (re)allocated, or `false` if an existing buffer
    /// was reused which already had enough capacity.
    ///
    /// [`reserve()`]: Self::reserve
    /// [`allocate()`]: Self::allocate
    /// [`buffer()`]: Self::buffer
    pub fn prepare_buffers(&mut self, render_device: &RenderDevice) -> bool {
        // Don't do anything if we still have some storage.
        let old_capacity = self.gpu_capacity();
        if self.used_size <= old_capacity {
            return false;
        }

        // Ensure we allocate at least 256 more entries than what we need this frame,
        // and round that to make it nicer for the GPU.
        let new_capacity = (self.used_size + 256).next_multiple_of(1024);
        if new_capacity <= old_capacity {
            return false;
        }

        // Save the old buffer, we will need to copy it to the new one later.
        assert!(self.old_buffer.is_none(), "Multiple calls to GpuTable::prepare_buffers() before write_buffers() was called to copy old content.");
        self.old_buffer = self.buffer.take();

        // Allocate a new buffer of the appropriate size.
        let byte_size = self.item_size() * new_capacity as usize;
        trace!(
            "prepare_buffers(): increase capacity from {} to {} elements, new size {} bytes",
            old_capacity,
            new_capacity,
            byte_size
        );
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: self.label.as_ref().map(|s| &s[..]),
            size: byte_size as BufferAddress,
            usage: BufferUsages::COPY_DST | self.buffer_usage,
            mapped_at_creation: false,
        });
        self.buffer = Some(BufferAndSize {
            buffer,
            size: new_capacity,
        });

        true
    }

    /// Schedule any pending buffer copy.
    ///
    /// If a new buffer was (re-)allocated this frame, this schedules a
    /// buffer-to-buffer copy from the old buffer to the new one, then releases
    /// the old buffer.
    ///
    /// This should be called once a frame after [`prepare_buffers()`]. This is
    /// a no-op if there's no need for a buffer copy.
    ///
    /// [`prepare_buffers()`]: Self::prepare_buffers
    pub fn write_buffers(&self, command_encoder: &mut CommandEncoder) {
        if let Some(old_buffer) = self.old_buffer.as_ref() {
            let new_buffer = self.buffer.as_ref().unwrap();
            assert!(
                new_buffer.size >= old_buffer.size,
                "Old buffer is smaller than the new one. This is unexpected."
            );
            command_encoder.copy_buffer_to_buffer(
                &old_buffer.buffer,
                0,
                &new_buffer.buffer,
                0,
                old_buffer.size as u64,
            );
        }
    }

    /// Clear any stale buffer used for resize in the previous frame during
    /// rendering while the data structure was immutable.
    ///
    /// This must be called before any new [`allocate()`].
    ///
    /// [`allocate()`]: Self::allocate
    pub fn clear_previous_frame_resizes(&mut self) {
        if let Some(old_buffer) = self.old_buffer.take() {
            old_buffer.buffer.destroy();
        }
    }
}
