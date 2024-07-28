use std::num::NonZeroU64;

use bevy::{
    log::trace,
    render::{
        render_resource::{
            Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoder, ShaderSize,
            ShaderType,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::{cast_slice, Pod};
use copyless::VecHelper;

use crate::next_multiple_of;

/// Index of a row in a [`BufferTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferTableId(pub(crate) u32); // TEMP: pub(crate)

#[derive(Debug)]
struct AllocatedBuffer {
    /// Currently allocated buffer, of size equal to `size`.
    buffer: Buffer,
    /// Size of the currently allocated buffer, in bytes.
    size: usize,
    /// Previously allocated buffer if any, cached until the next buffer write
    /// so that old data can be copied into the newly-allocated buffer.
    old_buffer: Option<Buffer>,
    /// Size of the old buffer if any, in bytes.
    old_size: usize,
}

impl AllocatedBuffer {
    /// Get the size in bytes of the currently allocated GPU buffer.
    ///
    /// On capacity grow, the size is valid until the next buffer swap.
    pub fn allocated_size(&self) -> usize {
        if self.old_buffer.is_some() {
            self.old_size
        } else {
            self.size
        }
    }
}

/// GPU buffer holding a table with concurrent interleaved CPU/GPU access.
///
/// The buffer table data structure represents a GPU buffer holding a table made
/// of individual rows. Each row of the table has the same layout (same size),
/// and can be allocated (assigned to an existing index) or free (available for
/// future allocation). The data structure manages a free list of rows, and copy
/// of rows modified on CPU to the GPU without touching other rows. This ensures
/// that existing rows in the GPU buffer can be accessed and modified by the GPU
/// without being overwritten by the CPU and without the need for the CPU to
/// read the data back from GPU into CPU memory.
///
/// The element type `T` needs to implement the following traits:
/// - [`Pod`] to allow copy.
/// - [`ShaderType`] because it needs to be mapped for a shader.
/// - [`ShaderSize`] to ensure a fixed footprint, to allow packing multiple
///   instances inside a single buffer. This therefore excludes any
///   runtime-sized array.
///
/// This is similar to a [`BufferVec`] or [`AlignedBufferVec`], but unlike those
/// data structures a buffer table preserves rows modified by the GPU without
/// overwriting. This is useful when the buffer is also modified by GPU shaders,
/// so neither the CPU side nor the GPU side have an up-to-date view of the
/// entire table, and so the CPU cannot re-upload the entire table on changes.
///
/// # Usage
///
/// - During the [`RenderStage::Prepare`] stage, call
///   [`clear_previous_frame_resizes()`] to clear any stale buffer from the
/// previous frame. Then insert new rows with [`insert()`] and if you made
/// changes call [`allocate_gpu()`] at the end to allocate any new buffer
/// needed.
/// - During the [`RenderStage::Render`] stage, call [`write_buffer()`] from a
///   command encoder before using any row, to perform any buffer resize copy
///   pending.
///
/// [`BufferVec`]: bevy::render::render_resource::BufferVec
/// [`AlignedBufferVec`]: bevy_haanabi::render::aligned_buffer_vec::AlignedBufferVec
#[derive(Debug)]
pub struct BufferTable<T: Pod + ShaderSize> {
    /// GPU buffer if already allocated, or `None` otherwise.
    buffer: Option<AllocatedBuffer>,
    /// GPU buffer usages.
    buffer_usage: BufferUsages,
    /// Optional GPU buffer name, for debugging.
    label: Option<String>,
    /// Size of a single buffer element, in bytes, in CPU memory (Rust layout).
    item_size: usize,
    /// Size of a single buffer element, in bytes, aligned to GPU memory
    /// constraints.
    aligned_size: usize,
    /// Capacity of the buffer, in number of rows.
    capacity: usize,
    /// Size of the "active" portion of the table, which includes allocated rows
    /// and any row in the free list. All other rows in the
    /// `active_size..capacity` range are implicitly unallocated.
    active_size: usize,
    /// Free list of rows available in the GPU buffer for a new allocation. This
    /// only contains indices in the `0..active_size` range; all row indices in
    /// `active_size..capacity` are assumed to be unallocated.
    free_indices: Vec<u32>,
    /// Pending values accumulated on CPU and not yet written to GPU, and their
    /// rows.
    pending_values: Vec<(u32, T)>,
    /// Extra pending values accumulated on CPU like `pending_values`, but for
    /// which there's not enough space in the current GPU buffer. Those values
    /// are sorted in index order, occupying the range `buffer.size..`.
    extra_pending_values: Vec<T>,
}

impl<T: Pod + ShaderSize> Default for BufferTable<T> {
    fn default() -> Self {
        let item_size = size_of::<T>();
        let aligned_size = <T as ShaderSize>::SHADER_SIZE.get() as usize;
        assert!(aligned_size >= item_size);
        Self {
            buffer: None,
            buffer_usage: BufferUsages::all(),
            label: None,
            item_size,
            aligned_size,
            capacity: 0,
            active_size: 0,
            free_indices: Vec::new(),
            pending_values: Vec::new(),
            extra_pending_values: Vec::new(),
        }
    }
}

impl<T: Pod + ShaderSize> BufferTable<T> {
    /// Create a new collection.
    ///
    /// `item_align` is an optional additional alignment for items in the
    /// collection. If greater than the natural alignment dictated by WGSL
    /// rules, this extra alignment is enforced. Otherwise it's ignored (so you
    /// can pass `0` to ignore).
    ///
    /// # Panics
    ///
    /// Panics if `buffer_usage` contains [`BufferUsages::UNIFORM`] and the
    /// layout of the element type `T` does not meet the requirements of the
    /// uniform address space, as tested by
    /// [`ShaderType::assert_uniform_compat()`].
    ///
    /// [`BufferUsages::UNIFORM`]: bevy::render::render_resource::BufferUsages::UNIFORM
    pub fn new(
        buffer_usage: BufferUsages,
        item_align: Option<NonZeroU64>,
        label: Option<String>,
    ) -> Self {
        // GPU-aligned item size, compatible with WGSL rules
        let item_size = <T as ShaderSize>::SHADER_SIZE.get() as usize;
        // Extra manual alignment for device constraints
        let aligned_size = if let Some(item_align) = item_align {
            let item_align = item_align.get() as usize;
            let aligned_size = next_multiple_of(item_size, item_align);
            assert!(aligned_size >= item_size);
            assert!(aligned_size % item_align == 0);
            aligned_size
        } else {
            item_size
        };
        trace!(
            "BufferTable[\"{}\"]: item_size={} aligned_size={}",
            label.as_ref().unwrap_or(&String::new()),
            item_size,
            aligned_size
        );
        if buffer_usage.contains(BufferUsages::UNIFORM) {
            <T as ShaderType>::assert_uniform_compat();
        }
        Self {
            // Need COPY_SRC and COPY_DST to copy from old to new buffer on resize
            buffer_usage: buffer_usage | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            aligned_size,
            label,
            ..Default::default()
        }
    }

    /// Reference to the GPU buffer, if already allocated.
    ///
    /// This reference corresponds to the currently allocated GPU buffer, which
    /// may not contain all data since the last [`insert()`] call, and could
    /// become invalid if a new larger buffer needs to be allocated to store the
    /// pending values inserted with [`insert()`].
    ///
    /// [`insert()]`: BufferTable::insert
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref().map(|ab| &ab.buffer)
    }

    /// Maximum number of rows the table can hold without reallocation.
    ///
    /// This is the maximum number of rows that can be added to the table
    /// without forcing a new GPU buffer to be allocated and a copy from the old
    /// to the new buffer.
    ///
    /// Note that this doesn't imply that no GPU buffer allocation will ever
    /// occur; if a GPU buffer was never allocated, and there are pending
    /// CPU rows to insert, then a new buffer will be allocated on next
    /// update with this capacity.
    #[inline]
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current number of rows in use in the table.
    #[inline]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.active_size - self.free_indices.len()
    }

    /// Size of a single row in the table, in bytes, aligned to GPU constraints.
    #[inline]
    #[allow(dead_code)]
    pub fn aligned_size(&self) -> usize {
        self.aligned_size
    }

    /// Is the table empty?
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.active_size == 0
    }

    /// Clear all rows of the table without deallocating any existing GPU
    /// buffer.
    ///
    /// This operation only updates the CPU cache of the table, without touching
    /// any GPU buffer. On next GPU buffer update, the GPU buffer will be
    /// deallocated.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.pending_values.clear();
        self.extra_pending_values.clear();
        self.free_indices.clear();
        self.active_size = 0;
    }

    /// Clear any stale buffer used for resize in the previous frame during
    /// rendering while the data structure was immutable.
    ///
    /// This must be called before any new [`insert()`].
    ///
    /// [`insert()`]: crate::BufferTable::insert
    pub fn clear_previous_frame_resizes(&mut self) {
        if let Some(ab) = self.buffer.as_mut() {
            ab.old_buffer = None;
            ab.old_size = 0;
        }
    }

    /// Insert a new row into the table.
    ///
    /// For performance reasons, this buffers the row content on the CPU until
    /// the next GPU update, to minimize the number of CPU to GPU transfers.
    pub fn insert(&mut self, value: T) -> BufferTableId {
        trace!(
            "Inserting into table buffer with {} free indices, capacity: {}, active_size: {}",
            self.free_indices.len(),
            self.capacity,
            self.active_size
        );
        let index = if self.free_indices.is_empty() {
            let index = self.active_size;
            if index == self.capacity {
                self.capacity += 1;
            }
            debug_assert!(index < self.capacity);
            self.active_size += 1;
            index as u32
        } else {
            // Note: this is inefficient O(n) but we need to apply the same logic as the
            // EffectCache because we rely on indices being in sync.
            self.free_indices.remove(0)
        } as usize;
        let allocated_size = self
            .buffer
            .as_ref()
            .map(|ab| ab.allocated_size())
            .unwrap_or(0);
        debug_assert!(allocated_size % self.aligned_size == 0);
        trace!(
            "Found free index {}, capacity: {}, active_size: {}, allocated_size: {}",
            index,
            self.capacity,
            self.active_size,
            allocated_size
        );
        let allocated_count = allocated_size / self.aligned_size;
        if index < allocated_count {
            self.pending_values.alloc().init((index as u32, value));
        } else {
            let extra_index = index - allocated_count;
            if extra_index < self.extra_pending_values.len() {
                self.extra_pending_values[extra_index] = value;
            } else {
                self.extra_pending_values.alloc().init(value);
            }
        }
        BufferTableId(index as u32)
    }

    /// Remove a row from the table.
    #[allow(dead_code)]
    pub fn remove(&mut self, id: BufferTableId) {
        let index = id.0;
        assert!(index < self.active_size as u32);

        // let allocated_size = self
        //     .buffer
        //     .as_ref()
        //     .map(|ab| ab.allocated_size())
        //     .unwrap_or(0);
        // debug_assert!(allocated_size % self.aligned_size == 0);
        // let allocated_count = allocated_size / self.aligned_size;

        // If this is the last item in the active zone, just shrink the active zone
        // (implicit free list).
        if index == self.active_size as u32 - 1 {
            self.active_size -= 1;
            self.capacity -= 1;
        } else {
            // This is very inefficient but we need to apply the same logic as the
            // EffectCache because we rely on indices being in sync.
            let pos = self
                .free_indices
                .binary_search(&index) // will fail
                .unwrap_or_else(|e| e); // will get position of insertion
            self.free_indices.insert(pos, index);
        }
    }

    /// Allocate any GPU buffer if needed, based on the most recent capacity
    /// requested.
    ///
    /// This should be called only once per frame after all allocation requests
    /// have been made via [`insert()`] but before the GPU buffer is actually
    /// updated. This is an optimization to enable allocating the GPU buffer
    /// earlier than it's actually needed. Calling this multiple times will work
    /// but will be inefficient and allocate GPU buffers for nothing. Not
    /// calling it is safe, as the next update will call it just-in-time anyway.
    ///
    /// # Returns
    ///
    /// Returns `true` if a new buffer was (re-)allocated, to indicate any bind
    /// group needs to be re-created.
    ///
    /// [`insert()]`: crate::render::BufferTable::insert
    pub fn allocate_gpu(&mut self, device: &RenderDevice, queue: &RenderQueue) -> bool {
        // The allocated capacity is the capacity of the currently allocated GPU buffer,
        // which can be different from the expected capacity (self.capacity) for next
        // update.
        let allocated_size = self.buffer.as_ref().map(|ab| ab.size).unwrap_or(0);
        let size = self.aligned_size * self.capacity;
        let reallocated = if size > allocated_size {
            trace!(
                "reserve: increase capacity from {} to {} elements, new size {} bytes",
                allocated_size / self.aligned_size,
                self.capacity,
                size
            );

            // Create the new buffer, swapping with the old one if any
            let new_buffer = device.create_buffer(&BufferDescriptor {
                label: self.label.as_ref().map(|s| &s[..]),
                size: size as BufferAddress,
                usage: self.buffer_usage,
                mapped_at_creation: false,
            });
            if let Some(ab) = self.buffer.as_mut() {
                // If there's any data currently in the GPU buffer, we need to copy it on next
                // update to preserve it, but only if there's no pending copy already.
                if self.active_size > 0 && ab.old_buffer.is_none() {
                    ab.old_buffer = Some(ab.buffer.clone()); // TODO: swap
                    ab.old_size = ab.size;
                }
                ab.buffer = new_buffer;
                ab.size = size;
            } else {
                self.buffer = Some(AllocatedBuffer {
                    buffer: new_buffer,
                    size,
                    old_buffer: None,
                    old_size: 0,
                });
            }

            true
        } else {
            false
        };

        // Immediately schedule a copy of all pending rows.
        // - For new rows, copy directly in the new buffer, which is the only one that
        //   can hold them since the old buffer was too small.
        // - For old rows, copy into the old buffer because the old-to-new buffer copy
        //   will be executed during a command queue while any CPU to GPU upload is
        //   prepended before the next command queue. To ensure things do get out of
        //   order with the CPU upload overwriting the GPU-to-GPU copy, make sure those
        //   two are disjoint.
        if let Some(ab) = self.buffer.as_ref() {
            let buffer = ab.old_buffer.as_ref().unwrap_or(&ab.buffer);
            for (index, content) in self.pending_values.drain(..) {
                let byte_size = self.aligned_size;
                let byte_offset = byte_size * index as usize;

                // Copy Rust value into a GPU-ready format, including GPU padding.
                // TODO - Do that in insert()!
                let mut aligned_buffer: Vec<u8> = vec![0; self.aligned_size];
                let src: &[u8] = cast_slice(std::slice::from_ref(&content));
                let dst_range = ..self.item_size;
                trace!(
                    "+ copy: index={} src={:?} dst={:?} byte_offset={} byte_size={}",
                    index,
                    src.as_ptr(),
                    dst_range,
                    byte_offset,
                    byte_size
                );
                let dst = &mut aligned_buffer[dst_range];
                dst.copy_from_slice(src);

                // Upload to GPU
                // TODO - Merge contiguous blocks into a single write_buffer()
                let bytes: &[u8] = cast_slice(&aligned_buffer);
                queue.write_buffer(buffer, byte_offset as u64, bytes);
            }

            // If there's any extra values, this means the buffer was (re)allocated, so we
            // can schedule copies of those rows into the new buffer which has enough
            // capacity for them.
            if !self.extra_pending_values.is_empty() {
                let base_size = ab.old_size;
                debug_assert!(base_size % self.aligned_size == 0);
                let base_count = base_size / self.aligned_size;
                let buffer = &ab.buffer;
                for (rel_index, content) in self.extra_pending_values.drain(..).enumerate() {
                    let index = base_count + rel_index;
                    let byte_size = self.aligned_size; // single row
                    let byte_offset = byte_size * index;

                    // Copy Rust value into a GPU-ready format, including GPU padding.
                    // TODO - Do that in insert()!
                    let mut aligned_buffer: Vec<u8> = vec![0; self.aligned_size];
                    let src: &[u8] = cast_slice(std::slice::from_ref(&content));
                    let dst_range = ..self.item_size;
                    trace!(
                        "+ copy: index={} src={:?} dst={:?} byte_offset={} byte_size={}",
                        index,
                        src.as_ptr(),
                        dst_range,
                        byte_offset,
                        byte_size
                    );
                    let dst = &mut aligned_buffer[dst_range];
                    dst.copy_from_slice(src);

                    // Upload to GPU
                    // TODO - Merge contiguous blocks into a single write_buffer()
                    // ESPECIALLY SINCE THIS IS TRIVIAL FOR THIS CASE!!!
                    let bytes: &[u8] = cast_slice(&aligned_buffer);
                    queue.write_buffer(buffer, byte_offset as u64, bytes);
                }
            }
        } else {
            debug_assert!(self.pending_values.is_empty());
            debug_assert!(self.extra_pending_values.is_empty());
        }

        reallocated
    }

    /// Write CPU data to the GPU buffer, (re)allocating it as needed.
    pub fn write_buffer(&self, encoder: &mut CommandEncoder) {
        // Check if there's any work to do: either some pending values to upload or some
        // existing buffer to copy into a newly-allocated one.
        if self.pending_values.is_empty()
            && self
                .buffer
                .as_ref()
                .map(|ab| ab.old_buffer.is_none())
                .unwrap_or(true)
        {
            return;
        }

        trace!(
            "write_buffer: pending_values.len={} item_size={} aligned_size={} buffer={:?}",
            self.pending_values.len(),
            self.item_size,
            self.aligned_size,
            self.buffer,
        );

        // If there's no more GPU buffer, there's nothing to do
        let Some(ab) = self.buffer.as_ref() else {
            return;
        };

        // Copy any old buffer into the new one, and clear the old buffer. Note that we
        // only clear the ref-counted reference to the buffer, not the actual buffer,
        // which stays alive until the copy is done (but we don't need to care about
        // keeping it alive, wgpu does that for us).
        if let Some(old_buffer) = ab.old_buffer.as_ref() {
            trace!("Copy old buffer id {:?} of size {} bytes into newly-allocated buffer {:?} of size {} bytes.", old_buffer.id(), ab.old_size, ab.buffer.id(), ab.size);
            encoder.copy_buffer_to_buffer(old_buffer, 0, &ab.buffer, 0, ab.old_size as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::Vec3;
    use bytemuck::{Pod, Zeroable};

    use super::*;

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
    pub(crate) struct GpuDummy {
        pub v: Vec3,
    }

    #[repr(C)]
    #[derive(Debug, Default, Clone, Copy, Pod, Zeroable, ShaderType)]
    pub(crate) struct GpuDummyComposed {
        pub simple: GpuDummy,
        pub tag: u32,
        // GPU padding to 16 bytes due to GpuDummy forcing align to 16 bytes
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Pod, Zeroable, ShaderType)]
    pub(crate) struct GpuDummyLarge {
        pub simple: GpuDummy,
        pub tag: u32,
        pub large: [f32; 128],
    }

    #[test]
    fn table_sizes() {
        // Rust
        assert_eq!(size_of::<GpuDummy>(), 12);
        assert_eq!(align_of::<GpuDummy>(), 4);
        assert_eq!(size_of::<GpuDummyComposed>(), 16); // tight packing
        assert_eq!(align_of::<GpuDummyComposed>(), 4);
        assert_eq!(size_of::<GpuDummyLarge>(), 132 * 4); // tight packing
        assert_eq!(align_of::<GpuDummyLarge>(), 4);

        // GPU
        assert_eq!(<GpuDummy as ShaderType>::min_size().get(), 16); // Vec3 gets padded to 16 bytes
        assert_eq!(<GpuDummy as ShaderSize>::SHADER_SIZE.get(), 16);
        assert_eq!(<GpuDummyComposed as ShaderType>::min_size().get(), 32); // align is 16 bytes, forces padding
        assert_eq!(<GpuDummyComposed as ShaderSize>::SHADER_SIZE.get(), 32);
        assert_eq!(<GpuDummyLarge as ShaderType>::min_size().get(), 544); // align is 16 bytes, forces padding
        assert_eq!(<GpuDummyLarge as ShaderSize>::SHADER_SIZE.get(), 544);

        for (item_align, expected_aligned_size) in [
            (0, 16),
            (4, 16),
            (8, 16),
            (16, 16),
            (32, 32),
            (256, 256),
            (512, 512),
        ] {
            let mut table = BufferTable::<GpuDummy>::new(
                BufferUsages::STORAGE,
                NonZeroU64::new(item_align),
                None,
            );
            assert_eq!(table.aligned_size(), expected_aligned_size);
            assert!(table.is_empty());
            table.insert(GpuDummy::default());
            assert!(!table.is_empty());
            assert_eq!(table.len(), 1);
        }

        for (item_align, expected_aligned_size) in [
            (0, 32),
            (4, 32),
            (8, 32),
            (16, 32),
            (32, 32),
            (256, 256),
            (512, 512),
        ] {
            let mut table = BufferTable::<GpuDummyComposed>::new(
                BufferUsages::STORAGE,
                NonZeroU64::new(item_align),
                None,
            );
            assert_eq!(table.aligned_size(), expected_aligned_size);
            assert!(table.is_empty());
            table.insert(GpuDummyComposed::default());
            assert!(!table.is_empty());
            assert_eq!(table.len(), 1);
        }

        for (item_align, expected_aligned_size) in [
            (0, 544),
            (4, 544),
            (8, 544),
            (16, 544),
            (32, 544),
            (256, 768),
            (512, 1024),
        ] {
            let mut table = BufferTable::<GpuDummyLarge>::new(
                BufferUsages::STORAGE,
                NonZeroU64::new(item_align),
                None,
            );
            assert_eq!(table.aligned_size(), expected_aligned_size);
            assert!(table.is_empty());
            table.insert(GpuDummyLarge {
                simple: Default::default(),
                tag: 0,
                large: [0.; 128],
            });
            assert!(!table.is_empty());
            assert_eq!(table.len(), 1);
        }
    }
}

#[cfg(all(test, feature = "gpu_tests"))]
mod gpu_tests {
    use std::fmt::Write;

    use bevy::render::render_resource::BufferSlice;
    use tests::*;
    use wgpu::{BufferView, CommandBuffer};

    use super::*;
    use crate::test_utils::MockRenderer;

    /// Read data from GPU back into CPU memory.
    ///
    /// This call blocks until the data is available on CPU. Used for testing
    /// only.
    fn read_back_gpu<'a>(device: &RenderDevice, slice: BufferSlice<'a>) -> BufferView<'a> {
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        let result = futures::executor::block_on(rx);
        assert!(result.is_ok());
        slice.get_mapped_range()
    }

    /// Submit a command buffer to GPU and wait for completion.
    ///
    /// This call blocks until the GPU executed the command buffer. Used for
    /// testing only.
    fn submit_gpu_and_wait(
        device: &RenderDevice,
        queue: &RenderQueue,
        command_buffer: CommandBuffer,
    ) {
        // Queue command
        queue.submit([command_buffer]);

        // Register callback to observe completion
        let (tx, rx) = futures::channel::oneshot::channel();
        queue.on_submitted_work_done(move || {
            tx.send(()).unwrap();
        });

        // Poll device, checking for completion and raising callback
        device.poll(wgpu::Maintain::Wait);

        // Wait for callback to be raised. This was need in previous versions, however
        // it's a bit unclear if it's still needed or if device.poll() is enough to
        // guarantee that the command was executed.
        let _ = futures::executor::block_on(rx);
    }

    /// Convert a byte slice to a string of hexadecimal values separated by a
    /// blank space.
    fn to_hex_string(slice: &[u8]) -> String {
        let len = slice.len();
        let num_chars = len * 3 - 1;
        let mut s = String::with_capacity(num_chars);
        for b in &slice[..len - 1] {
            write!(&mut s, "{:02x} ", *b).unwrap();
        }
        write!(&mut s, "{:02x}", slice[len - 1]).unwrap();
        debug_assert_eq!(s.len(), num_chars);
        s
    }

    fn write_buffers_and_wait<T: Pod + ShaderSize>(
        table: &BufferTable<T>,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test"),
        });
        table.write_buffer(&mut encoder);
        let command_buffer = encoder.finish();
        submit_gpu_and_wait(device, queue, command_buffer);
        println!("Buffer written to GPU");
    }

    #[test]
    fn table_write() {
        let renderer = MockRenderer::new();
        let device = renderer.device();
        let queue = renderer.queue();

        let item_align = device.limits().min_storage_buffer_offset_alignment as u64;
        println!("min_storage_buffer_offset_alignment = {item_align}");
        let mut table = BufferTable::<GpuDummyComposed>::new(
            BufferUsages::STORAGE | BufferUsages::MAP_READ,
            NonZeroU64::new(item_align),
            None,
        );
        let final_align = item_align.max(<GpuDummyComposed as ShaderSize>::SHADER_SIZE.get());
        assert_eq!(table.aligned_size(), final_align as usize);

        // Initial state
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert_eq!(table.capacity(), 0);
        assert!(table.buffer.is_none());

        // This has no effect while the table is empty
        table.clear_previous_frame_resizes();
        table.allocate_gpu(&device, &queue);
        write_buffers_and_wait(&table, &device, &queue);
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert_eq!(table.capacity(), 0);
        assert!(table.buffer.is_none());

        // New frame
        table.clear_previous_frame_resizes();

        // Insert some entries
        let len = 3;
        for i in 0..len as u32 {
            let row = table.insert(GpuDummyComposed {
                tag: i + 1,
                ..Default::default()
            });
            assert_eq!(row.0, i);
        }
        assert!(!table.is_empty());
        assert_eq!(table.len(), len);
        assert!(table.capacity() >= len); // contract: could over-allocate...
        assert!(table.buffer.is_none()); // not yet allocated on GPU

        // Allocate GPU buffer for current requested state
        table.allocate_gpu(&device, &queue);
        assert!(!table.is_empty());
        assert_eq!(table.len(), len);
        assert!(table.capacity() >= len);
        let ab = table
            .buffer
            .as_ref()
            .expect("GPU buffer should be allocated after allocate_gpu()");
        assert!(ab.old_buffer.is_none()); // no previous copy
        assert_eq!(ab.size, table.aligned_size() * len);
        println!(
            "Allocated buffer #{:?} of size {} bytes",
            ab.buffer.id(),
            ab.size
        );
        let ab_buffer = ab.buffer.clone();

        // Another allocate_gpu() is a no-op
        table.allocate_gpu(&device, &queue);
        assert!(!table.is_empty());
        assert_eq!(table.len(), len);
        assert!(table.capacity() >= len);
        let ab = table
            .buffer
            .as_ref()
            .expect("GPU buffer should be allocated after allocate_gpu()");
        assert!(ab.old_buffer.is_none()); // no previous copy
        assert_eq!(ab.size, table.aligned_size() * len);
        assert_eq!(ab_buffer.id(), ab.buffer.id()); // same buffer

        // Write buffer (CPU -> GPU)
        write_buffers_and_wait(&table, &device, &queue);

        {
            // Read back (GPU -> CPU)
            let buffer = table.buffer().expect("Buffer was not allocated").clone(); // clone() for lifetime
            {
                let slice = buffer.slice(..);
                let view = read_back_gpu(&device, slice);
                println!(
                    "GPU data read back to CPU for validation: {} bytes",
                    view.len()
                );

                // Validate content
                assert_eq!(view.len(), final_align as usize * table.capacity());
                for i in 0..len {
                    let offset = i * final_align as usize;
                    let item_size = size_of::<GpuDummyComposed>();
                    let src = &view[offset..offset + 16];
                    println!("{}", to_hex_string(src));
                    let dummy_composed: &[GpuDummyComposed] =
                        cast_slice(&view[offset..offset + item_size]);
                    assert_eq!(dummy_composed[0].tag, (i + 1) as u32);
                }
            }
            buffer.unmap();
        }

        // New frame
        table.clear_previous_frame_resizes();

        // Insert more entries
        let old_capacity = table.capacity();
        let mut len = len as u32;
        while table.capacity() == old_capacity {
            let row = table.insert(GpuDummyComposed {
                tag: len + 1,
                ..Default::default()
            });
            assert_eq!(row.0, len);
            len += 1;
        }
        println!(
            "Added {} rows to grow capacity from {} to {}.",
            len - 3,
            old_capacity,
            table.capacity()
        );
        let len = len as usize;

        // This re-allocates a new GPU buffer because the capacity changed
        table.allocate_gpu(&device, &queue);
        assert!(!table.is_empty());
        assert_eq!(table.len(), len);
        assert!(table.capacity() >= len);
        let ab = table
            .buffer
            .as_ref()
            .expect("GPU buffer should be allocated after allocate_gpu()");
        assert_eq!(ab.size, table.aligned_size() * len);
        assert!(ab.old_buffer.is_some()); // old buffer to copy
        assert_ne!(ab.old_buffer.as_ref().unwrap().id(), ab.buffer.id());
        println!(
            "Allocated new buffer #{:?} of size {} bytes",
            ab.buffer.id(),
            ab.size
        );

        // Write buffer (CPU -> GPU)
        write_buffers_and_wait(&table, &device, &queue);

        {
            // Read back (GPU -> CPU)
            let buffer = table.buffer().expect("Buffer was not allocated").clone(); // clone() for lifetime
            {
                let slice = buffer.slice(..);
                let view = read_back_gpu(&device, slice);
                println!(
                    "GPU data read back to CPU for validation: {} bytes",
                    view.len()
                );

                // Validate content
                assert_eq!(view.len(), final_align as usize * table.capacity());
                for i in 0..len {
                    let offset = i * final_align as usize;
                    let item_size = size_of::<GpuDummyComposed>();
                    let src = &view[offset..offset + 16];
                    println!("{}", to_hex_string(src));
                    let dummy_composed: &[GpuDummyComposed] =
                        cast_slice(&view[offset..offset + item_size]);
                    assert_eq!(dummy_composed[0].tag, (i + 1) as u32);
                }
            }
            buffer.unmap();
        }

        // New frame
        table.clear_previous_frame_resizes();

        // Delete the last allocated row
        let old_capacity = table.capacity();
        let len = len - 1;
        table.remove(BufferTableId(len as u32));
        println!(
            "Removed last row to shrink capacity from {} to {}.",
            old_capacity,
            table.capacity()
        );

        // This doesn't do anything since we removed a row only
        table.allocate_gpu(&device, &queue);
        assert!(!table.is_empty());
        assert_eq!(table.len(), len);
        assert!(table.capacity() >= len);
        let ab = table
            .buffer
            .as_ref()
            .expect("GPU buffer should be allocated after allocate_gpu()");
        assert_eq!(ab.size, table.aligned_size() * (len + 1)); // GPU buffer kept its size
        assert!(ab.old_buffer.is_none());

        // Write buffer (CPU -> GPU)
        write_buffers_and_wait(&table, &device, &queue);

        {
            // Read back (GPU -> CPU)
            let buffer = table.buffer().expect("Buffer was not allocated").clone(); // clone() for lifetime
            {
                let slice = buffer.slice(..);
                let view = read_back_gpu(&device, slice);
                println!(
                    "GPU data read back to CPU for validation: {} bytes",
                    view.len()
                );

                // Validate content
                assert!(view.len() >= final_align as usize * table.capacity()); // note the >=, the buffer is over-allocated
                for i in 0..len {
                    let offset = i * final_align as usize;
                    let item_size = size_of::<GpuDummyComposed>();
                    let src = &view[offset..offset + 16];
                    println!("{}", to_hex_string(src));
                    let dummy_composed: &[GpuDummyComposed] =
                        cast_slice(&view[offset..offset + item_size]);
                    assert_eq!(dummy_composed[0].tag, (i + 1) as u32);
                }
            }
            buffer.unmap();
        }

        // New frame
        table.clear_previous_frame_resizes();

        // Delete the first allocated row
        let old_capacity = table.capacity();
        let len = len - 1;
        table.remove(BufferTableId(0));
        println!(
            "Removed first row to shrink capacity from {} to {}.",
            old_capacity,
            table.capacity()
        );

        // This doesn't do anything since we removed a row only
        table.allocate_gpu(&device, &queue);
        assert!(!table.is_empty());
        assert_eq!(table.len(), len);
        assert!(table.capacity() >= len);
        let ab = table
            .buffer
            .as_ref()
            .expect("GPU buffer should be allocated after allocate_gpu()");
        assert_eq!(ab.size, table.aligned_size() * (len + 2)); // GPU buffer kept its size
        assert!(ab.old_buffer.is_none());

        // Write buffer (CPU -> GPU)
        write_buffers_and_wait(&table, &device, &queue);

        {
            // Read back (GPU -> CPU)
            let buffer = table.buffer().expect("Buffer was not allocated").clone(); // clone() for lifetime
            {
                let slice = buffer.slice(..);
                let view = read_back_gpu(&device, slice);
                println!(
                    "GPU data read back to CPU for validation: {} bytes",
                    view.len()
                );

                // Validate content
                assert!(view.len() >= final_align as usize * table.capacity()); // note the >=, the buffer is over-allocated
                for i in 0..len {
                    let offset = i * final_align as usize;
                    let item_size = size_of::<GpuDummyComposed>();
                    let src = &view[offset..offset + 16];
                    println!("{}", to_hex_string(src));
                    if i > 0 {
                        let dummy_composed: &[GpuDummyComposed] =
                            cast_slice(&view[offset..offset + item_size]);
                        assert_eq!(dummy_composed[0].tag, (i + 1) as u32);
                    }
                }
            }
            buffer.unmap();
        }

        // New frame
        table.clear_previous_frame_resizes();

        // Insert a row; this should get into row #0 in the buffer
        let mut len = len as u32;
        let row = table.insert(GpuDummyComposed {
            tag: 1,
            ..Default::default()
        });
        assert_eq!(row.0, 0);
        len += 1;
        println!(
            "Added 1 row to grow capacity from {} to {}.",
            old_capacity,
            table.capacity()
        );
        let len = len as usize;

        // This doesn't reallocate the GPU buffer since we used a free list entry
        table.allocate_gpu(&device, &queue);
        assert!(!table.is_empty());
        assert_eq!(table.len(), len);
        assert!(table.capacity() >= len);
        let ab = table
            .buffer
            .as_ref()
            .expect("GPU buffer should be allocated after allocate_gpu()");
        assert_eq!(ab.size, table.aligned_size() * 4); // 4 == last time we grew
        assert!(ab.old_buffer.is_none());

        // Write buffer (CPU -> GPU)
        write_buffers_and_wait(&table, &device, &queue);

        {
            // Read back (GPU -> CPU)
            let buffer = table.buffer().expect("Buffer was not allocated").clone(); // clone() for lifetime
            {
                let slice = buffer.slice(..);
                let view = read_back_gpu(&device, slice);
                println!(
                    "GPU data read back to CPU for validation: {} bytes",
                    view.len()
                );

                // Validate content
                assert!(view.len() >= final_align as usize * table.capacity());
                for i in 0..len {
                    let offset = i * final_align as usize;
                    let item_size = size_of::<GpuDummyComposed>();
                    let src = &view[offset..offset + 16];
                    println!("{}", to_hex_string(src));
                    let dummy_composed: &[GpuDummyComposed] =
                        cast_slice(&view[offset..offset + item_size]);
                    assert_eq!(dummy_composed[0].tag, (i + 1) as u32);
                }
            }
            buffer.unmap();
        }

        // New frame
        table.clear_previous_frame_resizes();

        // Insert a row; this should get into row #3 at the end of the allocated buffer
        let mut len = len as u32;
        let row = table.insert(GpuDummyComposed {
            tag: 4,
            ..Default::default()
        });
        assert_eq!(row.0, 3);
        len += 1;
        println!(
            "Added 1 row to grow capacity from {} to {}.",
            old_capacity,
            table.capacity()
        );
        let len = len as usize;

        // This doesn't reallocate the GPU buffer since we used an implicit free entry
        table.allocate_gpu(&device, &queue);
        assert!(!table.is_empty());
        assert_eq!(table.len(), len);
        assert!(table.capacity() >= len);
        let ab = table
            .buffer
            .as_ref()
            .expect("GPU buffer should be allocated after allocate_gpu()");
        assert_eq!(ab.size, table.aligned_size() * 4); // 4 == last time we grew
        assert!(ab.old_buffer.is_none());

        // Write buffer (CPU -> GPU)
        write_buffers_and_wait(&table, &device, &queue);

        {
            // Read back (GPU -> CPU)
            let buffer = table.buffer().expect("Buffer was not allocated").clone(); // clone() for lifetime
            {
                let slice = buffer.slice(..);
                let view = read_back_gpu(&device, slice);
                println!(
                    "GPU data read back to CPU for validation: {} bytes",
                    view.len()
                );

                // Validate content
                assert!(view.len() >= final_align as usize * table.capacity());
                for i in 0..len {
                    let offset = i * final_align as usize;
                    let item_size = size_of::<GpuDummyComposed>();
                    let src = &view[offset..offset + 16];
                    println!("{}", to_hex_string(src));
                    let dummy_composed: &[GpuDummyComposed] =
                        cast_slice(&view[offset..offset + item_size]);
                    assert_eq!(dummy_composed[0].tag, (i + 1) as u32);
                }
            }
            buffer.unmap();
        }
    }
}
