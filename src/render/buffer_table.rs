use std::{
    borrow::Cow,
    num::{NonZeroU32, NonZeroU64},
    ops::Range,
};

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

/// Round a range start down to a given alignment, and return the new range and
/// the start offset inside the new range of the old range.
fn round_range_start_down(range: Range<u64>, align: u64) -> (Range<u64>, u64) {
    assert!(align > 0);
    let delta = align - 1;
    if range.start >= delta {
        // Snap range start to previous multiple of align
        let old_start = range.start;
        let new_start = (range.start - delta).next_multiple_of(align);
        let offset = old_start - new_start;
        (new_start..range.end, offset)
    } else {
        // Snap range start to 0
        (0..range.end, range.start)
    }
}

/// Index of a row in a [`BufferTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferTableId(pub(crate) u32); // TEMP: pub(crate)

impl BufferTableId {
    /// An invalid value, often used as placeholder.
    pub const INVALID: BufferTableId = BufferTableId(u32::MAX);

    /// Check if the current ID is valid, that is, is different from
    /// [`INVALID`].
    ///
    /// [`INVALID`]: Self::INVALID
    #[inline]
    pub fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }

    /// Compute a new buffer table ID by offseting an existing one by `count`
    /// rows.
    #[inline]
    #[allow(dead_code)]
    pub fn offset(&self, count: u32) -> BufferTableId {
        debug_assert!(self.is_valid());
        BufferTableId(self.0 + count)
    }
}

impl Default for BufferTableId {
    fn default() -> Self {
        Self::INVALID
    }
}

#[derive(Debug)]
struct AllocatedBuffer {
    /// Currently allocated buffer, of size equal to `size`.
    buffer: Buffer,
    /// Size of the currently allocated buffer, in number of rows.
    count: u32,
    /// Previously allocated buffer if any, cached until the next buffer write
    /// so that old data can be copied into the newly-allocated buffer.
    old_buffer: Option<Buffer>,
    /// Size of the old buffer if any, in number of rows.
    old_count: u32,
}

impl AllocatedBuffer {
    /// Get the number of rows of the currently allocated GPU buffer.
    ///
    /// On capacity grow, the count is valid until the next buffer swap.
    pub fn allocated_count(&self) -> u32 {
        if self.old_buffer.is_some() {
            self.old_count
        } else {
            self.count
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
///   previous frame. Then insert new rows with [`insert()`] and if you made
///   changes call [`allocate_gpu()`] at the end to allocate any new buffer
///   needed.
/// - During the [`RenderStage::Render`] stage, call [`write_buffer()`] from a
///   command encoder before using any row, to perform any buffer resize copy
///   pending.
///
/// [`BufferVec`]: bevy::render::render_resource::BufferVec
/// [`AlignedBufferVec`]: crate::render::aligned_buffer_vec::AlignedBufferVec
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
    ///
    /// This is the expected capacity, as requested by CPU side allocations and
    /// deallocations. The GPU buffer might not have been resized yet to handle
    /// it, so might be allocated with a different size.
    capacity: u32,
    /// Size of the "active" portion of the table, which includes allocated rows
    /// and any row in the free list. All other rows in the
    /// `active_size..capacity` range are implicitly unallocated.
    active_count: u32,
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
        let item_size = std::mem::size_of::<T>();
        let aligned_size = <T as ShaderSize>::SHADER_SIZE.get() as usize;
        assert!(aligned_size >= item_size);
        Self {
            buffer: None,
            buffer_usage: BufferUsages::all(),
            label: None,
            item_size,
            aligned_size,
            capacity: 0,
            active_count: 0,
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
    /// can pass `None` to ignore). This is useful if for example you want to
    /// bind individual rows or any subset of the table, to ensure each row is
    /// aligned to the device constraints.
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
            let aligned_size = item_size.next_multiple_of(item_align);
            assert!(aligned_size >= item_size);
            assert!(aligned_size.is_multiple_of(item_align));
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

    /// Get a safe buffer label for debug display.
    ///
    /// Falls back to an empty string if no label was specified.
    pub fn safe_label(&self) -> Cow<'_, str> {
        self.label
            .as_ref()
            .map(|s| Cow::Borrowed(&s[..]))
            .unwrap_or(Cow::Borrowed(""))
    }

    /// Get a safe buffer name for debug display.
    ///
    /// Same as [`safe_label()`] but includes the buffer ID as well.
    ///
    /// [`safe_label()`]: self::BufferTable::safe_label
    pub fn safe_name(&self) -> String {
        let id = self
            .buffer
            .as_ref()
            .map(|ab| {
                let id: NonZeroU32 = ab.buffer.id().into();
                id.get()
            })
            .unwrap_or(0);
        format!("#{}:{}", id, self.safe_label())
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
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Current number of rows in use in the table.
    ///
    /// Note that rows in use are not necessarily contiguous. There may be gaps
    /// between used rows.
    #[inline]
    #[allow(dead_code)]
    pub fn len(&self) -> u32 {
        self.active_count - self.free_indices.len() as u32
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
        self.active_count == 0
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
        self.active_count = 0;
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
            ab.old_count = 0;
        }
    }

    /// Calculate the size in byte of `count` rows.
    #[inline]
    fn to_byte_size(&self, count: u32) -> usize {
        count as usize * self.aligned_size
    }

    /// Insert a new row into the table.
    ///
    /// For performance reasons, this buffers the row content on the CPU until
    /// the next GPU update, to minimize the number of CPU to GPU transfers.
    pub fn insert(&mut self, value: T) -> BufferTableId {
        trace!(
            "Inserting into table buffer '{}' with {} free indices, capacity: {}, active_size: {}",
            self.safe_name(),
            self.free_indices.len(),
            self.capacity,
            self.active_count
        );
        let index = if self.free_indices.is_empty() {
            let index = self.active_count;
            if index == self.capacity {
                self.capacity += 1;
            }
            debug_assert!(index < self.capacity);
            self.active_count += 1;
            index
        } else {
            self.free_indices.pop().unwrap()
        };
        let allocated_count = self
            .buffer
            .as_ref()
            .map(|ab| ab.allocated_count())
            .unwrap_or(0);
        trace!(
            "Found free index {}, capacity: {}, active_count: {}, allocated_count: {}",
            index,
            self.capacity,
            self.active_count,
            allocated_count
        );
        if index < allocated_count {
            self.pending_values.push((index, value));
        } else {
            let extra_index = index - allocated_count;
            if extra_index < self.extra_pending_values.len() as u32 {
                self.extra_pending_values[extra_index as usize] = value;
            } else {
                self.extra_pending_values.push(value);
            }
        }
        BufferTableId(index)
    }

    /// Calculate a dynamic byte offset for a bind group from a table entry.
    ///
    /// This returns the product of `id` by the internal [`aligned_size()`].
    ///
    /// # Panic
    ///
    /// Panics if the `index` is too large, producing a byte offset larger than
    /// `u32::MAX`.
    ///
    /// [`aligned_size()`]: Self::aligned_size
    #[inline]
    pub fn dynamic_offset(&self, id: BufferTableId) -> u32 {
        let offset = self.aligned_size * id.0 as usize;
        assert!(offset <= u32::MAX as usize);
        u32::try_from(offset).expect("BufferTable index out of bounds")
    }

    /// Update an existing row in the table.
    ///
    /// For performance reasons, this buffers the row content on the CPU until
    /// the next GPU update, to minimize the number of CPU to GPU transfers.
    ///
    /// Calling this function multiple times overwrites the previous value. Only
    /// the last value recorded each frame will be uploaded to GPU.
    ///
    /// # Panics
    ///
    /// Panics if the `id` is invalid.
    pub fn update(&mut self, id: BufferTableId, value: T) {
        assert!(id.is_valid());
        trace!(
            "Updating row {} of table buffer '{}'",
            id.0,
            self.safe_name(),
        );
        let allocated_count = self
            .buffer
            .as_ref()
            .map(|ab| ab.allocated_count())
            .unwrap_or(0);
        if id.0 < allocated_count {
            if let Some(idx) = self
                .pending_values
                .iter()
                .position(|&(index, _)| index == id.0)
            {
                // Overwrite a previous update. This ensures we never upload more than one
                // update per row, which would waste GPU bandwidth.
                self.pending_values[idx] = (id.0, value);
            } else {
                self.pending_values.push((id.0, value));
            }
        } else {
            let extra_index = (id.0 - allocated_count) as usize;
            assert!(extra_index < self.extra_pending_values.len());
            // Overwrite a previous update. This ensures we never upload more than one
            // update per row, which would waste GPU bandwidth.
            self.extra_pending_values[extra_index] = value;
        }
    }

    /// Insert several new contiguous rows into the table.
    ///
    /// For performance reasons, this buffers the row content on the CPU until
    /// the next GPU update, to minimize the number of CPU to GPU transfers.
    ///
    /// # Returns
    ///
    /// Returns the index of the first entry. Other entries follow right after
    /// it.
    #[allow(dead_code)] // unused but annoying to write, so keep if we need in the future
    pub fn insert_contiguous(&mut self, values: impl ExactSizeIterator<Item = T>) -> BufferTableId {
        let count = values.len() as u32;
        trace!(
            "Inserting {} contiguous values into table buffer '{}' with {} free indices, capacity: {}, active_size: {}",
            count,
            self.safe_name(),
            self.free_indices.len(),
            self.capacity,
            self.active_count
        );
        let first_index = if self.free_indices.is_empty() {
            let index = self.active_count;
            if index == self.capacity {
                self.capacity += count;
            }
            debug_assert!(index < self.capacity);
            self.active_count += count;
            index
        } else {
            let mut s = 0;
            let mut n = 1;
            let mut i = 1;
            while i < self.free_indices.len() {
                debug_assert!(self.free_indices[i] > self.free_indices[i - 1]); // always sorted
                if self.free_indices[i] == self.free_indices[i - 1] + 1 {
                    // contiguous
                    n += 1;
                    if n == count {
                        break;
                    }
                } else {
                    // non-contiguous; restart a new sequence
                    debug_assert!(n < count);
                    s = i;
                }
                i += 1;
            }
            if n == count {
                // Found a range of 'count' consecutive entries. Consume it.
                let index = self.free_indices[s];
                self.free_indices.splice(s..=i, []);
                index
            } else {
                // No free range for 'count' consecutive entries. Allocate at end instead.
                let index = self.active_count;
                if index == self.capacity {
                    self.capacity += count;
                }
                debug_assert!(index < self.capacity);
                self.active_count += count;
                index
            }
        };
        let allocated_count = self
            .buffer
            .as_ref()
            .map(|ab| ab.allocated_count())
            .unwrap_or(0);
        trace!(
            "Found {} free indices {}..{}, capacity: {}, active_count: {}, allocated_count: {}",
            count,
            first_index,
            first_index + count,
            self.capacity,
            self.active_count,
            allocated_count
        );
        for (i, value) in values.enumerate() {
            let index = first_index + i as u32;
            if index < allocated_count {
                self.pending_values.push((index, value));
            } else {
                let extra_index = index - allocated_count;
                if extra_index < self.extra_pending_values.len() as u32 {
                    self.extra_pending_values[extra_index as usize] = value;
                } else {
                    self.extra_pending_values.push(value);
                }
            }
        }
        BufferTableId(first_index)
    }

    /// Remove a row from the table.
    #[allow(dead_code)]
    pub fn remove(&mut self, id: BufferTableId) {
        let index = id.0;
        assert!(index < self.active_count);

        // If this is the last item in the active zone, just shrink the active zone
        // (implicit free list).
        if index == self.active_count - 1 {
            self.active_count -= 1;
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

    /// Remove a range of rows from the table.
    #[allow(dead_code)]
    pub fn remove_range(&mut self, first: BufferTableId, count: u32) {
        let index = first.0;
        assert!(index + count <= self.active_count);

        // If this is the last item in the active zone, just shrink the active zone
        // (implicit free list).
        if index == self.active_count - count {
            self.active_count -= count;
            self.capacity -= count;

            // Also try to remove free indices
            if self.free_indices.len() as u32 == self.active_count {
                // Easy case: everything is free, clear everything
                self.free_indices.clear();
                self.active_count = 0;
                self.capacity = 0;
            } else {
                // Some rows are still allocated. Dequeue from end while we have a contiguous
                // tail of free indices.
                let mut num_popped = 0;
                while let Some(idx) = self.free_indices.pop() {
                    if idx < self.active_count - 1 - num_popped {
                        self.free_indices.push(idx);
                        break;
                    }
                    num_popped += 1;
                }
                self.active_count -= num_popped;
                self.capacity -= num_popped;
            }
        } else {
            // This is very inefficient but we need to apply the same logic as the
            // EffectCache because we rely on indices being in sync.
            let pos = self
                .free_indices
                .binary_search(&index) // will fail
                .unwrap_or_else(|e| e); // will get position of insertion
            self.free_indices.splice(pos..pos, index..(index + count));
        }

        debug_assert!(
            (self.free_indices.is_empty() && self.active_count == 0)
                || (self.free_indices.len() as u32) < self.active_count
        );
    }

    /// Allocate any GPU buffer if needed, based on the most recent capacity
    /// requested.
    ///
    /// This should be called only once per frame after all allocation requests
    /// have been made via [`insert()`] but before the GPU buffer is actually
    /// updated. This is an optimization to enable allocating the GPU buffer
    /// earlier than it's actually needed. Calling this multiple times is not
    /// supported, and might assert. Not calling it is safe, as the next
    /// update will call it just-in-time anyway.
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
        let allocated_count = self.buffer.as_ref().map(|ab| ab.count).unwrap_or(0);
        let reallocated = if self.capacity > allocated_count {
            let byte_size = self.to_byte_size(self.capacity);
            trace!(
                "reserve('{}'): increase capacity from {} to {} elements, old size {} bytes, new size {} bytes",
                self.safe_name(),
                allocated_count,
                self.capacity,
                self.to_byte_size(allocated_count),
                byte_size
            );

            // Create the new buffer, swapping with the old one if any
            let has_init_data = !self.extra_pending_values.is_empty();
            let new_buffer = device.create_buffer(&BufferDescriptor {
                label: self.label.as_ref().map(|s| &s[..]),
                size: byte_size as BufferAddress,
                usage: self.buffer_usage,
                mapped_at_creation: has_init_data,
            });

            // Use any pending data to initialize the buffer. We only use CPU-available
            // data, which was inserted after the buffer was (re-)allocated and
            // has not been uploaded to GPU yet.
            if has_init_data {
                // Leave some space to copy the old buffer if any
                let base_size = self.to_byte_size(allocated_count) as u64;
                let extra_count = self.extra_pending_values.len() as u32;
                let extra_size = self.to_byte_size(extra_count) as u64;

                // Scope get_mapped_range_mut() to force a drop before unmap()
                {
                    // Note: get_mapped_range_mut() requires 8-byte alignment of the start offset.
                    let unaligned_range = base_size..(base_size + extra_size);
                    let (range, byte_offset) = round_range_start_down(unaligned_range, 8);

                    let dst_slice = &mut new_buffer.slice(range).get_mapped_range_mut();

                    let base_offset = byte_offset as usize;
                    let byte_size = self.aligned_size; // single row
                    for (index, content) in self.extra_pending_values.drain(..).enumerate() {
                        let byte_offset = base_offset + byte_size * index;

                        // Copy Rust value into a GPU-ready format, including GPU padding.
                        let src: &[u8] = cast_slice(std::slice::from_ref(&content));
                        let dst_range = byte_offset..(byte_offset + self.item_size);
                        trace!(
                            "+ init_copy: index={} src={:?} dst={:?} byte_offset={} byte_size={}",
                            index,
                            src.as_ptr(),
                            dst_range,
                            byte_offset,
                            byte_size
                        );
                        let dst = &mut dst_slice[dst_range];
                        dst.copy_from_slice(src);
                    }
                }

                new_buffer.unmap();
            }

            if let Some(ab) = self.buffer.as_mut() {
                // If there's any data currently in the GPU buffer, we need to copy it on next
                // update to preserve it.
                if self.active_count > 0 {
                    // Current buffer has value to preserve, save it into old_buffer before
                    // replacing it with the newly-allocated one.

                    // By design we can't have all active entries as free ones; we should have
                    // updated active_count=0 and cleared the free list if that was the case.
                    debug_assert!(self.free_indices.len() < self.active_count as usize);

                    // If we already have an old buffer, that means we already have scheduled a copy
                    // to preserve some values. And we can't do that twice per frame.
                    assert!(
                        ab.old_buffer.is_none(),
                        "allocate_gpu() called twice before write_buffer() took effect."
                    );

                    // Swap old <-> new
                    let mut old_buffer = new_buffer;
                    std::mem::swap(&mut old_buffer, &mut ab.buffer);
                    ab.old_buffer = Some(old_buffer);
                    ab.old_count = ab.count;
                } else {
                    // Current buffer is unused, so we don't need to preserve anything.

                    // It could happen we reallocate during the frame then immediately free the rows
                    // to preserve, such that we don't need in the end to preserve anything.
                    if let Some(old_buffer) = ab.old_buffer.take() {
                        old_buffer.destroy();
                    }

                    ab.buffer.destroy();
                    ab.buffer = new_buffer;
                }
                ab.count = self.capacity;
            } else {
                self.buffer = Some(AllocatedBuffer {
                    buffer: new_buffer,
                    count: self.capacity,
                    old_buffer: None,
                    old_count: 0,
                });
            }

            true
        } else {
            false
        };

        // Immediately schedule a copy of old rows.
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
                    "+ old_copy: index={} src={:?} dst={:?} byte_offset={} byte_size={}",
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
            trace!("write_buffer({}): nothing to do", self.safe_name());
            return;
        }

        trace!(
            "write_buffer({}): pending_values.len={} item_size={} aligned_size={} buffer={:?}",
            self.safe_name(),
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
            let old_size = self.to_byte_size(ab.old_count) as u64;
            trace!("Copy old buffer id {:?} of size {} bytes into newly-allocated buffer {:?} of size {} bytes.", old_buffer.id(), old_size, ab.buffer.id(), self.to_byte_size(ab.count));
            encoder.copy_buffer_to_buffer(old_buffer, 0, &ab.buffer, 0, old_size);
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::Vec3;
    use bytemuck::{Pod, Zeroable};

    use super::*;

    #[test]
    fn test_round_range_start_down() {
        // r8(0..7) : no-op
        {
            let (r, o) = round_range_start_down(0..7, 8);
            assert_eq!(r, 0..7);
            assert_eq!(o, 0);
        }

        // r8(2..7) = 0..7, +2
        {
            let (r, o) = round_range_start_down(2..7, 8);
            assert_eq!(r, 0..7);
            assert_eq!(o, 2);
        }

        // r8(7..32) = 0..32, +7
        {
            let (r, o) = round_range_start_down(7..32, 8);
            assert_eq!(r, 0..32);
            assert_eq!(o, 7);
        }

        // r8(8..32) = no-op
        {
            let (r, o) = round_range_start_down(8..32, 8);
            assert_eq!(r, 8..32);
            assert_eq!(o, 0);
        }
    }

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
    fn buffer_table_sizes() {
        // Rust
        assert_eq!(std::mem::size_of::<GpuDummy>(), 12);
        assert_eq!(std::mem::align_of::<GpuDummy>(), 4);
        assert_eq!(std::mem::size_of::<GpuDummyComposed>(), 16); // tight packing
        assert_eq!(std::mem::align_of::<GpuDummyComposed>(), 4);
        assert_eq!(std::mem::size_of::<GpuDummyLarge>(), 132 * 4); // tight packing
        assert_eq!(std::mem::align_of::<GpuDummyLarge>(), 4);

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

    #[test]
    fn buffer_table_insert() {
        let mut table =
            BufferTable::<GpuDummy>::new(BufferUsages::STORAGE, NonZeroU64::new(32), None);

        // [x]
        let id1 = table.insert(GpuDummy::default());
        assert_eq!(id1.0, 0);
        assert_eq!(table.active_count, 1);
        assert!(table.free_indices.is_empty());

        // [x x]
        let id2 = table.insert(GpuDummy::default());
        assert_eq!(id2.0, 1);
        assert_eq!(table.active_count, 2);
        assert!(table.free_indices.is_empty());

        // [- x]
        table.remove(id1);
        assert_eq!(table.active_count, 2);
        assert_eq!(table.free_indices.len(), 1);
        assert_eq!(table.free_indices[0], 0);

        // [- x x x]
        let id3 = table.insert_contiguous([GpuDummy::default(); 2].into_iter());
        assert_eq!(id3.0, 2); // at the end (doesn't fit in free slot #0)
        assert_eq!(table.active_count, 4);
        assert_eq!(table.free_indices.len(), 1);
        assert_eq!(table.free_indices[0], 0);

        // [- - x x]
        table.remove(id2);
        assert_eq!(table.active_count, 4);
        assert_eq!(table.free_indices.len(), 2);
        assert_eq!(table.free_indices[0], 0);
        assert_eq!(table.free_indices[1], 1);

        // [x x x x]
        let id4 = table.insert_contiguous([GpuDummy::default(); 2].into_iter());
        assert_eq!(id4.0, 0); // this times it fit into slot #0-#1
        assert_eq!(table.active_count, 4);
        assert!(table.free_indices.is_empty());

        // [- - x x]
        table.remove_range(id4, 2);
        assert_eq!(table.active_count, 4);
        assert_eq!(table.free_indices.len(), 2);
        assert_eq!(table.free_indices[0], 0);
        assert_eq!(table.free_indices[1], 1);

        // []
        table.remove_range(id3, 2);
        assert_eq!(table.active_count, 0);
        assert!(table.free_indices.is_empty());

        // [x x]
        let id5 = table.insert_contiguous([GpuDummy::default(); 2].into_iter());
        assert_eq!(id5.0, 0);
        assert_eq!(table.active_count, 2);
        assert!(table.free_indices.is_empty());

        // [x x x x]
        let id6 = table.insert_contiguous([GpuDummy::default(); 2].into_iter());
        assert_eq!(id6.0, 2);
        assert_eq!(table.active_count, 4);
        assert!(table.free_indices.is_empty());

        // [x x x x x x]
        let id7 = table.insert_contiguous([GpuDummy::default(); 2].into_iter());
        assert_eq!(id7.0, 4);
        assert_eq!(table.active_count, 6);
        assert!(table.free_indices.is_empty());

        // [x x - - x x]
        table.remove_range(id6, 2);
        assert_eq!(table.active_count, 6);
        assert_eq!(table.free_indices.len(), 2);
        assert_eq!(table.free_indices[0], 2);
        assert_eq!(table.free_indices[1], 3);

        // [x x]
        table.remove_range(id7, 2);
        assert_eq!(table.active_count, 2);
        assert!(table.free_indices.is_empty());
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
    fn read_back_gpu(device: &RenderDevice, slice: BufferSlice<'_>) -> BufferView {
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
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
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

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
        for i in 0..len {
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
        assert_eq!(ab.count, len);
        println!(
            "Allocated buffer #{:?} of {} rows",
            ab.buffer.id(),
            ab.count
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
        assert_eq!(ab.count, len);
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
                assert_eq!(view.len(), final_align as usize * table.capacity() as usize);
                for i in 0..len as usize {
                    let offset = i * final_align as usize;
                    let item_size = std::mem::size_of::<GpuDummyComposed>();
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
        let mut len = len;
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

        // This re-allocates a new GPU buffer because the capacity changed
        table.allocate_gpu(&device, &queue);
        assert!(!table.is_empty());
        assert_eq!(table.len(), len);
        assert!(table.capacity() >= len);
        let ab = table
            .buffer
            .as_ref()
            .expect("GPU buffer should be allocated after allocate_gpu()");
        assert_eq!(ab.count, len);
        assert!(ab.old_buffer.is_some()); // old buffer to copy
        assert_ne!(ab.old_buffer.as_ref().unwrap().id(), ab.buffer.id());
        println!(
            "Allocated new buffer #{:?} of {} rows",
            ab.buffer.id(),
            ab.count
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
                assert_eq!(view.len(), final_align as usize * table.capacity() as usize);
                for i in 0..len as usize {
                    let offset = i * final_align as usize;
                    let item_size = std::mem::size_of::<GpuDummyComposed>();
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
        table.remove(BufferTableId(len));
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
        assert_eq!(ab.count, len + 1); // GPU buffer kept its size
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
                assert!(view.len() >= final_align as usize * table.capacity() as usize); // note the >=, the buffer is over-allocated
                for i in 0..len as usize {
                    let offset = i * final_align as usize;
                    let item_size = std::mem::size_of::<GpuDummyComposed>();
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
        let mut len = len - 1;
        table.remove(BufferTableId(0));
        assert_eq!(old_capacity, table.capacity());
        println!(
            "Removed first row to shrink capacity from {} to {} (no change).",
            old_capacity,
            table.capacity()
        );

        // This doesn't do anything since we only removed a row
        table.allocate_gpu(&device, &queue);
        assert!(!table.is_empty());
        assert_eq!(table.len(), len);
        assert!(table.capacity() >= len);
        let ab = table
            .buffer
            .as_ref()
            .expect("GPU buffer should be allocated after allocate_gpu()");
        assert_eq!(ab.count, len + 2); // GPU buffer kept its size
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
                assert!(view.len() >= final_align as usize * table.capacity() as usize); // note the >=, the buffer is over-allocated
                for i in 0..len as usize {
                    let offset = i * final_align as usize;
                    let item_size = std::mem::size_of::<GpuDummyComposed>();
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

        // This doesn't reallocate the GPU buffer since we used a free list entry
        table.allocate_gpu(&device, &queue);
        assert!(!table.is_empty());
        assert_eq!(table.len(), len);
        assert!(table.capacity() >= len);
        let ab = table
            .buffer
            .as_ref()
            .expect("GPU buffer should be allocated after allocate_gpu()");
        assert_eq!(ab.count, 4); // 4 == last time we grew
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
                assert!(view.len() >= final_align as usize * table.capacity() as usize);
                for i in 0..len as usize {
                    let offset = i * final_align as usize;
                    let item_size = std::mem::size_of::<GpuDummyComposed>();
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

        // This doesn't reallocate the GPU buffer since we used an implicit free entry
        table.allocate_gpu(&device, &queue);
        assert!(!table.is_empty());
        assert_eq!(table.len(), len);
        assert!(table.capacity() >= len);
        let ab = table
            .buffer
            .as_ref()
            .expect("GPU buffer should be allocated after allocate_gpu()");
        assert_eq!(ab.count, 4); // 4 == last time we grew
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
                assert!(view.len() >= final_align as usize * table.capacity() as usize);
                for i in 0..len as usize {
                    let offset = i * final_align as usize;
                    let item_size = std::mem::size_of::<GpuDummyComposed>();
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
