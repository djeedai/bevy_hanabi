use std::{num::NonZeroU64, ops::Range};

use bevy::{
    log::trace,
    render::{
        render_resource::{
            BindingResource, Buffer, BufferAddress, BufferBinding, BufferDescriptor, BufferUsages,
            ShaderSize, ShaderType,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::{cast_slice, Pod};

/// Like Bevy's [`BufferVec`], but with extra per-item alignment.
///
/// This helper ensures the individual array elements are properly aligned,
/// depending on the device constraints and the WGSL rules. In general using
/// [`BufferVec`] is enough to ensure alignment; however when some array items
/// also need to be bound individually, then each item (not only the array
/// itself) needs to be aligned to the device requirements. This is admittedly a
/// very specific case, because the device alignment might be very large (256
/// bytes) and this causes a lot of wasted space (padding per-element, instead
/// of padding for the entire array).
///
/// For this buffer to work correctly and items be bindable individually, the
/// alignment must come from one of the [`WgpuLimits`]. For example for a
/// storage buffer, to be able to bind the entire buffer but also any subset of
/// it (including individual elements), the extra alignment must
/// be [`WgpuLimits::min_storage_buffer_offset_alignment`].
///
/// The element type `T` needs to implement the following traits:
/// - [`Pod`] to allow copy.
/// - [`ShaderType`] because it needs to be mapped for a shader.
/// - [`ShaderSize`] to ensure a fixed footprint, to allow packing multiple
///   instances inside a single buffer. This therefore excludes any
///   runtime-sized array.
///
/// [`BufferVec`]: bevy::render::render_resource::BufferVec
/// [`WgpuLimits`]: bevy::render::settings::WgpuLimits
pub struct AlignedBufferVec<T: Pod + ShaderSize> {
    /// Pending values accumulated on CPU and not yet written to GPU.
    values: Vec<T>,
    /// GPU buffer if already allocated, or `None` otherwise.
    buffer: Option<Buffer>,
    /// Capacity of the buffer, in number of elements.
    capacity: usize,
    /// Size of a single buffer element, in bytes, in CPU memory (Rust layout).
    item_size: usize,
    /// Size of a single buffer element, in bytes, aligned to GPU memory
    /// constraints.
    aligned_size: usize,
    /// GPU buffer usages.
    buffer_usage: BufferUsages,
    /// Optional GPU buffer name, for debugging.
    label: Option<String>,
}

impl<T: Pod + ShaderSize> Default for AlignedBufferVec<T> {
    fn default() -> Self {
        let item_size = std::mem::size_of::<T>();
        let aligned_size = <T as ShaderSize>::SHADER_SIZE.get() as usize;
        assert!(aligned_size >= item_size);
        Self {
            values: Vec::new(),
            buffer: None,
            capacity: 0,
            buffer_usage: BufferUsages::all(),
            item_size,
            aligned_size,
            label: None,
        }
    }
}

impl<T: Pod + ShaderSize> AlignedBufferVec<T> {
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
            let aligned_size = item_size.next_multiple_of(item_align);
            assert!(aligned_size >= item_size);
            assert!(aligned_size.is_multiple_of(item_align));
            aligned_size
        } else {
            item_size
        };
        trace!(
            "AlignedBufferVec['{}']: item_size={} aligned_size={}",
            label.as_ref().map(|s| &s[..]).unwrap_or(""),
            item_size,
            aligned_size
        );
        if buffer_usage.contains(BufferUsages::UNIFORM) {
            <T as ShaderType>::assert_uniform_compat();
        }
        Self {
            buffer_usage,
            aligned_size,
            label,
            ..Default::default()
        }
    }

    fn safe_label(&self) -> &str {
        self.label.as_ref().map(|s| &s[..]).unwrap_or("")
    }

    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Get a binding for the entire buffer.
    #[inline]
    #[allow(dead_code)]
    pub fn binding(&self) -> Option<BindingResource<'_>> {
        // FIXME - Return a Buffer wrapper first, which can be unwrapped, then from that
        // wrapper implement all the xxx_binding() helpers. That avoids a bunch of "if
        // let Some()" everywhere when we know the buffer is valid. The only reason the
        // buffer might not be valid is if it was not created, and in that case
        // we wouldn't be calling the xxx_bindings() helpers, we'd have earlied out
        // before.
        let buffer = self.buffer()?;
        Some(BindingResource::Buffer(BufferBinding {
            buffer,
            offset: 0,
            size: None, // entire buffer
        }))
    }

    /// Get a binding for a subset of the elements of the buffer.
    ///
    /// Returns a binding for the elements in the range `offset..offset+count`.
    ///
    /// # Panics
    ///
    /// Panics if `count` is zero.
    #[inline]
    #[allow(dead_code)]
    pub fn range_binding(&self, offset: u32, count: u32) -> Option<BindingResource<'_>> {
        assert!(count > 0);
        let buffer = self.buffer()?;
        let offset = self.aligned_size as u64 * offset as u64;
        let size = NonZeroU64::new(self.aligned_size as u64 * count as u64).unwrap();
        Some(BindingResource::Buffer(BufferBinding {
            buffer,
            offset,
            size: Some(size),
        }))
    }

    #[inline]
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Size in bytes of a single item in the buffer, aligned to the item
    /// alignment.
    #[inline]
    pub fn aligned_size(&self) -> usize {
        self.aligned_size
    }

    /// Calculate a dynamic byte offset for a bind group from an array element
    /// index.
    ///
    /// This returns the product of `index` by the internal [`aligned_size()`].
    ///
    /// # Panic
    ///
    /// Panics if the `index` is too large, producing a byte offset larger than
    /// `u32::MAX`.
    ///
    /// [`aligned_size()`]: crate::AlignedBufferVec::aligned_size
    #[inline]
    pub fn dynamic_offset(&self, index: usize) -> u32 {
        let offset = self.aligned_size * index;
        assert!(offset <= u32::MAX as usize);
        u32::try_from(offset).expect("AlignedBufferVec index out of bounds")
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Append a value to the buffer.
    ///
    /// The content is stored on the CPU and uploaded on the GPU once
    /// [`write_buffer()`] is called.
    ///
    /// [`write_buffer()`]: crate::AlignedBufferVec::write_buffer
    pub fn push(&mut self, value: T) -> usize {
        let index = self.values.len();
        self.values.push(value);
        index
    }

    /// Reserve some capacity into the buffer.
    ///
    /// If the buffer is reallocated, the old content (on the GPU) is lost, and
    /// needs to be re-uploaded to the newly-created buffer. This is done with
    /// [`write_buffer()`].
    ///
    /// # Returns
    ///
    /// `true` if the buffer was (re)allocated, or `false` if an existing buffer
    /// was reused which already had enough capacity.
    ///
    /// [`write_buffer()`]: crate::AlignedBufferVec::write_buffer
    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) -> bool {
        if capacity > self.capacity {
            let size = self.aligned_size * capacity;
            trace!(
                "reserve['{}']: increase capacity from {} to {} elements, new size {} bytes",
                self.safe_label(),
                self.capacity,
                capacity,
                size
            );
            self.capacity = capacity;
            if let Some(old_buffer) = self.buffer.take() {
                trace!(
                    "reserve['{}']: destroying old buffer #{:?}",
                    self.safe_label(),
                    old_buffer.id()
                );
                old_buffer.destroy();
            }
            let new_buffer = device.create_buffer(&BufferDescriptor {
                label: self.label.as_ref().map(|s| &s[..]),
                size: size as BufferAddress,
                usage: BufferUsages::COPY_DST | self.buffer_usage,
                mapped_at_creation: false,
            });
            trace!(
                "reserve['{}']: created new buffer #{:?}",
                self.safe_label(),
                new_buffer.id(),
            );
            self.buffer = Some(new_buffer);
            // FIXME - this discards the old content if any!!!
            true
        } else {
            false
        }
    }

    /// Schedule the buffer write to GPU.
    ///
    /// # Returns
    ///
    /// `true` if the buffer was (re)allocated, `false` otherwise.
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) -> bool {
        if self.values.is_empty() {
            return false;
        }
        trace!(
            "write_buffer['{}']: values.len={} item_size={} aligned_size={}",
            self.safe_label(),
            self.values.len(),
            self.item_size,
            self.aligned_size
        );
        let buffer_changed = self.reserve(self.values.len(), device);
        if let Some(buffer) = &self.buffer {
            let aligned_size = self.aligned_size * self.values.len();
            trace!(
                "aligned_buffer['{}']: size={} buffer={:?}",
                self.safe_label(),
                aligned_size,
                buffer.id(),
            );
            let mut aligned_buffer: Vec<u8> = vec![0; aligned_size];
            for i in 0..self.values.len() {
                let src: &[u8] = cast_slice(std::slice::from_ref(&self.values[i]));
                let dst_offset = i * self.aligned_size;
                let dst_range = dst_offset..dst_offset + self.item_size;
                trace!("+ copy: src={:?} dst={:?}", src.as_ptr(), dst_range);
                let dst = &mut aligned_buffer[dst_range];
                dst.copy_from_slice(src);
            }
            let bytes: &[u8] = cast_slice(&aligned_buffer);
            queue.write_buffer(buffer, 0, bytes);
        }
        buffer_changed
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }
}

impl<T: Pod + ShaderSize> std::ops::Index<usize> for AlignedBufferVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<T: Pod + ShaderSize> std::ops::IndexMut<usize> for AlignedBufferVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FreeRow(pub Range<u32>);

impl PartialOrd for FreeRow {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FreeRow {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.start.cmp(&other.0.start)
    }
}

/// Like [`AlignedBufferVec`], but for heterogenous data.
#[derive(Debug)]
pub struct HybridAlignedBufferVec {
    /// Pending values accumulated on CPU and not yet written to GPU.
    values: Vec<u8>,
    /// GPU buffer if already allocated, or `None` otherwise.
    buffer: Option<Buffer>,
    /// Capacity of the buffer, in bytes.
    capacity: usize,
    /// Alignment of each element, in bytes.
    item_align: usize,
    /// GPU buffer usages.
    buffer_usage: BufferUsages,
    /// Optional GPU buffer name, for debugging.
    label: Option<String>,
    /// Free ranges available for re-allocation. Those are row ranges; byte
    /// ranges are obtained by multiplying these by `item_align`.
    free_rows: Vec<FreeRow>,
    /// Is the GPU buffer stale and the CPU one need to be re-uploaded?
    is_stale: bool,
}

impl HybridAlignedBufferVec {
    /// Create a new collection.
    ///
    /// `item_align` is the alignment for items in the collection.
    pub fn new(buffer_usage: BufferUsages, item_align: NonZeroU64, label: Option<String>) -> Self {
        let item_align = item_align.get() as usize;
        trace!(
            "HybridAlignedBufferVec['{}']: item_align={} byte",
            label.as_ref().map(|s| &s[..]).unwrap_or(""),
            item_align,
        );
        Self {
            values: vec![],
            buffer: None,
            capacity: 0,
            item_align,
            buffer_usage,
            label,
            free_rows: vec![],
            is_stale: true,
        }
    }

    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Get a binding for the entire buffer.
    #[allow(dead_code)]
    #[inline]
    pub fn max_binding(&self) -> Option<BindingResource<'_>> {
        // FIXME - Return a Buffer wrapper first, which can be unwrapped, then from that
        // wrapper implement all the xxx_binding() helpers. That avoids a bunch of "if
        // let Some()" everywhere when we know the buffer is valid. The only reason the
        // buffer might not be valid is if it was not created, and in that case
        // we wouldn't be calling the xxx_bindings() helpers, we'd have earlied out
        // before.
        let buffer = self.buffer()?;
        Some(BindingResource::Buffer(BufferBinding {
            buffer,
            offset: 0,
            size: None, // entire buffer
        }))
    }

    /// Get a binding for the first `size` bytes of the buffer.
    ///
    /// # Panics
    ///
    /// Panics if `size` is zero.
    #[allow(dead_code)]
    #[inline]
    pub fn lead_binding(&self, size: u32) -> Option<BindingResource<'_>> {
        let buffer = self.buffer()?;
        let size = NonZeroU64::new(size as u64).unwrap();
        Some(BindingResource::Buffer(BufferBinding {
            buffer,
            offset: 0,
            size: Some(size),
        }))
    }

    /// Get a binding for a subset of the elements of the buffer.
    ///
    /// Returns a binding for the elements in the range `offset..offset+count`.
    ///
    /// # Panics
    ///
    /// Panics if `offset` is not a multiple of the alignment specified on
    /// construction.
    ///
    /// Panics if `size` is zero.
    #[allow(dead_code)]
    #[inline]
    pub fn range_binding(&self, offset: u32, size: u32) -> Option<BindingResource<'_>> {
        assert!((offset as usize).is_multiple_of(self.item_align));
        let buffer = self.buffer()?;
        let size = NonZeroU64::new(size as u64).unwrap();
        Some(BindingResource::Buffer(BufferBinding {
            buffer,
            offset: offset as u64,
            size: Some(size),
        }))
    }

    /// Capacity of the allocated GPU buffer, in bytes.
    ///
    /// This may be zero if the buffer was not allocated yet. In general, this
    /// can differ from the actual data size cached on CPU and waiting to be
    /// uploaded to GPU.
    #[inline]
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current buffer size, in bytes.
    ///
    /// This represents the size of the CPU data uploaded to GPU. Pending a GPU
    /// buffer re-allocation or re-upload, this size might differ from the
    /// actual GPU buffer size. But they're eventually consistent.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Alignment, in bytes, of all the elements.
    #[allow(dead_code)]
    #[inline]
    pub fn item_align(&self) -> usize {
        self.item_align
    }

    /// Calculate a dynamic byte offset for a bind group from an array element
    /// index.
    ///
    /// This returns the product of `index` by the internal [`item_align()`].
    ///
    /// # Panic
    ///
    /// Panics if the `index` is too large, producing a byte offset larger than
    /// `u32::MAX`.
    ///
    /// [`item_align()`]: crate::HybridAlignedBufferVec::item_align
    #[allow(dead_code)]
    #[inline]
    pub fn dynamic_offset(&self, index: usize) -> u32 {
        let offset = self.item_align * index;
        assert!(offset <= u32::MAX as usize);
        u32::try_from(offset).expect("HybridAlignedBufferVec index out of bounds")
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Append a value to the buffer.
    ///
    /// As with [`set_content()`], the content is stored on the CPU and uploaded
    /// on the GPU once [`write_buffers()`] is called.
    ///
    /// # Returns
    ///
    /// Returns a range starting at the byte offset at which the new element was
    /// inserted, which is guaranteed to be a multiple of [`item_align()`].
    /// The range span is the item byte size.
    ///
    /// [`item_align()`]: self::HybridAlignedBufferVec::item_align
    #[allow(dead_code)]
    pub fn push<T: Pod + ShaderSize>(&mut self, value: &T) -> Range<u32> {
        let src: &[u8] = cast_slice(std::slice::from_ref(value));
        assert_eq!(value.size().get() as usize, src.len());
        self.push_raw(src)
    }

    /// Append a slice of values to the buffer.
    ///
    /// The values are assumed to be tightly packed, and will be copied
    /// back-to-back into the buffer, without any padding between them. This
    /// means that the individul slice items must be properly aligned relative
    /// to the beginning of the slice.
    ///
    /// As with [`set_content()`], the content is stored on the CPU and uploaded
    /// on the GPU once [`write_buffers()`] is called.
    ///
    /// # Returns
    ///
    /// Returns a range starting at the byte offset at which the new element
    /// (the slice) was inserted, which is guaranteed to be a multiple of
    /// [`item_align()`]. The range span is the item byte size.
    ///
    /// # Panics
    ///
    /// Panics if the byte size of the element `T` is not at least a multiple of
    /// the minimum GPU alignment, which is 4 bytes. Note that this doesn't
    /// guarantee that the written data is well-formed for use on GPU, as array
    /// elements on GPU have other alignment requirements according to WGSL, but
    /// at least this catches obvious errors.
    ///
    /// [`item_align()`]: self::HybridAlignedBufferVec::item_align
    #[allow(dead_code)]
    pub fn push_many<T: Pod + ShaderSize>(&mut self, value: &[T]) -> Range<u32> {
        assert_eq!(size_of::<T>() % 4, 0);
        let src: &[u8] = cast_slice(value);
        self.push_raw(src)
    }

    pub fn push_raw(&mut self, src: &[u8]) -> Range<u32> {
        self.is_stale = true;

        // Calculate the number of (aligned) rows to allocate
        let num_rows = src.len().div_ceil(self.item_align) as u32;

        // Try to find a block of free rows which can accomodate it, and pick the
        // smallest one in order to limit wasted space.
        let mut best_slot: Option<(u32, usize)> = None;
        for (index, range) in self.free_rows.iter().enumerate() {
            let free_rows = range.0.end - range.0.start;
            if free_rows >= num_rows {
                let wasted_rows = free_rows - num_rows;
                // If we found a slot with the exact size, just use it already
                if wasted_rows == 0 {
                    best_slot = Some((0, index));
                    break;
                }
                // Otherwise try to find the smallest oversized slot to reduce wasted space
                if let Some(best_slot) = best_slot.as_mut() {
                    if wasted_rows < best_slot.0 {
                        *best_slot = (wasted_rows, index);
                    }
                } else {
                    best_slot = Some((wasted_rows, index));
                }
            }
        }

        // Insert into existing space
        if let Some((_, index)) = best_slot {
            let row_range = self.free_rows.remove(index);
            let offset = row_range.0.start as usize * self.item_align;
            let free_size = (row_range.0.end - row_range.0.start) as usize * self.item_align;
            let size = src.len();
            assert!(size <= free_size);

            let dst = self.values.as_mut_ptr();
            // SAFETY: dst is guaranteed to point to allocated bytes, which are already
            // initialized from a previous call, and are initialized by overwriting the
            // bytes with those of a POD type.
            #[allow(unsafe_code)]
            unsafe {
                let dst = dst.add(offset);
                dst.copy_from_nonoverlapping(src.as_ptr(), size);
            }

            let start = offset as u32;
            let end = start + size as u32;
            start..end
        }
        // Insert at end of vector, after resizing it
        else {
            // Calculate new aligned insertion offset and new capacity
            let offset = self.values.len().next_multiple_of(self.item_align);
            let size = src.len();
            let new_capacity = offset + size;
            if new_capacity > self.values.capacity() {
                let additional = new_capacity - self.values.len();
                self.values.reserve(additional)
            }

            // Insert padding if needed
            if offset > self.values.len() {
                self.values.resize(offset, 0);
            }

            // Insert serialized value
            // Dealing with safe code via Vec::spare_capacity_mut() is quite difficult
            // without the upcoming (unstable) additions to MaybeUninit to deal with arrays.
            // To prevent having to loop over individual u8, we use direct pointers instead.
            assert!(self.values.capacity() >= offset + size);
            assert_eq!(self.values.len(), offset);
            let dst = self.values.as_mut_ptr();
            // SAFETY: dst is guaranteed to point to allocated (offset+size) bytes, which
            // are written by copying a Pod type, so ensures those values are initialized,
            // and the final size is set to exactly (offset+size).
            #[allow(unsafe_code)]
            unsafe {
                let dst = dst.add(offset);
                dst.copy_from_nonoverlapping(src.as_ptr(), size);
                self.values.set_len(offset + size);
            }

            debug_assert_eq!(offset % self.item_align, 0);
            let start = offset as u32;
            let end = start + size as u32;
            start..end
        }
    }

    /// Remove a range of bytes previously added.
    ///
    /// Remove a range of bytes previously returned by adding one or more
    /// elements with [`push()`] or [`push_many()`].
    ///
    /// # Returns
    ///
    /// Returns `true` if the range was valid and the corresponding data was
    /// removed, or `false` otherwise. In that case, the buffer is not modified.
    ///
    /// [`push()`]: Self::push
    /// [`push_many()`]: Self::push_many
    pub fn remove(&mut self, range: Range<u32>) -> bool {
        // Can only remove entire blocks starting at an aligned size
        let align = self.item_align as u32;
        if !range.start.is_multiple_of(align) {
            return false;
        }

        // Check for out of bounds argument
        let end = self.values.len() as u32;
        if range.start >= end || range.end > end {
            return false;
        }

        // Note: See below, sometimes self.values() has some padding left we couldn't
        // recover earlier beause we didn't know the size of this allocation, but we
        // need to still deallocate the row here.
        if range.end == end || range.end.next_multiple_of(align) == end {
            // If the allocation is at the end of the buffer, shorten the CPU values. This
            // ensures is_empty() eventually returns true.
            let mut new_row_end = range.start.div_ceil(align);

            // Walk the (sorted) free list to also dequeue any range which is now at the end
            // of the buffer
            while let Some(free_row) = self.free_rows.pop() {
                if free_row.0.end == new_row_end {
                    new_row_end = free_row.0.start;
                } else {
                    self.free_rows.push(free_row);
                    break;
                }
            }

            // Note: we can't really recover any padding here because we don't know the
            // exact size of that allocation, only its row-aligned size.
            self.values.truncate((new_row_end * align) as usize);
        } else {
            // Otherwise, save the row into the free list.
            let start = range.start / align;
            let end = range.end.div_ceil(align);
            let free_row = FreeRow(start..end);

            // Insert as sorted
            if self.free_rows.is_empty() {
                // Special case to simplify below, and to avoid binary_search()
                self.free_rows.push(free_row);
            } else if let Err(index) = self.free_rows.binary_search(&free_row) {
                if index >= self.free_rows.len() {
                    // insert at end
                    let prev = self.free_rows.last_mut().unwrap(); // known
                    if prev.0.end == free_row.0.start {
                        // merge with last value
                        prev.0.end = free_row.0.end;
                    } else {
                        // insert last, with gap
                        self.free_rows.push(free_row);
                    }
                } else if index == 0 {
                    // insert at start
                    let next = &mut self.free_rows[0];
                    if free_row.0.end == next.0.start {
                        // merge with next
                        next.0.start = free_row.0.start;
                    } else {
                        // insert first, with gap
                        self.free_rows.insert(0, free_row);
                    }
                } else {
                    // insert between 2 existing elements
                    let prev = &mut self.free_rows[index - 1];
                    if prev.0.end == free_row.0.start {
                        // merge with previous value
                        prev.0.end = free_row.0.end;

                        let prev = self.free_rows[index - 1].clone();
                        let next = &mut self.free_rows[index];
                        if prev.0.end == next.0.start {
                            // also merge prev with next, and remove prev
                            next.0.start = prev.0.start;
                            self.free_rows.remove(index - 1);
                        }
                    } else {
                        let next = &mut self.free_rows[index];
                        if free_row.0.end == next.0.start {
                            // merge with next value
                            next.0.start = free_row.0.start;
                        } else {
                            // insert between 2 values, with gaps on both sides
                            self.free_rows.insert(0, free_row);
                        }
                    }
                }
            } else {
                // The range exists in the free list, this means it's already removed. This is a
                // duplicate; ignore it.
                return false;
            }
        }
        self.is_stale = true;
        true
    }

    /// Update an allocated entry with a new value.
    #[allow(dead_code)]
    #[inline]
    pub fn update<T: Pod + ShaderSize>(&mut self, offset: u32, value: &T) {
        let data: &[u8] = cast_slice(std::slice::from_ref(value));
        assert_eq!(value.size().get() as usize, data.len());
        self.update_raw(offset, data);
    }

    /// Update an allocated entry with new data.
    pub fn update_raw(&mut self, offset: u32, data: &[u8]) {
        // Can only update entire blocks starting at an aligned size
        let align = self.item_align as u32;
        if !offset.is_multiple_of(align) {
            return;
        }

        // Check for out of bounds argument
        let end = self.values.len() as u32;
        let data_end = offset + data.len() as u32;
        if offset >= end || data_end > end {
            return;
        }

        let dst: &mut [u8] = &mut self.values[offset as usize..data_end as usize];
        dst.copy_from_slice(data);

        self.is_stale = true;
    }

    /// Reserve some capacity into the buffer.
    ///
    /// If the buffer is reallocated, the old content (on the GPU) is lost, and
    /// needs to be re-uploaded to the newly-created buffer. This is done with
    /// [`write_buffer()`].
    ///
    /// # Returns
    ///
    /// `true` if the buffer was (re)allocated, or `false` if an existing buffer
    /// was reused which already had enough capacity.
    ///
    /// [`write_buffer()`]: crate::AlignedBufferVec::write_buffer
    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) -> bool {
        if capacity > self.capacity {
            trace!(
                "reserve: increase capacity from {} to {} bytes",
                self.capacity,
                capacity,
            );
            self.capacity = capacity;
            if let Some(buffer) = self.buffer.take() {
                buffer.destroy();
            }
            self.buffer = Some(device.create_buffer(&BufferDescriptor {
                label: self.label.as_ref().map(|s| &s[..]),
                size: capacity as BufferAddress,
                usage: BufferUsages::COPY_DST | self.buffer_usage,
                mapped_at_creation: false,
            }));
            self.is_stale = !self.values.is_empty();
            // FIXME - this discards the old content if any!!!
            true
        } else {
            false
        }
    }

    /// Schedule the buffer write to GPU.
    ///
    /// # Returns
    ///
    /// `true` if the buffer was (re)allocated, `false` otherwise. If the buffer
    /// was reallocated, all bind groups referencing the old buffer should be
    /// destroyed.
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) -> bool {
        if self.values.is_empty() || !self.is_stale {
            return false;
        }
        let size = self.values.len();
        trace!(
            "hybrid abv: write_buffer: size={}B item_align={}B",
            size,
            self.item_align,
        );
        let buffer_changed = self.reserve(size, device);
        if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, self.values.as_slice());
            self.is_stale = false;
        }
        buffer_changed
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        if !self.values.is_empty() {
            self.is_stale = true;
        }
        self.values.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

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
    fn abv_sizes() {
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
            let mut abv = AlignedBufferVec::<GpuDummy>::new(
                BufferUsages::STORAGE,
                NonZeroU64::new(item_align),
                None,
            );
            assert_eq!(abv.aligned_size(), expected_aligned_size);
            assert!(abv.is_empty());
            abv.push(GpuDummy::default());
            assert!(!abv.is_empty());
            assert_eq!(abv.len(), 1);
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
            let mut abv = AlignedBufferVec::<GpuDummyComposed>::new(
                BufferUsages::STORAGE,
                NonZeroU64::new(item_align),
                None,
            );
            assert_eq!(abv.aligned_size(), expected_aligned_size);
            assert!(abv.is_empty());
            abv.push(GpuDummyComposed::default());
            assert!(!abv.is_empty());
            assert_eq!(abv.len(), 1);
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
            let mut abv = AlignedBufferVec::<GpuDummyLarge>::new(
                BufferUsages::STORAGE,
                NonZeroU64::new(item_align),
                None,
            );
            assert_eq!(abv.aligned_size(), expected_aligned_size);
            assert!(abv.is_empty());
            abv.push(GpuDummyLarge {
                simple: Default::default(),
                tag: 0,
                large: [0.; 128],
            });
            assert!(!abv.is_empty());
            assert_eq!(abv.len(), 1);
        }
    }

    #[test]
    fn habv_remove() {
        let mut habv =
            HybridAlignedBufferVec::new(BufferUsages::STORAGE, NonZeroU64::new(32).unwrap(), None);
        assert!(habv.is_empty());
        assert_eq!(habv.item_align, 32);

        // +r -r
        {
            let r = habv.push(&42u32);
            assert_eq!(r, 0..4);
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 4);
            assert!(habv.free_rows.is_empty());

            assert!(habv.remove(r));
            assert!(habv.is_empty());
            assert!(habv.values.is_empty());
            assert!(habv.free_rows.is_empty());
        }

        // +r0 +r1 +r2 -r0 -r0 -r1 -r2
        {
            let r0 = habv.push(&42u32);
            let r1 = habv.push(&84u32);
            let r2 = habv.push(&84u32);
            assert_eq!(r0, 0..4);
            assert_eq!(r1, 32..36);
            assert_eq!(r2, 64..68);
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 68);
            assert!(habv.free_rows.is_empty());

            assert!(habv.remove(r0.clone()));
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 68);
            assert_eq!(habv.free_rows.len(), 1);
            assert_eq!(habv.free_rows[0], FreeRow(0..1));

            // dupe; no-op
            assert!(!habv.remove(r0));

            assert!(habv.remove(r1.clone()));
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 68);
            assert_eq!(habv.free_rows.len(), 1); // merged!
            assert_eq!(habv.free_rows[0], FreeRow(0..2));

            assert!(habv.remove(r2));
            assert!(habv.is_empty());
            assert_eq!(habv.values.len(), 0);
            assert!(habv.free_rows.is_empty());
        }

        // +r0 +r1 +r2 -r1 -r0 -r2
        {
            let r0 = habv.push(&42u32);
            let r1 = habv.push(&84u32);
            let r2 = habv.push(&84u32);
            assert_eq!(r0, 0..4);
            assert_eq!(r1, 32..36);
            assert_eq!(r2, 64..68);
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 68);
            assert!(habv.free_rows.is_empty());

            assert!(habv.remove(r1.clone()));
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 68);
            assert_eq!(habv.free_rows.len(), 1);
            assert_eq!(habv.free_rows[0], FreeRow(1..2));

            assert!(habv.remove(r0.clone()));
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 68);
            assert_eq!(habv.free_rows.len(), 1); // merged!
            assert_eq!(habv.free_rows[0], FreeRow(0..2));

            assert!(habv.remove(r2));
            assert!(habv.is_empty());
            assert_eq!(habv.values.len(), 0);
            assert!(habv.free_rows.is_empty());
        }

        // +r0 +r1 +r2 -r1 -r2 -r0
        {
            let r0 = habv.push(&42u32);
            let r1 = habv.push(&84u32);
            let r2 = habv.push(&84u32);
            assert_eq!(r0, 0..4);
            assert_eq!(r1, 32..36);
            assert_eq!(r2, 64..68);
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 68);
            assert!(habv.free_rows.is_empty());

            assert!(habv.remove(r1.clone()));
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 68);
            assert_eq!(habv.free_rows.len(), 1);
            assert_eq!(habv.free_rows[0], FreeRow(1..2));

            assert!(habv.remove(r2.clone()));
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 32); // can't recover exact alloc (4), only row-aligned size (32)
            assert!(habv.free_rows.is_empty()); // merged!

            assert!(habv.remove(r0));
            assert!(habv.is_empty());
            assert_eq!(habv.values.len(), 0);
            assert!(habv.free_rows.is_empty());
        }

        // +r0 +r1 +r2 +r3 +r4 -r3 -r1 -r2 -r4 r0
        {
            let r0 = habv.push(&42u32);
            let r1 = habv.push(&84u32);
            let r2 = habv.push(&84u32);
            let r3 = habv.push(&84u32);
            let r4 = habv.push(&84u32);
            assert_eq!(r0, 0..4);
            assert_eq!(r1, 32..36);
            assert_eq!(r2, 64..68);
            assert_eq!(r3, 96..100);
            assert_eq!(r4, 128..132);
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 132);
            assert!(habv.free_rows.is_empty());

            assert!(habv.remove(r3.clone()));
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 132);
            assert_eq!(habv.free_rows.len(), 1);
            assert_eq!(habv.free_rows[0], FreeRow(3..4));

            assert!(habv.remove(r1.clone()));
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 132);
            assert_eq!(habv.free_rows.len(), 2);
            assert_eq!(habv.free_rows[0], FreeRow(1..2)); // sorted!
            assert_eq!(habv.free_rows[1], FreeRow(3..4));

            assert!(habv.remove(r2.clone()));
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 132);
            assert_eq!(habv.free_rows.len(), 1); // merged!
            assert_eq!(habv.free_rows[0], FreeRow(1..4)); // merged!

            assert!(habv.remove(r4.clone()));
            assert!(!habv.is_empty());
            assert_eq!(habv.values.len(), 32); // can't recover exact alloc (4), only row-aligned size (32)
            assert!(habv.free_rows.is_empty());

            assert!(habv.remove(r0));
            assert!(habv.is_empty());
            assert_eq!(habv.values.len(), 0);
            assert!(habv.free_rows.is_empty());
        }
    }
}

#[cfg(all(test, feature = "gpu_tests"))]
mod gpu_tests {
    use tests::*;

    use super::*;
    use crate::test_utils::MockRenderer;

    #[test]
    fn abv_write() {
        let renderer = MockRenderer::new();
        let device = renderer.device();
        let queue = renderer.queue();

        // Create a dummy CommandBuffer to force the write_buffer() call to have any
        // effect.
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test"),
        });
        let command_buffer = encoder.finish();

        let item_align = device.limits().min_storage_buffer_offset_alignment as u64;
        let mut abv = AlignedBufferVec::<GpuDummyComposed>::new(
            BufferUsages::STORAGE | BufferUsages::MAP_READ,
            NonZeroU64::new(item_align),
            None,
        );
        let final_align = item_align.max(<GpuDummyComposed as ShaderSize>::SHADER_SIZE.get());
        assert_eq!(abv.aligned_size(), final_align as usize);

        const CAPACITY: usize = 42;

        // Write buffer (CPU -> GPU)
        abv.push(GpuDummyComposed {
            tag: 1,
            ..Default::default()
        });
        abv.push(GpuDummyComposed {
            tag: 2,
            ..Default::default()
        });
        abv.push(GpuDummyComposed {
            tag: 3,
            ..Default::default()
        });
        abv.reserve(CAPACITY, &device);
        abv.write_buffer(&device, &queue);
        // need a submit() for write_buffer() to be processed
        queue.submit([command_buffer]);
        let (tx, rx) = futures::channel::oneshot::channel();
        queue.on_submitted_work_done(move || {
            tx.send(()).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let _ = futures::executor::block_on(rx);
        println!("Buffer written");

        // Read back (GPU -> CPU)
        let buffer = abv.buffer();
        let buffer = buffer.as_ref().expect("Buffer was not allocated");
        let buffer = buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let _result = futures::executor::block_on(rx);
        let view = buffer.get_mapped_range();

        // Validate content
        assert_eq!(view.len(), final_align as usize * CAPACITY);
        for i in 0..3 {
            let offset = i * final_align as usize;
            let dummy_composed: &[GpuDummyComposed] =
                cast_slice(&view[offset..offset + std::mem::size_of::<GpuDummyComposed>()]);
            assert_eq!(dummy_composed[0].tag, (i + 1) as u32);
        }
    }
}
