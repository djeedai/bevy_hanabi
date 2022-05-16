use bevy::{
    core::{cast_slice, Pod},
    log::trace,
    render::{
        render_resource::{Buffer, BufferAddress, BufferDescriptor, BufferUsages},
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::cast_slice_mut;
use copyless::VecHelper;

// TODO - filler for usize.next_multiple_of()
fn next_multiple_of(value: usize, align: usize) -> usize {
    let count = (value + align - 1) / align;
    count * align
}

/// Like Bevy's [`BufferVec`], but with an explicit item alignment.
///
/// This is a helper to ensure the data is properly aligned when copied to GPU, depending
/// on the device constraints. Generally the alignment is one of the [`wgpu::Limits`].
pub struct AlignedBufferVec<T: Pod> {
    values: Vec<T>,
    buffer: Option<Buffer>,
    capacity: usize,
    item_size: usize,
    aligned_size: usize,
    buffer_usage: BufferUsages,
    label: Option<String>,
}

impl<T: Pod> Default for AlignedBufferVec<T> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            buffer: None,
            capacity: 0,
            buffer_usage: BufferUsages::all(),
            item_size: std::mem::size_of::<T>(),
            aligned_size: std::mem::size_of::<T>(),
            label: None,
        }
    }
}

impl<T: Pod> AlignedBufferVec<T> {
    pub fn new(buffer_usage: BufferUsages, item_align: usize, label: Option<String>) -> Self {
        let item_size = std::mem::size_of::<T>();
        //let aligned_size = item_size.next_multiple_of(item_align);
        let aligned_size = next_multiple_of(item_size, item_align);
        assert!(aligned_size >= item_size);
        assert!(aligned_size % item_align == 0);
        Self {
            buffer_usage,
            aligned_size,
            label,
            ..Default::default()
        }
    }

    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn aligned_size(&self) -> usize {
        self.aligned_size
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn push(&mut self, value: T) -> usize {
        let index = self.values.len();
        self.values.alloc().init(value);
        index
    }

    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) {
        if capacity > self.capacity {
            self.capacity = capacity;
            let size = self.aligned_size * capacity;
            self.buffer = Some(device.create_buffer(&BufferDescriptor {
                label: self.label.as_ref().map(|s| &s[..]),
                size: size as BufferAddress,
                usage: BufferUsages::COPY_DST | self.buffer_usage,
                mapped_at_creation: false,
            }));
        }
    }

    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.values.is_empty() {
            return;
        }
        trace!(
            "write_buffer: values.len={} item_size={} aligned_size={}",
            self.values.len(),
            self.item_size,
            self.aligned_size
        );
        self.reserve(self.values.len(), device);
        if let Some(buffer) = &self.buffer {
            let aligned_size = self.aligned_size * self.values.len();
            trace!("aligned_buffer: size={}", aligned_size);
            let mut aligned_buffer: Vec<u8> = Vec::with_capacity(aligned_size);
            aligned_buffer.resize(aligned_size, 0);
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
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::MockRenderer;

    const INTS: &[usize] = &[1, 2, 4, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33];

    /// Same as `INTS`, rounded up to 16
    const INTS16: &[usize] = &[16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 48];

    #[test]
    fn next_multiple() {
        // align-1 is no-op
        for &size in INTS {
            assert_eq!(size, next_multiple_of(size, 1));
        }

        // zero-sized is always aligned
        for &align in INTS {
            assert_eq!(0, next_multiple_of(0, align));
        }

        // size < align : rounds up to align
        for &size in INTS {
            assert_eq!(256, next_multiple_of(size, 256));
        }

        // size > align : actually aligns
        for (&size, &aligned_size) in INTS.iter().zip(INTS16) {
            assert_eq!(aligned_size, next_multiple_of(size, 16));
        }
    }

    #[test]
    fn abv_align() {
        for &align in INTS {
            let abv = AlignedBufferVec::<u8>::new(BufferUsages::STORAGE, align, None);
            assert_eq!(abv.aligned_size(), align);
        }

        for &align in INTS {
            let abv = AlignedBufferVec::<u32>::new(BufferUsages::STORAGE, align, None);
            assert_eq!(abv.aligned_size(), next_multiple_of(4, align));
        }

        for &align in INTS {
            let abv = AlignedBufferVec::<[u8; 27]>::new(BufferUsages::STORAGE, align, None);
            assert_eq!(abv.aligned_size(), next_multiple_of(27, align));
        }
    }

    #[test]
    fn abv_push() {
        const SIZE: usize = 27;
        const ALIGN: usize = 32;
        let mut abv = AlignedBufferVec::<[u8; SIZE]>::new(BufferUsages::STORAGE, ALIGN, None);
        assert_eq!(abv.aligned_size(), next_multiple_of(SIZE, ALIGN));
        assert!(abv.is_empty());
        abv.push([9; SIZE]);
        assert!(!abv.is_empty());
        assert_eq!(abv.len(), 1);
    }
}

#[cfg(all(test, feature = "gpu_tests"))]
mod gpu_tests {
    use super::*;
    use crate::test_utils::MockRenderer;

    #[test]
    fn abv_write() {
        let renderer = MockRenderer::new();
        let device = renderer.device();
        let queue = renderer.queue();

        // Create a dummy CommandBuffer to force the write_buffer() call to have any effect
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test"),
        });
        let command_buffer = encoder.finish();

        // Write buffer (CPU -> GPU)
        const SIZE: usize = 27;
        const ALIGN: usize = 32;
        const CAPACITY: usize = 16;
        let mut abv = AlignedBufferVec::<[u8; SIZE]>::new(
            BufferUsages::STORAGE | BufferUsages::MAP_READ,
            ALIGN,
            None,
        );
        abv.push([9; SIZE]);
        abv.push([6; SIZE]);
        abv.push([3; SIZE]);
        abv.reserve(CAPACITY, &device);
        abv.write_buffer(&device, &queue);
        // need a submit() for write_buffer() to be processed
        queue.submit([command_buffer]);
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(queue.on_submitted_work_done());
        println!("Buffer written");

        // Read back (GPU -> CPU)
        let buffer = abv.buffer();
        let buffer = buffer.as_ref().expect("Buffer was not allocated");
        let buffer = buffer.slice(..);
        let fut = buffer.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);
        pollster::block_on(fut).expect("Failed to map");
        let view = buffer.get_mapped_range();

        // Validate content
        assert_eq!(view.len(), ALIGN * CAPACITY);
        for i in 0..3 {
            let offset = i * ALIGN;
            let value: u8 = (9 - i * 3) as u8;
            let value: [u8; SIZE] = [value; SIZE];
            let vec: &[u8] = cast_slice(&view[offset..offset + SIZE]);
            assert_eq!(vec, value);
        }
    }
}
