use std::num::NonZeroU64;

use bevy::{
    log::trace,
    render::{
        render_resource::{
            Buffer, BufferAddress, BufferDescriptor, BufferUsages, ShaderSize, ShaderType,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::{cast_slice, Pod};
use copyless::VecHelper;

use crate::next_multiple_of;

/// Like Bevy's [`BufferVec`], but with correct item alignment.
///
/// This is a helper to ensure the data is properly aligned when copied to GPU,
/// depending on the device constraints and the WGSL rules. Generally the
/// alignment is one of the [`WgpuLimits`], and is also ensured to be
/// compatible with WGSL.
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

impl<T: Pod + ShaderType + ShaderSize> Default for AlignedBufferVec<T> {
    fn default() -> Self {
        let item_size = size_of::<T>();
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

impl<T: Pod + ShaderType + ShaderSize> AlignedBufferVec<T> {
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
            "AlignedBufferVec: item_size={} aligned_size={}",
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

    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
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

    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn push(&mut self, value: T) -> usize {
        let index = self.values.len();
        self.values.alloc().init(value);
        index
    }

    /// Reserve some capacity into the buffer.
    ///
    /// # Returns
    ///
    /// `true` if the buffer was (re)allocated, or `false` if an existing buffer
    /// was reused which already had enough capacity.
    pub fn reserve(&mut self, capacity: usize, device: &RenderDevice) -> bool {
        if capacity > self.capacity {
            let size = self.aligned_size * capacity;
            trace!(
                "reserve: increase capacity from {} to {} elements, new size {} bytes",
                self.capacity,
                capacity,
                size
            );
            self.capacity = capacity;
            self.buffer = Some(device.create_buffer(&BufferDescriptor {
                label: self.label.as_ref().map(|s| &s[..]),
                size: size as BufferAddress,
                usage: BufferUsages::COPY_DST | self.buffer_usage,
                mapped_at_creation: false,
            }));
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
            "write_buffer: values.len={} item_size={} aligned_size={}",
            self.values.len(),
            self.item_size,
            self.aligned_size
        );
        let buffer_changed = self.reserve(self.values.len(), device);
        if let Some(buffer) = &self.buffer {
            let aligned_size = self.aligned_size * self.values.len();
            trace!("aligned_buffer: size={}", aligned_size);
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

impl<T: Pod + ShaderType + ShaderSize> std::ops::Index<usize> for AlignedBufferVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<T: Pod + ShaderType + ShaderSize> std::ops::IndexMut<usize> for AlignedBufferVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
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
    fn abv_sizes() {
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
        device.poll(wgpu::Maintain::Wait);
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
        device.poll(wgpu::Maintain::Wait);
        let _result = futures::executor::block_on(rx);
        let view = buffer.get_mapped_range();

        // Validate content
        assert_eq!(view.len(), final_align as usize * CAPACITY);
        for i in 0..3 {
            let offset = i * final_align as usize;
            let dummy_composed: &[GpuDummyComposed] =
                cast_slice(&view[offset..offset + size_of::<GpuDummyComposed>()]);
            assert_eq!(dummy_composed[0].tag, (i + 1) as u32);
        }
    }
}
