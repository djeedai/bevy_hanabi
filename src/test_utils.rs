use bevy::render::renderer::{RenderDevice, RenderQueue};

/// Utility to compare floating-point values with a tolerance.
pub(crate) fn abs_diff_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// Assert that two floating-point quantities are approximately equal.
///
/// This macro asserts that the absolute difference between the two first
/// arguments is strictly less than a tolerance factor, which can be explicitly
/// passed as third argument or implicitly defaults to `1e-5`.
///
/// # Usage
///
/// ```
/// let x = 3.500009;
/// assert_approx_eq!(x, 3.5);       // default tolerance 1e-5
///
/// let x = 3.509;
/// assert_approx_eq!(x, 3.5, 0.01); // explicit tolerance
/// ```
macro_rules! assert_approx_eq {
    ($left:expr, $right:expr $(,)?) => {
        match (&$left, &$right) {
            (left_val, right_val) => {
                assert!(
                    abs_diff_eq(*left_val, *right_val, 1e-5),
                    "assertion failed: expected={} actual={} delta={} tol=1e-5(default)",
                    left_val,
                    right_val,
                    (left_val - right_val).abs(),
                );
            }
        }
    };
    ($left:expr, $right:expr, $tol:expr $(,)?) => {
        match (&$left, &$right, &$tol) {
            (left_val, right_val, tol_val) => {
                assert!(
                    abs_diff_eq(*left_val, *right_val, *tol_val),
                    "assertion failed: expected={} actual={} delta={} tol={}",
                    left_val,
                    right_val,
                    (left_val - right_val).abs(),
                    tol_val
                );
            }
        }
    };
}

pub(crate) use assert_approx_eq;

/// Mock renderer backed by any available WGPU backend, and simulating the real
/// Bevy backend to enable testing rendering-related features.
pub(crate) struct MockRenderer {
    /// WGPU instance backed by any available backend.
    instance: wgpu::Instance,
    /// Default WGPU adapter for the configured instance.
    adapter: wgpu::Adapter,
    /// Bevy render device abstracting the WGPU backend.
    device: RenderDevice,
    /// Bevy render queue sending commands to the render device.
    queue: RenderQueue,
}

impl MockRenderer {
    /// Create a new mock renderer with a default backend and adapter.
    pub fn new() -> Self {
        // Create the WGPU adapter
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter =
            futures::executor::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
            }))
            .expect("Failed to find an appropriate adapter");

        // Create the logical device and command queue
        let (device, queue) = futures::executor::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                // Request MAPPABLE_PRIMARY_BUFFERS to allow MAP_WRITE|COPY_DST.
                // FIXME - Should use a separate buffer from primary to support more platforms.
                features: wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .expect("Failed to create device");

        // Turn into Bevy objects
        let device = RenderDevice::from(std::sync::Arc::new(device));
        let queue = RenderQueue(std::sync::Arc::new(queue));

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }

    /// Get the Bevy render device of the mock renderer.
    pub fn device(&self) -> RenderDevice {
        self.device.clone()
    }

    /// Get the Bevy render queue of the mock renderer.
    pub fn queue(&self) -> RenderQueue {
        self.queue.clone()
    }
}
