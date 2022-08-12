use bevy::render::renderer::{RenderDevice, RenderQueue};

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
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .expect("Failed to create device");

        // Turn into Bevy objects
        let device = RenderDevice::from(std::sync::Arc::new(device));
        let queue = std::sync::Arc::new(queue);

        MockRenderer {
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
