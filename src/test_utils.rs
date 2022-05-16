use bevy::render::renderer::{RenderDevice, RenderQueue};

pub(crate) struct MockRenderer {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: RenderDevice,
    queue: RenderQueue,
}

impl MockRenderer {
    pub fn new() -> Self {
        // Create the WGPU adapter
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("Failed to find an appropriate adapter");

        // Create the logical device and command queue
        let (device, queue) = pollster::block_on(adapter.request_device(
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

    pub fn device(&self) -> RenderDevice {
        self.device.clone()
    }

    pub fn queue(&self) -> RenderQueue {
        self.queue.clone()
    }
}
