#![cfg(test)]

#[cfg(feature = "gpu_tests")]
use bevy::render::renderer::{RenderDevice, RenderQueue};

use bevy::prelude::{Quat, Vec2, Vec3, Vec4};
use std::ops::Sub;

/// Utility trait to compare floating-point values with a tolerance.
pub(crate) trait AbsDiffEq {
    /// Calculate the absolute value of the difference between two
    /// floating-point quantities. For non-scalar quantities, the maximum
    /// absolute difference for all components is returned.
    fn abs_diff(a: &Self, b: &Self) -> f32;

    /// Check if two floating-point quantities are approximately equal within a
    /// given tolerance. Non-scalar values are checked component-wise.
    fn abs_diff_eq(a: &Self, b: &Self, tol: f32) -> bool;
}

impl AbsDiffEq for f32 {
    fn abs_diff(a: &f32, b: &f32) -> f32 {
        (a - b).abs()
    }

    fn abs_diff_eq(a: &f32, b: &f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }
}

impl AbsDiffEq for f64 {
    fn abs_diff(a: &f64, b: &f64) -> f32 {
        (a - b).abs() as f32
    }

    fn abs_diff_eq(a: &f64, b: &f64, tol: f32) -> bool {
        (a - b).abs() < tol as f64
    }
}

impl AbsDiffEq for Vec2 {
    fn abs_diff(a: &Vec2, b: &Vec2) -> f32 {
        a.sub(*b).abs().max_element()
    }

    fn abs_diff_eq(a: &Vec2, b: &Vec2, tol: f32) -> bool {
        a.abs_diff_eq(*b, tol)
    }
}

impl AbsDiffEq for Vec3 {
    fn abs_diff(a: &Vec3, b: &Vec3) -> f32 {
        a.sub(*b).abs().max_element()
    }

    fn abs_diff_eq(a: &Vec3, b: &Vec3, tol: f32) -> bool {
        a.abs_diff_eq(*b, tol)
    }
}

impl AbsDiffEq for Vec4 {
    fn abs_diff(a: &Vec4, b: &Vec4) -> f32 {
        a.sub(*b).abs().max_element()
    }

    fn abs_diff_eq(a: &Vec4, b: &Vec4, tol: f32) -> bool {
        a.abs_diff_eq(*b, tol)
    }
}

impl AbsDiffEq for Quat {
    fn abs_diff(a: &Quat, b: &Quat) -> f32 {
        Vec4::from(*a).sub(Vec4::from(*b)).abs().max_element()
    }

    fn abs_diff_eq(a: &Quat, b: &Quat, tol: f32) -> bool {
        a.abs_diff_eq(*b, tol)
    }
}

/// Assert that two floating-point quantities are approximately equal.
///
/// This macro asserts that the absolute difference between the two first
/// arguments is strictly less than a tolerance factor, which can be explicitly
/// passed as third argument or implicitly defaults to `1e-5`.
///
/// The two quantities must implement the [`AbsDiffEq`] helper trait. This trait
/// is implemented for common floating-point quantities like `f32` or `f64`, or
/// associated vector math types like `Vec3` or `Quat`.
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
                    crate::test_utils::AbsDiffEq::abs_diff_eq(left_val, right_val, 1e-5),
                    "assertion failed: expected={} actual={} delta={} tol=1e-5(default)",
                    left_val,
                    right_val,
                    crate::test_utils::AbsDiffEq::abs_diff(left_val, right_val),
                );
            }
        }
    };
    ($left:expr, $right:expr, $tol:expr $(,)?) => {
        match (&$left, &$right, &$tol) {
            (left_val, right_val, tol_val) => {
                assert!(
                    crate::test_utils::AbsDiffEq::abs_diff_eq(left_val, right_val, *tol_val),
                    "assertion failed: expected={} actual={} delta={} tol={}",
                    left_val,
                    right_val,
                    crate::test_utils::AbsDiffEq::abs_diff(left_val, right_val),
                    tol_val
                );
            }
        }
    };
}

pub(crate) use assert_approx_eq;

/// Mock renderer backed by any available WGPU backend, and simulating the real
/// Bevy backend to enable testing rendering-related features.
#[cfg(feature = "gpu_tests")]
#[allow(dead_code)]
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

#[cfg(feature = "gpu_tests")]
impl MockRenderer {
    /// Create a new mock renderer with a default backend and adapter.
    pub fn new() -> Self {
        // Create the WGPU adapter. Use PRIMARY backends (Vulkan, Metal, DX12,
        // Browser+WebGPU) to ensure we get a backend that supports compute and other
        // modern features we might need.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
        });
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
                // Request downlevel_defaults() for maximum compatibility in testing. The actual
                // Hanabi library uses the default requested mode of the app.
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .expect("Failed to create device");

        // Turn into Bevy objects
        let device = RenderDevice::from(device);
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
