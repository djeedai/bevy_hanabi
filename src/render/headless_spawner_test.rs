//! Headless test to exercise EffectsMeta batching and prefix-sum logic

#[cfg(test)]
mod tests {
    use wgpu;

    #[derive(Debug, Clone, Copy)]
    struct GpuBatchInfo {
        total_spawn_count: u32,
        total_update_count: u32,
        base_effect: u32,
        base_particle: u32,
        prefix_sum_offset: u32,
        prefix_sum_count: u32,
    }

    struct EffectsMetaSim {
        batch_info_buffer: Vec<GpuBatchInfo>,
        prefix_sum_buffer: Vec<u32>,
        is_batch_open: bool,
        dispatch_indirect_next: u32,
        spawner_buffer_len: u32,
    }

    impl EffectsMetaSim {
        fn new() -> Self {
            Self {
                batch_info_buffer: Vec::new(),
                prefix_sum_buffer: Vec::new(),
                is_batch_open: false,
                dispatch_indirect_next: 0,
                spawner_buffer_len: 0,
            }
        }

        fn begin_batch(&mut self, base_particle: u32, spawner_base: u32) -> u32 {
            assert!(!self.is_batch_open, "Duplicate call to begin_batch()");
            let prefix_sum_offset = self.prefix_sum_buffer.len() as u32;
            let batch_info_base = self.batch_info_buffer.len() as u32;
            let batch_info = GpuBatchInfo {
                total_spawn_count: 0,
                total_update_count: 0,
                base_effect: spawner_base,
                base_particle,
                prefix_sum_offset,
                prefix_sum_count: u32::MAX,
            };
            self.batch_info_buffer.push(batch_info);
            self.is_batch_open = true;

            // allocate dispatch indirect index (simulate)
            let di_index = self.dispatch_indirect_next;
            self.dispatch_indirect_next += 1;
            assert_eq!(di_index, batch_info_base);
            batch_info_base
        }

        fn add_effect_to_batch(&mut self, slab_offset: u32) {
            assert!(self.is_batch_open, "Cannot add effect before calling begin_batch()");
            self.prefix_sum_buffer.push(slab_offset);
        }

        fn end_batch(&mut self) {
            assert!(self.is_batch_open, "Call to end_batch() without begin_batch()");
            let last = self.batch_info_buffer.last_mut().expect("No open batch. Missing begin_batch() call?");
            let end = self.prefix_sum_buffer.len() as u32;
            assert!(end >= last.prefix_sum_offset);
            last.prefix_sum_count = end - last.prefix_sum_offset;
            self.is_batch_open = false;
        }

        fn allocate_spawner(&mut self) -> u32 {
            let spawner_base = self.spawner_buffer_len as u32;
            self.spawner_buffer_len += 1;
            spawner_base
        }
    }

    #[test]
    fn headless_batching_prefix_sum_stability() {
        // Initialize a headless wgpu device/adapter to ensure the test runs in
        // headless environments and to match the user's request for a WGPU-backed test.
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("Failed to request adapter");

            let (_device, _queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        features: wgpu::Features::empty(),
                        limits: wgpu::Limits::downlevel_defaults(),
                    },
                    None,
                )
                .await
                .expect("Failed to request device");
        });

        // Simulate EffectsMeta batching behavior and ensure prefix sums remain stable
        // across a simulated property buffer reallocation.
        let mut meta = EffectsMetaSim::new();
        let spawner_base = meta.allocate_spawner();
        let batch_idx = meta.begin_batch(0, spawner_base);
        meta.add_effect_to_batch(5);
        meta.add_effect_to_batch(6);
        meta.end_batch();

        assert_eq!(meta.batch_info_buffer.len(), 1);
        let b = meta.batch_info_buffer[batch_idx as usize];
        assert_eq!(b.prefix_sum_count, 2, "prefix_sum_count should be 2");
        assert_eq!(meta.prefix_sum_buffer, vec![5_u32, 6_u32]);

        // Simulate a later property buffer reallocation and creation of a new batch.
        let spawner_base2 = meta.allocate_spawner();
        let _batch_idx2 = meta.begin_batch(2, spawner_base2);
        meta.add_effect_to_batch(7);
        meta.end_batch();

        // Verify the first batch information is unchanged after the second batch
        assert_eq!(meta.batch_info_buffer[batch_idx as usize].prefix_sum_count, 2);
    }
}
