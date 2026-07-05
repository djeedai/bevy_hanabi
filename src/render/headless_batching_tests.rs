#![cfg(test)]

mod tests {
    use crate::render::GpuDispatchIndirectArgs;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct BatchInfo {
        spawner_base: u32,
        base_particle: u32,
        prefix_sum_offset: u32,
        prefix_sum_count: u32,
        total_update_count: u32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct EffectLocation {
        effect_index: u32,
        base_particle: u32,
        update_index: u32,
    }

    fn run_prefix_sum_pass(
        batch_infos: &mut [BatchInfo],
        prefix_sum: &mut [u32],
        dispatch: &mut [GpuDispatchIndirectArgs],
    ) {
        for (batch_index, batch) in batch_infos.iter_mut().enumerate() {
            let offset = batch.prefix_sum_offset as usize;
            let end = offset + batch.prefix_sum_count as usize;
            let mut sum = 0_u32;
            for value in prefix_sum.iter_mut().take(end).skip(offset) {
                let count = *value;
                *value = sum;
                sum += count;
            }
            batch.total_update_count = sum;
            dispatch[batch_index] = GpuDispatchIndirectArgs {
                x: sum.div_ceil(64),
                y: 1,
                z: 1,
            };
        }
    }

    fn find_location_from_particle(
        batch: BatchInfo,
        prefix_sum: &[u32],
        slab_particle_index: u32,
    ) -> EffectLocation {
        // Compute passes use packed indices (alive-count domain), not slab indices.
        let update_particle_index = slab_particle_index;
        let mut lo = batch.prefix_sum_offset;
        let mut hi = lo + batch.prefix_sum_count;
        while lo < hi {
            let mid = (hi + lo) >> 1;
            let base_particle = prefix_sum[mid as usize];
            if update_particle_index >= base_particle {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let base_particle = prefix_sum[(lo - 1) as usize];
        let effect_index = lo - 1 - batch.prefix_sum_offset;
        let update_index = update_particle_index - base_particle;
        EffectLocation {
            effect_index,
            base_particle,
            update_index,
        }
    }

    #[test]
    fn batching_prefix_sum_contract() {
        // Simulate the vfx_indirect output: alive counts for each effect instance.
        let mut prefix_sum = vec![10, 5, 8, 6];
        let mut batches = vec![
            BatchInfo {
                spawner_base: 0,
                base_particle: 100,
                prefix_sum_offset: 0,
                prefix_sum_count: 3,
                total_update_count: 0,
            },
            BatchInfo {
                spawner_base: 3,
                base_particle: 500,
                prefix_sum_offset: 3,
                prefix_sum_count: 1,
                total_update_count: 0,
            },
        ];
        let mut dispatch = vec![
            GpuDispatchIndirectArgs { x: 0, y: 0, z: 0 },
            GpuDispatchIndirectArgs { x: 0, y: 0, z: 0 },
        ];

        run_prefix_sum_pass(&mut batches, &mut prefix_sum, &mut dispatch);

        // Batch #0: [10,5,8] -> exclusive prefix [0,10,15], total 23
        // Batch #1: [6] -> exclusive prefix [0], total 6
        assert_eq!(prefix_sum, vec![0, 10, 15, 0]);
        assert_eq!(batches[0].total_update_count, 23);
        assert_eq!(batches[1].total_update_count, 6);
        assert_eq!(dispatch[0], GpuDispatchIndirectArgs { x: 1, y: 1, z: 1 });
        assert_eq!(dispatch[1], GpuDispatchIndirectArgs { x: 1, y: 1, z: 1 });
    }

    #[test]
    fn location_mapping_uses_batch_base_particle() {
        // Exclusive prefix sums for 3 effects in a batch.
        let prefix_sum = vec![0, 10, 15];
        let batch = BatchInfo {
            spawner_base: 7,
            base_particle: 0,
            prefix_sum_offset: 0,
            prefix_sum_count: 3,
            total_update_count: 23,
        };

        // First effect
        assert_eq!(
            find_location_from_particle(batch, &prefix_sum, 0),
            EffectLocation {
                effect_index: 0,
                base_particle: 0,
                update_index: 0
            }
        );

        // Second effect starts at packed index 10
        assert_eq!(
            find_location_from_particle(batch, &prefix_sum, 10),
            EffectLocation {
                effect_index: 1,
                base_particle: 10,
                update_index: 0
            }
        );

        // Third effect starts at packed index 15
        assert_eq!(
            find_location_from_particle(batch, &prefix_sum, 20),
            EffectLocation {
                effect_index: 2,
                base_particle: 15,
                update_index: 5
            }
        );
    }

    #[test]
    fn render_batch_ids_must_be_initialized_for_all_merged_effects() {
        let mut render_batch_info_ids = vec![u32::MAX, u32::MAX, u32::MAX, u32::MAX];
        let mut next_id = 42_u32;
        for id in &mut render_batch_info_ids {
            if *id == u32::MAX {
                *id = next_id;
                next_id += 1;
            }
        }

        assert!(render_batch_info_ids.iter().all(|id| *id != u32::MAX));
        assert_eq!(render_batch_info_ids, vec![42, 43, 44, 45]);
    }
}
