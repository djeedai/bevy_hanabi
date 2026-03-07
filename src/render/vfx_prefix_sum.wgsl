#import bevy_hanabi::vfx_common::{BatchInfo, DispatchIndirectArgs}

@group(0) @binding(0) var<storage, read_write> batch_infos : array<BatchInfo>;
@group(0) @binding(1) var<storage, read_write> prefix_sum : array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch_indirect_buffer : array<DispatchIndirectArgs>;

/// Compute the prefix sum of each effect batch.
///
/// This is invoked once per effect batch each frame, before the update pass, to recalculate
/// the prefix sum of the alive particle counts per instance in that batch. Before this pass,
/// the prefix sum buffer contins the number of alive particles for each instance. After it,
/// it contains a prefix sum of that number.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // Get the batch to update
    let batch_index = global_invocation_id.x;
    let batch_count = arrayLength(&batch_infos);
    if (batch_index >= batch_count) {
        return;
    }

    // Find the slice of prefix sum to update inside the global shared buffer
    let offset = batch_infos[batch_index].prefix_sum_offset;
    let count = batch_infos[batch_index].prefix_sum_count;

    // Calculate the prefix sum
    let end = offset + count;
    var sum = 0u;
    for (var i = offset; i < end; i += 1u) {
        let count = prefix_sum[i];
        // Exclusive: store current sum before incrementing
        prefix_sum[i] = sum;
        sum = sum + count;
    }

    // Store the sum back into the batch
    batch_infos[batch_index].total_update_count = sum;

    // Update the dispatch indirect count
    dispatch_indirect_buffer[batch_index].x = (sum + 63u) >> 6u;
    dispatch_indirect_buffer[batch_index].y = 1u;
    dispatch_indirect_buffer[batch_index].z = 1u;
}
