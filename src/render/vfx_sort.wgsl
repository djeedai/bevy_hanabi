struct KeyValuePair {
    /// Sorting key.
    key: u32,
#ifdef HAS_DUAL_KEY
    /// Secondary sorting key. Sorts value with the same primary key.
    key2: u32,
#endif
    /// Value associated with the sort key(s), generally an index to some other data.
    /// Copied as is and otherwise ignored by the sorting algorithm.
    value: u32,
}

struct SortBuffer {
    count: i32,
    pairs: array<KeyValuePair>,
}

/// Check whether kv1 > kv2, comparing the key(s) of each pair.
fn compare_greater(kv1: KeyValuePair, kv2: KeyValuePair) -> bool {
    if (kv1.key > kv2.key) {
        return true;
    }
#ifdef HAS_DUAL_KEY
    if (kv1.key == kv2.key) {
        return kv1.key2 > kv2.key2;
    }
#endif
    return false;
}

@group(0) @binding(0) var<storage, read_write> sort_buffer: SortBuffer;

/// Naive insertion sort. TODO: replace with something faster.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // Naive single-threaded sort
    if (global_invocation_id.x != 0) {
        return;
    }

    // Insertion sort
    let num_items = sort_buffer.count;
    for (var i: i32 = 1; i < num_items; i += 1) {
        var kv = sort_buffer.pairs[i];
        var j = i;
        while (j > 0 && compare_greater(sort_buffer.pairs[j - 1], kv)) {
            sort_buffer.pairs[j] = sort_buffer.pairs[j - 1];
            j -= 1;
        }
        sort_buffer.pairs[j] = kv;
    }

    // Clear for next frame
    sort_buffer.count = 0;
}
