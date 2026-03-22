/// Key-value pair to sort, with optional secondary key.
///
/// Sorting operates on the key (and the secondary key as discriminant, if present,
/// and if the primary keys are equal). The value is carried over alongside the key(s),
/// unmodified. It generally represents or indexes the payload associated with the key(s).
struct KeyValuePair {
    /// Sorting key.
    key: u32,
#ifdef HAS_DUAL_KEY
    /// Secondary sorting key. Used for values with equal primary key.
    key2: u32,
#endif
    /// Value associated with the sort key(s), generally an index to some other data.
    /// Copied as is and otherwise ignored by the sorting algorithm.
    value: u32,
}

/// Buffer of key-value pairs to sort.
struct SortBuffer {
    /// Number of items in the buffer.
    count: i32,
    /// Pairs to sort. On output, contains the pairs in sorted order.
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

@group(0) @binding(0) var<storage, read_write> sort_buffer : SortBuffer;

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
