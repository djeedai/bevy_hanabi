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

struct MergeParams {
    /// Maximum number of items per list for this pass.
    max_list_size: u32,
    /// Half of the ping-pong buffer to read from: 0 for the first half, 1 for
    /// the second half.
    source_buffer: u32,
}

@group(0) @binding(0) var<storage, read_write> sort_buffer : SortBuffer;
@group(0) @binding(1) var<storage, read> merge_params : MergeParams;

/// Merge two sorted lists [a..a+num_a) and [b..b+num_b) into a single sorted list.
///
/// The source elements are read from the sort buffer starting at offsets (src + a)
/// and (src + b), and the merged list written starting at offset (dst + a).
fn merge_lists_serial(a: u32, num_a: u32, b: u32, num_b: u32, src: u32, dst: u32) {
    var ia = a;
    var ib = b;
    let a_end = a + num_a;
    let b_end = b + num_b;
    for (var i: u32 = 0u; i < num_a + num_b; i += 1u) {
        if ((ib >= b_end) || ((ia < a_end) && !compare_greater(sort_buffer.pairs[src + ia], sort_buffer.pairs[src + ib]))) {
            sort_buffer.pairs[dst + a + i] = sort_buffer.pairs[src + ia];
            ia += 1u;
        } else {
            sort_buffer.pairs[dst + a + i] = sort_buffer.pairs[src + ib];
            ib += 1u;
        }
    }
}

/// Sort each block in parallel on a separate workgroup.
@compute @workgroup_size(64)
fn merge_sort(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let tid = global_invocation_id.x;
    let total_num_items = u32(sort_buffer.count);
    let list_size = merge_params.max_list_size;
    let num_lists = (total_num_items + list_size - 1u) / list_size;
    let num_merge_operations = (num_lists + 1u) / 2u;
    if (tid < num_merge_operations) {
        // We always merge 2 consecutive lists of up to list_size items. The last list may have
        // less elements, if total_num_items is not a multiple of list_size (which is common).
        // An unmatched last list is copied unchanged to the other ping-pong half.
        let start_a = tid * list_size * 2u;
        let start_b = start_a + list_size;
        let num_a = min(list_size, total_num_items - start_a);
        let num_b = min(list_size, total_num_items - min(start_b, total_num_items));
        let src = merge_params.source_buffer * total_num_items;
        let dst = (1u - merge_params.source_buffer) * total_num_items;
        merge_lists_serial(start_a, num_a, start_b, num_b, src, dst);
    }
}
