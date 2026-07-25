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

/// Size of a block of KeyValuePair in workgroup memory.
const blockSize: u32 = 512u;

// Workgroup has at least 16kB memory (max_compute_workgroup_storage_size).
// Note that Vulkan on Windows 11 report 16352, not 16384 (so, lower than
// the default in the wgpu docs).
var<workgroup> arr0 : array<KeyValuePair, blockSize>;
var<workgroup> arr1 : array<KeyValuePair, blockSize>;

/// Find the index of an effect from the index of a particle.
///
/// This uses a binary search on the slab_offset field of the spawners array, which
/// represents a prefix sum of the particle count per effect (for previous effects;
/// the value is actually the base particle so the first entry is always 0).
fn find_effect_from_particle(num_effects: u32, particle_index: u32) -> u32 {
    var lo = 0u;
    var hi = num_effects;
    var nnn = 0;
    while (lo < hi) {
        let mid = (hi + lo) >> 1u;
        let base_particle = arr0[mid].key;
        if (particle_index >= base_particle) {
            lo = mid + 1u;
        } else if (particle_index < base_particle) {
            hi = mid;
        }
        nnn += 1;
        if (nnn >= 100) {
            return 0xDEADBEEFu;
        }
    }
    return lo - 1u;
}

#ifdef TEST

@compute @workgroup_size(64)
fn test_find_effect_from_particle(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let tid = global_invocation_id.x;
    if (tid >= 64) {
        return;
    }
    
    let num_particles = arrayLength(&sort_buffer.pairs);
    let num_effects = u32(sort_buffer.count);

    // Copy prefix sum into arr0[]
    if (tid < num_effects) {
        arr0[tid] = sort_buffer.pairs[tid];
    }

    workgroupBarrier();

    let particle_per_thread = (num_particles + 63u) >> 6u;
    let first_particle = particle_per_thread * tid;
    let last_particle = min(first_particle + particle_per_thread, num_particles);
    for (var i = first_particle; i < last_particle; i += 1u) {
        sort_buffer.pairs[i].value = find_effect_from_particle(num_effects, i);
    }
}

#endif

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
