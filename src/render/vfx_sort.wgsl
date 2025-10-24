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

@group(0) @binding(0) var<storage, read_write> sort_buffer : SortBuffer;

/// Size of a block of KeyValuePair in workgroup memory.
override blockSize = 1024;

// Workgroup has at least 16kB memory (max_compute_workgroup_storage_size).
// Note that Vulkan on Windows 11 report 16352, not 16384 (so, lower than
// the default in the wgpu docs).
var<workgroup> arr0 : array<KeyValuePair, blockSize>;
var<workgroup> arr1 : array<KeyValuePair, blockSize>;

/// Sort locally inside a single workgroup, using workgroup-shared memory.
///
/// Workgroup has up to 16kB, for 8B keys (dual) that's 2k elements. Since
/// we need 2 arrays for ping-pong, we can do up to 1024 elements at once.
fn block_sort() {
    // Merge-sort blocks of 'size' items
    for (var size: i32 = 2; size <= 512; size *= 2) {

    }
}

/// Calculate a single merge path for a subsequent parallel merge.
///
/// INPUTS:
/// - Values A : var<workgroup> arr0[num_a]
/// - Values B : var<workgroup> arr1[num_b]
/// - Index of diagonal constraint : diag
///
/// OUTPUT:
/// Index in A of the merge path end. The index in B is ib = (diag - ia),
/// since a merge path is always of length 'diag' so (ia + ib == diag).
///
/// The merge path is such that the merged sub-list [0..diag[ contains the
/// first 'i_a' elements of A and the first 'i_b' elements of B. This means
/// that we can load and merge in parallel exactly the elements between two
/// consecutive diagonals, since we know exactly in advance which elements
/// of the merged lists will be used, before we even start the actual merge.
/// So calling calc_merge_path() N times sub-divides some merge lists into
/// N+1 independent mergeable intervals. We can then recusrively merge down.
///
/// https://moderngpu.github.io/bulkinsert.html
fn calc_merge_path(num_a: u32, num_b: u32, diag: u32) -> u32 {
    // Intersection of cross-diagonal with the X axis (alongside A) at the
    // bottom of the merge matrix. This is the min A value to consider.
    var begin = diag - min(diag, num_b); // == min(0, diag - num_b) >= 0
    // Intersection of cross-diagonal with the X axis (alongside A) at the
    // top of the merge matrix. This is the max A value to consider.
    var end = min(diag, num_a);

    // Binary search the cross-diagonal intersection with the merge path,
    // which is the index where the values are sorted.
    while (begin < end) {
        let mid = (begin + end) >> 1u;
        // Only if b < a do we move B, otherwise in case of equality we
        // favor A, to ensure stability (since A is the first merge list,
        // so its elements are originally located before those of B before
        // we merge).
        if (compare_greater(arr0[mid], arr1[diag - 1u - mid])) {
            end = mid;
        } else {
            begin = mid + 1;
        }
    }

    return begin;
}

/// Batcher's odd-even merge sort.
///
/// INPUTS:
/// - Values A : var<workgroup> arr0[]
/// - Offset from the start of arr0 where the values to sort start : offset
/// - Number of values to sort : n
///
/// OUTPUTS:
/// - Values A in arr0[], sorted
///
/// This is a sorting network. Not asymptotically optimal, but reasonably efficient.
/// It's more of a reference implementation, from 1998. Not the fastest by today's
/// standards.
///
/// https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
fn batcher_odd_even_mergesort(offset: u32, n: u32) {
    for (var p: u32 = 1; p < n; p += p) {
        for (var k: u32 = p; k >= 1; k = k >> 1) {
            for (var j: u32 = k % p; j + k < n; j += 2 * k) {
                for (var i: u32 = 0; i < min(k, n - j - k); i += 1) {
                    let i0 = i + j;
                    let i1 = i + j + k;
                    let f0 = f32(i0) / f32(2 * p);
                    let f1 = f32(i1) / f32(2 * p);
                    // i0 < i1 < n - 1
                    if (floor(f0) == floor(f1)) {
                        let idx0: u32 = u32(offset + i0);
                        let idx1: u32 = u32(offset + i1);
                        if (compare_greater(arr0[idx0], arr0[idx1])) {
                            let kv = arr0[idx0];
                            arr0[idx0] = arr0[idx1];
                            arr0[idx1] = kv;
                        }
                    }
                }
            }
        }
    }
}

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

/// Test for batcher_odd_even_mergesort().
@compute @workgroup_size(64)
fn test_batcher_odd_even_mergesort(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let tid = global_invocation_id.x;
    if (tid >= 64) {
        return;
    }

    // 64 threads copy all items into arr0
    let total_num_items = u32(sort_buffer.count);
    let num_items_per_thread = (total_num_items + 63u) >> 6u;
    let base_item = tid * num_items_per_thread;
    let num_items = min(num_items_per_thread, total_num_items - base_item);
    for (var i: u32 = base_item; i < base_item + num_items; i += 1u) {
        arr0[i] = sort_buffer.pairs[i];
    }

    workgroupBarrier();

    // 64 threads block-sort all items
    batcher_odd_even_mergesort(base_item, num_items);

    workgroupBarrier();

    // 64 threads copy all items back into sort_buffer
    for (var i: u32 = base_item; i < base_item + num_items; i += 1u) {
        sort_buffer.pairs[i] = arr0[i];
    }
}

/// Test for calc_merge_path().
@compute @workgroup_size(64)
fn test_calc_merge_path(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let tid = global_invocation_id.x;
    if (tid >= 64u) {
        return;
    }

    // 64 threads copy all items into arr0 and arr1
    let total_num_items = u32(sort_buffer.count) >> 1u;
    let num_items_per_thread = (total_num_items + 63u) >> 6u;
    let base_item = tid * num_items_per_thread;
    let num_items = min(num_items_per_thread, total_num_items - base_item);
    for (var i: u32 = base_item; i < base_item + num_items; i += 1u) {
        // First half -> arr0
        arr0[i] = sort_buffer.pairs[i];
        // Second half -> arr1
        arr1[i] = sort_buffer.pairs[total_num_items + i];
    }

    workgroupBarrier();

    // Each of the 64 threads calculates one merge path of length (blockSize / 64).
    let path_len = u32(blockSize) / 64u;

    // The total path length is the sum of the lengths of the two lists, which here
    // because they're of the same size is (total_num_items * 2u).
    let total_path_len = total_num_items * 2u;

    // The number of paths is the ratio
    let num_paths = (total_path_len + path_len - 1) / path_len;

    // Calculate the merge path for each section
    if (tid < num_paths) {
        let diag = min((tid + 1u) * path_len, total_num_items);
        var a_i = 0xFFFFFFFFu;
        let num_a = total_num_items;
        let num_b = num_a;  // merging 2 lists of same size
        a_i = calc_merge_path(num_a, num_b, diag);

        // Copy results into sort_buffer
        sort_buffer.pairs[tid].key = diag;
        sort_buffer.pairs[tid].value = a_i;
    }
}

#endif

// GPU sorts
// https://linebender.org/wiki/gpu/sorting/
// https://moderngpu.github.io/mergesort.html
// https://moderngpu.github.io/mergesort.html#blocksort
// https://moderngpu.github.io/segsort.html

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
