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

/// Number of items sorted per thread, serially.
const numItemPerThread: u32 = 16u;

/// Number of threads per workgroup (block).
const numThreads: u32 = 64u;

/// Size of a block of KeyValuePair in workgroup memory.
const blockSize: u32 = numThreads * numItemPerThread;  // 1024

// Workgroup has at least 16kB memory (max_compute_workgroup_storage_size).
// Note that Vulkan on Windows 11 report 16352, not 16384 (so, lower than
// the default in the wgpu docs).
var<workgroup> shared_pairs : array<KeyValuePair, blockSize>;  // 12 kB

/// Offset inside the sort buffer pairs of the source half to read unsorted pairs from.
var<private> src: u32;

/// Offset inside the sort buffer pairs of the destination half to write sorted pairs to.
var<private> dst: u32;

/// Sort locally inside a single block (workgroup), using workgroup-shared memory.
///
/// `tid` is the local thread ID inside the block (0..64). The `src` and `dst` offsets are
/// the starts of blockSize-length memory blocks inside the sort buffer pairs, to read from
/// and write to, respectively. `total_num_items` is the number of items in the block; this
/// is typically equal to blockSize, except for the last block. `src == dst` is valid, and
/// simply sort in-place inside the storage buffer.
///
/// https://moderngpu.github.io/mergesort.html#blocksort
/// https://moderngpu.github.io/mergesort.html#sortnetworks
fn block_sort(tid: u32, src: u32, dst: u32, total_num_items: u32) {
    
    var pairs: array<KeyValuePair, numItemPerThread>;

    // Each thread copies numItemPerThread into a local register array, then sort
    // that array serially inside the thread, before copying the sorted result
    // into shared workgroup memory for the block-level merge sort.
    let num_threads = (total_num_items + numItemPerThread - 1) / numItemPerThread;
    let begin = tid * numItemPerThread;
    let end = min(total_num_items, begin + numItemPerThread);
    if (tid < num_threads) {
        // Copy pairs into register array
        for (var i: u32 = begin; i < end; i += 1u) {
            pairs[i - begin] = sort_buffer.pairs[src + i];
        }

        // Sort array serially inside this thread
        let num = end - begin;
        batcher_odd_even_mergesort(&pairs, num);

        // Copy the data into shared workgroup memory
        for (var i: u32 = begin; i < end; i += 1u) {
            shared_pairs[i] = pairs[i - begin];
        }
    }
    
    // Wait for all threads to have sorted their own thread-local values, and written
    // them into their own slice of shared memory, before we read back shared memory below.
    workgroupBarrier();

    // shared_pairs[] contains a series of numItemPerThread-length sorted sub-arrays.
    // Merge-sort in parallel all those sub-arrays into increasibly larger sub-arrays
    // with increasngly more (= coop) threads collaborating to process each merge.
    for (var coop: u32 = 2u; coop <= numThreads; coop = coop * 2u) {
        let list = ~(coop - 1) & tid;
        let diag = min(total_num_items, numItemPerThread * ((coop - 1) & tid));
        let start = numItemPerThread * list;
        let a0 = min(total_num_items, start);
        let b0 = min(total_num_items, start + numItemPerThread * (coop / 2));
        let b1 = min(total_num_items, start + numItemPerThread * coop);

        let num_a = b0 - a0;
        let num_b = b1 - b0;
        let a_i = calc_merge_path(a0, b0, num_a, num_b, diag);

        // Merge the 2 sorted lists
        var ia = a0 + a_i;
        var ib = b0 + diag - a_i;
        for (var i: u32 = 0; i < numItemPerThread; i += 1u) {
            if ((ib >= b1) || ((ia < b0) && !compare_greater(shared_pairs[ia], shared_pairs[ib]))) {
                pairs[i] = shared_pairs[ia];
                ia += 1u;
            } else {
                pairs[i] = shared_pairs[ib];
                ib += 1u;
            }
        }

        // We're about to write back exactly numItemPerThread elements into the shared array.
        // However by design threads may read more values in either of the two merged lists,
        // so we need to wait for all co-op threads to finish reading before we can write back.
        workgroupBarrier();

        // Copy back into shared memory for next iteration
        for (var i: u32 = 0; i < numItemPerThread; i += 1u) {
            shared_pairs[begin + i] = pairs[i];
        }

        // Next iteration (or the final write) need to read back from shared memory, so we need
        // to wait for all threads to finish writing there.
        workgroupBarrier();
    }

    // Copy sorted items from shared workgroup memory back into sort_buffer
    if (tid < num_threads) {
        for (var i: u32 = begin; i < end; i += 1u) {
            sort_buffer.pairs[dst + i] = shared_pairs[i];
        }
    }
}

/// Calculate a single merge path for a subsequent parallel merge.
///
/// INPUTS:
/// - Values A : var<workgroup> shared_pairs[num_a]
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
/// https://moderngpu.github.io/bulkinsert.html#mergepath
fn calc_merge_path(offset_a: u32, offset_b: u32, num_a: u32, num_b: u32, diag: u32) -> u32 {
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
        if (compare_greater(shared_pairs[offset_a + mid], shared_pairs[offset_b + diag - 1u - mid])) {
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
/// - Values A : var<workgroup> shared_pairs[]
/// - Offset from the start of shared_pairs where the values to sort start : offset
/// - Number of values to sort : n
///
/// OUTPUTS:
/// - Values A in shared_pairs[], sorted
///
/// This is a sorting network. Not asymptotically optimal, but reasonably efficient.
/// It's more of a reference implementation, from 1998. Not the fastest by today's
/// standards. It's not stable.
///
/// https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
fn batcher_odd_even_mergesort(data: ptr<function, array<KeyValuePair, numItemPerThread>>, num: u32) {
    // Pad with elements which always compare greater, so they end up at the end
    // of the array after all real elements, and when truncated we get back the
    // sorted original array.
    for (var i: u32 = num; i < numItemPerThread; i += 1u) {
        data[i].key = 0xFFFFFFFFu;
#ifdef HAS_DUAL_KEY
        data[i].key2 = 0xFFFFFFFFu;
#endif
    }

    // Actual numItemPerThread-length sort
    for (var p: u32 = 1; p < numItemPerThread; p += p) {
        for (var k: u32 = p; k >= 1; k = k >> 1) {
            for (var j: u32 = k % p; j + k < numItemPerThread; j += 2 * k) {
                for (var i: u32 = 0; i < min(k, numItemPerThread - j - k); i += 1) {
                    let i0 = i + j;
                    let i1 = i + j + k;
                    let f0 = f32(i0) / f32(2 * p);
                    let f1 = f32(i1) / f32(2 * p);
                    // i0 < i1 < numItemPerThread - 1
                    if (floor(f0) == floor(f1)) {
                        let idx0 = i0;
                        let idx1 = i1;
                        if (compare_greater(data[idx0], data[idx1])) {
                            let kv = data[idx0];
                            data[idx0] = data[idx1];
                            data[idx1] = kv;
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
        let base_particle = shared_pairs[mid].key;
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

/// Get the number N such that log2(value) == N rounded up.
fn find_log2(value: u32) -> u32 {
    var num = 31u - countLeadingZeros(value);
    if ((value & (value - 1u)) != 0u) {  // not a power of 2
        num += 1u;
    }
    return num;
}

/// Parallel merge-sort combining a block-level parallel sort followed by a serial mergesort.
fn block_parallel_merge_sort(thread_id: u32, block_id: u32) {
    let total_num_items = u32(sort_buffer.count);

    // Loop over all blocks and block-sort each serially (TODO - parallelize this too)
    let num_blocks = (total_num_items + blockSize - 1) / blockSize;
    {
        let offset = block_id * blockSize;
        let count = min(offset + blockSize, total_num_items) - offset;  // <= blockSize
        block_sort(thread_id, offset, offset, count);
    }

    // Wait for all threads to write per-block sorted lists into the storage sort buffer
    storageBarrier();
    workgroupBarrier();

    // Recursively merge the blockSize-length sorted lists into a single globally sorted one.
    // FIXME - parallelize this...
    if (thread_id == 0 && block_id == 0) {
        src = 0u;
        dst = total_num_items;
        merge_lists_serial(0u, blockSize, blockSize, blockSize, 0u, total_num_items);

        // for (var i: u32 = 0; i < total_num_items; i += 1u) {
        //     sort_buffer.pairs[total_num_items + i] = sort_buffer.pairs[i];
        // }

        // src = 0u;
        // dst = total_num_items;
        // var left = num_blocks;
        // while (left > 1u) {
        //     var step = 1u;
        //     while (step < left) {
        //         for (var i: u32 = 0u; i + step < left; i += step * 2u) {
        //             merge_lists_serial(i, step, i + step, step, src, dst);
        //         }
        //         step <<= 1u;

        //         //storageBarrier();

        //         // Swap source/destination lists for next iteration
        //         let tmp = src;
        //         src = dst;
        //         dst = tmp;
        //     }
        //     left >>= 1u;
        // }
    }
}

/// Sort each block in parallel on a separate workgroup.
@compute @workgroup_size(64)
fn parallel_block_sort(@builtin(local_invocation_index) thread_id: u32, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let block_id = workgroup_id.x;  // wgpu doesn't support @builtin(workgroup_index)
    let total_num_items = u32(sort_buffer.count);
    
    // Sort each block independently
    let offset = block_id * blockSize;
    let block_num = min(offset + blockSize, total_num_items) - offset;
    let block_src = src + offset;
    let block_dst = dst + offset;
    block_sort(thread_id, block_src, block_dst, block_num);
}

fn parallel_merge_sort(thread_id: u32, block_id: u32) {
    let total_num_items = u32(sort_buffer.count);
    let num_blocks = (total_num_items + blockSize - 1) / blockSize;
    let num_passes = find_log2(num_blocks);

    // Sort each block independently
    {
        let offset = block_id * blockSize;
        let block_num = min(offset + blockSize, total_num_items) - offset;
        let block_src = src + offset;
        let block_dst = dst + offset;
        block_sort(thread_id, block_src, block_dst, block_num);
    }

    // Merge-sort blocks into storage buffer by recusrively merging pairs of adacent sorted lists of increasing size
    src = 0u;
    dst = total_num_items;
    for (var ipass: u32 = 0u; ipass < num_passes; ipass += 1u) {
        let coop = 2u << ipass;

        let list = ~(coop - 1u) & block_id;
        let diag = min(total_num_items, blockSize * ((coop - 1u) & block_id));
        let start = blockSize * list;
        let a0 = min(total_num_items, start);
        let b0 = min(total_num_items, start + blockSize * (coop / 2u));
        let b1 = min(total_num_items, start + blockSize * coop);

        // TODO... calc merge path + do the ping-pong merge
    }
}

#ifdef TEST

/// Test for find_effect_from_particle().
@compute @workgroup_size(64)
fn test_find_effect_from_particle(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let tid = global_invocation_id.x;
    
    // Read test data from sort buffer
    let num_effects = u32(sort_buffer.count);
    let num_particles = arrayLength(&sort_buffer.pairs);
    if (tid < num_effects) {
        shared_pairs[tid] = sort_buffer.pairs[tid];
    }

    workgroupBarrier();

    // Execute the actual test
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
    let total_num_items = u32(sort_buffer.count);

    // Threads copy all items into a local register array
    var pairs: array<KeyValuePair, numItemPerThread>;
    let base_item = tid * numItemPerThread;
    let num_items = min(numItemPerThread, total_num_items - base_item);
    for (var i: u32 = base_item; i < base_item + num_items; i += 1u) {
        pairs[i - base_item] = sort_buffer.pairs[i];
    }

    workgroupBarrier();

    // Sort all items in the local array
    batcher_odd_even_mergesort(&pairs, numItemPerThread);

    workgroupBarrier();

    // Threads copy all items back into sort_buffer
    for (var i: u32 = base_item; i < base_item + num_items; i += 1u) {
        sort_buffer.pairs[i] = pairs[i - base_item];
    }
}

/// Test for calc_merge_path().
@compute @workgroup_size(64)
fn test_calc_merge_path(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let tid = global_invocation_id.x;

    // 64 threads copy all items into shared_pairs
    let total_num_items = u32(sort_buffer.count) >> 1u;
    let num_items_per_thread = (total_num_items + 63u) >> 6u;
    let base_item = tid * num_items_per_thread;
    let num_items = min(num_items_per_thread, total_num_items - base_item);
    for (var i: u32 = base_item; i < base_item + num_items; i += 1u) {
        // First half -> shared_pairs
        shared_pairs[i] = sort_buffer.pairs[i];
        // Second half -> shared_pairs
        shared_pairs[total_num_items + i] = sort_buffer.pairs[total_num_items + i];
    }

    workgroupBarrier();

    // The total path length is the sum of the lengths of the two lists, which here
    // because they're of the same size is (total_num_items * 2u).
    let total_path_len = total_num_items * 2u;

    // Divide the entire merge into one interval per thread. The last interval can
    // be shorter, which exercises the final partial merge tile.
    let path_len = (total_path_len + 63u) / 64u;
    let num_paths = (total_path_len + path_len - 1) / path_len;

    // Calculate the merge path for each section
    if (tid < num_paths) {
        let diag = min((tid + 1u) * path_len, total_path_len);
        let start_a = 0u;
        let num_a = total_num_items;
        let start_b = num_a;
        let num_b = num_a;  // merging 2 lists of same size
        let a_i = calc_merge_path(start_a, start_b, num_a, num_b, diag);

        // Copy results into sort_buffer
        sort_buffer.pairs[tid].key = diag;
        sort_buffer.pairs[tid].value = a_i;
    }
}

/// Test for block_sort().
@compute @workgroup_size(64)
fn test_block_sort(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let tid = global_invocation_id.x;
    let total_num_items = u32(sort_buffer.count);
    block_sort(tid, 0u, 0u, total_num_items);
}

/// Test for parallel_merge_sort().
@compute @workgroup_size(64)
fn test_block_parallel_merge_sort(@builtin(local_invocation_index) thread_id: u32, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let block_id = workgroup_id.x;  // wgpu doesn't support @builtin(workgroup_index)
    block_parallel_merge_sort(thread_id, block_id);
}

#endif

// GPU sorts
// https://linebender.org/wiki/gpu/sorting/
// https://moderngpu.github.io/mergesort.html
// https://moderngpu.github.io/mergesort.html#blocksort
// https://moderngpu.github.io/segsort.html

/// Naive insertion sort. TODO: replace with something faster.
// @compute @workgroup_size(64)
// fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
//     // Naive single-threaded sort
//     if (global_invocation_id.x != 0) {
//         return;
//     }

//     // Insertion sort
//     let num_items = sort_buffer.count;
//     for (var i: i32 = 1; i < num_items; i += 1) {
//         var kv = sort_buffer.pairs[i];
//         var j = i;
//         while (j > 0 && compare_greater(sort_buffer.pairs[j - 1], kv)) {
//             sort_buffer.pairs[j] = sort_buffer.pairs[j - 1];
//             j -= 1;
//         }
//         sort_buffer.pairs[j] = kv;
//     }

//     // Clear for next frame
//     sort_buffer.count = 0;
// }

/// Block-sort 1024 particles per workgroup. Larger inputs are merged by
/// vfx_sort_merge.wgsl in subsequent compute passes.
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) thread_id: u32, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let block_id = workgroup_id.x;  // wgpu doesn't support @builtin(workgroup_index)

    let total_num_items = u32(sort_buffer.count);

    let offset = block_id * blockSize;
    let count = min(offset + blockSize, total_num_items) - offset;  // <= blockSize
    block_sort(thread_id, offset, offset, count);
}
