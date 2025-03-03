/// Arguments for a block operation on a buffer.
struct BufferOperationArgs {
    /// Offset, as u32 count, where the operation starts in the source buffer.
    src_offset: u32,
    /// Stride, as u32 count, between elements in the source buffer.
    src_stride: u32,
    /// Offset, as u32 count, where the operation starts in the destination buffer.
    dst_offset: u32,
    /// Stride, as u32 count, between elements in the destination buffer.
    dst_stride: u32,
    /// Number of u32 elements to process for this operation.
    count: u32,
}

@group(0) @binding(0) var<uniform> args : BufferOperationArgs;
@group(0) @binding(1) var<storage, read> src_buffer : array<u32>;
@group(0) @binding(2) var<storage, read_write> dst_buffer : array<u32>;

/// Clear a buffer to zero. Each thread clears a single u32.
@compute @workgroup_size(64)
fn zero_buffer(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;
    if (thread_index >= args.count) {
        return;
    }
    let dst = args.dst_offset + thread_index;
    dst_buffer[dst] = 0u;
}

/// Copy a source buffer into a destination buffer. Each thread copies a single u32.
@compute @workgroup_size(64)
fn copy_buffer(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;
    if (thread_index >= args.count) {
        return;
    }
    let src = args.src_offset + thread_index * args.src_stride;
    let dst = args.dst_offset + thread_index * args.dst_stride;
    let value = src_buffer[src];
    dst_buffer[dst] = value;
}

/// Calculate the number of workgroups to dispatch for a given number of threads.
fn calc_workground_count(thread_count: u32) -> u32 {
    // We assume a workgroup size of 64. This is currently the case everywhere in Hanabi,
    // but would need to be adapted if we decided to vary the workgroup size.
    let workgroup_count = (thread_count + 63u) >> 6u;
    return workgroup_count;
}

/// Fill indirect dispatch arguments from a raw element count, by copying the element count
/// and rounding it up to the number of thread per workgroup. Each thread copies a single u32.
@compute @workgroup_size(64)
fn fill_dispatch_args(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;
    if (thread_index >= args.count) {
        return;
    }

    let src = args.src_offset + thread_index * args.src_stride;
    let dst = args.dst_offset + thread_index * args.dst_stride;
    let thread_count = src_buffer[src];
    let workgroup_count = calc_workground_count(thread_count);
    dst_buffer[dst] = workgroup_count;
    dst_buffer[dst + 1u] = 1u;
    dst_buffer[dst + 2u] = 1u;
}

/// Fill indirect dispatch arguments from a raw element count, by copying the element count
/// and rounding it up to the number of thread per workgroup. Each thread copies a single u32.
/// Specialized variant for the indirect init pass, where the destination offset is computed
/// via an indirection read from the source buffer.
@compute @workgroup_size(64)
fn init_fill_dispatch_args(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;
    if (thread_index >= args.count) {
        return;
    }

    // Note: we ignore both src_offset and dst_offset, as we:
    // - bind the proper view of the ChildInfo array (src), which otherwise doesn't have a constant stride
    // - bind the entire init indirect dispatch buffer (dst) and retrieve its offset here in the shader
    
    // Retrieve the source ChildInfo array buffer
    let src = thread_index * args.src_stride;
    let dispatch_indirect_index = src_buffer[src];
    let event_count = src_buffer[src + 1u];

    // Note: we always assume 12-byte dispatch indirect structs, which is the canonical struct
    // with its x/y/z values, without any extra field or padding.
    let dst = dispatch_indirect_index * 3u;
    let workgroup_count = calc_workground_count(event_count);
    dst_buffer[dst] = workgroup_count;
    dst_buffer[dst + 1u] = 1u;
    dst_buffer[dst + 2u] = 1u;
}

/// Fill indirect dispatch arguments from a raw element count, by copying the element count
/// and rounding it up to the number of thread per workgroup. Each thread copies a single
/// single u32. Same as fill_dispatch_args(), but both read and write are from the destination
/// buffer. The source buffer is ignored (no binding).
@compute @workgroup_size(64)
fn fill_dispatch_args_self(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;
    if (thread_index >= args.count) {
        return;
    }
    let src = args.src_offset + thread_index * args.src_stride;
    // Note: we always assume 12-byte dispatch indirect structs, which is the canonical struct
    // with its x/y/z values, without any extra field or padding.
    let dst = args.dst_offset + thread_index * 3u;
    // Note: only the dst_buffer is used
    let thread_count = dst_buffer[src];
    let workgroup_count = calc_workground_count(thread_count);
    dst_buffer[dst] = workgroup_count;
    dst_buffer[dst + 1u] = 1u;
    dst_buffer[dst + 2u] = 1u;
    // leave last entry untouched; sometimes it's used for unrelated things
}
