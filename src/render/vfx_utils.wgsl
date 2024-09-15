/// Arguments for a block operation on a buffer.
struct BufferOperationArgs {
    /// Offset, as u32 count, where the operation starts in the source buffer.
    src_offset: u32,
    /// Stride, as u32 count, between elements in the source buffer.
    src_stride: u32,
    /// Offset, as u32 count, where the operation starts in the destination buffer.
    dst_offset: u32,
    /// Number of u32 elements to process for this operation.
    count: u32,
}

@group(0) @binding(0) var<uniform> args : BufferOperationArgs;
@group(0) @binding(1) var<storage, read> src_buffer : array<u32>;
@group(0) @binding(2) var<storage, read_write> dst_buffer : array<u32>;

/// Clear a buffer to zero. Each thread clears a single u32.
@compute @workgroup_size(64)
fn zero_buffer(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let index = global_invocation_id.x;
    if (index >= args.count) {
        return;
    }
    let dst = args.dst_offset + index;
    dst_buffer[dst] = 0u;
}

/// Copy a source buffer into a destination buffer. Each thread copies a single u32.
@compute @workgroup_size(64)
fn copy_buffer(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let index = global_invocation_id.x;
    if (index >= args.count) {
        return;
    }
    let src = args.src_offset + index * args.src_stride;
    let dst = args.dst_offset + index;
    let value = src_buffer[src];
    dst_buffer[dst] = value;
}

/// Fill indirect dispatch arguments from a raw element count, by copying the element count
/// and rounding it up to the number of thread per workgroup. Each thread copies a single
/// argument struct.
@compute @workgroup_size(64)
fn fill_dispatch_args(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let index = global_invocation_id.x;
    if (index >= args.count) {
        return;
    }
    let src = args.src_offset + index * args.src_stride;
    let dst = args.dst_offset + index * 16u;
    let value = src_buffer[src];
    let workgroup_count = (value + 63u) >> 6u;
    dst_buffer[dst] = workgroup_count;
    dst_buffer[dst + 1u] = 1u;
    dst_buffer[dst + 2u] = 1u;
    // leave last entry untouched
}
