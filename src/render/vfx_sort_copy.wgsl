#import bevy_hanabi::vfx_common::EffectMetadata

/// Key-value pair for sorting, with optional second sort key.
struct KeyValuePair {
    /// Sorting key.
    key: u32,
    /// Secondary sorting key. Sorts value with the same primary key.
    key2: u32,
    /// Value associated with the sort key(s), generally an index to some other data.
    /// Copied as is and otherwise ignored by the sorting algorithm.
    value: u32,
}

struct SortBuffer {
    count: i32,
    pairs: array<KeyValuePair>,
}

struct IndirectIndexBuffer {
    data: array<u32>,
}

@group(0) @binding(0) var<storage, read_write> indirect_index_buffer : IndirectIndexBuffer;
@group(0) @binding(1) var<storage, read> sort_buffer : SortBuffer;
// Technically read-only, but the type contains atomic<> fields and wasm is strict about it
@group(0) @binding(2) var<storage, read_write> effect_metadata : EffectMetadata;

/// Copy the sorted particle indices back into the effect index buffer.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let row_index = global_invocation_id.x;
    let count = atomicLoad(&effect_metadata.instance_count); // TODO - atomic not needed
    if (row_index >= count) {
        return;
    }
    
    // Always write into ping, read from pong
    let write_index = effect_metadata.ping;

    let particle_index = sort_buffer.pairs[row_index].value;
    indirect_index_buffer.data[row_index * 3u + write_index] = particle_index;
}
