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
    count: atomic<i32>,
    pairs: array<KeyValuePair>,
}

/// Particle buffer as an array of u32. This prevents having to specialize this shader
/// for every single particle layout.
struct RawParticleBuffer {
    data: array<u32>,
}

@group(0) @binding(0) var<storage, read_write> sort_buffer : SortBuffer;
@group(0) @binding(1) var<storage, read> particle_buffer : RawParticleBuffer;
@group(0) @binding(2) var<storage, read> indirect_index_buffer : array<u32>;
// Technically read-only, but the type contains atomic<> fields and wasm is strict about it
@group(0) @binding(3) var<storage, read_write> effect_metadata : EffectMetadata;

/// Fill the sorting key-value pair buffer with data to prepare for actual sorting.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let thread_index = global_invocation_id.x;
    let count = atomicLoad(&effect_metadata.instance_count); // TODO - atomic not needed
    if (thread_index >= count) {
        return;
    }

    let read_index = effect_metadata.ping;
    let particle_index = indirect_index_buffer[thread_index * 3u + read_index];

    let particle_offset = particle_index * effect_metadata.particle_stride;
    let key_offset = particle_offset + effect_metadata.sort_key_offset;
    let key2_offset = particle_offset + effect_metadata.sort_key2_offset;

    let pair_index = atomicAdd(&sort_buffer.count, 1);
    sort_buffer.pairs[pair_index].key = particle_buffer.data[key_offset];
    sort_buffer.pairs[pair_index].key2 = particle_buffer.data[key2_offset];
    sort_buffer.pairs[pair_index].value = particle_index;
}
