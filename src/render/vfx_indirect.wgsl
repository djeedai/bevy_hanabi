
struct RenderIndirectBuffer {
    vertex_count: u32,
    instance_count: atomic<u32>,
    base_index: u32,
    vertex_offset: i32,
    base_instance: u32,
    alive_count: atomic<u32>,
    dead_count: atomic<u32>,
    __pad: u32,
};

struct DispatchIndirectBuffer {
    x: u32,
    y: u32,
    z: u32,
    __pad: u32,
};

@group(0) @binding(0) var<storage, read> render_indirect : RenderIndirectBuffer;
@group(0) @binding(1) var<storage, read_write> dispatch_indirect : DispatchIndirectBuffer;

/// Calculate the indirect workgroups counts based on the number of particles alive.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let index = global_invocation_id.x;
    if (index >= 1u) { // FIXME - batch all effects together in a single dispatch
        return;
    }

    let alive_count = atomicLoad(&render_indirect.alive_count);
    dispatch_indirect.x = (alive_count + 63u) / 64u;
}
