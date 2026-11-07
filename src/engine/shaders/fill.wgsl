// Per Gaussian: append its index to the list of every tile it covers (counting sort by tile) and record, in the
// Gaussian's own CSR range [goffset, goffset + gcount), the list slot of each of its entries (for the gradient
// reduction). Slots past the capacity are dropped and recorded as INVALID; the host grows the capacity.

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> rects: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> gcount: array<u32>;
@group(0) @binding(3) var<storage, read> goffset: array<u32>;
@group(0) @binding(4) var<storage, read> tile_start: array<u32>;
@group(0) @binding(5) var<storage, read_write> tile_cursor: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> tile_list: array<u32>;
@group(0) @binding(7) var<storage, read_write> entry_slot: array<u32>;

@compute @workgroup_size(256)
fn fill(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let i = linear_id(wid, nwg, li, 256u);
  if (i >= view.count) {
    return;
  }
  if (gcount[i] == 0u) {
    return;
  }
  let r = rects[i];
  let x0 = r.x & 0xffffu;
  let x1 = r.x >> 16u;
  let y0 = r.y & 0xffffu;
  let y1 = r.y >> 16u;
  var e = goffset[i];
  for (var ty = y0; ty <= y1; ty++) {
    for (var tx = x0; tx <= x1; tx++) {
      let t = ty * view.tiles.x + tx;
      let slot = tile_start[t] + atomicAdd(&tile_cursor[t], 1u);
      var keep = INVALID;
      if (slot < view.entry_cap) {
        tile_list[slot] = i;
        keep = slot;
      }
      if (e < view.entry_cap) {
        entry_slot[e] = keep;
      }
      e++;
    }
  }
}
