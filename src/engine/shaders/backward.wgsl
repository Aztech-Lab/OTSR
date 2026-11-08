// Backward pass, part 1: per tile, the gradient of the loss with respect to each listed Gaussian's screen-space
// quantities (centre, conic, colour, weight), summed over the tile's pixels. One workgroup per tile; the tile's
// per-pixel g = dL/dout / D and out are staged in workgroup memory, then P threads share one Gaussian (P chosen so
// that the 256 threads are busy) and their partial sums are reduced in workgroup memory. The result is written per
// tile-list slot (3 x vec4 per slot), so the per-Gaussian sum in reduce_grad.wgsl is deterministic: no atomics.
//
// With u = w alpha(q), D = sum u + EPS, out = sum u c / D and G = dL/dout:
//   dL/dc    = G u / D                 dL/du = q_i = sum_ch G_ch (c_ch - out_ch) / D
//   dL/dw    = sum q_i alpha           dL/dq = -q_i w exp(-q)      (alpha = exp(-q) - ALPHA_SHIFT for q < Q_CUT)
//   dq/dm    = (a dx + b dy, b dx + d dy)   dq/d(a, b, d) = (dx^2 / 2, dx dy, dy^2 / 2)   with dx = m - p

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> splats: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tile_start: array<u32>;
@group(0) @binding(3) var<storage, read> tile_list: array<u32>;
@group(0) @binding(4) var<storage, read> out: array<vec4<f32>>;   // r, g, b, D
@group(0) @binding(5) var<storage, read> gt: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> partial: array<vec4<f32>>;  // 3 per slot: (dmx, dmy, da, db), (dd, dcr, dcg, dcb), (dw, 0, 0, 0)

var<workgroup> pg: array<vec4<f32>, 256>;   // G / D per pixel
var<workgroup> po: array<vec4<f32>, 256>;   // out per pixel
var<workgroup> red: array<vec4<f32>, 256>;

fn reduce_lanes(v: vec4<f32>, li: u32, lane: u32, p: u32) -> vec4<f32> {
  workgroupBarrier();
  red[li] = v;
  workgroupBarrier();
  for (var s = p >> 1u; s > 0u; s >>= 1u) {
    if (lane < s) {
      red[li] = red[li] + red[li + s];
    }
    workgroupBarrier();
  }
  return red[li];
}

@compute @workgroup_size(256)
fn backward(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wid.y * view.tiles.x + wid.x;
  let org = wid.xy * TILE;
  let pix = org + vec2<u32>(li % TILE, li / TILE);
  if (pix.x < view.out_size.x && pix.y < view.out_size.y) {
    let idx = pix.y * view.out_size.x + pix.x;
    let o = out[idx];
    let d = o.xyz - gt[idx].xyz;
    var G = 2.0 * d * view.grad_scale;
    if (view.loss_kind == 1u) {
      G = d / sqrt(d * d + vec3<f32>(1e-6)) * view.grad_scale;
    }
    pg[li] = vec4<f32>(G / o.w, 0.0);
    po[li] = vec4<f32>(o.xyz, 0.0);
  } else {
    pg[li] = vec4<f32>(0.0);
    po[li] = vec4<f32>(0.0);
  }
  workgroupBarrier();
  let start = min(tile_start[t], view.entry_cap);
  let end = min(tile_start[t + 1u], view.entry_cap);
  let n = end - start;
  if (n == 0u) {
    return;
  }
  var P = 1u;                                   // threads per Gaussian: the largest power of two with n P <= 256
  while (P < 256u && n * (P * 2u) <= 256u) {
    P *= 2u;
  }
  let C = 256u / P;                             // Gaussians per chunk
  let gl = li / P;
  let lane = li % P;
  for (var base = start; base < end; base += C) {
    let slot = base + gl;
    var acc0 = vec4<f32>(0.0);                  // dmx, dmy, da, db
    var acc1 = vec4<f32>(0.0);                  // dd, dcr, dcg, dcb
    var accw = 0.0;
    if (slot < end) {
      let i = tile_list[slot] * 3u;
      let sa = splats[i];
      let sb = splats[i + 1u];
      let c = splats[i + 2u].xyz;
      let a = sa.z;
      let b = sa.w;
      let d = sb.x;
      let w = sb.y;
      for (var k = lane; k < 256u; k += P) {
        let p = vec2<f32>(org + vec2<u32>(k % TILE, k / TILE));
        let dx = sa.x - p.x;
        let dy = sa.y - p.y;
        let q = 0.5 * (a * dx * dx + d * dy * dy) + b * dx * dy;
        if (q < Q_CUT) {
          let e = exp(-q);
          let alpha = e - ALPHA_SHIFT;
          let u = w * alpha;
          let g = pg[k].xyz;
          let qi = dot(g, c - po[k].xyz);
          let dq = -qi * w * e;
          acc0 += vec4<f32>(dq * (a * dx + b * dy), dq * (b * dx + d * dy), dq * 0.5 * dx * dx, dq * dx * dy);
          acc1 += vec4<f32>(dq * 0.5 * dy * dy, g * u);
          accw += qi * alpha;
        }
      }
    }
    let r0 = reduce_lanes(acc0, li, lane, P);
    let r1 = reduce_lanes(acc1, li, lane, P);
    let r2 = reduce_lanes(vec4<f32>(accw, 0.0, 0.0, 0.0), li, lane, P);
    if (lane == 0u && slot < end) {
      partial[3u * slot] = r0;
      partial[3u * slot + 1u] = r1;
      partial[3u * slot + 2u] = r2;
    }
  }
}
