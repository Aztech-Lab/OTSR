// Forward rasteriser: one workgroup per 16 x 16 tile, one thread per output pixel. The tile's Gaussians are streamed
// through workgroup memory in batches of 256. Writes (r, g, b, D) per pixel with D = sum w alpha + EPS.

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> splats: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tile_start: array<u32>;
@group(0) @binding(3) var<storage, read> tile_list: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

var<workgroup> s0: array<vec4<f32>, 256>;  // mx, my, a, b
var<workgroup> s1: array<vec4<f32>, 256>;  // d, w, -, -
var<workgroup> s2: array<vec4<f32>, 256>;  // r, g, b, -

@compute @workgroup_size(16, 16)
fn forward(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(local_invocation_index) li: u32) {
  let t = wid.y * view.tiles.x + wid.x;
  let pix = wid.xy * TILE + lid.xy;
  let inside = pix.x < view.out_size.x && pix.y < view.out_size.y;
  let p = vec2<f32>(pix);
  let start = min(tile_start[t], view.entry_cap);
  let end = min(tile_start[t + 1u], view.entry_cap);
  var num = vec3<f32>(0.0);
  var den = 0.0;
  for (var base = start; base < end; base += 256u) {
    let j = base + li;
    if (j < end) {
      let i = tile_list[j] * 3u;
      s0[li] = splats[i];
      s1[li] = splats[i + 1u];
      s2[li] = splats[i + 2u];
    }
    workgroupBarrier();
    let n = min(end - base, 256u);
    if (inside) {
      for (var k = 0u; k < n; k++) {
        let g = s0[k];
        let dx = g.x - p.x;
        let dy = g.y - p.y;
        let q = 0.5 * (g.z * dx * dx + s1[k].x * dy * dy) + g.w * dx * dy;
        if (q < Q_CUT) {
          let u = s1[k].y * (exp(-q) - ALPHA_SHIFT);
          num += u * s2[k].xyz;
          den += u;
        }
      }
    }
    workgroupBarrier();
  }
  if (inside) {
    out[pix.y * view.out_size.x + pix.x] = vec4<f32>(num / (den + EPS), den + EPS);
  }
}
