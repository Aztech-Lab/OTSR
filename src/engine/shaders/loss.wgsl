// Per-workgroup partial sums of the squared error: (clamped output, raw output). The host sums the partials.

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> out: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gt: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> partial: array<vec2<f32>>;

var<workgroup> red: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn loss(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let wg = wid.y * nwg.x + wid.x;
  let i = wg * 256u + li;
  let n = view.out_size.x * view.out_size.y;
  var v = vec2<f32>(0.0);
  if (i < n) {
    let o = out[i].xyz;
    let g = gt[i].xyz;
    let dr = o - g;
    let dc = clamp(o, vec3<f32>(0.0), vec3<f32>(1.0)) - g;
    v = vec2<f32>(dot(dc, dc), dot(dr, dr));
  }
  red[li] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) {
      red[li] = red[li] + red[li + s];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    partial[wg] = red[0];
  }
}
