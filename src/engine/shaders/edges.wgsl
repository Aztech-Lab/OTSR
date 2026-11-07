// v3 distance-field edges (reference/otsr_ref/edges.py edge_field) in three passes over an S x render:
//   edge_h: per pixel, the luma maximum and minimum (value + pixel index) over the horizontal window [x - r, x + r]
//   edge_v: the same over the vertical window of the edge_h results = the (2r + 1)^2 window; writes
//           (imax, imin, t, range) with t = (Y - Ymin) / (Ymax - Ymin + 1e-6)
//   edge_final: w = 1/2 edge_px |grad t| (central differences, zero on the border), alpha = smoothstep(1/2 - w, 1/2 + w, t),
//           sharp = D + (B - D) alpha with D, B the colours at imin, imax; out = img + gate(range) (sharp - img)
// Ties pick the first pixel in row-major order, as torch.max_pool2d does.

struct EdgeParams {
  width: u32,
  height: u32,
  r: u32,
  _p0: u32,
  edge_px: f32,
  c0: f32,
  c1: f32,
  _p1: f32,
};

@group(0) @binding(0) var<uniform> ep: EdgeParams;
@group(0) @binding(1) var<storage, read> img: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tmp_in: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> tmp_out: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;

fn luma(i: u32) -> f32 {
  let c = img[i].xyz;
  return 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
}

fn smoothstep1(e0: f32, e1: f32, x: f32) -> f32 {
  let u = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return u * u * (3.0 - 2.0 * u);
}

@compute @workgroup_size(256)
fn edge_h(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
          @builtin(local_invocation_index) li: u32) {
  let i = linear_id(wid, nwg, li, 256u);
  if (i >= ep.width * ep.height) {
    return;
  }
  let x = i % ep.width;
  let y = i / ep.width;
  let x0 = select(x - ep.r, 0u, x < ep.r);
  let x1 = min(x + ep.r, ep.width - 1u);
  var ymax = -1.0;
  var imax = 0u;
  var ymin = 2.0;
  var imin = 0u;
  for (var xx = x0; xx <= x1; xx++) {
    let j = y * ep.width + xx;
    let v = luma(j);
    if (v > ymax) {
      ymax = v;
      imax = j;
    }
    if (v < ymin) {
      ymin = v;
      imin = j;
    }
  }
  tmp_out[i] = vec4<u32>(bitcast<u32>(ymax), imax, bitcast<u32>(ymin), imin);
}

@compute @workgroup_size(256)
fn edge_v(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
          @builtin(local_invocation_index) li: u32) {
  let i = linear_id(wid, nwg, li, 256u);
  if (i >= ep.width * ep.height) {
    return;
  }
  let x = i % ep.width;
  let y = i / ep.width;
  let y0 = select(y - ep.r, 0u, y < ep.r);
  let y1 = min(y + ep.r, ep.height - 1u);
  var ymax = -1.0;
  var imax = 0u;
  var ymin = 2.0;
  var imin = 0u;
  for (var yy = y0; yy <= y1; yy++) {
    let h = tmp_in[yy * ep.width + x];
    let vmax = bitcast<f32>(h.x);
    let vmin = bitcast<f32>(h.z);
    if (vmax > ymax) {
      ymax = vmax;
      imax = h.y;
    }
    if (vmin < ymin) {
      ymin = vmin;
      imin = h.w;
    }
  }
  let rng = ymax - ymin;
  let t = clamp((luma(i) - ymin) / (rng + 1e-6), 0.0, 1.0);
  tmp_out[i] = vec4<u32>(imax, imin, bitcast<u32>(t), bitcast<u32>(rng));
}

fn t_at(i: u32) -> f32 {
  return bitcast<f32>(tmp_in[i].z);
}

@compute @workgroup_size(256)
fn edge_final(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
              @builtin(local_invocation_index) li: u32) {
  let i = linear_id(wid, nwg, li, 256u);
  if (i >= ep.width * ep.height) {
    return;
  }
  let x = i % ep.width;
  let y = i / ep.width;
  let v = tmp_in[i];
  let t = bitcast<f32>(v.z);
  let rng = bitcast<f32>(v.w);
  var gx = 0.0;
  var gy = 0.0;
  if (x >= 1u && x + 1u < ep.width) {
    gx = (t_at(i + 1u) - t_at(i - 1u)) * 0.5;
  }
  if (y >= 1u && y + 1u < ep.height) {
    gy = (t_at(i + ep.width) - t_at(i - ep.width)) * 0.5;
  }
  let w = clamp(0.5 * ep.edge_px * (abs(gx) + abs(gy)), 1e-3, 0.5);
  let a = smoothstep1(0.5 - w, 0.5 + w, t);
  let B = img[v.x].xyz;
  let D = img[v.y].xyz;
  let sharp = D + (B - D) * a;
  let gate = smoothstep1(ep.c0, max(ep.c1, ep.c0 + 1e-6), rng);
  let c = img[i].xyz;
  out[i] = vec4<f32>(clamp(c + gate * (sharp - c), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
