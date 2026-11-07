// Log-domain Sinkhorn with a separable quadratic cost on an ny x nx grid over the box [0, ex] x [0, ey]:
// soft_min(pot)[a1, a2] = LSE_{b1, b2} [(pot[b1, b2] - Cy[a1, b1] - Cx[a2, b2]) / e] is computed as two 1-D
// log-sum-exps (lse_x over b2, then lse_y over b1). The same two kernels evaluate the barycentric projection
// with the optional log x / log y weights (wx, wy) and the affine output A * lse + B * base.

struct OtParams {
  nx: u32,
  ny: u32,
  wx: u32,       // 1: add log x_b2 inside lse_x (projection numerator for T_x)
  wy: u32,       // 1: add log y_b1 inside lse_y (projection numerator for T_y)
  ex: f32,
  ey: f32,
  e: f32,
  A: f32,        // lse_y writes A * lse + B * base
  B: f32,
  _p0: f32,
  _p1: f32,
  _p2: f32,
};

@group(0) @binding(0) var<uniform> op: OtParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> base: array<f32>;

const NEG_BIG: f32 = -3.0e38;

// inner[b1, a2] = LSE_{b2} ((src[b1, b2] - (x_a2 - x_b2)^2) / e + wx log x_b2)
@compute @workgroup_size(256)
fn lse_x(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
         @builtin(local_invocation_index) li: u32) {
  let i = linear_id(wid, nwg, li, 256u);
  if (i >= op.nx * op.ny) {
    return;
  }
  let b1 = i / op.nx;
  let a2 = i % op.nx;
  let xa = (f32(a2) + 0.5) / f32(op.nx) * op.ex;
  var m = NEG_BIG;
  var s = 0.0;
  for (var b2 = 0u; b2 < op.nx; b2++) {
    let xb = (f32(b2) + 0.5) / f32(op.nx) * op.ex;
    let d = xa - xb;
    var v = (src[b1 * op.nx + b2] - d * d) / op.e;
    if (op.wx == 1u) {
      v += log(xb);
    }
    if (v > m) {
      s = s * exp(m - v) + 1.0;
      m = v;
    } else {
      s += exp(v - m);
    }
  }
  dst[i] = m + log(s);
}

// dst[a1, a2] = A * LSE_{b1} (src[b1, a2] - (y_a1 - y_b1)^2 / e + wy log y_b1) + B * base[a1, a2]
@compute @workgroup_size(256)
fn lse_y(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
         @builtin(local_invocation_index) li: u32) {
  let i = linear_id(wid, nwg, li, 256u);
  if (i >= op.nx * op.ny) {
    return;
  }
  let a1 = i / op.nx;
  let a2 = i % op.nx;
  let ya = (f32(a1) + 0.5) / f32(op.ny) * op.ey;
  var m = NEG_BIG;
  var s = 0.0;
  for (var b1 = 0u; b1 < op.ny; b1++) {
    let yb = (f32(b1) + 0.5) / f32(op.ny) * op.ey;
    let d = ya - yb;
    var v = src[b1 * op.nx + a2] - d * d / op.e;
    if (op.wy == 1u) {
      v += log(yb);
    }
    if (v > m) {
      s = s * exp(m - v) + 1.0;
      m = v;
    } else {
      s += exp(v - m);
    }
  }
  dst[i] = op.A * (m + log(s)) + op.B * base[i];
}
