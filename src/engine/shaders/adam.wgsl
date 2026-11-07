// Adam step (torch.optim.Adam semantics) on the flat parameter array, one thread per float, followed by the clamps.
//   m = b1 m + (1 - b1) g;  v = b2 v + (1 - b2) g^2
//   p -= (lr / bc1) * m / (sqrt(v) / sqrt(bc2) + eps),  bc_k = 1 - b_k^t
// The host writes lr (already multiplied by the cosine schedule), bc1, sqrt(bc2), and the clamp bounds per float
// index k in 0..9: uv, uv, log_s, log_s, rot, r, g, b, log_w.

struct AdamParams {
  lr: array<vec4<f32>, 3>,
  lo: array<vec4<f32>, 3>,
  hi: array<vec4<f32>, 3>,
  bc1: f32,
  bc2_sqrt: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  n: u32,
  _p: vec2<u32>,
};

@group(0) @binding(0) var<uniform> ap: AdamParams;
@group(0) @binding(1) var<storage, read_write> params: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(256)
fn adam(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let i = linear_id(wid, nwg, li, 256u);
  if (i >= ap.n) {
    return;
  }
  let k = i % FPG;
  var g = grad[i];
  if (non_finite(g)) {
    g = 0.0;
  }
  let mi = ap.beta1 * m[i] + (1.0 - ap.beta1) * g;
  let vi = ap.beta2 * v[i] + (1.0 - ap.beta2) * g * g;
  m[i] = mi;
  v[i] = vi;
  let lr = ap.lr[k / 4u][k % 4u];
  let step = (lr / ap.bc1) * mi / (sqrt(vi) / ap.bc2_sqrt + ap.eps);
  let p = params[i] - step;
  params[i] = clamp(p, ap.lo[k / 4u][k % 4u], ap.hi[k / 4u][k % 4u]);
}
