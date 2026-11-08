// Display-only recolouring of a parameter set (the "vector view" and the fit animation): the same Gaussians
// (positions, shapes and weights straight from the optimiser) with each colour replaced by an independent random
// colour per Gaussian (0.15 + 0.85 * rand per channel, as in the research code's Gaussian-id figures) times the
// Gaussian's own luma, so the random colours still carry the image's structure. `amount` mixes it in from plain
// luma (0 = grey), sizes are scaled by exp(size_log) so the Gaussians can grow in, and the true colour is restored
// left of `scan_u` during the reveal. The fit itself never reads `dst`.

struct Fx {
  count: u32,
  seed: u32,
  amount: f32,     // 0: luma only (grey) .. 1: random colour x luma
  size_log: f32,   // added to both log_s (negative = smaller)
  scan_u: f32,     // Gaussians with uv.x < scan_u keep their true colour (uv units: long side = 1)
  _p0: f32,
  _p1: f32,
  _p2: f32,
};

@group(0) @binding(0) var<uniform> fx: Fx;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

const GAIN: f32 = 1.6; // the mean random channel is 0.575; this keeps the brightness of the image

fn hash01(x: u32) -> f32 {
  var h = x * 0x9e3779b9u;
  h ^= h >> 15u;
  h *= 0x85ebca6bu;
  h ^= h >> 13u;
  h *= 0xc2b2ae35u;
  h ^= h >> 16u;
  return f32(h) / 4294967296.0;
}

@compute @workgroup_size(256)
fn recolour(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
            @builtin(local_invocation_index) li: u32) {
  let i = linear_id(wid, nwg, li, 256u);
  if (i >= fx.count) {
    return;
  }
  let b = i * FPG;
  let ux = src[b];
  dst[b] = ux;
  dst[b + 1u] = src[b + 1u];
  dst[b + 2u] = src[b + 2u] + fx.size_log;
  dst[b + 3u] = src[b + 3u] + fx.size_log;
  dst[b + 4u] = src[b + 4u];
  let c = vec3<f32>(src[b + 5u], src[b + 6u], src[b + 7u]);
  var o = c;
  if (ux >= fx.scan_u) {
    let luma = dot(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.299, 0.587, 0.114));
    let k = i * 3u + fx.seed;
    let rnd = 0.15 + 0.85 * vec3<f32>(hash01(k), hash01(k + 1u), hash01(k + 2u));
    o = mix(vec3<f32>(luma), rnd * luma * GAIN, fx.amount);
  }
  dst[b + 5u] = o.x;
  dst[b + 6u] = o.y;
  dst[b + 7u] = o.z;
  dst[b + 8u] = src[b + 8u];
}
