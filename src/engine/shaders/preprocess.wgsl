// Per Gaussian: centre and conic in output pixels, the tile rectangle of its 4-sigma box, and the tile counts.

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> splats: array<vec4<f32>>;  // 3 per Gaussian: (mx, my, a, b), (d, w, -, -), (rgb, -)
@group(0) @binding(3) var<storage, read_write> rects: array<vec2<u32>>;   // tiles (x0 | x1 << 16, y0 | y1 << 16)
@group(0) @binding(4) var<storage, read_write> gcount: array<u32>;        // tiles covered by each Gaussian
@group(0) @binding(5) var<storage, read_write> tile_count: array<atomic<u32>>;

@compute @workgroup_size(256)
fn preprocess(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
              @builtin(local_invocation_index) li: u32) {
  let i = linear_id(wid, nwg, li, 256u);
  if (i >= view.count) {
    return;
  }
  let b = i * FPG;
  let z = view.zoom;
  let m = vec2<f32>(params[b], params[b + 1u]) * view.src_size * z + view.offset - 0.5;
  let s = exp(vec2<f32>(params[b + 2u], params[b + 3u])) * (view.long_side * z);
  let th = params[b + 4u];
  let c = cos(th);
  let sn = sin(th);
  let s2 = s * s;
  // covariance in output pixels: R diag(s^2) R^T with R = [[c, sn], [-sn, c]]
  var sxx = c * c * s2.x + sn * sn * s2.y;
  var syy = sn * sn * s2.x + c * c * s2.y;
  let sxy = c * sn * (s2.y - s2.x);
  var w = exp(params[b + 8u]);
  var conic: vec3<f32>;
  if (view.aa > 0.0) {
    // display prefilter: Sigma + aa^2 I, weight scaled by sqrt(det0 / det1) so each Gaussian keeps its integral
    let det0 = s2.x * s2.y;
    let a2 = view.aa * view.aa;
    sxx += a2;
    syy += a2;
    let det1 = sxx * syy - sxy * sxy;
    w *= sqrt(det0 / det1);
    conic = vec3<f32>(syy, -sxy, sxx) / det1;
  } else {
    let i1 = 1.0 / s2.x;
    let i2 = 1.0 / s2.y;
    conic = vec3<f32>(c * c * i1 + sn * sn * i2, c * sn * (i2 - i1), sn * sn * i1 + c * c * i2);
  }
  splats[3u * i] = vec4<f32>(m, conic.x, conic.y);
  splats[3u * i + 1u] = vec4<f32>(conic.z, w, 0.0, 0.0);
  splats[3u * i + 2u] = vec4<f32>(params[b + 5u], params[b + 6u], params[b + 7u], 0.0);

  // pixel centres that can lie inside the 4-sigma ellipse: |dx| <= K sqrt(Sigma_xx), |dy| <= K sqrt(Sigma_yy)
  let r = K_SIGMA * sqrt(vec2<f32>(sxx, syy));
  let lo = max(ceil(m - r), vec2<f32>(0.0));
  let hi = min(floor(m + r), vec2<f32>(view.out_size) - 1.0);
  let bad = non_finite(m.x) || non_finite(m.y) || non_finite(r.x) || non_finite(r.y) || non_finite(w) ||
            non_finite(conic.x) || non_finite(conic.y) || non_finite(conic.z);
  if (bad || lo.x > hi.x || lo.y > hi.y) {
    gcount[i] = 0u;
    rects[i] = vec2<u32>(0u);
    return;
  }
  let t0 = vec2<u32>(lo) / TILE;
  let t1 = vec2<u32>(hi) / TILE;
  gcount[i] = (t1.x - t0.x + 1u) * (t1.y - t0.y + 1u);
  rects[i] = vec2<u32>(t0.x | (t1.x << 16u), t0.y | (t1.y << 16u));
  for (var ty = t0.y; ty <= t1.y; ty++) {
    for (var tx = t0.x; tx <= t1.x; tx++) {
      atomicAdd(&tile_count[ty * view.tiles.x + tx], 1u);
    }
  }
}
