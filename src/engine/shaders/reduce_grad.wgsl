// Backward pass, part 2: per Gaussian, sum the partial gradients of its tile entries (in CSR order, deterministic)
// and apply the chain rule from screen-space quantities to the 9 parameters:
//   m = uv (W, H) zoom + offset - 1/2                      -> dL/duv = dL/dm (W, H) zoom
//   i_k = 1 / s_k^2, s_k = exp(log_s_k) L zoom             -> di_k / dlog_s_k = -2 i_k
//   a = c^2 i1 + s^2 i2, b = c s (i2 - i1), d = s^2 i1 + c^2 i2   (c = cos rot, s = sin rot)
//   dL/drot = (i2 - i1) (2 c s dL/da + (c^2 - s^2) dL/db - 2 c s dL/dd)
//   dL/dlog_w = dL/dw exp(log_w)

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read> gcount: array<u32>;
@group(0) @binding(3) var<storage, read> goffset: array<u32>;
@group(0) @binding(4) var<storage, read> entry_slot: array<u32>;
@group(0) @binding(5) var<storage, read> partial: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> grad: array<f32>;

@compute @workgroup_size(256)
fn reduce_grad(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_index) li: u32) {
  let i = linear_id(wid, nwg, li, 256u);
  if (i >= view.count) {
    return;
  }
  var s0 = vec4<f32>(0.0);
  var s1 = vec4<f32>(0.0);
  var sw = 0.0;
  let e0 = goffset[i];
  let e1 = min(e0 + gcount[i], view.entry_cap);
  for (var e = e0; e < e1; e++) {
    let slot = entry_slot[e];
    if (slot != INVALID) {
      s0 += partial[3u * slot];
      s1 += partial[3u * slot + 1u];
      sw += partial[3u * slot + 2u].x;
    }
  }
  let b = i * FPG;
  let s = exp(vec2<f32>(params[b + 2u], params[b + 3u])) * (view.long_side * view.zoom);
  let i1 = 1.0 / (s.x * s.x);
  let i2 = 1.0 / (s.y * s.y);
  let th = params[b + 4u];
  let c = cos(th);
  let sn = sin(th);
  let da = s0.z;
  let db = s0.w;
  let dd = s1.x;
  let di1 = da * c * c - db * c * sn + dd * sn * sn;
  let di2 = da * sn * sn + db * c * sn + dd * c * c;
  let duv = s0.xy * view.src_size * view.zoom;
  grad[b] = duv.x;
  grad[b + 1u] = duv.y;
  grad[b + 2u] = -2.0 * i1 * di1;
  grad[b + 3u] = -2.0 * i2 * di2;
  grad[b + 4u] = (i2 - i1) * (2.0 * c * sn * da + (c * c - sn * sn) * db - 2.0 * c * sn * dd);
  grad[b + 5u] = s1.y;
  grad[b + 6u] = s1.z;
  grad[b + 7u] = s1.w;
  grad[b + 8u] = sw * exp(params[b + 8u]);
}
