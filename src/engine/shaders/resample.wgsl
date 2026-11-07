// Classic resampling of the source image on a View raster, for side-by-side comparison with the Gaussian render:
// mode 0 nearest, 1 bilinear, 2 bicubic (Keys, a = -1/2, as PIL / OpenCV). Borders are clamped.

struct ResampleParams {
  mode: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> rp: ResampleParams;
@group(0) @binding(2) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;

fn fetch(x: i32, y: i32) -> vec3<f32> {
  let w = i32(view.src_size.x);
  let h = i32(view.src_size.y);
  let cx = clamp(x, 0, w - 1);
  let cy = clamp(y, 0, h - 1);
  return src[u32(cy * w + cx)].xyz;
}

fn keys(t: f32) -> f32 {
  let a = -0.5;
  let x = abs(t);
  if (x < 1.0) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  }
  if (x < 2.0) {
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
  }
  return 0.0;
}

@compute @workgroup_size(16, 16)
fn resample(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= view.out_size.x || gid.y >= view.out_size.y) {
    return;
  }
  // source position of the output pixel centre, in source pixel index coordinates
  let s = (vec2<f32>(gid.xy) + 0.5 - view.offset) / view.zoom - 0.5;
  var c = vec3<f32>(0.0);
  if (rp.mode == 0u) {
    c = fetch(i32(floor(s.x + 0.5)), i32(floor(s.y + 0.5)));
  } else if (rp.mode == 1u) {
    let f = floor(s);
    let t = s - f;
    let x0 = i32(f.x);
    let y0 = i32(f.y);
    c = mix(mix(fetch(x0, y0), fetch(x0 + 1, y0), t.x), mix(fetch(x0, y0 + 1), fetch(x0 + 1, y0 + 1), t.x), t.y);
  } else {
    let f = floor(s);
    let t = s - f;
    let x0 = i32(f.x);
    let y0 = i32(f.y);
    var wsum = 0.0;
    for (var j = -1; j <= 2; j++) {
      let wy = keys(f32(j) - t.y);
      for (var i = -1; i <= 2; i++) {
        let w = wy * keys(f32(i) - t.x);
        c += w * fetch(x0 + i, y0 + j);
        wsum += w;
      }
    }
    c /= wsum;
  }
  out[gid.y * view.out_size.x + gid.x] = vec4<f32>(c, 1.0);
}
