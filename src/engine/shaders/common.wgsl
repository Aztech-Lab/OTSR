// Shared by every OTSR kernel (prepended to each module). Constants mirror src/core/spec.ts.

const TILE: u32 = 16u;
const TILE_PIXELS: u32 = 256u;
const K_SIGMA: f32 = 4.0;
const Q_CUT: f32 = 8.0;                          // 0.5 * K_SIGMA^2
const ALPHA_SHIFT: f32 = 0.00033546262790251185; // exp(-Q_CUT)
const EPS: f32 = 1e-4;
const FPG: u32 = 9u;                             // floats per Gaussian: uv(2) log_s(2) rot(1) color(3) log_w(1)
const INVALID: u32 = 0xffffffffu;

// One raster: the source image function sampled on an output grid. Output pixel (i, j) has its centre at the
// source position ((j + .5 - offset.x) / zoom, (i + .5 - offset.y) / zoom) in source pixels.
struct View {
  src_size: vec2<f32>,   // W, H of the source image
  zoom: f32,
  long_side: f32,        // L = max(W, H)
  offset: vec2<f32>,
  aa: f32,               // isotropic prefilter std in output pixels (0 = sample the exact function)
  grad_scale: f32,       // d(loss)/d(sum of squared errors): 1 / (3 W H) for the mean-squared-error fit
  out_size: vec2<u32>,
  tiles: vec2<u32>,
  count: u32,            // number of Gaussians
  entry_cap: u32,        // capacity of the tile-entry arrays
  _pad: vec2<u32>,
};

// Linear invocation index for 1-D work laid out as a 2-D dispatch of `size`-wide workgroups (dispatch1d in util.ts).
fn linear_id(wid: vec3<u32>, nwg: vec3<u32>, lid: u32, size: u32) -> u32 {
  return (wid.y * nwg.x + wid.x) * size + lid;
}

// Infinite or NaN, robust against fast-math folding of x != x.
fn non_finite(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u;
}
