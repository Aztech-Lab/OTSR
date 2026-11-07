// Shows one or two (r, g, b, *) storage images on the canvas with a full-screen triangle: pixels left of `split`
// (output pixels) come from image B (the comparison), the rest from image A. Pixels outside the image rectangle
// [x0, x1) x [y0, y1) show the background colour. With no comparison, B is bound to A and split is 0.

struct PresentParams {
  width: u32,
  height: u32,
  split: f32,
  _p0: u32,
  rect: vec4<f32>,       // x0, y0, x1, y1
  background: vec4<f32>,
};

@group(0) @binding(0) var<uniform> pp: PresentParams;
@group(0) @binding(1) var<storage, read> img_a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> img_b: array<vec4<f32>>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let x = f32(i32(vi & 1u) * 4 - 1);
  let y = f32(i32(vi >> 1u) * 4 - 1);
  return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let x = u32(pos.x);
  let y = u32(pos.y);
  if (x >= pp.width || y >= pp.height || pos.x < pp.rect.x || pos.y < pp.rect.y || pos.x >= pp.rect.z || pos.y >= pp.rect.w) {
    return pp.background;
  }
  let i = y * pp.width + x;
  var c: vec3<f32>;
  if (pos.x < pp.split) {
    c = img_b[i].xyz;
  } else {
    c = img_a[i].xyz;
  }
  return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
