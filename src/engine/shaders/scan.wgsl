// Exclusive prefix sum of u32 arrays in blocks of 1024 (256 threads x 4 elements). scan.ts chains the levels:
// scan_blocks writes each block's total to block_sums, the totals are scanned the same way, and add_offsets adds
// the scanned totals back. The last level has one block, whose total is the grand total.

struct ScanParams {
  n: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<uniform> sp: ScanParams;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;

var<workgroup> sh: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_blocks(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_index) li: u32) {
  let blk = wid.y * nwg.x + wid.x;
  let base = blk * 1024u + li * 4u;
  var local: array<u32, 4>;
  var run = 0u;
  for (var k = 0u; k < 4u; k++) {
    var x = 0u;
    if (base + k < sp.n) {
      x = src[base + k];
    }
    local[k] = run;
    run += x;
  }
  sh[li] = run;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off <<= 1u) {
    var t = 0u;
    if (li >= off) {
      t = sh[li - off];
    }
    workgroupBarrier();
    sh[li] += t;
    workgroupBarrier();
  }
  let prefix = sh[li] - run;
  for (var k = 0u; k < 4u; k++) {
    if (base + k < sp.n) {
      dst[base + k] = prefix + local[k];
    }
  }
  if (li == 255u) {
    block_sums[blk] = sh[255u];
  }
}

@compute @workgroup_size(256)
fn add_offsets(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_index) li: u32) {
  let blk = wid.y * nwg.x + wid.x;
  let off = offsets[blk];
  let base = blk * 1024u + li * 4u;
  for (var k = 0u; k < 4u; k++) {
    if (base + k < sp.n) {
      dst[base + k] += off;
    }
  }
}
