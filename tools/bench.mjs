// End-to-end timing of the engine in Node on a real GPU (Dawn): OT initialisation, the fit, an S x export.
//   python reference/dump_image.py image.png image.rgb.bin
//   node tools/bench.mjs image.rgb.bin [--kpp=0.1] [--steps=3000] [--scale=4] [--adapter=4090] [--out=render.rgba]
// Prints a JSON line with the timings; docs/performance.md collects them.
import fs from 'node:fs';
import { create, globals } from 'webgpu';
import { createServer } from 'vite';

Object.assign(globalThis, globals);
const args = Object.fromEntries(process.argv.slice(3).filter(a => a.startsWith('--')).map(a => { const [k, v] = a.slice(2).split('='); return [k, v ?? 'true']; }));
const file = process.argv[2];
if (!file) throw new Error('usage: node tools/bench.mjs image.rgb.bin [--kpp=] [--steps=] [--scale=] [--adapter=] [--out=]');
const flags = args.adapter ? [`adapter=${args.adapter}`] : [];
const gpu = create(flags);
Object.defineProperty(globalThis, 'navigator', { value: { gpu }, configurable: true });

// load the TypeScript engine through Vite's module runner (handles the .wgsl?raw imports)
const vite = await createServer({ server: { middlewareMode: true }, optimizeDeps: { noDiscovery: true }, logLevel: 'error' });
const engine = await vite.ssrLoadModule('/src/engine/index.ts');

const buf = fs.readFileSync(file);
if (buf.toString('latin1', 0, 4) !== 'OTSI') throw new Error('not an OTSI dump (see reference/dump_image.py)');
const W = buf.readUInt32LE(4), H = buf.readUInt32LE(8);
const rgb = new Float32Array(buf.buffer.slice(buf.byteOffset + 16, buf.byteOffset + 16 + W * H * 12));
const rgba = new Float32Array(W * H * 4);
for (let i = 0; i < W * H; i++) rgba.set([rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], 1], 4 * i);
const kpp = Number(args.kpp ?? 0.1), steps = Number(args.steps ?? 3000), scale = Number(args.scale ?? 4);
const edges = args.edges !== '0' && args.edges !== 'false';   // --edges=0: plain Gaussian render

const ctx = await engine.acquireDevice(gpu);
const pipes = await engine.createPipelines(ctx.device);
const info = ctx.info;
console.log(`adapter: ${info.description} (${info.vendor} ${info.architecture}); image ${W}x${H}, ${kpp} Gaussians/px, ${steps} steps`);
const t0 = performance.now();
const init = await engine.otInit(ctx, pipes, rgb, W, H, { K: Math.round(kpp * W * H) });
const tInit = (performance.now() - t0) / 1000;
console.log(`init: ${init.count} Gaussians (${init.gy}x${init.gx}), grid ${init.grid.ny}x${init.grid.nx}, ${init.iters} Sinkhorn iterations in ${tInit.toFixed(2)} s (density ${init.ms.density.toFixed(0)} ms, transport ${init.ms.transport.toFixed(0)} ms, params ${init.ms.params.toFixed(0)} ms)`);
let last = null;
const curve = [];
const res = await engine.runFit(ctx, pipes, init.params, W, H, rgba, { steps, logEvery: Number(args.log ?? 100), onProgress: p => { last = p; curve.push({ step: p.step, psnr: p.psnr, ms: p.elapsedMs }); if (p.step % 500 === 0 || p.step === steps) console.log(`  step ${String(p.step).padStart(5)}  ${p.psnr.toFixed(2)} dB  ${(p.elapsedMs / 1000).toFixed(1)} s`); } });
if (args.curve) fs.writeFileSync(args.curve, JSON.stringify({ adapter: info.description, W, H, N: init.count, steps, init_s: tInit, curve }));
console.log(`fit: ${last.psnr.toFixed(2)} dB, ${steps} steps in ${res.seconds.toFixed(2)} s = ${(steps / res.seconds).toFixed(0)} it/s; ${res.entries} tile entries (${(res.entries / init.count).toFixed(2)} per Gaussian)`);
const params = engine.uploadBuffer(ctx.device, res.params, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
const t1 = performance.now();
const img = await engine.renderImage(ctx, pipes, params, init.count, W, H, scale, { edges });
const tRender = (performance.now() - t1) / 1000;
console.log(`render x${scale}: ${img.width}x${img.height} ${edges ? 'with' : 'without'} edges in ${tRender.toFixed(2)} s`);
if (args.out) { fs.writeFileSync(args.out, Buffer.concat([Buffer.from(new Uint32Array([img.width, img.height]).buffer), Buffer.from(img.rgba.buffer)])); console.log(`wrote ${args.out}`); }
const t2 = performance.now();
for (let i = 0; i < 5; i++) await engine.renderImage(ctx, pipes, params, init.count, W, H, 1, { edges: false });
const tView = (performance.now() - t2) / 5000;
console.log(`native render (no edges): ${(tView * 1000).toFixed(1)} ms per frame including readback`);
console.log(JSON.stringify({ adapter: info.description, W, H, kpp, N: init.count, steps, psnr: last.psnr, init_s: tInit, fit_s: res.seconds, it_per_s: steps / res.seconds, scale, render_s: tRender, native_ms: tView * 1000 }));
await vite.close();
delete globalThis.navigator;
