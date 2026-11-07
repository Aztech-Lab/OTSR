// Browser smoke test of the built app (dist/): starts WebGPU, loads a synthetic image through window.otsr, fits a
// small model, renders an export and takes a screenshot. Usage: npm run build && node tools/smoke-ui.mjs [--headful]
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import puppeteer from 'puppeteer-core';
import { findBrowser, webgpuArgs } from './browser.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const DIST = path.join(ROOT, 'dist');
const OUT = path.join(ROOT, 'tools', 'out');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.map': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.md': 'text/markdown' };
if (!fs.existsSync(path.join(DIST, 'index.html'))) throw new Error('dist/index.html missing: run npm run build first');
fs.mkdirSync(OUT, { recursive: true });

const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  let file = path.join(DIST, rel === '/' ? 'index.html' : rel);
  if (!file.startsWith(DIST) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/`;
const headful = process.argv.includes('--headful');
const browser = await puppeteer.launch({ executablePath: findBrowser(), headless: !headful, args: [...webgpuArgs(), '--window-size=1400,900'], defaultViewport: { width: 1400, height: 900 } });
const t0 = performance.now();
try {
  const page = await browser.newPage();
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.log(`  [browser ${m.type()}] ${m.text().slice(0, 300)}`); });
  page.on('pageerror', e => console.log(`  [page error] ${e.message}`));
  await page.goto(url);
  await page.waitForFunction('window.otsr !== undefined', { timeout: 30000 });
  const adapter = await page.evaluate(() => window.otsr.state().adapter);
  console.log(`adapter: ${adapter.description} (${adapter.vendor} ${adapter.architecture})${adapter.isFallback ? ' FALLBACK' : ''}`);
  // a 120 x 80 synthetic image: gradient background, two discs, a dark box, a thin diagonal line
  await page.evaluate(() => {
    const W = 120, H = 80, px = new Uint8ClampedArray(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let r = 40 + 180 * x / W, g = 90 + 120 * y / H, b = 200 - 150 * x / W;
      if ((x - 35) ** 2 + (y - 30) ** 2 < 300) { r = 240; g = 200; b = 60; }
      if ((x - 85) ** 2 + (y - 50) ** 2 < 500) { r = 30; g = 160; b = 220; }
      if (x > 50 && x < 70 && y > 10 && y < 26) { r = 20; g = 20; b = 30; }
      if (Math.abs((y - 10) - 0.5 * (x - 10)) < 0.8) { r = 250; g = 250; b = 250; }
      px.set([r, g, b, 255], 4 * (y * W + x));
    }
    window.otsr.loadRgba(px, W, H, 'synthetic.png');
  });
  const fit = await page.evaluate(() => window.otsr.fit({ kPerPx: 0.1, steps: 300 }));
  console.log(`fit: ${fit.psnr.toFixed(2)} dB, ${fit.steps} steps in ${fit.seconds.toFixed(2)} s`);
  if (!(fit.psnr > 28)) throw new Error(`PSNR too low: ${fit.psnr}`);
  const rendered = await page.evaluate(async () => { const r = await window.otsr.render(4); let dark = 0; for (let i = 0; i < r.rgba.length; i += 4) if (r.rgba[i] + r.rgba[i + 1] + r.rgba[i + 2] < 30) dark++; return { w: r.width, h: r.height, darkFrac: dark / (r.width * r.height) }; });
  console.log(`render x4: ${rendered.w}x${rendered.h}, dark pixels ${(rendered.darkFrac * 100).toFixed(1)} %`);
  if (rendered.w !== 480 || rendered.h !== 320 || rendered.darkFrac > 0.2) throw new Error('export looks wrong');
  const bytes = await page.evaluate(() => window.otsr.modelBytes()?.byteLength ?? 0);
  console.log(`.otsr bytes: ${bytes}`);
  await page.evaluate(() => { window.otsr.viewer.zoomAbout(6, 500, 300); });
  await new Promise(r => setTimeout(r, 300));
    // Headless Chrome does not composite WebGPU canvases into screenshots, so the view is checked through a snapshot
  // rendered into an offscreen texture (the same passes, the same presenter).
  // simple mode after the fit: the split comparison is on by default
  await page.screenshot({ path: path.join(OUT, 'smoke_page_simple.png') });
  fs.writeFileSync(path.join(OUT, 'smoke_view_split.png'), Buffer.from((await page.evaluate(() => window.otsr.snapshotPng(true))).split(',')[1], 'base64'));
  const st = await page.evaluate(() => window.otsr.state());
  console.log(`state: mode ${st.mode}, model ${st.model?.N} Gaussians, zoom ${st.zoom.toFixed(2)}`);
  // research mode with a region of interest drawn on the overlay
  await page.evaluate(() => { window.otsr.setMode('research'); window.otsr.viewer.compare = false; window.otsr.setRoi({ x: 20, y: 15, w: 40, h: 40 }); });
  await new Promise(r => setTimeout(r, 300));
  await page.screenshot({ path: path.join(OUT, 'smoke_page_research.png') });
  fs.writeFileSync(path.join(OUT, 'smoke_view_roi.png'), Buffer.from((await page.evaluate(() => window.otsr.snapshotPng(true))).split(',')[1], 'base64'));
  for (const [mode, file] of [['otsr', 'smoke_view_otsr.png'], ['raw', 'smoke_view_raw.png'], ['bicubic', 'smoke_view_bicubic.png']]) {
    const dataUrl = await page.evaluate(async (m) => { window.otsr.viewer.mode = m; return window.otsr.snapshotPng(false); }, mode);
    const png = Buffer.from(dataUrl.split(',')[1], 'base64');
    fs.writeFileSync(path.join(OUT, file), png);
    const mean = await page.evaluate(async () => { const s = await window.otsr.viewer.snapshot(); let sum = 0; for (let i = 0; i < s.rgba.length; i += 4) sum += s.rgba[i] + s.rgba[i + 1] + s.rgba[i + 2]; return sum / (s.rgba.length / 4) / 3; });
    console.log(`view ${mode.padEnd(7)} mean level ${mean.toFixed(1)} -> ${path.join(OUT, file)} (${(png.length / 1024).toFixed(0)} kB)`);
    if (mean < 5) throw new Error(`view ${mode} is blank`);
  }
  console.log(`OK in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
} finally {
  await browser.close();
  server.close();
}
