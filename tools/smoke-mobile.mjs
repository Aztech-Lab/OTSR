// Phone / tablet layout test of the built app (dist/) in headless Chrome with touch emulation: four viewports,
// the start screen with its system check, the default tier (Fast on phones), an image, the split run button with its
// tier menu and the readout strip, a Turbo fit (no fit animation), a short fit, the sheet (or the column, sideways),
// the menus, the dialogs, a two-finger pinch, and the technical report. Every floating element must stay inside the
// viewport and the page must not scroll sideways.
// Usage: npm run build && node tools/smoke-mobile.mjs [--vendor=nvidia] [--sample=AZ.PNG] [--steps=600]
// Screenshots go to tools/out/mobile/. Exit code 1 when a check fails.
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import puppeteer from 'puppeteer-core';
import { findBrowser, webgpuArgs } from './browser.mjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const DIST = path.join(ROOT, 'dist');
const OUT = path.join(ROOT, 'tools', 'out', 'mobile');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.map': 'application/json', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.md': 'text/markdown' };
if (!fs.existsSync(path.join(DIST, 'index.html'))) throw new Error('dist/index.html missing: run npm run build first');
fs.mkdirSync(OUT, { recursive: true });
const arg = (name, def) => process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? def;
const VENDOR = arg('vendor', '');
const SAMPLE = arg('sample', 'AZ.PNG');
const STEPS = Number(arg('steps', 600));

const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const file = path.join(DIST, rel === '/' ? 'index.html' : rel);
  if (!file.startsWith(DIST) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const VIEWPORTS = [
  { name: 'phone', width: 390, height: 844, deviceScaleFactor: 3, isMobile: true, hasTouch: true },
  { name: 'phone-narrow', width: 360, height: 800, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
  { name: 'phone-landscape', width: 844, height: 390, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
  { name: 'tablet', width: 820, height: 1180, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
];
const failures = [];
const check = (ok, what) => { console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) failures.push(what); };

/** Floating UI (and open dialogs) that reaches outside the viewport, plus any sideways page overflow. Content inside
 *  a scrolling panel may of course extend beyond it. */
const layoutProblems = () => {
  const vw = window.innerWidth, vh = window.innerHeight;
  const out = [];
  const name = (el) => `${el.tagName.toLowerCase()}${el.id ? '#' + el.id : ''}${el.className && typeof el.className === 'string' ? '.' + el.className.trim().split(/\s+/).join('.') : ''}`;
  const roots = [...document.querySelectorAll('.float, dialog[open]')];
  for (const root of roots) {
    for (const el of [root, ...root.querySelectorAll('*')]) {
      const cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden' || Number(cs.opacity) === 0) continue;
      if (el.closest('[hidden]')) continue;
      let scroller = false;
      for (let a = el.parentElement; a && a !== root.parentElement; a = a.parentElement) { const o = getComputedStyle(a).overflowY; if (o === 'auto' || o === 'scroll') { scroller = true; break; } }
      if (scroller) continue;
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      if (r.left < -1 || r.right > vw + 1 || r.top < -1 || r.bottom > vh + 1) out.push(`${name(el)} ${Math.round(r.left)},${Math.round(r.top)}–${Math.round(r.right)},${Math.round(r.bottom)}`);
    }
  }
  if (document.documentElement.scrollWidth > vw + 1) out.push(`page scrolls sideways: ${document.documentElement.scrollWidth} > ${vw}`);
  return out;
};
const rect = (sel) => { const el = document.querySelector(sel); if (!el) return null; const r = el.getBoundingClientRect(); return { l: r.left, t: r.top, r: r.right, b: r.bottom, w: r.width, h: r.height }; };
const overlap = (a, b) => !!a && !!b && a.l < b.r && b.l < a.r && a.t < b.b && b.t < a.b;
const imageRect = () => { const v = window.otsr.viewer, s = v.sourceSize, d = v.dpr; return { l: v.offsetX / d, t: v.offsetY / d, r: (v.offsetX + s.W * v.zoom) / d, b: (v.offsetY + s.H * v.zoom) / d }; };

const browser = await puppeteer.launch({ executablePath: findBrowser(), headless: true, protocolTimeout: 120000, args: [...webgpuArgs(), '--window-size=900,1200'] });
const t0 = performance.now();
try {
  for (const vp of VIEWPORTS) {
    const portrait = vp.height > vp.width;
    console.log(`\n${vp.name} ${vp.width}×${vp.height} (${portrait ? 'sheet' : 'column'} layout)`);
    const page = await browser.newPage();
    await page.setViewport(vp);
    page.on('pageerror', (e) => console.log(`  [page error] ${e.message}`));
    page.on('console', (m) => { if (m.type() === 'error') console.log(`  [browser error] ${m.text().slice(0, 200)}`); });
    const shot = (n) => page.screenshot({ path: path.join(OUT, `${vp.name}_${n}.png`) });
    const problems = async (what) => { const p = await page.evaluate(layoutProblems); check(p.length === 0, `${what}: layout inside the viewport${p.length ? ' — ' + p.slice(0, 4).join('; ') : ''}`); };
    const tapOn = async (sel) => { const r = await page.evaluate(rect, sel); if (!r) throw new Error(`${sel} missing`); await page.touchscreen.tap(r.l + r.w / 2, r.t + r.h / 2); };
    const folded = () => page.evaluate(() => document.body.classList.contains('left-closed'));
    await page.goto(url);
    await page.waitForFunction('window.otsr !== undefined', { timeout: 30000 });
    const adapter = await page.evaluate(() => window.otsr.state().adapter);
    if (vp === VIEWPORTS[0]) console.log(`  adapter: ${adapter.description} (${adapter.vendor})`);
    if (VENDOR && adapter.vendor !== VENDOR) throw new Error(`adapter vendor ${adapter.vendor} is not ${VENDOR}: stopping`);
    // a first visit: phones start on Fast and draw the viewer at most at 2x device pixels
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await page.waitForFunction('window.otsr !== undefined', { timeout: 30000 });
    const phoneVp = Math.min(vp.width, vp.height) < 600;
    const first = await page.evaluate(() => ({ preset: window.otsr.state().preset, phone: window.otsr.state().phone, dpr: window.otsr.viewer.dpr }));
    check(first.phone === phoneVp, `device class: ${first.phone ? 'phone' : 'not a phone'}`);
    check(first.preset === (phoneVp ? 'fast' : 'standard'), `first-visit tier: ${first.preset} (${phoneVp ? 'Fast on phones' : 'Standard elsewhere'})`);
    if (phoneVp) check(first.dpr === Math.min(vp.deviceScaleFactor, 2), `viewer at ${first.dpr}x device pixels on a ${vp.deviceScaleFactor}x screen`);
    await page.evaluate(() => { window.otsr.store.setPrefs({ theme: 'light', mode: 'simple', motion: true, fitAnim: true, panelLeft: true, scale: 4 }); document.documentElement.setAttribute('data-theme', 'light'); });
    await sleep(500);
    await shot('1_empty');
    await problems('start screen');
    const sc = await page.evaluate(() => { const el = document.getElementById('syscheck'); if (!el || el.hidden) return null; const r = el.getBoundingClientRect(); return { l: r.left, t: r.top, r: r.right, b: r.bottom, w: r.width, h: r.height, verdict: el.querySelector('.sc-verdict b')?.textContent ?? '', rows: el.querySelectorAll('.sc-rows dt').length, gpu: el.querySelectorAll('.sc-rows dd')[1]?.textContent ?? '' }; });
    check(!!sc && sc.rows === 6 && /Ready|Limited|Slow/.test(sc.verdict), `system check on the start screen: ${sc?.verdict}, ${sc?.rows} rows, GPU ${sc?.gpu}`);
    for (const sel of ['.instrument .center', '#brand', '.sysnav']) check(!(await page.evaluate(overlap, sc, await page.evaluate(rect, sel))), `system check clear of ${sel}`);
    check(await page.evaluate(() => { const b = document.getElementById('empty-sample'); const r = b && !b.hidden && b.getBoundingClientRect(); return !!r && r.width > 40; }), 'start screen: Load sample button present');
    check(!(await page.evaluate(overlap, await page.evaluate(rect, '#brand'), await page.evaluate(rect, '.sysnav'))), 'start screen: wordmark and system nav do not overlap');
    check(await page.evaluate(() => ['nav-open', 'settings-btn', 'theme-btn'].every((id) => { const r = document.getElementById(id).getBoundingClientRect(); return r.width >= 30 && r.height >= 30; })), 'start screen: nav buttons are at least 30 px');
    // a tap on Load sample (phone) or a scripted load with an absurd file name (others)
    if (vp === VIEWPORTS[0]) {
      await tapOn('#empty-sample');
      await page.waitForFunction(() => !!window.otsr.state().image, { timeout: 20000 });
    } else {
      await page.evaluate(() => { const W = 120, H = 80, px = new Uint8ClampedArray(W * H * 4); for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let r = 40 + 180 * x / W, g = 90 + 120 * y / H, b = 200 - 150 * x / W; if ((x - 35) ** 2 + (y - 30) ** 2 < 300) { r = 240; g = 200; b = 60; } if ((x - 85) ** 2 + (y - 50) ** 2 < 500) { r = 30; g = 160; b = 220; } if (x > 50 && x < 70 && y > 10 && y < 26) { r = 20; g = 20; b = 30; } px.set([r, g, b, 255], 4 * (y * W + x)); } window.otsr.loadRgba(px, W, H, 'a_very_long_file_name_that_must_not_push_the_replace_button_out_of_the_panel_2048x2048.png'); });
    }
    await sleep(1800);
    await shot('2_loaded');
    await problems('image loaded');
    check(!(await page.evaluate(overlap, await page.evaluate(rect, '#brand'), await page.evaluate(rect, '.sysnav'))), 'image loaded: wordmark and system nav do not overlap');
    if (portrait) {
      check(await folded(), 'sheet folded when the image arrives');
      let img = await page.evaluate(imageRect);
      const cmd0 = await page.evaluate(rect, '#command');
      check(img.t >= cmd0.b - 1, `image (top ${Math.round(img.t)}) clear of the command block (bottom ${Math.round(cmd0.b)})`);
      await tapOn('#console-toggle');
      await sleep(500);
      check(!(await folded()), 'tap on the title bar opens the sheet');
      await shot('2b_sheet');
      await problems('sheet open');
      const sheet = await page.evaluate(rect, '#inspector');
      const bar = await page.evaluate(rect, '#toolbar');
      const cmd = await page.evaluate(rect, '#command');
      check(sheet && sheet.h <= vp.height * 0.62, `sheet height ${Math.round(sheet?.h ?? 0)} ≤ 62 % of ${vp.height}`);
      check(bar && sheet && bar.b <= sheet.t + 2, 'zoom bar sits above the sheet');
      check(!(await page.evaluate(overlap, cmd, bar)), 'command block and zoom bar do not overlap');
      check(!(await page.evaluate(overlap, cmd, await page.evaluate(rect, '.sysnav'))), 'command block and system nav do not overlap');
      img = await page.evaluate(imageRect);
      check(img.t >= cmd.b - 1 && img.b <= bar.t + 1, `image (${Math.round(img.t)}–${Math.round(img.b)}) between the command block (${Math.round(cmd.b)}) and the zoom bar (${Math.round(bar.t)}) with the sheet open`);
    } else {
      const col = await page.evaluate(rect, '#inspector');
      const cmd = await page.evaluate(rect, '#command');
      const bar = await page.evaluate(rect, '#toolbar');
      check(col && col.l <= 1 && col.w >= 280 && col.w <= 320 && col.b >= vp.height - 1, `console is a column on the left (${Math.round(col?.w ?? 0)} px wide, to the bottom)`);
      check(cmd.l >= col.r - 1 && bar.l >= col.r - 1, 'command block and zoom bar are right of the column');
      const img = await page.evaluate(imageRect);
      check(img.l >= col.r - 1 && img.t >= cmd.b - 1 && img.b <= bar.t + 1, `image (${Math.round(img.l)},${Math.round(img.t)}–${Math.round(img.b)}) right of the column and between the command block (${Math.round(cmd.b)}) and the zoom bar (${Math.round(bar.t)})`);
    }
    // the source and the tier left the sheet: the action is a split button over a strip of readouts
    check(await page.evaluate(() => { const p = document.getElementById('panel'); const first = p.querySelector('.sec .sec-head .name')?.textContent; return !p.querySelector('.file') && first === 'Output' && !!p.querySelector('.switch input') && p.scrollWidth <= p.clientWidth + 1; }), 'console opens on Output (source and tier moved to the action), no sideways scroll');
    const strip = await page.evaluate(() => { const el = document.getElementById('cmd-info'); const r = el.getBoundingClientRect(); const t = document.getElementById('cmd-tier').getBoundingClientRect(); return { h: r.height, name: document.getElementById('ci-name').textContent, cells: el.querySelectorAll('.readout').length, tier: t.width, label: document.getElementById('cmd-tier-en').textContent, state: document.getElementById('ci-state').textContent }; });
    check(strip.tier > 40 && strip.label === (phoneVp ? 'Fast' : 'Standard'), `split run button shows the tier (${strip.label})`);
    check(strip.h > 20 && strip.cells === 4 && strip.name.length > 2 && strip.state === 'Idle', `readout strip under the action: ${strip.name} · ${strip.state}`);
    if (vp === VIEWPORTS[0]) {
      // the tier menu of the split button, and a Turbo fit: no fit animation
      await tapOn('#cmd-tier');
      await sleep(250);
      await shot('2c_tiermenu');
      await problems('tier menu open');
      const menu = await page.evaluate(() => { const m = document.getElementById('tier-menu'); return { open: !m.hidden, items: [...m.querySelectorAll('.tier')].map((b) => b.querySelector('b').textContent), on: m.querySelector('.tier.on b')?.textContent }; });
      check(menu.open && menu.items.join(',') === 'Turbo,Fast,Standard,Best' && menu.on === 'Fast', `tier menu: ${menu.items.join(' / ')}, ${menu.on} marked`);
      await tapOn('#tier-menu .tier:first-of-type');
      await sleep(250);
      const picked = await page.evaluate(() => ({ preset: window.otsr.state().preset, open: !document.getElementById('tier-menu').hidden, label: document.getElementById('cmd-tier-en').textContent, chosen: window.otsr.store.state.prefs.presetChosen }));
      check(picked.preset === 'turbo' && !picked.open && picked.label === 'Turbo' && picked.chosen, 'Turbo picked in the menu: label updated, menu closed, the choice is kept');
      await page.evaluate(() => { window.__show = false; new MutationObserver(() => { if (document.body.classList.contains('fitshow')) window.__show = true; }).observe(document.body, { attributes: true, attributeFilter: ['class'] }); void window.otsr.fit(); });
      await page.waitForFunction(() => !window.otsr.state().fitting && !!window.otsr.state().lastFit, { timeout: 60000 });
      const tf = await page.evaluate(() => ({ fit: window.otsr.state().lastFit, show: window.__show, state: document.getElementById('ci-state').textContent }));
      check(tf.fit.steps === 400 && !tf.show, `Turbo fit: ${tf.fit.steps} steps, ${tf.fit.psnr.toFixed(2)} dB in ${tf.fit.seconds.toFixed(2)} s, no fit animation`);
      check(/dB$/.test(tf.state), `strip shows the result (${tf.state})`);
      await sleep(600);
      await shot('2d_turbo_done');
      await problems('after the Turbo fit');
    }
    // folding
    await tapOn('#console-toggle');
    await sleep(450);
    const foldedBox = await page.evaluate(rect, '#inspector');
    check(await folded() && (portrait ? foldedBox && foldedBox.h < 60 : !foldedBox || foldedBox.w === 0), portrait ? `sheet folds to its title bar (${Math.round(foldedBox?.h ?? 0)} px)` : 'column closes to the edge tab');
    await shot('3_folded');
    await problems('folded');
    if (!portrait) { await tapOn('#left-tab'); await sleep(450); check(!(await folded()), 'the edge tab opens the column again'); }
    // a two-finger pinch over the image: the zoom must grow about the fingers
    await page.evaluate(() => window.otsr.viewer.fitToView());
    await sleep(200);
    const z0 = await page.evaluate(() => window.otsr.state().zoom);
    const img = await page.evaluate(imageRect);
    const cx = (img.l + img.r) / 2, cy = (img.t + img.b) / 2;
    const a = await page.touchscreen.touchStart(cx - 30, cy);
    const b = await page.touchscreen.touchStart(cx + 30, cy);
    for (let k = 1; k <= 8; k++) { await a.move(cx - 30 - k * 7, cy); await b.move(cx + 30 + k * 7, cy); await sleep(16); }
    await a.end();
    await b.end();
    await sleep(300);
    const z1 = await page.evaluate(() => window.otsr.state().zoom);
    check(z1 > z0 * 1.6 && z1 < z0 * 3.2, `pinch zooms in (${z0.toFixed(2)} → ${z1.toFixed(2)}, fingers 60 → 172 px apart)`);
    await page.evaluate(() => window.otsr.viewer.fitToView());
    // a fit: on the sheet layout the sheet folds while it runs and opens afterwards
    const steps = vp === VIEWPORTS[0] ? STEPS : 300;
    await page.evaluate((n) => { window.otsr.fit({ kPerPx: 0.1, steps: n }); }, steps);
    await page.waitForFunction(() => window.otsr.state().fitting, { timeout: 10000 });
    await sleep(900);
    if (portrait) check(await folded(), 'sheet folded during the fit');
    await shot('4_fitting');
    await problems('fitting');
    await page.waitForFunction(() => !window.otsr.state().fitting, { timeout: 120000 });
    const fit = await page.evaluate(() => window.otsr.state().lastFit);
    console.log(`  fit: ${fit.psnr.toFixed(2)} dB, ${fit.steps} steps in ${fit.seconds.toFixed(2)} s`);
    check(fit.psnr > 28, 'fit PSNR > 28 dB');
    await sleep(2600); // the reveal
    check(!(await folded()), portrait ? 'sheet open after the fit' : 'column open after the fit');
    await shot('5_fitted');
    await problems('fitted');
    check(await page.evaluate(() => /\d+\.\d\d dB/.test(document.getElementById('ci-state')?.textContent ?? '')), 'PSNR shown in the strip under Enhance');
    check(await page.evaluate(() => !document.querySelector('#panel .run') && !/Save model|Load model/.test(document.getElementById('panel').textContent)), 'no process log or model file buttons in the phone console');
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'no sideways scroll');
    // menus and dialogs
    await page.evaluate(() => document.querySelector('.splitbtn .menu').click());
    await sleep(250);
    await shot('6_exportmenu');
    await problems('export menu open');
    await page.evaluate(() => { document.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })); document.getElementById('tb-view').click(); });
    await sleep(250);
    await shot('7_viewmenu');
    await problems('view menu open');
    await page.evaluate(() => { document.getElementById('viewmenu').hidden = true; document.getElementById('settings').showModal(); });
    await sleep(250);
    await shot('8_settings');
    await problems('settings dialog');
    check(await page.evaluate(() => { const d = document.getElementById('settings'); return d.scrollHeight <= d.clientHeight + 1 || getComputedStyle(d).overflowY === 'auto'; }), 'settings dialog scrolls when taller than the screen');
    check(await page.evaluate(() => { const d = document.getElementById('settings'); return d.scrollWidth <= d.clientWidth && getComputedStyle(d).overflowX === 'hidden'; }), 'settings dialog: no sideways scroll');
    await page.evaluate(() => { document.getElementById('settings').close(); window.otsr.setMode('research'); });
    await sleep(400);
    await shot('9_research');
    await problems('research mode');
    await page.evaluate(() => { window.otsr.setMode('simple'); window.otsr.store.setPrefs({ theme: 'dark' }); document.documentElement.setAttribute('data-theme', 'dark'); window.otsr.viewer.background = [0.059, 0.067, 0.071]; window.otsr.viewer.requestRender(); });
    await sleep(400);
    await shot('10_dark');
    await problems('dark theme');
    // the report page
    await page.goto(`${url}report.html`);
    await page.waitForSelector('.article h1');
    await sleep(300);
    check(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), 'report: no sideways scroll');
    check(await page.evaluate(() => [...document.querySelectorAll('.fig svg, .fig img, pre, .tablewrap, .numbers, .top')].every((el) => el.getBoundingClientRect().right <= window.innerWidth + 1)), 'report: figures, code, tables and the top bar inside the viewport');
    await shot('11_report');
    await page.evaluate(() => document.getElementById('port').scrollIntoView());
    await sleep(600);
    await shot('12_report_figures');
    await page.close();
  }
  console.log(`\n${failures.length ? `${failures.length} FAILED` : 'OK'} in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  process.exitCode = failures.length ? 1 : 0;
} finally {
  await browser.close();
  server.close();
}
