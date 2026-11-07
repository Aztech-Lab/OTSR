// Checks that a locally installed Chrome/Edge can run WebGPU (headless), which the GPU test suite relies on.
// Usage: node tools/webgpu-probe.mjs [--headful] [--browser=<path to chrome.exe or msedge.exe>]
import http from 'node:http';
import puppeteer from 'puppeteer-core';
import { findBrowser, webgpuArgs } from './browser.mjs';

const headful = process.argv.includes('--headful');
const page_html = `<!doctype html><meta charset="utf-8"><title>probe</title><script>
(async () => {
  const out = { hasGPU: !!navigator.gpu, secure: window.isSecureContext };
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter) {
      const info = adapter.info || {};
      out.adapter = { vendor: info.vendor, architecture: info.architecture, description: info.description };
      out.features = [...adapter.features].sort();
      const l = adapter.limits;
      out.limits = { maxStorageBufferBindingSize: l.maxStorageBufferBindingSize, maxBufferSize: l.maxBufferSize,
        maxComputeWorkgroupStorageSize: l.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
        maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage,
        maxComputeWorkgroupsPerDimension: l.maxComputeWorkgroupsPerDimension };
      const device = await adapter.requestDevice();
      out.device = !!device;
    }
  } catch (e) { out.error = String(e); }
  window.__probe = out;
})();
</script>`;

const server = http.createServer((req, res) => { res.writeHead(200, { 'content-type': 'text/html' }); res.end(page_html); });
await new Promise(r => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/`;
const browser = await puppeteer.launch({ executablePath: findBrowser(), headless: !headful, args: webgpuArgs() });
try {
  const page = await browser.newPage();
  await page.goto(url);
  await page.waitForFunction('window.__probe !== undefined', { timeout: 20000 });
  console.log(JSON.stringify(await page.evaluate('window.__probe'), null, 2));
} finally {
  await browser.close();
  server.close();
}
