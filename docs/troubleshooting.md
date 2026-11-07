# Troubleshooting

## "WebGPU is not available in this browser"

`navigator.gpu` does not exist. Update the browser (Chrome/Edge 113+, Safari 26+, Firefox 141+ on Windows) or
check that WebGPU is not disabled: in Chrome open `chrome://flags/#enable-unsafe-webgpu` (should be *Default* or
*Enabled*) and `chrome://gpu` (look for "WebGPU: Hardware accelerated"). On Linux, Chrome enables WebGPU from version
121; on some distributions it still needs `chrome://flags/#enable-vulkan`.

## "No WebGPU adapter found"

The browser has WebGPU but no usable GPU. Causes:

- the GPU is on Chrome's blocklist (old drivers): update the driver, or start Chrome with
  `--ignore-gpu-blocklist` to test;
- the page is served from `file://`: serve it over HTTP (see getting-started.md);
- remote desktop sessions and some virtual machines expose no GPU;
- a transient failure right after the browser started: reload the page (the app already retries a few times).

## The wrong GPU is used (laptops and desktops with two GPUs)

The header shows which adapter runs OTSR. On Windows, Chrome ignores the "high-performance" request
([crbug.com/369219127](https://crbug.com/369219127)) and uses the GPU that drives the display; on a machine whose
monitor is connected to the integrated GPU, WebGPU therefore runs on the integrated GPU even if a discrete one is
installed. To use the discrete GPU:

- Windows: *Settings → System → Display → Graphics*, add the browser executable and set it to *High performance*,
  then restart the browser; or
- connect the monitor to the discrete GPU; or
- in the NVIDIA / AMD control panel, set the preferred GPU for the browser.

macOS laptops switch automatically. On Linux, `chrome://flags/#use-angle` and the `DRI_PRIME=1` environment variable
select the GPU.

## It runs, but slowly

- Check the adapter name in the header: an integrated GPU fits a 2-megapixel image in a minute or two, a discrete GPU
  in a few seconds.
- Lower the budget (0.05) or the steps (1500) for a quick look; fit at the full settings for the final export.
- Keep the browser tab in the foreground: background tabs are throttled.
- Very large images: the viewer renders only what is visible, but the fit always covers the whole image.

## "WebGPU device lost" or the page goes blank

The GPU process crashed or was reset (driver timeout, out of memory, the laptop switched GPUs). Reload the page.
If it happens on **Fit** with a large image, reduce the image size or the budget: during the fit the engine keeps
roughly `32 bytes × pixels + 350 bytes × Gaussians` on the GPU (see performance.md).

## Export fails or produces nothing

The browser's PNG encoder refuses images above 16384 pixels per side or roughly 120 megapixels; choose a smaller
scale. Large exports take a while: rendering is fast, encoding a 64-megapixel PNG in the browser is not.

## Headless browsers and automation

Headless Chrome needs `--enable-unsafe-webgpu --ignore-gpu-blocklist`, may choose the integrated GPU, does not
composite WebGPU canvases into screenshots, and sometimes fails the first adapter request after start-up. The
repository's smoke test (`tools/smoke-ui.mjs`) deals with all of this; for your own automation use the
`window.otsr` hook (development.md) and `otsr.snapshotPng()` instead of screenshots.
