# OTSR documentation

| Document | For whom | Contents |
|---|---|---|
| [getting-started.md](getting-started.md) | users | running the app, the workflow, settings, limits |
| [troubleshooting.md](troubleshooting.md) | users | WebGPU not available, wrong GPU on dual-GPU machines, memory, speed |
| [method.md](method.md) | everyone | the model, the optimal-transport initialisation, the fit, the edge pass |
| [engine.md](engine.md) | developers | the WebGPU pipeline: buffers, kernels, gradients, capacity handling, numerics |
| [file-format.md](file-format.md) | developers | the `.otsr` model file |
| [testing.md](testing.md) | developers | reference implementation, golden data, what each test proves, measured accuracy |
| [development.md](development.md) | developers | repository layout, scripts, conventions, the `window.otsr` scripting hook |
| [performance.md](performance.md) | everyone | measured timings, memory model, how to benchmark |
| [tutorial-porting.md](tutorial-porting.md) | developers | how the research code was ported step by step: derivations, decisions, mistakes, checklist |

The top-level [README](../README.md) gives the overview.
