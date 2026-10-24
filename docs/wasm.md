# WebAssembly support

tl;dr: WebAssembly is supported as of 🎆 Hanabi 0.13.

## Overview

⚠️ _Read carefully for limitations. This is not specific to 🎆 Hanabi, but extends to Bevy in general._

The WebAssembly target (_a.k.a._ "wasm"; cargo's `--target wasm32-unknown-unknown`) is supported.
However, because 🎆 Hanabi makes heavy usage of compute shaders,
like several other advanced Bevy features,
it **requires using the WebGPU render backend of Bevy instead of the WebGL2 one**.
This has limiting implications which app authors should understand before using 🎆 Hanabi.
In particular, as of 2024:

- Chrome and Edge only enable WebGPU from version 113 (April 2023);
- Firefox still disables WebGPU by default,
  and either requires a nightly build or the user changing its internal flags.

Mozilla has a [Browser compatibility](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API#browser_compatibility) table
to understand where and how WebGPU is available on web browsers.

**In addition, you need to enable the `bevy/webgpu` feature.** Otherwise you will get some error:

```txt
No wgpu backend feature that is implemented for the target platform was enabled. See `wgpu::Instance::enabled_backend_features()` for more information.
```

For more information about WebAssembly support, see also:

- Bevy's own explanations on how to run its examples
  on [WebGL2 and WebGPU](https://github.com/bevyengine/bevy/tree/main/examples#webgl2-and-webgpu).
- The [WebAssembly section](https://bevy-cheatbook.github.io/platforms/wasm.html)
  of the Unofficial Bevy Cheat Book.

## Running the examples

Build the unified example runner for WebAssembly and serve it over HTTP:

- Install [NodeJS](https://nodejs.org/)
- Run `./build_examples_wasm.sh` (or `build_examples_wasm.bat` on Windows)
- Execute `npx http-server examples\wasm`,
  which will run the `http-server` package without having to install it
- Open `http://127.0.0.1:8080/` in a compatible browser (e.g. Chrome or Edge)
- Choose an example from the in-app menu

See also the [WebAssembly section](https://bevy-cheatbook.github.io/platforms/wasm.html)
of the Unofficial Bevy Cheat Book
for more tools and ways to build and run Bevy apps on Web.

The desktop runner also accepts `--demo <id>` to skip the menu.
