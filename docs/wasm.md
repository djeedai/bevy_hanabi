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

For more information about WebAssembly support, see also:

- Bevy's own explanations on how to run its examples
  on [WebGL2 and WebGPU](https://github.com/bevyengine/bevy/tree/main/examples#webgl2-and-webgpu).
- The [WebAssembly section](https://bevy-cheatbook.github.io/platforms/wasm.html)
  of the Unofficial Bevy Cheat Book.

## Limitations

The `serde` feature,
which allows deriving the `Serialize` and `Deserialize` traits on asset-related types,
is not compatible with the `wasm` target.
This is due to the use of the `typetag` dependency to handle trait objects,
which itself is not available for `wasm`.

To disable the `serde` feature,
simply disable default features and explicitly list the features you need:

```sh
cargo b --target wasm32-unknown-unknown --no-default-features --features="2d 3d"
```

## Running the examples

The simplest way is to make use of the `http-server` NPM package,
combined with `npx` from NodeJS to execute it.

- Install [NodeJS](https://nodejs.org/)
- Copy the `assets/` folder into `examples\wasm\`;
  this is a quick and simple workaround for the HTTP server to find the assets
- Execute `npx http-server examples\wasm`,
  which will run the `http-server` package without having to install it
- Open a compatible browser (e.g. Chrome or Edge) at `http://127.0.0.1:8080/`
  (or the address `http-server` will print in the console)

See also the [WebAssembly section](https://bevy-cheatbook.github.io/platforms/wasm.html)
of the Unofficial Bevy Cheat Book
for more tools and ways to build and run Bevy apps on Web.
