#!/usr/bin/env sh
set -eu

wasm_bindgen_version=0.2.129
if ! wasm-bindgen --version | grep -q "$wasm_bindgen_version"; then
    echo "Install wasm-bindgen-cli $wasm_bindgen_version" >&2
    exit 1
fi

export RUSTFLAGS="${RUSTFLAGS:-} --cfg=web_sys_unstable_apis"
cargo build --release --example demo --target wasm32-unknown-unknown --features bevy/webgpu
wasm-bindgen \
    --out-name wasm_demo \
    --out-dir examples/wasm/target \
    --target web \
    target/wasm32-unknown-unknown/release/examples/demo.wasm
