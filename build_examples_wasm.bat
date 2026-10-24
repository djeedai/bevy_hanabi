@echo off
setlocal

set "WASM_BINDGEN_VERSION=0.2.129"
wasm-bindgen --version | findstr /C:"%WASM_BINDGEN_VERSION%" >nul
if errorlevel 1 (
    echo Install wasm-bindgen-cli %WASM_BINDGEN_VERSION% 1>&2
    exit /b 1
)

set RUSTFLAGS=--cfg web_sys_unstable_apis --cfg getrandom_backend="wasm_js"

cargo build --release --example demo --target wasm32-unknown-unknown --features="bevy/webgpu"
if errorlevel 1 exit /b %errorlevel%

wasm-bindgen --out-name wasm_demo --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/demo.wasm
if errorlevel 1 exit /b %errorlevel%

echo Done. Serve examples/wasm and open index.html.
endlocal
