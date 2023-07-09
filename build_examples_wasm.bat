@echo on

echo Setting RUSTFLAGS to enable unstable web_sys APIs
set RUSTFLAGS=--cfg=web_sys_unstable_apis

echo Build all examples for WASM
REM 3D
cargo b --release --example firework --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example portal --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example expr --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example spawn --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example multicam --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example visibility --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example random --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example spawn_on_command --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example activate --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example force_field --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example init --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example lifetime --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example instancing --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
REM 3D + PNG
cargo b --release --example gradient --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
cargo b --release --example circle --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
cargo b --release --example billboard --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
REM 2D
cargo b --release --example 2d --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_sprite 2d"

echo Bindgen all examples
wasm-bindgen --out-name wasm_firework --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/firework.wasm
wasm-bindgen --out-name wasm_portal --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/portal.wasm
wasm-bindgen --out-name wasm_expr --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/expr.wasm
wasm-bindgen --out-name wasm_spawn --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/spawn.wasm
wasm-bindgen --out-name wasm_multicam --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/multicam.wasm
wasm-bindgen --out-name wasm_visibility --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/visibility.wasm
wasm-bindgen --out-name wasm_random --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/random.wasm
wasm-bindgen --out-name wasm_spawn_on_command --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/spawn_on_command.wasm
wasm-bindgen --out-name wasm_activate --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/activate.wasm
wasm-bindgen --out-name wasm_force_field --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/force_field.wasm
wasm-bindgen --out-name wasm_init --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/init.wasm
wasm-bindgen --out-name wasm_lifetime --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/lifetime.wasm
wasm-bindgen --out-name wasm_instancing --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/instancing.wasm
wasm-bindgen --out-name wasm_gradient --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/gradient.wasm
wasm-bindgen --out-name wasm_circle --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/circle.wasm
wasm-bindgen --out-name wasm_billboard --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/billboard.wasm
wasm-bindgen --out-name wasm_2d --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/2d.wasm
