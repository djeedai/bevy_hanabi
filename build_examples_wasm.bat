@echo on

echo Check that wasm-bindgen -V returns 0.2.92, otherwise cargo install wasm-bindgen-cli --version 0.2.92

echo Setting RUSTFLAGS to enable unstable web_sys APIs...
set RUSTFLAGS=--cfg=web_sys_unstable_apis

echo Build all examples for WASM...
REM 3D
cargo b --release --example firework --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example portal --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example expr --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example spawn --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example multicam --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example visibility --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example random --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example spawn_on_command --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example activate --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/bevy_ui bevy/default_font 3d"
cargo b --release --example force_field --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example init --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example lifetime --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example ordering --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo b --release --example ribbon --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
REM 3D + PNG
cargo b --release --example gradient --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
cargo b --release --example circle --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
cargo b --release --example billboard --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
cargo b --release --example worms --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
cargo b --release --example instancing --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
REM 2D
cargo b --release --example 2d --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_sprite 2d"

wasm-bindgen --out-name wasm_firework --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/firework.wasm
wasm-bindgen --out-name wasm_portal --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/portal.wasm
wasm-bindgen --out-name wasm_expr --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/expr.wasm
wasm-bindgen --out-name wasm_spawn --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/spawn.wasm
wasm-bindgen --out-name wasm_multicam --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/multicam.wasm
wasm-bindgen --out-name wasm_visibility --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/visibility.wasm
wasm-bindgen --out-name wasm_random --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/random.wasm
wasm-bindgen --out-name wasm_spawn_on_command --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/spawn_on_command.wasm
wasm-bindgen --out-name wasm_activate --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/activate.wasm
wasm-bindgen --out-name wasm_force_field --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/force_field.wasm
wasm-bindgen --out-name wasm_init --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/init.wasm
wasm-bindgen --out-name wasm_lifetime --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/lifetime.wasm
wasm-bindgen --out-name wasm_ordering --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/ordering.wasm
wasm-bindgen --out-name wasm_ribbon --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/ribbon.wasm
wasm-bindgen --out-name wasm_gradient --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/gradient.wasm
wasm-bindgen --out-name wasm_circle --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/circle.wasm
wasm-bindgen --out-name wasm_billboard --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/billboard.wasm
wasm-bindgen --out-name wasm_worms --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/worms.wasm
wasm-bindgen --out-name wasm_instancing --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/instancing.wasm
wasm-bindgen --out-name wasm_2d --no-typescript --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/2d.wasm

echo Done. See docs/wasm.md for help on running the examples locally.
