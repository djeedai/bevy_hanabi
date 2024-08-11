export RUSTFLAGS := "--cfg=web_sys_unstable_apis"

_default:
    @just --list

[group('example')]
serve_examples: build_examples
    simple-http-server -c wasm,html,js -i --coop --coep --ip 127.0.0.1 examples/wasm

[group('example')]
build_examples: firework portal spawn expr multicam visibility random spawn_on_command force_field\
    example_init lifetime ordering ribbon gradient circle billboard worms instancing example_2d

_build_firework:
    cargo b --release --example firework --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_firework:
    wasm-bindgen --out-name wasm_firework --out-dir examples/wasm/target --target web --no-typescript target/wasm32-unknown-unknown/release/examples/firework.wasm

[group('3d')]
firework: _build_firework _bindgen_firework

_build_portal:
    cargo b --release --example portal --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_portal:
    wasm-bindgen --out-name wasm_portal --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/portal.wasm

[group('3d')]
portal: _build_portal _bindgen_portal

_build_expr:
    cargo b --release --example expr --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_expr:
    wasm-bindgen --out-name wasm_expr --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/expr.wasm

[group('3d')]
expr: _build_expr _bindgen_expr

_build_spawn:
    cargo b --release --example spawn --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_spawn:
    wasm-bindgen --out-name wasm_spawn --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/spawn.wasm

[group('3d')]
spawn: _build_spawn _bindgen_spawn

_build_multicam:
    cargo b --release --example multicam --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_multicam:
    wasm-bindgen --out-name wasm_multicam --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/multicam.wasm

[group('3d')]
multicam: _build_multicam _bindgen_multicam

_build_visibility:
    cargo b --release --example visibility --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_visibility:
    wasm-bindgen --out-name wasm_visibility --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/visibility.wasm

[group('3d')]
visibility: _build_visibility _bindgen_visibility

_build_random:
    cargo b --release --example random --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_random:
    wasm-bindgen --out-name wasm_random --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/random.wasm

[group('3d')]
random: _build_random _bindgen_random

_build_spawn_on_command:
    cargo b --release --example spawn_on_command --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_spawn_on_command:
    wasm-bindgen --out-name wasm_spawn_on_command --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/spawn_on_command.wasm

[group('3d')]
spawn_on_command: _build_spawn_on_command _bindgen_spawn_on_command

_build_activate:
    cargo b --release --example activate --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/bevy_ui bevy/default_font 3d"

_bindgen_activate:
    wasm-bindgen --out-name wasm_activate --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/activate.wasm

[group('3d')]
activate: _build_activate _bindgen_activate

_build_force_field:
    cargo b --release --example force_field --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_force_field:
    wasm-bindgen --out-name wasm_force_field --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/force_field.wasm

[group('3d')]
force_field: _build_force_field _bindgen_force_field

_build_init:
    cargo b --release --example init --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_init:
    wasm-bindgen --out-name wasm_init --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/init.wasm

[group('3d')]
example_init: _build_init _bindgen_init

_build_lifetime:
    cargo b --release --example lifetime --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_lifetime:
    wasm-bindgen --out-name wasm_lifetime --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/lifetime.wasm

[group('3d')]
lifetime: _build_lifetime _bindgen_lifetime

_build_ordering:
    cargo b --release --example ordering --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_ordering:
    wasm-bindgen --out-name wasm_ordering --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/ordering.wasm

[group('3d')]
ordering: _build_ordering _bindgen_ordering

_build_ribbon:
    cargo b --release --example ribbon --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"

_bindgen_ribbon:
    wasm-bindgen --out-name wasm_ribbon --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/ribbon.wasm

[group('3d')]
ribbon: _build_ribbon _bindgen_ribbon

_build_gradient:
    cargo b --release --example gradient --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"

_bindgen_gradient:
    wasm-bindgen --out-name wasm_gradient --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/gradient.wasm

[group('3d')]
[group('png')]
gradient: _build_gradient _bindgen_gradient

_build_circle:
    cargo b --release --example circle --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"

_bindgen_circle:
    wasm-bindgen --out-name wasm_circle --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/circle.wasm

[group('3d')]
[group('png')]
circle: _build_circle _bindgen_circle

_build_billboard:
    cargo b --release --example billboard --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"

_bindgen_billboard:
    wasm-bindgen --out-name wasm_billboard --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/billboard.wasm

[group('3d')]
[group('png')]
billboard: _build_billboard _bindgen_billboard

_build_worms:
    cargo b --release --example worms --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"

_bindgen_worms:
    wasm-bindgen --out-name wasm_worms --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/worms.wasm

[group('3d')]
[group('png')]
worms: _build_worms _bindgen_worms

_build_instancing:
    cargo b --release --example instancing --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"

_bindgen_instancing:
    wasm-bindgen --out-name wasm_instancing --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/instancing.wasm

[group('3d')]
[group('png')]
instancing: _build_instancing _bindgen_instancing

_build_2d:
    cargo b --release --example 2d --target wasm32-unknown-unknown --no-default-features --features="bevy/bevy_winit bevy/bevy_sprite 2d"

_bindgen_2d:
    wasm-bindgen --out-name wasm_2d --out-dir examples/wasm/target --target web target/wasm32-unknown-unknown/release/examples/2d.wasm

[group('2d')]
example_2d: _build_2d _bindgen_2d
