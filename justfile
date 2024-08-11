export RUSTFLAGS := "--cfg=web_sys_unstable_apis"

_default:
    @just --list

[group('example')]
serve_examples: build_examples
    simple-http-server -c wasm,html,js -i --coop --coep --ip 127.0.0.1 examples/wasm

[group('example')]
build_examples: firework portal spawn expr multicam visibility random spawn_on_command force_field\
    example_init lifetime ordering ribbon gradient circle billboard worms instancing example_2d

_build_example name *FEATURES:
    cargo b --release --example {{name}} --target wasm32-unknown-unknown --no-default-features --features="{{FEATURES}}"

_bindgen_example name:
    wasm-bindgen --out-name wasm_{{name}} --out-dir examples/wasm/target --target web --no-typescript target/wasm32-unknown-unknown/release/examples/{{name}}.wasm

_build_firework: (_build_example "firework" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_firework: (_bindgen_example "firework")
[group('3d')]
firework: _build_firework _bindgen_firework

_build_portal: (_build_example "portal" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_portal: (_bindgen_example "portal")
[group('3d')]
portal: _build_portal _bindgen_portal

_build_expr: (_build_example "expr" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_expr: (_bindgen_example "expr")
[group('3d')]
expr: _build_expr _bindgen_expr

_build_spawn: (_build_example "spawn" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_spawn: (_bindgen_example "spawn")
[group('3d')]
spawn: _build_spawn _bindgen_spawn

_build_multicam: (_build_example "multicam" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_multicam: (_bindgen_example "multicam")
[group('3d')]
multicam: _build_multicam _bindgen_multicam

_build_visibility: (_build_example "visibility" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_visibility: (_bindgen_example "visibility")
[group('3d')]
visibility: _build_visibility _bindgen_visibility

_build_random: (_build_example "random" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_random: (_bindgen_example "random")
[group('3d')]
random: _build_random _bindgen_random

_build_spawn_on_command: (_build_example "spawn_on_command" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_spawn_on_command: (_bindgen_example "spawn_on_command")
[group('3d')]
spawn_on_command: _build_spawn_on_command _bindgen_spawn_on_command

_build_activate: (_build_example "activate" "bevy/bevy_winit" "bevy/bevy_pbr" "bevy/bevy_ui" "bevy/default_font" "3d")
_bindgen_activate: (_bindgen_example "activate")
[group('3d')]
activate: _build_activate _bindgen_activate

_build_force_field: (_build_example "force_field" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_force_field: (_bindgen_example "force_field")
[group('3d')]
force_field: _build_force_field _bindgen_force_field

_build_init: (_build_example "init" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_init: (_bindgen_example "init")
[group('3d')]
example_init: _build_init _bindgen_init

_build_lifetime: (_build_example "lifetime" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_lifetime: (_bindgen_example "lifetime")
[group('3d')]
lifetime: _build_lifetime _bindgen_lifetime

_build_ordering: (_build_example "ordering" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_ordering: (_bindgen_example "ordering")
[group('3d')]
ordering: _build_ordering _bindgen_ordering

_build_ribbon: (_build_example "ribbon" "bevy/bevy_winit" "bevy/bevy_pbr" "3d")
_bindgen_ribbon: (_bindgen_example "ribbon")
[group('3d')]
ribbon: _build_ribbon _bindgen_ribbon

_build_gradient: (_build_example "gradient" "bevy/bevy_winit" "bevy/bevy_pbr" "bevy/png" "3d")
_bindgen_gradient: (_bindgen_example "gradient")
[group('3d')]
[group('png')]
gradient: _build_gradient _bindgen_gradient

_build_circle: (_build_example "circle" "bevy/bevy_winit" "bevy/bevy_pbr" "bevy/png" "3d")
_bindgen_circle: (_bindgen_example "circle")
[group('3d')]
[group('png')]
circle: _build_circle _bindgen_circle

_build_billboard: (_build_example "billboard" "bevy/bevy_winit" "bevy/bevy_pbr" "bevy/png" "3d")
_bindgen_billboard: (_bindgen_example "billboard")
[group('3d')]
[group('png')]
billboard: _build_billboard _bindgen_billboard

_build_worms: (_build_example "worms" "bevy/bevy_winit" "bevy/bevy_pbr" "bevy/png" "3d")
_bindgen_worms: (_bindgen_example "worms")
[group('3d')]
[group('png')]
worms: _build_worms _bindgen_worms

_build_instancing: (_build_example "instancing" "bevy/bevy_winit" "bevy/bevy_pbr" "bevy/png" "3d")
_bindgen_instancing: (_bindgen_example "instancing")
[group('3d')]
[group('png')]
instancing: _build_instancing _bindgen_instancing

_build_2d: (_build_example "2d" "bevy/bevy_winit" "bevy/bevy_sprite" "2d")
_bindgen_2d: (_bindgen_example "2d")
[group('2d')]
example_2d: _build_2d _bindgen_2d
