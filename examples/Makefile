# quickly run all examples at the same time with `make -j17`
CARGO_FLAGS ?= --release --offline

# default flags assume "--offline", so if your build fails remove the "--offline" flag

default: all-examples

run-%:
	cargo r --example $* ${CARGO_FLAGS} --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png bevy/bevy_sprite 3d"

build-%:
	cargo build --example $* --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png bevy/bevy_sprite 3d"
	cargo build --example $* --release --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png bevy/bevy_sprite 3d"

run-2d:
	cargo r --example 2d ${CARGO_FLAGS} --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png bevy/bevy_sprite 2d"

build-2d:
	cargo build --example 2d --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png bevy/bevy_sprite 2d"
	cargo build --example 2d --release --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png bevy/bevy_sprite 2d"

all-examples: run-firework\
               run-portal \
               run-expr \
               run-spawn \
               run-multicam \
               run-visibility \
               run-random \
               run-spawn_on_command \
               run-activate \
               run-force_field \
               run-lifetime \
               run-instancing \
               run-gradient \
               run-billboard \
               run-2d

broken-in-debug-examples: run-init \
                          run-circle
