@echo on
echo Run all examples
cargo r --example spawn --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo r --example random --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo r --example gradient --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
cargo r --example circle --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d"
cargo r --example spawn_on_command --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo r --example activate --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo r --example force_field --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"
cargo r --example 2d --no-default-features --features="bevy/bevy_winit bevy/bevy_sprite 2d"
cargo r --example lifetime --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d"