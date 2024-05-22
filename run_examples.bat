@echo on
echo Run all examples
REM 3D
cargo r --example firework --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
cargo r --example portal --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
cargo r --example expr --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
cargo r --example spawn --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
cargo r --example multicam --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
cargo r --example visibility --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
cargo r --example random --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
cargo r --example spawn_on_command --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
cargo r --example activate --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/bevy_ui bevy/default_font 3d examples_world_inspector"
cargo r --example force_field --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
cargo r --example init --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
cargo r --example lifetime --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
cargo r --example ordering --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
cargo r --example ribbon --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr 3d examples_world_inspector"
REM 3D + PNG
cargo r --example gradient --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d examples_world_inspector"
cargo r --example circle --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d examples_world_inspector"
cargo r --example billboard --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d examples_world_inspector"
cargo r --example worms --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d examples_world_inspector"
cargo r --example instancing --no-default-features --features="bevy/bevy_winit bevy/bevy_pbr bevy/png 3d examples_world_inspector"
REM 2D
cargo r --example 2d --no-default-features --features="bevy/bevy_winit bevy/bevy_sprite 2d examples_world_inspector"
