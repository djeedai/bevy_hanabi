# Setup

🎆 Hanabi is provided through the `bevy_hanabi` Rust package, published on [crates.io](https://crates.io/crates/bevy_hanabi). Installation into a library or executable follows the usual `cargo` process:

```sh
cargo add bevy_hanabi
```

Once the dependency is added, register the [`HanabiPlugin`](https://docs.rs/bevy_hanabi/latest/bevy_hanabi/struct.HanabiPlugin.html) with your Bevy app.

```rust
use bevy::prelude::*;
use bevy_hanabi::prelude::*;

App::default()
    // Default Bevy plugins
    .add_plugins(DefaultPlugins)
    // Hanabi VFX
    .add_plugins(HanabiPlugin)
    .run();
```

For a version compatibility list with Bevy's versions, see the [`README.md`](https://github.com/djeedai/bevy_hanabi/blob/main/README.md#compatible-bevy-versions).
