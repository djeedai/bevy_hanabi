# Release process

- Update `CHANGELOG` with date and version
- Update `Cargo.toml` with version
- Update `README.md` and other images to point to github raw content at commit SHA1 of current HEAD
- Update Bevy tracking version at top of `README.md` if needed

- `cargo fmt --all`

- `cargo build`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test`

- `cargo build --no-default-features --features="2d"`
- `cargo clippy --workspace --all-targets --no-default-features --features="2d" -- -D warnings`
- `cargo test --no-default-features --features="2d gpu_tests"`

- `cargo build --no-default-features --features="3d"`
- `cargo clippy --workspace --all-targets --no-default-features --features="3d" -- -D warnings`
- `cargo test --no-default-features --features="3d gpu_tests"`

- `cargo +nightly build --all-features` (for `docs.rs`)
- `cargo +nightly doc --no-deps --all-features` (for `docs.rs`)
