# Release process

- Update `CHANGELOG` with date and version
- Update `Cargo.toml` with version
- Update `README.md` and other images to point to github raw content at commit SHA1 of current HEAD
- `cargo fmt --all`
- `cargo build --features="2d"`
- `cargo build --features="3d"`
- `cargo clippy --workspace --all-targets --features="2d" -- -D warnings`
- `cargo clippy --workspace --all-targets --features="3d" -- -D warnings`
- `cargo test --features="2d"`
- `cargo test --features="3d"`
- `cargo +nightly doc --no-deps --features="3d"` (for `docs.rs`)
- `cargo +nightly build --features="3d"` (for `docs.rs`)
