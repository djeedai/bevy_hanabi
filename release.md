# Release process

- Update `CHANGELOG` with date and version
- Update `Cargo.toml` with version
- Update `README.md` and other images to point to github raw content at commit SHA1 of current HEAD
- `cargo fmt --all`
- `cargo build`
- `cargo build --no-default-features --features="2d"`
- `cargo build --no-default-features --features="3d"`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo clippy --workspace --all-targets --no-default-features --features="2d" -- -D warnings`
- `cargo clippy --workspace --all-targets --no-default-features --features="3d" -- -D warnings`
- `cargo test`
- `cargo test --no-default-features --features="2d"`
- `cargo test --no-default-features --features="3d"`
- `cargo +nightly doc --no-deps --all-features` (for `docs.rs`)
- `cargo +nightly build --all-features` (for `docs.rs`)
