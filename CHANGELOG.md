# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `SizeOverLifetimeModifier`.
- Add `PositionCircleModifier` to allow spawning from a circle or disc.
- Revamped spawning system:
    - `SpawnMode` is gone; `Spawner`s are constructed with associated functions `new`, `once`, `rate`, and `burst`.
    - Spawners can be reset with `Spawner::reset`. This gives control over when to spawn a burst of particles.
    - Spawners can be activated or deactivated with `Spawner::set_active`.
    - `ParticleEffectBundle`s can be initialized with a spawner with `ParticleEffectBundle::with_spawner`.

### Fixed

- Fixed depth sorting of particles relative to opaque objects. Particles are now correctly hidden when behind opaque objects.
- Fixed truncation in compute workgroup count preventing update of some particles, and in degenerate cases (`capacity < 64`) completely disabling update.

## [0.1.1] 2022-02-15

### Fixed

- Fix homepage link in `Cargo.toml`
- Bevy 0.6.1 fixed build on nightly, thereby fixing docs.rs builds

## [0.1.0] 2022-02-11

Initial alpha version. Lots of things missing, but the barebone functionality is there.
See the README.md for the list of planned and implemented features.
