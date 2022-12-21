//! Building blocks to create a visual effect.
//!
//! A **modifier** is a building block used to create effects. Particles effects
//! are composed of multiple modifiers, which put together and configured
//! produce the desired visual effect. Each modifier changes a specific part of
//! the behavior of an effect. Modifiers are grouped in three categories:
//!
//! - **Init modifiers** influence the initializing of particles when they
//!   spawn. They typically configure the initial position and/or velocity of
//!   particles. Init modifiers are grouped under the [`init`] module, and
//!   implement the [`InitModifier`] trait.
//! - **Update modifiers** influence the particle update loop each frame. For
//!   example, an update modifier can apply a gravity force to all particles.
//!   Update modifiers are grouped under the [`update`] module, and implement
//!   the [`UpdateModifier`] trait.
//! - **Render modifiers** influence the rendering of each particle. They can
//!   change the particle's color, or orient it to face the camera. Render
//!   modifiers are grouped under the [`render`] module, and implement the
//!   [`RenderModifier`] trait.
//!
//! [`InitModifier`]: crate::modifier::init::InitModifier
//! [`UpdateModifier`]: crate::modifier::update::UpdateModifier
//! [`RenderModifier`]: crate::modifier::render::RenderModifier

pub mod init;
pub mod render;
pub mod update;

pub use init::*;
pub use render::*;
pub use update::*;

use crate::Attribute;

/// The dimension of a shape to consider.
///
/// The exact meaning depends on the context where this enum is used.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ShapeDimension {
    /// Consider the surface of the shape only.
    Surface,
    /// Consider the entire shape volume.
    Volume,
}

impl Default for ShapeDimension {
    fn default() -> Self {
        Self::Surface
    }
}

/// Trait describing a modifier customizing an effect pipeline.
pub trait Modifier {
    /// Get the list of dependent attributes required for this modifier to be
    /// used.
    fn attributes(&self) -> &[&'static Attribute];
}
