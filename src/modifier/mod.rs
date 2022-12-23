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

use bevy::reflect::{FromReflect, Reflect};
use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect, FromReflect, Serialize, Deserialize)]
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

/// Enumeration of all the possible modifiers.
///
/// This is mainly a workaround for the impossibility to currently serialize and
/// deserialize some trait object in Bevy, which makes it impossible to directly
/// hold a `Box<dyn Modifier>` in an `EffectAsset`. Instead, we use an enum to
/// explicitly list the types and make the field a sized one that can be
/// serialized and deserialized.
#[derive(Debug, Clone, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Modifiers {
    /// [`PositionCircleModifier`].
    PositionCircle(PositionCircleModifier),
    /// [`PositionSphereModifier`].
    PositionSphere(PositionSphereModifier),
    /// [`PositionCone3dModifier`].
    PositionCone3d(PositionCone3dModifier),
    /// [`ParticleLifetimeModifier`].
    ParticleLifetime(ParticleLifetimeModifier),

    /// [`AccelModifier`].
    Accel(AccelModifier),
    /// [`ForceFieldModifier`].
    ForceField(ForceFieldModifier),
    /// [`LinearDragModifier`].
    LinearDrag(LinearDragModifier),

    /// [`ParticleTextureModifier`].
    ParticleTexture(ParticleTextureModifier),
    /// [`ColorOverLifetimeModifier`].
    ColorOverLifetime(ColorOverLifetimeModifier),
    /// [`SizeOverLifetimeModifier`].
    SizeOverLifetime(SizeOverLifetimeModifier),
    /// [`BillboardModifier`].
    Billboard(BillboardModifier),
}

impl Modifiers {
    /// Cast the enum value to the [`Modifier`] trait.
    pub fn modifier(&self) -> &dyn Modifier {
        match self {
            Modifiers::PositionCircle(modifier) => modifier,
            Modifiers::PositionSphere(modifier) => modifier,
            Modifiers::PositionCone3d(modifier) => modifier,
            Modifiers::ParticleLifetime(modifier) => modifier,
            Modifiers::Accel(modifier) => modifier,
            Modifiers::ForceField(modifier) => modifier,
            Modifiers::LinearDrag(modifier) => modifier,
            Modifiers::ParticleTexture(modifier) => modifier,
            Modifiers::ColorOverLifetime(modifier) => modifier,
            Modifiers::SizeOverLifetime(modifier) => modifier,
            Modifiers::Billboard(modifier) => modifier,
        }
    }
}

impl From<PositionCircleModifier> for Modifiers {
    fn from(modifier: PositionCircleModifier) -> Self {
        Self::PositionCircle(modifier)
    }
}

impl From<PositionSphereModifier> for Modifiers {
    fn from(modifier: PositionSphereModifier) -> Self {
        Self::PositionSphere(modifier)
    }
}

impl From<PositionCone3dModifier> for Modifiers {
    fn from(modifier: PositionCone3dModifier) -> Self {
        Self::PositionCone3d(modifier)
    }
}

impl From<ParticleLifetimeModifier> for Modifiers {
    fn from(modifier: ParticleLifetimeModifier) -> Self {
        Self::ParticleLifetime(modifier)
    }
}

impl From<AccelModifier> for Modifiers {
    fn from(modifier: AccelModifier) -> Self {
        Self::Accel(modifier)
    }
}

impl From<ForceFieldModifier> for Modifiers {
    fn from(modifier: ForceFieldModifier) -> Self {
        Self::ForceField(modifier)
    }
}

impl From<LinearDragModifier> for Modifiers {
    fn from(modifier: LinearDragModifier) -> Self {
        Self::LinearDrag(modifier)
    }
}

impl From<ParticleTextureModifier> for Modifiers {
    fn from(modifier: ParticleTextureModifier) -> Self {
        Self::ParticleTexture(modifier)
    }
}

impl From<ColorOverLifetimeModifier> for Modifiers {
    fn from(modifier: ColorOverLifetimeModifier) -> Self {
        Self::ColorOverLifetime(modifier)
    }
}

impl From<SizeOverLifetimeModifier> for Modifiers {
    fn from(modifier: SizeOverLifetimeModifier) -> Self {
        Self::SizeOverLifetime(modifier)
    }
}

impl From<BillboardModifier> for Modifiers {
    fn from(modifier: BillboardModifier) -> Self {
        Self::Billboard(modifier)
    }
}
