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
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Reflect, FromReflect, Serialize, Deserialize,
)]
pub enum ShapeDimension {
    /// Consider the surface of the shape only.
    #[default]
    Surface,
    /// Consider the entire shape volume.
    Volume,
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
    /// [`InitSizeModifier`]
    InitSize(InitSizeModifier),

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
            Modifiers::InitSize(modifier) => modifier,

            Modifiers::Accel(modifier) => modifier,
            Modifiers::ForceField(modifier) => modifier,
            Modifiers::LinearDrag(modifier) => modifier,

            Modifiers::ParticleTexture(modifier) => modifier,
            Modifiers::ColorOverLifetime(modifier) => modifier,
            Modifiers::SizeOverLifetime(modifier) => modifier,
            Modifiers::Billboard(modifier) => modifier,
        }
    }

    /// Cast the enum value to an [`InitModifier`] if possible.
    pub fn init_modifier(&self) -> Option<&dyn InitModifier> {
        match self {
            Modifiers::PositionCircle(modifier) => Some(modifier),
            Modifiers::PositionSphere(modifier) => Some(modifier),
            Modifiers::PositionCone3d(modifier) => Some(modifier),
            Modifiers::ParticleLifetime(modifier) => Some(modifier),
            Modifiers::InitSize(modifier) => Some(modifier),

            _ => None,
        }
    }
}

/// Implement `From<T> for Modifiers` for a [`Modifier`] type.
macro_rules! impl_modifier {
    ($e:ident, $t:ty) => {
        impl From<$t> for $crate::Modifiers {
            fn from(modifier: $t) -> Self {
                Self::$e(modifier)
            }
        }
    };
}

impl_modifier!(PositionCircle, PositionCircleModifier);
impl_modifier!(PositionSphere, PositionSphereModifier);
impl_modifier!(PositionCone3d, PositionCone3dModifier);
impl_modifier!(ParticleLifetime, ParticleLifetimeModifier);
impl_modifier!(InitSize, InitSizeModifier);

impl_modifier!(Accel, AccelModifier);
impl_modifier!(ForceField, ForceFieldModifier);
impl_modifier!(LinearDrag, LinearDragModifier);

impl_modifier!(ParticleTexture, ParticleTextureModifier);
impl_modifier!(ColorOverLifetime, ColorOverLifetimeModifier);
impl_modifier!(SizeOverLifetime, SizeOverLifetimeModifier);
impl_modifier!(Billboard, BillboardModifier);

#[cfg(test)]
mod tests {
    use bevy::prelude::*;

    use super::*;

    fn make_test_modifier() -> Modifiers {
        PositionSphereModifier {
            center: Vec3::ZERO,
            radius: 1.5,
            speed: 3.5.into(),
            dimension: ShapeDimension::Surface,
        }
        .into()
    }

    #[test]
    fn reflect() {
        let m = make_test_modifier();

        // Reflect
        let reflect: &dyn Reflect = &m;
        assert!(reflect.is::<Modifiers>());
        let m_reflect = reflect.downcast_ref::<Modifiers>().unwrap();
        assert_eq!(*m_reflect, m);
    }

    #[test]
    fn serde() {
        let m = make_test_modifier();
        let s = ron::to_string(&m).unwrap();
        println!("modifier: {:?}", s);
        let m_serde: Modifiers = ron::from_str(&s).unwrap();
        assert_eq!(m, m_serde);
    }
}
