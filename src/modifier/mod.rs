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

use crate::{Attribute, Property};

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

/// Context a modifier applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModifierContext {
    /// Particle initializing.
    Init,
    /// Particle simulation (update).
    Update,
    /// Particle rendering.
    Render,
}

/// Trait describing a modifier customizing an effect pipeline.
#[typetag::serde]
pub trait Modifier: Reflect + Send + Sync + 'static {
    /// Get the context this modifier applies to.
    fn context(&self) -> ModifierContext;

    /// Try to cast this modifier to an [`InitModifier`].
    fn init(&self) -> Option<&dyn InitModifier> {
        None
    }

    /// Try to cast this modifier to an [`InitModifier`].
    fn init_mut(&mut self) -> Option<&mut dyn InitModifier> {
        None
    }

    /// Try to cast this modifier to an [`UpdateModifier`].
    fn update(&self) -> Option<&dyn UpdateModifier> {
        None
    }

    /// Try to cast this modifier to an [`UpdateModifier`].
    fn update_mut(&mut self) -> Option<&mut dyn UpdateModifier> {
        None
    }

    /// Try to cast this modifier to a [`RenderModifier`].
    fn render(&self) -> Option<&dyn RenderModifier> {
        None
    }

    /// Try to cast this modifier to a [`RenderModifier`].
    fn render_mut(&mut self) -> Option<&mut dyn RenderModifier> {
        None
    }

    /// Get the list of dependent attributes required for this modifier to be
    /// used.
    fn attributes(&self) -> &[&'static Attribute];

    /// Attempt to resolve any property reference to the actual property in the
    /// effect.
    fn resolve_properties(&mut self, _properties: &[Property]) {}

    /// Clone self.
    fn boxed_clone(&self) -> BoxedModifier;
}

/// Boxed version of [`Modifier`].
pub type BoxedModifier = Box<dyn Modifier>;

impl Clone for BoxedModifier {
    fn clone(&self) -> Self {
        self.boxed_clone()
    }
}

#[cfg(test)]
mod tests {
    use bevy::prelude::*;

    use super::*;

    fn make_test_modifier() -> BoxedModifier {
        Box::new(PositionSphereModifier {
            center: Vec3::ZERO,
            radius: 1.5,
            speed: 3.5.into(),
            dimension: ShapeDimension::Surface,
        })
    }

    // #[test]
    // fn reflect() {
    //     let m = make_test_modifier();

    //     // Reflect
    //     let reflect: &dyn Reflect = &m;
    //     assert!(reflect.is::<Modifiers>());
    //     let m_reflect = reflect.downcast_ref::<Modifiers>().unwrap();
    //     assert_eq!(*m_reflect, m);
    // }

    #[test]
    fn serde() {
        let m = make_test_modifier();
        let s = ron::to_string(&m).unwrap();
        println!("modifier: {:?}", s);
        let _m_serde: BoxedModifier = ron::from_str(&s).unwrap();
        //assert_eq!(m, m_serde);
    }
}
