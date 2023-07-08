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

use bevy::reflect::Reflect;
use serde::{Deserialize, Serialize};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

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
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Reflect, Serialize, Deserialize)]
pub enum ShapeDimension {
    /// Consider the surface of the shape only.
    #[default]
    Surface,
    /// Consider the entire shape volume.
    Volume,
}

/// Calculate a function ID by hashing the given value representative of the
/// function.
pub(crate) fn calc_func_id<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
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
    fn as_init(&self) -> Option<&dyn InitModifier> {
        None
    }

    /// Try to cast this modifier to an [`InitModifier`].
    fn as_init_mut(&mut self) -> Option<&mut dyn InitModifier> {
        None
    }

    /// Try to cast this modifier to an [`UpdateModifier`].
    fn as_update(&self) -> Option<&dyn UpdateModifier> {
        None
    }

    /// Try to cast this modifier to an [`UpdateModifier`].
    fn as_update_mut(&mut self) -> Option<&mut dyn UpdateModifier> {
        None
    }

    /// Try to cast this modifier to a [`RenderModifier`].
    fn as_render(&self) -> Option<&dyn RenderModifier> {
        None
    }

    /// Try to cast this modifier to a [`RenderModifier`].
    fn as_render_mut(&mut self) -> Option<&mut dyn RenderModifier> {
        None
    }

    /// Get the list of dependent attributes required for this modifier to be
    /// used.
    fn attributes(&self) -> &[Attribute];

    /// Attempt to resolve any property reference to the actual property in the
    /// effect.
    /// WARNING - For internal use only.
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

    fn make_test_modifier() -> InitPositionSphereModifier {
        InitPositionSphereModifier {
            center: Vec3::new(1., -3.5, 42.42),
            radius: 1.5,
            dimension: ShapeDimension::Surface,
        }
    }

    #[test]
    fn reflect() {
        let m = make_test_modifier();

        // Reflect
        let reflect: &dyn Reflect = m.as_reflect();
        assert!(reflect.is::<InitPositionSphereModifier>());
        let m_reflect = reflect
            .downcast_ref::<InitPositionSphereModifier>()
            .unwrap();
        assert_eq!(*m_reflect, m);
    }

    #[test]
    fn serde() {
        let m = make_test_modifier();
        let bm: BoxedModifier = Box::new(m);

        // Ser
        let s = ron::to_string(&bm).unwrap();
        println!("modifier: {:?}", s);

        // De
        let m_serde: BoxedModifier = ron::from_str(&s).unwrap();

        let rm: &dyn Reflect = m.as_reflect();
        let rm_serde: &dyn Reflect = m_serde.as_reflect();
        assert_eq!(
            rm.get_represented_type_info().unwrap().type_id(),
            rm_serde.get_represented_type_info().unwrap().type_id()
        );

        assert!(rm_serde.is::<InitPositionSphereModifier>());
        let rm_reflect = rm_serde
            .downcast_ref::<InitPositionSphereModifier>()
            .unwrap();
        assert_eq!(*rm_reflect, m);
    }
}
