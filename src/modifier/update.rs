//! Modifiers to influence the update behavior of particle each frame.
//!
//! The update modifiers control how particles are updated each frame by the
//! update compute shader.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{Attribute, Modifier, UpdateLayout};

/// Trait to customize the updating of existing particles each frame.
pub trait UpdateModifier: Modifier {
    /// Apply the modifier to the update layout of the effect instance.
    fn apply(&self, update_layout: &mut UpdateLayout);
}

/// A modifier to apply a constant acceleration to all particles each frame.
/// Used to simulate gravity.
#[derive(Default, Debug, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct AccelModifier {
    /// The constant acceleration to apply to all particles in the effect each
    /// frame.
    pub accel: Vec3,
}

impl Modifier for AccelModifier {
    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::VELOCITY]
    }
}

impl UpdateModifier for AccelModifier {
    fn apply(&self, layout: &mut UpdateLayout) {
        layout.accel = self.accel;
    }
}

/// A single attraction or repulsion source of a [`ForceFieldModifier`].
///
/// The source applies a radial force field to all particles around its
/// position, with a decreasing intensity the further away from the source the
/// particle is. This force is added to the one(s) of all the other active
/// sources of a [`ForceFieldModifier`].
#[derive(Debug, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct ForceFieldSource {
    /// Position of the source.
    pub position: Vec3,
    /// Maximum radius of the sphere of influence, outside of which
    /// the contribution of this source to the force field is null.
    pub max_radius: f32,
    /// Minimum radius of the sphere of influence, inside of which
    /// the contribution of this source to the force field is null, avoiding the
    /// singularity at the source position.
    pub min_radius: f32,
    /// The intensity of the force of the source is proportional to its mass.
    /// Note that the update shader will ignore all subsequent force field
    /// sources after it encountered a source with a mass of zero. To change
    /// the force from an attracting one to a repulsive one, simply
    /// set the mass to a negative value.
    pub mass: f32,
    /// The source force is proportional to `1 / distance^force_exponent`.
    pub force_exponent: f32,
    /// If `true`, the particles which attempt to come closer than
    /// [`min_radius`] from the source position will conform to a sphere of
    /// radius [`min_radius`] around the source position, appearing like a
    /// recharging effect.
    ///
    /// [`min_radius`]: ForceFieldSource::min_radius
    pub conform_to_sphere: bool,
}

impl Default for ForceFieldSource {
    fn default() -> Self {
        // defaults to no force field (a mass of 0)
        Self {
            position: Vec3::new(0., 0., 0.),
            min_radius: 0.1,
            max_radius: 0.0,
            mass: 0.,
            force_exponent: 0.0,
            conform_to_sphere: false,
        }
    }
}

impl ForceFieldSource {
    /// Maximum number of sources in the force field.
    pub const MAX_SOURCES: usize = 16;
}

/// A modifier to apply a force field to all particles each frame. The force
/// field is made up of [`ForceFieldSource`]s.
///
/// The maximum number of sources is [`ForceFieldSource::MAX_SOURCES`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct ForceFieldModifier {
    /// Array of force field sources.
    ///
    /// A source can be disabled by setting its [`mass`] to zero. In that case,
    /// all other sources located after it in the array are automatically
    /// disabled too.
    ///
    /// [`mass`]: ForceFieldSource::mass
    pub sources: [ForceFieldSource; ForceFieldSource::MAX_SOURCES],
}

impl ForceFieldModifier {
    /// Instantiate a [`ForceFieldModifier`] with a set of sources.
    ///
    /// # Panics
    ///
    /// Panics if the number of sources exceeds [`MAX_SOURCES`].
    ///
    /// [`MAX_SOURCES`]: ForceFieldSource::MAX_SOURCES
    pub fn new<T>(sources: T) -> Self
    where
        T: IntoIterator<Item = ForceFieldSource>,
    {
        let mut source_array = [ForceFieldSource::default(); ForceFieldSource::MAX_SOURCES];

        for (index, p_attractor) in sources.into_iter().enumerate() {
            if index >= ForceFieldSource::MAX_SOURCES {
                panic!(
                    "Force field source count exceeded maximum of {}",
                    ForceFieldSource::MAX_SOURCES
                );
            }
            source_array[index] = p_attractor;
        }

        Self {
            sources: source_array,
        }
    }

    /// Overwrite the source at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than or equal to [`MAX_SOURCES`].
    ///
    /// [`MAX_SOURCES`]: ForceFieldSource::MAX_SOURCES
    pub fn add_or_replace(&mut self, source: ForceFieldSource, index: usize) {
        self.sources[index] = source;
    }
}

impl Modifier for ForceFieldModifier {
    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::POSITION, Attribute::VELOCITY]
    }
}

impl UpdateModifier for ForceFieldModifier {
    fn apply(&self, layout: &mut UpdateLayout) {
        layout.force_field = self.sources;
    }
}

/// A modifier to apply a linear drag force to all particles each frame. The
/// force slows down the particles without changing their direction.
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct LinearDragModifier {
    /// Drag coefficient. Higher values increase the drag force, and
    /// consequently decrease the particle's speed faster.
    pub drag: f32,
}

impl LinearDragModifier {
    /// Instantiate a [`LinearDragModifier`].
    pub fn new(drag: f32) -> Self {
        Self { drag }
    }
}

impl Modifier for LinearDragModifier {
    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::VELOCITY]
    }
}

impl UpdateModifier for LinearDragModifier {
    fn apply(&self, layout: &mut UpdateLayout) {
        layout.drag_coefficient = self.drag;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn mod_accel() {
        let accel = Vec3 {
            x: 1.,
            y: 2.,
            z: 3.,
        };
        let modifier = AccelModifier { accel };

        let mut layout = UpdateLayout::default();
        modifier.apply(&mut layout);

        assert_eq!(layout.accel, accel);
    }

    #[test]
    fn mod_force_field() {
        let position = Vec3 {
            x: 1.,
            y: 2.,
            z: 3.,
        };
        let mut sources = [ForceFieldSource::default(); 16];
        sources[0].position = position;
        sources[0].mass = 1.;
        let modifier = ForceFieldModifier { sources };

        let mut layout = UpdateLayout::default();
        modifier.apply(&mut layout);

        assert!(layout.force_field[0]
            .position
            .abs_diff_eq(sources[0].position, 1e-5));
    }

    #[test]
    fn mod_force_field_new() {
        let modifier = ForceFieldModifier::new((0..10).map(|i| {
            let position = Vec3 {
                x: i as f32,
                y: 0.,
                z: 0.,
            };
            ForceFieldSource {
                position,
                mass: 1.,
                ..default()
            }
        }));

        let mut layout = UpdateLayout::default();
        modifier.apply(&mut layout);

        for i in 0..16 {
            assert!(layout.force_field[i].position.abs_diff_eq(
                Vec3 {
                    x: if i < 10 { i as f32 } else { 0. },
                    y: 0.,
                    z: 0.,
                },
                1e-5
            ));

            let mass = if i < 10 { 1. } else { 0. };
            assert_approx_eq!(layout.force_field[i].mass, mass);
        }
    }

    #[test]
    #[should_panic]
    fn mod_force_field_new_too_many() {
        let count = ForceFieldSource::MAX_SOURCES + 1;
        ForceFieldModifier::new((0..count).map(|_| ForceFieldSource::default()));
    }

    #[test]
    fn mod_drag() {
        let modifier = LinearDragModifier { drag: 3.5 };

        let mut layout = UpdateLayout::default();
        modifier.apply(&mut layout);

        assert_eq!(layout.drag_coefficient, 3.5);
    }
}
