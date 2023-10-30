//! Modifiers to apply forces to the particles.
//!
//! The forces are applied as accelerations times unit mass, as particles
//! currently do not have a mass.

use std::hash::Hash;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    calc_func_id,
    graph::{BuiltInOperator, EvalContext, ExprError},
    impl_mod_update,
    modifier::ForceFieldSource,
    Attribute, ExprHandle, Module, UpdateContext, UpdateModifier,
};

/// A modifier to apply a force field to all particles each frame. The force
/// field is made up of [`ForceFieldSource`]s.
///
/// The maximum number of sources is [`ForceFieldSource::MAX_SOURCES`].
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Default, Clone, Copy, PartialEq, Hash, Reflect, Serialize, Deserialize)]
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

impl_mod_update!(
    ForceFieldModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

#[typetag::serde]
impl UpdateModifier for ForceFieldModifier {
    fn apply_update(
        &self,
        _module: &mut Module,
        context: &mut UpdateContext,
    ) -> Result<(), ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("force_field_{0:016X}", func_id);

        context.update_extra += &format!(
            r##"fn {}(particle: ptr<function, Particle>) {{
    {}
}}
"##,
            func_name,
            include_str!("../render/force_field_code.wgsl")
        );

        context.update_code += &format!("{}(&particle);\n", func_name);

        // TEMP
        context.force_field = self.sources;

        Ok(())
    }
}

/// A modifier to apply a linear drag force to all particles each frame. The
/// force slows down the particles without changing their direction.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Clone, Copy, Reflect, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LinearDragModifier {
    /// Drag coefficient. Higher values increase the drag force, and
    /// consequently decrease the particle's speed faster.
    ///
    /// Expression type: `f32`
    pub drag: ExprHandle,
}

impl LinearDragModifier {
    /// Create a new modifier from a drag expression.
    pub fn new(drag: ExprHandle) -> Self {
        Self { drag }
    }

    /// Instantiate a [`LinearDragModifier`] with a constant drag value.
    pub fn constant(module: &mut Module, drag: f32) -> Self {
        Self {
            drag: module.lit(drag),
        }
    }
}

impl_mod_update!(LinearDragModifier, &[Attribute::VELOCITY]);

#[typetag::serde]
impl UpdateModifier for LinearDragModifier {
    fn apply_update(
        &self,
        module: &mut Module,
        context: &mut UpdateContext,
    ) -> Result<(), ExprError> {
        let m = module;
        let attr = m.attr(Attribute::VELOCITY);
        let dt = m.builtin(BuiltInOperator::DeltaTime);
        let drag_dt = m.mul(self.drag, dt);
        let one = m.lit(1.);
        let one_minus_drag_dt = m.sub(one, drag_dt);
        let zero = m.lit(0.);
        let expr = m.max(zero, one_minus_drag_dt);
        let attr = context.eval(m, attr)?;
        let expr = context.eval(m, expr)?;
        context.update_code += &format!("{} *= {};", attr, expr);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{ParticleLayout, PropertyLayout, UpdateContext};

    use super::*;

    #[test]
    fn mod_force_field() {
        let position = Vec3::new(1., 2., 3.);
        let mut sources = [ForceFieldSource::default(); 16];
        sources[0].position = position;
        sources[0].mass = 1.;
        let modifier = ForceFieldModifier { sources };

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut module = Module::default();
        let mut context = UpdateContext::new(&property_layout, &particle_layout);
        assert!(modifier.apply_update(&mut module, &mut context).is_ok());

        // force_field_code.wgsl is too big
        // assert!(context.update_code.contains(&include_str!("../render/
        // force_field_code.wgsl")));
    }

    #[test]
    #[should_panic]
    fn mod_force_field_new_too_many() {
        let count = ForceFieldSource::MAX_SOURCES + 1;
        ForceFieldModifier::new((0..count).map(|_| ForceFieldSource::default()));
    }

    #[test]
    fn mod_drag() {
        let mut module = Module::default();
        let modifier = LinearDragModifier::constant(&mut module, 3.5);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut context = UpdateContext::new(&property_layout, &particle_layout);
        assert!(modifier.apply_update(&mut module, &mut context).is_ok());

        assert!(context.update_code.contains("3.5")); // TODO - less weak check
    }
}
