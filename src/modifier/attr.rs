//! Modifiers to manipulate a particle's attributes.
//!
//! Attribute modifiers initialize or update during simulation the attributes of
//! all particles of an effect. The provide the core functionality for
//! per-particle diversity / randomness, as attributes are the only quantities
//! stored per particle.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    graph::{EvalContext, ExprError},
    Attribute, BoxedModifier, ExprHandle, Modifier, ModifierContext, Module, ShaderWriter,
};

/// A modifier to assign a value to a particle attribute.
///
/// This modifier sets the value of an [`Attribute`] of the particle of a system
/// to a given graph expression, either when the particle spawns (if used as an
/// init modifier) or each frame when the particle is simulated (if used as an
/// update modifier).
///
/// This is a basic building block to create any complex effect. Most other init
/// modifiers are convenience helpers to achieve a behavior which can otherwise
/// be produced, more verbosely, by setting the individual attributes of a
/// particle with instances of this modifier.
///
/// # Example
///
/// ```
/// # use bevy::math::Vec3;
/// # use bevy_hanabi::*;
/// let mut module = Module::default();
///
/// // Set the position of the particle to (0,0,0) on spawn.
/// let pos = module.lit(Vec3::ZERO);
/// let init_pos = SetAttributeModifier::new(Attribute::POSITION, pos);
///
/// // Each frame, assign the value of the "my_velocity" property to the velocity
/// // of the particle. The property is assigned from CPU, and uploaded to GPU
/// // automatically when its value changed.
/// let my_velocity = module.add_property("my_velocity", Vec3::ZERO.into());
/// let vel = module.prop(my_velocity);
/// let update_vel = SetAttributeModifier::new(Attribute::VELOCITY, vel);
/// ```
///
/// # Warning
///
/// At the minute there is no validation that the type of the value is the same
/// as the type of the attribute. Users are advised to be careful, until more
/// safeguards are added.
///
/// # Attributes
///
/// This modifier requires the attribute specified in the `attribute` field.
#[derive(Debug, Clone, Copy, Reflect, Serialize, Deserialize)]
pub struct SetAttributeModifier {
    /// The attribute to initialize.
    ///
    /// See [`Attribute`] for the list of available attributes.
    pub attribute: Attribute,
    /// The initial value of the attribute.
    ///
    /// Expression type: same as the attribute.
    pub value: ExprHandle,
}

impl SetAttributeModifier {
    /// Create a new instance of a [`SetAttributeModifier`].
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy::math::Vec3;
    /// # use bevy_hanabi::*;
    /// let mut module = Module::default();
    /// let pos = module.lit(Vec3::ZERO);
    /// let init_pos = SetAttributeModifier::new(Attribute::POSITION, pos);
    /// ```
    pub fn new(attribute: Attribute, value: ExprHandle) -> Self {
        assert!(
            attribute != Attribute::ID,
            "The particle's ID is a read-only pseudo-attribute, cannot be assigned."
        );
        assert!(
            attribute != Attribute::PARTICLE_COUNTER,
            "The PARTICLE_COUNTER attribute is a read-only pseudo-attribute, cannot be assigned."
        );
        Self { attribute, value }
    }

    fn eval(
        &self,
        module: &mut Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        // Validate if possible that the expression produces the same type expected by
        // the attribute. This makes it much easier to diagnose issues.
        let expr = module.try_get(self.value)?;
        if let Some(value_type) = expr.value_type() {
            let attr_value_type = self.attribute.value_type();
            if value_type != attr_value_type {
                return Err(ExprError::TypeError(format!(
                    "Mismatching expression type in SetAttributeModifer: attribute '{}' requires an expression producing a value of type {}, but a value of type {} was produced instead",
                    self.attribute.name().to_uppercase(), attr_value_type, value_type)));
            }
        }

        // Generate the code
        let attr = module.attr(self.attribute);
        let attr = context.eval(module, attr)?;
        let expr = context.eval(module, self.value)?;
        Ok(format!("{} = {};\n", attr, expr))
    }
}

#[cfg_attr(feature = "serde", typetag::serde)]
impl Modifier for SetAttributeModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Init | ModifierContext::Update
    }

    fn attributes(&self) -> &[Attribute] {
        std::slice::from_ref(&self.attribute)
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn apply(&self, module: &mut Module, context: &mut ShaderWriter) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.main_code += &code;
        Ok(())
    }
}

/// Inherit the value of the given attribute when spawning a new particle.
///
/// This init modifier is used when spawning particles from other particles. The
/// attribute is inherited from the source particle by reading its value from
/// the previous frame's update pass and using that value to initialize the
/// particle being spawned. The source particle is the particle of the parent
/// effect which emitted the GPU spawn event this particle is spawned from. The
/// effect instance must have an [`EffectParent`] to declare its parent effect.
///
/// [`EffectParent`]: crate::EffectParent
#[derive(Debug, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize)]
pub struct InheritAttributeModifier {
    /// The attribute to inherit.
    ///
    /// See [`Attribute`] for the list of available attributes. The attribute
    /// must be present on the parent particle, and is added to the particles of
    /// the current effect too. During the init pass, the parent value is copied
    /// into the attribute of the newly spawned particle.
    pub attribute: Attribute,
}

impl InheritAttributeModifier {
    /// Create a new instance of a [`InheritAttributeModifier`].
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy::math::Vec3;
    /// # use bevy_hanabi::*;
    /// // Inherit the position of the parent particle which spawned this particle
    /// let inherit_pos = InheritAttributeModifier::new(Attribute::POSITION);
    /// ```
    pub fn new(attribute: Attribute) -> Self {
        Self { attribute }
    }

    fn eval(
        &self,
        module: &mut Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        let attr = module.attr(self.attribute);
        let attr = context.eval(module, attr)?;
        Ok(format!(
            "{} = parent_particle.{};\n",
            attr,
            self.attribute.name()
        ))
    }
}

#[cfg_attr(feature = "serde", typetag::serde)]
impl Modifier for InheritAttributeModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Init
    }

    fn attributes(&self) -> &[Attribute] {
        std::slice::from_ref(&self.attribute)
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn apply(&self, module: &mut Module, context: &mut ShaderWriter) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.main_code += &code;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::SetAttributeModifier;
    use crate::{
        Attribute, ExprError, ModifierContext, Module, ParticleLayout, PropertyLayout, ShaderWriter,
    };

    #[test]
    fn eval_validate() {
        let mut module = Module::default();
        let attr = Attribute::POSITION; // vec3<f32>
        let expr = module.lit(3.); // f32
        let attr = SetAttributeModifier::new(attr, expr);
        let property_layout = PropertyLayout::empty();
        let particle_layout = ParticleLayout::empty();
        let mut context =
            ShaderWriter::new(ModifierContext::Init, &property_layout, &particle_layout);
        assert!(matches!(
            attr.eval(&mut module, &mut context),
            Err(ExprError::TypeError(_))
        ));
    }
}
