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
    Attribute, BoxedModifier, ExprHandle, InitContext, InitModifier, Modifier, ModifierContext,
    Module, UpdateContext, UpdateModifier,
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
/// // of the particle.
/// let vel = module.prop("my_velocity");
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
        Self { attribute, value }
    }

    fn eval(
        &self,
        module: &mut Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        assert!(module.get(self.value).is_some());
        let attr = module.attr(self.attribute);
        let attr = context.eval(module, attr)?;
        let expr = context.eval(module, self.value)?;
        Ok(format!("{} = {};\n", attr, expr))
    }
}

#[typetag::serde]
impl Modifier for SetAttributeModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Init | ModifierContext::Update
    }

    fn as_init(&self) -> Option<&dyn InitModifier> {
        Some(self)
    }

    fn as_init_mut(&mut self) -> Option<&mut dyn InitModifier> {
        Some(self)
    }

    fn as_update(&self) -> Option<&dyn UpdateModifier> {
        Some(self)
    }

    fn as_update_mut(&mut self) -> Option<&mut dyn UpdateModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[Attribute] {
        std::slice::from_ref(&self.attribute)
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }
}

#[typetag::serde]
impl InitModifier for SetAttributeModifier {
    fn apply_init(&self, module: &mut Module, context: &mut InitContext) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.init_code += &code;
        Ok(())
    }
}

#[typetag::serde]
impl UpdateModifier for SetAttributeModifier {
    fn apply_update(
        &self,
        module: &mut Module,
        context: &mut UpdateContext,
    ) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.update_code += &code;
        Ok(())
    }
}
