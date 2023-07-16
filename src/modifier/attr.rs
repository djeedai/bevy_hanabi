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
    UpdateContext, UpdateModifier,
};

/// A modifier to set the initial value of any particle attribute.
///
/// This modifier initializes an [`Attribute`] of the particle of a system to a
/// given graph expression when the particle spawns.
///
/// This is a basic building block to create any complex effect. Most other init
/// modifiers are convenience helpers to achieve a behavior which can otherwise
/// be produced, more verbosely, by setting the individual attributes of a
/// particle with instances of this modifier.
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
    /// The name of the attribute to initialize.
    pub attribute: Attribute,
    /// The initial value of the attribute.
    ///
    /// Expression type: same as the attribute.
    pub value: ExprHandle,
}

impl SetAttributeModifier {
    /// Create a new instance of an [`SetAttributeModifier`].
    pub fn new(attribute: Attribute, value: ExprHandle) -> Self {
        Self { attribute, value }
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
    fn apply_init(&self, context: &mut InitContext) -> Result<(), ExprError> {
        assert!(context.module.get(self.value).is_some());
        let attr = context.module.attr(self.attribute);
        let attr = context.eval(attr)?;
        let expr = context.eval(self.value)?;
        context.init_code += &format!("{} = {};\n", attr, expr);
        Ok(())
    }
}

#[typetag::serde]
impl UpdateModifier for SetAttributeModifier {
    fn apply_update(&self, context: &mut UpdateContext) -> Result<(), ExprError> {
        assert!(context.module.get(self.value).is_some());
        let attr = context.module.attr(self.attribute);
        let attr = context.eval(attr)?;
        let expr = context.eval(self.value)?;
        context.update_code += &format!("{} = {};\n", attr, expr);
        Ok(())
    }
}
