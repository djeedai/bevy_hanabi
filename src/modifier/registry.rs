//! Type registration for modifiers.

use std::any::TypeId;

use bevy::{ecs::reflect::AppTypeRegistry, log::warn};

use crate::{BoxedModifier, Modifier, ModifierContext, Module};

/// Factory function to build a default modifier instance.
///
/// The factory can use the given [`Module`] to register some expressions in
/// order to default-initialize itself.
pub type ModifierFactory = fn(&mut Module) -> BoxedModifier;

/// Type data attached to a modifier type's registration in
/// [`AppTypeRegistry`]. Carrying both fields here means the editor
/// never needs to construct an instance just to query
/// `Modifier::context()`.
#[derive(Clone, Copy)]
pub struct ReflectModifier {
    /// Factory function.
    pub factory: ModifierFactory,
    /// Modifier context, cached from [`Modifier::context()`].
    pub context: ModifierContext,
}

impl ReflectModifier {}

/// Register a [`ReflectModifier`] type data for the given [`Modifier`].
pub fn register_reflect_modifier<T: Modifier>(
    type_registry: &AppTypeRegistry,
    factory: ModifierFactory,
) {
    // Create a dummy instance to cache its context. This also serves as a
    // validation layer to check the returned object type is correct.
    let context = {
        let mut module = Module::default();
        let modifier = factory(&mut module);
        let context = modifier.context();
        let any = modifier.into_any();
        assert_eq!(
            any.type_id(),
            TypeId::of::<T>(),
            "Factory for modifier type '{}' returned a different object of type.",
            std::any::type_name::<T>()
        );
        context
    };

    let reflect_modifier = ReflectModifier { factory, context };

    match type_registry.write().get_mut(TypeId::of::<T>()) {
        Some(type_registration) => type_registration.insert(reflect_modifier),
        None => warn!(
            "insert_reflect_modifier: type {} not found in TypeRegistry",
            std::any::type_name::<T>()
        ),
    }
}
