//! Type registration for modifiers.

use std::any::TypeId;

use bevy::{
    ecs::reflect::AppTypeRegistry,
    log::warn,
};

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

// Serde helpers: serialize/deserialize BoxedModifier via Bevy's TypeRegistry + reflect serde
// The approach below uses Bevy's ReflectSerializer / ReflectDeserializer which emit a
// single-entry map keyed by the full type path (e.g. "my_crate::mod::MyModifier": { ... }).
// On deserialization we use the type information produced by the reflect deserializer to
// lookup the ReflectModifier type-data (registered via register_reflect_modifier) and
// then construct a default instance using the registered factory and apply the deserialized
// reflect data into that instance via Reflect::set.
//
// This keeps a single source of truth (bevy's TypeRegistry) and avoids the old typetag hack.
#[cfg(feature = "serde")]
#[allow(missing_docs)]
pub mod serde_impl {
    use serde::de;
    use serde::de::DeserializeSeed;
    use serde::ser::Error as _;
    use serde::ser::SerializeMap as _;
    use serde::Serializer;

    use bevy::reflect::serde::{
        ReflectDeserializer, TypedReflectSerializer,
        ReflectSerializeWithRegistry, ReflectDeserializeWithRegistry,
    };
    use bevy::reflect::{PartialReflect, Reflect, ReflectFromReflect, TypeRegistry};

    use crate::{BoxedModifier, Module, ReflectModifier};

    // Serialize a Box<dyn Modifier> by delegating to the reflect serializer. The
    // reflect serializer emits the type tag and the inner data as a single map entry.
    impl bevy::reflect::serde::SerializeWithRegistry for BoxedModifier {
        fn serialize<S>(&self, serializer: S, registry: &TypeRegistry) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            // Get an &dyn Reflect for the boxed modifier
            let reflect: &dyn Reflect = Reflect::as_reflect(&**self);

            // Serialize as a single-entry map { "full::type::Path": <typed-data> }
            let type_path = reflect
                .get_represented_type_info()
                .ok_or_else(|| {
                    S::Error::custom("cannot serialize dynamic value without represented type")
                })?
                .type_path();

            let mut map = serializer.serialize_map(Some(1))?;
            map.serialize_entry(type_path, &TypedReflectSerializer::new(reflect, registry))?;
            map.end()
        }
    }

    // Deserialize a Box<dyn Modifier> using the reflect deserializer and the ReflectModifier
    // type data to construct the concrete Modifier instance.
    impl<'de> bevy::reflect::serde::DeserializeWithRegistry<'de> for BoxedModifier {
        fn deserialize<D>(deserializer: D, registry: &TypeRegistry) -> Result<Self, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            // First, use the generic reflect deserializer which expects a single-entry map
            // { "full::type::Path": <value> } and returns a Box<dyn PartialReflect> (e.g. DynamicStruct).
            let reflect_seed = ReflectDeserializer::new(registry);
            let boxed_partial: Box<dyn PartialReflect> = reflect_seed
                .deserialize(deserializer)
                .map_err(de::Error::custom)?;

            // Obtain the type id represented by the deserialized reflect value.
            let type_info = boxed_partial
                .get_represented_type_info()
                .ok_or_else(|| de::Error::custom("reflected value has no represented type info"))?;
            let type_id = type_info.type_id();

            // Lookup the ReflectModifier type data that must have been registered for this
            // concrete modifier type (register_reflect_modifier<T>). That type-data contains
            // the factory used to create a default instance and cached ModifierContext.
            let reflect_modifier = registry
                .get_type_data::<ReflectModifier>(type_id)
                .ok_or_else(|| {
                    de::Error::custom(format!(
                        "no ReflectModifier type data for '{}'",
                        type_info.type_path()
                    ))
                })?;

            // Convert the dynamic PartialReflect into a concrete Box<dyn Reflect> using
            // the ReflectFromReflect type data registered for the represented concrete type.
            let rfr = registry
                .get_type_data::<ReflectFromReflect>(type_id)
                .ok_or_else(|| de::Error::custom("no ReflectFromReflect data for type"))?;
            let concrete_reflect: Box<dyn Reflect> = rfr
                .from_reflect(boxed_partial.as_partial_reflect())
                .ok_or_else(|| de::Error::custom("from_reflect failed"))?;

            // Build a default instance using the stored factory. The factory currently takes
            // a &mut Module; construct a temporary Module (same as register_reflect_modifier).
            let mut module = Module::default();
            let mut modifier: BoxedModifier = (reflect_modifier.factory)(&mut module);

            // Assign the deserialized reflect value into the freshly created instance.
            // Reflect::set performs a type-checked assignment or applies fields appropriately.
            let reflect_mut: &mut dyn Reflect = Reflect::as_reflect_mut(&mut *modifier);
            reflect_mut.set(concrete_reflect).map_err(|_v| {
                de::Error::custom("failed to assign reflect value to modifier instance")
            })?;

            Ok(modifier)
        }
    }

    // Note: for this to be used automatically by the reflect-powered serializers/deserializers
    // the TypeRegistry must register the ReflectSerializeWithRegistry / ReflectDeserializeWithRegistry
    // type data for BoxedModifier (or the surrounding asset type that contains it) so that
    // reflection knows to call these impls. In practice this is done by registering the type
    // information for the asset and ensuring your AppTypeRegistry contains the ReflectModifier
    // entries for every concrete modifier type.

    use serde::de::{SeqAccess, Visitor};
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Serialize};
    use serde::ser::Serializer as SerSerializer;
    use serde::de::Deserializer as DeDeserializer;
    use std::fmt::Formatter;
    use std::ops::{Deref, DerefMut};

    #[derive(Default, Clone, Reflect)]
    #[reflect(SerializeWithRegistry, DeserializeWithRegistry, from_reflect = false)]
    pub struct Modifiers(#[reflect(ignore)] pub Vec<BoxedModifier>);

    impl Deref for Modifiers {
        type Target = Vec<BoxedModifier>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl DerefMut for Modifiers {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl bevy::reflect::serde::SerializeWithRegistry for Modifiers {
        fn serialize<S>(&self, serializer: S, registry: &TypeRegistry) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
            for m in &self.0 {
                seq.serialize_element(&TypedReflectSerializer::new(m.as_reflect(), registry))?;
            }
            seq.end()
        }
    }

    // Provide plain serde Serialize/Deserialize by delegating to the inner Vec<BoxedModifier>.
    impl Serialize for Modifiers {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: SerSerializer,
        {
            self.0.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Modifiers {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: DeDeserializer<'de>,
        {
            let vec: Vec<BoxedModifier> = Vec::deserialize(deserializer)?;
            Ok(Modifiers(vec))
        }
    }

    impl<'de> bevy::reflect::serde::DeserializeWithRegistry<'de> for Modifiers {
        fn deserialize<D>(deserializer: D, registry: &TypeRegistry) -> Result<Self, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            struct ModifiersVisitor<'a> {
                registry: &'a TypeRegistry,
            }

            impl<'a, 'de> Visitor<'de> for ModifiersVisitor<'a> {
                type Value = Modifiers;

                fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                    write!(formatter, "a list of modifiers")
                }

                fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                where
                    A: SeqAccess<'de>,
                {
                    let mut vec = Vec::new();
                    while let Some(boxed_partial) =
                        seq.next_element_seed(ReflectDeserializer::new(self.registry))?
                    {
                        let type_info =
                            boxed_partial.get_represented_type_info().ok_or_else(|| {
                                de::Error::custom("reflected value has no represented type info")
                            })?;
                        let type_id = type_info.type_id();

                        let reflect_modifier = self
                            .registry
                            .get_type_data::<ReflectModifier>(type_id)
                            .ok_or_else(|| {
                                de::Error::custom("no ReflectModifier type data for type")
                            })?;

                        let rfr = self
                            .registry
                            .get_type_data::<ReflectFromReflect>(type_id)
                            .ok_or_else(|| {
                                de::Error::custom("no ReflectFromReflect data for type")
                            })?;
                        let concrete_reflect: Box<dyn Reflect> = rfr
                            .from_reflect(boxed_partial.as_partial_reflect())
                            .ok_or_else(|| de::Error::custom("from_reflect failed"))?;

                        let mut module = Module::default();
                        let mut modifier: BoxedModifier = (reflect_modifier.factory)(&mut module);
                        let reflect_mut: &mut dyn Reflect = Reflect::as_reflect_mut(&mut *modifier);
                        reflect_mut.set(concrete_reflect).map_err(|_v| {
                            de::Error::custom("failed to assign reflect value to modifier instance")
                        })?;

                        vec.push(modifier);
                    }

                    Ok(Modifiers(vec))
                }
            }

            deserializer.deserialize_seq(ModifiersVisitor { registry })
        }
    }
}

// Re-export Modifiers at module root so other modules can refer to
// crate::modifier::registry::Modifiers
#[cfg(feature = "serde")]
pub use serde_impl::Modifiers;

// The Modifiers wrapper type — declared/implemented inside the serde_impl module
// so it can reuse the imports there.

#[cfg(test)]
mod tests {
    use bevy::math::Vec3;
    use bevy::reflect::serde::{ReflectDeserializer, ReflectSerializer};
    use serde::de::DeserializeSeed;

    use crate::{register_modifiers, AccelModifier, EffectAsset, SpawnerSettings};

    //use super::serde_impl::*;
    use super::*;

    #[test]
    fn ser() {
        let mut module = Module::default();
        let accel_mod = AccelModifier::new(module.lit(Vec3::X));
        let asset =
            EffectAsset::new(24, SpawnerSettings::once(3.0.into()), module).update(accel_mod);

        let type_registry = AppTypeRegistry::new_with_derived_types();
        register_modifiers(&type_registry);

        // ser
        let registry = type_registry.read();
        let serializer = ReflectSerializer::new(&asset, &registry);
        let json = ron::to_string(&serializer).unwrap();
        println!("{json}");

        // de
        let mut deserializer = ron::de::Deserializer::from_str(&json).unwrap();
        let reflect_deserializer = ReflectDeserializer::new(&registry);
        let reflect_value = reflect_deserializer.deserialize(&mut deserializer).unwrap();
        println!("{:?}", reflect_value.is_dynamic());
    }
}
