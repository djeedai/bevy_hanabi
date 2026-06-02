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

// Serde helpers: serialize/deserialize BoxedModifier via Bevy's TypeRegistry +
// reflect serde The approach below uses Bevy's ReflectSerializer /
// ReflectDeserializer which emit a single-entry map keyed by the full type path
// (e.g. "my_crate::mod::MyModifier": { ... }). On deserialization we use the
// type information produced by the reflect deserializer to lookup the
// ReflectModifier type-data (registered via register_reflect_modifier) and then
// construct a default instance using the registered factory and apply the
// deserialized reflect data into that instance via Reflect::set.
//
// This keeps a single source of truth (bevy's TypeRegistry) and avoids the old
// typetag hack.
#[cfg(feature = "serde")]
#[allow(missing_docs)]
pub mod serde_impl {
    use bevy::reflect::serde::{ReflectDeserializer, TypedReflectSerializer};
    use bevy::reflect::{PartialReflect, Reflect, TypeRegistry};
    use serde::de;
    use serde::de::DeserializeSeed;
    use serde::ser::Error as _;
    use serde::ser::SerializeMap as _;
    use serde::Serializer;

    use crate::{BoxedModifier, Module, ReflectModifier};

    // Serialize a Box<dyn Modifier> by delegating to the reflect serializer. The
    // reflect serializer emits the type tag and the inner data as a single map
    // entry.
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

    // Deserialize a Box<dyn Modifier> using the reflect deserializer and the
    // ReflectModifier type data to construct the concrete Modifier instance.
    impl<'de> bevy::reflect::serde::DeserializeWithRegistry<'de> for BoxedModifier {
        fn deserialize<D>(deserializer: D, registry: &TypeRegistry) -> Result<Self, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            // First, use the generic reflect deserializer which expects a single-entry map
            // { "full::type::Path": <value> } and returns a Box<dyn PartialReflect> (e.g.
            // DynamicStruct).
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
            // concrete modifier type (register_reflect_modifier<T>). That type-data
            // contains the factory used to create a default instance and cached
            // ModifierContext.
            let reflect_modifier = registry
                .get_type_data::<ReflectModifier>(type_id)
                .ok_or_else(|| {
                    de::Error::custom(format!(
                        "no ReflectModifier type data for '{}'",
                        type_info.type_path()
                    ))
                })?;

            // Build a default instance using the stored factory. The factory currently
            // takes a &mut Module; construct a temporary Module (same as
            // register_reflect_modifier).
            let mut module = Module::default();
            let mut modifier: BoxedModifier = (reflect_modifier.factory)(&mut module);

            // Apply the deserialized partial reflect value into the instance.
            let reflect_mut: &mut dyn Reflect = Reflect::as_reflect_mut(&mut *modifier);
            reflect_mut.apply(boxed_partial.as_partial_reflect());
            // .map_err(|e| {
            //     de::Error::custom(format!(
            //         "failed to apply reflect value to modifier instance: {:?}",
            //         e
            //     ))
            // })?;

            Ok(modifier)
        }
    }

    // Note: for this to be used automatically by the reflect-powered
    // serializers/deserializers the TypeRegistry must register the
    // ReflectSerializeWithRegistry / ReflectDeserializeWithRegistry
    // type data for BoxedModifier (or the surrounding asset type that contains it)
    // so that reflection knows to call these impls. In practice this is done by
    // registering the type information for the asset and ensuring your
    // AppTypeRegistry contains the ReflectModifier entries for every concrete
    // modifier type.

    use std::fmt::Formatter;
    use std::ops::{Deref, DerefMut};

    use bevy::reflect::serde::{ReflectDeserializeWithRegistry, ReflectSerializeWithRegistry};
    use serde::de::{SeqAccess, Visitor};
    use serde::ser::SerializeSeq;

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
            //use bevy::reflect::Reflect;
            //use serde::ser::SerializeMap;

            struct Elem<'a> {
                boxed: &'a BoxedModifier,
                //reflect: &'a dyn Reflect,
                registry: &'a TypeRegistry,
            }

            impl<'a> serde::Serialize for Elem<'a> {
                fn serialize<S2>(&self, serializer: S2) -> Result<S2::Ok, S2::Error>
                where
                    S2: serde::Serializer,
                {
                    use bevy::reflect::serde::SerializeWithRegistry;

                    // let type_path = self
                    //     .reflect
                    //     .get_represented_type_info()
                    //     .ok_or_else(|| {
                    //         S2::Error::custom(
                    //             "cannot serialize dynamic value without represented type",
                    //         )
                    //     })?
                    //     .type_path();
                    // let mut map = serializer.serialize_map(Some(1))?;
                    // In theory yes, but BoxedModifier is a type alias, and we don't register its
                    // SerializeWithRegistry as a proper type data, so
                    // TypedReflectSerializer won't find it. Invoke directly instead.
                    // map.serialize_entry(
                    //     type_path,
                    //     &TypedReflectSerializer::new(self.reflect, self.registry),
                    // )?;
                    // Directly invoke SerializeWithRegistry::serialize()
                    self.boxed.serialize(serializer, self.registry)
                    //map.end()
                }
            }

            let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
            for m in &self.0 {
                seq.serialize_element(&Elem {
                    boxed: m,
                    //reflect: m.as_reflect(),
                    registry,
                })?;
            }
            seq.end()
        }
    }

    impl<'de> bevy::reflect::serde::DeserializeWithRegistry<'de> for Modifiers {
        fn deserialize<D>(deserializer: D, registry: &TypeRegistry) -> Result<Self, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            // Deserialize the Modifiers collection, which is a sequence of single-item
            // {typename:value} maps.

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
                    // Each element is a single-entry map { "full::type::Path": <value> }.
                    // Implement a DeserializeSeed that parses that map into a BoxedModifier.
                    struct ElemSeed<'a> {
                        registry: &'a TypeRegistry,
                    }

                    impl<'de2, 'a> de::DeserializeSeed<'de2> for ElemSeed<'a> {
                        type Value = BoxedModifier;

                        fn deserialize<D2>(self, deserializer: D2) -> Result<Self::Value, D2::Error>
                        where
                            D2: de::Deserializer<'de2>,
                        {
                            use bevy::reflect::serde::DeserializeWithRegistry;

                            BoxedModifier::deserialize(deserializer, self.registry)

                            // // Use the reflect deserializer to parse the
                            // single-entry map // into
                            // a Box<dyn PartialReflect> and then construct the
                            // concrete // modifier
                            // using the ReflectModifier factory +
                            // ReflectFromReflect.
                            // let reflect_seed =
                            // ReflectDeserializer::new(self.registry);
                            // let boxed_partial: Box<dyn PartialReflect> =
                            // reflect_seed
                            //     .deserialize(deserializer)
                            //     .map_err(de::Error::custom)?;

                            // let type_info =
                            //     boxed_partial.get_represented_type_info().
                            // ok_or_else(|| {
                            //         de::Error::custom(
                            //             "reflected value has no represented
                            // type info",         )
                            //     })?;
                            // let type_id = type_info.type_id();

                            // let reflect_modifier = self
                            //     .registry
                            //     .get_type_data::<ReflectModifier>(type_id)
                            //     .ok_or_else(|| {
                            //         de::Error::custom("no ReflectModifier
                            // type data for type")
                            //     })?;

                            // let mut module = Module::default();
                            // let mut modifier: BoxedModifier =
                            //     (reflect_modifier.factory)(&mut module);
                            // let reflect_mut: &mut dyn Reflect =
                            //     Reflect::as_reflect_mut(&mut *modifier);
                            // reflect_mut
                            //     .apply(boxed_partial.as_partial_reflect());
                            //     // .map_err(|e| {
                            //     //     de::Error::custom(format!(
                            //     //         "failed to apply reflect value to
                            // modifier instance: {:?}",
                            //     //         e
                            //     //     ))
                            //     // })?;

                            // Ok(modifier)
                        }
                    }

                    let mut vec = Vec::new();
                    while let Some(modifier) = seq.next_element_seed(ElemSeed {
                        registry: self.registry,
                    })? {
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

// The Modifiers wrapper type — declared/implemented inside the serde_impl
// module so it can reuse the imports there.

#[cfg(test)]
mod tests {
    use bevy::reflect::serde::TypedReflectSerializer;
    use bevy::reflect::PartialReflect;
    use bevy::{math::Vec3, reflect::serde::TypedReflectDeserializer};
    use serde::de::DeserializeSeed;

    use super::*;
    use crate::{register_modifiers, AccelModifier, EffectAsset, RenderModifier, SpawnerSettings};

    /// Serialize and deserialize a [`Modifiers`] container.
    #[cfg(feature = "serde")]
    #[test]
    fn serde_modifiers() {
        use bevy::reflect::{
            serde::{TypedReflectDeserializer, TypedReflectSerializer},
            Reflect,
        };

        use crate::{Attribute, SetAttributeModifier, SetPositionSphereModifier, ShapeDimension};

        let mut module = Module::default();

        // Create a Modifiers container with 2 different modifiers
        let mut modifiers = vec![];
        let m1 = SetAttributeModifier::new(Attribute::SIZE, module.lit(2.));
        let bm1: BoxedModifier = Box::new(m1);
        modifiers.push(bm1);
        let m2 = SetPositionSphereModifier {
            center: module.lit(Vec3::ZERO),
            radius: module.lit(1.),
            dimension: ShapeDimension::Surface,
        };
        let bm2: BoxedModifier = Box::new(m2);
        modifiers.push(bm2);
        let modifiers = Modifiers(modifiers);

        // Prepare a type registry with all known modifiers registered.
        let type_registry = AppTypeRegistry::new_with_derived_types();
        register_modifiers(&type_registry);
        let registry = type_registry.read();

        // Serialize via reflection, so that the SerializeWithRegistry impl of Modifiers
        // get automatically picked up and used.
        let s = ron::to_string(&TypedReflectSerializer::new(
            modifiers.as_reflect(),
            &registry,
        ))
        .unwrap();
        println!("{s}");

        // The serialized string must contain the type names of all modifiers, because
        // they use dynamic dispatching.
        assert!(s.contains(std::any::type_name::<SetAttributeModifier>()));
        assert!(s.contains(std::any::type_name::<SetPositionSphereModifier>()));
        // The Modifiers wrapper itself does not necessarily appear in the serialized
        // form here because we're serializing the inner representation
        // directly.
        assert!(!s.contains(std::any::type_name::<Modifiers>()));

        // Deserialize via reflection again. This also picks up the
        // DeserializeWithRegistry impl of Modifiers automatically, which recovers the
        // concrete object via the ReflectModifier type data factory.
        let mut de = ron::de::Deserializer::from_str(&s).unwrap();
        let type_registration = registry.get(std::any::TypeId::of::<Modifiers>()).unwrap();
        let deserializer = TypedReflectDeserializer::new(type_registration, &registry);
        let mods = deserializer.deserialize(&mut de).unwrap();

        // Recover and check the deserialized Modifiers
        assert!(mods.represents::<Modifiers>());
        let mods = mods.try_downcast::<Modifiers>().unwrap();
        assert_eq!(mods.0.len(), modifiers.0.len());
        for (mi, smi) in mods.0.iter().zip(modifiers.0.iter()) {
            use std::any::Any;

            assert_eq!(
                mi.get_represented_type_info().type_id(),
                smi.get_represented_type_info().type_id()
            );
        }

        // Recover and check the individual modifiers
        let serde_m1 = mods.0[0]
            .as_reflect()
            .downcast_ref::<SetAttributeModifier>()
            .unwrap();
        let serde_m2 = mods.0[1]
            .as_reflect()
            .downcast_ref::<SetPositionSphereModifier>()
            .unwrap();
        assert_eq!(m1, *serde_m1);
        assert_eq!(m2, *serde_m2);
    }

    fn cmp_modifiers<'a>(
        a: impl Iterator<Item = &'a dyn Modifier>,
        b: impl Iterator<Item = &'a dyn Modifier>,
    ) {
        let a = a.collect::<Vec<_>>();
        let b = b.collect::<Vec<_>>();
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b.iter()) {
            assert_eq!(a.context(), b.context());
        }
    }

    fn cmp_render_modifiers<'a>(
        a: impl Iterator<Item = &'a dyn RenderModifier>,
        b: impl Iterator<Item = &'a dyn RenderModifier>,
    ) {
        let a = a.collect::<Vec<_>>();
        let b = b.collect::<Vec<_>>();
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b.iter()) {
            assert_eq!(a.context(), b.context());
        }
    }

    /// Serialize and deserialize a full [`EffectAsset`].
    #[test]
    fn serde_asset() {
        let mut module = Module::default();
        let accel_mod = AccelModifier::new(module.lit(Vec3::X));
        let asset =
            EffectAsset::new(24, SpawnerSettings::once(3.0.into()), module).update(accel_mod);

        let type_registry = AppTypeRegistry::new_with_derived_types();
        register_modifiers(&type_registry);

        // ser
        let registry = type_registry.read();
        let serializer = TypedReflectSerializer::new(&asset, &registry);
        let json = ron::to_string(&serializer).unwrap();
        println!("{json}");

        // de
        let mut deserializer = ron::de::Deserializer::from_str(&json).unwrap();
        let type_registration = registry.get(std::any::TypeId::of::<EffectAsset>()).unwrap();
        let reflect_deserializer = TypedReflectDeserializer::new(&type_registration, &registry);
        let reflect_value = reflect_deserializer.deserialize(&mut deserializer).unwrap();
        assert!(reflect_value.represents::<EffectAsset>());

        // Note: reflection-based deserializer produces a DynamicStruct, not an
        // EffectAsset (because FromReflect is not available). Manually apply the
        // data of the DynamicStruct to a real EffectAsset instance for convenience.
        let mut serde_asset = EffectAsset::default();
        serde_asset.try_apply(&*reflect_value).unwrap();

        // Validate what can be
        assert_eq!(asset.name, serde_asset.name);
        assert_eq!(asset.capacity(), serde_asset.capacity());
        assert_eq!(asset.spawner, serde_asset.spawner);
        assert_eq!(asset.z_layer_2d, serde_asset.z_layer_2d);
        assert_eq!(asset.simulation_space, serde_asset.simulation_space);
        assert_eq!(asset.simulation_condition, serde_asset.simulation_condition);
        assert_eq!(asset.prng_seed, serde_asset.prng_seed);
        assert_eq!(asset.motion_integration, serde_asset.motion_integration);
        assert_eq!(asset.alpha_mode, serde_asset.alpha_mode);
        cmp_modifiers(asset.init_modifiers(), serde_asset.init_modifiers());
        cmp_modifiers(asset.update_modifiers(), serde_asset.update_modifiers());
        cmp_render_modifiers(asset.render_modifiers(), serde_asset.render_modifiers());
    }
}
