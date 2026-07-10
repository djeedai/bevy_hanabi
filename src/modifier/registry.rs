//! Type registration for modifiers.
//!
//! This module contains utilities used to reflect and serialize
//! [`EffectAsset`]. In general, you don't need to directly interact with those;
//! instead, use [`EffectAsset::serialize`], as well as the
//! [`EffectAssetSerializer`] and [`EffectAssetDeserializer`].
//!
//! _Modifiers_ are objects implementing the [`Modifier`] trait. Because a trait
//! abstracts the concrete object type, it cannot be used as is with the Bevy
//! reflection and serialization infrastructure. Bevy does allow reflecting
//! traits with `#[reflect_trait]`, but due to other limitations (notably,
//! `Box<T>` not being reflected), this is not sufficient to easily make
//! modifiers reflected and serializable.
//!
//! This module contains some utilities to enable reflecting and serializing
//! modifiers. At the core, we register for each conrete modifier type some
//! [`ReflectModifier`] type data, which essentially contains a factory function
//! used to construct an instance of that type during deserialization. We also
//! define [`Modifiers`], an ordered collection (`Vec`) of [`BoxedModifier`]
//! objects, and provide some implementation of Bevy's [`SerializeWithRegistry`]
//! and [`DeserializeWithRegistry`]. Those trait are automatically picked up by
//! Bevy's reflect-based serialization utilities like [`TypedReflectSerializer`]
//! and [`TypedReflectDeserializer`]. They are in turn used by
//! [`EffectAssetSerializer`] and [`EffectAssetDeserializer`], which are custom
//! de/serializer implementations for [`EffectAsset`] which handle the
//! collections of modifiers for the init, update, and render passes of each
//! effect.
//!
//! [`EffectAsset`]: crate::EffectAsset
//! [`EffectAsset::serialize`]: crate::EffectAsset::serialize
//! [`EffectAssetSerializer`]: crate::EffectAssetSerializer
//! [`EffectAssetDeserializer`]: crate::EffectAssetDeserializer
//! [`SerializeWithRegistry`]: bevy::reflect::serde::SerializeWithRegistry
//! [`DeserializeWithRegistry`]: bevy::reflect::serde::DeserializeWithRegistry
//! [`TypedReflectSerializer`]: bevy::reflect::serde::TypedReflectSerializer
//! [`TypedReflectDeserializer`]: bevy::reflect::serde::TypedReflectDeserializer

use std::any::TypeId;

use bevy::{
    ecs::reflect::AppTypeRegistry,
    log::warn,
    reflect::{
        serde::{ReflectDeserializer, TypedReflectSerializer},
        PartialReflect, Reflect, TypeRegistry,
    },
};
use serde::{
    de::DeserializeSeed,
    ser::{Error as _, SerializeMap as _},
    Serializer,
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
        D: serde::de::Deserializer<'de>,
    {
        // First, use the generic reflect deserializer which expects a single-entry map
        // { "full::type::Path": <value> } and returns a Box<dyn PartialReflect> (e.g.
        // DynamicStruct).
        let reflect_seed = ReflectDeserializer::new(registry);
        let boxed_partial: Box<dyn PartialReflect> = reflect_seed
            .deserialize(deserializer)
            .map_err(serde::de::Error::custom)?;

        // Obtain the type id represented by the deserialized reflect value.
        let type_info = boxed_partial.get_represented_type_info().ok_or_else(|| {
            serde::de::Error::custom("reflected value has no represented type info")
        })?;
        let type_id = type_info.type_id();

        // Lookup the ReflectModifier type data that must have been registered for this
        // concrete modifier type (register_reflect_modifier<T>). That type-data
        // contains the factory used to create a default instance and cached
        // ModifierContext.
        let reflect_modifier = registry
            .get_type_data::<ReflectModifier>(type_id)
            .ok_or_else(|| {
                serde::de::Error::custom(format!(
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

        Ok(modifier)
    }
}

use std::fmt::Formatter;
use std::ops::{Deref, DerefMut};

use bevy::reflect::serde::{ReflectDeserializeWithRegistry, ReflectSerializeWithRegistry};
use serde::de::{SeqAccess, Visitor};
use serde::ser::SerializeSeq;

/// Ordered collection of [`BoxedModifier`] objects.
///
/// This is a wrapper over a `Vec<BoxedModifier>`, with additional reflection
/// and serialization features.
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
        struct Elem<'a> {
            boxed: &'a BoxedModifier,
            registry: &'a TypeRegistry,
        }

        impl<'a> serde::Serialize for Elem<'a> {
            fn serialize<S2>(&self, serializer: S2) -> Result<S2::Ok, S2::Error>
            where
                S2: serde::Serializer,
            {
                use bevy::reflect::serde::SerializeWithRegistry;
                self.boxed.serialize(serializer, self.registry)
            }
        }

        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for m in &self.0 {
            seq.serialize_element(&Elem { boxed: m, registry })?;
        }
        seq.end()
    }
}

impl<'de> bevy::reflect::serde::DeserializeWithRegistry<'de> for Modifiers {
    fn deserialize<D>(deserializer: D, registry: &TypeRegistry) -> Result<Self, D::Error>
    where
        D: serde::de::Deserializer<'de>,
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

                impl<'de2, 'a> serde::de::DeserializeSeed<'de2> for ElemSeed<'a> {
                    type Value = BoxedModifier;

                    fn deserialize<D2>(self, deserializer: D2) -> Result<Self::Value, D2::Error>
                    where
                        D2: serde::de::Deserializer<'de2>,
                    {
                        use bevy::reflect::serde::DeserializeWithRegistry;
                        BoxedModifier::deserialize(deserializer, self.registry)
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

        let modifiers: Self = deserializer.deserialize_seq(ModifiersVisitor { registry })?;
        Ok(modifiers)
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::Vec3;
    use ron::ser::PrettyConfig;
    use serde::de::DeserializeSeed;

    use super::*;
    use crate::{
        register_modifiers, AccelModifier, EffectAsset, EffectAssetDeserializer,
        EffectAssetSerializer, RenderModifier, SpawnerSettings,
    };

    /// Serialize and deserialize a [`Modifiers`] container.
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
        let s = ron::ser::to_string_pretty(
            &TypedReflectSerializer::new(modifiers.as_reflect(), &registry),
            PrettyConfig::default(),
        )
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
        let serializer = EffectAssetSerializer::new(&asset, &registry);
        let json = ron::ser::to_string_pretty(&serializer, PrettyConfig::default()).unwrap();
        println!("{json}");

        // de
        let mut deserializer = ron::de::Deserializer::from_str(&json).unwrap();
        let deserialize = EffectAssetDeserializer::new(&registry);
        let serde_asset = deserialize.deserialize(&mut deserializer).unwrap();

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
