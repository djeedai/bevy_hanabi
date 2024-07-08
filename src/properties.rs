//! Effect properties.
//!
//! An _effect property_ is a named variable stored per effect, and mutable at
//! runtime from the application (CPU). Unlike [attributes](crate::attributes),
//! all particles see the same property value while the effect simulates or
//! renders. As such, properties are a much more compact way to represent
//! time-varying quantities when there's no per-particle variation.
//!
//! # Definition
//!
//! An effect property is represented by the [`Property`] type. Each
//! [`Module`] can have zero or more properties.
//!
//! The property has a name unique within the expression module; different
//! effects can have a property with the same name, but a single effect cannot
//! have multiple properties with a duplicate name (since each effect has a
//! single expression module).
//!
//! The property is created with a default value used to initialize it. Like for
//! attributes, this can be any [`Value`]. The type of the default value also
//! defines the type of the property itself, which is immutable. Assigning a new
//! value to the property requires assigning a value of the same type as the
//! default value passed to [`Module::add_property()`].
//!
//! # Use case
//!
//! Use properties to ensure a value from an [expression](crate::expr) can be
//! mutated while the effect is instantiated, without having to destroy and
//! re-create a new effect. Using properties is a bit more costly in terms of
//! GPU processing than using a hard-coded constant, so avoid using properties
//! if a value doesn’t need to change dynamically at runtime, or changes very
//! infrequently.
//!
//! # Layout
//!
//! Like attributes, particle properties are tightly packed to be efficiently
//! uploaded from CPU to GPU when a change is detected. The [`PropertyLayout`]
//! defines this memory layout, with the same rules as attributes. See the
//! [attribute layout](crate::attributes#layout-1) section for more details.
//!
//! # Usage
//!
//! To use properties, create a new property using [`Module::add_property()`],
//! giving it a unique name and a default value. The default value determines
//! the type of the property.
//!
//! ```
//! # use bevy_hanabi::*;
//! # use bevy::prelude::*;
//! let mut module = Module::default();
//! module.add_property("my_color", LinearRgba::WHITE.as_u32().into());
//! ```
//!
//! Once the module is assigned to an [`EffectAsset`], any instance of that
//! effect (that is, any [`ParticleEffect`] component referencing that
//! [`EffectAsset`]) will have the defined a property.
//!
//! Property values are stored in the [`EffectProperties`] component, attached
//! to the same entity containing the [`ParticleEffect`] component. This
//! component can be mutated by a system you own to change the value of one or
//! more properties; when doing so, the Bevy change detection mechanism will
//! trigger a re-upload of the new property values from CPU to GPU. If the
//! [`EffectProperties`] component is missing, and the effect has properties
//! defined in the [`Module`] of its [`EffectAsset`], then 🎆 Hanabi will add
//! the component automatically, using the default value of each property. One
//! advantage of manually adding that component is that you can override the
//! initial value of some or all of the properties. Note that the component is
//! not automatically added if the [`Module`] doesn't declare any property.
//!
//! To change the value of a property, call [`EffectProperties::set()`] or
//! [`EffectProperties::set_if_changed()`]. The former is easier to use but
//! always triggers the change detection mechanism, even if the value assigned
//! is the same as the previous one. [`EffectProperties::set_if_changed()`] on
//! the other hand will compare the current value and only assign if different.
//! This minimizes the need for a GPU re-upload.
//!
//! ```
//! # use bevy_hanabi::*;
//! # use bevy::prelude::*;
//! fn change_property(mut query: Query<&mut EffectProperties>) {
//!     let mut effect_properties = query.single_mut();
//!     let color = LinearRgba::rgb(1., 0., 0.).as_u32();
//!     // If the current color is not already red, it will be updated, and
//!     // the properties will be re-uploaded to the GPU.
//!     EffectProperties::set_if_changed(effect_properties, "my_color", color.into());
//! }
//! ```
//!
//! [`Module`]: crate::Module
//! [`Module::add_property()`]: crate::Module::add_property
//! [`EffectAsset`]: crate::EffectAsset
//! [`add_property()`]: crate::Module::add_property
//! [`ParticleEffect`]: crate::ParticleEffect

use std::num::NonZeroU64;

use bevy::{
    ecs::{component::Component, reflect::ReflectComponent, world::Mut},
    log::trace,
    reflect::Reflect,
    utils::HashSet,
};
use serde::{Deserialize, Serialize};

use crate::{graph::Value, next_multiple_of, ToWgslString, ValueType};

/// A single property of an [`EffectAsset`].
///
/// See the [`properties`](crate::properties) module documentation for details.
///
/// [`EffectAsset`]: crate::EffectAsset
#[derive(Debug, Clone, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub struct Property {
    name: String,
    default_value: Value,
}

impl Property {
    /// Create a new property.
    ///
    /// In general properties are created internally by the [`EffectAsset`]
    /// they're defined on, when calling [`Module::add_property()`].
    ///
    /// [`EffectAsset`]: crate::EffectAsset
    /// [`Module::add_property()`]: crate::Module::add_property
    #[inline]
    pub fn new(name: impl Into<String>, default_value: impl Into<Value>) -> Self {
        Self {
            name: name.into(),
            default_value: default_value.into(),
        }
    }

    /// The property name.
    ///
    /// The name of a property is unique within a given effect, and corresponds
    /// to the name of the variable in the generated WGSL code.
    #[inline]
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    /// The default value of the property.
    ///
    /// The default value is used to initialize the property.
    #[inline]
    pub fn default_value(&self) -> &Value {
        &self.default_value
    }

    /// The property type.
    ///
    /// This is the type of the value stored in the property.
    #[inline]
    pub fn value_type(&self) -> ValueType {
        self.default_value.value_type()
    }

    /// The property size, in bytes.
    ///
    /// This is a shortcut for `self.value_type().size()`.
    #[inline]
    pub fn size(&self) -> usize {
        self.default_value.value_type().size()
    }
}

impl ToWgslString for Property {
    fn to_wgsl_string(&self) -> String {
        format!("properties.{}", self.name)
    }
}

/// Instance of a [`Property`] owned by a specific [`ParticleEffect`] component.
///
/// The property instance is stored inside an [`EffectProperties`].
///
/// [`ParticleEffect`]: crate::ParticleEffect
/// [`EffectProperties`]: crate::EffectProperties
#[derive(Debug, Clone, Reflect)]
pub(crate) struct PropertyInstance {
    /// The property definition, including its default value.
    pub def: Property,
    /// The current runtime value of the property.
    pub value: Value,
}

/// Runtime storage component for the properties of a [`ParticleEffect`].
///
/// This component stores the list of properties of a single [`ParticleEffect`]
/// instance and their current value. It represents the CPU side copy of the
/// values actually present in GPU memory and used by the particle effect.
///
/// A new value can be assigned to a property via [`set()`] or
/// [`set_if_changed()`], which will trigger a GPU (re-)upload
/// of the properties by reading them during the render extract phase.
///
/// # Asset changes
///
/// When a declared property is added to or removed from the underlying
/// [`EffectAsset`], an internal system automatically updates the component
/// during the [`EffectSystems::UpdatePropertiesFromAsset`] stage, which runs in
/// [`PostUpdate`] schedule. Note however that changing a declared property's
/// default value has no effect on the instance already stored in the
/// [`EffectProperties`], and will only affect other components spawned after
/// the change.
///
/// [`ParticleEffect`]: crate::ParticleEffect
/// [`EffectAsset`]: crate::asset::EffectAsset
/// [`set()`]: crate::EffectProperties::set
/// [`set_if_changed()`]: crate::EffectProperties::set_if_changed
/// [`EffectSystems::UpdatePropertiesFromAsset`]: crate::EffectSystems::UpdatePropertiesFromAsset
/// [`PostUpdate`]: bevy::app::PostUpdate
#[derive(Debug, Default, Clone, Component, Reflect)]
#[reflect(Component)]
pub struct EffectProperties {
    /// Instances of all declared properties, as well as any property manually
    /// added with [`set()`] this frame.
    properties: Vec<PropertyInstance>,
}

impl EffectProperties {
    /// Create or set some properties.
    ///
    /// Add new properties or overwrite the value of existing ones. This
    /// function takes an iterator of pairs of `(name, value)`, where `name`
    /// defines the property name and `value` its value.
    ///
    /// If a property with the same name doesn't already exist, a new property
    /// is created with a default value equal to the provided `value`.
    ///
    /// Conversely, if a property with the same name already exists in the
    /// `EffectProperties`, it's overwritten with the new value, unless its
    /// type mismatches in which case this function panics (to prevent
    /// unexpected overwrites). However the property keeps the default value
    /// initially set when first created.
    ///
    /// # Panics
    ///
    /// Panics if a property with the same name already exists in the
    /// `EffectProperties`, and the new property type, as derived from its
    /// default value, is different from the type of the existing property.
    pub fn with_properties(
        mut self,
        properties: impl IntoIterator<Item = (String, Value)>,
    ) -> Self {
        let iter = properties.into_iter();
        for (name, value) in iter {
            if let Some(index) = self.properties.iter().position(|p| p.def.name() == name) {
                assert_eq!(
                    self.properties[index].value.value_type(),
                    value.value_type(),
                    "Trying to overwrite existing property '{}' with value of type {:?}, but property has type {:?}",
                    name,
                    value.value_type(),
                    self.properties[index].value.value_type()
                );
                self.properties[index].value = value;
            } else {
                self.properties.push(PropertyInstance {
                    def: Property::new(name, value),
                    value,
                });
            }
        }
        self
    }

    /// Get all properties currently stored in this component.
    #[allow(dead_code)] // used in tests
    pub(crate) fn properties(&self) -> &[PropertyInstance] {
        &self.properties
    }

    /// Get the value of a stored property.
    ///
    /// The property will be matched by name against the properties already
    /// stored in this [`EffectProperties`] component. If no property exists
    /// with that name, `None` is returned, which either indicates that the
    /// [`Module`] of the effect does not declare such a property, or that the
    /// [`EffectProperties`] component didn't observe the property yet. This
    /// means that [`get_stored()`] is only relevant when called after a
    /// [`set()`] of the same property, or after the
    /// [`EffectSystems::UpdatePropertiesFromAsset`] stage has added any
    /// property declared in the [`Module`] of the effect but missing in the
    /// [`EffectProperties`]. This also means that [`get_stored()`] may
    /// return a property which was [`set()`] but is not in fact declared in
    /// the [`EffectAsset`].
    ///
    /// Note that this behavior is not symmetric with [`set()`], which allows
    /// setting any property even if not declared.
    ///
    /// [`Module`]: crate::Module
    /// [`EffectAsset`]: crate::asset::EffectAsset
    /// [`EffectSystems::UpdatePropertiesFromAsset`]: crate::EffectSystems::UpdatePropertiesFromAsset
    /// [`get_stored()`]: crate::EffectProperties::get_stored
    /// [`set()`]: crate::EffectProperties::set
    pub fn get_stored(&self, name: &str) -> Option<Value> {
        self.properties
            .iter()
            .find(|prop| prop.def.name() == name)
            .map(|prop| prop.value)
    }

    /// Set the value of a property.
    ///
    /// The property will be matched by name against the properties of the
    /// associated [`EffectAsset`] on next update. If no property exists with
    /// that name, the value will be discarded. Otherwise it will be used to
    /// replace the current property's value, and if different will trigger a
    /// GPU re-upload of the properties.
    ///
    /// Note that this behavior is not symmetric with [`get_stored()`], which
    /// only returns properties already stored in this [`EffectProperties`]
    /// component.
    ///
    /// [`EffectAsset`]: crate::asset::EffectAsset
    /// [`get_stored()`]: crate::EffectProperties::get_stored
    pub fn set(&mut self, name: &str, value: Value) {
        if let Some(index) = self
            .properties
            .iter()
            .position(|prop| prop.def.name() == name)
        {
            let prop = &mut self.properties[index];
            assert_eq!(
                prop.def.value_type(),
                value.value_type(),
                "Cannot assign value of type {:?} to property '{}' of type {:?}",
                value.value_type(),
                prop.def.name(),
                prop.def.value_type()
            );
            prop.value = value;
        } else {
            self.properties.push(PropertyInstance {
                def: Property::new(name, value),
                value,
            });
        }
    }

    /// Set the value of a property, only if it changed.
    ///
    /// This is similar to [`set()`], with the notable difference that this
    /// associated function takes a [`Mut`] reference, and will only trigger
    /// change detection on the target component if the property either isn't
    /// already stored, or has a different value than `value`. This means in
    /// particular that a full value comparison is performed, which is never the
    /// case with [`set()`].
    ///
    /// [`set()`]: crate::EffectProperties::set
    pub fn set_if_changed<'p>(
        mut this: Mut<'p, EffectProperties>,
        name: &str,
        value: Value,
    ) -> Mut<'p, EffectProperties> {
        if let Some(index) = this
            .properties
            .iter()
            .position(|prop| prop.def.name() == name)
        {
            let prop = &this.properties[index];
            assert_eq!(
                prop.def.value_type(),
                value.value_type(),
                "Cannot assign value of type {:?} to property '{}' of type {:?}",
                value.value_type(),
                prop.def.name(),
                prop.def.value_type()
            );
            if prop.value != value {
                this.properties[index].value = value;
            }
        } else {
            this.properties.push(PropertyInstance {
                def: Property::new(name, value),
                value,
            });
        }

        this
    }

    /// Update the properties from the asset.
    ///
    /// Compare the properties declared in the asset with the properties
    /// actually stored in the [`EffectProperties`] component, and update the
    /// latter:
    /// - Add any missing property, using their default value.
    /// - Remove any unknown property not declared in the asset.
    ///
    /// Change detection on the [`EffectProperties`] component is guaranteed not
    /// to trigger unless some property was added or removed.
    pub(crate) fn update(
        mut this: Mut<'_, EffectProperties>,
        asset_properties: &[Property],
        is_added: bool,
    ) {
        trace!(
            "Updating effect properties from asset (is_added: {})",
            is_added
        );

        let mut new_props = vec![];
        let mut intersect = HashSet::new();
        for prop in asset_properties {
            if this.properties.iter().any(|p| p.def.name() == prop.name()) {
                intersect.insert(prop.name());
                continue;
            }
            new_props.push(PropertyInstance {
                def: prop.clone(),
                value: *prop.default_value(),
            });
        }

        // Only mutate if needed to avoid triggering change detection
        if intersect.len() != this.properties.len() {
            // Delete instances for unknown properties
            this.properties
                .retain(|prop| intersect.contains(prop.def.name()));
        }

        // Only mutate if needed to avoid triggering change detection
        if !new_props.is_empty() {
            // Append new instances (with their default value) for missing properties
            this.properties.append(&mut new_props);
        }
    }

    /// Serialize properties into a binary blob ready for GPU upload.
    ///
    /// Return the binary blob where properties have been written according to
    /// the given property layout. The size of the output blob is guaranteed
    /// to be equal to the size of the layout.
    pub(crate) fn serialize(&self, layout: &PropertyLayout) -> Vec<u8> {
        let size = layout.size() as usize;
        let mut data = vec![0; size];
        // FIXME: O(n^2) search due to offset() being O(n) linear search already
        for property in &self.properties {
            if let Some(offset) = layout.offset(property.def.name()) {
                let offset = offset as usize;
                let size = property.def.size();
                let src = property.value.as_bytes();
                debug_assert_eq!(src.len(), size);
                let dst = &mut data[offset..offset + size];
                dst.copy_from_slice(src);
            }
        }
        data
    }
}

#[derive(Clone)]
struct PropertyLayoutEntry {
    property: Property,
    offset: u32,
}

impl PartialEq for PropertyLayoutEntry {
    fn eq(&self, other: &Self) -> bool {
        // Compare property's name and type, and offset inside layout, but not default
        // value since two properties cannot differ only by default value (the property
        // name is unique).
        self.property.name() == other.property.name()
            && self.property.value_type() == other.property.value_type()
            && self.offset == other.offset
    }
}

impl Eq for PropertyLayoutEntry {}

impl std::hash::Hash for PropertyLayoutEntry {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash property's name and type, and offset inside layout, but not default
        // value since two properties cannot differ only by default value (the property
        // name is unique).
        self.property.name().hash(state);
        self.property.value_type().hash(state);
        self.offset.hash(state);
    }
}

impl std::fmt::Debug for PropertyLayoutEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // "(+offset) name: type"
        f.write_fmt(format_args!(
            "(+{}) {}: {}",
            self.offset,
            self.property.name(),
            self.property.value_type().to_wgsl_string(),
        ))
    }
}

/// Layout of properties for an effect.
///
/// The `PropertyLayout` describes the memory layout of properties inside the
/// GPU buffer where their values are stored. This forms a contract between the
/// CPU side where properties are written each frame, and the GPU shaders where
/// they're subsequently read.
///
/// The layout is immutable once created. To create a different layout, build a
/// new `PropertyLayout` from scratch, typically with [`new()`].
///
/// # Example
///
/// ```
/// # use bevy_hanabi::{Property, PropertyLayout};
/// # use bevy::math::Vec3;
/// let layout = PropertyLayout::new(&[
///     Property::new("my_property", Vec3::ZERO),
///     Property::new("other_property", 3.4_f32),
/// ]);
/// assert_eq!(layout.size(), 16);
/// ```
///
/// [`new()`]: crate::PropertyLayout::new
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PropertyLayout {
    layout: Vec<PropertyLayoutEntry>,
}

impl PropertyLayout {
    /// Create a new empty property layout.
    ///
    /// An empty layout contains no property. This is often used as a
    /// placeholder where a property layout is expected but no property is
    /// available.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::PropertyLayout;
    /// let layout = PropertyLayout::empty();
    /// assert!(layout.is_empty());
    /// ```
    pub fn empty() -> Self {
        Self { layout: vec![] }
    }

    /// Create a new collection from an iterator.
    ///
    /// In general a property layout is directly built from an asset via
    /// [`EffectAsset::property_layout()`], so this method is mostly used for
    /// advanced use cases or testing.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::{Property, PropertyLayout};
    /// # use bevy::math::Vec3;
    /// let layout = PropertyLayout::new(&[
    ///     Property::new("my_property", Vec3::ZERO),
    ///     Property::new("other_property", 3.4_f32),
    /// ]);
    /// ```
    ///
    /// [`EffectAsset::property_layout()`]: crate::EffectAsset::property_layout
    pub fn new<'a>(iter: impl IntoIterator<Item = &'a Property>) -> Self {
        let mut properties = iter.into_iter().collect::<Vec<_>>();

        // Sort by size
        properties.sort_unstable_by_key(|prop| prop.size());
        let properties = properties; // un-mut

        let mut layout = vec![];
        let mut offset = 0;

        // Enqueue all Float4, which are already aligned
        let index4 = properties.partition_point(|prop| prop.size() < 16);
        for &prop in properties.iter().skip(index4) {
            let entry = PropertyLayoutEntry {
                property: prop.clone(),
                offset,
            };
            offset += 16;
            layout.push(entry);
        }

        // Enqueue paired { Float3 + Float1 }
        let index2 = properties.partition_point(|prop| prop.size() < 8);
        let num1 = index2;
        let index3 = properties.partition_point(|prop| prop.size() < 12);
        let num2 = (index2..index3).len();
        let num3 = (index3..index4).len();
        let num_pairs = num1.min(num3);
        for i in 0..num_pairs {
            // Float3
            let prop = properties[index3 + i];
            let entry = PropertyLayoutEntry {
                property: prop.clone(),
                offset,
            };
            offset += 12;
            layout.push(entry);

            // Float
            let prop = properties[i];
            let entry = PropertyLayoutEntry {
                property: prop.clone(),
                offset,
            };
            offset += 4;
            layout.push(entry);
        }
        let index1 = num_pairs;
        let index3 = index3 + num_pairs;
        let num1 = num1 - num_pairs;
        let num3 = num3 - num_pairs;

        // Enqueue paired { Float2 + Float2 }
        for i in 0..(num2 / 2) {
            for j in 0..2 {
                let prop = properties[index2 + i * 2 + j];
                let entry = PropertyLayoutEntry {
                    property: prop.clone(),
                    offset,
                };
                offset += 8;
                layout.push(entry);
            }
        }
        let index2 = index2 + (num2 / 2) * 2;
        let num2 = num2 % 2;

        // Enqueue { Float3, Float2 } or { Float2, Float1 }
        if num3 > num1 {
            // Float1 is done, some Float3 left, and at most one Float2
            debug_assert_eq!(num1, 0);

            // Try 3/3/2, fallback to 3/2
            let num3head = if num2 > 0 {
                debug_assert_eq!(num2, 1);
                let num3head = num3.min(2);
                for i in 0..num3head {
                    let prop = properties[index3 + i];
                    let entry = PropertyLayoutEntry {
                        property: prop.clone(),
                        offset,
                    };
                    offset += 12;
                    layout.push(entry);
                }
                let prop = properties[index2];
                let entry = PropertyLayoutEntry {
                    property: prop.clone(),
                    offset,
                };
                offset += 8;
                layout.push(entry);
                num3head
            } else {
                0
            };

            // End with remaining Float3
            for i in num3head..num3 {
                let prop = properties[index3 + i];
                let entry = PropertyLayoutEntry {
                    property: prop.clone(),
                    offset,
                };
                offset += 12;
                layout.push(entry);
            }
        } else {
            // Float3 is done, some Float1 left, and at most one Float2
            debug_assert_eq!(num3, 0);

            // Emit the single Float2 now if any
            if num2 > 0 {
                debug_assert_eq!(num2, 1);
                let prop = properties[index2];
                let entry = PropertyLayoutEntry {
                    property: prop.clone(),
                    offset,
                };
                offset += 8;
                layout.push(entry);
            }

            // End with remaining Float1
            for i in 0..num1 {
                let prop = properties[index1 + i];
                let entry = PropertyLayoutEntry {
                    property: prop.clone(),
                    offset,
                };
                offset += 4;
                layout.push(entry);
            }
        }

        Self { layout }
    }

    /// Returns `true` if the layout contains no properties.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::PropertyLayout;
    /// let layout = PropertyLayout::empty();
    /// assert!(layout.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }

    /// Get the size of the layout in bytes.
    ///
    /// The size of a layout is the sum of the offset and size of its last
    /// property. The last property doesn't have any padding, since padding's
    /// purpose is to align the next property in the layout.
    ///
    /// If the layout is empty, this returns zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::{Property, PropertyLayout};
    /// # use bevy::math::Vec3;
    /// let layout = PropertyLayout::new(&[
    ///     Property::new("my_property", Vec3::ZERO),
    ///     Property::new("other_property", 3.4_f32),
    /// ]);
    /// assert_eq!(layout.size(), 16);
    /// ```
    pub fn size(&self) -> u32 {
        if self.layout.is_empty() {
            0
        } else {
            let last_entry = self.layout.last().unwrap();
            last_entry.offset + last_entry.property.size() as u32
        }
    }

    /// Get the alignment of the layout in bytes.
    ///
    /// This is the largest alignment of all the properties. If the layout is
    /// empty, this returns zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::{Property, PropertyLayout};
    /// # use bevy::math::Vec3;
    /// let layout = PropertyLayout::new(&[
    ///     Property::new("my_property", Vec3::ZERO),
    ///     Property::new("other_property", 3.4_f32),
    /// ]);
    /// assert_eq!(layout.align(), 16);
    /// ```
    pub fn align(&self) -> usize {
        if self.layout.is_empty() {
            0
        } else {
            self.layout
                .iter()
                .map(|entry| entry.property.value_type().align())
                .max()
                .unwrap()
        }
    }

    /// Iterate over the properties of the layout and their byte offset.
    ///
    /// This returns an iterator over the offset, in bytes, of a property in the
    /// layout, and a reference to the corresponding property, as a pair.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::{Property, PropertyLayout};
    /// # use bevy::math::Vec3;
    /// let layout = PropertyLayout::new(&[
    ///     Property::new("my_property", Vec3::ZERO),
    ///     Property::new("other_property", 3.4_f32),
    /// ]);
    /// for (offset, property) in layout.properties() {
    ///     println!("+{}: {}", offset, property.name());
    /// }
    /// ```
    pub fn properties(&self) -> impl Iterator<Item = (u32, &Property)> {
        self.layout
            .iter()
            .map(|entry| (entry.offset, &entry.property))
    }

    /// Minimum binding size in bytes.
    ///
    /// This corresponds to the stride of the properties struct in WGSL when
    /// contained inside an array.
    ///
    /// # Panics
    ///
    /// Panics if the layout is empty, which is not valid once used on GPU.
    pub fn min_binding_size(&self) -> NonZeroU64 {
        assert!(!self.layout.is_empty());
        let size = self.size() as usize;
        let align = self.align();
        NonZeroU64::new(next_multiple_of(size, align) as u64).unwrap()
    }

    /// Check if the layout contains the property with the given name.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::{Property, PropertyLayout};
    /// # use bevy::math::Vec3;
    /// let layout = PropertyLayout::new(&[
    ///     Property::new("my_property", Vec3::ZERO),
    ///     Property::new("other_property", 3.4_f32),
    /// ]);
    /// assert!(layout.contains("other_property"));
    /// ```
    pub fn contains(&self, name: &str) -> bool {
        self.layout.iter().any(|entry| entry.property.name == name)
    }

    /// Generate the WGSL property code corresponding to the layout.
    ///
    /// This generates code declaring the `Properties` struct in WGSL, or an
    /// empty string if the layout is empty. The `Properties` struct contains
    /// the values of all the effect properties, as defined by this layout.
    pub fn generate_code(&self) -> String {
        // debug_assert!(self.layout.is_sorted_by_key(|entry| entry.offset));
        let content = self
            .layout
            .iter()
            .map(|entry| {
                format!(
                    "    {}: {},",
                    entry.property.name(),
                    entry.property.value_type().to_wgsl_string()
                )
            })
            .fold(String::new(), |mut a, b| {
                a.reserve(b.len() + 1);
                a.push_str(&b);
                a.push('\n');
                a
            });
        if content.is_empty() {
            String::new()
        } else {
            format!("struct Properties {{\n{}}}\n", content)
        }
    }

    /// Get the offset in byte of the property with the given name.
    ///
    /// If no property with the given name is found, this returns `None`.
    pub(crate) fn offset(&self, name: &str) -> Option<u32> {
        self.layout.iter().find_map(|entry| {
            if entry.property.name == name {
                Some(entry.offset)
            } else {
                None
            }
        })
    }
}

impl Default for PropertyLayout {
    fn default() -> Self {
        PropertyLayout::empty()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };

    use bevy::{
        ecs::component::Tick,
        math::{Vec2, Vec3, Vec4},
    };

    use super::*;

    #[test]
    fn property_basic() {
        let value = Value::Scalar(3_f32.into());
        let p = Property::new("my_prop", value);
        assert_eq!(p.name(), "my_prop");
        assert_eq!(*p.default_value(), value);
        assert_eq!(p.value_type(), value.value_type());
        assert_eq!(p.size(), value.value_type().size());
        assert_eq!(p.to_wgsl_string(), format!("properties.{}", p.name()));
    }

    #[test]
    fn property_serde() {
        let p = Property::new("my_prop", Value::Scalar(3_f32.into()));
        let s = ron::to_string(&p).unwrap();
        println!("property: {:?}", s);
        let p_serde: Property = ron::from_str(&s).unwrap();
        assert_eq!(p_serde, p);
    }

    /// Hash the given PropertyLayoutEntry.
    fn hash_ple(ple: &PropertyLayoutEntry) -> u64 {
        let mut hasher = DefaultHasher::default();
        ple.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn property_layout_entry() {
        // self == self
        let prop1 = Property::new("my_prop", Vec3::NEG_X);
        let entry1 = PropertyLayoutEntry {
            property: prop1,
            offset: 16,
        };
        assert_eq!(
            format!("{:?}", entry1),
            "(+16) my_prop: vec3<f32>".to_string()
        );
        assert_eq!(entry1, entry1);
        assert_eq!(hash_ple(&entry1), hash_ple(&entry1));

        // same name, offset, type
        let prop1b = Property::new("my_prop", Vec3::X);
        let entry1b = PropertyLayoutEntry {
            property: prop1b,
            offset: 16,
        };
        assert_eq!(
            format!("{:?}", entry1b),
            "(+16) my_prop: vec3<f32>".to_string()
        );
        assert_eq!(entry1, entry1b);
        assert_eq!(hash_ple(&entry1), hash_ple(&entry1b));

        // different name (shouldn't happen within same layout)
        let prop2 = Property::new("other_prop", Vec3::Y);
        let entry2 = PropertyLayoutEntry {
            property: prop2,
            offset: 16,
        };
        assert_eq!(
            format!("{:?}", entry2),
            "(+16) other_prop: vec3<f32>".to_string()
        );
        assert_ne!(entry1, entry2);
        assert_ne!(hash_ple(&entry1), hash_ple(&entry2));
        assert_ne!(entry1b, entry2);
        assert_ne!(hash_ple(&entry1b), hash_ple(&entry2));

        // different type (shouldn't happen within same layout)
        let prop3 = Property::new("my_prop", 3.4_f32);
        let entry3 = PropertyLayoutEntry {
            property: prop3,
            offset: 16,
        };
        assert_eq!(format!("{:?}", entry3), "(+16) my_prop: f32".to_string());
        assert_ne!(entry1, entry3);
        assert_ne!(hash_ple(&entry1), hash_ple(&entry3));
        assert_ne!(entry1b, entry3);
        assert_ne!(hash_ple(&entry1b), hash_ple(&entry3));

        // different offset (shouldn't happen within same layout)
        let prop4 = Property::new("my_prop", Vec3::NEG_X);
        let entry4 = PropertyLayoutEntry {
            property: prop4,
            offset: 24,
        };
        assert_eq!(
            format!("{:?}", entry4),
            "(+24) my_prop: vec3<f32>".to_string()
        );
        assert_ne!(entry1, entry4);
        assert_ne!(hash_ple(&entry1), hash_ple(&entry4));
        assert_ne!(entry1b, entry4);
        assert_ne!(hash_ple(&entry1b), hash_ple(&entry4));
    }

    #[test]
    fn layout_empty() {
        let l = PropertyLayout::empty();
        assert!(l.is_empty());
        assert_eq!(l.size(), 0);
        assert_eq!(l.align(), 0);
        assert_eq!(l.properties().next(), None);
        let s = l.generate_code();
        assert!(s.is_empty());
    }

    #[test]
    fn layout_valid() {
        let prop1 = Property::new("f32", 3.4_f32);
        let prop2 = Property::new("vec3", Vec3::ZERO);
        let prop3 = Property::new("vec2", Vec2::NEG_Y);
        let prop4 = Property::new("vec4", Vec4::Y);
        let layout = PropertyLayout::new([&prop1, &prop2, &prop3, &prop4]);
        assert!(!layout.is_empty());
        assert_eq!(layout.size(), 40);
        assert_eq!(layout.align(), 16);
        assert_eq!(layout.min_binding_size(), NonZeroU64::new(48).unwrap());
        let mut it = layout.properties();
        // vec4 goes first as it doesn't disrupt the align
        assert_eq!(it.next(), Some((0, &prop4)));
        // vec3 and f32 go next, in pairs, for same reason
        assert_eq!(it.next(), Some((16, &prop2)));
        assert_eq!(it.next(), Some((28, &prop1)));
        // anything left go last
        assert_eq!(it.next(), Some((32, &prop3)));
        assert_eq!(it.next(), None);
        let s = layout.generate_code();
        assert_eq!(
            s,
            r#"struct Properties {
    vec4: vec4<f32>,
    vec3: vec3<f32>,
    f32: f32,
    vec2: vec2<f32>,
}
"#
        );
    }

    #[test]
    fn layout_tail_332() {
        let prop1 = Property::new("vec2", Vec2::NEG_Y);
        let prop2 = Property::new("vec3a", Vec3::ZERO);
        let prop3 = Property::new("vec3b", Vec3::NEG_X);
        let layout = PropertyLayout::new([&prop1, &prop2, &prop3]);
        assert!(!layout.is_empty());
        assert_eq!(layout.size(), 32);
        assert_eq!(layout.align(), 16);
        assert_eq!(layout.min_binding_size(), NonZeroU64::new(32).unwrap());
        let mut it = layout.properties();
        // 3/3/2
        assert_eq!(it.next(), Some((0, &prop2)));
        assert_eq!(it.next(), Some((12, &prop3)));
        assert_eq!(it.next(), Some((24, &prop1)));
        assert_eq!(it.next(), None);
        let s = layout.generate_code();
        assert_eq!(
            s,
            r#"struct Properties {
    vec3a: vec3<f32>,
    vec3b: vec3<f32>,
    vec2: vec2<f32>,
}
"#
        );
    }

    #[test]
    fn layout_tail_32() {
        let prop1 = Property::new("vec2", Vec2::NEG_Y);
        let prop2 = Property::new("vec3", Vec3::ZERO);
        let layout = PropertyLayout::new([&prop1, &prop2]);
        assert!(!layout.is_empty());
        assert_eq!(layout.size(), 20);
        assert_eq!(layout.align(), 16);
        assert_eq!(layout.min_binding_size(), NonZeroU64::new(32).unwrap());
        let mut it = layout.properties();
        // 3/2
        assert_eq!(it.next(), Some((0, &prop2)));
        assert_eq!(it.next(), Some((12, &prop1)));
        assert_eq!(it.next(), None);
        let s = layout.generate_code();
        assert_eq!(
            s,
            r#"struct Properties {
    vec3: vec3<f32>,
    vec2: vec2<f32>,
}
"#
        );
    }

    #[test]
    fn layout_tail_21() {
        let prop1 = Property::new("f32", 3.4_f32);
        let prop2 = Property::new("vec2", Vec2::NEG_Y);
        let layout = PropertyLayout::new([&prop1, &prop2]);
        assert!(!layout.is_empty());
        assert_eq!(layout.size(), 12);
        assert_eq!(layout.align(), 8);
        assert_eq!(layout.min_binding_size(), NonZeroU64::new(16).unwrap());
        let mut it = layout.properties();
        // 3/2
        assert_eq!(it.next(), Some((0, &prop2)));
        assert_eq!(it.next(), Some((8, &prop1)));
        assert_eq!(it.next(), None);
        let s = layout.generate_code();
        assert_eq!(
            s,
            r#"struct Properties {
    vec2: vec2<f32>,
    f32: f32,
}
"#
        );
    }

    #[test]
    fn effect_properties_with_properties() {
        let ep = EffectProperties::default()
            .with_properties([
                ("a".to_string(), 3.0.into()),
                ("b".to_string(), Vec3::ZERO.into()),
            ])
            .with_properties([
                ("a".to_string(), 7.0.into()),
                ("c".to_string(), Vec2::ONE.into()),
            ]);

        assert_eq!(ep.properties.len(), 3);

        assert_eq!(ep.properties[0].def.name(), "a");
        // kept original 3.0 default
        assert_eq!(
            ep.properties[0].def.default_value(),
            &Value::Scalar(3.0.into())
        );
        // overwrote current value as 7.0
        assert_eq!(ep.properties[0].value, 7.0.into());

        assert_eq!(ep.properties[1].def.name(), "b");
        assert_eq!(
            ep.properties[1].def.default_value(),
            &Value::Vector(Vec3::ZERO.into())
        );
        assert_eq!(ep.properties[1].value, Vec3::ZERO.into());

        assert_eq!(ep.properties[2].def.name(), "c");
        assert_eq!(
            ep.properties[2].def.default_value(),
            &Value::Vector(Vec2::ONE.into())
        );
        assert_eq!(ep.properties[2].value, Vec2::ONE.into());
    }

    #[test]
    #[should_panic]
    fn effect_properties_with_properties_type_mismatch() {
        let _ = EffectProperties::default()
            .with_properties([("a".to_string(), 3.0.into())])
            // Mismatching type for existing property; will panic
            .with_properties([("a".to_string(), Vec2::ONE.into())]);
    }

    #[test]
    fn effect_properties_get_stored() {
        let ep = EffectProperties::default()
            .with_properties([
                ("a".to_string(), 3.0.into()),
                ("b".to_string(), Vec3::ZERO.into()),
            ])
            .with_properties([
                ("a".to_string(), 7.0.into()),
                ("c".to_string(), Vec2::ONE.into()),
            ]);

        assert!(ep.get_stored("a").is_some());
        assert!(ep.get_stored("b").is_some());
        assert!(ep.get_stored("c").is_some());
        assert!(ep.get_stored("x").is_none());
    }

    #[test]
    fn effect_properties_set() {
        let mut ep = EffectProperties::default().with_properties([
            ("a".to_string(), 3.0.into()),
            ("b".to_string(), Vec3::ZERO.into()),
        ]);

        ep.set("a", 7.0.into());
        ep.set("x", 3.0.into());
    }

    #[test]
    #[should_panic]
    fn effect_properties_set_type_mismatch() {
        let mut ep = EffectProperties::default().with_properties([("a".to_string(), 3.0.into())]);
        ep.set("a", Vec3::ZERO.into());
    }

    #[test]
    fn effect_properties_update_empty() {
        // Empty asset vs. empty runtime == empty
        let mut ep = EffectProperties::default();
        let mut added = Tick::new(0);
        let last_changed_prev = Tick::new(0u32.wrapping_sub(1u32));
        let mut last_changed = last_changed_prev;
        let last_run = last_changed;
        let this_run = added;
        let asset_properties = vec![];
        {
            let this = Mut::new(&mut ep, &mut added, &mut last_changed, last_run, this_run);
            let is_added = true;
            EffectProperties::update(this, &asset_properties, is_added);
        }
        assert!(ep.properties.is_empty());
        assert_eq!(last_changed, last_changed_prev); // unchanged (no-op)
    }

    #[test]
    fn effect_properties_update_added() {
        // Some asset vs. empty runtime == some
        let mut ep = EffectProperties::default();
        let mut added = Tick::new(0);
        let last_changed_prev = Tick::new(0u32.wrapping_sub(1u32));
        let mut last_changed = last_changed_prev;
        let last_run = last_changed;
        let this_run = added;
        let asset_properties = vec![Property::new("prop1", 32.)];
        {
            let this = Mut::new(&mut ep, &mut added, &mut last_changed, last_run, this_run);
            let is_added = true;
            EffectProperties::update(this, &asset_properties, is_added);
        }
        assert_eq!(ep.properties.len(), 1);
        assert_eq!(ep.properties[0].def, asset_properties[0]);
        assert_eq!(last_changed, this_run); // changed (added missing property)
    }

    #[test]
    fn effect_properties_update_removed() {
        // Empty asset vs. some runtime == empty
        let mut ep = EffectProperties::default();
        ep.set("unknown", 3.into());
        let mut added = Tick::new(0);
        let last_changed_prev = Tick::new(0u32.wrapping_sub(1u32));
        let mut last_changed = last_changed_prev;
        let last_run = last_changed;
        let this_run = added;
        let asset_properties = vec![];
        {
            let this = Mut::new(&mut ep, &mut added, &mut last_changed, last_run, this_run);
            let is_added = true;
            EffectProperties::update(this, &asset_properties, is_added);
        }
        assert!(ep.properties.is_empty());
        assert_eq!(last_changed, this_run); // changed (removed unknown
                                            // property)
    }

    #[test]
    fn effect_properties_update_override() {
        // Some asset vs. same runtime == same(runtime)
        let mut ep = EffectProperties::default();
        ep.set("prop1", 5_f32.into());
        let mut added = Tick::new(0);
        let last_changed_prev = Tick::new(0u32.wrapping_sub(1u32));
        let mut last_changed = last_changed_prev;
        let last_run = last_changed;
        let this_run = added;
        let asset_properties = vec![Property::new("prop1", 32.)];
        {
            let this = Mut::new(&mut ep, &mut added, &mut last_changed, last_run, this_run);
            let is_added = true;
            EffectProperties::update(this, &asset_properties, is_added);
        }
        assert_eq!(ep.properties.len(), 1);
        assert_eq!(ep.properties[0].def.name(), asset_properties[0].name());
        assert_eq!(ep.properties[0].value, 5_f32.into());
        assert_eq!(last_changed, last_changed_prev); // unchanged
    }

    #[test]
    fn effect_properties_update_mixed() {
        // Some asset vs. some runtime, one override and one default
        let mut ep = EffectProperties::default();
        ep.set("prop1", 5_f32.into());
        let mut added = Tick::new(0);
        let last_changed_prev = Tick::new(0u32.wrapping_sub(1u32));
        let mut last_changed = last_changed_prev;
        let last_run = last_changed;
        let this_run = added;
        let asset_properties = vec![Property::new("prop1", 32.), Property::new("prop2", false)];
        {
            let this = Mut::new(&mut ep, &mut added, &mut last_changed, last_run, this_run);
            let is_added = true;
            EffectProperties::update(this, &asset_properties, is_added);
        }
        assert_eq!(ep.properties.len(), 2);
        assert_eq!(ep.properties[0].def.name(), asset_properties[0].name());
        assert_eq!(ep.properties[0].value, 5_f32.into());
        assert_eq!(ep.properties[1].def, asset_properties[1]);
        assert_eq!(last_changed, this_run); // changed (added missing property)
    }

    #[test]
    fn effect_properties_serialize() {
        let ep = EffectProperties::default().with_properties([
            ("a".to_string(), 3.0.into()),
            ("b".to_string(), Vec3::ONE.into()),
        ]);
        let layout = PropertyLayout::new(ep.properties().iter().map(|pi| &pi.def));
        let blob = ep.serialize(&layout);
        assert_eq!(blob.len(), layout.size() as usize);

        let pi_a = &ep.properties()[0];
        let size = pi_a.def.size();
        assert_eq!(size, 4); // f32
        let offset = layout.offset(pi_a.def.name()).unwrap() as usize;
        let raw = &blob[offset..offset + size];
        #[allow(unsafe_code)]
        let raw_ref: &[u8; 4] = unsafe { std::mem::transmute(&[3.0_f32]) };
        assert_eq!(raw, raw_ref);

        let pi_b = &ep.properties()[1];
        let size = pi_b.def.size();
        assert_eq!(size, 12); // Vec3
        let offset = layout.offset(pi_b.def.name()).unwrap() as usize;
        let raw = &blob[offset..offset + size];
        #[allow(unsafe_code)]
        let raw_ref: &[u8; 12] = unsafe { std::mem::transmute(&[1_f32, 1_f32, 1_f32]) };
        assert_eq!(raw, raw_ref);
    }
}
