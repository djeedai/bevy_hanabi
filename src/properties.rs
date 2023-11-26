use std::num::NonZeroU64;

use bevy::reflect::Reflect;
use serde::{Deserialize, Serialize};

use crate::{graph::Value, next_multiple_of, ToWgslString, ValueType};

/// Effect property.
///
/// Properties are named variables mutable at runtime from the application
/// (CPU). Their value is uploaded to the GPU each frame.
///
/// Use properties to ensure a value from a modifier can be dynamically mutated
/// while the effect is instantiated, without having to destroy and re-create a
/// new effect. Using properties is a bit more costly than using a hard-coded
/// constant, so avoid using properties if a value doesn't need to change
/// dynamically at runtime, or changes very infrequently.
#[derive(Debug, Clone, PartialEq, Reflect, Serialize, Deserialize)]
pub struct Property {
    name: String,
    default_value: Value,
}

impl Property {
    /// Create a new property.
    ///
    /// In general properties are created internally by the [`EffectAsset`]
    /// they're defined on, when calling [`EffectAsset::with_property()`] or
    /// [`EffectAsset::add_property()`].
    ///
    /// [`EffectAsset`]: crate::EffectAsset
    /// [`EffectAsset::with_property()`]: crate::EffectAsset::with_property
    /// [`EffectAsset::add_property()`]: crate::EffectAsset::add_property
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
        self.layout
            .iter()
            .filter_map(|entry| {
                if entry.property.name == name {
                    Some(entry.offset)
                } else {
                    None
                }
            })
            .next()
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

    use bevy::math::{Vec2, Vec3, Vec4};

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
}
