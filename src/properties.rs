//! Effect properties.
//!
//! Properties are named variables mutable at runtime. Use properties to ensure
//! a value from a modifier can be dynamically mutated while the effect is
//! instantiated, without having to destroy and re-create a new effect. Using
//! properties is a bit more costly than using a hard-coded constant, so avoid
//! using properties if a value doesn't need to change dynamically at runtime,
//! or changes very infrequently. In general any value accepting a property
//! reference also alternatively accepts a constant.

use std::num::NonZeroU64;

use bevy::reflect::{FromReflect, Reflect};
use serde::{Deserialize, Serialize};

use crate::{graph::Value, next_multiple_of, ToWgslString, ValueType};

/// Effect property.
#[derive(Debug, Clone, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct Property {
    name: String,
    default_value: Value,
}

impl Property {
    /// Create a new property.
    pub(crate) fn new(name: impl Into<String>, default_value: Value) -> Self {
        Self {
            name: name.into(),
            default_value,
        }
    }

    /// The property name.
    ///
    /// The name of a property is unique within a given effect, and corresponds
    /// to the name of the variable in the generated WGSL code.
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    /// The default value of the property.
    pub fn default_value(&self) -> &Value {
        &self.default_value
    }

    /// The property type.
    pub fn value_type(&self) -> ValueType {
        self.default_value.value_type()
    }

    /// The property size, in bytes.
    ///
    /// This is a shortcut for `self.value_type().size()`.
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
/// [`ParticleEffect`]: crate::ParticleEffect
#[derive(Debug, Clone, Reflect, FromReflect)]
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
        // Compare property's name and type, but not default value
        self.property.name() == other.property.name()
            && self.property.value_type() == other.property.value_type()
            && self.offset == other.offset
    }
}

impl Eq for PropertyLayoutEntry {}

impl std::hash::Hash for PropertyLayoutEntry {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash property's name and type, but not default value
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
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PropertyLayout {
    layout: Vec<PropertyLayoutEntry>,
}

impl PropertyLayout {
    /// Create a new empty property layout.
    pub fn empty() -> Self {
        Self { layout: vec![] }
    }

    /// Create a new collection from an iterator.
    pub(crate) fn new<'a>(iter: impl Iterator<Item = &'a Property>) -> Self {
        let mut properties = iter.collect::<Vec<_>>();

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
    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }

    /// Get the size of the layout in bytes.
    pub fn size(&self) -> u32 {
        if self.layout.is_empty() {
            0
        } else {
            let last_entry = self.layout.last().unwrap();
            last_entry.offset + last_entry.property.size() as u32
        }
    }

    /// Get the alignment of the layout in bytes.
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

    /// Minimum binding size in bytes.
    ///
    /// This corresponds to the stride of the properties struct in WGSL when
    /// contained inside an array.
    pub fn min_binding_size(&self) -> NonZeroU64 {
        assert!(!self.layout.is_empty());
        let size = self.size() as usize;
        let align = self.align();
        NonZeroU64::new(next_multiple_of(size, align) as u64).unwrap()
    }

    /// Generate the WGSL attribute code corresponding to the layout.
    pub fn generate_code(&self) -> String {
        // assert!(self.layout.is_sorted_by_key(|entry| entry.offset));
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
            format!("struct Properties {{\n{}\n}}\n", content)
        }
    }

    /// Get the offset for the property with the given name.
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
    use super::*;

    #[test]
    fn serde() {
        let p = Property::new("my_prop", Value::Float(3.));
        let s = ron::to_string(&p).unwrap();
        println!("property: {:?}", s);
        let p_serde: Property = ron::from_str(&s).unwrap();
        assert_eq!(p_serde, p);
    }
}
