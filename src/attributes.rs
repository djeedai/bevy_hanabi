use std::{borrow::Cow, num::NonZeroU64};

use crate::{next_multiple_of, ToWgslString};

/// Type of an [`Attribute`]'s value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValueType {
    /// Single `f32` value.
    Float,
    /// Vector of two `f32` values (`vec2<f32>`). Equivalent to `Vec2` on the
    /// CPU side.
    Float2,
    /// Vector of three `f32` values (`vec3<f32>`). Equivalent to `Vec3` on the
    /// CPU side.
    Float3,
    /// Vector of four `f32` values (`vec4<f32>`). Equivalent to `Vec4` on the
    /// CPU side.
    Float4,
    /// Single `u32` value.
    Uint,
}

impl ValueType {
    /// Size of a value of this type, in bytes.
    pub fn size(&self) -> usize {
        match self {
            ValueType::Float => 4,
            ValueType::Float2 => 8,
            ValueType::Float3 => 12,
            ValueType::Float4 => 16,
            ValueType::Uint => 4,
        }
    }

    /// Alignment of a value of this type, in bytes.
    ///
    /// This corresponds to the alignment of a variable of that type when part
    /// of a struct in WGSL.
    pub fn align(&self) -> usize {
        match self {
            ValueType::Float => 4,
            ValueType::Float2 => 8,
            ValueType::Float3 => 16,
            ValueType::Float4 => 16,
            ValueType::Uint => 4,
        }
    }
}

impl ToWgslString for ValueType {
    fn to_wgsl_string(&self) -> String {
        match self {
            ValueType::Float => "f32",
            ValueType::Float2 => "vec2<f32>",
            ValueType::Float3 => "vec3<f32>",
            ValueType::Float4 => "vec4<f32>",
            ValueType::Uint => "u32",
        }
        .to_string()
    }
}

/// An attribute of a particle simulated for an effect.
///
/// Effects are composed of many simulated particles. Each particle is in turn
/// composed of a set of attributes, which are used to simulate and render it.
/// Common attributes include the particle's position, its age, or its color.
/// See [`Attribute::ALL`] for a list of supported attributes. Custom attributes
/// are not supported.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Attribute {
    name: Cow<'static, str>,
    value_type: ValueType,
}

impl Attribute {
    /// The particle position in [simulation space].
    ///
    /// # Type
    ///
    /// [`ValueType::Float3`] representing the XYZ coordinates of the position.
    ///
    /// [simulation space]: crate::SimulationSpace
    pub const POSITION: &'static Attribute =
        &Attribute::new(Cow::Borrowed("position"), ValueType::Float3);

    /// The particle velocity in [simulation space].
    ///
    /// # Type
    ///
    /// [`ValueType::Float3`] representing the XYZ coordinates of the velocity.
    ///
    /// [simulation space]: crate::SimulationSpace
    pub const VELOCITY: &'static Attribute =
        &Attribute::new(Cow::Borrowed("velocity"), ValueType::Float3);

    /// The age of the particle.
    ///
    /// When the age of the particle exceeds its lifetime (either a per-effect
    /// constant value, or a per-particle value stored in the
    /// [`Attribute::LIFETIME`] attribute), the particle dies and is not
    /// simulated nor rendered anymore.
    ///
    /// # Type
    ///
    /// [`ValueType::Float`]
    pub const AGE: &'static Attribute = &Attribute::new(Cow::Borrowed("age"), ValueType::Float);

    /// The lifetime of the particle.
    ///
    /// This attribute stores a per-particle lifetime, which compared to the
    /// particle's age allows determining if the particle needs to be
    /// simulated and rendered. This requires the [`Attribute::AGE`]
    /// attribute to be used too.
    ///
    /// # Type
    ///
    /// [`ValueType::Float`]
    pub const LIFETIME: &'static Attribute =
        &Attribute::new(Cow::Borrowed("lifetime"), ValueType::Float);

    /// The particle's base color.
    ///
    /// This attribute stores a per-particle color, which can be used for
    /// various purposes, generally as the base color for rendering the
    /// particle.
    ///
    /// # Type
    ///
    /// [`ValueType::Uint`] representing the RGBA components of the color
    /// encoded as 0xAABBGGRR, with a single byte per component.
    pub const COLOR: &'static Attribute = &Attribute::new(Cow::Borrowed("color"), ValueType::Uint);

    /// The particle's base color (HDR).
    ///
    /// This attribute stores a per-particle HDR color, which can be used for
    /// various purposes, generally as the base color for rendering the
    /// particle.
    ///
    /// # Type
    ///
    /// [`ValueType::Float4`] representing the RGBA components of the color.
    /// Values are not clamped, and can be outside the \[0:1\] range to
    /// represent HDR values.
    pub const HDR_COLOR: &'static Attribute =
        &Attribute::new(Cow::Borrowed("hdr_color"), ValueType::Float4);

    /// The particle's transparency (alpha).
    ///
    /// Type: [`ValueType::Float`]
    pub const ALPHA: &'static Attribute = &Attribute::new(Cow::Borrowed("alpha"), ValueType::Float);

    /// The particle's uniform size.
    ///
    /// The particle is uniformly scaled by this size.
    ///
    /// # Type
    ///
    /// [`ValueType::Float`]
    pub const SIZE: &'static Attribute = &Attribute::new(Cow::Borrowed("size"), ValueType::Float);

    /// The particle's 2D size, for quad rendering.
    ///
    /// The particle, when drawn as a quad, is scaled along its local X and Y
    /// axes by these values.
    ///
    /// # Type
    ///
    /// [`ValueType::Float2`] representing the XY sizes of the particle.
    pub const SIZE2: &'static Attribute =
        &Attribute::new(Cow::Borrowed("size2"), ValueType::Float2);

    /// Collection of all the existing particle attributes.
    pub const ALL: [&'static Attribute; 9] = [
        Attribute::POSITION,
        Attribute::VELOCITY,
        Attribute::AGE,
        Attribute::LIFETIME,
        Attribute::COLOR,
        Attribute::HDR_COLOR,
        Attribute::ALPHA,
        Attribute::SIZE,
        Attribute::SIZE2,
    ];

    /// Retrieve an attribute by its name.
    ///
    /// See [`Attribute::ALL`] for the list of attributes, and the
    /// [`Attribute::name()`] method of each attribute for their name.
    pub fn from_name<'a>(name: &'a str) -> Option<&'static Attribute> {
        Attribute::ALL
            .iter()
            .find(|&&attr| attr.name() == name)
            .copied()
    }

    /// Create a new attribute.
    pub(crate) const fn new(name: Cow<'static, str>, value_type: ValueType) -> Self {
        Self { name, value_type }
    }

    /// The attribute's name.
    ///
    /// The name of an attribute is unique, and corresponds to the name of the
    /// variable in the generated WGSL code.
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    /// The attribute's type.
    pub fn value_type(&self) -> ValueType {
        self.value_type
    }

    /// Size of this attribute, in bytes.
    ///
    /// This is a shortcut for [`ValueType::size()`].
    pub fn size(&self) -> usize {
        self.value_type.size()
    }

    /// Alignment of this attribute, in bytes.
    ///
    /// This is a shortcut for [`ValueType::align()`].
    pub fn align(&self) -> usize {
        self.value_type.align()
    }
}

/// Layout for a single [`Attribute`] inside a [`ParticleLayout`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AttributeLayout {
    pub attribute: &'static Attribute,
    pub offset: u32,
}

impl std::fmt::Debug for AttributeLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // "(+offset) name: type"
        f.write_fmt(format_args!(
            "(+{}) {}: {}",
            self.offset,
            self.attribute.name(),
            self.attribute.value_type().to_wgsl_string(),
        ))
    }
}

/// Builder helper to create a new [`ParticleLayout`].
///
/// Use [`ParticleLayout::new()`] to create a new empty builder.
#[derive(Debug, Default, Clone)]
pub struct ParticleLayoutBuilder {
    layout: Vec<AttributeLayout>,
}

impl ParticleLayoutBuilder {
    /// Add a new attribute to the layout builder.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let mut builder = ParticleLayout::new();
    /// builder.add(Attribute::POSITION);
    /// ```
    pub fn add(mut self, attribute: &'static Attribute) -> Self {
        self.layout.push(AttributeLayout {
            attribute,
            offset: 0, // fixed up by build()
        });
        self
    }

    /// Finalize the builder pattern and build the layout from the existing
    /// attributes.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let layout = ParticleLayout::new()
    ///     .add(Attribute::POSITION)
    ///     .build();
    /// ```
    pub fn build(mut self) -> ParticleLayout {
        // Remove duplicates
        self.layout.sort_unstable_by_key(|la| la.attribute.name());
        self.layout.dedup_by_key(|la| la.attribute.name());

        // Sort by size
        self.layout.sort_unstable_by_key(|la| la.attribute.size());

        let mut layout = vec![];
        let mut offset = 0;

        // Enqueue all Float4, which are already aligned
        let index4 = self
            .layout
            .partition_point(|attr| attr.attribute.size() < 16);
        for i in index4..self.layout.len() {
            let mut attr = self.layout[i];
            attr.offset = offset;
            offset += 16;
            layout.push(attr);
        }

        // Enqueue paired { Float3 + Float1 }
        let index2 = self
            .layout
            .partition_point(|attr| attr.attribute.size() < 8);
        let num1 = index2;
        let index3 = self
            .layout
            .partition_point(|attr| attr.attribute.size() < 12);
        let num2 = (index2..index3).len();
        let num3 = (index3..index4).len();
        let num_pairs = num1.min(num3);
        for i in 0..num_pairs {
            // Float3
            let mut attr = self.layout[index3 + i];
            attr.offset = offset;
            offset += 12;
            layout.push(attr);

            // Float
            let mut attr = self.layout[i];
            attr.offset = offset;
            offset += 4;
            layout.push(attr);
        }
        let index1 = num_pairs;
        let index3 = index3 + num_pairs;
        let num1 = num1 - num_pairs;
        let num3 = num3 - num_pairs;

        // Enqueue paired { Float2 + Float2 }
        for i in 0..(num2 / 2) {
            for j in 0..2 {
                let mut attr = self.layout[index2 + i * 2 + j];
                attr.offset = offset;
                offset += 8;
                layout.push(attr);
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
                    let mut attr = self.layout[index3 + i];
                    attr.offset = offset;
                    offset += 12;
                    layout.push(attr);
                }
                let mut attr = self.layout[index2];
                attr.offset = offset;
                offset += 8;
                layout.push(attr);
                num3head
            } else {
                0
            };

            // End with remaining Float3
            for i in num3head..num3 {
                let mut attr = self.layout[index3 + i];
                attr.offset = offset;
                offset += 12;
                layout.push(attr);
            }
        } else {
            // Float3 is done, some Float1 left, and at most one Float2
            debug_assert_eq!(num3, 0);

            // Emit the single Float2 now if any
            if num2 > 0 {
                debug_assert_eq!(num2, 1);
                let mut attr = self.layout[index2];
                attr.offset = offset;
                offset += 8;
                layout.push(attr);
            }

            // End with remaining Float1
            for i in 0..num1 {
                let mut attr = self.layout[index1 + i];
                attr.offset = offset;
                offset += 4;
                layout.push(attr);
            }
        }

        ParticleLayout { layout }
    }
}

impl From<&ParticleLayout> for ParticleLayoutBuilder {
    fn from(layout: &ParticleLayout) -> Self {
        Self {
            layout: layout.layout.clone(),
        }
    }
}

/// Particle layout of an effect.
///
/// The particle layout describes the set of attributes used by the particles of
/// an effect, and the relative positioning of those attributes inside the
/// particle GPU buffer.
///
/// Effects with a compatible particle layout can be simulated or rendered
/// together in a single call, therefore it is recommended to minimize the
/// layout variations across effects and attempt to reuse the same layout for
/// multiple effects, which can be benefical for performance.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ParticleLayout {
    layout: Vec<AttributeLayout>,
}

impl std::fmt::Debug for ParticleLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Output a compact list of all layout entries
        f.debug_list().entries(self.layout.iter()).finish()
    }
}

impl Default for ParticleLayout {
    fn default() -> Self {
        // Default layout: { position, age, velocity, lifetime }
        ParticleLayout::new()
            .add(Attribute::POSITION)
            .add(Attribute::AGE)
            .add(Attribute::VELOCITY)
            .add(Attribute::LIFETIME)
            .build()
    }
}

impl ParticleLayout {
    /// Create an empty finalized layout.
    ///
    /// The layout is immutable. This is mostly used as a placeholder while a
    /// valid layout is not available yet. To create a new non-finalized layout
    /// which can be mutated, use [`ParticleLayout::new()`] instead.
    pub fn empty() -> ParticleLayout {
        Self { layout: vec![] }
    }

    /// Create a new empty layout.
    ///
    /// This returns a builder for the particle layout. Use
    /// [`ParticleLayoutBuilder::build()`] to finalize the builder and create
    /// the actual (immutable) [`ParticleLayout`].
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let layout = ParticleLayout::new()
    ///     .add(Attribute::POSITION)
    ///     .add(Attribute::AGE)
    ///     .add(Attribute::LIFETIME)
    ///     .build();
    /// ```
    pub fn new() -> ParticleLayoutBuilder {
        ParticleLayoutBuilder::default()
    }

    /// Build a new particle layout from the current one merged with a new set
    /// of attributes.
    pub fn merged_with(
        &self,
        //attributes: impl IntoIterator<Item = &'static Attribute>,
        attributes: &[&'static Attribute],
    ) -> ParticleLayout {
        let mut builder = ParticleLayoutBuilder::from(self);
        //for attr in attributes.into_iter() {
        for attr in attributes {
            builder = builder.add(*attr);
        }
        builder.build()
    }

    /// Get the size of the layout in bytes.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let layout = ParticleLayout::new()
    ///     .add(Attribute::POSITION) // vec3<f32>
    ///     .build();
    /// assert_eq!(layout.size(), 12);
    /// ```
    pub fn size(&self) -> u32 {
        if self.layout.is_empty() {
            0
        } else {
            let last_attr = self.layout.last().unwrap();
            last_attr.offset + last_attr.attribute.size() as u32
        }
    }

    /// Get the alignment of the layout in bytes.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let layout = ParticleLayout::new()
    ///     .add(Attribute::POSITION) // vec3<f32>
    ///     .build();
    /// assert_eq!(layout.align(), 16);
    /// ```
    pub fn align(&self) -> usize {
        self.layout
            .iter()
            .map(|attr| attr.attribute.value_type().align())
            .max()
            .unwrap()
    }

    /// Minimum binding size in bytes.
    ///
    /// This corresponds to the stride of the attribute struct in WGSL when
    /// contained inside an array.
    pub fn min_binding_size(&self) -> NonZeroU64 {
        let size = self.size() as usize;
        let align = self.align();
        NonZeroU64::new(next_multiple_of(size, align) as u64).unwrap()
    }

    pub(crate) fn attributes(&self) -> &[AttributeLayout] {
        &self.layout
    }

    /// Check if the layout contains the specified [`Attribute`].
    ///
    /// # Example
    ///
    /// ```
    /// # let layout = AttributeLayout::new();
    /// let has_size = layout.contains(Attribute::SIZE);
    /// ```
    pub fn contains(&self, attribute: &'static Attribute) -> bool {
        self.layout
            .iter()
            .any(|&entry| entry.attribute.name() == attribute.name())
    }

    /// Generate the WGSL attribute code corresponding to the layout.
    pub fn generate_code(&self) -> String {
        //assert!(self.layout.is_sorted_by_key(|entry| entry.offset));
        self.layout
            .iter()
            .map(|entry| {
                format!(
                    "    {}: {},",
                    entry.attribute.name(),
                    entry.attribute.value_type().to_wgsl_string()
                )
            })
            .fold(String::new(), |mut a, b| {
                a.reserve(b.len() + 1);
                a.push_str(&b);
                a.push('\n');
                a
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_name() {
        for attr in Attribute::ALL {
            assert_eq!(Attribute::from_name(attr.name()), Some(attr));
        }
    }

    const F1: &'static Attribute = &Attribute::new(Cow::Borrowed("F1"), ValueType::Float);
    const F1B: &'static Attribute = &Attribute::new(Cow::Borrowed("F1B"), ValueType::Float);
    const F2: &'static Attribute = &Attribute::new(Cow::Borrowed("F2"), ValueType::Float2);
    const F2B: &'static Attribute = &Attribute::new(Cow::Borrowed("F2B"), ValueType::Float2);
    const F3: &'static Attribute = &Attribute::new(Cow::Borrowed("F3"), ValueType::Float3);
    const F3B: &'static Attribute = &Attribute::new(Cow::Borrowed("F3B"), ValueType::Float3);
    const F4: &'static Attribute = &Attribute::new(Cow::Borrowed("F4"), ValueType::Float4);
    const F4B: &'static Attribute = &Attribute::new(Cow::Borrowed("F4B"), ValueType::Float4);

    #[test]
    fn test_layout_build() {
        // empty
        let layout = ParticleLayout::new().build();
        assert_eq!(layout.layout.len(), 0);
        assert_eq!(layout.generate_code(), String::new());

        // single
        for attr in Attribute::ALL {
            let layout = ParticleLayout::new().add(attr).build();
            assert_eq!(layout.layout.len(), 1);
            let attr0 = &layout.layout[0];
            assert_eq!(attr0.offset, 0);
            assert_eq!(
                layout.generate_code(),
                format!(
                    "    {}: {},\n",
                    attr0.attribute.name(),
                    attr0.attribute.value_type.to_wgsl_string()
                )
            );
        }

        // dedup
        for attr in [F1, F2, F3, F4] {
            let mut layout = ParticleLayout::new();
            for _ in 0..3 {
                layout = layout.add(attr);
            }
            let layout = layout.build();
            assert_eq!(layout.layout.len(), 1); // unique
            let attr = &layout.layout[0];
            assert_eq!(attr.offset, 0);
        }

        // homogenous
        for attr in [[F1, F1B], [F2, F2B], [F3, F3B], [F4, F4B]] {
            let mut layout = ParticleLayout::new();
            for i in 0..2 {
                layout = layout.add(attr[i]);
            }
            let layout = layout.build();
            assert_eq!(layout.layout.len(), 2);
            let attr_0 = &layout.layout[0];
            let size = attr_0.attribute.size();
            assert_eq!(attr_0.offset as usize, 0 * size);
            let attr_1 = &layout.layout[1];
            assert_eq!(attr_1.offset as usize, 1 * size);
            assert_eq!(attr_1.attribute.size(), size);
        }

        // [3, 1, 3, 2] -> [3 1 3 2]
        {
            let mut layout = ParticleLayout::new();
            for attr in &[F1, F3, F2, F3B] {
                layout = layout.add(attr);
            }
            let layout = layout.build();
            assert_eq!(layout.layout.len(), 4);
            let mut i = 0;
            for (off, a) in &[(0, F3), (12, F1), (16, F3B), (28, F2)] {
                let attr_i = layout.layout[i];
                assert_eq!(attr_i.offset, *off as u32);
                assert_eq!(&attr_i.attribute, a);
                i += 1;
            }
        }

        // [1, 4, 3, 2, 2, 3] -> [4 3 1 2 2 3]
        {
            let mut layout = ParticleLayout::new();
            for attr in &[F1, F4, F3, F2, F2B, F3B] {
                layout = layout.add(attr);
            }
            let layout = layout.build();
            assert_eq!(layout.layout.len(), 6);
            let mut i = 0;
            for (off, a) in &[(0, F4), (16, F3), (28, F1), (32, F2), (40, F2B), (48, F3B)] {
                let attr_i = layout.layout[i];
                assert_eq!(attr_i.offset, *off as u32);
                assert_eq!(&attr_i.attribute, a);
                i += 1;
            }
        }
    }
}
