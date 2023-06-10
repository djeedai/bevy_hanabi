use std::{any::Any, borrow::Cow, num::NonZeroU64};

use bevy::{
    math::{Vec2, Vec3, Vec4},
    reflect::{
        utility::NonGenericTypeInfoCell, DynamicStruct, FieldIter, FromReflect, NamedField,
        Reflect, ReflectMut, ReflectOwned, ReflectRef, Struct, StructInfo, TypeInfo, Typed,
    },
};
use serde::{Deserialize, Serialize};

use crate::{graph::Value, next_multiple_of, ToWgslString};

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

#[derive(Debug, Clone, Reflect, FromReflect)]
pub(crate) struct AttributeInner {
    name: Cow<'static, str>,
    default_value: Value,
}

impl PartialEq for AttributeInner {
    fn eq(&self, other: &Self) -> bool {
        // Compare attributes by name since it's unique.
        self.name == other.name
    }
}

impl Eq for AttributeInner {}

impl std::hash::Hash for AttributeInner {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Keep consistent with PartialEq and Eq
        self.name.hash(state);
    }
}

impl AttributeInner {
    pub const POSITION: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("position"), Value::Float3(Vec3::ZERO));
    pub const VELOCITY: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("velocity"), Value::Float3(Vec3::ZERO));
    pub const AGE: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("age"), Value::Float(0.));
    pub const LIFETIME: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("lifetime"), Value::Float(1.));
    pub const COLOR: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("color"), Value::Uint(0xFFFFFFFFu32));
    pub const HDR_COLOR: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("hdr_color"), Value::Float4(Vec4::ONE));
    pub const ALPHA: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("alpha"), Value::Float(1.));
    pub const SIZE: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("size"), Value::Float(1.));
    pub const SIZE2: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("size2"), Value::Float2(Vec2::ONE));
    pub const AXIS_X: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("axis_x"), Value::Float3(Vec3::X));
    pub const AXIS_Y: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("axis_y"), Value::Float3(Vec3::Y));
    pub const AXIS_Z: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("axis_z"), Value::Float3(Vec3::Z));

    #[inline]
    pub(crate) const fn new(name: Cow<'static, str>, default_value: Value) -> Self {
        Self {
            name,
            default_value,
        }
    }
}

/// An attribute of a particle simulated for an effect.
///
/// Effects are composed of many simulated particles. Each particle is in turn
/// composed of a set of attributes, which are used to simulate and render it.
/// Common attributes include the particle's position, its age, or its color.
/// See [`Attribute::ALL`] for a list of supported attributes. Custom attributes
/// are not supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "&str", into = "&'static str")]
pub struct Attribute(pub(crate) &'static AttributeInner);

impl TryFrom<&str> for Attribute {
    type Error = &'static str;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Attribute::from_name(value).ok_or("Unknown attribute name.")
    }
}

impl From<Attribute> for &'static str {
    fn from(value: Attribute) -> Self {
        value.name()
    }
}

impl Typed for Attribute {
    fn type_info() -> &'static TypeInfo {
        static CELL: NonGenericTypeInfoCell = NonGenericTypeInfoCell::new();
        CELL.get_or_set(|| {
            let fields = [
                NamedField::new::<Cow<str>>("name"),
                NamedField::new::<Value>("default_value"),
            ];
            let info = StructInfo::new::<Self>("Attribute", &fields);
            TypeInfo::Struct(info)
        })
    }
}

impl Struct for Attribute {
    fn field(&self, name: &str) -> Option<&dyn Reflect> {
        match name {
            "name" => Some(&self.0.name),
            "default_value" => Some(&self.0.default_value),
            _ => None,
        }
    }

    fn field_mut(&mut self, _name: &str) -> Option<&mut dyn Reflect> {
        // Attributes are immutable
        None
    }

    fn field_at(&self, index: usize) -> Option<&dyn Reflect> {
        match index {
            0 => Some(&self.0.name),
            1 => Some(&self.0.default_value),
            _ => None,
        }
    }

    fn field_at_mut(&mut self, _index: usize) -> Option<&mut dyn Reflect> {
        // Attributes are immutable
        None
    }

    fn name_at(&self, index: usize) -> Option<&str> {
        match index {
            0 => Some("name"),
            1 => Some("default_value"),
            _ => None,
        }
    }

    fn field_len(&self) -> usize {
        2
    }

    fn iter_fields(&self) -> FieldIter {
        FieldIter::new(self)
    }

    fn clone_dynamic(&self) -> DynamicStruct {
        let mut dynamic = DynamicStruct::default();
        dynamic.set_name(::std::string::ToString::to_string(Reflect::type_name(self)));
        dynamic.insert_boxed("name", Reflect::clone_value(&self.0.name));
        dynamic.insert_boxed("default_value", Reflect::clone_value(&self.0.default_value));
        dynamic
    }
}

impl Reflect for Attribute {
    #[inline]
    fn type_name(&self) -> &str {
        ::core::any::type_name::<Attribute>()
    }

    #[inline]
    fn get_type_info(&self) -> &'static TypeInfo {
        <Attribute as Typed>::type_info()
    }

    #[inline]
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    #[inline]
    fn into_reflect(self: Box<Self>) -> Box<dyn Reflect> {
        self
    }

    #[inline]
    fn as_reflect(&self) -> &dyn Reflect {
        self
    }

    #[inline]
    fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
        self
    }

    #[inline]
    fn clone_value(&self) -> Box<dyn Reflect> {
        Box::new(*self)
    }

    #[inline]
    fn set(&mut self, value: Box<dyn Reflect>) -> Result<(), Box<dyn Reflect>> {
        *self = <dyn Reflect>::take(value)?;
        Ok(())
    }

    #[inline]
    fn apply(&mut self, value: &dyn Reflect) {
        let value = Reflect::as_any(value);
        if let Some(value) = <dyn Any>::downcast_ref::<Self>(value) {
            *self = *value;
        } else {
            panic!("Value is not {}.", ::core::any::type_name::<Self>());
        }
    }

    fn reflect_ref(&self) -> ReflectRef {
        ReflectRef::Struct(self)
    }

    fn reflect_mut(&mut self) -> ReflectMut {
        ReflectMut::Struct(self)
    }

    fn reflect_owned(self: Box<Self>) -> ReflectOwned {
        ReflectOwned::Struct(self)
    }
}

impl FromReflect for Attribute {
    fn from_reflect(reflect: &dyn Reflect) -> Option<Self> {
        if let Some(name) = reflect.as_any().downcast_ref::<String>() {
            Attribute::from_name(name)
        } else {
            None
        }
    }
}

impl Attribute {
    /// The particle position in [simulation space].
    ///
    /// # Type
    ///
    /// [`ValueType::Float3`] representing the XYZ coordinates of the position.
    ///
    /// [simulation space]: crate::SimulationSpace
    pub const POSITION: Attribute = Attribute(AttributeInner::POSITION);

    /// The particle velocity in [simulation space].
    ///
    /// # Type
    ///
    /// [`ValueType::Float3`] representing the XYZ coordinates of the velocity.
    ///
    /// [simulation space]: crate::SimulationSpace
    pub const VELOCITY: Attribute = Attribute(AttributeInner::VELOCITY);

    /// The age of the particle.
    ///
    /// Each time the particle is updated, the current simualtion delta time is
    /// added to the particle's age. The age can be used to animate some other
    /// quantities; see the [`ColorOverLifetimeModifier`] for example.
    ///
    /// If the particle also has a lifetime (either a per-effect
    /// constant value, or a per-particle value stored in the
    /// [`Attribute::LIFETIME`] attribute), then when the age of the particle
    /// exceeds its lifetime, the particle dies and is not simulated nor
    /// rendered anymore.
    ///
    /// # Type
    ///
    /// [`ValueType::Float`]
    ///
    /// [`ColorOverLifetimeModifier`]: crate::modifier::render::ColorOverLifetimeModifier
    pub const AGE: Attribute = Attribute(AttributeInner::AGE);

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
    pub const LIFETIME: Attribute = Attribute(AttributeInner::LIFETIME);

    /// The particle's base color.
    ///
    /// This attribute stores a per-particle color, which can be used for
    /// various purposes, generally as the base color for rendering the
    /// particle.
    ///
    /// # Type
    ///
    /// [`ValueType::Uint`] representing the RGBA components of the color
    /// encoded as `0xAABBGGRR`, with a single byte per component.
    pub const COLOR: Attribute = Attribute(AttributeInner::COLOR);

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
    pub const HDR_COLOR: Attribute = Attribute(AttributeInner::HDR_COLOR);

    /// The particle's transparency (alpha).
    ///
    /// Type: [`ValueType::Float`]
    pub const ALPHA: Attribute = Attribute(AttributeInner::ALPHA);

    /// The particle's uniform size.
    ///
    /// The particle is uniformly scaled by this size.
    ///
    /// # Type
    ///
    /// [`ValueType::Float`]
    pub const SIZE: Attribute = Attribute(AttributeInner::SIZE);

    /// The particle's 2D size, for quad rendering.
    ///
    /// The particle, when drawn as a quad, is scaled along its local X and Y
    /// axes by these values.
    ///
    /// # Type
    ///
    /// [`ValueType::Float2`] representing the XY sizes of the particle.
    pub const SIZE2: Attribute = Attribute(AttributeInner::SIZE2);

    /// The local X axis of the particle.
    ///
    /// This attribute stores a per-particle X axis, which defines the
    /// horizontal direction of a quad particle. This is generally used to
    /// re-orient the particle during rendering, for example to face the camera
    /// or another point of interest.
    ///
    /// # Type
    ///
    /// [`ValueType::Float3`]
    pub const AXIS_X: Attribute = Attribute(AttributeInner::AXIS_X);

    /// The local Y axis of the particle.
    ///
    /// This attribute stores a per-particle Y axis, which defines the vertical
    /// direction of a quad particle. This is generally used to re-orient the
    /// particle during rendering, for example to face the camera or another
    /// point of interest.
    ///
    /// # Type
    ///
    /// [`ValueType::Float3`]
    pub const AXIS_Y: Attribute = Attribute(AttributeInner::AXIS_Y);

    /// The local Z axis of the particle.
    ///
    /// This attribute stores a per-particle Z axis, which defines the normal to
    /// a quad particle's plane. This is generally used to re-orient the
    /// particle during rendering, for example to face the camera or another
    /// point of interest.
    ///
    /// # Type
    ///
    /// [`ValueType::Float3`]
    pub const AXIS_Z: Attribute = Attribute(AttributeInner::AXIS_Z);

    /// Collection of all the existing particle attributes.
    pub const ALL: [Attribute; 12] = [
        Attribute::POSITION,
        Attribute::VELOCITY,
        Attribute::AGE,
        Attribute::LIFETIME,
        Attribute::COLOR,
        Attribute::HDR_COLOR,
        Attribute::ALPHA,
        Attribute::SIZE,
        Attribute::SIZE2,
        Attribute::AXIS_X,
        Attribute::AXIS_Y,
        Attribute::AXIS_Z,
    ];

    /// Retrieve an attribute by its name.
    ///
    /// See [`Attribute::ALL`] for the list of attributes, and the
    /// [`Attribute::name()`] method of each attribute for their name.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let attr = Attribute::from_name("position").unwrap();
    /// assert_eq!(attr, Attribute::POSITION);
    /// ```
    pub fn from_name(name: &str) -> Option<Attribute> {
        Attribute::ALL
            .iter()
            .find(|&attr| attr.name() == name)
            .copied()
    }

    /// The attribute's name.
    ///
    /// The name of an attribute is unique, and corresponds to the name of the
    /// variable in the generated WGSL code.
    #[inline]
    pub fn name(&self) -> &'static str {
        self.0.name.as_ref()
    }

    /// The attribute's default value.
    #[inline]
    pub fn default_value(&self) -> Value {
        self.0.default_value
    }

    /// The attribute's type.
    ///
    /// This is a shortcut for `default_value().value_type()`.
    #[inline]
    pub fn value_type(&self) -> ValueType {
        self.0.default_value.value_type()
    }

    /// Size of this attribute, in bytes.
    ///
    /// This is a shortcut for `value_type().size()`.
    #[inline]
    pub fn size(&self) -> usize {
        self.value_type().size()
    }

    /// Alignment of this attribute, in bytes.
    ///
    /// This is a shortcut for `value_type().align()`.
    #[inline]
    pub fn align(&self) -> usize {
        self.value_type().align()
    }
}

/// Layout for a single [`Attribute`] inside a [`ParticleLayout`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct AttributeLayout {
    pub attribute: Attribute,
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
    /// builder.append(Attribute::POSITION);
    /// ```
    pub fn append(mut self, attribute: Attribute) -> Self {
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
    ///     .append(Attribute::POSITION)
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
            .append(Attribute::POSITION)
            .append(Attribute::AGE)
            .append(Attribute::VELOCITY)
            .append(Attribute::LIFETIME)
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
    ///     .append(Attribute::POSITION)
    ///     .append(Attribute::AGE)
    ///     .append(Attribute::LIFETIME)
    ///     .build();
    /// ```
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> ParticleLayoutBuilder {
        ParticleLayoutBuilder::default()
    }

    /// Build a new particle layout from the current one merged with a new set
    /// of attributes.
    pub fn merged_with(
        &self,
        // attributes: impl IntoIterator<Item = Attribute>,
        attributes: &[Attribute],
    ) -> ParticleLayout {
        let mut builder = ParticleLayoutBuilder::from(self);
        // for attr in attributes.into_iter() {
        for attr in attributes {
            builder = builder.append(*attr);
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
    ///     .append(Attribute::POSITION) // vec3<f32>
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
    ///     .append(Attribute::POSITION) // vec3<f32>
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
    /// # use bevy_hanabi::*;
    /// let layout = ParticleLayout::new()
    ///     .append(Attribute::SIZE)
    ///     .build();
    /// let has_size = layout.contains(Attribute::SIZE);
    /// assert!(has_size);
    /// ```
    pub fn contains(&self, attribute: Attribute) -> bool {
        self.layout
            .iter()
            .any(|&entry| entry.attribute.name() == attribute.name())
    }

    /// Generate the WGSL attribute code corresponding to the layout.
    pub fn generate_code(&self) -> String {
        // assert!(self.layout.is_sorted_by_key(|entry| entry.offset));
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

    use bevy::{
        math::{Vec2, Vec3, Vec4},
        reflect::TypeRegistration,
    };
    use naga::{front::wgsl::Parser, proc::Layouter};

    // Ensure the size and alignment of all types conforms to the WGSL spec by
    // querying naga as a reference.
    #[test]
    fn value_type_align() {
        let mut parser = Parser::new();
        for (value_type, value) in &[
            (ValueType::Float, crate::graph::Value::Float(0.)),
            (
                ValueType::Float2,
                crate::graph::Value::Float2(Vec2::new(-0.5, 3.458)),
            ),
            (
                ValueType::Float3,
                crate::graph::Value::Float3(Vec3::new(-0.5, 3.458, -53.)),
            ),
            (
                ValueType::Float4,
                crate::graph::Value::Float4(Vec4::new(-0.5, 3.458, 0., -53.)),
            ),
            (ValueType::Uint, crate::graph::Value::Uint(42_u32)),
        ] {
            assert_eq!(value.value_type(), *value_type);

            // Create a tiny WGSL snippet with the Value(Type) and parse it
            let src = format!("const x = {};", value.to_wgsl_string());
            let res = parser.parse(&src);
            if let Err(err) = &res {
                println!("Error: {:?}", err);
            }
            assert!(res.is_ok());
            let m = res.unwrap();
            // println!("Module: {:?}", m);

            // Retrieve the "x" constant and the size/align of its type
            let (_cst_handle, cst) = m
                .constants
                .iter()
                .find(|c| c.1.name == Some("x".to_string()))
                .unwrap();
            let (size, align) = if let naga::ConstantInner::Scalar { width, value: _ } = &cst.inner
            {
                // Scalar types have the same size and align
                (
                    *width as u32,
                    naga::proc::Alignment::new(*width as u32).unwrap(),
                )
            } else {
                // For non-scalar types, calculate the type layout according to WGSL
                let type_handle = cst.inner.resolve_type().handle().unwrap();
                let mut layouter = Layouter::default();
                assert!(layouter.update(&m.types, &m.constants).is_ok());
                let layout = layouter[type_handle];
                (layout.size, layout.alignment)
            };

            // Compare WGSL layout with the one of Value(Type)
            assert_eq!(size, value_type.size() as u32);
            assert_eq!(
                align,
                naga::proc::Alignment::new(value_type.align() as u32).unwrap()
            );
        }
    }

    const TEST_ATTR_NAME: &str = "test_attr";
    const TEST_ATTR_INNER: &AttributeInner =
        &AttributeInner::new(Cow::Borrowed(TEST_ATTR_NAME), Value::Float3(Vec3::ONE));

    #[test]
    fn attr_new() {
        let attr = Attribute(TEST_ATTR_INNER);
        assert_eq!(attr.name(), TEST_ATTR_NAME);
        assert_eq!(attr.size(), 12);
        assert_eq!(attr.align(), 16);
        assert_eq!(attr.value_type(), ValueType::Float3);
        assert_eq!(attr.default_value(), Value::Float3(Vec3::ONE));
    }

    #[test]
    fn attr_from_name() {
        for attr in Attribute::ALL {
            assert_eq!(Attribute::from_name(attr.name()), Some(attr));
        }
    }

    #[test]
    fn attr_reflect() {
        let mut attr = Attribute(TEST_ATTR_INNER);

        let r = attr.as_reflect();
        assert_eq!(
            TypeRegistration::of::<Attribute>().type_name(),
            r.type_name()
        );
        match r.reflect_ref() {
            ReflectRef::Struct(s) => {
                assert_eq!(2, s.field_len());

                assert_eq!(Some("name"), s.name_at(0));
                assert_eq!(Some("default_value"), s.name_at(1));
                assert_eq!(None, s.name_at(2));
                assert_eq!(None, s.name_at(9999));

                assert_eq!(
                    Some("alloc::borrow::Cow<str>"),
                    s.field("name").map(|f| f.type_name())
                );
                assert_eq!(
                    Some("bevy_hanabi::graph::Value"),
                    s.field("default_value").map(|f| f.type_name())
                );
                assert!(s.field("DUMMY").is_none());
                assert!(s.field("").is_none());

                for f in s.iter_fields() {
                    assert!(
                        f.type_name().contains("alloc::borrow::Cow<str>")
                            || f.type_name().contains("bevy_hanabi::graph::Value")
                    );
                }

                let d = s.clone_dynamic();
                assert_eq!(TypeRegistration::of::<Attribute>().type_name(), d.name());
                assert_eq!(Some(0), d.index_of("name"));
                assert_eq!(Some(1), d.index_of("default_value"));
            }
            _ => panic!("Attribute should be reflected as a Struct"),
        }

        // Mutating operators are not implemented by design; only hard-coded built-in
        // attributes are supported. In any case that won't matter because you
        // cannot call `as_reflect_mut()` since you cannot obtain a mutable reference to
        // an attribute.
        let r = attr.as_reflect_mut();
        match r.reflect_mut() {
            ReflectMut::Struct(s) => {
                assert!(s.field_mut("name").is_none());
                assert!(s.field_mut("default_value").is_none());
                assert!(s.field_at_mut(0).is_none());
                assert!(s.field_at_mut(1).is_none());
            }
            _ => panic!("Attribute should be reflected as a Struct"),
        }
    }

    #[test]
    fn attr_from_reflect() {
        for attr in Attribute::ALL {
            let s: String = attr.name().into();
            let r = s.as_reflect();
            let r_attr = Attribute::from_reflect(r).expect("Cannot find attribute by name");
            assert_eq!(r_attr, attr);
        }

        assert_eq!(
            None,
            Attribute::from_reflect("test".to_string().as_reflect())
        );
    }

    #[test]
    fn attr_serde() {
        // All existing attributes can round-trip via serialization
        for attr in Attribute::ALL {
            // Serialize; this produces just the name of the attribute, which uniquely
            // identifies it. The default value is never serialized.
            let ron = ron::to_string(&attr).unwrap();
            assert_eq!(ron, format!("\"{}\"", attr.name()));

            // Deserialize; this recovers the Attribute from its name using
            // Attribute::from_name().
            let s: Attribute = ron::from_str(&ron).unwrap();
            assert_eq!(s, attr);
        }

        // Any other attribute name cannot deserialize
        assert!(ron::from_str::<Attribute>("\"\"").is_err());
        assert!(ron::from_str::<Attribute>("\"UNKNOWN\"").is_err());
    }

    const F1_INNER: &AttributeInner = &AttributeInner::new(Cow::Borrowed("F1"), Value::Float(3.));
    const F1B_INNER: &AttributeInner = &AttributeInner::new(Cow::Borrowed("F1B"), Value::Float(5.));
    const F2_INNER: &AttributeInner =
        &AttributeInner::new(Cow::Borrowed("F2"), Value::Float2(Vec2::ZERO));
    const F2B_INNER: &AttributeInner =
        &AttributeInner::new(Cow::Borrowed("F2B"), Value::Float2(Vec2::ONE));
    const F3_INNER: &AttributeInner =
        &AttributeInner::new(Cow::Borrowed("F3"), Value::Float3(Vec3::ZERO));
    const F3B_INNER: &AttributeInner =
        &AttributeInner::new(Cow::Borrowed("F3B"), Value::Float3(Vec3::ONE));
    const F4_INNER: &AttributeInner =
        &AttributeInner::new(Cow::Borrowed("F4"), Value::Float4(Vec4::ZERO));
    const F4B_INNER: &AttributeInner =
        &AttributeInner::new(Cow::Borrowed("F4B"), Value::Float4(Vec4::ONE));

    const F1: Attribute = Attribute(F1_INNER);
    const F1B: Attribute = Attribute(F1B_INNER);
    const F2: Attribute = Attribute(F2_INNER);
    const F2B: Attribute = Attribute(F2B_INNER);
    const F3: Attribute = Attribute(F3_INNER);
    const F3B: Attribute = Attribute(F3B_INNER);
    const F4: Attribute = Attribute(F4_INNER);
    const F4B: Attribute = Attribute(F4B_INNER);

    #[test]
    fn test_layout_build() {
        // empty
        let layout = ParticleLayout::new().build();
        assert_eq!(layout.layout.len(), 0);
        assert_eq!(layout.generate_code(), String::new());

        // single
        for attr in Attribute::ALL {
            let layout = ParticleLayout::new().append(attr).build();
            assert_eq!(layout.layout.len(), 1);
            let attr0 = &layout.layout[0];
            assert_eq!(attr0.offset, 0);
            assert_eq!(
                layout.generate_code(),
                format!(
                    "    {}: {},\n",
                    attr0.attribute.name(),
                    attr0.attribute.value_type().to_wgsl_string()
                )
            );
        }

        // dedup
        for attr in [F1, F2, F3, F4] {
            let mut layout = ParticleLayout::new();
            for _ in 0..3 {
                layout = layout.append(attr);
            }
            let layout = layout.build();
            assert_eq!(layout.layout.len(), 1); // unique
            let attr = &layout.layout[0];
            assert_eq!(attr.offset, 0);
        }

        // homogenous
        for attrs in [[F1, F1B], [F2, F2B], [F3, F3B], [F4, F4B]] {
            let mut layout = ParticleLayout::new();
            for &attr in &attrs {
                layout = layout.append(attr);
            }
            let layout = layout.build();
            assert_eq!(layout.layout.len(), 2);
            let attr_0 = &layout.layout[0];
            let size = attr_0.attribute.size();
            assert_eq!(attr_0.offset as usize, 0);
            let attr_1 = &layout.layout[1];
            assert_eq!(attr_1.offset as usize, size);
            assert_eq!(attr_1.attribute.size(), size);
        }

        // [3, 1, 3, 2] -> [3 1 3 2]
        {
            let mut layout = ParticleLayout::new();
            for &attr in &[F1, F3, F2, F3B] {
                layout = layout.append(attr);
            }
            let layout = layout.build();
            assert_eq!(layout.layout.len(), 4);
            for (i, (off, a)) in [(0, F3), (12, F1), (16, F3B), (28, F2)].iter().enumerate() {
                let attr_i = layout.layout[i];
                assert_eq!(attr_i.offset, *off);
                assert_eq!(attr_i.attribute, *a);
            }
        }

        // [1, 4, 3, 2, 2, 3] -> [4 3 1 2 2 3]
        {
            let mut layout = ParticleLayout::new();
            for &attr in &[F1, F4, F3, F2, F2B, F3B] {
                layout = layout.append(attr);
            }
            let layout = layout.build();
            assert_eq!(layout.layout.len(), 6);
            for (i, (off, a)) in [(0, F4), (16, F3), (28, F1), (32, F2), (40, F2B), (48, F3B)]
                .iter()
                .enumerate()
            {
                let attr_i = layout.layout[i];
                assert_eq!(attr_i.offset, *off);
                assert_eq!(attr_i.attribute, *a);
            }
        }
    }
}
