use std::{any::Any, borrow::Cow, num::NonZeroU64};

use bevy::{
    math::{Vec2, Vec3, Vec4},
    reflect::{
        utility::{GenericTypePathCell, NonGenericTypeInfoCell},
        DynamicStruct, FieldIter, FromReflect, NamedField, Reflect, ReflectMut, ReflectOwned,
        ReflectRef, Struct, StructInfo, TypeInfo, TypePath, Typed,
    },
};
use serde::{Deserialize, Serialize};

use crate::{
    graph::{ScalarValue, Value, VectorValue},
    next_multiple_of, ToWgslString,
};

/// Scalar types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ScalarType {
    /// Boolean value (`bool`).
    ///
    /// The size of a `bool` is undefined in the WGSL specification, but fixed
    /// at 4 bytes here.
    Bool,
    /// Floating point value (`f32`).
    Float,
    /// Signed 32-bit integer value (`i32`).
    Int,
    /// Unsigned 32-bit integer value (`u32`).
    Uint,
}

impl ScalarType {
    /// Check if this type is a numeric type.
    ///
    /// A numeric type can be used in various math operators etc. All scalar
    /// types are numeric, except `ScalarType::Bool`.
    pub fn is_numeric(&self) -> bool {
        !(matches!(self, ScalarType::Bool))
    }

    /// Size of a value of this type, in bytes.
    ///
    /// This corresponds to the size of a variable of that type when part of a
    /// struct in WGSL. For `bool`, this is always 4 bytes (undefined in WGSL
    /// spec).
    pub const fn size(&self) -> usize {
        4
    }

    /// Alignment of a value of this type, in bytes.
    ///
    /// This corresponds to the alignment of a variable of that type when part
    /// of a struct in WGSL. For `bool`, this is always 4 bytes (undefined in
    /// WGSL spec).
    pub const fn align(&self) -> usize {
        4
    }
}

impl ToWgslString for ScalarType {
    fn to_wgsl_string(&self) -> String {
        match self {
            ScalarType::Bool => "bool",
            ScalarType::Float => "f32",
            ScalarType::Int => "i32",
            ScalarType::Uint => "u32",
        }
        .to_string()
    }
}

/// Vector type (`vecN<T>`).
///
/// Describes the type of a vector, which is composed of 2 to 4 components of a
/// same scalar type. This type corresponds to one of the valid vector types in
/// the WGSL specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct VectorType {
    /// Type of all elements (components) of the vector.
    elem_type: ScalarType,
    /// Number of components. Always 2/3/4.
    count: u8,
}

impl VectorType {
    /// Boolean vector with 2 components (`vec2<bool>`).
    pub const VEC2B: VectorType = VectorType::new(ScalarType::Bool, 2);
    /// Boolean vector with 3 components (`vec3<bool>`).
    pub const VEC3B: VectorType = VectorType::new(ScalarType::Bool, 3);
    /// Boolean vector with 4 components (`vec4<bool>`).
    pub const VEC4B: VectorType = VectorType::new(ScalarType::Bool, 4);
    /// Floating-point vector with 2 components (`vec2<f32>`).
    pub const VEC2F: VectorType = VectorType::new(ScalarType::Float, 2);
    /// Floating-point vector with 3 components (`vec3<f32>`).
    pub const VEC3F: VectorType = VectorType::new(ScalarType::Float, 3);
    /// Floating-point vector with 4 components (`vec4<f32>`).
    pub const VEC4F: VectorType = VectorType::new(ScalarType::Float, 4);
    /// Vector with 2 signed integer components (`vec2<i32>`).
    pub const VEC2I: VectorType = VectorType::new(ScalarType::Int, 2);
    /// Vector with 3 signed integer components (`vec3<i32>`).
    pub const VEC3I: VectorType = VectorType::new(ScalarType::Int, 3);
    /// Vector with 4 signed integer components (`vec4<i32>`).
    pub const VEC4I: VectorType = VectorType::new(ScalarType::Int, 4);
    /// Vector with 2 unsigned integer components (`vec2<u32>`).
    pub const VEC2U: VectorType = VectorType::new(ScalarType::Uint, 2);
    /// Vector with 3 unsigned integer components (`vec3<u32>`).
    pub const VEC3U: VectorType = VectorType::new(ScalarType::Uint, 3);
    /// Vector with 4 unsigned integer components (`vec4<u32>`).
    pub const VEC4U: VectorType = VectorType::new(ScalarType::Uint, 4);

    /// Create a new vector type.
    ///
    /// # Panic
    ///
    /// Panics if the component `count` is not 2/3/4.
    pub const fn new(elem_type: ScalarType, count: u8) -> Self {
        assert!(count >= 2 && count <= 4);
        Self { elem_type, count }
    }

    /// Scalar type of the individual vector elements (components).
    pub const fn elem_type(&self) -> ScalarType {
        self.elem_type
    }

    /// Number of components.
    pub const fn count(&self) -> usize {
        self.count as usize
    }

    /// Is the type a numeric type?
    ///
    /// See [`ScalarType::is_numeric()`] for a definition of a numeric type.
    ///
    /// [`ScalarType::is_numeric()`]: crate::ScalarType::is_numeric
    pub fn is_numeric(&self) -> bool {
        self.elem_type.is_numeric()
    }

    /// Size of a value of this type, in bytes.
    ///
    /// This corresponds to the size of a variable of that type when part of a
    /// struct in WGSL.
    pub const fn size(&self) -> usize {
        // https://gpuweb.github.io/gpuweb/wgsl/#alignment-and-size
        self.count() * self.elem_type.size()
    }

    /// Alignment of a value of this type, in bytes.
    ///
    /// This corresponds to the alignment of a variable of that type when part
    /// of a struct in WGSL.
    pub const fn align(&self) -> usize {
        // https://gpuweb.github.io/gpuweb/wgsl/#alignment-and-size
        if self.count >= 3 {
            4 * self.elem_type.align()
        } else if self.count == 2 {
            2 * self.elem_type.align()
        } else {
            self.elem_type.align()
        }
    }
}

impl ToWgslString for VectorType {
    fn to_wgsl_string(&self) -> String {
        format!("vec{}<{}>", self.count, self.elem_type.to_wgsl_string())
    }
}

/// Floating-point matrix type (`matCxR<f32>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct MatrixType {
    rows: u8,
    cols: u8,
}

impl MatrixType {
    /// Floating-point matrix of size 2x2 (`mat2x2<f32>`).
    pub const MAT2X2F: MatrixType = MatrixType::new(2, 2);
    /// Floating-point matrix of size 3x2 (`mat3x2<f32>`).
    pub const MAT3X2F: MatrixType = MatrixType::new(3, 2);
    /// Floating-point matrix of size 4x2 (`mat4x2<f32>`).
    pub const MAT4X2F: MatrixType = MatrixType::new(4, 2);
    /// Floating-point matrix of size 2x3 (`mat2x3<f32>`).
    pub const MAT2X3F: MatrixType = MatrixType::new(2, 3);
    /// Floating-point matrix of size 3x3 (`mat3x3<f32>`).
    pub const MAT3X3F: MatrixType = MatrixType::new(3, 3);
    /// Floating-point matrix of size 4x3 (`mat4x3<f32>`).
    pub const MAT4X3F: MatrixType = MatrixType::new(4, 3);
    /// Floating-point matrix of size 2x4 (`mat2x4<f32>`).
    pub const MAT2X4F: MatrixType = MatrixType::new(2, 4);
    /// Floating-point matrix of size 3x4 (`mat3x4<f32>`).
    pub const MAT3X4F: MatrixType = MatrixType::new(3, 4);
    /// Floating-point matrix of size 4x4 (`mat4x4<f32>`).
    pub const MAT4X4F: MatrixType = MatrixType::new(4, 4);

    /// Create a new matrix type.
    ///
    /// # Panic
    ///
    /// Panics if the number of columns or rows is not 2/3/4.
    pub const fn new(cols: u8, rows: u8) -> Self {
        Self { cols, rows }
    }

    /// Number of columns in the matrix.
    pub const fn cols(&self) -> usize {
        self.cols as usize
    }

    /// Number of rows in the matrix.
    pub const fn rows(&self) -> usize {
        self.rows as usize
    }

    /// Size of a value of this type, in bytes.
    ///
    /// This corresponds to the size of a variable of that type when part of a
    /// struct in WGSL.
    pub const fn size(&self) -> usize {
        // SizeOf(array<vecR, C>), which means matCx3 and matCx4 have same size
        // https://gpuweb.github.io/gpuweb/wgsl/#alignment-and-size
        if self.rows >= 3 {
            self.cols() * 4 * ScalarType::Float.size()
        } else {
            self.cols() * self.rows() * ScalarType::Float.size()
        }
    }

    /// Alignment of a value of this type, in bytes.
    ///
    /// This corresponds to the alignment of a variable of that type when part
    /// of a struct in WGSL.
    pub const fn align(&self) -> usize {
        // AlignOf(vecR), which means matCx3 and matCx4 have same align
        // https://gpuweb.github.io/gpuweb/wgsl/#alignment-and-size
        VectorType::new(ScalarType::Float, self.rows).align()
    }
}

impl ToWgslString for MatrixType {
    fn to_wgsl_string(&self) -> String {
        format!(
            "mat{}x{}<{}>",
            self.cols,
            self.rows,
            ScalarType::Float.to_wgsl_string()
        )
    }
}

/// Type of an [`Attribute`]'s value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ValueType {
    /// A scalar type (single value).
    Scalar(ScalarType),
    /// A vector type with 2 to 4 components.
    Vector(VectorType),
    /// A floating-point matrix type of size between 2x2 and 4x4.
    Matrix(MatrixType),
}

impl ValueType {
    /// Is the type a numeric type?
    pub fn is_numeric(&self) -> bool {
        match self {
            ValueType::Scalar(s) => s.is_numeric(),
            ValueType::Vector(v) => v.is_numeric(),
            ValueType::Matrix(_) => true,
        }
    }

    /// Is the type a scalar type?
    pub fn is_scalar(&self) -> bool {
        matches!(self, ValueType::Scalar(_))
    }

    /// Is the type a vector type?
    pub fn is_vector(&self) -> bool {
        matches!(self, ValueType::Vector(_))
    }

    /// Is the type a matrix type?
    pub fn is_matrix(&self) -> bool {
        matches!(self, ValueType::Matrix(_))
    }

    /// Size of a value of this type, in bytes.
    pub fn size(&self) -> usize {
        match self {
            ValueType::Scalar(s) => s.size(),
            ValueType::Vector(v) => v.size(),
            ValueType::Matrix(m) => m.size(),
        }
    }

    /// Alignment of a value of this type, in bytes.
    ///
    /// This corresponds to the alignment of a variable of that type when part
    /// of a struct in WGSL.
    pub fn align(&self) -> usize {
        match self {
            ValueType::Scalar(s) => s.align(),
            ValueType::Vector(v) => v.align(),
            ValueType::Matrix(m) => m.align(),
        }
    }
}

impl From<ScalarType> for ValueType {
    fn from(value: ScalarType) -> Self {
        ValueType::Scalar(value)
    }
}

impl From<VectorType> for ValueType {
    fn from(value: VectorType) -> Self {
        ValueType::Vector(value)
    }
}

impl From<MatrixType> for ValueType {
    fn from(value: MatrixType) -> Self {
        ValueType::Matrix(value)
    }
}

impl ToWgslString for ValueType {
    fn to_wgsl_string(&self) -> String {
        match self {
            ValueType::Scalar(s) => s.to_wgsl_string(),
            ValueType::Vector(v) => v.to_wgsl_string(),
            ValueType::Matrix(m) => m.to_wgsl_string(),
        }
    }
}

#[derive(Debug, Clone, Reflect)]
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
    pub const POSITION: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("position"),
        Value::Vector(VectorValue::new_vec3(Vec3::ZERO)),
    );

    pub const VELOCITY: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("velocity"),
        Value::Vector(VectorValue::new_vec3(Vec3::ZERO)),
    );

    pub const AGE: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("age"), Value::Scalar(ScalarValue::Float(0.)));

    pub const LIFETIME: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("lifetime"),
        Value::Scalar(ScalarValue::Float(1.)),
    );

    pub const COLOR: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("color"),
        Value::Scalar(ScalarValue::Uint(0xFFFFFFFFu32)),
    );

    pub const HDR_COLOR: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("hdr_color"),
        Value::Vector(VectorValue::new_vec4(Vec4::ONE)),
    );

    pub const ALPHA: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("alpha"),
        Value::Scalar(ScalarValue::Float(1.)),
    );

    pub const SIZE: &'static AttributeInner =
        &AttributeInner::new(Cow::Borrowed("size"), Value::Scalar(ScalarValue::Float(1.)));

    pub const SIZE2: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("size2"),
        Value::Vector(VectorValue::new_vec2(Vec2::ONE)),
    );

    pub const AXIS_X: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("axis_x"),
        Value::Vector(VectorValue::new_vec3(Vec3::X)),
    );

    pub const AXIS_Y: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("axis_y"),
        Value::Vector(VectorValue::new_vec3(Vec3::Y)),
    );

    pub const AXIS_Z: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("axis_z"),
        Value::Vector(VectorValue::new_vec3(Vec3::Z)),
    );

    pub const SPRITE_INDEX: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("sprite_index"),
        Value::Scalar(ScalarValue::Int(0)),
    );

    pub const F32_0: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("f32_0"),
        Value::Scalar(ScalarValue::Float(0.)),
    );

    pub const F32_1: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("f32_1"),
        Value::Scalar(ScalarValue::Float(0.)),
    );

    pub const F32_2: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("f32_2"),
        Value::Scalar(ScalarValue::Float(0.)),
    );

    pub const F32_3: &'static AttributeInner = &AttributeInner::new(
        Cow::Borrowed("f32_3"),
        Value::Scalar(ScalarValue::Float(0.)),
    );

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
/// See [`Attribute::all()`] for a list of supported attributes. Custom
/// attributes are not supported.
///
/// Attributes are indirectly added to an effect by adding [modifiers] requiring
/// them. Each modifier documents its required attributes. You can force a
/// single attribute by adding the [`SetAttributeModifier`].
///
/// [modifiers]: crate::modifier
/// [`SetAttributeModifier`]: crate::modifier::SetAttributeModifier
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

impl TypePath for Attribute {
    fn type_path() -> &'static str {
        static CELL: GenericTypePathCell = GenericTypePathCell::new();
        CELL.get_or_insert::<Self, _>(|| "bevy_hanabi::attribute::Attribute".to_owned())
    }

    fn short_type_path() -> &'static str {
        static CELL: GenericTypePathCell = GenericTypePathCell::new();
        CELL.get_or_insert::<Self, _>(|| "Attribute".to_owned())
    }

    fn type_ident() -> Option<&'static str> {
        Some("Attribute")
    }

    fn crate_name() -> Option<&'static str> {
        Some("bevy_hanabi")
    }

    fn module_path() -> Option<&'static str> {
        Some("bevy_hanabi::attribute")
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
            let info = StructInfo::new::<Self>(&fields);
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
        dynamic.set_represented_type(self.get_represented_type_info());
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

    fn get_represented_type_info(&self) -> Option<&'static TypeInfo> {
        Some(<Self as Typed>::type_info())
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
    /// # Name
    ///
    /// `position`
    ///
    /// # Type
    ///
    /// [`VectorType::VEC3F`] representing the XYZ coordinates of the position.
    ///
    /// [simulation space]: crate::SimulationSpace
    pub const POSITION: Attribute = Attribute(AttributeInner::POSITION);

    /// The particle velocity in [simulation space].
    ///
    /// # Name
    ///
    /// `velocity`
    ///
    /// # Type
    ///
    /// [`VectorType::VEC3F`] representing the XYZ coordinates of the velocity.
    ///
    /// [simulation space]: crate::SimulationSpace
    pub const VELOCITY: Attribute = Attribute(AttributeInner::VELOCITY);

    /// The age of the particle.
    ///
    /// Each time the particle is updated, the current simulation delta time is
    /// added to the particle's age. The age can be used to animate some other
    /// quantities; see the [`ColorOverLifetimeModifier`] for example.
    ///
    /// If the particle also has a lifetime (either a per-effect
    /// constant value, or a per-particle value stored in the
    /// [`Attribute::LIFETIME`] attribute), then when the age of the particle
    /// exceeds its lifetime, the particle dies and is not simulated nor
    /// rendered anymore.
    ///
    /// # Name
    ///
    /// `age`
    ///
    /// # Type
    ///
    /// [`ScalarType::Float`]
    ///
    /// [`ColorOverLifetimeModifier`]: crate::modifier::output::ColorOverLifetimeModifier
    pub const AGE: Attribute = Attribute(AttributeInner::AGE);

    /// The lifetime of the particle.
    ///
    /// This attribute stores a per-particle lifetime, which compared to the
    /// particle's age allows determining if the particle needs to be
    /// simulated and rendered. This requires the [`Attribute::AGE`]
    /// attribute to be used too.
    ///
    /// # Name
    ///
    /// `lifetime`
    ///
    /// # Type
    ///
    /// [`ScalarType::Float`]
    pub const LIFETIME: Attribute = Attribute(AttributeInner::LIFETIME);

    /// The particle's base color.
    ///
    /// This attribute stores a per-particle color, which can be used for
    /// various purposes, generally as the base color for rendering the
    /// particle.
    ///
    /// # Name
    ///
    /// `color`
    ///
    /// # Type
    ///
    /// [`ScalarType::Uint`] representing the RGBA components of the color
    /// encoded as `0xAABBGGRR`, with a single byte per component.
    pub const COLOR: Attribute = Attribute(AttributeInner::COLOR);

    /// The particle's base color (HDR).
    ///
    /// This attribute stores a per-particle HDR color, which can be used for
    /// various purposes, generally as the base color for rendering the
    /// particle.
    ///
    /// # Name
    ///
    /// `hdr_color`
    ///
    /// # Type
    ///
    /// [`VectorType::VEC4F`] representing the RGBA components of the color.
    /// Values are not clamped, and can be outside the \[0:1\] range to
    /// represent HDR values.
    pub const HDR_COLOR: Attribute = Attribute(AttributeInner::HDR_COLOR);

    /// The particle's opacity (alpha).
    ///
    /// This is a value in \[0:1\], where `0` corresponds to a fully transparent
    /// particle, and `1` to a fully opaque one.
    ///
    /// # Name
    ///
    /// `alpha`
    ///
    /// # Type
    ///
    /// [`ScalarType::Float`]
    pub const ALPHA: Attribute = Attribute(AttributeInner::ALPHA);

    /// The particle's uniform size.
    ///
    /// The particle is uniformly scaled by this size.
    ///
    /// # Name
    ///
    /// `size`
    ///
    /// # Type
    ///
    /// [`ScalarType::Float`]
    pub const SIZE: Attribute = Attribute(AttributeInner::SIZE);

    /// The particle's 2D size, for quad rendering.
    ///
    /// The particle, when drawn as a quad, is scaled along its local X and Y
    /// axes by these values.
    ///
    /// # Name
    ///
    /// `size2`
    ///
    /// # Type
    ///
    /// [`VectorType::VEC2F`] representing the XY sizes of the particle.
    pub const SIZE2: Attribute = Attribute(AttributeInner::SIZE2);

    /// The local X axis of the particle.
    ///
    /// This attribute stores a per-particle X axis, which defines the
    /// horizontal direction of a quad particle. This is generally used to
    /// re-orient the particle during rendering, for example to face the camera
    /// or another point of interest. For example, the [`OrientModifier`]
    /// modifies this attribute to make the particle face a specific item.
    ///
    /// # Name
    ///
    /// `axis_x`
    ///
    /// # Type
    ///
    /// [`VectorType::VEC3F`]
    ///
    /// [`OrientModifier`]: crate::modifier::output::OrientModifier
    pub const AXIS_X: Attribute = Attribute(AttributeInner::AXIS_X);

    /// The local Y axis of the particle.
    ///
    /// This attribute stores a per-particle Y axis, which defines the vertical
    /// direction of a quad particle. This is generally used to re-orient the
    /// particle during rendering, for example to face the camera or another
    /// point of interest. For example, the [`OrientModifier`] modifies this
    /// attribute to make the particle face a specific item.
    ///
    /// # Name
    ///
    /// `axis_y`
    ///
    /// # Type
    ///
    /// [`VectorType::VEC3F`]
    ///
    /// [`OrientModifier`]: crate::modifier::output::OrientModifier
    pub const AXIS_Y: Attribute = Attribute(AttributeInner::AXIS_Y);

    /// The local Z axis of the particle.
    ///
    /// This attribute stores a per-particle Z axis, which defines the normal to
    /// a quad particle's plane. This is generally used to re-orient the
    /// particle during rendering, for example to face the camera or another
    /// point of interest. For example, the [`OrientModifier`] modifies this
    /// attribute to make the particle face a specific item.
    ///
    /// # Name
    ///
    /// `axis_z`
    ///
    /// # Type
    ///
    /// [`VectorType::VEC3F`]
    ///
    /// [`OrientModifier`]: crate::modifier::output::OrientModifier
    pub const AXIS_Z: Attribute = Attribute(AttributeInner::AXIS_Z);

    /// The sprite index in a flipbook animation.
    ///
    /// This attribute stores the index of the sprite of a flipbook animation.
    /// This is used with the [`FlipbookModifier`].
    ///
    /// # Name
    ///
    /// `sprite_index`
    ///
    /// # Type
    ///
    /// [`ScalarType::Int`]
    ///
    /// [`FlipbookModifier`]: crate::modifier::output::FlipbookModifier
    pub const SPRITE_INDEX: Attribute = Attribute(AttributeInner::SPRITE_INDEX);

    /// A generic scalar float attribute.
    ///
    /// This attribute can be used for anything. It has no specific meaning. You
    /// can store whatever per-particle value you want in it (for example, at
    /// spawn time) and read it back later.
    ///
    /// # Name
    ///
    /// `f32_0`
    ///
    /// # Type
    ///
    /// [`ScalarType::Float`]
    pub const F32_0: Attribute = Attribute(AttributeInner::F32_0);

    /// A generic scalar float attribute.
    ///
    /// This attribute can be used for anything. It has no specific meaning. You
    /// can store whatever per-particle value you want in it (for example, at
    /// spawn time) and read it back later.
    ///
    /// # Name
    ///
    /// `f32_1`
    ///
    /// # Type
    ///
    /// [`ScalarType::Float`]
    pub const F32_1: Attribute = Attribute(AttributeInner::F32_1);

    /// A generic scalar float attribute.
    ///
    /// This attribute can be used for anything. It has no specific meaning. You
    /// can store whatever per-particle value you want in it (for example, at
    /// spawn time) and read it back later.
    ///
    /// # Name
    ///
    /// `f32_2`
    ///
    /// # Type
    ///
    /// [`ScalarType::Float`]
    pub const F32_2: Attribute = Attribute(AttributeInner::F32_2);

    /// A generic scalar float attribute.
    ///
    /// This attribute can be used for anything. It has no specific meaning. You
    /// can store whatever per-particle value you want in it (for example, at
    /// spawn time) and read it back later.
    ///
    /// # Name
    ///
    /// `f32_3`
    ///
    /// # Type
    ///
    /// [`ScalarType::Float`]
    pub const F32_3: Attribute = Attribute(AttributeInner::F32_3);

    /// Collection of all the existing particle attributes.
    const ALL: [Attribute; 17] = [
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
        Attribute::SPRITE_INDEX,
        Attribute::F32_0,
        Attribute::F32_1,
        Attribute::F32_2,
        Attribute::F32_3,
    ];

    /// Retrieve an attribute by its name.
    ///
    /// See [`Attribute::all()`] for the list of attributes, and the
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

    /// Get the list of all existing attributes.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// for attr in Attribute::all() {
    ///     println!("{}", attr.name());
    /// }
    /// ```
    pub fn all() -> &'static [Attribute] {
        &Self::ALL
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
///
/// # Construction
///
/// To create a particle layout you can either:
/// - Use [`ParticleLayout::default()`] to create the default layout, which
///   contains some default attributes commonly used for effects
///   ([`Attribute::POSITION`], [`Attribute::VELOCITY`], [`Attribute::AGE`],
///   [`Attribute::LIFETIME`]).
/// - Use [`ParticleLayout::empty()`] to create an empty layout without any
///   attribute.
/// - Use [`ParticleLayout::new()`] to create a [`ParticleLayoutBuilder`] and
///   append the necessary attributes manually then call [`build()`] to complete
///   the layout.
///
/// [`build()`]: crate::ParticleLayoutBuilder::build
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
    pub const fn empty() -> ParticleLayout {
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
    use naga::{front::wgsl::Frontend, proc::Layouter};

    // Ensure the size and alignment of all types conforms to the WGSL spec by
    // querying naga as a reference.
    #[test]
    fn value_type_align() {
        let mut frontend = Frontend::new();
        for (value_type, value) in &[
            (
                ValueType::Scalar(ScalarType::Float),
                Value::Scalar(ScalarValue::Float(0.)),
            ),
            // FIXME - We use a constant below, which has a size of 1 byte. For a field
            // inside a struct, the size of bool is undefined in WGSL/naga, and 4 bytes
            // in Hanabi. We probably can't test bool with naga here anyway.
            // (
            //     ValueType::Scalar(ScalarType::Bool),
            //     Value::Scalar(ScalarValue::Bool(true)),
            // ),
            (
                ValueType::Scalar(ScalarType::Int),
                Value::Scalar(ScalarValue::Int(-42)),
            ),
            (
                ValueType::Scalar(ScalarType::Uint),
                Value::Scalar(ScalarValue::Uint(999)),
            ),
            (
                ValueType::Vector(VectorType {
                    elem_type: ScalarType::Float,
                    count: 2,
                }),
                Value::Vector(VectorValue::new_vec2(Vec2::new(-0.5, 3.458))),
            ),
            (
                ValueType::Vector(VectorType {
                    elem_type: ScalarType::Float,
                    count: 3,
                }),
                Value::Vector(VectorValue::new_vec3(Vec3::new(-0.5, 3.458, -53.))),
            ),
            (
                ValueType::Vector(VectorType {
                    elem_type: ScalarType::Float,
                    count: 4,
                }),
                Value::Vector(VectorValue::new_vec4(Vec4::new(-0.5, 3.458, 0., -53.))),
            ),
        ] {
            assert_eq!(value.value_type(), *value_type);

            // Create a tiny WGSL snippet with the Value(Type) and parse it
            let src = format!("const x = {};", value.to_wgsl_string());
            let res = frontend.parse(&src);
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
            let (size, align) = {
                // Calculate the type layout according to WGSL
                let type_handle = cst.ty;
                let mut layouter = Layouter::default();
                assert!(layouter.update(m.to_ctx()).is_ok());
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
    const TEST_ATTR_INNER: &AttributeInner = &AttributeInner::new(
        Cow::Borrowed(TEST_ATTR_NAME),
        Value::Vector(VectorValue::new_vec3(Vec3::ONE)),
    );

    #[test]
    fn attr_new() {
        let attr = Attribute(TEST_ATTR_INNER);
        assert_eq!(attr.name(), TEST_ATTR_NAME);
        assert_eq!(attr.size(), 12);
        assert_eq!(attr.align(), 16);
        assert_eq!(
            attr.value_type(),
            ValueType::Vector(VectorType {
                elem_type: ScalarType::Float,
                count: 3
            })
        );
        assert_eq!(
            attr.default_value(),
            Value::Vector(VectorValue::new_vec3(Vec3::ONE))
        );
    }

    #[test]
    fn attr_from_name() {
        for attr in Attribute::all() {
            assert_eq!(Attribute::from_name(attr.name()), Some(*attr));
        }
    }

    #[test]
    fn attr_reflect() {
        let mut attr = Attribute(TEST_ATTR_INNER);

        let r = attr.as_reflect();
        assert_eq!(TypeRegistration::of::<Attribute>().type_id(), r.type_id());
        match r.reflect_ref() {
            ReflectRef::Struct(s) => {
                assert_eq!(2, s.field_len());

                assert_eq!(Some("name"), s.name_at(0));
                assert_eq!(Some("default_value"), s.name_at(1));
                assert_eq!(None, s.name_at(2));
                assert_eq!(None, s.name_at(9999));

                assert_eq!(
                    Some("alloc::borrow::Cow<str>"),
                    s.field("name")
                        .map(|f| f.get_represented_type_info().unwrap().type_path())
                );
                assert_eq!(
                    Some("bevy_hanabi::graph::Value"),
                    s.field("default_value")
                        .map(|f| f.get_represented_type_info().unwrap().type_path())
                );
                assert!(s.field("DUMMY").is_none());
                assert!(s.field("").is_none());

                for f in s.iter_fields() {
                    let tp = f.get_represented_type_info().unwrap().type_path();
                    assert!(
                        tp.contains("alloc::borrow::Cow<str>")
                            || tp.contains("bevy_hanabi::graph::Value")
                    );
                }

                let d = s.clone_dynamic();
                assert_eq!(
                    TypeRegistration::of::<Attribute>().type_id(),
                    d.get_represented_type_info().unwrap().type_id()
                );
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
            let r_attr = Attribute::from_reflect(r).expect(
                "Cannot find
    attribute by name",
            );
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
            // Serialize; this produces just the name of the attribute, which    uniquely
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

    const F1_INNER: &AttributeInner =
        &AttributeInner::new(Cow::Borrowed("F1"), Value::Scalar(ScalarValue::Float(3.)));
    const F1B_INNER: &AttributeInner =
        &AttributeInner::new(Cow::Borrowed("F1B"), Value::Scalar(ScalarValue::Float(5.)));
    const F2_INNER: &AttributeInner = &AttributeInner::new(
        Cow::Borrowed("F2"),
        Value::Vector(VectorValue::new_vec2(Vec2::ZERO)),
    );
    const F2B_INNER: &AttributeInner = &AttributeInner::new(
        Cow::Borrowed("F2B"),
        Value::Vector(VectorValue::new_vec2(Vec2::ONE)),
    );
    const F3_INNER: &AttributeInner = &AttributeInner::new(
        Cow::Borrowed("F3"),
        Value::Vector(VectorValue::new_vec3(Vec3::ZERO)),
    );
    const F3B_INNER: &AttributeInner = &AttributeInner::new(
        Cow::Borrowed("F3B"),
        Value::Vector(VectorValue::new_vec3(Vec3::ONE)),
    );
    const F4_INNER: &AttributeInner = &AttributeInner::new(
        Cow::Borrowed("F4"),
        Value::Vector(VectorValue::new_vec4(Vec4::ZERO)),
    );
    const F4B_INNER: &AttributeInner = &AttributeInner::new(
        Cow::Borrowed("F4B"),
        Value::Vector(VectorValue::new_vec4(Vec4::ONE)),
    );

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
