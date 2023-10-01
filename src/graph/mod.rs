//! Effect graph and language definition.
//!
//! This module contains the elements used to build an _effect graph_, a fully
//! customizable description of a visual effect.
//!
//! The effect graph API is composed of two layers:
//! - The [**Expression API**] provides a micro-language to build shader
//!   expressions ([`Expr`]) via code. Those expressions are composed together
//!   into complex behaviors assigned to the input values of some modifiers.
//!   This enables complete customizing of the modifiers. This API is focused on
//!   runtime execution and asset serialization. It provides the user with a way
//!   to _indirectly_ write effect shaders without any knowledge about shaders,
//!   and with a framework which guarantees the shader code generated is
//!   correct.
//! - The [**Node API**] provides a higher-level API built on top of the
//!   Expression API. Its use is entirely optional. It defines a node graph
//!   ([`Graph`]), where each node ([`Node`]) represents an expression or a
//!   modifier. Nodes are linked together to implicitly build expressions. This
//!   API focuses on asset editing, with the explicit intent to be used to build
//!   a (visual) effect editor.
//!
//! # API status
//!
//! Currently effect graphs are not fully available yet; only some preview
//! elements exist.
//!
//! The Expression API already contains a good set of expressions, and some
//! modifiers have already been converted to accept expressions for their input
//! fields. Its generally in a reasonable shape for early adoption.
//!
//! The Node API contains a basic node and graph definition, which is entirely
//! experimental at this stage.
//!
//! We recommend starting to familiarize yourself with effect graphs, and
//! starting to port your code to use expressions ([`Expr`]), as the entire
//! library is moving in the direction of adopting effect graphs across the
//! board.
//!
//! [**Expression API**]: crate::graph::expr
//! [**Node API**]: crate::graph::node

use std::fmt::Debug;

use bevy::{
    math::{BVec2, BVec3, BVec4, IVec2, IVec3, IVec4, Mat2, Mat3, Mat4, Vec2, Vec3, Vec3A, Vec4},
    reflect::Reflect,
    utils::FloatOrd,
};
use serde::{Deserialize, Serialize};

use crate::{MatrixType, ScalarType, ToWgslString, ValueType, VectorType};

pub mod expr;
pub mod node;

pub use expr::{
    AttributeExpr, BinaryOperator, BuiltInExpr, BuiltInOperator, EvalContext, Expr, ExprError,
    ExprHandle, ExprWriter, LiteralExpr, Module, PropertyExpr, UnaryOperator, WriterExpr,
};
pub use node::{
    AddNode, AttributeNode, DivNode, Graph, MulNode, Node, NormalizeNode, Slot, SlotDir, SlotId,
    SubNode, TimeNode,
};

/// Variant storage for a scalar value.
#[derive(Debug)]
#[non_exhaustive]
pub enum ScalarValueMut<'a> {
    /// Single `bool` value.
    Bool(&'a mut bool),
    /// Single `f32` value.
    Float(&'a mut f32),
    /// Single `i32` value.
    Int(&'a mut i32),
    /// Single `u32` value.
    Uint(&'a mut u32),
}

/// Variant storage for a single (scalar) value.
///
/// The value implements total equality and hashing. For [`ScalarValue::Float`],
/// the total equality is based on [`FloatOrd`]. Values of different types
/// compare inequally to each other, even if the values would be equal after
/// casting, and hash differently. To compare two [`ScalarValue`] taking into
/// account casting, use [`cast_eq()`].
///
/// [`cast_eq()`]: crate::graph::ScalarValue::cast_eq
#[derive(Debug, Clone, Copy, Reflect, Serialize, Deserialize)]
#[non_exhaustive] // f16 not supported yet
pub enum ScalarValue {
    /// Single `bool` value.
    Bool(bool),
    /// Single `f32` value.
    Float(f32),
    /// Single `i32` value.
    Int(i32),
    /// Single `u32` value.
    Uint(u32),
}

impl ScalarValue {
    /// The value `false` when a boolean value is stored internally.
    pub(crate) const BOOL_FALSE_STORAGE: u32 = 0u32;

    /// The value `true` when a boolean value is stored internally.
    pub(crate) const BOOL_TRUE_STORAGE: u32 = 0xFF_FF_FF_FFu32;

    /// The value `false` and `true` when a boolean value is stored internally.
    pub(crate) const BOOL_STORAGE: [u32; 2] = [Self::BOOL_FALSE_STORAGE, Self::BOOL_TRUE_STORAGE];

    /// Convert this value to a `bool` value.
    ///
    /// Any non-zero value converts to `true`, while a zero value converts to
    /// `false`.
    pub fn as_bool(&self) -> bool {
        match *self {
            ScalarValue::Bool(b) => b,
            ScalarValue::Float(f) => f != 0f32,
            ScalarValue::Int(i) => i != 0,
            ScalarValue::Uint(u) => u != 0,
        }
    }

    /// Convert this value to a floating-point value.
    ///
    /// Boolean values convert to 0 or 1. Numeric values are cast to `f32`.
    pub fn as_f32(&self) -> f32 {
        match *self {
            ScalarValue::Bool(b) => {
                if b {
                    1f32
                } else {
                    0f32
                }
            }
            ScalarValue::Float(f) => f,
            ScalarValue::Int(i) => i as f32,
            ScalarValue::Uint(u) => u as f32,
        }
    }

    /// Convert this value to an `i32` value.
    ///
    /// Boolean values convert to 0 or 1. Numeric values are cast to `i32`.
    pub fn as_i32(&self) -> i32 {
        match *self {
            ScalarValue::Bool(b) => {
                if b {
                    1i32
                } else {
                    0i32
                }
            }
            ScalarValue::Float(f) => f as i32,
            ScalarValue::Int(i) => i,
            ScalarValue::Uint(u) => u as i32,
        }
    }

    /// Convert this value to an `u32` value.
    ///
    /// Boolean values convert to 0 or 1. Numeric values are cast to `u32`.
    pub fn as_u32(&self) -> u32 {
        match *self {
            ScalarValue::Bool(b) => {
                if b {
                    1u32
                } else {
                    0u32
                }
            }
            ScalarValue::Float(f) => f as u32,
            ScalarValue::Int(i) => i as u32,
            ScalarValue::Uint(u) => u,
        }
    }

    /// Get the value as a binary blob ready for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            ScalarValue::Bool(_) => panic!("Cannot convert scalar bool to byte slice."),
            ScalarValue::Float(f) => bytemuck::cast_slice::<f32, u8>(std::slice::from_ref(f)),
            ScalarValue::Int(i) => bytemuck::cast_slice::<i32, u8>(std::slice::from_ref(i)),
            ScalarValue::Uint(u) => bytemuck::cast_slice::<u32, u8>(std::slice::from_ref(u)),
        }
    }

    /// Get the raw internal storage value representing this scalar value.
    ///
    /// Used internally for some conversion operations. The representation is
    /// not guaranteed to be stable.
    fn as_storage(&self) -> u32 {
        match self {
            ScalarValue::Bool(b) => {
                if *b {
                    Self::BOOL_TRUE_STORAGE
                } else {
                    0u32
                }
            }
            ScalarValue::Float(f) => bytemuck::cast::<f32, u32>(*f),
            ScalarValue::Int(i) => bytemuck::cast::<i32, u32>(*i),
            ScalarValue::Uint(u) => *u,
        }
    }

    /// Get the scalar type of this value.
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            ScalarValue::Bool(_) => ScalarType::Bool,
            ScalarValue::Float(_) => ScalarType::Float,
            ScalarValue::Int(_) => ScalarType::Int,
            ScalarValue::Uint(_) => ScalarType::Uint,
        }
    }

    /// Check equality with another value by casting the other value to this
    /// value's type.
    ///
    /// Floating point values ([`ScalarValue::Float`]) use [`FloatOrd`] for
    /// total equality.
    pub fn cast_eq(&self, other: &Self) -> bool {
        match *self {
            ScalarValue::Bool(b) => b == other.as_bool(),
            ScalarValue::Float(f) => FloatOrd(f) == FloatOrd(other.as_f32()),
            ScalarValue::Int(i) => i == other.as_i32(),
            ScalarValue::Uint(u) => u == other.as_u32(),
        }
    }

    // fn binary_op(&self, other: &Self, op: BinaryOperator) -> Self {
    //     match *self {
    //         ScalarValue::Bool(_) => panic!("Cannot apply binary operation to
    // boolean value."),         ScalarValue::Float(f) =>
    // ScalarValue::Float(op.apply_f32(f, other.as_f32())),
    //         ScalarValue::Int(i) => ScalarValue::Int(op.apply_i32(i,
    // other.as_i32())),         ScalarValue::Uint(u) =>
    // ScalarValue::Uint(op.apply_u32(u, other.as_u32())),     }
    // }
}

impl PartialEq for ScalarValue {
    fn eq(&self, other: &Self) -> bool {
        match *self {
            ScalarValue::Bool(b) => match *other {
                ScalarValue::Bool(b2) => b == b2,
                _ => false,
            },
            ScalarValue::Float(f) => match *other {
                ScalarValue::Float(f2) => FloatOrd(f) == FloatOrd(f2),
                _ => false,
            },
            ScalarValue::Int(i) => match *other {
                ScalarValue::Int(i2) => i == i2,
                _ => false,
            },
            ScalarValue::Uint(u) => match *other {
                ScalarValue::Uint(u2) => u == u2,
                _ => false,
            },
        }
    }
}

impl Eq for ScalarValue {}

impl std::hash::Hash for ScalarValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash some u8 to encode the enum variant, then the actual value
        match *self {
            ScalarValue::Bool(b) => {
                1_u8.hash(state);
                b.hash(state)
            }
            ScalarValue::Float(f) => {
                2_u8.hash(state);
                FloatOrd(f).hash(state);
            }
            ScalarValue::Int(i) => {
                3_u8.hash(state);
                i.hash(state);
            }
            ScalarValue::Uint(u) => {
                4_u8.hash(state);
                u.hash(state);
            }
        }
    }
}

impl ToWgslString for ScalarValue {
    fn to_wgsl_string(&self) -> String {
        match *self {
            ScalarValue::Bool(b) => b.to_wgsl_string(),
            ScalarValue::Float(f) => f.to_wgsl_string(),
            ScalarValue::Int(i) => i.to_wgsl_string(),
            ScalarValue::Uint(u) => u.to_wgsl_string(),
        }
    }
}

// impl BinaryOperation for ScalarValue {
//     fn apply(&self, other: &Self, op: BinaryOperator) -> Self {
//         match *self {
//             ScalarValue::Bool(_) => panic!("Cannot apply binary operation to
// boolean value."),             ScalarValue::Float(f) =>
// ScalarValue::Float(op.apply_f32(f, other.as_f32())),
// ScalarValue::Int(i) => ScalarValue::Int(op.apply_i32(i, other.as_i32())),
//             ScalarValue::Uint(u) => ScalarValue::Uint(op.apply_u32(u,
// other.as_u32())),         }
//     }
// }

impl From<bool> for ScalarValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<f32> for ScalarValue {
    fn from(value: f32) -> Self {
        Self::Float(value)
    }
}

impl From<i32> for ScalarValue {
    fn from(value: i32) -> Self {
        Self::Int(value)
    }
}

impl From<u32> for ScalarValue {
    fn from(value: u32) -> Self {
        Self::Uint(value)
    }
}

/// Trait to convert the elements of a vector.
pub trait ElemType {
    /// The zero element of the type.
    const ZERO: Self;

    /// Get the N-th component of the vector given its raw storage.
    fn get_n<const N: usize>(storage: &[u32; 4]) -> Self;

    /// Get the given component of the vector given its raw storage.
    fn get(index: usize, storage: &[u32; 4]) -> Self;

    /// Get a slice of all the components of the vector given its raw storage.
    ///
    /// This is only valid for numeric types, and will panic for a boolean type.
    fn get_all(storage: &[u32; 4], count: usize) -> &[Self]
    where
        Self: Sized;

    /// Get a mutable reference to the given component of the vector from within
    /// its raw storage.
    fn get_mut(index: usize, storage: &mut [u32; 4]) -> &mut Self;
}

impl ElemType for bool {
    const ZERO: Self = false;

    fn get_n<const N: usize>(storage: &[u32; 4]) -> Self {
        storage[N] != 0
    }

    fn get(index: usize, storage: &[u32; 4]) -> Self {
        storage[index] != 0
    }

    fn get_all(_storage: &[u32; 4], _count: usize) -> &[Self] {
        panic!("Cannot get bool element type as slice.");
    }

    fn get_mut(_index: usize, _storage: &mut [u32; 4]) -> &mut Self {
        panic!("Cannot get bool element type as mutable reference.");
    }
}

impl ElemType for f32 {
    const ZERO: Self = 0f32;

    fn get_n<const N: usize>(storage: &[u32; 4]) -> Self {
        bytemuck::cast_slice::<u32, f32>(storage)[N]
    }

    fn get(index: usize, storage: &[u32; 4]) -> Self {
        bytemuck::cast_slice::<u32, f32>(storage)[index]
    }

    fn get_all(storage: &[u32; 4], count: usize) -> &[Self] {
        &bytemuck::cast_slice::<u32, f32>(storage)[..count]
    }

    fn get_mut(index: usize, storage: &mut [u32; 4]) -> &mut Self {
        bytemuck::cast_mut::<u32, f32>(&mut storage[index])
    }
}

impl ElemType for i32 {
    const ZERO: Self = 0i32;

    fn get_n<const N: usize>(storage: &[u32; 4]) -> Self {
        bytemuck::cast_slice::<u32, i32>(storage)[N]
    }

    fn get(index: usize, storage: &[u32; 4]) -> Self {
        bytemuck::cast_slice::<u32, i32>(storage)[index]
    }

    fn get_all(storage: &[u32; 4], count: usize) -> &[Self] {
        &bytemuck::cast_slice::<u32, i32>(storage)[..count]
    }

    fn get_mut(index: usize, storage: &mut [u32; 4]) -> &mut Self {
        bytemuck::cast_mut::<u32, i32>(&mut storage[index])
    }
}

impl ElemType for u32 {
    const ZERO: Self = 0u32;

    fn get_n<const N: usize>(storage: &[u32; 4]) -> Self {
        storage[N]
    }

    fn get(index: usize, storage: &[u32; 4]) -> Self {
        storage[index]
    }

    fn get_all(storage: &[u32; 4], count: usize) -> &[Self] {
        &storage[..count]
    }

    fn get_mut(index: usize, storage: &mut [u32; 4]) -> &mut Self {
        &mut storage[index]
    }
}

/// Variant storage for a vector value.
#[derive(Debug, Clone, Copy, Reflect, Serialize, Deserialize)]
pub struct VectorValue {
    vector_type: VectorType,
    storage: [u32; 4],
}

impl VectorValue {
    /// Workaround for `impl const From<BVec2>`.
    #[allow(unsafe_code)]
    pub const fn new_bvec2(value: BVec2) -> Self {
        Self {
            vector_type: VectorType::VEC2B,
            storage: [
                if value.x {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0u32
                },
                if value.y {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0u32
                },
                0u32,
                0u32,
            ],
        }
    }

    /// Workaround for `impl const From<BVec3>`.
    #[allow(unsafe_code)]
    pub const fn new_bvec3(value: BVec3) -> Self {
        Self {
            vector_type: VectorType::VEC3B,
            storage: [
                if value.x {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0u32
                },
                if value.y {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0u32
                },
                if value.z {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0u32
                },
                0u32,
            ],
        }
    }

    /// Workaround for `impl const From<BVec4>`.
    #[allow(unsafe_code)]
    pub const fn new_bvec4(value: BVec4) -> Self {
        Self {
            vector_type: VectorType::VEC4B,
            storage: [
                if value.x {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0u32
                },
                if value.y {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0u32
                },
                if value.z {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0u32
                },
                if value.w {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0u32
                },
            ],
        }
    }

    /// Workaround for `impl const From<Vec2>`.
    #[allow(unsafe_code)]
    pub const fn new_vec2(value: Vec2) -> Self {
        Self {
            vector_type: VectorType::VEC2F,
            storage: [
                unsafe { std::mem::transmute(value.x) },
                unsafe { std::mem::transmute(value.y) },
                0u32,
                0u32,
            ],
        }
    }

    /// Workaround for `impl const From<Vec3>`.
    #[allow(unsafe_code)]
    pub const fn new_vec3(value: Vec3) -> Self {
        Self {
            vector_type: VectorType::VEC3F,
            storage: [
                unsafe { std::mem::transmute(value.x) },
                unsafe { std::mem::transmute(value.y) },
                unsafe { std::mem::transmute(value.z) },
                0u32,
            ],
        }
    }

    /// Workaround for `impl const From<Vec4>`.
    #[allow(unsafe_code)]
    pub const fn new_vec4(value: Vec4) -> Self {
        Self {
            vector_type: VectorType::VEC4F,
            storage: unsafe { std::mem::transmute(value.to_array()) },
        }
    }

    /// Workaround for `impl const From<IVec2>`.
    #[allow(unsafe_code)]
    pub const fn new_ivec2(value: IVec2) -> Self {
        Self {
            vector_type: VectorType::VEC2I,
            storage: [
                unsafe { std::mem::transmute(value.x) },
                unsafe { std::mem::transmute(value.y) },
                0u32,
                0u32,
            ],
        }
    }

    /// Workaround for `impl const From<IVec3>`.
    #[allow(unsafe_code)]
    pub const fn new_ivec3(value: IVec3) -> Self {
        Self {
            vector_type: VectorType::VEC3I,
            storage: [
                unsafe { std::mem::transmute(value.x) },
                unsafe { std::mem::transmute(value.y) },
                unsafe { std::mem::transmute(value.z) },
                0u32,
            ],
        }
    }

    /// Workaround for `impl const From<IVec4>`.
    #[allow(unsafe_code)]
    pub const fn new_ivec4(value: IVec4) -> Self {
        Self {
            vector_type: VectorType::VEC4I,
            storage: unsafe { std::mem::transmute(value.to_array()) },
        }
    }

    /// Create a new [`VectorValue`] from a 2D vector of `u32`.
    ///
    /// Note that due to the lack of `UVec2` in glam, this method takes
    /// individual vector components instead.
    pub const fn new_uvec2(x: u32, y: u32) -> Self {
        Self {
            vector_type: VectorType::VEC2U,
            storage: [x, y, 0, 0],
        }
    }

    /// Create a new [`VectorValue`] from a 3D vector of `u32`.
    ///
    /// Note that due to the lack of `UVec3` in glam, this method takes
    /// individual vector components instead.
    pub const fn new_uvec3(x: u32, y: u32, z: u32) -> Self {
        Self {
            vector_type: VectorType::VEC3U,
            storage: [x, y, z, 0],
        }
    }

    /// Create a new [`VectorValue`] from a 4D vector of `u32`.
    ///
    /// Note that due to the lack of `UVec4` in glam, this method takes
    /// individual vector components instead.
    pub const fn new_uvec4(x: u32, y: u32, z: u32, w: u32) -> Self {
        Self {
            vector_type: VectorType::VEC4U,
            storage: [x, y, z, w],
        }
    }

    /// Create a new vector by "splatting" a scalar value into all components.
    ///
    /// # Panic
    ///
    /// Panics if the component `count` is not 2/3/4.
    pub fn splat(value: &ScalarValue, count: u8) -> Self {
        let raw_value = value.as_storage();
        Self {
            vector_type: VectorType::new(value.scalar_type(), count),
            storage: [raw_value; 4],
        }
    }

    /// Get the type of the vector elements.
    ///
    /// This is a convenience shortcut for `self.vector_type().elem_type()`.
    pub fn elem_type(&self) -> ScalarType {
        self.vector_type.elem_type()
    }

    /// Get the vector type itself.
    pub fn vector_type(&self) -> VectorType {
        self.vector_type
    }

    /// Get the scalar value of the N-th element of the vector.
    pub fn value_n<const N: usize>(&self) -> ScalarValue {
        match self.elem_type() {
            ScalarType::Bool => ScalarValue::Bool(self.get_n::<bool, N>()),
            ScalarType::Float => ScalarValue::Float(self.get_n::<f32, N>()),
            ScalarType::Int => ScalarValue::Int(self.get_n::<i32, N>()),
            ScalarType::Uint => ScalarValue::Uint(self.get_n::<u32, N>()),
        }
    }

    /// Get the scalar value of an element of the vector.
    pub fn value(&self, index: usize) -> ScalarValue {
        match self.elem_type() {
            ScalarType::Bool => ScalarValue::Bool(self.get::<bool>(index)),
            ScalarType::Float => ScalarValue::Float(self.get::<f32>(index)),
            ScalarType::Int => ScalarValue::Int(self.get::<i32>(index)),
            ScalarType::Uint => ScalarValue::Uint(self.get::<u32>(index)),
        }
    }

    /// Get the scalar value of an element of the vector.
    pub fn value_mut(&mut self, index: usize) -> ScalarValueMut {
        match self.elem_type() {
            ScalarType::Bool => ScalarValueMut::Bool(self.get_mut::<bool>(index)),
            ScalarType::Float => ScalarValueMut::Float(self.get_mut::<f32>(index)),
            ScalarType::Int => ScalarValueMut::Int(self.get_mut::<i32>(index)),
            ScalarType::Uint => ScalarValueMut::Uint(self.get_mut::<u32>(index)),
        }
    }

    /// Get the value of the N-th element of the vector.
    pub fn get_n<T: ElemType, const N: usize>(&self) -> T {
        if self.vector_type.count() > N {
            T::get_n::<N>(&self.storage)
        } else {
            T::ZERO
        }
    }

    /// Get the value of an element of the vector.
    pub fn get<T: ElemType>(&self, index: usize) -> T {
        if index < self.vector_type.count() {
            T::get(index, &self.storage)
        } else {
            T::ZERO
        }
    }

    /// Get the value of an element of the vector.
    fn get_mut<T: ElemType>(&mut self, index: usize) -> &mut T {
        assert!(index < self.vector_type.count());
        T::get_mut(index, &mut self.storage)
    }

    /// Get a slice of all the values of the vector.
    ///
    /// This is only valid for numeric types, and will panic for a boolean type.
    pub fn get_all<T: ElemType>(&self) -> &[T] {
        T::get_all(&self.storage, self.vector_type.count())
    }

    /// Cast this vector value to a [`BVec2`].
    ///
    /// # Panic
    ///
    /// Panics if the current vector type is not [`VectorType::VEC2B`].
    pub fn as_bvec2(&self) -> BVec2 {
        assert_eq!(self.vector_type, VectorType::VEC2B);
        BVec2::new(self.storage[0] != 0, self.storage[1] != 0)
    }

    /// Cast this vector value to a [`BVec3`].
    ///
    /// # Panic
    ///
    /// Panics if the current vector type is not [`VectorType::VEC3B`].
    pub fn as_bvec3(&self) -> BVec3 {
        assert_eq!(self.vector_type, VectorType::VEC3B);
        BVec3::new(
            self.storage[0] != 0,
            self.storage[1] != 0,
            self.storage[2] != 0,
        )
    }

    /// Cast this vector value to a [`BVec4`].
    ///
    /// # Panic
    ///
    /// Panics if the current vector type is not [`VectorType::VEC4B`].
    pub fn as_bvec4(&self) -> BVec4 {
        assert_eq!(self.vector_type, VectorType::VEC4B);
        BVec4::new(
            self.storage[0] != 0,
            self.storage[1] != 0,
            self.storage[2] != 0,
            self.storage[3] != 0,
        )
    }

    /// Cast this vector value to a [`Vec2`].
    ///
    /// # Panic
    ///
    /// Panics if the current vector type is not [`VectorType::VEC2F`].
    pub fn as_vec2(&self) -> Vec2 {
        assert_eq!(self.vector_type, VectorType::VEC2F);
        Vec2::from_slice(bytemuck::cast_slice::<u32, f32>(&self.storage))
    }

    /// Cast this vector value to a [`Vec3`].
    ///
    /// # Panic
    ///
    /// Panics if the current vector type is not [`VectorType::VEC3F`].
    pub fn as_vec3(&self) -> Vec3 {
        assert_eq!(self.vector_type, VectorType::VEC3F);
        Vec3::from_slice(bytemuck::cast_slice::<u32, f32>(&self.storage))
    }

    /// Cast this vector value to a [`Vec3A`].
    ///
    /// # Panic
    ///
    /// Panics if the current vector type is not [`VectorType::VEC3F`].
    pub fn as_vec3a(&self) -> Vec3A {
        assert_eq!(self.vector_type, VectorType::VEC3F);
        Vec3A::from_slice(bytemuck::cast_slice::<u32, f32>(&self.storage))
    }

    /// Cast this vector value to a [`Vec4`].
    ///
    /// # Panic
    ///
    /// Panics if the current vector type is not [`VectorType::VEC4F`].
    pub fn as_vec4(&self) -> Vec4 {
        assert_eq!(self.vector_type, VectorType::VEC4F);
        Vec4::from_slice(bytemuck::cast_slice::<u32, f32>(&self.storage))
    }

    /// Cast this vector value to a [`IVec2`].
    ///
    /// # Panic
    ///
    /// Panics if the current vector type is not [`VectorType::VEC2I`].
    pub fn as_ivec2(&self) -> IVec2 {
        assert_eq!(self.vector_type, VectorType::VEC2I);
        IVec2::from_slice(bytemuck::cast_slice::<u32, i32>(&self.storage))
    }

    /// Cast this vector value to a [`IVec3`].
    ///
    /// # Panic
    ///
    /// Panics if the current vector type is not [`VectorType::VEC3I`].
    pub fn as_ivec3(&self) -> IVec3 {
        assert_eq!(self.vector_type, VectorType::VEC3I);
        IVec3::from_slice(bytemuck::cast_slice::<u32, i32>(&self.storage))
    }

    /// Cast this vector value to a [`IVec4`].
    ///
    /// # Panic
    ///
    /// Panics if the current vector type is not [`VectorType::VEC4I`].
    pub fn as_ivec4(&self) -> IVec4 {
        assert_eq!(self.vector_type, VectorType::VEC4I);
        IVec4::from_slice(bytemuck::cast_slice::<u32, i32>(&self.storage))
    }

    /// Get the value as a binary blob ready for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        let count = self.vector_type.count();
        bytemuck::cast_slice::<u32, u8>(&self.storage[..count])
    }

    /// Check equality with another value by casting the other value to this
    /// value's type.
    ///
    /// Floating point values ([`ScalarValue::Float`]) use [`FloatOrd`] for
    /// total equality.
    ///
    /// Vectors of different size always compare inequal (returns `false`).
    pub fn cast_eq(&self, other: &Self) -> bool {
        let count = self.vector_type().count();
        if count != other.vector_type().count() {
            return false;
        }
        match self.elem_type() {
            ScalarType::Bool => {
                let s = self.cast_bool();
                let o = other.cast_bool();
                s[..count] == o[..count]
            }
            ScalarType::Float => {
                let s = self.cast_f32();
                let o = other.cast_f32();
                s[..count] == o[..count]
            }
            ScalarType::Int => {
                let s = self.cast_i32();
                let o = other.cast_i32();
                s[..count] == o[..count]
            }
            ScalarType::Uint => {
                let s = self.cast_u32();
                let o = other.cast_u32();
                s[..count] == o[..count]
            }
        }
    }

    fn cast_bool(&self) -> [u32; 4] {
        match self.elem_type() {
            ScalarType::Bool => self.storage,
            ScalarType::Float => [
                ScalarValue::BOOL_STORAGE
                    [(bytemuck::cast::<u32, f32>(self.storage[0]) != 0.) as usize],
                ScalarValue::BOOL_STORAGE
                    [(bytemuck::cast::<u32, f32>(self.storage[1]) != 0.) as usize],
                ScalarValue::BOOL_STORAGE
                    [(bytemuck::cast::<u32, f32>(self.storage[2]) != 0.) as usize],
                ScalarValue::BOOL_STORAGE
                    [(bytemuck::cast::<u32, f32>(self.storage[3]) != 0.) as usize],
            ],
            ScalarType::Int => [
                ScalarValue::BOOL_STORAGE
                    [(bytemuck::cast::<u32, i32>(self.storage[0]) != 0) as usize],
                ScalarValue::BOOL_STORAGE
                    [(bytemuck::cast::<u32, i32>(self.storage[1]) != 0) as usize],
                ScalarValue::BOOL_STORAGE
                    [(bytemuck::cast::<u32, i32>(self.storage[2]) != 0) as usize],
                ScalarValue::BOOL_STORAGE
                    [(bytemuck::cast::<u32, i32>(self.storage[3]) != 0) as usize],
            ],
            ScalarType::Uint => [
                ScalarValue::BOOL_STORAGE[(self.storage[0] != 0) as usize],
                ScalarValue::BOOL_STORAGE[(self.storage[1] != 0) as usize],
                ScalarValue::BOOL_STORAGE[(self.storage[2] != 0) as usize],
                ScalarValue::BOOL_STORAGE[(self.storage[3] != 0) as usize],
            ],
        }
    }

    fn cast_f32(&self) -> [FloatOrd; 4] {
        match self.elem_type() {
            ScalarType::Bool => {
                let ft = [FloatOrd(0.), FloatOrd(1.)];
                [
                    ft[(self.storage[0] != 0) as usize],
                    ft[(self.storage[1] != 0) as usize],
                    ft[(self.storage[2] != 0) as usize],
                    ft[(self.storage[3] != 0) as usize],
                ]
            }
            ScalarType::Float => [
                FloatOrd(bytemuck::cast::<u32, f32>(self.storage[0])),
                FloatOrd(bytemuck::cast::<u32, f32>(self.storage[1])),
                FloatOrd(bytemuck::cast::<u32, f32>(self.storage[2])),
                FloatOrd(bytemuck::cast::<u32, f32>(self.storage[3])),
            ],
            ScalarType::Int => [
                FloatOrd(bytemuck::cast::<u32, i32>(self.storage[0]) as f32),
                FloatOrd(bytemuck::cast::<u32, i32>(self.storage[1]) as f32),
                FloatOrd(bytemuck::cast::<u32, i32>(self.storage[2]) as f32),
                FloatOrd(bytemuck::cast::<u32, i32>(self.storage[3]) as f32),
            ],
            ScalarType::Uint => [
                FloatOrd(self.storage[0] as f32),
                FloatOrd(self.storage[1] as f32),
                FloatOrd(self.storage[2] as f32),
                FloatOrd(self.storage[3] as f32),
            ],
        }
    }

    fn cast_i32(&self) -> [i32; 4] {
        match self.elem_type() {
            ScalarType::Bool => {
                let ft = [0, 1];
                [
                    ft[(self.storage[0] != 0) as usize],
                    ft[(self.storage[1] != 0) as usize],
                    ft[(self.storage[2] != 0) as usize],
                    ft[(self.storage[3] != 0) as usize],
                ]
            }
            ScalarType::Float => [
                bytemuck::cast::<u32, f32>(self.storage[0]) as i32,
                bytemuck::cast::<u32, f32>(self.storage[1]) as i32,
                bytemuck::cast::<u32, f32>(self.storage[2]) as i32,
                bytemuck::cast::<u32, f32>(self.storage[3]) as i32,
            ],
            ScalarType::Int => [
                bytemuck::cast::<u32, i32>(self.storage[0]),
                bytemuck::cast::<u32, i32>(self.storage[1]),
                bytemuck::cast::<u32, i32>(self.storage[2]),
                bytemuck::cast::<u32, i32>(self.storage[3]),
            ],
            ScalarType::Uint => [
                self.storage[0] as i32,
                self.storage[1] as i32,
                self.storage[2] as i32,
                self.storage[3] as i32,
            ],
        }
    }

    fn cast_u32(&self) -> [u32; 4] {
        match self.elem_type() {
            ScalarType::Bool => {
                let ft = [0, 1];
                [
                    ft[(self.storage[0] != 0) as usize],
                    ft[(self.storage[1] != 0) as usize],
                    ft[(self.storage[2] != 0) as usize],
                    ft[(self.storage[3] != 0) as usize],
                ]
            }
            ScalarType::Float => [
                bytemuck::cast::<u32, f32>(self.storage[0]) as u32,
                bytemuck::cast::<u32, f32>(self.storage[1]) as u32,
                bytemuck::cast::<u32, f32>(self.storage[2]) as u32,
                bytemuck::cast::<u32, f32>(self.storage[3]) as u32,
            ],
            ScalarType::Int => [
                bytemuck::cast::<u32, i32>(self.storage[0]) as u32,
                bytemuck::cast::<u32, i32>(self.storage[1]) as u32,
                bytemuck::cast::<u32, i32>(self.storage[2]) as u32,
                bytemuck::cast::<u32, i32>(self.storage[3]) as u32,
            ],
            ScalarType::Uint => self.storage,
        }
    }

    // fn binary_op(&self, other: &Self, op: BinaryOperator) -> Self {
    //     let count = self.vector_type.count();
    //     let mut v = *self;
    //     // component-wise op
    //     for i in 0..count {
    //         v.value_mut(i).binary_op(&other.value(i), op);
    //     }
    //     v
    // }
}

impl PartialEq for VectorValue {
    fn eq(&self, other: &Self) -> bool {
        if self.vector_type() != other.vector_type() {
            return false;
        }
        let count = self.vector_type().count();
        match self.elem_type() {
            ScalarType::Float => {
                let s = self.cast_f32();
                let o = other.cast_f32();
                s[..count] == o[..count]
            }
            _ => self.storage[..count] == other.storage[..count],
        }
    }
}

impl Eq for VectorValue {}

impl std::hash::Hash for VectorValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.vector_type.hash(state);
        // Only compare the subset of storage actually in use
        let count = self.vector_type.count();
        let elem_type = self.elem_type();
        if elem_type == ScalarType::Float {
            for i in 0..count {
                FloatOrd(self.get::<f32>(i)).hash(state);
            }
        } else {
            self.storage[..count].hash(state);
        }
    }
}

impl ToWgslString for VectorValue {
    fn to_wgsl_string(&self) -> String {
        let mut vals = format!(
            "{}({},{}",
            self.vector_type().to_wgsl_string(),
            self.value_n::<0>().to_wgsl_string(),
            self.value_n::<1>().to_wgsl_string()
        );
        let count = self.vector_type.count();
        if count > 2 {
            vals.push(',');
            vals.push_str(&self.value_n::<2>().to_wgsl_string());
            if count > 3 {
                vals.push(',');
                vals.push_str(&self.value_n::<3>().to_wgsl_string());
            }
        }
        vals.push(')');
        vals
    }
}

impl From<BVec2> for VectorValue {
    fn from(value: BVec2) -> Self {
        Self {
            vector_type: VectorType::new(ScalarType::Bool, 2),
            storage: [
                if value.x {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0
                },
                if value.y {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0
                },
                0,
                0,
            ],
        }
    }
}

impl From<BVec3> for VectorValue {
    fn from(value: BVec3) -> Self {
        Self {
            vector_type: VectorType::new(ScalarType::Bool, 3),
            storage: [
                if value.x {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0
                },
                if value.y {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0
                },
                if value.z {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0
                },
                0,
            ],
        }
    }
}

impl From<BVec4> for VectorValue {
    fn from(value: BVec4) -> Self {
        Self {
            vector_type: VectorType::new(ScalarType::Bool, 4),
            storage: [
                if value.x {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0
                },
                if value.y {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0
                },
                if value.z {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0
                },
                if value.w {
                    ScalarValue::BOOL_TRUE_STORAGE
                } else {
                    0
                },
            ],
        }
    }
}

impl From<Vec2> for VectorValue {
    fn from(value: Vec2) -> Self {
        let mut s = Self {
            vector_type: VectorType::VEC2F,
            storage: [0u32; 4],
        };
        let v = bytemuck::cast_slice_mut::<u32, f32>(&mut s.storage);
        value.write_to_slice(v);
        s
    }
}

impl From<Vec3> for VectorValue {
    fn from(value: Vec3) -> Self {
        let mut s = Self {
            vector_type: VectorType::VEC3F,
            storage: [0u32; 4],
        };
        let v = bytemuck::cast_slice_mut::<u32, f32>(&mut s.storage);
        value.write_to_slice(v);
        s
    }
}

impl From<Vec3A> for VectorValue {
    fn from(value: Vec3A) -> Self {
        let mut s = Self {
            vector_type: VectorType::VEC3F,
            storage: [0u32; 4],
        };
        let v = bytemuck::cast_slice_mut::<u32, f32>(&mut s.storage);
        value.write_to_slice(v);
        s
    }
}

impl From<Vec4> for VectorValue {
    fn from(value: Vec4) -> Self {
        let mut s = Self {
            vector_type: VectorType::VEC4F,
            storage: [0u32; 4],
        };
        let v = bytemuck::cast_slice_mut::<u32, f32>(&mut s.storage);
        value.write_to_slice(v);
        s
    }
}

impl From<IVec2> for VectorValue {
    fn from(value: IVec2) -> Self {
        let mut s = Self {
            vector_type: VectorType::VEC2I,
            storage: [0u32; 4],
        };
        let v = bytemuck::cast_slice_mut::<u32, i32>(&mut s.storage);
        value.write_to_slice(v);
        s
    }
}

impl From<IVec3> for VectorValue {
    fn from(value: IVec3) -> Self {
        let mut s = Self {
            vector_type: VectorType::VEC3I,
            storage: [0u32; 4],
        };
        let v = bytemuck::cast_slice_mut::<u32, i32>(&mut s.storage);
        value.write_to_slice(v);
        s
    }
}

impl From<IVec4> for VectorValue {
    fn from(value: IVec4) -> Self {
        let mut s = Self {
            vector_type: VectorType::VEC4I,
            storage: [0u32; 4],
        };
        let v = bytemuck::cast_slice_mut::<u32, i32>(&mut s.storage);
        value.write_to_slice(v);
        s
    }
}

/// Floating-point matrix value.
#[derive(Debug, Clone, Copy, Reflect, Serialize, Deserialize)]
pub struct MatrixValue {
    /// Matrix type.
    matrix_type: MatrixType,
    /// Raw storage for up to 16 values. Actual usage depends on matrix size.
    storage: [f32; 16],
}

impl MatrixValue {
    /// Scalar type of the elements of the matrix.
    ///
    /// This always returns [`ScalarType::Float`]. This method is provided for
    /// consistency.
    pub const fn elem_type(&self) -> ScalarType {
        ScalarType::Float
    }

    /// Matrix type.
    pub fn matrix_type(&self) -> MatrixType {
        self.matrix_type
    }

    /// Get the scalar value of the matrix element in the R-th row and C-th
    /// column.
    pub fn value_n<const R: usize, const C: usize>(&self) -> ScalarValue {
        ScalarValue::Float(self.get_n::<R, C>())
    }

    /// Get the scalar value of a matrix element.
    pub fn value(&self, row: usize, col: usize) -> ScalarValue {
        ScalarValue::Float(self.get(row, col))
    }

    /// Get the scalar value of an element of the matrix.
    pub fn value_mut(&mut self, row: usize, col: usize) -> &mut f32 {
        self.get_mut(row, col)
    }

    /// Get the floating-point value of the matrix element in the R-th row and
    /// C-th column.
    pub fn get_n<const R: usize, const C: usize>(&self) -> f32 {
        if R < self.matrix_type.rows() && C < self.matrix_type.cols() {
            self.storage[self.matrix_type.rows() * C + R]
        } else {
            0f32
        }
    }

    /// Get the floating-point value of a matrix element.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        if row < self.matrix_type.rows() && col < self.matrix_type.cols() {
            self.storage[self.matrix_type.rows() * col + row]
        } else {
            0f32
        }
    }

    /// Get the value of an element of the matrix.
    fn get_mut(&mut self, row: usize, col: usize) -> &mut f32 {
        assert!(row < self.matrix_type.rows());
        assert!(col < self.matrix_type.cols());
        &mut self.storage[self.matrix_type.rows() * col + row]
    }

    /// Get the value as a binary blob ready for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        let count = self.matrix_type.rows() * self.matrix_type.cols();
        bytemuck::cast_slice::<f32, u8>(&self.storage[..count])
    }

    // fn binary_op(&self, other: &Self, op: BinaryOperator) -> Self {
    //     let mut m = *self;
    //     // component-wise op
    //     for j in 0..self.matrix_type.cols() {
    //         for i in 0..self.matrix_type.rows() {
    //             let dst = m.value_mut(i, j);
    //             let a = *dst;
    //             let b = other.get(i, j);
    //             *dst = op.apply_f32(a, b);
    //         }
    //     }
    //     m
    // }
}

impl PartialEq for MatrixValue {
    fn eq(&self, other: &Self) -> bool {
        if self.matrix_type != other.matrix_type {
            return false;
        }
        let count = self.matrix_type.rows() * self.matrix_type.cols();
        self.storage[..count] == other.storage[..count]
    }
}

impl std::hash::Hash for MatrixValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.matrix_type.hash(state);
        let count = self.matrix_type.rows() * self.matrix_type.cols();
        for i in 0..count {
            FloatOrd(self.storage[i]).hash(state);
        }
    }
}

impl ToWgslString for MatrixValue {
    fn to_wgsl_string(&self) -> String {
        let mut vals = format!("{}(", self.matrix_type().to_wgsl_string(),);
        for j in 0..self.matrix_type.cols() {
            for i in 0..self.matrix_type.rows() {
                vals.push_str(&self.value(i, j).to_wgsl_string());
                vals.push(',');
            }
        }
        vals.pop(); // Remove the last comma
        vals.push(')');
        vals
    }
}

impl From<Mat2> for MatrixValue {
    fn from(value: Mat2) -> Self {
        let mut s = Self {
            matrix_type: MatrixType::MAT2X2F,
            storage: [0.; 16],
        };
        value.write_cols_to_slice(&mut s.storage);
        s
    }
}

impl From<Mat3> for MatrixValue {
    fn from(value: Mat3) -> Self {
        let mut s = Self {
            matrix_type: MatrixType::MAT3X3F,
            storage: [0.; 16],
        };
        value.write_cols_to_slice(&mut s.storage);
        s
    }
}

impl From<Mat4> for MatrixValue {
    fn from(value: Mat4) -> Self {
        let mut s = Self {
            matrix_type: MatrixType::MAT4X4F,
            storage: [0.; 16],
        };
        value.write_cols_to_slice(&mut s.storage);
        s
    }
}

/// Variant storage for a simple value.
///
/// The variant can store a scalar, vector, or matrix value.
#[derive(Debug, Clone, Copy, PartialEq, Hash, Reflect, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Value {
    /// Scalar value.
    Scalar(ScalarValue),
    /// Vector value with 2 to 4 components.
    ///
    /// Note that 1-component vectors are invalid; [`Value::Scalar`] must be
    /// used instead. Similarly, vectors with more than 4 components are
    /// invalid.
    Vector(VectorValue),
    /// Floating-point matrix value of size 2x2 to 4x4.
    ///
    /// Note that single-row or single-column matrices are invalid;
    /// [`Value::Vector`] must be used instead, or [`Value::Scalar`] for a
    /// 1x1 "matrix". Similarly, matrices with more than 4 rows or columns are
    /// invalid.
    Matrix(MatrixValue),
}

impl Value {
    /// Get the value as a binary blob ready for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Value::Scalar(s) => s.as_bytes(),
            Value::Vector(v) => v.as_bytes(),
            Value::Matrix(m) => m.as_bytes(),
        }
    }

    /// Get the type of the value.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let value = graph::Value::Scalar(3_f32.into());
    /// assert_eq!(ValueType::Scalar(ScalarType::Float), value.value_type());
    /// ```
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::Scalar(s) => ValueType::Scalar(s.scalar_type()),
            Value::Vector(v) => ValueType::Vector(v.vector_type()),
            Value::Matrix(m) => ValueType::Matrix(m.matrix_type()),
        }
    }

    /// Cast this value to a [`ScalarValue`].
    ///
    /// # Panic
    ///
    /// Panics if this value is not a [`ScalarValue`].
    pub fn as_scalar(&self) -> &ScalarValue {
        match self {
            Value::Scalar(s) => s,
            _ => panic!("Cannot cast from {:?} to ScalarType.", self.value_type()),
        }
    }

    /// Cast this value to a [`VectorValue`].
    ///
    /// # Panic
    ///
    /// Panics if this value is not a [`VectorValue`].
    pub fn as_vector(&self) -> &VectorValue {
        match self {
            Value::Vector(v) => v,
            _ => panic!("Cannot cast from {:?} to VectorType.", self.value_type()),
        }
    }

    /// Cast this value to a [`MatrixValue`].
    ///
    /// # Panic
    ///
    /// Panics if this value is not a [`MatrixValue`].
    pub fn as_matrix(&self) -> &MatrixValue {
        match self {
            Value::Matrix(m) => m,
            _ => panic!("Cannot cast from {:?} to MatrixType.", self.value_type()),
        }
    }

    // /// Apply a binary arithmetic operator between self and another operand.
    // pub fn binary_op(&self, other: &Value, op: BinaryOperator) -> Value {
    //     match self {
    //         Value::Scalar(s) => Value::Scalar(s.binary_op(other.as_scalar(),
    // op)),         Value::Vector(v) =>
    // Value::Vector(v.binary_op(other.as_vector(), op)),
    //         Value::Matrix(m) => Value::Matrix(m.binary_op(other.as_matrix(),
    // op)),     }
    // }
}

impl ToWgslString for Value {
    fn to_wgsl_string(&self) -> String {
        match self {
            Value::Scalar(s) => s.to_wgsl_string(),
            Value::Vector(v) => v.to_wgsl_string(),
            Value::Matrix(m) => m.to_wgsl_string(),
        }
    }
}

impl From<ScalarValue> for Value {
    fn from(value: ScalarValue) -> Self {
        Self::Scalar(value)
    }
}

impl From<VectorValue> for Value {
    fn from(value: VectorValue) -> Self {
        Self::Vector(value)
    }
}

impl From<MatrixValue> for Value {
    fn from(value: MatrixValue) -> Self {
        Self::Matrix(value)
    }
}

macro_rules! impl_scalar_value {
    ($t: ty, $sv: ident) => {
        impl From<$t> for Value {
            fn from(value: $t) -> Self {
                Self::Scalar(value.into())
            }
        }

        impl TryInto<$t> for ScalarValue {
            type Error = ExprError;

            fn try_into(self) -> Result<$t, Self::Error> {
                match self {
                    ScalarValue::$sv(b) => Ok(b),
                    _ => Err(ExprError::TypeError(format!(
                        "Expected {:?} type, found {:?} instead.",
                        ScalarType::$sv,
                        self.scalar_type()
                    ))),
                }
            }
        }

        impl TryInto<$t> for Value {
            type Error = ExprError;

            fn try_into(self) -> Result<$t, Self::Error> {
                match self {
                    Value::Scalar(s) => s.try_into(),
                    _ => Err(ExprError::TypeError(format!(
                        "Expected ValueType::Scalar type, found {:?} instead.",
                        self.value_type()
                    ))),
                }
            }
        }
    };
}

impl_scalar_value!(bool, Bool);
impl_scalar_value!(f32, Float);
impl_scalar_value!(i32, Int);
impl_scalar_value!(u32, Uint);

macro_rules! impl_vec_value {
    ($t: ty, $vt: ident, $cast: tt) => {
        impl From<$t> for Value {
            fn from(value: $t) -> Self {
                Self::Vector(value.into())
            }
        }

        impl TryInto<$t> for VectorValue {
            type Error = ExprError;

            fn try_into(self) -> Result<$t, Self::Error> {
                if self.vector_type() == VectorType::$vt {
                    Ok(self.$cast())
                } else {
                    Err(ExprError::TypeError(format!(
                        "Expected {:?} type, found {:?} instead.",
                        VectorType::$vt,
                        self.vector_type()
                    )))
                }
            }
        }

        impl TryInto<$t> for Value {
            type Error = ExprError;

            fn try_into(self) -> Result<$t, Self::Error> {
                match self {
                    Value::Vector(v) => v.try_into(),
                    _ => Err(ExprError::TypeError(format!(
                        "Expected ValueType::Scalar type, found {:?} instead.",
                        self.value_type()
                    ))),
                }
            }
        }
    };
}

impl_vec_value!(BVec2, VEC2B, as_bvec2);
impl_vec_value!(BVec3, VEC3B, as_bvec3);
impl_vec_value!(BVec4, VEC4B, as_bvec4);
impl_vec_value!(Vec2, VEC2F, as_vec2);
impl_vec_value!(Vec3, VEC3F, as_vec3);
impl_vec_value!(Vec3A, VEC3F, as_vec3a);
impl_vec_value!(Vec4, VEC4F, as_vec4);
impl_vec_value!(IVec2, VEC2I, as_ivec2);
impl_vec_value!(IVec3, VEC3I, as_ivec3);
impl_vec_value!(IVec4, VEC4I, as_ivec4);

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };

    #[test]
    fn as_bytes() {
        // let v = Value::Scalar(true.into());
        // let b = v.as_bytes();
        // assert_eq!(b, &[0xFFu8, 0xFFu8, 0xFFu8, 0xFFu8]);

        let v = Value::Scalar(3f32.into());
        let b = v.as_bytes();
        assert_eq!(b, &[0u8, 0u8, 0x40u8, 0x40u8]); // 0x40400000

        let v = Value::Scalar(0x12FF89ACu32.into());
        let b = v.as_bytes();
        assert_eq!(b, &[0xACu8, 0x89u8, 0xFFu8, 0x12u8]);

        let v = Value::Scalar(0x12FF89ACi32.into());
        let b = v.as_bytes();
        assert_eq!(b, &[0xACu8, 0x89u8, 0xFFu8, 0x12u8]);

        let v = Value::Vector(Vec2::new(-2., 3.).into());
        let b = v.as_bytes();
        assert_eq!(b, &[0u8, 0u8, 0u8, 0xC0u8, 0u8, 0u8, 0x40u8, 0x40u8]); // 0xc0000000 0x40400000

        let v = Value::Vector(Vec3::new(-2., 3., 4.).into());
        let b = v.as_bytes();
        assert_eq!(
            b,
            &[0u8, 0u8, 0u8, 0xC0u8, 0u8, 0u8, 0x40u8, 0x40u8, 0u8, 0u8, 0x80u8, 0x40u8]
        ); // 0xc0000000 0x40400000 0x40800000

        let v = Value::Vector(Vec4::new(-2., 3., 4., -5.).into());
        let b = v.as_bytes();
        assert_eq!(
            b,
            &[
                0u8, 0u8, 0u8, 0xC0u8, 0u8, 0u8, 0x40u8, 0x40u8, 0u8, 0u8, 0x80u8, 0x40u8, 0u8,
                0u8, 0xa0u8, 0xc0u8
            ]
        ); // 0xc0000000 0x40400000 0x40800000 0xc0a00000
    }

    #[test]
    fn value_type() {
        assert_eq!(
            Value::Scalar(true.into()).value_type(),
            ValueType::Scalar(ScalarType::Bool)
        );
        assert_eq!(
            Value::Scalar(0f32.into()).value_type(),
            ValueType::Scalar(ScalarType::Float)
        );
        assert_eq!(
            Value::Scalar(0i32.into()).value_type(),
            ValueType::Scalar(ScalarType::Int)
        );
        assert_eq!(
            Value::Scalar(0u32.into()).value_type(),
            ValueType::Scalar(ScalarType::Uint)
        );
        assert_eq!(
            Value::Vector(Vec2::ZERO.into()).value_type(),
            ValueType::Vector(VectorType::VEC2F)
        );
        assert_eq!(
            Value::Vector(Vec3::ZERO.into()).value_type(),
            ValueType::Vector(VectorType::VEC3F)
        );
        assert_eq!(
            Value::Vector(Vec4::ZERO.into()).value_type(),
            ValueType::Vector(VectorType::VEC4F)
        );
        assert_eq!(
            Value::Vector(BVec2::TRUE.into()).value_type(),
            ValueType::Vector(VectorType::VEC2B)
        );
        assert_eq!(
            Value::Vector(BVec3::TRUE.into()).value_type(),
            ValueType::Vector(VectorType::VEC3B)
        );
        assert_eq!(
            Value::Vector(BVec4::TRUE.into()).value_type(),
            ValueType::Vector(VectorType::VEC4B)
        );
        assert_eq!(
            Value::Matrix(Mat2::IDENTITY.into()).value_type(),
            ValueType::Matrix(MatrixType::MAT2X2F)
        );
        assert_eq!(
            Value::Matrix(Mat3::IDENTITY.into()).value_type(),
            ValueType::Matrix(MatrixType::MAT3X3F)
        );
        assert_eq!(
            Value::Matrix(Mat4::IDENTITY.into()).value_type(),
            ValueType::Matrix(MatrixType::MAT4X4F)
        );
    }

    #[test]
    fn to_wgsl_string() {
        for b in [true, false] {
            assert_eq!(Value::Scalar(b.into()).to_wgsl_string(), b.to_wgsl_string());
        }
        for f in [0_f32, -1., 1., 1e-5] {
            assert_eq!(Value::Scalar(f.into()).to_wgsl_string(), f.to_wgsl_string());
        }
        for u in [0_u32, 1, 42, 999999] {
            assert_eq!(Value::Scalar(u.into()).to_wgsl_string(), u.to_wgsl_string());
        }
        for i in [0_i32, -1, 1, -42, 42, -100000, 100000] {
            assert_eq!(Value::Scalar(i.into()).to_wgsl_string(), i.to_wgsl_string());
        }
        for v in [
            Vec2::ZERO,
            Vec2::ONE,
            Vec2::NEG_ONE,
            Vec2::X,
            Vec2::Y,
            Vec2::NEG_X,
            Vec2::NEG_Y,
            Vec2::new(-42.578, 663.449),
        ] {
            assert_eq!(Value::Vector(v.into()).to_wgsl_string(), v.to_wgsl_string());
        }
        for v in [
            Vec3::ZERO,
            Vec3::ONE,
            Vec3::NEG_ONE,
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::NEG_X,
            Vec3::NEG_Y,
            Vec3::NEG_Z,
            Vec3::new(-42.578, 663.449, -42558.35),
        ] {
            assert_eq!(Value::Vector(v.into()).to_wgsl_string(), v.to_wgsl_string());
        }
        for v in [
            Vec4::ZERO,
            Vec4::ONE,
            Vec4::NEG_ONE,
            Vec4::X,
            Vec4::Y,
            Vec4::Z,
            Vec4::NEG_X,
            Vec4::NEG_Y,
            Vec4::NEG_Z,
            Vec4::new(-42.578, 663.449, -42558.35, -4.2),
        ] {
            assert_eq!(Value::Vector(v.into()).to_wgsl_string(), v.to_wgsl_string());
        }

        for (m, expected) in [
            (Mat3::IDENTITY, "mat3x3<f32>(1.,0.,0.,0.,1.,0.,0.,0.,1.)"),
            (Mat3::ZERO, "mat3x3<f32>(0.,0.,0.,0.,0.,0.,0.,0.,0.)"),
            (
                Mat3::from_cols(
                    Vec3::new(1., 2., 3.),
                    Vec3::new(4., 5., 6.),
                    Vec3::new(7., 8., 9.),
                ),
                "mat3x3<f32>(1.,2.,3.,4.,5.,6.,7.,8.,9.)",
            ),
        ] {
            assert_eq!(
                Value::Matrix(m.into()).to_wgsl_string(),
                expected.to_string()
            );
        }
    }

    #[test]
    fn vector_value() {
        let v = Vec2::new(-3.2, 5.);
        let vv = VectorValue::new_vec2(v);
        assert_eq!(v.x.to_bits(), vv.storage[0]);
        assert_eq!(v.y.to_bits(), vv.storage[1]);
        assert_eq!(0u32, vv.storage[2]);
        assert_eq!(0u32, vv.storage[3]);

        let v = Vec3::new(-3.2, 5., 64.5);
        let vv = VectorValue::new_vec3(v);
        assert_eq!(v.x.to_bits(), vv.storage[0]);
        assert_eq!(v.y.to_bits(), vv.storage[1]);
        assert_eq!(v.z.to_bits(), vv.storage[2]);
        assert_eq!(0u32, vv.storage[3]);

        let v = Vec4::new(-3.2, 5., 64.5, -42.);
        let vv = VectorValue::new_vec4(v);
        assert_eq!(v.x.to_bits(), vv.storage[0]);
        assert_eq!(v.y.to_bits(), vv.storage[1]);
        assert_eq!(v.z.to_bits(), vv.storage[2]);
        assert_eq!(v.w.to_bits(), vv.storage[3]);

        let v = BVec2::new(false, true);
        let vv = VectorValue::new_bvec2(v);
        assert_eq!(0u32, vv.storage[0]);
        assert_eq!(ScalarValue::BOOL_TRUE_STORAGE, vv.storage[1]);
        assert_eq!(0u32, vv.storage[2]);
        assert_eq!(0u32, vv.storage[3]);

        let v = BVec3::new(false, true, false);
        let vv = VectorValue::new_bvec3(v);
        assert_eq!(0u32, vv.storage[0]);
        assert_eq!(ScalarValue::BOOL_TRUE_STORAGE, vv.storage[1]);
        assert_eq!(0u32, vv.storage[2]);
        assert_eq!(0u32, vv.storage[3]);

        let v = BVec4::new(false, true, false, true);
        let vv = VectorValue::new_bvec4(v);
        assert_eq!(0u32, vv.storage[0]);
        assert_eq!(ScalarValue::BOOL_TRUE_STORAGE, vv.storage[1]);
        assert_eq!(0u32, vv.storage[2]);
        assert_eq!(ScalarValue::BOOL_TRUE_STORAGE, vv.storage[3]);
    }

    #[test]
    fn from() {
        assert_eq!(Value::Scalar(ScalarValue::Bool(true)), true.into());
        assert_eq!(Value::Scalar(ScalarValue::Float(0.)), 0_f32.into());
        assert_eq!(Value::Scalar(ScalarValue::Int(-42)), (-42_i32).into());
        assert_eq!(Value::Scalar(ScalarValue::Uint(0)), 0_u32.into());

        assert_eq!(
            Value::Vector(VectorValue::new_vec2(Vec2::Y)),
            Vec2::Y.into()
        );
        assert_eq!(
            Value::Vector(VectorValue::new_vec3(Vec3::Z)),
            Vec3::Z.into()
        );
        assert_eq!(
            Value::Vector(VectorValue::new_vec4(Vec4::W)),
            Vec4::W.into()
        );

        let v = BVec2::new(false, true);
        let vv: VectorValue = v.into();
        assert_eq!(vv, VectorValue::new_bvec2(v));
        assert_eq!(0, vv.storage[0]);
        assert_eq!(ScalarValue::BOOL_TRUE_STORAGE, vv.storage[1]);
        assert_eq!(0, vv.storage[2]);
        assert_eq!(0, vv.storage[3]);

        let v = BVec3::new(false, true, false);
        let vv: VectorValue = v.into();
        assert_eq!(vv, VectorValue::new_bvec3(v));
        assert_eq!(0, vv.storage[0]);
        assert_eq!(ScalarValue::BOOL_TRUE_STORAGE, vv.storage[1]);
        assert_eq!(0, vv.storage[2]);
        assert_eq!(0, vv.storage[3]);

        let v = BVec4::new(false, true, false, true);
        let vv: VectorValue = v.into();
        assert_eq!(vv, VectorValue::new_bvec4(v));
        assert_eq!(0, vv.storage[0]);
        assert_eq!(ScalarValue::BOOL_TRUE_STORAGE, vv.storage[1]);
        assert_eq!(0, vv.storage[2]);
        assert_eq!(ScalarValue::BOOL_TRUE_STORAGE, vv.storage[3]);
    }

    fn calc_hash<H: Hash>(value: &H) -> u64 {
        let mut hasher = DefaultHasher::default();
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn calc_f32_vector_hash(vector_type: VectorType, values: &[f32]) -> u64 {
        let mut hasher = DefaultHasher::default();
        vector_type.hash(&mut hasher);
        for f in values {
            FloatOrd(*f).hash(&mut hasher);
        }
        hasher.finish()
    }

    fn calc_i32_vector_hash(vector_type: VectorType, values: &[i32]) -> u64 {
        let mut hasher = DefaultHasher::default();
        vector_type.hash(&mut hasher);
        let v = bytemuck::cast_slice::<i32, u32>(values);
        let c = vector_type.count();
        v[..c].hash(&mut hasher);
        hasher.finish()
    }

    fn calc_u32_vector_hash(vector_type: VectorType, values: &[u32]) -> u64 {
        let mut hasher = DefaultHasher::default();
        vector_type.hash(&mut hasher);
        let c = vector_type.count();
        values[..c].hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn hash() {
        // Different types must not be equal
        let zeros = [
            Into::<ScalarValue>::into(false),
            Into::<ScalarValue>::into(0_f32),
            Into::<ScalarValue>::into(0_u32),
            Into::<ScalarValue>::into(0_i32),
        ];
        let ones = [
            Into::<ScalarValue>::into(true),
            Into::<ScalarValue>::into(1_f32),
            Into::<ScalarValue>::into(1_u32),
            Into::<ScalarValue>::into(1_i32),
        ];
        for arr in [zeros, ones] {
            for i in 0..=3 {
                for j in 0..=3 {
                    if i == j {
                        // Equal to self
                        assert_eq!(arr[i], arr[j]);
                        assert_eq!(calc_hash(&arr[i]), calc_hash(&arr[j]));
                    } else {
                        // Different types must be different and hash to different values
                        assert_ne!(arr[i], arr[j]);
                        assert_ne!(calc_hash(&arr[i]), calc_hash(&arr[j]));
                    }
                    // With casting however, values can be equal
                    assert!(arr[i].cast_eq(&arr[j]));
                }
            }
        }

        // Different types must not be equal
        let vecs = [
            VectorValue::new_vec2(Vec2::new(1., 0.)),
            VectorValue::new_ivec2(IVec2::new(1, 0)),
            VectorValue::new_uvec2(1, 0),
            VectorValue::new_bvec2(BVec2::new(true, false)),
        ];
        for i in 0..=3 {
            for j in 0..=3 {
                if i == j {
                    // Equal to self
                    assert_eq!(vecs[i], vecs[j]);
                    assert_eq!(calc_hash(&vecs[i]), calc_hash(&vecs[j]));
                } else {
                    // Different types must be different and hash to different values
                    assert_ne!(vecs[i], vecs[j]);
                    assert_ne!(calc_hash(&vecs[i]), calc_hash(&vecs[j]));
                }
                // With casting however, values can be equal
                assert!(vecs[i].cast_eq(&vecs[j]));
            }
        }

        // Vectors with different sizes are always inequal
        assert_ne!(
            VectorValue::new_vec2(Vec2::ZERO),
            VectorValue::new_vec3(Vec3::ZERO)
        );
        assert_ne!(
            VectorValue::new_vec2(Vec2::ZERO),
            VectorValue::new_vec4(Vec4::ZERO)
        );
        assert_ne!(
            VectorValue::new_vec3(Vec3::ZERO),
            VectorValue::new_vec4(Vec4::ZERO)
        );

        assert_eq!(
            calc_hash(&Into::<VectorValue>::into(Vec2::new(3.5, -42.))),
            calc_f32_vector_hash(VectorType::VEC2F, &[3.5, -42.])
        );
        assert_eq!(
            calc_hash(&Into::<VectorValue>::into(Vec3::new(3.5, -42., 999.99))),
            calc_f32_vector_hash(VectorType::VEC3F, &[3.5, -42., 999.99])
        );
        assert_eq!(
            calc_hash(&Into::<VectorValue>::into(Vec4::new(
                3.5, -42., 999.99, -0.01,
            ))),
            calc_f32_vector_hash(VectorType::VEC4F, &[3.5, -42., 999.99, -0.01])
        );

        assert_eq!(
            calc_hash(&Into::<VectorValue>::into(IVec2::new(3, -42))),
            calc_i32_vector_hash(VectorType::VEC2I, &[3, -42])
        );
        assert_eq!(
            calc_hash(&Into::<VectorValue>::into(IVec3::new(3, -42, 999))),
            calc_i32_vector_hash(VectorType::VEC3I, &[3, -42, 999])
        );
        assert_eq!(
            calc_hash(&Into::<VectorValue>::into(IVec4::new(3, -42, 999, -1))),
            calc_i32_vector_hash(VectorType::VEC4I, &[3, -42, 999, -1])
        );

        assert_eq!(
            calc_hash(&VectorValue::new_uvec2(3, 42)),
            calc_u32_vector_hash(VectorType::VEC2U, &[3, 42])
        );
        assert_eq!(
            calc_hash(&VectorValue::new_uvec3(3, 42, 999)),
            calc_u32_vector_hash(VectorType::VEC3U, &[3, 42, 999])
        );
        assert_eq!(
            calc_hash(&VectorValue::new_uvec4(3, 42, 999, 1)),
            calc_u32_vector_hash(VectorType::VEC4U, &[3, 42, 999, 1])
        );
    }

    #[test]
    fn try_into() {
        let b: Value = true.into();
        let ret: Result<bool, _> = b.try_into();
        assert_eq!(ret, Ok(true));
        assert!(matches!(
            TryInto::<f32>::try_into(b),
            Err(ExprError::TypeError(_))
        ));
        assert!(matches!(
            TryInto::<i32>::try_into(b),
            Err(ExprError::TypeError(_))
        ));
        assert!(matches!(
            TryInto::<Vec3>::try_into(b),
            Err(ExprError::TypeError(_))
        ));

        let f: Value = 3.4_f32.into();
        let ret: Result<f32, _> = f.try_into();
        assert_eq!(ret, Ok(3.4_f32));
        assert!(matches!(
            TryInto::<bool>::try_into(f),
            Err(ExprError::TypeError(_))
        ));
        assert!(matches!(
            TryInto::<u32>::try_into(f),
            Err(ExprError::TypeError(_))
        ));
        assert!(matches!(
            TryInto::<Vec2>::try_into(f),
            Err(ExprError::TypeError(_))
        ));

        let u: Value = 42_u32.into();
        let ret: Result<u32, _> = u.try_into();
        assert_eq!(ret, Ok(42_u32));
        assert!(matches!(
            TryInto::<bool>::try_into(u),
            Err(ExprError::TypeError(_))
        ));
        assert!(matches!(
            TryInto::<f32>::try_into(u),
            Err(ExprError::TypeError(_))
        ));
        assert!(matches!(
            TryInto::<Vec4>::try_into(u),
            Err(ExprError::TypeError(_))
        ));

        let i: Value = 42_i32.into();
        let ret: Result<i32, _> = i.try_into();
        assert_eq!(ret, Ok(42_i32));
        assert!(matches!(
            TryInto::<bool>::try_into(i),
            Err(ExprError::TypeError(_))
        ));
        assert!(matches!(
            TryInto::<f32>::try_into(i),
            Err(ExprError::TypeError(_))
        ));
        assert!(matches!(
            TryInto::<Vec4>::try_into(i),
            Err(ExprError::TypeError(_))
        ));
    }

    #[test]
    fn splat() {
        let b = true;
        for c in 2..=4 {
            let x = VectorValue::splat(&b.into(), c);
            assert_eq!(x.elem_type(), ScalarType::Bool);
            assert_eq!(x.vector_type().count(), c as usize);
            if c == 2 {
                assert_eq!(TryInto::<BVec2>::try_into(x), Ok(BVec2::splat(b)));
            } else if c == 3 {
                assert_eq!(TryInto::<BVec3>::try_into(x), Ok(BVec3::splat(b)));
            } else {
                assert_eq!(TryInto::<BVec4>::try_into(x), Ok(BVec4::splat(b)));
            }
        }

        let f = 3.4_f32;
        for c in 2..=4 {
            let x = VectorValue::splat(&f.into(), c);
            assert_eq!(x.elem_type(), ScalarType::Float);
            assert_eq!(x.vector_type().count(), c as usize);
            if c == 2 {
                assert_eq!(TryInto::<Vec2>::try_into(x), Ok(Vec2::splat(f)));
            } else if c == 3 {
                assert_eq!(TryInto::<Vec3>::try_into(x), Ok(Vec3::splat(f)));
            } else {
                assert_eq!(TryInto::<Vec4>::try_into(x), Ok(Vec4::splat(f)));
            }
        }

        let i = -46458_i32;
        for c in 2..=4 {
            let x = VectorValue::splat(&i.into(), c);
            assert_eq!(x.elem_type(), ScalarType::Int);
            assert_eq!(x.vector_type().count(), c as usize);
            if c == 2 {
                assert_eq!(TryInto::<IVec2>::try_into(x), Ok(IVec2::splat(i)));
            } else if c == 3 {
                assert_eq!(TryInto::<IVec3>::try_into(x), Ok(IVec3::splat(i)));
            } else {
                assert_eq!(TryInto::<IVec4>::try_into(x), Ok(IVec4::splat(i)));
            }
        }
    }
}
