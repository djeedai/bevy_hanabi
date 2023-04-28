//! Effect graph and language definition.
//!
//! This module contains the elements used to build a effect graph, a fully
//! customizable description of a visual effect.
//!
//! Currently effect graphs are not yet available; only some preview elements
//! exist. So this module is a bit empty and of little interest.

use std::fmt::Debug;

use bevy::{
    math::{BVec2, BVec3, BVec4, Vec2, Vec3, Vec4},
    reflect::{FromReflect, Reflect},
    utils::FloatOrd,
};
use serde::{Deserialize, Serialize};

use crate::{MatrixType, ScalarType, ToWgslString, ValueType, VectorType};

mod expr;

pub use expr::{AddExpr, BoxedExpr, Expr, Literal};

/// Variant storage for a scalar value.
#[derive(Debug, Clone, Copy, Reflect, FromReflect, Serialize, Deserialize)]
#[non_exhaustive]
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
    /// Convert this value to a `bool` value.
    ///
    /// Any non-zero value converts to `true`, while a zero value converts to `false`.
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

    /// Get the scalar type of this value.
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            ScalarValue::Bool(_) => ScalarType::Bool,
            ScalarValue::Float(_) => ScalarType::Float,
            ScalarValue::Int(_) => ScalarType::Int,
            ScalarValue::Uint(_) => ScalarType::Uint,
        }
    }
}

impl PartialEq for ScalarValue {
    fn eq(&self, other: &Self) -> bool {
        match *self {
            ScalarValue::Bool(b) => b == other.as_bool(),
            ScalarValue::Float(f) => FloatOrd(f) == FloatOrd(other.as_f32()),
            ScalarValue::Int(i) => i == other.as_i32(),
            ScalarValue::Uint(u) => u == other.as_u32(),
        }
    }
}

impl std::hash::Hash for ScalarValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match *self {
            ScalarValue::Bool(b) => b.hash(state),
            ScalarValue::Float(f) => FloatOrd(f).hash(state),
            ScalarValue::Int(i) => i.hash(state),
            ScalarValue::Uint(u) => u.hash(state),
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
}

/// Variant storage for a vector value.
#[derive(Debug, Clone, Copy, Reflect, FromReflect, Serialize, Deserialize)]
pub struct VectorValue {
    vector_type: VectorType,
    storage: [u32; 4],
}

impl VectorValue {
    /// Workaround for "impl const From<Vec2>".
    #[allow(unsafe_code)]
    pub const fn new_vec2(value: Vec2) -> Self {
        let mut s = Self {
            vector_type: VectorType::new(ScalarType::Float, 2),
            storage: [0u32; 4],
        };
        s.storage[0] = unsafe { std::mem::transmute(value.x) };
        s.storage[1] = unsafe { std::mem::transmute(value.y) };
        // let v = bytemuck::cast_slice_mut::<u32, f32>(&mut s.storage);
        // value.write_to_slice(v);
        s
    }

    /// Workaround for "impl const From<Vec3>".
    #[allow(unsafe_code)]
    pub const fn new_vec3(value: Vec3) -> Self {
        let mut s = Self {
            vector_type: VectorType::new(ScalarType::Float, 3),
            storage: [0u32; 4],
        };
        s.storage[0] = unsafe { std::mem::transmute(value.x) };
        s.storage[1] = unsafe { std::mem::transmute(value.y) };
        s.storage[2] = unsafe { std::mem::transmute(value.z) };
        // let v = bytemuck::cast_slice_mut::<u32, f32>(&mut s.storage);
        // value.write_to_slice(v);
        s
    }

    /// Workaround for "impl const From<Vec4>".
    #[allow(unsafe_code)]
    pub const fn new_vec4(value: Vec4) -> Self {
        let mut s = Self {
            vector_type: VectorType::new(ScalarType::Float, 4),
            storage: [0u32; 4],
        };
        s.storage[0] = unsafe { std::mem::transmute(value.to_array()[0]) };
        s.storage[1] = unsafe { std::mem::transmute(value.to_array()[1]) };
        s.storage[2] = unsafe { std::mem::transmute(value.to_array()[2]) };
        s.storage[3] = unsafe { std::mem::transmute(value.to_array()[3]) };
        // let v = bytemuck::cast_slice_mut::<u32, f32>(&mut s.storage);
        // value.write_to_slice(v);
        s
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

    /// Get the value of the N-th element of the vector.
    pub fn get_n<T: ElemType, const N: usize>(&self) -> T {
        if self.vector_type.size() > N {
            T::get_n::<N>(&self.storage)
        } else {
            T::ZERO
        }
    }

    /// Get the value of an element of the vector.
    pub fn get<T: ElemType>(&self, index: usize) -> T {
        if index < self.vector_type.size() {
            T::get(index, &self.storage)
        } else {
            T::ZERO
        }
    }

    /// Get a slice of all the values of the vector.
    ///
    /// This is only valid for numeric types, and will panic for a boolean type.
    pub fn get_all<T: ElemType>(&self) -> &[T] {
        T::get_all(&self.storage, self.vector_type.size())
    }

    /// Get the value as a binary blob ready for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        if self.elem_type() == ScalarType::Bool {
            panic!("Cannot convert bool vector to byte slice.");
        }
        let size = self.vector_type.size();
        bytemuck::cast_slice::<u32, u8>(&self.storage[..size])
    }
}

impl PartialEq for VectorValue {
    fn eq(&self, other: &Self) -> bool {
        let size = self.vector_type.size();
        if size != other.vector_type.size() {
            return false;
        }
        let elem_type = self.elem_type();
        if elem_type == ScalarType::Bool {
            &self.storage[..size] == &other.storage[..size]
        } else if elem_type == ScalarType::Float {
            let mut eq = true;
            for i in 0..size {
                eq = eq && (FloatOrd(self.get::<f32>(i)) == FloatOrd(other.get::<f32>(i)));
            }
            eq
        } else {
            let size = size * 4;
            &self.storage[..size] == &other.storage[..size]
        }
    }
}

impl std::hash::Hash for VectorValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.vector_type.hash(state);
        let size = self.vector_type.size();
        let elem_type = self.elem_type();
        if elem_type == ScalarType::Bool {
            self.storage[..size].hash(state);
        } else if elem_type == ScalarType::Float {
            for i in 0..size {
                FloatOrd(self.get::<f32>(i)).hash(state);
            }
        } else {
            let size = size * 4;
            self.storage[..size].hash(state);
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
        let size = self.vector_type.size();
        if size > 2 {
            vals.push(',');
            vals.push_str(&self.value_n::<2>().to_wgsl_string());
            if size > 3 {
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
        let mut s = Self {
            vector_type: VectorType::new(ScalarType::Bool, 2),
            storage: [0u32; 4],
        };
        let v = bytemuck::cast_slice_mut::<u32, u8>(&mut s.storage);
        v[0] = if value.x { 1 } else { 0 };
        v[1] = if value.y { 1 } else { 0 };
        s
    }
}

impl From<BVec3> for VectorValue {
    fn from(value: BVec3) -> Self {
        let mut s = Self {
            vector_type: VectorType::new(ScalarType::Bool, 3),
            storage: [0u32; 4],
        };
        let v = bytemuck::cast_slice_mut::<u32, u8>(&mut s.storage);
        v[0] = if value.x { 1 } else { 0 };
        v[1] = if value.y { 1 } else { 0 };
        v[2] = if value.z { 1 } else { 0 };
        s
    }
}

impl From<BVec4> for VectorValue {
    fn from(value: BVec4) -> Self {
        let mut s = Self {
            vector_type: VectorType::new(ScalarType::Bool, 4),
            storage: [0u32; 4],
        };
        let v = bytemuck::cast_slice_mut::<u32, u8>(&mut s.storage);
        v[0] = if value.x { 1 } else { 0 };
        v[1] = if value.y { 1 } else { 0 };
        v[2] = if value.z { 1 } else { 0 };
        v[3] = if value.w { 1 } else { 0 };
        s
    }
}

impl From<Vec2> for VectorValue {
    fn from(value: Vec2) -> Self {
        let mut s = Self {
            vector_type: VectorType::new(ScalarType::Float, 2),
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
            vector_type: VectorType::new(ScalarType::Float, 3),
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
            vector_type: VectorType::new(ScalarType::Float, 4),
            storage: [0u32; 4],
        };
        let v = bytemuck::cast_slice_mut::<u32, f32>(&mut s.storage);
        value.write_to_slice(v);
        s
    }
}

/// Floating-point matrix value.
#[derive(Debug, Clone, Copy, Reflect, FromReflect, Serialize, Deserialize)]
pub struct MatrixValue {
    /// Matrix type.
    matrix_type: MatrixType,
    /// Raw storage for up to 16 values. Actual usage depends on matrix size.
    storage: [f32; 16],
}

impl MatrixValue {
    /// Scalar type of the elements of the matrix.
    ///
    /// This always returns [`ScalarType::Float`]. This method is provided for consistency.
    pub const fn elem_type(&self) -> ScalarType {
        ScalarType::Float
    }

    /// Matrix type.
    pub fn matrix_type(&self) -> MatrixType {
        self.matrix_type
    }

    /// Get the scalar value of the matrix element in the R-th row and C-th column.
    pub fn value_n<const R: usize, const C: usize>(&self) -> ScalarValue {
        ScalarValue::Float(self.get_n::<R, C>())
    }

    /// Get the scalar value of a matrix element.
    pub fn value(&self, row: usize, col: usize) -> ScalarValue {
        ScalarValue::Float(self.get(row, col))
    }

    /// Get the floating-point value of the matrix element in the R-th row and C-th column.
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

    /// Get the value as a binary blob ready for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        let count = self.matrix_type.rows() * self.matrix_type.cols();
        bytemuck::cast_slice::<f32, u8>(&self.storage[..count])
    }
}

impl PartialEq for MatrixValue {
    fn eq(&self, other: &Self) -> bool {
        if self.matrix_type != other.matrix_type {
            return false;
        }
        let count = self.matrix_type.rows() * self.matrix_type.cols();
        &self.storage[..count] == &other.storage[..count]
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
        let mut vals = format!(
            "{}({}",
            self.matrix_type().to_wgsl_string(),
            self.value_n::<0, 0>().to_wgsl_string()
        );
        for j in 0..self.matrix_type.cols() {
            for i in 0..self.matrix_type.rows() {
                vals.push(',');
                vals.push_str(&self.value(i, j).to_wgsl_string());
            }
        }
        vals.push(')');
        vals
    }
}

/// Variant storage for a simple value.
#[derive(Debug, Clone, Copy, PartialEq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Value {
    /// Scalar value.
    Scalar(ScalarValue),
    /// Vector value with 2 to 4 components.
    Vector(VectorValue),
    /// Floating-point matrix value of size 2x2 to 4x4.
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

    /// Type of the value.
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::Scalar(s) => ValueType::Scalar(s.scalar_type()),
            Value::Vector(v) => ValueType::Vector(v.vector_type()),
            Value::Matrix(m) => ValueType::Matrix(m.matrix_type()),
        }
    }
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

// impl TryInto<f32> for Value {
//     type Error = ExprError;

//     fn try_into(self) -> Result<f32, Self::Error> {
//         match self {
//             Value::Float(f) => Ok(f),
//             _ => Err(ExprError::TypeError(format!(
//                 "Expected {:?} type, found {:?} instead.",
//                 ValueType::Float,
//                 self.value_type()
//             ))),
//         }
//     }
// }

// impl From<Vec2> for Value {
//     fn from(v: Vec2) -> Self {
//         Self::Float2(v)
//     }
// }

// impl TryInto<Vec2> for Value {
//     type Error = ExprError;

//     fn try_into(self) -> Result<Vec2, Self::Error> {
//         match self {
//             Value::Float2(v) => Ok(v),
//             _ => Err(ExprError::TypeError(format!(
//                 "Expected {:?} type, found {:?} instead.",
//                 ValueType::Float2,
//                 self.value_type()
//             ))),
//         }
//     }
// }

// impl From<Vec3> for Value {
//     fn from(v: Vec3) -> Self {
//         Self::Float3(v)
//     }
// }

// impl TryInto<Vec3> for Value {
//     type Error = ExprError;

//     fn try_into(self) -> Result<Vec3, Self::Error> {
//         match self {
//             Value::Float3(v) => Ok(v),
//             _ => Err(ExprError::TypeError(format!(
//                 "Expected {:?} type, found {:?} instead.",
//                 ValueType::Float3,
//                 self.value_type()
//             ))),
//         }
//     }
// }

// impl From<Vec4> for Value {
//     fn from(v: Vec4) -> Self {
//         Self::Float4(v)
//     }
// }

// impl TryInto<Vec4> for Value {
//     type Error = ExprError;

//     fn try_into(self) -> Result<Vec4, Self::Error> {
//         match self {
//             Value::Float4(v) => Ok(v),
//             _ => Err(ExprError::TypeError(format!(
//                 "Expected {:?} type, found {:?} instead.",
//                 ValueType::Float4,
//                 self.value_type()
//             ))),
//         }
//     }
// }

// impl From<u32> for Value {
//     fn from(u: u32) -> Self {
//         Self::Uint(u)
//     }
// }

// impl TryInto<u32> for Value {
//     type Error = ExprError;

//     fn try_into(self) -> Result<u32, Self::Error> {
//         match self {
//             Value::Uint(v) => Ok(v),
//             _ => Err(ExprError::TypeError(format!(
//                 "Expected {:?} type, found {:?} instead.",
//                 ValueType::Uint,
//                 self.value_type()
//             ))),
//         }
//     }
// }

#[cfg(test)]
mod tests {
    // use std::{
    //     collections::hash_map::DefaultHasher,
    //     hash::{Hash, Hasher},
    // };

    use super::*;

    #[test]
    fn as_bytes() {
        let v = Value::Scalar(3f32.into());
        let b = v.as_bytes();
        assert_eq!(b, &[0u8, 0u8, 0x40u8, 0x40u8]); // 0x40400000

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

        let v = Value::Scalar(0x12FF89ACu32.into());
        let b = v.as_bytes();
        assert_eq!(b, &[0xACu8, 0x89u8, 0xFFu8, 0x12u8]);
    }

    #[test]
    fn value_type() {
        assert_eq!(
            Value::Scalar(0f32.into()).value_type(),
            ValueType::Scalar(ScalarType::Float)
        );
        assert_eq!(
            Value::Vector(Vec2::ZERO.into()).value_type(),
            ValueType::Vector(VectorType::FLOAT2)
        );
        assert_eq!(
            Value::Vector(Vec3::ZERO.into()).value_type(),
            ValueType::Vector(VectorType::FLOAT3)
        );
        assert_eq!(
            Value::Vector(Vec4::ZERO.into()).value_type(),
            ValueType::Vector(VectorType::FLOAT4)
        );
        assert_eq!(
            Value::Scalar(0u32.into()).value_type(),
            ValueType::Scalar(ScalarType::Uint)
        );
    }

    #[test]
    fn to_wgsl_string() {
        assert_eq!(
            Value::Scalar(0f32.into()).to_wgsl_string(),
            0_f32.to_wgsl_string()
        );
        assert_eq!(
            Value::Vector(Vec2::ZERO.into()).to_wgsl_string(),
            Vec2::ZERO.to_wgsl_string()
        );
        assert_eq!(
            Value::Vector(Vec3::ZERO.into()).to_wgsl_string(),
            Vec3::ZERO.to_wgsl_string()
        );
        assert_eq!(
            Value::Vector(Vec4::ZERO.into()).to_wgsl_string(),
            Vec4::ZERO.to_wgsl_string()
        );
        assert_eq!(
            Value::Scalar(0u32.into()).to_wgsl_string(),
            0_u32.to_wgsl_string()
        );
    }

    // #[test]
    // fn from() {
    //     assert_eq!(Value::Float(0.), 0_f32.into());
    //     assert_eq!(Value::Float2(Vec2::ZERO), Vec2::ZERO.into());
    //     assert_eq!(Value::Float3(Vec3::ZERO), Vec3::ZERO.into());
    //     assert_eq!(Value::Float4(Vec4::ZERO), Vec4::ZERO.into());
    //     assert_eq!(Value::Uint(0), 0_u32.into());
    // }

    // fn scalar_hash<H: Hash>(value: &H) -> u64 {
    //     let mut hasher = DefaultHasher::default();
    //     value.hash(&mut hasher);
    //     hasher.finish()
    // }

    // fn vector_hash(values: &[f32]) -> u64 {
    //     let mut hasher = DefaultHasher::default();
    //     for f in values {
    //         FloatOrd(*f).hash(&mut hasher);
    //     }
    //     hasher.finish()
    // }

    // #[test]
    // fn hash() {
    //     assert_eq!(
    //         scalar_hash(&Value::Float(0.)),
    //         scalar_hash(&FloatOrd(0_f32))
    //     );
    //     assert_eq!(scalar_hash(&Value::Uint(0)), scalar_hash(&0_u32));
    //     assert_eq!(
    //         scalar_hash(&Value::Float2(Vec2::new(3.5, -42.))),
    //         vector_hash(&[3.5, -42.])
    //     );
    //     assert_eq!(
    //         scalar_hash(&Value::Float3(Vec3::new(3.5, -42., 999.99))),
    //         vector_hash(&[3.5, -42., 999.99])
    //     );
    //     assert_eq!(
    //         scalar_hash(&Value::Float4(Vec4::new(3.5, -42., 999.99, -0.01))),
    //         vector_hash(&[3.5, -42., 999.99, -0.01])
    //     );
    // }
}
