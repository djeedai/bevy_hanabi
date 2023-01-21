//! Effect graph and language definition.

use std::fmt::Debug;

use bevy::{
    math::{Vec2, Vec3, Vec4},
    reflect::{FromReflect, Reflect},
};
use serde::{Deserialize, Serialize};

use crate::{ToWgslString, ValueType};

/// Variant storage for a simple value.
#[derive(Debug, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Value {
    /// Single `f32` value.
    Float(f32),
    /// Vector of two `f32` values (`vec2<f32>`).
    Float2(Vec2),
    /// Vector of three `f32` values (`vec3<f32>`).
    Float3(Vec3),
    /// Vector of four `f32` values (`vec4<f32>`).
    Float4(Vec4),
    /// Single `u32` value.
    Uint(u32),
}

impl Value {
    /// Type of the value.
    pub fn value_type(&self) -> ValueType {
        match *self {
            Value::Float(_) => ValueType::Float,
            Value::Float2(_) => ValueType::Float2,
            Value::Float3(_) => ValueType::Float3,
            Value::Float4(_) => ValueType::Float4,
            Value::Uint(_) => ValueType::Uint,
        }
    }
}

impl ToWgslString for Value {
    fn to_wgsl_string(&self) -> String {
        match *self {
            Value::Float(f) => f.to_wgsl_string(),
            Value::Float2(v2) => v2.to_wgsl_string(),
            Value::Float3(v3) => v3.to_wgsl_string(),
            Value::Float4(v4) => v4.to_wgsl_string(),
            Value::Uint(u) => u.to_wgsl_string(),
        }
    }
}

impl From<f32> for Value {
    fn from(f: f32) -> Self {
        Self::Float(f)
    }
}

impl From<Vec2> for Value {
    fn from(v: Vec2) -> Self {
        Self::Float2(v)
    }
}

impl From<Vec3> for Value {
    fn from(v: Vec3) -> Self {
        Self::Float3(v)
    }
}

impl From<Vec4> for Value {
    fn from(v: Vec4) -> Self {
        Self::Float4(v)
    }
}

impl From<u32> for Value {
    fn from(u: u32) -> Self {
        Self::Uint(u)
    }
}

/// Language expression producing a value.
#[typetag::serde]
pub trait Expr: Debug + ToWgslString + Send + Sync + 'static {
    /// Is the expression resulting in a compile-time constant?
    fn is_const(&self) -> bool {
        false
    }

    /// The type of the value produced by the expression.
    fn value_type(&self) -> ValueType;

    /// Create a boxed clone of self.
    fn boxed_clone(&self) -> BoxedExpr;
}

/// Boxed [`Expr`] for storing an expression.
pub type BoxedExpr = Box<dyn Expr>;

impl Clone for BoxedExpr {
    fn clone(&self) -> Self {
        self.boxed_clone()
    }
}

impl ToWgslString for BoxedExpr {
    fn to_wgsl_string(&self) -> String {
        let d: &dyn Expr = self.as_ref();
        d.to_wgsl_string()
    }
}

/// A literal constant expression like `3.0` or `vec3<f32>(1.0, 2.0, 3.0)`.
#[derive(Debug, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct Literal {
    value: Value,
}

impl Literal {
    /// Create a new literal expression from a [`Value`].
    pub fn new<V>(value: V) -> Self
    where
        Value: From<V>,
    {
        Self {
            value: value.into(),
        }
    }
}

impl ToWgslString for Literal {
    fn to_wgsl_string(&self) -> String {
        self.value.to_wgsl_string()
    }
}

#[typetag::serde]
impl Expr for Literal {
    fn is_const(&self) -> bool {
        true
    }

    fn value_type(&self) -> ValueType {
        self.value.value_type()
    }

    fn boxed_clone(&self) -> BoxedExpr {
        Box::new(Literal { value: self.value })
    }
}

impl From<Value> for Literal {
    fn from(value: Value) -> Self {
        Self { value }
    }
}

impl From<&Value> for Literal {
    fn from(value: &Value) -> Self {
        Self { value: *value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde() {
        let l: Literal = Value::Float(3.0).into();
        let s = ron::to_string(&l).unwrap();
        println!("literal: {:?}", s);
        let l_serde: Literal = ron::from_str(&s).unwrap();
        assert_eq!(l_serde, l);

        let b: BoxedExpr = Box::new(l);
        let s = ron::to_string(&b).unwrap();
        println!("boxed literal: {:?}", s);
        let b_serde: BoxedExpr = ron::from_str(&s).unwrap();
        assert!(b_serde.is_const());
    }
}
