//! Effect graph and language definition.
//!
//! This module contains the elements used to build a effect graph, a fully
//! customizable description of a visual effect.
//!
//! Currently effect graphs are not yet available; only some preview elements
//! exist. So this module is a bit empty and of little interest.

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
    /// Get the value as a binary blob ready for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Value::Float(f) => bytemuck::cast_slice::<f32, u8>(std::slice::from_ref(f)),
            Value::Float2(v) => bytemuck::cast_slice::<Vec2, u8>(std::slice::from_ref(v)),
            Value::Float3(v) => bytemuck::cast_slice::<Vec3, u8>(std::slice::from_ref(v)),
            Value::Float4(v) => bytemuck::cast_slice::<Vec4, u8>(std::slice::from_ref(v)),
            Value::Uint(u) => bytemuck::cast_slice::<u32, u8>(std::slice::from_ref(u)),
        }
    }
}

impl Value {
    /// Type of the value.
    pub fn value_type(&self) -> ValueType {
        match self {
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
        match self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_bytes() {
        let v = Value::Float(3.);
        let b = v.as_bytes();
        assert_eq!(b, &[0u8, 0u8, 0x40u8, 0x40u8]); // 0x40400000

        let v = Value::Float2(Vec2::new(-2., 3.));
        let b = v.as_bytes();
        assert_eq!(b, &[0u8, 0u8, 0u8, 0xC0u8, 0u8, 0u8, 0x40u8, 0x40u8]); // 0xc0000000 0x40400000

        let v = Value::Float3(Vec3::new(-2., 3., 4.));
        let b = v.as_bytes();
        assert_eq!(
            b,
            &[0u8, 0u8, 0u8, 0xC0u8, 0u8, 0u8, 0x40u8, 0x40u8, 0u8, 0u8, 0x80u8, 0x40u8]
        ); // 0xc0000000 0x40400000 0x40800000

        let v = Value::Float4(Vec4::new(-2., 3., 4., -5.));
        let b = v.as_bytes();
        assert_eq!(
            b,
            &[
                0u8, 0u8, 0u8, 0xC0u8, 0u8, 0u8, 0x40u8, 0x40u8, 0u8, 0u8, 0x80u8, 0x40u8, 0u8,
                0u8, 0xa0u8, 0xc0u8
            ]
        ); // 0xc0000000 0x40400000 0x40800000 0xc0a00000

        let v = Value::Uint(0x12FF89ACu32);
        let b = v.as_bytes();
        assert_eq!(b, &[0xACu8, 0x89u8, 0xFFu8, 0x12u8]);
    }

    #[test]
    fn value_type() {
        assert_eq!(Value::Float(0.).value_type(), ValueType::Float);
        assert_eq!(Value::Float2(Vec2::ZERO).value_type(), ValueType::Float2);
        assert_eq!(Value::Float3(Vec3::ZERO).value_type(), ValueType::Float3);
        assert_eq!(Value::Float4(Vec4::ZERO).value_type(), ValueType::Float4);
        assert_eq!(Value::Uint(0).value_type(), ValueType::Uint);
    }

    #[test]
    fn to_wgsl_string() {
        assert_eq!(Value::Float(0.).to_wgsl_string(), 0_f32.to_wgsl_string());
        assert_eq!(
            Value::Float2(Vec2::ZERO).to_wgsl_string(),
            Vec2::ZERO.to_wgsl_string()
        );
        assert_eq!(
            Value::Float3(Vec3::ZERO).to_wgsl_string(),
            Vec3::ZERO.to_wgsl_string()
        );
        assert_eq!(
            Value::Float4(Vec4::ZERO).to_wgsl_string(),
            Vec4::ZERO.to_wgsl_string()
        );
        assert_eq!(Value::Uint(0).to_wgsl_string(), 0_u32.to_wgsl_string());
    }

    #[test]
    fn from() {
        assert_eq!(Value::Float(0.), 0_f32.into());
        assert_eq!(Value::Float2(Vec2::ZERO), Vec2::ZERO.into());
        assert_eq!(Value::Float3(Vec3::ZERO), Vec3::ZERO.into());
        assert_eq!(Value::Float4(Vec4::ZERO), Vec4::ZERO.into());
        assert_eq!(Value::Uint(0), 0_u32.into());
    }
}
