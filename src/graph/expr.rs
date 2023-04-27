use std::any::Any;

use bevy::{
    prelude::*,
    reflect::{FromReflect, Reflect, ReflectMut, ReflectOwned, ReflectRef, TypeInfo},
    utils::thiserror::Error,
};
use serde::{Deserialize, Serialize};

use crate::{ToWgslString, ValueType};

use super::Value;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ExprError {
    #[error("Type error: {0:?}")]
    TypeError(String),
    #[error("Syntax error: {0:?}")]
    SyntaxError(String),
}

/// Language expression producing a value.
#[typetag::serde]
pub trait Expr: std::fmt::Debug + ToWgslString + Send + Sync + Reflect + 'static {
    /// Get an expression.
    fn as_expr(&self) -> &dyn Expr;

    /// Is the expression resulting in a compile-time constant which can be hard-coded into a shader's code?
    fn is_const(&self) -> bool {
        false
    }

    /// The type of the value produced by the expression.
    fn value_type(&self) -> ValueType;

    /// Evaluate the expression.
    fn eval(&self) -> Result<Value, ExprError>;

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

impl Reflect for BoxedExpr {
    fn type_name(&self) -> &str {
        (**self).type_name()
    }

    fn get_type_info(&self) -> &'static TypeInfo {
        (**self).get_type_info()
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        unimplemented!(); // TODO -- (**self).into_any()
    }

    fn as_any(&self) -> &dyn Any {
        let this: &dyn Expr = &**self;
        this.as_any()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        let this: &mut dyn Expr = &mut **self;
        this.as_any_mut()
    }

    fn into_reflect(self: Box<Self>) -> Box<dyn Reflect> {
        unimplemented!(); // TODO -- (**self).into_reflect()
    }

    fn as_reflect(&self) -> &dyn Reflect {
        (**self).as_reflect()
    }

    fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
        (**self).as_reflect_mut()
    }

    fn apply(&mut self, value: &dyn Reflect) {
        (**self).apply(value);
    }

    fn set(&mut self, value: Box<dyn Reflect>) -> Result<(), Box<dyn Reflect>> {
        (**self).set(value)
    }

    fn reflect_ref(&self) -> ReflectRef {
        (**self).reflect_ref()
    }

    fn reflect_mut(&mut self) -> ReflectMut {
        (**self).reflect_mut()
    }

    fn reflect_owned(self: Box<Self>) -> ReflectOwned {
        unimplemented!(); // TODO -- (**self).reflect_owned()
    }

    fn clone_value(&self) -> Box<dyn Reflect> {
        (**self).clone_value()
    }
}

impl FromReflect for BoxedExpr {
    fn from_reflect(_reflect: &dyn Reflect) -> Option<Self> {
        unimplemented!(); // TODO
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
    fn as_expr(&self) -> &dyn Expr {
        self
    }

    fn is_const(&self) -> bool {
        true
    }

    fn value_type(&self) -> ValueType {
        self.value.value_type()
    }

    fn eval(&self) -> Result<Value, ExprError> {
        Ok(self.value)
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

/// Addition expression between two expressions of the same type.
#[derive(Debug, Clone, Reflect, FromReflect, Serialize, Deserialize)]
pub struct AddExpr {
    left: BoxedExpr,
    right: BoxedExpr,
}

impl AddExpr {
    /// Create a new addition expression between two boxed expressions.
    #[inline]
    pub fn new<L: Expr, R: Expr>(lhs: L, rhs: R) -> Self {
        assert_eq!(lhs.value_type(), rhs.value_type());
        Self {
            left: Box::new(lhs),
            right: Box::new(rhs),
        }
    }
}

#[typetag::serde]
impl Expr for AddExpr {
    fn as_expr(&self) -> &dyn Expr {
        self
    }

    fn is_const(&self) -> bool {
        self.left.is_const() && self.right.is_const()
    }

    fn value_type(&self) -> ValueType {
        self.left.value_type() // TODO: cast left/right? or always gave same as invariant?
    }

    fn eval(&self) -> Result<Value, ExprError> {
        let value_type = self.left.value_type();
        if value_type != self.right.value_type() {
            return Err(ExprError::TypeError(format!(
                "Mismatching L/R types: {:?} != {:?}",
                value_type,
                self.right.value_type()
            )));
        }

        let l = self.left.eval()?;
        let r = self.right.eval()?;

        todo!();
        // match value_type {
        //     ValueType::Scalar(s) => match s {
        //         ScalarType::Bool => {
        //             let r: bool = r.try_into()?;
        //             Ok(Value::Uint(l + r))
        //         }
        //         ScalarType::Float => {
        //             let r: f32 = r.try_into()?;
        //             Ok(Value::Float(l + r))
        //         }
        //         ScalarType::Int => {
        //             let r: f32 = r.try_into()?;
        //             Ok(Value::Float(l + r))
        //         }
        //         ScalarType::Uint => {
        //             let r: f32 = r.try_into()?;
        //             Ok(Value::Float(l + r))
        //         }
        //     },
        // }
    }

    fn boxed_clone(&self) -> BoxedExpr {
        Box::new(AddExpr {
            left: self.left.boxed_clone(),
            right: self.right.boxed_clone(),
        })
    }
}

impl ToWgslString for AddExpr {
    fn to_wgsl_string(&self) -> String {
        format!(
            "{} + {}",
            self.left.to_wgsl_string(),
            self.right.to_wgsl_string()
        )
    }
}

impl std::ops::Add for Literal {
    type Output = AddExpr;

    fn add(self, rhs: Self) -> Self::Output {
        AddExpr::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn err() {
    //     let l = Value::Scalar(3.5_f32.into());
    //     let r: Result<Vec2, ExprError> = l.try_into();
    //     assert!(r.is_err());
    //     assert!(matches!(r, Err(ExprError::TypeError(_))));
    // }

    #[test]
    fn serde() {
        let v = Value::Scalar(3.0_f32.into());
        let l: Literal = v.into();
        assert_eq!(Ok(v), l.eval());
        let s = ron::to_string(&l).unwrap();
        println!("literal: {:?}", s);
        let l_serde: Literal = ron::from_str(&s).unwrap();
        assert_eq!(l_serde, l);

        let b: BoxedExpr = Box::new(l);
        let s = ron::to_string(&b).unwrap();
        println!("boxed literal: {:?}", s);
        let b_serde: BoxedExpr = ron::from_str(&s).unwrap();
        assert!(b_serde.is_const());
        assert_eq!(b_serde.to_wgsl_string(), b.to_wgsl_string());

        let v0 = Value::Scalar(3.0_f32.into());
        let v1 = Value::Scalar(2.5_f32.into());
        let l0: Literal = v0.into();
        let l1: Literal = v1.into();
        let a = l0 + l1;
        assert!(a.is_const());
        assert_eq!(Ok(Value::Scalar(5.5_f32.into())), a.eval());
        let s = ron::to_string(&a).unwrap();
        println!("add: {:?}", s);
        let a_serde: AddExpr = ron::from_str(&s).unwrap();
        println!("a_serde: {:?}", a_serde);
        assert_eq!(a_serde.left.to_wgsl_string(), l0.to_wgsl_string());
        assert_eq!(a_serde.right.to_wgsl_string(), l1.to_wgsl_string());
    }
}
