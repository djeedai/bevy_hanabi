use std::any::Any;

use bevy::{
    prelude::*,
    reflect::{FromReflect, Reflect, ReflectMut, ReflectOwned, ReflectRef, TypeInfo},
    utils::thiserror::Error,
};
use serde::{Deserialize, Serialize};

use crate::{Attribute, ScalarType, ToWgslString, ValueType};

use super::{BinaryOperator, Value, VectorValue};

///
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ExprError {
    ///
    #[error("Type error: {0:?}")]
    TypeError(String),
    ///
    #[error("Syntax error: {0:?}")]
    SyntaxError(String),
    ///
    #[error("Graph evaluation error: {0:?}")]
    GraphEvalError(String),
}

/// Language expression producing a value.
#[typetag::serde]
pub trait Expr: std::fmt::Debug + ToWgslString + Send + Sync + Reflect + 'static {
    /// Get an expression.
    fn as_expr(&self) -> &dyn Expr;

    /// Is the expression resulting in a compile-time constant which can be
    /// hard-coded into a shader's code?
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

impl<T: Expr> From<T> for BoxedExpr {
    fn from(value: T) -> Self {
        Box::new(value)
    }
}

/// A literal constant expression like `3.0` or `vec3<f32>(1.0, 2.0, 3.0)`.
#[derive(Debug, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct LiteralExpr {
    value: Value,
}

impl LiteralExpr {
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

impl ToWgslString for LiteralExpr {
    fn to_wgsl_string(&self) -> String {
        self.value.to_wgsl_string()
    }
}

#[typetag::serde]
impl Expr for LiteralExpr {
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
        Box::new(LiteralExpr { value: self.value })
    }
}

impl From<&Value> for LiteralExpr {
    fn from(value: &Value) -> Self {
        Self { value: *value }
    }
}

impl<T: Into<Value>> From<T> for LiteralExpr {
    fn from(value: T) -> Self {
        Self {
            value: value.into(),
        }
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
    pub fn new<L: Into<BoxedExpr>, R: Into<BoxedExpr>>(lhs: L, rhs: R) -> Self {
        let lhs: BoxedExpr = lhs.into();
        let rhs: BoxedExpr = rhs.into();
        Self {
            left: lhs,
            right: rhs,
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
        self.left.value_type() // TODO: cast left/right? or always gave same as
                               // invariant?
    }

    fn eval(&self) -> Result<Value, ExprError> {
        let value_type = self.left.value_type();
        if !value_type.is_numeric() || !self.right.value_type().is_numeric() {
            return Err(ExprError::TypeError(format!(
                "Cannot apply Add binary operator to boolean type: {:?} and {:?}",
                value_type,
                self.right.value_type()
            )));
        }
        if value_type != self.right.value_type() {
            // Special case: mixed scalar and vector operands
            // https://www.w3.org/TR/WGSL/#arithmetic-expr
            if value_type.is_scalar() && self.right.value_type().is_vector() {
                // es + ev => V(es) + ev
            } else if value_type.is_vector() && self.right.value_type().is_scalar() {
                // ev + es => ev + V(es)
            } else {
                return Err(ExprError::TypeError(format!(
                    "Mismatching L/R types: {:?} != {:?}",
                    value_type,
                    self.right.value_type()
                )));
            }
        }

        let l = self.left.eval()?;
        let r = self.right.eval()?;

        if value_type != self.right.value_type() {
            if value_type.is_scalar() {
                // es + ev => V(es) + ev
                let l = Value::Vector(VectorValue::splat(
                    l.as_scalar(),
                    r.as_vector().vector_type().count() as u8,
                ));
                Ok(l.binary_op(&r, BinaryOperator::Add))
            } else {
                // ev + es => ev + V(es)
                let r = Value::Vector(VectorValue::splat(
                    r.as_scalar(),
                    l.as_vector().vector_type().count() as u8,
                ));
                Ok(l.binary_op(&r, BinaryOperator::Add))
            }
        } else {
            Ok(l.binary_op(&r, BinaryOperator::Add))
        }

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
            "({}) + ({})",
            self.left.to_wgsl_string(),
            self.right.to_wgsl_string()
        )
    }
}

/// Subtraction expression between two expressions of the same type.
#[derive(Debug, Clone, Reflect, FromReflect, Serialize, Deserialize)]
pub struct SubExpr {
    left: BoxedExpr,
    right: BoxedExpr,
}

impl SubExpr {
    /// Create a new addition expression between two boxed expressions.
    #[inline]
    pub fn new<L: Into<BoxedExpr>, R: Into<BoxedExpr>>(lhs: L, rhs: R) -> Self {
        let lhs: BoxedExpr = lhs.into();
        let rhs: BoxedExpr = rhs.into();
        assert_eq!(lhs.value_type(), rhs.value_type());
        Self {
            left: lhs,
            right: rhs,
        }
    }
}

#[typetag::serde]
impl Expr for SubExpr {
    fn as_expr(&self) -> &dyn Expr {
        self
    }

    fn is_const(&self) -> bool {
        self.left.is_const() && self.right.is_const()
    }

    fn value_type(&self) -> ValueType {
        self.left.value_type() // TODO: cast left/right? or always gave same as
                               // invariant?
    }

    fn eval(&self) -> Result<Value, ExprError> {
        let value_type = self.left.value_type();
        if !value_type.is_numeric() || !self.right.value_type().is_numeric() {
            return Err(ExprError::TypeError(format!(
                "Cannot apply Sub binary operator to boolean type: {:?} and {:?}",
                value_type,
                self.right.value_type()
            )));
        }
        if value_type != self.right.value_type() {
            return Err(ExprError::TypeError(format!(
                "Mismatching L/R types: {:?} != {:?}",
                value_type,
                self.right.value_type()
            )));
        }

        if !value_type.is_numeric() {
            return Err(ExprError::TypeError(format!(
                "Non-numeric type in Add expression."
            )));
        }

        let l = self.left.eval()?;
        let r = self.right.eval()?;

        Ok(l.binary_op(&r, BinaryOperator::Sub))

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
        Box::new(SubExpr {
            left: self.left.boxed_clone(),
            right: self.right.boxed_clone(),
        })
    }
}

impl ToWgslString for SubExpr {
    fn to_wgsl_string(&self) -> String {
        format!(
            "({}) - ({})",
            self.left.to_wgsl_string(),
            self.right.to_wgsl_string()
        )
    }
}

/// Multiply expression between two expressions of the same type.
#[derive(Debug, Clone, Reflect, FromReflect, Serialize, Deserialize)]
pub struct MulExpr {
    left: BoxedExpr,
    right: BoxedExpr,
}

impl MulExpr {
    /// Create a new addition expression between two boxed expressions.
    #[inline]
    pub fn new<L: Into<BoxedExpr>, R: Into<BoxedExpr>>(lhs: L, rhs: R) -> Self {
        let lhs: BoxedExpr = lhs.into();
        let rhs: BoxedExpr = rhs.into();
        assert_eq!(lhs.value_type(), rhs.value_type());
        Self {
            left: lhs,
            right: rhs,
        }
    }
}

#[typetag::serde]
impl Expr for MulExpr {
    fn as_expr(&self) -> &dyn Expr {
        self
    }

    fn is_const(&self) -> bool {
        self.left.is_const() && self.right.is_const()
    }

    fn value_type(&self) -> ValueType {
        self.left.value_type() // TODO: cast left/right? or always gave same as
                               // invariant?
    }

    fn eval(&self) -> Result<Value, ExprError> {
        let value_type = self.left.value_type();
        if !value_type.is_numeric() || !self.right.value_type().is_numeric() {
            return Err(ExprError::TypeError(format!(
                "Cannot apply Mul binary operator to boolean type: {:?} and {:?}",
                value_type,
                self.right.value_type()
            )));
        }
        if value_type != self.right.value_type() {
            return Err(ExprError::TypeError(format!(
                "Mismatching L/R types: {:?} != {:?}",
                value_type,
                self.right.value_type()
            )));
        }

        if !value_type.is_numeric() {
            return Err(ExprError::TypeError(format!(
                "Non-numeric type in Add expression."
            )));
        }

        let l = self.left.eval()?;
        let r = self.right.eval()?;

        Ok(l.binary_op(&r, BinaryOperator::Mul))

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
        Box::new(MulExpr {
            left: self.left.boxed_clone(),
            right: self.right.boxed_clone(),
        })
    }
}

impl ToWgslString for MulExpr {
    fn to_wgsl_string(&self) -> String {
        format!(
            "({}) * ({})",
            self.left.to_wgsl_string(),
            self.right.to_wgsl_string()
        )
    }
}

/// Divide expression between two expressions of the same type.
#[derive(Debug, Clone, Reflect, FromReflect, Serialize, Deserialize)]
pub struct DivExpr {
    left: BoxedExpr,
    right: BoxedExpr,
}

impl DivExpr {
    /// Create a new addition expression between two boxed expressions.
    #[inline]
    pub fn new<L: Into<BoxedExpr>, R: Into<BoxedExpr>>(lhs: L, rhs: R) -> Self {
        let lhs: BoxedExpr = lhs.into();
        let rhs: BoxedExpr = rhs.into();
        assert_eq!(lhs.value_type(), rhs.value_type());
        Self {
            left: lhs,
            right: rhs,
        }
    }
}

#[typetag::serde]
impl Expr for DivExpr {
    fn as_expr(&self) -> &dyn Expr {
        self
    }

    fn is_const(&self) -> bool {
        self.left.is_const() && self.right.is_const()
    }

    fn value_type(&self) -> ValueType {
        self.left.value_type() // TODO: cast left/right? or always gave same as
                               // invariant?
    }

    fn eval(&self) -> Result<Value, ExprError> {
        let value_type = self.left.value_type();
        if !value_type.is_numeric() || !self.right.value_type().is_numeric() {
            return Err(ExprError::TypeError(format!(
                "Cannot apply Div binary operator to boolean type: {:?} and {:?}",
                value_type,
                self.right.value_type()
            )));
        }
        if value_type != self.right.value_type() {
            return Err(ExprError::TypeError(format!(
                "Mismatching L/R types: {:?} != {:?}",
                value_type,
                self.right.value_type()
            )));
        }

        if !value_type.is_numeric() {
            return Err(ExprError::TypeError(format!(
                "Non-numeric type in Add expression."
            )));
        }

        let l = self.left.eval()?;
        let r = self.right.eval()?;

        Ok(l.binary_op(&r, BinaryOperator::Div))

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
        Box::new(DivExpr {
            left: self.left.boxed_clone(),
            right: self.right.boxed_clone(),
        })
    }
}

impl ToWgslString for DivExpr {
    fn to_wgsl_string(&self) -> String {
        format!(
            "({}) / ({})",
            self.left.to_wgsl_string(),
            self.right.to_wgsl_string()
        )
    }
}

/// Attribute expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub struct AttributeExpr {
    attr: Attribute,
}

impl AttributeExpr {
    /// Create a new attribute expression.
    #[inline]
    pub fn new(attr: Attribute) -> Self {
        Self { attr }
    }
}

#[typetag::serde]
impl Expr for AttributeExpr {
    fn as_expr(&self) -> &dyn Expr {
        self
    }

    fn is_const(&self) -> bool {
        false
    }

    fn value_type(&self) -> ValueType {
        self.attr.value_type()
    }

    fn eval(&self) -> Result<Value, ExprError> {
        unimplemented!();
    }

    fn boxed_clone(&self) -> BoxedExpr {
        Box::new(AttributeExpr { attr: self.attr })
    }
}

impl ToWgslString for AttributeExpr {
    fn to_wgsl_string(&self) -> String {
        format!("particle.{}", self.attr.name())
    }
}

impl From<Attribute> for AttributeExpr {
    fn from(value: Attribute) -> Self {
        AttributeExpr::new(value)
    }
}

/// Built-in operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub enum BuiltInOperator {
    /// Current effect system time since startup, in seconds.
    Time,
    /// Delta time, in seconds, since last effect system update.
    DeltaTime,
}

impl BuiltInOperator {
    /// Array of all the built-in operators available.
    pub const ALL: [BuiltInOperator; 2] = [BuiltInOperator::Time, BuiltInOperator::DeltaTime];

    /// Get the operator name.
    pub fn name(&self) -> &str {
        match self {
            BuiltInOperator::Time => "time",
            BuiltInOperator::DeltaTime => "deltaTime",
        }
    }

    /// Get the type of the value of a built-in operator.
    pub fn value_type(&self) -> ValueType {
        match self {
            BuiltInOperator::Time => ValueType::Scalar(ScalarType::Float),
            BuiltInOperator::DeltaTime => ValueType::Scalar(ScalarType::Float),
        }
    }

    // /// Evaluate the result of the operator as an expression.
    // pub fn eval(&self) -> Result<Value, ExprError> {
    //     match self {
    //         BuiltInOperator::Time => Value::Scalar(Scal)
    //     }
    // }
}

impl ToWgslString for BuiltInOperator {
    fn to_wgsl_string(&self) -> String {
        match self {
            BuiltInOperator::Time => "sim_params.time".to_string(),
            BuiltInOperator::DeltaTime => "sim_params.dt".to_string(),
        }
    }
}

/// Expression for getting built-in quantities related to the effect system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub struct BuiltInExpr {
    operator: BuiltInOperator,
}

impl BuiltInExpr {
    /// Create a new built-in operator expression.
    #[inline]
    pub fn new(operator: BuiltInOperator) -> Self {
        Self { operator }
    }
}

#[typetag::serde]
impl Expr for BuiltInExpr {
    fn as_expr(&self) -> &dyn Expr {
        self
    }

    fn is_const(&self) -> bool {
        false
    }

    fn value_type(&self) -> ValueType {
        self.operator.value_type()
    }

    fn eval(&self) -> Result<Value, ExprError> {
        unimplemented!() //self.operator.eval()
    }

    fn boxed_clone(&self) -> BoxedExpr {
        Box::new(BuiltInExpr {
            operator: self.operator,
        })
    }
}

impl ToWgslString for BuiltInExpr {
    fn to_wgsl_string(&self) -> String {
        self.operator.to_wgsl_string()
    }
}

/// Expression for normalizing a vector.
#[derive(Debug, Clone, Reflect, FromReflect, Serialize, Deserialize)]
pub struct NormalizeExpr {
    input: BoxedExpr,
}

impl NormalizeExpr {
    /// Create a new normalize expression.
    #[inline]
    pub fn new(input: BoxedExpr) -> Self {
        Self { input }
    }
}

#[typetag::serde]
impl Expr for NormalizeExpr {
    fn as_expr(&self) -> &dyn Expr {
        self
    }

    fn is_const(&self) -> bool {
        self.input.is_const()
    }

    fn value_type(&self) -> ValueType {
        self.input.value_type()
    }

    fn eval(&self) -> Result<Value, ExprError> {
        unimplemented!() //self.operator.eval()
    }

    fn boxed_clone(&self) -> BoxedExpr {
        Box::new(NormalizeExpr {
            input: self.input.boxed_clone(),
        })
    }
}

impl ToWgslString for NormalizeExpr {
    fn to_wgsl_string(&self) -> String {
        format!("normalize({})", self.input.to_wgsl_string())
    }
}

/// Implement the binary operators for the given concrete expression type.
macro_rules! impl_binary_ops {
    ($t: ty) => {
        impl<T: Into<BoxedExpr>> std::ops::Add<T> for $t {
            type Output = AddExpr;

            fn add(self, rhs: T) -> Self::Output {
                let this: BoxedExpr = Box::new(self);
                AddExpr::new(this, rhs)
            }
        }

        impl<T: Into<BoxedExpr>> std::ops::Sub<T> for $t {
            type Output = SubExpr;

            fn sub(self, rhs: T) -> Self::Output {
                let this: BoxedExpr = Box::new(self);
                SubExpr::new(this, rhs)
            }
        }

        impl<T: Into<BoxedExpr>> std::ops::Mul<T> for $t {
            type Output = MulExpr;

            fn mul(self, rhs: T) -> Self::Output {
                let this: BoxedExpr = Box::new(self);
                MulExpr::new(this, rhs)
            }
        }

        impl<T: Into<BoxedExpr>> std::ops::Div<T> for $t {
            type Output = DivExpr;

            fn div(self, rhs: T) -> Self::Output {
                let this: BoxedExpr = Box::new(self);
                DivExpr::new(this, rhs)
            }
        }
    };
}

impl_binary_ops!(LiteralExpr);
impl_binary_ops!(AddExpr);
impl_binary_ops!(SubExpr);
impl_binary_ops!(MulExpr);
impl_binary_ops!(DivExpr);
impl_binary_ops!(AttributeExpr);
impl_binary_ops!(BuiltInExpr);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn err() {
        let l = Value::Scalar(3.5_f32.into());
        let r: Result<Vec2, ExprError> = l.try_into();
        assert!(r.is_err());
        assert!(matches!(r, Err(ExprError::TypeError(_))));
    }

    #[test]
    fn add_expr() {
        // f32 + f32
        let x: LiteralExpr = 3_f32.into();
        let y: LiteralExpr = 42_f32.into();
        let a = (x + y).eval();
        assert!(a.is_ok());
        let b = (y + x).eval();
        assert!(b.is_ok());
        assert_eq!(a, b);

        // Cannot Add bool
        let z: LiteralExpr = true.into();
        assert!((x + z).eval().is_err());
        assert!((z + x).eval().is_err());

        // Cannot Add a different scalar
        let z: LiteralExpr = 8_u32.into();
        assert!((x + z).eval().is_err());
        assert!((z + x).eval().is_err());

        // f32 + vec3<f32>
        let x: LiteralExpr = 3_f32.into();
        let y: LiteralExpr = Vec3::ONE.into();
        let a = (x + y).eval();
        assert!(a.is_ok());
        let b = (y + x).eval();
        assert!(b.is_ok());
        assert_eq!(a, b);
        assert!(matches!(a.unwrap(), Value::Vector(_)));
    }

    #[test]
    fn math_expr() {
        let x: AttributeExpr = Attribute::POSITION.into();
        let y = LiteralExpr::new(Vec3::ONE);

        let a = x + y;
        assert_eq!(
            a.to_wgsl_string(),
            format!(
                "(particle.{}) + (vec3<f32>(1.,1.,1.))",
                Attribute::POSITION.name()
            )
        );

        let b = y + x;
        assert_eq!(
            b.to_wgsl_string(),
            format!(
                "(vec3<f32>(1.,1.,1.)) + (particle.{})",
                Attribute::POSITION.name()
            )
        );

        let a = x - y;
        assert_eq!(
            a.to_wgsl_string(),
            format!(
                "(particle.{}) - (vec3<f32>(1.,1.,1.))",
                Attribute::POSITION.name()
            )
        );

        let b = y - x;
        assert_eq!(
            b.to_wgsl_string(),
            format!(
                "(vec3<f32>(1.,1.,1.)) - (particle.{})",
                Attribute::POSITION.name()
            )
        );

        let a = x * y;
        assert_eq!(
            a.to_wgsl_string(),
            format!(
                "(particle.{}) * (vec3<f32>(1.,1.,1.))",
                Attribute::POSITION.name()
            )
        );

        let b = y * x;
        assert_eq!(
            b.to_wgsl_string(),
            format!(
                "(vec3<f32>(1.,1.,1.)) * (particle.{})",
                Attribute::POSITION.name()
            )
        );

        let a = x / y;
        assert_eq!(
            a.to_wgsl_string(),
            format!(
                "(particle.{}) / (vec3<f32>(1.,1.,1.))",
                Attribute::POSITION.name()
            )
        );

        let b = y / x;
        assert_eq!(
            b.to_wgsl_string(),
            format!(
                "(vec3<f32>(1.,1.,1.)) / (particle.{})",
                Attribute::POSITION.name()
            )
        );
    }

    #[test]
    fn serde() {
        let v = Value::Scalar(3.0_f32.into());
        let l: LiteralExpr = v.into();
        assert_eq!(Ok(v), l.eval());
        let s = ron::to_string(&l).unwrap();
        println!("literal: {:?}", s);
        let l_serde: LiteralExpr = ron::from_str(&s).unwrap();
        assert_eq!(l_serde, l);

        let b: BoxedExpr = Box::new(l);
        let s = ron::to_string(&b).unwrap();
        println!("boxed literal: {:?}", s);
        let b_serde: BoxedExpr = ron::from_str(&s).unwrap();
        assert!(b_serde.is_const());
        assert_eq!(b_serde.to_wgsl_string(), b.to_wgsl_string());

        let v0 = Value::Scalar(3.0_f32.into());
        let v1 = Value::Scalar(2.5_f32.into());
        let l0: LiteralExpr = v0.into();
        let l1: LiteralExpr = v1.into();
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
