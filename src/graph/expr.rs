use std::any::Any;

use bevy::{
    prelude::*,
    reflect::{FromReflect, Reflect, ReflectMut, ReflectOwned, ReflectRef, TypeInfo},
    utils::thiserror::Error,
};
use serde::{Deserialize, Serialize};

use crate::{Attribute, PropertyLayout, ScalarType, ToWgslString, ValueType};

use super::Value;

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
    ///
    #[error("Property error: {0:?}")]
    PropertyError(String),
}

/// Evaluation context for [`Expr::eval()`].
pub trait EvalContext {
    /// Get the property layout of the effect.
    fn property_layout(&self) -> &PropertyLayout;
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
    fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError>;

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

    fn eval(&self, _context: &dyn EvalContext) -> Result<String, ExprError> {
        Ok(self.value.to_wgsl_string())
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

    fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError> {
        let lhs = self.left.eval(context)?;
        let rhs = self.right.eval(context)?;
        Ok(format!("({}) + ({})", lhs, rhs))

        // let value_type = self.left.value_type();
        // if !value_type.is_numeric() || !self.right.value_type().is_numeric()
        // {     return Err(ExprError::TypeError(format!(
        //         "Cannot apply Add binary operator to boolean type: {:?} and
        // {:?}",         value_type,
        //         self.right.value_type()
        //     )));
        // }
        // if value_type != self.right.value_type() {
        //     // Special case: mixed scalar and vector operands
        //     // https://www.w3.org/TR/WGSL/#arithmetic-expr
        //     if value_type.is_scalar() && self.right.value_type().is_vector()
        // {         // es + ev => V(es) + ev
        //     } else if value_type.is_vector() &&
        // self.right.value_type().is_scalar() {         // ev + es =>
        // ev + V(es)     } else {
        //         return Err(ExprError::TypeError(format!(
        //             "Mismatching L/R types: {:?} != {:?}",
        //             value_type,
        //             self.right.value_type()
        //         )));
        //     }
        // }

        // let l = self.left.eval()?;
        // let r = self.right.eval()?;

        // if value_type != self.right.value_type() {
        //     if value_type.is_scalar() {
        //         // es + ev => V(es) + ev
        //         let l = Value::Vector(VectorValue::splat(
        //             l.as_scalar(),
        //             r.as_vector().vector_type().count() as u8,
        //         ));
        //         Ok(l.binary_op(&r, BinaryOperator::Add))
        //     } else {
        //         // ev + es => ev + V(es)
        //         let r = Value::Vector(VectorValue::splat(
        //             r.as_scalar(),
        //             l.as_vector().vector_type().count() as u8,
        //         ));
        //         Ok(l.binary_op(&r, BinaryOperator::Add))
        //     }
        // } else {
        //     Ok(l.binary_op(&r, BinaryOperator::Add))
        // }

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

    fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError> {
        let lhs = self.left.eval(context)?;
        let rhs = self.right.eval(context)?;
        Ok(format!("({}) - ({})", lhs, rhs))

        // let value_type = self.left.value_type();
        // if !value_type.is_numeric() || !self.right.value_type().is_numeric()
        // {     return Err(ExprError::TypeError(format!(
        //         "Cannot apply Sub binary operator to boolean type: {:?} and
        // {:?}",         value_type,
        //         self.right.value_type()
        //     )));
        // }
        // if value_type != self.right.value_type() {
        //     return Err(ExprError::TypeError(format!(
        //         "Mismatching L/R types: {:?} != {:?}",
        //         value_type,
        //         self.right.value_type()
        //     )));
        // }

        // if !value_type.is_numeric() {
        //     return Err(ExprError::TypeError(format!(
        //         "Non-numeric type in Add expression."
        //     )));
        // }

        // let l = self.left.eval()?;
        // let r = self.right.eval()?;

        // Ok(l.binary_op(&r, BinaryOperator::Sub))

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

    fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError> {
        let lhs = self.left.eval(context)?;
        let rhs = self.right.eval(context)?;
        Ok(format!("({}) * ({})", lhs, rhs))

        // let value_type = self.left.value_type();
        // if !value_type.is_numeric() || !self.right.value_type().is_numeric()
        // {     return Err(ExprError::TypeError(format!(
        //         "Cannot apply Mul binary operator to boolean type: {:?} and
        // {:?}",         value_type,
        //         self.right.value_type()
        //     )));
        // }
        // if value_type != self.right.value_type() {
        //     return Err(ExprError::TypeError(format!(
        //         "Mismatching L/R types: {:?} != {:?}",
        //         value_type,
        //         self.right.value_type()
        //     )));
        // }

        // if !value_type.is_numeric() {
        //     return Err(ExprError::TypeError(format!(
        //         "Non-numeric type in Add expression."
        //     )));
        // }

        // let l = self.left.eval()?;
        // let r = self.right.eval()?;

        // Ok(l.binary_op(&r, BinaryOperator::Mul))

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

    fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError> {
        let lhs = self.left.eval(context)?;
        let rhs = self.right.eval(context)?;
        Ok(format!("({}) / ({})", lhs, rhs))

        // let value_type = self.left.value_type();
        // if !value_type.is_numeric() || !self.right.value_type().is_numeric()
        // {     return Err(ExprError::TypeError(format!(
        //         "Cannot apply Div binary operator to boolean type: {:?} and
        // {:?}",         value_type,
        //         self.right.value_type()
        //     )));
        // }
        // if value_type != self.right.value_type() {
        //     return Err(ExprError::TypeError(format!(
        //         "Mismatching L/R types: {:?} != {:?}",
        //         value_type,
        //         self.right.value_type()
        //     )));
        // }

        // if !value_type.is_numeric() {
        //     return Err(ExprError::TypeError(format!(
        //         "Non-numeric type in Add expression."
        //     )));
        // }

        // let l = self.left.eval()?;
        // let r = self.right.eval()?;

        // Ok(l.binary_op(&r, BinaryOperator::Div))

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

/// Expression representing the value of an attribute of a particle.
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

    fn eval(&self, _context: &dyn EvalContext) -> Result<String, ExprError> {
        Ok(self.to_wgsl_string())
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

/// Expression representing the value of a property of an effect.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub struct PropertyExpr {
    property_name: String,
}

impl PropertyExpr {
    /// Create a new property expression.
    #[inline]
    pub fn new(property_name: impl Into<String>) -> Self {
        Self {
            property_name: property_name.into(),
        }
    }
}

#[typetag::serde]
impl Expr for PropertyExpr {
    fn as_expr(&self) -> &dyn Expr {
        self
    }

    fn is_const(&self) -> bool {
        false
    }

    fn value_type(&self) -> ValueType {
        ValueType::Scalar(ScalarType::Bool) // FIXME - This is unknown until
                                            // properties are resolved with the
                                            // effect, when code is generated...
    }

    fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError> {
        if !context.property_layout().contains(&self.property_name) {
            return Err(ExprError::PropertyError(format!(
                "Unknown property '{}' in evaluation context.",
                self.property_name
            )));
        }

        Ok(self.to_wgsl_string())
    }

    fn boxed_clone(&self) -> BoxedExpr {
        Box::new(PropertyExpr {
            property_name: self.property_name.clone(),
        })
    }
}

impl ToWgslString for PropertyExpr {
    fn to_wgsl_string(&self) -> String {
        format!("properties.{}", self.property_name)
    }
}

impl From<String> for PropertyExpr {
    fn from(property_name: String) -> Self {
        PropertyExpr::new(property_name)
    }
}

/// Built-in operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub enum BuiltInOperator {
    /// Current effect system simulation time since startup, in seconds.
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
            BuiltInOperator::DeltaTime => "delta_time",
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
    // pub fn eval(&self, _context: &dyn EvalContext) -> Result<String, ExprError> {
    //     match self {
    //         BuiltInOperator::Time => Value::Scalar(Scal)
    //     }
    // }
}

impl ToWgslString for BuiltInOperator {
    fn to_wgsl_string(&self) -> String {
        format!("sim_params.{}", self.name())
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

    fn eval(&self, _context: &dyn EvalContext) -> Result<String, ExprError> {
        Ok(self.to_wgsl_string())
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

/// Unary numeric operator.
///
/// The operator can be used with any numeric type or vector of numeric types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub enum UnaryNumericOperator {
    /// Absolute value operator.
    Abs,
    /// Logical ALL operator for bool vectors.
    /// FIXME - This is not numeric...
    All,
    /// Logical ANY operator for bool vectors.
    /// FIXME - This is not numeric...
    Any,
    /// Vector normalizing operator.
    Normalize,
}

impl ToWgslString for UnaryNumericOperator {
    fn to_wgsl_string(&self) -> String {
        match *self {
            UnaryNumericOperator::Abs => "abs".to_string(),
            UnaryNumericOperator::All => "all".to_string(),
            UnaryNumericOperator::Any => "any".to_string(),
            UnaryNumericOperator::Normalize => "normalize".to_string(),
        }
    }
}

/// Unary numeric operation expression.
#[derive(Debug, Clone, Reflect, FromReflect, Serialize, Deserialize)]
pub struct UnaryNumericOpExpr {
    input: BoxedExpr,
    op: UnaryNumericOperator,
}

impl UnaryNumericOpExpr {
    /// Create a new unary numeric operation expression.
    #[inline]
    pub fn new(input: impl Into<BoxedExpr>, op: UnaryNumericOperator) -> Self {
        Self {
            input: input.into(),
            op,
        }
    }
}

#[typetag::serde]
impl Expr for UnaryNumericOpExpr {
    fn as_expr(&self) -> &dyn Expr {
        self
    }

    fn is_const(&self) -> bool {
        self.input.is_const()
    }

    fn value_type(&self) -> ValueType {
        match self.op {
            UnaryNumericOperator::Abs => self.input.value_type(),
            UnaryNumericOperator::All | UnaryNumericOperator::Any => {
                ValueType::Scalar(ScalarType::Bool)
            }
            UnaryNumericOperator::Normalize => ValueType::Scalar(ScalarType::Float),
        }
    }

    fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError> {
        let expr = self.input.eval(context);

        // if self.input.value_type() != self.value_type() {
        //     return Err(ExprError::TypeError(format!(
        //         "Cannot apply normalize() function to non-vector expression: {}",
        //         expr.unwrap_or("(error evaluating expression)".to_string())
        //     )));
        // }

        expr.map(|s| format!("{}({})", self.op.to_wgsl_string(), s))
    }

    fn boxed_clone(&self) -> BoxedExpr {
        Box::new(UnaryNumericOpExpr {
            input: self.input.boxed_clone(),
            op: self.op,
        })
    }
}

impl ToWgslString for UnaryNumericOpExpr {
    fn to_wgsl_string(&self) -> String {
        format!(
            "{}({})",
            self.op.to_wgsl_string(),
            self.input.to_wgsl_string()
        )
    }
}

/// Binary numeric operator.
///
/// The operator can be used with any numeric type or vector of numeric types
/// (component-wise).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub enum BinaryNumericOperator {
    /// Minimum operator.
    Min,
    /// Maximum operator.
    Max,
    /// Less-than operator.
    LessThan,
    /// Less-than-or-equal operator.
    LessThanOrEqual,
    /// Greater-than operator.
    GreaterThan,
    /// Greater-than-or-equal operator.
    GreaterThanOrEqual,
}

impl BinaryNumericOperator {
    /// Check if a binary operator is called via a functional-style call.
    ///
    /// Functional-style calls are in the form `op(lhs, rhs)` (like `min(a,
    /// b)`), while non-functional ones are in the form `lhs op rhs` (like `a +
    /// b`).
    pub fn is_functional(&self) -> bool {
        match *self {
            BinaryNumericOperator::Min | BinaryNumericOperator::Max => true,
            BinaryNumericOperator::LessThan
            | BinaryNumericOperator::LessThanOrEqual
            | BinaryNumericOperator::GreaterThan
            | BinaryNumericOperator::GreaterThanOrEqual => false,
        }
    }
}

impl ToWgslString for BinaryNumericOperator {
    fn to_wgsl_string(&self) -> String {
        match *self {
            BinaryNumericOperator::Min => "min".to_string(),
            BinaryNumericOperator::Max => "max".to_string(),
            BinaryNumericOperator::LessThan => "<".to_string(),
            BinaryNumericOperator::LessThanOrEqual => "<=".to_string(),
            BinaryNumericOperator::GreaterThan => ">".to_string(),
            BinaryNumericOperator::GreaterThanOrEqual => ">=".to_string(),
        }
    }
}

/// Binary numeric operation expression.
#[derive(Debug, Clone, Reflect, FromReflect, Serialize, Deserialize)]
pub struct BinaryNumericOpExpr {
    lhs: BoxedExpr,
    rhs: BoxedExpr,
    op: BinaryNumericOperator,
}

impl BinaryNumericOpExpr {
    /// Create a new binary numeric operation expression.
    #[inline]
    pub fn new(
        lhs: impl Into<BoxedExpr>,
        rhs: impl Into<BoxedExpr>,
        op: BinaryNumericOperator,
    ) -> Self {
        Self {
            lhs: lhs.into(),
            rhs: rhs.into(),
            op,
        }
    }
}

#[typetag::serde]
impl Expr for BinaryNumericOpExpr {
    fn as_expr(&self) -> &dyn Expr {
        self
    }

    fn is_const(&self) -> bool {
        self.lhs.is_const() && self.rhs.is_const()
    }

    fn value_type(&self) -> ValueType {
        self.lhs.value_type() // FIXME - need to handle casts
    }

    fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError> {
        let lhs = self.lhs.eval(context)?;
        let rhs = self.rhs.eval(context)?;

        // if !self.input.value_type().is_vector() {
        //     return Err(ExprError::TypeError(format!(
        //         "Cannot apply normalize() function to non-vector expression: {}",
        //         expr.unwrap_or("(error evaluating expression)".to_string())
        //     )));
        // }

        Ok(if self.op.is_functional() {
            format!("{}({}, {})", self.op.to_wgsl_string(), lhs, rhs)
        } else {
            format!("({}) {} ({})", lhs, self.op.to_wgsl_string(), rhs)
        })
    }

    fn boxed_clone(&self) -> BoxedExpr {
        Box::new(BinaryNumericOpExpr {
            lhs: self.lhs.boxed_clone(),
            rhs: self.rhs.boxed_clone(),
            op: self.op,
        })
    }
}

impl ToWgslString for BinaryNumericOpExpr {
    fn to_wgsl_string(&self) -> String {
        if self.op.is_functional() {
            format!(
                "{}({}, {})",
                self.op.to_wgsl_string(),
                self.lhs.to_wgsl_string(),
                self.rhs.to_wgsl_string()
            )
        } else {
            format!(
                "({}) {} ({})",
                self.lhs.to_wgsl_string(),
                self.op.to_wgsl_string(),
                self.rhs.to_wgsl_string()
            )
        }
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
impl_binary_ops!(UnaryNumericOpExpr);
impl_binary_ops!(BinaryNumericOpExpr);

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

    // #[test]
    // fn add_expr() {
    //     // f32 + f32
    //     let x: LiteralExpr = 3_f32.into();
    //     let y: LiteralExpr = 42_f32.into();
    //     let a = (x + y).eval();
    //     assert!(a.is_ok());
    //     let b = (y + x).eval();
    //     assert!(b.is_ok());
    //     assert_eq!(a, b);

    //     // Cannot Add bool
    //     let z: LiteralExpr = true.into();
    //     assert!((x + z).eval().is_err());
    //     assert!((z + x).eval().is_err());

    //     // Cannot Add a different scalar
    //     let z: LiteralExpr = 8_u32.into();
    //     assert!((x + z).eval().is_err());
    //     assert!((z + x).eval().is_err());

    //     // f32 + vec3<f32>
    //     let x: LiteralExpr = 3_f32.into();
    //     let y: LiteralExpr = Vec3::ONE.into();
    //     let a = (x + y).eval();
    //     assert!(a.is_ok());
    //     let b = (y + x).eval();
    //     assert!(b.is_ok());
    //     assert_eq!(a, b);
    //     assert!(matches!(a.unwrap(), Value::Vector(_)));
    // }

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

    // #[test]
    // fn serde() {
    //     let v = Value::Scalar(3.0_f32.into());
    //     let l: LiteralExpr = v.into();
    //     assert_eq!(Ok(v), l.eval());
    //     let s = ron::to_string(&l).unwrap();
    //     println!("literal: {:?}", s);
    //     let l_serde: LiteralExpr = ron::from_str(&s).unwrap();
    //     assert_eq!(l_serde, l);

    //     let b: BoxedExpr = Box::new(l);
    //     let s = ron::to_string(&b).unwrap();
    //     println!("boxed literal: {:?}", s);
    //     let b_serde: BoxedExpr = ron::from_str(&s).unwrap();
    //     assert!(b_serde.is_const());
    //     assert_eq!(b_serde.to_wgsl_string(), b.to_wgsl_string());

    //     let v0 = Value::Scalar(3.0_f32.into());
    //     let v1 = Value::Scalar(2.5_f32.into());
    //     let l0: LiteralExpr = v0.into();
    //     let l1: LiteralExpr = v1.into();
    //     let a = l0 + l1;
    //     assert!(a.is_const());
    //     assert_eq!(Ok(Value::Scalar(5.5_f32.into())), a.eval());
    //     let s = ron::to_string(&a).unwrap();
    //     println!("add: {:?}", s);
    //     let a_serde: AddExpr = ron::from_str(&s).unwrap();
    //     println!("a_serde: {:?}", a_serde);
    //     assert_eq!(a_serde.left.to_wgsl_string(), l0.to_wgsl_string());
    //     assert_eq!(a_serde.right.to_wgsl_string(), l1.to_wgsl_string());
    // }
}
