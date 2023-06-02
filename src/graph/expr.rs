use std::{cell::RefCell, cmp::Ordering, fmt, marker::PhantomData, num::NonZeroU32, rc::Rc};

use bevy::{
    reflect::{FromReflect, Reflect},
    utils::thiserror::Error,
};
use serde::{Deserialize, Serialize};

use crate::{Attribute, PropertyLayout, ScalarType, ToWgslString, ValueType};

use super::Value;

type Index = NonZeroU32;

#[derive(Reflect, FromReflect, Serialize, Deserialize)]
pub struct Handle<T: Send + Sync + 'static> {
    index: Index,
    #[serde(skip)]
    #[reflect(ignore)]
    marker: PhantomData<T>,
}

impl<T: Send + Sync + 'static> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle {
            index: self.index,
            marker: self.marker,
        }
    }
}

impl<T: Send + Sync + 'static> Copy for Handle<T> {}

impl<T: Send + Sync + 'static> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T: Send + Sync + 'static> Eq for Handle<T> {}

impl<T: Send + Sync + 'static> PartialOrd for Handle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.index.partial_cmp(&other.index)
    }
}

impl<T: Send + Sync + 'static> Ord for Handle<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index.cmp(&other.index)
    }
}

impl<T: Send + Sync + 'static> fmt::Debug for Handle<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "[{}]", self.index)
    }
}

impl<T: Send + Sync + 'static> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        self.index.hash(hasher)
    }
}

impl<T: Send + Sync + 'static> Handle<T> {
    fn new(index: Index) -> Handle<T> {
        Handle {
            index,
            marker: PhantomData,
        }
    }

    fn index(&self) -> usize {
        (self.index.get() - 1) as usize
    }
}

/// Handle of an expression inside a [`Module`].
pub type ExprHandle = Handle<Expr>;

/// Container for expressions.
#[derive(Debug, Default, Clone, PartialEq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub struct Module {
    expressions: Vec<Expr>,
}

impl Module {
    /// Create a new module from an existing collection of expressions.
    pub fn from_raw(expr: Vec<Expr>) -> Self {
        Self { expressions: expr }
    }

    /// Append a new expression to the module.
    pub fn push(&mut self, expr: impl Into<Expr>) -> ExprHandle {
        #[allow(unsafe_code)]
        let index: Index = unsafe { NonZeroU32::new_unchecked(self.expressions.len() as u32 + 1) };
        self.expressions.push(expr.into());
        Handle::new(index)
    }

    /// Build a literal expression and append it to the module.
    #[inline]
    pub fn lit<V>(&mut self, value: V) -> ExprHandle
    where
        Value: From<V>,
    {
        self.push(Expr::Literal(LiteralExpr::new(value)))
    }

    /// Build an attribute expression and append it to the module.
    #[inline]
    pub fn attr(&mut self, attr: Attribute) -> ExprHandle {
        self.push(Expr::Attribute(AttributeExpr::new(attr)))
    }

    /// Build a built-in expression and append it to the module.
    #[inline]
    pub fn builtin(&mut self, op: BuiltInOperator) -> ExprHandle {
        self.push(Expr::BuiltIn(BuiltInExpr::new(op)))
    }

    /// Build a unary expression and append it to the module.
    #[inline]
    pub fn unary(&mut self, op: UnaryNumericOperator, inner: ExprHandle) -> ExprHandle {
        assert!(inner.index() < self.expressions.len());
        self.push(Expr::Unary { op, expr: inner })
    }

    /// Build an `abs()` unary expression and append it to the module.
    #[inline]
    pub fn abs(&mut self, inner: ExprHandle) -> ExprHandle {
        self.unary(UnaryNumericOperator::Abs, inner)
    }

    /// Build an `all()` unary expression and append it to the module.
    #[inline]
    pub fn all(&mut self, inner: ExprHandle) -> ExprHandle {
        self.unary(UnaryNumericOperator::All, inner)
    }

    /// Build an `any()` unary expression and append it to the module.
    #[inline]
    pub fn any(&mut self, inner: ExprHandle) -> ExprHandle {
        self.unary(UnaryNumericOperator::Any, inner)
    }

    /// Build a binary expression and append it to the module.
    #[inline]
    pub fn binary(
        &mut self,
        op: BinaryOperator,
        left: ExprHandle,
        right: ExprHandle,
    ) -> ExprHandle {
        assert!(left.index() < self.expressions.len());
        assert!(right.index() < self.expressions.len());
        self.push(Expr::Binary { op, left, right })
    }

    /// Build an `add()` binary expression and append it to the module.
    #[inline]
    pub fn add(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::Add, left, right)
    }

    /// Build a `sub()` binary expression and append it to the module.
    #[inline]
    pub fn sub(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::Sub, left, right)
    }

    /// Build a `mul()` binary expression and append it to the module.
    #[inline]
    pub fn mul(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::Mul, left, right)
    }

    /// Build a `div()` binary expression and append it to the module.
    #[inline]
    pub fn div(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::Div, left, right)
    }

    /// Build a `min()` binary expression and append it to the module.
    #[inline]
    pub fn min(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::Min, left, right)
    }

    /// Build a `max()` binary expression and append it to the module.
    #[inline]
    pub fn max(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::Max, left, right)
    }

    /// Build a less-than binary expression and append it to the module.
    #[inline]
    pub fn lt(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::LessThan, left, right)
    }

    /// Build a less-than-or-equal binary expression and append it to the module.
    #[inline]
    pub fn le(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::LessThanOrEqual, left, right)
    }

    /// Build a greater-than binary expression and append it to the module.
    #[inline]
    pub fn gt(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::GreaterThan, left, right)
    }

    /// Build a greater-than-or-equal binary expression and append it to the module.
    #[inline]
    pub fn ge(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::GreaterThanOrEqual, left, right)
    }

    /// Build a `uniform()` binary expression and append it to the module.
    #[inline]
    pub fn uniform(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::UniformRand, left, right)
    }

    /// Get an existing expression.
    #[inline]
    pub fn get(&self, expr: ExprHandle) -> Option<&Expr> {
        let index = expr.index();
        self.expressions.get(index)
    }

    /// Get an existing expression.
    #[inline]
    pub fn get_mut(&mut self, expr: ExprHandle) -> Option<&mut Expr> {
        let index = expr.index();
        self.expressions.get_mut(index)
    }

    /// Is the expression resulting in a compile-time constant which can be
    /// hard-coded into a shader's code?
    #[inline]
    pub fn is_const(&self, expr: ExprHandle) -> bool {
        let expr = self.get(expr).unwrap();
        expr.is_const(self)
    }
}

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
    /// Get the module the evaluation is taking place in.
    fn module(&self) -> &Module;

    /// Get the property layout of the effect.
    fn property_layout(&self) -> &PropertyLayout;

    /// Resolve an expression handle its the underlying expression.
    fn expr(&self, handle: Handle<Expr>) -> Result<&Expr, ExprError>;

    /// Evaluate an expression.
    fn eval(&self, handle: Handle<Expr>) -> Result<String, ExprError>;
}

/// Language expression producing a value.
#[derive(Debug, Clone, PartialEq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub enum Expr {
    /// Built-in expression (`BuiltInExpr`).
    BuiltIn(BuiltInExpr),
    /// Literal expression (`LiteralExpr`).
    Literal(LiteralExpr),
    /// Property expression (`PropertyExpr`).
    Property(PropertyExpr),
    /// Attribute expression (`AttributeExpr`).
    Attribute(AttributeExpr),
    /// Unary operation expression.
    Unary {
        /// Unary operator.
        op: UnaryNumericOperator,
        /// Operand the unary operation applies to.
        expr: Handle<Expr>,
    },
    /// Binary operation expression.
    Binary {
        /// Binary operator.
        op: BinaryOperator,
        /// Left-hand side operand the binary operation applies to.
        left: Handle<Expr>,
        /// Right-hand side operand the binary operation applies to.
        right: Handle<Expr>,
    },
}

impl Expr {
    /// Is the expression resulting in a compile-time constant which can be
    /// hard-coded into a shader's code?
    pub fn is_const(&self, module: &Module) -> bool {
        match self {
            Expr::BuiltIn(expr) => expr.is_const(),
            Expr::Literal(expr) => expr.is_const(),
            Expr::Property(expr) => expr.is_const(),
            Expr::Attribute(expr) => expr.is_const(),
            Expr::Unary { expr, .. } => module.is_const(*expr),
            Expr::Binary { left, right, .. } => module.is_const(*left) && module.is_const(*right),
        }
    }

    /// The type of the value produced by the expression.
    pub fn value_type(&self) -> ValueType {
        unimplemented!()
    }

    /// Evaluate the expression.
    pub fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError> {
        match self {
            Expr::BuiltIn(expr) => expr.eval(context),
            Expr::Literal(expr) => expr.eval(context),
            Expr::Property(expr) => expr.eval(context),
            Expr::Attribute(expr) => expr.eval(context),
            Expr::Unary { op, expr } => {
                let expr = context.expr(*expr)?.eval(context);

                // if expr.value_type() != self.value_type() {
                //     return Err(ExprError::TypeError(format!(
                //         "Cannot apply normalize() function to non-vector expression: {}",
                //         expr.unwrap_or("(error evaluating expression)".to_string())
                //     )));
                // }

                expr.map(|s| format!("{}({})", op.to_wgsl_string(), s))
            }
            Expr::Binary { op, left, right } => {
                let left = context.expr(*left)?.eval(context)?;
                let right = context.expr(*right)?.eval(context)?;

                // if !self.input.value_type().is_vector() {
                //     return Err(ExprError::TypeError(format!(
                //         "Cannot apply normalize() function to non-vector expression: {}",
                //         expr.unwrap_or("(error evaluating expression)".to_string())
                //     )));
                // }

                Ok(if op.is_functional() {
                    format!("{}({}, {})", op.to_wgsl_string(), left, right)
                } else {
                    format!("({}) {} ({})", left, op.to_wgsl_string(), right)
                })
            }
        }
    }
}

// impl ToWgslString for Expr {
//     fn to_wgsl_string(&self) -> String {
//         match self {
//             Expr::BuiltIn(_) => unimplemented!(),
//             Expr::Literal(lit) => lit.to_wgsl_string(),
//             Expr::Property(prop) => prop.to_wgsl_string(),
//             Expr::Attribute(attr) => attr.to_wgsl_string(),
//             Expr::Unary { .. } => unimplemented!(),
//             Expr::Binary { .. } => unimplemented!(),
//         }
//     }
// }

/// A literal constant expression like `3.0` or `vec3<f32>(1.0, 2.0, 3.0)`.
#[derive(Debug, Clone, Copy, PartialEq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
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

    /// Is the expression resulting in a compile-time constant which can be
    /// hard-coded into a shader's code?
    pub fn is_const(&self) -> bool {
        true
    }

    /// Get the value type of the expression.
    pub fn value_type(&self) -> ValueType {
        self.value.value_type()
    }

    /// Evaluate the expression in the given context.
    pub fn eval(&self, _context: &dyn EvalContext) -> Result<String, ExprError> {
        Ok(self.value.to_wgsl_string())
    }
}

impl ToWgslString for LiteralExpr {
    fn to_wgsl_string(&self) -> String {
        self.value.to_wgsl_string()
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

    /// Is the expression resulting in a compile-time constant which can be
    /// hard-coded into a shader's code?
    pub fn is_const(&self) -> bool {
        false
    }

    /// Get the value type of the expression.
    pub fn value_type(&self) -> ValueType {
        self.attr.value_type()
    }

    /// Evaluate the expression in the given context.
    pub fn eval(&self, _context: &dyn EvalContext) -> Result<String, ExprError> {
        Ok(self.to_wgsl_string())
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

    /// Is the expression resulting in a compile-time constant which can be
    /// hard-coded into a shader's code?
    fn is_const(&self) -> bool {
        false
    }

    /// Get the value type of the expression.
    #[allow(dead_code)]
    fn value_type(&self) -> ValueType {
        ValueType::Scalar(ScalarType::Bool) // FIXME - This is unknown until
                                            // properties are resolved with the
                                            // effect, when code is generated...
    }

    /// Evaluate the expression in the given context.
    fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError> {
        if !context.property_layout().contains(&self.property_name) {
            return Err(ExprError::PropertyError(format!(
                "Unknown property '{}' in evaluation context.",
                self.property_name
            )));
        }

        Ok(self.to_wgsl_string())
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

    /// Is the expression resulting in a compile-time constant which can be
    /// hard-coded into a shader's code?
    pub fn is_const(&self) -> bool {
        false
    }

    /// Get the value type of the expression.
    pub fn value_type(&self) -> ValueType {
        self.operator.value_type()
    }

    /// Evaluate the expression in the given context.
    pub fn eval(&self, _context: &dyn EvalContext) -> Result<String, ExprError> {
        Ok(self.to_wgsl_string())
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

/// Binary numeric operator.
///
/// The operator can be used with any numeric type or vector of numeric types
/// (component-wise).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub enum BinaryOperator {
    /// Addition operator.
    Add,
    /// Subtraction operator.
    Sub,
    /// Multiply operator.
    Mul,
    /// Division operator.
    Div,
    /// Less-than operator.
    LessThan,
    /// Less-than-or-equal operator.
    LessThanOrEqual,
    /// Greater-than operator.
    GreaterThan,
    /// Greater-than-or-equal operator.
    GreaterThanOrEqual,
    /// Minimum operator.
    Min,
    /// Maximum operator.
    Max,
    /// Uniform random number operator.
    UniformRand,
}

impl BinaryOperator {
    /// Check if a binary operator is called via a functional-style call.
    ///
    /// Functional-style calls are in the form `op(lhs, rhs)` (like `min(a,
    /// b)`), while non-functional ones are in the form `lhs op rhs` (like `a +
    /// b`).
    pub fn is_functional(&self) -> bool {
        match *self {
            BinaryOperator::Add
            | BinaryOperator::Sub
            | BinaryOperator::Mul
            | BinaryOperator::Div
            | BinaryOperator::LessThan
            | BinaryOperator::LessThanOrEqual
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterThanOrEqual => false,
            BinaryOperator::Min | BinaryOperator::Max | BinaryOperator::UniformRand => true,
        }
    }
}

impl ToWgslString for BinaryOperator {
    fn to_wgsl_string(&self) -> String {
        match *self {
            BinaryOperator::Add => "+".to_string(),
            BinaryOperator::Sub => "-".to_string(),
            BinaryOperator::Mul => "*".to_string(),
            BinaryOperator::Div => "/".to_string(),
            BinaryOperator::LessThan => "<".to_string(),
            BinaryOperator::LessThanOrEqual => "<=".to_string(),
            BinaryOperator::GreaterThan => ">".to_string(),
            BinaryOperator::GreaterThanOrEqual => ">=".to_string(),
            BinaryOperator::Min => "min".to_string(),
            BinaryOperator::Max => "max".to_string(),
            BinaryOperator::UniformRand => "rand_uniform".to_string(),
        }
    }
}

/// Expression writer.
///
/// Utility to write expressions with a simple functional syntax.
///
/// # Example
///
/// ```
/// type W = ExprWriter;
/// let w = (W::lit(5.) + W::attr(Attribute::POSITION)).max(W::prop("my_prop"));
/// let expr = w.expr();
/// assert_eq!(expr.to_wgsl_string(), "max((5.) + (particle.position), properties.my_prop)");
/// ```
#[derive(Debug)]
pub struct ExprWriter {
    module: Rc<RefCell<Module>>,
}

#[allow(dead_code)]
impl ExprWriter {
    /// Create a new writer.
    pub fn new() -> Self {
        Self {
            module: Rc::new(RefCell::new(Module::default())),
        }
    }

    /// Create a new writer from an existing module.
    pub fn from_module(module: Rc<RefCell<Module>>) -> Self {
        Self { module }
    }

    /// Push a new expression into the writer.
    pub fn push(&self, expr: impl Into<Expr>) -> WriterExpr {
        let expr = {
            let mut m = self.module.borrow_mut();
            m.push(expr.into())
        };
        WriterExpr {
            expr,
            module: Rc::clone(&self.module),
        }
    }

    /// Create a new writer from a literal constant.
    pub fn lit(&self, value: impl Into<Value>) -> WriterExpr {
        self.push(Expr::Literal(LiteralExpr {
            value: value.into(),
        }))
    }

    /// Create a new writer from an attribute expression.
    pub fn attr(&self, attr: Attribute) -> WriterExpr {
        self.push(Expr::Attribute(AttributeExpr::new(attr)))
    }

    /// Create a new writer from a property expression.
    pub fn prop(&self, name: impl Into<String>) -> WriterExpr {
        self.push(Expr::Property(PropertyExpr::new(name)))
    }

    /// Finish
    pub fn finish(self) -> Module {
        self.module.take()
    }
}

/// Intermediate expression from an [`ExprWriter`].
#[derive(Debug)]
pub struct WriterExpr {
    expr: Handle<Expr>,
    module: Rc<RefCell<Module>>,
}

impl WriterExpr {
    fn unary_op(self, op: UnaryNumericOperator) -> Self {
        let expr = self.module.borrow_mut().push(Expr::Unary {
            op,
            expr: self.expr,
        });
        WriterExpr {
            expr,
            module: self.module,
        }
    }

    /// Take the absolute value of the current expression.
    pub fn abs(self) -> Self {
        self.unary_op(UnaryNumericOperator::Abs)
    }

    /// Apply the logical operator "all" to the current bool vector expression.
    pub fn all(self) -> Self {
        self.unary_op(UnaryNumericOperator::All)
    }

    /// Apply the logical operator "any" to the current bool vector expression.
    pub fn any(self) -> Self {
        self.unary_op(UnaryNumericOperator::Any)
    }

    fn binary_op(self, other: Self, op: BinaryOperator) -> Self {
        assert_eq!(self.module, other.module);
        let left = self.expr;
        let right = other.expr;
        let expr = self
            .module
            .borrow_mut()
            .push(Expr::Binary { op, left, right });
        WriterExpr {
            expr,
            module: self.module,
        }
    }

    /// Take the minimum value of the current expression and another expression.
    pub fn min(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Min)
    }

    /// Take the maximum value of the current expression and another expression.
    pub fn max(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Max)
    }

    /// Add the current expression with another expression.
    pub fn add(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Add)
    }

    /// Subtract another expression from the current expression.
    pub fn sub(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Sub)
    }

    /// Multiply the current expression with another expression.
    pub fn mul(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Mul)
    }

    /// Divide the current expression by another expression.
    pub fn div(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Div)
    }

    /// Apply the logical operator "less than or equal" to this expression and another expression.
    pub fn le(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::LessThanOrEqual)
    }

    /// Apply the logical operator "less than" to this expression and another expression.
    pub fn lt(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::LessThan)
    }

    /// Apply the logical operator "greater than or equal" to this expression and another expression.
    pub fn ge(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::GreaterThanOrEqual)
    }

    /// Apply the logical operator "greater than" to this expression and another expression.
    pub fn gt(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::GreaterThan)
    }

    /// Apply the logical operator "uniform" to this expression and another expression.
    pub fn uniform(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::UniformRand)
    }

    /// Finalize the writer and return the accumulated expression.
    pub fn expr(self) -> Handle<Expr> {
        self.expr
    }
}

// impl std::ops::Add<ExprWriter> for ExprWriter {
//     type Output = ExprWriter;

//     fn add(self, mut rhs: ExprWriter) -> Self::Output {
//         self.add(&mut rhs)
//     }
// }

// impl std::ops::Sub<ExprWriter> for ExprWriter {
//     type Output = ExprWriter;

//     fn sub(self, mut rhs: ExprWriter) -> Self::Output {
//         self.sub(&mut rhs)
//     }
// }

// impl std::ops::Mul<ExprWriter> for ExprWriter {
//     type Output = ExprWriter;

//     fn mul(self, mut rhs: ExprWriter) -> Self::Output {
//         self.mul(&mut rhs)
//     }
// }

// impl std::ops::Mul<&mut ExprWriter> for ExprWriter {
//     type Output = ExprWriter;

//     fn mul(self, rhs: &mut ExprWriter) -> Self::Output {
//         self.mul(rhs)
//     }
// }

// impl std::ops::Mul<&mut ExprWriter> for &mut ExprWriter {
//     type Output = ExprWriter;

//     fn mul(self, rhs: &mut ExprWriter) -> Self::Output {
//         self.mul(rhs)
//     }
// }

// impl std::ops::Div<ExprWriter> for ExprWriter {
//     type Output = ExprWriter;

//     fn div(self, mut rhs: ExprWriter) -> Self::Output {
//         self.div(&mut rhs)
//     }
// }

impl std::ops::Add<WriterExpr> for WriterExpr {
    type Output = WriterExpr;

    fn add(self, rhs: WriterExpr) -> Self::Output {
        self.add(rhs)
    }
}

impl std::ops::Sub<WriterExpr> for WriterExpr {
    type Output = WriterExpr;

    fn sub(self, rhs: WriterExpr) -> Self::Output {
        self.sub(rhs)
    }
}

impl std::ops::Mul<WriterExpr> for WriterExpr {
    type Output = WriterExpr;

    fn mul(self, rhs: WriterExpr) -> Self::Output {
        self.mul(rhs)
    }
}

impl std::ops::Div<WriterExpr> for WriterExpr {
    type Output = WriterExpr;

    fn div(self, rhs: WriterExpr) -> Self::Output {
        self.div(rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::Property, InitContext};

    use super::*;
    use bevy::math::Vec2;

    #[test]
    fn writer() {
        // Get a module and its writer
        let w = ExprWriter::new();

        // Build some expression
        let x = w.lit(3.).abs().max(w.attr(Attribute::POSITION) * w.lit(2.))
            + w.lit(-4.).min(w.prop("my_prop"));
        let x = x.expr();

        // Create an evaluation context
        let property_layout = PropertyLayout::new(&[Property::new(
            "my_prop",
            Value::Scalar(crate::ScalarValue::Float(3.)),
        )]);
        let mut m = w.finish();
        let context = InitContext::new(&mut m, &property_layout);

        // Evaluate the expression
        let s = context.expr(x).unwrap().eval(&context).unwrap();
        assert_eq!(
            "(max(abs(3.), (particle.position) * (2.))) + (min(-4., properties.my_prop))"
                .to_string(),
            s
        );
    }

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

    // #[test]
    // fn math_expr() {
    //     let x: AttributeExpr = Attribute::POSITION.into();
    //     let y = LiteralExpr::new(Vec3::ONE);

    //     let a = x + y;
    //     assert_eq!(
    //         a.to_wgsl_string(),
    //         format!(
    //             "(particle.{}) + (vec3<f32>(1.,1.,1.))",
    //             Attribute::POSITION.name()
    //         )
    //     );

    //     let b = y + x;
    //     assert_eq!(
    //         b.to_wgsl_string(),
    //         format!(
    //             "(vec3<f32>(1.,1.,1.)) + (particle.{})",
    //             Attribute::POSITION.name()
    //         )
    //     );

    //     let a = x - y;
    //     assert_eq!(
    //         a.to_wgsl_string(),
    //         format!(
    //             "(particle.{}) - (vec3<f32>(1.,1.,1.))",
    //             Attribute::POSITION.name()
    //         )
    //     );

    //     let b = y - x;
    //     assert_eq!(
    //         b.to_wgsl_string(),
    //         format!(
    //             "(vec3<f32>(1.,1.,1.)) - (particle.{})",
    //             Attribute::POSITION.name()
    //         )
    //     );

    //     let a = x * y;
    //     assert_eq!(
    //         a.to_wgsl_string(),
    //         format!(
    //             "(particle.{}) * (vec3<f32>(1.,1.,1.))",
    //             Attribute::POSITION.name()
    //         )
    //     );

    //     let b = y * x;
    //     assert_eq!(
    //         b.to_wgsl_string(),
    //         format!(
    //             "(vec3<f32>(1.,1.,1.)) * (particle.{})",
    //             Attribute::POSITION.name()
    //         )
    //     );

    //     let a = x / y;
    //     assert_eq!(
    //         a.to_wgsl_string(),
    //         format!(
    //             "(particle.{}) / (vec3<f32>(1.,1.,1.))",
    //             Attribute::POSITION.name()
    //         )
    //     );

    //     let b = y / x;
    //     assert_eq!(
    //         b.to_wgsl_string(),
    //         format!(
    //             "(vec3<f32>(1.,1.,1.)) / (particle.{})",
    //             Attribute::POSITION.name()
    //         )
    //     );
    // }

    // #[test]
    // fn serde() {
    //     let v = Value::Scalar(3.0_f32.into());
    //     let l: LiteralExpr = v.into();
    //     assert_eq!(Ok(v), l.eval());
    //     let s = ron::to_string(&l).unwrap();
    //     println!("literal: {:?}", s);
    //     let l_serde: LiteralExpr = ron::from_str(&s).unwrap();
    //     assert_eq!(l_serde, l);

    //     let b: Handle<Expr> = Box::new(l);
    //     let s = ron::to_string(&b).unwrap();
    //     println!("boxed literal: {:?}", s);
    //     let b_serde: Handle<Expr> = ron::from_str(&s).unwrap();
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
