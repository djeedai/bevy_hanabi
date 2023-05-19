use std::{cmp::Ordering, fmt, marker::PhantomData, num::NonZeroU32};

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

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Module {
    expressions: Vec<Expr>,
}

impl Module {
    #[allow(unsafe_code)]
    pub fn push(&mut self, expr: Expr) -> Handle<Expr> {
        let index = unsafe { NonZeroU32::new_unchecked(self.expressions.len() as u32 + 1) };
        self.expressions.push(expr);
        Handle::new(index)
    }

    #[inline]
    pub fn get(&self, expr: Handle<Expr>) -> Option<&Expr> {
        let index = expr.index();
        self.expressions.get(index)
    }

    #[inline]
    pub fn get_mut(&mut self, expr: Handle<Expr>) -> Option<&mut Expr> {
        let index = expr.index();
        self.expressions.get_mut(index)
    }

    #[inline]
    pub fn is_const(&self, expr: Handle<Expr>) -> bool {
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
    BuiltIn(BuiltInExpr),
    Literal(LiteralExpr),
    Property(PropertyExpr),
    Attribute(AttributeExpr),
    Unary {
        op: UnaryNumericOperator,
        expr: Handle<Expr>,
    },
    Binary {
        op: BinaryOperator,
        left: Handle<Expr>,
        right: Handle<Expr>,
    },
}

impl Expr {
    /// Is the expression resulting in a compile-time constant which can be
    /// hard-coded into a shader's code?
    pub fn is_const(&self, module: &Module) -> bool {
        match *self {
            Expr::BuiltIn(expr) => expr.is_const(),
            Expr::Literal(expr) => expr.is_const(),
            Expr::Property(expr) => expr.is_const(),
            Expr::Attribute(expr) => expr.is_const(),
            Expr::Unary { op, expr } => module.is_const(expr),
            Expr::Binary { op, left, right } => module.is_const(left) && module.is_const(right),
        }
    }

    /// The type of the value produced by the expression.
    pub fn value_type(&self) -> ValueType {
        unimplemented!()
    }

    /// Evaluate the expression.
    pub fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError> {
        match *self {
            Expr::BuiltIn(expr) => expr.eval(context),
            Expr::Literal(expr) => expr.eval(context),
            Expr::Property(expr) => expr.eval(context),
            Expr::Attribute(expr) => expr.eval(context),
            Expr::Unary { op, expr } => {
                let expr = context.expr(expr)?.eval(context);

                // if expr.value_type() != self.value_type() {
                //     return Err(ExprError::TypeError(format!(
                //         "Cannot apply normalize() function to non-vector expression: {}",
                //         expr.unwrap_or("(error evaluating expression)".to_string())
                //     )));
                // }

                expr.map(|s| format!("{}({})", op.to_wgsl_string(), s))
            }
            Expr::Binary { op, left, right } => {
                let left = context.expr(left)?.eval(context)?;
                let right = context.expr(right)?.eval(context)?;

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

impl ToWgslString for Expr {
    fn to_wgsl_string(&self) -> String {
        match *self {
            Expr::BuiltIn(_) => unimplemented!(),
            Expr::Literal(lit) => lit.to_wgsl_string(),
            Expr::Property(prop) => prop.to_wgsl_string(),
            Expr::Attribute(attr) => attr.to_wgsl_string(),
            Expr::Unary { op, expr } => unimplemented!(),
            Expr::Binary { op, left, right } => unimplemented!(),
        }
    }
}

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

    pub fn is_const(&self) -> bool {
        true
    }

    pub fn value_type(&self) -> ValueType {
        self.value.value_type()
    }

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

    pub fn is_const(&self) -> bool {
        false
    }

    pub fn value_type(&self) -> ValueType {
        self.attr.value_type()
    }

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

    pub fn is_const(&self) -> bool {
        false
    }

    pub fn value_type(&self) -> ValueType {
        self.operator.value_type()
    }

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
            BinaryOperator::Min | BinaryOperator::Max => true,
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
pub struct ExprWriter<'m> {
    module: &'m Module,
}

#[derive(Debug)]
pub struct WriterExpr<'m, 'w> {
    expr: Handle<Expr>,
    writer: &'w mut ExprWriter<'m>,
}

#[allow(dead_code)]
impl<'m> ExprWriter<'m> {
    /// Create a new writer starting from any generic expression.
    pub fn new(module: &'m Module) -> Self {
        Self { module }
    }

    /// Create a new writer from a literal constant.
    pub fn lit<'w>(&'w mut self, value: impl Into<Value>) -> WriterExpr<'m, 'w> {
        let expr = self.module.push(Expr::Literal(LiteralExpr {
            value: value.into(),
        }));
        WriterExpr { expr, writer: self }
    }

    /// Create a new writer from an attribute expression.
    pub fn attr<'w>(&'w mut self, attr: Attribute) -> WriterExpr<'m, 'w> {
        let expr = self.module.push(Expr::Attribute(AttributeExpr::new(attr)));
        WriterExpr { expr, writer: self }
    }

    /// Create a new writer from a property expression.
    pub fn prop<'w>(&'w mut self, name: impl Into<String>) -> WriterExpr<'m, 'w> {
        let expr = self.module.push(Expr::Property(PropertyExpr::new(name)));
        WriterExpr { expr, writer: self }
    }
}

impl<'m, 'w> WriterExpr<'m, 'w> {
    fn unary_op(mut self, op: UnaryNumericOperator) -> Self {
        let expr = self.writer.module.push(Expr::Unary {
            op,
            expr: self.expr,
        });
        self
    }

    /// Take the absolute value of the current expression.
    pub fn abs(mut self) -> Self {
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

    fn binary_op(mut self, other: &mut Self, op: BinaryOperator) -> Self {
        assert_eq!(self.writer, other.writer);
        let left = self.expr;
        let right = other.expr;
        let expr = self.writer.module.push(Expr::Binary { op, left, right });
        self
    }

    /// Take the minimum value of the current expression and another expression.
    pub fn min(mut self, other: &mut Self) -> Self {
        self.binary_op(other, BinaryOperator::Min)
    }

    /// Take the maximum value of the current expression and another expression.
    pub fn max(mut self, other: &mut Self) -> Self {
        self.binary_op(other, BinaryOperator::Max)
    }

    /// Add the current expression with another expression.
    pub fn add(mut self, other: &mut Self) -> Self {
        self.binary_op(other, BinaryOperator::Add)
    }

    /// Subtract another expression from the current expression.
    pub fn sub(mut self, other: &mut Self) -> Self {
        self.binary_op(other, BinaryOperator::Sub)
    }

    /// Multiply the current expression with another expression.
    pub fn mul(mut self, other: &mut Self) -> Self {
        self.binary_op(other, BinaryOperator::Mul)
    }

    /// Divide the current expression by another expression.
    pub fn div(mut self, other: &mut Self) -> Self {
        self.binary_op(other, BinaryOperator::Div)
    }

    /// Apply the logical operator "less than or equal" to this expression and another expression.
    pub fn le(mut self, other: &mut Self) -> Self {
        self.binary_op(other, BinaryOperator::LessThanOrEqual)
    }

    /// Apply the logical operator "less than" to this expression and another expression.
    pub fn lt(mut self, other: &mut Self) -> Self {
        self.binary_op(other, BinaryOperator::LessThan)
    }

    /// Apply the logical operator "greater than or equal" to this expression and another expression.
    pub fn ge(mut self, other: &mut Self) -> Self {
        self.binary_op(other, BinaryOperator::GreaterThanOrEqual)
    }

    /// Apply the logical operator "greater than" to this expression and another expression.
    pub fn gt(mut self, other: &mut Self) -> Self {
        self.binary_op(other, BinaryOperator::GreaterThan)
    }

    /// Finalize the writer and return the accumulated expression.
    pub fn expr(mut self) -> Handle<Expr> {
        assert_eq!(1, self.stack.len());
        self.stack.pop().unwrap()
    }
}

impl<'m> std::ops::Add<ExprWriter<'m>> for ExprWriter<'m> {
    type Output = ExprWriter<'m>;

    fn add(self, mut rhs: ExprWriter) -> Self::Output {
        self.add(&mut rhs)
    }
}

impl<'m> std::ops::Sub<ExprWriter<'m>> for ExprWriter<'m> {
    type Output = ExprWriter<'m>;

    fn sub(self, mut rhs: ExprWriter) -> Self::Output {
        self.sub(&mut rhs)
    }
}

impl<'m> std::ops::Mul<ExprWriter<'m>> for ExprWriter<'m> {
    type Output = ExprWriter<'m>;

    fn mul(self, mut rhs: ExprWriter) -> Self::Output {
        self.mul(&mut rhs)
    }
}

impl<'m> std::ops::Mul<&mut ExprWriter<'m>> for ExprWriter<'m> {
    type Output = ExprWriter<'m>;

    fn mul(self, rhs: &mut ExprWriter) -> Self::Output {
        self.mul(rhs)
    }
}

impl<'m> std::ops::Mul<&mut ExprWriter<'m>> for &mut ExprWriter<'m> {
    type Output = ExprWriter<'m>;

    fn mul(self, rhs: &mut ExprWriter) -> Self::Output {
        self.mul(rhs)
    }
}

impl<'m> std::ops::Div<ExprWriter<'m>> for ExprWriter<'m> {
    type Output = ExprWriter<'m>;

    fn div(self, mut rhs: ExprWriter) -> Self::Output {
        self.div(&mut rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::{Vec2, Vec3};

    #[test]
    fn writer() {
        let mut m = Module::default();
        let w = ExprWriter::new(&m);
        let w = w.lit(3.).abs().max(w.attr(Attribute::POSITION) * w.lit(2.))
            + w.lit(-4.).min(w.prop("my_prop"));
        let x = w.expr();
        let s = m.get(x).unwrap().to_wgsl_string();
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
