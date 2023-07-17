//! Expression API
//!
//! This module contains the low-level _Expression API_, designed to produce
//! highly customizable modifier behaviors through a code-first API focused on
//! runtime and serialization. For asset editing, the higher-level [Node API]
//! offers an easier-to-use abstraction built on top of the Expression API.
//!
//! # Modules and expressions
//!
//! A particle effect is composed of a series [`Modifier`]s decribing how to
//! initialize and update (simulate) the particles of the effect. Choosing which
//! modifier to add to an effect provides the user some limited level of
//! customizing. However modifiers alone cannot provide enough customizing to
//! build visual effects. For this reason, modifier inputs can be further
//! customized with _expressions_. An expression produces a value which is
//! assigned to the input. That value can be constant, in which case it will be
//! hard-coded into the generated WGSL shader, for performance reasons.
//! Alternatively, that value can vary based on other quantities, like an effect
//! property, a particle attribute, or some built-in simulation variable like
//! the simulation time.
//!
//! An expression is represented by the [`Expr`] enum. Expressions can be
//! combined together to form more complex expression; for example, the Add
//! expression computes the sum between two other expressions. [`Expr`]
//! represents a form of abstraction over the actual WGSL shader code, and is
//! generally closely related to the actual expressions of the WGSL language
//! itself.
//!
//! An expression often refers to other expressions. However, [`Expr`] as an
//! enum cannot directly contain other [`Expr`], otherwise the type would become
//! infinitely recursive. Instead, each expression is stored into a [`Module`]
//! and indexed by an [`ExprHandle`], a non-zero index referencing the
//! expression inside the module. This indirection avoids the recursion issue.
//! This means all expressions are implicitly associated with a unique module,
//! and care must be taken to not mix exressions from different modules.
//!
//! Each [`EffectAsset`] contains a single [`Module`] storing all the [`Expr`]
//! used in all its modifiers.
//!
//! # Kinds of expressions
//!
//! Expressions can be grouped into various kinds, for the sake of
//! comprehension:
//! - Literal expressions represent a constant, which will be hard-coded into
//!   the final WGSL shader code. Expressions like `1.42` or `vec3<f32>(0.)` are
//!   literal expressions in WGSL, and are represented by a [`LiteralExpr`].
//! - Built-in expressions represent specific built-in values provided by the
//!   simulation context. For example, the current simulation time is a built-in
//!   expression accessible from the shader code of any visual effect to animate
//!   it. A built-in expression is represented by a [`BuiltInExpr`].
//! - Attribute expressions represent the value of an attribute of a particle. A
//!   typical example is the particle position, represented by
//!   [`Attribute::POSITION`], which can be obtained as an expression through an
//!   [`AttributeExpr`].
//! - Property expressions represent the value of a visual effect property, a
//!   quantity assigned by the user on the CPU side and uploaded each frame into
//!   the GPU for precise per-frame control over an effect. It's represented by
//!   a [`PropertyExpr`].
//! - Unary and binary operations are expressions taking one or two operand
//!   expressions and transforming them. A typical example is the Add operator,
//!   which takes two operand expressions and produces their sum.
//!
//! # Building expressions
//!
//! The fundamental way to build expressions is to directly write them into a
//! [`Module`] itself. The [`Module`] type contains various methods to create
//! new expressions and immediately write them.
//!
//! ```
//! # use bevy_hanabi::*;
//! let mut module = Module::default();
//!
//! // Build and write a literal expression into the module.
//! let expr = module.lit(3.42);
//! ```
//!
//! Due to the code-first nature of the Expression API however, that approach
//! can be very verbose. Instead, users are encouraged to use an [`ExprWriter`],
//! a simple utility to build expressions with a shortened syntax. Once an
//! expression is built, it can be written into the underlying [`Module`]. This
//! approach generally makes the code more readable, and is therefore highly
//! encouraged, but is not mandatory.
//!
//! ```
//! # use bevy_hanabi::*;
//! // Create a writer owning a new Module
//! let mut w = ExprWriter::new();
//!
//! // Build a complex expression: max(3.42, properties.my_prop)
//! let expr = w.lit(3.42).max(w.prop("my_prop"));
//!
//! // Finalize the expression and write it into the Module. The returned handle can
//! // be assign to a modifier input.
//! let handle = expr.expr();
//!
//! // Finish using the writer and recover the Module with all written expressions
//! let module = w.finish();
//! ```
//!
//! [Node API]: crate::graph::node
//! [`Modifier`]: crate::Modifier
//! [`EffectAsset`]: crate::EffectAsset

use std::{cell::RefCell, num::NonZeroU32, rc::Rc};

use bevy::{reflect::Reflect, utils::thiserror::Error};
use serde::{Deserialize, Serialize};

use crate::{Attribute, PropertyLayout, ScalarType, ToWgslString, ValueType};

use super::Value;

type Index = NonZeroU32;

/// Handle of an expression inside a given [`Module`].
///
/// A handle uniquely references an [`Expr`] stored inside a [`Module`]. It's a
/// lightweight representation, similar to a simple array index. For this
/// reason, it's easily copyable. However it's also lacking any kind of error
/// checking, and mixing handles to different modules produces undefined
/// behaviors (like an index does when indexing the wrong array).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect, Serialize, Deserialize,
)]
pub struct ExprHandle {
    index: Index,
}

impl ExprHandle {
    /// Create a new handle from a 1-based [`Index`].
    fn new(index: Index) -> Self {
        Self { index }
    }

    /// Get the zero-based index into the array of the module.
    fn index(&self) -> usize {
        (self.index.get() - 1) as usize
    }
}

/// Container for expressions.
///
/// A module represents a storage for a set of expressions used in a single
/// [`EffectAsset`]. Modules are not reusable accross effect assets; each effect
/// asset owns a single module, containing all the expressions used in all the
/// modifiers attached to that asset. However, for convenience, a module can be
/// cloned into an unrelated module, and the clone can be assigned to another
/// effect asset.
///
/// Modules are built incrementally. Expressions are written into the module
/// through convenience helpers like [`lit()`] or [`attr()`]. Alternatively, an
/// [`ExprWriter`] can be used to populate a new or existing module. Either way,
/// once an expression is written into a module, it cannot be modified or
/// deleted. Modules are not designed to be used as editing structures, but as
/// storage and serialization ones.
///
/// [`EffectAsset`]: crate::EffectAsset
/// [`lit()`]: Module::lit
/// [`attr()`]: Module::attr
#[derive(Debug, Default, Clone, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub struct Module {
    expressions: Vec<Expr>,
}

impl Module {
    /// Create a new module from an existing collection of expressions.
    pub fn from_raw(expr: Vec<Expr>) -> Self {
        Self { expressions: expr }
    }

    /// Append a new expression to the module.
    fn push(&mut self, expr: impl Into<Expr>) -> ExprHandle {
        #[allow(unsafe_code)]
        let index: Index = unsafe { NonZeroU32::new_unchecked(self.expressions.len() as u32 + 1) };
        self.expressions.push(expr.into());
        ExprHandle::new(index)
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

    /// Build a property expression and append it to the module.
    #[inline]
    pub fn prop(&mut self, property_name: impl Into<String>) -> ExprHandle {
        self.push(Expr::Property(PropertyExpr::new(property_name)))
    }

    /// Build a built-in expression and append it to the module.
    #[inline]
    pub fn builtin(&mut self, op: BuiltInOperator) -> ExprHandle {
        self.push(Expr::BuiltIn(BuiltInExpr::new(op)))
    }

    /// Build a unary expression and append it to the module.
    ///
    /// The handle to the expression representing the operand of the unary
    /// operation must be valid, that is reference an expression
    /// contained in the current [`Module`].
    ///
    /// # Panics
    ///
    /// Panics in some cases if the operand handle do
    /// not reference an existing expression in the current module. Note however
    /// that this check can miss some invalid handles (false negative), so only
    /// represents an extra safety net that users shouldn't rely exclusively on
    /// to ensure the operand handles are valid. Instead, it's the
    /// responsibility of the user to ensure the operand handle references an
    /// existing expression in the current [`Module`].
    #[inline]
    pub fn unary(&mut self, op: UnaryOperator, inner: ExprHandle) -> ExprHandle {
        assert!(inner.index() < self.expressions.len());
        self.push(Expr::Unary { op, expr: inner })
    }

    /// Build an `abs()` unary expression and append it to the module.
    #[inline]
    pub fn abs(&mut self, inner: ExprHandle) -> ExprHandle {
        self.unary(UnaryOperator::Abs, inner)
    }

    /// Build an `all()` unary expression and append it to the module.
    #[inline]
    pub fn all(&mut self, inner: ExprHandle) -> ExprHandle {
        self.unary(UnaryOperator::All, inner)
    }

    /// Build an `any()` unary expression and append it to the module.
    #[inline]
    pub fn any(&mut self, inner: ExprHandle) -> ExprHandle {
        self.unary(UnaryOperator::Any, inner)
    }

    /// Build a `normalize()` unary expression and append it to the module.
    #[inline]
    pub fn normalize(&mut self, inner: ExprHandle) -> ExprHandle {
        self.unary(UnaryOperator::Normalize, inner)
    }

    /// Build a `sin()` unary expression and append it to the module.
    #[inline]
    pub fn sin(&mut self, inner: ExprHandle) -> ExprHandle {
        self.unary(UnaryOperator::Sin, inner)
    }

    /// Build a `cos()` unary expression and append it to the module.
    #[inline]
    pub fn cos(&mut self, inner: ExprHandle) -> ExprHandle {
        self.unary(UnaryOperator::Cos, inner)
    }

    /// Build a binary expression and append it to the module.
    ///
    /// The handles to the expressions representing the left and right operands
    /// of the binary operation must be valid, that is reference expressions
    /// contained in the current [`Module`].
    ///
    /// # Panics
    ///
    /// Panics in some cases if either of the left or right operand handles do
    /// not reference existing expressions in the current module. Note however
    /// that this check can miss some invalid handles (false negative), so only
    /// represents an extra safety net that users shouldn't rely exclusively on
    /// to ensure the operand handles are valid. Instead, it's the
    /// responsibility of the user to ensure handles reference existing
    /// expressions in the current [`Module`].
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

    /// Build a `dot()` binary expression and append it to the module.
    #[inline]
    pub fn dot(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::Dot, left, right)
    }

    /// Build a less-than binary expression and append it to the module.
    #[inline]
    pub fn lt(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::LessThan, left, right)
    }

    /// Build a less-than-or-equal binary expression and append it to the
    /// module.
    #[inline]
    pub fn le(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::LessThanOrEqual, left, right)
    }

    /// Build a greater-than binary expression and append it to the module.
    #[inline]
    pub fn gt(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::GreaterThan, left, right)
    }

    /// Build a greater-than-or-equal binary expression and append it to the
    /// module.
    #[inline]
    pub fn ge(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::GreaterThanOrEqual, left, right)
    }

    /// Build a `uniform()` binary expression and append it to the module.
    #[inline]
    pub fn uniform(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
        self.binary(BinaryOperator::UniformRand, left, right)
    }

    /// Get an existing expression from its handle.
    #[inline]
    pub fn get(&self, expr: ExprHandle) -> Option<&Expr> {
        let index = expr.index();
        self.expressions.get(index)
    }

    /// Get an existing expression from its handle.
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

/// Errors raised when manipulating expressions [`Expr`] and node graphs
/// [`Graph`].
///
/// [`Graph`]: crate::graph::Graph
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExprError {
    /// Expression type error.
    ///
    /// Generally used for invalid type conversion (casting).
    #[error("Type error: {0:?}")]
    TypeError(String),

    /// Expression syntax error.
    #[error("Syntax error: {0:?}")]
    SyntaxError(String),

    /// Generic graph evaluation error.
    #[error("Graph evaluation error: {0:?}")]
    GraphEvalError(String),

    /// Error resolving a property.
    ///
    /// An unknown property was not defined in the evaluation context, which
    /// usually means that the property was not defined
    /// with [`EffectAsset::with_property()`] or
    /// [`EffectAsset::add_property()`].
    ///
    /// [`EffectAsset::with_property()`]: crate::EffectAsset::with_property
    /// [`EffectAsset::add_property()`]: crate::EffectAsset::add_property
    #[error("Property error: {0:?}")]
    PropertyError(String),

    /// Invalid expression handle not referencing any existing [`Expr`] in the
    /// evaluation [`Module`].
    ///
    /// This error is commonly raised when using an [`ExprWriter`] and
    /// forgetting to transfer the underlying [`Module`] where the expressions
    /// are written to the [`EffectAsset`]. See [`ExprWriter`] for details.
    ///
    /// [`EffectAsset`]: crate::EffectAsset
    #[error("Invalid expression handle: {0:?}")]
    InvalidExprHandleError(String),
}

/// Evaluation context for transforming expressions into WGSL code.
///
/// The evaluation context references a [`Module`] storing all [`Expr`] in use,
/// as well as a [`PropertyLayout`] defining existing properties and their
/// layout in memory. These together define the context within which expressions
/// are evaluated.
///
/// A same expression can be valid in one context and invalid in another. The
/// most common example are [`PropertyExpr`] which are only valid if the
/// property is actually defined in the evaluation context.
pub trait EvalContext {
    /// Get the module the evaluation is taking place in.
    fn module(&self) -> &Module;

    /// Get the property layout of the effect.
    fn property_layout(&self) -> &PropertyLayout;

    /// Resolve an expression handle its the underlying expression.
    fn expr(&self, handle: ExprHandle) -> Result<&Expr, ExprError>;

    /// Evaluate an expression, returning its WGSL shader code.
    fn eval(&self, handle: ExprHandle) -> Result<String, ExprError>;
}

/// Language expression producing a value.
#[derive(Debug, Clone, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub enum Expr {
    /// Built-in expression ([`BuiltInExpr`]) providing access to some internal
    /// quantities like the simulation time.
    BuiltIn(BuiltInExpr),
    /// Literal expression ([`LiteralExpr`]) representing shader constants.
    Literal(LiteralExpr),
    /// Property expression ([`PropertyExpr`]) representing the value of an
    /// [`EffectAsset`]'s property.
    ///
    /// [`EffectAsset`]: crate::EffectAsset
    Property(PropertyExpr),
    /// Attribute expression ([`AttributeExpr`]) representing the value of an
    /// attribute for a particle, like its position or velocity.
    Attribute(AttributeExpr),
    /// Unary operation expression, transforming an expression into another
    /// expression.
    Unary {
        /// Unary operator.
        op: UnaryOperator,
        /// Operand the unary operation applies to.
        expr: ExprHandle,
    },
    /// Binary operation expression, composing two expressions into a third one.
    Binary {
        /// Binary operator.
        op: BinaryOperator,
        /// Left-hand side operand the binary operation applies to.
        left: ExprHandle,
        /// Right-hand side operand the binary operation applies to.
        right: ExprHandle,
    },
}

impl Expr {
    /// Is the expression resulting in a compile-time constant which can be
    /// hard-coded into a shader's code?
    ///
    /// The [`Module`] passed as argument is the owning module of the
    /// expression, which is used to recursively evaluate the const-ness of the
    /// entire expression. For some expressions like literals or attributes or
    /// properties their const-ness is intrinsic to the expression type, but for
    /// other expressions like binary operations (addition, ...) their
    /// const-ness depends in turn on the const-ness of their sub-expressions
    /// (left and right operands), which requires a [`Module`] to be retrieved
    /// and evaluated.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # let module = Module::default();
    /// // Literals are always constant by definition.
    /// assert!(Expr::Literal(LiteralExpr::new(1.)).is_const(&module));
    ///
    /// // Properties and attributes are never constant, since they're by definition used
    /// // to provide runtime customization.
    /// assert!(!Expr::Property(PropertyExpr::new("my_prop")).is_const(&module));
    /// assert!(!Expr::Attribute(AttributeExpr::new(Attribute::POSITION)).is_const(&module));
    /// ```
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
    ///
    /// If the type is variable and depends on the runtime evaluation context,
    /// this returns `None`. In that case the type needs to be obtained by
    /// evaluating the expression with [`eval()`].
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// // Literal expressions always have a constant, build-time value type.
    /// let expr = Expr::Literal(LiteralExpr::new(1.));
    /// assert_eq!(expr.value_type(), Some(ValueType::Scalar(ScalarType::Float)));
    /// ```
    ///
    /// [`eval()`]: crate::graph::Expr::eval
    pub fn value_type(&self) -> Option<ValueType> {
        match self {
            Expr::BuiltIn(expr) => Some(expr.value_type()),
            Expr::Literal(expr) => Some(expr.value_type()),
            Expr::Property(_) => None,
            Expr::Attribute(expr) => Some(expr.value_type()),
            Expr::Unary { .. } => None,
            Expr::Binary { .. } => None,
        }
    }

    /// Evaluate the expression in the given context.
    ///
    /// Evaluate the full expression as part of the given evaluation context,
    /// returning the WGSL string representation of the expression on success.
    ///
    /// The evaluation context is used to resolve some quantities related to the
    /// effect asset, like its properties. It also holds the [`Module`] that the
    /// expression is part of, to allow resolving sub-expressions of operators.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # let mut module = Module::default();
    /// # let pl = PropertyLayout::empty();
    /// # let context = InitContext::new(&mut module, &pl);
    /// let expr = Expr::Literal(LiteralExpr::new(1.));
    /// assert_eq!(Ok("1.".to_string()), expr.eval(&context));
    /// ```
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

/// A literal constant expression like `3.0` or `vec3<f32>(1.0, 2.0, 3.0)`.
///
/// Literal expression are compile-time constants. They are always constant
/// ([`is_const()`] is `true`) and have a value type equal to the type of the
/// constant itself.
///
/// [`is_const()`]: LiteralExpr::is_const
#[derive(Debug, Clone, Copy, PartialEq, Hash, Reflect, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum BuiltInOperator {
    /// Current effect system simulation time since startup, in seconds.
    Time,
    /// Delta time, in seconds, since last effect system update.
    DeltaTime,
    /// Random unit value of the given type.
    ///
    /// The type can be any scalar or vector type. Matrix types are not
    /// supported. The random values generated are uniformly distributed in
    /// `[0:1]`. For vectors, each component is sampled separately.
    ///
    /// The random number generator is from [the PCG family] of PRNGs, and is
    /// implemented directly inside the shader, and running on the GPU
    /// exclusively. It's seeded by the [`Spawner`].
    ///
    /// [the PCG family]: https://www.pcg-random.org/
    /// [`Spawner`]: crate::Spawner
    Rand(ValueType),
}

impl BuiltInOperator {
    /// Get the operator name.
    pub fn name(&self) -> &str {
        match self {
            BuiltInOperator::Time => "time",
            BuiltInOperator::DeltaTime => "delta_time",
            BuiltInOperator::Rand(value_type) => match value_type {
                ValueType::Scalar(s) => match s {
                    ScalarType::Bool => "brand",
                    ScalarType::Float => "frand",
                    ScalarType::Int => "irand",
                    ScalarType::Uint => "urand",
                },
                ValueType::Vector(vector_type) => {
                    match (vector_type.elem_type(), vector_type.count()) {
                        (ScalarType::Bool, 2) => "brand2",
                        (ScalarType::Bool, 3) => "brand3",
                        (ScalarType::Bool, 4) => "brand4",
                        (ScalarType::Float, 2) => "frand2",
                        (ScalarType::Float, 3) => "frand3",
                        (ScalarType::Float, 4) => "frand4",
                        (ScalarType::Int, 2) => "irand2",
                        (ScalarType::Int, 3) => "irand3",
                        (ScalarType::Int, 4) => "irand4",
                        (ScalarType::Uint, 2) => "urand2",
                        (ScalarType::Uint, 3) => "urand3",
                        (ScalarType::Uint, 4) => "urand4",
                        _ => panic!("Invalid vector type {:?}", vector_type),
                    }
                }
                ValueType::Matrix(_) => panic!("Invalid BuiltInOperator::Rand(ValueType::Matrix)."),
            },
        }
    }

    /// Get the type of the value of a built-in operator.
    pub fn value_type(&self) -> ValueType {
        match self {
            BuiltInOperator::Time => ValueType::Scalar(ScalarType::Float),
            BuiltInOperator::DeltaTime => ValueType::Scalar(ScalarType::Float),
            BuiltInOperator::Rand(value_type) => *value_type,
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
        match self {
            BuiltInOperator::Rand(_) => format!("{}()", self.name()),
            _ => format!("sim_params.{}", self.name()),
        }
    }
}

/// Expression for getting built-in quantities related to the effect system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct BuiltInExpr {
    operator: BuiltInOperator,
}

impl BuiltInExpr {
    /// Create a new built-in operator expression.
    ///
    /// # Panics
    ///
    /// Panics on invalid [`BuiltInOperator`], like `Rand(MatrixType)`. See
    /// each [`BuiltInOperator`] variant for more details.
    #[inline]
    pub fn new(operator: BuiltInOperator) -> Self {
        if let BuiltInOperator::Rand(value_type) = operator {
            assert!(!matches!(value_type, ValueType::Matrix(_)));
        }
        Self { operator }
    }

    /// Is the expression resulting in a compile-time constant?
    ///
    /// Constant expressions can be hard-coded into a shader's code, making them
    /// more efficient and open to shader compiler optimizing.
    pub fn is_const(&self) -> bool {
        false
    }

    /// Get the value type of the expression.
    ///
    /// The value type of the expression is the type of the value(s) that an
    /// expression produces.
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

/// Unary operator.
///
/// Operator applied to a single operand to produce another value. The type of
/// the operand and the result are not necessarily the same. Valid operand types
/// depend on the operator itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum UnaryOperator {
    /// Absolute value operator.
    ///
    /// Return the absolute value of the operand, component-wise for vectors.
    /// Only valid for numeric operands.
    Abs,

    /// Logical ALL operator for bool vectors.
    ///
    /// Return `true` if all the components of the bool vector operand are
    /// `true`. Invalid for any other type of operand.
    All,

    /// Logical ANY operator for bool vectors.
    ///
    /// Return `true` if any component of the bool vector operand is `true`.
    /// Invalid for any other type of operand.
    Any,

    /// Vector normalizing operator.
    ///
    /// Normalize the given numeric vector. Only valid for numeric vector
    /// operands.
    Normalize,

    /// Cosine operator.
    Cos,

    /// Sine operator.
    Sin,
}

impl ToWgslString for UnaryOperator {
    fn to_wgsl_string(&self) -> String {
        match *self {
            UnaryOperator::Abs => "abs".to_string(),
            UnaryOperator::All => "all".to_string(),
            UnaryOperator::Any => "any".to_string(),
            UnaryOperator::Normalize => "normalize".to_string(),
            UnaryOperator::Cos => "cos".to_string(),
            UnaryOperator::Sin => "sin".to_string(),
        }
    }
}

/// Binary operator.
///
/// Operator applied between two operands, generally denoted "left" and "right".
/// The type of the operands and the result are not necessarily the same. Valid
/// operand types depend on the operator itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum BinaryOperator {
    /// Addition operator.
    ///
    /// Returns the sum of its operands. Only valid for numeric operands.
    Add,

    /// Subtraction operator.
    ///
    /// Returns the difference between its left and right operands. Only valid
    /// for numeric operands.
    Sub,

    /// Multiply operator.
    ///
    /// Returns the product of its operands. Only valid for numeric operands.
    Mul,

    /// Division operator.
    ///
    /// Returns the left operand divided by the right operand. Only valid for
    /// numeric operands.
    Div,

    /// Less-than operator.
    ///
    /// Returns `true` if the left operand is strictly less than the right
    /// operand. Only valid for numeric types. If the operands are vectors,
    /// they must be of the same rank, and the result is a bool vector of
    /// that rank.
    LessThan,

    /// Less-than-or-equal operator.
    ///
    /// Returns `true` if the left operand is less than or equal to the right
    /// operand. Only valid for numeric types. If the operands are vectors,
    /// they must be of the same rank, and the result is a bool vector of
    /// that rank.
    LessThanOrEqual,

    /// Greater-than operator.
    ///
    /// Returns `true` if the left operand is strictly greater than the right
    /// operand. Only valid for numeric types. If the operands are vectors,
    /// they must be of the same rank, and the result is a bool vector of
    /// that rank.
    GreaterThan,

    /// Greater-than-or-equal operator.
    ///
    /// Returns `true` if the left operand is greater than or equal to the right
    /// operand. Only valid for numeric types. If the operands are vectors,
    /// they must be of the same rank, and the result is a bool vector of
    /// that rank.
    GreaterThanOrEqual,

    /// Minimum operator.
    ///
    /// Returns the minimum value of its left and right operands. Only valid for
    /// numeric types. If the operands are vectors, they must be of the same
    /// rank, and the result is a vector of that rank and same element
    /// scalar type.
    Min,

    /// Maximum operator.
    ///
    /// Returns the maximum value of its left and right operands. Only valid for
    /// numeric types. If the operands are vectors, they must be of the same
    /// rank, and the result is a vector of that rank and same element
    /// scalar type.
    Max,

    /// Dot product operator.
    ///
    /// Returns the dot product of the left and right operands. Only valid for
    /// vector type operands. Always produce a scalar floating-point result.
    Dot,

    /// Cross product operator.
    ///
    /// Returns the cross product of the left and right operands. Only valid for
    /// vector type operands of size 3. Always produce a vector result of the
    /// same size.
    Cross,

    /// Uniform random number operator.
    ///
    /// Returns a value generated by a fast non-cryptographically-secure
    /// pseudo-random number generator (PRNG) whose statistical characteristics
    /// are undefined and generally focused around speed. The random value is
    /// uniformly distributed between the left and right operands, which must be
    /// numeric types. If the operands are vectors, they must be of the same
    /// rank, and the result is a vector of that rank and same element
    /// scalar type.
    UniformRand,
}

impl BinaryOperator {
    /// Check if a binary operator is called via a functional-style call.
    ///
    /// Functional-style calls are in the form `op(lhs, rhs)`, like `min(a,
    /// b)` for example, while non-functional ones are in the form `lhs op rhs`,
    /// like `a + b` for example. This check is used for formatting the WGSL
    /// code emitted during evaluation of a binary operation expression.
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
            BinaryOperator::Min
            | BinaryOperator::Max
            | BinaryOperator::Dot
            | BinaryOperator::Cross
            | BinaryOperator::UniformRand => true,
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
            BinaryOperator::Dot => "dot".to_string(),
            BinaryOperator::Cross => "cross".to_string(),
            BinaryOperator::UniformRand => "rand_uniform".to_string(),
        }
    }
}

/// Expression writer.
///
/// Utility to write expressions with a simple functional syntax. Expressions
/// created with the writer are gatherned into a [`Module`] which can be
/// transfered once [`finish()`]ed to initialize an [`EffectAsset`].
///
/// Because an [`EffectAsset`] contains a single [`Module`], you generally want
/// to keep using the same [`ExprWriter`] to write all the expressions used by
/// all the [`Modifer`]s assigned to a given [`EffectAsset`], and only then once
/// done call [`finish()`] to recover the [`ExprWriter`]'s underlying [`Module`]
/// to assign it to the [`EffectAsset`].
///
/// # Example
///
/// ```
/// # use bevy_hanabi::*;
/// // Create a writer
/// let w = ExprWriter::new();
///
/// // Create a new expression: max(5. + particle.position, properties.my_prop)
/// let expr = (w.lit(5.) + w.attr(Attribute::POSITION)).max(w.prop("my_prop"));
///
/// // Finalize the expression and write it down into the `Module` as an `Expr`
/// let expr: ExprHandle = expr.expr();
///
/// // Create a modifier and assign the expression to one of its input(s)
/// let init_modifier = SetAttributeModifier::new(Attribute::LIFETIME, expr);
///
/// // Create an EffectAsset with the modifier and the Module from the writer
/// let effect = EffectAsset::new(1024, Spawner::rate(32_f32.into()), w.finish())
///     .init(init_modifier);
/// ```
///
/// [`finish()`]: ExprWriter::finish
/// [`EffectAsset`]: crate::EffectAsset
/// [`Modifer`]: crate::Modifier
#[derive(Debug, Default, Clone)]
pub struct ExprWriter {
    module: Rc<RefCell<Module>>,
}

#[allow(dead_code)]
impl ExprWriter {
    /// Create a new writer.
    ///
    /// The writer owns a new [`Module`] internally, and write all expressions
    /// to it. The module can be released to the user with [`finish()`] once
    /// done using the writer.
    ///
    /// [`finish()`]: ExprWriter::finish
    pub fn new() -> Self {
        Self {
            module: Rc::new(RefCell::new(Module::default())),
        }
    }

    /// Create a new writer from an existing module.
    ///
    /// This is an advanced use entry point to write expressions into an
    /// existing [`Module`]. In general, users should prefer using
    /// [`ExprWriter::new()`] to create a new [`Module`], and keep using the
    /// same [`ExprWriter`] to write all expressions of the same
    /// [`EffectAsset`].
    ///
    /// [`EffectAsset`]: crate::EffectAsset
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

    /// Create a new writer expression from a literal constant.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let mut w = ExprWriter::new();
    /// let x = w.lit(-3.5); // x = -3.5;
    /// ```
    pub fn lit(&self, value: impl Into<Value>) -> WriterExpr {
        self.push(Expr::Literal(LiteralExpr {
            value: value.into(),
        }))
    }

    /// Create a new writer expression from an attribute.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let mut w = ExprWriter::new();
    /// let x = w.attr(Attribute::POSITION); // x = particle.position;
    /// ```
    pub fn attr(&self, attr: Attribute) -> WriterExpr {
        self.push(Expr::Attribute(AttributeExpr::new(attr)))
    }

    /// Create a new writer expression from a property.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let mut w = ExprWriter::new();
    /// let x = w.prop("my_prop"); // x = properties.my_prop;
    /// ```
    pub fn prop(&self, name: impl Into<String>) -> WriterExpr {
        self.push(Expr::Property(PropertyExpr::new(name)))
    }

    /// Create a new writer expression representing the current simulation time.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let mut w = ExprWriter::new();
    /// let x = w.time(); // x = sim_params.time;
    /// ```
    pub fn time(&self) -> WriterExpr {
        self.push(Expr::BuiltIn(BuiltInExpr::new(BuiltInOperator::Time)))
    }

    /// Create a new writer expression representing the simulation delta time
    /// since last frame.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let mut w = ExprWriter::new();
    /// let x = w.delta_time(); // x = sim_params.delta_time;
    /// ```
    pub fn delta_time(&self) -> WriterExpr {
        self.push(Expr::BuiltIn(BuiltInExpr::new(BuiltInOperator::DeltaTime)))
    }

    /// Create a new writer expression representing a random value of the given
    /// type.
    ///
    /// The type can be any scalar or vector type. Matrix types are not
    /// supported. The random values generated are uniformly distributed in
    /// `[0:1]`. For vectors, each component is sampled separately.
    ///
    /// # Panics
    ///
    /// Panics in the same cases as [`BuiltInExpr::new()`] does.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let mut w = ExprWriter::new();
    /// let x = w.rand(VectorType::VEC3F); // x = frand3();
    /// ```
    pub fn rand(&self, value_type: impl Into<ValueType>) -> WriterExpr {
        self.push(Expr::BuiltIn(BuiltInExpr::new(BuiltInOperator::Rand(
            value_type.into(),
        ))))
    }

    /// Finish using the writer, and recover the [`Module`] where all [`Expr`]
    /// were written by the writer.
    ///
    /// This module is typically passed to [`EffectAsset::new()`] before adding
    /// to that effect the modifiers which use the expressions created by this
    /// writer.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # let spawner = Spawner::default();
    /// let mut w = ExprWriter::new();
    /// // [...]
    /// let module = w.finish();
    /// let asset = EffectAsset::new(256, spawner, module);
    /// ```
    ///
    /// [`EffectAsset::new()`]: crate::EffectAsset::new()
    pub fn finish(self) -> Module {
        self.module.take()
    }
}

/// Intermediate expression from an [`ExprWriter`].
///
/// A writer expression [`WriterExpr`] is equivalent to an [`ExprHandle`], but
/// retains a reference to the underlying [`Module`] and therefore can easily be
/// chained with other [`WriterExpr`] via a concise syntax, at the expense of
/// being more heavyweight and locking the underlying [`Module`] under a
/// ref-counted interior mutability (`Rc<RefCell<Module>>`). [`ExprHandle`] by
/// opposition is a very lightweight type, similar to a simple index. And like
/// an array index, [`ExprHandle`] doesn't explicitly reference its associated
/// storage ([`Module`]) which needs to be remembered by the user explicitly.
///
/// In addition, [`WriterExpr`] implements several numerical operators like the
/// [`std::ops::Add`] trait, making it simpler to combine it with another
/// [`WriterExpr`].
///
/// ```
/// # use bevy_hanabi::*;
/// let mut w = ExprWriter::new();
/// let x = w.lit(-3.5);
/// let y = w.lit(78.);
/// let z = x + y; // == 74.5
/// ```
///
/// In general the [`WriterExpr`] type is not used directly, but inferred from
/// calling [`ExprWriter`] methods and combining [`WriterExpr`] together.
///
/// ```
/// # use bevy_hanabi::*;
/// let mut w = ExprWriter::new();
///
/// // x = max(-3.5 + 1., properties.my_prop) * 0.5 - particle.position;
/// let x = (w.lit(-3.5) + w.lit(1.)).max(w.prop("my_prop")).mul(w.lit(0.5))
///     .sub(w.attr(Attribute::POSITION));
///
/// let handle: ExprHandle = x.expr();
/// ```
#[derive(Debug, Clone)]
pub struct WriterExpr {
    expr: ExprHandle,
    module: Rc<RefCell<Module>>,
}

impl WriterExpr {
    fn unary_op(self, op: UnaryOperator) -> Self {
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
    ///
    /// This is a unary operator, which applies component-wise to vector and
    /// matrix operand expressions.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = -3.5;`.
    /// let x = w.lit(-3.5);
    ///
    /// // The absolute value `y = abs(x);`.
    /// let y = x.abs(); // == 3.5
    /// ```
    pub fn abs(self) -> Self {
        self.unary_op(UnaryOperator::Abs)
    }

    /// Apply the logical operator "all" to the current bool vector expression.
    ///
    /// This is a unary operator, which applies to vector operand expressions to
    /// produce a scalar boolean.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::BVec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec3<bool>(true, false, true);`.
    /// let x = w.lit(BVec3::new(true, false, true));
    ///
    /// // Check if all components are true `y = all(x);`.
    /// let y = x.all(); // == false
    /// ```
    pub fn all(self) -> Self {
        self.unary_op(UnaryOperator::All)
    }

    /// Apply the logical operator "any" to the current bool vector expression.
    ///
    /// This is a unary operator, which applies to vector operand expressions to
    /// produce a scalar boolean.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::BVec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec3<bool>(true, false, true);`.
    /// let x = w.lit(BVec3::new(true, false, true));
    ///
    /// // Check if any components is true `y = any(x);`.
    /// let y = x.any(); // == true
    /// ```
    pub fn any(self) -> Self {
        self.unary_op(UnaryOperator::Any)
    }

    /// Apply the "normalize" operator to the current float vector expression.
    ///
    /// This is a unary operator, which applies to float vector operand
    /// expressions to produce another float vector with unit length
    /// (normalized).
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec3<f32>(1., 1., 1.);`.
    /// let x = w.lit(Vec3::ONE);
    ///
    /// // Normalize: `y = normalize(x);`
    /// let y = x.normalized();
    /// ```
    pub fn normalized(self) -> Self {
        self.unary_op(UnaryOperator::Normalize)
    }

    /// Apply the "sin" operator to the current float scalar or vector
    /// expression.
    ///
    /// This is a unary operator, which applies to float scalar or vector
    /// operand expressions to produce a float scalar or vector. It applies
    /// component-wise to vector operand expressions.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec3<f32>(1., 1., 1.);`.
    /// let x = w.lit(Vec3::ONE);
    ///
    /// // Sin: `y = sin(x);`
    /// let y = x.sin();
    /// ```
    pub fn sin(self) -> Self {
        self.unary_op(UnaryOperator::Sin)
    }

    /// Apply the "cos" operator to the current float scalar or vector
    /// expression.
    ///
    /// This is a unary operator, which applies to float scalar or vector
    /// operand expressions to produce a float scalar or vector. It applies
    /// component-wise to vector operand expressions.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec3<f32>(1., 1., 1.);`.
    /// let x = w.lit(Vec3::ONE);
    ///
    /// // Sin: `y = cos(x);`
    /// let y = x.cos();
    /// ```
    pub fn cos(self) -> Self {
        self.unary_op(UnaryOperator::Cos)
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
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec2;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec2<f32>(3., -2.);`.
    /// let x = w.lit(Vec2::new(3., -2.));
    ///
    /// // Another literal expression `y = vec2<f32>(1., 5.);`.
    /// let y = w.lit(Vec2::new(1., 5.));
    ///
    /// // The minimum of both vectors `z = min(x, y);`.
    /// let z = x.min(y); // == vec2<f32>(1., -2.)
    /// ```
    pub fn min(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Min)
    }

    /// Take the maximum value of the current expression and another expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec2;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec2<f32>(3., -2.);`.
    /// let x = w.lit(Vec2::new(3., -2.));
    ///
    /// // Another literal expression `y = vec2<f32>(1., 5.);`.
    /// let y = w.lit(Vec2::new(1., 5.));
    ///
    /// // The maximum of both vectors `z = max(x, y);`.
    /// let z = x.max(y); // == vec2<f32>(3., 5.)
    /// ```
    pub fn max(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Max)
    }

    /// Add the current expression with another expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// You can also use the [`std::ops::Add`] trait directly, via the `+`
    /// symbol, as an alternative to calling this method directly.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec2;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec2<f32>(3., -2.);`.
    /// let x = w.lit(Vec2::new(3., -2.));
    ///
    /// // Another literal expression `y = vec2<f32>(1., 5.);`.
    /// let y = w.lit(Vec2::new(1., 5.));
    ///
    /// // The sum of both vectors `z = x + y;`.
    /// let z = x.add(y); // == vec2<f32>(4., 3.)
    /// // -OR-
    /// // let z = x + y;
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Add)
    }

    /// Subtract another expression from the current expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// You can also use the [`std::ops::Sub`] trait directly, via the `-`
    /// symbol, as an alternative to calling this method directly.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec2;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec2<f32>(3., -2.);`.
    /// let x = w.lit(Vec2::new(3., -2.));
    ///
    /// // Another literal expression `y = vec2<f32>(1., 5.);`.
    /// let y = w.lit(Vec2::new(1., 5.));
    ///
    /// // The difference of both vectors `z = x - y;`.
    /// let z = x.sub(y); // == vec2<f32>(2., -7.)
    /// // -OR-
    /// // let z = x - y;
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Sub)
    }

    /// Multiply the current expression with another expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// You can also use the [`std::ops::Mul`] trait directly, via the `*`
    /// symbol, as an alternative to calling this method directly.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec2;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec2<f32>(3., -2.);`.
    /// let x = w.lit(Vec2::new(3., -2.));
    ///
    /// // Another literal expression `y = vec2<f32>(1., 5.);`.
    /// let y = w.lit(Vec2::new(1., 5.));
    ///
    /// // The product of both vectors `z = x * y;`.
    /// let z = x.mul(y); // == vec2<f32>(3., -10.)
    /// // -OR-
    /// // let z = x * y;
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Mul)
    }

    /// Divide the current expression by another expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// You can also use the [`std::ops::Div`] trait directly, via the `/`
    /// symbol, as an alternative to calling this method directly.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec2;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec2<f32>(3., -2.);`.
    /// let x = w.lit(Vec2::new(3., -2.));
    ///
    /// // Another literal expression `y = vec2<f32>(1., 5.);`.
    /// let y = w.lit(Vec2::new(1., 5.));
    ///
    /// // The quotient of both vectors `z = x / y;`.
    /// let z = x.div(y); // == vec2<f32>(3., -0.4)
    /// // -OR-
    /// // let z = x / y;
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Div)
    }

    /// Calculate the dot product of the current expression by another
    /// expression.
    ///
    /// This is a binary operator, which applies to vector operands of same size
    /// only, and always produces a floating point scalar.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec2;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec2<f32>(3., -2.);`.
    /// let x = w.lit(Vec2::new(3., -2.));
    ///
    /// // Another literal expression `y = vec2<f32>(1., 5.);`.
    /// let y = w.lit(Vec2::new(1., 5.));
    ///
    /// // The dot product of both vectors `z = dot(x, y);`.
    /// let z = x.dot(y);
    /// ```
    pub fn dot(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Dot)
    }

    /// Calculate the cross product of the current expression by another
    /// expression.
    ///
    /// This is a binary operator, which applies to vector operands of size 3
    /// only, and always produces a vector of the same size.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec3<f32>(3., -2., 1.);`.
    /// let x = w.lit(Vec3::new(3., -2., 1.));
    ///
    /// // Another literal expression `y = vec3<f32>(1., 5., 0.);`.
    /// let y = w.lit(Vec3::new(1., 5., 0.));
    ///
    /// // The cross product of both vectors `z = cross(x, y);`.
    /// let z = x.cross(y);
    /// ```
    pub fn cross(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Cross)
    }

    /// Apply the logical operator "less than or equal" to this expression and
    /// another expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec3<f32>(3., -2., 7.);`.
    /// let x = w.lit(Vec3::new(3., -2., 7.));
    ///
    /// // Another literal expression `y = vec3<f32>(1., 5., 7.);`.
    /// let y = w.lit(Vec3::new(1., 5., 7.));
    ///
    /// // The boolean result of the less than or equal operation `z = (x <= y);`.
    /// let z = x.le(y); // == vec3<bool>(false, true, true)
    /// ```
    pub fn le(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::LessThanOrEqual)
    }

    /// Apply the logical operator "less than" to this expression and another
    /// expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec3<f32>(3., -2., 7.);`.
    /// let x = w.lit(Vec3::new(3., -2., 7.));
    ///
    /// // Another literal expression `y = vec3<f32>(1., 5., 7.);`.
    /// let y = w.lit(Vec3::new(1., 5., 7.));
    ///
    /// // The boolean result of the less than operation `z = (x < y);`.
    /// let z = x.lt(y); // == vec3<bool>(false, true, false)
    /// ```
    pub fn lt(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::LessThan)
    }

    /// Apply the logical operator "greater than or equal" to this expression
    /// and another expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec3<f32>(3., -2., 7.);`.
    /// let x = w.lit(Vec3::new(3., -2., 7.));
    ///
    /// // Another literal expression `y = vec3<f32>(1., 5., 7.);`.
    /// let y = w.lit(Vec3::new(1., 5., 7.));
    ///
    /// // The boolean result of the greater than or equal operation `z = (x >= y);`.
    /// let z = x.ge(y); // == vec3<bool>(true, false, true)
    /// ```
    pub fn ge(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::GreaterThanOrEqual)
    }

    /// Apply the logical operator "greater than" to this expression and another
    /// expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec3<f32>(3., -2., 7.);`.
    /// let x = w.lit(Vec3::new(3., -2., 7.));
    ///
    /// // Another literal expression `y = vec3<f32>(1., 5., 7.);`.
    /// let y = w.lit(Vec3::new(1., 5., 7.));
    ///
    /// // The boolean result of the greater than operation `z = (x > y);`.
    /// let z = x.gt(y); // == vec3<bool>(true, false, false)
    /// ```
    pub fn gt(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::GreaterThan)
    }

    /// Apply the logical operator "uniform" to this expression and another
    /// expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions. That is, for vectors, this produces a vector of
    /// random values where each component is uniformly distributed within the
    /// bounds of the related component of both operands.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec3<f32>(3., -2., 7.);`.
    /// let x = w.lit(Vec3::new(3., -2., 7.));
    ///
    /// // Another literal expression `y = vec3<f32>(1., 5., 7.);`.
    /// let y = w.lit(Vec3::new(1., 5., 7.));
    ///
    /// // A random variable uniformly distributed in [1:3]x[-2:5]x[7:7].
    /// let z = x.uniform(y);
    /// ```
    pub fn uniform(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::UniformRand)
    }

    /// Finalize an expression chain and return the accumulated expression.
    ///
    /// The returned handle indexes the [`Module`] owned by the [`ExprWriter`]
    /// this intermediate expression was built from.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = -3.5;`.
    /// let x = w.lit(-3.5);
    ///
    /// // Retrieve the ExprHandle for that expression.
    /// let handle = x.expr();
    /// ```
    pub fn expr(self) -> ExprHandle {
        self.expr
    }
}

impl std::ops::Add<WriterExpr> for WriterExpr {
    type Output = WriterExpr;

    #[inline]
    fn add(self, rhs: WriterExpr) -> Self::Output {
        self.add(rhs)
    }
}

impl std::ops::Sub<WriterExpr> for WriterExpr {
    type Output = WriterExpr;

    #[inline]
    fn sub(self, rhs: WriterExpr) -> Self::Output {
        self.sub(rhs)
    }
}

impl std::ops::Mul<WriterExpr> for WriterExpr {
    type Output = WriterExpr;

    #[inline]
    fn mul(self, rhs: WriterExpr) -> Self::Output {
        self.mul(rhs)
    }
}

impl std::ops::Div<WriterExpr> for WriterExpr {
    type Output = WriterExpr;

    #[inline]
    fn div(self, rhs: WriterExpr) -> Self::Output {
        self.div(rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::Property, InitContext, ScalarValue, VectorType};

    use super::*;
    use bevy::prelude::*;

    #[test]
    fn writer() {
        // Get a module and its writer
        let w = ExprWriter::new();

        // Build some expression
        let x = w.lit(3.).abs().max(w.attr(Attribute::POSITION) * w.lit(2.))
            + w.lit(-4.).min(w.prop("my_prop"));
        let x = x.expr();

        // Create an evaluation context
        let property_layout =
            PropertyLayout::new(&[Property::new("my_prop", ScalarValue::Float(3.))]);
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
    fn type_error() {
        let l = Value::Scalar(3.5_f32.into());
        let r: Result<Vec2, ExprError> = l.try_into();
        assert!(r.is_err());
        assert!(matches!(r, Err(ExprError::TypeError(_))));
    }

    #[test]
    fn math_expr() {
        let mut m = Module::default();

        let x = m.attr(Attribute::POSITION);
        let y = m.lit(Vec3::ONE);

        let add = m.add(x, y);
        let sub = m.sub(x, y);
        let mul = m.mul(x, y);
        let div = m.div(x, y);
        let lt = m.lt(x, y);
        let le = m.le(x, y);
        let gt = m.gt(x, y);
        let ge = m.ge(x, y);

        let pl = PropertyLayout::default();
        let ctx = InitContext {
            module: &mut m,
            init_code: String::new(),
            init_extra: String::new(),
            property_layout: &pl,
        };

        for (expr, op) in [
            (add, "+"),
            (sub, "-"),
            (mul, "*"),
            (div, "/"),
            (lt, "<"),
            (le, "<="),
            (gt, ">"),
            (ge, ">="),
        ] {
            let expr = ctx.eval(expr);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            assert_eq!(
                expr,
                format!(
                    "(particle.{}) {} (vec3<f32>(1.,1.,1.))",
                    Attribute::POSITION.name(),
                    op,
                )
            );
        }
    }

    #[test]
    fn builtin_expr() {
        let mut m = Module::default();

        for op in [BuiltInOperator::Time, BuiltInOperator::DeltaTime] {
            let value = m.builtin(op);

            let pl = PropertyLayout::default();
            let ctx = InitContext {
                module: &mut m,
                init_code: String::new(),
                init_extra: String::new(),
                property_layout: &pl,
            };

            let expr = ctx.eval(value);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            assert_eq!(expr, format!("sim_params.{}", op.name()));
        }

        for (scalar_type, prefix) in [
            (ScalarType::Bool, "b"),
            (ScalarType::Float, "f"),
            (ScalarType::Int, "i"),
            (ScalarType::Uint, "u"),
        ] {
            let value = m.builtin(BuiltInOperator::Rand(scalar_type.into()));

            let pl = PropertyLayout::default();
            let ctx = InitContext {
                module: &mut m,
                init_code: String::new(),
                init_extra: String::new(),
                property_layout: &pl,
            };

            let expr = ctx.eval(value);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            assert_eq!(expr, format!("{}rand()", prefix));

            for count in 2..=4 {
                let vec = m.builtin(BuiltInOperator::Rand(
                    VectorType::new(scalar_type, count).into(),
                ));

                let pl = PropertyLayout::default();
                let ctx = InitContext {
                    module: &mut m,
                    init_code: String::new(),
                    init_extra: String::new(),
                    property_layout: &pl,
                };

                let expr = ctx.eval(vec);
                assert!(expr.is_ok());
                let expr = expr.unwrap();
                assert_eq!(expr, format!("{}rand{}()", prefix, count));
            }
        }
    }

    #[test]
    fn unary_expr() {
        let mut m = Module::default();

        let x = m.attr(Attribute::POSITION);
        let y = m.lit(Vec3::new(1., -3.1, 6.99));
        let z = m.lit(BVec3::new(false, true, false));

        let abs = m.abs(x);
        let norm = m.normalize(y);
        let any = m.any(z);
        let all = m.all(z);

        let pl = PropertyLayout::default();
        let ctx = InitContext {
            module: &mut m,
            init_code: String::new(),
            init_extra: String::new(),
            property_layout: &pl,
        };

        for (expr, op, inner) in [
            (
                abs,
                "abs",
                &format!("particle.{}", Attribute::POSITION.name())[..],
            ),
            (norm, "normalize", "vec3<f32>(1.,-3.1,6.99)"),
            (any, "any", "vec3<bool>(false,true,false)"),
            (all, "all", "vec3<bool>(false,true,false)"),
        ] {
            let expr = ctx.eval(expr);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            assert_eq!(expr, format!("{}({})", op, inner,));
        }
    }

    #[test]
    fn binary_expr() {
        let mut m = Module::default();

        let x = m.attr(Attribute::POSITION);
        let y = m.lit(Vec3::ONE);

        let min = m.min(x, y);
        let max = m.max(x, y);

        let pl = PropertyLayout::default();
        let ctx = InitContext {
            module: &mut m,
            init_code: String::new(),
            init_extra: String::new(),
            property_layout: &pl,
        };

        for (expr, op) in [(min, "min"), (max, "max")] {
            let expr = ctx.eval(expr);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            assert_eq!(
                expr,
                format!(
                    "{}(particle.{}, vec3<f32>(1.,1.,1.))",
                    op,
                    Attribute::POSITION.name(),
                )
            );
        }
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

    //     let b: ExprHandle = Box::new(l);
    //     let s = ron::to_string(&b).unwrap();
    //     println!("boxed literal: {:?}", s);
    //     let b_serde: ExprHandle = ron::from_str(&s).unwrap();
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
