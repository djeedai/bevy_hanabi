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

use crate::{
    Attribute, ModifierContext, ParticleLayout, PropertyLayout, ScalarType, ToWgslString, ValueType,
};

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

macro_rules! impl_module_unary {
    ($t: ident, $T: ident) => {
        #[doc = concat!("Build a [`UnaryOperator::", stringify!($T), "`](crate::graph::expr::UnaryOperator::", stringify!($T),") unary expression and append it to the module.\n\nThis is a shortcut for [`unary(UnaryOperator::", stringify!($T), ", inner)`](crate::graph::expr::Module::unary).")]
        #[inline]
        pub fn $t(&mut self, inner: ExprHandle) -> ExprHandle {
            self.unary(UnaryOperator::$T, inner)
        }
    };
}

macro_rules! impl_module_binary {
    ($t: ident, $T: ident) => {
        #[doc = concat!("Build a [`BinaryOperator::", stringify!($T), "`](crate::graph::expr::BinaryOperator::", stringify!($T),") binary expression and append it to the module.\n\nThis is a shortcut for [`binary(BinaryOperator::", stringify!($T), ", left, right)`](crate::graph::expr::Module::binary).")]
        #[inline]
        pub fn $t(&mut self, left: ExprHandle, right: ExprHandle) -> ExprHandle {
            self.binary(BinaryOperator::$T, left, right)
        }
    };
}

macro_rules! impl_module_ternary {
    ($t: ident, $T: ident) => {
        #[doc = concat!("Build a [`TernaryOperator::", stringify!($T), "`](crate::graph::expr::TernaryOperator::", stringify!($T),") ternary expression and append it to the module.\n\nThis is a shortcut for [`ternary(TernaryOperator::", stringify!($T), ", first, second, third)`](crate::graph::expr::Module::ternary).")]
        #[inline]
        pub fn $t(&mut self, first: ExprHandle, second: ExprHandle, third: ExprHandle) -> ExprHandle {
            self.ternary(TernaryOperator::$T, first, second, third)
        }
    };
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
    /// Panics in some cases if the operand handle do not reference an existing
    /// expression in the current module. Note however that this check can
    /// miss some invalid handles (false negative), so only represents an
    /// extra safety net that users shouldn't rely exclusively on
    /// to ensure the operand handles are valid. Instead, it's the
    /// responsibility of the user to ensure the operand handle references an
    /// existing expression in the current [`Module`].
    #[inline]
    pub fn unary(&mut self, op: UnaryOperator, inner: ExprHandle) -> ExprHandle {
        assert!(inner.index() < self.expressions.len());
        self.push(Expr::Unary { op, expr: inner })
    }

    impl_module_unary!(abs, Abs);
    impl_module_unary!(all, All);
    impl_module_unary!(any, Any);
    impl_module_unary!(ceil, Ceil);
    impl_module_unary!(cos, Cos);
    impl_module_unary!(exp, Exp);
    impl_module_unary!(exp2, Exp2);
    impl_module_unary!(floor, Floor);
    impl_module_unary!(fract, Fract);
    impl_module_unary!(length, Length);
    impl_module_unary!(log, Log);
    impl_module_unary!(log2, Log2);
    impl_module_unary!(normalize, Normalize);
    impl_module_unary!(pack4x8snorm, Pack4x8snorm);
    impl_module_unary!(pack4x8unorm, Pack4x8unorm);
    impl_module_unary!(saturate, Saturate);
    impl_module_unary!(sign, Sign);
    impl_module_unary!(sin, Sin);
    impl_module_unary!(tan, Tan);
    impl_module_unary!(unpack4x8snorm, Unpack4x8snorm);
    impl_module_unary!(unpack4x8unorm, Unpack4x8unorm);
    impl_module_unary!(w, W);
    impl_module_unary!(x, X);
    impl_module_unary!(y, Y);
    impl_module_unary!(z, Z);

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

    impl_module_binary!(add, Add);
    impl_module_binary!(cross, Cross);
    impl_module_binary!(distance, Distance);
    impl_module_binary!(div, Div);
    impl_module_binary!(dot, Dot);
    impl_module_binary!(ge, GreaterThanOrEqual);
    impl_module_binary!(gt, GreaterThan);
    impl_module_binary!(le, LessThanOrEqual);
    impl_module_binary!(lt, LessThan);
    impl_module_binary!(max, Max);
    impl_module_binary!(min, Min);
    impl_module_binary!(mul, Mul);
    impl_module_binary!(rem, Remainder);
    impl_module_binary!(step, Step);
    impl_module_binary!(sub, Sub);
    impl_module_binary!(uniform, UniformRand);

    /// Build a ternary expression and append it to the module.
    ///
    /// The handles to the expressions representing the three operands of the
    /// ternary operation must be valid, that is reference expressions
    /// contained in the current [`Module`].
    ///
    /// # Panics
    ///
    /// Panics in some cases if any of the operand handles do not reference
    /// existing expressions in the current module. Note however
    /// that this check can miss some invalid handles (false negative), so only
    /// represents an extra safety net that users shouldn't rely exclusively on
    /// to ensure the operand handles are valid. Instead, it's the
    /// responsibility of the user to ensure handles reference existing
    /// expressions in the current [`Module`].
    #[inline]
    pub fn ternary(
        &mut self,
        op: TernaryOperator,
        first: ExprHandle,
        second: ExprHandle,
        third: ExprHandle,
    ) -> ExprHandle {
        assert!(first.index() < self.expressions.len());
        assert!(second.index() < self.expressions.len());
        assert!(third.index() < self.expressions.len());
        self.push(Expr::Ternary {
            op,
            first,
            second,
            third,
        })
    }

    impl_module_ternary!(mix, Mix);
    impl_module_ternary!(smoothstep, SmoothStep);

    /// Build a cast expression and append it to the module.
    ///
    /// The handle to the expressions representing the operand of the cast
    /// operation must be valid, that is reference expressions contained in
    /// the current [`Module`].
    ///
    /// # Panics
    ///
    /// Panics in some cases if the operand handle does not reference existing
    /// expressions in the current module. Note however that this check can
    /// miss some invalid handles (false negative), so only represents an
    /// extra safety net that users shouldn't rely exclusively on
    /// to ensure the operand handles are valid. Instead, it's the
    /// responsibility of the user to ensure handles reference existing
    /// expressions in the current [`Module`].
    ///
    /// Panics if the resulting cast expression is not valid. See
    /// [`CastExpr::is_valid()`] for the exact meaning.
    pub fn cast(&mut self, expr: ExprHandle, target: impl Into<ValueType>) -> ExprHandle {
        assert!(expr.index() < self.expressions.len());
        let target = target.into();
        let expr = CastExpr::new(expr, target);
        if let Some(valid) = expr.is_valid(self) {
            assert!(valid);
        }
        self.push(Expr::Cast(expr))
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

    /// Get an existing expression from its handle.
    #[inline]
    pub fn try_get(&self, expr: ExprHandle) -> Result<&Expr, ExprError> {
        let index = expr.index();
        self.expressions
            .get(index)
            .ok_or(ExprError::InvalidExprHandleError(format!(
                "Cannot find expression with handle {:?} in the current module. Check that the Module used to build the expression was the same used in the EvalContext or the original EffectAsset.", expr)))
    }

    /// Get an existing expression from its handle.
    #[inline]
    pub fn try_get_mut(&mut self, expr: ExprHandle) -> Result<&mut Expr, ExprError> {
        let index = expr.index();
        self.expressions
            .get_mut(index)
            .ok_or(ExprError::InvalidExprHandleError(format!(
                "Cannot find expression with handle {:?} in the current module. Check that the Module used to build the expression was the same used in the EvalContext or the original EffectAsset.", expr)))
    }

    /// Is the expression resulting in a compile-time constant which can be
    /// hard-coded into a shader's code?
    ///
    /// # Panics
    ///
    /// Panics if `expr` doesn't refer to an expression of this module.
    #[inline]
    pub fn is_const(&self, expr: ExprHandle) -> bool {
        let expr = self.get(expr).unwrap();
        expr.is_const(self)
    }

    /// Has the expression any side-effect?
    ///
    /// Expressions with side-effect need to be stored into temporary variables
    /// when the shader code is emitted, so that the side effect is only applied
    /// once when the expression is reused in multiple locations.
    ///
    /// # Panics
    ///
    /// Panics if `expr` doesn't refer to an expression of this module.
    pub fn has_side_effect(&self, expr: ExprHandle) -> bool {
        let expr = self.get(expr).unwrap();
        expr.has_side_effect(self)
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
    /// Get the modifier context of the evaluation.
    fn modifier_context(&self) -> ModifierContext;

    /// Get the particle layout of the effect.
    fn particle_layout(&self) -> &ParticleLayout;

    /// Get the property layout of the effect.
    fn property_layout(&self) -> &PropertyLayout;

    /// Evaluate an expression, returning its WGSL shader code.
    ///
    /// The evaluation is guaranteed to be unique. Calling `eval()` multiple
    /// times with the same handle will return the same result. If the
    /// expression referenced by `handle` has side effects, it's evaluated only
    /// once on first call, stored in a local variable, and the variable cached
    /// and returned on subsequent calls.
    fn eval(&mut self, module: &Module, handle: ExprHandle) -> Result<String, ExprError>;

    /// Generate a unique local variable name.
    ///
    /// Each time this function is called, a new unique name is generated. The
    /// name is guaranteed to be unique within the current evaluation context
    /// only. Do not use for global top-level identifiers.
    fn make_local_var(&mut self) -> String;

    /// Push an intermediate statement during an evaluation.
    ///
    /// Intermediate statements are inserted before the expression evaluation
    /// which produced them. They're generally used to define temporary local
    /// variables, for example to store the result of expressions with side
    /// effects.
    fn push_stmt(&mut self, stmt: &str);

    /// Create a function.
    ///
    /// Create a new function with the given `func_name` inside the given
    /// [`Module`]. The function takes a list of arguments `args`, which are
    /// copied verbatim into the shader code without any validation. The body of
    /// the function is generated by invoking the given closure once with the
    /// input `module` and a temporary `EvalContext` local to the function. The
    /// closure must return the generated shader code of the function body. Any
    /// statement pushed to the temporary function context with
    /// [`EvalContext::push_stmt()`] is emitted inside the function body before
    /// the returned code. The function can subsequently be called from the
    /// parent context by generating code to call `func_name`, with the correct
    /// arguments.
    fn make_fn(
        &mut self,
        func_name: &str,
        args: &str,
        module: &mut Module,
        f: &mut dyn FnMut(&mut Module, &mut dyn EvalContext) -> Result<String, ExprError>,
    ) -> Result<(), ExprError>;
}

/// Language expression producing a value.
#[derive(Debug, Clone, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub enum Expr {
    /// Built-in expression ([`BuiltInExpr`]).
    ///
    /// A built-in expression provides access to some internal
    /// quantities like the simulation time.
    BuiltIn(BuiltInExpr),

    /// Literal expression ([`LiteralExpr`]).
    ///
    /// A literal expression represents a shader constants.
    Literal(LiteralExpr),

    /// Property expression ([`PropertyExpr`]).
    ///
    /// A property expression represents the value of an [`EffectAsset`]'s
    /// property.
    ///
    /// [`EffectAsset`]: crate::EffectAsset
    Property(PropertyExpr),

    /// Attribute expression ([`AttributeExpr`]).
    ///
    /// An attribute expression represents the value of an attribute for a
    /// particle, like its position or velocity.
    Attribute(AttributeExpr),

    /// Unary operation expression.
    ///
    /// A unary operation transforms an expression into another expression.
    Unary {
        /// Unary operator.
        op: UnaryOperator,
        /// Operand the unary operation applies to.
        expr: ExprHandle,
    },

    /// Binary operation expression.
    ///
    /// A binary operation composes two expressions into a third one.
    Binary {
        /// Binary operator.
        op: BinaryOperator,
        /// Left-hand side operand the binary operation applies to.
        left: ExprHandle,
        /// Right-hand side operand the binary operation applies to.
        right: ExprHandle,
    },

    /// Ternary operation expression.
    ///
    /// A ternary operation composes three expressions into a fourth one.
    Ternary {
        /// Ternary operator.
        op: TernaryOperator,
        /// First operand the ternary operation applies to.
        first: ExprHandle,
        /// Second operand the ternary operation applies to.
        second: ExprHandle,
        /// Third operand the ternary operation applies to.
        third: ExprHandle,
    },

    /// Cast expression.
    ///
    /// An expression to cast an expression to another type.
    Cast(CastExpr),
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
            Expr::Ternary {
                first,
                second,
                third,
                ..
            } => module.is_const(*first) && module.is_const(*second) && module.is_const(*third),
            Expr::Cast(expr) => module.is_const(expr.inner),
        }
    }

    /// Has the expression any side-effect?
    ///
    /// Expressions with side-effect need to be stored into temporary variables
    /// when the shader code is emitted, so that the side effect is only applied
    /// once when the expression is reused in multiple locations.
    pub fn has_side_effect(&self, module: &Module) -> bool {
        match self {
            Expr::BuiltIn(expr) => expr.has_side_effect(),
            Expr::Literal(_) => false,
            Expr::Property(_) => false,
            Expr::Attribute(_) => false,
            Expr::Unary { expr, .. } => module.has_side_effect(*expr),
            Expr::Binary { left, right, op } => {
                (*op == BinaryOperator::UniformRand)
                    || module.has_side_effect(*left)
                    || module.has_side_effect(*right)
            }
            Expr::Ternary {
                first,
                second,
                third,
                ..
            } => {
                module.has_side_effect(*first)
                    || module.has_side_effect(*second)
                    || module.has_side_effect(*third)
            }
            Expr::Cast(expr) => module.has_side_effect(expr.inner),
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
            Expr::Ternary { .. } => None,
            Expr::Cast(expr) => Some(expr.value_type()),
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
    /// let mut module = Module::default();
    /// # let pl = PropertyLayout::empty();
    /// # let pal = ParticleLayout::default();
    /// # let mut context = InitContext::new(&pl, &pal);
    /// let handle = module.lit(1.);
    /// let expr = module.get(handle).unwrap();
    /// assert_eq!(Ok("1.".to_string()), expr.eval(&module, &mut context));
    /// ```
    pub fn eval(
        &self,
        module: &Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        match self {
            Expr::BuiltIn(expr) => expr.eval(context),
            Expr::Literal(expr) => expr.eval(context),
            Expr::Property(expr) => expr.eval(context),
            Expr::Attribute(expr) => expr.eval(context),
            Expr::Unary { op, expr } => {
                // Recursively evaluate child expressions throught the context to ensure caching
                let expr = context.eval(module, *expr)?;

                // if expr.value_type() != self.value_type() {
                //     return Err(ExprError::TypeError(format!(
                //         "Cannot apply normalize() function to non-vector expression: {}",
                //         expr.unwrap_or("(error evaluating expression)".to_string())
                //     )));
                // }

                Ok(if op.is_functional() {
                    format!("{}({})", op.to_wgsl_string(), expr)
                } else {
                    format!("{}.{}", expr, op.to_wgsl_string())
                })
            }
            Expr::Binary { op, left, right } => {
                // Recursively evaluate child expressions throught the context to ensure caching
                let left = context.eval(module, *left)?;
                let right = context.eval(module, *right)?;

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
            Expr::Ternary {
                op,
                first,
                second,
                third,
            } => {
                // Recursively evaluate child expressions throught the context to ensure caching
                let first = context.eval(module, *first)?;
                let second = context.eval(module, *second)?;
                let third = context.eval(module, *third)?;

                // if !self.input.value_type().is_vector() {
                //     return Err(ExprError::TypeError(format!(
                //         "Cannot apply normalize() function to non-vector expression: {}",
                //         expr.unwrap_or("(error evaluating expression)".to_string())
                //     )));
                // }

                Ok(format!(
                    "{}({}, {}, {})",
                    op.to_wgsl_string(),
                    first,
                    second,
                    third
                ))
            }
            Expr::Cast(expr) => {
                // Recursively evaluate child expressions throught the context to ensure caching
                let inner = context.eval(module, expr.inner)?;

                Ok(format!("{}({})", expr.target.to_wgsl_string(), inner))
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

/// Expression to cast an expression to another type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct CastExpr {
    /// The operand expression to cast.
    inner: ExprHandle,
    /// The target type to cast to.
    target: ValueType,
}

impl CastExpr {
    /// Create a new cast expression.
    #[inline]
    pub fn new(inner: ExprHandle, target: impl Into<ValueType>) -> Self {
        Self {
            inner,
            target: target.into(),
        }
    }

    /// Get the value type of the expression.
    pub fn value_type(&self) -> ValueType {
        self.target
    }

    /// Try to evaluate if the cast expression is valid.
    ///
    /// The evaluation fails if the value type of the operand cannot be
    /// determined. In that case, the function returns `None`.
    ///
    /// Valid cast expressions are:
    /// - scalar to scalar
    /// - scalar to vector
    /// - vector to vector
    /// - matrix to matrix
    pub fn is_valid(&self, module: &Module) -> Option<bool> {
        let Some(inner) = module.get(self.inner) else {
            return Some(false);
        };
        if let Some(inner_type) = inner.value_type() {
            match self.target {
                ValueType::Scalar(_) => {
                    // scalar -> scalar only
                    Some(matches!(inner_type, ValueType::Scalar(_)))
                }
                ValueType::Vector(_) => {
                    // {scalar, vector} -> vector
                    Some(!matches!(inner_type, ValueType::Matrix(_)))
                }
                ValueType::Matrix(_) => {
                    // matrix -> matrix only
                    Some(matches!(inner_type, ValueType::Matrix(_)))
                }
            }
        } else {
            None
        }
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
    /// Value of the alpha cutoff for alpha masking.
    ///
    /// This value is only available in the render context. It represents the
    /// current threshold, generally in \[0:1\], which the particle's fragment
    /// alpha value will be compared against to determine alpha masking.
    ///
    /// The value is initalized at the beginning of the fragment shader to the
    /// expression stored in [`AlphaMode::Mask`].
    ///
    /// [`AlphaMode::Mask`]: crate::AlphaMode::Mask
    AlphaCutoff,
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
            BuiltInOperator::AlphaCutoff => "alpha_cutoff",
        }
    }

    /// Get the type of the value of a built-in operator.
    pub fn value_type(&self) -> ValueType {
        match self {
            BuiltInOperator::Time => ValueType::Scalar(ScalarType::Float),
            BuiltInOperator::DeltaTime => ValueType::Scalar(ScalarType::Float),
            BuiltInOperator::Rand(value_type) => *value_type,
            BuiltInOperator::AlphaCutoff => ValueType::Scalar(ScalarType::Float),
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

    /// Has the expression any side-effect?
    ///
    /// Expressions with side-effect need to be stored into temporary variables
    /// when the shader code is emitted, so that the side effect is only applied
    /// once when the expression is reused in multiple locations.
    pub fn has_side_effect(&self) -> bool {
        matches!(self.operator, BuiltInOperator::Rand(_))
    }

    /// Get the value type of the expression.
    ///
    /// The value type of the expression is the type of the value(s) that an
    /// expression produces.
    pub fn value_type(&self) -> ValueType {
        self.operator.value_type()
    }

    /// Evaluate the expression in the given context.
    pub fn eval(&self, context: &mut dyn EvalContext) -> Result<String, ExprError> {
        if self.has_side_effect() {
            let var_name = context.make_local_var();
            context.push_stmt(&format!("let {} = {};", var_name, self.to_wgsl_string()));
            Ok(var_name)
        } else {
            Ok(self.to_wgsl_string())
        }
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

    /// Ceiling operator.
    ///
    /// Return the unique integral number `k` such that `k-1 < x <= k`, where
    /// `x` is the operand which the operator applies to.
    Ceil,

    /// Cosine operator.
    Cos,

    /// Natural exponent operator.
    ///
    /// Return the natural exponentiation of the operand (`e^x`), component-wise
    /// for vectors.
    Exp,

    /// Base-2 exponent operator.
    ///
    /// Return two raised to the power of the operand (`2^x`), component-wise
    /// for vectors.
    Exp2,

    /// Floor operator.
    ///
    /// Return the unique integral number `k` such that `k <= x < k+1`, where
    /// `x` is the operand which the operator applies to.
    Floor,

    /// Fractional part operator.
    ///
    /// Return the fractional part of the operand, which is equal to `x -
    /// floor(x)`, component-wise for vectors.
    Fract,

    /// Length operator.
    ///
    /// Return the length of a floating point scalar or vector. The "length" of
    /// a scalar is taken as its absolute value. The length of a vector is the
    /// Euclidian distance `sqrt(x^2 + ...)` (square root of the sum of the
    /// squared components).
    ///
    /// The output is always a floating point scalar.
    Length,

    /// Natural logarithm operator.
    ///
    /// Return the natural logarithm of the operand (`log(x)`), component-wise
    /// for vectors.
    Log,

    /// Base-2 logarithm operator.
    ///
    /// Return the base-2 logarithm of the operand (`log2(x)`), component-wise
    /// for vectors.
    Log2,

    /// Vector normalizing operator.
    ///
    /// Normalize the given numeric vector. Only valid for numeric vector
    /// operands.
    Normalize,

    /// Packing operator from `vec4<f32>` to `u32` (signed normalized).
    ///
    /// Convert the four components of a signed normalized floating point vector
    /// into a signed integral `i8` value in `[-128:127]`, then pack those
    /// four values into a single `u32`. Each vector component should be in
    /// `[-1:1]` before packing; values outside this range are clamped.
    Pack4x8snorm,

    /// Packing operator from `vec4<f32>` to `u32` (unsigned normalized).
    ///
    /// Convert the four components of an unsigned normalized floating point
    /// vector into an unsigned integral `u8` value in `[0:255]`, then pack
    /// those four values into a single `u32`. Each vector component should
    /// be in `[0:1]` before packing; values outside this range are clamped.
    Pack4x8unorm,

    /// Saturate operator.
    ///
    /// Clamp the value of the operand to the \[0:1\] range, component-wise for
    /// vectors.
    Saturate,

    /// Sign operator.
    ///
    /// Return a value representing the sign of a floating point scalar or
    /// vector input:
    /// - `1.` if the operand is > 0
    /// - `0.` if the operand is = 0
    /// - `-1.` if the operand is < 0
    ///
    /// Applies component-wise for vectors.
    Sign,

    /// Sine operator.
    Sin,

    /// Tangent operator.
    Tan,

    /// Unpacking operator from `u32` to `vec4<f32>` (signed normalized).
    ///
    /// Unpack the `u32` into four signed integral `i8` value in `[-128:127]`,
    /// then convert each value to a signed normalized `f32` value in `[-1:1]`.
    Unpack4x8snorm,

    /// Unpacking operator from `u32` to `vec4<f32>` (unsigned normalized).
    ///
    /// Unpack the `u32` into four unsigned integral `u8` value in `[0:255]`,
    /// then convert each value to an unsigned normalized `f32` value in
    /// `[0:1]`.
    Unpack4x8unorm,

    /// Get the fourth component of a vector.
    ///
    /// This is only valid for vectors of rank 4.
    W,

    /// Get the first component of a scalar or vector.
    ///
    /// For scalar, return the value itself. For vectors, return the first
    /// component.
    X,

    /// Get the second component of a vector.
    Y,

    /// Get the third component of a vector.
    ///
    /// This is only valid for vectors of rank 3 or more.
    Z,
}

impl UnaryOperator {
    /// Check if a unary operator is called via a functional-style call.
    ///
    /// Functional-style calls are in the form `op(inner)`, like `abs(x)` for
    /// example, while non-functional ones are in the form `inner.op`,
    /// like `v.x` for example. This check is used for formatting the WGSL
    /// code emitted during evaluation of a binary operation expression.
    pub fn is_functional(&self) -> bool {
        !matches!(
            *self,
            UnaryOperator::X | UnaryOperator::Y | UnaryOperator::Z | UnaryOperator::W
        )
    }
}

impl ToWgslString for UnaryOperator {
    fn to_wgsl_string(&self) -> String {
        match *self {
            UnaryOperator::Abs => "abs".to_string(),
            UnaryOperator::All => "all".to_string(),
            UnaryOperator::Any => "any".to_string(),
            UnaryOperator::Ceil => "ceil".to_string(),
            UnaryOperator::Cos => "cos".to_string(),
            UnaryOperator::Exp => "exp".to_string(),
            UnaryOperator::Exp2 => "exp2".to_string(),
            UnaryOperator::Floor => "floor".to_string(),
            UnaryOperator::Fract => "fract".to_string(),
            UnaryOperator::Length => "length".to_string(),
            UnaryOperator::Log => "log".to_string(),
            UnaryOperator::Log2 => "log2".to_string(),
            UnaryOperator::Normalize => "normalize".to_string(),
            UnaryOperator::Pack4x8snorm => "pack4x8snorm".to_string(),
            UnaryOperator::Pack4x8unorm => "pack4x8unorm".to_string(),
            UnaryOperator::Saturate => "saturate".to_string(),
            UnaryOperator::Sign => "sign".to_string(),
            UnaryOperator::Sin => "sin".to_string(),
            UnaryOperator::Tan => "tan".to_string(),
            UnaryOperator::Unpack4x8snorm => "unpack4x8snorm".to_string(),
            UnaryOperator::Unpack4x8unorm => "unpack4x8unorm".to_string(),
            UnaryOperator::W => "w".to_string(),
            UnaryOperator::X => "x".to_string(),
            UnaryOperator::Y => "y".to_string(),
            UnaryOperator::Z => "z".to_string(),
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

    /// Cross product operator.
    ///
    /// Returns the cross product of the left and right operands. Only valid for
    /// vector type operands of size 3. Always produce a vector result of the
    /// same size.
    Cross,

    /// Distance operator.
    ///
    /// Returns the distance between two floating point scalar or vectors, that
    /// is `length(right - left)`.
    Distance,

    /// Division operator.
    ///
    /// Returns the left operand divided by the right operand. Only valid for
    /// numeric operands.
    Div,

    /// Dot product operator.
    ///
    /// Returns the dot product of the left and right operands. Only valid for
    /// vector type operands. Always produce a scalar floating-point result.
    Dot,

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

    /// Maximum operator.
    ///
    /// Returns the maximum value of its left and right operands. Only valid for
    /// numeric types. If the operands are vectors, they must be of the same
    /// rank, and the result is a vector of that rank and same element
    /// scalar type.
    Max,

    /// Minimum operator.
    ///
    /// Returns the minimum value of its left and right operands. Only valid for
    /// numeric types. If the operands are vectors, they must be of the same
    /// rank, and the result is a vector of that rank and same element
    /// scalar type.
    Min,

    /// Multiply operator.
    ///
    /// Returns the product of its operands. Only valid for numeric operands.
    Mul,

    /// Remainder operator.
    ///
    /// Returns the remainder of the division of the first operand by the
    /// second. Only valid for numeric types. If the operands are vectors,
    /// they must be of the same rank, and the result is a vector of that
    /// rank and same element scalar type.
    Remainder,

    /// Stepping operator.
    ///
    /// Returns `1.0` if the left operand is less than or equal to the right
    /// operand, or `0.0` otherwise. Only valid for floating scalar or vectors
    /// of the same rank, and applied component-wise for vectors.
    Step,

    /// Subtraction operator.
    ///
    /// Returns the difference between its left and right operands. Only valid
    /// for numeric operands.
    Sub,

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
            | BinaryOperator::Div
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterThanOrEqual
            | BinaryOperator::LessThan
            | BinaryOperator::LessThanOrEqual
            | BinaryOperator::Mul
            | BinaryOperator::Remainder
            | BinaryOperator::Sub => false,
            BinaryOperator::Cross
            | BinaryOperator::Distance
            | BinaryOperator::Dot
            | BinaryOperator::Max
            | BinaryOperator::Min
            | BinaryOperator::Step
            | BinaryOperator::UniformRand => true,
        }
    }
}

impl ToWgslString for BinaryOperator {
    fn to_wgsl_string(&self) -> String {
        match *self {
            BinaryOperator::Add => "+".to_string(),
            BinaryOperator::Cross => "cross".to_string(),
            BinaryOperator::Distance => "distance".to_string(),
            BinaryOperator::Div => "/".to_string(),
            BinaryOperator::Dot => "dot".to_string(),
            BinaryOperator::GreaterThan => ">".to_string(),
            BinaryOperator::GreaterThanOrEqual => ">=".to_string(),
            BinaryOperator::LessThan => "<".to_string(),
            BinaryOperator::LessThanOrEqual => "<=".to_string(),
            BinaryOperator::Max => "max".to_string(),
            BinaryOperator::Min => "min".to_string(),
            BinaryOperator::Mul => "*".to_string(),
            BinaryOperator::Remainder => "%".to_string(),
            BinaryOperator::Step => "step".to_string(),
            BinaryOperator::Sub => "-".to_string(),
            BinaryOperator::UniformRand => "rand_uniform".to_string(),
        }
    }
}

/// Ternary operator.
///
/// Operator applied between three operands. The type of the operands and the
/// result are not necessarily the same. Valid operand types depend on the
/// operator itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum TernaryOperator {
    /// Linear blend ("mix") operator.
    ///
    /// Returns the linear blend between the first and second argument, based on
    /// the fraction of the third argument. If the operands are vectors, they
    /// must be of the same rank, and the result is a vector of that rank
    /// and same element scalar type.
    ///
    /// The linear blend of `x` and `y` with fraction `t` is equivalent to `x *
    /// (1 - t) + y * t`.
    Mix,

    /// Smooth stepping operator.
    ///
    /// Returns the smooth Hermitian interpolation between the first and second
    /// argument, calculated at the third argument. If the operands are vectors,
    /// they must be of the same rank, and the result is a vector of that
    /// rank and same element scalar type.
    ///
    /// The smooth stepping of `low` and `high` at position `x` is equivalent to
    /// `t * t * (3. - 2. * t)` where `t = clamp((x - low) / (high - low))`
    /// represents the fractional position of `x` between `low` and `high`.
    ///
    /// The result is always a floating point scalar in \[0:1\].
    SmoothStep,
}

impl ToWgslString for TernaryOperator {
    fn to_wgsl_string(&self) -> String {
        match *self {
            TernaryOperator::Mix => "mix".to_string(),
            TernaryOperator::SmoothStep => "smoothstep".to_string(),
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

    /// Create a new writer expression representing the alpha cutoff value used
    /// for alpha masking.
    ///
    /// This expression is only valid when used in the context of the fragment
    /// shader, in the render context.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let mut w = ExprWriter::new();
    /// let x = w.alpha_cutoff(); // x = alpha_cutoff;
    /// ```
    pub fn alpha_cutoff(&self) -> WriterExpr {
        self.push(Expr::BuiltIn(BuiltInExpr::new(
            BuiltInOperator::AlphaCutoff,
        )))
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
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn any(self) -> Self {
        self.unary_op(UnaryOperator::Any)
    }

    /// Apply the "ceil" operator to the current float scalar or vector
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
    /// // Ceil: `y = ceil(x);`
    /// let y = x.ceil();
    /// ```
    #[inline]
    pub fn ceil(self) -> Self {
        self.unary_op(UnaryOperator::Ceil)
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
    /// // Cos: `y = cos(x);`
    /// let y = x.cos();
    /// ```
    #[inline]
    pub fn cos(self) -> Self {
        self.unary_op(UnaryOperator::Cos)
    }

    /// Apply the "exp" operator to the current float scalar or vector
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
    /// // Exp: `y = exp(x);`
    /// let y = x.exp();
    /// ```
    #[inline]
    pub fn exp(self) -> Self {
        self.unary_op(UnaryOperator::Exp)
    }

    /// Apply the "exp2" operator to the current float scalar or vector
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
    /// // Exp2: `y = exp2(x);`
    /// let y = x.exp2();
    /// ```
    #[inline]
    pub fn exp2(self) -> Self {
        self.unary_op(UnaryOperator::Exp2)
    }

    /// Apply the "floor" operator to the current float scalar or vector
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
    /// // Floor: `y = floor(x);`
    /// let y = x.floor();
    /// ```
    #[inline]
    pub fn floor(self) -> Self {
        self.unary_op(UnaryOperator::Floor)
    }

    /// Apply the "fract" operator to the current float scalar or vector
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
    /// // Fract: `y = fract(x);`
    /// let y = x.fract();
    /// ```
    #[inline]
    pub fn fract(self) -> Self {
        self.unary_op(UnaryOperator::Fract)
    }

    /// Apply the "length" operator to the current float scalar or vector
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
    /// // Length: `y = length(x);`
    /// let y = x.length();
    /// ```
    #[inline]
    pub fn length(self) -> Self {
        self.unary_op(UnaryOperator::Length)
    }

    /// Apply the "log" operator to the current float scalar or vector
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
    /// // Log: `y = log(x);`
    /// let y = x.log();
    /// ```
    #[inline]
    pub fn log(self) -> Self {
        self.unary_op(UnaryOperator::Log)
    }

    /// Apply the "log2" operator to the current float scalar or vector
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
    /// // Log2: `y = log2(x);`
    /// let y = x.log2();
    /// ```
    #[inline]
    pub fn log2(self) -> Self {
        self.unary_op(UnaryOperator::Log2)
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
    #[inline]
    pub fn normalized(self) -> Self {
        self.unary_op(UnaryOperator::Normalize)
    }

    /// Apply the "pack4x8snorm" operator to the current 4-component float
    /// vector expression.
    ///
    /// This is a unary operator, which applies to 4-component float vector
    /// operand expressions to produce a single `u32` scalar expression.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec4;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec4<f32>(-1., 1., 0., 7.2);`.
    /// let x = w.lit(Vec4::new(-1., 1., 0., 7.2));
    ///
    /// // Pack: `y = pack4x8snorm(x);`
    /// let y = x.pack4x8snorm(); // 0x7F007FFFu32
    /// ```
    #[inline]
    pub fn pack4x8snorm(self) -> Self {
        self.unary_op(UnaryOperator::Pack4x8snorm)
    }

    /// Apply the "pack4x8unorm" operator to the current 4-component float
    /// vector expression.
    ///
    /// This is a unary operator, which applies to 4-component float vector
    /// operand expressions to produce a single `u32` scalar expression.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec4;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `x = vec4<f32>(-1., 1., 0., 7.2);`.
    /// let x = w.lit(Vec4::new(-1., 1., 0., 7.2));
    ///
    /// // Pack: `y = pack4x8unorm(x);`
    /// let y = x.pack4x8unorm(); // 0xFF00FF00u32
    /// ```
    #[inline]
    pub fn pack4x8unorm(self) -> Self {
        self.unary_op(UnaryOperator::Pack4x8unorm)
    }

    /// Apply the "sign" operator to the current float scalar or vector
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
    /// // Sign: `y = sign(x);`
    /// let y = x.sign();
    /// ```
    #[inline]
    pub fn sign(self) -> Self {
        self.unary_op(UnaryOperator::Sign)
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
    #[inline]
    pub fn sin(self) -> Self {
        self.unary_op(UnaryOperator::Sin)
    }

    /// Apply the "tan" operator to the current float scalar or vector
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
    /// // Tan: `y = tan(x);`
    /// let y = x.tan();
    /// ```
    #[inline]
    pub fn tan(self) -> Self {
        self.unary_op(UnaryOperator::Tan)
    }

    /// Apply the "unpack4x8snorm" operator to the current `u32` scalar
    /// expression.
    ///
    /// This is a unary operator, which applies to `u32` scalar operand
    /// expressions to produce a 4-component floating point vector of signed
    /// normalized components in `[-1:1]`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `y = 0x7F007FFFu32;`.
    /// let y = w.lit(0x7F007FFFu32);
    ///
    /// // Unpack: `x = unpack4x8snorm(y);`
    /// let x = y.unpack4x8snorm(); // vec4<f32>(-1., 1., 0., 7.2)
    /// ```
    #[inline]
    pub fn unpack4x8snorm(self) -> Self {
        self.unary_op(UnaryOperator::Unpack4x8snorm)
    }

    /// Apply the "unpack4x8unorm" operator to the current `u32` scalar
    /// expression.
    ///
    /// This is a unary operator, which applies to `u32` scalar operand
    /// expressions to produce a 4-component floating point vector of unsigned
    /// normalized components in `[0:1]`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `y = 0xFF00FF00u32;`.
    /// let y = w.lit(0xFF00FF00u32);
    ///
    /// // Unpack: `x = unpack4x8unorm(y);`
    /// let x = y.unpack4x8unorm(); // vec4<f32>(-1., 1., 0., 7.2)
    /// ```
    #[inline]
    pub fn unpack4x8unorm(self) -> Self {
        self.unary_op(UnaryOperator::Unpack4x8unorm)
    }

    /// Apply the "saturate" operator to the current float scalar or vector
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
    /// // Saturate: `y = saturate(x);`
    /// let y = x.saturate();
    /// ```
    #[inline]
    pub fn saturate(self) -> Self {
        self.unary_op(UnaryOperator::Saturate)
    }

    /// Get the first component of a scalar or vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `v = vec3<f32>(1., 1., 1.);`.
    /// let v = w.lit(Vec3::ONE);
    ///
    /// // f = v.x;`
    /// let f = v.x();
    /// ```
    #[inline]
    pub fn x(self) -> Self {
        self.unary_op(UnaryOperator::X)
    }

    /// Get the second component of a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `v = vec3<f32>(1., 1., 1.);`.
    /// let v = w.lit(Vec3::ONE);
    ///
    /// // f = v.y;`
    /// let f = v.y();
    /// ```
    #[inline]
    pub fn y(self) -> Self {
        self.unary_op(UnaryOperator::Y)
    }

    /// Get the third component of a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `v = vec3<f32>(1., 1., 1.);`.
    /// let v = w.lit(Vec3::ONE);
    ///
    /// // f = v.z;`
    /// let f = v.z();
    /// ```
    #[inline]
    pub fn z(self) -> Self {
        self.unary_op(UnaryOperator::Z)
    }

    /// Get the fourth component of a vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `v = vec3<f32>(1., 1., 1.);`.
    /// let v = w.lit(Vec3::ONE);
    ///
    /// // f = v.w;`
    /// let f = v.w();
    /// ```
    #[inline]
    pub fn w(self) -> Self {
        self.unary_op(UnaryOperator::W)
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
    #[inline]
    pub fn add(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Add)
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
    #[inline]
    pub fn cross(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Cross)
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
    #[inline]
    pub fn dot(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Dot)
    }

    /// Calculate the distance between the current expression and another
    /// expression.
    ///
    /// This is a binary operator.
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
    /// // The distance between the vectors `z = distance(x, y);`.
    /// let z = x.distance(y);
    /// ```
    #[inline]
    pub fn distance(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Distance)
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
    #[inline]
    pub fn div(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Div)
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
    #[inline]
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
    #[inline]
    pub fn gt(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::GreaterThan)
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
    #[inline]
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
    #[inline]
    pub fn lt(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::LessThan)
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
    #[inline]
    pub fn max(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Max)
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
    #[inline]
    pub fn min(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Min)
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
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Mul)
    }

    /// Calculate the remainder of the division of the current expression by
    /// another expression.
    ///
    /// This is a binary operator.
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
    /// // The remainder of the division `z = x % y;`.
    /// let z = x.rem(y);
    /// ```
    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub fn rem(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Remainder)
    }

    /// Calculate the step of a value with respect to a reference.
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
    /// // A literal expression `x = vec3<f32>(3., -2.);`.
    /// let x = w.lit(Vec3::new(3., -2., 8.));
    ///
    /// // An edge reference `e = vec3<f32>(1., 5.);`.
    /// let e = w.lit(Vec3::new(1., 5., 8.));
    ///
    /// // The step value
    /// let s = x.step(e); // == vec3<f32>(1., 0., 1.)
    /// ```
    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub fn step(self, edge: Self) -> Self {
        // Note: order is step(edge, x) but x.step(edge)
        edge.binary_op(self, BinaryOperator::Step)
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
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::Sub)
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
    #[inline]
    pub fn uniform(self, other: Self) -> Self {
        self.binary_op(other, BinaryOperator::UniformRand)
    }

    fn ternary_op(self, second: Self, third: Self, op: TernaryOperator) -> Self {
        assert_eq!(self.module, second.module);
        assert_eq!(self.module, third.module);
        let first = self.expr;
        let second = second.expr;
        let third = third.expr;
        let expr = self.module.borrow_mut().push(Expr::Ternary {
            op,
            first,
            second,
            third,
        });
        WriterExpr {
            expr,
            module: self.module,
        }
    }

    /// Blending linearly ("mix") two expressions with the fraction provided by
    /// a third expression.
    ///
    /// This is a ternary operator, which applies component-wise to vector
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
    /// // Another literal expression `y = vec2<f32>(1., 4.);`.
    /// let y = w.lit(Vec2::new(1., 4.));
    ///
    /// // A fraction `t = 0.5;`.
    /// let t = w.lit(0.25);
    ///
    /// // The linear blend of x and y via t: z = (1 - t) * x + y * t
    /// let z = x.mix(y, t); // == vec2<f32>(2.5, -0.5)
    /// ```
    #[inline]
    pub fn mix(self, other: Self, fraction: Self) -> Self {
        self.ternary_op(other, fraction, TernaryOperator::Mix)
    }

    /// Calculate the smooth Hermite interpolation in \[0:1\] of the current
    /// value taken between the given bounds.
    ///
    /// This is a ternary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec2;
    /// # let mut w = ExprWriter::new();
    /// // A literal expression `low = vec2<f32>(3., -2.);`.
    /// let low = w.lit(Vec2::new(3., -2.));
    ///
    /// // Another literal expression `high = vec2<f32>(1., 4.);`.
    /// let high = w.lit(Vec2::new(1., 4.));
    ///
    /// // A point `x = vec2<f32>(2., 1.);` between `low` and `high`.
    /// let x = w.lit(Vec2::new(2., 1.));
    ///
    /// // The smooth Hermite interpolation: `t = smoothstep(low, high, x)`
    /// let t = x.smoothstep(low, high); // == 0.5
    /// ```
    #[inline]
    pub fn smoothstep(self, low: Self, high: Self) -> Self {
        // Note: order is smoothstep(low, high, x) but x.smoothstep(low, high)
        low.ternary_op(high, self, TernaryOperator::SmoothStep)
    }

    /// Cast an expression to a different type.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec2;
    /// # let mut w = ExprWriter::new();
    /// let x = w.lit(Vec2::new(3., -2.));
    /// let y = x.cast(VectorType::VEC3I); // x = vec3<i32>(particle.position);
    /// ```
    pub fn cast(self, target: impl Into<ValueType>) -> Self {
        let target = target.into();
        let expr = self
            .module
            .borrow_mut()
            .push(Expr::Cast(CastExpr::new(self.expr, target)));
        WriterExpr {
            expr,
            module: self.module,
        }
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
    #[inline]
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

impl std::ops::Rem<WriterExpr> for WriterExpr {
    type Output = WriterExpr;

    #[inline]
    fn rem(self, rhs: WriterExpr) -> Self::Output {
        self.rem(rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::Property, InitContext, MatrixType, ScalarValue, VectorType};

    use super::*;
    use bevy::{prelude::*, utils::HashSet};

    #[test]
    fn module() {
        let mut m = Module::default();

        let unknown = ExprHandle::new(NonZeroU32::new(1).unwrap());
        assert!(m.get(unknown).is_none());
        assert!(m.get_mut(unknown).is_none());
        assert!(matches!(
            m.try_get(unknown),
            Err(ExprError::InvalidExprHandleError(_))
        ));
        assert!(matches!(
            m.try_get_mut(unknown),
            Err(ExprError::InvalidExprHandleError(_))
        ));

        let x = m.lit(5.);
        let mut expected = Expr::Literal(LiteralExpr::new(5.));
        assert_eq!(m.get(x), Some(&expected));
        assert_eq!(m.get_mut(x), Some(&mut expected));
        assert_eq!(m.try_get(x), Ok(&expected));
        assert_eq!(m.try_get_mut(x), Ok(&mut expected));
    }

    #[test]
    fn local_var() {
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut ctx = InitContext::new(&property_layout, &particle_layout);
        let mut h = HashSet::new();
        for _ in 0..100 {
            let v = ctx.make_local_var();
            assert!(h.insert(v));
        }
    }

    #[test]
    fn make_fn() {
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut ctx = InitContext::new(&property_layout, &particle_layout);
        let mut module = Module::default();

        // Make a function
        let func_name = "my_func";
        let args = "arg0: i32, arg1: f32";
        assert!(ctx
            .make_fn(func_name, args, &mut module, &mut |m, ctx| {
                m.lit(3.);
                let v = ctx.make_local_var();
                assert_eq!(v, "var0");
                let code = String::new();
                Ok(code)
            })
            .is_ok());

        // The local function context doesn't influence the outer caller context
        let v = ctx.make_local_var();
        assert_eq!(v, "var0");

        // However the module is common to the outer caller and the function
        assert!(!module.expressions.is_empty());
    }

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
        let particle_layout = ParticleLayout::default();
        let m = w.finish();
        let mut context = InitContext::new(&property_layout, &particle_layout);

        // Evaluate the expression
        let x = m.try_get(x).unwrap();
        let s = x.eval(&m, &mut context).unwrap();
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
        let rem = m.rem(x, y);
        let lt = m.lt(x, y);
        let le = m.le(x, y);
        let gt = m.gt(x, y);
        let ge = m.ge(x, y);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut ctx = InitContext::new(&property_layout, &particle_layout);

        for (expr, op) in [
            (add, "+"),
            (sub, "-"),
            (mul, "*"),
            (div, "/"),
            (rem, "%"),
            (lt, "<"),
            (le, "<="),
            (gt, ">"),
            (ge, ">="),
        ] {
            let expr = ctx.eval(&m, expr);
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

            let property_layout = PropertyLayout::default();
            let particle_layout = ParticleLayout::default();
            let mut ctx = InitContext::new(&property_layout, &particle_layout);

            let expr = ctx.eval(&m, value);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            assert_eq!(expr, format!("sim_params.{}", op.name()));
        }

        // BuiltInOperator::Rand (which has side effect)
        for (scalar_type, prefix) in [
            (ScalarType::Bool, "b"),
            (ScalarType::Float, "f"),
            (ScalarType::Int, "i"),
            (ScalarType::Uint, "u"),
        ] {
            let value = m.builtin(BuiltInOperator::Rand(scalar_type.into()));

            // Scalar form
            {
                let property_layout = PropertyLayout::default();
                let particle_layout = ParticleLayout::default();
                let mut ctx = InitContext::new(&property_layout, &particle_layout);

                let expr = ctx.eval(&m, value);
                assert!(expr.is_ok());
                let expr = expr.unwrap();
                assert_eq!(expr, "var0");
                assert_eq!(ctx.init_code, format!("let var0 = {}rand();\n", prefix));
            }

            // Vector form
            for count in 2..=4 {
                let vec = m.builtin(BuiltInOperator::Rand(
                    VectorType::new(scalar_type, count).into(),
                ));

                let property_layout = PropertyLayout::default();
                let particle_layout = ParticleLayout::default();
                let mut ctx = InitContext::new(&property_layout, &particle_layout);

                let expr = ctx.eval(&m, vec);
                assert!(expr.is_ok());
                let expr = expr.unwrap();
                assert_eq!(expr, "var0");
                assert_eq!(
                    ctx.init_code,
                    format!("let var0 = {}rand{}();\n", prefix, count)
                );
            }
        }
    }

    #[test]
    fn unary_expr() {
        let mut m = Module::default();

        let x = m.attr(Attribute::POSITION);
        let y = m.lit(Vec3::new(1., -3.1, 6.99));
        let z = m.lit(BVec3::new(false, true, false));
        let w = m.lit(Vec4::W);
        let v = m.lit(Vec4::new(-1., 1., 0., 7.2));
        let us = m.lit(0x0u32);
        let uu = m.lit(0x0u32);

        let abs = m.abs(x);
        let all = m.all(z);
        let any = m.any(z);
        let ceil = m.ceil(y);
        let cos = m.cos(y);
        let exp = m.exp(y);
        let exp2 = m.exp2(y);
        let floor = m.floor(y);
        let fract = m.fract(y);
        let length = m.length(y);
        let log = m.log(y);
        let log2 = m.log2(y);
        let norm = m.normalize(y);
        let pack4x8snorm = m.pack4x8snorm(v);
        let pack4x8unorm = m.pack4x8unorm(v);
        let saturate = m.saturate(y);
        let sign = m.sign(y);
        let sin = m.sin(y);
        let tan = m.tan(y);
        let unpack4x8snorm = m.unpack4x8snorm(us);
        let unpack4x8unorm = m.unpack4x8unorm(uu);
        let comp_x = m.x(w);
        let comp_y = m.y(w);
        let comp_z = m.z(w);
        let comp_w = m.w(w);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut ctx = InitContext::new(&property_layout, &particle_layout);

        for (expr, op, inner) in [
            (
                abs,
                "abs",
                &format!("particle.{}", Attribute::POSITION.name())[..],
            ),
            (all, "all", "vec3<bool>(false,true,false)"),
            (any, "any", "vec3<bool>(false,true,false)"),
            (ceil, "ceil", "vec3<f32>(1.,-3.1,6.99)"),
            (cos, "cos", "vec3<f32>(1.,-3.1,6.99)"),
            (exp, "exp", "vec3<f32>(1.,-3.1,6.99)"),
            (exp2, "exp2", "vec3<f32>(1.,-3.1,6.99)"),
            (floor, "floor", "vec3<f32>(1.,-3.1,6.99)"),
            (fract, "fract", "vec3<f32>(1.,-3.1,6.99)"),
            (length, "length", "vec3<f32>(1.,-3.1,6.99)"),
            (log, "log", "vec3<f32>(1.,-3.1,6.99)"),
            (log2, "log2", "vec3<f32>(1.,-3.1,6.99)"),
            (norm, "normalize", "vec3<f32>(1.,-3.1,6.99)"),
            (pack4x8snorm, "pack4x8snorm", "vec4<f32>(-1.,1.,0.,7.2)"),
            (pack4x8unorm, "pack4x8unorm", "vec4<f32>(-1.,1.,0.,7.2)"),
            (saturate, "saturate", "vec3<f32>(1.,-3.1,6.99)"),
            (sign, "sign", "vec3<f32>(1.,-3.1,6.99)"),
            (sin, "sin", "vec3<f32>(1.,-3.1,6.99)"),
            (tan, "tan", "vec3<f32>(1.,-3.1,6.99)"),
            (unpack4x8snorm, "unpack4x8snorm", "0"),
            (unpack4x8unorm, "unpack4x8unorm", "0"),
        ] {
            let expr = ctx.eval(&m, expr);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            assert_eq!(expr, format!("{}({})", op, inner));
        }

        for (expr, op, inner) in [
            (comp_x, "x", "vec4<f32>(0.,0.,0.,1.)"),
            (comp_y, "y", "vec4<f32>(0.,0.,0.,1.)"),
            (comp_z, "z", "vec4<f32>(0.,0.,0.,1.)"),
            (comp_w, "w", "vec4<f32>(0.,0.,0.,1.)"),
        ] {
            let expr = ctx.eval(&m, expr);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            assert_eq!(expr, format!("{}.{}", inner, op));
        }
    }

    #[test]
    fn binary_expr() {
        let mut m = Module::default();

        let x = m.attr(Attribute::POSITION);
        let y = m.lit(Vec3::ONE);

        let cross = m.cross(x, y);
        let dist = m.distance(x, y);
        let dot = m.dot(x, y);
        let min = m.min(x, y);
        let max = m.max(x, y);
        let step = m.step(x, y);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut ctx = InitContext::new(&property_layout, &particle_layout);

        for (expr, op) in [
            (cross, "cross"),
            (dist, "distance"),
            (dot, "dot"),
            (min, "min"),
            (max, "max"),
            (step, "step"),
        ] {
            let expr = ctx.eval(&m, expr);
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

    #[test]
    fn ternary_expr() {
        let mut m = Module::default();

        let x = m.attr(Attribute::POSITION);
        let y = m.lit(Vec3::ONE);
        let t = m.lit(0.3);

        let mix = m.mix(x, y, t);
        let smoothstep = m.smoothstep(x, y, x);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut ctx = InitContext::new(&property_layout, &particle_layout);

        for (expr, op, third) in [(mix, "mix", t), (smoothstep, "smoothstep", x)] {
            let expr = ctx.eval(&m, expr);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            let third = ctx.eval(&m, third).unwrap();
            assert_eq!(
                expr,
                format!(
                    "{}(particle.{}, vec3<f32>(1.,1.,1.), {})",
                    op,
                    Attribute::POSITION.name(),
                    third
                )
            );
        }
    }

    #[test]
    fn cast_expr() {
        let mut m = Module::default();

        let x = m.attr(Attribute::POSITION);
        let y = m.lit(IVec2::ONE);
        let z = m.lit(0.3);
        let w = m.lit(false);

        let cx = m.cast(x, VectorType::VEC3I);
        let cy = m.cast(y, VectorType::VEC2U);
        let cz = m.cast(z, ScalarType::Int);
        let cw = m.cast(w, ScalarType::Uint);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut ctx = InitContext::new(&property_layout, &particle_layout);

        for (expr, cast, target) in [
            (x, cx, ValueType::Vector(VectorType::VEC3I)),
            (y, cy, VectorType::VEC2U.into()),
            (z, cz, ScalarType::Int.into()),
            (w, cw, ScalarType::Uint.into()),
        ] {
            let expr = ctx.eval(&m, expr);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            let cast = ctx.eval(&m, cast);
            assert!(cast.is_ok());
            let cast = cast.unwrap();
            assert_eq!(cast, format!("{}({})", target.to_wgsl_string(), expr));
        }
    }

    #[test]
    #[should_panic]
    fn invalid_cast_vector_to_scalar() {
        let mut m = Module::default();
        let x = m.lit(Vec2::ONE);
        let _ = m.cast(x, ScalarType::Float);
    }

    #[test]
    #[should_panic]
    fn invalid_cast_matrix_to_scalar() {
        let mut m = Module::default();
        let x = m.lit(Value::Matrix(Mat4::ZERO.into()));
        let _ = m.cast(x, ScalarType::Float);
    }

    #[test]
    #[should_panic]
    fn invalid_cast_matrix_to_vector() {
        let mut m = Module::default();
        let x = m.lit(Value::Matrix(Mat4::ZERO.into()));
        let _ = m.cast(x, VectorType::VEC4F);
    }

    #[test]
    #[should_panic]
    fn invalid_cast_scalar_to_matrix() {
        let mut m = Module::default();
        let x = m.lit(3.);
        let _ = m.cast(x, MatrixType::MAT3X3F);
    }

    #[test]
    #[should_panic]
    fn invalid_cast_vector_to_matrix() {
        let mut m = Module::default();
        let x = m.lit(Vec3::ZERO);
        let _ = m.cast(x, MatrixType::MAT2X4F);
    }

    #[test]
    fn cast_expr_new() {
        let mut m = Module::default();

        let x = m.attr(Attribute::POSITION);
        let c = CastExpr::new(x, VectorType::VEC3F);
        assert_eq!(c.value_type(), ValueType::Vector(VectorType::VEC3F));
        assert_eq!(c.is_valid(&m), Some(true));

        let x = m.attr(Attribute::POSITION);
        let c = CastExpr::new(x, ScalarType::Bool);
        assert_eq!(c.value_type(), ValueType::Scalar(ScalarType::Bool));
        assert_eq!(c.is_valid(&m), Some(false)); // invalid cast vector -> scalar

        let y = m.prop("my_prop");
        let c = CastExpr::new(y, MatrixType::MAT2X3F);
        assert_eq!(c.value_type(), ValueType::Matrix(MatrixType::MAT2X3F));
        assert_eq!(c.is_valid(&m), None); // properties' value_type() is unknown
    }

    #[test]
    fn side_effect() {
        let mut m = Module::default();

        // Adding the same cloned expression with side effect to itself should yield
        // twice the value, and not two separate evaluations of the expression.
        // CORRECT:
        //   let r = frand();
        //   r + r
        // INCORRECT:
        //   frand() + frand()

        let r = m.builtin(BuiltInOperator::Rand(ScalarType::Float.into()));
        let r2 = r;
        let r3 = r2;
        let a = m.add(r, r2);
        let b = m.mix(r, r2, r3);
        let c = m.abs(a);

        {
            let property_layout = PropertyLayout::default();
            let particle_layout = ParticleLayout::default();
            let mut ctx = InitContext::new(&property_layout, &particle_layout);
            let value = ctx.eval(&m, a).unwrap();
            assert_eq!(value, "(var0) + (var0)");
            assert_eq!(ctx.init_code, "let var0 = frand();\n");
        }

        {
            let property_layout = PropertyLayout::default();
            let particle_layout = ParticleLayout::default();
            let mut ctx = InitContext::new(&property_layout, &particle_layout);
            let value = ctx.eval(&m, b).unwrap();
            assert_eq!(value, "mix(var0, var0, var0)");
            assert_eq!(ctx.init_code, "let var0 = frand();\n");
        }

        {
            let property_layout = PropertyLayout::default();
            let particle_layout = ParticleLayout::default();
            let mut ctx = InitContext::new(&property_layout, &particle_layout);
            let value = ctx.eval(&m, c).unwrap();
            assert_eq!(value, "abs((var0) + (var0))");
            assert_eq!(ctx.init_code, "let var0 = frand();\n");
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
