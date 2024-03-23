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
//! let prop = w.add_property("my_property", 3.0.into());
//! let expr = w.lit(3.42).max(w.prop(prop));
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

use bevy::{
    reflect::{FromReflect, Reflect},
    utils::thiserror::Error,
};
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, hash::Hash, num::NonZeroU32, rc::Rc};

use crate::{
    calc_func_id, gradient::Lerp, Attribute, Gradient, GradientKey, ModifierContext,
    ParticleLayout, Property, PropertyLayout, ScalarType, ToWgslString, ValueType, ValueTypeOf,
    VectorType,
};

use super::Value;

/// A one-based ID into a collection of a [`Module`].
type Id = NonZeroU32;

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
#[repr(transparent)]
#[serde(transparent)]
pub struct ExprHandle {
    id: Id,
}

impl ExprHandle {
    /// Create a new handle from a 1-based [`Id`].
    #[allow(dead_code)]
    fn new(id: Id) -> Self {
        Self { id }
    }

    /// Create a new handle from a 1-based [`Id`] as a `usize`, for cases where
    /// the index is known to be non-zero already.
    #[allow(unsafe_code)]
    unsafe fn new_unchecked(id: usize) -> Self {
        debug_assert!(id != 0);
        Self {
            id: NonZeroU32::new_unchecked(id as u32),
        }
    }

    /// Get the zero-based index into the array of the module.
    fn index(&self) -> usize {
        (self.id.get() - 1) as usize
    }
}

/// Handle of a property inside a given [`Module`].
///
/// A handle uniquely references a [`Property`] stored inside a [`Module`]. It's
/// a lightweight representation, similar to a simple array index. For this
/// reason, it's easily copyable. However it's also lacking any kind of error
/// checking, and mixing handles to different modules produces undefined
/// behaviors (like an index does when indexing the wrong array).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect, Serialize, Deserialize,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct PropertyHandle {
    id: Id,
}

impl PropertyHandle {
    /// Create a new handle from a 1-based [`Id`].
    #[allow(dead_code)]
    fn new(id: Id) -> Self {
        Self { id }
    }

    /// Create a new handle from a 1-based [`Id`] as a `usize`, for cases where
    /// the index is known to be non-zero already.
    #[allow(unsafe_code)]
    unsafe fn new_unchecked(id: usize) -> Self {
        debug_assert!(id != 0);
        Self {
            id: NonZeroU32::new_unchecked(id as u32),
        }
    }

    /// Get the zero-based index into the array of the module.
    fn index(&self) -> usize {
        (self.id.get() - 1) as usize
    }
}

/// Handle of a function inside a given [`Module`].
///
/// A handle uniquely references a [`Func`] stored inside a [`Module`]. It's a
/// lightweight representation, similar to a simple array index. For this
/// reason, it's easily copyable. However it's also lacking any kind of error
/// checking, and mixing handles to different modules produces undefined
/// behaviors (like an index does when indexing the wrong array).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct FuncHandle {
    id: Id,
}

impl FuncHandle {
    /// Create a new handle from a 1-based [`Id`].
    fn new(id: Id) -> Self {
        Self { id }
    }

    /// Create a new handle from a 1-based [`Id`] as a `usize`, for cases where
    /// the index is known to be non-zero already.
    #[allow(unsafe_code)]
    unsafe fn new_unchecked(id: usize) -> Self {
        debug_assert!(id != 0);
        Self {
            id: NonZeroU32::new_unchecked(id as u32),
        }
    }

    /// Get the zero-based index into the array of the module.
    fn index(&self) -> usize {
        (self.id.get() - 1) as usize
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
    /// Expressions defined in the module.
    expressions: Vec<Expr>,
    /// Properties used as part of a [`PropertyExpr`].
    properties: Vec<Property>,
    /// Functions used by the module.
    functions: Vec<Func>,
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

impl Module {
    /// Create a new module from an existing collection of expressions.
    pub fn from_raw(expr: Vec<Expr>) -> Self {
        Self {
            expressions: expr,
            properties: vec![],
            functions: vec![],
        }
    }

    /// Add a new property to the module.
    ///
    /// See [`Property`] for more details on what effect properties are.
    ///
    /// # Panics
    ///
    /// Panics if a property with the same name already exists.
    pub fn add_property(
        &mut self,
        name: impl Into<String>,
        default_value: Value,
    ) -> PropertyHandle {
        let name = name.into();
        assert!(!self.properties.iter().any(|p| p.name() == name));
        self.properties.push(Property::new(name, default_value));
        // SAFETY - We just pushed a new property into the array, so its length is
        // non-zero.
        #[allow(unsafe_code)]
        unsafe {
            PropertyHandle::new_unchecked(self.properties.len())
        }
    }

    /// Get an existing property by handle.
    ///
    /// Existing properties are properties previously created with
    /// [`add_property()`].
    ///
    /// [`add_property()`]: crate::Module::add_property
    pub fn get_property(&self, property: PropertyHandle) -> Option<&Property> {
        self.properties.get(property.index())
    }

    /// Get an existing property by name.
    ///
    /// Existing properties are properties previously created with
    /// [`add_property()`].
    ///
    /// [`add_property()`]: crate::Module::add_property
    pub fn get_property_by_name(&self, name: &str) -> Option<PropertyHandle> {
        self.properties
            .iter()
            .enumerate()
            .find(|(_, prop)| prop.name() == name)
            .map(|(index, _)| PropertyHandle::new(NonZeroU32::new(index as u32 + 1).unwrap()))
    }

    /// Get the list of existing properties.
    pub fn properties(&self) -> &[Property] {
        &self.properties
    }

    /// Append a new expression to the module.
    fn push(&mut self, expr: impl Into<Expr>) -> ExprHandle {
        self.expressions.push(expr.into());
        #[allow(unsafe_code)]
        unsafe {
            ExprHandle::new_unchecked(self.expressions.len())
        }
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
    ///
    /// A property expression retrieves the value of the given property.
    #[inline]
    pub fn prop(&mut self, property: PropertyHandle) -> ExprHandle {
        self.push(Expr::Property(PropertyExpr::new(property)))
    }

    /// Build an analytical gradient sampling function and append it to the
    /// module.
    ///
    /// This is only supported for gradients of `f32`, `Vec2`, and `Vec3`. The
    /// function generated accepts a single parameter `x: f32` representing the
    /// sampling point of the gradient, and returns the value of the gradient
    /// sampled at that point. The function is a pure function with no side
    /// effect.
    #[inline]
    pub fn gradient<T>(&mut self, gradient: &Gradient<T>) -> FuncHandle
    where
        T: Lerp + FromReflect + Default + ToWgslString + ValueTypeOf,
        GradientKey<T>: Hash,
    {
        let func_id = calc_func_id(gradient);
        let func_name = format!("sample_gradient_{0:016X}", func_id);

        let mut body = String::new();
        for key in gradient.keys() {
            if !body.is_empty() {
                body += &"else ";
            }
            body += &format!(
                "if x <= {} {{ {} }}",
                key.ratio(),
                key.value.to_wgsl_string()
            );
        }
        let body = self.raw_code(body, Some(ValueType::Scalar(ScalarType::Float)));

        let parameters = vec![FuncParam {
            name: "x".to_string(),
            value_type: ScalarType::Float.into(),
        }];

        self.add_fn(&func_name, parameters, Some(T::value_type()), move |_| {
            Ok(body)
        })
        .expect(&format!(
            "Body generator for function {} returned an error.",
            func_name
        ))
    }

    /// Build a built-in expression and append it to the module.
    #[inline]
    pub fn builtin(&mut self, op: BuiltInOperator) -> ExprHandle {
        self.push(Expr::BuiltIn(BuiltInExpr::new(op)))
    }

    /// Build a math function call expression and append it to the module.
    ///
    /// The expression calls a [`MathFunction`] with the given arguments. The
    /// number of arguments required depends on the actual function.
    #[inline]
    pub fn math_fn(&mut self, func: MathFunction, args: &[ExprHandle]) -> ExprHandle {
        assert!(args.len() >= 1 && args.len() <= 4);
        self.push(Expr::Math(MathExpr {
            func,
            arg0: args[0],
            arg1: args.get(1).cloned(),
            arg2: args.get(2).cloned(),
            arg3: args.get(3).cloned(),
        }))
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
    impl_module_binary!(div, Div);
    impl_module_binary!(ge, GreaterThanOrEqual);
    impl_module_binary!(gt, GreaterThan);
    impl_module_binary!(le, LessThanOrEqual);
    impl_module_binary!(lt, LessThan);
    impl_module_binary!(mul, Mul);
    impl_module_binary!(rem, Remainder);
    impl_module_binary!(sub, Sub);
    impl_module_binary!(uniform, UniformRand);
    impl_module_binary!(vec2, Vec2);

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

    /// Add a new function definition to the module.
    ///
    /// The function body is generated with the `gen_body` closure.
    ///
    /// # Panics
    ///
    /// Panics if the module already contains a function with the same
    /// `func_name`.
    ///
    /// Panics if the `gen_body` closure returns an error.
    pub fn add_fn(
        &mut self,
        func_name: &str,
        parameters: Vec<FuncParam>,
        return_type: Option<ValueType>,
        gen_body: impl FnOnce(&mut Module) -> Result<ExprHandle, ExprError>,
    ) -> Result<FuncHandle, ExprError> {
        assert!(
            self.functions
                .iter()
                .find(|f| f.name == func_name)
                .is_none(),
            "Cannot add duplicate function named {} to Module.",
            func_name
        );

        let body = gen_body(self)?;

        self.functions.push(Func {
            name: func_name.to_string(),
            parameters,
            return_type,
            body: Some(body),
            kind: FunctionKind::User,
        });

        #[allow(unsafe_code)]
        let id: Id = unsafe { NonZeroU32::new_unchecked(self.functions.len() as u32) };

        Ok(FuncHandle::new(id))
    }

    fn add_builtin_function(
        &mut self,
        func_name: &str,
        parameters: Vec<FuncParam>,
        return_type: Option<ValueType>,
    ) -> FuncHandle {
        // Built-in function may be generic (like abs(f32) -> f32 vs. abs(i32) -> i32)
        // so we skip any unicity check on the name.

        let func = Func {
            name: func_name.to_string(),
            parameters,
            return_type,
            body: None, // built-in
            kind: FunctionKind::Builtin,
        };

        assert!(!self.functions.iter().any(|f| *f == func));

        self.functions.push(func);

        #[allow(unsafe_code)]
        let id: Id = unsafe { NonZeroU32::new_unchecked(self.functions.len() as u32) };

        FuncHandle::new(id)
    }

    fn get_builtin_func(&self, func_name: &str, parameters: &[ValueType]) -> Option<FuncHandle> {
        self.functions
            .iter()
            .position(|f| {
                f.name == func_name
                    // Compare the value type of parameters, which form the function signature.
                    // Ignore the parameter name, which is only used in the definition.
                    // The return type in theory is part of the signature, but two functions
                    // cannot differ only by it, so it can be ignored
                    && f.parameters
                        .iter()
                        .map(|p| p.value_type)
                        .eq(parameters.iter().cloned())
            })
            .map(|index| {
                #[allow(unsafe_code)]
                unsafe {
                    FuncHandle::new_unchecked(index + 1)
                }
            })
    }

    fn get_or_create_builtin_function(
        &mut self,
        func_name: &str,
        parameters: &[ValueType],
    ) -> FuncHandle {
        if let Some(func) = self.get_builtin_func(func_name, parameters) {
            return func;
        }

        match func_name {
            // logical operators on vec<bool>
            "all" | "any" => self.add_builtin_function(
                func_name,
                vec![FuncParam::new("e", parameters[0])],
                Some(ScalarType::Bool.into()),
            ),
            // single-argument numeric scalar and component-wise vector
            "abs" | "sign" => self.add_builtin_function(
                func_name,
                vec![FuncParam::new("e", parameters[0])],
                Some(parameters[0]),
            ),
            // single-argument floating-point scalar and component-wise vector
            "acos" | "acosh" | "asin" | "asinh" | "atan" | "atanh" | "ceil" | "cos" | "cosh"
            | "degrees" | "exp" | "exp2" | "floor" | "fract" | "inverseSqrt" | "length" | "log"
            | "log2" | "saturate" | "sqrt" => self.add_builtin_function(
                func_name,
                vec![FuncParam::new("e", parameters[0])],
                Some(parameters[0]),
            ),
            // two-argument numeric scalar and component-wise vector returning a scalar
            "distance" => self.add_builtin_function(
                func_name,
                vec![
                    FuncParam::new("e1", parameters[0]),
                    FuncParam::new("e2", parameters[1]),
                ],
                Some(ScalarType::Float.into()),
            ),
            // atan2(y, x) for f32 or vecX<f32>
            "atan2" => self.add_builtin_function(
                func_name,
                vec![
                    // atan2(y, x) conventionally because it's atan(y/x)
                    FuncParam::new("y", parameters[0]),
                    FuncParam::new("x", parameters[1]),
                ],
                Some(parameters[0]),
            ),
            // cross(vec3, vec3) -> vec3
            "cross" => self.add_builtin_function(
                func_name,
                vec![
                    FuncParam::new("e1", VectorType::VEC3F),
                    FuncParam::new("e2", VectorType::VEC3F),
                ],
                Some(VectorType::VEC3F.into()),
            ),
            // three-argument numeric scalar and component-wise vector
            "clamp" => self.add_builtin_function(
                func_name,
                vec![
                    FuncParam::new("e", parameters[0]),
                    FuncParam::new("low", parameters[1]),
                    FuncParam::new("high", parameters[2]),
                ],
                Some(parameters[0]),
            ),
            // dot(vec<numeric>, vec<numeric>) -> numeric
            "dot" => self.add_builtin_function(
                func_name,
                vec![
                    FuncParam::new("e1", parameters[0]),
                    FuncParam::new("e2", parameters[1]),
                ],
                Some(parameters[0]),
            ),
            // dot4U8Packed(u32, u32) -> u32
            "dot4U8Packed" => self.add_builtin_function(
                func_name,
                vec![
                    FuncParam::new("e1", ScalarType::Uint),
                    FuncParam::new("e2", ScalarType::Uint),
                ],
                Some(ScalarType::Uint.into()),
            ),
            // dot4I8Packed(u32, u32) -> i32
            "dot4I8Packed" => self.add_builtin_function(
                func_name,
                vec![
                    FuncParam::new("e1", ScalarType::Uint),
                    FuncParam::new("e2", ScalarType::Uint),
                ],
                Some(ScalarType::Int.into()),
            ),
            // extractBits(i32/vec<i32>, u32, u32) -> i32/vec<i32>
            // extractBits(u32/vec<u32>, u32, u32) -> u32/vec<u32>
            "extractBits " => self.add_builtin_function(
                func_name,
                vec![
                    FuncParam::new("e", parameters[0]),
                    FuncParam::new("offset", ScalarType::Uint),
                    FuncParam::new("count", ScalarType::Uint),
                ],
                Some(parameters[0]),
            ),
            // normalize(vec<f32>) -> vec<f32>
            "normalize" => self.add_builtin_function(
                func_name,
                vec![FuncParam::new("e", ScalarType::Float)],
                Some(ScalarType::Float.into()),
            ),
            // pack4x8snorm(vec4<f32>) -> u32
            // pack4x8unorm(vec4<f32>) -> u32
            "pack4x8snorm" | "pack4x8unorm" => self.add_builtin_function(
                func_name,
                vec![FuncParam::new("e", VectorType::VEC4F)],
                Some(ScalarType::Uint.into()),
            ),
            // unpack4x8snorm(u32) -> vec4<f32>
            // unpack4x8unorm(u32) -> vec4<f32>
            "unpack4x8snorm" | "unpack4x8unorm" => self.add_builtin_function(
                func_name,
                vec![FuncParam::new("e", ScalarType::Uint)],
                Some(VectorType::VEC4F.into()),
            ),
            // "countLeadingZeros" | "countOneBits" | "countTrailingZeros" | "determinant"
            _ => unimplemented!("Built-in function {} not implemented.", func_name),
        }
    }

    /// Get an existing function from its handle.
    #[inline]
    pub fn get_fn(&self, func: FuncHandle) -> Option<&Func> {
        let index = func.index();
        self.functions.get(index)
    }

    /// Build a function call expression and append it to the module.
    #[inline]
    pub fn call_fn(&mut self, func: FuncHandle, args: &[ExprHandle]) -> ExprHandle {
        let func_def = self.get_fn(func).expect("Unknown function handle.");
        // TODO - more validation...
        self.push(Expr::Call(CallExpr {
            func,
            args: args.to_vec(),
            return_type: func_def.return_type,
        }))
    }

    /// Build a raw code block expression and append it to the module.
    ///
    /// The raw `code` is expected to produce a value of type `value_type`, or
    /// no value if `None`.
    pub fn raw_code(&mut self, code: String, value_type: Option<ValueType>) -> ExprHandle {
        self.push(Expr::RawCode(RawCodeExpr { code, value_type }))
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
    /// with [`Module::add_property()`].
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

    /// Invalid modifier context.
    ///
    /// The operation was expecting a given [`ModifierContext`], but instead
    /// another [`ModifierContext`] was available.
    #[error("Invalid modifier context {0}, expected {1} instead.")]
    InvalidModifierContext(ModifierContext, ModifierContext),
}

/// Evaluation context for transforming expressions into WGSL code.
///
/// The evaluation context references a [`Module`] storing all [`Expr`] in use,
/// as well as a [`ParticleLayout`] defining the existing attributes of each
/// particle and their layout in memory, and a [`PropertyLayout`] defining
/// existing properties and their layout in memory. These together define the
/// context within which expressions are evaluated.
///
/// A same expression can be valid in one context and invalid in another. The
/// most obvious example is a [`PropertyExpr`] which is only valid if the
/// property is actually defined in the property layout of the evaluation
/// context.
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
    ///
    /// The variable name is not registered automatically in the [`Module`]. If
    /// you call `make_local_var()` but doesn't use the returned name, it won't
    /// appear in the shader.
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
    /// input `module` and a temporary [`EvalContext`] local to the function.
    /// The closure must return the generated shader code of the function
    /// body. Any statement pushed to the temporary function context with
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

    /// Check if the particle attribute struct is a pointer?
    ///
    /// In some context the attribute struct (named 'particle' in WGSL code) is
    /// a pointer instead of being a struct instance. This happens in particular
    /// when defining a function for a modifier, and passing the attribute
    /// struct to be modified. In that case the generated code needs to emit
    /// a pointer indirection code to access the fields of the struct.
    fn is_attribute_pointer(&self) -> bool;
}

/// Built-in WGSL shader function.
///
/// This is similar to [`naga::MathFunction`], but includes some relational
/// functions (`all()`, `any()`) and excludes functions not supported by WGSL.
/// This enum is also reflected (implements Bevy's [`Reflect`] trait).
///
/// [`Reflect`]: bevy::reflect::Reflect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Reflect)]
#[allow(missing_docs)]
pub enum MathFunction {
    // comparison
    Abs,
    Min,
    Max,
    Clamp,
    Saturate,
    // trigonometry
    Cos,
    Cosh,
    Sin,
    Sinh,
    Tan,
    Tanh,
    Acos,
    Asin,
    Atan,
    Atan2,
    Asinh,
    Acosh,
    Atanh,
    Radians,
    Degrees,
    // decomposition
    Ceil,
    Floor,
    Round,
    Fract,
    Trunc,
    Modf,
    Frexp,
    Ldexp,
    // exponent
    Exp,
    Exp2,
    Log,
    Log2,
    Pow,
    // geometry
    Dot,
    Cross,
    Distance,
    Length,
    Normalize,
    FaceForward,
    Reflect,
    Refract,
    // computational
    Sign,
    Fma,
    Mix,
    Step,
    SmoothStep,
    Sqrt,
    InverseSqrt,
    Transpose,
    Determinant,
    // bits
    CountTrailingZeros,
    CountLeadingZeros,
    CountOneBits,
    ReverseBits,
    ExtractBits,
    InsertBits,
    FindLsb,
    FindMsb,
    // data packing
    Pack4x8snorm,
    Pack4x8unorm,
    Pack2x16snorm,
    Pack2x16unorm,
    Pack2x16float,
    // data unpacking
    Unpack4x8snorm,
    Unpack4x8unorm,
    Unpack2x16snorm,
    Unpack2x16unorm,
    Unpack2x16float,
    // relational
    All,
    Any,
}

impl ToWgslString for MathFunction {
    fn to_wgsl_string(&self) -> String {
        match *self {
            MathFunction::Abs => "abs",
            MathFunction::Acos => "acos",
            MathFunction::Acosh => "acosh",
            MathFunction::All => "all",
            MathFunction::Any => "any",
            MathFunction::Asin => "asin",
            MathFunction::Asinh => "asinh",
            MathFunction::Atan => "atan",
            MathFunction::Atan2 => "atan2",
            MathFunction::Atanh => "atanh",
            MathFunction::Ceil => "ceil",
            MathFunction::Clamp => "clamp",
            MathFunction::CountLeadingZeros => "countLeadingZeros",
            MathFunction::CountOneBits => "countOneBits",
            MathFunction::CountTrailingZeros => "countTrailingZeros",
            MathFunction::Cos => "cos",
            MathFunction::Cosh => "cosh",
            MathFunction::Cross => "cross",
            MathFunction::Degrees => "degrees",
            MathFunction::Determinant => "determinant",
            MathFunction::Distance => "distance",
            MathFunction::Dot => "dot",
            MathFunction::Exp => "exp",
            MathFunction::Exp2 => "exp2",
            MathFunction::ExtractBits => "extractBits",
            MathFunction::FaceForward => "faceForward",
            MathFunction::FindLsb => "findLsb",
            MathFunction::FindMsb => "findMsb",
            MathFunction::Floor => "floor",
            MathFunction::Fma => "fma",
            MathFunction::Fract => "fract",
            MathFunction::Frexp => "frexp",
            MathFunction::InsertBits => "insertBits",
            MathFunction::InverseSqrt => "inverseSqrt",
            MathFunction::Ldexp => "ldexp",
            MathFunction::Length => "length",
            MathFunction::Log => "log",
            MathFunction::Log2 => "log2",
            MathFunction::Max => "max",
            MathFunction::Min => "min",
            MathFunction::Mix => "mix",
            MathFunction::Modf => "modf",
            MathFunction::Normalize => "normalize",
            MathFunction::Pack2x16float => "pack2x16float",
            MathFunction::Pack2x16snorm => "pack2x16snorm",
            MathFunction::Pack2x16unorm => "pack2x16unorm",
            MathFunction::Pack4x8snorm => "pack4x8snorm",
            MathFunction::Pack4x8unorm => "pack4x8unorm",
            MathFunction::Pow => "pow",
            MathFunction::Radians => "radians",
            MathFunction::Reflect => "reflect",
            MathFunction::Refract => "refract",
            MathFunction::ReverseBits => "reverseBits",
            MathFunction::Round => "round",
            MathFunction::Saturate => "saturate",
            MathFunction::Sign => "sign",
            MathFunction::Sin => "sin",
            MathFunction::Sinh => "sinh",
            MathFunction::SmoothStep => "smoothstep",
            MathFunction::Sqrt => "sqrt",
            MathFunction::Step => "step",
            MathFunction::Tan => "tan",
            MathFunction::Tanh => "tanh",
            MathFunction::Transpose => "transpose",
            MathFunction::Trunc => "trunc",
            MathFunction::Unpack2x16float => "unpack2x16float",
            MathFunction::Unpack2x16snorm => "unpack2x16snorm",
            MathFunction::Unpack2x16unorm => "unpack2x16unorm",
            MathFunction::Unpack4x8snorm => "unpack4x8snorm",
            MathFunction::Unpack4x8unorm => "unpack4x8unorm",
        }
        .to_string()
    }
}

/// Language expression producing a value.
#[derive(Debug, Clone, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub enum Expr {
    /// Built-in expression ([`BuiltInExpr`]).
    ///
    /// A built-in expression provides access to some internal
    /// quantities like the simulation time, or built-in functions like `max()`.
    BuiltIn(BuiltInExpr),

    /// Literal expression ([`LiteralExpr`]).
    ///
    /// A literal expression represents a shader constant, like `3.0` or
    /// `vec3<f32>(-5.0, 2.5, 3.2)`.
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

    /// Call to a [`MathFunction`].
    ///
    /// Math functions are built-in functions provided by the WGSL language.
    Math(MathExpr),

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

    /// Function call expression.
    ///
    /// An expression to call a function with some arguments, and optionally
    /// retrieve a result.
    Call(CallExpr),

    /// Raw code expression.
    ///
    /// An expression encapsulating a raw WGSL block of code. This acts as an
    /// escape hatch for features not yet supported.
    RawCode(RawCodeExpr),
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
    /// # let mut module = Module::default();
    /// // Literals are always constant by definition.
    /// assert!(Expr::Literal(LiteralExpr::new(1.)).is_const(&module));
    ///
    /// // Properties and attributes are never constant, since they're by definition used
    /// // to provide runtime customization.
    /// let prop = module.add_property("my_property", 3.0.into());
    /// assert!(!Expr::Property(PropertyExpr::new(prop)).is_const(&module));
    /// assert!(!Expr::Attribute(AttributeExpr::new(Attribute::POSITION)).is_const(&module));
    /// ```
    pub fn is_const(&self, module: &Module) -> bool {
        match self {
            Expr::BuiltIn(expr) => expr.is_const(),
            Expr::Literal(expr) => expr.is_const(),
            Expr::Property(expr) => expr.is_const(),
            Expr::Attribute(expr) => expr.is_const(),
            Expr::Math(expr) => expr.is_const(module),
            Expr::Unary { expr, .. } => module.is_const(*expr),
            Expr::Binary { left, right, .. } => module.is_const(*left) && module.is_const(*right),
            Expr::Ternary {
                first,
                second,
                third,
                ..
            } => module.is_const(*first) && module.is_const(*second) && module.is_const(*third),
            Expr::Cast(expr) => module.is_const(expr.inner),
            Expr::Call(_) => false, // TODO - handle pure functions with const arguments?
            Expr::RawCode(_) => false,
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
            Expr::Math { .. } => true, // TODO - handle pure functions?
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
            Expr::Call(_) => true, // TODO - handle pure functions?
            Expr::RawCode(_) => true,
        }
    }

    /// The type of the value produced by the expression.
    ///
    /// If the expression produced no value, like for example an `Expr::Call` to
    /// a function without a return value, then this returns `None`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # let mut module = Module::default();
    /// // Literal expressions always have a constant, build-time value type.
    /// let expr = Expr::Literal(LiteralExpr::new(1.));
    /// assert_eq!(expr.value_type(&module), Some(ValueType::Scalar(ScalarType::Float)));
    /// ```
    pub fn value_type(&self, module: &Module) -> Option<ValueType> {
        match self {
            Expr::BuiltIn(expr) => Some(expr.value_type()),
            Expr::Literal(expr) => Some(expr.value_type()),
            Expr::Property(expr) => Some(expr.value_type(module)),
            Expr::Attribute(expr) => Some(expr.value_type()),
            Expr::Math(expr) => Some(expr.value_type(module)),
            Expr::Unary { expr, .. } => module.get(*expr).unwrap().value_type(module),
            Expr::Binary { left, .. } => module.get(*left).unwrap().value_type(module),
            Expr::Ternary { first, .. } => module.get(*first).unwrap().value_type(module),
            Expr::Cast(expr) => Some(expr.value_type()),
            Expr::Call(expr) => expr.return_type,
            Expr::RawCode(expr) => expr.value_type,
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
    /// # let mut context = ShaderWriter::new(ModifierContext::Update, &pl, &pal);
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
            Expr::Property(expr) => expr.eval(module, context),
            Expr::Attribute(expr) => expr.eval(context),
            Expr::Math(expr) => expr.eval(module, context),
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
                let compiled_left = context.eval(module, *left)?;
                let compiled_right = context.eval(module, *right)?;

                // if !self.input.value_type().is_vector() {
                //     return Err(ExprError::TypeError(format!(
                //         "Cannot apply normalize() function to non-vector expression: {}",
                //         expr.unwrap_or("(error evaluating expression)".to_string())
                //     )));
                // }

                if op.is_functional() {
                    if op.needs_type_suffix() {
                        let lhs_type = module.get(*left).and_then(|arg| arg.value_type(module));
                        let rhs_type = module.get(*right).and_then(|arg| arg.value_type(module));
                        if lhs_type.is_none() || rhs_type.is_none() {
                            return Err(ExprError::TypeError(
                                "Can't determine the type of the operand".to_string(),
                            ));
                        }
                        if lhs_type != rhs_type {
                            return Err(ExprError::TypeError("Mismatched types".to_string()));
                        }
                        let value_type = lhs_type.unwrap();
                        let suffix = match value_type {
                            ValueType::Scalar(ScalarType::Float) => "f",
                            ValueType::Vector(vector_type)
                                if vector_type.elem_type() == ScalarType::Float =>
                            {
                                match vector_type.count() {
                                    2 => "vec2",
                                    3 => "vec3",
                                    4 => "vec4",
                                    _ => unreachable!(),
                                }
                            }
                            _ => {
                                // Add more types here as needed.
                                return Err(ExprError::TypeError("Unsupported type".to_string()));
                            }
                        };

                        Ok(format!(
                            "{}_{}({}, {})",
                            op.to_wgsl_string(),
                            suffix,
                            compiled_left,
                            compiled_right
                        ))
                    } else {
                        Ok(format!(
                            "{}({}, {})",
                            op.to_wgsl_string(),
                            compiled_left,
                            compiled_right
                        ))
                    }
                } else {
                    Ok(format!(
                        "({}) {} ({})",
                        compiled_left,
                        op.to_wgsl_string(),
                        compiled_right
                    ))
                }
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
            Expr::Call(expr) => expr.eval(module, context),
            Expr::RawCode(expr) => Ok(expr.code.clone()),
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
#[serde(transparent)]
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
    pub fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError> {
        if context.is_attribute_pointer() {
            Ok(format!("(*particle).{}", self.attr.name()))
        } else {
            Ok(format!("particle.{}", self.attr.name()))
        }
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

/// Expression representing a call to a built-in [`MathFunction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct MathExpr {
    /// Function to call.
    pub func: MathFunction,
    /// First function argument.
    pub arg0: ExprHandle,
    /// Optional second function argument.
    pub arg1: Option<ExprHandle>,
    /// Optional third function argument.
    pub arg2: Option<ExprHandle>,
    /// Optional fourth function argument.
    pub arg3: Option<ExprHandle>,
}

impl MathExpr {
    /// Is the expression resulting in a compile-time constant which can be
    /// hard-coded into a shader's code?
    pub fn is_const(&self, module: &Module) -> bool {
        // All math functions are pure functions, so the output is const if all inputs
        // are
        module.is_const(self.arg0)
            && self.arg1.map(|expr| module.is_const(expr)).unwrap_or(false)
            && self.arg2.map(|expr| module.is_const(expr)).unwrap_or(false)
            && self.arg3.map(|expr| module.is_const(expr)).unwrap_or(false)
    }

    /// Get the value type of the expression.
    pub fn value_type(&self, module: &Module) -> ValueType {
        match self.func {
            // Note: we list explicitly to make sure we don't forget any
            MathFunction::Abs
            | MathFunction::Acos
            | MathFunction::Acosh
            | MathFunction::Asin
            | MathFunction::Asinh
            | MathFunction::Atan
            | MathFunction::Atan2
            | MathFunction::Atanh
            | MathFunction::Ceil
            | MathFunction::Clamp
            | MathFunction::CountLeadingZeros
            | MathFunction::CountOneBits
            | MathFunction::CountTrailingZeros
            | MathFunction::Cos
            | MathFunction::Cosh
            | MathFunction::Degrees
            | MathFunction::Dot
            | MathFunction::Exp
            | MathFunction::Exp2
            | MathFunction::ExtractBits
            | MathFunction::FaceForward
            | MathFunction::FindLsb
            | MathFunction::FindMsb
            | MathFunction::Floor
            | MathFunction::Fma
            | MathFunction::Fract
            | MathFunction::InsertBits
            | MathFunction::InverseSqrt
            | MathFunction::Ldexp
            | MathFunction::Length
            | MathFunction::Log
            | MathFunction::Log2
            | MathFunction::Max
            | MathFunction::Min
            | MathFunction::Mix
            | MathFunction::Pow
            | MathFunction::Radians
            | MathFunction::Reflect
            | MathFunction::Refract
            | MathFunction::ReverseBits
            | MathFunction::Round
            | MathFunction::Saturate
            | MathFunction::Sign
            | MathFunction::Sin
            | MathFunction::Sinh
            | MathFunction::SmoothStep
            | MathFunction::Sqrt
            | MathFunction::Step
            | MathFunction::Tan
            | MathFunction::Tanh
            | MathFunction::Trunc => module.get(self.arg0).unwrap().value_type(module).unwrap(),
            MathFunction::All | MathFunction::Any => ScalarType::Bool.into(),
            MathFunction::Cross => VectorType::VEC3F.into(),
            MathFunction::Normalize => ScalarType::Float.into(),
            MathFunction::Pack4x8snorm
            | MathFunction::Pack4x8unorm
            | MathFunction::Pack2x16snorm
            | MathFunction::Pack2x16unorm
            | MathFunction::Pack2x16float => ScalarType::Uint.into(),
            MathFunction::Unpack4x8snorm | MathFunction::Unpack4x8unorm => VectorType::VEC4F.into(),
            MathFunction::Unpack2x16float
            | MathFunction::Unpack2x16snorm
            | MathFunction::Unpack2x16unorm => VectorType::VEC2F.into(),
            MathFunction::Determinant => todo!(),
            MathFunction::Distance => todo!(),
            MathFunction::Transpose => todo!(),
            MathFunction::Modf | MathFunction::Frexp => unimplemented!(
                "Can't implement right now, this returns an implementation-defined type"
            ),
        }
    }

    /// Evaluate the expression in the given context.
    pub fn eval(
        &self,
        module: &Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        let mut args = context.eval(module, self.arg0)?;
        if let Some(arg1) = self.arg1 {
            args += &", ";
            args += &context.eval(module, arg1)?;
        }
        if let Some(arg2) = self.arg2 {
            args += &", ";
            args += &context.eval(module, arg2)?;
        }
        if let Some(arg3) = self.arg3 {
            args += &", ";
            args += &context.eval(module, arg3)?;
        }
        Ok(format!("{}({})", self.func.to_wgsl_string(), args))
    }
}

/// Expression representing the value of a property of an effect.
///
/// A property expression represents the value of the property at the time the
/// expression appears. In shader, the expression yields a read from the
/// property memory location.
///
/// To create a property to reference with an expression, use
/// [`Module::add_property()`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct PropertyExpr {
    property: PropertyHandle,
}

impl PropertyExpr {
    /// Create a new property expression.
    #[inline]
    pub fn new(property: PropertyHandle) -> Self {
        Self { property }
    }

    /// Is the expression resulting in a compile-time constant which can be
    /// hard-coded into a shader's code?
    fn is_const(&self) -> bool {
        false
    }

    fn value_type(&self, module: &Module) -> ValueType {
        module.get_property(self.property).unwrap().value_type()
    }

    /// Evaluate the expression in the given context.
    fn eval(&self, module: &Module, context: &dyn EvalContext) -> Result<String, ExprError> {
        let prop = module
            .get_property(self.property)
            .ok_or(ExprError::PropertyError(format!(
                "Unknown property handle {:?} in evaluation module.",
                self.property
            )))?;
        if !context.property_layout().contains(prop.name()) {
            return Err(ExprError::PropertyError(format!(
                "Unknown property '{}' in evaluation layout.",
                prop.name()
            )));
        }

        Ok(format!("properties.{}", prop.name()))
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
        if let Some(inner_type) = inner.value_type(module) {
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

/// Expression representing a call to a function.
///
/// The call expressions represents a single call to the given function,
/// represented by its [`FuncHandle`]. The function must be declared first via
/// [`Module::add_fn()`].
///
/// The function call takes the given arguments, and optionally return a result,
/// depending on the actual function signature.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct CallExpr {
    /// Function to call.
    pub func: FuncHandle,
    /// Arguments.
    pub args: Vec<ExprHandle>,
    /// Type of the value return by the function, if any.
    pub return_type: Option<ValueType>,
}

impl CallExpr {
    /// Evaluate the expression in the given context.
    fn eval(&self, module: &Module, context: &mut dyn EvalContext) -> Result<String, ExprError> {
        // Recursively evaluate argument expressions throught the context to ensure
        // caching
        let args = self.args.iter().try_fold(String::new(), |mut acc, arg| {
            let str = context.eval(module, *arg)?;
            if !acc.is_empty() {
                acc += &", ";
            }
            acc += &str;
            Ok(acc)
        })?;

        let func = module
            .get_fn(self.func)
            .ok_or(ExprError::InvalidExprHandleError(
                "Unknown function".to_string(),
            ))?;

        Ok(format!("{}({})", func.name(), args))
    }
}

/// Expression representing a raw WGSL code block.
///
/// This acts as an escape hatch for any feature not otherwise implemented as an
/// expression, but should be used as last resort as there's no validation
/// whatsoever that the emitted code is well formed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct RawCodeExpr {
    /// WGSL code.
    pub code: String,
    /// Type of the value produced by the expresion, if any.
    pub value_type: Option<ValueType>,
}

/// Built-in operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum BuiltInOperator {
    /// Current effect system simulation time since startup, in seconds.
    /// This is based on the
    /// [`Time<EffectSimulation>`](crate::time::EffectSimulation) clock.
    Time,
    /// Delta time, in seconds, since last effect system update.
    DeltaTime,
    /// Current virtual time since startup, in seconds.
    /// This is based on the [`Time<Virtual>`](bevy::time::Virtual) clock.
    VirtualTime,
    /// Virtual delta time, in seconds, since last effect system update.
    VirtualDeltaTime,
    /// Current real time since startup, in seconds.
    /// This is based on the [`Time<Time>`](bevy::time::Real) clock.
    RealTime,
    /// Real delta time, in seconds, since last effect system update.
    RealDeltaTime,
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
            BuiltInOperator::VirtualTime => "virtual_time",
            BuiltInOperator::VirtualDeltaTime => "virtual_delta_time",
            BuiltInOperator::RealTime => "real_time",
            BuiltInOperator::RealDeltaTime => "real_delta_time",
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
            BuiltInOperator::VirtualTime => ValueType::Scalar(ScalarType::Float),
            BuiltInOperator::VirtualDeltaTime => ValueType::Scalar(ScalarType::Float),
            BuiltInOperator::RealTime => ValueType::Scalar(ScalarType::Float),
            BuiltInOperator::RealDeltaTime => ValueType::Scalar(ScalarType::Float),
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

    /// Division operator.
    ///
    /// Returns the left operand divided by the right operand. Only valid for
    /// numeric operands.
    Div,

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

    /// Constructor for 2-element vectors.
    ///
    /// Given two scalar elements `x` and `y`, returns the vector consisting of
    /// those two elements `(x, y)`.
    Vec2,
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
            BinaryOperator::UniformRand | BinaryOperator::Vec2 => true,
        }
    }

    /// Check if a binary operator needs a type suffix.
    ///
    /// This is currently just for `rand_uniform`
    /// (`BinaryOperator::UniformRand`), which is a function we define
    /// ourselves. WGSL doesn't support user-defined function overloading, so
    /// we need a suffix to disambiguate the types.
    pub fn needs_type_suffix(&self) -> bool {
        *self == BinaryOperator::UniformRand
    }
}

impl ToWgslString for BinaryOperator {
    fn to_wgsl_string(&self) -> String {
        match *self {
            BinaryOperator::Add => "+",
            BinaryOperator::Div => "/",
            BinaryOperator::GreaterThan => ">",
            BinaryOperator::GreaterThanOrEqual => ">=",
            BinaryOperator::LessThan => "<",
            BinaryOperator::LessThanOrEqual => "<=",
            BinaryOperator::Mul => "*",
            BinaryOperator::Remainder => "%",
            BinaryOperator::Sub => "-",
            BinaryOperator::UniformRand => "rand_uniform",
            BinaryOperator::Vec2 => "vec2",
        }
        .to_string()
    }
}

/// Ternary operator.
///
/// Operator applied between three operands. The type of the operands and the
/// result are not necessarily the same. Valid operand types depend on the
/// operator itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum TernaryOperator {
    /// Constructor for 3-element vectors.
    ///
    /// Given three scalar elements `x`, `y`, and `z`, returns the vector
    /// consisting of those three elements `(x, y, z)`.
    Vec3,
}

impl ToWgslString for TernaryOperator {
    fn to_wgsl_string(&self) -> String {
        match *self {
            TernaryOperator::Vec3 => "vec3".to_string(),
        }
    }
}

/// Expression writer.
///
/// Utility to write expressions with a simple functional syntax. Expressions
/// created with the writer are gatherned into a [`Module`] which can be
/// retrieved with [`finish()`] once done, for example to initialize an
/// [`EffectAsset`].
///
/// Because an [`EffectAsset`] contains a single [`Module`], you generally want
/// to keep using the same [`ExprWriter`] to write all the expressions used by
/// all the [`Modifer`]s assigned to a given [`EffectAsset`], and only then once
/// done call [`finish()`] to recover the [`ExprWriter`]'s underlying [`Module`]
/// to assign it to the [`EffectAsset`]. Alternatively, you can re-wrap an
/// existing [`Module`] with `From<Module>`.
///
/// # Example
///
/// ```
/// # use bevy_hanabi::*;
/// # use bevy::prelude::*;
/// // Create a writer
/// let w = ExprWriter::new();
///
/// // Create a new expression: max(5. + particle.position, properties.my_prop)
/// let prop = w.add_property("my_property", Vec3::ONE.into());
/// let expr = (w.lit(5.) + w.attr(Attribute::POSITION)).max(w.prop(prop));
///
/// // Finalize the expression and write it down into the `Module` as an `Expr`
/// let expr: ExprHandle = expr.expr();
///
/// // Create a modifier and assign the expression to one of its input(s)
/// let init_modifier = SetAttributeModifier::new(Attribute::LIFETIME, expr);
///
/// // Finalize the writer and release the written module
/// let module = w.finish();
///
/// // Create an EffectAsset with the modifier and the Module from the writer
/// let effect = EffectAsset::new(vec![1024], Spawner::rate(32_f32.into()), module)
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
    /// to it. The module can be released with [`finish()`] once done using the
    /// writer.
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let mut w = ExprWriter::new();
    /// // [...]
    /// let module = w.finish();
    /// ```
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
    /// Alternatively, use `From<Module>` to wrap an existing [`Module`] into a
    /// new [`ExprWriter`].
    ///
    /// [`EffectAsset`]: crate::EffectAsset
    pub fn from_module(module: Rc<RefCell<Module>>) -> Self {
        Self { module }
    }

    /// Add a new property.
    ///
    /// Declare a new property and add it to the underlying [`Module`]. See
    /// [`Property`] for more details on what effect properties are.
    ///
    /// # Panics
    ///
    /// Panics if a property with the same name already exists.
    pub fn add_property(&self, name: impl Into<String>, default_value: Value) -> PropertyHandle {
        self.module.borrow_mut().add_property(name, default_value)
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
    /// let prop = w.add_property("my_prop", 3.0.into());
    /// let x = w.prop(prop); // x = properties.my_prop;
    /// ```
    pub fn prop(&self, handle: PropertyHandle) -> WriterExpr {
        self.push(Expr::Property(PropertyExpr::new(handle)))
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
    /// let asset = EffectAsset::new(vec![256], spawner, module);
    /// ```
    ///
    /// [`EffectAsset::new()`]: crate::EffectAsset::new()
    pub fn finish(self) -> Module {
        self.module.take()
    }
}

impl From<Module> for ExprWriter {
    fn from(value: Module) -> Self {
        Self {
            module: Rc::new(RefCell::new(value)),
        }
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
/// let my_prop = w.add_property("my_prop", 3.0.into());
///
/// // x = max(-3.5 + 1., properties.my_prop) * 0.5 - particle.position;
/// let x = (w.lit(-3.5) + w.lit(1.)).max(w.prop(my_prop)).mul(w.lit(0.5))
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

    fn unary_math_fn(self, func: MathFunction) -> Self {
        let expr = self.module.borrow_mut().math_fn(func, &[self.expr]);
        WriterExpr {
            expr,
            module: self.module,
        }
    }

    fn binary_math_fn(self, func: MathFunction, arg1: Self) -> Self {
        assert_eq!(self.module, arg1.module);
        let arg0 = self.expr;
        let arg1 = arg1.expr;
        let expr = self.module.borrow_mut().math_fn(func, &[arg0, arg1]);
        WriterExpr {
            expr,
            module: self.module,
        }
    }

    fn ternary_math_fn(self, func: MathFunction, arg1: Self, arg2: Self) -> Self {
        assert_eq!(self.module, arg1.module);
        assert_eq!(self.module, arg2.module);
        let arg0 = self.expr;
        let arg1 = arg1.expr;
        let arg2 = arg2.expr;
        let expr = self.module.borrow_mut().math_fn(func, &[arg0, arg1, arg2]);
        WriterExpr {
            expr,
            module: self.module,
        }
    }

    /// Call a single-argument function with the given name.
    ///
    /// The argument is the expression contained in this [`WriterExpr`]. Its
    /// value type determines the function overload called.
    #[allow(dead_code)]
    fn call_unary_fn(self, func_name: &str) -> Self {
        let inner_type = {
            let m = self.module.borrow();
            let inner = m.get(self.expr).unwrap();
            inner.value_type(&*m).unwrap()
        };
        let func = self
            .module
            .borrow_mut()
            .get_or_create_builtin_function(func_name, &[inner_type]);
        let return_type = self.module.borrow().get_fn(func).unwrap().return_type();
        let expr = self.module.borrow_mut().push(Expr::Call(CallExpr {
            func,
            args: vec![self.expr],
            return_type,
        }));
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
        self.unary_math_fn(MathFunction::Abs)
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
        self.unary_math_fn(MathFunction::All)
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
        self.unary_math_fn(MathFunction::Any)
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
        self.unary_math_fn(MathFunction::Ceil)
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
        self.unary_math_fn(MathFunction::Cos)
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
        self.unary_math_fn(MathFunction::Exp)
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
        self.unary_math_fn(MathFunction::Exp2)
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
        self.unary_math_fn(MathFunction::Floor)
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
        self.unary_math_fn(MathFunction::Fract)
    }

    /// Apply the "inverseSqrt" (inverse square root) operator to the current
    /// float scalar or vector expression.
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
    /// // Inverse square root: `y = inverseSqrt(x) = 1.0 / sqrt(x);`
    /// let y = x.inverse_sqrt();
    /// ```
    #[inline]
    pub fn inverse_sqrt(self) -> Self {
        self.unary_math_fn(MathFunction::InverseSqrt)
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
        self.unary_math_fn(MathFunction::Length)
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
        self.unary_math_fn(MathFunction::Log)
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
        self.unary_math_fn(MathFunction::Log2)
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
        self.unary_math_fn(MathFunction::Normalize)
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
        self.unary_math_fn(MathFunction::Pack4x8snorm)
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
        self.unary_math_fn(MathFunction::Pack4x8unorm)
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
        self.unary_math_fn(MathFunction::Sign)
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
        self.unary_math_fn(MathFunction::Sin)
    }

    /// Apply the "sqrt" (square root) operator to the current float scalar or
    /// vector expression.
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
    /// // Square root: `y = sqrt(x);`
    /// let y = x.sqrt();
    /// ```
    #[inline]
    pub fn sqrt(self) -> Self {
        self.unary_math_fn(MathFunction::Sqrt)
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
        self.unary_math_fn(MathFunction::Tan)
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
        self.unary_math_fn(MathFunction::Unpack4x8snorm)
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
        self.unary_math_fn(MathFunction::Unpack4x8unorm)
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
        self.unary_math_fn(MathFunction::Saturate)
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
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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

    /// Angle in radians, in the \[-π,π\] interval, whose tangent is `y / x`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is from a different [`Module`] than self.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::math::Vec3;
    /// # let mut w = ExprWriter::new();
    /// let x = w.lit(3.);
    /// let y = w.lit(5.);
    /// // z = atan2(y, x)
    /// let z = y.atan2(x);
    /// ```
    ///
    /// # WGSL spec
    ///
    /// See <https://www.w3.org/TR/WGSL/#atan2-builtin> for details.
    #[inline]
    pub fn atan2(self, x: WriterExpr) -> Self {
        self.binary_math_fn(MathFunction::Atan2, x)
    }

    /// Calculate the cross product of the current expression by another
    /// expression.
    ///
    /// This is a binary operator, which applies to vector operands of size 3
    /// only, and always produces a vector of the same size.
    ///
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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
        self.binary_math_fn(MathFunction::Cross, other)
    }

    /// Calculate the dot product of the current expression by another
    /// expression.
    ///
    /// This is a binary operator, which applies to vector operands of same size
    /// only, and always produces a floating point scalar.
    ///
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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
        self.binary_math_fn(MathFunction::Dot, other)
    }

    /// Calculate the distance between the current expression and another
    /// expression.
    ///
    /// This is a binary operator.
    ///
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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
        self.binary_math_fn(MathFunction::Distance, other)
    }

    /// Divide the current expression by another expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// You can also use the [`std::ops::Div`] trait directly, via the `/`
    /// symbol, as an alternative to calling this method directly.
    ///
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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
        self.binary_math_fn(MathFunction::Max, other)
    }

    /// Take the minimum value of the current expression and another expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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
        self.binary_math_fn(MathFunction::Min, other)
    }

    /// Multiply the current expression with another expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// You can also use the [`std::ops::Mul`] trait directly, via the `*`
    /// symbol, as an alternative to calling this method directly.
    ///
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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
    /// Returns `0.0` before the edge  (`x < edge`) and `1.0` after it (`x >=
    /// edge`).
    ///
    /// # Panics
    ///
    /// Panics if `edge` is from a different [`Module`] than self.
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
    ///
    /// # WGSL spec
    ///
    /// See <https://www.w3.org/TR/WGSL/#step-builtin> for details.
    #[inline]
    pub fn step(self, edge: Self) -> Self {
        // Note: order is step(edge, x) but called as x.step(edge)
        edge.binary_math_fn(MathFunction::Step, self)
    }

    /// Subtract another expression from the current expression.
    ///
    /// This is a binary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// You can also use the [`std::ops::Sub`] trait directly, via the `-`
    /// symbol, as an alternative to calling this method directly.
    ///
    /// # Panics
    ///
    /// Panics if `other` is from a different [`Module`] than self.
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

    /// Clamp a value in a given range.
    ///
    /// Returns `min(max(e, low), high)`.
    ///
    /// # Panics
    ///
    /// Panics if any argument is from a different [`Module`] than self.
    ///
    /// # WGSL spec
    ///
    /// See <https://www.w3.org/TR/WGSL/#clamp> for details.
    #[inline]
    pub fn clamp(self, low: WriterExpr, high: WriterExpr) -> Self {
        self.ternary_math_fn(MathFunction::Clamp, low, high)
    }

    /// Blending linearly ("mix") two expressions with the fraction provided by
    /// a third expression.
    ///
    /// This is a ternary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// # Panics
    ///
    /// Panics if any argument is from a different [`Module`] than self.
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
    ///
    /// # WGSL spec
    ///
    /// See <https://www.w3.org/TR/WGSL/#mix-builtin> for details.
    #[inline]
    pub fn mix(self, other: Self, fraction: Self) -> Self {
        self.ternary_math_fn(MathFunction::Mix, other, fraction)
    }

    /// Calculate the smooth Hermite interpolation in \[0:1\] of the current
    /// value taken between the given bounds.
    ///
    /// This is a ternary operator, which applies component-wise to vector
    /// operand expressions.
    ///
    /// Returns `t * t * (3.0 - 2.0 * t)`, where `t = saturate((x - low) / (high
    /// - low))`.
    ///
    /// # Panics
    ///
    /// Panics if any argument is from a different [`Module`] than self.
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
    ///
    /// # WGSL spec
    ///
    /// See <https://www.w3.org/TR/WGSL/#smoothstep-builtin> for details.
    #[inline]
    pub fn smoothstep(self, low: Self, high: Self) -> Self {
        // Note: order is smoothstep(low, high, x) but x.smoothstep(low, high)
        low.ternary_math_fn(MathFunction::SmoothStep, high, self)
    }

    /// Insert bits from one integer into the other.
    ///
    /// Select `count` bits from `new_bits` starting at offset `offset`, and
    /// copy other bits from `e`. Return the masked copy.
    ///
    /// See <https://www.w3.org/TR/WGSL/#insertBits-builtin> for details.
    ///
    /// # Panics
    ///
    /// Panics if any argument is from a different [`Module`] than self.
    ///
    /// # WGSL spec
    ///
    /// See <https://www.w3.org/TR/WGSL/#insertBits-builtin> for details.
    #[inline]
    pub fn insert_bits(self, new_bits: WriterExpr, offset: WriterExpr, count: WriterExpr) -> Self {
        assert_eq!(self.module, new_bits.module);
        assert_eq!(self.module, offset.module);
        assert_eq!(self.module, count.module);
        let new_bits = new_bits.expr;
        let offset = offset.expr;
        let count = count.expr;
        let expr = self.module.borrow_mut().math_fn(
            MathFunction::InsertBits,
            &[self.expr, new_bits, offset, count],
        );
        WriterExpr {
            expr,
            module: self.module,
        }
    }

    /// Construct a `Vec2` from two scalars.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # let mut w = ExprWriter::new();
    /// let theta = w.add_property("theta", 0.0.into());
    /// // Convert the angular property `theta` to a 2D vector.
    /// let (cos_theta, sin_theta) = (w.prop(theta).cos(), w.prop(theta).sin());
    /// let circle_pos = cos_theta.vec2(sin_theta);
    /// ```
    #[inline]
    pub fn vec2(self, y: Self) -> Self {
        self.binary_op(y, BinaryOperator::Vec2)
    }

    /// Construct a `Vec3` from two scalars.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # let mut w = ExprWriter::new();
    /// let theta = w.add_property("theta", 0.0.into());
    /// // Convert the angular property `theta` to a 3D vector in a flat plane.
    /// let (cos_theta, sin_theta) = (w.prop(theta).cos(), w.prop(theta).sin());
    /// let circle_pos = cos_theta.vec3(w.lit(0.0), sin_theta);
    /// ```
    #[inline]
    pub fn vec3(self, y: Self, z: Self) -> Self {
        self.ternary_op(y, z, TernaryOperator::Vec3)
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

/// Shader function parameter.
///
/// Defines the name and type of a single parameter to a [`Func`].
#[derive(Debug, Clone, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub struct FuncParam {
    /// Parameter name.
    pub name: String,
    /// Parameter value.
    pub value_type: ValueType,
}

impl FuncParam {
    /// Create a new function parameter.
    #[inline]
    pub fn new(name: impl Into<String>, value_type: impl Into<ValueType>) -> Self {
        Self {
            name: name.into(),
            value_type: value_type.into(),
        }
    }
}

/// Kind of a [`Func`] function of a [`Module`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum FunctionKind {
    /// Built-in function defined in the WGSL language.
    Builtin,
    /// Function defined by Hanabi through modifiers.
    Modifier,
    /// User-defined function.
    User,
}

/// Shader function.
#[derive(Debug, Clone, PartialEq, Hash, Reflect, Serialize, Deserialize)]
pub struct Func {
    /// Function name.
    name: String,
    /// List of parameters, possibly empty.
    parameters: Vec<FuncParam>,
    /// Optional return type of the function.
    return_type: Option<ValueType>,
    /// Expression representing the function body.
    ///
    /// This is `None` for built-in functions provided by the WGSL language.
    body: Option<ExprHandle>,
    /// Function kind
    kind: FunctionKind,
}

impl Func {
    /// Get the function name.
    ///
    /// The function name is the WGSL identifier used to call the function.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the kind of this function.
    pub fn kind(&self) -> FunctionKind {
        self.kind
    }

    /// Get the function parameters.
    pub fn params(&self) -> &[FuncParam] {
        &self.parameters[..]
    }

    /// Get the function return type, if any.
    ///
    /// This returns the type of the function return value, or `None` if the
    /// function has no return value.
    pub fn return_type(&self) -> Option<ValueType> {
        self.return_type
    }
}

#[cfg(test)]
mod tests {
    use crate::{MatrixType, ScalarValue, ShaderWriter, VectorType};

    use super::*;
    use bevy::{prelude::*, utils::HashSet};

    #[test]
    fn module() {
        let mut m = Module::default();

        #[allow(unsafe_code)]
        let unknown = unsafe { ExprHandle::new_unchecked(1) };
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
        let mut ctx =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
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
        let mut ctx =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
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
    fn property() {
        let mut m = Module::default();

        let _my_prop = m.add_property("my_prop", Value::Scalar(345_u32.into()));
        let _other_prop = m.add_property(
            "other_prop",
            Value::Vector(Vec3::new(3., -7.5, 42.42).into()),
        );

        assert!(m.properties().iter().any(|p| p.name() == "my_prop"));
        assert!(m.properties().iter().any(|p| p.name() == "other_prop"));
        assert!(!m.properties().iter().any(|p| p.name() == "do_not_exist"));
    }

    #[test]
    fn writer() {
        // Get a module and its writer
        let w = ExprWriter::new();
        let my_prop = w.add_property("my_prop", 3.0.into());

        // Build some expression
        let x = w.lit(3.).abs().max(w.attr(Attribute::POSITION) * w.lit(2.))
            + w.lit(-4.).min(w.prop(my_prop));
        let x = x.expr();

        // Create an evaluation context
        let property_layout =
            PropertyLayout::new(&[Property::new("my_prop", ScalarValue::Float(3.))]);
        let particle_layout = ParticleLayout::default();
        let m = w.finish();
        let mut context =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);

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
        let mut ctx =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);

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
            let mut ctx =
                ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);

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
                let mut ctx =
                    ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);

                let expr = ctx.eval(&m, value);
                assert!(expr.is_ok());
                let expr = expr.unwrap();
                assert_eq!(expr, "var0");
                assert_eq!(ctx.main_code, format!("let var0 = {}rand();\n", prefix));
            }

            // Vector form
            for count in 2..=4 {
                let vec = m.builtin(BuiltInOperator::Rand(
                    VectorType::new(scalar_type, count).into(),
                ));

                let property_layout = PropertyLayout::default();
                let particle_layout = ParticleLayout::default();
                let mut ctx =
                    ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);

                let expr = ctx.eval(&m, vec);
                assert!(expr.is_ok());
                let expr = expr.unwrap();
                assert_eq!(expr, "var0");
                assert_eq!(
                    ctx.main_code,
                    format!("let var0 = {}rand{}();\n", prefix, count)
                );
            }
        }
    }

    #[test]
    fn module_unary_expr() {
        let mut m = Module::default();

        let w = m.lit(Vec4::W);
        let comp_x = m.x(w);
        let comp_y = m.y(w);
        let comp_z = m.z(w);
        let comp_w = m.w(w);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut ctx =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);

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
    fn writer_unary_math() {
        let w = ExprWriter::new();

        let x = w.attr(Attribute::POSITION);
        let y = w.lit(Vec3::new(1., -3.1, 6.99));
        let z = w.lit(BVec3::new(false, true, false));
        let ww = w.lit(Vec4::W);
        let v = w.lit(Vec4::new(-1., 1., 0., 7.2));
        let us = w.lit(0x0u32);
        let uu = w.lit(0x0u32);

        let abs = x.abs().expr();
        let all = z.clone().all().expr();
        let any = z.any().expr();
        let ceil = y.clone().ceil().expr();
        let cos = y.clone().cos().expr();
        let exp = y.clone().exp().expr();
        let exp2 = y.clone().exp2().expr();
        let floor = y.clone().floor().expr();
        let fract = y.clone().fract().expr();
        let inv_sqrt = y.clone().inverse_sqrt().expr();
        let length = y.clone().length().expr();
        let log = y.clone().log().expr();
        let log2 = y.clone().log2().expr();
        let norm = y.clone().normalized().expr();
        let pack4x8snorm = v.clone().pack4x8snorm().expr();
        let pack4x8unorm = v.clone().pack4x8unorm().expr();
        let saturate = y.clone().saturate().expr();
        let sign = y.clone().sign().expr();
        let sin = y.clone().sin().expr();
        let sqrt = y.clone().sqrt().expr();
        let tan = y.tan().expr();
        let unpack4x8snorm = us.unpack4x8snorm().expr();
        let unpack4x8unorm = uu.unpack4x8unorm().expr();
        let comp_x = ww.clone().x().expr();
        let comp_y = ww.clone().y().expr();
        let comp_z = ww.clone().z().expr();
        let comp_w = ww.w().expr();

        let module = w.finish();

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut ctx =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);

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
            (inv_sqrt, "inverseSqrt", "vec3<f32>(1.,-3.1,6.99)"),
            (length, "length", "vec3<f32>(1.,-3.1,6.99)"),
            (log, "log", "vec3<f32>(1.,-3.1,6.99)"),
            (log2, "log2", "vec3<f32>(1.,-3.1,6.99)"),
            (norm, "normalize", "vec3<f32>(1.,-3.1,6.99)"),
            (pack4x8snorm, "pack4x8snorm", "vec4<f32>(-1.,1.,0.,7.2)"),
            (pack4x8unorm, "pack4x8unorm", "vec4<f32>(-1.,1.,0.,7.2)"),
            (saturate, "saturate", "vec3<f32>(1.,-3.1,6.99)"),
            (sign, "sign", "vec3<f32>(1.,-3.1,6.99)"),
            (sin, "sin", "vec3<f32>(1.,-3.1,6.99)"),
            (sqrt, "sqrt", "vec3<f32>(1.,-3.1,6.99)"),
            (tan, "tan", "vec3<f32>(1.,-3.1,6.99)"),
            (unpack4x8snorm, "unpack4x8snorm", "0u"),
            (unpack4x8unorm, "unpack4x8unorm", "0u"),
        ] {
            let expr = ctx.eval(&module, expr);
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
            let expr = ctx.eval(&module, expr);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            assert_eq!(expr, format!("{}.{}", inner, op));
        }
    }

    #[test]
    fn writer_binary_math() {
        let w = ExprWriter::new();

        let x = w.attr(Attribute::POSITION);
        let y = w.lit(Vec3::ONE);

        let cross = x.clone().cross(y.clone()).expr();
        let dist = x.clone().distance(y.clone()).expr();
        let dot = x.clone().dot(y.clone()).expr();
        let min = x.clone().min(y.clone()).expr();
        let max = x.clone().max(y.clone()).expr();
        let step = x.step(y).expr();

        let module = w.finish();

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut ctx =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);

        for (expr, op) in [
            (cross, "cross"),
            (dist, "distance"),
            (dot, "dot"),
            (min, "min"),
            (max, "max"),
            (step, "step"),
        ] {
            let expr = ctx.eval(&module, expr);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            let expected = if op == "step" {
                // swapped: x.step(edge) yields step(edge, x)
                format!(
                    "{}(vec3<f32>(1.,1.,1.), particle.{})",
                    op,
                    Attribute::POSITION.name(),
                )
            } else {
                format!(
                    "{}(particle.{}, vec3<f32>(1.,1.,1.))",
                    op,
                    Attribute::POSITION.name(),
                )
            };
            assert_eq!(expr, expected);
        }
    }

    #[test]
    fn writer_ternary_math() {
        let w = ExprWriter::new();

        let x = w.attr(Attribute::POSITION);
        let y = w.lit(Vec3::ONE);
        let t = w.lit(0.3);

        let mix = x.clone().mix(y.clone(), t.clone()).expr();
        let smoothstep = x.clone().smoothstep(x.clone(), y).expr();

        let module = w.finish();

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut ctx =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);

        for (expr, op, third) in [(mix, "mix", t.expr()), (smoothstep, "smoothstep", x.expr())] {
            let expr = ctx.eval(&module, expr);
            assert!(expr.is_ok());
            let expr = expr.unwrap();
            let third = ctx.eval(&module, third).unwrap();
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
        let mut ctx =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);

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
    fn func() {
        let mut m = Module::default();

        let func_name = "my_func";
        let parameters = vec![
            FuncParam::new("x", ScalarType::Float),
            FuncParam::new("other", VectorType::VEC3F),
        ];
        let return_type = Some(ValueType::Scalar(ScalarType::Bool));
        let handle = m.add_fn(func_name, parameters.clone(), return_type, |m| {
            Ok(m.raw_code("".to_string(), return_type))
        });

        assert!(handle.is_ok());
        let handle = handle.unwrap();
        let func = m.get_fn(handle);
        assert!(func.is_some());
        let func = func.unwrap();
        assert_eq!(func.name, func_name);
        assert_eq!(func.parameters, parameters);
        assert_eq!(func.return_type, return_type);

        let pt = m.lit(0.75);
        let call = m.call_fn(handle, &[pt]);

        let property_layout = PropertyLayout::empty();
        let particle_layout = ParticleLayout::new().build();
        let mut ctx =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
        let s = ctx
            .eval(&m, call)
            .expect("Failed to evaluate function call");
        assert_eq!(s, format!("{}(0.75)", func_name));
    }

    #[test]
    fn attribute_pointer() {
        let mut m = Module::default();
        let x = m.attr(Attribute::POSITION);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut ctx =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);

        let res = ctx.eval(&m, x);
        assert!(res.is_ok());
        let xx = res.ok().unwrap();
        assert_eq!(xx, format!("particle.{}", Attribute::POSITION.name()));

        // Use a different context; it's invalid to reuse a mutated context, as the
        // expression cache will have been generated with the wrong context.
        let mut ctx =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout)
                .with_attribute_pointer();

        let res = ctx.eval(&m, x);
        assert!(res.is_ok());
        let xx = res.ok().unwrap();
        assert_eq!(xx, format!("(*particle).{}", Attribute::POSITION.name()));
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

        let p = m.add_property("my_prop", 3.0.into());
        let y = m.prop(p);
        let c = CastExpr::new(y, MatrixType::MAT2X3F);
        assert_eq!(c.value_type(), ValueType::Matrix(MatrixType::MAT2X3F));
        assert_eq!(c.is_valid(&m), Some(false)); // invalid cast * -> matrix
    }

    #[test]
    fn side_effect() {
        let w = ExprWriter::new();

        // Adding the same cloned expression with side effect to itself should yield
        // twice the value, and not two separate evaluations of the expression.
        // CORRECT:
        //   let r = frand();
        //   r + r
        // INCORRECT:
        //   frand() + frand()

        let r = w.push(Expr::BuiltIn(BuiltInExpr::new(BuiltInOperator::Rand(
            ScalarType::Float.into(),
        ))));
        let r2 = r.clone();
        let r3 = r2.clone();
        let a = r.clone().add(r2.clone());
        let b = r.mix(r2, r3).expr();
        let c = a.clone().abs().expr();
        let a = a.expr();

        let module = w.finish();

        {
            let property_layout = PropertyLayout::default();
            let particle_layout = ParticleLayout::default();
            let mut ctx =
                ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
            let value = ctx.eval(&module, a).unwrap();
            assert_eq!(value, "(var0) + (var0)");
            assert_eq!(ctx.main_code, "let var0 = frand();\n");
        }

        {
            let property_layout = PropertyLayout::default();
            let particle_layout = ParticleLayout::default();
            let mut ctx =
                ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
            let value = ctx.eval(&module, b).unwrap();
            assert_eq!(value, "mix(var0, var0, var0)");
            assert_eq!(ctx.main_code, "let var0 = frand();\n");
        }

        {
            let property_layout = PropertyLayout::default();
            let particle_layout = ParticleLayout::default();
            let mut ctx =
                ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
            let value = ctx.eval(&module, c).unwrap();
            assert_eq!(value, "abs((var0) + (var0))");
            assert_eq!(ctx.main_code, "let var0 = frand();\n");
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
