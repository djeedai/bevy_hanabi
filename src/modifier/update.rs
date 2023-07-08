//! Modifiers to influence the update behavior of particle each frame.
//!
//! The update modifiers control how particles are updated each frame by the
//! update compute shader.

use std::hash::{Hash, Hasher};

use bevy::{prelude::*, utils::FloatOrd};
use serde::{Deserialize, Serialize};

use crate::{
    calc_func_id,
    graph::{BuiltInExpr, BuiltInOperator, EvalContext, Expr, ExprError, Value},
    Attribute, BoxedModifier, ExprHandle, Modifier, ModifierContext, Module, Property,
    PropertyLayout, ToWgslString,
};

/// Particle update shader code generation context.
#[derive(Debug, PartialEq)]
pub struct UpdateContext<'a> {
    /// Module being populated with new expressions from modifiers.
    pub module: &'a mut Module,
    /// Main particle update code, which needs to update the fields of the
    /// `particle` struct instance.
    pub update_code: String,
    /// Extra functions emitted at top level, which `update_code` can call.
    pub update_extra: String,
    /// Layout of properties for the current effect.
    pub property_layout: &'a PropertyLayout,

    // TEMP
    /// Array of force field components with a maximum number of components
    /// determined by [`ForceFieldSource::MAX_SOURCES`].
    pub force_field: [ForceFieldSource; ForceFieldSource::MAX_SOURCES],
}

impl<'a> UpdateContext<'a> {
    /// Create a new update context.
    pub fn new(module: &'a mut Module, property_layout: &'a PropertyLayout) -> Self {
        Self {
            module,
            update_code: String::new(),
            update_extra: String::new(),
            property_layout,
            force_field: [ForceFieldSource::default(); ForceFieldSource::MAX_SOURCES],
        }
    }
}

impl<'a> EvalContext for UpdateContext<'a> {
    fn module(&self) -> &Module {
        self.module
    }

    fn property_layout(&self) -> &PropertyLayout {
        self.property_layout
    }

    fn expr(&self, expr: ExprHandle) -> Result<&Expr, ExprError> {
        self.module
            .get(expr)
            .ok_or(ExprError::GraphEvalError("Unknown expression.".to_string()))
    }

    fn eval(&self, handle: ExprHandle) -> Result<String, ExprError> {
        self.expr(handle)?.eval(self)
    }
}

/// Trait to customize the updating of existing particles each frame.
#[typetag::serde]
pub trait UpdateModifier: Modifier {
    /// Append the update code.
    fn apply(&self, context: &mut UpdateContext) -> Result<(), ExprError>;
}

/// Macro to implement the [`Modifier`] trait for an update modifier.
macro_rules! impl_mod_update {
    ($t:ty, $attrs:expr) => {
        #[typetag::serde]
        impl Modifier for $t {
            fn context(&self) -> ModifierContext {
                ModifierContext::Update
            }

            fn as_update(&self) -> Option<&dyn UpdateModifier> {
                Some(self)
            }

            fn as_update_mut(&mut self) -> Option<&mut dyn UpdateModifier> {
                Some(self)
            }

            fn attributes(&self) -> &[Attribute] {
                $attrs
            }

            fn boxed_clone(&self) -> BoxedModifier {
                Box::new(self.clone())
            }
        }
    };
}

/// Constant value or reference to a named property.
///
/// This enumeration either directly stores a constant value assigned at
/// creation time, or a reference to an effect property the value is derived
/// from.
#[derive(Debug, Clone, PartialEq, Hash, Reflect, FromReflect, Serialize, Deserialize)]
pub enum ValueOrProperty {
    /// Constant value.
    Value(Value),
    /// Unresolved reference to a named property the value is derived from.
    Property(String),
    /// Reference to a named property the value is derived from, resolved to the
    /// index of that property in the owning [`EffectAsset`].
    ///
    /// [`EffectAsset`]: crate::EffectAsset
    ResolvedProperty((usize, String)),
}

impl From<&str> for ValueOrProperty {
    fn from(value: &str) -> Self {
        Self::Property(value.to_string())
    }
}

impl ToWgslString for ValueOrProperty {
    fn to_wgsl_string(&self) -> String {
        match self {
            ValueOrProperty::Value(value) => value.to_wgsl_string(),
            ValueOrProperty::Property(name) => format!("properties.{}", name),
            ValueOrProperty::ResolvedProperty((_, name)) => format!("properties.{}", name),
        }
    }
}

/// A modifier to apply a uniform acceleration to all particles each frame, to
/// simulate gravity or any other global force.
///
/// The acceleration is the same for all particles of the effect, and is applied
/// each frame to modify the particle's velocity based on the simulation
/// timestep.
///
/// ```txt
/// particle.velocity += acceleration * simulation.delta_time;
/// ```
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Clone, Copy, Reflect, FromReflect, Serialize, Deserialize)]
pub struct AccelModifier {
    /// The acceleration to apply to all particles in the effect each frame.
    accel: ExprHandle,
}

impl AccelModifier {
    /// Create a new modifier from an acceleration expression.
    pub fn new(accel: ExprHandle) -> Self {
        Self { accel }
    }

    /// Create a new modifier with an acceleration derived from a property.
    pub fn via_property(module: &mut Module, property_name: impl Into<String>) -> Self {
        Self {
            accel: module.prop(property_name),
        }
    }

    /// Create a new modifier with a constant acceleration.
    pub fn constant(module: &mut Module, acceleration: Vec3) -> Self {
        Self {
            accel: module.lit(acceleration),
        }
    }
}

#[typetag::serde]
impl Modifier for AccelModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Update
    }

    fn as_update(&self) -> Option<&dyn UpdateModifier> {
        Some(self)
    }

    fn as_update_mut(&mut self) -> Option<&mut dyn UpdateModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[Attribute] {
        &[Attribute::VELOCITY]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn resolve_properties(&mut self, _properties: &[Property]) {
        // if let Some(index) = properties
        //     .iter()
        //     .position(|p| p.name() == self.property_name)
        // {
        //     let name = std::mem::take(&mut self.property_name);
        //     self.accel = ValueOrProperty::ResolvedProperty((index, name));
        // } else {
        //     panic!("Cannot resolve property '{}' in effect. Ensure you have
        // added a property with `with_property()` or `add_property()` before
        // trying to reference it from a modifier.", name); }
    }
}

#[typetag::serde]
impl UpdateModifier for AccelModifier {
    fn apply(&self, context: &mut UpdateContext) -> Result<(), ExprError> {
        let attr = context.module.attr(Attribute::VELOCITY);
        let attr = context.eval(attr)?;
        let expr = context.eval(self.accel)?;
        let dt = BuiltInExpr::new(crate::graph::BuiltInOperator::DeltaTime).eval(context)?;
        context.update_code += &format!("{} += ({}) * {};", attr, expr, dt);
        Ok(())
    }
}

/// A modifier to apply a radial acceleration to all particles each frame.
///
/// The acceleration is the same for all particles of the effect, and is applied
/// each frame to modify the particle's velocity based on the simulation
/// timestep.
///
/// ```txt
/// particle.velocity += acceleration * simulation.delta_time;
/// ```
///
/// In the absence of other modifiers, the radial acceleration alone, if
/// oriented toward the center point, makes particles rotate around that center
/// point. The radial direction is calculated as the direction from the modifier
/// origin to the particle position.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Clone, Reflect, FromReflect, Serialize, Deserialize)]
pub struct RadialAccelModifier {
    /// The acceleration to apply to all particles in the effect each frame.
    accel: ValueOrProperty,
    /// The center point the tangent direction is calculated from.
    origin: Vec3,
}

impl PartialEq for RadialAccelModifier {
    fn eq(&self, other: &Self) -> bool {
        self.accel == other.accel
            && FloatOrd(self.origin.x) == FloatOrd(other.origin.x)
            && FloatOrd(self.origin.y) == FloatOrd(other.origin.y)
            && FloatOrd(self.origin.z) == FloatOrd(other.origin.z)
    }
}

impl Hash for RadialAccelModifier {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.accel.hash(state);
        FloatOrd(self.origin.x).hash(state);
        FloatOrd(self.origin.y).hash(state);
        FloatOrd(self.origin.z).hash(state);
    }
}

impl RadialAccelModifier {
    /// Create a new modifier with an acceleration derived from a property.
    pub fn via_property(origin: Vec3, property_name: impl Into<String>) -> Self {
        Self {
            accel: ValueOrProperty::Property(property_name.into()),
            origin,
        }
    }

    /// Create a new modifier with a constant acceleration.
    pub fn constant(origin: Vec3, acceleration: f32) -> Self {
        Self {
            accel: ValueOrProperty::Value(Value::Scalar(acceleration.into())),
            origin,
        }
    }
}

#[typetag::serde]
impl Modifier for RadialAccelModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Update
    }

    fn as_update(&self) -> Option<&dyn UpdateModifier> {
        Some(self)
    }

    fn as_update_mut(&mut self) -> Option<&mut dyn UpdateModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[Attribute] {
        &[Attribute::POSITION, Attribute::VELOCITY]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(self.clone())
    }

    fn resolve_properties(&mut self, properties: &[Property]) {
        if let ValueOrProperty::Property(name) = &mut self.accel {
            if let Some(index) = properties.iter().position(|p| p.name() == name) {
                let name = std::mem::take(name);
                self.accel = ValueOrProperty::ResolvedProperty((index, name));
            } else {
                panic!("Cannot resolve property '{}' in effect. Ensure you have added a property with `with_property()` or `add_property()` before trying to reference it from a modifier.", name);
            }
        }
    }
}

#[typetag::serde]
impl UpdateModifier for RadialAccelModifier {
    fn apply(&self, context: &mut UpdateContext) -> Result<(), ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("radial_accel_{0:016X}", func_id);

        context.update_extra += &format!(
            r##"fn {}(particle: ptr<function, Particle>) {{
    let radial = normalize((*particle).{} - {});
    (*particle).{} += radial * (({}) * sim_params.delta_time);
}}
"##,
            func_name,
            Attribute::POSITION.name(),
            self.origin.to_wgsl_string(),
            Attribute::VELOCITY.name(),
            self.accel.to_wgsl_string(),
        );

        context.update_code += &format!("{}(&particle);\n", func_name);

        Ok(())
    }
}

/// A modifier to apply a tangential acceleration to all particles each frame.
///
/// The acceleration is the same for all particles of the effect, and is applied
/// each frame to modify the particle's velocity based on the simulation
/// timestep.
///
/// ```txt
/// particle.velocity += acceleration * simulation.delta_time;
/// ```
///
/// In the absence of other modifiers, the tangential acceleration alone makes
/// particles rotate around a center point. The tangent direction is calculated
/// as the cross product of the rotation plane axis and the direction from the
/// modifier origin to the particle position.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Clone, Reflect, FromReflect, Serialize, Deserialize)]
pub struct TangentAccelModifier {
    /// The acceleration to apply to all particles in the effect each frame.
    accel: ValueOrProperty,
    /// The center point the tangent direction is calculated from.
    origin: Vec3,
    /// The axis defining the rotation plane and orientation.
    axis: Vec3,
}

impl PartialEq for TangentAccelModifier {
    fn eq(&self, other: &Self) -> bool {
        self.accel == other.accel
            && FloatOrd(self.origin.x) == FloatOrd(other.origin.x)
            && FloatOrd(self.origin.y) == FloatOrd(other.origin.y)
            && FloatOrd(self.origin.z) == FloatOrd(other.origin.z)
            && FloatOrd(self.axis.x) == FloatOrd(other.axis.x)
            && FloatOrd(self.axis.y) == FloatOrd(other.axis.y)
            && FloatOrd(self.axis.z) == FloatOrd(other.axis.z)
    }
}

impl Hash for TangentAccelModifier {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.accel.hash(state);
        FloatOrd(self.origin.x).hash(state);
        FloatOrd(self.origin.y).hash(state);
        FloatOrd(self.origin.z).hash(state);
        FloatOrd(self.axis.x).hash(state);
        FloatOrd(self.axis.y).hash(state);
        FloatOrd(self.axis.z).hash(state);
    }
}

impl TangentAccelModifier {
    /// Create a new modifier with an acceleration derived from a property.
    pub fn via_property(origin: Vec3, axis: Vec3, property_name: impl Into<String>) -> Self {
        Self {
            accel: ValueOrProperty::Property(property_name.into()),
            origin,
            axis,
        }
    }

    /// Create a new modifier with a constant acceleration.
    pub fn constant(origin: Vec3, axis: Vec3, acceleration: f32) -> Self {
        Self {
            accel: ValueOrProperty::Value(Value::Scalar(acceleration.into())),
            origin,
            axis,
        }
    }
}

#[typetag::serde]
impl Modifier for TangentAccelModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Update
    }

    fn as_update(&self) -> Option<&dyn UpdateModifier> {
        Some(self)
    }

    fn as_update_mut(&mut self) -> Option<&mut dyn UpdateModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[Attribute] {
        &[Attribute::POSITION, Attribute::VELOCITY]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(self.clone())
    }

    fn resolve_properties(&mut self, properties: &[Property]) {
        if let ValueOrProperty::Property(name) = &mut self.accel {
            if let Some(index) = properties.iter().position(|p| p.name() == name) {
                let name = std::mem::take(name);
                self.accel = ValueOrProperty::ResolvedProperty((index, name));
            } else {
                panic!("Cannot resolve property '{}' in effect. Ensure you have added a property with `with_property()` or `add_property()` before trying to reference it from a modifier.", name);
            }
        }
    }
}

#[typetag::serde]
impl UpdateModifier for TangentAccelModifier {
    fn apply(&self, context: &mut UpdateContext) -> Result<(), ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("tangent_accel_{0:016X}", func_id);

        context.update_extra += &format!(
            r##"fn {}(particle: ptr<function, Particle>) {{
    let radial = normalize((*particle).{} - {});
    let tangent = normalize(cross({}, radial));
    (*particle).{} += tangent * (({}) * sim_params.delta_time);
}}
"##,
            func_name,
            Attribute::POSITION.name(),
            self.origin.to_wgsl_string(),
            self.axis.to_wgsl_string(),
            Attribute::VELOCITY.name(),
            self.accel.to_wgsl_string(),
        );

        context.update_code += &format!("{}(&particle);\n", func_name);

        Ok(())
    }
}

/// A single attraction or repulsion source of a [`ForceFieldModifier`].
///
/// The source applies a radial force field to all particles around its
/// position, with a decreasing intensity the further away from the source the
/// particle is. This force is added to the one(s) of all the other active
/// sources of a [`ForceFieldModifier`].
#[derive(Debug, Clone, Copy, Reflect, FromReflect, Serialize, Deserialize)]
pub struct ForceFieldSource {
    /// Position of the source.
    pub position: Vec3,
    /// Maximum radius of the sphere of influence, outside of which
    /// the contribution of this source to the force field is null.
    pub max_radius: f32,
    /// Minimum radius of the sphere of influence, inside of which
    /// the contribution of this source to the force field is null, avoiding the
    /// singularity at the source position.
    pub min_radius: f32,
    /// The intensity of the force of the source is proportional to its mass.
    /// Note that the update shader will ignore all subsequent force field
    /// sources after it encountered a source with a mass of zero. To change
    /// the force from an attracting one to a repulsive one, simply
    /// set the mass to a negative value.
    pub mass: f32,
    /// The source force is proportional to `1 / distance^force_exponent`.
    pub force_exponent: f32,
    /// If `true`, the particles which attempt to come closer than
    /// [`min_radius`] from the source position will conform to a sphere of
    /// radius [`min_radius`] around the source position, appearing like a
    /// recharging effect.
    ///
    /// [`min_radius`]: ForceFieldSource::min_radius
    pub conform_to_sphere: bool,
}

impl PartialEq for ForceFieldSource {
    fn eq(&self, other: &Self) -> bool {
        FloatOrd(self.position.x) == FloatOrd(other.position.x)
            && FloatOrd(self.position.y) == FloatOrd(other.position.y)
            && FloatOrd(self.position.z) == FloatOrd(other.position.z)
            && FloatOrd(self.max_radius) == FloatOrd(other.max_radius)
            && FloatOrd(self.min_radius) == FloatOrd(other.min_radius)
            && FloatOrd(self.mass) == FloatOrd(other.mass)
            && FloatOrd(self.force_exponent) == FloatOrd(other.force_exponent)
            && self.conform_to_sphere == other.conform_to_sphere
    }
}

impl Hash for ForceFieldSource {
    fn hash<H: Hasher>(&self, state: &mut H) {
        FloatOrd(self.position.x).hash(state);
        FloatOrd(self.position.y).hash(state);
        FloatOrd(self.position.z).hash(state);
        FloatOrd(self.max_radius).hash(state);
        FloatOrd(self.min_radius).hash(state);
        FloatOrd(self.mass).hash(state);
        FloatOrd(self.force_exponent).hash(state);
        self.conform_to_sphere.hash(state);
    }
}

impl Default for ForceFieldSource {
    fn default() -> Self {
        // defaults to no force field (a mass of 0)
        Self {
            position: Vec3::new(0., 0., 0.),
            min_radius: 0.1,
            max_radius: 0.0,
            mass: 0.,
            force_exponent: 0.0,
            conform_to_sphere: false,
        }
    }
}

impl ForceFieldSource {
    /// Maximum number of sources in the force field.
    pub const MAX_SOURCES: usize = 16;
}

/// A modifier to apply a force field to all particles each frame. The force
/// field is made up of [`ForceFieldSource`]s.
///
/// The maximum number of sources is [`ForceFieldSource::MAX_SOURCES`].
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
/// - [`Attribute::VELOCITY`]
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Hash, Reflect, FromReflect, Serialize, Deserialize,
)]
pub struct ForceFieldModifier {
    /// Array of force field sources.
    ///
    /// A source can be disabled by setting its [`mass`] to zero. In that case,
    /// all other sources located after it in the array are automatically
    /// disabled too.
    ///
    /// [`mass`]: ForceFieldSource::mass
    pub sources: [ForceFieldSource; ForceFieldSource::MAX_SOURCES],
}

impl ForceFieldModifier {
    /// Instantiate a [`ForceFieldModifier`] with a set of sources.
    ///
    /// # Panics
    ///
    /// Panics if the number of sources exceeds [`MAX_SOURCES`].
    ///
    /// [`MAX_SOURCES`]: ForceFieldSource::MAX_SOURCES
    pub fn new<T>(sources: T) -> Self
    where
        T: IntoIterator<Item = ForceFieldSource>,
    {
        let mut source_array = [ForceFieldSource::default(); ForceFieldSource::MAX_SOURCES];

        for (index, p_attractor) in sources.into_iter().enumerate() {
            if index >= ForceFieldSource::MAX_SOURCES {
                panic!(
                    "Force field source count exceeded maximum of {}",
                    ForceFieldSource::MAX_SOURCES
                );
            }
            source_array[index] = p_attractor;
        }

        Self {
            sources: source_array,
        }
    }

    /// Overwrite the source at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than or equal to [`MAX_SOURCES`].
    ///
    /// [`MAX_SOURCES`]: ForceFieldSource::MAX_SOURCES
    pub fn add_or_replace(&mut self, source: ForceFieldSource, index: usize) {
        self.sources[index] = source;
    }
}

impl_mod_update!(
    ForceFieldModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

#[typetag::serde]
impl UpdateModifier for ForceFieldModifier {
    fn apply(&self, context: &mut UpdateContext) -> Result<(), ExprError> {
        let func_id = calc_func_id(self);
        let func_name = format!("force_field_{0:016X}", func_id);

        context.update_extra += &format!(
            r##"fn {}(particle: ptr<function, Particle>) {{
    {}
}}
"##,
            func_name,
            include_str!("../render/force_field_code.wgsl")
        );

        context.update_code += &format!("{}(&particle);\n", func_name);

        // TEMP
        context.force_field = self.sources;

        Ok(())
    }
}

/// A modifier to apply a linear drag force to all particles each frame. The
/// force slows down the particles without changing their direction.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::VELOCITY`]
#[derive(Debug, Clone, Copy, Reflect, FromReflect, Serialize, Deserialize)]
pub struct LinearDragModifier {
    /// Drag coefficient. Higher values increase the drag force, and
    /// consequently decrease the particle's speed faster.
    pub drag: ExprHandle,
}

impl LinearDragModifier {
    /// Create a new modifier from a drag expression.
    pub fn new(drag: ExprHandle) -> Self {
        Self { drag }
    }

    /// Instantiate a [`LinearDragModifier`] with a constant drag value.
    pub fn constant(module: &mut Module, drag: f32) -> Self {
        Self {
            drag: module.lit(drag),
        }
    }
}

impl_mod_update!(LinearDragModifier, &[Attribute::VELOCITY]);

#[typetag::serde]
impl UpdateModifier for LinearDragModifier {
    fn apply(&self, context: &mut UpdateContext) -> Result<(), ExprError> {
        let m = &mut context.module;
        let attr = m.attr(Attribute::VELOCITY);
        let dt = m.builtin(BuiltInOperator::DeltaTime);
        let drag_dt = m.mul(self.drag, dt);
        let one = m.lit(1.);
        let one_minus_drag_dt = m.sub(one, drag_dt);
        let zero = m.lit(0.);
        let expr = m.max(zero, one_minus_drag_dt);
        let attr = context.eval(attr)?;
        let expr = context.eval(expr)?;
        context.update_code += &format!("{} *= {};", attr, expr);
        Ok(())
    }
}

/// A modifier killing all particles that enter or exit an AABB.
///
/// This enables confining particles to a region in space, or preventing
/// particles to enter that region.
///
/// # Attributes
///
/// This modifier requires the following particle attributes:
/// - [`Attribute::POSITION`]
#[derive(Debug, Clone, Copy, Reflect, FromReflect, Serialize, Deserialize)]
pub struct AabbKillModifier {
    /// Center of the AABB.
    pub center: ExprHandle,
    /// Half-size of the AABB.
    pub half_size: ExprHandle,
    /// If `true`, invert the kill condition and kill all particles inside the
    /// AABB. If `false` (default), kill all particles outside the AABB.
    pub kill_inside: bool,
}

impl AabbKillModifier {
    /// Create a new instance of an [`AabbKillModifier`] from an AABB center and
    /// half extents.
    ///
    /// The created instance has a default `kill_inside = false` value.
    pub fn new(center: impl Into<ExprHandle>, half_size: impl Into<ExprHandle>) -> Self {
        Self {
            center: center.into(),
            half_size: half_size.into(),
            kill_inside: false,
        }
    }

    /// Set whether particles are killed when inside the AABB or not.
    pub fn with_kill_inside(mut self, kill_inside: bool) -> Self {
        self.kill_inside = kill_inside;
        self
    }
}

impl_mod_update!(AabbKillModifier, &[Attribute::POSITION]);

#[typetag::serde]
impl UpdateModifier for AabbKillModifier {
    fn apply(&self, context: &mut UpdateContext) -> Result<(), ExprError> {
        let pos = context.module.attr(Attribute::POSITION);
        let diff = context.module.sub(pos, self.center);
        let dist = context.module.abs(diff);
        let cmp = if self.kill_inside {
            context.module.lt(dist, self.half_size)
        } else {
            context.module.gt(dist, self.half_size)
        };
        let reduce = if self.kill_inside {
            context.module.all(cmp)
        } else {
            context.module.any(cmp)
        };
        let expr = context.eval(reduce)?;

        context.update_code += &format!(
            r#"if ({}) {{
    is_alive = false;
}}
"#,
            expr
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{ExprWriter, ParticleLayout};

    use super::*;

    use naga::front::wgsl::Frontend;

    #[test]
    fn mod_accel() {
        let mut module = Module::default();
        let accel = Vec3::new(1., 2., 3.);
        let modifier = AccelModifier::constant(&mut module, accel);

        let property_layout = PropertyLayout::default();
        let mut context = UpdateContext::new(&mut module, &property_layout);
        assert!(modifier.apply(&mut context).is_ok());

        assert!(context.update_code.contains(&accel.to_wgsl_string()));
    }

    #[test]
    fn mod_force_field() {
        let position = Vec3::new(1., 2., 3.);
        let mut sources = [ForceFieldSource::default(); 16];
        sources[0].position = position;
        sources[0].mass = 1.;
        let modifier = ForceFieldModifier { sources };

        let property_layout = PropertyLayout::default();
        let mut module = Module::default();
        let mut context = UpdateContext::new(&mut module, &property_layout);
        assert!(modifier.apply(&mut context).is_ok());

        // force_field_code.wgsl is too big
        // assert!(context.update_code.contains(&include_str!("../render/
        // force_field_code.wgsl")));
    }

    #[test]
    #[should_panic]
    fn mod_force_field_new_too_many() {
        let count = ForceFieldSource::MAX_SOURCES + 1;
        ForceFieldModifier::new((0..count).map(|_| ForceFieldSource::default()));
    }

    #[test]
    fn mod_drag() {
        let mut module = Module::default();
        let modifier = LinearDragModifier::constant(&mut module, 3.5);

        let property_layout = PropertyLayout::default();
        let mut context = UpdateContext::new(&mut module, &property_layout);
        assert!(modifier.apply(&mut context).is_ok());

        assert!(context.update_code.contains("3.5")); // TODO - less weak check
    }

    #[test]
    fn validate() {
        let writer = ExprWriter::new();
        let modifiers: &[&dyn UpdateModifier] = &[
            &AccelModifier::new(writer.lit(Vec3::ONE).expr()),
            &RadialAccelModifier::constant(Vec3::ZERO, 1.),
            &TangentAccelModifier::constant(Vec3::ZERO, Vec3::Z, 1.),
            &ForceFieldModifier::default(),
            &LinearDragModifier::new(writer.lit(3.5).expr()),
            &AabbKillModifier::new(writer.lit(Vec3::ZERO).expr(), writer.lit(Vec3::ONE).expr()),
        ];
        let mut module = writer.finish();
        for &modifier in modifiers.iter() {
            let property_layout = PropertyLayout::default();
            let mut context = UpdateContext::new(&mut module, &property_layout);
            assert!(modifier.apply(&mut context).is_ok());
            let update_code = context.update_code;
            let update_extra = context.update_extra;

            let mut particle_layout = ParticleLayout::new();
            for &attr in modifier.attributes() {
                particle_layout = particle_layout.append(attr);
            }
            let particle_layout = particle_layout.build();
            let attributes_code = particle_layout.generate_code();

            let code = format!(
                r##"fn rand() -> f32 {{
    return 0.0;
}}

const tau: f32 = 6.283185307179586476925286766559;

struct Particle {{
    {attributes_code}
}};

struct ParticleBuffer {{
    particles: array<Particle>,
}};

struct SimParams {{
    delta_time: f32,
    time: f32,
}};

struct ForceFieldSource {{
    position: vec3<f32>,
    max_radius: f32,
    min_radius: f32,
    mass: f32,
    force_exponent: f32,
    conform_to_sphere: f32,
}};

struct Spawner {{
    transform: mat3x4<f32>, // transposed (row-major)
    spawn: atomic<i32>,
    seed: u32,
    count_unused: u32,
    effect_index: u32,
    force_field: array<ForceFieldSource, 16>,
}};

fn proj(u: vec3<f32>, v: vec3<f32>) -> vec3<f32> {{
    return dot(v, u) / dot(u,u) * u;
}}

{update_extra}

@group(0) @binding(0) var<uniform> sim_params : SimParams;
@group(1) @binding(0) var<storage, read_write> particle_buffer : ParticleBuffer;
@group(2) @binding(0) var<storage, read_write> spawner : Spawner; // NOTE - same group as init

@compute @workgroup_size(64)
fn main() {{
    var particle: Particle = particle_buffer.particles[0];
    var transform: mat4x4<f32> = mat4x4<f32>();
    var is_alive = true;
{update_code}
}}"##
            );

            let mut frontend = Frontend::new();
            let res = frontend.parse(&code);
            if let Err(err) = &res {
                println!("Modifier: {:?}", modifier.type_name());
                println!("Code: {:?}", code);
                println!("Err: {:?}", err);
            }
            assert!(res.is_ok());
        }
    }
}
