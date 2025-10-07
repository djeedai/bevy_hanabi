//! Building blocks to create a visual effect.
//!
//! A **modifier** is a building block used to define the behavior of an effect.
//! Particles effects are composed of multiple modifiers, which put together and
//! configured produce the desired visual effect. Each modifier changes a
//! specific part of the behavior of an effect. Modifiers are grouped in three
//! categories:
//!
//! - **Init modifiers** influence the initializing of particles when they
//!   spawn. They typically configure the initial position and/or velocity of
//!   particles. Init modifiers implement the [`Modifier`] trait, and act on the
//!   [`ModifierContext::Init`] modifier context.
//! - **Update modifiers** influence the particle update loop each frame. For
//!   example, an update modifier can apply a gravity force to all particles.
//!   Update modifiers implement the [`Modifier`] trait, and act on the
//!   [`ModifierContext::Update`] modifier context.
//! - **Render modifiers** influence the rendering of each particle. They can
//!   change the particle's color, or orient it to face the camera. Render
//!   modifiers implement the [`RenderModifier`] trait, and act on the
//!   [`ModifierContext::Render`] modifier context.
//!
//! A single modifier can be part of multiple categories. For example, the
//! [`SetAttributeModifier`] can be used either to initialize a particle's
//! attribute on spawning, or to assign a value to that attribute each frame
//! during simulation (update).
//!
//! # Modifiers and expressions
//!
//! Modifiers are configured by assigning values to their field(s). Some values
//! are compile-time constants, like which attribute a [`SetAttributeModifier`]
//! mutates. Others however can take the form of
//! [expressions](crate::graph::expr), which form a mini language designed to
//! emit shader code and provide extended customization. For example, a 3D
//! vector position can be assigned to a [property](crate::properties) and
//! mutated each frame, giving CPU-side control over the behavior of the GPU
//! particle effect. See [expressions](crate::graph::expr) for more details.
//!
//! # Limitations
//!
//! At this time, serialization and deserialization of modifiers is not
//! supported on Wasm. This means assets authored and saved on a non-Wasm target
//! cannot be read back into an application running on Wasm.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use bevy::{
    asset::Handle,
    image::Image,
    math::{UVec2, Vec3, Vec4},
    platform::collections::HashMap,
    reflect::Reflect,
};
use bitflags::bitflags;
use serde::{Deserialize, Serialize};

pub mod accel;
pub mod attr;
pub mod force;
pub mod kill;
pub mod output;
pub mod position;
pub mod velocity;

pub use accel::*;
pub use attr::*;
pub use force::*;
pub use kill::*;
pub use output::*;
pub use position::*;
pub use velocity::*;

use crate::{
    Attribute, EvalContext, ExprError, ExprHandle, Gradient, Module, ParticleLayout,
    PropertyLayout, TextureLayout,
};

/// The dimension of a shape to consider.
///
/// The exact meaning depends on the context where this enum is used.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum ShapeDimension {
    /// Consider the surface of the shape only.
    #[default]
    Surface,
    /// Consider the entire shape volume.
    Volume,
}

/// Calculate a function ID by hashing the given value representative of the
/// function.
pub(crate) fn calc_func_id<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

bitflags! {
    /// Context a modifier applies to.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ModifierContext : u8 {
        /// Particle initializing on spawning.
        ///
        /// Modifiers in the init context are executed for each newly spawned
        /// particle, to initialize that particle.
        const Init = 0b001;
        /// Particle simulation (update).
        ///
        /// Modifiers in the update context are executed each frame to simulate
        /// the particle behavior.
        const Update = 0b010;
        /// Particle rendering.
        ///
        /// Modifiers in the render context are executed for each view (camera)
        /// where a particle is visible, each frame.
        const Render = 0b100;
    }
}

impl std::fmt::Display for ModifierContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = if self.contains(ModifierContext::Init) {
            "Init".to_string()
        } else {
            String::new()
        };
        if self.contains(ModifierContext::Update) {
            if s.is_empty() {
                s = "Update".to_string();
            } else {
                s += " | Update";
            }
        }
        if self.contains(ModifierContext::Render) {
            if s.is_empty() {
                s = "Render".to_string();
            } else {
                s += " | Render";
            }
        }
        if s.is_empty() {
            s = "None".to_string();
        }
        write!(f, "{}", s)
    }
}

/// Trait describing a modifier customizing an effect pipeline.
#[cfg_attr(feature = "serde", typetag::serde)]
pub trait Modifier: Reflect + Send + Sync + 'static {
    /// Get the context this modifier applies to.
    fn context(&self) -> ModifierContext;

    /// Try to cast this modifier to a [`RenderModifier`].
    fn as_render(&self) -> Option<&dyn RenderModifier> {
        None
    }

    /// Try to cast this modifier to a [`RenderModifier`].
    fn as_render_mut(&mut self) -> Option<&mut dyn RenderModifier> {
        None
    }

    /// Get the list of attributes required for this modifier to be used.
    fn attributes(&self) -> &[Attribute];

    /// Clone self.
    fn boxed_clone(&self) -> BoxedModifier;

    /// Apply the modifier to generate code.
    fn apply(&self, module: &mut Module, context: &mut ShaderWriter) -> Result<(), ExprError>;
}

/// Boxed version of [`Modifier`].
pub type BoxedModifier = Box<dyn Modifier>;

impl Clone for BoxedModifier {
    fn clone(&self) -> Self {
        self.boxed_clone()
    }
}

/// Shader code writer.
///
/// Writer utility to generate shader code. The writer works in a defined
/// context, for a given [`ModifierContext`] and a particular effect setup
/// ([`ParticleLayout`] and [`PropertyLayout`]).
#[derive(Debug, PartialEq)]
pub struct ShaderWriter<'a> {
    /// Main shader compute code emitted.
    ///
    /// This is the WGSL code emitted into the target [`ModifierContext`]. The
    /// context dictates what variables are available (this is currently
    /// implicit and requires knownledge of the target context; there's little
    /// validation that the emitted code is valid).
    pub main_code: String,
    /// Extra functions emitted at shader top level.
    ///
    /// This contains optional WGSL code emitted at shader top level. This
    /// generally contains functions called from `main_code`.
    pub extra_code: String,
    /// Layout of properties for the current effect.
    pub property_layout: &'a PropertyLayout,
    /// Layout of attributes of a particle for the current effect.
    pub particle_layout: &'a ParticleLayout,
    /// Modifier context the writer is being used from.
    modifier_context: ModifierContext,
    /// Counter for unique variable names.
    var_counter: u32,
    /// Cache of evaluated expressions.
    expr_cache: HashMap<ExprHandle, String>,
    /// Is the attribute struct a pointer?
    is_attribute_pointer: bool,
    /// Is the shader using GPU spawn events?
    emits_gpu_spawn_events: Option<bool>,
}

impl<'a> ShaderWriter<'a> {
    /// Create a new init context.
    pub fn new(
        modifier_context: ModifierContext,
        property_layout: &'a PropertyLayout,
        particle_layout: &'a ParticleLayout,
    ) -> Self {
        Self {
            main_code: String::new(),
            extra_code: String::new(),
            property_layout,
            particle_layout,
            modifier_context,
            var_counter: 0,
            expr_cache: Default::default(),
            is_attribute_pointer: false,
            emits_gpu_spawn_events: None,
        }
    }

    /// Mark the attribute struct as being available through a pointer.
    pub fn with_attribute_pointer(mut self) -> Self {
        self.is_attribute_pointer = true;
        self
    }

    /// Mark the shader as emitting GPU spawn events.
    ///
    /// This is used by the [`EmitSpawnEventModifier`] to declare that the
    /// current effect emits GPU spawn events, and therefore needs an event
    /// buffer to be allocated and the appropriate compute work to be executed
    /// to fill that buffer with events.
    ///
    /// # Returns
    ///
    /// Returns an error if another modifier previously called this function
    /// with a different value of `use_events`. Calling this function with the
    /// same value is a no-op, and doesn't generate any error.
    pub fn set_emits_gpu_spawn_events(&mut self, use_events: bool) -> Result<(), ExprError> {
        if let Some(was_using_events) = self.emits_gpu_spawn_events {
            if was_using_events == use_events {
                Ok(())
            } else {
                Err(ExprError::GraphEvalError(
                    "Conflicting use of GPU spawn events.".to_string(),
                ))
                // FIXME - Should probably be a validation error instead...
                // Err(ShaderGenerateError::Validate(
                //     "Conflicting use of GPU spawn events.".to_string(),
                // ))
            }
        } else {
            self.emits_gpu_spawn_events = Some(use_events);
            Ok(())
        }
    }

    /// Check whether this shader emits GPU spawn events.
    ///
    /// If no modifier called [`set_emits_gpu_spawn_events()`], this returns
    /// `None`. Otherwise this returns `Some(value)` where `value` was the value
    /// passed to [`set_emits_gpu_spawn_events()`].
    ///
    /// [`set_emits_gpu_spawn_events()`]: crate::ShaderWriter::set_emits_gpu_spawn_events
    pub fn emits_gpu_spawn_events(&self) -> Option<bool> {
        self.emits_gpu_spawn_events
    }
}

impl EvalContext for ShaderWriter<'_> {
    fn modifier_context(&self) -> ModifierContext {
        self.modifier_context
    }

    fn property_layout(&self) -> &PropertyLayout {
        self.property_layout
    }

    fn particle_layout(&self) -> &ParticleLayout {
        self.particle_layout
    }

    fn eval(&mut self, module: &Module, handle: ExprHandle) -> Result<String, ExprError> {
        // On cache hit, don't re-evaluate the expression to prevent any duplicate
        // side-effect.
        if let Some(s) = self.expr_cache.get(&handle) {
            Ok(s.clone())
        } else {
            module.try_get(handle)?.eval(module, self).inspect(|s| {
                self.expr_cache.insert(handle, s.clone());
            })
        }
    }

    fn make_local_var(&mut self) -> String {
        let index = self.var_counter;
        self.var_counter += 1;
        format!("var{}", index)
    }

    fn push_stmt(&mut self, stmt: &str) {
        self.main_code += stmt;
        self.main_code += "\n";
    }

    fn make_fn(
        &mut self,
        func_name: &str,
        args: &str,
        module: &mut Module,
        f: &mut dyn FnMut(&mut Module, &mut dyn EvalContext) -> Result<String, ExprError>,
    ) -> Result<(), ExprError> {
        // Generate a temporary context for the function content itself
        // FIXME - Dynamic with_attribute_pointer()!
        let mut ctx = ShaderWriter::new(
            self.modifier_context,
            self.property_layout,
            self.particle_layout,
        )
        .with_attribute_pointer();

        // Evaluate the function content
        let body = f(module, &mut ctx)?;

        // Append any extra
        self.extra_code += &ctx.extra_code;

        // Append the function itself
        self.extra_code += &format!(
            r##"fn {0}({1}) {{
{2}{3}}}"##,
            func_name, args, ctx.main_code, body
        );

        Ok(())
    }

    fn is_attribute_pointer(&self) -> bool {
        self.is_attribute_pointer
    }
}

/// Particle rendering shader code generation context.
#[derive(Debug, PartialEq)]
pub struct RenderContext<'a> {
    /// Layout of properties for the current effect.
    pub property_layout: &'a PropertyLayout,
    /// Layout of attributes of a particle for the current effect.
    pub particle_layout: &'a ParticleLayout,
    /// Main particle rendering code for the vertex shader.
    pub vertex_code: String,
    /// Main particle rendering code for the fragment shader.
    pub fragment_code: String,
    /// Extra functions emitted at top level, which `vertex_code` and
    /// `fragment_code` can call.
    pub render_extra: String,
    /// Texture layout.
    pub texture_layout: &'a TextureLayout,
    /// Effect textures.
    pub textures: Vec<Handle<Image>>,
    /// Flipbook sprite sheet grid size, if any.
    pub sprite_grid_size: Option<UVec2>,
    /// Color gradients.
    pub gradients: HashMap<u64, Gradient<Vec4>>,
    /// Size gradients.
    pub size_gradients: HashMap<u64, Gradient<Vec3>>,
    /// The particle needs UV coordinates to sample one or more texture(s).
    pub needs_uv: bool,
    /// The particle needs normals for lighting effects.
    pub needs_normal: bool,
    /// The particle needs access to its data in the fragment shader.
    pub needs_particle_fragment: bool,
    /// Counter for unique variable names.
    var_counter: u32,
    /// Cache of evaluated expressions.
    expr_cache: HashMap<ExprHandle, String>,
    /// Is the attriubute struct a pointer?
    is_attribute_pointer: bool,
}

impl<'a> RenderContext<'a> {
    /// Create a new update context.
    pub fn new(
        property_layout: &'a PropertyLayout,
        particle_layout: &'a ParticleLayout,
        texture_layout: &'a TextureLayout,
    ) -> Self {
        Self {
            property_layout,
            particle_layout,
            vertex_code: String::new(),
            fragment_code: String::new(),
            render_extra: String::new(),
            texture_layout,
            textures: vec![],
            sprite_grid_size: None,
            gradients: HashMap::default(),
            size_gradients: HashMap::default(),
            needs_uv: false,
            needs_normal: false,
            needs_particle_fragment: false,
            var_counter: 0,
            expr_cache: Default::default(),
            is_attribute_pointer: false,
        }
    }

    /// Mark the rendering shader as needing UVs.
    pub fn set_needs_uv(&mut self) {
        self.needs_uv = true;
    }

    /// Mark the rendering shader as needing normals.
    pub fn set_needs_normal(&mut self) {
        self.needs_normal = true;
    }

    /// Mark the rendering shader as needing particle data in the fragment
    /// shader.
    pub fn set_needs_particle_fragment(&mut self) {
        self.needs_particle_fragment = true;
    }

    /// Add a color gradient.
    ///
    /// # Returns
    ///
    /// Returns the unique name of the gradient, to be used as function name in
    /// the shader code.
    fn add_color_gradient(&mut self, gradient: Gradient<Vec4>) -> String {
        let func_id = calc_func_id(&gradient);
        self.gradients.insert(func_id, gradient);
        let func_name = format!("color_gradient_{0:016X}", func_id);
        func_name
    }

    /// Add a size gradient.
    ///
    /// # Returns
    ///
    /// Returns the unique name of the gradient, to be used as function name in
    /// the shader code.
    fn add_size_gradient(&mut self, gradient: Gradient<Vec3>) -> String {
        let func_id = calc_func_id(&gradient);
        self.size_gradients.insert(func_id, gradient);
        let func_name = format!("size_gradient_{0:016X}", func_id);
        func_name
    }

    /// Mark the attribute struct as being available through a pointer.
    pub fn with_attribute_pointer(mut self) -> Self {
        self.is_attribute_pointer = true;
        self
    }
}

impl EvalContext for RenderContext<'_> {
    fn modifier_context(&self) -> ModifierContext {
        ModifierContext::Render
    }

    fn property_layout(&self) -> &PropertyLayout {
        self.property_layout
    }

    fn particle_layout(&self) -> &ParticleLayout {
        self.particle_layout
    }

    fn eval(&mut self, module: &Module, handle: ExprHandle) -> Result<String, ExprError> {
        // On cache hit, don't re-evaluate the expression to prevent any duplicate
        // side-effect.
        if let Some(s) = self.expr_cache.get(&handle) {
            Ok(s.clone())
        } else {
            module.try_get(handle)?.eval(module, self).inspect(|s| {
                self.expr_cache.insert(handle, s.clone());
            })
        }
    }

    fn make_local_var(&mut self) -> String {
        let index = self.var_counter;
        self.var_counter += 1;
        format!("var{}", index)
    }

    fn push_stmt(&mut self, stmt: &str) {
        // FIXME - vertex vs. fragment code, can't differentiate here currently
        self.vertex_code += stmt;
        self.vertex_code += "\n";
    }

    fn make_fn(
        &mut self,
        func_name: &str,
        args: &str,
        module: &mut Module,
        f: &mut dyn FnMut(&mut Module, &mut dyn EvalContext) -> Result<String, ExprError>,
    ) -> Result<(), ExprError> {
        // Generate a temporary context for the function content itself
        // FIXME - Dynamic with_attribute_pointer()!
        let texture_layout = module.texture_layout();
        let mut ctx =
            RenderContext::new(self.property_layout, self.particle_layout, &texture_layout)
                .with_attribute_pointer();

        // Evaluate the function content
        let body = f(module, &mut ctx)?;

        // Append any extra
        self.render_extra += &ctx.render_extra;

        // Append the function itself
        self.render_extra += &format!(
            r##"fn {0}({1}) {{
            {2};
        }}
        "##,
            func_name, args, body
        );

        Ok(())
    }

    fn is_attribute_pointer(&self) -> bool {
        self.is_attribute_pointer
    }
}

/// Trait to customize the rendering of alive particles each frame.
#[cfg_attr(feature = "serde", typetag::serde)]
pub trait RenderModifier: Modifier {
    /// Apply the rendering code.
    fn apply_render(
        &self,
        module: &mut Module,
        context: &mut RenderContext,
    ) -> Result<(), ExprError>;

    /// Clone into boxed self.
    fn boxed_render_clone(&self) -> Box<dyn RenderModifier>;

    /// Upcast to [`Modifier`] trait.
    fn as_modifier(&self) -> &dyn Modifier;
}

impl Clone for Box<dyn RenderModifier> {
    fn clone(&self) -> Self {
        self.boxed_render_clone()
    }
}

/// Macro to implement the [`Modifier`] trait for a render modifier.
macro_rules! impl_mod_render {
    ($t:ty, $attrs:expr) => {
        #[cfg_attr(feature = "serde", typetag::serde)]
        impl $crate::Modifier for $t {
            fn context(&self) -> $crate::ModifierContext {
                $crate::ModifierContext::Render
            }

            fn as_render(&self) -> Option<&dyn $crate::RenderModifier> {
                Some(self)
            }

            fn as_render_mut(&mut self) -> Option<&mut dyn $crate::RenderModifier> {
                Some(self)
            }

            fn attributes(&self) -> &[$crate::Attribute] {
                $attrs
            }

            fn boxed_clone(&self) -> $crate::BoxedModifier {
                Box::new(self.clone())
            }

            fn apply(
                &self,
                _module: &mut Module,
                context: &mut ShaderWriter,
            ) -> Result<(), ExprError> {
                Err(ExprError::InvalidModifierContext(
                    context.modifier_context(),
                    ModifierContext::Render,
                ))
            }
        }
    };
}

pub(crate) use impl_mod_render;

/// Condition to emit a GPU spawn event.
///
/// Determines when a GPU spawn event is emitted by a parent effect. See
/// the [`EffectParent`] component for details about the parent-child effect
/// relationship and its use.
///
/// [`EffectParent`]: crate::EffectParent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum EventEmitCondition {
    /// Always emit events each time the particle is updated, each simulation
    /// frame.
    Always,
    /// Only emit events if the particle died during this frame update.
    OnDie,
}

/// Emit GPU spawn events to spawn new particle(s) in a child effect.
///
/// This update modifier is used to spawn new particles into a child effect
/// instance based on a condition applied to particles of the current effect
/// instance. The most common use case is to spawn one or more child particles
/// into a child effect when a particle in this effect dies; this is achieved
/// with [`EventEmitCondition::OnDie`].
///
/// An effect instance with this modifier will emit GPU spawn events. Those
/// events are read by all child effects (those effects with an [`EffectParent`]
/// component pointing at the current effect instance). GPU spawn events are
/// stored internally in a GPU buffer; they're **unrelated** to Bevy ECS events.
///
/// [`EffectParent`]: crate::EffectParent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct EmitSpawnEventModifier {
    /// Emit condition for the GPU spawn events.
    pub condition: EventEmitCondition,
    /// The number of particles to spawn if the emit condition is met.
    ///
    /// Expression type: `Uint`
    pub count: ExprHandle,
    /// Index of the event channel / child the events are emitted into.
    ///
    /// GPU spawn events emitted by this parent event are associated with a
    /// single event channel. When the N-th child effect of a parent effect
    /// consumes those event, it implicitly reads events from channel #N. In
    /// general if a parent has a single child, use `0` here.
    pub child_index: u32,
}

impl EmitSpawnEventModifier {
    fn eval(
        &self,
        module: &mut Module,
        context: &mut dyn EvalContext,
    ) -> Result<String, ExprError> {
        // FIXME - mixing (ex-)channel and event buffer index; this should be automated
        let channel_index = self.child_index;
        // TODO - validate GPU spawn events are in use in the eval context...

        let count_val = context.eval(module, self.count)?;
        let count_var = context.make_local_var();
        context.push_stmt(&format!("let {} = {};", count_var, count_val));

        let cond = match self.condition {
            EventEmitCondition::Always => format!(
                "if (is_alive) {{ append_spawn_events_{channel_index}((*effect_metadata).base_child_index, particle_index, {}); }}",
                count_var
            ),
            EventEmitCondition::OnDie => format!(
                "if (was_alive && !is_alive) {{ append_spawn_events_{channel_index}((*effect_metadata).base_child_index, particle_index, {}); }}",
                count_var
            ),
        };
        Ok(cond)
    }
}

#[cfg_attr(feature = "serde", typetag::serde)]
impl Modifier for EmitSpawnEventModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Update
    }

    fn attributes(&self) -> &[Attribute] {
        &[]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn apply(&self, module: &mut Module, context: &mut ShaderWriter) -> Result<(), ExprError> {
        let code = self.eval(module, context)?;
        context.main_code += &code;
        context.set_emits_gpu_spawn_events(true)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use bevy::prelude::*;
    use naga::front::wgsl::Frontend;

    use super::*;
    use crate::{BuiltInOperator, ExprWriter, ScalarType};

    fn make_test_modifier() -> SetPositionSphereModifier {
        // We use a dummy module here because we don't care about the values and won't
        // evaluate the modifier.
        let mut m = Module::default();
        SetPositionSphereModifier {
            center: m.lit(Vec3::ZERO),
            radius: m.lit(1.),
            dimension: ShapeDimension::Surface,
        }
    }

    #[test]
    fn modifier_context_display() {
        assert_eq!("None", format!("{}", ModifierContext::empty()));
        assert_eq!("Init", format!("{}", ModifierContext::Init));
        assert_eq!("Update", format!("{}", ModifierContext::Update));
        assert_eq!("Render", format!("{}", ModifierContext::Render));
        assert_eq!(
            "Init | Update",
            format!("{}", ModifierContext::Init | ModifierContext::Update)
        );
        assert_eq!(
            "Update | Render",
            format!("{}", ModifierContext::Update | ModifierContext::Render)
        );
        assert_eq!(
            "Init | Render",
            format!("{}", ModifierContext::Init | ModifierContext::Render)
        );
        assert_eq!(
            "Init | Update | Render",
            format!("{}", ModifierContext::all())
        );
    }

    #[test]
    fn reflect() {
        let m = make_test_modifier();

        // Reflect
        let reflect: &dyn Reflect = m.as_reflect();
        assert!(reflect.is::<SetPositionSphereModifier>());
        let m_reflect = reflect.downcast_ref::<SetPositionSphereModifier>().unwrap();
        assert_eq!(*m_reflect, m);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde() {
        let m = make_test_modifier();
        let bm: BoxedModifier = Box::new(m);

        // Ser
        let s = ron::to_string(&bm).unwrap();
        println!("modifier: {:?}", s);

        // De
        let m_serde: BoxedModifier = ron::from_str(&s).unwrap();

        let rm: &dyn Reflect = m.as_reflect();
        let rm_serde: &dyn Reflect = m_serde.as_reflect();
        assert_eq!(
            rm.get_represented_type_info().unwrap().type_id(),
            rm_serde.get_represented_type_info().unwrap().type_id()
        );

        assert!(rm_serde.is::<SetPositionSphereModifier>());
        let rm_reflect = rm_serde
            .downcast_ref::<SetPositionSphereModifier>()
            .unwrap();
        assert_eq!(*rm_reflect, m);
    }

    #[test]
    fn validate_init() {
        let mut module = Module::default();
        let center = module.lit(Vec3::ZERO);
        let axis = module.lit(Vec3::Y);
        let radius = module.lit(1.);
        let modifiers: &[&dyn Modifier] = &[
            &SetPositionCircleModifier {
                center,
                axis,
                radius,
                dimension: ShapeDimension::Volume,
            },
            &SetPositionSphereModifier {
                center,
                radius,
                dimension: ShapeDimension::Volume,
            },
            &SetPositionCone3dModifier {
                base_radius: radius,
                top_radius: radius,
                height: radius,
                dimension: ShapeDimension::Volume,
            },
            &SetVelocityCircleModifier {
                center,
                axis,
                speed: radius,
            },
            &SetVelocitySphereModifier {
                center,
                speed: radius,
            },
            &SetVelocityTangentModifier {
                origin: center,
                axis,
                speed: radius,
            },
        ];
        for &modifier in modifiers.iter() {
            assert!(modifier.context().contains(ModifierContext::Init));
            let property_layout = PropertyLayout::default();
            let particle_layout = ParticleLayout::default();
            let mut context =
                ShaderWriter::new(ModifierContext::Init, &property_layout, &particle_layout);
            assert!(modifier.apply(&mut module, &mut context).is_ok());
            let main_code = context.main_code;
            let extra_code = context.extra_code;

            let mut particle_layout = ParticleLayout::new();
            for &attr in modifier.attributes() {
                particle_layout = particle_layout.append(attr);
            }
            let particle_layout = particle_layout.build();
            let attributes_code = particle_layout.generate_code();

            let code = format!(
                r##"fn frand() -> f32 {{
    return 0.0;
}}

const tau: f32 = 6.283185307179586476925286766559;

struct Particle {{
    {attributes_code}
}};

{extra_code}

@compute @workgroup_size(64)
fn main() {{
    var particle = Particle();
    var transform: mat4x4<f32> = mat4x4<f32>();
{main_code}
}}"##
            );
            // println!("code: {:?}", code);

            let mut frontend = Frontend::new();
            let res = frontend.parse(&code);
            if let Err(err) = &res {
                println!(
                    "Modifier: {:?}",
                    modifier.get_represented_type_info().unwrap().type_path()
                );
                println!("Code: {:?}", code);
                println!("Err: {:?}", err);
            }
            assert!(res.is_ok());
        }
    }

    #[test]
    fn validate_update() {
        let writer = ExprWriter::new();
        let origin = writer.lit(Vec3::ZERO).expr();
        let center = origin;
        let axis = origin;
        let y_axis = writer.lit(Vec3::Y).expr();
        let one = writer.lit(1.).expr();
        let radius = one;
        let modifiers: &[&dyn Modifier] = &[
            &AccelModifier::new(origin),
            &RadialAccelModifier::new(origin, one),
            &TangentAccelModifier::new(origin, y_axis, one),
            &ConformToSphereModifier::new(origin, one, one, one, one),
            &LinearDragModifier::new(writer.lit(3.5).expr()),
            &KillAabbModifier::new(writer.lit(Vec3::ZERO).expr(), writer.lit(Vec3::ONE).expr()),
            &SetPositionCircleModifier {
                center,
                axis,
                radius,
                dimension: ShapeDimension::Volume,
            },
            &SetPositionSphereModifier {
                center,
                radius,
                dimension: ShapeDimension::Volume,
            },
            &SetPositionCone3dModifier {
                base_radius: radius,
                top_radius: radius,
                height: radius,
                dimension: ShapeDimension::Volume,
            },
            &SetVelocityCircleModifier {
                center,
                axis,
                speed: radius,
            },
            &SetVelocitySphereModifier {
                center,
                speed: radius,
            },
            &SetVelocityTangentModifier {
                origin: center,
                axis,
                speed: radius,
            },
        ];
        let mut module = writer.finish();
        for &modifier in modifiers.iter() {
            assert!(modifier.context().contains(ModifierContext::Update));
            let property_layout = PropertyLayout::default();
            let particle_layout = ParticleLayout::default();
            let mut context =
                ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
            assert!(modifier.apply(&mut module, &mut context).is_ok());
            let update_code = context.main_code;
            let update_extra = context.extra_code;

            let mut particle_layout = ParticleLayout::new();
            for &attr in modifier.attributes() {
                particle_layout = particle_layout.append(attr);
            }
            let particle_layout = particle_layout.build();
            let attributes_code = particle_layout.generate_code();

            let code = format!(
                r##"fn frand() -> f32 {{
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
    virtual_delta_time: f32,
    virtual_time: f32,
    real_delta_time: f32,
    real_time: f32,
}};

struct Spawner {{
    transform: mat3x4<f32>, // transposed (row-major)
    spawn: atomic<i32>,
    seed: u32,
    count_unused: u32,
    effect_index: u32,
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
                println!(
                    "Modifier: {:?}",
                    modifier.get_represented_type_info().unwrap().type_path()
                );
                println!("Code: {:?}", code);
                println!("Err: {:?}", err);
            }
            assert!(res.is_ok());
        }
    }

    #[test]
    fn validate_render() {
        let mut base_module = Module::default();
        let slot_zero = base_module.lit(0u32);
        let modifiers: &[&dyn RenderModifier] = &[
            &ParticleTextureModifier::new(slot_zero),
            &ColorOverLifetimeModifier::default(),
            &SizeOverLifetimeModifier::default(),
            &OrientModifier::new(OrientMode::ParallelCameraDepthPlane),
            &OrientModifier::new(OrientMode::FaceCameraPosition),
            &OrientModifier::new(OrientMode::AlongVelocity),
        ];
        for &modifier in modifiers.iter() {
            let mut module = base_module.clone();
            let property_layout = PropertyLayout::default();
            let particle_layout = ParticleLayout::default();
            let texture_layout = module.texture_layout();
            let mut context =
                RenderContext::new(&property_layout, &particle_layout, &texture_layout);
            modifier
                .apply_render(&mut module, &mut context)
                .expect("Failed to apply modifier to render context.");
            let vertex_code = context.vertex_code;
            let fragment_code = context.fragment_code;
            let render_extra = context.render_extra;

            let mut particle_layout = ParticleLayout::new();
            for &attr in modifier.attributes() {
                particle_layout = particle_layout.append(attr);
            }
            let particle_layout = particle_layout.build();
            let attributes_code = particle_layout.generate_code();

            let code = format!(
                r##"
struct ColorGrading {{
    balance: mat3x3<f32>,
    saturation: vec3<f32>,
    contrast: vec3<f32>,
    gamma: vec3<f32>,
    gain: vec3<f32>,
    lift: vec3<f32>,
    midtone_range: vec2<f32>,
    exposure: f32,
    hue: f32,
    post_saturation: f32,
}}

struct View {{
    clip_from_world: mat4x4<f32>,
    unjittered_clip_from_world: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    world_from_view: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    clip_from_view: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
    world_position: vec3<f32>,
    exposure: f32,
    // viewport(x_origin, y_origin, width, height)
    viewport: vec4<f32>,
    frustum: array<vec4<f32>, 6>,
    color_grading: ColorGrading,
    mip_bias: f32,
}}

fn frand() -> f32 {{ return 0.0; }}
fn get_camera_position_effect_space() -> vec3<f32> {{ return vec3<f32>(); }}
fn get_camera_rotation_effect_space() -> mat3x3<f32> {{ return mat3x3<f32>(); }}

const tau: f32 = 6.283185307179586476925286766559;

struct Particle {{
    {attributes_code}
}};

struct VertexOutput {{
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}};

@group(0) @binding(0) var<uniform> view: View;

{render_extra}

@compute @workgroup_size(64)
fn main() {{
    var particle = Particle();
    var position = vec3<f32>(0.0, 0.0, 0.0);
    var velocity = vec3<f32>(0.0, 0.0, 0.0);
    var size = vec3<f32>(1.0, 1.0, 1.0);
    var axis_x = vec3<f32>(1.0, 0.0, 0.0);
    var axis_y = vec3<f32>(0.0, 1.0, 0.0);
    var axis_z = vec3<f32>(0.0, 0.0, 1.0);
    var color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
{vertex_code}
    var out: VertexOutput;
    return out;
}}


@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {{
    var color = vec4<f32>(0.0);
    var uv = vec2<f32>(0.0);
{fragment_code}
    return vec4<f32>(1.0);
}}"##
            );

            let mut frontend = Frontend::new();
            let res = frontend.parse(&code);
            if let Err(err) = &res {
                println!(
                    "Modifier: {:?}",
                    modifier.get_represented_type_info().unwrap().type_path()
                );
                println!("Code: {:?}", code);
                println!("Err: {:?}", err);
            }
            assert!(res.is_ok());
        }
    }

    #[test]
    fn eval_cached() {
        let mut module = Module::default();
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let x = module.builtin(BuiltInOperator::Rand(ScalarType::Float.into()));
        let texture_layout = module.texture_layout();
        let init: &mut dyn EvalContext =
            &mut ShaderWriter::new(ModifierContext::Init, &property_layout, &particle_layout);
        let update: &mut dyn EvalContext =
            &mut ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
        let render: &mut dyn EvalContext =
            &mut RenderContext::new(&property_layout, &particle_layout, &texture_layout);
        for ctx in [init, update, render] {
            // First evaluation is cached inside a local variable 'var0'
            let s = ctx.eval(&module, x).unwrap();
            assert_eq!(s, "var0");
            // Second evaluation return the same variable
            let s2 = ctx.eval(&module, x).unwrap();
            assert_eq!(s2, s);
        }
    }
}
