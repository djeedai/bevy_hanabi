//! Building blocks to create a visual effect.
//!
//! A **modifier** is a building block used to create effects. Particles effects
//! are composed of multiple modifiers, which put together and configured
//! produce the desired visual effect. Each modifier changes a specific part of
//! the behavior of an effect. Modifiers are grouped in three categories:
//!
//! - **Init modifiers** influence the initializing of particles when they
//!   spawn. They typically configure the initial position and/or velocity of
//!   particles. Init modifiers implement the [`InitModifier`] trait.
//! - **Update modifiers** influence the particle update loop each frame. For
//!   example, an update modifier can apply a gravity force to all particles.
//!   Update modifiers implement the [`UpdateModifier`] trait.
//! - **Render modifiers** influence the rendering of each particle. They can
//!   change the particle's color, or orient it to face the camera. Render
//!   modifiers implement the [`RenderModifier`] trait.
//!
//! A single modifier can be part of multiple categories. For example, the
//! [`SetAttributeModifier`] can be used either to initialize a particle's
//! attribute on spawning, or to assign a value to that attribute each frame
//! during simulation (update).
//!
//! [`InitModifier`]: crate::modifier::InitModifier
//! [`UpdateModifier`]: crate::modifier::UpdateModifier
//! [`RenderModifier`]: crate::modifier::RenderModifier

use bevy::{
    asset::Handle,
    math::{Vec2, Vec3, Vec4},
    reflect::Reflect,
    render::texture::Image,
    utils::{FloatOrd, HashMap},
};
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

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
    Attribute, EvalContext, Expr, ExprError, ExprHandle, Gradient, Module, PropertyLayout,
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
    let mut hasher = DefaultHasher::default();
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

/// Trait describing a modifier customizing an effect pipeline.
#[typetag::serde]
pub trait Modifier: Reflect + Send + Sync + 'static {
    /// Get the context this modifier applies to.
    fn context(&self) -> ModifierContext;

    /// Try to cast this modifier to an [`InitModifier`].
    fn as_init(&self) -> Option<&dyn InitModifier> {
        None
    }

    /// Try to cast this modifier to an [`InitModifier`].
    fn as_init_mut(&mut self) -> Option<&mut dyn InitModifier> {
        None
    }

    /// Try to cast this modifier to an [`UpdateModifier`].
    fn as_update(&self) -> Option<&dyn UpdateModifier> {
        None
    }

    /// Try to cast this modifier to an [`UpdateModifier`].
    fn as_update_mut(&mut self) -> Option<&mut dyn UpdateModifier> {
        None
    }

    /// Try to cast this modifier to a [`RenderModifier`].
    fn as_render(&self) -> Option<&dyn RenderModifier> {
        None
    }

    /// Try to cast this modifier to a [`RenderModifier`].
    fn as_render_mut(&mut self) -> Option<&mut dyn RenderModifier> {
        None
    }

    /// Get the list of dependent attributes required for this modifier to be
    /// used.
    fn attributes(&self) -> &[Attribute];

    /// Clone self.
    fn boxed_clone(&self) -> BoxedModifier;
}

/// Boxed version of [`Modifier`].
pub type BoxedModifier = Box<dyn Modifier>;

impl Clone for BoxedModifier {
    fn clone(&self) -> Self {
        self.boxed_clone()
    }
}

/// Particle initializing shader code generation context.
#[derive(Debug, PartialEq)]
pub struct InitContext<'a> {
    /// Module being populated with new expressions from modifiers.
    pub module: &'a mut Module,
    /// Main particle initializing code, which needs to assign the fields of the
    /// `particle` struct instance.
    pub init_code: String,
    /// Extra functions emitted at top level, which `init_code` can call.
    pub init_extra: String,
    /// Layout of properties for the current effect.
    pub property_layout: &'a PropertyLayout,
}

impl<'a> InitContext<'a> {
    /// Create a new init context.
    pub fn new(module: &'a mut Module, property_layout: &'a PropertyLayout) -> Self {
        Self {
            module,
            init_code: String::new(),
            init_extra: String::new(),
            property_layout,
        }
    }
}

impl<'a> EvalContext for InitContext<'a> {
    fn module(&self) -> &Module {
        self.module
    }

    fn property_layout(&self) -> &PropertyLayout {
        self.property_layout
    }

    fn expr(&self, expr: ExprHandle) -> Result<&Expr, ExprError> {
        self.module
            .get(expr)
            .ok_or(ExprError::InvalidExprHandleError(format!(
                "Cannot find expression with handle {:?} in the current module. Check that the Module used to build the expression was the same used in the EvalContext or the original EffectAsset.",
                expr
            )))
    }

    fn eval(&self, handle: ExprHandle) -> Result<String, ExprError> {
        self.expr(handle)?.eval(self)
    }
}

/// Trait to customize the initializing of newly spawned particles.
#[typetag::serde]
pub trait InitModifier: Modifier {
    /// Append the initializing code.
    fn apply_init(&self, context: &mut InitContext) -> Result<(), ExprError>;
}

/// A single attraction or repulsion source of a [`ForceFieldModifier`].
///
/// The source applies a radial force field to all particles around its
/// position, with a decreasing intensity the further away from the source the
/// particle is. This force is added to the one(s) of all the other active
/// sources of a [`ForceFieldModifier`].
#[derive(Debug, Clone, Copy, Reflect, Serialize, Deserialize)]
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
    fn apply_update(&self, context: &mut UpdateContext) -> Result<(), ExprError>;
}

/// Particle rendering shader code generation context.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RenderContext {
    /// Main particle rendering code for the vertex shader.
    pub vertex_code: String,
    /// Main particle rendering code for the fragment shader.
    pub fragment_code: String,
    /// Extra functions emitted at top level, which `vertex_code` and
    /// `fragment_code` can call.
    pub render_extra: String,
    /// Texture modulating the particle color.
    pub particle_texture: Option<Handle<Image>>,
    /// Color gradients.
    pub gradients: HashMap<u64, Gradient<Vec4>>,
    /// Size gradients.
    pub size_gradients: HashMap<u64, Gradient<Vec2>>,
    /// Are particles using a fixed screen-space size (in logical pixels)? If
    /// `true` then the particle size is not affected by the camera projection,
    /// and in particular by the distance to the camera.
    pub screen_space_size: bool,
}

impl RenderContext {
    /// Set the main texture used to color particles.
    fn set_particle_texture(&mut self, handle: Handle<Image>) {
        self.particle_texture = Some(handle);
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
    fn add_size_gradient(&mut self, gradient: Gradient<Vec2>) -> String {
        let func_id = calc_func_id(&gradient);
        self.size_gradients.insert(func_id, gradient);
        let func_name = format!("size_gradient_{0:016X}", func_id);
        func_name
    }
}

/// Trait to customize the rendering of alive particles each frame.
#[typetag::serde]
pub trait RenderModifier: Modifier {
    /// Apply the rendering code.
    fn apply_render(&self, context: &mut RenderContext);
}

/// Macro to implement the [`Modifier`] trait for an init modifier.
// macro_rules! impl_mod_init {
//     ($t:ty, $attrs:expr) => {
//         #[typetag::serde]
//         impl $crate::Modifier for $t {
//             fn context(&self) -> $crate::ModifierContext {
//                 $crate::ModifierContext::Init
//             }

//             fn as_init(&self) -> Option<&dyn $crate::InitModifier> {
//                 Some(self)
//             }

//             fn as_init_mut(&mut self) -> Option<&mut dyn $crate::InitModifier> {
//                 Some(self)
//             }

//             fn attributes(&self) -> &[$crate::Attribute] {
//                 $attrs
//             }

//             fn boxed_clone(&self) -> $crate::BoxedModifier {
//                 Box::new(*self)
//             }
//         }
//     };
// }

// pub(crate) use impl_mod_init;

/// Macro to implement the [`Modifier`] trait for an update modifier.
macro_rules! impl_mod_update {
    ($t:ty, $attrs:expr) => {
        #[typetag::serde]
        impl $crate::Modifier for $t {
            fn context(&self) -> $crate::ModifierContext {
                $crate::ModifierContext::Update
            }

            fn as_update(&self) -> Option<&dyn $crate::UpdateModifier> {
                Some(self)
            }

            fn as_update_mut(&mut self) -> Option<&mut dyn $crate::UpdateModifier> {
                Some(self)
            }

            fn attributes(&self) -> &[$crate::Attribute] {
                $attrs
            }

            fn boxed_clone(&self) -> $crate::BoxedModifier {
                Box::new(self.clone())
            }
        }
    };
}

pub(crate) use impl_mod_update;

/// Macro to implement the [`Modifier`] trait for a modifier which can be used
/// both in init and update contexts.
macro_rules! impl_mod_init_update {
    ($t:ty, $attrs:expr) => {
        #[typetag::serde]
        impl $crate::Modifier for $t {
            fn context(&self) -> $crate::ModifierContext {
                $crate::ModifierContext::Init | $crate::ModifierContext::Update
            }

            fn as_init(&self) -> Option<&dyn $crate::InitModifier> {
                Some(self)
            }

            fn as_init_mut(&mut self) -> Option<&mut dyn $crate::InitModifier> {
                Some(self)
            }

            fn as_update(&self) -> Option<&dyn $crate::UpdateModifier> {
                Some(self)
            }

            fn as_update_mut(&mut self) -> Option<&mut dyn $crate::UpdateModifier> {
                Some(self)
            }

            fn attributes(&self) -> &[$crate::Attribute] {
                $attrs
            }

            fn boxed_clone(&self) -> $crate::BoxedModifier {
                Box::new(self.clone())
            }
        }
    };
}

pub(crate) use impl_mod_init_update;

/// Macro to implement the [`Modifier`] trait for a render modifier.
macro_rules! impl_mod_render {
    ($t:ty, $attrs:expr) => {
        #[typetag::serde]
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
        }
    };
}

pub(crate) use impl_mod_render;

#[cfg(test)]
mod tests {
    use bevy::prelude::*;
    use naga::front::wgsl::Frontend;

    use crate::{ExprWriter, ParticleLayout};

    use super::*;

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
    fn reflect() {
        let m = make_test_modifier();

        // Reflect
        let reflect: &dyn Reflect = m.as_reflect();
        assert!(reflect.is::<SetPositionSphereModifier>());
        let m_reflect = reflect.downcast_ref::<SetPositionSphereModifier>().unwrap();
        assert_eq!(*m_reflect, m);
    }

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
        let modifiers: &[&dyn InitModifier] = &[
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
            let property_layout = PropertyLayout::default();
            let mut context = InitContext::new(&mut module, &property_layout);
            assert!(modifier.apply_init(&mut context).is_ok());
            let init_code = context.init_code;
            let init_extra = context.init_extra;

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

{init_extra}

@compute @workgroup_size(64)
fn main() {{
    var particle = Particle();
    var transform: mat4x4<f32> = mat4x4<f32>();
{init_code}
}}"##
            );
            // println!("code: {:?}", code);

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

    #[test]
    fn validate_update() {
        let writer = ExprWriter::new();
        let origin = writer.lit(Vec3::ZERO).expr();
        let center = origin;
        let axis = origin;
        let y_axis = writer.lit(Vec3::Y).expr();
        let one = writer.lit(1.).expr();
        let radius = one;
        let modifiers: &[&dyn UpdateModifier] = &[
            &AccelModifier::new(origin),
            &RadialAccelModifier::new(origin, one),
            &TangentAccelModifier::new(origin, y_axis, one),
            &ForceFieldModifier::default(),
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
            let property_layout = PropertyLayout::default();
            let mut context = UpdateContext::new(&mut module, &property_layout);
            assert!(modifier.apply_update(&mut context).is_ok());
            let update_code = context.update_code;
            let update_extra = context.update_extra;

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

    #[test]
    fn validate_render() {
        let modifiers: &[&dyn RenderModifier] = &[
            &ParticleTextureModifier::default(),
            &ColorOverLifetimeModifier::default(),
            &SizeOverLifetimeModifier::default(),
            &BillboardModifier,
            &OrientAlongVelocityModifier,
        ];
        for &modifier in modifiers.iter() {
            let mut context = RenderContext::default();
            modifier.apply_render(&mut context);
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
struct View {{
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    world_position: vec3<f32>,
    width: f32,
    height: f32,
}};

fn frand() -> f32 {{
    return 0.0;
}}

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
    var size = vec2<f32>(1.0, 1.0);
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
{fragment_code}
    return vec4<f32>(1.0);
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
