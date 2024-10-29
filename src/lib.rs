#![deny(
    warnings,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    missing_docs,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    clippy::suboptimal_flops,
    clippy::imprecise_flops,
    clippy::branches_sharing_code,
    clippy::suspicious_operation_groupings,
    clippy::useless_let_if_seq
)]
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

//! 🎆 Hanabi -- a GPU particle system plugin for the Bevy game engine.
//!
//! The 🎆 Hanabi particle system is a GPU-based particle system integrated with
//! the built-in Bevy renderer. It allows creating complex visual effects with
//! millions of particles simulated in real time, by leveraging the power of
//! compute shaders and offloading most of the work of particle simulating to
//! the GPU.
//!
//! _Note: This library makes heavy use of compute shaders to offload work to
//! the GPU in a performant way. Support for compute shaders on the `wasm`
//! target (WebAssembly) via WebGPU is only available in Bevy in general since
//! the Bevy v0.11, and is not yet available in this library.
//! See [#41](https://github.com/djeedai/bevy_hanabi/issues/41) for details on
//! progress._
//!
//! # 2D vs. 3D
//!
//! 🎆 Hanabi integrates both with the 2D and the 3D core pipelines of Bevy. The
//! 2D pipeline integration is controlled by the `2d` cargo feature, while the
//! 3D pipeline integration is controlled by the `3d` cargo feature. Both
//! features are enabled by default for convenience. As an optimization, users
//! can disable default features and re-enable only one of the two modes.
//!
//! ```toml
//! # Example: enable only 3D integration
//! bevy_hanabi = { version = "0.12", default-features = false, features = ["3d"] }
//! ```
//!
//! # Example
//!
//! Add the 🎆 Hanabi plugin to your app:
//!
//! ```no_run
//! # use bevy::prelude::*;
//! use bevy_hanabi::prelude::*;
//!
//! App::default()
//!     .add_plugins(DefaultPlugins)
//!     .add_plugins(HanabiPlugin)
//!     .run();
//! ```
//!
//! Create an [`EffectAsset`] describing a visual effect:
//!
//! ```
//! # use bevy::prelude::*;
//! # use bevy_hanabi::prelude::*;
//! fn setup(mut effects: ResMut<Assets<EffectAsset>>) {
//!     // Define a color gradient from red to transparent black
//!     let mut gradient = Gradient::new();
//!     gradient.add_key(0.0, Vec4::new(1., 0., 0., 1.));
//!     gradient.add_key(1.0, Vec4::splat(0.));
//!
//!     // Create a new expression module
//!     let mut module = Module::default();
//!
//!     // On spawn, randomly initialize the position of the particle
//!     // to be over the surface of a sphere of radius 2 units.
//!     let init_pos = SetPositionSphereModifier {
//!         center: module.lit(Vec3::ZERO),
//!         radius: module.lit(2.),
//!         dimension: ShapeDimension::Surface,
//!     };
//!
//!     // Also initialize a radial initial velocity to 6 units/sec
//!     // away from the (same) sphere center.
//!     let init_vel = SetVelocitySphereModifier {
//!         center: module.lit(Vec3::ZERO),
//!         speed: module.lit(6.),
//!     };
//!
//!     // Initialize the total lifetime of the particle, that is
//!     // the time for which it's simulated and rendered. This modifier
//!     // is almost always required, otherwise the particles won't show.
//!     let lifetime = module.lit(10.); // literal value "10.0"
//!     let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);
//!
//!     // Every frame, add a gravity-like acceleration downward
//!     let accel = module.lit(Vec3::new(0., -3., 0.));
//!     let update_accel = AccelModifier::new(accel);
//!
//!     // Create the effect asset
//!     let effect = EffectAsset::new(
//!         // Maximum number of particles alive at a time
//!         32768,
//!         // Spawn at a rate of 5 particles per second
//!         Spawner::rate(5.0.into()),
//!         // Move the expression module into the asset
//!         module,
//!     )
//!     .with_name("MyEffect")
//!     .init(init_pos)
//!     .init(init_vel)
//!     .init(init_lifetime)
//!     .update(update_accel)
//!     // Render the particles with a color gradient over their
//!     // lifetime. This maps the gradient key 0 to the particle spawn
//!     // time, and the gradient key 1 to the particle death (10s).
//!     .render(ColorOverLifetimeModifier { gradient });
//!
//!     // Insert into the asset system
//!     let effect_asset = effects.add(effect);
//! }
//! ```
//!
//! Then add an instance of that effect to an entity by spawning a
//! [`ParticleEffect`] component referencing the asset. The simplest way is
//! to use the [`ParticleEffectBundle`] to ensure all required components are
//! spawned together.
//!
//! ```
//! # use bevy::prelude::*;
//! # use bevy_hanabi::prelude::*;
//! # fn spawn_effect(mut commands: Commands) {
//! #   let effect_asset = Handle::<EffectAsset>::default();
//! commands.spawn(ParticleEffectBundle {
//!     effect: ParticleEffect::new(effect_asset),
//!     transform: Transform::from_translation(Vec3::Y),
//!     ..Default::default()
//! });
//! # }
//! ```
//!
//! # Workflow
//!
//! Authoring and using a particle effect follows this workflow:
//!
//! 1. Create an [`EffectAsset`] representing the definition of the particle
//!    effect. This asset is a proper Bevy [`Asset`], expected to be authored in
//!    advance, serialized, and shipped with your application. Creating an
//!    [`EffectAsset`] at runtime while the application is running is also
//!    supported, though. In any case however, the asset doesn't do anything by
//!    itself.
//! 2. At runtime, when the application is running, create an actual particle
//!    effect instance by spawning a [`ParticleEffect`] component (or use the
//!    [`ParticleEffectBundle`] for simplicity). The component references the
//!    [`EffectAsset`] via its `handle` field. Multiple instances can reference
//!    the same asset at the same time, and some changes to the asset are
//!    reflected to its instances, although not all changes are supported. In
//!    general, avoid changing an [`EffectAsset`] while it's in use by one or
//!    more [`ParticleEffect`].
//! 3. Also spawn an [`EffectProperties`] component on the same entity. This is
//!    required even if the effect doesn't use properties. See [properties] for
//!    more details.
//! 4. If actually using properties, update them through the
//!    [`EffectProperties`] at any time while the effect is active. This allows
//!    some moderate CPU-side control over the simulation and rendering of the
//!    effect, without having to destroy the effect and re-create a new one.

use std::fmt::Write as _;

#[cfg(feature = "2d")]
use bevy::math::FloatOrd;
use bevy::{prelude::*, utils::HashSet};
use serde::{Deserialize, Serialize};
use spawn::Initializer;
use thiserror::Error;

mod asset;
pub mod attributes;
mod bundle;
mod gradient;
pub mod graph;
pub mod modifier;
mod plugin;
pub mod properties;
mod render;
mod spawn;
mod time;

#[cfg(test)]
mod test_utils;

pub use asset::{AlphaMode, EffectAsset, MotionIntegration, SimulationCondition};
pub use attributes::*;
pub use bundle::ParticleEffectBundle;
pub use gradient::{Gradient, GradientKey};
pub use graph::*;
pub use modifier::*;
pub use plugin::{EffectSystems, HanabiPlugin};
pub use properties::*;
pub use render::{LayoutFlags, ShaderCache};
pub use spawn::{
    tick_spawners, Cloner, CpuValue, EffectCloner, EffectInitializer, EffectInitializers,
    EffectSpawner, Random, Spawner,
};
pub use time::{EffectSimulation, EffectSimulationTime};

#[allow(missing_docs)]
pub mod prelude {
    #[doc(hidden)]
    pub use crate::*;
}

#[cfg(not(any(feature = "2d", feature = "3d")))]
compile_error!(
    "You need to enable at least one of the '2d' or '3d' features for anything to happen."
);

/// Get the smallest multiple of align greater than or equal to value, where
/// `align` must be a power of two.
///
/// # Panics
///
/// Panics if `align` is not a power of two.
// TODO - filler for usize.next_multiple_of()
// https://github.com/rust-lang/rust/issues/88581
pub(crate) fn next_multiple_of(value: usize, align: usize) -> usize {
    assert!(align & (align - 1) == 0); // power of 2
    let count = (value + align - 1) / align;
    count * align
}

/// Extension trait to convert an object to WGSL code.
///
/// This is mainly used for floating-point constants. This is required because
/// WGSL doesn't support a floating point constant without a decimal separator
/// (_e.g._ `0.` instead of `0`), which would be what a regular string
/// formatting function like [`format!()`] would produce, but which is
/// interpreted as an integral type by WGSL instead.
///
/// # Example
///
/// ```
/// # use bevy_hanabi::ToWgslString;
/// let x = 2.0_f32;
/// assert_eq!("let x = 2.;", format!("let x = {};", x.to_wgsl_string()));
/// ```
///
/// # Remark
///
/// This trait is soft-deprecated. It serves the same purpose as
/// [`EvalContext::eval()`], however it lacks any context for the evaluation of
/// an expression producing WGSL code, so its use is limited. It's still useful
/// for constant (literal) expressions, which do not require any context to
/// evaluate.
///
/// [`format!()`]: std::format
/// [`EvalContext::eval()`]: crate::graph::expr::EvalContext::eval
pub trait ToWgslString {
    /// Convert an object to a string representing its WGSL code.
    fn to_wgsl_string(&self) -> String;
}

impl ToWgslString for f32 {
    fn to_wgsl_string(&self) -> String {
        let s = format!("{self:.6}");
        s.trim_end_matches('0').to_string()
    }
}

impl ToWgslString for f64 {
    fn to_wgsl_string(&self) -> String {
        let s = format!("{self:.15}");
        s.trim_end_matches('0').to_string()
    }
}

impl ToWgslString for Vec2 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec2<f32>({0},{1})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string()
        )
    }
}

impl ToWgslString for Vec3 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec3<f32>({0},{1},{2})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string(),
            self.z.to_wgsl_string()
        )
    }
}

impl ToWgslString for Vec4 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec4<f32>({0},{1},{2},{3})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string(),
            self.z.to_wgsl_string(),
            self.w.to_wgsl_string()
        )
    }
}

impl ToWgslString for bool {
    fn to_wgsl_string(&self) -> String {
        if *self {
            "true".to_string()
        } else {
            "false".to_string()
        }
    }
}

impl ToWgslString for BVec2 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec2<bool>({0},{1})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string()
        )
    }
}

impl ToWgslString for BVec3 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec3<bool>({0},{1},{2})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string(),
            self.z.to_wgsl_string()
        )
    }
}

impl ToWgslString for BVec4 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec4<bool>({0},{1},{2},{3})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string(),
            self.z.to_wgsl_string(),
            self.w.to_wgsl_string()
        )
    }
}

impl ToWgslString for i32 {
    fn to_wgsl_string(&self) -> String {
        format!("{}", self)
    }
}

impl ToWgslString for IVec2 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec2<i32>({0},{1})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string()
        )
    }
}

impl ToWgslString for IVec3 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec3<i32>({0},{1},{2})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string(),
            self.z.to_wgsl_string()
        )
    }
}

impl ToWgslString for IVec4 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec4<i32>({0},{1},{2},{3})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string(),
            self.z.to_wgsl_string(),
            self.w.to_wgsl_string()
        )
    }
}

impl ToWgslString for u32 {
    fn to_wgsl_string(&self) -> String {
        format!("{}u", self)
    }
}

impl ToWgslString for UVec2 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec2<u32>({0},{1})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string()
        )
    }
}

impl ToWgslString for UVec3 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec3<u32>({0},{1},{2})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string(),
            self.z.to_wgsl_string()
        )
    }
}

impl ToWgslString for UVec4 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec4<u32>({0},{1},{2},{3})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string(),
            self.z.to_wgsl_string(),
            self.w.to_wgsl_string()
        )
    }
}

impl ToWgslString for CpuValue<f32> {
    fn to_wgsl_string(&self) -> String {
        match self {
            Self::Single(x) => x.to_wgsl_string(),
            Self::Uniform((a, b)) => format!(
                "(frand() * ({1} - {0}) + {0})",
                a.to_wgsl_string(),
                b.to_wgsl_string(),
            ),
        }
    }
}

impl ToWgslString for CpuValue<Vec2> {
    fn to_wgsl_string(&self) -> String {
        match self {
            Self::Single(v) => v.to_wgsl_string(),
            Self::Uniform((a, b)) => format!(
                "(frand2() * ({1} - {0}) + {0})",
                a.to_wgsl_string(),
                b.to_wgsl_string(),
            ),
        }
    }
}

impl ToWgslString for CpuValue<Vec3> {
    fn to_wgsl_string(&self) -> String {
        match self {
            Self::Single(v) => v.to_wgsl_string(),
            Self::Uniform((a, b)) => format!(
                "(frand3() * ({1} - {0}) + {0})",
                a.to_wgsl_string(),
                b.to_wgsl_string(),
            ),
        }
    }
}

impl ToWgslString for CpuValue<Vec4> {
    fn to_wgsl_string(&self) -> String {
        match self {
            Self::Single(v) => v.to_wgsl_string(),
            Self::Uniform((a, b)) => format!(
                "(frand4() * ({1} - {0}) + {0})",
                a.to_wgsl_string(),
                b.to_wgsl_string(),
            ),
        }
    }
}

/// Simulation space for the particles of an effect.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Reflect, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SimulationSpace {
    /// Particles are simulated in global space.
    ///
    /// The global space is the Bevy world space. Particles simulated in global
    /// space are "detached" from the emitter when they spawn, and not
    /// influenced anymore by the emitter's [`Transform`] after spawning. The
    /// particle's [`Attribute::POSITION`] is the world space position of the
    /// particle.
    ///
    /// This is the default.
    #[default]
    Global,

    /// Particles are simulated in local effect space.
    ///
    /// The local space is the space associated with the [`Transform`] of the
    /// [`ParticleEffect`] component being simulated. Particles simulated in
    /// local effect space are "attached" to the effect, and will be affected by
    /// its [`Transform`]. The particle's [`Attribute::POSITION`] is the
    /// position of the particle relative to the effect's [`Transform`].
    Local,
}

impl SimulationSpace {
    /// Evaluate the simulation space expression.
    ///
    /// - In the init and udpate contexts, this expression transforms the
    ///   particle's position from simulation space to storage space.
    /// - In the render context, this expression transforms the particle's
    ///   position from simulation space to view space.
    pub fn eval(&self, context: &dyn EvalContext) -> Result<String, ExprError> {
        match context.modifier_context() {
            ModifierContext::Init | ModifierContext::Update => match *self {
                SimulationSpace::Global => {
                    if !context.particle_layout().contains(Attribute::POSITION) {
                        return Err(ExprError::GraphEvalError(format!("Global-space simulation requires that the particles have a {} attribute.", Attribute::POSITION.name())));
                    }
                    Ok(format!(
                        "particle.{} += transform[3].xyz;", // TODO: get_view_position()
                        Attribute::POSITION.name()
                    ))
                }
                SimulationSpace::Local => Ok("".to_string()),
            },
            ModifierContext::Render => Ok(match *self {
                // TODO: cast vec3 -> vec4 auomatically
                SimulationSpace::Global => "vec4<f32>(local_position, 1.0)",
                // TODO: transform_world_to_view(...)
                SimulationSpace::Local => "transform * vec4<f32>(local_position, 1.0)",
            }
            .to_string()),
            _ => Err(ExprError::GraphEvalError(
                "Invalid modifier context value.".to_string(),
            )),
        }
    }
}

/// Value a user wants to assign to a property with
/// [`EffectProperties::set()`] before the instance had a chance
/// to inspect its underlying asset and check the asset's defined properties.
///
/// A property with this name might not exist, in which case the value will be
/// discarded silently when the instance is initialized from its asset.
#[derive(Debug, Clone, PartialEq, Reflect)]
pub struct PropertyValue {
    /// Name of the property the value should be assigned to.
    name: String,

    /// The property value to assign, instead of the default value of the
    /// property.
    value: Value,
}

impl From<PropertyInstance> for PropertyValue {
    fn from(prop: PropertyInstance) -> Self {
        Self {
            name: prop.def.name().to_string(),
            value: prop.value,
        }
    }
}

impl From<&PropertyInstance> for PropertyValue {
    fn from(prop: &PropertyInstance) -> Self {
        Self {
            name: prop.def.name().to_string(),
            value: prop.value,
        }
    }
}

/// Visual effect made of particles.
///
/// The particle effect component represent a single instance of a visual
/// effect. The visual effect itself is described by a handle to an
/// [`EffectAsset`]. This instance is associated to an [`Entity`], inheriting
/// its [`Transform`] as the origin frame for its particle spawning.
///
/// # Content
///
/// The values in this component, with the exception of the handle to the
/// underlying asset itself ([`ParticleEffect::handle`]), are optional
/// per-instance overrides taking precedence over the fallback value set in the
/// [`EffectAsset`].
///
/// Note that the effect graph itself including all its modifiers cannot be
/// overridden per instance. For minor variations use different assets. If you
/// need too many variants try to use a property instead.
///
/// # Dependencies
///
/// This component must always be paired with a [`CompiledParticleEffect`]
/// component. Failure to do so will prevent the effect instance from working.
///
/// When spawning a new [`ParticleEffect`], consider using the
/// [`ParticleEffectBundle`] to ensure all the necessary components are present
/// on the entity for the effect to render correctly.
///
/// # Change detection
///
/// The [`CompiledParticleEffect`] component located on the same [`Entity`] as
/// the current [`ParticleEffect`] component is automatically updated when a
/// change occurs to this component. This separation is designed to leverage the
/// change detection system of Bevy, in order to ensure that any manual change
/// by the user is performed on this component and any automated change is
/// performed on the [`CompiledParticleEffect`] component. This allows
/// efficiently detecting when a user change requires an update to the compiled
/// particle effect, while ensuring conversely that internal mutations do not
/// invalidate the compiled effect and do not trigger a costly shader rebuild
/// for example.
#[derive(Debug, Default, Clone, Component, Reflect)]
#[reflect(Component)]
pub struct ParticleEffect {
    /// Handle of the effect to instantiate.
    pub handle: Handle<EffectAsset>,
    /// For 2D rendering, override the value of the Z coordinate of the layer at
    /// which the particles are rendered.
    ///
    /// This value is passed to the render pipeline and used when sorting
    /// transparent items to render, to order them. As a result, effects
    /// with different Z values cannot be batched together, which may
    /// negatively affect performance.
    ///
    /// Note that this value is shared by all particles of the effect instance.
    ///
    /// This is only available with the `2d` feature.
    #[cfg(feature = "2d")]
    pub z_layer_2d: Option<f32>,
}

impl ParticleEffect {
    /// Create a new particle effect without a spawner or any modifier.
    pub fn new(handle: Handle<EffectAsset>) -> Self {
        Self {
            handle,
            #[cfg(feature = "2d")]
            z_layer_2d: None,
        }
    }

    /// Set the value of the Z layer used when rendering in 2D mode.
    ///
    /// In 2D mode, the Bevy renderer sorts all render items according to their
    /// Z layer value, from back (negative) to front (positive). This
    /// function sets the value assigned to the current particle effect, to
    /// order it relative to any other 2D render item (including non-effects).
    /// Setting the value to `None` reverts to the default sorting and put the
    /// effect back into the default layer.
    ///
    /// This function has no effect when rendering in 3D mode.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// # use bevy::asset::Handle;
    /// # let asset = Handle::<EffectAsset>::default();
    /// // Always render the effect in front of the default layer (z=0)
    /// let effect = ParticleEffect::new(asset).with_z_layer_2d(Some(0.1));
    /// ```
    #[cfg(feature = "2d")]
    pub fn with_z_layer_2d(mut self, z_layer_2d: Option<f32>) -> Self {
        self.z_layer_2d = z_layer_2d;
        self
    }
}

/// Material for an effect instance.
///
/// A material component contains the render resources (textures) actually bound
/// to the slots defined in an effect's [`Module`]. That way, a same effect can
/// be rendered multiple times with different sets of textures.
///
/// The [`EffectMaterial`] component needs to be spawned on the same entity as
/// the [`ParticleEffect`].
#[derive(Debug, Default, Clone, Component)]
pub struct EffectMaterial {
    /// List of images to use to render the effect instance.
    pub images: Vec<Handle<Image>>,
}

/// Texture slot of a [`Module`].
///
/// A texture slot defines a named bind point where a texture can be attached
/// and sampled by an effect during rendering.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct TextureSlot {
    /// Unique slot name.
    pub name: String,
}

/// Texture layout.
///
/// Defines the list of texture slots for an effect.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub struct TextureLayout {
    /// The list of slots.
    pub layout: Vec<TextureSlot>,
}

/// Effect shader.
///
/// Contains the configured shaders for the init, update, and render passes.
#[derive(Debug, Default, Clone)]
pub(crate) struct EffectShader {
    pub init: Handle<Shader>,
    pub update: Handle<Shader>,
    pub render: Handle<Shader>,
}

/// Source code (WGSL) of an effect.
///
/// The source code is generated from an [`EffectAsset`] by applying all
/// modifiers. The resulting source code is configured (the Hanabi variables
/// `{{VARIABLE}}` are replaced by the relevant WGSL code) but is not
/// specialized (the conditional directives like `#if` are still present).
#[derive(Debug)]
struct EffectShaderSource {
    pub shaders: Vec<EffectGroupShaderSource>,
    pub layout_flags: LayoutFlags,
}

#[derive(Debug)]
struct EffectGroupShaderSource {
    init: String,
    update: String,
    render: String,
}

/// Error resulting from the generating of the WGSL shader code of an
/// [`EffectAsset`].
#[derive(Debug, Error)]
enum ShaderGenerateError {
    #[error("Expression error: {0:?}")]
    Expr(ExprError),

    #[error("Validation error: {0:?}")]
    Validate(String),
}

impl EffectShaderSource {
    /// Generate the effect shader WGSL source code.
    ///
    /// This takes a base asset effect and generate the WGSL code for the
    /// various shaders (init/update/render).
    pub fn generate(asset: &EffectAsset) -> Result<EffectShaderSource, ShaderGenerateError> {
        let particle_layout = asset.particle_layout();

        // The particle layout cannot be empty currently because we always emit some
        // Particle{} struct and it needs at least one field. There's probably no use
        // case for an empty layout anyway.
        if particle_layout.size() == 0 {
            return Err(ShaderGenerateError::Validate(format!(
                "Asset {} has invalid empty particle layout.",
                asset.name
            )));
        }

        // Currently the POSITION attribute is mandatory, as it's always used by the
        // render shader.
        if !particle_layout.contains(Attribute::POSITION) {
            return Err(ShaderGenerateError::Validate(format!(
                "The particle layout of asset {} is missing the {} attribute. Add a modifier using that attribute, for example the SetAttributeModifier.",
                asset.name, Attribute::POSITION.name()
            )));
        }

        // Generate the WGSL code declaring all the attributes inside the Particle
        // struct.
        let attributes_code = particle_layout.generate_code();

        // For the renderer, assign all its inputs to the values of the attributes
        // present, or a default value.
        let mut inputs_code = String::new();
        // All required attributes, except the size/color which are variadic
        let required_attributes =
            HashSet::from_iter([Attribute::AXIS_X, Attribute::AXIS_Y, Attribute::AXIS_Z]);
        let mut present_attributes = HashSet::new();
        let mut has_size = false;
        let mut has_color = false;
        for attr_layout in particle_layout.attributes() {
            let attr = attr_layout.attribute;
            if attr == Attribute::SIZE {
                if !has_size {
                    inputs_code += &format!(
                        "var size = vec2<f32>(particle.{0}, particle.{0});\n",
                        Attribute::SIZE.name()
                    );
                    has_size = true;
                } else {
                    warn!("Attribute SIZE conflicts with another size attribute; ignored.");
                }
            } else if attr == Attribute::SIZE2 {
                if !has_size {
                    inputs_code += &format!("var size = particle.{0};\n", Attribute::SIZE2.name());
                    has_size = true;
                } else {
                    warn!("Attribute SIZE2 conflicts with another size attribute; ignored.");
                }
            } else if attr == Attribute::HDR_COLOR {
                if !has_color {
                    inputs_code +=
                        &format!("var color = particle.{};\n", Attribute::HDR_COLOR.name());
                    has_color = true;
                } else {
                    warn!("Attribute HDR_COLOR conflicts with another color attribute; ignored.");
                }
            } else if attr == Attribute::COLOR {
                if !has_color {
                    inputs_code += &format!(
                        "var color = unpack4x8unorm(particle.{0});\n",
                        Attribute::COLOR.name()
                    );
                    has_color = true;
                } else {
                    warn!("Attribute COLOR conflicts with another color attribute; ignored.");
                }
            } else {
                inputs_code += &format!("var {0} = particle.{0};\n", attr.name());
                present_attributes.insert(attr);
            }
        }
        // For all attributes required by the render shader, if they're not explicitly
        // stored in the particle layout, define a variable with their default value.
        if !has_size {
            inputs_code += &format!(
                "var size = {0};\n",
                Attribute::SIZE2.default_value().to_wgsl_string() // TODO - or SIZE?
            );
        }
        if !has_color {
            inputs_code += &format!(
                "var color = {};\n",
                Attribute::HDR_COLOR.default_value().to_wgsl_string() // TODO - or COLOR?
            );
        }
        for &attr in required_attributes.difference(&present_attributes) {
            inputs_code += &format!(
                "var {} = {};\n",
                attr.name(),
                attr.default_value().to_wgsl_string()
            );
        }

        // Generate the shader code defining the per-effect properties, if any
        let property_layout = asset.property_layout();
        let properties_code = property_layout.generate_code();
        let properties_binding_code = if property_layout.is_empty() {
            "// (no properties)".to_string()
        } else {
            "@group(1) @binding(3) var<storage, read> properties : Properties;".to_string()
        };

        // Start from the base module containing the expressions actually serialized in
        // the asset. We will add the ones created on-the-fly by applying the
        // modifiers to the contexts.
        let mut module = asset.module().clone();

        let mut layout_flags = LayoutFlags::NONE;
        if asset.simulation_space == SimulationSpace::Local {
            layout_flags |= LayoutFlags::LOCAL_SPACE_SIMULATION;
        }
        if let AlphaMode::Mask(_) = &asset.alpha_mode {
            layout_flags |= LayoutFlags::USE_ALPHA_MASK;
        }
        if asset.ribbon_group.is_some() {
            layout_flags |= LayoutFlags::RIBBONS;
        }

        let mut group_shader_sources = vec![];

        // Configure the init shader template, and make sure a corresponding shader
        // asset exists
        for dest_group_index in 0..(asset.init.len() as u32) {
            // Generate the shader code for the initializing shader
            let (init_code, init_extra, init_sim_space_transform_code) = {
                let mut init_context =
                    ShaderWriter::new(ModifierContext::Init, &property_layout, &particle_layout);
                for m in asset.init_modifiers_for_group(dest_group_index) {
                    if let Err(err) = m.apply(&mut module, &mut init_context) {
                        error!("Failed to compile effect, error in init context: {:?}", err);
                        return Err(ShaderGenerateError::Expr(err));
                    }
                }

                let sim_space_transform_code =
                    asset.simulation_space.eval(&init_context).map_err(|err| {
                        error!("Failed to compile effect's simulation space: {:?}", err);
                        ShaderGenerateError::Expr(err)
                    })?;

                (
                    init_context.main_code,
                    init_context.extra_code,
                    sim_space_transform_code,
                )
            };

            let src_group_index = match asset.init[dest_group_index as usize] {
                Initializer::Spawner(_) => 0,
                Initializer::Cloner(cloner) => cloner.src_group_index,
            };

            let init_shader_source = PARTICLES_INIT_SHADER_TEMPLATE
                .replace("{{ATTRIBUTES}}", &attributes_code)
                .replace("{{INIT_CODE}}", &init_code)
                .replace("{{INIT_EXTRA}}", &init_extra)
                .replace("{{PROPERTIES}}", &properties_code)
                .replace("{{PROPERTIES_BINDING}}", &properties_binding_code)
                .replace("{{SRC_GROUP_INDEX}}", &src_group_index.to_string())
                .replace("{{DEST_GROUP_INDEX}}", &dest_group_index.to_string())
                .replace(
                    "{{SIMULATION_SPACE_TRANSFORM_PARTICLE}}",
                    &init_sim_space_transform_code,
                );

            // Generate the shader code for the update shader
            let (mut update_code, update_extra) = {
                let mut update_context =
                    ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
                for m in asset.update_modifiers_for_group(dest_group_index) {
                    if let Err(err) = m.apply(&mut module, &mut update_context) {
                        error!(
                            "Failed to compile effect, error in update context: {:?}",
                            err
                        );
                        return Err(ShaderGenerateError::Expr(err));
                    }
                }
                (update_context.main_code, update_context.extra_code)
            };

            // Insert Euler motion integration if needed.
            let has_position = present_attributes.contains(&Attribute::POSITION);
            let has_velocity = present_attributes.contains(&Attribute::VELOCITY);
            if asset.motion_integration != MotionIntegration::None {
                if has_position && has_velocity {
                    // Note the prepended "\n" to prevent appending to a comment line.
                    let code = format!(
                        "\nparticle.{0} += particle.{1} * sim_params.delta_time;\n",
                        Attribute::POSITION.name(),
                        Attribute::VELOCITY.name()
                    );
                    if asset.motion_integration == MotionIntegration::PreUpdate {
                        update_code.insert_str(0, &code);
                    } else {
                        update_code += &code;
                    }
                } else {
                    warn!(
                        "Asset {} specifies motion integration but is missing {}.",
                        asset.name,
                        if has_position {
                            "Attribute::VELOCITY"
                        } else {
                            "Attribute::POSITION"
                        }
                    )
                }
            }

            // Generate the shader code for the render shader
            let (
                vertex_code,
                fragment_code,
                render_extra,
                alpha_cutoff_code,
                flipbook_scale_code,
                flipbook_row_count_code,
                material_bindings_code,
            ) = {
                let texture_layout = module.texture_layout();
                let mut render_context =
                    RenderContext::new(&property_layout, &particle_layout, &texture_layout);
                for m in asset.render_modifiers_for_group(dest_group_index) {
                    m.apply_render(&mut module, &mut render_context)
                        .map_err(ShaderGenerateError::Expr)?;
                }

                if render_context.needs_uv {
                    layout_flags |= LayoutFlags::NEEDS_UV;
                }

                let alpha_cutoff_code = if let AlphaMode::Mask(cutoff) = &asset.alpha_mode {
                    render_context.eval(&module, *cutoff).unwrap_or_else(|err| {
                        error!(
                            "Failed to evaluate the expression for AlphaMode::Mask, error: {:?}",
                            err
                        );

                        // In Debug, show everything to help diagnosing
                        #[cfg(debug_assertions)]
                        return 1_f32.to_wgsl_string();

                        // In Release, hide everything with an error
                        #[cfg(not(debug_assertions))]
                        return 0_f32.to_wgsl_string();
                    })
                } else {
                    String::new()
                };

                let (flipbook_scale_code, flipbook_row_count_code) =
                    if let Some(grid_size) = render_context.sprite_grid_size {
                        layout_flags |= LayoutFlags::FLIPBOOK;
                        // Note: row_count needs to be i32, not u32, because of sprite_index
                        let flipbook_row_count_code = (grid_size.x as i32).to_wgsl_string();
                        let flipbook_scale_code =
                            Vec2::new(1.0 / grid_size.x as f32, 1.0 / grid_size.y as f32)
                                .to_wgsl_string();
                        (flipbook_scale_code, flipbook_row_count_code)
                    } else {
                        (String::new(), String::new())
                    };

                trace!(
                    "Generating material bindings code for layout: {:?}",
                    texture_layout
                );
                let mut material_bindings_code = String::new();
                for (slot, _) in texture_layout.layout.iter().enumerate() {
                    material_bindings_code.push_str(&format!(
                        "@group(2) @binding(0) var material_texture_{slot}: texture_2d<f32>;
@group(2) @binding(1) var material_sampler_{slot}: sampler;
"
                    ));
                }

                (
                    render_context.vertex_code,
                    render_context.fragment_code,
                    render_context.render_extra,
                    alpha_cutoff_code,
                    flipbook_scale_code,
                    flipbook_row_count_code,
                    material_bindings_code,
                )
            };

            // Configure aging code
            let has_age = present_attributes.contains(&Attribute::AGE);
            let has_lifetime = present_attributes.contains(&Attribute::LIFETIME);
            let alive_init_code = if has_age && has_lifetime {
                format!(
                    "var is_alive = particle.{0} < particle.{1};",
                    Attribute::AGE.name(),
                    Attribute::LIFETIME.name()
                )
            } else {
                // Since we're using a dead index buffer, all particles that make it to the
                // update compute shader are guaranteed to be alive (we never
                // simulate dead particles).
                "var is_alive = true;".to_string()
            };
            let age_code = if has_age {
                format!(
                    "particle.{0} = particle.{0} + sim_params.delta_time;",
                    Attribute::AGE.name()
                )
            } else {
                "".to_string()
            } + "\n    "
                + &alive_init_code;

            // Configure reaping code
            let reap_code = if has_age && has_lifetime {
                format!(
                    "is_alive = is_alive && (particle.{0} < particle.{1});",
                    Attribute::AGE.name(),
                    Attribute::LIFETIME.name()
                )
            } else {
                "".to_string()
            };

            // Assign attributes individually instead of using struct
            // assignment. Otherwise we might race on `PREV` and `NEXT`
            // attributes, which might be updated behind our back when adjacent
            // particles die.
            let mut writeback_code = "".to_owned();
            for attribute in present_attributes.iter().filter(|attribute| {
                **attribute != Attribute::PREV && **attribute != Attribute::NEXT
            }) {
                writeln!(
                    &mut writeback_code,
                    "    particle_buffer.particles[index].{0} = particle.{0};",
                    attribute.name()
                )
                .unwrap();
            }

            let dest_group_index_code = format!("{}", dest_group_index);

            // Configure the update shader template, and make sure a corresponding shader
            // asset exists
            let update_shader_source = PARTICLES_UPDATE_SHADER_TEMPLATE
                .replace("{{ATTRIBUTES}}", &attributes_code)
                .replace("{{AGE_CODE}}", &age_code)
                .replace("{{REAP_CODE}}", &reap_code)
                .replace("{{UPDATE_CODE}}", &update_code)
                .replace("{{WRITEBACK_CODE}}", &writeback_code)
                .replace("{{UPDATE_EXTRA}}", &update_extra)
                .replace("{{PROPERTIES}}", &properties_code)
                .replace("{{PROPERTIES_BINDING}}", &properties_binding_code)
                .replace("{{GROUP_INDEX}}", &dest_group_index_code);
            trace!("Configured update shader:\n{}", update_shader_source);

            // Configure the render shader template, and make sure a corresponding shader
            // asset exists
            let render_shader_source = PARTICLES_RENDER_SHADER_TEMPLATE
                .replace("{{ATTRIBUTES}}", &attributes_code)
                .replace("{{INPUTS}}", &inputs_code)
                .replace("{{MATERIAL_BINDINGS}}", &material_bindings_code)
                .replace("{{VERTEX_MODIFIERS}}", &vertex_code)
                .replace("{{FRAGMENT_MODIFIERS}}", &fragment_code)
                .replace("{{RENDER_EXTRA}}", &render_extra)
                .replace("{{ALPHA_CUTOFF}}", &alpha_cutoff_code)
                .replace("{{FLIPBOOK_SCALE}}", &flipbook_scale_code)
                .replace("{{FLIPBOOK_ROW_COUNT}}", &flipbook_row_count_code);
            trace!("Configured render shader:\n{}", render_shader_source);

            group_shader_sources.push(EffectGroupShaderSource {
                init: init_shader_source,
                update: update_shader_source,
                render: render_shader_source,
            });
        }

        Ok(EffectShaderSource {
            shaders: group_shader_sources,
            layout_flags,
        })
    }
}

/// Compiled data for a [`ParticleEffect`].
///
/// This component is managed automatically, and generally should not be
/// accessed manually. It contains data generated from the associated
/// [`ParticleEffect`] component located on the same [`Entity`]. The data is
/// split into this component in particular for change detection reasons, and
/// any change to the associated [`ParticleEffect`] will cause the values of
/// this component to be recalculated. Otherwise the data is cached
/// frame-to-frame for performance.
///
/// All [`ParticleEffect`]s are compiled by the system running in the
/// [`EffectSystems::CompileEffects`] set every frame when they're spawned or
/// when they change, irrelevant of whether the entity if visible
/// ([`Visibility::Visible`]).
#[derive(Debug, Clone, Component)]
pub struct CompiledParticleEffect {
    /// Handle to the underlying asset.
    asset: Handle<EffectAsset>,
    /// Cached simulation condition, to avoid having to query the asset each
    /// time we need it.
    simulation_condition: SimulationCondition,
    /// Handle to the effect shaders for his effect instance (one per group), if
    /// configured.
    effect_shaders: Vec<EffectShader>,
    /// Textures used by the effect, if any.
    textures: Vec<Handle<Image>>,
    /// 2D layer for the effect instance.
    #[cfg(feature = "2d")]
    z_layer_2d: FloatOrd,
    /// Layout flags.
    layout_flags: LayoutFlags,
    /// Alpha mode.
    alpha_mode: AlphaMode,
}

impl Default for CompiledParticleEffect {
    fn default() -> Self {
        Self {
            asset: default(),
            simulation_condition: SimulationCondition::default(),
            effect_shaders: vec![],
            textures: vec![],
            #[cfg(feature = "2d")]
            z_layer_2d: FloatOrd(0.0),
            layout_flags: LayoutFlags::NONE,
            alpha_mode: default(),
        }
    }
}

impl CompiledParticleEffect {
    /// Clear the compiled data from this component.
    pub(crate) fn clear(&mut self) {
        self.asset = Handle::default();
        self.effect_shaders.clear();
        self.textures.clear();
    }

    /// Update the compiled effect from its asset and instance.
    pub(crate) fn update(
        &mut self,
        rebuild: bool,
        #[cfg(feature = "2d")] z_layer_2d: FloatOrd,
        instance: &ParticleEffect,
        material: Option<&EffectMaterial>,
        asset: &EffectAsset,
        shaders: &mut ResMut<Assets<Shader>>,
        shader_cache: &mut ResMut<ShaderCache>,
    ) {
        trace!(
            "Updating (rebuild:{}) compiled particle effect '{}' ({:?})",
            rebuild,
            asset.name,
            instance.handle,
        );

        // #289 - Panic in fn extract_effects
        // We now keep a strong handle. Since CompiledParticleEffect is kept in sync
        // with the source ParticleEffect, this shouldn't produce any strong cyclic
        // dependency.
        debug_assert!(instance.handle.is_strong());

        // Note: if something marked the ParticleEffect as changed (via Mut for example)
        // but didn't actually change anything, or at least didn't change the asset,
        // then we may end up here with the same asset handle. Don't try to be
        // too smart, and rebuild everything anyway, it's easier than trying to
        // diff what may or may not have changed.
        self.asset = instance.handle.clone();
        self.simulation_condition = asset.simulation_condition;

        // Check if the instance changed. If so, rebuild some data from this compiled
        // effect based on the new data of the effect instance.
        if rebuild {
            // Clear the compiled effect if the effect instance changed. We could try to get
            // smarter here, only invalidate what changed, but for now just wipe everything
            // and rebuild from scratch all three shaders together.
            self.effect_shaders.clear();

            // Update the 2D layer
            #[cfg(feature = "2d")]
            {
                self.z_layer_2d = z_layer_2d;
            }
        }

        // If the shaders are already compiled, there's nothing more to do
        if !self.effect_shaders.is_empty() {
            return;
        }

        let shader_source = match EffectShaderSource::generate(asset) {
            Ok(shader_source) => shader_source,
            Err(err) => {
                error!(
                    "Failed to generate shaders for effect asset {}: {:?}",
                    asset.name, err
                );
                return;
            }
        };

        self.layout_flags = shader_source.layout_flags;
        self.alpha_mode = asset.alpha_mode;

        // TODO - Replace with Option<EffectShader { handle: Handle<Shader>, hash:
        // u64 }> where the hash takes into account the code and extra code
        // for each pass (and any other varying item). We don't need to keep
        // around the entire shader code, only a hash of it for compare (or, maybe safer
        // to avoid hash collisions, an index into a shader cache). The only
        // use is to be able to compare 2 instances and see if they can be
        // batched together.
        self.effect_shaders = shader_source
            .shaders
            .iter()
            .map(|effect_group_shader_source| {
                let init = shader_cache.get_or_insert(
                    &asset.name,
                    &effect_group_shader_source.init,
                    shaders,
                );
                let update = shader_cache.get_or_insert(
                    &asset.name,
                    &effect_group_shader_source.update,
                    shaders,
                );
                let render = shader_cache.get_or_insert(
                    &asset.name,
                    &effect_group_shader_source.render,
                    shaders,
                );
                EffectShader {
                    init,
                    update,
                    render,
                }
            })
            .collect();

        trace!(
            "CompiledParticleEffect::update(): shaders={:?} texture_count={} layout_flags={:?}",
            self.effect_shaders,
            material.map(|mat| mat.images.len()).unwrap_or(0),
            self.layout_flags,
        );

        self.textures = material.map(|mat| &mat.images).cloned().unwrap_or_default();
    }

    /// Get the effect shader if configured, or `None` otherwise.
    pub(crate) fn get_configured_shaders(&self) -> &[EffectShader] {
        &self.effect_shaders
    }
}

const PARTICLES_INIT_SHADER_TEMPLATE: &str = include_str!("render/vfx_init.wgsl");
const PARTICLES_UPDATE_SHADER_TEMPLATE: &str = include_str!("render/vfx_update.wgsl");
const PARTICLES_RENDER_SHADER_TEMPLATE: &str = include_str!("render/vfx_render.wgsl");

/// Trait to convert any data structure to its equivalent shader code.
trait ShaderCode {
    /// Generate the shader code for the current state of the object.
    fn to_shader_code(&self, input: &str) -> String;
}

impl ShaderCode for Gradient<Vec2> {
    fn to_shader_code(&self, input: &str) -> String {
        if self.keys().is_empty() {
            return String::new();
        }
        let mut s: String = self
            .keys()
            .iter()
            .enumerate()
            .map(|(index, key)| {
                format!(
                    "let t{0} = {1};\nlet v{0} = {2};",
                    index,
                    key.ratio().to_wgsl_string(),
                    key.value.to_wgsl_string()
                )
            })
            .fold("// Gradient\n".into(), |s, key| s + &key + "\n");
        if self.keys().len() == 1 {
            s + "return v0;\n"
        } else {
            s += &format!("if ({input} <= t0) {{ return v0; }}\n");
            let mut s = self
                .keys()
                .iter()
                .skip(1)
                .enumerate()
                .map(|(index, _key)| {
                    format!(
                        "else if ({input} <= t{1}) {{ return mix(v{0}, v{1}, ({input} - t{0}) / (t{1} - t{0})); }}\n",
                        index,
                        index + 1
                    )
                })
                .fold(s, |s, key| s + &key);
            let _ = writeln!(s, "else {{ return v{}; }}", self.keys().len() - 1);
            s
        }
    }
}

impl ShaderCode for Gradient<Vec4> {
    fn to_shader_code(&self, input: &str) -> String {
        if self.keys().is_empty() {
            return String::new();
        }
        let mut s: String = self
            .keys()
            .iter()
            .enumerate()
            .map(|(index, key)| {
                format!(
                    "let t{0} = {1};\nlet c{0} = {2};",
                    index,
                    key.ratio().to_wgsl_string(),
                    key.value.to_wgsl_string()
                )
            })
            .fold("// Gradient\n".into(), |s, key| s + &key + "\n");
        if self.keys().len() == 1 {
            s + "return c0;\n"
        } else {
            s += &format!("if ({input} <= t0) {{ return c0; }}\n");
            let mut s = self
                .keys()
                .iter()
                .skip(1)
                .enumerate()
                .map(|(index, _key)| {
                    format!(
                        "else if ({input} <= t{1}) {{ return mix(c{0}, c{1}, ({input} - t{0}) / (t{1} - t{0})); }}\n",
                        index,
                        index + 1
                    )
                })
                .fold(s, |s, key| s + &key);
            let _ = writeln!(s, "else {{ return c{}; }}", self.keys().len() - 1);
            s
        }
    }
}

/// Compile all the [`ParticleEffect`] components into a
/// [`CompiledParticleEffect`].
///
/// This system runs in the [`EffectSystems::CompileEffects`] set of the
/// [`PostUpdate`] schedule. It gathers all new instances of [`ParticleEffect`],
/// as well as instances which changed, and (re)compile them into an optimized
/// form saved into the [`CompiledParticleEffect`] component.
///
/// Hidden instances are compiled like visible ones, both to allow users to
/// compile "in the background" by spawning a hidden effect, and also to prevent
/// having mixed state where only some effects are compiled, and effects
/// becoming visible later need to be special casing. If you want to avoid
/// compiling an effect, don't spawn it.
fn compile_effects(
    effects: Res<Assets<EffectAsset>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut shader_cache: ResMut<ShaderCache>,
    mut q_effects: Query<(
        Entity,
        Ref<ParticleEffect>,
        Option<Ref<EffectMaterial>>,
        &mut CompiledParticleEffect,
    )>,
) {
    trace!("compile_effects");

    // Loop over all existing effects to update them, including invisible ones
    for (asset, entity, effect, material, mut compiled_effect) in
        q_effects
            .iter_mut()
            .filter_map(|(entity, effect, material, compiled_effect)| {
                // Check if asset is available, otherwise silently ignore as we can't check for
                // changes, and conceptually it makes no sense to render a particle effect whose
                // asset was unloaded.
                let asset = effects.get(&effect.handle)?;

                Some((asset, entity, effect, material, compiled_effect))
            })
    {
        // If the ParticleEffect didn't change, and the compiled one is for the correct
        // asset, then there's nothing to do.
        let need_rebuild =
            effect.is_changed() || material.as_ref().map_or(false, |r| r.is_changed());
        if !need_rebuild && (compiled_effect.asset == effect.handle) {
            continue;
        }

        if need_rebuild {
            debug!("Invalidating the compiled cache for effect on entity {:?} due to changes in the ParticleEffect component. If you see this message too much, then performance might be affected. Find why the change detection of the ParticleEffect is triggered.", entity);
        }

        #[cfg(feature = "2d")]
        let z_layer_2d = effect
            .z_layer_2d
            .map_or(FloatOrd(asset.z_layer_2d), |z_layer_2d| {
                FloatOrd(z_layer_2d)
            });

        compiled_effect.update(
            need_rebuild,
            #[cfg(feature = "2d")]
            z_layer_2d,
            &effect,
            material.map(|r| r.into_inner()),
            asset,
            &mut shaders,
            &mut shader_cache,
        );
    }

    // Clear removed effects, to allow them to be released by the asset server
    for (_, effect, _, mut compiled_effect) in q_effects.iter_mut() {
        if effects.get(&effect.handle).is_none() {
            compiled_effect.clear();
        }
    }
}

/// Update all properties of a [`ParticleEffect`] into its associated
/// [`EffectProperties`].
///
/// This system runs in the [`EffectSystems::UpdatePropertiesFromAsset`] set of
/// the [`PostUpdate`] schedule. It gathers all new instances of
/// [`ParticleEffect`], as well as instances which changed, and update their
/// associated [`EffectProperties`] component.
///
/// Hidden instances are processed like visible ones, both to allow users to
/// compile "in the background" by spawning a hidden effect, and also to prevent
/// having mixed state where only some effects are compiled, and effects
/// becoming visible later need to be special casing. If you want to avoid
/// compiling an effect, don't spawn it.
fn update_properties_from_asset(
    assets: Res<Assets<EffectAsset>>,
    mut q_effects: Query<(Ref<ParticleEffect>, &mut EffectProperties), Changed<ParticleEffect>>,
) {
    trace!("update_properties_from_asset");

    // Loop over all existing effects, including invisible ones
    for (effect, properties) in q_effects.iter_mut() {
        // Check if the asset is available, otherwise silently ignore as we can't check
        // for changes, and conceptually it makes no sense to render a particle
        // effect whose asset was unloaded.
        let Some(asset) = assets.get(&effect.handle) else {
            continue;
        };

        EffectProperties::update(properties, asset.properties(), effect.is_added());
    }
}

/// Event sent by [`gather_removed_effects()`] with the list of effects removed
/// during this frame.
///
/// The event is consumed during the extract phase by the [`extract_effects()`]
/// system, to clean-up unused GPU resources.
///
/// [`extract_effects()`]: crate::render::extract_effects
#[derive(Event)]
struct RemovedEffectsEvent {
    entities: Vec<Entity>,
}

/// Gather all the removed [`ParticleEffect`] components to allow cleaning-up
/// unused GPU resources.
///
/// This system executes inside the [`EffectSystems::GatherRemovedEffects`]
/// set of the [`PostUpdate`] schedule.
fn gather_removed_effects(
    mut removed_effects: RemovedComponents<ParticleEffect>,
    mut removed_effects_event_writer: EventWriter<RemovedEffectsEvent>,
) {
    let entities: Vec<Entity> = removed_effects.read().collect();
    if !entities.is_empty() {
        removed_effects_event_writer.send(RemovedEffectsEvent { entities });
    }
}

#[cfg(test)]
mod tests {
    use std::ops::DerefMut;

    use bevy::{
        asset::{
            io::{
                memory::{Dir, MemoryAssetReader},
                AssetSourceBuilder, AssetSourceBuilders, AssetSourceId,
            },
            AssetServerMode,
        },
        render::view::{VisibilityPlugin, VisibilitySystems},
        tasks::{IoTaskPool, TaskPoolBuilder},
    };
    use naga_oil::compose::{Composer, NagaModuleDescriptor, ShaderDefValue};

    use super::*;
    use crate::spawn::new_rng;

    const INTS: &[usize] = &[1, 2, 4, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33];
    const INTS_POW2: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

    /// Same as `INTS`, rounded up to 16
    const INTS16: &[usize] = &[16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 48];

    #[test]
    fn next_multiple() {
        // align-1 is no-op
        for &size in INTS {
            assert_eq!(size, next_multiple_of(size, 1));
        }

        // zero-sized is always aligned
        for &align in INTS_POW2 {
            assert_eq!(0, next_multiple_of(0, align));
        }

        // size < align : rounds up to align
        for &size in INTS {
            assert_eq!(256, next_multiple_of(size, 256));
        }

        // size > align : actually aligns
        for (&size, &aligned_size) in INTS.iter().zip(INTS16) {
            assert_eq!(aligned_size, next_multiple_of(size, 16));
        }
    }

    #[test]
    fn to_wgsl_f32() {
        let s = 1.0_f32.to_wgsl_string();
        assert_eq!(s, "1.");
        let s = (-1.0_f32).to_wgsl_string();
        assert_eq!(s, "-1.");
        let s = 1.5_f32.to_wgsl_string();
        assert_eq!(s, "1.5");
        let s = 0.5_f32.to_wgsl_string();
        assert_eq!(s, "0.5");
        let s = 0.123_456_78_f32.to_wgsl_string();
        assert_eq!(s, "0.123457"); // 6 digits
    }

    #[test]
    fn to_wgsl_f64() {
        let s = 1.0_f64.to_wgsl_string();
        assert_eq!(s, "1.");
        let s = (-1.0_f64).to_wgsl_string();
        assert_eq!(s, "-1.");
        let s = 1.5_f64.to_wgsl_string();
        assert_eq!(s, "1.5");
        let s = 0.5_f64.to_wgsl_string();
        assert_eq!(s, "0.5");
        let s = 0.123_456_789_012_345_67_f64.to_wgsl_string();
        assert_eq!(s, "0.123456789012346"); // 15 digits
    }

    #[test]
    fn to_wgsl_vec() {
        let s = Vec2::new(1., 2.).to_wgsl_string();
        assert_eq!(s, "vec2<f32>(1.,2.)");
        let s = Vec3::new(1., 2., -1.).to_wgsl_string();
        assert_eq!(s, "vec3<f32>(1.,2.,-1.)");
        let s = Vec4::new(1., 2., -1., 2.).to_wgsl_string();
        assert_eq!(s, "vec4<f32>(1.,2.,-1.,2.)");
    }

    #[test]
    fn to_wgsl_ivec() {
        let s = IVec2::new(1, 2).to_wgsl_string();
        assert_eq!(s, "vec2<i32>(1,2)");
        let s = IVec3::new(1, 2, -1).to_wgsl_string();
        assert_eq!(s, "vec3<i32>(1,2,-1)");
        let s = IVec4::new(1, 2, -1, 2).to_wgsl_string();
        assert_eq!(s, "vec4<i32>(1,2,-1,2)");
    }

    #[test]
    fn to_wgsl_uvec() {
        let s = UVec2::new(1, 2).to_wgsl_string();
        assert_eq!(s, "vec2<u32>(1u,2u)");
        let s = UVec3::new(1, 2, 42).to_wgsl_string();
        assert_eq!(s, "vec3<u32>(1u,2u,42u)");
        let s = UVec4::new(1, 2, 42, 5).to_wgsl_string();
        assert_eq!(s, "vec4<u32>(1u,2u,42u,5u)");
    }

    #[test]
    fn to_wgsl_bvec() {
        let s = BVec2::new(false, true).to_wgsl_string();
        assert_eq!(s, "vec2<bool>(false,true)");
        let s = BVec3::new(false, true, true).to_wgsl_string();
        assert_eq!(s, "vec3<bool>(false,true,true)");
        let s = BVec4::new(false, true, true, false).to_wgsl_string();
        assert_eq!(s, "vec4<bool>(false,true,true,false)");
    }

    #[test]
    fn to_wgsl_value_f32() {
        let s = CpuValue::Single(1.0_f32).to_wgsl_string();
        assert_eq!(s, "1.");
        let s = CpuValue::Uniform((1.0_f32, 2.0_f32)).to_wgsl_string();
        assert_eq!(s, "(frand() * (2. - 1.) + 1.)");
    }

    #[test]
    fn to_wgsl_value_vec2() {
        let s = CpuValue::Single(Vec2::ONE).to_wgsl_string();
        assert_eq!(s, "vec2<f32>(1.,1.)");
        let s = CpuValue::Uniform((Vec2::ZERO, Vec2::ONE)).to_wgsl_string();
        assert_eq!(
            s,
            "(frand2() * (vec2<f32>(1.,1.) - vec2<f32>(0.,0.)) + vec2<f32>(0.,0.))"
        );
    }

    #[test]
    fn to_wgsl_value_vec3() {
        let s = CpuValue::Single(Vec3::ONE).to_wgsl_string();
        assert_eq!(s, "vec3<f32>(1.,1.,1.)");
        let s = CpuValue::Uniform((Vec3::ZERO, Vec3::ONE)).to_wgsl_string();
        assert_eq!(
            s,
            "(frand3() * (vec3<f32>(1.,1.,1.) - vec3<f32>(0.,0.,0.)) + vec3<f32>(0.,0.,0.))"
        );
    }

    #[test]
    fn to_wgsl_value_vec4() {
        let s = CpuValue::Single(Vec4::ONE).to_wgsl_string();
        assert_eq!(s, "vec4<f32>(1.,1.,1.,1.)");
        let s = CpuValue::Uniform((Vec4::ZERO, Vec4::ONE)).to_wgsl_string();
        assert_eq!(s, "(frand4() * (vec4<f32>(1.,1.,1.,1.) - vec4<f32>(0.,0.,0.,0.)) + vec4<f32>(0.,0.,0.,0.))");
    }

    #[test]
    fn to_shader_code() {
        let mut grad = Gradient::new();
        assert_eq!("", grad.to_shader_code("key"));

        grad.add_key(0.0, Vec4::splat(0.0));
        assert_eq!(
            "// Gradient\nlet t0 = 0.;\nlet c0 = vec4<f32>(0.,0.,0.,0.);\nreturn c0;\n",
            grad.to_shader_code("key")
        );

        grad.add_key(1.0, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(
            r#"// Gradient
let t0 = 0.;
let c0 = vec4<f32>(0.,0.,0.,0.);
let t1 = 1.;
let c1 = vec4<f32>(1.,0.,0.,1.);
if (key <= t0) { return c0; }
else if (key <= t1) { return mix(c0, c1, (key - t0) / (t1 - t0)); }
else { return c1; }
"#,
            grad.to_shader_code("key")
        );
    }

    #[test]
    fn test_simulation_space_eval() {
        let particle_layout = ParticleLayout::empty();
        let property_layout = PropertyLayout::default();
        {
            // Local is always available
            let ctx =
                ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
            assert!(SimulationSpace::Local.eval(&ctx).is_ok());
            assert!(SimulationSpace::Global.eval(&ctx).is_err());

            // Global requires storing the particle's position
            let particle_layout = ParticleLayout::new().append(Attribute::POSITION).build();
            let ctx =
                ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
            assert!(SimulationSpace::Local.eval(&ctx).is_ok());
            assert!(SimulationSpace::Global.eval(&ctx).is_ok());
        }
        {
            // Local is always available
            let ctx =
                ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
            assert!(SimulationSpace::Local.eval(&ctx).is_ok());
            assert!(SimulationSpace::Global.eval(&ctx).is_err());

            // Global requires storing the particle's position
            let particle_layout = ParticleLayout::new().append(Attribute::POSITION).build();
            let ctx =
                ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
            assert!(SimulationSpace::Local.eval(&ctx).is_ok());
            assert!(SimulationSpace::Global.eval(&ctx).is_ok());
        }
        {
            // In the render context, the particle position is always available (either
            // stored or not), so the simulation space can always be evaluated.
            let texture_layout = TextureLayout::default();
            let ctx = RenderContext::new(&property_layout, &particle_layout, &texture_layout);
            assert!(SimulationSpace::Local.eval(&ctx).is_ok());
            assert!(SimulationSpace::Global.eval(&ctx).is_ok());
        }
    }

    fn make_test_app() -> App {
        IoTaskPool::get_or_init(|| {
            TaskPoolBuilder::default()
                .num_threads(1)
                .thread_name("Hanabi test IO Task Pool".to_string())
                .build()
        });

        let mut app = App::new();

        let watch_for_changes = false;
        let mut builders = app
            .world_mut()
            .get_resource_or_insert_with::<AssetSourceBuilders>(Default::default);
        let dir = Dir::default();
        let dummy_builder = AssetSourceBuilder::default()
            .with_reader(move || Box::new(MemoryAssetReader { root: dir.clone() }));
        builders.insert(AssetSourceId::Default, dummy_builder);
        let sources = builders.build_sources(watch_for_changes, false);
        let asset_server =
            AssetServer::new(sources, AssetServerMode::Unprocessed, watch_for_changes);

        app.insert_resource(asset_server);
        // app.add_plugins(DefaultPlugins);
        app.init_asset::<Mesh>();
        app.init_asset::<Shader>();
        app.add_plugins(VisibilityPlugin);
        app.init_resource::<ShaderCache>();
        app.insert_resource(Random(new_rng()));
        app.init_asset::<EffectAsset>();
        app.add_systems(
            PostUpdate,
            compile_effects.after(VisibilitySystems::CheckVisibility),
        );

        app
    }

    /// Test case for `tick_spawners()`.
    struct TestCase {
        /// Initial entity visibility on spawn. If `None`, do not add a
        /// [`Visibility`] component.
        visibility: Option<Visibility>,
    }

    impl TestCase {
        fn new(visibility: Option<Visibility>) -> Self {
            Self { visibility }
        }
    }

    #[test]
    fn test_effect_shader_source() {
        // Empty particle layout
        let module = Module::default();
        let asset = EffectAsset::new(256, Spawner::rate(32.0.into()), module)
            .with_simulation_space(SimulationSpace::Local);
        assert_eq!(asset.simulation_space, SimulationSpace::Local);
        let res = EffectShaderSource::generate(&asset);
        assert!(res.is_err());
        let err = res.err().unwrap();
        assert!(matches!(err, ShaderGenerateError::Validate(_)));

        // Missing Attribute::POSITION, currently mandatory for all effects
        let mut module = Module::default();
        let zero = module.lit(Vec3::ZERO);
        let asset = EffectAsset::new(256, Spawner::rate(32.0.into()), module)
            .init(SetAttributeModifier::new(Attribute::VELOCITY, zero));
        assert!(asset.particle_layout().size() > 0);
        let res = EffectShaderSource::generate(&asset);
        assert!(res.is_err());
        let err = res.err().unwrap();
        assert!(matches!(err, ShaderGenerateError::Validate(_)));

        // Valid
        let mut module = Module::default();
        let zero = module.lit(Vec3::ZERO);
        let asset = EffectAsset::new(256, Spawner::rate(32.0.into()), module)
            .with_simulation_space(SimulationSpace::Local)
            .init(SetAttributeModifier::new(Attribute::POSITION, zero));
        assert_eq!(asset.simulation_space, SimulationSpace::Local);
        let res = EffectShaderSource::generate(&asset);
        assert!(res.is_ok());
        let shader_source = res.unwrap();
        for (name, code) in shader_source
            .shaders
            .iter()
            .map(|shader| ("Init", &*shader.init))
            .chain(
                shader_source
                    .shaders
                    .iter()
                    .map(|shader| ("Update", &*shader.update)),
            )
            .chain(
                shader_source
                    .shaders
                    .iter()
                    .map(|shader| ("Render", &*shader.render)),
            )
        {
            println!("{} shader:\n\n{}", name, code);

            let mut shader_defs = std::collections::HashMap::<String, ShaderDefValue>::new();
            shader_defs.insert("LOCAL_SPACE_SIMULATION".into(), ShaderDefValue::Bool(true));
            shader_defs.insert("NEEDS_UV".into(), ShaderDefValue::Bool(true));
            shader_defs.insert("RENDER_NEEDS_SPAWNER".into(), ShaderDefValue::Bool(true));
            shader_defs.insert(
                "PARTICLE_SCREEN_SPACE_SIZE".into(),
                ShaderDefValue::Bool(true),
            );
            if name == "Update" {
                shader_defs.insert("REM_MAX_SPAWN_ATOMIC".into(), ShaderDefValue::Bool(true));
            }
            let mut composer = Composer::default();

            // Import bevy_render::view for the render shader
            {
                // It's reasonably hard to retrieve the source code for view.wgsl in
                // bevy_render. We use a few tricks to get a Shader that we can
                // then convert into a composable module (which is how imports work in Bevy
                // itself).
                let mut dummy_app = App::new();
                dummy_app.init_resource::<Assets<Shader>>();
                dummy_app.add_plugins(bevy::render::view::ViewPlugin);
                let shaders = dummy_app.world().get_resource::<Assets<Shader>>().unwrap();
                let view_shader = shaders.get(&bevy::render::view::VIEW_TYPE_HANDLE).unwrap();

                let res = composer.add_composable_module(view_shader.into());
                assert!(res.is_ok());
            }

            // Import bevy_hanabi::vfx_common
            {
                let min_storage_buffer_offset_alignment = 256;
                let common_shader =
                    HanabiPlugin::make_common_shader(min_storage_buffer_offset_alignment);
                let res = composer.add_composable_module((&common_shader).into());
                assert!(res.is_ok());
            }

            match composer.make_naga_module(NagaModuleDescriptor {
                source: code,
                file_path: &format!("{}.wgsl", name),
                shader_defs,
                ..Default::default()
            }) {
                Ok(module) => {
                    // println!("shader: {:#?}", module);
                    let info = naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        naga::valid::Capabilities::default(),
                    )
                    .validate(&module)
                    .unwrap();
                    let wgsl = naga::back::wgsl::write_string(
                        &module,
                        &info,
                        naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
                    )
                    .unwrap();
                    println!("Final wgsl from naga:\n\n{}", wgsl);
                    // Ok(module)
                }
                Err(e) => {
                    panic!("{}", e.emit_to_string(&composer));
                    // Err(e)
                }
            }

            // let mut frontend = Frontend::new();
            // let res = frontend.parse(code);
            // if let Err(err) = &res {
            //     println!("{} code: {}", name, code);
            //     println!("Err: {:?}", err);
            // }
            // assert!(res.is_ok());
        }
    }

    // Regression test for #343
    #[test]
    fn test_compile_effect_invalid_handle() {
        let mut app = make_test_app();

        let effect_entity = {
            let world = app.world_mut();

            // Spawn particle effect
            let entity = world.spawn(ParticleEffectBundle::default()).id();

            // Spawn a camera, otherwise ComputedVisibility stays at HIDDEN
            world.spawn(Camera3dBundle::default());

            entity
        };

        // Tick once
        app.update();

        // Check
        {
            let world = app.world_mut();

            let (entity, particle_effect, compiled_particle_effect) = world
                .query::<(Entity, &ParticleEffect, &CompiledParticleEffect)>()
                .iter(world)
                .next()
                .unwrap();
            assert_eq!(entity, effect_entity);
            assert_eq!(particle_effect.handle, Handle::<EffectAsset>::default());

            // `compile_effects()` cannot update the CompiledParticleEffect because the
            // asset is invalid
            assert_eq!(
                compiled_particle_effect.asset,
                Handle::<EffectAsset>::default()
            );
            assert!(compiled_particle_effect.effect_shaders.is_empty());
        }
    }

    // Regression test for #228
    #[test]
    fn test_compile_effect_changed() {
        let spawner = Spawner::once(32.0.into(), true);

        let mut app = make_test_app();

        let (effect_entity, handle) = {
            let world = app.world_mut();

            // Add effect asset
            let mut assets = world.resource_mut::<Assets<EffectAsset>>();
            let mut module = Module::default();
            let init_pos = module.lit(Vec3::ZERO);
            let mut asset = EffectAsset::new(64, spawner, module)
                .init(SetAttributeModifier::new(Attribute::POSITION, init_pos));
            asset.simulation_condition = SimulationCondition::Always;
            let handle = assets.add(asset);

            // Spawn particle effect
            let entity = world
                .spawn((
                    ParticleEffect::new(handle.clone()),
                    CompiledParticleEffect::default(),
                ))
                .id();

            // Spawn a camera, otherwise ComputedVisibility stays at HIDDEN
            world.spawn(Camera3dBundle::default());

            (entity, handle)
        };

        // Tick once
        app.update();

        // Check
        {
            let world = app.world_mut();

            let (entity, particle_effect, compiled_particle_effect) = world
                .query::<(Entity, &ParticleEffect, &CompiledParticleEffect)>()
                .iter(world)
                .next()
                .unwrap();
            assert_eq!(entity, effect_entity);
            assert_eq!(particle_effect.handle, handle);

            // `compile_effects()` always updates the CompiledParticleEffect
            assert_eq!(compiled_particle_effect.asset, handle);
            assert!(compiled_particle_effect.asset.is_strong());
            assert!(!compiled_particle_effect.effect_shaders.is_empty());
        }

        // Mark as changed without actually changing anything
        {
            let world = app.world_mut();

            let mut particle_effect = world
                .query::<&mut ParticleEffect>()
                .iter_mut(world)
                .next()
                .unwrap();

            // Force via Mut to mark the component as changed
            particle_effect.deref_mut();
        }

        // Tick once - Regression test for #228, this should not panic
        app.update();

        // Check again, nothing changed
        {
            let world = app.world_mut();

            let (entity, particle_effect, compiled_particle_effect) = world
                .query::<(Entity, &ParticleEffect, &CompiledParticleEffect)>()
                .iter(world)
                .next()
                .unwrap();
            assert_eq!(entity, effect_entity);
            assert_eq!(particle_effect.handle, handle);

            // `compile_effects()` always updates the CompiledParticleEffect
            assert_eq!(compiled_particle_effect.asset, handle);
            assert!(compiled_particle_effect.asset.is_strong());
            assert!(!compiled_particle_effect.effect_shaders.is_empty());
        }
    }

    #[test]
    fn test_compile_effect_visibility() {
        let spawner = Spawner::once(32.0.into(), true);

        for test_case in &[
            TestCase::new(None),
            TestCase::new(Some(Visibility::Hidden)),
            TestCase::new(Some(Visibility::Visible)),
        ] {
            let mut app = make_test_app();

            let (effect_entity, handle) = {
                let world = app.world_mut();

                // Add effect asset
                let mut assets = world.resource_mut::<Assets<EffectAsset>>();
                let mut module = Module::default();
                let init_pos = module.lit(Vec3::ZERO);
                let mut asset = EffectAsset::new(64, spawner, module)
                    .init(SetAttributeModifier::new(Attribute::POSITION, init_pos));
                asset.simulation_condition = if test_case.visibility.is_some() {
                    SimulationCondition::WhenVisible
                } else {
                    SimulationCondition::Always
                };
                // Use local simulation space so we don't need to store Attribute::POSITION for
                // particles
                asset.simulation_space = SimulationSpace::Local;
                let handle = assets.add(asset);

                // Spawn particle effect
                let entity = if let Some(visibility) = test_case.visibility {
                    world
                        .spawn((
                            visibility,
                            InheritedVisibility::default(),
                            ParticleEffect::new(handle.clone()),
                            CompiledParticleEffect::default(),
                        ))
                        .id()
                } else {
                    world
                        .spawn((
                            ParticleEffect::new(handle.clone()),
                            CompiledParticleEffect::default(),
                        ))
                        .id()
                };

                // Spawn a camera, otherwise ComputedVisibility stays at HIDDEN
                world.spawn(Camera3dBundle::default());

                (entity, handle)
            };

            // Tick once
            app.update();

            let world = app.world_mut();

            // Check the state of the components after `tick_spawners()` ran
            if let Some(test_visibility) = test_case.visibility {
                // Simulated-when-visible effect (SimulationCondition::WhenVisible)

                let (
                    entity,
                    visibility,
                    inherited_visibility,
                    particle_effect,
                    compiled_particle_effect,
                ) = world
                    .query::<(
                        Entity,
                        &Visibility,
                        &InheritedVisibility,
                        &ParticleEffect,
                        &CompiledParticleEffect,
                    )>()
                    .iter(world)
                    .next()
                    .unwrap();
                assert_eq!(entity, effect_entity);
                assert_eq!(visibility, test_visibility);
                assert_eq!(
                    inherited_visibility.get(),
                    test_visibility == Visibility::Visible
                );
                assert_eq!(particle_effect.handle, handle);

                // `compile_effects()` always updates the CompiledParticleEffect of new effects,
                // even if hidden
                assert_eq!(compiled_particle_effect.asset, handle);
                assert!(compiled_particle_effect.asset.is_strong());
                assert!(!compiled_particle_effect.effect_shaders.is_empty());

                // Toggle visibility and tick once more; this shouldn't panic (regression; #182)
                let (mut visibility, _) = world
                    .query::<(&mut Visibility, &ParticleEffect)>()
                    .iter_mut(world)
                    .next()
                    .unwrap();
                if *visibility == Visibility::Visible {
                    *visibility = Visibility::Hidden;
                } else {
                    *visibility = Visibility::Visible;
                }
                app.update();
            } else {
                // Always-simulated effect (SimulationCondition::Always)

                let (entity, particle_effect, compiled_particle_effect) = world
                    .query::<(Entity, &ParticleEffect, &CompiledParticleEffect)>()
                    .iter(world)
                    .next()
                    .unwrap();
                assert_eq!(entity, effect_entity);
                assert_eq!(particle_effect.handle, handle);

                // `compile_effects()` always updates the CompiledParticleEffect
                assert_eq!(compiled_particle_effect.asset, handle);
                assert!(compiled_particle_effect.asset.is_strong());
                assert!(!compiled_particle_effect.effect_shaders.is_empty());
            }
        }
    }
}
