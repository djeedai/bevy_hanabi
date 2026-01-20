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
//! The library supports the `wasm` target (WebAssembly) via the WebGPU renderer
//! only. Compute shaders are not available via the legacy WebGL2 renderer.
//! See Bevy's own [WebGL2 and WebGPU](https://github.com/bevyengine/bevy/tree/latest/examples#webgl2-and-webgpu)
//! section of the examples README for more information on how to run Wasm
//! builds with WebGPU.
//!
//! # 2D vs. 3D
//!
//! 🎆 Hanabi integrates both with the 2D and the 3D core pipelines of Bevy. The
//! 2D pipeline integration is controlled by the `2d` cargo feature, while the
//! 3D pipeline integration is controlled by the `3d` cargo feature. Both
//! features are enabled by default for convenience. As an optimization, users
//! can disable default features and re-enable only one of the two modes. At
//! least one of the `2d` or `3d` features must be enabled.
//!
//! ```toml
//! # Example: enable only 3D integration
//! bevy_hanabi = { version = "0.18", default-features = false, features = ["3d"] }
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
//!     let mut gradient = bevy_hanabi::Gradient::new();
//!     gradient.add_key(0.0, Vec4::new(1., 0., 0., 1.));
//!     gradient.add_key(1.0, Vec4::ZERO);
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
//!     // is almost always required, otherwise the particles will stay
//!     // alive forever, and new particles can't be spawned instead.
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
//!         SpawnerSettings::rate(5.0.into()),
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
//!     .render(ColorOverLifetimeModifier {
//!         gradient,
//!         blend: ColorBlendMode::Overwrite,
//!         mask: ColorBlendMask::RGBA,
//!     });
//!
//!     // Insert into the asset system
//!     let effect_asset = effects.add(effect);
//! }
//! ```
//!
//! Then add an instance of that effect to an entity by spawning a
//! [`ParticleEffect`] component referencing the asset.
//!
//! ```
//! # use bevy::prelude::*;
//! # use bevy_hanabi::prelude::*;
//! # fn spawn_effect(mut commands: Commands) {
//! #   let effect_asset = Handle::<EffectAsset>::default();
//! commands.spawn((
//!     ParticleEffect::new(effect_asset),
//!     Transform::from_translation(Vec3::Y),
//! ));
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
//!    effect instance by spawning a [`ParticleEffect`] component. The component
//!    references the [`EffectAsset`] via its `handle` field. Multiple instances
//!    can reference the same asset at the same time, and some changes to the
//!    asset are reflected to its instances, although not all changes are
//!    supported. In general, avoid changing an [`EffectAsset`] while it's in
//!    use by one or more [`ParticleEffect`].
//! 3. If using properties, spawn an [`EffectProperties`] component on the same
//!    entity. Then update properties through that component at any time while
//!    the effect is active. This allows some moderate CPU-side control over the
//!    simulation and rendering of the effect, without having to destroy the
//!    effect and re-create a new one.
//! 4. If using textures, spawn an [`EffectMaterial`] component to define which
//!    texture is bound to which slot in the effect. An [`EffectAsset`] only
//!    defines "slots" of textures, not the actual assets bound to those slots.
//!    This way, you can reuse the same effect asset multiple times with
//!    different textures, like you'd do with a regular rendering mesh.
//! 5. For advanced VFX composed of multiple hierarchical effects, where two or
//!    more effects are connected to each other in a parent-child relationship,
//!    spawn an [`EffectParent`] on any child effect instance to specify its
//!    parent instance. See also the [`EmitSpawnEventModifier`].
//!
//! The [`EffectAsset`] is intended to be the serialized effect format, which
//! authors can save to disk and ship with their application. At this time
//! however serialization and deserialization is still a work in progress. In
//! particular, serialization and deserialization of all
//! [modifiers](crate::modifier) is currently not supported on `wasm` target.

use std::fmt::Write as _;

use bevy::{
    asset::AsAssetId,
    camera::visibility::VisibilityClass,
    platform::collections::{HashMap, HashSet},
    prelude::*,
    render::{extract_component::ExtractComponent, sync_world::SyncToRenderWorld},
};
use rand::{Rng, SeedableRng as _};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod asset;
pub mod attributes;
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

pub use asset::{
    AlphaMode, DefaultMesh, EffectAsset, EffectParent, MotionIntegration, SimulationCondition,
};
pub use attributes::*;
pub use gradient::{Gradient, GradientKey};
pub use graph::*;
pub use modifier::*;
pub use plugin::{EffectSystems, HanabiPlugin};
pub use properties::*;
pub use render::{DebugSettings, LayoutFlags, ShaderCache};
pub use spawn::{tick_spawners, CpuValue, EffectSpawner, Random, SpawnerSettings};
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
    /// influenced anymore by the emitter's [`GlobalTransform`] after spawning.
    /// The particle's [`Attribute::POSITION`] is the world space position
    /// of the particle.
    ///
    /// This is the default.
    #[default]
    Global,

    /// Particles are simulated in local effect space.
    ///
    /// The local space is the space associated with the [`GlobalTransform`] of
    /// the [`ParticleEffect`] component being simulated. Particles
    /// simulated in local effect space are "attached" to the effect, and
    /// will be affected by its [`GlobalTransform`]. The particle's
    /// [`Attribute::POSITION`] is the position of the particle relative to
    /// the effect's [`GlobalTransform`].
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

/// The [`VisibilityClass`] used for all particle effects.
#[derive(Default, Clone, Copy, Component, ExtractComponent)]
pub struct EffectVisibilityClass;

/// Particle-based visual effect instance.
///
/// The particle effect component represents a single instance of a visual
/// effect. The visual effect itself is described by a handle to an
/// [`EffectAsset`].
///
/// This instance is associated to an [`Entity`], inheriting
/// its [`GlobalTransform`] as the origin frame for its particle spawning.
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
/// # Component dependencies
///
/// ## Mandatory components
///
/// The [`ParticleEffect`] component requires, in the sense of ECS, some other
/// components for the particle effect to work. Those mandatory components are
/// automatically added by Bevy if not otherwise provided during spawning.
///
/// - A [`CompiledParticleEffect`] component, which contains some precomputed
///   data and allocated resources, in particular GPU resources.
/// - A [`Visibility`] component (and the components it requires in turn) to
///   make use of the Bevy visibility system and optimize rendering of the
///   effects. This influences simulation when using
///   [`SimulationCondition::WhenVisible`].
/// - A [`Transform`] component to define the position of the particle emitter.
///
/// ## Optional components
///
/// - The [`EffectMaterial`] defines the "material" of the particle effect,
///   which contains for example the textures used by the particle system. This
///   component is optional, because not all effects make use of a material.
/// - The [`EffectParent`] defines the parent effect of this effect. This is
///   used for hierarchical effect construction, and to use GPU spawn events to
///   make the parent effect trigger spawning particles in this effect.
/// - The [`EffectProperties`] defines the runtime values of the properties of
///   the effect. If the effect doesn't use properties, this component is not
///   used.
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
#[require(
    CompiledParticleEffect,
    Transform,
    Visibility,
    VisibilityClass,
    SyncToRenderWorld
)]
#[component(on_add = bevy::camera::visibility::add_visibility_class::<EffectVisibilityClass>)]
pub struct ParticleEffect {
    /// Handle of the effect to instantiate.
    pub handle: Handle<EffectAsset>,
    /// Optional per-instance PRNG seed.
    ///
    /// Set this value to `Some(seed)` to override the default value set in
    /// [`EffectAsset::prng_seed`] for this effect instance. By default this is
    /// `None`, and the instance uses the same PRNG seed as its [`EffectAsset`].
    pub prng_seed: Option<u32>,
}

impl ParticleEffect {
    /// Create a new particle effect instance from an existing asset.
    pub fn new(handle: Handle<EffectAsset>) -> Self {
        Self {
            handle,
            prng_seed: None,
        }
    }
}

impl AsAssetId for ParticleEffect {
    type Asset = EffectAsset;

    fn as_asset_id(&self) -> AssetId<Self::Asset> {
        self.handle.id()
    }
}

/// Material for an effect instance.
///
/// A material component contains the render resources (textures) for a single
/// effect instance. Those textures are automatically bound during rendering to
/// the slots defined with [`Module::add_texture_slot()`]. Using this, multiple
/// effect instances sharing a same source [`EffectAsset`] can be instantiated
/// and rendered with different sets of textures, without changing the asset.
///
/// The [`EffectMaterial`] component needs to be spawned on the same entity as
/// the [`ParticleEffect`] component representing the effect instance.
#[derive(Debug, Default, Clone, Component)]
pub struct EffectMaterial {
    /// List of texture images to use to render the effect instance.
    ///
    /// The images are ordered by [slot index] into the corresponding
    /// [`TextureLayout`].
    ///
    /// [slot index]: crate::TextureLayout::get_slot_by_name
    pub images: Vec<Handle<Image>>,
}

/// Texture slot of a [`Module`].
///
/// A texture slot defines a named bind point where a texture can be attached
/// and sampled by an effect during rendering. A slot also has an implicit
/// unique index corresponding to its position in the [`TextureLayout::layout`]
/// array of the effect.
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
    ///
    /// The slots index corresponds to the index inside this array. Each image
    /// in [`EffectMaterial::images`] maps by index to a unique slot in this
    /// array.
    pub layout: Vec<TextureSlot>,
}

impl TextureLayout {
    /// Find a texture slot by name, and return its index.
    ///
    /// The index uniquely identify the slot (like its name), and indexes into
    /// the [`EffectMaterial::images`] array to determine which texture is bound
    /// to it.
    pub fn get_slot_by_name(&self, name: &str) -> Option<usize> {
        self.layout.iter().position(|slot| slot.name == name)
    }
}

/// Effect shaders.
///
/// Contains the configured shaders for the init, update, and render passes.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct EffectShader {
    pub init: Handle<Shader>,
    pub update: Handle<Shader>,
    pub render: Handle<Shader>,
}

/// Source code (WGSL) of an effect.
///
/// The source code is generated from an [`EffectAsset`] by applying all
/// modifiers. The resulting source code is _configured_ (the Hanabi variables
/// `{{VARIABLE}}` are replaced with the relevant WGSL code) but is not
/// _specialized_ (the conditional directives like `#if` are still present).
#[derive(Debug)]
struct EffectShaderSource {
    pub init_shader_source: String,
    pub update_shader_source: String,
    pub render_shader_source: String,
    pub layout_flags: LayoutFlags,
}

/// Error resulting from the generating of the WGSL shader code of an
/// [`EffectAsset`].
#[derive(Debug, Error)]
pub enum ShaderGenerateError {
    /// Error related to an [`Expr`].
    #[error("Expression error: {0}")]
    Expr(ExprError),

    /// Shader validation error.
    #[error("Validation error: {0}")]
    Validate(String),
}

impl EffectShaderSource {
    /// Generate the effect shader WGSL source code.
    ///
    /// This takes a base asset effect and generate the WGSL code for the
    /// various shaders (init/update/render).
    pub fn generate(
        asset: &EffectAsset,
        // Note: ideally the below fields are folded into the asset, but currently the parent
        // relationship and GPU event one are not encoded in assets.
        parent_layout: Option<&ParticleLayout>,
        num_event_bindings: u32,
    ) -> Result<EffectShaderSource, ShaderGenerateError> {
        trace!(
            "Generating shader sources for asset '{}' with {} event bindings",
            asset.name,
            num_event_bindings
        );

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
        if let Some(parent_layout) = parent_layout {
            if parent_layout.size() == 0 {
                return Err(ShaderGenerateError::Validate(format!(
                    "Effect using asset {} has invalid empty parent particle layout.",
                    asset.name
                )));
            }
        }

        // Currently the POSITION attribute is mandatory, as it's always used by the
        // render shader.
        if !particle_layout.contains(Attribute::POSITION) {
            return Err(ShaderGenerateError::Validate(format!(
                "The particle layout of asset '{}' is missing the '{}' attribute. Add a modifier using that attribute, for example the SetAttributeModifier.",
                asset.name, Attribute::POSITION.name().to_ascii_uppercase()
            )));
        }

        // Currently ribbon rendering requires AGE, so warn if it's missing because
        // everything will break with some weird error aboud bind groups or whatnot.
        if particle_layout.contains(Attribute::RIBBON_ID)
            && !particle_layout.contains(Attribute::AGE)
        {
            return Err(ShaderGenerateError::Validate(format!(
                "The particle layout of asset '{}' uses ribbons (has the '{}' attribute), but is missing the '{}' attribute, which is mandatory for ribbons. Add a modifier using that attribute, for example the SetAttributeModifier.",
                asset.name, Attribute::RIBBON_ID.name().to_ascii_uppercase(), Attribute::AGE.name().to_ascii_uppercase()
            )));
        }

        // Generate the WGSL code declaring all the attributes inside the Particle
        // struct.
        let attributes_code = particle_layout.generate_code();
        let parent_attributes_code = parent_layout
            .map(|layout| layout.generate_code())
            .unwrap_or_default();

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
                        "var size = vec3<f32>(particle.{0}, particle.{0}, particle.{0});\n",
                        Attribute::SIZE.name()
                    );
                    has_size = true;
                    present_attributes.insert(attr);
                } else {
                    warn!("Attribute SIZE conflicts with another size attribute; ignored.");
                }
            } else if attr == Attribute::SIZE2 {
                if !has_size {
                    inputs_code += &format!(
                        "var size = vec3<f32>(particle.{0}, 1.0);\n",
                        Attribute::SIZE2.name()
                    );
                    has_size = true;
                    present_attributes.insert(attr);
                } else {
                    warn!("Attribute SIZE2 conflicts with another size attribute; ignored.");
                }
            } else if attr == Attribute::SIZE3 {
                if !has_size {
                    inputs_code += &format!("var size = particle.{0};\n", Attribute::SIZE3.name());
                    has_size = true;
                    present_attributes.insert(attr);
                } else {
                    warn!("Attribute SIZE3 conflicts with another size attribute; ignored.");
                }
            } else if attr == Attribute::HDR_COLOR {
                if !has_color {
                    inputs_code +=
                        &format!("var color = particle.{};\n", Attribute::HDR_COLOR.name());
                    has_color = true;
                    present_attributes.insert(attr);
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
                    present_attributes.insert(attr);
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
                Attribute::SIZE3.default_value().to_wgsl_string() // TODO - or SIZE?
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
        let properties_code = property_layout.generate_code().unwrap_or_default();
        let properties_binding_code = if property_layout.is_empty() {
            "// (no properties)".to_string()
        } else {
            "@group(2) @binding(1) var<storage, read> properties : Properties;".to_string()
        };

        // Event buffer bindings for the update pass, if the effect emits GPU events to
        // one or more other effects.
        let mut emit_event_buffer_bindings_code = String::with_capacity(256);
        emit_event_buffer_bindings_code.push_str(
            "@group(3) @binding(1) var<storage, read_write> child_info_buffer : ChildInfoBuffer;\n",
        );
        let mut emit_event_buffer_append_funcs_code = String::with_capacity(1024);
        let base_binding_index = 2;
        for i in 0..num_event_bindings {
            let binding_index = base_binding_index + i;
            emit_event_buffer_bindings_code.push_str(&format!(
                "@group(3) @binding({binding_index}) var<storage, read_write> event_buffer_{i} : EventBuffer;\n"));
            emit_event_buffer_append_funcs_code.push_str(&format!(
                r##"/// Append one or more spawn events to the event buffer.
fn append_spawn_events_{0}(base_child_index: u32, particle_index: u32, count: u32) {{
    // Optimize this case.
    if (count == 0u) {{
        return;
    }}

    let capacity = arrayLength(&event_buffer_{0}.spawn_events);
    let base = min(u32(atomicAdd(&child_info_buffer.rows[base_child_index + {0}].event_count, i32(count))), capacity);
    let capped_count = min(count, capacity - base);
    for (var i = 0u; i < capped_count; i += 1u) {{
        event_buffer_{0}.spawn_events[base + i].particle_index = particle_index;
    }}
}}
"##,
                i,
            ));
        }
        emit_event_buffer_bindings_code.pop();
        if emit_event_buffer_bindings_code.is_empty() {
            emit_event_buffer_bindings_code = "// (not emitting GPU events)".into();
        }
        emit_event_buffer_append_funcs_code.pop();
        if emit_event_buffer_append_funcs_code.is_empty() {
            emit_event_buffer_append_funcs_code = "// (not emitting GPU events)".into();
        }

        // Start from the base module containing the expressions actually serialized in
        // the asset. We will add the ones created on-the-fly by applying the
        // modifiers to the contexts.
        let mut module = asset.module().clone();

        let mut layout_flags = LayoutFlags::NONE;
        if asset.simulation_space == SimulationSpace::Local {
            layout_flags |= LayoutFlags::LOCAL_SPACE_SIMULATION;
        }
        match &asset.alpha_mode {
            AlphaMode::Mask(_) => layout_flags.insert(LayoutFlags::USE_ALPHA_MASK),
            AlphaMode::Opaque => layout_flags.insert(LayoutFlags::OPAQUE),
            _ => layout_flags.remove(LayoutFlags::USE_ALPHA_MASK | LayoutFlags::OPAQUE),
        }
        if particle_layout.contains(Attribute::RIBBON_ID) {
            layout_flags |= LayoutFlags::RIBBONS;
        }
        if parent_layout.is_some() {
            layout_flags |= LayoutFlags::READ_PARENT_PARTICLE;
        }

        // Generate the shader code for the initializing shader
        let (init_code, init_extra, init_sim_space_transform_code, consume_gpu_spawn_events) = {
            // Apply all the init modifiers
            let mut init_context =
                ShaderWriter::new(ModifierContext::Init, &property_layout, &particle_layout);
            for m in asset.init_modifiers() {
                if let Err(err) = m.apply(&mut module, &mut init_context) {
                    error!(
                        "Failed to compile effect '{}', error in init context: {}",
                        asset.name, err
                    );
                    return Err(ShaderGenerateError::Expr(err));
                }
            }

            let sim_space_transform_code =
                asset.simulation_space.eval(&init_context).map_err(|err| {
                    error!("Failed to compile effect's simulation space: {}", err);
                    ShaderGenerateError::Expr(err)
                })?;

            // Effect uses GPU spawn events if it produces them, or if it has a parent and
            // consumes them from that parent.
            let consume_gpu_spawn_events =
                init_context.emits_gpu_spawn_events().unwrap_or(false) || parent_layout.is_some();

            (
                init_context.main_code,
                init_context.extra_code,
                sim_space_transform_code,
                consume_gpu_spawn_events,
            )
        };

        let init_shader_source = PARTICLES_INIT_SHADER_TEMPLATE
            .replace("{{ATTRIBUTES}}", &attributes_code)
            .replace("{{PARENT_ATTRIBUTES}}", &parent_attributes_code)
            .replace("{{INIT_CODE}}", &init_code)
            .replace("{{INIT_EXTRA}}", &init_extra)
            .replace("{{PROPERTIES}}", &properties_code)
            .replace("{{PROPERTIES_BINDING}}", &properties_binding_code)
            .replace(
                "{{SIMULATION_SPACE_TRANSFORM_PARTICLE}}",
                &init_sim_space_transform_code,
            );
        trace!(
            "Configured init shader for '{}':\n{}",
            asset.name,
            init_shader_source
        );

        // Generate the shader code for the update shader
        let (mut update_code, update_extra, emit_gpu_spawn_events) = {
            let mut update_context =
                ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
            for m in asset.update_modifiers() {
                if let Err(err) = m.apply(&mut module, &mut update_context) {
                    error!(
                        "Failed to compile effect '{}', error in update context: {}",
                        asset.name, err
                    );
                    return Err(ShaderGenerateError::Expr(err));
                }
            }

            let emit_gpu_spawn_events = update_context.emits_gpu_spawn_events().unwrap_or(false);

            (
                update_context.main_code,
                update_context.extra_code,
                emit_gpu_spawn_events,
            )
        };

        if consume_gpu_spawn_events {
            layout_flags |= LayoutFlags::CONSUME_GPU_SPAWN_EVENTS;
        }
        if emit_gpu_spawn_events {
            layout_flags |= LayoutFlags::EMIT_GPU_SPAWN_EVENTS;
        }

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
                        "Asset '{}' specifies motion integration but is missing {}. Particles won't move unless the POSITION attribute is explicitly assigned. Set MotionIntegration::None to remove this warning.",
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
            for m in asset.render_modifiers() {
                m.apply_render(&mut module, &mut render_context)
                    .map_err(ShaderGenerateError::Expr)?;
            }

            if render_context.needs_uv {
                layout_flags |= LayoutFlags::NEEDS_UV;
            }
            if render_context.needs_normal {
                layout_flags |= LayoutFlags::NEEDS_NORMAL;
            }
            if render_context.needs_particle_fragment {
                layout_flags |= LayoutFlags::NEEDS_PARTICLE_FRAGMENT;
            }

            let alpha_cutoff_code = if let AlphaMode::Mask(cutoff) = &asset.alpha_mode {
                render_context.eval(&module, *cutoff).unwrap_or_else(|err| {
                    error!(
                        "Failed to evaluate the expression for AlphaMode::Mask, error: {}",
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

            let (flipbook_scale_code, flipbook_row_count_code) = if let Some(grid_size) =
                render_context.sprite_grid_size
            {
                layout_flags |= LayoutFlags::FLIPBOOK;
                // Note: row_count needs to be i32, not u32, because of sprite_index
                let flipbook_row_count_code = (grid_size.x as i32).to_wgsl_string();
                let flipbook_scale_code =
                    Vec2::new(1.0 / grid_size.x as f32, 1.0 / grid_size.y as f32).to_wgsl_string();
                (flipbook_scale_code, flipbook_row_count_code)
            } else {
                (String::new(), String::new())
            };

            trace!(
                "Generating material bindings code for layout: {:?}",
                texture_layout
            );
            let mut material_bindings_code = String::new();
            let mut bind_index = 0;
            for (slot, _) in texture_layout.layout.iter().enumerate() {
                let tex_index = bind_index;
                let sampler_index = bind_index + 1;
                material_bindings_code.push_str(&format!(
                    "@group(2) @binding({tex_index}) var material_texture_{slot}: texture_2d<f32>;
@group(2) @binding({sampler_index}) var material_sampler_{slot}: sampler;
"
                ));
                bind_index += 2;
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
        let mut age_code = String::new();
        if has_age {
            if has_lifetime {
                age_code += &format!(
                    "\n    let was_alive = particle.{0} < particle.{1};",
                    Attribute::AGE.name(),
                    Attribute::LIFETIME.name()
                );
            }

            age_code += &format!(
                "\n    particle.{0} = particle.{0} + sim_params.delta_time;",
                Attribute::AGE.name()
            );

            if has_lifetime {
                age_code += &format!(
                    "\n    var is_alive = particle.{0} < particle.{1};",
                    Attribute::AGE.name(),
                    Attribute::LIFETIME.name()
                );
            }
        } else {
            // Since we're using a dead index buffer, all particles that make it to the
            // update compute shader are guaranteed to be alive (we never
            // simulate dead particles).
            age_code = "\n    let was_alive = true;\n    var is_alive = true;".to_string();
        }

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
        for attribute in present_attributes
            .iter()
            .filter(|attribute| **attribute != Attribute::PREV && **attribute != Attribute::NEXT)
        {
            writeln!(
                &mut writeback_code,
                "    particle_buffer.particles[base_particle + particle_index].{0} = particle.{0};",
                attribute.name()
            )
            .unwrap();
        }

        // Configure the update shader template, and make sure a corresponding shader
        // asset exists
        let update_shader_source = PARTICLES_UPDATE_SHADER_TEMPLATE
            .replace("{{ATTRIBUTES}}", &attributes_code)
            .replace("{{PARENT_ATTRIBUTES}}", &parent_attributes_code)
            .replace("{{AGE_CODE}}", &age_code)
            .replace("{{REAP_CODE}}", &reap_code)
            .replace("{{UPDATE_CODE}}", &update_code)
            .replace("{{WRITEBACK_CODE}}", &writeback_code)
            .replace("{{UPDATE_EXTRA}}", &update_extra)
            .replace("{{PROPERTIES}}", &properties_code)
            .replace("{{PROPERTIES_BINDING}}", &properties_binding_code)
            .replace(
                "{{EMIT_EVENT_BUFFER_BINDINGS}}",
                &emit_event_buffer_bindings_code,
            )
            .replace(
                "{{EMIT_EVENT_BUFFER_APPEND_FUNCS}}",
                &emit_event_buffer_append_funcs_code,
            );
        trace!(
            "Configured update shader for '{}':\n{}",
            asset.name,
            update_shader_source
        );

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
        trace!(
            "Configured render shader for '{}':\n{}",
            asset.name,
            render_shader_source
        );

        Ok(EffectShaderSource {
            init_shader_source,
            update_shader_source,
            render_shader_source,
            layout_flags,
        })
    }
}

/// Compiled data for a [`ParticleEffect`].
///
/// This component is managed automatically, and should not be accessed
/// manually. It contains data generated from the associated [`ParticleEffect`]
/// component located on the same [`Entity`]. The data is split into this
/// component for change detection reasons, and any change to the associated
/// [`ParticleEffect`] will cause the values of this component to be
/// recalculated. Otherwise the data is cached frame-to-frame for performance.
///
/// All [`ParticleEffect`]s are compiled by the system running in the
/// [`EffectSystems::CompileEffects`] set every frame when they're spawned or
/// when they change, irrelevant of whether the entity if visible
/// ([`Visibility::Visible`]).
#[derive(Debug, Clone, Component)]
pub struct CompiledParticleEffect {
    /// Handle to the underlying asset.
    asset: Handle<EffectAsset>,
    /// Parent effect, if any.
    parent: Option<Entity>,
    /// Child effects.
    children: Vec<Entity>,
    /// Cached simulation condition, to avoid having to query the asset each
    /// time we need it.
    simulation_condition: SimulationCondition,
    /// A custom mesh for this effect, if specified.
    mesh: Option<Handle<Mesh>>,
    /// Handle to the effect shaders for his effect instance, if configured.
    effect_shader: Option<EffectShader>,
    /// Textures used by the effect, if any.
    textures: Vec<Handle<Image>>,
    /// Layout flags.
    layout_flags: LayoutFlags,
    /// Alpha mode.
    alpha_mode: AlphaMode,
    /// Particle layout of the parent effect, if any.
    parent_particle_layout: Option<ParticleLayout>,
    /// PRNG seed.
    prng_seed: u32,
    /// Ready state reported by the render world.
    is_ready: bool,
}

impl Default for CompiledParticleEffect {
    fn default() -> Self {
        Self {
            asset: default(),
            parent: None,
            children: vec![],
            simulation_condition: SimulationCondition::default(),
            mesh: None,
            effect_shader: None,
            textures: vec![],
            layout_flags: LayoutFlags::NONE,
            alpha_mode: default(),
            parent_particle_layout: None,
            prng_seed: 0,
            is_ready: false,
        }
    }
}

impl CompiledParticleEffect {
    /// Check if the effect is ready.
    #[inline(always)]
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }

    #[cfg(test)]
    pub(crate) fn with_ready_for_tests(mut self) -> Self {
        self.is_ready = true;
        self
    }

    /// Clear the compiled data from this component.
    pub(crate) fn clear(&mut self) {
        self.asset = Handle::default();
        self.effect_shader = None;
        self.textures.clear();
    }

    /// Update the compiled effect from its asset and instance.
    pub(crate) fn update(
        &mut self,
        rebuild: bool,
        instance: &ParticleEffect,
        material: Option<&EffectMaterial>,
        asset: &EffectAsset,
        parent_entity: Option<Entity>,
        child_entities: Vec<Entity>,
        parent_layout: Option<ParticleLayout>,
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
        self.prng_seed = instance.prng_seed.unwrap_or(asset.prng_seed);

        // Check if the instance changed. If so, rebuild some data from this compiled
        // effect based on the new data of the effect instance.
        if rebuild {
            // Clear the compiled effect if the effect instance changed. We could try to get
            // smarter here, only invalidate what changed, but for now just wipe everything
            // and rebuild from scratch all three shaders together.
            self.effect_shader = None;
        }

        // If the shaders are already compiled, there's nothing more to do
        if self.effect_shader.is_some() {
            return;
        }

        self.parent = parent_entity;
        self.children = child_entities;

        let num_event_bindings = self.children.len() as u32;
        let shader_source =
            match EffectShaderSource::generate(asset, parent_layout.as_ref(), num_event_bindings) {
                Ok(shader_source) => shader_source,
                Err(err) => {
                    error!(
                        "Failed to generate shaders for effect asset '{}': {}",
                        asset.name, err
                    );
                    return;
                }
            };

        self.layout_flags = shader_source.layout_flags;
        self.alpha_mode = asset.alpha_mode;
        self.parent_particle_layout = parent_layout;
        trace!(
            "Compiled effect sources: layout_flags={:?} alpha_mode={:?}",
            self.layout_flags,
            self.alpha_mode
        );

        // TODO - Replace with Option<EffectShader { handle: Handle<Shader>, hash:
        // u64 }> where the hash takes into account the code and extra code
        // for each pass (and any other varying item). We don't need to keep
        // around the entire shader code, only a hash of it for compare (or, maybe safer
        // to avoid hash collisions, an index into a shader cache). The only
        // use is to be able to compare 2 instances and see if they can be
        // batched together.
        self.effect_shader = Some(EffectShader {
            init: shader_cache.get_or_insert(
                &asset.name,
                "init",
                &shader_source.init_shader_source,
                shaders,
            ),
            update: shader_cache.get_or_insert(
                &asset.name,
                "update",
                &shader_source.update_shader_source,
                shaders,
            ),
            render: shader_cache.get_or_insert(
                &asset.name,
                "render",
                &shader_source.render_shader_source,
                shaders,
            ),
        });

        trace!(
            "CompiledParticleEffect::update(): shaders={:?} texture_count={} layout_flags={:?}",
            self.effect_shader.as_ref().unwrap(),
            material.map(|mat| mat.images.len()).unwrap_or(0),
            self.layout_flags,
        );

        self.mesh = asset.mesh.clone();

        self.textures = material.map(|mat| &mat.images).cloned().unwrap_or_default();
    }

    /// Get the effect shader if configured, or `None` otherwise.
    pub(crate) fn get_configured_shaders(&self) -> Option<&EffectShader> {
        self.effect_shader.as_ref()
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

impl ShaderCode for Gradient<Vec3> {
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
        Option<Ref<EffectParent>>,
        &mut CompiledParticleEffect,
    )>,
) {
    trace!("compile_effects: {} effect(s)", q_effects.iter().len());

    // Loop over all existing effects and collect the valid ones. We do a separate
    // pass because we can't borrow mutably while doing a double lookup on the
    // query. This map is used to lookup valid parents, and filter out effects with
    // a declared parent but unresolved parent asset.
    let particle_layouts_and_parents: HashMap<Entity, (ParticleLayout, Option<Entity>)> = q_effects
        .iter()
        .filter_map(|(entity, effect, _, parent, _)| {
            effects
                .get(&effect.handle)
                .map(|asset| (entity, (asset.particle_layout(), parent.map(|p| p.entity))))
        })
        .collect();

    // Count children
    let mut children: HashMap<Entity, Vec<Entity>> =
        HashMap::with_capacity_and_hasher(particle_layouts_and_parents.len(), Default::default());
    for (child, (_, parent)) in particle_layouts_and_parents.iter() {
        if let Some(parent) = parent.as_ref() {
            children.entry(*parent).or_default().push(*child);
        }
    }

    // Loop over all existing effects to update them, including invisible ones
    for (asset, entity, effect, material, parent_entity, parent_layout, mut compiled_effect) in
        q_effects
            .iter_mut()
            .filter_map(|(entity, effect, material, parent, compiled_effect)| {
                // Check if asset is available, otherwise silently ignore as we can't check for
                // changes, and conceptually it makes no sense to render a particle effect whose
                // asset was unloaded.
                let asset = effects.get(&effect.handle)?;

                // Same for the parent asset, if any.
                let (parent_entity, parent_layout) = if let Some(parent) = &parent {
                    let Some((parent_layout, _)) = particle_layouts_and_parents.get(&parent.entity)
                    else {
                        // There's a parent declared, but not found. Skip the current asset.
                        return None;
                    };
                    // Declared parent with found parent asset, child asset is valid.
                    (Some(parent.entity), Some(parent_layout.clone()))
                } else {
                    // No declared parent, asset is valid.
                    (None, None)
                };

                Some((
                    asset,
                    entity,
                    effect,
                    material,
                    parent_entity,
                    parent_layout,
                    compiled_effect,
                ))
            })
    {
        let child_entities = children
            .get_mut(&entity)
            .map(std::mem::take)
            .unwrap_or_default();

        // If the ParticleEffect didn't change, and the compiled one is for the correct
        // asset, then there's nothing to do.
        let material_changed = material.as_ref().is_some_and(|r| r.is_changed());
        let need_rebuild =
            effect.is_changed() || material_changed || compiled_effect.children != child_entities;
        if need_rebuild || (compiled_effect.asset != effect.handle) {
            if need_rebuild {
                debug!("Invalidating the compiled cache for effect on entity {:?} due to changes in the ParticleEffect component. If you see this message too much, then performance might be affected. Find why the change detection of the ParticleEffect is triggered.", entity);
            }

            compiled_effect.update(
                need_rebuild,
                &effect,
                material.map(|r| r.into_inner()),
                asset,
                parent_entity,
                child_entities,
                parent_layout,
                &mut shaders,
                &mut shader_cache,
            );
        } else {
            // Update the PRNG seed. Unfortunately at the minute the "seed" (which
            // really is the internal PRNG state rather) is not cached on GPU, and
            // is re-uploaded each frame, so if it's not changed every frame then
            // there's no randomness anymore, because the uses of the previous frame
            // are "forgotten".
            let mut rng = rand::rngs::StdRng::seed_from_u64(compiled_effect.prng_seed as u64);
            compiled_effect.prng_seed = rng.random();
        }
    }

    // Clear removed effects, to allow them to be released by the asset server
    for (_, effect, _, parent, mut compiled_effect) in q_effects.iter_mut() {
        // If the effect has no asset, clear its compilation
        if effects.get(&effect.handle).is_none() {
            compiled_effect.clear();
        }

        // If the effect has a parent, and that parent has no asset, also clear the
        // child's compilation.
        if let Some(parent) = parent {
            if particle_layouts_and_parents.get(&parent.entity).is_none() {
                compiled_effect.clear();
            }
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
    #[cfg(feature = "trace")]
    let _span = bevy::log::info_span!("update_properties_from_asset").entered();
    trace!("update_properties_from_asset()");

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

#[cfg(test)]
mod tests {
    use std::ops::DerefMut;

    use bevy::{
        asset::{
            io::{
                memory::{Dir, MemoryAssetReader},
                AssetSourceBuilder, AssetSourceBuilders, AssetSourceId,
            },
            AssetServerMode, LoadState, UnapprovedPathMode,
        },
        camera::visibility::{VisibilityPlugin, VisibilitySystems},
        shader::ShaderLoader,
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
            assert_eq!(size, size.next_multiple_of(1));
        }

        // zero-sized is always aligned
        for &align in INTS_POW2 {
            assert_eq!(0, 0usize.next_multiple_of(align));
        }

        // size < align : rounds up to align
        for &size in INTS {
            assert_eq!(256, size.next_multiple_of(256));
        }

        // size > align : actually aligns
        for (&size, &aligned_size) in INTS.iter().zip(INTS16) {
            assert_eq!(aligned_size, size.next_multiple_of(16));
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
        let dummy_builder =
            AssetSourceBuilder::new(move || Box::new(MemoryAssetReader { root: dir.clone() }));
        builders.insert(AssetSourceId::Default, dummy_builder);
        let sources = builders.build_sources(watch_for_changes, false);
        let asset_server = AssetServer::new(
            sources.into(),
            AssetServerMode::Unprocessed,
            watch_for_changes,
            UnapprovedPathMode::Forbid,
        );

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
        let asset = EffectAsset::new(256, SpawnerSettings::rate(32.0.into()), module)
            .with_simulation_space(SimulationSpace::Local);
        assert_eq!(asset.simulation_space, SimulationSpace::Local);
        let res = EffectShaderSource::generate(&asset, None, 0);
        assert!(res.is_err());
        let err = res.err().unwrap();
        assert!(matches!(err, ShaderGenerateError::Validate(_)));

        // Missing Attribute::POSITION, currently mandatory for all effects
        let mut module = Module::default();
        let zero = module.lit(Vec3::ZERO);
        let asset = EffectAsset::new(256, SpawnerSettings::rate(32.0.into()), module)
            .init(SetAttributeModifier::new(Attribute::VELOCITY, zero));
        assert!(asset.particle_layout().size() > 0);
        let res = EffectShaderSource::generate(&asset, None, 0);
        assert!(res.is_err());
        let err = res.err().unwrap();
        assert!(matches!(err, ShaderGenerateError::Validate(_)));

        // Valid
        let mut module = Module::default();
        let zero = module.lit(Vec3::ZERO);
        let asset = EffectAsset::new(256, SpawnerSettings::rate(32.0.into()), module)
            .with_simulation_space(SimulationSpace::Local)
            .init(SetAttributeModifier::new(Attribute::POSITION, zero));
        assert_eq!(asset.simulation_space, SimulationSpace::Local);
        let res = EffectShaderSource::generate(&asset, None, 0);
        assert!(res.is_ok());
        let shader_source = res.unwrap();
        for (name, code) in [
            ("Init", shader_source.init_shader_source),
            ("Update", shader_source.update_shader_source),
            ("Render", shader_source.render_shader_source),
        ] {
            println!("{} shader:\n\n{}", name, code);

            let mut shader_defs = std::collections::HashMap::<String, ShaderDefValue>::new();
            shader_defs.insert("LOCAL_SPACE_SIMULATION".into(), ShaderDefValue::Bool(true));
            shader_defs.insert("NEEDS_UV".into(), ShaderDefValue::Bool(true));
            shader_defs.insert("NEEDS_NORMAL".into(), ShaderDefValue::Bool(false));
            shader_defs.insert(
                "NEEDS_PARTICLE_FRAGMENT".into(),
                ShaderDefValue::Bool(false),
            );
            shader_defs.insert(
                "PARTICLE_SCREEN_SPACE_SIZE".into(),
                ShaderDefValue::Bool(true),
            );
            if name == "Update" {
                shader_defs.insert("EM_MAX_SPAWN_ATOMIC".into(), ShaderDefValue::Bool(true));
            }
            let mut composer = Composer::default();

            // Import bevy_render::view for the render shader
            {
                // It's reasonably hard to retrieve the source code for view.wgsl in
                // bevy_render. We use a few tricks to get a Shader that we can
                // then convert into a composable module (which is how imports work in Bevy
                // itself).
                IoTaskPool::get_or_init(|| {
                    TaskPoolBuilder::default()
                        .num_threads(1)
                        .thread_name("Hanabi test IO Task Pool".to_string())
                        .build()
                });
                let mut dummy_app = App::new();
                dummy_app.add_plugins(bevy::asset::AssetPlugin::default());
                dummy_app
                    .init_asset::<Shader>()
                    .init_asset_loader::<ShaderLoader>();
                dummy_app.add_plugins(bevy::render::view::ViewPlugin);
                let asset_server = dummy_app.world().resource::<AssetServer>();
                let view_shader_handle =
                    asset_server.load::<Shader>("embedded://bevy_render/view/view.wgsl");

                // Need at least one frame tick for the loaded asset to send a message to the
                // asset server to get registered
                let mut max_frames = 10000; // it takes a decent amount of time to load async the asset, even if embedded
                while max_frames > 0 {
                    dummy_app.update();

                    let asset_server = dummy_app.world().resource::<AssetServer>();
                    let load_state = asset_server.get_load_state(&view_shader_handle).unwrap();
                    if let LoadState::Failed(err) = load_state {
                        panic!("Load failed: {:?}", err);
                    }
                    if matches!(load_state, LoadState::Loaded) {
                        break;
                    }

                    max_frames -= 1;
                }
                assert!(max_frames > 0);

                let shaders = dummy_app.world().get_resource::<Assets<Shader>>().unwrap();
                for (id, shader) in shaders.iter() {
                    println!("[{id:?}] {shader:?}");
                }
                let view_shader = shaders.get(&view_shader_handle).unwrap();

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
                source: &code[..],
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
            let entity = world.spawn(ParticleEffect::default()).id();

            // Spawn a camera, otherwise ComputedVisibility stays at HIDDEN
            world.spawn(Camera3d::default());

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
            assert!(compiled_particle_effect.effect_shader.is_none());
        }
    }

    // Regression test for #228
    #[test]
    fn test_compile_effect_changed() {
        let spawner = SpawnerSettings::once(32.0.into());

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
            world.spawn(Camera3d::default());

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
            assert!(compiled_particle_effect.effect_shader.is_some());
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
            assert!(compiled_particle_effect.effect_shader.is_some());
        }
    }

    #[test]
    fn test_compile_effect_visibility() {
        let spawner = SpawnerSettings::once(32.0.into());

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
                world.spawn(Camera3d::default());

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
                assert!(compiled_particle_effect.effect_shader.is_some());

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
                assert!(compiled_particle_effect.effect_shader.is_some());
            }
        }
    }
}
