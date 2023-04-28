#![deny(
    warnings,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications,
    missing_docs,
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
//! _Note: Because it uses compute shaders, this library is incompatible with
//! wasm. This is a limitation of webgpu/wasm._
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
//! bevy_hanabi = { version = "0.5", default-features = false, features = ["3d"] }
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
//!     .add_plugin(HanabiPlugin)
//!     .run();
//! ```
//!
//! Create an [`EffectAsset`] describing a visual effect:
//!
//! ```
//! # use bevy::prelude::*;
//! # use bevy_hanabi::prelude::*;
//! fn create_asset(mut effects: ResMut<Assets<EffectAsset>>) {
//!     // Define a color gradient from red to transparent black
//!     let mut gradient = Gradient::new();
//!     gradient.add_key(0.0, Vec4::new(1., 0., 0., 1.));
//!     gradient.add_key(1.0, Vec4::splat(0.));
//!
//!     // Create the effect asset
//!     let effect = effects.add(EffectAsset {
//!         name: "MyEffect".to_string(),
//!         // Maximum number of particles alive at a time
//!         capacity: 32768,
//!         // Spawn at a rate of 5 particles per second
//!         spawner: Spawner::rate(5.0.into()),
//!         ..Default::default()
//!     }
//!     // On spawn, randomly initialize the position of the particle
//!     // to be over the surface of a sphere of radius 2 units.
//!     .init(InitPositionSphereModifier {
//!         center: Vec3::ZERO,
//!         radius: 2.,
//!         dimension: ShapeDimension::Surface,
//!     })
//!     // Also initialize a radial initial velocity to 6 units/sec
//!     // away from the (same) sphere center.
//!     .init(InitVelocitySphereModifier {
//!         center: Vec3::ZERO,
//!         speed: 6.0.into(),
//!     })
//!     // Also initialize the total lifetime of the particle, that is
//!     // the time for which it's simulated and rendered. This modifier
//!     // is mandatory, otherwise the particles won't show up.
//!     .init(InitLifetimeModifier { lifetime: 10_f32.into() })
//!     // Every frame, add a gravity-like acceleration downward
//!     .update(AccelModifier::constant(Vec3::new(0., -3., 0.)))
//!     // Render the particles with a color gradient over their
//!     // lifetime. This maps the gradient key 0 to the particle spawn
//!     // time, and the gradient key 1 to the particle death (here, 10s).
//!     .render(ColorOverLifetimeModifier { gradient })
//!     );
//!
//!     // [...]
//! }
//! ```
//!
//! Then add an instance of that effect to an entity by spawning a
//! [`ParticleEffect`] component referencing the asset:
//!
//! ```
//! # use bevy::{prelude::*, asset::HandleId};
//! # use bevy_hanabi::prelude::*;
//! # fn spawn_effect(mut commands: Commands) {
//! #   let effect = Handle::weak(HandleId::random::<EffectAsset>());
//! commands
//!     .spawn((
//!         Name::new("MyEffectInstance"),
//!         ParticleEffectBundle {
//!             effect: ParticleEffect::new(effect),
//!             transform: Transform::from_translation(Vec3::new(0., 1., 0.)),
//!             ..Default::default()
//!         },
//!     ));
//! # }
//! ```

#[cfg(feature = "2d")]
use bevy::utils::FloatOrd;
use bevy::{prelude::*, utils::HashSet};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _; // import without risk of name clashing

mod asset;
mod attributes;
mod bundle;
mod gradient;
pub mod graph;
pub mod modifier;
mod plugin;
mod properties;
mod render;
mod spawn;

#[cfg(test)]
mod test_utils;

use properties::{Property, PropertyInstance};

pub use asset::{EffectAsset, MotionIntegration, SimulationCondition};
pub use attributes::*;
pub use bundle::ParticleEffectBundle;
pub use gradient::{Gradient, GradientKey};
pub use modifier::*;
pub use plugin::HanabiPlugin;
pub use properties::PropertyLayout;
pub use render::{EffectSystems, ShaderCache};
pub use spawn::{tick_spawners, DimValue, EffectSpawner, Random, Spawner, Value};

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

/// Extension trait to write a floating point scalar or vector constant in a
/// format matching the WGSL grammar.
///
/// This is required because WGSL doesn't support a floating point constant
/// without a decimal separator (_e.g._ `0.` instead of `0`), which would be
/// what a regular string formatting function like [`format!()`] would produce,
/// but which is interpreted as an integral type by WGSL instead.
///
/// # Example
///
/// ```
/// # use bevy_hanabi::ToWgslString;
/// let x = 2.0_f32;
/// assert_eq!("let x = 2.;", format!("let x = {};", x.to_wgsl_string()));
/// ```
///
/// [`format!()`]: std::format
pub trait ToWgslString {
    /// Convert a floating point scalar or vector to a string representing a
    /// WGSL constant.
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
            "vec2<f32>({0}, {1})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string()
        )
    }
}

impl ToWgslString for Vec3 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec3<f32>({0}, {1}, {2})",
            self.x.to_wgsl_string(),
            self.y.to_wgsl_string(),
            self.z.to_wgsl_string()
        )
    }
}

impl ToWgslString for Vec4 {
    fn to_wgsl_string(&self) -> String {
        format!(
            "vec4<f32>({0}, {1}, {2}, {3})",
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

impl ToWgslString for i32 {
    fn to_wgsl_string(&self) -> String {
        format!("{}", self)
    }
}

impl ToWgslString for u32 {
    fn to_wgsl_string(&self) -> String {
        format!("{}", self)
    }
}

impl ToWgslString for Value<f32> {
    fn to_wgsl_string(&self) -> String {
        match self {
            Self::Single(x) => x.to_wgsl_string(),
            Self::Uniform((a, b)) => format!(
                "(rand() * ({1} - {0}) + {0})",
                a.to_wgsl_string(),
                b.to_wgsl_string(),
            ),
        }
    }
}

impl ToWgslString for Value<Vec2> {
    fn to_wgsl_string(&self) -> String {
        match self {
            Self::Single(v) => v.to_wgsl_string(),
            Self::Uniform((a, b)) => format!(
                "(rand2() * ({1} - {0}) + {0})",
                a.to_wgsl_string(),
                b.to_wgsl_string(),
            ),
        }
    }
}

impl ToWgslString for Value<Vec3> {
    fn to_wgsl_string(&self) -> String {
        match self {
            Self::Single(v) => v.to_wgsl_string(),
            Self::Uniform((a, b)) => format!(
                "(rand3() * ({1} - {0}) + {0})",
                a.to_wgsl_string(),
                b.to_wgsl_string(),
            ),
        }
    }
}

impl ToWgslString for Value<Vec4> {
    fn to_wgsl_string(&self) -> String {
        match self {
            Self::Single(v) => v.to_wgsl_string(),
            Self::Uniform((a, b)) => format!(
                "(rand4() * ({1} - {0}) + {0})",
                a.to_wgsl_string(),
                b.to_wgsl_string(),
            ),
        }
    }
}

/// Simulation space for the particles of an effect.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Reflect, FromReflect, Serialize, Deserialize,
)]
#[non_exhaustive]
pub enum SimulationSpace {
    /// Particles are simulated in global space.
    ///
    /// The global space is the Bevy world space. Particles simulated in global
    /// space are "detached" from the emitter when they spawn, and not
    /// influenced anymore by the emitter's [`Transform`] after spawning.
    #[default]
    Global,
    // Local, // TODO
}

/// Value a user wants to assign to a property with
/// [`CompiledParticleEffect::set_property()`] before the instance had a chance
/// to inspect its underlying asset and check the asset's defined properties.
///
/// A property with this name might not exist, in which case the value will be
/// discarded silently when the instance is initialized from its asset.
#[derive(Debug, Clone, PartialEq, Reflect, FromReflect)]
pub struct PropertyValue {
    /// Name of the property the value should be assigned to.
    name: String,

    /// The property value to assign, instead of the default value of the
    /// property.
    value: graph::Value,
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
    /// which the particles are rendered present in the effect asset.
    ///
    /// This value is passed to the render pipeline and used when sorting
    /// transparent items to render, to order them. As a result, effects
    /// with different Z values cannot be batched together, which may
    /// negatively affect performance.
    ///
    /// This is only available with the `2d` feature.
    #[cfg(feature = "2d")]
    pub z_layer_2d: Option<f32>,
    /// Optional particle spawner override for this instance.
    ///
    /// If set, this overrides the spawner configured in the [`EffectAsset`].
    /// Otherwise the spawner from the effect asset will be copied here when the
    /// component is first processed.
    pub spawner: Option<Spawner>,
    /// Optional initial values for some or all of the effect properties.
    ///
    /// This contains values for some properties you want to assign when the
    /// [`CompiledParticleEffect`] is first initialized. If empty, all
    /// properties are initialized to their default value as specified when
    /// defined in the [`EffectAsset`]. Any value specified here will override
    /// the default property value.
    ///
    /// The properties will be matched by name when the effect instance is
    /// compiled into a [`CompiledParticleEffect`]. Any name not corresponding
    /// to an existing asset property will be ignored. If there's any
    /// duplicate name, the first value will be used.
    ///
    /// This is used as a convenience to avoid having to wait for the
    /// [`CompiledParticleEffect`] to effectively be initialized. The behavior
    /// is equivalent to waiting for  the [`CompiledParticleEffect`] to be
    /// initialized then calling the [`set_property()`] method.
    ///
    /// [`set_property()`]: crate::CompiledParticleEffect::set_property
    pub properties: Vec<PropertyValue>,
}

impl ParticleEffect {
    /// Create a new particle effect without a spawner or any modifier.
    pub fn new(handle: Handle<EffectAsset>) -> Self {
        Self {
            handle,
            #[cfg(feature = "2d")]
            z_layer_2d: None,
            spawner: None,
            properties: vec![],
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
    /// # use bevy::asset::{Handle, HandleId};
    /// # let asset = Handle::weak(HandleId::random::<EffectAsset>());
    /// // Always render the effect in front of the default layer (z=0)
    /// let effect = ParticleEffect::new(asset).with_z_layer_2d(Some(0.1));
    /// ```
    #[cfg(feature = "2d")]
    pub fn with_z_layer_2d(mut self, z_layer_2d: Option<f32>) -> Self {
        self.z_layer_2d = z_layer_2d;
        self
    }

    /// Set the spawner of this particle effect instance.
    ///
    /// By default particle effect instances inherit the spawner of the
    /// [`EffectAsset`] they're derived from. This allows overriding the spawner
    /// configuration per instance.
    pub fn with_spawner(mut self, spawner: Spawner) -> Self {
        self.spawner = Some(spawner);
        self
    }

    /// Set the initial value of some properties.
    ///
    /// See the [`properties`] field for more details.
    ///
    /// [`properties`]: crate::ParticleEffect::properties
    pub fn with_properties<P>(
        mut self,
        properties: impl IntoIterator<Item = (String, graph::Value)>,
    ) -> Self {
        let iter = properties.into_iter();
        for (name, value) in iter {
            if let Some(index) = self.properties.iter().position(|p| p.name == name) {
                self.properties[index].value = value;
            } else {
                self.properties.push(PropertyValue { name, value });
            }
        }
        self
    }
}

/// Compiled variant of a [`ParticleEffect`].
///
/// This component is managed automatically, and generally should not be
/// accessed manually, with the exception of setting property values via
/// [`set_property()`]. It contains data generated from the associated
/// [`ParticleEffect`] component located on the same [`Entity`]. The data is
/// split into this component in particular for change detection reasons, and
/// any change to the associated [`ParticleEffect`] will cause the values of
/// this component to be recalculated. Otherwise the data is cached
/// frame-to-frame for performance.
///
/// The component also contains the current values of all properties. Those
/// values are uploaded to the GPU each frame, to allow controling some
/// behaviors of the effect.
///
/// [`set_property()`]: crate::CompiledParticleEffect::set_property
#[derive(Debug, Clone, Component)]
pub struct CompiledParticleEffect {
    /// Weak handle to the underlying asset.
    asset: Handle<EffectAsset>,
    /// Cached simulation condition, to avoid having to query the asset each
    /// time we need it.
    simulation_condition: SimulationCondition,
    /// Handle to the configured init shader for his effect instance, if
    /// configured.
    configured_init_shader: Option<Handle<Shader>>,
    /// Handle to the configured update shader for his effect instance, if
    /// configured.
    configured_update_shader: Option<Handle<Shader>>,
    /// Handle to the configured render shader for his effect instance, if
    /// configured.
    configured_render_shader: Option<Handle<Shader>>,
    /// Instances of all exposed properties.
    properties: Vec<PropertyInstance>,
    /// Force field modifier values.
    force_field: [ForceFieldSource; ForceFieldSource::MAX_SOURCES],
    /// Main particle texture.
    particle_texture: Option<Handle<Image>>,
    /// 2D layer for the effect instance.
    #[cfg(feature = "2d")]
    z_layer_2d: FloatOrd,
}

impl Default for CompiledParticleEffect {
    fn default() -> Self {
        Self {
            asset: default(),
            simulation_condition: SimulationCondition::default(),
            configured_init_shader: None,
            configured_update_shader: None,
            configured_render_shader: None,
            properties: vec![],
            force_field: default(),
            particle_texture: None,
            #[cfg(feature = "2d")]
            z_layer_2d: FloatOrd(0.0),
        }
    }
}

impl CompiledParticleEffect {
    /// Update the compiled effect from its asset and instance.
    pub(crate) fn update(
        &mut self,
        rebuild: bool,
        properties: &[PropertyValue],
        #[cfg(feature = "2d")] z_layer_2d: FloatOrd,
        weak_handle: Handle<EffectAsset>,
        asset: &EffectAsset,
        shaders: &mut ResMut<Assets<Shader>>,
        shader_cache: &mut ResMut<ShaderCache>,
    ) {
        self.asset = weak_handle;
        self.simulation_condition = asset.simulation_condition;

        // Check if the instance changed. If so, rebuild some data from this compiled
        // effect based on the new data of the effect instance.
        if rebuild {
            // Clear the compiled effect if the effect instance changed. We could try to get
            // smarter here, only invalidate what changed, but for now just wipe everything
            // and rebuild from scratch all three shaders together.
            self.configured_init_shader = None;
            self.configured_update_shader = None;
            self.configured_render_shader = None;

            // Re-resolve all properties by looping on the properties defined in the asset
            // (which are the source of truth) and trying to map a value set by the user,
            // falling back to the property's default value if not found.
            self.properties = asset
                .properties
                .iter()
                .map(|def| PropertyInstance {
                    def: def.clone(),
                    value: properties
                        .iter()
                        .find_map(|u| {
                            // Try to find an unresolved property by name
                            if u.name == def.name() {
                                // If found, use the value specified by the user
                                Some(u.value)
                            } else {
                                // Otherwise fallback to default value from asset's property
                                None
                            }
                        })
                        .unwrap_or(*def.default_value()),
                })
                .collect();

            // Update the 2D layer
            #[cfg(feature = "2d")]
            {
                self.z_layer_2d = z_layer_2d;
            }
        }

        // If the shaders are already compiled, there's nothing more to do
        if self.configured_init_shader.is_some()
            && self.configured_update_shader.is_some()
            && self.configured_render_shader.is_some()
        {
            return;
        }

        // Generate the shader code defining the particle attributes
        let particle_layout = asset.particle_layout();
        let attributes_code = particle_layout.generate_code();

        // For the renderer, assign all its inputs to the values of the attributes
        // present, or a default value
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
        // Assign default values if not present
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
            "@group(1) @binding(2) var<storage, read> properties : Properties;".to_string()
        };

        // Generate the shader code for the initializing shader
        let mut init_context = InitContext::default();
        for m in asset.modifiers.iter().filter_map(|m| m.as_init()) {
            m.apply(&mut init_context);
        }
        // Warn in debug if the shader doesn't initialize the particle lifetime
        #[cfg(debug_assertions)]
        if !init_context
            .init_code
            .contains(&format!("particle.{}", Attribute::LIFETIME.name()))
        {
            warn!("Effect '{}' does not initialize the particle lifetime; particles will have a default lifetime of zero, and will immediately die after spawning. Add an InitLifetimeModifier to initialize the lifetime to a non-zero value.", asset.name);
        }

        // Generate the shader code for the update shader
        let mut update_context = UpdateContext::default();
        for m in asset.modifiers.iter().filter_map(|m| m.as_update()) {
            m.apply(&mut update_context);
        }

        // Insert Euler motion integration if needed.
        let has_position = present_attributes.contains(&Attribute::POSITION);
        let has_velocity = present_attributes.contains(&Attribute::VELOCITY);
        if asset.motion_integration != MotionIntegration::None {
            if has_position && has_velocity {
                // Note the prepended "\n" to prevent appending to a comment line.
                let code = format!(
                    "\nparticle.{0} += particle.{1} * sim_params.dt;\n",
                    Attribute::POSITION.name(),
                    Attribute::VELOCITY.name()
                );
                if asset.motion_integration == MotionIntegration::PreUpdate {
                    update_context.update_code.insert_str(0, &code);
                } else {
                    update_context.update_code += &code;
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
        let mut render_context = RenderContext::default();
        for m in asset.modifiers.iter().filter_map(|m| m.as_render()) {
            m.apply(&mut render_context);
        }

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
                "particle.{0} = particle.{0} + sim_params.dt;",
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

        // Configure the init shader template, and make sure a corresponding shader
        // asset exists
        let init_shader_source = PARTICLES_INIT_SHADER_TEMPLATE
            .replace("{{ATTRIBUTES}}", &attributes_code)
            .replace("{{INIT_CODE}}", &init_context.init_code)
            .replace("{{INIT_EXTRA}}", &init_context.init_extra)
            .replace("{{PROPERTIES}}", &properties_code)
            .replace("{{PROPERTIES_BINDING}}", &properties_binding_code);
        let init_shader = shader_cache.get_or_insert(&init_shader_source, shaders);
        trace!("Configured init shader:\n{}", init_shader_source);

        // Configure the update shader template, and make sure a corresponding shader
        // asset exists
        let update_shader_source = PARTICLES_UPDATE_SHADER_TEMPLATE
            .replace("{{ATTRIBUTES}}", &attributes_code)
            .replace("{{AGE_CODE}}", &age_code)
            .replace("{{REAP_CODE}}", &reap_code)
            .replace("{{UPDATE_CODE}}", &update_context.update_code)
            .replace("{{UPDATE_EXTRA}}", &update_context.update_extra)
            .replace("{{PROPERTIES}}", &properties_code)
            .replace("{{PROPERTIES_BINDING}}", &properties_binding_code);
        let update_shader = shader_cache.get_or_insert(&update_shader_source, shaders);
        trace!("Configured update shader:\n{}", update_shader_source);

        // Configure the render shader template, and make sure a corresponding shader
        // asset exists
        let render_shader_source = PARTICLES_RENDER_SHADER_TEMPLATE
            .replace("{{ATTRIBUTES}}", &attributes_code)
            .replace("{{INPUTS}}", &inputs_code)
            .replace("{{VERTEX_MODIFIERS}}", &render_context.vertex_code)
            .replace("{{FRAGMENT_MODIFIERS}}", &render_context.fragment_code)
            .replace("{{RENDER_EXTRA}}", &render_context.render_extra);
        let render_shader = shader_cache.get_or_insert(&render_shader_source, shaders);
        trace!("Configured render shader:\n{}", render_shader_source);

        trace!(
            "tick_spawners: init_shader={:?} update_shader={:?} render_shader={:?} has_image={}",
            init_shader,
            update_shader,
            render_shader,
            if render_context.particle_texture.is_some() {
                "Y"
            } else {
                "N"
            },
        );

        // TODO - Replace with Option<ConfiguredShader { handle: Handle<Shader>, hash:
        // u64 }> where the hash takes into account the code and extra code
        // for each pass (and any other varying item). We don't need to keep
        // around the entire shader code, only a hash of it for compare (or, maybe safer
        // to avoid hash collisions, an index into a shader cache). The only
        // use is to be able to compare 2 instances and see if they can be
        // batched together.
        self.configured_init_shader = Some(init_shader);
        self.configured_update_shader = Some(update_shader);
        self.configured_render_shader = Some(render_shader);

        self.force_field = update_context.force_field;
        self.particle_texture = render_context.particle_texture.clone();
    }

    /// Get the init, update, and render shaders if they're all configured, or
    /// `None` otherwise.
    pub(crate) fn get_configured_shaders(
        &self,
    ) -> Option<(Handle<Shader>, Handle<Shader>, Handle<Shader>)> {
        let init_shader = if let Some(init_shader) = &self.configured_init_shader {
            init_shader.clone()
        } else {
            return None;
        };
        let update_shader = if let Some(update_shader) = &self.configured_update_shader {
            update_shader.clone()
        } else {
            return None;
        };
        let render_shader = if let Some(render_shader) = &self.configured_render_shader {
            render_shader.clone()
        } else {
            return None;
        };
        Some((init_shader, update_shader, render_shader))
    }

    /// Set the value of a property associated with this effect.
    ///
    /// A property must exist which has been added to the source
    /// [`EffectAsset`].
    pub fn set_property(&mut self, name: &str, value: graph::Value) {
        if let Some(index) = self
            .properties
            .iter()
            .position(|prop| prop.def.name() == name)
        {
            self.properties[index].value = value;
        }
    }

    /// Write all properties into a binary buffer ready for GPU upload.
    ///
    /// Return the binary buffer where properties have been written according to
    /// the given property layout. The size of the output buffer is guaranteed
    /// to be equal to the size of the layout.
    fn write_properties(&self, layout: &PropertyLayout) -> Vec<u8> {
        let size = layout.size() as usize;
        let mut data = vec![0; size];
        // FIXME: O(n^2) search due to offset() being O(n) linear search already
        for property in &self.properties {
            if let Some(offset) = layout.offset(property.def.name()) {
                let offset = offset as usize;
                let size = property.def.size();
                let src = property.value.as_bytes();
                debug_assert_eq!(src.len(), size);
                let dst = &mut data[offset..offset + size];
                dst.copy_from_slice(src);
            }
        }
        data
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

/// Compile all the visible [`ParticleEffect`] components.
///
/// This system runs in the [`CoreSet::PostUpdate`] stage, after the visibility
/// system has updated the [`ComputedVisibility`] of each effect instance (see
/// [`VisibilitySystems::CheckVisibility`]). Hidden instances are not compiled,
/// unless their [`EffectAsset::simulation_condition`] is set to
/// [`SimulationCondition::Always`].
///
/// [`VisibilitySystems::CheckVisibility`]: bevy::render::view::VisibilitySystems::CheckVisibility
/// [`EffectAsset::simulation_condition`]: crate::EffectAsset::simulation_condition
fn compile_effects(
    effects: Res<Assets<EffectAsset>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut shader_cache: ResMut<ShaderCache>,
    mut q_always: Query<
        (Entity, Ref<ParticleEffect>, &mut CompiledParticleEffect),
        Without<ComputedVisibility>,
    >,
    mut q_when_visible: Query<(
        Entity,
        &ComputedVisibility,
        Ref<ParticleEffect>,
        &mut CompiledParticleEffect,
    )>,
) {
    trace!("compile_effects");

    // Loop over all existing effects to update them:
    // - Always-simulated effects with an available asset;
    // - Other effects with and available asset only when visible.
    for (asset, entity, effect, mut compiled_effect) in q_always
        .iter_mut()
        .filter_map(|(entity, effect, compiled_effect)| {
            // Check if asset is available, otherwise silently ignore as we can't check for
            // changes, and conceptually it makes no sense to render a particle effect whose
            // asset was unloaded.
            let Some(asset) = effects.get(&effect.handle) else { return None; };

            Some((asset, entity, effect, compiled_effect))
        })
        .chain(q_when_visible.iter_mut().filter_map(
            |(entity, computed_visibility, effect, compiled_effect)| {
                // Check if asset is available, otherwise silently ignore as we can't check for
                // changes, and conceptually it makes no sense to render a particle effect whose
                // asset was unloaded.
                let Some(asset) = effects.get(&effect.handle) else { return None; };

                if asset.simulation_condition == SimulationCondition::WhenVisible
                    && !computed_visibility.is_visible()
                {
                    return None;
                }

                Some((asset, entity, effect, compiled_effect))
            },
        ))
    {
        #[cfg(feature = "2d")]
        let z_layer_2d = effect
            .z_layer_2d
            .map_or(FloatOrd(asset.z_layer_2d), |z_layer_2d| {
                FloatOrd(z_layer_2d)
            });

        // Update the compiled effect
        let need_rebuild = effect.is_changed();
        if need_rebuild {
            debug!("Invalidating the compiled cache for effect on entity {:?} due to changes in the ParticleEffect component. If you see this message too much, then performance might be affected. Find why the change detection of the ParticleEffect is triggered.", entity);
        }
        compiled_effect.update(
            need_rebuild,
            &effect.properties,
            #[cfg(feature = "2d")]
            z_layer_2d,
            effect.handle.clone_weak(),
            asset,
            &mut shaders,
            &mut shader_cache,
        );
    }
}

/// Event sent by [`gather_removed_effects()`] with the list of effects removed
/// during this frame.
///
/// The event is consumed during the extract phase by the [`extract_effects()`]
/// system, to clean-up unused GPU resources.
///
/// [`extract_effects()`]: crate::render::extract_effects
struct RemovedEffectsEvent {
    entities: Vec<Entity>,
}

/// Gather all the removed [`ParticleEffect`] components to allow cleaning-up
/// unused GPU resources.
///
/// This system executes inside the [`EffectSystems::GatherRemovedEffects`]
/// system set.
fn gather_removed_effects(
    mut removed_effects: RemovedComponents<ParticleEffect>,
    mut removed_effects_event_writer: EventWriter<RemovedEffectsEvent>,
) {
    let entities: Vec<Entity> = removed_effects.iter().collect();
    if !entities.is_empty() {
        removed_effects_event_writer.send(RemovedEffectsEvent { entities });
    }
}

#[cfg(test)]
mod tests {
    use bevy::{
        render::view::{VisibilityPlugin, VisibilitySystems},
        tasks::IoTaskPool,
    };

    use crate::{spawn::new_rng, test_utils::DummyAssetIo};

    use super::*;

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
        assert_eq!(s, "vec2<f32>(1., 2.)");
        let s = Vec3::new(1., 2., -1.).to_wgsl_string();
        assert_eq!(s, "vec3<f32>(1., 2., -1.)");
        let s = Vec4::new(1., 2., -1., 2.).to_wgsl_string();
        assert_eq!(s, "vec4<f32>(1., 2., -1., 2.)");
    }

    #[test]
    fn to_wgsl_value_f32() {
        let s = Value::Single(1.0_f32).to_wgsl_string();
        assert_eq!(s, "1.");
        let s = Value::Uniform((1.0_f32, 2.0_f32)).to_wgsl_string();
        assert_eq!(s, "(rand() * (2. - 1.) + 1.)");
    }

    #[test]
    fn to_wgsl_value_vec2() {
        let s = Value::Single(Vec2::ONE).to_wgsl_string();
        assert_eq!(s, "vec2<f32>(1., 1.)");
        let s = Value::Uniform((Vec2::ZERO, Vec2::ONE)).to_wgsl_string();
        assert_eq!(
            s,
            "(rand2() * (vec2<f32>(1., 1.) - vec2<f32>(0., 0.)) + vec2<f32>(0., 0.))"
        );
    }

    #[test]
    fn to_wgsl_value_vec3() {
        let s = Value::Single(Vec3::ONE).to_wgsl_string();
        assert_eq!(s, "vec3<f32>(1., 1., 1.)");
        let s = Value::Uniform((Vec3::ZERO, Vec3::ONE)).to_wgsl_string();
        assert_eq!(
            s,
            "(rand3() * (vec3<f32>(1., 1., 1.) - vec3<f32>(0., 0., 0.)) + vec3<f32>(0., 0., 0.))"
        );
    }

    #[test]
    fn to_wgsl_value_vec4() {
        let s = Value::Single(Vec4::ONE).to_wgsl_string();
        assert_eq!(s, "vec4<f32>(1., 1., 1., 1.)");
        let s = Value::Uniform((Vec4::ZERO, Vec4::ONE)).to_wgsl_string();
        assert_eq!(s, "(rand4() * (vec4<f32>(1., 1., 1., 1.) - vec4<f32>(0., 0., 0., 0.)) + vec4<f32>(0., 0., 0., 0.))");
    }

    #[test]
    fn to_shader_code() {
        let mut grad = Gradient::new();
        assert_eq!("", grad.to_shader_code("key"));

        grad.add_key(0.0, Vec4::splat(0.0));
        assert_eq!(
            "// Gradient\nlet t0 = 0.;\nlet c0 = vec4<f32>(0., 0., 0., 0.);\nreturn c0;\n",
            grad.to_shader_code("key")
        );

        grad.add_key(1.0, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(
            r#"// Gradient
let t0 = 0.;
let c0 = vec4<f32>(0., 0., 0., 0.);
let t1 = 1.;
let c1 = vec4<f32>(1., 0., 0., 1.);
if (key <= t0) { return c0; }
else if (key <= t1) { return mix(c0, c1, (key - t0) / (t1 - t0)); }
else { return c1; }
"#,
            grad.to_shader_code("key")
        );
    }

    fn make_test_app() -> App {
        IoTaskPool::init(Default::default);
        let asset_server = AssetServer::new(DummyAssetIo {});

        let mut app = App::new();
        app.insert_resource(asset_server);
        // app.add_plugins(DefaultPlugins);
        app.add_asset::<Mesh>();
        app.add_asset::<Shader>();
        app.add_plugin(VisibilityPlugin);
        app.init_resource::<ShaderCache>();
        app.insert_resource(Random(new_rng()));
        app.add_asset::<EffectAsset>();
        app.add_system(
            compile_effects
                .in_base_set(CoreSet::PostUpdate)
                .after(VisibilitySystems::CheckVisibility),
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
    fn test_compile_effects() {
        let spawner = Spawner::once(32.0.into(), true);

        for test_case in &[
            TestCase::new(None),
            TestCase::new(Some(Visibility::Hidden)),
            TestCase::new(Some(Visibility::Visible)),
        ] {
            let mut app = make_test_app();

            let (effect_entity, handle) = {
                let world = &mut app.world;

                // Add effect asset
                let mut assets = world.resource_mut::<Assets<EffectAsset>>();
                let handle = assets.add(EffectAsset {
                    capacity: 64,
                    spawner,
                    simulation_condition: if test_case.visibility.is_some() {
                        SimulationCondition::WhenVisible
                    } else {
                        SimulationCondition::Always
                    },
                    ..default()
                });

                // Spawn particle effect
                let entity = if let Some(visibility) = test_case.visibility {
                    world
                        .spawn((
                            visibility,
                            ComputedVisibility::default(),
                            ParticleEffect {
                                handle: handle.clone(),
                                ..default()
                            },
                            CompiledParticleEffect::default(),
                        ))
                        .id()
                } else {
                    world
                        .spawn((
                            ParticleEffect {
                                handle: handle.clone(),
                                ..default()
                            },
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

            let world = &mut app.world;

            // Check the state of the components after `tick_spawners()` ran
            if let Some(test_visibility) = test_case.visibility {
                // Simulated-when-visible effect (SimulationCondition::WhenVisible)

                let (
                    entity,
                    visibility,
                    computed_visibility,
                    particle_effect,
                    compiled_particle_effect,
                ) = world
                    .query::<(
                        Entity,
                        &Visibility,
                        &ComputedVisibility,
                        &ParticleEffect,
                        &CompiledParticleEffect,
                    )>()
                    .iter(world)
                    .next()
                    .unwrap();
                assert_eq!(entity, effect_entity);
                assert_eq!(visibility, test_visibility);
                assert_eq!(
                    computed_visibility.is_visible(),
                    test_visibility == Visibility::Visible
                );
                assert_eq!(particle_effect.handle, handle);
                if computed_visibility.is_visible() {
                    // If visible, `compile_effects()` updates the CompiledParticleEffect
                    assert_eq!(compiled_particle_effect.asset, handle);
                    assert!(compiled_particle_effect.asset.is_weak());
                    assert!(compiled_particle_effect.configured_init_shader.is_some());
                    assert!(compiled_particle_effect.configured_update_shader.is_some());
                    assert!(compiled_particle_effect.configured_render_shader.is_some());
                } else {
                    // If not visible, `compile_effects()` skips the effect entirely so won't update
                    // the CompiledParticleEffect
                    assert_ne!(compiled_particle_effect.asset, handle);
                    assert!(compiled_particle_effect.configured_init_shader.is_none());
                    assert!(compiled_particle_effect.configured_update_shader.is_none());
                    assert!(compiled_particle_effect.configured_render_shader.is_none());
                }
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
                assert!(compiled_particle_effect.asset.is_weak());
                assert!(compiled_particle_effect.configured_init_shader.is_some());
                assert!(compiled_particle_effect.configured_update_shader.is_some());
                assert!(compiled_particle_effect.configured_render_shader.is_some());
            }
        }
    }
}
