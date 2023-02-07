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
#![allow(dead_code)] // TEMP
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
//! ## Example: enable only 3D integration
//! bevy_hanabi = { version = "0.4", default-features = false, features = ["3d"] }
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
//!     // to be over the surface of a sphere of radius 2 units, with
//!     // a radial initial velocity of 6 units/sec away from the
//!     // sphere center.
//!     .init(PositionSphereModifier {
//!         center: Vec3::ZERO,
//!         radius: 2.,
//!         dimension: ShapeDimension::Surface,
//!         speed: 6.0.into(),
//!     })
//!     // Every frame, add a gravity-like acceleration downward
//!     .update(AccelModifier {
//!         accel: Vec3::new(0., -3., 0.),
//!     })
//!     // Render the particles with a color gradient over their
//!     // lifetime.
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

use bevy::prelude::*;
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

use properties::PropertyInstance;
use render::EffectCacheId;

pub use asset::{EffectAsset, RenderLayout, UpdateLayout};
pub use attributes::*;
pub use bundle::ParticleEffectBundle;
pub use gradient::{Gradient, GradientKey};
pub use modifier::*;
pub use plugin::HanabiPlugin;
pub use properties::{Property, PropertyLayout};
pub use render::ShaderCache;
pub use spawn::{DimValue, Random, Spawner, Value};

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
                "rand() * ({1} - {0}) + {0}",
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
                "rand2() * ({1} - {0}) + {0}",
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
                "rand3() * ({1} - {0}) + {0}",
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
                "rand4() * ({1} - {0}) + {0}",
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
    //Local, // TODO
}

/// Visual effect made of particles.
///
/// The particle effect component represent a single instance of a visual
/// effect. The visual effect itself is described by a handle to an
/// [`EffectAsset`]. This instance is associated to an [`Entity`], inheriting
/// its [`Transform`] as the origin frame for its particle spawning.
///
/// When spawning a new [`ParticleEffect`], consider using the
/// [`ParticleEffectBundle`] to ensure all the necessary components are present
/// on the entity for the effect to render correctly.
#[derive(Debug, Default, Clone, Component, Reflect)]
#[reflect(Component)]
pub struct ParticleEffect {
    /// Handle of the effect to instantiate.
    handle: Handle<EffectAsset>,
    /// For 2D rendering, override the value of the Z coordinate of the layer at
    /// which the particles are rendered present in the effect asset.
    ///
    /// This value is passed to the render pipeline and used when sorting
    /// transparent items to render, to order them. As a result, effects
    /// with different Z values cannot be batched together, which may
    /// negatively affect performance.
    ///
    /// Ignored for 3D rendering.
    z_layer_2d: Option<f32>,
    /// Internal effect cache ID of the effect once allocated.
    #[reflect(ignore)]
    effect: EffectCacheId,
    /// Particle spawning descriptor.
    spawner: Option<Spawner>,
    /// Handle to the configured init shader for his effect instance, if
    /// configured.
    #[reflect(ignore)]
    configured_init_shader: Option<Handle<Shader>>,
    /// Handle to the configured update shader for his effect instance, if
    /// configured.
    #[reflect(ignore)]
    configured_update_shader: Option<Handle<Shader>>,
    /// Handle to the configured render shader for his effect instance, if
    /// configured.
    #[reflect(ignore)]
    configured_render_shader: Option<Handle<Shader>>,
    /// Instances of all exposed properties.
    properties: Vec<PropertyInstance>,

    // bunch of stuff that should move, which we store here temporarily between tick_spawners()
    // ticking the spawner and the extract/prepare/queue render stages consuming them.
    spawn_count: u32,
    init_code: String,
    init_extra: String,
    update_code: String,
    update_extra: String,
}

impl ParticleEffect {
    /// Create a new particle effect without a spawner or any modifier.
    pub fn new(handle: Handle<EffectAsset>) -> Self {
        Self {
            handle,
            z_layer_2d: None,
            effect: EffectCacheId::INVALID,
            spawner: None,
            configured_init_shader: None,
            configured_update_shader: None,
            configured_render_shader: None,
            properties: vec![],
            //
            spawn_count: 0,
            init_code: String::default(),
            init_extra: String::default(),
            update_code: String::default(),
            update_extra: String::default(),
        }
    }

    ///
    pub fn with_z_layer_2d(mut self, z_layer_2d: Option<f32>) -> Self {
        self.z_layer_2d = z_layer_2d;
        self
    }

    /// Set the spawner of this particle effect instance.
    ///
    /// By default particle effect instances inherit the spawner of the
    /// [`EffectAsset`] they're derived from. This allows overriding the spawner
    /// configuration per instance.
    pub fn set_spawner(&mut self, spawner: Spawner) {
        self.spawner = Some(spawner);
    }

    /// Get the spawner of this particle effect.
    ///
    /// Returns `None` if the effect has not yet been copied from the
    /// [`EffectAsset`] nor overridden by [`set_spawner()`].
    ///
    /// [`set_spawner()`]: ParticleEffect::set_spawner
    pub fn maybe_spawner(&mut self) -> Option<&mut Spawner> {
        self.spawner.as_mut()
    }

    /// Initialize the instance from its asset.
    ///
    /// This is called internally when an instance is created, to initialize it
    /// from its source asset.
    pub(crate) fn init_from_asset(&mut self, asset: &EffectAsset) {
        self.properties = asset
            .properties
            .iter()
            .map(|def| PropertyInstance {
                def: def.clone(),
                value: def.default_value().clone(),
            })
            .collect();

        self.spawner = Some(asset.spawner);
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

    /// Write all properties into a binary buffer ready for GPU upload.
    ///
    /// Return the binary buffer where properties have been written according to
    /// the given property layout. The size of the output buffer is guaranteed
    /// to be equal to the size of the layout.
    fn write_properties(&self, layout: &PropertyLayout) -> Vec<u8> {
        let size = layout.size() as usize;
        let mut data = Vec::with_capacity(size);
        data.resize(size, 0u8);
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

const FORCE_FIELD_CODE: &str = include_str!("render/force_field_code.wgsl");

const ENABLED_BILLBOARD_CODE: &str = r##"
    let camera_right = view.view[0];
    let camera_up = view.view[1];

    let world_position = vec4<f32>(particle.position, 1.0)
        + camera_right * vertex_position.x * size.x
        + camera_up * vertex_position.y * size.y;
"##;

const DISABLED_BILLBOARD_CODE: &str = r##"
    let vpos = vertex_position * vec3<f32>(size.x, size.y, 1.0);
    let world_position = vec4<f32>(particle.position + vpos, 1.0);
"##;

/// Trait to convert any data structure to its equivalent shader code.
trait ShaderCode {
    /// Generate the shader code for the current state of the object.
    fn to_shader_code(&self) -> String;
}

impl ShaderCode for Gradient<Vec2> {
    fn to_shader_code(&self) -> String {
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
            s + "size = v0;\n"
        } else {
            // FIXME - particle.age and particle.lifetime are unrelated to Gradient<Vec4>
            s += "let life = particle.age / particle.lifetime;\nif (life <= t0) { size = v0; }\n";
            let mut s = self
                .keys()
                .iter()
                .skip(1)
                .enumerate()
                .map(|(index, _key)| {
                    format!(
                        "else if (life <= t{1}) {{ size = mix(v{0}, v{1}, (life - t{0}) / (t{1} - t{0})); }}\n",
                        index,
                        index + 1
                    )
                })
                .fold(s, |s, key| s + &key);
            let _ = writeln!(s, "else {{ size = v{}; }}", self.keys().len() - 1);
            s
        }
    }
}

impl ShaderCode for Gradient<Vec4> {
    fn to_shader_code(&self) -> String {
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
            s + "out.color = c0;\n"
        } else {
            // FIXME - particle.age and particle.lifetime are unrelated to Gradient<Vec4>
            s += "let life = particle.age / particle.lifetime;\nif (life <= t0) { out.color = c0; }\n";
            let mut s = self
                .keys()
                .iter()
                .skip(1)
                .enumerate()
                .map(|(index, _key)| {
                    format!(
                        "else if (life <= t{1}) {{ out.color = mix(c{0}, c{1}, (life - t{0}) / (t{1} - t{0})); }}\n",
                        index,
                        index + 1
                    )
                })
                .fold(s, |s, key| s + &key);
            let _ = writeln!(s, "else {{ out.color = c{}; }}", self.keys().len() - 1);
            s
        }
    }
}

/// Tick all the spawners of the visible [`ParticleEffect`] components.
///
/// This system runs in the [`CoreStage::PostUpdate`] stage, after the
/// visibility system has updated the [`ComputedVisibility`] of each effect
/// instance (see [`VisibilitySystems::CheckVisibility`]). Hidden instances are
/// not updated.
///
/// [`VisibilitySystems::CheckVisibility`]: bevy::render::view::VisibilitySystems::CheckVisibility
fn tick_spawners(
    time: Res<Time>,
    effects: Res<Assets<EffectAsset>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut shader_cache: ResMut<ShaderCache>,
    mut rng: ResMut<Random>,
    mut query: ParamSet<(
        // All existing ParticleEffect components
        Query<(
            &ComputedVisibility,
            &mut ParticleEffect, /* TODO - Split EffectAsset::Spawner (desc) and
                                  * ParticleEffect::SpawnerData (runtime data), and init the
                                  * latter on component add without a need for the former */
        )>,
        // Changed ParticleEffect components
        Query<&mut ParticleEffect, Changed<ParticleEffect>>,
    )>,
) {
    trace!("tick_spawners");

    // Clear configured shaders if the effect changed (usually due to changes in
    // modifiers)
    for mut effect in query.p1().iter_mut() {
        effect.configured_init_shader = None;
        effect.configured_update_shader = None;
        effect.configured_render_shader = None;
    }

    let dt = time.delta_seconds();

    // Loop over all existing effects to update them
    for (computed_visibility, mut effect) in query.p0().iter_mut() {
        // Hidden effects are entirely skipped for performance reasons
        if !computed_visibility.is_visible() {
            continue;
        }

        // Assign asset if not already done
        let spawner = if let Some(spawner) = effect.maybe_spawner() {
            spawner
        } else {
            // Check if asset is available, otherwise silently ignore
            let asset = if let Some(asset) = effects.get(&effect.handle) {
                asset
            } else {
                continue;
            };

            effect.init_from_asset(&asset);

            effect.spawner.as_mut().unwrap()
        };

        // Tick the effect's spawner to determine the spawn count for this frame
        let spawn_count = spawner.tick(dt, &mut rng.0);

        // TEMP
        effect.spawn_count = spawn_count;

        // Lazily configure shaders on first use (or after some changes occurred)
        // TODO - Reconfigure only the shader that changed, not both
        if effect.configured_init_shader.is_none()
            || effect.configured_update_shader.is_none()
            || effect.configured_render_shader.is_none()
        {
            let asset = effects.get(&effect.handle).unwrap(); // must succeed since it did above

            // Generate the shader code defining the particle attributes
            let particle_layout = asset.particle_layout();
            let attributes_code = particle_layout.generate_code();

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
            for m in asset.modifiers.iter().filter_map(|m| m.init()) {
                m.apply(&mut init_context);
            }
            // Warn in debug if the shader doesn't initialize the particle lifetime
            #[cfg(debug_assertions)]
            if !init_context
                .init_code
                .contains(&format!("particle.{}", Attribute::LIFETIME.name()))
            {
                warn!("Effect '{}' does not initialize the particle lifetime; particles will have a default lifetime of zero, and will immediately die after spawning. Add a modifier like ParticleLifetimeModifier to initialize the lifetime to a non-zero value.", asset.name);
            }

            // Generate the shader code for the update shader
            let mut update_context = UpdateContext::default();
            for m in asset.modifiers.iter().filter_map(|m| m.update()) {
                m.apply(&mut update_context);
            }
            // Append Euler integration (TODO - Do we want to make this explicit?)
            update_context.update_code +=
                &format!("(*particle).position += (*particle).velocity * sim_params.dt;\n");

            // Generate the shader code for the color over lifetime gradient.
            // TODO - Move that to a pre-pass, not each frame!
            let mut vertex_modifiers = asset
                .render_layout
                .lifetime_color_gradient
                .as_ref()
                .map_or_else(String::new, |grad| grad.to_shader_code());
            if let Some(grad) = &asset.render_layout.lifetime_size_gradient {
                vertex_modifiers += &grad.to_shader_code();
            }

            if asset.render_layout.billboard {
                vertex_modifiers += ENABLED_BILLBOARD_CODE
            } else {
                vertex_modifiers += DISABLED_BILLBOARD_CODE
            }

            trace!("vertex_modifiers={}", vertex_modifiers);

            let has_per_particle_size = particle_layout.contains(Attribute::SIZE);
            let has_per_particle_size2 = particle_layout.contains(Attribute::SIZE2);

            // Configure the init shader template, and make sure a corresponding shader
            // asset exists
            let init_shader_source = PARTICLES_INIT_SHADER_TEMPLATE
                .replace("{{ATTRIBUTES}}", &attributes_code)
                .replace("{{INIT_CODE}}", &init_context.init_code)
                .replace("{{INIT_EXTRA}}", &init_context.init_extra);
            let init_shader = shader_cache.get_or_insert(&init_shader_source, &mut shaders);

            // Configure the update shader template, and make sure a corresponding shader
            // asset exists
            let update_shader_source = PARTICLES_UPDATE_SHADER_TEMPLATE
                .replace("{{ATTRIBUTES}}", &attributes_code)
                .replace("{{UPDATE_CODE}}", &update_context.update_code)
                .replace("{{UPDATE_EXTRA}}", &update_context.update_extra)
                .replace("{{PROPERTIES}}", &properties_code)
                .replace("{{PROPERTIES_BINDING}}", &properties_binding_code);
            let update_shader = shader_cache.get_or_insert(&update_shader_source, &mut shaders);

            // Configure the render shader template, and make sure a corresponding shader
            // asset exists
            let mut render_shader_source =
                PARTICLES_RENDER_SHADER_TEMPLATE.replace("{{VERTEX_MODIFIERS}}", &vertex_modifiers);
            render_shader_source = render_shader_source.replace("{{ATTRIBUTES}}", &attributes_code);
            if has_per_particle_size {
                render_shader_source = render_shader_source.replace(
                    "{{SIZE}}",
                    &format!("size = vec2<f32>(particle.{});", Attribute::SIZE.name()),
                );
            } else if has_per_particle_size2 {
                render_shader_source = render_shader_source.replace(
                    "{{SIZE}}",
                    &format!("size = particle.{};", Attribute::SIZE2.name()),
                );
            } else {
                render_shader_source = render_shader_source.replace("{{SIZE}}", "");
            }
            let render_shader = shader_cache.get_or_insert(&render_shader_source, &mut shaders);

            trace!(
                "tick_spawners: handle={:?} init_shader={:?} update_shader={:?} render_shader={:?} has_image={}",
                effect.handle,
                init_shader,
                update_shader,
                render_shader,
                if asset.render_layout.particle_texture.is_some() {
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
            effect.configured_init_shader = Some(init_shader);
            effect.configured_update_shader = Some(update_shader);
            effect.configured_render_shader = Some(render_shader);

            // TEMP - Should disappear after fixing the above TODO.
            effect.init_code = init_context.init_code;
            effect.init_extra = init_context.init_extra;
            effect.update_code = update_context.update_code;
            effect.update_extra = update_context.update_extra;
        }
    }
}

struct RemovedEffectsEvent {
    entities: Vec<Entity>,
}

fn gather_removed_effects(
    removed_effects: RemovedComponents<ParticleEffect>,
    mut removed_effects_event_writer: EventWriter<RemovedEffectsEvent>,
) {
    let entities: Vec<Entity> = removed_effects.iter().collect();
    if !entities.is_empty() {
        removed_effects_event_writer.send(RemovedEffectsEvent { entities });
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(s, "rand() * (2. - 1.) + 1.");
    }

    #[test]
    fn to_shader_code() {
        let mut grad = Gradient::new();
        assert_eq!("", grad.to_shader_code());

        grad.add_key(0.0, Vec4::splat(0.0));
        assert_eq!(
            "// Gradient\nlet t0 = 0.;\nlet c0 = vec4<f32>(0., 0., 0., 0.);\nout.color = c0;\n",
            grad.to_shader_code()
        );

        grad.add_key(1.0, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(
            r#"// Gradient
let t0 = 0.;
let c0 = vec4<f32>(0., 0., 0., 0.);
let t1 = 1.;
let c1 = vec4<f32>(1., 0., 0., 1.);
let life = particle.age / particle.lifetime;
if (life <= t0) { out.color = c0; }
else if (life <= t1) { out.color = mix(c0, c1, (life - t0) / (t1 - t0)); }
else { out.color = c1; }
"#,
            grad.to_shader_code()
        );
    }
}
