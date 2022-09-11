#![deny(
    warnings,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications,
    missing_docs
)]
#![allow(dead_code)] // TEMP
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

//! 🎆 Hanabi -- a GPU particle system plugin for the Bevy game engine.
//!
//! This library provides a GPU-based particle system for the Bevy game engine.
//!
//! # Example
//!
//! Add the Hanabi plugin to your app:
//!
//! ```no_run
//! # use bevy::prelude::*;
//! # use bevy_hanabi::*;
//! App::default()
//!     .add_plugins(DefaultPlugins)
//!     .add_plugin(HanabiPlugin)
//!     .run();
//! ```
//!
//! Create an [`EffectAsset`] describing a visual effect, then add an
//! instance of that effect to an entity:
//!
//! ```
//! # use bevy::prelude::*;
//! # use bevy_hanabi::*;
//! fn setup(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
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
//!     // On spawn, randomly initialize the position and velocity
//!     // of the particle over a sphere of radius 2 units, with a
//!     // radial initial velocity of 6 units/sec away from the
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
//!     commands
//!         .spawn()
//!         .insert(Name::new("MyEffectInstance"))
//!         .insert_bundle(ParticleEffectBundle {
//!             effect: ParticleEffect::new(effect),
//!             transform: Transform::from_translation(Vec3::new(0., 1., 0.)),
//!             ..Default::default()
//!         });
//! }
//! ```

use bevy::prelude::*;
use std::fmt::Write as _; // import without risk of name clashing

mod asset;
mod bundle;
mod gradient;
mod modifiers;
mod plugin;
mod render;
mod spawn;

#[cfg(test)]
mod test_utils;

pub use asset::{EffectAsset, InitLayout, RenderLayout, UpdateLayout};
pub use bundle::ParticleEffectBundle;
pub use gradient::{Gradient, GradientKey};
pub use modifiers::{
    AccelModifier, BillboardModifier, ColorOverLifetimeModifier, ForceFieldModifier,
    ForceFieldParam, InitModifier, ParticleLifetimeModifier, ParticleTextureModifier,
    PositionCircleModifier, PositionCone3dModifier, PositionSphereModifier, RenderModifier,
    ShapeDimension, SizeOverLifetimeModifier, UpdateModifier, FFNUM,
};
pub use plugin::HanabiPlugin;
pub use render::{EffectCacheId, PipelineRegistry};
pub use spawn::{Random, Spawner, Value};

#[cfg(not(any(feature = "2d", feature = "3d")))]
compile_error!(
    "You need to enable at least one of the '2d' or '3d' features for anything to happen."
);

/// Extension trait to write a floating point scalar or vector constant in a
/// format matching the WGSL grammar.
///
/// This is required because WGSL doesn't support a floating point constant
/// without a decimal separator (e.g. `0.` instead of `0`), which would be what
/// a regular float to string formatting produces, but is interpreted as an
/// integral type by WGSL.
///
/// # Example
///
/// ```
/// # use bevy_hanabi::ToWgslString;
/// let x = 2.0_f32;
/// assert_eq!("let x = 2.;", format!("let x = {};", x.to_wgsl_string()));
/// ```
pub trait ToWgslString {
    /// Convert a floating point scalar or vector to a string representing a
    /// WGSL constant.
    fn to_wgsl_string(&self) -> String;
}

impl ToWgslString for f32 {
    fn to_wgsl_string(&self) -> String {
        let s = format!("{:.6}", self);
        s.trim_end_matches('0').to_string()
    }
}

impl ToWgslString for f64 {
    fn to_wgsl_string(&self) -> String {
        let s = format!("{:.15}", self);
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

/// Visual effect made of particles.
///
/// The particle effect component represent a single instance of a visual
/// effect. The visual effect itself is described by a handle to an
/// [`EffectAsset`]. This instance is associated to an [`Entity`], inheriting
/// its [`Transform`] as the origin frame for its particle spawning.
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
    #[reflect(ignore)]
    spawner: Option<Spawner>,
    /// Handle to the configured compute shader for his effect instance, if
    /// configured.
    #[reflect(ignore)]
    configured_compute_shader: Option<Handle<Shader>>,
    /// Handle to the configured render shader for his effect instance, if
    /// configured.
    #[reflect(ignore)]
    configured_render_shader: Option<Handle<Shader>>,

    // bunch of stuff that should move, which we store here temporarily between tick_spawners()
    // ticking the spawner and the extract/prepare/queue render stages consuming them.
    spawn_count: u32,
    accel: Vec3,
    #[reflect(ignore)]
    force_field: [ForceFieldParam; FFNUM],
    position_code: String,
    force_field_code: String,
    lifetime_code: String,
}

impl ParticleEffect {
    /// Create a new particle effect without a spawner or any modifier.
    pub fn new(handle: Handle<EffectAsset>) -> Self {
        ParticleEffect {
            handle,
            z_layer_2d: None,
            effect: EffectCacheId::INVALID,
            spawner: None,
            configured_compute_shader: None,
            configured_render_shader: None,
            //
            spawn_count: 0,
            accel: Vec3::ZERO,
            force_field: [ForceFieldParam::default(); FFNUM],
            position_code: String::default(),
            force_field_code: String::default(),
            lifetime_code: String::default(),
        }
    }

    ///
    pub fn with_z_layer_2d(mut self, z_layer_2d: Option<f32>) -> Self {
        self.z_layer_2d = z_layer_2d;
        self
    }

    /// Sets the spawner of this particle effect.
    pub fn set_spawner(&mut self, spawner: Spawner) {
        self.spawner = Some(spawner);
    }

    /// Configure the spawner of a new particle effect.
    ///
    /// The call returns a reference to the added spawner, allowing to chain
    /// adding modifiers to the effect.
    pub fn spawner(&mut self, spawner: &Spawner) -> &mut Spawner {
        if self.spawner.is_none() {
            self.spawner = Some(*spawner);
        }
        self.spawner.as_mut().unwrap()
    }

    /// Get the spawner of this particle effect.
    ///
    /// Returns None if `with_spawner` was not called
    /// and the effect has not rendered yet.
    pub fn maybe_spawner(&mut self) -> Option<&mut Spawner> {
        self.spawner.as_mut()
    }
}

const PARTICLES_UPDATE_SHADER_TEMPLATE: &str = include_str!("render/particles_update.wgsl");
const PARTICLES_RENDER_SHADER_TEMPLATE: &str = include_str!("render/particles_render.wgsl");

const DEFAULT_POSITION_CODE: &str = r##"
    ret.pos = vec3<f32>(0., 0., 0.);
    var dir = rand3() * 2. - 1.;
    dir = normalize(dir);
    var speed = 2.;
    ret.vel = dir * speed;
"##;

const DEFAULT_LIFETIME_CODE: &str = r##"
ret = 5.0;
"##;

const DEFAULT_FORCE_FIELD_CODE: &str = r##"
    vVel = vVel + (spawner.accel * sim_params.dt);
    vPos = vPos + vVel * sim_params.dt;
"##;

const FORCE_FIELD_CODE: &str = include_str!("render/force_field_code.wgsl");

const ENABLED_BILLBOARD_CODE: &str = r##"
    let camera_up = view.view * vec4<f32>(0.0, 1.0, 0.0, 1.0);
    let camera_right = view.view * vec4<f32>(1.0, 0.0, 0.0, 1.0);

    let world_position = vec4<f32>(particle.pos, 1.0)
        + camera_right * vertex_position.x * size.x
        + camera_up * vertex_position.y * size.y;
"##;

const DISABLED_BILLBOARD_CODE: &str = r##"
    let vpos = vertex_position * vec3<f32>(size.x, size.y, 1.0);
    let world_position = vec4<f32>(particle.pos + vpos, 1.0);
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
fn tick_spawners(
    time: Res<Time>,
    effects: Res<Assets<EffectAsset>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut pipeline_registry: ResMut<PipelineRegistry>,
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
        effect.configured_compute_shader = None;
        effect.configured_render_shader = None;
    }

    let dt = time.delta_seconds();

    // Loop over all existing effects to update them
    for (computed_visibility, mut effect) in query.p0().iter_mut() {
        // Check if visible. Hidden effects are entirely skipped for performance
        // reasons.
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

            effect.spawner(&asset.spawner)
        };

        // Tick the effect's spawner to determine the spawn count for this frame
        let spawn_count = spawner.tick(dt, &mut rng.0);

        // TEMP
        effect.spawn_count = spawn_count;

        // Lazily configure shaders on first use (or after some changes occurred)
        // TODO - Reconfigure only the shader that changed, not both
        if effect.configured_compute_shader.is_none() || effect.configured_render_shader.is_none() {
            let asset = effects.get(&effect.handle).unwrap(); // must succeed since it did above

            // Extract the acceleration
            let accel = asset.update_layout.accel;
            let force_field = asset.update_layout.force_field;

            // Generate the shader code for the position initializing of newly emitted
            // particles TODO - Move that to a pre-pass, not each frame!
            let position_code = &asset.init_layout.position_code;
            let position_code = if position_code.is_empty() {
                DEFAULT_POSITION_CODE.to_owned()
            } else {
                position_code.clone()
            };

            // Generate the shader code for the lifetime initializing of newly emitted
            // particles TODO - Move that to a pre-pass, not each frame!
            let lifetime_code = &asset.init_layout.lifetime_code;
            let lifetime_code = if lifetime_code.is_empty() {
                DEFAULT_LIFETIME_CODE.to_owned()
            } else {
                lifetime_code.clone()
            };

            // Generate the shader code for the force field of newly emitted particles
            // TODO - Move that to a pre-pass, not each frame!
            // let force_field_code = &asset.init_layout.force_field_code;
            // let force_field_code = if force_field_code.is_empty() {
            let force_field_code = if 0.0 == asset.update_layout.force_field[0].force_exponent {
                DEFAULT_FORCE_FIELD_CODE.to_owned()
            } else {
                FORCE_FIELD_CODE.to_owned()
            };

            // Generate the shader code for the color over lifetime gradient.
            // TODO - Move that to a pre-pass, not each frame!
            let mut vertex_modifiers =
                if let Some(grad) = &asset.render_layout.lifetime_color_gradient {
                    grad.to_shader_code()
                } else {
                    String::new()
                };
            if let Some(grad) = &asset.render_layout.size_color_gradient {
                vertex_modifiers += &grad.to_shader_code();
            }

            if asset.render_layout.billboard {
                vertex_modifiers += ENABLED_BILLBOARD_CODE
            } else {
                vertex_modifiers += DISABLED_BILLBOARD_CODE
            }

            trace!("vertex_modifiers={}", vertex_modifiers);

            // Configure the compute shader template, and make sure a corresponding shader
            // asset exists
            let mut compute_shader_source =
                PARTICLES_UPDATE_SHADER_TEMPLATE.replace("{{INIT_POS_VEL}}", &position_code);
            compute_shader_source =
                compute_shader_source.replace("{{FORCE_FIELD_CODE}}", &force_field_code);
            compute_shader_source =
                compute_shader_source.replace("{{INIT_LIFETIME}}", &lifetime_code);
            let compute_shader = pipeline_registry.configure(&compute_shader_source, &mut shaders);

            // Configure the render shader template, and make sure a corresponding shader
            // asset exists
            let render_shader_source =
                PARTICLES_RENDER_SHADER_TEMPLATE.replace("{{VERTEX_MODIFIERS}}", &vertex_modifiers);
            let render_shader = pipeline_registry.configure(&render_shader_source, &mut shaders);

            trace!(
                "tick_spawners: handle={:?} compute_shader={:?} render_shader={:?} has_image={} position_code={} force_field_code={} lifetime_code={}",
                effect.handle,
                compute_shader,
                render_shader,
                if asset.render_layout.particle_texture.is_some() {
                    "Y"
                } else {
                    "N"
                },
                position_code,
                force_field_code,
                lifetime_code,
            );

            effect.configured_compute_shader = Some(compute_shader);
            effect.configured_render_shader = Some(render_shader);

            // TEMP
            effect.accel = accel;
            effect.force_field = force_field;
            effect.position_code = position_code;
            effect.force_field_code = force_field_code;
            effect.lifetime_code = lifetime_code;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
