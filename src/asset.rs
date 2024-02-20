use bevy::{
    asset::{io::Reader, Asset, AssetLoader, AsyncReadExt, LoadContext},
    reflect::Reflect,
    utils::{default, thiserror::Error, BoxedFuture, HashSet},
};
use serde::{Deserialize, Serialize};

use crate::{
    graph::Value,
    modifier::{InitModifier, RenderModifier, UpdateModifier},
    BoxedModifier, ExprHandle, Module, ParticleLayout, Property, PropertyLayout, SimulationSpace,
    Spawner,
};

/// Type of motion integration applied to the particles of a system.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum MotionIntegration {
    /// No motion integration. The [`Attribute::POSITION`] of the particles
    /// needs to be explicitly assigned by a modifier for the particles to move.
    ///
    /// [`Attribute::POSITION`]: crate::Attribute::POSITION
    None,

    /// Apply Euler motion integration each simulation update before all
    /// modifiers are applied.
    ///
    /// Not to be confused with Bevy's `PreUpdate` phase. Here "update" refers
    /// to the particle update on the GPU via a compute shader.
    PreUpdate,

    /// Apply Euler motion integration each simulation update after all
    /// modifiers are applied. This is the default.
    ///
    /// Not to be confused with Bevy's `PostUpdate` phase. Here "update" refers
    /// to the particle update on the GPU via a compute shader.
    #[default]
    PostUpdate,
}

/// Simulation condition for an effect.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
pub enum SimulationCondition {
    /// Simulate the effect only when visible.
    ///
    /// The visibility is determined by the [`Visibility`], the
    /// [`InheritedVisibility`], and the [`ViewVisibility`] components.
    ///
    /// This is the default for all assets, and is the most performant option,
    /// allowing to have many effects in the scene without the need to simulate
    /// all of them if they're not visible.
    ///
    /// Note that any [`ParticleEffect`] spawned is always compiled into a
    /// [`CompiledParticleEffect`], even when it's not visible and even when
    /// that variant is selected.
    ///
    /// Note also that AABB culling is not currently available. Only boolean
    /// ON/OFF visibility is used.
    ///
    /// [`Visibility`]: bevy::render::view::Visibility
    /// [`InheritedVisibility`]: bevy::render::view::InheritedVisibility
    /// [`ViewVisibility`]: bevy::render::view::ViewVisibility
    /// [`ParticleEffect`]: crate::ParticleEffect
    /// [`CompiledParticleEffect`]: crate::CompiledParticleEffect
    #[default]
    WhenVisible,

    /// Always simulate the effect, whether visible or not.
    ///
    /// For performance reasons, it's recommended to only simulate visible
    /// particle effects (that is, use [`SimulationCondition::WhenVisible`]).
    /// However occasionally it may be needed to continue the simulation
    /// when the effect is not visible, to ensure some temporal continuity when
    /// the effect is made visible again. This is an uncommon case, and you
    /// should be aware of the performance implications of using this
    /// condition, and only use it when strictly necessary.
    ///
    /// Any [`Visibility`], [`InheritedVisibility`], or [`ViewVisibility`]
    /// component is ignored. You may want to spawn the particle effect
    /// components manually instead of using the [`ParticleEffectBundle`] to
    /// avoid adding those components.
    ///
    /// [`Visibility`]: bevy::render::view::Visibility
    /// [`InheritedVisibility`]: bevy::render::view::InheritedVisibility
    /// [`ViewVisibility`]: bevy::render::view::ViewVisibility
    /// [`ParticleEffectBundle`]: crate::ParticleEffectBundle
    Always,
}

/// Alpha mode for rendering an effect.
///
/// The alpha mode determines how the alpha value of a particle is used to
/// render it. In general effects use semi-transparent particles. However, there
/// are multiple alpha blending techniques available, producing different
/// results.
///
/// This is very similar to the `bevy::pbr::AlphaMode` of the `bevy_pbr` crate,
/// except that a different set of values is supported which reflects what this
/// library currently supports.
///
/// The alpha mode only affects the render phase that particles are rendered
/// into when rendering 3D views. For 2D views, all particle effects are
/// rendered during the [`Transparent2d`] render phase.
///
/// [`Transparent2d`]: bevy::core_pipeline::core_2d::Transparent2d
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AlphaMode {
    /// Render the effect with alpha blending.
    ///
    /// This is the most common mode for handling transparency. It uses the
    /// "blend" or "over" formula, where the color of each particle fragment is
    /// accumulated into the destination render target after being modulated by
    /// its alpha value.
    ///
    /// ```txt
    /// dst_color = src_color * (1 - particle_alpha) + particle_color * particle_alpha;
    /// dst_alpha = src_alpha * (1 - particle_alpha) + particle_alpha
    /// ```
    ///
    /// This is the default blending mode.
    ///
    /// For 3D views, effects with this mode are rendered during the
    /// [`Transparent3d`] render phase.
    ///
    /// [`Transparent3d`]: bevy::core_pipeline::core_3d::Transparent3d
    #[default]
    Blend,

    /// Render the effect with alpha masking.
    ///
    /// With this mode, the final alpha value computed per particle fragment is
    /// compared against the cutoff value stored in this enum. Any fragment
    /// with a value under the cutoff is discarded, while any fragment with
    /// a value equal or over the cutoff becomes fully opaque. The end result is
    /// an opaque particle with a cutout shape.
    ///
    /// ```txt
    /// if src_alpha >= cutoff {
    ///     dst_color = particle_color;
    ///     dst_alpha = 1;
    /// } else {
    ///     discard;
    /// }
    /// ```
    ///
    /// The assigned expression must yield a scalar floating-point value,
    /// typically in the \[0:1\] range. This expression is assigned at the
    /// beginning of the fragment shader to the special built-in `alpha_cutoff`
    /// variable, which can be further accessed and modified by render
    /// modifiers.
    ///
    /// The cutoff threshold comparison of the fragment's alpha value against
    /// `alpha_cutoff` is performed as the last operation in the fragment
    /// shader. This allows modifiers to affect the alpha value of the
    /// particle before it's tested against the cutoff value stored in
    /// `alpha_cutoff`.
    ///
    /// For 3D views, effects with this mode are rendered during the
    /// [`AlphaMask3d`] render phase.
    ///
    /// [`AlphaMask3d`]: bevy::core_pipeline::core_3d::AlphaMask3d
    Mask(ExprHandle),
}

/// Asset describing a visual effect.
///
/// The effect can be instanciated with a [`ParticleEffect`] component, or a
/// [`ParticleEffectBundle`].
///
/// [`ParticleEffect`]: crate::ParticleEffect
/// [`ParticleEffectBundle`]: crate::ParticleEffectBundle
#[derive(Asset, Default, Clone, Reflect, Serialize, Deserialize)]
#[reflect(from_reflect = false)]
pub struct EffectAsset {
    /// Display name of the effect.
    ///
    /// This has no internal use, and is mostly for the user to identify an
    /// effect or for display in some tool UI. It's however used in serializing
    /// the asset.
    pub name: String,
    /// Maximum number of concurrent particles.
    ///
    /// The capacity is the maximum number of particles that can be alive at the
    /// same time. It determines the size of various GPU resources, most notably
    /// the particle buffer itself. To prevent wasting GPU resources, users
    /// should keep this quantity as close as possible to the maximum number of
    /// particles they expect to render.
    capacity: u32,
    /// Spawner.
    pub spawner: Spawner,
    /// For 2D rendering, the Z coordinate used as the sort key.
    ///
    /// This value is passed to the render pipeline and used when sorting
    /// transparent items to render, to order them. As a result, effects
    /// with different Z values cannot be batched together, which may
    /// negatively affect performance.
    ///
    /// Ignored for 3D rendering.
    pub z_layer_2d: f32,
    /// Particle simulation space.
    pub simulation_space: SimulationSpace,
    /// Condition under which the effect is simulated.
    pub simulation_condition: SimulationCondition,
    /// Init modifier defining the effect.
    #[reflect(ignore)]
    // TODO - Can't manage to implement FromReflect for BoxedModifier in a nice way yet
    init_modifiers: Vec<BoxedModifier>,
    /// update modifiers defining the effect.
    #[reflect(ignore)]
    // TODO - Can't manage to implement FromReflect for BoxedModifier in a nice way yet
    update_modifiers: Vec<BoxedModifier>,
    /// Render modifiers defining the effect.
    #[reflect(ignore)]
    // TODO - Can't manage to implement FromReflect for BoxedModifier in a nice way yet
    render_modifiers: Vec<BoxedModifier>,
    /// Properties of the effect.
    ///
    /// Properties must have a unique name. Manually adding two or more
    /// properties with the same name will result in an invalid asset and
    /// undefined behavior are runtime. Prefer using the [`with_property()`] and
    /// [`add_property()`] methods for safety.
    ///
    /// [`with_property()`]: crate::EffectAsset::with_property
    /// [`add_property()`]: crate::EffectAsset::add_property
    properties: Vec<Property>,
    /// Type of motion integration applied to the particles of a system.
    pub motion_integration: MotionIntegration,
    /// Expression module for this effect.
    module: Module,
    /// Alpha mode.
    pub alpha_mode: AlphaMode,
}

impl EffectAsset {
    /// Create a new effect asset.
    ///
    /// The effect assets requires 2 essential pieces:
    /// - The capacity of the effect, which represents the maximum number of
    ///   particles which can be stored and simulated at the same time. The
    ///   capacity must be non-zero, and should be the smallest possible value
    ///   which allows you to author the effect. This value directly impacts the
    ///   GPU memory consumption of the effect, which will allocate some buffers
    ///   to store that many particles for as long as the effect exists. The
    ///   capacity of an effect is immutable. See also [`capacity()`] for more
    ///   details.
    /// - The [`Spawner`], which defines when particles are emitted.
    ///
    /// Additionally, if any modifier added to this effect uses some [`Expr`] to
    /// customize its behavior, then those [`Expr`] are stored into a [`Module`]
    /// which should be passed to this method. If expressions are not used, just
    /// pass an empty module [`Module::default()`].
    ///
    /// # Examples
    ///
    /// Create a new effect asset without any modifier. This effect doesn't
    /// really do anything because _e.g._ the particles have a zero lifetime.
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let spawner = Spawner::rate(5_f32.into()); // 5 particles per second
    /// let module = Module::default();
    /// let effect = EffectAsset::new(32768, spawner, module);
    /// ```
    ///
    /// Create a new effect asset with a modifier holding an expression. The
    /// expression is stored inside the [`Module`] transfered to the
    /// [`EffectAsset`].
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let spawner = Spawner::rate(5_f32.into()); // 5 particles per second
    ///
    /// let mut module = Module::default();
    ///
    /// // Create a modifier that initialized the particle lifetime to 10 seconds.
    /// let lifetime = module.lit(10.); // literal value "10.0"
    /// let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);
    ///
    /// let effect = EffectAsset::new(32768, spawner, module);
    /// ```
    ///
    /// [`capacity()`]: crate::EffectAsset::capacity
    /// [`Expr`]: crate::graph::expr::Expr
    pub fn new(capacity: u32, spawner: Spawner, module: Module) -> Self {
        Self {
            capacity,
            spawner,
            module,
            ..default()
        }
    }

    /// Get the capacity of the effect, in number of particles.
    ///
    /// This represents the number of particles stored in GPU memory at all
    /// time, even if unused, so you should try to minimize this value. However,
    /// the [`Spawner`] cannot emit more particles than this capacity. Whatever
    /// the spanwer settings, if the number of particles reaches the capacity,
    /// no new particle can be emitted. Setting an appropriate capacity for an
    /// effect is therefore a compromise between more particles available for
    /// visuals and more GPU memory usage.
    ///
    /// Common values range from 256 or less for smaller effects, to several
    /// hundreds of thousands for unique effects consuming a large portion of
    /// the GPU memory budget. Hanabi has been tested with over a million
    /// particles, however the performance will largely depend on the actual GPU
    /// hardware and available memory, so authors are encouraged not to go too
    /// crazy with the capacity.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Get the expression module storing all expressions in use by modifiers of
    /// this effect.
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Set the effect name.
    ///
    /// The effect name is used when serializing the effect.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the effect's simulation condition.
    pub fn with_simulation_condition(mut self, simulation_condition: SimulationCondition) -> Self {
        self.simulation_condition = simulation_condition;
        self
    }

    /// Set the effect's simulation space.
    pub fn with_simulation_space(mut self, simulation_space: SimulationSpace) -> Self {
        self.simulation_space = simulation_space;
        self
    }

    /// Set the alpha mode.
    pub fn with_alpha_mode(mut self, alpha_mode: AlphaMode) -> Self {
        self.alpha_mode = alpha_mode;
        self
    }

    /// Add a new property to the asset.
    ///
    /// See [`Property`] for more details on what effect properties are.
    ///
    /// # Panics
    ///
    /// Panics if a property with the same name already exists.
    pub fn with_property(mut self, name: impl Into<String>, default_value: Value) -> Self {
        self.add_property(name, default_value);
        self
    }

    /// Add a new property to the asset.
    ///
    /// See [`Property`] for more details on what effect properties are.
    ///
    /// # Panics
    ///
    /// Panics if a property with the same name already exists.
    pub fn add_property(&mut self, name: impl Into<String>, default_value: Value) {
        let name = name.into();
        assert!(!self.properties.iter().any(|p| p.name() == name));
        self.properties.push(Property::new(name, default_value));
    }

    /// Get the list of existing properties.
    pub fn properties(&self) -> &[Property] {
        &self.properties
    }

    /// Add an initialization modifier to the effect.
    ///
    /// [`with_property()`]: crate::EffectAsset::with_property
    /// [`add_property()`]: crate::EffectAsset::add_property
    #[inline]
    pub fn init<M>(mut self, modifier: M) -> Self
    where
        M: InitModifier + Send + Sync + 'static,
    {
        self.init_modifiers.push(Box::new(modifier));
        self
    }

    /// Add an update modifier to the effect.
    ///
    /// [`with_property()`]: crate::EffectAsset::with_property
    /// [`add_property()`]: crate::EffectAsset::add_property
    #[inline]
    pub fn update<M>(mut self, modifier: M) -> Self
    where
        M: UpdateModifier + Send + Sync + 'static,
    {
        self.update_modifiers.push(Box::new(modifier));
        self
    }

    /// Add a render modifier to the effect.
    ///
    /// [`with_property()`]: crate::EffectAsset::with_property
    /// [`add_property()`]: crate::EffectAsset::add_property
    #[inline]
    pub fn render<M>(mut self, modifier: M) -> Self
    where
        M: RenderModifier + Send + Sync + 'static,
    {
        self.render_modifiers.push(Box::new(modifier));
        self
    }

    /// Get a list of all the modifiers of this effect.
    pub fn modifiers(&self) -> impl Iterator<Item = &BoxedModifier> {
        self.init_modifiers
            .iter()
            .chain(self.update_modifiers.iter())
            .chain(self.render_modifiers.iter())
    }

    /// Get a list of all the init modifiers of this effect.
    ///
    /// This is a filtered list of all modifiers, retaining only modifiers
    /// executing in the [`ModifierContext::Init`] context.
    ///
    /// [`ModifierContext::Init`]: crate::ModifierContext::Init
    pub fn init_modifiers(&self) -> impl Iterator<Item = &dyn InitModifier> {
        self.init_modifiers.iter().filter_map(|m| m.as_init())
    }

    /// Get a list of all the update modifiers of this effect.
    ///
    /// This is a filtered list of all modifiers, retaining only modifiers
    /// executing in the [`ModifierContext::Update`] context.
    ///
    /// [`ModifierContext::Update`]: crate::ModifierContext::Update
    pub fn update_modifiers(&self) -> impl Iterator<Item = &dyn UpdateModifier> {
        self.update_modifiers.iter().filter_map(|m| m.as_update())
    }

    /// Get a list of all the render modifiers of this effect.
    ///
    /// This is a filtered list of all modifiers, retaining only modifiers
    /// executing in the [`ModifierContext::Render`] context.
    ///
    /// [`ModifierContext::Render`]: crate::ModifierContext::Render
    pub fn render_modifiers(&self) -> impl Iterator<Item = &dyn RenderModifier> {
        self.render_modifiers.iter().filter_map(|m| m.as_render())
    }

    /// Build the particle layout of the asset based on its modifiers.
    ///
    /// This method calculates the particle layout of the effect based on the
    /// currently existing particles, and return it as a newly allocated
    /// [`ParticleLayout`] object.
    pub fn particle_layout(&self) -> ParticleLayout {
        // Build the set of unique attributes required for all modifiers
        let mut set = HashSet::new();
        for modifier in self.modifiers() {
            for &attr in modifier.attributes() {
                set.insert(attr);
            }
        }

        // Build the layout
        let mut layout = ParticleLayout::new();
        for attr in set {
            layout = layout.append(attr);
        }
        layout.build()
    }

    /// Build the property layout of the asset based on its properties.
    ///
    /// This method calculates the property layout of the effect based on the
    /// currently existing properties, and return it as a newly allocated
    /// [`PropertyLayout`] object.
    pub fn property_layout(&self) -> PropertyLayout {
        PropertyLayout::new(self.properties.iter())
    }
}

/// Asset loader for [`EffectAsset`].
///
/// Effet assets take the `.effect` extension.
#[derive(Default)]
pub struct EffectAssetLoader;

/// Error for the [`EffectAssetLoader`] loading an [`EffectAsset`].
#[derive(Error, Debug)]
pub enum EffectAssetLoaderError {
    /// I/O error reading the asset source.
    #[error("An IO error occurred during loading of a particle effect")]
    Io(#[from] std::io::Error),

    /// Error during RON format parsing.
    #[error("A RON format error occurred during loading of a particle effect")]
    Ron(#[from] ron::error::SpannedError),
}

impl AssetLoader for EffectAssetLoader {
    type Asset = EffectAsset;

    type Settings = ();

    type Error = EffectAssetLoaderError;

    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _settings: &'a Self::Settings,
        _load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;
            let custom_asset = ron::de::from_bytes::<EffectAsset>(&bytes)?;
            Ok(custom_asset)
        })
    }

    fn extensions(&self) -> &[&str] {
        &["effect"]
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    use super::*;

    #[test]
    fn property() {
        let mut effect = EffectAsset {
            name: "Effect".into(),
            capacity: 4096,
            spawner: Spawner::rate(30.0.into()),
            ..Default::default()
        }
        .with_property("my_prop", graph::Value::Scalar(345_u32.into()));

        effect.add_property(
            "other_prop",
            graph::Value::Vector(Vec3::new(3., -7.5, 42.42).into()),
        );

        assert!(effect.properties().iter().any(|p| p.name() == "my_prop"));
        assert!(effect.properties().iter().any(|p| p.name() == "other_prop"));
        assert!(!effect
            .properties()
            .iter()
            .any(|p| p.name() == "do_not_exist"));

        let layout = effect.property_layout();
        assert_eq!(layout.size(), 16);
        assert_eq!(layout.align(), 16);
        assert_eq!(layout.offset("my_prop"), Some(12));
        assert_eq!(layout.offset("other_prop"), Some(0));
        assert_eq!(layout.offset("unknown"), None);
    }

    #[test]
    fn test_apply_modifiers() {
        let mut module = Module::default();
        let one = module.lit(1.);
        let init_age = SetAttributeModifier::new(Attribute::AGE, one);
        let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, one);
        let init_pos_sphere = SetPositionSphereModifier {
            center: module.lit(Vec3::ZERO),
            radius: module.lit(1.),
            dimension: ShapeDimension::Volume,
        };
        let init_vel_sphere = SetVelocitySphereModifier {
            center: module.lit(Vec3::ZERO),
            speed: module.lit(1.),
        };

        let effect = EffectAsset {
            name: "Effect".into(),
            capacity: 4096,
            spawner: Spawner::rate(30.0.into()),
            ..Default::default()
        }
        .init(init_pos_sphere)
        .init(init_vel_sphere)
        //.update(AccelModifier::default())
        .update(LinearDragModifier::new(module.lit(1.)))
        .update(ForceFieldModifier::default())
        .render(ParticleTextureModifier::default())
        .render(ColorOverLifetimeModifier::default())
        .render(SizeOverLifetimeModifier::default())
        .render(OrientModifier::new(OrientMode::ParallelCameraDepthPlane))
        .render(OrientModifier::new(OrientMode::FaceCameraPosition))
        .render(OrientModifier::new(OrientMode::AlongVelocity));

        assert_eq!(effect.capacity, 4096);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut init_context = InitContext::new(&property_layout, &particle_layout);
        assert!(init_pos_sphere
            .apply_init(&mut module, &mut init_context)
            .is_ok());
        assert!(init_vel_sphere
            .apply_init(&mut module, &mut init_context)
            .is_ok());
        assert!(init_age.apply_init(&mut module, &mut init_context).is_ok());
        assert!(init_lifetime
            .apply_init(&mut module, &mut init_context)
            .is_ok());
        // assert_eq!(effect., init_context.init_code);

        let mut module = Module::default();
        let accel_mod = AccelModifier::constant(&mut module, Vec3::ONE);
        let drag_mod = LinearDragModifier::constant(&mut module, 3.5);
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut update_context = UpdateContext::new(&property_layout, &particle_layout);
        assert!(accel_mod
            .apply_update(&mut module, &mut update_context)
            .is_ok());
        assert!(drag_mod
            .apply_update(&mut module, &mut update_context)
            .is_ok());
        assert!(ForceFieldModifier::default()
            .apply_update(&mut module, &mut update_context)
            .is_ok());
        // assert_eq!(effect.update_layout, update_layout);

        let mut module = Module::default();
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut render_context = RenderContext::new(&property_layout, &particle_layout);
        ParticleTextureModifier::default().apply_render(&mut module, &mut render_context);
        ColorOverLifetimeModifier::default().apply_render(&mut module, &mut render_context);
        SizeOverLifetimeModifier::default().apply_render(&mut module, &mut render_context);
        OrientModifier::new(OrientMode::ParallelCameraDepthPlane)
            .apply_render(&mut module, &mut render_context);
        OrientModifier::new(OrientMode::FaceCameraPosition)
            .apply_render(&mut module, &mut render_context);
        OrientModifier::new(OrientMode::AlongVelocity)
            .apply_render(&mut module, &mut render_context);
        // assert_eq!(effect.render_layout, render_layout);
    }

    #[test]
    fn test_serde_ron() {
        let effect = EffectAsset {
            name: "Effect".into(),
            capacity: 4096,
            spawner: Spawner::rate(30.0.into()),
            ..Default::default()
        };

        let s = ron::to_string(&effect).unwrap();
        let _effect_serde: EffectAsset = ron::from_str(&s).unwrap();
        // assert_eq!(effect, effect_serde);
    }
}
