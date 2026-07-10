use bevy::asset::AssetPath;
use bevy::asset::{io::Reader, AssetLoader, LoadContext};
use bevy::reflect::TypeRegistry;
use bevy::reflect::{TypePath, TypeRegistryArc};
use bevy::{
    asset::{Asset, Assets, Handle},
    log::trace,
    math::{Vec2, Vec3},
    platform::collections::HashSet,
    prelude::{Component, Entity, FromWorld, Mesh, Plane3d, Resource, World},
    reflect::Reflect,
    utils::default,
};
use bevy::{ecs::reflect::AppTypeRegistry, reflect::serde::TypedReflectSerializer};
use serde::de::DeserializeSeed as _;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wgpu::{BlendComponent, BlendFactor, BlendOperation, BlendState};

use crate::Modifiers;
use crate::{
    modifier::{Modifier, RenderModifier},
    ExprHandle, ModifierContext, Module, ParticleLayout, Property, PropertyLayout, SimulationSpace,
    SpawnerSettings, TextureLayout,
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
    /// The visibility is determined by the [`InheritedVisibility`] and the
    /// [`ViewVisibility`] components, if present. The effect is assumed to be
    /// visible if those components are absent.
    ///
    /// This is the default for all assets, and is the most performant option,
    /// allowing to have many effects in the scene without the need to simulate
    /// all of them if they're not visible.
    ///
    /// Note that any [`ParticleEffect`] spawned is always compiled into a
    /// [`CompiledParticleEffect`], even when it's not visible and even when
    /// that variant is selected. That means it consumes GPU resources (memory,
    /// in particular).
    ///
    /// Note also that AABB culling is not currently available. Only boolean
    /// ON/OFF visibility is used.
    ///
    /// [`Visibility`]: bevy::camera::visibility::Visibility
    /// [`InheritedVisibility`]: bevy::camera::visibility::InheritedVisibility
    /// [`ViewVisibility`]: bevy::camera::visibility::ViewVisibility
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
    /// Any [`InheritedVisibility`] or [`ViewVisibility`] component is ignored.
    ///
    /// [`Visibility`]: bevy::camera::visibility::Visibility
    /// [`InheritedVisibility`]: bevy::camera::visibility::InheritedVisibility
    /// [`ViewVisibility`]: bevy::camera::visibility::ViewVisibility
    Always,
}

/// Alpha mode for rendering an effect.
///
/// The alpha mode determines how the alpha value of a particle is used to
/// render it. In general effects use semi-transparent particles. However, there
/// are multiple alpha blending techniques available, producing different
/// results.
///
/// This is very similar to the `bevy::prelude::AlphaMode` of the `bevy_pbr`
/// crate, except that a different set of values is supported which reflects
/// what this library currently supports.
///
/// The alpha mode only affects the render phase that particles are rendered
/// into when rendering 3D views. For 2D views, all particle effects are
/// rendered during the [`Transparent2d`] render phase.
///
/// [`Transparent2d`]: bevy::core_pipeline::core_2d::Transparent2d
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect, Serialize, Deserialize)]
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

    /// Similar to [`AlphaMode::Blend`], however assumes RGB channel values are
    /// [premultiplied](https://en.wikipedia.org/wiki/Alpha_compositing#Straight_versus_premultiplied).
    ///
    /// For otherwise constant RGB values, behaves more like
    /// [`AlphaMode::Blend`] for alpha values closer to 1.0, and more like
    /// [`AlphaMode::Add`] for alpha values closer to 0.0.
    ///
    /// Can be used to avoid “border” or “outline” artifacts that can occur
    /// when using plain alpha-blended textures.
    Premultiply,

    /// Combines the color of the fragments with the colors behind them in an
    /// additive process, (i.e. like light) producing lighter results.
    ///
    /// Black produces no effect. Alpha values can be used to modulate the
    /// result.
    ///
    /// Useful for effects like holograms, ghosts, lasers and other energy
    /// beams.
    Add,

    /// Combines the color of the fragments with the colors behind them in a
    /// multiplicative process, (i.e. like pigments) producing darker results.
    ///
    /// White produces no effect. Alpha values can be used to modulate the
    /// result.
    ///
    /// Useful for effects like stained glass, window tint film and some colored
    /// liquids.
    Multiply,

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

    /// Render the effect with no alpha, and update the depth buffer.
    ///
    /// Use this mode when every pixel covered by the particle's mesh is fully
    /// opaque.
    Opaque,
}

impl From<AlphaMode> for BlendState {
    fn from(value: AlphaMode) -> Self {
        match value {
            AlphaMode::Blend => BlendState::ALPHA_BLENDING,
            AlphaMode::Premultiply => BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            AlphaMode::Add => BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::Zero,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
            },
            AlphaMode::Multiply => BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::Dst,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent::OVER,
            },
            _ => BlendState::ALPHA_BLENDING,
        }
    }
}

/// Default particle mesh, if not otherwise specified in [`EffectAsset::mesh`].
///
/// This defaults to a unit quad facing the Z axis.
///
/// [`EffectAsset`]: crate::EffectAsset
#[derive(Debug, Clone, Resource)]
pub struct DefaultMesh(pub Handle<Mesh>);

impl FromWorld for DefaultMesh {
    fn from_world(world: &mut World) -> Self {
        let mut meshes = world.resource_mut::<Assets<Mesh>>();
        let handle = meshes.add(Plane3d::new(Vec3::Z, Vec2::splat(0.5)));
        trace!("Created DefaultMesh(Plane3d/Z): handle={handle:?}");
        Self(handle)
    }
}

/// Asset describing a visual effect.
///
/// An effect asset represents the description of an effect, intended to be
/// authored during development and instantiated once or more during the
/// application execution.
///
/// An actual effect instance can be spanwed with a [`ParticleEffect`]
/// component which references the [`EffectAsset`].
///
/// [`ParticleEffect`]: crate::ParticleEffect
/// [`EffectAsset`]: crate::EffectAsset
#[derive(Asset, Default, Clone, Reflect)]
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
    /// The CPU spawner for this effect.
    pub spawner: SpawnerSettings,
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
    /// Seed for the pseudo-random number generator.
    ///
    /// This value is used as the default value for all [`ParticleEffect`]
    /// instances based on this asset. You can override this on a per-instance
    /// basis by setting [`ParticleEffect::prng_seed`]. The resulting value
    /// is uploaded to GPU and used for the various random expressions and
    /// quantities computed in shaders.
    ///
    /// [`ParticleEffect`]: crate::ParticleEffect
    /// [`ParticleEffect::prng_seed`]: crate::ParticleEffect::prng_seed
    pub prng_seed: u32,
    /// Init modifier defining the effect.
    init_modifiers: Modifiers,
    /// Update modifiers defining the effect.
    update_modifiers: Modifiers,
    /// Render modifiers defining the effect.
    render_modifiers: Modifiers,
    /// Type of motion integration applied to the particles of a system.
    pub motion_integration: MotionIntegration,
    /// Expression module for this effect.
    module: Module,
    /// Alpha mode.
    pub alpha_mode: AlphaMode,
    /// The mesh that each particle renders.
    ///
    /// The asset path must reference a [`Mesh`] asset. If `None`, the effect
    /// uses the [`DefaultMesh`].
    ///
    /// Unlike previous versions of Hanabi, you cannot anymore assign a runtime
    /// [`Handle<Mesh>`] to an [`EffectAsset`]; although this might be
    /// convenient, this prevents serializing the asset, because not all assets
    /// in the [`Assets<Mesh>`] resource correspond to an actual asset with a
    /// path.
    pub mesh: Option<AssetPath<'static>>,
}

impl EffectAsset {
    /// Create a new effect asset.
    ///
    /// The effect assets requires 2 essential pieces:
    /// - The capacity of the effect, which represents the maximum number of
    ///   particles which can be stored and simulated at the same time for each
    ///   group. Each group has its own capacity, in number of particles. All
    ///   capacities must be non-zero and should be the smallest possible values
    ///   which allow you to author the effect. These values directly impact the
    ///   GPU memory consumption of the effect, which will allocate some buffers
    ///   to store that many particles for as long as the effect exists. The
    ///   capacities of an effect are immutable. See also [`capacity()`] for
    ///   more details.
    /// - The [`SpawnerSettings`], which defines when particles are emitted.
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
    /// let spawner = SpawnerSettings::rate(5_f32.into()); // 5 particles per second
    /// let module = Module::default();
    /// let capacity = 1024; // max 1024 particles alive at any time
    /// let effect = EffectAsset::new(capacity, spawner, module);
    /// ```
    ///
    /// Create a new effect asset with a modifier holding an expression. The
    /// expression is stored inside the [`Module`] transfered to the
    /// [`EffectAsset`], which owns the module once created.
    ///
    /// ```
    /// # use bevy_hanabi::*;
    /// let spawner = SpawnerSettings::rate(5_f32.into()); // 5 particles per second
    ///
    /// let mut module = Module::default();
    ///
    /// // Create a modifier that initialized the particle lifetime to 10 seconds.
    /// let lifetime = module.lit(10.); // literal value "10.0"
    /// let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);
    ///
    /// let capacity = 1024; // max 1024 particles alive at any time
    /// let effect = EffectAsset::new(capacity, spawner, module);
    /// ```
    ///
    /// [`capacity()`]: self::EffectAsset::capacity
    /// [`Expr`]: crate::graph::expr::Expr
    pub fn new(capacity: u32, spawner: SpawnerSettings, module: Module) -> Self {
        Self {
            capacity,
            spawner,
            module,
            ..default()
        }
    }

    /// Get the capacity of the effect, in number of particles.
    ///
    /// This value represents the number of particles stored in GPU memory at
    /// all times, even if unused, so you should try to minimize it.
    /// However, the library cannot emit more particles than the effect
    /// capacity. Whatever the spawner settings, if the number of particles
    /// reaches the capacity, no new particle can be emitted. Choosing an
    /// appropriate capacity for an effect is therefore a compromise between
    /// more particles available for visuals and more GPU memory usage.
    ///
    /// Common values range from 256 or less for smaller effects, to several
    /// hundreds of thousands for unique effects consuming a large portion of
    /// the GPU memory budget. Hanabi has been tested with over a million
    /// particles, however the performance will largely depend on the actual GPU
    /// hardware and available memory, so authors are encouraged not to go too
    /// crazy with the capacity.
    ///
    /// [`EffectSpawner`]: crate::EffectSpawner
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

    /// Set the effect's motion integration.
    pub fn with_motion_integration(mut self, motion_integration: MotionIntegration) -> Self {
        self.motion_integration = motion_integration;
        self
    }

    /// Get the list of existing properties.
    ///
    /// This is a shortcut for `self.module().properties()`.
    pub fn properties(&self) -> &[Property] {
        self.module.properties()
    }

    /// Add an initialization modifier to the effect.
    ///
    /// Initialization modifiers apply to all particles that are spawned or
    /// cloned.
    ///
    /// # Panics
    ///
    /// Panics if the modifier doesn't support the init context (that is,
    /// `modifier.context()` returns a flag which doesn't include
    /// [`ModifierContext::Init`]).
    #[inline]
    pub fn init<M>(mut self, modifier: M) -> Self
    where
        M: Modifier + Send + Sync,
    {
        assert!(modifier.context().contains(ModifierContext::Init));
        self.init_modifiers.push(Box::new(modifier));
        self
    }

    /// Add an update modifier to the effect.
    ///
    /// # Panics
    ///
    /// Panics if the modifier doesn't support the update context (that is,
    /// `modifier.context()` returns a flag which doesn't include
    /// [`ModifierContext::Update`]).
    #[inline]
    pub fn update<M>(mut self, modifier: M) -> Self
    where
        M: Modifier + Send + Sync,
    {
        assert!(modifier.context().contains(ModifierContext::Update));
        self.update_modifiers.push(Box::new(modifier));
        self
    }

    /// Add a [`BoxedModifier`] to the specific context.
    ///
    /// # Panics
    ///
    /// Panics if the context is [`ModifierContext::Render`]; use
    /// [`add_render_modifier()`] instead.
    ///
    /// Panics if the input `context` contains more than one context (the
    /// bitfield contains more than 1 bit set) or no context at all (zero bit
    /// set).
    ///
    /// Panics if the modifier doesn't support the context specified (that is,
    /// `modifier.context()` returns a flag which doesn't include `context`).
    ///
    /// [`BoxedModifier`]: crate::BoxedModifier
    /// [`add_render_modifier()`]: crate::EffectAsset::add_render_modifier
    pub fn add_modifier(mut self, context: ModifierContext, modifier: Box<dyn Modifier>) -> Self {
        assert!(context == ModifierContext::Init || context == ModifierContext::Update);
        assert!(modifier.context().contains(context));
        if context == ModifierContext::Init {
            self.init_modifiers.push(modifier);
        } else {
            self.update_modifiers.push(modifier);
        }
        self
    }

    /// Add a render modifier to the effect.
    ///
    /// # Panics
    ///
    /// Panics if the modifier doesn't support the render context (that is,
    /// `modifier.context()` returns a flag which doesn't include
    /// [`ModifierContext::Render`]).
    #[inline]
    pub fn render<M>(mut self, modifier: M) -> Self
    where
        M: RenderModifier + Send + Sync,
    {
        assert!(modifier.context().contains(ModifierContext::Render));
        self.render_modifiers.push(Box::new(modifier));
        self
    }

    /// Add a [`RenderModifier`] to the render context.
    ///
    /// # Panics
    ///
    /// Panics if the modifier doesn't support the render context (that is,
    /// `modifier.context()` returns a flag which doesn't include
    /// [`ModifierContext::Render`]).
    pub fn add_render_modifier(mut self, modifier: Box<dyn RenderModifier>) -> Self {
        assert!(modifier.context().contains(ModifierContext::Render));
        self.render_modifiers.push(modifier.boxed_clone());
        self
    }

    /// Get a list of all the modifiers of this effect.
    pub fn modifiers(&self) -> impl Iterator<Item = &dyn Modifier> {
        self.init_modifiers
            .iter()
            .map(|bm| &**bm)
            .chain(self.update_modifiers.iter().map(|bm| &**bm))
            .chain(self.render_modifiers.iter().map(|bm| &**bm))
    }

    /// Get a list of all the init modifiers of this effect.
    ///
    /// This is a filtered list of all modifiers, retaining only modifiers
    /// executing in the [`ModifierContext::Init`] context.
    ///
    /// [`ModifierContext::Init`]: crate::ModifierContext::Init
    pub fn init_modifiers(&self) -> impl Iterator<Item = &dyn Modifier> {
        self.init_modifiers.iter().map(|bm| &**bm)
    }

    /// Get a list of all the update modifiers of this effect.
    ///
    /// This is a filtered list of all modifiers, retaining only modifiers
    /// executing in the [`ModifierContext::Update`] context.
    ///
    /// [`ModifierContext::Update`]: crate::ModifierContext::Update
    pub fn update_modifiers(&self) -> impl Iterator<Item = &dyn Modifier> {
        self.update_modifiers.iter().map(|bm| &**bm)
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
    /// currently existing modifiers, and return it as a newly allocated
    /// [`ParticleLayout`] object.
    pub fn particle_layout(&self) -> ParticleLayout {
        // Build the set of unique attributes required for all modifiers
        let mut set = HashSet::new();
        for modifier in self.modifiers() {
            for &attr in modifier.attributes() {
                set.insert(attr);
            }
        }

        // Add all attributes used by expressions. Those are indirectly used by
        // modifiers, but may not have been added directly yet.
        self.module.gather_attributes(&mut set);

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
        PropertyLayout::new(self.properties().iter())
    }

    /// Get the texture layout of the module of this effect.
    pub fn texture_layout(&self) -> TextureLayout {
        self.module.texture_layout()
    }

    /// Sets the mesh that each particle will render.
    ///
    /// Note: this asset path must reference a [`Mesh`] asset.
    pub fn mesh(mut self, mesh: AssetPath<'static>) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Serialize this effect asset.
    ///
    /// This uses the canonical Hanabi serialization format, which is internally
    /// based on RON (implementation detail). The type registry must contain all
    /// types this [`EffectAsset`] references, including all concrete types of
    /// [`Modifier`] objects. In general, you should pass the app's own
    /// [`TypeRegistry`] found in the [`AppTypeRegistry`] resource.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy::prelude::*;
    /// # use bevy_hanabi::*;
    /// fn serialize_effect(type_registry: Res<AppTypeRegistry>) {
    ///     let asset = EffectAsset::default();
    ///     // [...]
    ///     let type_registry = type_registry.read();
    ///     let s = asset.serialize(&type_registry).unwrap();
    ///     // [...]
    /// }
    /// ```
    ///
    /// # Advanced
    ///
    /// For more advanced serialization, for example to another format, see also
    /// the [`EffectAssetSerializer`] which implements [`serde::Serialize`].
    pub fn serialize(&self, type_registry: &TypeRegistry) -> Result<String, ron::Error> {
        let serializer = EffectAssetSerializer::new(self, type_registry);
        let pretty_config = ron::ser::PrettyConfig::default()
            .indentor("  ".to_string())
            .new_line("\n".to_string());
        ron::ser::to_string_pretty(&serializer, pretty_config)
    }

    /// Deserialize an effect asset from string.
    ///
    /// This uses the canonical Hanabi serialization format, which is internally
    /// based on RON (implementation detail). The type registry must contain all
    /// types the serialized [`EffectAsset`] references, including all concrete
    /// types of [`Modifier`] objects. In general, you should pass the app's
    /// own [`TypeRegistry`] found in the [`AppTypeRegistry`] resource.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy::prelude::*;
    /// # use bevy_hanabi::*;
    /// # fn get_asset_string() -> String { unimplemented!() }
    /// fn deserialize_effect(type_registry: Res<AppTypeRegistry>) {
    ///     // [...]
    ///     let s: String = get_asset_string();
    ///     let type_registry = type_registry.read();
    ///     let asset = EffectAsset::deserialize(&s[..], &type_registry).unwrap();
    ///     // [...]
    /// }
    /// ```
    ///
    /// # Advanced
    ///
    /// For more advanced deserialization, for example to another format, see
    /// also the [`EffectAssetDeserializer`] which implements
    /// [`serde::de::DeserializeSeed`].
    pub fn deserialize(s: &str, type_registry: &TypeRegistry) -> Result<Self, ron::Error> {
        let mut deserializer = ron::de::Deserializer::from_str(s)?;
        let deserialize = EffectAssetDeserializer::new(type_registry);
        let asset = deserialize.deserialize(&mut deserializer)?;
        Ok(asset)
    }
}

impl bevy::reflect::serde::SerializeWithRegistry for EffectAsset {
    fn serialize<S>(&self, serializer: S, registry: &TypeRegistry) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct as _;

        let mut s = serializer.serialize_struct("EffectAsset", 14)?;
        s.serialize_field("name", &self.name)?;
        s.serialize_field("capacity", &self.capacity)?;
        s.serialize_field("spawner", &self.spawner)?;
        s.serialize_field("z_layer_2d", &self.z_layer_2d)?;
        s.serialize_field("simulation_space", &self.simulation_space)?;
        s.serialize_field("simulation_condition", &self.simulation_condition)?;
        s.serialize_field("prng_seed", &self.prng_seed)?;
        s.serialize_field(
            "init_modifiers",
            &TypedReflectSerializer::new(&self.init_modifiers, registry),
        )?;
        s.serialize_field(
            "update_modifiers",
            &TypedReflectSerializer::new(&self.update_modifiers, registry),
        )?;
        s.serialize_field(
            "render_modifiers",
            &TypedReflectSerializer::new(&self.render_modifiers, registry),
        )?;
        s.serialize_field("motion_integration", &self.motion_integration)?;
        s.serialize_field("module", &self.module)?;
        s.serialize_field("alpha_mode", &self.alpha_mode)?;
        s.serialize_field("mesh", &self.mesh)?;
        s.end()
    }
}

impl<'de> bevy::reflect::serde::DeserializeWithRegistry<'de> for EffectAsset {
    fn deserialize<D>(deserializer: D, registry: &TypeRegistry) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Name,
            Capacity,
            Spawner,
            #[serde(rename = "z_layer_2d")]
            ZLayer2d,
            SimulationSpace,
            SimulationCondition,
            PrngSeed,
            InitModifiers,
            UpdateModifiers,
            RenderModifiers,
            MotionIntegration,
            Module,
            AlphaMode,
            Mesh,
        }

        struct SerializedEffectAssetVisitor<'a> {
            pub registry: &'a TypeRegistry,
        }

        impl<'a, 'de> serde::de::Visitor<'de> for SerializedEffectAssetVisitor<'a> {
            type Value = EffectAsset;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct EffectAsset")
            }

            fn visit_map<V>(self, mut map: V) -> Result<EffectAsset, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let modifiers = self
                    .registry
                    .get(std::any::TypeId::of::<Modifiers>())
                    .ok_or_else(|| {
                        serde::de::Error::custom("Failed to find type registration for Modifiers.")
                    })?;

                let mut name = None;
                let mut capacity = None;
                let mut spawner = None;
                let mut z_layer_2d = None;
                let mut simulation_space = None;
                let mut simulation_condition = None;
                let mut prng_seed = None;
                let mut init_modifiers = None;
                let mut update_modifiers = None;
                let mut render_modifiers = None;
                let mut motion_integration = None;
                let mut module = None;
                let mut alpha_mode = None;
                let mut mesh = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Name => {
                            if name.is_some() {
                                return Err(serde::de::Error::duplicate_field("name"));
                            }
                            name = Some(map.next_value()?);
                        }
                        Field::Capacity => {
                            if capacity.is_some() {
                                return Err(serde::de::Error::duplicate_field("capacity"));
                            }
                            capacity = Some(map.next_value()?);
                        }
                        Field::Spawner => {
                            if spawner.is_some() {
                                return Err(serde::de::Error::duplicate_field("spawner"));
                            }
                            spawner = Some(map.next_value()?);
                        }
                        Field::ZLayer2d => {
                            if z_layer_2d.is_some() {
                                return Err(serde::de::Error::duplicate_field("z_layer_2d"));
                            }
                            z_layer_2d = Some(map.next_value()?);
                        }
                        Field::SimulationSpace => {
                            if simulation_space.is_some() {
                                return Err(serde::de::Error::duplicate_field("simulation_space"));
                            }
                            simulation_space = Some(map.next_value()?);
                        }
                        Field::SimulationCondition => {
                            if simulation_condition.is_some() {
                                return Err(serde::de::Error::duplicate_field(
                                    "simulation_condition",
                                ));
                            }
                            simulation_condition = Some(map.next_value()?);
                        }
                        Field::PrngSeed => {
                            if prng_seed.is_some() {
                                return Err(serde::de::Error::duplicate_field("prng_seed"));
                            }
                            prng_seed = Some(map.next_value()?);
                        }
                        Field::InitModifiers => {
                            if init_modifiers.is_some() {
                                return Err(serde::de::Error::duplicate_field("init_modifiers"));
                            }
                            init_modifiers = Some(map.next_value_seed(
                                bevy::reflect::serde::TypedReflectDeserializer::new(
                                    modifiers,
                                    self.registry,
                                ),
                            )?);
                        }
                        Field::UpdateModifiers => {
                            if update_modifiers.is_some() {
                                return Err(serde::de::Error::duplicate_field("update_modifiers"));
                            }
                            update_modifiers = Some(map.next_value_seed(
                                bevy::reflect::serde::TypedReflectDeserializer::new(
                                    modifiers,
                                    self.registry,
                                ),
                            )?);
                        }
                        Field::RenderModifiers => {
                            if render_modifiers.is_some() {
                                return Err(serde::de::Error::duplicate_field("render_modifiers"));
                            }
                            render_modifiers = Some(map.next_value_seed(
                                bevy::reflect::serde::TypedReflectDeserializer::new(
                                    modifiers,
                                    self.registry,
                                ),
                            )?);
                        }
                        Field::MotionIntegration => {
                            if motion_integration.is_some() {
                                return Err(serde::de::Error::duplicate_field(
                                    "motion_integration",
                                ));
                            }
                            motion_integration = Some(map.next_value()?);
                        }
                        Field::Module => {
                            if module.is_some() {
                                return Err(serde::de::Error::duplicate_field("module"));
                            }
                            module = Some(map.next_value()?);
                        }
                        Field::AlphaMode => {
                            if alpha_mode.is_some() {
                                return Err(serde::de::Error::duplicate_field("alpha_mode"));
                            }
                            alpha_mode = Some(map.next_value()?);
                        }
                        Field::Mesh => {
                            if mesh.is_some() {
                                return Err(serde::de::Error::duplicate_field("mesh"));
                            }
                            mesh = Some(map.next_value()?);
                        }
                    }
                }

                // Recover the concrete field objects
                let name = name.ok_or_else(|| serde::de::Error::missing_field("name"))?;
                let capacity =
                    capacity.ok_or_else(|| serde::de::Error::missing_field("capacity"))?;
                let spawner = spawner.ok_or_else(|| serde::de::Error::missing_field("spawner"))?;
                let z_layer_2d =
                    z_layer_2d.ok_or_else(|| serde::de::Error::missing_field("z_layer_2d"))?;
                let simulation_space = simulation_space
                    .ok_or_else(|| serde::de::Error::missing_field("simulation_space"))?;
                let simulation_condition = simulation_condition
                    .ok_or_else(|| serde::de::Error::missing_field("simulation_condition"))?;
                let prng_seed =
                    prng_seed.ok_or_else(|| serde::de::Error::missing_field("prng_seed"))?;
                let motion_integration = motion_integration
                    .ok_or_else(|| serde::de::Error::missing_field("motion_integration"))?;
                let module = module.ok_or_else(|| serde::de::Error::missing_field("module"))?;
                let alpha_mode =
                    alpha_mode.ok_or_else(|| serde::de::Error::missing_field("alpha_mode"))?;
                let mesh = mesh.ok_or_else(|| serde::de::Error::missing_field("mesh"))?;
                // Modifiers uses ReflectModifier type data to construct a concrete type. So we
                // can directly try_take() here from the PartialReflect to recover that concrete
                // object. This should always succeed.
                let init_modifiers = init_modifiers
                    .map(|m| m.try_take::<Modifiers>())
                    .transpose()
                    .map_err(|_| serde::de::Error::custom("Failed to get Modifiers"))?
                    .unwrap_or_default();
                let update_modifiers = update_modifiers
                    .map(|m| m.try_take::<Modifiers>())
                    .transpose()
                    .map_err(|_| serde::de::Error::custom("Failed to get Modifiers"))?
                    .unwrap_or_default();
                let render_modifiers = render_modifiers
                    .map(|m| m.try_take::<Modifiers>())
                    .transpose()
                    .map_err(|_| serde::de::Error::custom("Failed to get Modifiers"))?
                    .unwrap_or_default();

                // Rebuild the concrete EffectAsset object
                Ok(EffectAsset {
                    name,
                    capacity,
                    spawner,
                    z_layer_2d,
                    simulation_space,
                    simulation_condition,
                    prng_seed,
                    init_modifiers,
                    update_modifiers,
                    render_modifiers,
                    motion_integration,
                    module,
                    alpha_mode,
                    mesh,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "name",
            "capacity",
            "spawner",
            "z_layer_2d",
            "simulation_space",
            "simulation_condition",
            "prng_seed",
            "init_modifiers",
            "update_modifiers",
            "render_modifiers",
            "motion_integration",
            "module",
            "alpha_mode",
            "mesh",
        ];
        deserializer.deserialize_struct(
            "EffectAsset",
            FIELDS,
            SerializedEffectAssetVisitor { registry },
        )
    }
}

/// Serializer for an [`EffectAsset`].
pub struct EffectAssetSerializer<'a> {
    asset: &'a EffectAsset,
    type_registry: &'a TypeRegistry,
}

impl<'a> EffectAssetSerializer<'a> {
    /// Create a new serializer for a given [`EffectAsset`].
    ///
    /// The `type_registry` must contain all types referenced by the
    /// [`EffectAsset`], and in particular all concrete [`Modifier`] types.
    pub fn new(asset: &'a EffectAsset, type_registry: &'a TypeRegistry) -> Self {
        Self {
            asset,
            type_registry,
        }
    }
}

impl<'a> serde::Serialize for EffectAssetSerializer<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use bevy::reflect::serde::SerializeWithRegistry;

        <EffectAsset as SerializeWithRegistry>::serialize(
            self.asset,
            serializer,
            self.type_registry,
        )
    }
}

/// Deserializer for an [`EffectAsset`].
pub struct EffectAssetDeserializer<'a> {
    type_registry: &'a TypeRegistry,
}

impl<'a> EffectAssetDeserializer<'a> {
    /// Create a new deserializer for [`EffectAsset`].
    ///
    /// The `type_registry` must contain all types that could be referenced by
    /// the [`EffectAsset`] to deserialize, and in particular all concrete
    /// [`Modifier`] types.
    pub fn new(type_registry: &'a TypeRegistry) -> Self {
        Self { type_registry }
    }
}

impl<'a, 'de> serde::de::DeserializeSeed<'de> for EffectAssetDeserializer<'a> {
    type Value = EffectAsset;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        <EffectAsset as bevy::reflect::serde::DeserializeWithRegistry>::deserialize(
            deserializer,
            self.type_registry,
        )
    }
}

/// Asset loader for [`EffectAsset`].
///
/// Effet assets take the `.effect` extension.
#[derive(Debug, TypePath)]
pub struct EffectAssetLoader {
    /// The type registry passed to [`EffectAsset::deserialize()`].
    pub type_registry: TypeRegistryArc,
}

impl FromWorld for EffectAssetLoader {
    fn from_world(world: &mut World) -> Self {
        let type_registry = world.resource::<AppTypeRegistry>();
        EffectAssetLoader {
            type_registry: type_registry.0.clone(),
        }
    }
}

/// Error for the [`EffectAssetLoader`] loading an [`EffectAsset`].
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum EffectAssetLoaderError {
    /// I/O error reading the asset source.
    #[error("An IO error occurred during loading of a particle effect")]
    Io(#[from] std::io::Error),

    /// UTF-8 error converting the asset serialized content.
    #[error("An encoding error occurred during loading of a particle effect")]
    Encoding(#[from] std::string::FromUtf8Error),

    /// Error during RON format parsing.
    #[error("A RON format error occurred during loading of a particle effect")]
    RonSpan(#[from] ron::error::SpannedError),

    /// Error during RON format parsing.
    #[error("A RON error occurred during loading of a particle effect")]
    Ron(#[from] ron::error::Error),
}

impl AssetLoader for EffectAssetLoader {
    type Asset = EffectAsset;

    type Settings = ();

    type Error = EffectAssetLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let s = String::from_utf8(bytes)?;
        let type_registry = self.type_registry.read();
        let asset = EffectAsset::deserialize(&s, &type_registry)?;
        Ok(asset)
    }

    fn extensions(&self) -> &[&str] {
        &["effect"]
    }
}

/// Component defining the parent effect of the current effect.
///
/// This component is optional. When present, on the same entity as the
/// [`ParticleEffect`], it defines the "parent effect" of that effect. The
/// particles of the parent effect are accessible from the init pass of this
/// effect, to allow the particles from the current effect to inherit some
/// attributes (position, velocity, ...) from the parent particle which
/// triggered its spawning via GPU spawn events.
///
/// Adding this component automatically makes the current particle effect
/// instance use GPU spawn events emitted by its parent, and automatically makes
/// the parent effect instance emits such events.
///
/// An effect has at most one parent, defined by this component, but a parent
/// effect can have multiple children. For example, a parent effect can emit GPU
/// spawn events continuously ([`EventEmitCondition::Always`]) to generate some
/// kind of trail, and also emit GPU spawn events when its particles die
/// ([`EventEmitCondition::OnDie`]) for any explosion-like effect.
///
/// [`ParticleEffect`]: crate::ParticleEffect
/// [`EventEmitCondition::Always`]: crate::EventEmitCondition::Always
/// [`EventEmitCondition::OnDie`]: crate::EventEmitCondition::OnDie
#[derive(Debug, Clone, Copy, Component, Reflect)]
pub struct EffectParent {
    /// Entity of the parent effect.
    pub entity: Entity,
}

impl EffectParent {
    /// Create a new component with the given entity as parent.
    pub fn new(parent: Entity) -> Self {
        Self { entity: parent }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize)]
pub struct ParticleTrails {
    pub spawn_period: f32,
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use bevy::{
        asset::{
            io::{
                memory::{Dir, MemoryAssetReader, MemoryAssetWriter},
                AssetSourceBuilder, AssetSourceId,
            },
            LoadState,
        },
        diagnostic::DiagnosticsPlugin,
    };

    use super::*;
    use crate::*;

    #[test]
    fn add_modifiers() {
        let mut m = Module::default();
        let expr = m.lit(3.);

        for modifier_context in [ModifierContext::Init, ModifierContext::Update] {
            let effect = EffectAsset::default().add_modifier(
                modifier_context,
                Box::new(SetAttributeModifier::new(Attribute::POSITION, expr)),
            );
            assert_eq!(effect.modifiers().count(), 1);
            let m = effect.modifiers().next().unwrap();
            assert!(m.context().contains(modifier_context));
        }

        {
            let effect = EffectAsset::default().add_render_modifier(Box::new(SetColorModifier {
                color: CpuValue::Single(Vec4::ONE),
                blend: ColorBlendMode::Overwrite,
                mask: ColorBlendMask::RGBA,
            }));
            assert_eq!(effect.modifiers().count(), 1);
            let m = effect.modifiers().next().unwrap();
            assert!(m.context().contains(ModifierContext::Render));
        }
    }

    #[test]
    fn test_apply_modifiers() {
        let mut module = Module::default();
        let origin = module.lit(Vec3::ZERO);
        let one = module.lit(1.);
        let slot_zero = module.lit(0u32);
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

        let mut effect = EffectAsset::new(4096, SpawnerSettings::rate(30.0.into()), module)
            .init(init_pos_sphere)
            .init(init_vel_sphere)
            //.update(AccelModifier::default())
            .update(LinearDragModifier::new(one))
            .update(ConformToSphereModifier::new(origin, one, one, one, one))
            .render(ParticleTextureModifier::new(slot_zero))
            .render(ColorOverLifetimeModifier::default())
            .render(SizeOverLifetimeModifier::default())
            .render(OrientModifier::new(OrientMode::ParallelCameraDepthPlane))
            .render(OrientModifier::new(OrientMode::FaceCameraPosition))
            .render(OrientModifier::new(OrientMode::AlongVelocity));

        assert_eq!(effect.capacity, 4096);

        let module = &mut effect.module;
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut init_context =
            ShaderWriter::new(ModifierContext::Init, &property_layout, &particle_layout);
        assert!(init_pos_sphere.apply(module, &mut init_context).is_ok());
        assert!(init_vel_sphere.apply(module, &mut init_context).is_ok());
        assert!(init_age.apply(module, &mut init_context).is_ok());
        assert!(init_lifetime.apply(module, &mut init_context).is_ok());
        // assert_eq!(effect., init_context.init_code);

        let accel_mod = AccelModifier::constant(module, Vec3::ONE);
        let drag_mod = LinearDragModifier::constant(module, 3.5);
        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let mut update_context =
            ShaderWriter::new(ModifierContext::Update, &property_layout, &particle_layout);
        assert!(accel_mod.apply(module, &mut update_context).is_ok());
        assert!(drag_mod.apply(module, &mut update_context).is_ok());
        assert!(ConformToSphereModifier::new(origin, one, one, one, one)
            .apply(module, &mut update_context)
            .is_ok());
        // assert_eq!(effect.update_layout, update_layout);

        let property_layout = PropertyLayout::default();
        let particle_layout = ParticleLayout::default();
        let texture_layout = TextureLayout::default();
        let mut render_context =
            RenderContext::new(&property_layout, &particle_layout, &texture_layout);
        ParticleTextureModifier::new(slot_zero)
            .apply_render(module, &mut render_context)
            .unwrap();
        ColorOverLifetimeModifier::default()
            .apply_render(module, &mut render_context)
            .unwrap();
        SizeOverLifetimeModifier::default()
            .apply_render(module, &mut render_context)
            .unwrap();
        OrientModifier::new(OrientMode::ParallelCameraDepthPlane)
            .apply_render(module, &mut render_context)
            .unwrap();
        OrientModifier::new(OrientMode::FaceCameraPosition)
            .apply_render(module, &mut render_context)
            .unwrap();
        OrientModifier::new(OrientMode::AlongVelocity)
            .apply_render(module, &mut render_context)
            .unwrap();
        // assert_eq!(effect.render_layout, render_layout);
    }

    /// Round-trip EffectAsset through its own functions serialize() and
    /// deserialize().
    #[test]
    fn serde_asset() {
        let w = ExprWriter::new();

        let pos = w.lit(Vec3::new(1.2, -3.45, 87.54485));
        let x = w.lit(BVec2::new(false, true));
        let _ = x + pos.clone();
        let mod_pos = SetAttributeModifier::new(Attribute::POSITION, pos.expr());

        let mut module = w.finish();
        let prop = module.add_property("my_prop", Vec3::new(1.2, -2.3, 55.32).into());
        let prop = module.prop(prop);
        let _ = module.abs(prop);

        let effect = EffectAsset {
            name: "Effect".into(),
            capacity: 4096,
            spawner: SpawnerSettings::rate(30.0.into()),
            module,
            z_layer_2d: 1.5,
            simulation_space: SimulationSpace::Local,
            simulation_condition: SimulationCondition::Always,
            prng_seed: 4284,
            motion_integration: MotionIntegration::PreUpdate,
            alpha_mode: AlphaMode::Multiply,
            ..Default::default()
        }
        .init(mod_pos);

        let type_registry = AppTypeRegistry::new_with_derived_types();
        register_modifiers(&type_registry);
        let registry = type_registry.read();

        // Round-trip
        let s = effect.serialize(&registry).unwrap();
        eprintln!("{}", s);
        let effect_serde = EffectAsset::deserialize(&s, &registry).unwrap();

        assert_eq!(effect.name, effect_serde.name);
        assert_eq!(effect.capacity, effect_serde.capacity);
        assert_eq!(effect.spawner, effect_serde.spawner);
        assert_eq!(effect.z_layer_2d, effect_serde.z_layer_2d);
        assert_eq!(effect.simulation_space, effect_serde.simulation_space);
        assert_eq!(
            effect.simulation_condition,
            effect_serde.simulation_condition
        );
        assert_eq!(effect.motion_integration, effect_serde.motion_integration);
        assert_eq!(effect.module, effect_serde.module);
        assert_eq!(effect.alpha_mode, effect_serde.alpha_mode);
        assert_eq!(
            effect.init_modifiers().count(),
            effect_serde.init_modifiers().count()
        );
        assert_eq!(
            effect.update_modifiers().count(),
            effect_serde.update_modifiers().count()
        );
        assert_eq!(
            effect.render_modifiers().count(),
            effect_serde.render_modifiers().count()
        );
    }

    #[test]
    fn alpha_mode_blend_state() {
        assert_eq!(BlendState::ALPHA_BLENDING, AlphaMode::Blend.into());
        assert_eq!(
            BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            AlphaMode::Premultiply.into()
        );

        let blend_state = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
        };
        assert_eq!(blend_state, AlphaMode::Add.into());

        let blend_state = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::Dst,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent::OVER,
        };
        assert_eq!(blend_state, AlphaMode::Multiply.into());

        let expr = Module::default().lit(0.5);
        assert_eq!(BlendState::ALPHA_BLENDING, AlphaMode::Mask(expr).into());
    }

    // Regression test for #440
    #[test]
    fn transitive_attr() {
        let mut m = Module::default();
        let age = m.attr(Attribute::F32_0);
        let modifier = SetAttributeModifier::new(Attribute::AGE, age);
        let asset = EffectAsset::new(32, SpawnerSettings::once(3.0.into()), m).init(modifier);
        let particle_layout = asset.particle_layout();
        assert!(particle_layout.contains(Attribute::AGE)); // direct
        assert!(particle_layout.contains(Attribute::F32_0)); // transitive
    }

    /// Creates a basic asset app and an in-memory file system.
    fn create_app() -> (App, Dir) {
        let mut app = App::new();
        let dir = Dir::default();
        let dir1 = dir.clone();
        let dir2 = dir.clone();
        app.register_asset_source(
            AssetSourceId::Default,
            AssetSourceBuilder::new(move || Box::new(MemoryAssetReader { root: dir1.clone() }))
                .with_writer(move |_| Some(Box::new(MemoryAssetWriter { root: dir2.clone() }))),
        )
        .add_plugins((
            TaskPoolPlugin::default(),
            AssetPlugin {
                watch_for_changes_override: Some(false),
                use_asset_processor_override: Some(false),
                ..Default::default()
            },
            DiagnosticsPlugin,
        ));
        (app, dir)
    }

    fn run_app_until(app: &mut App, mut predicate: impl FnMut(&mut World) -> Option<()>) {
        for _ in 0..10_000 {
            app.update();
            if predicate(app.world_mut()).is_some() {
                return;
            }
        }
        panic!("Ran out of loops to return `Some` from `predicate`");
    }

    /// Check that the [`EffectAssetLoader`] can load an asset.
    #[test]
    fn loader() {
        let (mut app, dir) = create_app();
        app.init_asset::<EffectAsset>()
            .init_asset_loader::<EffectAssetLoader>();

        // Create an effect, serialize it, and write it to the in-memory file system
        let asset_ref = {
            let type_registry = app.world().resource::<AppTypeRegistry>().clone();
            let type_registry = type_registry.read();
            let spawner = SpawnerSettings::rate(3.0.into());
            let module = Module::default();
            let effect = EffectAsset::new(256, spawner, module);
            let s = effect.serialize(&type_registry).unwrap();
            dir.insert_asset_text(Path::new("test.effect"), &s[..]);
            effect
        };

        // Load the asset through the AssetServer
        let asset_server = app.world().resource::<AssetServer>().clone();
        let handle = asset_server.load::<EffectAsset>("test.effect");
        run_app_until(&mut app, |world| {
            let asset_server = world.resource::<AssetServer>();
            match asset_server.get_load_state(&handle).unwrap() {
                LoadState::NotLoaded | LoadState::Loading => None,
                LoadState::Loaded => Some(()),
                LoadState::Failed(err) => panic!("Failed to load asset: {err:?}"),
            }
        });

        // Compare the loaded asset deserialized from (in-memory) "disk", to the
        // original reference asset.
        let asset = app
            .world()
            .resource::<Assets<EffectAsset>>()
            .get(&handle)
            .unwrap();
        assert_eq!(asset.capacity(), asset_ref.capacity());
        assert_eq!(asset.spawner, asset_ref.spawner);
    }
}
