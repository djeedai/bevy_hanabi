use std::ops::Deref;

#[cfg(feature = "serde")]
use bevy::asset::{io::Reader, AssetLoader, LoadContext};
use bevy::{
    asset::{Asset, Handle},
    prelude::Mesh,
    reflect::Reflect,
    utils::{default, HashSet},
};
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde")]
use thiserror::Error;
use wgpu::{BlendComponent, BlendFactor, BlendOperation, BlendState};

use crate::{
    modifier::{Modifier, RenderModifier},
    spawn::{Cloner, Initializer},
    Attribute, CpuValue, ExprHandle, GroupedModifier, ModifierContext, Module, ParticleGroupSet,
    ParticleLayout, Property, PropertyLayout, SimulationSpace, Spawner, TextureLayout,
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

/// Asset describing a visual effect.
///
/// An effect asset represents the description of an effect, intended to be
/// authored during development and instantiated once or more during the
/// application execution.
///
/// An actual effect instance can be spanwed with a [`ParticleEffect`]
/// component, or a [`ParticleEffectBundle`], which references the
/// [`EffectAsset`].
///
/// # Groups, trails, and ribbons
///
/// Typically, an effect asset describes a single type of particles. At this
/// time, 🎆 Hanabi doesn't yet support complex effects involving multiple
/// sub-effects (sometimes called _systems_ in some other engines). This means
/// most parameters relating to an effect asset affect all particles.
///
/// However, for technical reasons, the implementation of trails and ribbons
/// requires treating different groups of particles in a different way.
/// - Trails refer to the visual effect of a group of particles appearing to
///   follow each other, leaving a visual trail. The implementation in fact
///   doesn't update the position of those particles; instead, it spawns at
///   regular interval a new particle, while older particles die after reaching
///   their lifetime. To give the appearance of a trail, the newly spawned
///   particles are _cloned_ from an existing "head" particle. Because those two
///   kinds of particles need to be treated differently by the implementation,
///   they are split into separate groups inside the same effect.
/// - Ribbons refer to a similar visual effect as trails, but in addition
///   particles are rendered by stitching consecutive trail particles together
///   to form a continuous visual trail called a ribbon. To achieve this, the
///   implementation needs to chain particles together and keep track of the
///   previous and/or next particle of each particle. This is achieved via the
///   [`Attribute::PREV`] and [`Attribute::NEXT`] attributes, stored per
///   particle. Because each particle can only store one set of attributes, this
///   means there can only be one ribbon per effect.
///
/// In general, groups are largely a technical implementation detail for trails
/// and ribbons, and you should simply rely on helper functions like
/// [`with_trails()`] or [`with_ribbons()`]. Groups were first introduced with a
/// powerful but complex API, which has been since then greatly simplified, with
/// the intent to completely hide/eliminate them in the future.
///
/// [`ParticleEffect`]: crate::ParticleEffect
/// [`ParticleEffectBundle`]: crate::ParticleEffectBundle
/// [`EffectAsset`]: crate::EffectAsset
/// [`with_trails()`]: crate::EffectAsset::with_trails
/// [`with_ribbons()`]: crate::EffectAsset::with_ribbons
#[derive(Asset, Default, Clone, Reflect)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    capacities: Vec<u32>,
    /// The initializer for each group.
    ///
    /// Each initializer contains either a spawner or a cloner.
    pub init: Vec<Initializer>,
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
    init_modifiers: Vec<GroupedModifier>,
    /// update modifiers defining the effect.
    #[reflect(ignore)]
    // TODO - Can't manage to implement FromReflect for BoxedModifier in a nice way yet
    update_modifiers: Vec<GroupedModifier>,
    /// Render modifiers defining the effect.
    #[reflect(ignore)]
    // TODO - Can't manage to implement FromReflect for BoxedModifier in a nice way yet
    render_modifiers: Vec<GroupedModifier>,
    /// Type of motion integration applied to the particles of a system.
    pub motion_integration: MotionIntegration,
    /// Expression module for this effect.
    module: Module,
    /// Alpha mode.
    pub alpha_mode: AlphaMode,
    /// The mesh that each particle renders.
    ///
    /// This defaults to a quad facing the Z axis.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub mesh: Option<Handle<Mesh>>,
    /// Which group is to render as ribbons.
    ///
    /// There can be only one such group, because there's only one set of
    /// next/previous pointers.
    pub ribbon_group: Option<usize>,
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
    ///   capacities of an effect are immutable. See also [`capacities()`] for
    ///   more details.
    /// - The [`Initializer`], which defines when particles are emitted.
    ///   Initializers can be either spawners, to spawn new particles, or
    ///   cloners, to clone particles from one group into another.
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
    /// let spawner = Spawner::rate(5_f32.into()); // 5 particles per second
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
    /// [`capacities()`]: crate::EffectAsset::capacities
    /// [`Expr`]: crate::graph::expr::Expr
    pub fn new(capacity: u32, spawner: Spawner, module: Module) -> Self {
        Self {
            capacities: vec![capacity],
            init: vec![spawner.into()],
            module,
            ..default()
        }
    }

    /// Creates a new particle group with the given capacity and initializer.
    ///
    /// Initializers can be spawners or cloners. Particle group indices are
    /// assigned sequentially; thus, the first time you call this function (or
    /// one of the convenience functions like [`with_trails()`] or
    /// [`with_ribbons()`]), the ID of the resulting group will be 1, the
    /// second time will create a group with ID 2, and so forth. Any asset
    /// always have a group 0 implicitly created by [`EffectAsset::new()`].
    ///
    /// For a less verbose way to create trails and ribbons, see
    /// [`with_trails()`] and [`with_ribbons()`] respectively.
    ///
    /// [`with_trails()`]: Self::with_trails
    /// [`with_ribbons()`]: Self::with_ribbons
    pub fn with_group(mut self, capacity: u32, initializer: impl Into<Initializer>) -> Self {
        self.capacities.push(capacity);
        self.init.push(initializer.into());
        self
    }

    /// Get the capacities of the effect, in number of particles per group.
    ///
    /// For example, if this function returns `&[256, 512]`, then this effect
    /// has two groups, the first of which has a maximum of 256 particles and
    /// the second of which has a maximum of 512 particles.
    ///
    /// Each value in the array represents the number of particles stored in GPU
    /// memory at all time for the group with the corresponding index, even if
    /// unused, so you should try to minimize this value. However, the
    /// [`Spawner`] cannot emit more particles than the capacity of group 0.
    /// Whatever the spawner settings, if the number of particles reaches the
    /// capacity, no new particle can be emitted. Setting an appropriate
    /// capacity for an effect is therefore a compromise between more particles
    /// available for visuals and more GPU memory usage.
    ///
    /// Common values range from 256 or less for smaller effects, to several
    /// hundreds of thousands for unique effects consuming a large portion of
    /// the GPU memory budget. Hanabi has been tested with over a million
    /// particles, however the performance will largely depend on the actual GPU
    /// hardware and available memory, so authors are encouraged not to go too
    /// crazy with the capacities.
    pub fn capacities(&self) -> &[u32] {
        &self.capacities
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

    /// Adds a new particle group that clones particles at an interval to
    /// produce a trail.
    ///
    /// Trails allow your particles to emit copies of themselves at fixed
    /// intervals, creating the effect of particles that follow one another.
    /// Trails consist of particles that are disconnected from one another; to
    /// visually connect the trail particles together, use a
    /// [ribbon](Self::with_ribbons) instead.
    ///
    /// You may have as many trails as you wish per particle effect, up to the
    /// limit on the number of groups.
    ///
    /// Particle group indices are assigned sequentially. The first group,
    /// automatically created when you create an effect, has ID 0. Additional
    /// groups, which functions like this one create, are assigned ID 1, 2, 3,
    /// etc.
    ///
    /// `capacity` represents the maximum number of particles in the group.
    /// `period` represents the fixed interval between clone operations.
    /// `lifetime` represents how long each particle in the trail lives;
    /// currently, it must be a fixed number of seconds. `src_group_index` is
    /// the group from which the particles are to be cloned; most of the time,
    /// you will want to pass 0 here to target the first group.
    pub fn with_trails(
        mut self,
        capacity: u32,
        period: impl Into<CpuValue<f32>>,
        lifetime: f32,
        src_group_index: u32,
    ) -> Self {
        self.capacities.push(capacity);
        self.init.push(Initializer::Cloner(Cloner {
            src_group_index,
            period: period.into(),
            lifetime,
            starts_active: true,
        }));
        self
    }

    /// Adds a new particle group that creates a ribbon following particles from
    /// another group.
    ///
    /// A ribbon is a connected string of quads that trail behind particles
    /// from the source group. Hanabi emits new quads on a fixed interval given
    /// by `period`. Ribbons are similar to [trails](Self::with_trails), but
    /// while trail particles are disconnected, ribbon particles are connected.
    ///
    /// Because ribbons internally use a doubly-linked list, of which there's at
    /// most one per effect, you may have at most one ribbon per particle
    /// effect.
    ///
    /// Particle group indices are assigned sequentially. The first group,
    /// automatically created when you create an effect, has ID 0. Additional
    /// groups, which functions like this one create, are assigned ID 1, 2, 3,
    /// etc.
    ///
    /// `capacity` represents the maximum number of ribbon segments in the
    /// group. `period` represents the amount of time that Hanabi will wait
    /// before spawning a new ribbon segment. `lifetime` represents the number
    /// of seconds that each ribbon segment will persist for.
    /// `src_group_index` is the group containing the particles that the ribbon
    /// segments will follow; most of the time, you will want to pass 0 here to
    /// target the first group.
    pub fn with_ribbons(
        mut self,
        capacity: u32,
        period: impl Into<CpuValue<f32>>,
        lifetime: f32,
        src_group_index: u32,
    ) -> Self {
        debug_assert!(self.ribbon_group.is_none());
        self.ribbon_group = Some(self.capacities.len());
        let period: CpuValue<f32> = period.into();
        self.with_trails(capacity, period, lifetime, src_group_index)
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
        self.init_modifiers.push(GroupedModifier {
            modifier: Box::new(modifier),
            groups: ParticleGroupSet::all(),
        });
        self
    }

    /// Add an initialization modifier to a specific set of groups.
    ///
    /// Initialization modifiers apply to all particles within those groups that
    /// are spawned or cloned.
    ///
    /// # Panics
    ///
    /// Panics if the modifier doesn't support the init context (that is,
    /// `modifier.context()` returns a flag which doesn't include
    /// [`ModifierContext::Init`]).
    pub fn init_groups<M>(mut self, modifier: M, groups: ParticleGroupSet) -> Self
    where
        M: Modifier + Send + Sync,
    {
        self.init_modifiers.push(GroupedModifier {
            modifier: Box::new(modifier),
            groups,
        });
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
        self.update_modifiers.push(GroupedModifier {
            modifier: Box::new(modifier),
            groups: ParticleGroupSet::all(),
        });
        self
    }

    /// Add an update modifier to the effect targeting only a subset of groups.
    ///
    /// [`with_property()`]: crate::EffectAsset::with_property
    /// [`add_property()`]: crate::EffectAsset::add_property
    #[inline]
    pub fn update_groups<M>(mut self, modifier: M, groups: ParticleGroupSet) -> Self
    where
        M: Modifier + Send + Sync,
    {
        self.update_modifiers.push(GroupedModifier {
            modifier: Box::new(modifier),
            groups,
        });
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
    pub fn add_modifier(self, context: ModifierContext, modifier: Box<dyn Modifier>) -> Self {
        self.add_modifier_to_groups(context, modifier, ParticleGroupSet::all())
    }

    /// Add a [`BoxedModifier`] to the specific context, in a specific set of
    /// groups.
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
    pub fn add_modifier_to_groups(
        mut self,
        context: ModifierContext,
        modifier: Box<dyn Modifier>,
        groups: ParticleGroupSet,
    ) -> Self {
        assert!(context == ModifierContext::Init || context == ModifierContext::Update);
        assert!(modifier.context().contains(context));
        let grouped_modifier = GroupedModifier { modifier, groups };
        if context == ModifierContext::Init {
            self.init_modifiers.push(grouped_modifier);
        } else {
            self.update_modifiers.push(grouped_modifier);
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
        self.render_modifiers.push(GroupedModifier {
            modifier: Box::new(modifier),
            groups: ParticleGroupSet::all(),
        });
        self
    }

    /// Add a render modifier to specific groups of this effect.
    ///
    /// # Panics
    ///
    /// Panics if the modifier doesn't support the render context (that is,
    /// `modifier.context()` returns a flag which doesn't include
    /// [`ModifierContext::Render`]).
    #[inline]
    pub fn render_groups<M>(mut self, modifier: M, groups: ParticleGroupSet) -> Self
    where
        M: RenderModifier + Send + Sync,
    {
        assert!(modifier.context().contains(ModifierContext::Render));
        self.render_modifiers.push(GroupedModifier {
            modifier: Box::new(modifier),
            groups,
        });
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
        self.render_modifiers.push(GroupedModifier {
            modifier: modifier.boxed_clone(),
            groups: ParticleGroupSet::all(),
        });
        self
    }

    /// Add a [`RenderModifier`] to the render context targeting a specific set
    /// of groups.
    ///
    /// # Panics
    ///
    /// Panics if the modifier doesn't support the render context (that is,
    /// `modifier.context()` returns a flag which doesn't include
    /// [`ModifierContext::Render`]).
    pub fn add_render_modifier_to_groups(
        mut self,
        modifier: Box<dyn RenderModifier>,
        groups: ParticleGroupSet,
    ) -> Self {
        assert!(modifier.context().contains(ModifierContext::Render));
        self.render_modifiers.push(GroupedModifier {
            modifier: modifier.boxed_clone(),
            groups,
        });
        self
    }

    /// Get a list of all the modifiers of this effect.
    pub fn modifiers(&self) -> impl Iterator<Item = &dyn Modifier> {
        self.init_modifiers
            .iter()
            .map(|grouped_modifier| &*grouped_modifier.modifier)
            .chain(
                self.update_modifiers
                    .iter()
                    .map(|grouped_modifier| &*grouped_modifier.modifier),
            )
            .chain(
                self.render_modifiers
                    .iter()
                    .map(|grouped_modifier| &*grouped_modifier.modifier),
            )
    }

    /// Get a list of all the init modifiers of this effect.
    ///
    /// This is a filtered list of all modifiers, retaining only modifiers
    /// executing in the [`ModifierContext::Init`] context.
    ///
    /// [`ModifierContext::Init`]: crate::ModifierContext::Init
    pub fn init_modifiers(&self) -> impl Iterator<Item = &dyn Modifier> {
        self.init_modifiers.iter().filter_map(|gm| {
            if gm.modifier.context().contains(ModifierContext::Init) {
                Some(gm.modifier.deref())
            } else {
                None
            }
        })
    }

    /// Get a list of all the init modifiers in a single group.
    ///
    /// This is a filtered list of all modifiers, retaining only modifiers
    /// executing in the [`ModifierContext::Init`] context and affecting the
    /// specified group.
    ///
    /// [`ModifierContext::Init`]: crate::ModifierContext::Init
    pub fn init_modifiers_for_group(
        &self,
        group_index: u32,
    ) -> impl Iterator<Item = &dyn Modifier> {
        self.init_modifiers.iter().filter_map(move |gm| {
            if gm.groups.contains(group_index)
                && gm.modifier.context().contains(ModifierContext::Init)
            {
                Some(gm.modifier.deref())
            } else {
                None
            }
        })
    }

    /// Get a list of all the update modifiers of this effect.
    ///
    /// This is a filtered list of all modifiers, retaining only modifiers
    /// executing in the [`ModifierContext::Update`] context.
    ///
    /// [`ModifierContext::Update`]: crate::ModifierContext::Update
    pub fn update_modifiers(&self) -> impl Iterator<Item = &dyn Modifier> {
        self.update_modifiers.iter().filter_map(|gm| {
            if gm.modifier.context().contains(ModifierContext::Update) {
                Some(gm.modifier.deref())
            } else {
                None
            }
        })
    }

    /// Get a list of all the update modifiers in a single group.
    ///
    /// This is a filtered list of all modifiers, retaining only modifiers
    /// executing in the [`ModifierContext::Update`] context and affecting the
    /// specified group.
    ///
    /// [`ModifierContext::Update`]: crate::ModifierContext::Update
    pub fn update_modifiers_for_group(
        &self,
        group_index: u32,
    ) -> impl Iterator<Item = &dyn Modifier> {
        self.update_modifiers.iter().filter_map(move |gm| {
            if gm.groups.contains(group_index)
                && gm.modifier.context().contains(ModifierContext::Update)
            {
                Some(gm.modifier.deref())
            } else {
                None
            }
        })
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

    /// Get a list of all the render modifiers of this effect that affect a
    /// specific group.
    ///
    /// This is a filtered list of all modifiers, retaining only modifiers
    /// executing in the [`ModifierContext::Render`] context and that affect the
    /// given group.
    ///
    /// [`ModifierContext::Render`]: crate::ModifierContext::Render
    pub fn render_modifiers_for_group(
        &self,
        group_index: u32,
    ) -> impl Iterator<Item = &dyn RenderModifier> {
        self.render_modifiers.iter().filter_map(move |m| {
            if m.groups.contains(group_index) {
                m.modifier.as_render()
            } else {
                None
            }
        })
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

        // If we're using ribbons, we need a linked list.
        if self.ribbon_group.is_some() {
            set.insert(Attribute::PREV);
            set.insert(Attribute::NEXT);
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
        PropertyLayout::new(self.properties().iter())
    }

    /// Get the texture layout of the module of this effect.
    pub fn texture_layout(&self) -> TextureLayout {
        self.module.texture_layout()
    }

    /// Sets the mesh that each particle will render.
    pub fn mesh(mut self, mesh: Handle<Mesh>) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Computes the group evaluation order, which ensures that cloners run
    /// before spawners.
    ///
    /// This makes sure that we don't spawn a particle and immediately clone it,
    /// which looks bad.
    pub(crate) fn calculate_group_order(&self) -> Vec<u32> {
        let mut group_order = Vec::with_capacity(self.init.len());
        for (group_index, init) in self.init.iter().enumerate() {
            if let Initializer::Cloner(_) = init {
                group_order.push(group_index as u32);
            }
        }
        for (group_index, init) in self.init.iter().enumerate() {
            if let Initializer::Spawner(_) = init {
                group_order.push(group_index as u32);
            }
        }
        group_order
    }
}

/// Asset loader for [`EffectAsset`].
///
/// Effet assets take the `.effect` extension.
#[cfg(feature = "serde")]
#[derive(Default)]
pub struct EffectAssetLoader;

/// Error for the [`EffectAssetLoader`] loading an [`EffectAsset`].
#[cfg(feature = "serde")]
#[derive(Error, Debug)]
pub enum EffectAssetLoaderError {
    /// I/O error reading the asset source.
    #[error("An IO error occurred during loading of a particle effect")]
    Io(#[from] std::io::Error),

    /// Error during RON format parsing.
    #[error("A RON format error occurred during loading of a particle effect")]
    Ron(#[from] ron::error::SpannedError),
}

#[cfg(feature = "serde")]
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
        let custom_asset = ron::de::from_bytes::<EffectAsset>(&bytes)?;
        Ok(custom_asset)
    }

    fn extensions(&self) -> &[&str] {
        &["effect"]
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParticleTrails {
    pub spawn_period: f32,
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "serde")]
    use ron::ser::PrettyConfig;

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

        let mut effect = EffectAsset::new(4096, Spawner::rate(30.0.into()), module)
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

        assert_eq!(&effect.capacities, &[4096]);

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

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_ron() {
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
            capacities: vec![4096],
            init: vec![Spawner::rate(30.0.into()).into()],
            module,
            ..Default::default()
        }
        .init(mod_pos);

        let s = ron::ser::to_string_pretty(&effect, PrettyConfig::new().new_line("\n".to_string()))
            .unwrap();
        eprintln!("{}", s);
        assert_eq!(
            s,
            r#"(
    name: "Effect",
    capacities: [
        4096,
    ],
    init: [
        Spawner((
            count: Single(30.0),
            spawn_duration: Single(1.0),
            period: Single(1.0),
            starts_active: true,
            starts_immediately: true,
        )),
    ],
    z_layer_2d: 0.0,
    simulation_space: Global,
    simulation_condition: WhenVisible,
    init_modifiers: [
        (
            modifier: {
                "SetAttributeModifier": (
                    attribute: "position",
                    value: 1,
                ),
            },
            groups: (4294967295),
        ),
    ],
    update_modifiers: [],
    render_modifiers: [],
    motion_integration: PostUpdate,
    module: (
        expressions: [
            Literal(Vector(Vec3((1.2, -3.45, 87.54485)))),
            Literal(Vector(BVec2((false, true)))),
            Binary(
                op: Add,
                left: 2,
                right: 1,
            ),
            Property(1),
            Unary(
                op: Abs,
                expr: 4,
            ),
        ],
        properties: [
            (
                name: "my_prop",
                default_value: Vector(Vec3((1.2, -2.3, 55.32))),
            ),
        ],
        texture_layout: (
            layout: [],
        ),
    ),
    alpha_mode: Blend,
    ribbon_group: None,
)"#
        );
        let effect_serde: EffectAsset = ron::from_str(&s).unwrap();
        assert_eq!(effect.name, effect_serde.name);
        assert_eq!(effect.capacities, effect_serde.capacities);
        assert_eq!(effect.init, effect_serde.init);
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
}
