//! Modifiers to influence the update behavior of particle each frame.
//!
//! The update modifiers control how particles are updated each frame by the
//! update compute shader.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    graph::Value, Attribute, BoxedModifier, Modifier, ModifierContext, Property, ToWgslString,
};

/// Particle update shader code generation context.
#[derive(Debug, Default, Clone, PartialEq)] // Eq
pub struct UpdateContext {
    /// Main particle update code, which needs to update the fields of the
    /// `particle` struct instance.
    pub update_code: String,
    /// Extra functions emitted at top level, which `update_code` can call.
    pub update_extra: String,

    //TEMP
    /// Array of force field components with a maximum number of components
    /// determined by [`ForceFieldSource::MAX_SOURCES`].
    pub force_field: [ForceFieldSource; ForceFieldSource::MAX_SOURCES],
}

/// Trait to customize the updating of existing particles each frame.
#[typetag::serde]
pub trait UpdateModifier: Modifier {
    /// Append the update code.
    fn apply(&self, context: &mut UpdateContext);
}

/// Macro to implement the [`Modifier`] trait for an update modifier.
macro_rules! impl_mod_update {
    ($t:ty, $attrs:expr) => {
        #[typetag::serde]
        impl Modifier for $t {
            fn context(&self) -> ModifierContext {
                ModifierContext::Update
            }

            fn as_update(&self) -> Option<&dyn UpdateModifier> {
                Some(self)
            }

            fn as_update_mut(&mut self) -> Option<&mut dyn UpdateModifier> {
                Some(self)
            }

            fn attributes(&self) -> &[&'static Attribute] {
                $attrs
            }

            fn boxed_clone(&self) -> BoxedModifier {
                Box::new(*self)
            }
        }
    };
}

/// Constant value or reference to a named property.
///
/// This enumeration either directly stores a constant value assigned at
/// creation time, or a reference to an effect property the value is derived
/// from.
#[derive(Debug, Clone, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub enum ValueOrProperty {
    /// Constant value.
    Value(Value),
    /// Unresolved reference to a named property the value is derived from.
    Property(String),
    /// Reference to a named property the value is derived from, resolved to the
    /// index of that property in the owning [`EffectAsset`].
    ///
    /// [`EffectAsset`]: crate::EffectAsset
    ResolvedProperty((usize, String)),
}

impl ToWgslString for ValueOrProperty {
    fn to_wgsl_string(&self) -> String {
        match self {
            ValueOrProperty::Value(value) => value.to_wgsl_string(),
            ValueOrProperty::Property(name) => format!("properties.{}", name),
            ValueOrProperty::ResolvedProperty((_, name)) => format!("properties.{}", name),
        }
    }
}

/// A modifier to apply a uniform acceleration to all particles each frame, to
/// simulate gravity or any other global force.
///
/// The acceleration is the same for all particles of the effect, and is applied
/// each frame to modify the particle's velocity based on the simulation
/// timestep.
///
/// ```txt
/// particle.velocity += acceleration * simulation.delta_time;
/// ```
#[derive(Debug, Clone, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct AccelModifier {
    /// The acceleration to apply to all particles in the effect each frame.
    accel: ValueOrProperty,
}

impl AccelModifier {
    /// Create a new modifier with an acceleration derived from a property.
    pub fn via_property(property_name: impl Into<String>) -> Self {
        Self {
            accel: ValueOrProperty::Property(property_name.into()),
        }
    }

    /// Create a new modifier with a constant acceleration.
    pub fn constant(acceleration: Vec3) -> Self {
        Self {
            accel: ValueOrProperty::Value(acceleration.into()),
        }
    }
}

#[typetag::serde]
impl Modifier for AccelModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Update
    }

    fn as_update(&self) -> Option<&dyn UpdateModifier> {
        Some(self)
    }

    fn as_update_mut(&mut self) -> Option<&mut dyn UpdateModifier> {
        Some(self)
    }

    fn attributes(&self) -> &[&'static Attribute] {
        &[Attribute::VELOCITY]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(self.clone())
    }

    fn resolve_properties(&mut self, properties: &[Property]) {
        if let ValueOrProperty::Property(name) = &mut self.accel {
            if let Some(index) = properties.iter().position(|p| p.name() == name) {
                let name = std::mem::take(name);
                self.accel = ValueOrProperty::ResolvedProperty((index, name));
            } else {
                panic!("Cannot resolve property '{}' in effect. Ensure you have added a property with `with_property()` or `add_property()` before trying to reference it from a modifier.", name);
            }
        }
    }
}

#[typetag::serde]
impl UpdateModifier for AccelModifier {
    fn apply(&self, context: &mut UpdateContext) {
        context.update_code += &format!(
            "(*particle).velocity += {} * sim_params.dt;\n",
            self.accel.to_wgsl_string()
        );
    }
}

/// A single attraction or repulsion source of a [`ForceFieldModifier`].
///
/// The source applies a radial force field to all particles around its
/// position, with a decreasing intensity the further away from the source the
/// particle is. This force is added to the one(s) of all the other active
/// sources of a [`ForceFieldModifier`].
#[derive(Debug, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
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

/// A modifier to apply a force field to all particles each frame. The force
/// field is made up of [`ForceFieldSource`]s.
///
/// The maximum number of sources is [`ForceFieldSource::MAX_SOURCES`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct ForceFieldModifier {
    /// Array of force field sources.
    ///
    /// A source can be disabled by setting its [`mass`] to zero. In that case,
    /// all other sources located after it in the array are automatically
    /// disabled too.
    ///
    /// [`mass`]: ForceFieldSource::mass
    pub sources: [ForceFieldSource; ForceFieldSource::MAX_SOURCES],
}

impl ForceFieldModifier {
    /// Instantiate a [`ForceFieldModifier`] with a set of sources.
    ///
    /// # Panics
    ///
    /// Panics if the number of sources exceeds [`MAX_SOURCES`].
    ///
    /// [`MAX_SOURCES`]: ForceFieldSource::MAX_SOURCES
    pub fn new<T>(sources: T) -> Self
    where
        T: IntoIterator<Item = ForceFieldSource>,
    {
        let mut source_array = [ForceFieldSource::default(); ForceFieldSource::MAX_SOURCES];

        for (index, p_attractor) in sources.into_iter().enumerate() {
            if index >= ForceFieldSource::MAX_SOURCES {
                panic!(
                    "Force field source count exceeded maximum of {}",
                    ForceFieldSource::MAX_SOURCES
                );
            }
            source_array[index] = p_attractor;
        }

        Self {
            sources: source_array,
        }
    }

    /// Overwrite the source at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is greater than or equal to [`MAX_SOURCES`].
    ///
    /// [`MAX_SOURCES`]: ForceFieldSource::MAX_SOURCES
    pub fn add_or_replace(&mut self, source: ForceFieldSource, index: usize) {
        self.sources[index] = source;
    }
}

impl_mod_update!(
    ForceFieldModifier,
    &[Attribute::POSITION, Attribute::VELOCITY]
);

#[typetag::serde]
impl UpdateModifier for ForceFieldModifier {
    fn apply(&self, context: &mut UpdateContext) {
        context.update_code += include_str!("../render/force_field_code.wgsl");

        //TEMP
        context.force_field = self.sources;
    }
}

/// A modifier to apply a linear drag force to all particles each frame. The
/// force slows down the particles without changing their direction.
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct LinearDragModifier {
    /// Drag coefficient. Higher values increase the drag force, and
    /// consequently decrease the particle's speed faster.
    pub drag: f32,
}

impl LinearDragModifier {
    /// Instantiate a [`LinearDragModifier`].
    pub fn new(drag: f32) -> Self {
        Self { drag }
    }
}

impl_mod_update!(LinearDragModifier, &[Attribute::VELOCITY]);

#[typetag::serde]
impl UpdateModifier for LinearDragModifier {
    fn apply(&self, context: &mut UpdateContext) {
        context.update_code += &format!(
            "(*particle).velocity *= max(0., (1. - {} * sim_params.dt));\n",
            self.drag.to_wgsl_string()
        );
    }
}

/// A modifier killing all particles that enter or exit an AABB.
///
/// This enables confining particles to a region in space, or preventing
/// particles to enter that region.
#[derive(Debug, Default, Clone, Copy, PartialEq, Reflect, FromReflect, Serialize, Deserialize)]
pub struct AabbKillModifier {
    /// Min corner of the AABB.
    pub min: Vec3,
    /// Max corner of the AABB.
    pub max: Vec3,
    /// If `true`, invert the kill condition and kill all particles inside the
    /// AABB. If `false` (default), kill all particles outside the AABB.
    pub kill_inside: bool,
}

impl_mod_update!(AabbKillModifier, &[Attribute::POSITION]);

#[typetag::serde]
impl UpdateModifier for AabbKillModifier {
    fn apply(&self, context: &mut UpdateContext) {
        let center = (self.min + self.max) / 2.;
        let half_size = (self.max - self.min) / 2.;
        let cmp = if self.kill_inside { "<" } else { ">" };
        let reduce = if self.kill_inside { "all" } else { "any" };
        context.update_code += &format!(
            r#"if ({}(abs((*particle).position - {}) {} {})) {{
    is_alive = false;
}}
"#,
            reduce,
            center.to_wgsl_string(),
            cmp,
            half_size.to_wgsl_string()
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::ParticleLayout;

    use super::*;

    use naga::front::wgsl::Parser;

    #[test]
    fn mod_accel() {
        let accel = Vec3::new(1., 2., 3.);
        let modifier = AccelModifier::constant(accel);

        let mut context = UpdateContext::default();
        modifier.apply(&mut context);

        assert!(context.update_code.contains(&accel.to_wgsl_string()));
    }

    #[test]
    fn mod_force_field() {
        let position = Vec3::new(1., 2., 3.);
        let mut sources = [ForceFieldSource::default(); 16];
        sources[0].position = position;
        sources[0].mass = 1.;
        let modifier = ForceFieldModifier { sources };

        let mut context = UpdateContext::default();
        modifier.apply(&mut context);

        // force_field_code.wgsl is too big
        //assert!(context.update_code.contains(&include_str!("../render/
        // force_field_code.wgsl")));
    }

    #[test]
    #[should_panic]
    fn mod_force_field_new_too_many() {
        let count = ForceFieldSource::MAX_SOURCES + 1;
        ForceFieldModifier::new((0..count).map(|_| ForceFieldSource::default()));
    }

    #[test]
    fn mod_drag() {
        let modifier = LinearDragModifier { drag: 3.5 };

        let mut context = UpdateContext::default();
        modifier.apply(&mut context);

        assert!(context.update_code.contains("3.5")); // TODO - less weak check
    }

    #[test]
    fn validate() {
        let modifiers: &[&dyn UpdateModifier] = &[
            &AccelModifier::constant(Vec3::ONE),
            &ForceFieldModifier::default(),
            &LinearDragModifier::default(),
            &AabbKillModifier::default(),
        ];
        for &modifier in modifiers.iter() {
            let mut context = UpdateContext::default();
            modifier.apply(&mut context);
            let update_code = context.update_code;
            let update_extra = context.update_extra;

            let mut particle_layout = ParticleLayout::new();
            for &attr in modifier.attributes() {
                particle_layout = particle_layout.append(attr);
            }
            let particle_layout = particle_layout.build();
            let attributes_code = particle_layout.generate_code();

            let code = format!(
                r##"fn rand() -> f32 {{
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
    dt: f32,
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
    let particle: ptr<storage, Particle, read_write> = &particle_buffer.particles[0];
    var transform: mat4x4<f32> = mat4x4<f32>();
    var is_alive = true;
{update_code}
}}"##
            );

            let mut parser = Parser::new();
            let res = parser.parse(&code);
            if let Err(err) = &res {
                println!("Modifier: {:?}", modifier.type_name());
                println!("Code: {:?}", code);
                println!("Err: {:?}", err);
            }
            assert!(res.is_ok());
        }
    }
}
