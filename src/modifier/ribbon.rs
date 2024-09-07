//! Renders particles as ribbons.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    impl_mod_render, Attribute, EvalContext, ExprError, Modifier, ModifierContext, Module,
    RenderContext, RenderModifier, ShaderWriter,
};

/// Renders particles as ribbons, drawing a quad in between each particle
/// instead of at each particle.
///
/// Internally, this threads particles into a linked list, using the
/// [`Attribute::PREV`] and [`Attribute::NEXT`] fields.
#[derive(Debug, Clone, Copy, Reflect, Serialize, Deserialize)]
pub struct RibbonModifier;

impl_mod_render!(RibbonModifier, &[Attribute::PREV, Attribute::NEXT]);

#[cfg_attr(feature = "serde", typetag::serde)]
impl RenderModifier for RibbonModifier {
    fn apply_render(&self, _: &mut Module, context: &mut RenderContext) -> Result<(), ExprError> {
        context.vertex_code += r##"
    let next_index = particle.next;
    if (next_index >= arrayLength(&particle_buffer.particles)) {
        out.position = vec4(0.0);
        return out;
    }

    let next_particle = particle_buffer.particles[next_index];
    var delta = next_particle.position - particle.position;

    axis_x = normalize(delta);
    axis_y = normalize(cross(axis_x, axis_z));
    axis_z = cross(axis_x, axis_y);

    position = mix(next_particle.position, particle.position, 0.5);
    size = vec3(length(delta), size.y, 1.0);
"##;

        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(*self)
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}
