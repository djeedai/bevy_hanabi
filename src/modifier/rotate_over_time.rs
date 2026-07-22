use bevy::reflect::Reflect;

use crate::{
    Attribute, BoxedModifier, BuiltInExpr, EvalContext, ExprError, ExprHandle, Modifier,
    ModifierContext, Module, ShaderWriter,
};

/// Rotates particles over time.
#[derive(Clone, Copy, Reflect)]
pub struct RotateOverTimeModifier {
    /// Rotation that the particle will have in a second.
    ///
    /// Expr type: Mat4
    pub rotation: ExprHandle,
}

impl Modifier for RotateOverTimeModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Update
    }

    fn attributes(&self) -> &[Attribute] {
        &[Attribute::AXIS_X, Attribute::AXIS_Y, Attribute::AXIS_Z]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn apply(&self, module: &mut Module, context: &mut ShaderWriter) -> Result<(), ExprError> {
        let rotation = context.eval(module, self.rotation)?;
        let dt = BuiltInExpr::new(crate::graph::BuiltInOperator::DeltaTime).eval(context)?;
        context.main_code += &format!(
            r#"    {{
        let rotation = {rotation};
        particle.{0} = normalize(mix(particle.{0}, (vec4(particle.{0}, 1) * rotation).xyz, {dt}));
        particle.{1} = normalize(mix(particle.{1}, (vec4(particle.{1}, 1) * rotation).xyz, {dt}));
        particle.{2} = normalize(mix(particle.{2}, (vec4(particle.{2}, 1) * rotation).xyz, {dt}));
    }}
"#,
            Attribute::AXIS_X.name(),
            Attribute::AXIS_Y.name(),
            Attribute::AXIS_Z.name(),
        );
        Ok(())
    }
}
