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
    /// The rotation is defined as a Euler rotation, applied
    /// in XYZ order. Angles must be in radians.
    ///
    /// Expr type: Vec3
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
        let euler_angles = {rotation} * {dt};
        let cx = cos(euler_angles.x);
        let sx = sin(euler_angles.x);
        let cy = cos(euler_angles.y);
        let sy = sin(euler_angles.y);
        let cz = cos(euler_angles.z);
        let sz = sin(euler_angles.z);

        // Individual axes matrices (Column-major format)
        let rx = mat3x3<f32>(
            1.0, 0.0, 0.0,
            0.0, cx,  sx,
            0.0, -sx, cx
        );

        let ry = mat3x3<f32>(
            cy,  0.0, -sy,
            0.0, 1.0, 0.0,
            sy,  0.0, cy
        );

        let rz = mat3x3<f32>(
            cz,  sz,  0.0,
            -sz, cz,  0.0,
            0.0, 0.0, 1.0
        );

        // Combines rotations (Applies X, then Y, then Z)
        let rotation = rz * ry * rx;
        particle.{0} = particle.{0} * rotation;
        particle.{1} = particle.{1} * rotation;
        particle.{2} = particle.{2} * rotation;
    }}
"#,
            Attribute::AXIS_X.name(),
            Attribute::AXIS_Y.name(),
            Attribute::AXIS_Z.name(),
        );
        Ok(())
    }
}
