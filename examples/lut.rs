//! Example of using a texture as a look-up table (LUT).
//!
//! This example spawns an effect which emits particles with a large initial
//! velocity, then decelerate due to drag. The particle color is looked up from
//! a 2D array texture based on its current age (U) and velocity (V), together
//! forming a texture coordinate (U,V) where to read. The array layer itself is
//! read from a property, and therefore can be dynamically changed by pressing
//! the UP/DOWN arrow or SPACE keys. In this example we use 4 layers,
//! representing 4 different color themes.
//!
//! Note that the example demonstrates reading a LUT texture from the Update
//! pass. For this read to be any useful, it writes the calculated particle's
//! color into the Attribute::COLOR, which occupies 4 bytes per particle in the
//! particle buffer. However, because the particle color is only used during
//! rendering, and the lookup is deterministic, this approach is NOT the
//! recommended way to achieve the visual result of this example. In a
//! production context, you should instead read the LUT directly inside the
//! Render pass and use the value there, without storing it, to avoid wasting 4
//! bytes per particle. This example is for demonstration only.

use bevy::{asset::RenderAssetUsages, core_pipeline::tonemapping::Tonemapping, prelude::*};
use bevy_hanabi::{expr::TextureLoadExpr, prelude::*};

mod utils;
use utils::*;
use wgpu::{Extent3d, TextureFormat};

const DEMO_DESC: &str = include_str!("lut.txt");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_exit = utils::DemoApp::new("lut")
        .with_desc(DEMO_DESC)
        .build()
        .add_systems(Startup, setup)
        .add_systems(Update, update)
        .run();
    app_exit.into_result()
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Make a 2d-array image. Each layer contains a "color theme" for the particles.
    // Within a layer, the U coordinate (alongside X) corresponds to the particle
    // normalized age (= age / lifetime). The V coordinate corresponds to the
    // normalized velocity.
    let mut data = vec![0u8; 128 * 128 * 4 * 4];
    for ilayer in 0..4 {
        for j in 0..128 {
            for i in 0..128 {
                let offset = ilayer * 128 * 128 + j * 128 + i;
                let c = match ilayer {
                    0 => Color::hsl(i as f32 / 127.0 * 60.0, 0.8, 0.3 + j as f32 / 127.0 * 0.7)
                        .to_linear(),
                    1 => Color::hsl(
                        i as f32 / 127.0 * 60.0 + 120.0,
                        0.8,
                        0.3 + j as f32 / 127.0 * 0.7,
                    )
                    .to_linear(),
                    2 => Color::hsl(
                        i as f32 / 127.0 * 60.0 + 240.0,
                        0.8,
                        0.3 + j as f32 / 127.0 * 0.7,
                    )
                    .to_linear(),
                    3 => Color::hsl(
                        i as f32 / 127.0 * 60.0 + 300.0,
                        0.8,
                        0.3 + j as f32 / 127.0 * 0.7,
                    )
                    .to_linear(),
                    _ => panic!(),
                };
                data[offset * 4] = (c.red * 255.0) as u8;
                data[offset * 4 + 1] = (c.green * 255.0) as u8;
                data[offset * 4 + 2] = (c.blue * 255.0) as u8;
                data[offset * 4 + 3] = 255;
            }
        }
    }
    let image = Image::new(
        Extent3d {
            width: 128,
            height: 128,
            depth_or_array_layers: 4,
        },
        wgpu::TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    let image = images.add(image);

    // Main 3D camera
    commands.spawn((
        Transform::from_translation(Vec3::Z * 100.),
        Camera3d::default(),
        Tonemapping::None,
    ));

    // Some light
    commands.spawn(DirectionalLight {
        color: Color::WHITE,
        // Crank the illuminance way (too) high to make the reference cube clearly visible
        illuminance: 100000.,
        shadow_maps_enabled: false,
        ..Default::default()
    });

    // A reference cube showing the position of the particle emitter. This is only
    // for reference; this is not needed for the effect to actually work.
    let cube = meshes.add(Cuboid {
        half_size: Vec3::splat(0.5),
    });
    let mat = materials.add(utils::COLOR_PURPLE);

    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(5.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let drag = writer.lit(2.0).expr();
    let update_drag = LinearDragModifier::new(drag);

    let init_pos = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(1.).expr(),
        dimension: ShapeDimension::Volume,
    };

    let init_vel = SetVelocitySphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        speed: writer.lit(100.).expr(),
    };

    // x = u32(clamp(age / lifetime, 0.0, 1.0) * 127.0)
    let x = ((writer.attr(Attribute::AGE) / writer.attr(Attribute::LIFETIME))
        .clamp(writer.lit(0.), writer.lit(1.))
        * writer.lit(127.))
    .cast(ValueType::Scalar(ScalarType::Uint))
    .expr();
    // y = u32(min(1.0, velocity / 50.0) * 127.0)
    let y = ((writer.attr(Attribute::VELOCITY).length() / writer.lit(50.)).min(writer.lit(1.0))
        * writer.lit(127.0))
    .cast(ValueType::Scalar(ScalarType::Uint))
    .expr();
    // coordinates = vec2<u32>(x, y)
    let coordinates = writer
        .push(Expr::Binary {
            op: BinaryOperator::Vec2,
            left: x,
            right: y,
        })
        .expr();
    // array_index = property.layer
    let prop_layer = writer.add_property("layer", Value::Scalar(ScalarValue::Uint(0)));
    let array_index = writer.prop(prop_layer).expr();
    // lut_color = textureLoad(texture#0, coordinates, array_index)
    let lut_color = writer
        .push(Expr::TextureLoad(TextureLoadExpr {
            slot_index: 0,
            slot_dimension: SlotDimension::D2Array,
            coordinates,
            array_index: Some(array_index),
            mip_level: None,
        }))
        .expr();
    // convert HDR vec4<f32>(r,g,b,a) -> LDR u32(0xAABBGGRR); this allows storing
    // only 4 bytes instead 16 bytes per particle for the color.
    let lut_color = writer
        .push(Expr::Unary {
            op: UnaryOperator::Pack4x8unorm,
            expr: lut_color,
        })
        .expr();
    let update_lut = SetAttributeModifier::new(Attribute::COLOR, lut_color);

    let mut module = writer.finish();
    let _ = module.add_texture_slot("lut", SlotDimension::D2Array);

    let effect = effects.add(
        EffectAsset::new(32768, SpawnerSettings::rate(30.0.into()), module)
            .with_name("lut")
            .init(init_pos)
            .init(init_vel)
            .init(init_age)
            .init(init_lifetime)
            .update(update_drag)
            .update(update_lut),
    );

    commands
        .spawn((
            Name::new("emit:random"),
            ParticleEffect::new(effect),
            Transform::from_translation(Vec3::new(0., 0., 0.)),
            EffectProperties::default(),
            EffectMaterial {
                // Assign our LUT image to texture slot #0
                images: vec![image],
            },
        ))
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn((Mesh3d(cube), MeshMaterial3d(mat)));
        });
}

fn update(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut q_properties: Query<&mut EffectProperties>,
) {
    let Ok(mut properties) = q_properties.single_mut() else {
        return;
    };

    // Read the current 2D array texture layer index stored in the "layer" property
    let layer_index = properties
        .get_stored("layer")
        .map(|v| v.as_scalar().as_u32())
        .unwrap_or(0);

    // Update the index if the user press a key
    let mut new_index = layer_index;
    if keyboard_input.just_pressed(KeyCode::ArrowDown)
        || keyboard_input.just_pressed(KeyCode::Space)
    {
        new_index = (new_index + 1) % 4;
    } else if keyboard_input.just_pressed(KeyCode::ArrowUp) {
        new_index = (new_index + 3) % 4;
    }

    // If the value changed, write down the new value, and it gets uploaded to GPU
    // automatically by Hanabi.
    if new_index != layer_index {
        properties.set("layer", new_index.into());
    }
}
