use std::f32::consts::FRAC_PI_2;

use super::DemoCommands as Commands;
use bevy::{
    camera::{visibility::RenderLayers, Viewport},
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    math::EulerRot,
    prelude::*,
    window::WindowResized,
};
use bevy_hanabi::prelude::*;

use crate::utils;

const DEMO_DESC: &str = include_str!("multicam.txt");

declare_demo!("multicam");

fn effects() -> Vec<EffectAsset> {
    vec![
        make_effect(utils::COLOR_RED),
        make_effect(utils::COLOR_GREEN),
        make_effect(utils::COLOR_BLUE),
    ]
}

fn install(app: &mut App) {
    app.add_systems(Update, setup.run_if(super::selected_once("multicam")))
        .add_systems(
            Update,
            update_camera_viewports.run_if(super::selected_demo("multicam")),
        );
}

#[derive(Component)]
struct SplitCamera {
    /// Grid position of the camera.
    pos: UVec2,
}

fn make_effect(color: Color) -> EffectAsset {
    let mut size_gradient = bevy_hanabi::Gradient::new();
    size_gradient.add_key(0.0, Vec3::splat(1.0));
    size_gradient.add_key(0.5, Vec3::splat(5.0));
    size_gradient.add_key(0.8, Vec3::splat(0.8));
    size_gradient.add_key(1.0, Vec3::splat(0.0));

    let mut color_gradient = bevy_hanabi::Gradient::new();
    color_gradient.add_key(0.0, Vec4::splat(1.0));
    color_gradient.add_key(
        0.4,
        Vec4::new(
            color.to_linear().red,
            color.to_linear().green,
            color.to_linear().blue,
            1.0,
        ),
    );
    color_gradient.add_key(1.0, Vec4::splat(0.0));

    let writer = ExprWriter::new();

    let age = writer.lit(0.).expr();
    let init_age = SetAttributeModifier::new(Attribute::AGE, age);

    let lifetime = writer.lit(5.).expr();
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);

    let accel = writer.lit(Vec3::Y * -3.).expr();
    let update_accel = AccelModifier::new(accel);

    let init_pos = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(2.).expr(),
        dimension: ShapeDimension::Surface,
    };

    let init_vel = SetVelocitySphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        speed: writer.lit(6.).expr(),
    };

    EffectAsset::new(32768, SpawnerSettings::rate(5.0.into()), writer.finish())
        .with_name("effect")
        .init(init_pos)
        .init(init_vel)
        .init(init_age)
        .init(init_lifetime)
        .update(update_accel)
        .render(ColorOverLifetimeModifier::new(color_gradient))
        .render(SizeOverLifetimeModifier {
            gradient: size_gradient.clone(),
            screen_space_size: false,
        })
        .render(OrientModifier::new(OrientMode::FaceCameraPosition))
}

fn setup(
    mut commands: Commands,
    demos: Query<&super::DemoInfo>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    window: Single<&Window, With<bevy::window::PrimaryWindow>>,
) {
    // Render layers for the 4 cameras, using a mix and match to see the differences
    let layers = [
        RenderLayers::layer(0),
        RenderLayers::layer(0).with(2),
        RenderLayers::layer(1).with(2),
        RenderLayers::from_layers(&[0, 1, 2, 3]),
    ];

    // Spawn 4 cameras in grid, "4-player couch co-op"-style
    for (i, layer) in layers.iter().enumerate() {
        let x = (i % 2) as f32 * 100. - 50.;
        let z = (i / 2) as f32 * 100. - 50.;
        let pos = UVec2::new(i as u32 % 2, i as u32 / 2);
        commands.spawn((
            Transform::from_translation(Vec3::new(x, 100.0, z)).looking_at(Vec3::ZERO, Vec3::Y),
            Camera {
                viewport: Some(camera_viewport(&window, pos)),
                // Have a different order for each camera to ensure determinism
                order: i as isize,
                // Only clear render target from first camera, others additively render on same
                // target
                clear_color: if i == 0 {
                    ClearColorConfig::Default
                } else {
                    ClearColorConfig::None
                },
                ..default()
            },
            Camera3d::default(),
            Tonemapping::None,
            DebandDither::Disabled,
            SplitCamera { pos },
            layer.clone(),
        ));
    }

    commands.spawn((
        Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 1.7, 2.4, 0.)),
        DirectionalLight {
            color: Color::WHITE,
            // Crank the illuminance way (too) high to make the reference cube clearly visible
            illuminance: 100000.,
            shadow_maps_enabled: false,
            ..Default::default()
        },
        // The light affects all the views
        RenderLayers::from_layers(&[0, 1, 2, 3]),
    ));

    let cube = meshes.add(Cuboid {
        half_size: Vec3::splat(0.5),
    });
    let plane = meshes.add(Rectangle {
        half_size: Vec2::splat(200.0),
    });
    let mat = materials.add(utils::COLOR_PURPLE);
    let ground_mat = materials.add(utils::COLOR_OLIVE);

    let effect1 = super::demo_effect(&demos, "multicam", 0);

    // Ground plane to make it easier to see the different cameras
    commands.spawn((
        Transform::from_translation(Vec3::Y * -20.)
            * Transform::from_scale(Vec3::new(0.4, 1., 1.))
            * Transform::from_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
        Mesh3d(plane),
        MeshMaterial3d(ground_mat),
        Name::new("ground"),
        RenderLayers::from_layers(&[0, 1, 2, 3]),
    ));

    commands
        .spawn((
            Name::new("0"),
            ParticleEffect::new(effect1),
            Transform::from_translation(Vec3::new(-30., 0., 0.)),
            RenderLayers::layer(0),
        ))
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn((
                Mesh3d(cube.clone()),
                MeshMaterial3d(mat.clone()),
                Name::new("source"),
                RenderLayers::layer(0),
            ));
        });

    let effect2 = super::demo_effect(&demos, "multicam", 1);

    commands
        .spawn((
            Name::new("1"),
            ParticleEffect::new(effect2),
            Transform::from_translation(Vec3::new(0., 0., 0.)),
            RenderLayers::layer(1),
        ))
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn((
                Mesh3d(cube.clone()),
                MeshMaterial3d(mat.clone()),
                Name::new("source"),
                RenderLayers::layer(1),
            ));
        });

    let effect3 = super::demo_effect(&demos, "multicam", 2);

    commands
        .spawn((
            Name::new("2"),
            ParticleEffect::new(effect3),
            Transform::from_translation(Vec3::new(30., 0., 0.)),
            RenderLayers::layer(2),
        ))
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn((
                Mesh3d(cube.clone()),
                MeshMaterial3d(mat.clone()),
                Name::new("source"),
                RenderLayers::layer(2),
            ));
        });
}

fn update_camera_viewports(
    window: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut resize_events: MessageReader<WindowResized>,
    mut query: Query<(&mut Camera, &SplitCamera)>,
) {
    for resize_event in resize_events.read() {
        let Ok(window) = window.get(resize_event.window) else {
            continue;
        };
        for (mut camera, split_camera) in query.iter_mut() {
            camera.viewport = Some(camera_viewport(window, split_camera.pos));
        }
    }
}

fn camera_viewport(window: &Window, pos: UVec2) -> Viewport {
    let width = window.physical_width() / 2;
    let height = window.physical_height() / 2;
    Viewport {
        physical_position: UVec2::new(width * pos.x, height * (1 - pos.y)),
        physical_size: UVec2::new(width, height),
        ..default()
    }
}
