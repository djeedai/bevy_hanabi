use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::settings::{WgpuLimits, WgpuSettings},
};
use bevy_hanabi::prelude::*;

mod utils;
use utils::*;

/// Set this to `true` to enable WGPU downlevel constraints. This is disabled by
/// default to prevent the example from failing to start on devices with a
/// monitor resolution larger than the maximum resolution imposed by the
/// downlevel settings of WGPU.
const USE_LOW_LIMITS: bool = false;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Optional; test that a stronger constraint is handled correctly.
    // For example, on macOS the alignment for storage buffer offsets is commonly
    // 256 bytes, whereas on Desktop GPUs it can be much smaller, like 16 bytes
    // only. Force the downlevel limits here, and as an example of how
    // to force a particular limit, and to show Hanabi works with those settings.
    let mut wgpu_settings = WgpuSettings::default();
    if USE_LOW_LIMITS {
        let limits = WgpuLimits::downlevel_defaults();
        wgpu_settings.constrained_limits = Some(limits);
    }

    let app_exit = utils::make_test_app_with_settings("spawn", wgpu_settings)
        .add_systems(Startup, setup)
        .add_systems(Update, update_accel)
        .run();
    app_exit.into_result()
}

/// A simple marker component to identify the effect using a dynamic
/// property-based acceleration that the `update_accel()` system will control at
/// runtime.
#[derive(Component)]
struct DynamicRuntimeAccel;

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Transform::from_translation(Vec3::Z * 100.),
        Camera3d::default(),
        Tonemapping::None,
    ));

    commands.spawn(DirectionalLight {
        color: Color::WHITE,
        // Crank the illuminance way (too) high to make the reference cube clearly visible
        illuminance: 100000.,
        shadows_enabled: false,
        ..default()
    });

    let cube = meshes.add(Cuboid {
        half_size: Vec3::splat(0.5),
    });
    let mat = materials.add(utils::COLOR_PURPLE);

    let mut color_gradient1 = Gradient::new();
    color_gradient1.add_key(0.0, Vec4::splat(1.0));
    color_gradient1.add_key(0.1, Vec4::new(1.0, 1.0, 0.0, 1.0));
    color_gradient1.add_key(0.4, Vec4::new(1.0, 0.0, 0.0, 1.0));
    color_gradient1.add_key(1.0, Vec4::splat(0.0));

    let mut size_gradient1 = Gradient::new();
    size_gradient1.add_key(0.0, Vec3::splat(0.1));
    size_gradient1.add_key(0.5, Vec3::splat(0.5));
    size_gradient1.add_key(0.8, Vec3::splat(0.08));
    size_gradient1.add_key(1.0, Vec3::splat(0.0));

    let writer1 = ExprWriter::new();

    let age1 = writer1.lit(0.).expr();
    let init_age1 = SetAttributeModifier::new(Attribute::AGE, age1);

    let lifetime1 = writer1.lit(5.).expr();
    let init_lifetime1 = SetAttributeModifier::new(Attribute::LIFETIME, lifetime1);

    // Add constant downward acceleration to simulate gravity
    let accel1 = writer1.lit(Vec3::Y * -3.).expr();
    let update_accel1 = AccelModifier::new(accel1);

    let init_pos1 = SetPositionCone3dModifier {
        base_radius: writer1.lit(0.).expr(),
        top_radius: writer1.lit(10.).expr(),
        height: writer1.lit(20.).expr(),
        dimension: ShapeDimension::Volume,
    };

    let init_vel1 = SetVelocitySphereModifier {
        center: writer1.lit(Vec3::ZERO).expr(),
        speed: writer1.lit(10.).expr(),
    };

    let effect1 = effects.add(
        EffectAsset::new(32768, Spawner::rate(500.0.into()), writer1.finish())
            .with_name("emit:rate")
            .init(init_pos1)
            // Make spawned particles move away from the emitter origin
            .init(init_vel1)
            .init(init_age1)
            .init(init_lifetime1)
            .update(update_accel1)
            .render(ColorOverLifetimeModifier {
                gradient: color_gradient1,
            })
            .render(SizeOverLifetimeModifier {
                gradient: size_gradient1,
                screen_space_size: false,
            }),
    );

    commands
        .spawn((
            Name::new("emit:rate"),
            ParticleEffectBundle {
                effect: ParticleEffect::new(effect1),
                transform: Transform::from_translation(Vec3::new(-30., 0., 0.))
                    .with_rotation(Quat::from_rotation_z(1.)),
                ..Default::default()
            },
        ))
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn((
                Mesh3d(cube.clone()),
                MeshMaterial3d(mat.clone()),
                Name::new("source"),
            ));
        });

    let mut gradient2 = Gradient::new();
    gradient2.add_key(0.0, Vec4::new(0.0, 0.7, 0.0, 1.0));
    gradient2.add_key(1.0, Vec4::splat(0.0));

    let writer2 = ExprWriter::new();
    let age2 = writer2.lit(0.).expr();
    let init_age2 = SetAttributeModifier::new(Attribute::AGE, age2);
    let lifetime2 = writer2.lit(5.).expr();
    let init_lifetime2 = SetAttributeModifier::new(Attribute::LIFETIME, lifetime2);
    let init_pos2 = SetPositionSphereModifier {
        center: writer2.lit(Vec3::ZERO).expr(),
        radius: writer2.lit(5.).expr(),
        dimension: ShapeDimension::Volume,
    };
    let init_vel2 = SetVelocitySphereModifier {
        center: writer2.lit(Vec3::ZERO).expr(),
        speed: writer2.lit(2.).expr(),
    };
    let effect2 = effects.add(
        EffectAsset::new(32768, Spawner::once(1000.0.into(), true), writer2.finish())
            .with_name("emit:once")
            .init(init_pos2)
            .init(init_vel2)
            .init(init_age2)
            .init(init_lifetime2)
            .render(ColorOverLifetimeModifier {
                gradient: gradient2,
            }),
    );

    commands
        .spawn((
            Name::new("emit:once"),
            ParticleEffectBundle {
                effect: ParticleEffect::new(effect2),
                transform: Transform::from_translation(Vec3::new(0., 0., 0.)),
                ..Default::default()
            },
        ))
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn((
                Mesh3d(cube.clone()),
                MeshMaterial3d(mat.clone()),
                Name::new("source"),
            ));
        });

    // Note: same as gradient2, will yield shared render shader between effects #2
    // and #3
    let mut gradient3 = Gradient::new();
    gradient3.add_key(0.0, Vec4::new(0.0, 0.0, 1.0, 1.0));
    gradient3.add_key(1.0, Vec4::splat(0.0));

    let writer3 = ExprWriter::new();

    let age3 = writer3.lit(0.).expr();
    let init_age3 = SetAttributeModifier::new(Attribute::AGE, age3);

    let lifetime3 = writer3.lit(5.).expr();
    let init_lifetime3 = SetAttributeModifier::new(Attribute::LIFETIME, lifetime3);

    // Initialize size with a random value between 0.3 and 0.7: size = frand() * 0.4
    // + 0.3
    let size3 = (writer3.rand(ScalarType::Float) * writer3.lit(0.4) + writer3.lit(0.3)).expr();
    let init_size3 = SetAttributeModifier::new(Attribute::SIZE, size3);

    // Add property-driven acceleration
    let my_accel = writer3.add_property("my_accel", Vec3::new(0., -3., 0.).into());
    let accel3 = writer3.prop(my_accel).expr();
    let update_accel3 = AccelModifier::new(accel3);

    let init_pos3 = SetPositionSphereModifier {
        center: writer3.lit(Vec3::ZERO).expr(),
        radius: writer3.lit(5.).expr(),
        dimension: ShapeDimension::Volume,
    };

    let init_vel3 = SetVelocitySphereModifier {
        center: writer3.lit(Vec3::ZERO).expr(),
        speed: writer3.lit(2.).expr(),
    };

    let effect3 = effects.add(
        EffectAsset::new(
            32768,
            Spawner::burst(400.0.into(), 3.0.into()),
            writer3.finish(),
        )
        .with_name("emit:burst")
        .init(init_pos3)
        .init(init_vel3)
        .init(init_age3)
        .init(init_lifetime3)
        .init(init_size3)
        .update(update_accel3)
        .render(ColorOverLifetimeModifier {
            gradient: gradient3,
        }),
    );

    commands
        .spawn((
            Name::new("emit:burst"),
            ParticleEffectBundle {
                effect: ParticleEffect::new(effect3),
                transform: Transform::from_translation(Vec3::new(30., 0., 0.)),
                ..Default::default()
            },
            DynamicRuntimeAccel,
        ))
        .with_children(|p| {
            // Reference cube to visualize the emit origin
            p.spawn((
                Mesh3d(cube.clone()),
                MeshMaterial3d(mat.clone()),
                Name::new("source"),
            ));
        });
}

fn update_accel(
    time: Res<Time>,
    mut query: Query<&mut EffectProperties, With<DynamicRuntimeAccel>>,
) {
    let mut properties = query.single_mut();
    let accel0 = 10.;
    let (s, c) = (time.elapsed_secs() * 0.3).sin_cos();
    let accel = Vec3::new(c * accel0, s * accel0, 0.);
    properties.set("my_accel", accel.into());
}
