use bevy::{
    prelude::*,
    render::{mesh::shape::Cube, render_resource::WgpuFeatures, settings::WgpuSettings},
};
//use bevy_inspector_egui::WorldInspectorPlugin;
use std::f32::consts::PI;

use bevy_hanabi::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = WgpuSettings::default();
    options
        .features
        .set(WgpuFeatures::VERTEX_WRITABLE_STORAGE, true);
    // options
    //     .features
    //     .set(WgpuFeatures::MAPPABLE_PRIMARY_BUFFERS, false);
    // println!("wgpu options: {:?}", options.features);
    App::default()
        .insert_resource(options)
        .insert_resource(bevy::log::LogSettings {
            level: bevy::log::Level::WARN,
            filter: "bevy_hanabi=error,spawn=trace".to_string(),
        })
        .add_plugins(DefaultPlugins)
        .add_system(bevy::input::system::exit_on_esc_system)
        .add_plugin(HanabiPlugin)
        //.add_plugin(WorldInspectorPlugin::new())
        .add_startup_system(setup)
        .add_system(update)
        .run();

    Ok(())
}

fn setup(
    asset_server: Res<AssetServer>,
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut camera = PerspectiveCameraBundle::new_3d();
    camera.transform.translation = Vec3::new(0.0, 0.0, 100.0);
    commands.spawn_bundle(camera);

    let texture_handle: Handle<Image> = asset_server.load("cloud.png");

    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::splat(1.0));
    gradient.add_key(0.1, Vec4::new(1.0, 1.0, 0.0, 1.0));
    gradient.add_key(0.4, Vec4::new(1.0, 0.0, 0.0, 1.0));
    gradient.add_key(1.0, Vec4::splat(0.0));

    let effect = effects.add(
        EffectAsset {
            name: "Gradient".to_string(),
            capacity: 32768,
            spawner: Spawner::rate(1000.0.into()),
            ..Default::default()
        }
        .render(ParticleTextureModifier {
            texture: texture_handle.clone(),
        })
        .render(ColorOverLifetimeModifier { gradient }),
    );

    commands
        .spawn()
        .insert(Name::new("effect"))
        .insert_bundle(ParticleEffectBundle::new(effect))
        .with_children(|p| {
            p.spawn().insert_bundle(PbrBundle {
                mesh: meshes.add(Mesh::from(Cube { size: 1.0 })),
                material: materials.add(Color::RED.into()),
                ..Default::default()
            });
        });
}

/// Calculate a position over a Lemniscate of Bernoulli curve ("infinite symbol").
///
/// The fractional part of `time` determines the parametric position over the curve.
/// The `radius` of the Lemniscate curve is the distance from the center to the edge
/// of any of the left or right loops. The curve loops extend from -X to +X.
fn lemniscate(time: f32, radius: f32) -> Vec2 {
    // The Lemniscate is defined in polar coordinates by the equation r² = a² ⨯ cos(2θ),
    // where a is the radius of the Lemniscate (distance from origin to loop edge).
    // This equation is defined only for values of θ in the [-π/4:π/4] range. Each value
    // yields two possible values for r, one positive and one negative, corresponding to
    // the two loops of the Lemniscates (left and right).
    // So we solve for θ ∈ [-π/4:π/4], and make θ vary back and forth in the [-π/4:π/4]
    // range. Then depending on the direction we flip the sign of r. This produces a
    // continuous parametrization of the curve.

    const TWO_PI: f32 = PI * 2.0;
    const PI_OVER_4: f32 = PI / 4.0;

    // Scale the parametric position over the curve to the [0:2*π] range
    let theta = time.fract() * TWO_PI;

    // This variant produces a linear parametrization of theta (triangular signal). Because
    // the parameter r changes much faster around 0 when solving the polar equation, this makes
    // the position "go faster" around the center, which is generally not wanted.
    // let (theta, sign) = if theta <= PI_OVER_2 {
    //     (theta - PI_OVER_4, 1.0)
    // } else {
    //     (3.0 * PI_OVER_4 - theta, -1.0)
    // };

    // That alternative variant "slows down" the parametric position around zero by converting
    // the linear θ variations with a sine function, making it move slower around the edges ±π/4
    // where r tends to zero. This does not produce an exact constant speed, but is visually close.
    let sign = theta.cos().signum();
    let theta = theta.sin() * PI_OVER_4;

    // Solve the polar equation to build the parametric position
    let r2 = radius * radius * (theta * 2.0).cos();
    let r = r2.sqrt().copysign(sign);

    // Convert to cartesian coordinates
    let x = r * theta.cos();
    let y = r * theta.sin();
    Vec2::new(x, y)
}

fn update(time: Res<Time>, mut query: Query<&mut Transform, With<ParticleEffect>>) {
    const ALPHA_OFFSET: f32 = PI * 0.41547;
    const SPEED_OFFSET: f32 = 2.57;
    let mut alpha_off = 0.0;
    let mut speed = 4.25;
    for mut transform in query.iter_mut() {
        let alpha = time.seconds_since_startup() as f32 * PI / speed + alpha_off;
        let radius = 50.0;
        transform.translation = lemniscate(alpha, radius).extend(0.0);
        alpha_off += ALPHA_OFFSET;
        speed += SPEED_OFFSET;
    }
}
