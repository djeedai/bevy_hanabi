use std::collections::HashMap;
use std::error::Error;
use std::ops::{Deref, DerefMut};
#[cfg(not(target_arch = "wasm32"))]
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use bevy::ui_widgets::{Activate, Button};
use bevy::{camera::visibility::RenderLayers, prelude::*};
use bevy::{ecs::system::SystemParam, picking::hover::Hovered};
use bevy_hanabi::prelude::*;

mod utils;
use utils::AppExitIntoResult;

#[derive(Component, Clone)]
pub(crate) struct DemoInfo {
    id: &'static str,
    effects: Vec<Handle<EffectAsset>>,
}

#[derive(Component, Clone, Copy)]
pub(crate) struct Demo {
    install: fn(&mut App, Entity),
    effects: fn() -> Vec<EffectAsset>,
}

impl Demo {
    const fn new(install: fn(&mut App, Entity), effects: fn() -> Vec<EffectAsset>) -> Self {
        Self { install, effects }
    }
}

/// Display name of a demo.
#[derive(Component, Clone, Copy)]
pub(crate) struct DemoName(pub(crate) &'static str);

/// Description text of a demo.
#[derive(Component, Clone, Copy)]
pub(crate) struct DemoDescription(pub(crate) &'static str);

/// Declare a demo.
///
/// The demo module must contain:
/// - `DEMO_DESC: &'str` -- the demo description text
/// - `install()` -- the installation system
/// - `effects()` -- the function returning the list of [`EffectAsset`] used by
///   the demo.
macro_rules! declare_demo {
    ($name:literal) => {
        pub(crate) fn insert_demo(world: &mut World) {
            world.spawn((
                super::Demo::new(install, effects),
                super::DemoName($name),
                super::DemoDescription(DEMO_DESC),
            ));
        }
    };
}

macro_rules! declare_demos {
    ($($module:ident => $path:literal),* $(,)?) => {
        $(
            #[path = $path]
            mod $module;
        )*

        fn insert_demos(world: &mut World) {
            $($module::insert_demo(world);)*
        }
    };
}

declare_demos! {
    d2 => "demo/2d.rs",
    activate => "demo/activate.rs",
    billboard => "demo/billboard.rs",
    circle => "demo/circle.rs",
    expr => "demo/expr.rs",
    firework => "demo/firework.rs",
    force_field => "demo/force_field.rs",
    gradient => "demo/gradient.rs",
    init => "demo/init.rs",
    instancing => "demo/instancing.rs",
    lifetime => "demo/lifetime.rs",
    lightning => "demo/lightning.rs",
    lut => "demo/lut.rs",
    multicam => "demo/multicam.rs",
    ordering => "demo/ordering.rs",
    portal => "demo/portal.rs",
    puffs => "demo/puffs.rs",
    random => "demo/random.rs",
    ribbon => "demo/ribbon.rs",
    spawn => "demo/spawn.rs",
    spawn_on_command => "demo/spawn_on_command.rs",
    visibility => "demo/visibility.rs",
    worms => "demo/worms.rs",
}

#[path = "demo/texutils.rs"]
mod texutils;

pub(crate) fn demo_effect(
    demos: &Query<&DemoInfo>,
    demo_id: &'static str,
    index: usize,
) -> Handle<EffectAsset> {
    demos
        .iter()
        .find(|demo| demo.id == demo_id)
        .and_then(|demo| demo.effects.get(index))
        .unwrap_or_else(|| panic!("missing effect #{index} for demo '{demo_id}'"))
        .clone()
}

/// Install demo systems and preload their effects from the demo entities.
fn install_demos(app: &mut App) {
    let demos = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &Demo, &DemoName)>();
        query
            .iter(world)
            .map(|(entity, demo, name)| (entity, *demo, *name))
            .collect::<Vec<_>>()
    };

    for (entity, demo, name) in demos {
        // Create all effect assets and register them
        let handles = (demo.effects)()
            .into_iter()
            .map(|effect| {
                app.world_mut()
                    .resource_mut::<Assets<EffectAsset>>()
                    .add(effect)
            })
            .collect();

        // Install the demo
        (demo.install)(app, entity);
        app.world_mut().entity_mut(entity).insert(DemoInfo {
            id: name.0,
            effects: handles,
        });
    }
}

#[derive(Component)]
struct MenuButton(Entity);

#[derive(Component)]
struct MenuRoot;

#[derive(Component)]
struct MenuCamera;

#[derive(States, Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DemoState {
    #[default]
    Loading,
    Running,
    Unloading,
}

#[derive(Resource)]
pub(crate) struct DemoSelection(pub(crate) Entity);

#[derive(Resource, Default)]
struct PendingDemoSelection(Option<Entity>);

pub(crate) fn demo_selected(entity: Entity) -> impl Fn(Res<DemoSelection>) -> bool + Clone {
    move |selection: Res<DemoSelection>| selection.0 == entity
}

pub(crate) fn demo_running(
    entity: Entity,
) -> impl Fn(Res<State<DemoState>>, Res<DemoSelection>) -> bool + Clone {
    move |state: Res<State<DemoState>>, selection: Res<DemoSelection>| {
        *state.get() == DemoState::Running && selection.0 == entity
    }
}

#[derive(Component)]
struct DemoContent;

#[derive(SystemParam)]
pub(crate) struct DemoCommands<'w, 's> {
    commands: Commands<'w, 's>,
}

impl DemoCommands<'_, '_> {
    pub(crate) fn spawn<B: Bundle>(&mut self, bundle: B) -> EntityCommands<'_> {
        self.commands.spawn((DemoContent, bundle))
    }
}

impl<'w, 's> Deref for DemoCommands<'w, 's> {
    type Target = Commands<'w, 's>;

    fn deref(&self) -> &Self::Target {
        &self.commands
    }
}

impl DerefMut for DemoCommands<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.commands
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    match parse_cli()? {
        #[cfg(not(target_arch = "wasm32"))]
        CliMode::WriteEffects => process_effect_artifacts(true),
        #[cfg(not(target_arch = "wasm32"))]
        CliMode::CheckEffects => process_effect_artifacts(false),
        CliMode::Run(selected_demo) => run_demo_app(selected_demo),
    }
}

fn run_demo_app(selected_demo: Option<String>) -> Result<(), Box<dyn Error>> {
    let selected = selected_demo
        .map(|id| {
            let mut world = World::new();
            insert_demos(&mut world);
            let mut query = world.query::<&DemoName>();
            query
                .iter(&world)
                .find(|demo| demo.0 == id)
                .map(|demo| demo.0.to_string())
                .ok_or_else(|| format!("unknown demo ID: {id}"))
        })
        .transpose()?;

    let mut app = utils::DemoApp::new("Demo Browser")
        .with_desc("Select a demo from the menu.")
        .build();
    insert_demos(app.world_mut());
    app.world_mut().resource_mut::<utils::Demo>().show_ui = false;
    let selected = selected.as_deref().unwrap_or("firework");
    let selected = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &DemoName)>();
        query
            .iter(world)
            .find(|(_, demo)| demo.0 == selected)
            .map(|(entity, _)| entity)
            .expect("selected demo was validated")
    };
    app.insert_resource(DemoSelection(selected))
        .init_resource::<PendingDemoSelection>()
        .init_state::<DemoState>();

    install_demo_state_systems(&mut app);
    app.add_systems(Startup, (spawn_menu, spawn_menu_camera))
        .add_systems(Update, highlight_menu_buttons)
        .add_systems(Update, sync_demo_info.before(utils::update_demo_ui))
        .add_observer(choose_demo);

    install_demos(&mut app);
    app.run().into_result()
}

enum CliMode {
    Run(Option<String>),
    #[cfg(not(target_arch = "wasm32"))]
    WriteEffects,
    #[cfg(not(target_arch = "wasm32"))]
    CheckEffects,
}

fn parse_cli() -> Result<CliMode, Box<dyn Error>> {
    #[cfg(target_arch = "wasm32")]
    {
        Ok(CliMode::Run(None))
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        match args.as_slice() {
            [] => Ok(CliMode::Run(None)),
            [mode] if mode == "--write-effects" => Ok(CliMode::WriteEffects),
            [mode] if mode == "--check-effects" => Ok(CliMode::CheckEffects),
            [mode, id] if mode == "--demo" => Ok(CliMode::Run(Some(id.to_string()))),
            [flag] if flag == "--help" || flag == "-h" => {
                println!("Usage: cargo run --example demo [-- --demo <ID> | --write-effects | --check-effects]");
                println!("Available demos: {}", {
                    let mut world = World::new();
                    insert_demos(&mut world);
                    let mut query = world.query::<&DemoName>();
                    query
                        .iter(&world)
                        .map(|demo| demo.0)
                        .collect::<Vec<_>>()
                        .join(", ")
                });
                std::process::exit(0);
            }
            [arg, ..] => Err(format!("invalid arguments starting with {arg}").into()),
        }
    }
}

fn begin_demo(mut next_state: ResMut<NextState<DemoState>>) {
    next_state.set(DemoState::Running);
}

fn install_demo_state_systems(app: &mut App) {
    app.add_systems(OnEnter(DemoState::Loading), begin_demo)
        .add_systems(
            OnEnter(DemoState::Unloading),
            (clear_demo_content, prepare_next_demo).chain(),
        );
}

fn prepare_next_demo(
    mut selection: ResMut<DemoSelection>,
    mut pending: ResMut<PendingDemoSelection>,
    mut next_state: ResMut<NextState<DemoState>>,
) {
    if let Some(entity) = pending.0.take() {
        selection.0 = entity;
    }
    next_state.set(DemoState::Loading);
}

fn sync_demo_info(
    selection: Res<DemoSelection>,
    demos: Query<(&DemoName, Option<&DemoDescription>)>,
    mut demo: ResMut<utils::Demo>,
) {
    if !selection.is_changed() && demo.show_ui {
        return;
    }
    if let Ok((name, description)) = demos.get(selection.0) {
        demo.name = name.0.to_string();
        demo.desc = description
            .map_or("", |description| description.0)
            .to_string();
        demo.show_ui = true;
    }
}

fn spawn_menu(mut commands: Commands, demos: Query<(Entity, &DemoName)>) {
    commands
        .spawn((
            Node {
                width: Val::Px(220.),
                height: Val::Percent(100.),
                position_type: PositionType::Absolute,
                top: Val::Px(0.),
                right: Val::Px(0.),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Start,
                row_gap: Val::Px(4.),
                padding: UiRect::all(Val::Px(8.)),
                overflow: Overflow::clip_y(),
                ..default()
            },
            BackgroundColor(Color::srgba(0.035, 0.045, 0.07, 0.92)),
            ZIndex(1),
            MenuRoot,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Hanabi demos"),
                TextFont::from_font_size(20.),
                TextColor(Color::WHITE),
                Node {
                    margin: UiRect::bottom(Val::Px(6.)),
                    ..default()
                },
            ));

            parent
                .spawn(Node {
                    width: Val::Percent(100.),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Start,
                    row_gap: Val::Px(4.),
                    ..default()
                })
                .with_children(|grid| {
                    for (entity, demo) in &demos {
                        grid.spawn((
                            Button,
                            Node {
                                width: Val::Percent(100.),
                                min_height: Val::Px(24.),
                                padding: UiRect::all(Val::Px(4.)),
                                align_items: AlignItems::Center,
                                justify_content: JustifyContent::Center,
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.12, 0.16, 0.24)),
                            Hovered::default(),
                            MenuButton(entity),
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new(demo.0),
                                TextFont::from_font_size(13.),
                                TextColor(Color::WHITE),
                            ));
                        });
                    }
                });
        });
}

fn spawn_menu_camera(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            clear_color: Color::NONE.into(),
            output_mode: utils::ui_camera_output_mode(),
            ..default()
        },
        RenderLayers::layer(utils::UI_RENDER_LAYER),
        Name::new("menu camera"),
        MenuCamera,
    ));
}

fn clear_demo_content(
    mut commands: Commands,
    mut ambient_light: ResMut<GlobalAmbientLight>,
    content: Query<Entity, With<DemoContent>>,
) {
    *ambient_light = default();
    for entity in &content {
        commands.entity(entity).despawn();
    }
}

fn highlight_menu_buttons(
    mut buttons: Query<(&Hovered, &mut BackgroundColor), (Changed<Hovered>, With<MenuButton>)>,
) {
    for (hovered, mut background) in &mut buttons {
        background.0 = if hovered.get() {
            Color::srgb(0.22, 0.36, 0.56)
        } else {
            Color::srgb(0.12, 0.16, 0.24)
        };
    }
}

fn choose_demo(
    activate: On<Activate>,
    buttons: Query<&MenuButton>,
    selection: Res<DemoSelection>,
    mut pending: ResMut<PendingDemoSelection>,
    mut next_state: ResMut<NextState<DemoState>>,
) {
    if let Ok(button) = buttons.get(activate.entity) {
        if button.0 != selection.0 {
            pending.0 = Some(button.0);
            next_state.set(DemoState::Unloading);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn process_effect_artifacts(write: bool) -> Result<(), Box<dyn Error>> {
    let mut app = headless_app();
    let mut generated = Vec::new();
    let mut expected_files = HashSet::new();
    let demo_infos = {
        let world = app.world_mut();
        let mut query = world.query::<&DemoInfo>();
        query.iter(world).cloned().collect::<Vec<_>>()
    };
    let assets = app.world().resource::<Assets<EffectAsset>>();

    for demo in &demo_infos {
        let type_registry = app.world().resource::<AppTypeRegistry>().clone();
        let type_registry = type_registry.read();
        let mut effects = demo
            .effects
            .iter()
            .enumerate()
            .map(|(index, handle)| {
                let effect = assets.get(handle).ok_or_else(|| {
                    format!("demo '{}' effect {index} was not registered", demo.id)
                })?;
                let serialized = effect.serialize(&type_registry)?;
                EffectAsset::deserialize(&serialized, &type_registry)?;
                Ok((effect.name.clone(), serialized))
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        effects.sort_by(|a, b| a.0.cmp(&b.0));
        if effects.is_empty() {
            return Err(format!("demo '{}' constructed no EffectAssets", demo.id).into());
        }

        let mut names = HashMap::new();
        for (index, (name, content)) in effects.into_iter().enumerate() {
            let stem = if name.trim().is_empty() {
                format!("effect-{}", index + 1)
            } else {
                effect_stem(&name)?
            };
            let count = names.entry(stem.clone()).or_insert(0usize);
            *count += 1;
            let filename = if *count == 1 {
                format!("{stem}.effect")
            } else {
                format!("{stem}-{}.effect", *count)
            };
            let relative_path = PathBuf::from("examples/effects")
                .join(demo.id)
                .join(filename);
            expected_files.insert(relative_path.clone());
            generated.push((relative_path, content));
        }
    }

    for (relative_path, content) in generated {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(&relative_path);
        if write {
            fs::create_dir_all(path.parent().expect("artifact has parent"))?;
            fs::write(&path, content)?;
            println!("wrote {}", relative_path.display());
        } else {
            let expected = fs::read_to_string(&path)
                .map_err(|error| format!("cannot read {}: {error}", relative_path.display()))?;
            if expected != content {
                return Err(format!(
                    "{} differs from regenerated output; run --write-effects",
                    relative_path.display()
                )
                .into());
            }
        }
    }
    if !write {
        for demo in &demo_infos {
            let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("examples/effects")
                .join(demo.id);
            if !directory.exists() {
                continue;
            }
            for entry in fs::read_dir(&directory)? {
                let entry = entry?;
                let path = entry.path();
                if path
                    .extension()
                    .is_some_and(|extension| extension == "effect")
                {
                    let relative_path =
                        path.strip_prefix(env!("CARGO_MANIFEST_DIR"))?.to_path_buf();
                    if !expected_files.contains(&relative_path) {
                        return Err(format!(
                            "unexpected effect artifact {}; remove it or add it to the demo",
                            relative_path.display()
                        )
                        .into());
                    }
                }
            }
        }
    }
    Ok(())
}

/// Build a headless app for effect asset generation or validation.
#[cfg(not(target_arch = "wasm32"))]
fn headless_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        TaskPoolPlugin::default(),
        AssetPlugin::default(),
        bevy::state::app::StatesPlugin,
    ))
    .init_asset::<EffectAsset>()
    .init_asset::<Mesh>()
    .init_asset::<Image>()
    .init_asset::<StandardMaterial>()
    .init_asset::<ColorMaterial>()
    .init_asset::<bevy::world_serialization::WorldAsset>()
    .register_type::<EffectAsset>()
    .register_type::<SpawnerSettings>();
    register_modifiers(app.world().resource::<AppTypeRegistry>());
    insert_demos(app.world_mut());
    let selected = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &DemoName)>();
        query
            .iter(world)
            .find(|(_, name)| name.0 == "firework")
            .map(|(entity, _)| entity)
            .expect("firework demo is registered")
    };
    app.insert_resource(DemoSelection(selected))
        .init_resource::<PendingDemoSelection>()
        .init_state::<DemoState>();
    install_demo_state_systems(&mut app);
    install_demos(&mut app);
    app
}

#[cfg(not(target_arch = "wasm32"))]
fn effect_stem(name: &str) -> Result<String, Box<dyn Error>> {
    let stem = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    let stem = stem.trim_matches('-');
    if stem.is_empty() {
        return Err(format!("effect name '{name}' cannot be used as a filename").into());
    }
    Ok(stem.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spawn_demo_content(mut commands: DemoCommands) {
        commands
            .spawn(Name::new("demo content"))
            .with_children(|parent| {
                parent.spawn(Name::new("demo child"));
            });
    }

    #[test]
    fn demo_entities_have_metadata_and_named_effects() {
        let mut world = World::new();
        insert_demos(&mut world);
        let mut query = world.query::<(&Demo, &DemoName, &DemoDescription)>();
        for (demo, name, description) in query.iter(&world) {
            assert!(!name.0.trim().is_empty(), "demo has no name");
            assert!(
                !description.0.trim().is_empty(),
                "demo '{}' has no description",
                name.0
            );
            let effects = (demo.effects)();
            assert!(!effects.is_empty(), "demo '{}' has no effects", name.0);
            for effect in effects {
                assert!(
                    !effect.name.trim().is_empty(),
                    "demo '{}' has an unnamed effect",
                    name.0
                );
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn registered_demos_store_handles_to_preloaded_effects() {
        let mut app = headless_app();
        let demo_infos = {
            let world = app.world_mut();
            let mut query = world.query::<&DemoInfo>();
            query.iter(world).cloned().collect::<Vec<_>>()
        };
        let demo_count = {
            let world = app.world_mut();
            let mut query = world.query::<&Demo>();
            query.iter(world).count()
        };
        let assets = app.world().resource::<Assets<EffectAsset>>();

        assert_eq!(demo_infos.len(), demo_count);
        for demo in demo_infos {
            assert!(
                !demo.effects.is_empty(),
                "demo '{}' registered no effects",
                demo.id
            );
            for handle in demo.effects {
                assert!(
                    assets.get(&handle).is_some(),
                    "demo '{}' references an effect that was not preloaded",
                    demo.id
                );
            }
        }
    }

    #[test]
    fn unloading_clears_demo_content_before_loading_again() {
        let mut app = App::new();
        app.add_plugins(bevy::state::app::StatesPlugin)
            .insert_resource(DemoSelection(Entity::PLACEHOLDER))
            .init_resource::<PendingDemoSelection>()
            .init_resource::<GlobalAmbientLight>()
            .init_state::<DemoState>()
            .add_systems(
                OnEnter(DemoState::Running),
                spawn_demo_content.run_if(demo_selected(Entity::PLACEHOLDER)),
            )
            .add_systems(OnEnter(DemoState::Loading), begin_demo)
            .add_systems(
                OnEnter(DemoState::Unloading),
                (clear_demo_content, prepare_next_demo).chain(),
            );
        app.update();
        app.update();

        let mut content = app.world_mut().query_filtered::<Entity, With<Name>>();
        assert_eq!(content.iter(app.world()).count(), 2);

        app.world_mut()
            .resource_mut::<NextState<DemoState>>()
            .set(DemoState::Unloading);
        app.update();
        assert_eq!(content.iter(app.world()).count(), 0);

        app.update();
        app.update();
        assert_eq!(content.iter(app.world()).count(), 2);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn selecting_demo_keeps_menu_visible_and_runs_only_selected_demo() {
        let mut app = headless_app();
        app.add_plugins(bevy::time::TimePlugin)
            .init_asset::<AnimationGraph>()
            .init_resource::<Time<EffectSimulation>>()
            .add_observer(choose_demo);
        app.world_mut().spawn((Node::default(), MenuRoot));
        let original_menu_camera = app.world_mut().spawn((Camera2d, MenuCamera)).id();
        app.update();

        for id in [
            "expr",
            "lightning",
            "puffs",
            "portal",
            "spawn",
            "puffs",
            "gradient",
            "ribbon",
            "firework",
        ] {
            let demo_entity = {
                let world = app.world_mut();
                let mut demos = world.query::<(Entity, &DemoName)>();
                demos
                    .iter(world)
                    .find(|(_, name)| name.0 == id)
                    .map(|(entity, _)| entity)
                    .unwrap()
            };
            let button = app.world_mut().spawn(MenuButton(demo_entity)).id();
            app.world_mut().trigger(Activate { entity: button });
            for _ in 0..5 {
                app.update();
            }

            let mut cameras = app
                .world_mut()
                .query_filtered::<&Camera, (With<DemoContent>, With<Camera3d>)>();
            assert_eq!(
                cameras
                    .iter(app.world())
                    .filter(|camera| camera.is_active)
                    .count(),
                1,
                "demo '{id}' did not spawn an active camera"
            );

            let mut menu = app.world_mut().query_filtered::<Entity, With<MenuRoot>>();
            assert_eq!(menu.iter(app.world()).count(), 1);

            let mut menu_cameras = app
                .world_mut()
                .query_filtered::<(Entity, &Camera), With<MenuCamera>>();
            let (menu_camera, camera) = menu_cameras.single(app.world()).unwrap();
            assert_eq!(menu_camera, original_menu_camera);
            assert!(camera.is_active);
            assert_eq!(app.world().resource::<DemoSelection>().0, demo_entity);
            if id == "puffs" {
                let mut effects = app
                    .world_mut()
                    .query_filtered::<Entity, (With<ParticleEffect>, With<DemoContent>)>();
                assert_eq!(effects.iter(app.world()).count(), 1);
            }
            assert_eq!(
                *app.world().resource::<State<DemoState>>().get(),
                DemoState::Running
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn initially_selected_hdr_demos_spawn_their_camera() {
        for id in ["expr", "firework"] {
            let mut app = headless_app();
            let demo_entity = {
                let world = app.world_mut();
                let mut demos = world.query::<(Entity, &DemoName)>();
                demos
                    .iter(world)
                    .find(|(_, name)| name.0 == id)
                    .map(|(entity, _)| entity)
                    .unwrap()
            };
            app.world_mut().resource_mut::<DemoSelection>().0 = demo_entity;
            app.add_plugins(bevy::time::TimePlugin)
                .init_resource::<Time<EffectSimulation>>();

            for _ in 0..3 {
                app.update();
            }

            let mut cameras = app
                .world_mut()
                .query_filtered::<&Camera, (With<DemoContent>, With<Camera3d>)>();
            assert_eq!(
                cameras
                    .iter(app.world())
                    .filter(|camera| camera.is_active)
                    .count(),
                1,
                "demo '{id}' did not spawn its camera on initial selection"
            );
        }
    }
}
