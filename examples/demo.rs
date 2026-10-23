use bevy::prelude::*;
use bevy::ui_widgets::{Activate, Button};
use bevy::{ecs::system::SystemParam, picking::hover::Hovered};
use bevy_hanabi::prelude::*;
use std::error::Error;
use std::ops::{Deref, DerefMut};
#[cfg(not(target_arch = "wasm32"))]
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

mod utils;
use utils::AppExitIntoResult;

#[derive(Clone)]
struct DemoInfo {
    id: &'static str,
    description: &'static str,
}

pub(crate) trait Demo: Send + Sync {
    fn id(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn register(&self, app: &mut App);
}

macro_rules! declare_demos {
    ($($module:ident => $path:literal),* $(,)?) => {
        $(
            #[path = $path]
            mod $module;
        )*

        fn demos() -> Vec<Box<dyn Demo>> {
            vec![$(Box::new($module::register()),)*]
        }
    };
}

declare_demos! {
    two_d => "demo/2d.rs",
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

/// Simple concrete type to implement the [`Demo`] trait.
pub(crate) struct DemoRegistration {
    id: &'static str,
    description: &'static str,
    register: fn(&mut App),
}

impl DemoRegistration {
    pub(crate) fn new(id: &'static str, description: &'static str, register: fn(&mut App)) -> Self {
        Self {
            id,
            description,
            register,
        }
    }
}

impl Demo for DemoRegistration {
    fn id(&self) -> &'static str {
        self.id
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn register(&self, app: &mut App) {
        (self.register)(app);
    }
}

/// Catalog of all registered demos.
#[derive(Resource, Default)]
struct DemoCatalog(Vec<DemoInfo>);

/// Register all demos with the given app.
fn register_demos(app: &mut App, demos: Vec<Box<dyn Demo>>) {
    app.insert_resource(DemoCatalog(
        demos
            .iter()
            .map(|demo| DemoInfo {
                id: demo.id(),
                description: demo.description(),
            })
            .collect(),
    ));
    for demo in demos {
        demo.register(app);
    }
}

#[derive(Component)]
struct MenuButton(&'static str);

#[derive(Component)]
struct MenuRoot;

#[derive(Component)]
struct MenuCamera;

#[derive(Resource, Default)]
pub(crate) struct DemoSelection {
    id: Option<String>,
    artifact_mode: bool,
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

pub(crate) fn selected_demo(id: &'static str) -> impl Fn(Res<DemoSelection>) -> bool + Clone {
    move |selection: Res<DemoSelection>| {
        selection.id.as_deref() == Some(id) && !selection.artifact_mode
    }
}

pub(crate) fn selected_once(
    id: &'static str,
) -> impl FnMut(Res<DemoSelection>, Local<bool>) -> bool + Clone {
    move |selection: Res<DemoSelection>, mut has_run: Local<bool>| {
        if selection.id.as_deref() == Some(id) {
            if !*has_run {
                *has_run = true;
                return true;
            }
        } else {
            *has_run = false;
        }
        false
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
    let demos = demos();
    let selected = selected_demo
        .map(|id| {
            demos
                .iter()
                .find(|demo| demo.id() == id)
                .map(|demo| demo.id().to_string())
                .ok_or_else(|| format!("unknown demo ID: {id}"))
        })
        .transpose()?;

    let mut app = utils::DemoApp::new("Demo Browser")
        .with_desc("Select a demo from the menu.")
        .build();
    app.world_mut().resource_mut::<utils::Demo>().show_ui = false;
    app.insert_resource(DemoSelection {
        id: None,
        artifact_mode: false,
    });

    app.add_systems(
        Startup,
        (
            spawn_menu.run_if(no_demo_selected),
            spawn_menu_camera_if_missing.run_if(no_demo_selected),
        )
            .chain(),
    )
    .add_systems(
        Update,
        (
            return_to_menu_on_esc,
            clear_demo_content.run_if(no_demo_selected),
            spawn_menu_if_missing.run_if(no_demo_selected),
            spawn_menu_camera_if_missing.run_if(no_demo_selected),
        )
            .chain(),
    )
    .add_systems(Update, highlight_menu_buttons)
    .add_systems(Update, sync_demo_info.before(utils::update_demo_ui))
    .add_observer(choose_demo);

    register_demos(&mut app, demos);
    app.world_mut().resource_mut::<DemoSelection>().id = selected;

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
                println!(
                    "Available demos: {}",
                    demos()
                        .iter()
                        .map(|demo| demo.id())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                std::process::exit(0);
            }
            [arg, ..] => Err(format!("invalid arguments starting with {arg}").into()),
        }
    }
}

fn no_demo_selected(selection: Res<DemoSelection>) -> bool {
    selection.id.is_none()
}

fn return_to_menu_on_esc(input: Res<ButtonInput<KeyCode>>, mut selection: ResMut<DemoSelection>) {
    if input.just_pressed(KeyCode::Escape) {
        selection.id = None;
    }
}

fn sync_demo_info(
    selection: Res<DemoSelection>,
    catalog: Res<DemoCatalog>,
    mut demo: ResMut<utils::Demo>,
) {
    let Some(id) = selection.id.as_deref() else {
        if demo.show_ui {
            demo.show_ui = false;
        }
        return;
    };
    if !selection.is_changed() && demo.show_ui {
        return;
    }
    if let Some(entry) = catalog.0.iter().find(|entry| entry.id == id) {
        demo.name = entry.id.to_string();
        demo.desc = entry.description.to_string();
        demo.show_ui = true;
    }
}

fn spawn_menu(mut commands: Commands, catalog: Res<DemoCatalog>) {
    commands
        .spawn((
            Node {
                width: Val::Percent(100.),
                height: Val::Percent(100.),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                row_gap: Val::Px(8.),
                overflow: Overflow::clip_y(),
                ..default()
            },
            BackgroundColor(Color::srgb(0.035, 0.045, 0.07)),
            MenuRoot,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Hanabi demos"),
                TextFont::from_font_size(32.),
                TextColor(Color::WHITE),
                Node {
                    margin: UiRect::bottom(Val::Px(14.)),
                    ..default()
                },
            ));

            parent
                .spawn(Node {
                    width: Val::Vw(80.),
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                    column_gap: Val::Px(8.),
                    row_gap: Val::Px(8.),
                    ..default()
                })
                .with_children(|grid| {
                    for demo in &catalog.0 {
                        grid.spawn((
                            Button,
                            Node {
                                width: Val::Px(240.),
                                min_height: Val::Px(48.),
                                padding: UiRect::all(Val::Px(8.)),
                                align_items: AlignItems::Center,
                                justify_content: JustifyContent::Center,
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.12, 0.16, 0.24)),
                            Hovered::default(),
                            MenuButton(demo.id),
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new(demo.id),
                                TextFont::from_font_size(16.),
                                TextColor(Color::WHITE),
                            ));
                        });
                    }
                });
        });
}

fn spawn_menu_if_missing(
    commands: Commands,
    catalog: Res<DemoCatalog>,
    menu: Query<(), With<MenuRoot>>,
) {
    if menu.is_empty() {
        spawn_menu(commands, catalog);
    }
}

fn spawn_menu_camera_if_missing(mut commands: Commands, cameras: Query<(), With<MenuCamera>>) {
    if cameras.is_empty() {
        commands.spawn((Camera2d, Name::new("menu camera"), MenuCamera));
    }
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
    mut commands: Commands,
    mut selection: ResMut<DemoSelection>,
    q_delete: Query<Entity, Or<(With<Node>, With<Camera>)>>,
) {
    if let Ok(button) = buttons.get(activate.entity) {
        selection.id = Some(button.0.to_string());

        // Clean-up everything that could leak into the new demo about to start
        for entity in &q_delete {
            commands.entity(entity).despawn();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn process_effect_artifacts(write: bool) -> Result<(), Box<dyn Error>> {
    let mut app = headless_app();
    let mut generated = Vec::new();
    let mut expected_files = HashSet::new();
    let demos = app.world().resource::<DemoCatalog>().0.clone();

    for demo in &demos {
        let before = app
            .world()
            .resource::<Assets<EffectAsset>>()
            .iter()
            .map(|(id, _)| id)
            .collect::<HashSet<_>>();
        app.world_mut().resource_mut::<DemoSelection>().id = Some(demo.id.to_string());
        app.world_mut()
            .resource_mut::<DemoSelection>()
            .artifact_mode = true;
        app.world_mut().run_schedule(Update);

        let type_registry = app.world().resource::<AppTypeRegistry>().clone();
        let type_registry = type_registry.read();
        let assets = app.world().resource::<Assets<EffectAsset>>();
        let mut effects = assets
            .iter()
            .filter(|(id, _)| !before.contains(id))
            .map(|(_, effect)| {
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
        for demo in &demos {
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
    .init_resource::<DemoSelection>()
    .init_asset::<EffectAsset>()
    .init_asset::<Mesh>()
    .init_asset::<Image>()
    .init_asset::<StandardMaterial>()
    .init_asset::<ColorMaterial>()
    .init_asset::<bevy::world_serialization::WorldAsset>()
    .register_type::<EffectAsset>()
    .register_type::<SpawnerSettings>();
    register_modifiers(app.world().resource::<AppTypeRegistry>());
    register_demos(&mut app, demos());
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
    fn returning_to_menu_clears_demo_content_and_allows_reentry() {
        let mut app = App::new();
        app.insert_resource(DemoSelection::default())
            .init_resource::<GlobalAmbientLight>()
            .add_systems(
                Update,
                spawn_demo_content.run_if(selected_once("test-demo")),
            )
            .add_systems(Update, clear_demo_content.run_if(no_demo_selected));

        app.world_mut().resource_mut::<DemoSelection>().id = Some("test-demo".to_string());
        app.update();

        let mut content = app.world_mut().query_filtered::<Entity, With<Name>>();
        assert_eq!(content.iter(app.world()).count(), 2);

        app.world_mut().resource_mut::<DemoSelection>().id = None;
        app.update();
        assert_eq!(content.iter(app.world()).count(), 0);

        app.world_mut().resource_mut::<DemoSelection>().id = Some("test-demo".to_string());
        app.update();
        assert_eq!(content.iter(app.world()).count(), 2);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn selected_demos_spawn_their_camera() {
        for id in ["expr", "firework", "gradient", "ribbon"] {
            let mut app = headless_app();
            app.add_plugins(bevy::time::TimePlugin)
                .init_resource::<Time<EffectSimulation>>()
                .add_systems(
                    Update,
                    (
                        clear_demo_content.run_if(no_demo_selected),
                        spawn_menu_if_missing.run_if(no_demo_selected),
                        spawn_menu_camera_if_missing.run_if(no_demo_selected),
                    )
                        .chain(),
                )
                .add_observer(choose_demo);
            app.world_mut().spawn((Node::default(), MenuRoot));
            let original_menu_camera = app.world_mut().spawn((Camera2d, MenuCamera)).id();
            let button = app.world_mut().spawn(MenuButton(id)).id();
            app.world_mut().trigger(Activate { entity: button });
            app.world_mut().run_schedule(Update);

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
            assert_eq!(menu.iter(app.world()).count(), 0);
            let mut nodes = app.world_mut().query_filtered::<Entity, With<Node>>();
            assert_eq!(nodes.iter(app.world()).count(), 0);

            let mut menu_cameras = app
                .world_mut()
                .query_filtered::<(Entity, &Camera), With<MenuCamera>>();
            assert_eq!(menu_cameras.iter(app.world()).count(), 0);
            assert!(app.world().get_entity(original_menu_camera).is_err());

            app.world_mut().resource_mut::<DemoSelection>().id = None;
            app.world_mut().run_schedule(Update);

            let mut cameras = app
                .world_mut()
                .query_filtered::<Entity, (With<DemoContent>, With<Camera3d>)>();
            assert_eq!(cameras.iter(app.world()).count(), 0);
            assert_eq!(menu.iter(app.world()).count(), 1);
            let (menu_camera, camera) = menu_cameras.single(app.world()).unwrap();
            assert_ne!(menu_camera, original_menu_camera);
            assert!(camera.is_active);
        }
    }
}
