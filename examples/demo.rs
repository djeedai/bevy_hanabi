use std::{
    error::Error,
    ops::{Deref, DerefMut},
};
#[cfg(not(target_arch = "wasm32"))]
use std::{
    fs,
    path::{Path, PathBuf},
};

#[cfg(not(target_arch = "wasm32"))]
use bevy::platform::collections::HashMap;
use bevy::{
    camera::visibility::RenderLayers,
    color::Luminance,
    ecs::system::SystemParam,
    input::{
        keyboard::{KeyCode, KeyboardInput},
        ButtonState,
    },
    input_focus::{
        tab_navigation::{TabGroup, TabIndex, TabNavigationPlugin},
        FocusGained, FocusedInput, InputFocus, InputFocusVisible,
    },
    picking::{hover::Hovered, Pickable},
    platform::collections::HashSet,
    prelude::*,
    text::{EditableText, FontWeight, TextCursorStyle, TextEditChange},
    ui::{ComputedNode, FocusPolicy, UiGlobalTransform, UiSystems},
    ui_widgets::{
        Activate, Button, ControlOrientation, ScrollArea, ScrollIntoView, Scrollbar,
        ScrollbarThumb, TextInput,
    },
};
use bevy_hanabi::prelude::*;

mod utils;
use utils::AppExitIntoResult;

/// Default number of frames to render in --bootstrap mode, after all effects
/// reported ready. Override with --bootstrap=N.
const BOOTSTRAP_FRAMES: usize = 60;

/// Timeout, in frames, to wait in --bootstrap mode for effects to become ready.
/// This depends on the GPU (PSO compilation, notably) so we keep this fairly
/// large.
const BOOTSTRAP_READY_TIMEOUT_FRAMES: usize = 600;

const MENU_WIDTH: f32 = 320.;
const MENU_GAP: f32 = 16.;
const MENU_FOCUS_COLOR: Color = Color::oklab(0.67, 0.08, -0.12);
const DISABLED_TAG_COLOR: Color = Color::oklch(0.38, 0.01, 258.);
const DISABLED_TAG_TEXT_COLOR: Color = Color::srgb(0.62, 0.62, 0.64);

#[derive(Component, Clone)]
pub(crate) struct DemoInfo {
    id: &'static str,
    effects: Vec<Handle<EffectAsset>>,
}

/// Component representing a demo entry.
#[derive(Component, Clone, Copy)]
pub(crate) struct Demo {
    /// Callback to install the demo into the browser's app. This is executed
    /// only once.
    install: fn(&mut App, Entity),
    /// Callback to create all the [`EffectAsset`] that the demo will use. Those
    /// assets are automatically registered by the demo app into
    /// [`Assets<EffectAsset>`]. They are also used for automated testing of
    /// serialization.
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

/// Short menu summary of a demo.
#[derive(Component, Clone, Copy)]
pub(crate) struct DemoSummary(pub(crate) &'static str);

/// Concepts demonstrated by a demo.
#[derive(Component, Clone, Copy)]
pub(crate) struct DemoTags(pub(crate) &'static [DemoTag]);

/// A consistent tag used to categorize demos.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DemoTag {
    D2,
    AlphaMask,
    Attributes,
    Blending,
    Camera,
    Expr,
    Flipbook,
    Hdr,
    Mesh,
    Modifier,
    Perf,
    Properties,
    Spawner,
    Texture,
    Trail,
    Visibility,
}

impl DemoTag {
    const ALL: &[Self] = &[
        Self::D2,
        Self::AlphaMask,
        Self::Attributes,
        Self::Blending,
        Self::Camera,
        Self::Expr,
        Self::Flipbook,
        Self::Hdr,
        Self::Mesh,
        Self::Modifier,
        Self::Perf,
        Self::Properties,
        Self::Spawner,
        Self::Texture,
        Self::Trail,
        Self::Visibility,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::D2 => "2d",
            Self::AlphaMask => "alphamask",
            Self::Attributes => "attributes",
            Self::Blending => "blending",
            Self::Camera => "camera",
            Self::Expr => "expr",
            Self::Flipbook => "flipbook",
            Self::Hdr => "hdr",
            Self::Mesh => "mesh",
            Self::Modifier => "modifier",
            Self::Perf => "perf",
            Self::Properties => "properties",
            Self::Spawner => "spawner",
            Self::Texture => "texture",
            Self::Trail => "trail",
            Self::Visibility => "visibility",
        }
    }

    fn color(self) -> Color {
        const LIGHTNESS: f32 = 0.48;
        const CHROMA: f32 = 0.14;

        let hue = match self {
            Self::D2 => 0.,
            Self::AlphaMask => 22.5,
            Self::Attributes => 45.,
            Self::Blending => 67.5,
            Self::Camera => 90.,
            Self::Expr => 112.5,
            Self::Flipbook => 135.,
            Self::Hdr => 157.5,
            Self::Mesh => 180.,
            Self::Modifier => 202.5,
            Self::Perf => 225.,
            Self::Properties => 247.5,
            Self::Spawner => 270.,
            Self::Texture => 292.5,
            Self::Trail => 315.,
            Self::Visibility => 337.5,
        };

        Color::oklch(LIGHTNESS, CHROMA, hue)
    }
}

/// Declare a demo.
///
/// The demo module must contain:
/// - `DEMO_DESC: &'str` -- the demo description text
/// - `DEMO_SUMMARY: &'str` -- the short menu summary
/// - `DEMO_TAGS: &[DemoTag]` -- concepts demonstrated by the demo
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
                super::DemoSummary(DEMO_SUMMARY),
                super::DemoTags(DEMO_TAGS),
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
struct MenuDemoList;

#[derive(Component)]
struct MenuRoot;

#[derive(Component)]
struct MenuDemoEntry {
    name: &'static str,
    tags: &'static [DemoTag],
}

#[derive(Component)]
struct MenuTagButton(DemoTag);

#[derive(Component)]
struct MenuTagLabel(DemoTag);

#[derive(Component)]
struct MenuEmptyState;

#[derive(Component)]
struct MenuSearchInput;

#[derive(Component)]
struct MenuSearchInputContent;

#[derive(Component)]
struct MenuSearchPlaceholder;

#[derive(Component)]
struct MenuSearchFocusIndicator;

#[derive(Component)]
struct MenuCamera;

#[derive(Resource, Default)]
struct MenuSearch {
    name: String,
    disabled_tags: HashSet<DemoTag>,
}

#[derive(States, Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DemoState {
    #[default]
    Loading,
    Running,
    Unloading,
}

/// Currently select demo, referenced by the entity holding the [`Demo`]
/// component.
#[derive(Resource)]
pub(crate) struct DemoSelection(pub(crate) Entity);

/// Next demo to select once unloading of current demo is done.
#[derive(Resource, Default)]
struct NextDemoSelection(Option<Entity>);

/// Active menu entry in the list of demos.
#[derive(Resource)]
struct ActiveMenuDemo(Entity);

/// Runner condition for systems that need to run once a specific demo is
/// selected. This is generally paired with the`OnEnter(DemoState::Running)`
/// schedule to run a one-time setup each time a demo starts.
pub(crate) fn demo_selected(entity: Entity) -> impl Fn(Res<DemoSelection>) -> bool + Clone {
    move |selection: Res<DemoSelection>| selection.0 == entity
}

/// Runner condition for systems that need to run while a specific demo is
/// running. This is generally used for the `update()` system of a demo, to
/// ensure it only runs while the demo is active.
pub(crate) fn demo_running(
    entity: Entity,
) -> impl Fn(Res<State<DemoState>>, Res<DemoSelection>) -> bool + Clone {
    move |state: Res<State<DemoState>>, selection: Res<DemoSelection>| {
        *state.get() == DemoState::Running && selection.0 == entity
    }
}

/// Marker for entities spawned by a demo, which need to be cleaned-up when
/// changing demo.
#[derive(Component)]
struct DemoContent;

/// Wrapper around [`Commands`] to automatically insert [`DemoContent`] when
/// spawning an entity inside a demo.
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
        CliMode::Run {
            selected_demo,
            bootstrap_frames,
        } => run_demo_app(selected_demo, bootstrap_frames),
    }
}

/// Run the main demo GUI app with the menu.
fn run_demo_app(
    selected_demo: Option<String>,
    bootstrap_frames: Option<usize>,
) -> Result<(), Box<dyn Error>> {
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
        .with_desc_position(utils::DescPosition::RightColumn {
            left: Val::Px(MENU_WIDTH + MENU_GAP),
            top: Val::Px(MENU_GAP),
        })
        .build();
    insert_demos(app.world_mut());
    app.world_mut().resource_mut::<utils::Demo>().show_ui = false;
    let (demo_entities, selected) = {
        let world = app.world_mut();
        let mut query = world.query::<(Entity, &DemoName)>();
        let mut demos = query
            .iter(world)
            .map(|(entity, demo)| (entity, demo.0))
            .collect::<Vec<_>>();
        demos.sort_by_key(|(_, name)| *name);
        let selected_name = if bootstrap_frames.is_some() {
            demos.first().map(|(_, name)| *name)
        } else {
            Some(selected.as_deref().unwrap_or("firework"))
        }
        .expect("at least one demo is registered");
        let selected = demos
            .iter()
            .find(|(_, name)| *name == selected_name)
            .map(|(entity, _)| *entity)
            .expect("selected demo was validated");
        (
            demos
                .into_iter()
                .map(|(entity, _)| entity)
                .collect::<Vec<_>>(),
            selected,
        )
    };
    app.insert_resource(DemoSelection(selected))
        .insert_resource(ActiveMenuDemo(selected))
        .init_resource::<NextDemoSelection>()
        .init_resource::<MenuSearch>()
        .init_state::<DemoState>();

    install_demo_state_systems(&mut app);
    if let Some(frames_per_demo) = bootstrap_frames {
        app.insert_resource(Bootstrap {
            demos: demo_entities,
            current: 0,
            frames: 0,
            frames_per_demo,
            waited_frames: 0,
            timed_out: false,
        })
        .add_systems(Update, bootstrap_demo);
    }
    app.add_plugins(TabNavigationPlugin);
    app.add_systems(Startup, (spawn_menu, spawn_menu_camera))
        .add_systems(
            Update,
            (
                highlight_menu_buttons,
                update_menu_filter,
                update_menu_search_focus_indicator,
            ),
        )
        .add_systems(Update, sync_demo_info.before(utils::update_demo_ui))
        .add_systems(
            PostUpdate,
            scroll_initial_active_demo_into_view.after(UiSystems::Layout),
        )
        .add_observer(search_text_changed)
        .add_observer(toggle_menu_tag)
        .add_observer(choose_demo)
        .add_observer(navigate_demo_list);

    install_demos(&mut app);
    app.run().into_result()
}

/// App run mode determined by CLI args.
enum CliMode {
    /// Run the normal GUI app.
    Run {
        selected_demo: Option<String>,
        bootstrap_frames: Option<usize>,
    },

    /// Serialize all demo's effects to `.effect` files. This is used to
    /// regenerate the reference files when a demo changed.
    #[cfg(not(target_arch = "wasm32"))]
    WriteEffects,

    /// Serialize all demo's effects and compare them with checked-in `.effect`
    /// files. This is used to validate the serialization format didn't change
    /// unintentionally.
    #[cfg(not(target_arch = "wasm32"))]
    CheckEffects,
}

/// Parse the CLI args and return the [`CliMode`] the app should run in.
fn parse_cli() -> Result<CliMode, Box<dyn Error>> {
    #[cfg(target_arch = "wasm32")]
    {
        Ok(CliMode::Run {
            selected_demo: None,
            bootstrap_frames: None,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        match args.as_slice() {
            [] => Ok(CliMode::Run {
                selected_demo: None,
                bootstrap_frames: None,
            }),
            [mode] if mode == "--write-effects" => Ok(CliMode::WriteEffects),
            [mode] if mode == "--check-effects" => Ok(CliMode::CheckEffects),
            [flag] if *flag == "--bootstrap" || flag.starts_with("--bootstrap=") => {
                let frame_count = if *flag == "--bootstrap" {
                    None
                } else {
                    flag.strip_prefix("--bootstrap=")
                };
                Ok(CliMode::Run {
                    selected_demo: None,
                    bootstrap_frames: Some(parse_bootstrap_frames(frame_count)?),
                })
            }
            [mode, id] if mode == "--demo" => Ok(CliMode::Run {
                selected_demo: Some(id.to_string()),
                bootstrap_frames: None,
            }),
            [flag] if flag == "--help" || flag == "-h" => {
                println!("Usage: cargo run --example demo [-- --demo <ID> | --bootstrap[=<FRAMES>] | --write-effects | --check-effects]");
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

/// Parse the `--bootstrap=N` CLI arg frame count.
#[cfg(not(target_arch = "wasm32"))]
fn parse_bootstrap_frames(value: Option<&str>) -> Result<usize, Box<dyn Error>> {
    let Some(value) = value else {
        return Ok(BOOTSTRAP_FRAMES);
    };
    let frames = value
        .parse::<usize>()
        .map_err(|_| format!("invalid bootstrap frame count: {value}"))?;
    if frames == 0 {
        return Err("bootstrap frame count must be greater than zero".into());
    }
    Ok(frames)
}

/// Bootstrap readiness for the currently running demo.
enum BootstrapReadiness {
    /// Waiting for the demo's effect(s) to become ready.
    Waiting,
    /// Demo effects are ready.
    Ready,
    /// Timed out while waiting for effect(s) to become ready.
    TimedOut,
}

enum BootstrapProgress {
    Waiting,
    Next(Entity),
    Finished,
}

/// Bootstrap mode parameters.
#[derive(Resource)]
struct Bootstrap {
    /// Demo(s) to run in the bootstrap.
    demos: Vec<Entity>,
    /// Index of the currently running demo.
    current: usize,
    /// Number of frames simulated in the currently running demo.
    frames: usize,
    /// Maximum number of frames to simulate per demo.
    frames_per_demo: usize,
    /// Number of frames the current demo spent waiting for its effect(s) to
    /// become ready.
    waited_frames: usize,
    /// True if the currently running demo timed out waiting for its effect(s)
    /// to become ready.
    timed_out: bool,
}

impl Bootstrap {
    fn wait_for_effects(&mut self, effects_ready: bool) -> BootstrapReadiness {
        if effects_ready {
            return BootstrapReadiness::Ready;
        }
        if self.timed_out {
            return BootstrapReadiness::TimedOut;
        }

        self.waited_frames += 1;
        if self.waited_frames >= BOOTSTRAP_READY_TIMEOUT_FRAMES {
            self.timed_out = true;
            BootstrapReadiness::TimedOut
        } else {
            BootstrapReadiness::Waiting
        }
    }

    fn advance(&mut self) -> BootstrapProgress {
        self.frames += 1;
        if self.frames < self.frames_per_demo {
            return BootstrapProgress::Waiting;
        }
        self.frames = 0;

        if let Some(next) = self.demos.get(self.current + 1).copied() {
            self.current += 1;
            self.waited_frames = 0;
            self.timed_out = false;
            BootstrapProgress::Next(next)
        } else {
            BootstrapProgress::Finished
        }
    }
}

/// Drive the demo bootstrap loop, switching between demos as the complete.
fn bootstrap_demo(
    mut bootstrap: ResMut<Bootstrap>,
    state: Res<State<DemoState>>,
    effects: Query<Option<&CompiledParticleEffect>, With<ParticleEffect>>,
    demo_names: Query<&DemoName>,
    mut next_demo: ResMut<NextDemoSelection>,
    mut next_state: ResMut<NextState<DemoState>>,
    mut app_exit: MessageWriter<AppExit>,
) {
    if *state.get() != DemoState::Running {
        return;
    }

    let mut effect_count = 0;
    let mut effects_ready = true;
    for compiled in effects.iter() {
        effect_count += 1;
        effects_ready &= compiled.is_some_and(CompiledParticleEffect::is_ready);
    }
    effects_ready &= effect_count > 0;
    let was_timed_out = bootstrap.timed_out;
    match bootstrap.wait_for_effects(effects_ready) {
        BootstrapReadiness::Waiting => {
            bootstrap.frames = 0;
            return;
        }
        BootstrapReadiness::TimedOut if !was_timed_out => {
            let demo = bootstrap.demos[bootstrap.current];
            let name = demo_names
                .get(demo)
                .map_or("<unknown>", |demo_name| demo_name.0);
            warn!(
                "Timed out waiting for particle effect pipelines for demo '{name}' after \
                 {BOOTSTRAP_READY_TIMEOUT_FRAMES} frames; continuing bootstrap."
            );
        }
        BootstrapReadiness::Ready | BootstrapReadiness::TimedOut => {}
    }

    match bootstrap.advance() {
        BootstrapProgress::Waiting => {}
        BootstrapProgress::Next(next) => {
            next_demo.0 = Some(next);
            next_state.set(DemoState::Unloading);
        }
        BootstrapProgress::Finished => {
            app_exit.write(AppExit::Success);
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
    mut pending: ResMut<NextDemoSelection>,
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

/// Spawn the UI for the main GUI app's menu, which shows the list of demos.
fn spawn_menu(mut commands: Commands, demos: Query<(Entity, &DemoName, &DemoSummary, &DemoTags)>) {
    let mut demos = demos.iter().collect::<Vec<_>>();
    demos.sort_by_key(|(_, name, _, _)| name.0);

    commands
        .spawn((
            Node {
                width: Val::Px(MENU_WIDTH),
                height: Val::Percent(100.),
                position_type: PositionType::Absolute,
                top: Val::Px(0.),
                left: Val::Px(0.),
                border: UiRect::right(Val::Px(1.)),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Stretch,
                justify_content: JustifyContent::Start,
                padding: UiRect::all(Val::Px(0.)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.055, 0.055, 0.06, 0.95)),
            BorderColor::all(Color::srgb(0.22, 0.22, 0.23)),
            ZIndex(1),
            TabGroup::default(),
            MenuRoot,
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.),
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::Stretch,
                        row_gap: Val::Px(8.),
                        padding: UiRect::all(Val::Px(12.)),
                        margin: UiRect::bottom(Val::Px(10.)),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.14, 0.16, 0.18)),
                    BorderColor::all(Color::srgb(0.2, 0.2, 0.21)),
                ))
                .with_children(|header| {
                    header.spawn((
                        Text::new("Hanabi demos"),
                        utils::ui_font(19.),
                        TextColor(Color::WHITE),
                    ));

                    header
                        .spawn((
                            Node {
                                width: Val::Percent(100.),
                                height: Val::Px(30.),
                                border: UiRect::all(Val::Px(1.)),
                                border_radius: BorderRadius::all(Val::Px(4.)),
                                flex_direction: FlexDirection::Column,
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.19, 0.21, 0.23)),
                            BorderColor::all(Color::srgb(0.25, 0.26, 0.27)),
                        ))
                        .with_children(|search| {
                            search
                                .spawn((
                                    Node {
                                        width: Val::Percent(100.),
                                        flex_grow: 1.,
                                        align_items: AlignItems::Stretch,
                                        ..default()
                                    },
                                    MenuSearchInputContent,
                                ))
                                .with_children(|content| {
                                    content.spawn((
                                        Node {
                                            width: Val::Percent(100.),
                                            padding: UiRect::axes(Val::Px(3.), Val::Px(7.)),
                                            align_self: AlignSelf::Center,
                                            ..default()
                                        },
                                        TextInput,
                                        EditableText::new(""),
                                        TextCursorStyle {
                                            color: Color::srgb(0.9, 0.9, 0.9),
                                            selection_color: Color::oklch(0.48, 0.12, 300.),
                                            unfocused_selection_color: Color::oklch(
                                                0.35, 0.08, 300.,
                                            ),
                                            ..default()
                                        },
                                        TextLayout::no_wrap(),
                                        TabIndex(0),
                                        utils::ui_font(12.),
                                        TextColor(Color::WHITE),
                                        MenuSearchInput,
                                    ));
                                });
                            search
                                .spawn((
                                    Node {
                                        position_type: PositionType::Absolute,
                                        left: Val::Px(7.),
                                        right: Val::Px(3.),
                                        top: Val::Px(0.),
                                        bottom: Val::Px(0.),
                                        overflow: Overflow::clip(),
                                        align_items: AlignItems::Center,
                                        ..default()
                                    },
                                    Pickable::IGNORE,
                                    MenuSearchPlaceholder,
                                ))
                                .with_children(|placeholder| {
                                    placeholder.spawn((
                                        Text::new("Search by demo name..."),
                                        utils::ui_font(12.),
                                        TextColor(Color::srgb(0.7, 0.7, 0.71)),
                                        TextLayout::no_wrap(),
                                        Pickable::IGNORE,
                                    ));
                                });
                            search.spawn((
                                // Focus indicator: we need a half-pill shape, that clips to the
                                // rounded corners of the parent. Bevy clamps the border radius to
                                // the min(half-width, half-height) so we can't set a bottom radius
                                // equal to the total height. So we use a shape twice as tall,
                                // remove its top and side borders, make it transparent, and only
                                // use its bottom border as shape.
                                Node {
                                    position_type: PositionType::Absolute,
                                    left: Val::Px(0.),
                                    right: Val::Px(0.),
                                    bottom: Val::Px(0.),
                                    height: Val::Px(4.),
                                    border: UiRect::bottom(Val::Px(2.)),
                                    border_radius: BorderRadius::new(
                                        Val::Px(0.),
                                        Val::Px(0.),
                                        Val::Px(2.0),
                                        Val::Px(2.0),
                                    ),
                                    ..default()
                                },
                                BackgroundColor(Color::NONE),
                                BorderColor::all(Color::NONE),
                                FocusPolicy::Pass,
                                MenuSearchFocusIndicator,
                            ));
                        });

                    header
                        .spawn(Node {
                            width: Val::Percent(100.),
                            flex_direction: FlexDirection::Row,
                            flex_wrap: FlexWrap::Wrap,
                            align_items: AlignItems::Center,
                            column_gap: Val::Px(4.),
                            row_gap: Val::Px(4.),
                            ..default()
                        })
                        .with_children(|tags| {
                            for tag in DemoTag::ALL {
                                tags.spawn((
                                    Button,
                                    Node {
                                        padding: UiRect {
                                            left: Val::Px(5.),
                                            right: Val::Px(5.),
                                            top: Val::Px(3.),
                                            bottom: Val::Px(3.),
                                        },
                                        border: UiRect::all(Val::Px(1.)),
                                        border_radius: BorderRadius::all(Val::Percent(50.)),
                                        ..default()
                                    },
                                    BackgroundColor(tag.color()),
                                    BorderColor::all(tag.color().lighter(0.24)),
                                    Hovered::default(),
                                    MenuTagButton(*tag),
                                ))
                                .with_children(|pill| {
                                    pill.spawn((
                                        Text::new(tag.label()),
                                        menu_tag_font(11.),
                                        TextColor(Color::WHITE),
                                        MenuTagLabel(*tag),
                                    ));
                                });
                            }
                        });
                });

            parent
                .spawn(Node {
                    flex_grow: 1.,
                    min_height: Val::Px(0.),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Stretch,
                    justify_content: JustifyContent::Start,
                    column_gap: Val::Px(6.),
                    padding: UiRect {
                        left: Val::Px(12.),
                        right: Val::Px(12.),
                        top: Val::Px(0.),
                        bottom: Val::Px(12.),
                    },
                    ..default()
                })
                .with_children(|body| {
                    let scroll_area = body
                        .spawn((
                            Node {
                                width: Val::Percent(100.),
                                flex_grow: 1.,
                                min_width: Val::Px(0.),
                                min_height: Val::Px(0.),
                                flex_direction: FlexDirection::Column,
                                align_items: AlignItems::Stretch,
                                row_gap: Val::Px(5.),
                                overflow: Overflow::scroll_y(),
                                ..default()
                            },
                            TabIndex(1),
                            MenuDemoList,
                            ScrollArea,
                            ScrollPosition::default(),
                        ))
                        .observe(scroll_active_demo_into_view)
                        .with_children(|grid| {
                            grid.spawn((
                                Node {
                                    display: Display::None,
                                    padding: UiRect::all(Val::Px(8.)),
                                    ..default()
                                },
                                Text::new("No demos match your search."),
                                utils::ui_font(11.),
                                TextColor(Color::srgb(0.7, 0.7, 0.71)),
                                MenuEmptyState,
                            ));

                            for (entity, demo, summary, tags) in &demos {
                                grid.spawn((
                                    Button,
                                    Node {
                                        width: Val::Percent(100.),
                                        min_height: Val::Px(58.),
                                        padding: UiRect::all(Val::Px(8.)),
                                        border: UiRect::all(Val::Px(1.)),
                                        border_radius: BorderRadius::all(Val::Px(5.)),
                                        flex_direction: FlexDirection::Column,
                                        align_items: AlignItems::Stretch,
                                        justify_content: JustifyContent::Start,
                                        row_gap: Val::Px(3.),
                                        ..default()
                                    },
                                    BackgroundColor(Color::srgb(0.12, 0.12, 0.13)),
                                    BorderColor::all(Color::srgb(0.2, 0.2, 0.21)),
                                    Hovered::default(),
                                    MenuButton(*entity),
                                    MenuDemoEntry {
                                        name: demo.0,
                                        tags: tags.0,
                                    },
                                ))
                                .with_children(|button| {
                                    button
                                        .spawn(Node {
                                            width: Val::Percent(100.),
                                            height: Val::Px(22.),
                                            flex_direction: FlexDirection::Row,
                                            align_items: AlignItems::Center,
                                            column_gap: Val::Px(5.),
                                            ..default()
                                        })
                                        .with_children(|title| {
                                            title.spawn((
                                                Text::new(demo.0),
                                                utils::ui_font(12.),
                                                TextColor(Color::WHITE),
                                            ));
                                            for tag in tags.0 {
                                                title
                                                    .spawn((
                                                        Node {
                                                            padding: UiRect {
                                                                left: Val::Px(4.),
                                                                right: Val::Px(4.),
                                                                top: Val::Px(2.),
                                                                bottom: Val::Px(2.),
                                                            },
                                                            border_radius: BorderRadius::all(
                                                                Val::Percent(50.),
                                                            ),
                                                            ..default()
                                                        },
                                                        BackgroundColor(tag.color()),
                                                    ))
                                                    .with_children(|chip| {
                                                        chip.spawn((
                                                            Text::new(tag.label()),
                                                            menu_tag_font(11.),
                                                            TextColor(Color::WHITE),
                                                        ));
                                                    });
                                            }
                                        });
                                    button.spawn((
                                        Node {
                                            min_height: Val::Px(16.),
                                            ..default()
                                        },
                                        Text::new(summary.0),
                                        utils::ui_font(11.),
                                        TextColor(Color::srgb(0.7, 0.7, 0.71)),
                                    ));
                                });
                            }
                        })
                        .id();

                    body.spawn((
                        Node {
                            width: Val::Px(6.),
                            height: Val::Percent(100.),
                            border_radius: BorderRadius::all(Val::Percent(50.)),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.13, 0.13, 0.14)),
                        Scrollbar::new(scroll_area, ControlOrientation::Vertical, 24.),
                    ))
                    .with_children(|track| {
                        track.spawn((
                            ScrollbarThumb {
                                border_radius: BorderRadius::all(Val::Percent(50.)),
                                border: UiRect::all(Val::Px(0.)),
                            },
                            BackgroundColor(Color::srgb(0.55, 0.55, 0.57)),
                        ));
                    });
                });
        });
}

/// Get a [`TextFont`] for a given font size, for use with the demo tag pills.
fn menu_tag_font(size: f32) -> TextFont {
    TextFont {
        weight: FontWeight::SEMIBOLD,
        ..utils::ui_font(size)
    }
}

fn search_text_changed(
    on: On<TextEditChange>,
    inputs: Query<&EditableText, With<MenuSearchInput>>,
    mut search: ResMut<MenuSearch>,
    mut placeholders: Query<&mut Visibility, With<MenuSearchPlaceholder>>,
) {
    let Ok(input) = inputs.get(on.event_target()) else {
        return;
    };

    search.name.clear();
    for part in input.value() {
        search.name.push_str(part);
    }

    for mut visibility in &mut placeholders {
        *visibility = if search.name.is_empty() {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

fn update_menu_search_focus_indicator(
    input_focus: Res<InputFocus>,
    inputs: Query<&ChildOf, With<MenuSearchInput>>,
    input_content: Query<&ChildOf, With<MenuSearchInputContent>>,
    mut indicators: Query<(&ChildOf, &mut BorderColor), With<MenuSearchFocusIndicator>>,
) {
    if !input_focus.is_changed() {
        return;
    }

    let focused_parent = input_focus
        .get()
        .and_then(|focused| inputs.get(focused).ok())
        .and_then(|input_parent| input_content.get(input_parent.parent()).ok())
        .map(ChildOf::parent);

    for (parent, mut border_color) in &mut indicators {
        *border_color = BorderColor::all(if Some(parent.parent()) == focused_parent {
            MENU_FOCUS_COLOR
        } else {
            Color::NONE
        });
    }
}

fn toggle_menu_tag(
    activate: On<Activate>,
    buttons: Query<&MenuTagButton>,
    mut search: ResMut<MenuSearch>,
) {
    let Ok(button) = buttons.get(activate.entity) else {
        return;
    };

    if !search.disabled_tags.remove(&button.0) {
        search.disabled_tags.insert(button.0);
    }
}

fn update_menu_filter(
    search: Res<MenuSearch>,
    active_demo: Option<ResMut<ActiveMenuDemo>>,
    mut demos: Query<
        (Entity, &MenuDemoEntry, &mut Node, Option<&MenuButton>),
        Without<MenuEmptyState>,
    >,
    mut empty_state: Query<&mut Node, With<MenuEmptyState>>,
    mut tag_buttons: Query<(&MenuTagButton, &mut BackgroundColor, &mut BorderColor)>,
    mut tag_labels: Query<(&MenuTagLabel, &mut TextColor)>,
) {
    if !search.is_changed() {
        return;
    }

    let name = search.name.to_lowercase();
    let mut visible_count = 0;
    let mut first_visible: Option<(Entity, &'static str)> = None;
    let mut active_is_visible = false;
    for (_, demo, mut node, menu_button) in &mut demos {
        let name_matches = name.is_empty() || demo.name.to_lowercase().contains(&name);
        let tag_matches = demo
            .tags
            .iter()
            .any(|tag| !search.disabled_tags.contains(tag));
        let visible = name_matches && tag_matches;
        node.display = if visible {
            Display::Flex
        } else {
            Display::None
        };
        visible_count += usize::from(visible);
        if visible {
            if let Some(menu_button) = menu_button {
                active_is_visible |= active_demo
                    .as_ref()
                    .is_some_and(|active| active.0 == menu_button.0);
                if first_visible.is_none_or(|(_, first_name)| demo.name < first_name) {
                    first_visible = Some((menu_button.0, demo.name));
                }
            }
        }
    }
    if !active_is_visible {
        if let (Some((entity, _)), Some(mut active_demo)) = (first_visible, active_demo) {
            active_demo.0 = entity;
        }
    }
    if let Ok(mut node) = empty_state.single_mut() {
        node.display = if visible_count == 0 {
            Display::Flex
        } else {
            Display::None
        };
    }

    for (tag, mut background, mut border) in &mut tag_buttons {
        let active = !search.disabled_tags.contains(&tag.0);
        let tag_color = if active {
            tag.0.color()
        } else {
            DISABLED_TAG_COLOR
        };
        background.0 = tag_color;
        border.set_all(if active {
            tag.0.color().darker(0.24)
        } else {
            DISABLED_TAG_COLOR.darker(0.2)
        });
    }

    for (tag, mut text_color) in &mut tag_labels {
        text_color.0 = if search.disabled_tags.contains(&tag.0) {
            DISABLED_TAG_TEXT_COLOR
        } else {
            Color::WHITE
        };
    }
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

/// Clear all demo-specific content when a demo finished, before the next demo
/// is loaded.
///
/// This relies on the [`DemoContent`] to be correctly applied to all per-demo
/// entities.
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

/// Highlight menu buttons when hovered.
fn highlight_menu_buttons(
    selection: Res<DemoSelection>,
    active_demo: Res<ActiveMenuDemo>,
    input_focus: Res<InputFocus>,
    lists: Query<Entity, With<MenuDemoList>>,
    mut buttons: Query<(
        &MenuButton,
        &Hovered,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    let list_focused = lists.iter().any(|entity| input_focus.get() == Some(entity));
    for (button, hovered, mut background, mut border) in &mut buttons {
        let (background_color, border_color) = if button.0 == selection.0 {
            (Color::srgb(0.19, 0.19, 0.2), Color::srgb(0.44, 0.44, 0.46))
        } else if hovered.get() {
            (Color::srgb(0.15, 0.15, 0.16), Color::srgb(0.3, 0.3, 0.31))
        } else {
            (Color::srgb(0.12, 0.12, 0.13), Color::srgb(0.2, 0.2, 0.21))
        };
        background.0 = background_color;
        border.set_all(if list_focused && button.0 == active_demo.0 {
            MENU_FOCUS_COLOR
        } else {
            border_color
        });
    }
}

/// Change the current demo when a menu list button is pressed, from any input
/// source. This transitions to [`DemoState::Unloading`] and saves the next demo
/// to load in [`NextDemoSelection`].
fn choose_demo(
    activate: On<Activate>,
    buttons: Query<&MenuButton>,
    selection: Res<DemoSelection>,
    mut active_demo: ResMut<ActiveMenuDemo>,
    mut next_demo: ResMut<NextDemoSelection>,
    mut next_state: ResMut<NextState<DemoState>>,
) {
    if let Ok(button) = buttons.get(activate.entity) {
        active_demo.0 = button.0;
        if button.0 != selection.0 {
            next_demo.0 = Some(button.0);
            next_state.set(DemoState::Unloading);
        }
    }
}

/// Handle navigation of the demo list from keyboard.
fn navigate_demo_list(
    mut event: On<FocusedInput<KeyboardInput>>,
    mut active_demo: ResMut<ActiveMenuDemo>,
    mut focus_visible: ResMut<InputFocusVisible>,
    mut commands: Commands,
    lists: Query<&ComputedNode, With<MenuDemoList>>,
    buttons: Query<(Entity, &MenuButton, &MenuDemoEntry, &Node, &ComputedNode), With<MenuButton>>,
) {
    if event.input.state != ButtonState::Pressed {
        return;
    }
    let Ok(list_node) = lists.get(event.focused_entity) else {
        return;
    };

    let mut visible_buttons = buttons
        .iter()
        .filter(|(_, _, _, node, _)| node.display != Display::None)
        .map(|(entity, button, demo, _, computed)| (entity, button.0, demo.name, computed.size().y))
        .collect::<Vec<_>>();
    if visible_buttons.is_empty() {
        return;
    }
    visible_buttons.sort_by_key(|(_, _, name, _)| *name);

    let current_index = visible_buttons
        .iter()
        .position(|(_, demo, _, _)| *demo == active_demo.0)
        .unwrap_or(0);
    let last_index = visible_buttons.len() - 1;
    let page_step = if visible_buttons[current_index].3 > 0. {
        ((list_node.size().y / visible_buttons[current_index].3).floor() as usize)
            .saturating_sub(1)
            .max(1)
    } else {
        1
    };

    let next_index = match event.input.key_code {
        KeyCode::ArrowUp => current_index.saturating_sub(1),
        KeyCode::ArrowDown => current_index.saturating_add(1).min(last_index),
        KeyCode::PageUp => current_index.saturating_sub(page_step),
        KeyCode::PageDown => current_index.saturating_add(page_step).min(last_index),
        KeyCode::Home => 0,
        KeyCode::End => last_index,
        KeyCode::Enter | KeyCode::Space => {
            event.propagate(false);
            if let Some((row, _, _, _)) = visible_buttons
                .iter()
                .find(|(_, demo, _, _)| *demo == active_demo.0)
            {
                commands.trigger(Activate { entity: *row });
            }
            return;
        }
        _ => return,
    };

    event.propagate(false);
    let (row, demo, _, _) = visible_buttons[next_index];
    active_demo.0 = demo;
    commands.trigger(ScrollIntoView { entity: row });
    focus_visible.0 = true;
}

/// Ensure the demo list scrolls to the active demo when it gains focus.
fn scroll_active_demo_into_view(
    event: On<FocusGained>,
    active_demo: Res<ActiveMenuDemo>,
    q_list: Query<(), With<MenuDemoList>>,
    q_buttons: Query<(Entity, &MenuButton)>,
    mut commands: Commands,
) {
    if !q_list.contains(event.entity) {
        return;
    }
    if let Some((row, _)) = q_buttons
        .iter()
        .find(|(_, button)| button.0 == active_demo.0)
    {
        commands.trigger(ScrollIntoView { entity: row });
    }
}

/// Ensure the demo list scrolls to the active demo on start. This is especially
/// important when using `--demo <id>`, which could select a demo down the list.
#[allow(clippy::type_complexity)]
fn scroll_initial_active_demo_into_view(
    mut scrolled: Local<bool>,
    active_demo: Res<ActiveMenuDemo>,
    buttons: Query<(Entity, &MenuButton), (With<ComputedNode>, With<UiGlobalTransform>)>,
    mut commands: Commands,
) {
    if *scrolled {
        return;
    }
    if let Some((row, _)) = buttons.iter().find(|(_, button)| button.0 == active_demo.0) {
        commands.trigger(ScrollIntoView { entity: row });
        *scrolled = true;
    }
}

/// Validate or regenerate all effect files from all demos.
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
                    "{} differs from generated .effect; check the difference, and if expected run --write-effects to regenerate it.",
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
                            "unexpected effect artifact {}; remove it, or register it with the demo.",
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
        .init_resource::<NextDemoSelection>()
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

    #[test]
    fn bootstrap() {
        let mut world = World::new();
        let first = world.spawn_empty().id();
        let second = world.spawn_empty().id();
        let mut bootstrap = Bootstrap {
            demos: vec![first, second],
            current: 0,
            frames: 0,
            frames_per_demo: 3,
            waited_frames: 0,
            timed_out: false,
        };

        assert!(matches!(
            bootstrap.wait_for_effects(false),
            BootstrapReadiness::Waiting
        ));
        assert!(matches!(
            bootstrap.wait_for_effects(true),
            BootstrapReadiness::Ready
        ));
        assert!(matches!(bootstrap.advance(), BootstrapProgress::Waiting));
        assert!(matches!(bootstrap.advance(), BootstrapProgress::Waiting));
        assert!(matches!(
            bootstrap.advance(),
            BootstrapProgress::Next(entity) if entity == second
        ));
        assert!(matches!(bootstrap.advance(), BootstrapProgress::Waiting));
        assert!(matches!(bootstrap.advance(), BootstrapProgress::Waiting));
        assert!(matches!(bootstrap.advance(), BootstrapProgress::Finished));
    }

    #[test]
    fn bootstrap_timeout() {
        let mut bootstrap = Bootstrap {
            demos: vec![],
            current: 0,
            frames: 0,
            frames_per_demo: BOOTSTRAP_FRAMES,
            waited_frames: 0,
            timed_out: false,
        };

        for _ in 1..BOOTSTRAP_READY_TIMEOUT_FRAMES {
            assert!(matches!(
                bootstrap.wait_for_effects(false),
                BootstrapReadiness::Waiting
            ));
        }
        assert!(matches!(
            bootstrap.wait_for_effects(false),
            BootstrapReadiness::TimedOut
        ));
        assert!(matches!(
            bootstrap.wait_for_effects(false),
            BootstrapReadiness::TimedOut
        ));
    }

    #[test]
    fn bootstrap_parse_frames() {
        assert_eq!(parse_bootstrap_frames(None).unwrap(), BOOTSTRAP_FRAMES);
        assert_eq!(parse_bootstrap_frames(Some("20")).unwrap(), 20);
        assert!(parse_bootstrap_frames(Some("0")).is_err());
        assert!(parse_bootstrap_frames(Some("invalid")).is_err());
    }

    fn spawn_demo_content(mut commands: DemoCommands) {
        commands
            .spawn(Name::new("demo content"))
            .with_children(|parent| {
                parent.spawn(Name::new("demo child"));
            });
    }

    /// Validate the content of all registered demos.
    #[test]
    fn validate_demos() {
        let mut world = World::new();
        insert_demos(&mut world);
        let mut query =
            world.query::<(&Demo, &DemoName, &DemoDescription, &DemoSummary, &DemoTags)>();
        for (demo, name, description, summary, tags) in query.iter(&world) {
            assert!(!name.0.trim().is_empty(), "demo has no name");
            assert!(
                !description.0.trim().is_empty(),
                "demo '{}' has no description",
                name.0
            );
            assert!(
                !summary.0.trim().is_empty(),
                "demo '{}' has no menu summary",
                name.0
            );
            assert!(!tags.0.is_empty(), "demo '{}' has no tags", name.0);
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

    /// Validate that all registered demos preload their effects and store
    /// handles to them.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn validate_demo_effects() {
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
    fn menu_search() {
        let mut app = App::new();
        app.init_resource::<MenuSearch>()
            .add_systems(Update, update_menu_filter);
        app.world_mut().spawn((
            Node::default(),
            MenuDemoEntry {
                name: "Firework",
                tags: &[DemoTag::Expr, DemoTag::Hdr],
            },
        ));
        app.world_mut().spawn((
            Node::default(),
            MenuDemoEntry {
                name: "Puffs",
                tags: &[DemoTag::Mesh],
            },
        ));
        app.world_mut().spawn((
            Node::default(),
            MenuDemoEntry {
                name: "Expression-only",
                tags: &[DemoTag::Expr],
            },
        ));
        for tag in DemoTag::ALL {
            app.world_mut().spawn((
                MenuTagButton(*tag),
                BackgroundColor(tag.color()),
                BorderColor::all(tag.color().darker(0.24)),
            ));
            app.world_mut()
                .spawn((MenuTagLabel(*tag), TextColor(Color::WHITE)));
        }

        let visible_names = |app: &mut App| {
            let mut entries = app.world_mut().query::<(&MenuDemoEntry, &Node)>();
            entries
                .iter(app.world())
                .filter(|(_, node)| node.display != Display::None)
                .map(|(entry, _)| entry.name)
                .collect::<HashSet<_>>()
        };
        let highlighted_tags = |app: &mut App| {
            let mut tags = app
                .world_mut()
                .query::<(&MenuTagButton, &BackgroundColor)>();
            tags.iter(app.world())
                .filter(|(tag, background)| background.0 == tag.0.color())
                .map(|(tag, _)| tag.0)
                .collect::<HashSet<_>>()
        };

        app.update();
        assert_eq!(
            visible_names(&mut app),
            HashSet::from(["Firework", "Puffs", "Expression-only"])
        );
        assert_eq!(
            highlighted_tags(&mut app),
            DemoTag::ALL.iter().copied().collect()
        );

        app.world_mut().resource_mut::<MenuSearch>().name = "FIRE".to_string();
        app.update();
        assert_eq!(visible_names(&mut app), HashSet::from(["Firework"]));

        {
            let mut search = app.world_mut().resource_mut::<MenuSearch>();
            search.name.clear();
            search.disabled_tags.insert(DemoTag::Expr);
        }
        app.update();
        assert_eq!(
            visible_names(&mut app),
            HashSet::from(["Firework", "Puffs"])
        );
        assert!(!highlighted_tags(&mut app).contains(&DemoTag::Expr));
        let disabled_background = {
            let mut tags = app
                .world_mut()
                .query::<(&MenuTagButton, &BackgroundColor)>();
            tags.iter(app.world())
                .find(|(tag, _)| tag.0 == DemoTag::Expr)
                .map(|(_, background)| background.0)
        };
        assert_eq!(disabled_background, Some(DISABLED_TAG_COLOR));
        let label_colors = {
            let mut labels = app.world_mut().query::<(&MenuTagLabel, &TextColor)>();
            labels
                .iter(app.world())
                .map(|(tag, color)| (tag.0, color.0))
                .collect::<HashMap<_, _>>()
        };
        assert_eq!(
            label_colors.get(&DemoTag::Expr),
            Some(&DISABLED_TAG_TEXT_COLOR)
        );
        assert_eq!(label_colors.get(&DemoTag::Hdr), Some(&Color::WHITE));

        app.world_mut()
            .resource_mut::<MenuSearch>()
            .disabled_tags
            .insert(DemoTag::Hdr);
        app.update();
        assert_eq!(visible_names(&mut app), HashSet::from(["Puffs"]));
    }

    /// Ensure demo tag colors have sufficient contrast to be readable. This was
    /// a problem during the demo development, so keeping this to prevent
    /// regression.
    #[test]
    fn demo_tag_contrast() {
        for background in DemoTag::ALL
            .iter()
            .map(|tag| tag.color())
            .chain([DISABLED_TAG_COLOR])
        {
            let background_luminance = background.to_linear().luminance();
            let text_luminance = Color::WHITE.to_linear().luminance();
            let (lighter, darker) = if background_luminance > text_luminance {
                (background_luminance, text_luminance)
            } else {
                (text_luminance, background_luminance)
            };
            let contrast_ratio = (lighter + 0.05) / (darker + 0.05);
            assert!(
                contrast_ratio >= 4.5,
                "text contrast ratio {contrast_ratio} is too low"
            );
        }
    }

    /// Check demo transition sequencing (unload old demo, then load the new
    /// one).
    #[test]
    fn demo_transition() {
        let mut app = App::new();
        app.add_plugins(bevy::state::app::StatesPlugin)
            .insert_resource(DemoSelection(Entity::PLACEHOLDER))
            .init_resource::<NextDemoSelection>()
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
}
