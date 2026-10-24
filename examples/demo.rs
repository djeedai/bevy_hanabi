use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::ops::{Deref, DerefMut};
#[cfg(not(target_arch = "wasm32"))]
use std::{
    fs,
    path::{Path, PathBuf},
};

use bevy::color::Luminance;
use bevy::input_focus::tab_navigation::{TabGroup, TabIndex, TabNavigationPlugin};
use bevy::input_focus::InputFocus;
use bevy::text::{EditableText, FontSource, FontWeight, TextCursorStyle, TextEditChange};
use bevy::ui::FocusPolicy;
use bevy::ui_widgets::{
    Activate, Button, ControlOrientation, ScrollArea, Scrollbar, ScrollbarThumb, TextInput,
};
use bevy::{camera::visibility::RenderLayers, prelude::*};
use bevy::{ecs::system::SystemParam, picking::hover::Hovered};
use bevy_hanabi::prelude::*;

mod utils;
use utils::AppExitIntoResult;

const MENU_WIDTH: f32 = 320.;
const MENU_GAP: f32 = 16.;
const DISABLED_TAG_COLOR: Color = Color::oklch(0.38, 0.01, 258.);
const DISABLED_TAG_TEXT_COLOR: Color = Color::srgb(0.62, 0.62, 0.64);

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
        .with_desc_position(utils::DescPosition::RightColumn {
            left: Val::Px(MENU_WIDTH + MENU_GAP),
            top: Val::Px(MENU_GAP),
        })
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
        .init_resource::<MenuSearch>()
        .init_state::<DemoState>();

    install_demo_state_systems(&mut app);
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
        .add_observer(search_text_changed)
        .add_observer(toggle_menu_tag)
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
                        menu_font(19.),
                        TextColor(Color::WHITE),
                    ));

                    header.spawn((
                        Text::new("Search by name"),
                        menu_font(11.),
                        TextColor(Color::srgb(0.7, 0.7, 0.71)),
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
                                            ..default()
                                        },
                                        TextLayout::no_wrap(),
                                        TabIndex(0),
                                        menu_font(12.),
                                        TextColor(Color::WHITE),
                                        MenuSearchInput,
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
                            ScrollArea,
                            ScrollPosition::default(),
                        ))
                        .with_children(|grid| {
                            grid.spawn((
                                Node {
                                    display: Display::None,
                                    padding: UiRect::all(Val::Px(8.)),
                                    ..default()
                                },
                                Text::new("No demos match your search."),
                                menu_font(11.),
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
                                                menu_font(12.),
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
                                        menu_font(11.),
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

fn menu_font(size: f32) -> TextFont {
    TextFont {
        font: FontSource::list([FontSource::system_ui(), FontSource::default()]),
        ..TextFont::from_font_size(size)
    }
}

fn menu_tag_font(size: f32) -> TextFont {
    TextFont {
        weight: FontWeight::SEMIBOLD,
        ..menu_font(size)
    }
}

fn search_text_changed(
    on: On<TextEditChange>,
    inputs: Query<&EditableText, With<MenuSearchInput>>,
    mut search: ResMut<MenuSearch>,
) {
    let Ok(input) = inputs.get(on.event_target()) else {
        return;
    };

    search.name.clear();
    for part in input.value() {
        search.name.push_str(part);
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
            Color::oklab(0.67, 0.08, -0.12)
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
    mut demos: Query<(&MenuDemoEntry, &mut Node), Without<MenuEmptyState>>,
    mut empty_state: Query<&mut Node, With<MenuEmptyState>>,
    mut tag_buttons: Query<(&MenuTagButton, &mut BackgroundColor, &mut BorderColor)>,
    mut tag_labels: Query<(&MenuTagLabel, &mut TextColor)>,
) {
    if !search.is_changed() {
        return;
    }

    let name = search.name.to_lowercase();
    let mut visible_count = 0;
    for (demo, mut node) in &mut demos {
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
    selection: Res<DemoSelection>,
    mut buttons: Query<(
        &MenuButton,
        &Hovered,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    for (button, hovered, mut background, mut border) in &mut buttons {
        let (background_color, border_color) = if button.0 == selection.0 {
            (Color::srgb(0.19, 0.19, 0.2), Color::srgb(0.44, 0.44, 0.46))
        } else if hovered.get() {
            (Color::srgb(0.15, 0.15, 0.16), Color::srgb(0.3, 0.3, 0.31))
        } else {
            (Color::srgb(0.12, 0.12, 0.13), Color::srgb(0.2, 0.2, 0.21))
        };
        background.0 = background_color;
        border.set_all(border_color);
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
    fn menu_search_combines_case_insensitive_name_and_tag_filters() {
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

    #[test]
    fn demo_tag_palette_uses_uniform_oklch_lightness() {
        let mut hues = Vec::new();
        for tag in DemoTag::ALL {
            let Color::Oklcha(color) = tag.color() else {
                panic!("demo tag colors should use Oklch");
            };
            assert_eq!(color.lightness, 0.48);
            assert_eq!(color.chroma, 0.14);
            hues.push(color.hue);
        }
        hues.sort_by(f32::total_cmp);
        hues.dedup();
        assert_eq!(hues.len(), DemoTag::ALL.len());
    }

    #[test]
    fn demo_tag_text_colors_have_sufficient_contrast() {
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
