//! Node editor

use bevy::{
    core_pipeline::{bloom::BloomSettings, clear_color::ClearColorConfig},
    diagnostic::FrameTimeDiagnosticsPlugin,
    log::LogPlugin,
    prelude::*,
    ui::FocusPolicy,
    window::{PresentMode, PrimaryWindow},
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

use bevy_hanabi::{
    node::{Node, SlotDef},
    prelude::*,
};

const MENU_BACKGROUND: Color = Color::rgb(0.1, 0.1, 0.1);
const MENU_BORDERS: Color = Color::rgb(0.15, 0.15, 0.15);

const NODE_BACKGROUND: Color = Color::rgb(0.1, 0.1, 0.1);
const NODE_BORDERS: Color = Color::rgb(0.15, 0.15, 0.15);
const NODE_BACKGROUND_HOVERED: Color = Color::rgb(0.2, 0.2, 0.2);

const NORMAL_BUTTON: Color = Color::NONE;
const HOVERED_BUTTON: Color = Color::rgb(0.25, 0.25, 0.25);
const PRESSED_BUTTON: Color = Color::rgb(0.35, 0.75, 0.35);

const SLOT_BACKGROUND: Color = Color::rgb(0.7, 0.2, 0.2);
const SLOT_BACKGROUND_HOVERED: Color = Color::rgb(0.9, 0.4, 0.4);

trait SlotEx {
    fn background_color(&self) -> Color;
    fn hover_color(&self) -> Color;
}

impl SlotEx for SlotDef {
    fn background_color(&self) -> Color {
        if let Some(value_type) = &self.value_type() {
            match *value_type {
                ValueType::Scalar(_) => Color::rgb(0.7, 0.2, 0.2),
                ValueType::Vector(_) => Color::rgb(0.7, 0.7, 0.2),
                ValueType::Matrix(_) => Color::rgb(0.7, 0.2, 0.7),
                _ => Color::rgb(0.7, 0.7, 0.7),
            }
        } else {
            Color::rgb(0.7, 0.7, 0.7)
        }
    }

    fn hover_color(&self) -> Color {
        if let Some(value_type) = &self.value_type() {
            match *value_type {
                ValueType::Scalar(_) => Color::rgb(0.9, 0.4, 0.4),
                ValueType::Vector(_) => Color::rgb(0.9, 0.9, 0.4),
                ValueType::Matrix(_) => Color::rgb(0.9, 0.4, 0.9),
                _ => Color::rgb(0.9, 0.9, 0.9),
            }
        } else {
            Color::rgb(0.9, 0.9, 0.9)
        }
    }
}

struct NodeEntry {
    pub node_name: String,
    pub create: Box<dyn Fn() -> Box<dyn Node> + Send + Sync + 'static>,
}

impl NodeEntry {
    pub fn new<N: Node + Default + 'static>(name: impl Into<String>) -> Self {
        Self {
            node_name: name.into(),
            create: Box::new(|| Box::new(N::default())),
        }
    }
}

#[derive(Resource)]
struct NodeRegistry {
    nodes: Vec<NodeEntry>,
}

impl Default for NodeRegistry {
    fn default() -> Self {
        let nodes = vec![
            NodeEntry::new::<AddNode>("Add"),
            NodeEntry::new::<SubNode>("Subtract"),
            NodeEntry::new::<MulNode>("Multiply"),
            NodeEntry::new::<DivNode>("Divide"),
            //NodeEntry::new::<PropertyNode>("Property"), // TODO
            NodeEntry::new::<AttributeNode>("Attribute"),
        ];
        Self { nodes }
    }
}

impl NodeRegistry {
    pub fn create(&self, node_name: &str) -> Option<Box<dyn Node>> {
        if let Some(entry) = self
            .nodes
            .iter()
            .find(|&entry| entry.node_name == node_name)
        {
            Some((entry.create)())
        } else {
            None
        }
    }

    pub fn node_names(&self) -> impl Iterator<Item = &str> {
        self.nodes.iter().map(|entry| &entry.node_name[..])
    }
}

#[derive(Debug, Default, Component)]
struct CreateNodeMenu;

impl CreateNodeMenu {
    pub fn spawn(
        commands: &mut Commands,
        asset_server: Res<AssetServer>,
        node_registry: &NodeRegistry,
    ) {
        let text_style = TextStyle {
            font: asset_server.load("fonts/FiraSans-Regular.ttf"),
            font_size: 14.,
            color: Color::rgb(0.9, 0.9, 0.9),
        };

        commands
            .spawn((
                NodeBundle {
                    style: Style {
                        position_type: PositionType::Absolute,
                        position: UiRect {
                            left: Val::Px(0.),
                            top: Val::Px(0.),
                            ..default()
                        },
                        size: Size::AUTO,
                        padding: UiRect::all(Val::Px(2.)),
                        flex_direction: FlexDirection::Column,
                        ..default()
                    },
                    background_color: MENU_BORDERS.into(),
                    visibility: Visibility::Hidden,
                    z_index: ZIndex::Global(10000), // above everything
                    ..default()
                },
                CreateNodeMenu::default(),
                Name::new("CreateNodeMenu"),
                Interaction::default(),
            ))
            .with_children(|p| {
                p.spawn(NodeBundle {
                    style: Style {
                        position_type: PositionType::Relative,
                        position: UiRect::all(Val::Px(0.)),
                        size: Size::all(Val::Percent(100.)),
                        min_size: Size {
                            width: Val::Px(120.),
                            height: Val::Auto,
                        },
                        padding: UiRect::all(Val::Px(3.)),
                        flex_direction: FlexDirection::Column,
                        ..default()
                    },
                    background_color: MENU_BACKGROUND.into(),
                    ..default()
                })
                .with_children(|p| {
                    for name in node_registry.node_names() {
                        p.spawn((
                            NodeBundle {
                                style: Style {
                                    position_type: PositionType::Relative,
                                    size: Size {
                                        width: Val::Auto,
                                        height: Val::Auto,
                                    },
                                    padding: UiRect {
                                        left: Val::Px(8.),
                                        right: Val::Px(8.),
                                        top: Val::Px(4.),
                                        bottom: Val::Px(4.),
                                    },
                                    ..default()
                                },
                                background_color: NORMAL_BUTTON.into(),
                                focus_policy: FocusPolicy::Block,
                                ..default()
                            },
                            Interaction::default(),
                            CreateNodeButton {
                                node_name: name.to_string(),
                            },
                        ))
                        .with_children(|p| {
                            p.spawn(TextBundle {
                                text: Text::from_section(name, text_style.clone()),
                                ..default()
                            });
                        });
                    }
                });
            });
    }
}

//#[derive(Event)]
struct CreateNodeEvent {
    pub node_name: String,
    pub initial_pos: Vec2,
}

#[derive(Component)]
struct CreateNodeButton {
    pub node_name: String,
}

#[derive(Component)]
struct NodeWidget;

#[derive(Debug, Default, Component)]
struct NodeWidgetTitle {
    drag_pos: Option<Vec2>,
}

#[derive(Debug, Component)]
struct SlotUI {
    def: SlotDef,
}

impl SlotUI {
    pub fn def(&self) -> &SlotDef {
        &self.def
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    App::default()
        .add_plugins(
            DefaultPlugins
                .set(LogPlugin {
                    level: bevy::log::Level::WARN,
                    filter: "bevy_hanabi=warn,node_editor=trace".to_string(),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "🎆 Hanabi - Editor".into(),
                        present_mode: PresentMode::AutoVsync,
                        fit_canvas_to_parent: true,
                        prevent_default_event_handling: false,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .init_resource::<NodeRegistry>()
        .add_event::<CreateNodeEvent>()
        .add_system(bevy::window::close_on_esc)
        .add_plugin(FrameTimeDiagnosticsPlugin)
        .add_plugin(HanabiPlugin)
        .add_plugin(WorldInspectorPlugin::default())
        .add_startup_system(setup)
        .add_system(input)
        .add_system(create_node_menu_entry)
        .add_system(spawn_node_widget)
        .add_system(node_interaction)
        .add_system(slot_interaction)
        .run();

    Ok(())
}

fn setup(
    asset_server: Res<AssetServer>,
    node_registry: Res<NodeRegistry>,
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0., 0., 50.)),
            camera: Camera {
                hdr: true,
                ..default()
            },
            camera_3d: Camera3d {
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            ..default()
        },
        BloomSettings::default(),
    ));

    let mut color_gradient1 = Gradient::new();
    color_gradient1.add_key(0.0, Vec4::new(4.0, 4.0, 4.0, 1.0));
    color_gradient1.add_key(0.1, Vec4::new(4.0, 4.0, 0.0, 1.0));
    color_gradient1.add_key(0.9, Vec4::new(4.0, 0.0, 0.0, 1.0));
    color_gradient1.add_key(1.0, Vec4::new(4.0, 0.0, 0.0, 0.0));

    let mut size_gradient1 = Gradient::new();
    size_gradient1.add_key(0.0, Vec2::splat(0.1));
    size_gradient1.add_key(0.3, Vec2::splat(0.1));
    size_gradient1.add_key(1.0, Vec2::splat(0.0));

    let writer = ExprWriter::new();

    // Give a bit of variation by randomizing the age per particle. This will
    // control the starting color and starting size of particles.
    let age = writer.lit(0.).uniform(writer.lit(0.2)).expr();
    let init_age = InitAttributeModifier::new(Attribute::AGE, age);

    // Give a bit of variation by randomizing the lifetime per particle
    let lifetime = writer.lit(0.8).uniform(writer.lit(1.2)).expr();
    let init_lifetime = InitAttributeModifier::new(Attribute::LIFETIME, lifetime);

    // Add constant downward acceleration to simulate gravity
    let accel = writer.lit(Vec3::Y * -8.).expr();
    let update_accel = AccelModifier::new(accel);

    // Add drag to make particles slow down a bit after the initial explosion
    let drag = writer.lit(5.).expr();
    let update_drag = LinearDragModifier::new(drag);

    let effect = EffectAsset {
        name: "firework".to_string(),
        capacity: 32768,
        spawner: Spawner::burst(2500.0.into(), 2.0.into()),
        module: writer.finish(),
        ..Default::default()
    }
    .init(InitPositionSphereModifier {
        center: Vec3::ZERO,
        radius: 2.,
        dimension: ShapeDimension::Volume,
    })
    .init(InitVelocitySphereModifier {
        center: Vec3::ZERO,
        // Give a bit of variation by randomizing the initial speed
        speed: Value::Uniform((65., 75.)),
    })
    .init(init_age)
    .init(init_lifetime)
    .update(update_drag)
    .update(update_accel)
    .render(ColorOverLifetimeModifier {
        gradient: color_gradient1,
    })
    .render(SizeOverLifetimeModifier {
        gradient: size_gradient1,
    });

    let effect1 = effects.add(effect);

    commands.spawn((
        Name::new("firework"),
        ParticleEffectBundle {
            effect: ParticleEffect::new(effect1),
            transform: Transform::IDENTITY,
            ..Default::default()
        },
    ));

    CreateNodeMenu::spawn(&mut commands, asset_server, &node_registry);
}

fn get_cursor_pos(window: &Window) -> Vec2 {
    let mut cursor_pos = window.cursor_position().unwrap_or(Vec2::ZERO);
    cursor_pos.y = window.resolution.height() - cursor_pos.y;
    cursor_pos
}

fn input(
    keyboard_input: Res<Input<KeyCode>>,
    mouse_button_input: Res<Input<MouseButton>>,
    mut windows: Query<&mut Window>,
    mut q_create_node_menu: Query<
        (&mut Visibility, &mut Style, &Interaction),
        With<CreateNodeMenu>,
    >,
) {
    let (mut visibility, mut style, interaction) = q_create_node_menu.single_mut();

    if keyboard_input.just_pressed(KeyCode::Space)
        || mouse_button_input.just_pressed(MouseButton::Right)
    {
        let window = windows.single_mut();
        let cursor_pos = get_cursor_pos(&*window);
        style.position.left = Val::Px(cursor_pos.x);
        style.position.top = Val::Px(cursor_pos.y);
        *visibility = Visibility::Visible;
    } else if mouse_button_input.just_pressed(MouseButton::Left)
        && (*visibility == Visibility::Visible)
        && (*interaction != Interaction::Hovered)
    {
        *visibility = Visibility::Hidden;
    }
}

fn create_node_menu_entry(
    mut q_create_node_menu: Query<&mut Visibility, With<CreateNodeMenu>>,
    mut q_entry: Query<
        (&CreateNodeButton, &Interaction, &mut BackgroundColor),
        Changed<Interaction>,
    >,
    q_window: Query<&Window, With<PrimaryWindow>>,
    mut ev_create_node: EventWriter<CreateNodeEvent>,
) {
    for (button, interaction, mut background_color) in &mut q_entry {
        match *interaction {
            Interaction::Clicked => {
                // Highlight button
                *background_color = PRESSED_BUTTON.into();

                // Hide menu
                *q_create_node_menu.single_mut() = Visibility::Hidden;

                // Get cursor position
                let primary_window = q_window.single();
                let cursor_pos = get_cursor_pos(primary_window);

                // Send event to create actual node
                ev_create_node.send(CreateNodeEvent {
                    node_name: button.node_name.clone(),
                    initial_pos: cursor_pos,
                });
            }
            Interaction::Hovered => {
                *background_color = HOVERED_BUTTON.into();
            }
            Interaction::None => {
                *background_color = NORMAL_BUTTON.into();
            }
        }
    }
}

fn calc_ui_node_position(
    node: &bevy::ui::Node,
    global_transform: &GlobalTransform,
    calculated_clip: Option<&CalculatedClip>,
) -> Vec2 {
    let position = global_transform.translation();
    let ui_position = position.truncate();
    let extents = node.size() / 2.0;
    let mut min = ui_position - extents;
    if let Some(clip) = calculated_clip {
        min = Vec2::max(min, clip.clip.min);
    }
    min
}

fn node_interaction(
    mut q_node: Query<(
        &bevy::ui::Node,
        &Style,
        &Interaction,
        &mut NodeWidgetTitle,
        &mut BackgroundColor,
        &Parent,
        &GlobalTransform,
        Option<&CalculatedClip>,
    )>,
    mut q_widget: Query<
        (&mut BackgroundColor, &mut Style),
        (With<NodeWidget>, Without<NodeWidgetTitle>),
    >,
    q_window: Query<&Window, With<PrimaryWindow>>,
) {
    let primary_window = q_window.single();

    for (
        ui_node,
        style,
        interaction,
        mut widget_title,
        mut background_color,
        parent,
        global_transform,
        calculated_clip,
    ) in &mut q_node
    {
        let mut delta = None;
        match *interaction {
            Interaction::Clicked => {
                let node_pos = calc_ui_node_position(ui_node, global_transform, calculated_clip);
                let cursor_pos = get_cursor_pos(primary_window);
                if widget_title.drag_pos.is_none() {
                    // Start dragging
                    widget_title.drag_pos = Some(cursor_pos - node_pos);
                } else {
                    // Calculate delta since drag started
                    delta = Some(cursor_pos - node_pos - widget_title.drag_pos.unwrap());
                }
            }
            Interaction::Hovered => {
                *background_color = NODE_BACKGROUND_HOVERED.into();
                if widget_title.drag_pos.is_some() {
                    // End drag
                    widget_title.drag_pos = None;
                }
            }
            Interaction::None => {
                *background_color = NODE_BORDERS.into();
                if widget_title.drag_pos.is_some() {
                    // End drag
                    widget_title.drag_pos = None;
                }
            }
        }

        if let Ok((mut frame_background_color, mut style)) = q_widget.get_mut(parent.get()) {
            // Update widget borders (frame) to the same color as its title bar
            *frame_background_color = *background_color;

            // Apply delta between previous and current mouse cursor
            if let Some(delta) = delta {
                style
                    .position
                    .left
                    .try_add_assign(Val::Px(delta.x))
                    .unwrap();
                style.position.top.try_add_assign(Val::Px(delta.y)).unwrap();
            }
        }
    }
}

const NODE_TITLE_BAR_HEIGHT: f32 = 30.;

fn spawn_node_widget(
    node_registry: Res<NodeRegistry>,
    mut commands: Commands,
    mut ev_create_node: EventReader<CreateNodeEvent>,
    asset_server: Res<AssetServer>, // FIXME - load font once and for all
) {
    for ev in ev_create_node.iter() {
        if let Some(node) = node_registry.create(&ev.node_name) {
            let text_style = TextStyle {
                font: asset_server.load("fonts/FiraSans-Regular.ttf"),
                font_size: 14.,
                color: Color::rgb(0.9, 0.9, 0.9),
            };

            // Compute node height based on number of input and output slots
            let mut hi = 0_f32;
            let mut ho = 0_f32;
            for slot_def in node.slots() {
                if slot_def.is_input() {
                    hi += 20.;
                } else {
                    ho += 20.;
                }
            }
            let height = hi.max(ho) + NODE_TITLE_BAR_HEIGHT + 10.;

            // Border
            commands
                .spawn((
                    NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            position: UiRect {
                                left: Val::Px(ev.initial_pos.x),
                                top: Val::Px(ev.initial_pos.y),
                                ..default()
                            },
                            size: Size {
                                width: Val::Px(100.),
                                height: Val::Px(height),
                            },
                            padding: UiRect::all(Val::Px(2.)),
                            flex_direction: FlexDirection::Column,
                            ..default()
                        },
                        background_color: NODE_BORDERS.into(),
                        ..default()
                    },
                    Name::new(format!("node:{}", ev.node_name)),
                    NodeWidget,
                ))
                .with_children(|p| {
                    // Title bar
                    p.spawn((
                        NodeBundle {
                            style: Style {
                                position_type: PositionType::Relative,
                                size: Size {
                                    width: Val::Auto,
                                    height: Val::Px(NODE_TITLE_BAR_HEIGHT),
                                },
                                padding: UiRect::all(Val::Px(8.)),
                                ..default()
                            },
                            background_color: Color::NONE.into(),
                            ..default()
                        },
                        Interaction::default(),
                        NodeWidgetTitle::default(),
                    ))
                    .with_children(|p| {
                        p.spawn(TextBundle {
                            text: Text::from_section(ev.node_name.clone(), text_style.clone()),
                            ..default()
                        });
                    });

                    // Content
                    p.spawn(NodeBundle {
                        style: Style {
                            position_type: PositionType::Relative,
                            position: UiRect::all(Val::Px(0.)),
                            size: Size::all(Val::Percent(100.)),
                            padding: UiRect::all(Val::Px(6.)),
                            flex_direction: FlexDirection::Column,
                            ..default()
                        },
                        background_color: NODE_BACKGROUND.into(),
                        ..default()
                    });

                    // Slots (above content, overlaps borders)
                    let mut iy = 40.;
                    let mut oy = 40.;
                    for slot_def in node.slots() {
                        let position = if slot_def.is_input() {
                            let rect = UiRect {
                                left: Val::Px(-5.),
                                top: Val::Px(iy),
                                ..default()
                            };
                            iy += 20.;
                            rect
                        } else {
                            let rect = UiRect {
                                right: Val::Px(-5.),
                                top: Val::Px(oy),
                                ..default()
                            };
                            oy += 20.;
                            rect
                        };
                        p.spawn((
                            NodeBundle {
                                style: Style {
                                    position_type: PositionType::Absolute,
                                    position,
                                    size: Size::all(Val::Px(10.)),
                                    ..default()
                                },
                                background_color: SLOT_BACKGROUND.into(),
                                ..default()
                            },
                            Name::new(format!("slot:{}", slot_def.name())),
                            Interaction::default(),
                            SlotUI {
                                def: slot_def.clone(),
                            },
                        ));

                        let mut position = position;
                        if slot_def.is_input() {
                            position.left = Val::Px(10.);
                        } else {
                            position.right = Val::Px(10.);
                        }

                        p.spawn((
                            NodeBundle {
                                style: Style {
                                    position_type: PositionType::Absolute,
                                    position,
                                    size: Size::AUTO,
                                    ..default()
                                },
                                background_color: Color::NONE.into(),
                                ..default()
                            },
                            Name::new(format!("text:{}", slot_def.name())),
                        ))
                        .with_children(|p| {
                            p.spawn(TextBundle {
                                text: Text::from_section(slot_def.name(), text_style.clone()),
                                ..default()
                            });
                        });
                    }
                });
        }
    }
}

fn slot_interaction(
    mut q_slot: Query<(
        &bevy::ui::Node,
        &Style,
        &Interaction,
        &mut BackgroundColor,
        &GlobalTransform,
        Option<&CalculatedClip>,
        &SlotUI,
    )>,
    q_window: Query<&Window, With<PrimaryWindow>>,
) {
    let primary_window = q_window.single();

    for (
        ui_node,
        style,
        interaction,
        mut background_color,
        global_transform,
        calculated_clip,
        slot_ui,
    ) in &mut q_slot
    {
        //let mut delta = None;
        match *interaction {
            Interaction::Clicked => {
                // let node_pos = calc_ui_node_position(ui_node, global_transform, calculated_clip);
                // let cursor_pos = get_cursor_pos(primary_window);
                // if widget_title.drag_pos.is_none() {
                //     // Start dragging
                //     widget_title.drag_pos = Some(cursor_pos - node_pos);
                // } else {
                //     // Calculate delta since drag started
                //     delta = Some(cursor_pos - node_pos - widget_title.drag_pos.unwrap());
                // }
            }
            Interaction::Hovered => {
                *background_color = slot_ui.def().hover_color().into();
                // if widget_title.drag_pos.is_some() {
                //     // End drag
                //     widget_title.drag_pos = None;
                // }
            }
            Interaction::None => {
                *background_color = slot_ui.def().background_color().into();
                // if widget_title.drag_pos.is_some() {
                //     // End drag
                //     widget_title.drag_pos = None;
                // }
            }
        }
    }
}
