#![allow(unused)]

use bevy::prelude::*;

pub fn close_on_esc(mut ev_app_exit: EventWriter<AppExit>, input: Res<ButtonInput<KeyCode>>) {
    if input.just_pressed(KeyCode::Escape) {
        ev_app_exit.send(AppExit::Success);
    }
}

pub const COLOR_RED: Color = Color::linear_rgb(1., 0., 0.);
pub const COLOR_GREEN: Color = Color::linear_rgb(0., 1., 0.);
pub const COLOR_BLUE: Color = Color::linear_rgb(0., 0., 1.);
pub const COLOR_YELLOW: Color = Color::linear_rgb(1., 1., 0.);
pub const COLOR_CYAN: Color = Color::linear_rgb(0., 1., 1.);
pub const COLOR_OLIVE: Color = Color::linear_rgb(0.5, 0.5, 0.);
pub const COLOR_PURPLE: Color = Color::linear_rgb(0.5, 0., 0.5);
