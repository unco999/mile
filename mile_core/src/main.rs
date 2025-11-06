use crate::run::{App, AppEvent};
use mile_gpu_dsl::prelude::Expr;
use mile_graphics::structs::GlobalState;
use mile_ui::prelude::*;
use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use winit::event_loop::{self, EventLoop};
pub mod run;

fn main() {
    // Application entry point wires the event loop and GPU/UI subsystems.
    // Event loop dispatches window/user events into our app state.
    let mut event_loop = EventLoop::<AppEvent>::with_user_event();

    let event_loop_main = event_loop.build().unwrap();
    let proxy = event_loop_main.create_proxy();

    // GlobalState keeps GPU/device handles shared across threads.
    let global_state: Arc<Mutex<GlobalState>> = Arc::new(Mutex::new(GlobalState::new()));
    let gs = global_state.clone();

    // App bundles hubs, fonts, rendering context, and timing info.
    let mut app = App {
        mile_font: None,
        wgpu_context: None,
        mui_runtime: None,
        global_state: global_state,
        last_frame_time: Instant::now(),
        last_tick: Instant::now(),
        delta_time: Duration::from_secs_f32(0.0),
        tick_interval: Duration::from_secs_f64(1.0 / 60.0),
        kennel: None,
        frame_index: 0,
    };

    event_loop_main.run_app(&mut app).unwrap();
}
