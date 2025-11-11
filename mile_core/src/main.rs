use crate::run::{App, AppEvent};
use mile_graphics::structs::GlobalState;
use std::sync::{Arc, Mutex};

use winit::event_loop::EventLoop;
pub mod run;
fn main() {
    // Application entry point wires the event loop and GPU/UI subsystems.
    // Event loop dispatches window/user events into our app state.
    let mut event_loop = EventLoop::<AppEvent>::with_user_event();

    let event_loop_main = event_loop.build().unwrap();
    let _proxy = event_loop_main.create_proxy();

    // GlobalState keeps GPU/device handles shared across threads.

    // App bundles hubs, fonts, rendering context, and timing info.
    let mut app = App::new();

    event_loop_main.run_app(&mut app).unwrap();
}
