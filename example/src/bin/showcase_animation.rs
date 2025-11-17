use example::register_animation_demo;
use mile_core::{App, Mile};

fn main() {
    Mile::new()
        .add_demo(|| {
            register_animation_demo().expect("animation demo");
        })
        .run();
}
