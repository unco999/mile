use example::register_basic_layout;
use mile_core::{App, Mile};

fn main() {
    Mile::new()
        .add_demo(|| {
            register_basic_layout().expect("animation demo");
        })
        .run();
}
