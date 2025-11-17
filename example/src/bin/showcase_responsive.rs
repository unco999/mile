use example::register_responsive_layout;
use mile_core::{App, Mile};

fn main() {
   Mile::new()
        .add_demo(|| {
            register_responsive_layout().expect("animation demo");
        })
        .run();
}
