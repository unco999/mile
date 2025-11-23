use example::register_text_demo;
use mile_core::{App, Mile};

fn main() {
    Mile::new()
        .add_demo(|| {
            register_text_demo().expect("animation demo");
        })  
        .run();
}
    