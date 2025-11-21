use example::register_test;
use mile_core::{App, Mile};

fn main() {
    Mile::new()
        .add_demo(|| {
            register_test().expect("drag demo");
        })
        .run();
}
