use example::register_float_layout;
use mile_core::Mile;

fn main() {
    Mile::new()
        .add_demo(|| {
            register_float_layout().expect("float layout demo");
        })
        .run();
}
