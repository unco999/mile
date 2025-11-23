use example::register_free_layout;
use mile_core::Mile;

fn main() {
    Mile::new()
        .add_demo(|| {
            register_free_layout().expect("free layout demo");
        })
        .run();
}
