use example::register_state_demo;
use mile_core::Mile;

fn main() {
    Mile::new()
        .add_demo(|| {
            register_state_demo().expect("state demo");
        })
        .run();
}
