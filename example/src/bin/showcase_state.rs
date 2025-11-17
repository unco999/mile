use example::register_state_demo;
use mile_core::Mile;

fn main() {
    Mile::new()
        .add_demo(|| {
            //这个是AI写的3状态轮换
            register_state_demo().expect("state demo");
        })
        .run();
}
