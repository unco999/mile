use example::{register_state_demo, register_test};
use mile_core::Mile;

fn main() {
    Mile::new()
        .add_demo(|| register_test().expect("测试错误"))
        .run();
}
