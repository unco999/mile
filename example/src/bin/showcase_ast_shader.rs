use example::register_ast_shader_demo;
use mile_core::{App, Mile};

fn main() {
    Mile::new()
        .add_demo(|| {
            register_ast_shader_demo().expect("animation demo");
        })
        .run();
}
