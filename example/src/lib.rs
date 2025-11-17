//! Mile UI Showcase Examples
//! - Basic Layout Container
//! - Animation
//! - Responsive Layout
//! - Text Rendering
//! - AST Shader (GPU DSL)
//!
//! Each example exposes a `register_*` function that builds panels into the global DB.
//! You can run the tests to invoke these builders; visual output appears when running
//! the main app since runtime picks up DB contents.

mod showcase_basic_layout;
mod showcase_animation;
mod showcase_responsive;
mod showcase_text;
mod showcase_ast_shader;

pub use showcase_basic_layout::register_basic_layout;
pub use showcase_animation::register_animation_demo;
pub use showcase_responsive::register_responsive_layout;
pub use showcase_text::register_text_demo;
pub use showcase_ast_shader::register_ast_shader_demo;

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests only build panels into the DB.
    // Run the main app to see visuals; runtime will ingest DB on the next frame.

    #[test]
    fn example_basic_layout() {
        register_basic_layout().expect("basic layout");
    }

    #[test]
    fn example_animation() {
        register_animation_demo().expect("animation");
    }

    #[test]
    fn example_responsive() {
        register_responsive_layout().expect("responsive");
    }

    #[test]
    fn example_text() {
        register_text_demo().expect("text");
    }

    #[test]
    fn example_ast_shader() {
        register_ast_shader_demo().expect("ast shader");
    }
}
