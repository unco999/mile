pub mod gpu_struct;
pub mod structs;
pub mod test;

pub mod event {
    use crate::structs::FontStyle;

    pub struct BatchFontEntry {
        pub str: &'static str,
        pub font_file_path: &'static str,
    }

    /**
     * 批量文件的plan
     * 需要写清楚
     */
    pub struct BatchRenderFont<'style, ID: Into<u32>> {
        pub str: &'static str,
        pub font_file_path: &'static str,
        pub parent: ID,
        pub font_style: &'style FontStyle,
    }
}

pub mod prelude {
    pub use crate::gpu_struct::*;
    pub use crate::structs::*;
}
