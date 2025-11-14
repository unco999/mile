pub mod gpu_struct;
pub mod minimal_runtime;

pub mod event {
    use std::{rc::Rc, sync::Arc};

    use mile_api::prelude::_ty::PanelId;

    use crate::prelude::FontStyle;

    #[derive(Debug)]
    pub struct BatchFontEntry {
        pub text: Arc<str>,
        pub font_file_path: Arc<str>,
    }

    
    /**
     * 批量文件的plan
     * 需要写清楚
     */
    pub struct BatchRenderFont {
        pub text: Arc<str>,
        pub font_file_path: Arc<str>,    
        pub parent: PanelId,
        pub font_style: Arc<FontStyle>,
    }
}

pub mod prelude {
    pub use crate::gpu_struct::*;
}
