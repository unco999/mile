pub mod gpu_struct;
pub mod minimal_runtime;

pub const DEFAULT_FONT_PATH: &str = "tf/STXIHEI.ttf";

pub mod event {
    use std::{rc::Rc, sync::Arc};

    use mile_api::prelude::_ty::PanelId;

    use crate::prelude::FontStyle;

    pub fn font_str<'a>(text: &'a str) -> Arc<str> {
        Arc::from(text)
    }

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

    /// 移除指定面板的文字渲染缓存（不清理 SDF 纹理/描述表）
    /// 用于面板切换字体或重置文本时，丢弃先前生成的 GpuText/实例。
    #[derive(Debug)]
    pub struct RemoveRenderFont {
        pub parent: PanelId,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct ResetFontRuntime;
}

pub mod prelude {
    pub use crate::gpu_struct::*;
}
