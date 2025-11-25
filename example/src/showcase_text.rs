use std::sync::Arc;

use glam::{vec2, vec4};
use mile_db::DbError;
use mile_font::prelude::{FontStyle, TextAlign};
use mile_ui::mui_prototype::{Mui, UiPanelData, UiState};

// Simple text rendering demo using the font event bridge in EventFlow::text.
pub fn register_text_demo() -> Result<(), DbError> {
    Mui::<UiPanelData>::new("ex_text_panel")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .z_index(5)
                .position(vec2(120.0, 100.0))
                .color(vec4(0.1,0.2,0.5,0.9))
                .size(vec2(540.0, 280.0))
                .events()
                .on_event(mile_ui::mui_prototype::UiEventKind::Init, |flow| {
                    flow.text(
                        "点一下变换",
                        FontStyle {
                            font_size: 12,
                            font_file_path:Arc::from("tf/LXGWWenKaiMono-Light.ttf"),
                            font_line_height: 0,
                            text_align:TextAlign::Center,
                            ..Default::default()
                        },
                    );
                })
                .on_event(mile_ui::mui_prototype::UiEventKind::Click, |flow| {
                    flow.clear_texts();
                    flow.text(
                        "测试文本测试文本测试文本测试文本测试文本测试文本测试文本测试文本测试文本测试文本测试文本",
                        FontStyle{
                            font_file_path:Arc::from("tf/LXGWWenKaiMono-Light.ttf"),
                            font_size:50,
                            font_line_height:0,
                            first_weight:0.0,
                            ..Default::default()
                        }
                    );
                })
                .finish()
        })
        .build()?;
    Ok(())
}
