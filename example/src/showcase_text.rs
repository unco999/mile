use glam::{vec2};
use mile_db::DbError;
use mile_ui::mui_prototype::{Mui, UiPanelData, UiState};

// Simple text rendering demo using the font event bridge in EventFlow::text.
pub fn register_text_demo() -> Result<(), DbError> {
    Mui::<UiPanelData>::new("ex_text_panel")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .z_index(5)
                .position(vec2(120.0, 100.0))
                .size(vec2(540.0, 280.0))
                .events()
                .on_event(mile_ui::mui_prototype::UiEventKind::Init, |flow| {
                    flow.text(
                        "Mile UI — Text Demo\n点击可触发更多文本",
                        "tf/NotoSansSC-Regular.otf".into(),
                        24,
                        [0.95, 0.95, 0.95, 1.0],
                        400,
                        28,
                    );
                })
                .on_event(mile_ui::mui_prototype::UiEventKind::Click, |flow| {
                    flow.clear_texts();
                    flow.text(
                        "已点击：更新内容 & 样式",
                        "tf/NotoSansSC-Regular.otf".into(),
                        26,
                        [0.90, 0.60, 0.20, 1.0],
                        700,
                        28,
                    );
                })
                .finish()
        })
        .build()?;
    Ok(())
}

