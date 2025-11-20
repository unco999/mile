use glam::{vec2, vec4};
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
                .color(vec4(0.1,0.2,0.5,0.9))
                .size(vec2(540.0, 280.0))
                .events()
                .on_event(mile_ui::mui_prototype::UiEventKind::Init, |flow| {
                    flow.text(
                        "Mile UI — Text Demo 点击渲染文本",
                        "tf/STXIHEI.ttf".into(),
                        72,
                        [0.95, 0.95, 0.95, 1.0],
                        400,
                        90,
                    );
                })
                .on_event(mile_ui::mui_prototype::UiEventKind::Click, |flow| {
                    flow.clear_texts();
                    flow.text(
                        "长篇小说是小说形式之一，以篇幅长（通常超过十万字）、容量大、情节复杂为特征,通过分章节或分卷形式构建叙事结构。其核心要素包括人物塑造、故事情节和环境描写，涵盖社会生活的广泛性及人物性格多样性，常运用叙述时间处置、空间观照和视角选择等手法，部分作品涉及传统文",
                        "tf/STXIHEI.ttf".into(),
                        26,
                        [0.90, 0.60, 0.20, 1.0],
                        700,
                        40,
                    );
                })
                .finish()
        })
        .build()?;
    Ok(())
}
