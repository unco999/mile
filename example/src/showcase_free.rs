use glam::{vec2, vec4};
use mile_db::DbError;
use mile_font::prelude::FontStyle;
use mile_ui::{
    mui_prototype::{BorderStyle, Mui, StateStageBuilder, UiEventKind, UiPanelData, UiState},
    mui_rel::RelLayoutKind,
};

const FREE_CONTAINER_ID: &str = "free_demo_container";

pub fn register_free_layout() -> Result<(), DbError> {
    build_free_container()?;
    build_free_children()?;
    Ok(())
}

fn build_free_container() -> Result<(), DbError> {
    Mui::<UiPanelData>::new(FREE_CONTAINER_ID)?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .z_index(1)
                .position(vec2(120.0, 120.0))
                .size(vec2(720.0, 480.0))
                .drag(true)
                .color(vec4(0.06, 0.08, 0.12, 0.95))
                .border(BorderStyle {
                    color: [0.35, 0.65, 0.90, 1.0],
                    width: 3.0,
                    radius: 16.0,
                })
                .container_style()
                .padding([16.0, 16.0, 16.0, 16.0])
                .size_container(vec2(720.0, 480.0))
                .layout(RelLayoutKind::Free)
                .finish()
                .events()
                .on_event(UiEventKind::Init, |flow| {
                    flow.text(
                        "Free Container: 子面板位置完全由自身 position 决定",
                        FontStyle {
                            font_size: 26,
                            font_line_height: 32,
                            ..Default::default()
                        },
                    );
                })
                .finish()
        })
        .build()?;
    Ok(())
}

fn build_free_children() -> Result<(), DbError> {
    let demos = [
        (vec2(40.0, 80.0), vec2(200.0, 96.0), [0.85, 0.35, 0.35, 0.95], "左上角固定"),
        (vec2(420.0, 90.0), vec2(220.0, 120.0), [0.32, 0.55, 0.90, 0.95], "右侧标签"),
        (
            vec2(180.0, 250.0),
            vec2(320.0, 140.0),
            [0.25, 0.75, 0.55, 0.92],
            "中央卡片 - 测试 absolute",
        ),
    ];

    for (idx, (pos, size, color, label)) in demos.into_iter().enumerate() {
        let panel_id = format!("free_demo_child_{}", idx);
        Mui::<UiPanelData>::new(Box::leak(panel_id.into_boxed_str()))?
            .default_state(UiState(0))
            .state(
                UiState(0),
                move |mut builder: StateStageBuilder<UiPanelData>| {
                    builder.rel().container_with::<UiPanelData>(FREE_CONTAINER_ID);
                    builder
                        .z_index(5 + idx as i32)
                        .position(pos)
                        .size(size)
                        .color(vec4(color[0], color[1], color[2], color[3]))
                        .border(BorderStyle {
                            color: [0.05, 0.05, 0.05, 0.35],
                            width: 1.0,
                            radius: 12.0,
                        })
                        .drag(true)
                        .hover(true)
                        .events()
                        .on_event(UiEventKind::Init, move |flow| {
                            flow.text(
                                label,
                                FontStyle {
                                    font_size: 20,
                                    font_line_height: 26,
                                    ..Default::default()
                                },
                            );
                        })
                        .finish()
                },
            )
            .build()?;
    }

    Ok(())
}
