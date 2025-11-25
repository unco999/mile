use glam::{vec2, vec4};
use mile_db::DbError;
use mile_font::prelude::FontStyle;
use mile_ui::{
    mui_prototype::{BorderStyle, Mui, StateStageBuilder, UiEventKind, UiPanelData, UiState},
    mui_rel::{RelFloatAxis, RelLayoutKind},
};

const FLOAT_CONTAINER: &str = "float_demo_primary";
const STACK_CONTAINER: &str = "float_demo_vertical";

pub fn register_float_layout() -> Result<(), DbError> {
    build_primary_float_container()?;
    build_vertical_float_container()?;
    build_primary_children()?;
    build_vertical_children()?;
    Ok(())
}

fn build_primary_float_container() -> Result<(), DbError> {
    Mui::<UiPanelData>::new(FLOAT_CONTAINER)?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .z_index(2)
                .position(vec2(80.0, 80.0))
                .size(vec2(640.0, 420.0))
                .color(vec4(0.09, 0.13, 0.18, 0.92))
                .border(BorderStyle {
                    color: [0.35, 0.55, 0.85, 1.0],
                    width: 3.0,
                    radius: 12.0,
                })
                .container_style()
                .padding([20.0, 20.0, 20.0, 20.0])
                .slot_size(vec2(120.0, 72.0))
                .size_container(vec2(640.0, 420.0))
                .layout(RelLayoutKind::Float {
                    axis: RelFloatAxis::Horizontal,
                    spacing: [16.0, 16.0],
                    align_center: false,
                })
                .finish()
                .events()
                .on_event(UiEventKind::Init, |flow| {
                    flow.text(
                        "Float 容器 (水平) - 宽度不足时换行排布",
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

fn build_vertical_float_container() -> Result<(), DbError> {
    Mui::<UiPanelData>::new(STACK_CONTAINER)?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .z_index(2)
                .position(vec2(760.0, 120.0))
                .size(vec2(320.0, 500.0))
                .color(vec4(0.16, 0.12, 0.18, 0.92))
                .border(BorderStyle {
                    color: [0.80, 0.45, 0.75, 1.0],
                    width: 2.0,
                    radius: 18.0,
                })
                .container_style()
                .padding([16.0, 18.0, 16.0, 18.0])
                .slot_size(vec2(140.0, 64.0))
                .size_container(vec2(320.0, 500.0))
                .layout(RelLayoutKind::Float {
                    axis: RelFloatAxis::Vertical,
                    spacing: [12.0, 18.0],
                    align_center: true,
                })
                .finish()
                .events()
                .on_event(UiEventKind::Init, |flow| {
                    flow.text(
                        "Float 容器 (垂直) - 自动开新列",
                        FontStyle {
                            font_size: 22,
                            font_line_height: 28,
                            ..Default::default()
                        },
                    );
                })
                .finish()
        })
        .build()?;
    Ok(())
}

fn build_primary_children() -> Result<(), DbError> {
    for idx in 0..18 {
        let id = format!("float_demo_chip_{idx}");
        Mui::<UiPanelData>::new(Box::leak(id.into_boxed_str()))?
            .default_state(UiState(0))
            .state(
                UiState(0),
                move |mut stage: StateStageBuilder<UiPanelData>| {
                    stage.rel().container_with::<UiPanelData>(FLOAT_CONTAINER);
                    let hue = 0.15 + 0.03 * (idx as f32);
                    stage
                        .z_index(4 + idx)
                        .size(vec2(120.0, 72.0))
                        .color(vec4(0.25 + hue * 0.2, 0.45, 0.35 + hue * 0.3, 0.95))
                        .border(BorderStyle {
                            color: [0.05, 0.05, 0.05, 0.35],
                            width: 1.0,
                            radius: 10.0,
                        })
                        .events()
                        .on_event(UiEventKind::Init, move |flow| {
                            flow.text(
                                &format!("Chip {}", idx + 1),
                                FontStyle {
                                    font_size: 20,
                                    font_line_height: 24,
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

fn build_vertical_children() -> Result<(), DbError> {
    for idx in 0..12 {
        let id = format!("float_demo_card_{idx}");
        Mui::<UiPanelData>::new(Box::leak(id.into_boxed_str()))?
            .default_state(UiState(0))
            .state(
                UiState(0),
                move |mut stage: StateStageBuilder<UiPanelData>| {
                    stage.rel().container_with::<UiPanelData>(STACK_CONTAINER);
                    let t = idx as f32 / 12.0;
                    stage
                        .z_index(3 + idx)
                        .size(vec2(140.0, 64.0 + (t * 24.0)))
                        .color(vec4(0.35 + t * 0.3, 0.2 + t * 0.4, 0.55 + t * 0.2, 0.92))
                        .border(BorderStyle {
                            color: [0.95, 0.95, 0.95, 0.35],
                            width: 1.0,
                            radius: 14.0,
                        })
                        .events()
                        .on_event(UiEventKind::Init, move |flow| {
                            flow.text(
                                &format!("Badge {}", idx + 1),
                                FontStyle {
                                    font_size: 18,
                                    font_line_height: 24,
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
