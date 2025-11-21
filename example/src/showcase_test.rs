use glam::{vec2, vec4};
use mile_db::DbError;
use mile_ui::{
    mui_prototype::{Mui, UiEventKind, UiPanelData, UiState},
    structs::PanelField,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct SourceData {
    count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct TargetData;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct TData;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct DragTest;

pub fn register_test() -> Result<(), DbError> {
    // 拖拽源面板：开始拖拽时写 payload，拖动&结束时打印。
    Mui::<SourceData>::new("drag_source")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .z_index(3)
                .position(vec2(120.0, 160.0))
                .size(vec2(160.0, 110.0))
                .color(vec4(0.2, 0.6, 0.9, 0.9))
                .events()
                .on_event(UiEventKind::SourceDragStart, |_flow| {
                    println!("[source] drag start");
                })
                .source_drag_start(|flow| {
                    println!("当前拖拽开始了");
                    flow.style_add(PanelField::SIZE_X.bits(), [50.0, 0.0, 0.0, 0.0]);
                    flow.style_add(PanelField::SIZE_Y.bits(), [50.0, 0.0, 0.0, 0.0]);
                    flow.set_drag_payload(DragTest);
                })
                .source_drag_over(|_flow, delta| {
                    println!("[source] drag delta {:?}", delta);
                })
                .source_drag_drop(|_flow| {
                    println!("[source] drag drop");
                    _flow.style_add(PanelField::SIZE_X.bits(), [160.0, 0.0, 0.0, 0.0]);
                    _flow.style_add(PanelField::SIZE_Y.bits(), [110.0, 0.0, 0.0, 0.0]);
                })
                .finish()
        })
        .build()?;

    // 拖拽接收面板：enter/over/leave/drop 打印 payload。
    Mui::<TargetData>::new("drag_target")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .z_index(2)
                .position(vec2(360.0, 220.0))
                .size(vec2(220.0, 150.0))
                .color(vec4(0.8, 0.3, 0.3, 0.9))
                .events()
                .on_event(UiEventKind::Hover, |flow| {})
                .on_event(UiEventKind::TargetDragDrop, |flow| {
                    println!("2号被置入");
                })
                .finish()
        })
        .build()?;

    Mui::<TData>::new("Ttest")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .z_index(2)
                .position(vec2(660.0, 220.0))
                .size(vec2(220.0, 150.0))
                .color(vec4(0.8, 0.3, 0.3, 0.9))
                .events()
                .on_event(UiEventKind::Hover, |flow| {})
                .target_drag_drop::<DragTest, _>(|flow, target_flow| {
                    println!("3号被drop");
                })
                .finish()
        })
        .build()?;

    Ok(())
}
