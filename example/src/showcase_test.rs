use glam::{vec2, vec4};
use mile_db::DbError;
use mile_ui::mui_prototype::{Mui, UiPanelData, UiState};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct SourceData {
    count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct TargetData;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct DragPayload {
    value: u32,
}

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
                .source_drag_start_with_payload(|_flow| {
                    println!("[source] drag start");
                    DragPayload { value: 7 }
                })
                .source_drag_over(|_flow, delta| {
                    println!("[source] drag delta {:?}", delta);
                })
                .source_drag_drop(|_flow| {
                    println!("[source] drag drop");
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
                .target_drag_enter::<SourceData, _>(|_flow, payload| {
                    println!("[target] enter payload {:?}", payload);
                })
                .target_drag_over::<SourceData, _>(|_flow, delta, payload| {
                    println!("[target] over delta {:?} payload {:?}", delta, payload);
                })
                .target_drag_leave::<SourceData, _>(|_flow, payload| {
                    println!("[target] leave payload {:?}", payload);
                })
                .target_drag_drop::<SourceData, _>(|_flow, payload| {
                    println!("[target] drop payload {:?}", payload);
                })
                .finish()
        })
        .build()?;

    Ok(())
}
