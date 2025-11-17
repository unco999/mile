use glam::{Vec2, vec2, vec4};
use mile_db::DbError;
use mile_ui::mui_anim::Easing;
use mile_ui::mui_prototype::{BorderStyle, Mui, StateStageBuilder, UiEventData, UiEventKind, UiPanelData, UiState};
use mile_ui::mui_rel::{RelLayoutKind, RelSpace};
use mile_ui::prelude::Field;
use mile_ui::structs::PanelField;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct DataTest {
    count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
struct UIDefault;


#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct TestCustomData {
    pub count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
struct TestUi{
    pub test_count:f32,
    pub lock_state:bool
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
struct SliderLock{
    pub lock:bool
}

// Responsive-like behavior using container percent and row wrap.
pub fn register_responsive_layout() -> Result<(), DbError> {
      let color_panel = Mui::<TestUi>::new("demo_color_target")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            let state = state
                .z_index(2)
                .position(vec2(320.0, 180.0))
                .size(vec2(240.0, 140.0))
                .color(vec4(0.12, 0.28, 0.60, 1.0))
                // Respond to brightness changes via observers (data-driven)
                .events()
                    .on_data_change::<UiPanelData,_>(None,|target,flow|{
                        println!("当前color面版监听的事件 {:?}",target);
                        if(flow.payload_ref().lock_state){ return; }
                        flow.style_set(PanelField::TRANSPARENT.bits(), [target.brightness,0.0,0.0,0.0]);
                    })
                    .on_event(UiEventKind::Click, |flow|{
                        let position = Vec2::from_array(flow.record.snapshot.position);
                        println!("当前position {:?}",position);
                        let _ = Mui::<UIDefault>::new("nb").unwrap()
                            .default_state(UiState(0))
                            .state(UiState(0), |state|{
                                let state = state
                                    .color(vec4(0.0, 1.0, 1.0, 1.0))
                                    .size(vec2(300.0, 300.0))
                                    .with_trigger_mouse_pos()
                                    ;
                                state
                            })
                            .build();
                    })
                .finish();
            state
        })
            .build()?
        ;

    // Slider "thumb" – draggable only on X, clamped into [120, 580] with 5px steps
    let x_min = 120.0_f32;
    let x_max = 580.0_f32;
    let slider_thumb = Mui::<UiPanelData>::new("demo_slider_thumb")?
        .default_state(UiState(0))
        .state(UiState(0), move |state: StateStageBuilder<UiPanelData>| {
            let state = state
                .z_index(3)
                .position(vec2(x_min, 420.0))
                .size(vec2(36.0, 36.0))
                .color(vec4(0.85, 0.55, 0.22, 1.0))
                // X-only with relative budget [0..(x_max-x_min)], step 5px; origin from snapshot
                .clamp_offset(Field::OnlyPositionX, [0.0, (x_max - x_min - 36.0), 5.0])
                .events()
                .on_data_change::<SliderLock,_>(None, |target,flow|{
                    if(target.lock){
                        flow.set_state(UiState(1));
                    }
                })
                .on_event_with(UiEventKind::Drag, move |flow,drag_detla| {
                    // 将滑块位置映射为 [0,1] 的进度，并写入“颜色面板”的 payload.brightness，
                    // 触发本面板的 DB 提交（只允许改自己）。
                    let pos = Vec2::from_array(flow.args().record_snapshot.snapshot.position);
                    let len = (x_max - x_min - 36.0).max(1.0);
                    let t = ((pos.x - x_min) / len).clamp(0.0, 1.0);
                    println!("拖拽了");
                    match drag_detla {
                        UiEventData::Vec2(vec2) => {
                            flow.update_self_payload(|data| {
                                 data.brightness += vec2.x / x_max;
                            });
                        },
                        _=>{}
                    }

           
                })
                .finish();
            state
        })
        .state(UiState(1), move |state|{
            let state = state
                 .color(vec4(1.0, 0.0, 0.22, 1.0))
                 .clamp_offset(Field::OnlyPositionX, [0.0, (x_max - x_min - 36.0), 5.0])
                 .events()
                 .on_data_change::<SliderLock,_>(None, |target,flow|{
                    if(target.lock == false){
                         println!("回到state 0");
                         flow.set_state(UiState(0));
                    }
                    flow.update_self_payload(|payload|{
                        payload.brightness = 0.0;
                    });
                  })
                 .finish()
                ;
            state
        })
        .build()?;

    // Slider track: a background bar aligned with the slider thumb's origin and height,
    // width equals the clamp budget (x_max - x_min).
    let slider_track = Mui::<UiPanelData>::new("demo_slider_track")?
        .default_state(UiState(0))
        .state(UiState(0), move |state| {
            let state = state
                .z_index(2)
                .position(vec2(x_min, 420.0))
                .size(vec2((x_max - x_min), 36.0))
                .color(vec4(0.2, 0.2, 0.2, 0.65))
                .events()
                .finish();
            state
        })
        .build()?;

    // Update button: clicking it will bump the color target's brightness and trigger its observers.
    let update_button = Mui::<SliderLock>::new("demo_update_button")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            let state = state
                .z_index(4)
                .position(vec2(x_min + (x_max - x_min) + 20.0, 420.0))
                .size(vec2(80.0, 36.0))
                .color(vec4(0.25, 0.6, 0.3, 0.9))
                .events()
                .on_event(UiEventKind::Click, |flow| {
                    // 只允许改自己的 payload，提交 DB 触发观察者
                    flow.update_self_payload(|data| {
                        data.lock = !data.lock;
                    });
                })
                .finish();
            state
        })
        .build()?;
   Ok(())
}

