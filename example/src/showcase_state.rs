use glam::{vec2, vec4};
use mile_db::DbError;
use mile_ui::mui_prototype::{Mui, UiEventKind, UiPanelData, UiState};

// Three-state panel cycling position/size on each click.
pub fn register_state_demo() -> Result<(), DbError> {
    Mui::<UiPanelData>::new("ex_state_panel")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .z_index(4)
                .position(vec2(160.0, 150.0))
                .size(vec2(220.0, 140.0))
                .color(vec4(0.18, 0.36, 0.82, 1.0))
                .events()
                .on_event(UiEventKind::Click, |flow| {
                    flow.set_state(UiState(1));
                })
                .finish()
        })
        .state(UiState(1), |state| {
            state
                .z_index(5)
                .position(vec2(360.0, 240.0))
                .size(vec2(280.0, 180.0))
                .color(vec4(0.88, 0.42, 0.32, 1.0))
                .events()
                .on_event(UiEventKind::Click, |flow| {
                    flow.set_state(UiState(2));
                })
                .finish()
        })
        .state(UiState(2), |state| {
            state
                .z_index(6)
                .position(vec2(220.0, 360.0))
                .size(vec2(200.0, 120.0))
                .color(vec4(0.18, 0.68, 0.46, 1.0))
                .events()
                .on_event(UiEventKind::Click, |flow| {
                    flow.set_state(UiState(0));
                })
                .finish()
        })
        .build()?;
    Ok(())
}
