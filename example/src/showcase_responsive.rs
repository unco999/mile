use glam::{vec2, vec4};
use mile_db::DbError;
use mile_ui::mui_prototype::{Mui, UiPanelData, UiState};
use mile_ui::mui_rel::{RelLayoutKind, RelSpace};

// Responsive-like behavior using container percent and row wrap.
pub fn register_responsive_layout() -> Result<(), DbError> {
    // Parent uses percent-of-parent sizing to simulate responsiveness.
    Mui::<UiPanelData>::new("ex_resp_parent")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            let state = state
                .z_index(1)
                .position(vec2(80.0, 80.0))
                .size(vec2(640.0, 400.0))
                .color(vec4(0.06, 0.08, 0.10, 0.95))
                .container_style()
                .origin(vec2(16.0, 16.0))
                .size_container(vec2(608.0, 368.0))
                .slot_size(vec2(140.0, 96.0))
                .layout(RelLayoutKind::ring(10.0))
                .finish();
            state
        })
        .build()?;

    for i in 0..8 {
        let id = format!("ex_resp_child_{}", i);
        Mui::<UiPanelData>::new(Box::leak(id.into_boxed_str()))?
            .default_state(UiState(0))
            .state(UiState(0), move |mut state| {
                state.rel().container_with::<UiPanelData>("ex_resp_parent");
                state
                    .z_index(2 + i)
                    .rel_position_in(RelSpace::Parent, vec2(0.0, 0.0))
                    .size(vec2(140.0, 96.0))
                    .color(vec4(0.20 + i as f32 * 0.06, 0.35, 0.60, 1.0))
            })
            .build()?;
    }
    Ok(())
}

