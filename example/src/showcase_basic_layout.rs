use glam::{vec2, vec4};
use mile_db::DbError;
use mile_ui::mui_prototype::{
    Mui, PanelPayload, UiPanelData, UiState,
};

// Simple parent container with 3 children laid out horizontally via container spec.
pub fn register_basic_layout() -> Result<(), DbError> {
    // Parent container
    Mui::<UiPanelData>::new("ex_basic_parent")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            let state = state
                .z_index(1)
                .position(vec2(220.0, 160.0))
                .size(vec2(520.0, 220.0))
                .color(vec4(0.08, 0.10, 0.12, 0.9))
                .container_style()
                .origin(vec2(24.0, 24.0))
                .size_container(vec2(472.0, 172.0))
                .slot_size(vec2(148.0, 148.0))
                .layout(mile_ui::mui_rel::RelLayoutKind::ring(8.0))
                .finish();
            state
        })
        .build()?;

    // Children
    for i in 0..3 {
        let id = format!("ex_basic_child_{}", i);
        Mui::<UiPanelData>::new(Box::leak(id.into_boxed_str()))?
            .default_state(UiState(0))
            .state(UiState(0), move |mut state| {
                state.rel().container_with::<UiPanelData>("ex_basic_parent");
                state
                    .z_index(2 + i)
                    .position(vec2(0.0, 0.0))
                    .size(vec2(148.0, 148.0))
                    .color(vec4(0.18 + 0.06 * i as f32, 0.24, 0.45, 1.0))
                    .events()
                    .on_event(mile_ui::mui_prototype::UiEventKind::Hover, |flow| {
                        flow.position_anim()
                            .from_current()
                            .to_offset(vec2(0.0, -8.0))
                            .duration(0.12)
                            .easing(mile_ui::mui_anim::Easing::QuadraticOut)
                            .push(flow);
                    })
                    .on_event(mile_ui::mui_prototype::UiEventKind::Out, |flow| {
                        flow.position_anim()
                            .from_current()
                            .to_snapshot()
                            .duration(0.12)
                            .easing(mile_ui::mui_anim::Easing::QuadraticIn)
                            .push(flow);
                    })
                    .finish()
            })
            .build()?;
    }
    Ok(())
}

