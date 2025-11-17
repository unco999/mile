use glam::{vec2, vec4};
use mile_db::DbError;
use mile_ui::mui_prototype::{Mui, UiEventKind, UiPanelData, UiState};
use mile_ui::mui_anim::Easing;

// A panel that animates position and color on hover/click.
pub fn register_animation_demo() -> Result<(), DbError> {
    Mui::<UiPanelData>::new("ex_anim_panel")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .z_index(3)
                .position(vec2(320.0, 220.0))
                .size(vec2(240.0, 140.0))
                .color(vec4(0.10, 0.28, 0.75, 1.0))
                .events()
                .on_event(UiEventKind::Hover, |flow| {
                    flow.position_anim()
                        .from_current()
                        .to_offset(vec2(0.0, -10.0))
                        .duration(0.18)
                        .easing(Easing::QuadraticOut)
                        .push(flow);
                })
                .on_event(UiEventKind::Out, |flow| {
                    flow.position_anim()
                        .from_current()
                        .to_snapshot()
                        .duration(0.18)
                        .easing(Easing::QuadraticIn)
                        .push(flow);
                })
                .on_event(UiEventKind::Click, |flow| {
                    flow.color_anim()
                        .from_current()
                        .to(vec4(0.85, 0.25, 0.35, 1.0))
                        .duration(0.25)
                        .easing(Easing::CubicOut)
                        .push(flow);
                })
                .finish()
        })
        .build()?;
    Ok(())
}

