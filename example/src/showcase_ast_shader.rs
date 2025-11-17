use glam::{vec2, vec4};
use mile_db::DbError;
use mile_ui::mui_anim::Easing;
use mile_ui::mui_prototype::{
    Mui, ShaderStage, UiEventKind, UiPanelData, UiState,
};
use mile_gpu_dsl::prelude::Expr;
use mile_gpu_dsl::dsl::{cv, rv, smoothstep};
use mile_gpu_dsl::core::dsl::{sin, wvec4};

fn frag_wave() -> Expr {
    let uv = rv("uv");
    let t = cv("time");
    let wave = (sin(uv.x() * 8.0 + t.clone() * 1.2) + sin(uv.y() * 9.0 - t * 1.1)) * 0.5;
    let crest = smoothstep(0.55, 0.9, (wave + 1.0) * 0.5);
    wvec4(0.10 + 0.8 * crest.clone(), 0.20 + 0.3 * crest.clone(), 0.28 + crest.clone(), 1.0)
}

pub fn register_ast_shader_demo() -> Result<(), DbError> {
    Mui::<UiPanelData>::new("ex_ast_shader")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .z_index(6)
                .position(vec2(380.0, 260.0))
                .size(vec2(280.0, 180.0))
                .color(vec4(0.05, 0.05, 0.06, 1.0))
                .events()
                .on_event(UiEventKind::Init, |flow| {
                    // 请求片元着色器（AST）
                    flow.request_fragment_shader(|_| frag_wave());
                })
                .on_event(UiEventKind::Hover, |flow| {
                    flow.position_anim()
                        .from_current()
                        .to_offset(vec2(0.0, -10.0))
                        .duration(0.16)
                        .easing(Easing::QuadraticOut)
                        .push(flow);
                })
                .on_event(UiEventKind::Out, |flow| {
                    flow.position_anim()
                        .from_current()
                        .to_snapshot()
                        .duration(0.16)
                        .easing(Easing::QuadraticIn)
                        .push(flow);
                })
                .finish()
        })
        .build()?;
    Ok(())
}

