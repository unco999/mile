use std::sync::Arc;

use glam::{vec2, vec4};
use mile_db::DbError;
use mile_font::prelude::{FontStyle, TextAlign};
use mile_ui::mui_prototype::{Mui, UiPanelData, UiState};

// Simple text rendering demo using the font event bridge in EventFlow::text.
pub fn register_text_demo() -> Result<(), DbError> {
    Mui::<UiPanelData>::new("ex_text_panel")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .z_index(5)
                .position(vec2(120.0, 100.0))
                .color(vec4(0.1,0.2,0.5,0.9))
                .size(vec2(540.0, 280.0))
                .events()
                .on_event(mile_ui::mui_prototype::UiEventKind::Init, |flow| {
                    flow.text(
                        "我,",
                        FontStyle {
                            font_size: 50,
                            font_file_path:Arc::from("tf/LXGWWenKaiMono-Light.ttf"),
                            font_line_height: 50,
                            text_align:TextAlign::Center,
                            ..Default::default()
                        },
                    );
                })
                .on_event(mile_ui::mui_prototype::UiEventKind::Click, |flow| {
                    flow.clear_texts();
                    flow.text(
                        "那只鸟生来一翼破损,风也不凑巧,天黑了,鸟群南迁,猎人们在平原上谈笑,自信的狂风中有迎难而上的鸟群,她蜷缩在枯草里 发出悲鸣,没人在意宇宙中微弱的信号,恐惧开始蔓延,无数颗星星掉落",
                        FontStyle{
                            font_file_path:Arc::from("tf/LXGWWenKaiMono-Light.ttf"),
                            font_size:50,
                            font_line_height:0,
                            first_weight:150.0,
                            ..Default::default()
                        }
                    );
                })
                .finish()
        })
        .build()?;
    Ok(())
}
