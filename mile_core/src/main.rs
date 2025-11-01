use std::{cell::RefCell, collections::HashMap, sync::{Arc, Mutex}, time::{Duration, Instant}};

use mile_api::{GlobalEventHub, ModuleEvent, ModuleParmas};
use mile_font::structs::MileFont;
use mile_gpu_dsl::{core::Expr, pipeline::RenderPlan};
use mile_graphics::structs::{GlobalState, WGPUContext};
use mile_ui::{structs::{AnimOp, EasingMask, PanelField, PanelInteractionHold}, TransformAnimFieldInfo};
use winit::event_loop::{self, EventLoop};
use crate::structs::{App, AppEvent};
pub mod structs;



fn main() {
 // ✅ 正确：创建带用户事件的 EventLoop
    let mut event_loop = EventLoop::<AppEvent>::with_user_event();

    // ✅ 创建 proxy
    let event_loop_main = event_loop.build().unwrap();
    let proxy = event_loop_main.create_proxy();

    let global_state: Arc<Mutex<GlobalState>> = Arc::new(Mutex::new(GlobalState::new()));
    let gs = global_state.clone();


    let global_hub = Arc::new(GlobalEventHub::<ModuleEvent<ModuleParmas<Expr>,RenderPlan>>::new());
    

    
    // ✅ 初始化 app，并保存 proxy
    let mut app = App {
        global_hub,
        gpu_kennel:None,
        mile_font:None,
        proxy: Some(proxy.clone()),
        wgpu_context: None,
        wgpu_gpu_ui:None,
        global_state:global_state,
        last_frame_time:Instant::now(),
        last_tick:Instant::now(),
        delta_time:Duration::from_secs_f32(0.0),
        tick_interval: Duration::from_secs_f64(1.0 / 60.0),
    };

    // std::thread::spawn(move || {
    //     let frame_duration = std::time::Duration::from_secs_f32(1.0 / 60.0);
    //     let mut last_tick = std::time::Instant::now();

    //     loop {
    //         let now = std::time::Instant::now();
    //         if now.duration_since(last_tick) >= frame_duration {
    //             proxy.send_event(AppEvent::Tick).ok();
    //             last_tick += frame_duration; // 累积补偿，防止漂移
    //         } else {
    //             // 只 sleep 剩余时间，减轻 CPU
    //             std::thread::sleep(frame_duration - now.duration_since(last_tick));
    //         }
    //         // 线程可继续处理其他事情，非阻塞主线程
    //     }
    // });

    // ✅ 启动主事件循环
    event_loop_main.run_app(&mut app).unwrap();
}
