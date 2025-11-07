use mile_api::prelude::{
    global_event_bus, Computeable, CpuGlobalUniform, GlobalUniform, Renderable,
};
use mile_font::{
    event::{BatchFontEntry, BatchRenderFont},
    structs::MileFont,
};
use mile_gpu_dsl::prelude::{
    gpu_ast_compute_pipeline::ComputePipelineConfig,
    kennel::{Kennel, KennelConfig},
};
use mile_graphics::structs::WGPUContext;
use mile_ui::{
    mui_prototype::build_demo_panel,
    prelude::*,
    runtime::{BufferArenaConfig, MuiRuntime},
};
use std::{
    cell::RefCell,
    mem::offset_of,
    rc::Rc,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use wgpu::SurfaceError;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, KeyEvent, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::{self, Window, WindowAttributes},
};
use winit::{
    event::MouseButton,
    event_loop::{ActiveEventLoop, ControlFlow},
};

use crate::GlobalState;

#[derive(Clone)]
pub struct App {
    pub wgpu_context: Option<WGPUContext>,
    pub mui_runtime: Option<Arc<RefCell<MuiRuntime>>>,
    pub mile_font: Option<Arc<RefCell<MileFont>>>,
    pub kennel: Option<Arc<RefCell<Kennel>>>,
    pub global_state: Arc<Mutex<GlobalState>>,
    pub last_tick: Instant,
    pub tick_interval: Duration,
    pub last_frame_time: Instant,
    pub delta_time: Duration,
    pub frame_index: u32,
}

impl App {
    pub fn new(global_state: Arc<Mutex<GlobalState>>) -> Self {
        Self {
            wgpu_context: None,
            mui_runtime: None,
            mile_font: None,
            kennel: None,
            global_state,
            last_tick: Instant::now(),
            tick_interval: Duration::from_secs_f64(1.0 / 60.0),
            last_frame_time: Instant::now(),
            delta_time: Duration::from_secs_f32(0.0),
            frame_index: 0,
        }
    }

    fn update_frame_time(&mut self) {
        let now = Instant::now();
        self.delta_time = now - self.last_frame_time;
        self.last_frame_time = now;
    }

    fn update_runtime(&mut self) {
        let (ctx, runtime_cell, mile_font) =
            match (&self.wgpu_context, &self.mui_runtime, &self.mile_font) {
                (Some(ctx), Some(runtime), Some(mile_font)) => (ctx, runtime, mile_font),
                _ => return,
            };

        let mut runtime = runtime_cell.borrow_mut();
        runtime.begin_frame(self.frame_index, self.delta_time.as_secs_f32());

        if !runtime.panel_cache.is_empty() {
            runtime.refresh_registered_payloads(&ctx.device, &ctx.queue);
            runtime.upload_panel_instances(&ctx.device, &ctx.queue);
        }

        let mut mile_font = mile_font.borrow_mut();
        mile_font.evnet_polling(&ctx.device, &ctx.queue);
        // runtime.copy_interaction_swap_frame();
        runtime.tick_frame_update_data(&ctx.queue);
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    fn app_first(&mut self) {
        let Some(ctx) = &self.wgpu_context else {
            return;
        };

        let mut mui_runtime = self.mui_runtime.as_ref().unwrap().borrow_mut();
        mui_runtime.mouse_press_tick_first(&ctx.queue);
        mui_runtime.copy_interaction_swap_frame(&ctx.device, &ctx.queue);
    }

    fn app_post(&mut self) {
        let Some(ctx) = &self.wgpu_context else {
            return;
        };

        let mut mui_runtime = self.mui_runtime.as_ref().unwrap().borrow_mut();
        mui_runtime.mouse_press_tick_post(&ctx.queue);
    }

    fn compute(&mut self) {
        let Some(ctx) = &self.wgpu_context else {
            return;
        };

        if let Some(font) = &self.mile_font {
            let mut font = font.borrow_mut();
            font.batch_enqueue_compute(&ctx.device, &ctx.queue);
            font.copy_store_texture_to_render_texture(&ctx.device, &ctx.queue);
        }

        if let Some(kennel) = &self.kennel {
            let mut kennel = kennel.borrow_mut();
            kennel.compute(&ctx.device, &ctx.queue);
            kennel.debug_readback(&ctx.device, &ctx.queue);
            kennel.process_global_events(&ctx.queue, &ctx.device);
        }

        let mut mui_runtime = self.mui_runtime.as_ref().unwrap();
        ctx.compute(&[&*mui_runtime.borrow()]);
    }

    fn event_first(&mut self) {
        let mut mui_runtime = self.mui_runtime.as_ref().unwrap().borrow_mut();
        mui_runtime.event_poll();
    }

    fn render(&mut self) {
        let (ctx, runtime_cell, mile_font) =
            match (&self.wgpu_context, &self.mui_runtime, &self.mile_font) {
                (Some(ctx), Some(runtime), Some(mile_font)) => (ctx, runtime, mile_font),
                _ => return,
            };

        let runtime = runtime_cell.borrow();
        let mile_font = mile_font.borrow();
        ctx.render(&[&*runtime, &*mile_font]);
    }

    fn build_resources(&mut self) {
        let Some(ctx) = &self.wgpu_context else {
            return;
        };

        if let Some(font) = &self.mile_font {
            let mut font = font.borrow_mut();
            font.load_font_file();
            font.init_buffer(&ctx.device);
            font.create_template_render_texture_and_layout(&ctx.device, None);
            font.create_render_pipeline(&ctx.device, ctx.config.format);
            font.create_batch_enqueue_font_compute_cahce(&ctx.device);
        }

        if let Some(runtime_cell) = &self.mui_runtime {
            let mut runtime = runtime_cell.borrow_mut();
            runtime.read_all_texture();
            runtime.rebuild_texture_bindings(&ctx.device, &ctx.queue);
            runtime.ensure_render_pipeline(&ctx.device, &ctx.queue, ctx.config.format);
        }
    }
}

fn cursor_to_normalized(window: &Window, position: PhysicalPosition<f64>) -> [f32; 2] {
    let size = window.inner_size();
    let x = position.x as f32 / size.width as f32;
    let y = position.y as f32 / size.height as f32;
    [x * 2.0 - 1.0, 1.0 - y * 2.0]
}

#[derive(Debug, Clone)]
pub enum AppEvent {
    Tick,
}

impl ApplicationHandler<AppEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(WindowAttributes::default())
            .expect("failed to create window");

        let window = Arc::new(window);
        let ctx = WGPUContext::new(window.clone(), Arc::clone(&self.global_state));
        let global_uniform = Rc::new(CpuGlobalUniform::new(&ctx.device, &window));

        let kennel = Arc::new(RefCell::new(Kennel::new(
            &ctx.device,
            &ctx.queue,
            Rc::clone(&global_uniform),
            KennelConfig {
                compute_config: ComputePipelineConfig {
                    max_nodes: 512,
                    max_imports: 32,
                    workgroup_size: (8, 8, 1),
                },
                readback_interval_secs: 2,
            },
        )));

        let runtime = MuiRuntime::new(
            &ctx.device,
            BufferArenaConfig {
                max_panels: 1024,
                max_animation_fields: 512,
                max_collections: 128,
                max_relations: 128,
            },
        );

        self.wgpu_context = Some(ctx.clone());
        self.mui_runtime = Some(Arc::new(RefCell::new(runtime)));
        self.mile_font = Some(Arc::new(RefCell::new(MileFont::new(global_uniform))));
        self.kennel = Some(kennel);

        self.build_resources();
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppEvent) {
        if matches!(event, AppEvent::Tick) {
            self.update_frame_time();
            self.update_runtime();
            self.compute();
            self.render();
            self.app_post();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        if now - self.last_tick >= self.tick_interval {
            self.last_tick = now;
            self.app_first();
            self.update_frame_time();
            self.update_runtime();
            self.compute();
            self.event_first();
            self.render();
            self.app_post();
            event_loop.set_control_flow(ControlFlow::Poll);
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(now + self.tick_interval));
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => std::process::exit(0),
            WindowEvent::CursorMoved { position, .. } => {
                let ctx = &self.wgpu_context.as_ref().expect("没有渲染上下文");
                if let Some(mui_runtime) = &self.mui_runtime {
                    let mui_runtime = mui_runtime.borrow_mut();
                    mui_runtime.write_global_buffer(
                        &ctx.queue,
                        offset_of!(GlobalUniform, mouse_pos) as u64,
                        [position.x as f32, position.y as f32],
                    )
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(ctx) = self.wgpu_context.as_mut() {
                    ctx.resize(size);
                    if let Some(runtime_cell) = &self.mui_runtime {
                        let mut runtime = runtime_cell.borrow_mut();
                        runtime.resize(size, &ctx.queue, &ctx.device);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let (Some(ctx), Some(runtime_cell)) = (&self.wgpu_context, &self.mui_runtime) {
                    let mut runtime = runtime_cell.borrow_mut();
                    match (button, state) {
                        (MouseButton::Left, ElementState::Pressed) => {
                            runtime.update_mouse_state(&ctx.queue, MouseState::LEFT_DOWN);
                        }
                        (MouseButton::Left, ElementState::Released) => {
                            runtime.update_mouse_state(&ctx.queue, MouseState::LEFT_UP);
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if matches!(event.state, ElementState::Pressed)
                    && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Space))
                {
                    global_event_bus().publish(BatchFontEntry {
                        str: "道",
                        font_file_path: "../ttf/BIZUDPGothic-Regular.ttf",
                    });
                    // global_event_bus().publish(BatchRenderFont {
                    //     str: "道",
                    //     font_file_path: "../ttf/BIZUDPGothic-Regular.ttf",
                    // });
                }
                if matches!(event.state, ElementState)
                    && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Enter))
                {
                    if let Some(runtime_cell) = &self.mui_runtime {
                        if runtime_cell.borrow().panel_instances.is_empty() {
                            if build_demo_panel().is_ok() {
                                let ctx = self.wgpu_context.as_ref().unwrap();
                                let mut runtime = runtime_cell.borrow_mut();
                                runtime.refresh_registered_payloads(&ctx.device, &ctx.queue);
                                runtime.upload_panel_instances(&ctx.device, &ctx.queue);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
