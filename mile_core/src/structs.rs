use std::{
    cell::RefCell,
    fs,
    path::Path,
    rc::Rc,
    slice::Windows,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use mile_api::{
    Computeable, CpuGlobalUniform, GlobalEventHub, GpuDebug, ModuleEvent, ModuleParmas, Renderable,
};
use mile_font::structs::MileFont;
use mile_gpu_dsl::{
    core::{
        Expr,
        dsl::{var, wvec4},
    },
    dsl::cv,
    mat::{
        gpu_ast_compute_pipeline::ComputePipelineConfig,
        kennel::{Kennel, KennelConfig},
        op::ImportRegistry,
    },
};
use mile_graphics::structs::WGPUContext;
use mile_ui::{
    GpuUi, Panel,
    mile_ui_wgsl::mile_test,
    structs::{AnimOp, MouseState, PanelEvent, PanelField, PanelInteraction},
};
use rand::{Rng, rng};
use wgpu::BindGroupLayout;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::EventLoopProxy,
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
    pub global_hub: Arc<GlobalEventHub<ModuleEvent<ModuleParmas<Expr>, u32>>>,
    pub wgpu_context: Option<WGPUContext>,
    pub wgpu_gpu_ui: Option<Arc<RefCell<GpuUi>>>,
    pub mile_font: Option<Arc<RefCell<MileFont>>>,
    pub proxy: Option<EventLoopProxy<AppEvent>>,
    pub kennel: Option<Arc<RefCell<Kennel>>>,
    pub global_state: Arc<Mutex<GlobalState>>,
    pub last_tick: Instant,
    pub tick_interval: Duration,
    pub last_frame_time: Instant,
    pub delta_time: Duration,
}

impl App {
    pub fn secs_tick(&mut self) {}

    pub fn render(&self) {
        if let (Some(ui_cell), Some(mile_font)) = (&self.wgpu_gpu_ui, &self.mile_font) {
            let ui_ref = ui_cell.borrow();
            let mile_font_ref = mile_font.borrow();

            let renderables: Vec<&dyn Renderable> = vec![&*ui_ref, &*mile_font_ref];

            self.wgpu_context.as_ref().unwrap().render(&renderables);
        } // ui_ref 和 mile_font_ref 生命周期足够
    }

    pub fn compute(&self) {
        if let (Some(ui_cell), Some(mile_font), Some(kennel)) =
            (&self.wgpu_gpu_ui, &self.mile_font, &self.kennel)
        {
            let ctx = self.wgpu_context.as_ref().unwrap();
            ui_cell.borrow_mut().mouse_press_tick_first(&ctx.queue);
            ui_cell
                .borrow_mut()
                .interaction_compute(&ctx.device, &ctx.queue);
            ui_cell
                .borrow_mut()
                .net_work_compute(&ctx.device, &ctx.queue);
            ui_cell
                .borrow_mut()
                .animtion_compute(&ctx.device, &ctx.queue);
            ui_cell.borrow_mut().mouse_press_tick_post(&ctx.queue);
            mile_font
                .borrow()
                .batch_enqueue_compute(&ctx.device, &ctx.queue);
            mile_font
                .borrow_mut()
                .copy_store_texture_to_render_texture(&ctx.device, &ctx.queue);
            kennel.borrow_mut().compute(&ctx.device, &ctx.queue);
            kennel.borrow_mut().compute(&ctx.device, &ctx.queue);
        }
    }

    pub fn tick(&self) {
        let ctx = self.wgpu_context.as_ref().unwrap();

        if let Some(ui_cell) = &self.wgpu_gpu_ui {
            let mut ui_cell = ui_cell.borrow_mut();
            ui_cell.process_global_events(&ctx.queue, &ctx.device);
            ui_cell.process_ui_events(&self.wgpu_context.as_ref().unwrap().queue, &ctx.device);
            ui_cell.update_frame(&ctx.queue, &ctx.device);
            ui_cell.update_network_dirty_entries(&ctx.queue, &ctx.device);
            ui_cell.update_global_unifrom_time(&ctx.queue, self.delta_time.as_secs_f32());
            ui_cell.global_unifrom_clear_tick(&ctx.queue);
            ui_cell.update_dt(self.delta_time.as_secs_f32(), &ctx.queue);
            // ui_cell.borrow_mut().readback(&ctx.device,&ctx.queue);
        }

        if let Some(kennel) = &self.kennel {
            let mut kennel = kennel.borrow_mut();
            kennel.compute(&ctx.device, &ctx.queue);
            kennel.debug_readback(&ctx.device, &ctx.queue);
            kennel.process_global_events(&ctx.queue, &ctx.device);
            // kennel.read_call_back_cpu(&ctx.device,&ctx.queue);
            // kennel.process_ui_events(&ctx.device,&ctx.queue);
        }
    }

    fn update_frame_time(&mut self) {
        let now = Instant::now();
        self.delta_time = now - self.last_frame_time;
        self.last_frame_time = now;
    }

    pub fn ui_build(&self) {
        let ctx = self.wgpu_context.as_ref().unwrap();

        if let Some(mile_font_arc) = &self.mile_font {
            {
                let mut mile_font = mile_font_arc.borrow_mut();
                mile_font.init_buffer(&ctx.device);
                mile_font.create_template_render_texture_and_layout(&ctx.device, None);
                mile_font.create_render_pipeline(&ctx.device, ctx.config.format);
                mile_font.create_batch_enqueue_font_compute_cahce(&ctx.device);
            }
        }

        if let Some(ui_cell) = &self.wgpu_gpu_ui {
            {
                let mut ui = ui_cell.borrow_mut();
                ui.read_all_image();
                ui.create_render_bind_layout(&ctx.device, &ctx.queue);
                ui.create_texture_bind_layout(&ctx.device, &ctx.queue);
                ui.create_render_pipeline(&ctx.device, &ctx.queue, ctx.config.format);
                ui.create_animtion_compute_pipeline(&ctx.device, &ctx.queue);
                ui.create_net_work_compute_pipeline(&ctx.device, &ctx.queue);
                ui.create_interaction_compute_pipeline(&ctx.device);
                ui.createa_animtion_compute_pipeline_two(&ctx.device);
            }
            // ui_cell.borrow_mut().test_gpu_collection(device,queue);
            // ui_cell.borrow_mut().create_compute_pipeline(device, queue);
        }

        if let Some(ui_cell) = &self.wgpu_gpu_ui {
            mile_test(ui_cell.clone(), &ctx.queue, &ctx.device);

            // ui_cell.borrow_mut().test_gpu_collection(device,queue);
            // ui_cell.borrow_mut().create_compute_pipeline(device, queue);
        }
    }

    pub fn get_path_by_index(&self, index: u32) -> Option<String> {
        if let Some(ui_cell) = &self.wgpu_gpu_ui {
            let gpu_ui = ui_cell.borrow();
            let output = gpu_ui.ui_texture_map.get_path_by_index(index);
            println!("当前0号位的texture地址 {:?}", output);
        }
        None
    }

    pub fn read_all_image(&mut self) {
        // 遍历 ./texture 目录
        let texture_dir = Path::new("./texture");
        if !texture_dir.exists() {
            eprintln!("纹理目录 {:?} 不存在", texture_dir);
            return;
        }

        // 收集所有支持的图片文件
        let supported_ext = ["png", "jpg", "jpeg", "bmp"];

        let mut image_paths = Vec::new();
        if let Ok(entries) = fs::read_dir(texture_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if supported_ext.contains(&ext.to_lowercase().as_str()) {
                        image_paths.push(path);
                    }
                }
            }
        }

        if image_paths.is_empty() {
            println!("未找到任何纹理文件");
            return;
        }

        // 逐个调用 gpu_ui.read_img
        if let Some(ui_cell) = &self.wgpu_gpu_ui {
            let mut gpu_ui = ui_cell.borrow_mut();
            for path in image_paths {
                println!("读取纹理文件: {:?}", path);
                gpu_ui.read_img(path.as_path());
            }
        } else {
            eprintln!("wgpu_gpu_ui 尚未初始化");
        }
    }

    // pub fn load_get_texture_view_bind_layout(&mut self,queue:&wgpu::Queue,device:&wgpu::Device,bind_group_layout:BindGroupLayout){
    //     if let Some(ui_cell) = &self.wgpu_gpu_ui {
    //         let mut cell = ui_cell.borrow_mut();

    //         // 上传所有纹理
    //         for (_, ui_texture_atlas) in cell.ui_texture_map.data.iter_mut() {
    //             ui_texture_atlas.upload_to_gpu(device, queue);
    //         }
    //         // 🧱 2️⃣ 收集所有 texture views 和 samplers
    //         let mut texture_views = vec![];
    //         let mut samplers = vec![];

    //         for (_, atlas) in cell.ui_texture_map.data.iter() {
    //             if let (Some(view), Some(sampler)) = (&atlas.texture_view, &atlas.sampler) {
    //                 texture_views.push(view);
    //                 samplers.push(sampler);
    //             }
    //         }

    //         // 🧱 3️⃣ 生成 bind group entries
    //         // 每个纹理各有两个 binding（view + sampler）
    //         let mut entries = Vec::new();
    //         for (i, (view, sampler)) in texture_views.iter().zip(samplers.iter()).enumerate() {
    //             entries.push(wgpu::BindGroupEntry {
    //                 binding: (i * 2) as u32,
    //                 resource: wgpu::BindingResource::TextureView(view),
    //             });
    //             entries.push(wgpu::BindGroupEntry {
    //                 binding: (i * 2 + 1) as u32,
    //                 resource: wgpu::BindingResource::Sampler(sampler),
    //             });
    //         }

    //         // 🧱 4️⃣ 创建 BindGroup
    //         let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    //             label: Some("UI Texture Array BindGroup"),
    //             layout: &bind_group_layout,
    //             entries: &entries,
    //         });

    //         // 🧱 5️⃣ 保存结果到 GPU UI 结构中
    //         cell.ui_texture_bind_group = Some(bind_group);
    //     }
    // }
}

fn cursor_to_normalized(window: &Window, position: PhysicalPosition<f64>) -> [f32; 2] {
    let size = window.inner_size();
    let x = position.x as f32 / size.width as f32; // 0~1
    let y = position.y as f32 / size.height as f32; // 0~1

    // 如果 GPU shader 使用 NDC [-1,1] 坐标，可以做转换
    let ndc_x = x * 2.0 - 1.0;
    let ndc_y = 1.0 - y * 2.0; // y 反向
    [ndc_x, ndc_y]
}

#[derive(Debug, Clone)]
pub enum AppEvent {
    Tick,
}

impl ApplicationHandler<AppEvent> for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let attribute = WindowAttributes::default();
        let window = event_loop
            .create_window(attribute)
            .expect("Failed to create window");

        let window = Arc::new(window);
        let ctx = WGPUContext::new(window.clone(), self.global_state.clone());
        let global_unifrom = Rc::new(CpuGlobalUniform::new(&ctx.device, &window));

        let global_hub = Arc::new(GlobalEventHub::new());

        let kennel = Arc::new(RefCell::new(Kennel::new(
            &ctx.device,
            &ctx.queue,
            global_unifrom.clone(),
            KennelConfig {
                compute_config: ComputePipelineConfig {
                    max_nodes: 512,
                    max_imports: 32,
                    workgroup_size: (8, 8, 1),
                },
                readback_interval_secs: 2,
            },
            global_hub.clone(),
        )));

        let mut gpu_ui = GpuUi::new(
            &ctx.device,
            ctx.config.format,
            self.global_state.clone(),
            global_unifrom.clone(),
            &window,
            global_hub.clone(),
            kennel.clone(),
        );

        self.wgpu_context = Some(ctx.clone());
        self.wgpu_gpu_ui = Some(Arc::new(RefCell::new(gpu_ui)));
        self.mile_font = Some(Arc::new(RefCell::new(MileFont::new(
            global_unifrom.clone(),
        ))));
        self.ui_build();
        self.kennel = Some(kennel)

        // self.read_all_image();
        // self.load_get_texture_view_bind_layout(&ctx.queue, &ctx.device,bind_group_layout);
        // self.get_path_by_index(0);
    }

    fn user_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        app_event: AppEvent,
    ) {
        // match app_event {
        //     AppEvent::Tick => {
        //         // let renderables: &[&dyn Renderable] =
        //         //     &[&self.wgpu_gpu_ui.as_ref().unwrap().clone()];
        //         // if let Some(ui_cell) = &self.wgpu_gpu_ui {
        //         //     let mut cell = ui_cell.borrow_mut();
        //         //     cell.print_mouse_state();
        //         // }
        //         self.compute();
        //         self.tick();
        //         self.render();
        //     }
        // }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();

        if now - self.last_tick >= self.tick_interval {
            self.last_tick = now;
            self.update_frame_time();
            // 直接在这里调用 Tick 逻辑
            self.tick();
            self.compute();
            self.render();

            // 告诉 event_loop 立即再轮询，不阻塞
            event_loop.set_control_flow(ControlFlow::Poll);
        } else {
            // 等待下一个事件（阻塞模式）
            event_loop.set_control_flow(ControlFlow::WaitUntil(now + self.tick_interval));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("窗口关闭请求");
                std::process::exit(0); // 退出程序
            }
            WindowEvent::Resized(new_size) => {
                self.wgpu_context.as_ref().unwrap().clone().resize(new_size);
                if let Some(ui_cell) = &self.wgpu_gpu_ui {
                    let ctx = self.wgpu_context.as_ref().unwrap();
                    ui_cell.borrow_mut().window_resized(
                        new_size.width,
                        new_size.height,
                        &ctx.queue,
                    );
                }
            }
            WindowEvent::RedrawRequested => {}
            WindowEvent::CursorMoved {
                device_id,
                position,
            } => {
                if let Some(ui_cell) = &self.wgpu_gpu_ui {
                    let mut cell = ui_cell.borrow_mut();
                    let ctx = self.wgpu_context.as_ref().unwrap().clone();
                    let cursor_postion = cursor_to_normalized(&ctx.window, position);
                    cell.update_mouse_pos(&ctx.queue, cursor_postion);
                }
                //    self.compute();
            }

            WindowEvent::MouseInput {
                device_id: _,
                state,
                button,
            } => {
                if let Some(ui_cell) = &self.wgpu_gpu_ui {
                    let mut gpu_ui = ui_cell.borrow_mut(); // RefMut<GpuUi>
                    let ctx = self.wgpu_context.as_ref().unwrap();
                    match (button, state) {
                        (MouseButton::Left, ElementState::Pressed) => {
                            gpu_ui.update_mouse_state(&ctx.queue, MouseState::LEFT_DOWN);
                        }
                        (MouseButton::Left, ElementState::Released) => {
                            gpu_ui.update_mouse_state(&ctx.queue, MouseState::LEFT_UP);
                        }
                        (MouseButton::Right, ElementState::Pressed) => {
                            gpu_ui.update_mouse_state(&ctx.queue, MouseState::RIGHT_DOWN);
                        }
                        (MouseButton::Right, ElementState::Released) => {
                            gpu_ui.update_mouse_state(&ctx.queue, MouseState::RIGHT_UP);
                        }
                        _ => {}
                    }
                }
                //    self.compute();
            }
            WindowEvent::KeyboardInput {
                device_id: _, // 这里暂时不关心
                event:
                    KeyEvent {
                        physical_key,
                        state,
                        repeat,
                        ..
                    },
                is_synthetic: _, // 忽略是否合成事件
            } => {
                if !repeat && state == ElementState::Pressed {
                    match physical_key {
                        PhysicalKey::Code(KeyCode::Backspace) => {
                            if let Some(ui_cell) = &self.wgpu_gpu_ui {
                                let ctx = self.wgpu_context.as_ref().unwrap();

                                //  let anim = TransformAnim {
                                //     field_id: (PanelField::POSITION_X | PanelField::POSITION_Y).bits(),
                                //     field_len: 1,
                                //     start_value: 0.0,
                                //     end_value: 50.0,
                                //     easing_mask: 1,
                                //     _pad1: 0,
                                //     duration: 0.33,
                                //     elapsed: 0.0,
                                //     instance_id: 0,
                                //     op: AnimOp::ADD.bits(),
                                //     _pad2:0,
                                //     _pad3:0,
                                //     last_applied:0.0,
                                //     _pad4: [0u32;3],
                                // };
                                // ui_cell.borrow_mut().add_animation(&ctx.queue, anim);
                            }
                        }
                        PhysicalKey::Code(KeyCode::Space) => {
                            let ctx = self.wgpu_context.as_ref().unwrap();

                            if let Some(mile_font) = &self.mile_font {
                                mile_font.borrow_mut().test_entry(&ctx.queue);
                                mile_font.borrow_mut().test_entry_text(&ctx.queue);
                            }

                            // if let Some(kennel) = &self.kennel{
                            //     let mut kennel = kennel.borrow_mut();

                            // }
                        }
                        PhysicalKey::Code(KeyCode::Escape) => {
                            println!("按下 Esc，退出程序");
                            std::process::exit(0);
                        }
                        _ => {}
                    }
                }
            }

            _ => {}
        }
    }
}
