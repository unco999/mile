use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;
use std::{borrow::Cow, sync::Arc};

use mile_api::{Computeable, Renderable};
use pollster::block_on;
use wgpu::{BindGroup, Buffer, RenderPass, VertexStepMode::*};
use wgpu::util::DeviceExt;
use wgpu::{ShaderSource, Surface, util::BufferInitDescriptor};
use winit::{
    event_loop::{self, EventLoop},
    window::Window,
};
#[allow(dead_code)]
#[derive(Clone)]
pub struct WGPUContext {
    pub window: Arc<Window>,
    pub surface: Arc<Surface<'static>>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub config: Arc<wgpu::SurfaceConfiguration>,
    pub render_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub app_state:AppState,
    pub global_uniform_buffer:Buffer,
    pub bindgroup:BindGroup,
    pub global_state:Arc<Mutex<GlobalState>>
}

#[derive(Debug)]
pub struct GlobalState {
    pub flags: HashMap<GlobalStateType, GlobalStateRecord>, // 标识 -> 累计帧数
}
#[derive(PartialEq, PartialOrd,Eq,Hash,Debug)]
pub enum GlobalStateType{
    ComputeTickDuration,
    RenderTickDuration,
}

#[derive(Clone,Debug)]
pub enum GlobalStateRecord{
    TickFrame(u32),
    TickDuration(f32)
}

impl GlobalState {
    pub fn new() -> Self {
        Self { flags: HashMap::new() }
    }

    pub fn tick(&mut self) {
    // 每帧累加计数并清理超过 60 的标识
    println!("{:?}",self.flags);

    self.flags.retain(|_, record| {
        let bool = match record{
            GlobalStateRecord::TickFrame(frame) => {
                if(*frame == 0){
                    return true;
                }else {
                    *frame -= 1;
                    return false;
                }

            },
            GlobalStateRecord::TickDuration(time) => {
                *time -= 0.016;
                if(*time == 0.0){
                    return true;
                }else {
                    return false;
                }
            },
        };
    });
}

    /// 检查是否超过最大值，超过就插入新值
    pub fn check_and_insert(&mut self, key: GlobalStateType, record: GlobalStateRecord) {
        let need_insert = match self.flags.get(&key) {
            Some(existing) => match (existing, &record) {
                (GlobalStateRecord::TickFrame(old), GlobalStateRecord::TickFrame(new)) => *old  > *new,
                (GlobalStateRecord::TickDuration(old), GlobalStateRecord::TickDuration(new)) => *old > *new,
                _ => false, // 类型不匹配就不插入
            },
            None => true, // 不存在就插入
        };

        if need_insert {
            self.set_flag(key, record);
        }
    }

    pub fn set_flag(&mut self, key:GlobalStateType,val:GlobalStateRecord) {
        self.flags.insert(key, val); // 新标识从 0 帧开始计数
    }
}

/// Vertex data structure for use in vertex buffers
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BackGroundUnitform {
    pub time:f32
}


#[derive(Clone)]

struct AppState {
    start_time: Instant,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    pub fn update_time(&self) -> f32 {
        // 返回从 start_time 开始到现在的秒数
        self.start_time.elapsed().as_secs_f32()
    }
}

pub const VERTEX_LIST: &[Vertex] = &[
    Vertex {
        position: [0.0, 1.0, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.0, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];

impl WGPUContext {
    pub fn new(window: Arc<Window>,global_state:Arc<Mutex<GlobalState>>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = Arc::new(instance.create_surface(Arc::clone(&window)).unwrap());

        // 使用 pollster 阻塞等待异步函数完成
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("Failed to find an appropriate adapter");

        
        let limits = adapter.limits();
        
        println!("Max sampled textures per stage: {}", limits.max_sampled_textures_per_shader_stage);

        let required_features = wgpu::Features::TEXTURE_BINDING_ARRAY | wgpu::Features::VERTEX_WRITABLE_STORAGE | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: required_features, // <--- 允许
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("Failed to create device");

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let surface_config = surface
            .get_default_config(&adapter, width, height)
            .expect("Failed to get default surface config");


        surface.configure(&device, &surface_config);

        // 如果有 pipeline 等初始化，也可以在这里同步创建
        let (render_pipeline, bind_group_layout, bindgroup,buffer) = WGPUContext::create_pipeline(&device, surface_config.format);

        let bytes: &[u8] = bytemuck::cast_slice(&VERTEX_LIST);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            window,
            surface,
            device: Arc::new(device),
            queue: Arc::new(queue),
            config: Arc::new(surface_config),
            render_pipeline,
            vertex_buffer,
            app_state:AppState{ start_time:Instant::now() },
            global_uniform_buffer:buffer,
            bindgroup,
            global_state
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        let width = new_size.width.max(1);
        let height = new_size.height.max(1);
        let mut config = Arc::make_mut(&mut self.config);
        config.width = width;
        config.height = height;
        self.surface.configure(&self.device, &config);
    }


    pub fn compute(&self, computeables: &[&dyn Computeable]) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Command Encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hover Compute Pass"),
                ..Default::default()
            });
            for c in computeables {
                c.compute(&mut cpass);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        for c in computeables {
            c.readback(&self.device, &self.queue);
        }
    }


   
    pub fn render(&self,renderables: &[&dyn Renderable])->&Self {
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => return self,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        ..Default::default()
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            let time_sec = self.app_state.update_time();
            self.queue.write_buffer(
                &self.global_uniform_buffer,
                0,
                bytemuck::cast_slice(&[time_sec]),
            );

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.bindgroup, &[]); // <-- 绑定 uniform
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.draw(0..VERTEX_LIST.len() as u32, 0..1);
            for r in renderables {
                r.render(&self.device,&self.queue,&view,&mut rpass);
                r.readback(&self.device, &self.queue);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        self
    }

    fn create_pipeline(
        device: &wgpu::Device,
        swap_chain_format: wgpu::TextureFormat,
    ) -> ((wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::BindGroup,wgpu::Buffer)) {
        // Load the shaders from disk
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

         // Uniform layout
    let uniform_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(std::num::NonZeroU64::new(
                        std::mem::size_of::<BackGroundUnitform>() as u64,
                    ).unwrap()),
                },
                count: None,
            }],
        });

    // Uniform buffer
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Background Uniform Buffer"),
        contents: bytemuck::cast_slice(&[BackGroundUnitform { time: 0.0 }]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Uniform Bind Group"),
        layout: &uniform_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    // Pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Background Pipeline Layout"),
        bind_group_layouts: &[&uniform_bind_group_layout],
        push_constant_ranges: &[],
    });


        let pipeline =device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Self::create_vertex_buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: swap_chain_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        (pipeline, uniform_bind_group_layout, uniform_bind_group,uniform_buffer)
    }

    fn create_vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    // 这里的偏移，是要偏移position的字节长度
                    offset: size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1, // 我们把颜色信息数据指定为location = 1的地方
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}
