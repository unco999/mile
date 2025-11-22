use bitflags::*;
use bytemuck::{Pod, Zeroable, bytes_of, bytes_of_mut};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    collections::HashMap,
    fmt::Debug,
    rc::Rc,
    time::{Duration, Instant},
};
use wgpu::{
    RenderPass,
    util::{BufferInitDescriptor, DeviceExt, DownloadBuffer},
};
use winit::dpi::PhysicalSize;

pub mod _ty {
    pub type LayerID = u32;

    #[derive(Clone, Copy, Debug)]
    pub struct PanelId(pub u32);

    impl Into<u32> for PanelId {
        fn into(self) -> u32 {
            self.0
        }
    }
}

bitflags! {
    /// Interaction Mask
    pub struct ModuleEventType: u32 {
        const DEFUALT = 1 << 0;
        const Push = 1 << 1;
        const PanelCustomRead = 1 << 3;
        const Vertex = 1 << 4;
        const Frag = 1 << 5;
    }
}

pub trait Renderable {
    fn render<'a>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_view: &wgpu::TextureView,
        pass: &mut RenderPass<'a>,
    );
    fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue);

    fn resize(
        &mut self,
        size: winit::dpi::PhysicalSize<u32>,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
    );
}

pub trait Computeable {
    fn encode(&self, pass: &mut wgpu::ComputePass<'_>);
    fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue);
    fn is_dirty(&self) -> bool;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct GpuDebugReadCallBack {
    pub floats: [f32; 32], // 16 * 4 = 64 字节
    pub uints: [u32; 32],  // 16 * 4 = 64 字节
}

pub struct GpuDebug {
    import_name: &'static str,
    structs: GpuDebugReadCallBack,
    pub buffer: Option<wgpu::Buffer>,
    last_print: Cell<Instant>, // 上一次打印时间
    print_interval: Duration,  // 最小间隔
}

impl GpuDebugReadCallBack {
    pub fn print(name: &'static str, data: &GpuDebugReadCallBack) {
        // let mut has_nonzero = false;

        // print!("GpuDebug:[{name}]: {{\n  floats: [");
        // for (i, &f) in data.floats.iter().enumerate() {
        //     if f != 0.0 {
        //         has_nonzero = true;
        //         print!("{}: {:.4}, ", i, f);
        //     }
        // }
        // println!("]");

        // print!("  uints: [");
        // for (i, &u) in data.uints.iter().enumerate() {
        //     if u != 0 {
        //         has_nonzero = true;
        //         print!("{}: {}, ", i, u);
        //     }
        // }
        // println!("]");

        // if !has_nonzero {
        //     println!("  全部为 0");
        // }

        // println!("}}");
    }
}

impl GpuDebug {
    pub fn new(name: &'static str) -> Self {
        Self {
            import_name: name,
            buffer: None,
            last_print: Cell::new(Instant::now()),
            print_interval: Duration::from_millis(1333),
            structs: GpuDebugReadCallBack::default(), // 每 200ms 打印一次
        }
    }

    pub fn create_buffer(&mut self, device: &wgpu::Device) {
        let out = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Shared State"),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST // CPU 写入
                | wgpu::BufferUsages::COPY_SRC,
            contents: bytemuck::bytes_of(&self.structs),
        });
        self.buffer = Some(out);
    }

    pub fn raw_debug(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let name = self.import_name;
        DownloadBuffer::read_buffer(
            device,
            queue,
            &self.buffer.as_ref().unwrap().slice(..),
            move |e| {
                if let Ok(downloadBuffer) = e {
                    let bytes = downloadBuffer;
                    let data: &[GpuDebugReadCallBack] = bytemuck::cast_slice(&bytes);
                    for data in data {
                        GpuDebugReadCallBack::print(name, data);
                    }
                }
            },
        );
    }

    pub fn check(&self) -> bool {
        if self.last_print.get().elapsed() < self.print_interval {
            self.last_print.set(Instant::now()); // 更新上次打印时间
            return false;
        }
        true
    }

    pub fn debug(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.last_print.get().elapsed() < self.print_interval {
            return; // 太快就跳过
        }

        let name = self.import_name;
        if(self.buffer.is_none()) {return;}
        DownloadBuffer::read_buffer(
            device,
            queue,
            &self.buffer.as_ref().unwrap().slice(..),
            move |e| {
                if let Ok(downloadBuffer) = e {
                    let bytes = downloadBuffer;
                    let data: &[GpuDebugReadCallBack] = bytemuck::cast_slice(&bytes);
                    for data in data {
                        GpuDebugReadCallBack::print(name, data);
                    }
                }
            },
        );

        self.last_print.set(Instant::now()); // 更新上次打印时间
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
pub struct GlobalUniform {
    // === 16-byte block 1: atomic z/layouts ===
    pub click_layout_z: u32,
    pub click_layout_id: u32,
    pub hover_layout_id: u32,
    pub hover_layout_z: u32, // 16 bytes

    // === 16-byte block 2: atomic drag ===
    pub drag_layout_id: u32,
    pub drag_layout_z: u32,
    pub pad_atomic1: u32,
    pub pad_atomic2: u32, // 16 bytes

    // === 16-byte block 3: delta / dt ===
    pub dt: f32,
    pub pad1: f32,
    pub pad2: f32,
    pub pad3: f32, // 16 bytes

    // === 16-byte block 4: mouse ===
    pub mouse_pos: [f32; 2],
    pub mouse_state: u32,
    pub frame: u32, // 16 bytes

    // === 16-byte block 5: screen info ===
    pub screen_size: [u32; 2],
    pub press_duration: f32,
    pub time: f32, // 16 bytes

    // === 16-byte block 6: event points ===
    pub event_point: [f32; 2],
    pub extra1: [f32; 2], // 16 bytes

    // === 16-byte block 7: extra data ===
    pub extra2: [f32; 2],
    pub pad_extra: [f32; 2], // 16 bytes
}

impl GlobalUniform {
    pub fn new() -> GlobalUniform {
        GlobalUniform {
            pad_atomic1: u32::MAX,
            ..Default::default()
        }
    }
}
pub struct CpuGlobalUniform {
    inner: Rc<RefCell<GlobalUniform>>,
    buffer: wgpu::Buffer,
}

impl CpuGlobalUniform {
    pub fn get_struct(&self) -> Rc<RefCell<GlobalUniform>> {
        return self.inner.clone();
    }

    pub fn get_buffer(&self) -> wgpu::Buffer {
        return self.buffer.clone();
    }

    pub fn borrow(&self) -> std::cell::Ref<'_, GlobalUniform> {
        self.inner.borrow()
    }

    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, GlobalUniform> {
        self.inner.borrow_mut()
    }

    pub fn write_field<T: Pod>(&self, queue: &wgpu::Queue, offset: wgpu::BufferAddress, value: &T) {
        queue.write_buffer(&self.buffer, offset, bytes_of(value));
        let mut guard = self.inner.borrow_mut();
        let cpu_bytes = bytes_of_mut(&mut *guard);
        let start = offset as usize;
        let end = start + bytes_of(value).len();
        cpu_bytes[start..end].copy_from_slice(bytes_of(value));
    }

    pub fn flush(&self, queue: &wgpu::Queue) {
        let snapshot = self.inner.borrow();
        queue.write_buffer(&self.buffer, 0, bytes_of(&*snapshot));
    }

    pub fn new(device: &wgpu::Device, window: &winit::window::Window) -> Self {
        let size = window.inner_size(); // 返回 PhysicalSize<u32>
        let width = size.width;
        let height = size.height;

        println!("目前的gpu w:{} h:{}", width, height);

        let mut global_uniform = GlobalUniform::new();
        global_uniform.screen_size = [width, height];

        let global_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GlobalUniformBuffer"),
            contents: bytemuck::bytes_of(&global_uniform),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            inner: Rc::new(RefCell::new(global_uniform)),
            buffer: global_buffer,
        }
    }
}

pub struct ImportRegistry {
    pub render_imports: std::collections::HashMap<String, (u32, ImportHandler)>,
    pub compute_imports: std::collections::HashMap<String, (u32, ImportHandler)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImportType {
    Render(&'static str),  // 渲染管线导入，如UV、屏幕坐标等
    Compute(&'static str), // 计算管线导入，如缓存数据等
}

#[derive(Debug, Clone)]
pub struct ImportInfo {
    pub import_type: ImportType,
    pub mask: u32,    // 比如 01 是uv, 11 是pos+uv
    pub index: usize, // 在V中的索引位置
}

pub type ImportHandler = Box<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>;

impl ImportRegistry {
    pub fn new() -> Self {
        Self {
            render_imports: HashMap::new(),
            compute_imports: HashMap::new(),
        }
    }

    // 注册渲染导入（如UV坐标）
    pub fn register_render_import(&mut self, name: &str, mask: u32, handler: ImportHandler) {
        self.render_imports
            .insert(name.to_string(), (mask, handler));
    }

    // 注册计算导入（如缓存数据）
    pub fn register_compute_import(&mut self, name: &str, mask: u32, handler: ImportHandler) {
        self.compute_imports
            .insert(name.to_string(), (mask, handler));
    }

    // 获取导入信息
    pub fn get_import_info(&self, name: &'static str) -> Option<(ImportType, u32)> {
        if let Some((mask, _)) = self.render_imports.get(name) {
            Some((ImportType::Render(name), *mask))
        } else if let Some((mask, _)) = self.compute_imports.get(name) {
            Some((ImportType::Compute(name), *mask))
        } else {
            None
        }
    }

    // 执行导入处理
    pub fn execute_import(&self, import_type: &ImportType, input: &[f32]) -> Vec<f32> {
        match import_type {
            ImportType::Render(name) => {
                if let Some((_, handler)) = self.render_imports.get(*name) {
                    handler(input)
                } else {
                    vec![0.0; input.len()]
                }
            }
            ImportType::Compute(name) => {
                if let Some((_, handler)) = self.compute_imports.get(*name) {
                    handler(input)
                } else {
                    vec![0.0; input.len()]
                }
            }
        }
    }
}
