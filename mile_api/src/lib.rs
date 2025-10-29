use std::{cell::RefCell, default, rc::Rc, time::{Duration, Instant}};

use bytemuck::{Pod, Zeroable};
use wgpu::{util::{BufferInitDescriptor, DeviceExt, DownloadBuffer}, wgc::device::queue, RenderPass};

pub trait Renderable {
    fn render<'a>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_view: &wgpu::TextureView,
        pass: &mut RenderPass<'a>,
    );
    fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue);
}

pub trait Computeable {
    fn compute(&self, pass: &mut wgpu::ComputePass<'_>);
    fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue);
    fn is_upate(&self)->bool;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct GpuDebugReadCallBack {
    pub floats: [f32; 32], // 16 * 4 = 64 字节
    pub uints: [u32; 32],  // 16 * 4 = 64 字节
}

pub struct GpuDebug{
    import_name:&'static str,
    structs:GpuDebugReadCallBack,
    pub buffer:Option<wgpu::Buffer>,
    last_print: Instant,      // 上一次打印时间
    print_interval: Duration, // 最小间隔
}

impl GpuDebugReadCallBack {
    pub fn print(name:&'static str,data: &GpuDebugReadCallBack) {
        let mut has_nonzero = false;

        print!("GpuDebug:[{name}]: {{\n  floats: [");
        for (i, &f) in data.floats.iter().enumerate() {
            if f != 0.0 {
                has_nonzero = true;
                print!("{}: {:.4}, ", i, f);
            }
        }
        println!("]");

        print!("  uints: [");
        for (i, &u) in data.uints.iter().enumerate() {
            if u != 0{
                has_nonzero = true;
                print!("{}: {}, ", i, u);
            }
        }
        println!("]");

        if !has_nonzero {
            println!("  全部为 0");
        }

        println!("}}");
    }
}

impl GpuDebug {

    pub fn new(name:&'static str)->Self{
        Self{
            import_name:name,
            buffer: None,
            last_print: Instant::now(),
            print_interval: Duration::from_millis(1333),
            structs:GpuDebugReadCallBack::default(), // 每 200ms 打印一次
        }
    }

    pub fn create_buffer(&mut self,device:&wgpu::Device){
        let out = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Shared State"),
            usage: 
                 wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST // CPU 写入
                | wgpu::BufferUsages::COPY_SRC
            ,
            contents: bytemuck::bytes_of(&self.structs),
        });
        self.buffer = Some(
            out
        );
    }

    pub fn debug(&mut self,device:&wgpu::Device,queue:&wgpu::Queue) {
         if self.last_print.elapsed() < self.print_interval {
            return; // 太快就跳过
        }

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
                        GpuDebugReadCallBack::print(name,data);
                    }
                }
            },
        );

        self.last_print = Instant::now(); // 更新上次打印时间
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
    pub pad_atomic2: u32,    // 16 bytes

    // === 16-byte block 3: delta / dt ===
    pub dt: f32,
    pub pad1: f32,
    pub pad2: f32,
    pub pad3: f32,           // 16 bytes

    // === 16-byte block 4: mouse ===
    pub mouse_pos: [f32; 2],
    pub mouse_state: u32,
    pub frame: u32,          // 16 bytes

    // === 16-byte block 5: screen info ===
    pub screen_size: [u32; 2],
    pub press_duration: f32,
    pub time: f32,           // 16 bytes

    // === 16-byte block 6: event points ===
    pub event_point: [f32; 2],
    pub extra1: [f32; 2],    // 16 bytes

    // === 16-byte block 7: extra data ===
    pub extra2: [f32; 2],
    pub pad_extra: [f32; 2], // 16 bytes
}

pub struct CpuGlobalUniform{
    inner:Rc<RefCell<GlobalUniform>>,
    buffer:wgpu::Buffer
}

impl CpuGlobalUniform  {

    pub fn get_struct(&self)->Rc<RefCell<GlobalUniform>>{
        return self.inner.clone();
    }

    pub fn get_buffer(&self)->wgpu::Buffer{
        return self.buffer.clone()
    }

    pub fn new(        
        device: &wgpu::Device,
        window: &winit::window::Window,
    )->Self{
        let size = window.inner_size(); // 返回 PhysicalSize<u32>
        let width = size.width;
        let height = size.height;

        println!("目前的gpu w:{} h:{}", width, height);

        let mut global_uniform = GlobalUniform::default();
        global_uniform.screen_size = [width, height];

        let global_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GlobalUniformBuffer"),
            contents: bytemuck::bytes_of(&global_uniform),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        Self { inner: Rc::new(RefCell::new(global_uniform)) ,buffer:global_buffer}
    }
}
