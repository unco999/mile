use bitflags::*;
use bytemuck::{Pod, Zeroable};
use flume::{Receiver, Sender};
use std::{
    cell::RefCell,
    collections::HashMap,
    default,
    fmt::Debug,
    hash::{DefaultHasher, Hash, Hasher},
    rc::Rc,
    time::{Duration, Instant},
};
use wgpu::{
    RenderPass,
    util::{BufferInitDescriptor, DeviceExt, DownloadBuffer},
    wgc::device::queue,
};

use lazy_static::lazy_static;

// 使用 lazy_static 创建一个全局的 HashMap
lazy_static! {
    static ref GLOBAL_VARIABLES: Mutex<HashMap<u32, String>> = Mutex::new(HashMap::new());
}

fn get_variable_hash(name: &str) -> u32 {
    // 计算字符串的哈希值
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish() as u32
}

fn store_variable(name: &str, value: &str) {
    let hash = get_variable_hash(name);
    let mut globals = GLOBAL_VARIABLES.lock().unwrap();
    globals.insert(hash, value.to_string());
}

fn get_variable(hash: u32) -> Option<String> {
    let globals = GLOBAL_VARIABLES.lock().unwrap();
    globals.get(&hash).cloned()
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
}

pub trait Computeable {
    fn compute(&self, pass: &mut wgpu::ComputePass<'_>);
    fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue);
    fn is_upate(&self) -> bool;
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
    last_print: Instant,      // 上一次打印时间
    print_interval: Duration, // 最小间隔
}

pub struct Tick {
    interval: Duration, // 设定的间隔时间
    last_tick: Instant, // 上次 tick 的时间
}

impl Tick {
    // 创建一个新的 Tick，设置间隔时间
    pub fn new(interval_seconds: u64) -> Self {
        Tick {
            interval: Duration::from_secs(interval_seconds),
            last_tick: Instant::now(),
        }
    }

    // 检查当前时间是否超过设定的间隔，若是，返回 true，并重置计时器
    pub fn tick(&mut self) -> bool {
        if self.last_tick.elapsed() >= self.interval {
            self.last_tick = Instant::now(); // 重置计时器
            return true;
        }
        false
    }
}

impl GpuDebugReadCallBack {
    pub fn print(name: &'static str, data: &GpuDebugReadCallBack) {
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
            if u != 0 {
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
    pub fn new(name: &'static str) -> Self {
        Self {
            import_name: name,
            buffer: None,
            last_print: Instant::now(),
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

    pub fn debug(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
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
                        GpuDebugReadCallBack::print(name, data);
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

    pub fn new(device: &wgpu::Device, window: &winit::window::Window) -> Self {
        let size = window.inner_size(); // 返回 PhysicalSize<u32>
        let width = size.width;
        let height = size.height;

        println!("目前的gpu w:{} h:{}", width, height);

        let mut global_uniform = GlobalUniform::default();
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

/**
 * 这里是一个模块使用的 des的集合情况
 * 用于响应事件时 根据id或者其他索引来获取计算结果值所用的存储结构
 */
#[derive(Debug, Clone, Default)]
pub struct KennelReadDesPool<T: Hash + Clone + Eq> {
    pool: HashMap<T, MileResultDes>,
}

impl<T: Hash + Clone + Eq> KennelReadDesPool<T> {
    pub fn get_des(&mut self, key: T) -> Option<[u32; 4]> {
        self.pool.get(&key).map(|e| e.row_start)
    }

    pub fn insert(&mut self, k: T, v: MileResultDes) {
        self.pool.insert(k, v);
    }
}

/**
 * instance_id 来获取MileSimpleGPU里面的计算结果 共给vertex 和 frag 使用
 */
#[derive(Debug, Clone)]
pub struct MileResultDes {
    pub row_start: [u32; 4],
}

use std::sync::{Arc, Condvar, Mutex};

// 假设 ModuleEvent 和 Expr 是你定义的类型
#[derive(Clone, Debug)]
pub enum ModuleEvent<T, T1> {
    KennelPush(ModuleParmas<T>), //向Kennel写入处理事件
    KennelPushResultReadDes(ModuleParmas<T1>),
}

#[derive(Clone, Debug)]
pub struct ModuleParmas<T> {
    pub module_name: &'static str,
    pub idx: u32,
    pub data: T,
    pub _ty: u32,
}

pub type LayerID = u32;

pub struct GlobalEventHub<T: Debug + Clone> {
    pub sender: Sender<T>,
    receiver: Receiver<T>,
    pre_hover_panel_id: Option<u32>,
    event_condvar: Arc<(Mutex<bool>, Condvar)>, // 用于通知有新事件
}

impl<T: Debug + Clone> GlobalEventHub<T> {
    // 创建一个新的 GlobalEventHub 实例
    pub fn new() -> Self {
        let (sender, receiver) = flume::unbounded();
        let event_condvar = Arc::new((Mutex::new(false), Condvar::new())); // 初始状态为无事件

        Self {
            sender,
            receiver,
            pre_hover_panel_id: None,
            event_condvar,
        }
    }

    // 将事件推送到发送队列
    pub fn push(&self, event: T) {
        let _ = self.sender.send(event);
        // 发送事件后通知等待的线程
        let (lock, cvar) = &*self.event_condvar;
        let mut event_available = lock.lock().unwrap();
        *event_available = true; // 新事件到达
        cvar.notify_all(); // 通知所有等待的线程
    }

    pub fn poll(&self) -> Vec<T> {
        let mut events = Vec::new();
        while let Ok(ev) = self.receiver.try_recv() {
            events.push(ev);
        }
        events
    }
}
