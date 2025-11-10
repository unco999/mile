use bytemuck::{Pod, Zeroable, cast_slice, offset_of};
use flume::{Receiver, Sender};
use glam::{Vec2, vec2};
use image::{Frame, ImageReader, RgbaImage, imageops::overlay};
use itertools::Itertools;

use mile_api::{
    prelude::{EventStream, GlobalUniform},
    util::WGPU,
};
use mile_gpu_dsl::{core::Expr, prelude::kennel::Kennel};
use mile_graphics::structs::{GlobalState, GlobalStateRecord, GlobalStateType};
use std::{
    any::Any, cell::RefCell, collections::{self, HashMap}, default, fs, hash::Hash, marker::PhantomData, num::{NonZeroU32, NonZeroU64}, path::Path, rc::Rc, sync::{Arc, Mutex}, u32
};


use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferUsages, Device, RenderPass,
    SamplerBindingType, ShaderStages, TextureSampleType, TextureView, TextureViewDimension,
    VertexFormat,
    util::{BufferInitDescriptor, DeviceExt, DownloadBuffer},
    wgc::device::{self, queue},
};
use winit::window::Window;

use crate::{
    mui_prototype::UiState,
    prelude::*,
    runtime::{StateConfigDes, StateNetWorkConfigDes, WgslResult},
};

const PADDING: u32 = 2; // 每张图像间的像素间距，防止GPU采样溢出
const DEFAULT_ATLAS_SIZE: u32 = 2048;

type PANEL_ID = u32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}


const QUAD_INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
pub struct Panel {
    // === 16-byte 块 1 ===
    pub position: [f32; 2], // 8
    pub size: [f32; 2],     // 8

    // === 16-byte 块 2 ===
    pub uv_offset: [f32; 2], // 8
    pub uv_scale: [f32; 2],  // 8

    // === 16-byte 块 3 ===
    pub z_index: u32,      // 4
    pub pass_through: u32, // 4
    pub id: u32,           // 4
    pub interaction: u32,  // 4

    // === 16-byte 块 4 ===
    pub event_mask: u32,  // 4
    pub state_mask: u32,  // 4
    pub transparent: f32, // 4
    pub texture_id: u32,  // 4

    // === 16-byte 块 5 ===
    pub state: u32, // 4
    pub collection_state: u32,
    pub kennel_des_id: u32, // 12, 补齐到 16
    pub pad_1: u32,         // 12, 补齐到 16
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
pub struct PanelAnimDelta {
    // --- Position delta ---
    pub delta_position: [f32; 2], // 8
    pub delta_size: [f32; 2],     // 8

    // --- UV delta ---
    pub delta_uv_offset: [f32; 2],
    pub delta_uv_scale: [f32; 2],

    // --- Panel attributes ---
    pub delta_z_index: i32,
    pub delta_pass_through: i32,
    pub panel_id: u32,
    pub _pad0: u32, // 对齐填充

    // --- 状态相关 ---
    pub delta_interaction: u32,
    pub delta_event_mask: u32,
    pub delta_state_mask: u32,
    pub _pad1: u32, // 对齐填充

    // --- 透明度/texture ---
    pub delta_transparent: f32,
    pub delta_texture_id: i32,
    pub _pad2: [f32; 2], // ✅ 对齐到16字节

    // --- 起始位置 ---
    pub start_position: [f32; 2],
    pub _pad3: [f32; 2], // ✅ 对齐补齐
}

impl PanelAnimDelta {
    /// 转成 GPU buffer
    pub fn write_to_buffer(
        &self,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        offset: wgpu::BufferAddress,
    ) {
        let raw_bytes = bytemuck::cast_slice(std::slice::from_ref(self));
        queue.write_buffer(buffer, offset, raw_bytes);
    }

    /// 全局初始化 GPU buffer，固定 1024 个元素
    pub fn global_init(device: &wgpu::Device) -> wgpu::Buffer {
        let buffer_size = 1024 * std::mem::size_of::<PanelAnimDelta>() as wgpu::BufferAddress;

        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PanelAnimDelta Global Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

#[derive(Debug, Clone)]
/// 单字段动画描述
pub struct TransformAnimFieldInfo {
    pub field_id: u32,          // PanelField 位标记
    pub start_value: Vec<f32>,  // 起始值
    pub target_value: Vec<f32>, // 结束值
    pub duration: f32,          // 持续时间
    pub easing: EasingMask,     // 渐变函数
    pub op: AnimOp,             // 叠加方式
    pub hold: u32,              // 动画结束后是否保持最后值
    pub delay: f32,             // 延迟时间
    pub loop_count: u32,        // 循环次数，0 = 无限
    pub ping_pong: u32,         // 往返动画
    pub on_complete: u32,       // 完成回调，参数 panel_id
}

impl TransformAnimFieldInfo {
    /// 拆分多位 field_id，生成 GPU 用偏移指针
    /// cache_start_offset: 动画缓存起始偏移
    ///
    /// 拆分多位 field_id，并生成 GPU buffer 原始 f32 数组
    pub fn split_write_field(&self, panel_id: u32) -> Vec<AnimtionFieldOffsetPtr> {
        let mut result = Vec::new();
        let mut mask = self.field_id;
        let mut bit_index = 0;
        let mut value_index = 0;

        while mask != 0 {
            if mask & 1 != 0 {
                result.push(AnimtionFieldOffsetPtr {
                    field_id: 1 << bit_index,
                    start_value: self.start_value[value_index],
                    target_value: self.target_value[value_index],
                    elapsed: 0.0,
                    duration: self.duration,
                    op: self.op.bits(),
                    hold: self.hold,
                    delay: self.delay,
                    loop_count: self.loop_count,
                    ping_pong: self.ping_pong,
                    on_complete: self.on_complete,
                    panel_id,
                    death: 0,
                    easy_fn: self.easing.bits(),
                    _pad: [0],
                });

                value_index += 1;
            }

            mask >>= 1;
            bit_index += 1;
        }

        result
    }
}

//这里是每个实例 分配一个GPU线程
//我们把TransformAnim的字段 分拆这个实例
//比如TransformAnim里面有N个字段的动画 在TransformAnim里面写成一个实例
//但是实际上分拆N个AnimtionFieldOffsetPtr
//我们就可以方便的去调用buffer
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
pub struct AnimtionFieldOffsetPtr {
    pub field_id: u32,     // 字段标识
    pub start_value: f32,  // 起始值
    pub target_value: f32, // 目标值
    pub elapsed: f32,      // 已经过的时间
    pub duration: f32,     // 动画持续时间
    pub op: u32,           // 操作类型（SET/ADD/MUL/…）
    pub hold: u32,         // hold 时间
    pub delay: f32,        // 延迟时间
    pub loop_count: u32,   // 循环次数
    pub ping_pong: u32,    // 往返标记
    pub on_complete: u32,  // 回调标识
    pub panel_id: u32,     // Panel ID
    pub death: u32,        // 是否结束
    pub easy_fn: u32,      // easing 函数标识
    pub _pad: [u32; 1],    // 补齐16字节对齐
}

// impl TransformAnimInfo{
//     fn to_transform_anim(&self,start_offset:u32) -> TransformAnim{
//         let end_offset = self.field_id.bits().count_ones();
//         TransformAnim{
//             field_id: self.field_id.bits(),
//             field_len: 0u32,
//             start_offset,
//             end_offset,
//             easing_mask: self.easing_mask.bits(),
//             hold: self.hold.bits(),
//             duration: self.duration,
//             elapsed: 0.0,
//             panel_id: self.panel_id,
//             op: self.op.bits(),
//             _pad2: 0u32,
//             _pad3: 0u32,
//             last_applied: 0.0,
//             _pad4: [0u32;3],
//         }
//     }

//     fn to_gpu(&self,queue:&wgpu::Queue,device:&wgpu::Device,gpu_ui:&mut GpuUi){
//         //原始数据动画偏移量下标 代表最新的可写偏移量位置
//         let animation_field_raw_point = gpu_ui.animation_pipe_line_cahce.animtion_field_point;
//         let animtion_mata = self.to_transform_anim(animation_field_raw_point as u32);

//         // 先把动画存到 CPU 端 cache
//         let list = gpu_ui
//             .transfrom_anim_cache
//             .entry(animtion_mata.panel_id)
//             .or_insert_with(Vec::new);
//         list.push(animtion_mata);
//         let len = list.len();

//         // 偏移字节

//         let anim_count = gpu_ui.animation_pipe_line_cahce
//             .gpu_animation_des
//             .animation_count;

//         let delta_offset_bytes = std::mem::size_of::<PanelAnimDelta>() as wgpu::BufferAddress
//             * anim_count as wgpu::BufferAddress;

//         println!("当前的动画动画偏移量:{:?}", delta_offset_bytes);
//         PanelAnimDelta::default().write_to_buffer(
//             queue,
//             &gpu_ui
//                 .animation_pipe_line_cahce
//                 .panel_anim_delta_buffer
//                 .as_ref()
//                 .unwrap(),
//             delta_offset_bytes,
//         );

//         gpu_ui.animation_pipe_line_cahce
//             .gpu_animation_des
//             .animation_count += 1;

//         let len = (animtion_mata.end_offset - animtion_mata.start_offset) * 4;

//         // 创建一个全零的临时 Vec<u8>
//         let zero_bytes = vec![0u8; len as usize];

//         gpu_ui.animation_pipe_line_cahce.animtion_field_point += len as u64;

//         queue.write_buffer(
//             gpu_ui.animation_pipe_line_cahce.animtion_field_buffer
//                     .as_ref()
//                     .unwrap(),
//                     animation_field_raw_point,
//                     bytemuck::cast_slice(zero_bytes.as_slice())
//             );

//         gpu_ui.animation_pipe_line_cahce
//             .gpu_animation_des
//             .write_to_buffer(
//                 queue,
//                 gpu_ui.animation_pipe_line_cahce
//                     .gpu_animation_des_buffer
//                     .as_ref()
//                     .unwrap(),
//             );
//         // 写入 GPU buffer
//         queue.write_buffer(
//             &gpu_ui
//                 .animation_pipe_line_cahce
//                 .animtion_buffer
//                 .as_ref()
//                 .unwrap(),
//             delta_offset_bytes,
//             bytemuck::bytes_of(&animtion_mata),
//         );
//     }
// }
pub struct GlobalLayout {
    pub global_unitform_struct: Rc<RefCell<GlobalUniform>>,
    pub global_unitform_buffer: Buffer,
    pub shared_structs: GlobalUiState,
    pub shared_buffer: Buffer,
}

pub struct GpuUi {
    pub event_stream: EventStream<EventTest>,
    pub kennel: Arc<RefCell<Kennel>>,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub num_instances: u32,
    pub pipeline: Option<wgpu::RenderPipeline>,
    pub render_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub texture_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub compute_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub instances: Vec<Panel>,
    pub indirects_buffer: Buffer,
    pub instance_pool_index: u32,
    pub compute_pipeline: Option<wgpu::ComputePipeline>,
    pub compute_bind_group: Option<BindGroup>,
    pub event_hub: Arc<UIEventHub>,
    pub global_state: Arc<Mutex<GlobalState>>,
    pub surface: TextureView,
    pub ui_texture_map: TextureAtlasSet,
    pub render_bind_group: Option<wgpu::BindGroup>,
    pub ui_texture_bind_group: Option<wgpu::BindGroup>,
    pub ui_panel_extra_buffer: Option<wgpu::Buffer>,
    pub animation_pipe_line_cahce: AnimationPipelineCache,
    pub interaction_pipeline_cache: InteractionPipelineCache,
    pub panel_interaction_trigger: PanelInteractionTrigger,
    pub new_work_store: Option<NetworkStore>,
    pub global_layout: Option<GlobalLayout>,
    pub custom_wgsl: Box<[CustomWgsl; 1024]>,
    pub custom_wgsl_buffer: wgpu::Buffer,
    indirects_len: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
pub struct CustomWgsl {
    pub frag: [[f32; 4]; 16],
    pub vertex: [[f32; 4]; 16],
}

impl GpuNetWorkDes {
    /// 转成 GPU buffer
    pub fn to_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuNetWorkDes Buffer"),
            contents: bytemuck::cast_slice(std::slice::from_ref(self)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn write_to_buffer(&self, queue: &wgpu::Queue, buffer: &wgpu::Buffer) {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(std::slice::from_ref(self)));
    }
}
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
pub struct GpuNetWorkDes {
    pub net_work_count: u32, // 4 bytes
    pub ids_len: u32,        // 4 bytes
    pub pad0: u32,           // 4 bytes padding
    pub pad1: u32,           // 4 bytes padding -> 对齐到16字节
    pub pad2: [u32; 4],      // 16 bytes
}

#[derive(Default)]
pub struct AnimationPipelineCache {
    pub panel_anim_delta_buffer: Option<wgpu::Buffer>,
    pub pipe_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub pipe_bind_group: Option<wgpu::BindGroup>,
    pub pipeline: Option<wgpu::ComputePipeline>,
    //这个是动画的描述和索引表
    pub animtion_field_offset_buffer: Option<Buffer>,
    pub animtion_field_offset_ptr_point: wgpu::BufferAddress,
    pub gpu_animation_des: GpuAnimationDes,
    pub gpu_animation_des_buffer: Option<wgpu::Buffer>,
    //这个是纯排列的原始f32实际数值
    pub animtion_raw_buffer: Option<wgpu::Buffer>,
    pub animtion_raw_buffer_point: wgpu::BufferAddress,
    pub net_work_layout: Option<wgpu::BindGroupLayout>,
    pub net_work_bind_group: Option<wgpu::BindGroup>,
    pub field_cahce: AnimtionFieldCache,
}

type AnimtionIdx = u32;
type PanelFieldBit = u32;
type PanelIdx = u32;
#[derive(Default)]
struct AnimtionFieldCache {
    pub hash: HashMap<(AnimtionIdx, PanelIdx), PanelFieldBit>,
}

#[derive(Debug, Clone, Default)]
pub struct InteractionPipelineCache {
    pub pipe_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub pipe_bind_group: Option<wgpu::BindGroup>,
    pub pipeline: Option<wgpu::ComputePipeline>,
    pub interaction_struct: Option<InteractionFrameCache>,
    pub gpu_interaction_struct: Option<GpuInteractionFrameCache>,
    pub gpu_Interaction_buffer: Option<wgpu::Buffer>,
}

#[derive(Debug, Clone)]
pub struct InteractionFrameCache {
    pub pre: InteractionFrame,
    pub curr: InteractionFrame,
}

impl InteractionFrameCache {
    /// 转换为 GPU 可用缓存
    pub fn to_gpu(&self) -> GpuInteractionFrameCache {
        GpuInteractionFrameCache {
            pre: self.pre.to_gpu(),
            curr: self.curr.to_gpu(),
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
pub struct GpuInteractionFrameCache {
    pub pre: GpuInteractionFrame,
    pub curr: GpuInteractionFrame,
}

impl GpuInteractionFrameCache {
    pub fn to_buffer_vec(&self, device: &wgpu::Device) -> wgpu::Buffer {
        // 把 pre 和 curr 展平为一个 Vec
        let frames = vec![self.pre, self.curr];
        println!(
            "Size of frames vec (2 elements): {}",
            size_of_val(&frames[..])
        );

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuInteractionFrameBuffer"),
            contents: bytemuck::cast_slice(&frames),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    }

    pub fn copy_interaction_swap_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_buffer: &wgpu::Buffer,
        frame: u32,
    ) {
        let frame_size = std::mem::size_of::<GpuInteractionFrame>() as wgpu::BufferAddress;

        // 创建 encoder
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CopyCurrToPreEncoder"),
        });

        // 临时 buffer
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TempFrameBuffer"),
            size: frame_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.curr.frame = frame;

        let new_curr_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("cur frame".into()),
            contents: bytemuck::cast_slice(&[self.curr]),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // 1️⃣ 把 curr (offset = frame_size) 复制到 temp
        encoder.copy_buffer_to_buffer(frame_buffer, frame_size, &temp_buffer, 0, frame_size);

        // 2️⃣ 把 temp buffer 写回 pre (offset = 0)
        encoder.copy_buffer_to_buffer(&temp_buffer, 0, frame_buffer, 0, frame_size);

        // 清空 [0..frame_size) 部分
        encoder.copy_buffer_to_buffer(&new_curr_buffer, 0, frame_buffer, frame_size, frame_size);
        // 提交
        queue.submit(Some(encoder.finish()));
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct GpuInteractionFrame {
    pub frame: u32,
    pub drag_id: u32,
    pub hover_id: u32,
    pub click_id: u32,

    pub mouse_pos: [f32; 2],
    pub trigger_panel_state: u32,
    pub _pad1: u32,

    pub mouse_state: u32,
    pub _pad2: [u32; 3],

    pub drag_delta: [f32; 2],
    pub _pad3: [f32; 2],

    pub pinch_delta: f32,
    pub pass_through_depth: u32,
    pub _pad4: [u32; 2],

    pub _pad5: [u32; 8], // 填充到 128 bytes
}

#[derive(Debug, Clone, Default)]
pub struct InteractionFrame {
    /// 当前帧时间（秒或任意时间单位）
    pub frame: u32,

    /// 当前被拖拽的面板 ID，如果没有拖拽为 None
    pub drag_id: Option<u32>,

    /// 当前鼠标悬停的面板 ID，如果没有悬停为 None
    pub hover_id: Option<u32>,

    /// 当前鼠标点击的面板 ID，如果没有点击为 None
    pub click_id: Option<u32>,

    /// 鼠标位置，屏幕坐标
    pub mouse_pos: [f32; 2],

    /// 鼠标状态位掩码（可扩展，例如按下、抬起、右键、滚轮等）
    pub mouse_state: u32,

    /// 可选记录：本帧拖拽偏移（屏幕坐标）
    pub drag_delta: Option<[f32; 2]>,

    /// 可选记录：缩放手势
    pub pinch_delta: Option<f32>,

    /// 当前鼠标穿透层数（pass-through 层数）
    pub pass_through_depth: u32,
}

impl InteractionFrame {
    /// 转换为 GPU 可用结构
    pub fn to_gpu(&self) -> GpuInteractionFrame {
        GpuInteractionFrame {
            frame: self.frame,
            drag_id: u32::MAX,
            hover_id: u32::MAX,
            click_id: u32::MAX,
            mouse_pos: self.mouse_pos,
            mouse_state: self.mouse_state,
            drag_delta: [f32::MAX; 2],
            pinch_delta: f32::MAX,
            pass_through_depth: self.pass_through_depth,
            _pad2: [0u32; 3],
            _pad4: [0u32; 2],
            _pad1: 0u32,
            _pad3: [0f32; 2],
            _pad5: [0u32; 8],
            trigger_panel_state: 0u32,
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct GpuAnimationDes {
    pub animation_count: u32, // 当前帧动画数量
    pub frame_count: u32,     // 每个动画帧数
    pub start_index: u32,     // 动画在全局 buffer 的起始索引
    pub _pad0: u32,           // padding 16字节对齐

    pub delta_time: f32, // 单帧时间增量
    pub total_time: f32, // 动画总时长
    pub _pad1: [f32; 2], // 补齐16字节对齐
}

impl GpuAnimationDes {
    /// 转成 GPU buffer
    pub fn to_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuAnimationDes Buffer"),
            contents: bytemuck::cast_slice(std::slice::from_ref(self)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn write_to_buffer(&self, queue: &wgpu::Queue, buffer: &wgpu::Buffer) {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(std::slice::from_ref(self)));
    }
}

impl GpuUi {
/// 写入“整个字段”：F 必须是 Pod
    pub fn write_field<S, F: bytemuck::Pod>(
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        field: field_offset::FieldOffset<S, F>,
        value: &F,
    ) {
        let offset = field.get_byte_offset() as wgpu::BufferAddress;
        debug_assert_eq!(offset % 4, 0); // wgpu 写入偏移建议 4 字节对齐
        queue.write_buffer(buffer, offset, bytemuck::bytes_of(value));
    }

    pub fn set_frag_slot(
        &mut self,
        panel_id: usize,
        slot: usize,
        value: [f32; 4],
        queue: &wgpu::Queue,
    ) {
        if panel_id >= self.custom_wgsl.len() || slot >= 16 {
            return;
        }

        // 更新 CPU 缓存
        self.custom_wgsl[panel_id].frag[slot] = value;

        // 计算 GPU buffer 偏移
        let panel_offset = panel_id as wgpu::BufferAddress * 512;
        let frag_offset = panel_offset + (slot as wgpu::BufferAddress * 16);

        // 写入 GPU
        queue.write_buffer(&self.custom_wgsl_buffer, frag_offset, cast_slice(&[value]));
    }

    pub fn set_vertex_slot(
        &mut self,
        panel_id: usize,
        slot: usize,
        value: [f32; 4],
        queue: &wgpu::Queue,
    ) {
        if panel_id >= self.custom_wgsl.len() || slot >= 16 {
            return;
        }

        self.custom_wgsl[panel_id].vertex[slot] = value;

        let panel_offset = panel_id as wgpu::BufferAddress * 512;
        let vertex_offset = panel_offset + 256 + (slot as wgpu::BufferAddress * 16);

        queue.write_buffer(
            &self.custom_wgsl_buffer,
            vertex_offset,
            cast_slice(&[value]),
        );
    }

    pub fn update_global_unifrom_time(&mut self, queue: &wgpu::Queue, dt: f32) {
        let global = self.global_layout.as_mut().unwrap();
        let mut unitfrom_struct = global.global_unitform_struct.borrow_mut();
        unitfrom_struct.time += dt;
        let offset = offset_of!(GlobalUniform, time) as wgpu::BufferAddress;
        queue.write_buffer(
            &global.global_unitform_buffer,
            offset,
            bytemuck::bytes_of(&unitfrom_struct.time),
        );
    }

    // pub fn new(
    //     device: &wgpu::Device,
    //     format: wgpu::TextureFormat,
    //     global_state: Arc<Mutex<GlobalState>>,
    //     cpu_global_uniform: Rc<CpuGlobalUniform>,
    //     window: &winit::window::Window,
    //     global_hub: Arc<GlobalEventHub<ModuleEvent<Expr, LayerID>>>,
    //     kennel: Arc<RefCell<Kennel>>,
    // ) -> Self {
    //     let event_stream = global_event_bus().subscribe::<EventTest>();

    //     println!(
    //         "size = {}, align = {}",
    //         std::mem::size_of::<Panel>(),
    //         std::mem::align_of::<Panel>()
    //     );
    //     // 顶点 buffer
    //     let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //         label: Some("UI Quad Vertex Buffer"),
    //         contents: bytemuck::cast_slice(QUAD_VERTICES),
    //         usage: wgpu::BufferUsages::VERTEX,
    //     });

    //     let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //         label: Some("UI Quad Index Buffer"),
    //         contents: bytemuck::cast_slice(QUAD_INDICES),
    //         usage: wgpu::BufferUsages::INDEX,
    //     });

    //     // 默认一个 instance
    //     let instances = [];
    //     let max_instances = 1000;
    //     let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    //         label: Some("UI Instance Buffer"),
    //         size: (max_instances * std::mem::size_of::<Panel>()) as wgpu::BufferAddress,
    //         usage: wgpu::BufferUsages::VERTEX
    //             | wgpu::BufferUsages::STORAGE
    //             | wgpu::BufferUsages::COPY_DST,
    //         mapped_at_creation: false,
    //     });

    //     let global_shared_struct = GlobalUiState {
    //         mouse_pos: [0.0, 0.0],
    //         hover_id: u32::MAX,
    //         hover_blocked: 0,
    //         _pad0: 0,
    //         hover_pos: [0.0f32; 2],
    //         _pad1: [0u32; 2],
    //         current_depth: 0,
    //         _pad2: 0,
    //         history_id: u32::MAX,
    //         _pad3: 0,
    //         _pad4: [0u32; 2],
    //         mouse_state: MouseState::DEFAULT.bits(),
    //         click_id: u32::MAX,
    //         click_blocked: 0,
    //         drag_id: u32::MAX,
    //         drag_blocked: 0,
    //     };

    //     let shared_buffer = device.create_buffer_init(&BufferInitDescriptor {
    //         label: Some("Shared State"),
    //         usage: wgpu::BufferUsages::UNIFORM
    //             | wgpu::BufferUsages::STORAGE
    //             | wgpu::BufferUsages::COPY_DST, // CPU 写入
    //         contents: bytemuck::bytes_of(&global_shared_struct),
    //     });

    //     let max_layers = 32; // 或根据你的 UI 层数需求

    //     let indirects_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    //         label: Some("Indirect Draw Buffer"),
    //         size: (max_layers * std::mem::size_of::<DrawIndexedIndirect>()) as u64,
    //         usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
    //         mapped_at_creation: false,
    //     });

    //     let size = window.inner_size(); // 返回 PhysicalSize<u32>
    //     let width = size.width;
    //     let height = size.height;

    //     println!("目前的gpu w:{} h:{}", width, height);

    //     let max_animtion = 1024; // 或根据你的 UI 层数需求

    //     let wgsl_structs = Box::new(
    //         [CustomWgsl {
    //             frag: [[1.0f32; 4]; 16],
    //             vertex: Default::default(),
    //         }; 1024],
    //     );

    //     let wgsl_buffer = device.create_buffer_init(&BufferInitDescriptor {
    //         label: Some("Shared State"),
    //         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, // CPU 写入
    //         contents: bytemuck::cast_slice(&*wgsl_structs),
    //     });

    //     let mut gpu_debug = GpuDebug::new("mile_ui");
    //     gpu_debug.create_buffer(device);

    //     Self {
    //         event_stream,
    //         global_hub,
    //         kennel,
    //         ui_kennel_des: KennelReadDesPool::default(),
    //         gpu_debug: RefCell::new(gpu_debug),
    //         custom_wgsl: wgsl_structs,
    //         custom_wgsl_buffer: wgsl_buffer,
    //         global_layout: Some(GlobalLayout {
    //             global_unitform_struct: cpu_global_uniform.get_struct(),
    //             global_unitform_buffer: cpu_global_uniform.get_buffer(),
    //             shared_structs: global_shared_struct,
    //             shared_buffer: shared_buffer,
    //         }),
    //         new_work_store: None,
    //         ui_panel_extra_buffer: None,
    //         global_state,
    //         vertex_buffer,
    //         index_buffer,
    //         instance_buffer,
    //         num_indices: QUAD_INDICES.len() as u32,
    //         num_instances: instances.len() as u32,
    //         pipeline: None,
    //         ui_texture_bind_group: None,
    //         indirects_len: 0,
    //         indirects_buffer,
    //         render_bind_group: None,
    //         instances: instances.to_vec(),
    //         compute_bind_group_layout: None,
    //         compute_pipeline: None,
    //         compute_bind_group: None,
    //         instance_pool_index: 0,
    //         event_hub: Arc::new(UIEventHub::new()),
    //         surface: GpuUi::create_surface(device, format),
    //         ui_texture_map: TextureAtlasSet::new(),
    //         render_bind_group_layout: None,
    //         texture_bind_group_layout: None,
    //         animation_pipe_line_cahce: AnimationPipelineCache::default(),
    //         interaction_pipeline_cache: InteractionPipelineCache::default(),
    //         panel_interaction_trigger: PanelInteractionTrigger::default(),
    //     }
    // }

    /**
     * Panel 内存偏移对显存偏移的唯一创造函数  所有的附加信息 全部在这里创造
     * 只要compute idx 访问  可以通过这个idx 访问这个gpu数据结构里 任意的新增加附加信息
     *
     */

    pub fn update_dt(&mut self, dt: f32, queue: &wgpu::Queue) {
        let global = self.global_layout.as_mut().unwrap();
        let mut unitfrom_struct = global.global_unitform_struct.borrow_mut();

        unitfrom_struct.dt = dt;
        let dt_offset = offset_of!(GlobalUniform, dt) as wgpu::BufferAddress;
        queue.write_buffer(
            &global.global_unitform_buffer,
            dt_offset,
            bytemuck::bytes_of(&dt),
        );
    }

    pub fn window_resized(&mut self, width: u32, height: u32, queue: &wgpu::Queue) {
        let global = self.global_layout.as_mut().unwrap();
        let mut unitfrom_struct = global.global_unitform_struct.borrow_mut();

        unitfrom_struct.screen_size = [width, height];
        let state_offset = offset_of!(GlobalUniform, screen_size) as wgpu::BufferAddress;
        queue.write_buffer(
            &global.global_unitform_buffer,
            state_offset,
            bytemuck::bytes_of(&[width, height]),
        );
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
        for path in image_paths {
            println!("读取纹理文件: {:?}", path);
            self.read_img(path.as_path());
        }
    }

    // pub fn create_interaction_compute_pipeline(&mut self, device: &wgpu::Device) {
    //     let compute_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    //         label: Some("Interaction Compute Shader"),
    //         source: wgpu::ShaderSource::Wgsl(include_str!("interaction_compute.wgsl").into()),
    //     });

    //     self.interaction_pipeline_cache.interaction_struct = Some(InteractionFrameCache {
    //         pre: InteractionFrame::default(),
    //         curr: InteractionFrame::default(),
    //     });

    //     let gpu_interaction_struct = self
    //         .interaction_pipeline_cache
    //         .interaction_struct
    //         .as_ref()
    //         .unwrap()
    //         .to_gpu();

    //     self.interaction_pipeline_cache.gpu_interaction_struct = Some(gpu_interaction_struct);

    //     let mut interaction_frame_buffer = gpu_interaction_struct.to_buffer_vec(device);

    //     self.interaction_pipeline_cache.gpu_Interaction_buffer =
    //         Some(interaction_frame_buffer.clone());

    //     let Interaction_bind_group_layout =
    //         device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    //             label: Some("Compute Bind Group Layout"),
    //             entries: &[
    //                 // 0️⃣ instance buffer
    //                 wgpu::BindGroupLayoutEntry {
    //                     binding: 0,
    //                     visibility: wgpu::ShaderStages::COMPUTE,
    //                     ty: wgpu::BindingType::Buffer {
    //                         ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                         has_dynamic_offset: false,
    //                         min_binding_size: None,
    //                     },
    //                     count: None,
    //                 },
    //                 // 1️⃣ shared buffer
    //                 wgpu::BindGroupLayoutEntry {
    //                     binding: 1,
    //                     visibility: wgpu::ShaderStages::COMPUTE,
    //                     ty: wgpu::BindingType::Buffer {
    //                         ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                         has_dynamic_offset: false,
    //                         min_binding_size: None,
    //                     },
    //                     count: None,
    //                 },
    //                 // 2️⃣ global uniform
    //                 wgpu::BindGroupLayoutEntry {
    //                     binding: 2,
    //                     visibility: wgpu::ShaderStages::COMPUTE,
    //                     ty: wgpu::BindingType::Buffer {
    //                         ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                         has_dynamic_offset: false,
    //                         min_binding_size: None,
    //                     },
    //                     count: None,
    //                 },
    //                 // 3️⃣ animation buffer
    //                 wgpu::BindGroupLayoutEntry {
    //                     binding: 3,
    //                     visibility: wgpu::ShaderStages::COMPUTE,
    //                     ty: wgpu::BindingType::Buffer {
    //                         ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                         has_dynamic_offset: false,
    //                         min_binding_size: None,
    //                     },
    //                     count: None,
    //                 },
    //                 wgpu::BindGroupLayoutEntry {
    //                     binding: 4,
    //                     visibility: wgpu::ShaderStages::COMPUTE,
    //                     ty: wgpu::BindingType::Buffer {
    //                         ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                         has_dynamic_offset: false,
    //                         min_binding_size: None,
    //                     },
    //                     count: None,
    //                 },
    //             ],
    //         });

    //     let Interaction_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    //         label: Some("Compute Bind Group"),
    //         layout: &Interaction_bind_group_layout,
    //         entries: &[
    //             wgpu::BindGroupEntry {
    //                 binding: 0,
    //                 resource: self.instance_buffer.as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 1,
    //                 resource: self
    //                     .global_layout
    //                     .as_ref()
    //                     .unwrap()
    //                     .global_unitform_buffer
    //                     .as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 2,
    //                 resource: interaction_frame_buffer.as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 3,
    //                 resource: self
    //                     .gpu_debug
    //                     .borrow()
    //                     .buffer
    //                     .as_ref()
    //                     .unwrap()
    //                     .as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 4,
    //                 resource: self
    //                     .animation_pipe_line_cahce
    //                     .panel_anim_delta_buffer
    //                     .as_ref()
    //                     .unwrap()
    //                     .as_entire_binding(),
    //             },
    //         ],
    //     });

    //     let interaction_compute_pipeline =
    //         device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    //             label: Some("interaction_compute_pipeline_pipeline"),
    //             layout: Some(
    //                 &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    //                     label: Some("interaction_compute_pipeline Layout"),
    //                     bind_group_layouts: &[&Interaction_bind_group_layout],
    //                     push_constant_ranges: &[],
    //                 }),
    //             ),
    //             module: &compute_shader_module,
    //             entry_point: Some("main"),
    //             compilation_options: Default::default(),
    //             cache: Default::default(),
    //         });

    //     self.interaction_pipeline_cache.pipe_bind_group_layout =
    //         Some(Interaction_bind_group_layout);
    //     self.interaction_pipeline_cache.pipe_bind_group = Some(Interaction_bind_group);
    //     self.interaction_pipeline_cache.pipeline = Some(interaction_compute_pipeline);
    // }

    // pub fn interaction_compute(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
    //     let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
    //         label: Some("interaction_compute Encoder"),
    //     });
    //     {
    //         let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //             label: Some("UI interaction_compute pass"),
    //             timestamp_writes: Default::default(),
    //         });
    //         // encoder.copy_buffer_to_buffer(&self.shared_buffer, 0, &self.shared_buffer_readback_buffer, 0, self.shared_buffer.size());

    //         cpass.set_pipeline(&self.interaction_pipeline_cache.pipeline.as_ref().unwrap());
    //         cpass.set_bind_group(0, &self.interaction_pipeline_cache.pipe_bind_group, &[]);

    //         let panel_count = self.instances.len() as u32;

    //         let workgroups = (panel_count + 63) / 64;

    //         cpass.dispatch_workgroups(workgroups, 1, 1);
    //     }
    //     queue.submit(Some(encoder.finish()));
    //     self.interaction_cpu_trigger(device, queue);
    // }

    // pub fn createa_animtion_compute_pipeline_two(&mut self, device: &wgpu::Device) {
    //     let bind_layout = self
    //         .animation_pipe_line_cahce
    //         .net_work_layout
    //         .as_ref()
    //         .unwrap();
    //     let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    //         label: Some("Compute Bind Group"),
    //         layout: &bind_layout,
    //         entries: &[
    //             wgpu::BindGroupEntry {
    //                 binding: 0,
    //                 resource: self
    //                     .new_work_store
    //                     .as_ref()
    //                     .unwrap()
    //                     .buffer_meta
    //                     .as_ref()
    //                     .unwrap()
    //                     .as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 1,
    //                 resource: self
    //                     .new_work_store
    //                     .as_ref()
    //                     .unwrap()
    //                     .buffer_rels
    //                     .as_ref()
    //                     .unwrap()
    //                     .as_entire_binding(),
    //             },
    //         ],
    //     });
    //     self.animation_pipe_line_cahce.net_work_bind_group = Some(compute_bind_group)
    // }

    // pub fn create_animtion_compute_pipeline(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
    //     let max_animtion = 8192;
    //     let animtion_des_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    //         label: Some("TransformAnim Buffer"),
    //         size: (max_animtion * std::mem::size_of::<AnimtionFieldOffsetPtr>()) as u64,
    //         usage: wgpu::BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    //         mapped_at_creation: false,
    //     });

    //     let max_animation_field = 8192;
    //     let animtion_raw_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    //         label: Some("max_animation_field Buffer"),
    //         size: (max_animation_field * std::mem::size_of::<f32>()) as u64,
    //         usage: wgpu::BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    //         mapped_at_creation: false,
    //     });

    //     let compute_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    //         label: Some("compute_shader_module Compute Shader"),
    //         source: wgpu::ShaderSource::Wgsl(include_str!("animtion_compute.wgsl").into()),
    //     });

    //     let panel_anim_delta_buffer = PanelAnimDelta::global_init(device);
    //     let animtion_gpu_des_buffer = GpuAnimationDes::default().to_buffer(device);

    //     // === Bind Group Layout ===
    //     let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    //         label: Some("Compute Bind Group Layout"),
    //         entries: &[
    //             // 0️⃣ instance buffer
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 0,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             // 1️⃣ shared buffer
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 1,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             // 2️⃣ global uniform
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 2,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             // 3️⃣ animation buffer
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 3,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 4,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Uniform,
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 5,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //         ],
    //     });

    //     // === Bind Group Layout ===
    //     let net_work_bind_group_layout =
    //         device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    //             label: Some("Compute Bind Group Layout"),
    //             entries: &[
    //                 // 0️⃣ instance buffer
    //                 wgpu::BindGroupLayoutEntry {
    //                     binding: 0,
    //                     visibility: wgpu::ShaderStages::COMPUTE,
    //                     ty: wgpu::BindingType::Buffer {
    //                         ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                         has_dynamic_offset: false,
    //                         min_binding_size: None,
    //                     },
    //                     count: None,
    //                 },
    //                 // 1️⃣ shared buffer
    //                 wgpu::BindGroupLayoutEntry {
    //                     binding: 1,
    //                     visibility: wgpu::ShaderStages::COMPUTE,
    //                     ty: wgpu::BindingType::Buffer {
    //                         ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                         has_dynamic_offset: false,
    //                         min_binding_size: None,
    //                     },
    //                     count: None,
    //                 },
    //             ],
    //         });
    //     // let net_work = self.ui_net_work.borrow();
    //     // let buffer_set = net_work.buffer_set.as_ref().unwrap();

    //     // === Bind Group ===
    //     let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    //         label: Some("Compute Bind Group"),
    //         layout: &bind_group_layout,
    //         entries: &[
    //             wgpu::BindGroupEntry {
    //                 binding: 0,
    //                 resource: self.instance_buffer.as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 1,
    //                 resource: self
    //                     .global_layout
    //                     .as_ref()
    //                     .unwrap()
    //                     .global_unitform_buffer
    //                     .as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 2,
    //                 resource: animtion_des_buffer.as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 3,
    //                 resource: panel_anim_delta_buffer.as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 4,
    //                 resource: animtion_gpu_des_buffer.as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 5,
    //                 resource: self
    //                     .gpu_debug
    //                     .borrow()
    //                     .buffer
    //                     .as_ref()
    //                     .unwrap()
    //                     .as_entire_binding(),
    //             },
    //             // wgpu::BindGroupEntry {
    //             //     binding: 6,
    //             //     resource: buffer_set.collection_buffer.as_ref().unwrap().as_entire_binding(),
    //             // },
    //             // wgpu::BindGroupEntry {
    //             //     binding: 7,
    //             //     resource: buffer_set.panel_ids_buffer.as_ref().unwrap().as_entire_binding(),
    //             // },
    //             // wgpu::BindGroupEntry {
    //             //     binding: 8,
    //             //     resource: buffer_set.ui_rel_buffer.as_ref().unwrap().as_entire_binding(),
    //             // },
    //             // wgpu::BindGroupEntry {
    //             //     binding: 9,
    //             //     resource: buffer_set.panel_id_to_rel_buffer.as_ref().unwrap().as_entire_binding(),
    //             // },
    //         ],
    //     });

    //     // === Compute Pipeline ===
    //     let animation_compute_pipeline =
    //         device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    //             label: Some("animation_compute_pipeline"),
    //             layout: Some(
    //                 &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    //                     label: Some("Compute Pipeline Layout"),
    //                     bind_group_layouts: &[&bind_group_layout, &net_work_bind_group_layout],
    //                     push_constant_ranges: &[],
    //                 }),
    //             ),
    //             module: &compute_shader_module,
    //             entry_point: Some("main"),
    //             compilation_options: Default::default(),
    //             cache: Default::default(),
    //         });

    //     self.animation_pipe_line_cahce.panel_anim_delta_buffer = Some(panel_anim_delta_buffer);
    //     self.animation_pipe_line_cahce.pipe_bind_group_layout = Some(bind_group_layout);
    //     self.animation_pipe_line_cahce.pipe_bind_group = Some(compute_bind_group);
    //     self.animation_pipe_line_cahce.pipeline = Some(animation_compute_pipeline);
    //     self.animation_pipe_line_cahce.animtion_field_offset_buffer = Some(animtion_des_buffer);
    //     self.animation_pipe_line_cahce.gpu_animation_des_buffer = Some(animtion_gpu_des_buffer);
    //     self.animation_pipe_line_cahce.animtion_raw_buffer = Some(animtion_raw_buffer);
    //     self.animation_pipe_line_cahce.net_work_layout = Some(net_work_bind_group_layout);
    // }

    // pub fn create_net_work_compute_pipeline(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
    //     self.new_work_store = Some(NetworkStore::new(
    //         device,
    //         &self.global_layout.as_ref().unwrap(),
    //         self.animation_pipe_line_cahce
    //             .panel_anim_delta_buffer
    //             .as_ref()
    //             .unwrap(),
    //         &self.instance_buffer,
    //         self.gpu_debug.borrow().buffer.as_ref().unwrap(),
    //     ));
    // }

    pub fn interaction_cpu_trigger(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let hub = self.event_hub.clone(); // Arc clone
        // let frame =  self.global_layout.as_ref().unwrap().global_unitform_struct.frame;

        // if frame % 24 == 0 {
        //     DownloadBuffer::read_buffer(
        //         device,
        //         queue,
        //         &self.global_layout.as_ref().unwrap().debug_buffer.slice(..),
        //         move |e|{
        //             if let Ok(downloadBuffer) = e {
        //                 let bytes = downloadBuffer;

        //                 // cast bytes -> &[MyStruct]
        //                 let data: &[GpuUiDebugReadCallBack] = bytemuck::cast_slice(&bytes);

        //                 println!("DEBUG {:?}",data);
        //             }
        //         }
        //     );
        // }

        DownloadBuffer::read_buffer(
            device,
            queue,
            &self
                .interaction_pipeline_cache
                .gpu_Interaction_buffer
                .as_ref()
                .unwrap()
                .slice(..),
            move |e| {
                if let Ok(downloadBuffer) = e {
                    let bytes = downloadBuffer;

                    // cast bytes -> &[MyStruct]
                    let data: &[GpuInteractionFrame] = bytemuck::cast_slice(&bytes);

                    let old_frame = data[0];
                    let new_frame = data[1];

                    // println!("cpu记录帧 {:?}", frame);
                    // println!("显卡记录帧 {:?}", new_frame.frame);

                    if new_frame.click_id != u32::MAX {
                        hub.push(CpuPanelEvent::Click((
                            new_frame.frame, 
                            UiInteractionScope {
                                panel_id: new_frame.click_id,
                                state: new_frame.trigger_panel_state,
                            },
                        )));
                    }
                    if new_frame.drag_id != u32::MAX {
                        println("当前拖拽的目标 {:?}",new_frame.drag_id);
                        hub.push(CpuPanelEvent::Drag((
                            new_frame.frame,
                            UiInteractionScope {
                                panel_id: new_frame.drag_id,
                                state: new_frame.trigger_panel_state,
                            },
                        )));
                    }

                    if new_frame.hover_id != u32::MAX && old_frame.hover_id != new_frame.hover_id {
                        hub.push(CpuPanelEvent::Hover((
                            new_frame.frame,
                            UiInteractionScope {
                                panel_id: new_frame.hover_id,
                                state: new_frame.trigger_panel_state,
                            },
                        )));
                    }

                    if (new_frame.hover_id != old_frame.hover_id) {
                        println!(
                            "当前退出了out {:?} {:?}",
                            old_frame.hover_id, old_frame.trigger_panel_state
                        );
                        hub.push(CpuPanelEvent::OUT((
                            new_frame.frame,
                            UiInteractionScope {
                                panel_id: old_frame.hover_id,
                                state: old_frame.trigger_panel_state,
                            },
                        )));
                    }

                    // if new_frame.hover_id != u32::MAX {
                    //     hub.push(CpuPanelEvent::Hover((new_frame.frame, new_frame.hover_id)));
                    // }
                }
            },
        );
    }

    pub fn animtion_compute(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Animation Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("UI animation pass"),
                timestamp_writes: Default::default(),
            });
            // encoder.copy_buffer_to_buffer(&self.shared_buffer, 0, &self.shared_buffer_readback_buffer, 0, self.shared_buffer.size());

            cpass.set_pipeline(&self.animation_pipe_line_cahce.pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, &self.animation_pipe_line_cahce.pipe_bind_group, &[]);
            cpass.set_bind_group(1, &self.animation_pipe_line_cahce.net_work_bind_group, &[]);
            let animation_count = self
                .animation_pipe_line_cahce
                .gpu_animation_des
                .animation_count;
            let workgroups = (animation_count + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    pub fn net_work_compute(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("NetWork Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NetWork pass"),
                timestamp_writes: Default::default(),
            });
            // encoder.copy_buffer_to_buffer(&self.shared_buffer, 0, &self.shared_buffer_readback_buffer, 0, self.shared_buffer.size());

            cpass.set_pipeline(
                &self
                    .new_work_store
                    .as_ref()
                    .unwrap()
                    .pipeline
                    .as_ref()
                    .unwrap(),
            );
            cpass.set_bind_group(0, &self.new_work_store.as_ref().unwrap().bind_group, &[]);
            let net_work_count = self.new_work_store.as_ref().unwrap().panels_ref.len() as u32;
            let workgroups = (net_work_count + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
    }
    // pub fn create_compute_pipeline(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
    //     let compute_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    //         label: Some("compute_shader_module Compute Shader"),
    //         source: wgpu::ShaderSource::Wgsl(include_str!("compute_shader_module.wgsl").into()),
    //     });

    //     let ui_buffers = self.ui_relation_network_buffers.as_ref().unwrap();

    //     // === Bind Group Layout ===
    //     let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    //         label: Some("Compute Bind Group Layout"),
    //         entries: &[
    //             // 0️⃣ instance buffer
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 0,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             // 1️⃣ shared buffer
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 1,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             // 2️⃣ global uniform
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 2,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Uniform,
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             // 3️⃣ animation buffer
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 3,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             // 4️⃣ collections
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 4,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: true },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             // 5️⃣ relations
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 5,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: true },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             // 6️⃣ influences
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 6,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: true },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             // 7️⃣ ids
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 7,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: true },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 8,
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: wgpu::BindingType::Buffer {
    //                     ty: wgpu::BufferBindingType::Storage { read_only: false },
    //                     has_dynamic_offset: false,
    //                     min_binding_size: None,
    //                 },
    //                 count: None,
    //             },
    //         ],
    //     });

    //     // === Bind Group ===
    //     let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    //         label: Some("Compute Bind Group"),
    //         layout: &bind_group_layout,
    //         entries: &[
    //             wgpu::BindGroupEntry {
    //                 binding: 0,
    //                 resource: self.instance_buffer.as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 1,
    //                 resource: self.shared_buffer.as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 2,
    //                 resource: self.global_unitform_buffer.as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 3,
    //                 resource: self.animation_pipe_line_cahce.panel_anim_delta_buffer.as_ref().unwrap().as_entire_binding(),
    //             },
    //             // UiRelationNetwork buffers
    //             wgpu::BindGroupEntry {
    //                 binding: 4,
    //                 resource: self
    //                     .ui_relation_network_buffers
    //                     .as_ref()
    //                     .unwrap()
    //                     .collection_buf
    //                     .as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 5,
    //                 resource: self
    //                     .ui_relation_network_buffers
    //                     .as_ref()
    //                     .unwrap()
    //                     .relation_buf
    //                     .as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 6,
    //                 resource: self
    //                     .ui_relation_network_buffers
    //                     .as_ref()
    //                     .unwrap()
    //                     .influence_buf
    //                     .as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 7,
    //                 resource: self
    //                     .ui_relation_network_buffers
    //                     .as_ref()
    //                     .unwrap()
    //                     .id_buf
    //                     .as_entire_binding(),
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 8,
    //                 resource: self.ui_relation_network_debug_buffer.as_entire_binding(),
    //             },

    //         ],
    //     });

    //     // === Compute Pipeline ===
    //     let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    //         label: Some("Hover Compute Pipeline"),
    //         layout: Some(
    //             &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    //                 label: Some("Compute Pipeline Layout"),
    //                 bind_group_layouts: &[&bind_group_layout],
    //                 push_constant_ranges: &[],
    //             }),
    //         ),
    //         module: &compute_shader_module,
    //         entry_point: Some("main"),
    //         compilation_options: Default::default(),
    //         cache: Default::default(),
    //     });

    //     self.compute_pipeline = Some(compute_pipeline);
    //     self.compute_bind_group = Some(compute_bind_group);
    //     self.compute_bind_group_layout = Some(bind_group_layout);
    // }

    pub fn create_render_pipeline(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
    ) {
        // 创建 pipeline (Vertex + Fragment shader)
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ui.wgsl").into()),
        });

        let mut kennel_ref = self.kennel.borrow_mut();
        if kennel_ref.render_binding_resources().is_none() {
            kennel_ref.reserve_render_layers(device, 256);
        }

        let kennel_bind = kennel_ref
            .rebuild_render_bindings(device, queue)
            .expect("kennel render bindings");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UI Pipeline Layout"),
            bind_group_layouts: &[
                &self.render_bind_group_layout.clone().unwrap(),
                &self.texture_bind_group_layout.clone().unwrap(),
                &kennel_bind.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main".into(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Panel>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            // === Block 1 ===
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x2, // position
                            },
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32x2, // size
                            },
                            // === Block 2 ===
                            wgpu::VertexAttribute {
                                offset: 16,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32x2, // uv_offset
                            },
                            wgpu::VertexAttribute {
                                offset: 24,
                                shader_location: 5,
                                format: wgpu::VertexFormat::Float32x2, // uv_scale
                            },
                            // === Block 3 ===
                            wgpu::VertexAttribute {
                                offset: 32,
                                shader_location: 6,
                                format: wgpu::VertexFormat::Uint32, // z_index
                            },
                            wgpu::VertexAttribute {
                                offset: 36,
                                shader_location: 7,
                                format: wgpu::VertexFormat::Uint32, // pass_through
                            },
                            wgpu::VertexAttribute {
                                offset: 40,
                                shader_location: 8,
                                format: wgpu::VertexFormat::Uint32, // id
                            },
                            wgpu::VertexAttribute {
                                offset: 44,
                                shader_location: 9,
                                format: wgpu::VertexFormat::Uint32, // interaction
                            },
                            // === Block 4 ===
                            wgpu::VertexAttribute {
                                offset: 48,
                                shader_location: 10,
                                format: wgpu::VertexFormat::Uint32, // event_mask
                            },
                            wgpu::VertexAttribute {
                                offset: 52,
                                shader_location: 11,
                                format: wgpu::VertexFormat::Uint32, // state_mask
                            },
                            wgpu::VertexAttribute {
                                offset: 56,
                                shader_location: 12,
                                format: wgpu::VertexFormat::Float32, // transparent
                            },
                            wgpu::VertexAttribute {
                                offset: 60,
                                shader_location: 13,
                                format: wgpu::VertexFormat::Uint32, // texture_id
                            },
                            // === Block 5 ===
                            wgpu::VertexAttribute {
                                offset: 64,
                                shader_location: 14,
                                format: wgpu::VertexFormat::Uint32, // state
                            },
                            wgpu::VertexAttribute {
                                offset: 68,
                                shader_location: 15,
                                format: wgpu::VertexFormat::Uint32, // pad[0]
                            },
                            wgpu::VertexAttribute {
                                offset: 72,
                                shader_location: 16,
                                format: wgpu::VertexFormat::Uint32, // pad[1]
                            },
                            wgpu::VertexAttribute {
                                offset: 76,
                                shader_location: 17,
                                format: wgpu::VertexFormat::Uint32, // pad[2]
                            },
                        ],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main".into(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), // 或 OVER
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        self.pipeline = Some(pipeline)
    }

    pub fn create_render_bind_layout(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shared State BindGroup Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shared State BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self
                        .global_layout
                        .as_ref()
                        .unwrap()
                        .shared_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self
                        .global_layout
                        .as_ref()
                        .unwrap()
                        .global_unitform_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self
                        .global_layout
                        .as_ref()
                        .unwrap()
                        .shared_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.custom_wgsl_buffer.as_entire_binding(),
                },
            ],
        });

        self.render_bind_group = Some(bind_group);
        self.render_bind_group_layout = Some(bind_group_layout);

        // 🧱 5️⃣ 保存结果到 GPU UI 结构中
    }

    pub fn create_texture_bind_layout(&mut self, device: &Device, queue: &wgpu::Queue) {
        // 1️⃣ 上传所有 atlas 到 GPU（保持不变）
        for (_, atlas) in self.ui_texture_map.data.iter_mut() {
            atlas.upload_to_gpu(device, queue);
        }

        // 2️⃣ 按稳定顺序收集 atlas keys
        let mut atlas_keys: Vec<u32> = self.ui_texture_map.data.keys().cloned().collect();
        atlas_keys.sort_unstable();

        // 3️⃣ 按相同顺序构建 texture_views & samplers，并记录 atlas -> slot 映射
        let mut texture_views = Vec::new();
        let mut samplers = Vec::new();
        let mut atlas_slot_map: HashMap<u32, u32> = HashMap::new(); // atlas_id -> slot_index

        for (slot, atlas_key) in atlas_keys.iter().enumerate() {
            if let Some(atlas) = self.ui_texture_map.data.get(atlas_key) {
                if let (Some(view), Some(sampler)) = (&atlas.texture_view, &atlas.sampler) {
                    atlas_slot_map.insert(*atlas_key, slot as u32);
                    texture_views.push(view.clone()); // clone the view handle
                    samplers.push(sampler.clone());
                } else {
                    // atlas 未上传成功或没有视图/采样器，跳过（或处理错误）
                    eprintln!("atlas {} missing view/sampler, skipping", atlas_key);
                }
            }
        }

        if texture_views.is_empty() || samplers.is_empty() {
            eprintln!("没有找到纹理或采样器，BindGroup 将不会创建");
            return;
        }

        // 4️⃣ 创建 BindGroupLayout（binding 0: texture array, 1: sampler array, 2: subimage storage）
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("UI Texture BindGroup Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: Some(std::num::NonZeroU32::new(texture_views.len() as u32).unwrap()),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: Some(std::num::NonZeroU32::new(samplers.len() as u32).unwrap()),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        self.texture_bind_group_layout = Some(bind_group_layout);

        let mut gpu_sub_image_structs: Vec<GpuUiTextureInfo> = vec![
            GpuUiTextureInfo::default();
            self.ui_texture_map.curr_ui_texture_info_index
                as usize
        ];

        for atlas_key in atlas_keys.iter() {
            if let Some(atlas) = self.ui_texture_map.data.get(atlas_key) {
                let slot = atlas_slot_map.get(atlas_key).cloned().unwrap_or(0u32);

                for sub_image in atlas.map.values() {
                    let mut gpu_struct = sub_image.to_gpu_struct();
                    gpu_struct.parent_index = slot;

                    // 按 UiTextureInfo.index 放入正确位置
                    gpu_sub_image_structs[sub_image.index as usize] = gpu_struct;
                }
            }
        }

        // 6️⃣ 创建 GPU buffer（storage）
        let gpu_sub_image_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UI SubImage Structs"),
            contents: bytemuck::cast_slice(&gpu_sub_image_structs),
            usage: wgpu::BufferUsages::STORAGE, // 如需 CPU 更新，可加 COPY_DST
        });
        let texture_view_refs: Vec<&wgpu::TextureView> = texture_views.iter().collect();
        let samplers_refs: Vec<&wgpu::Sampler> = samplers.iter().collect();
        // 7️⃣ 创建 bind group，注意这里传入 texture_views/samplers 的 slice 要与 layout 的 count 一致
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("UI Texture Array BindGroup"),
            layout: self.texture_bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&texture_view_refs),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::SamplerArray(&samplers_refs),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu_sub_image_buffer.as_entire_binding(),
                },
            ],
        });

        self.ui_texture_bind_group = Some(bind_group);
        println!(
            "成功创建 UI Texture BindGroup，atlas_count: {}, sub_images: {}",
            texture_views.len(),
            gpu_sub_image_structs.len()
        );
    }

    pub fn create_surface(device: &wgpu::Device, format: wgpu::TextureFormat) -> TextureView {
        let ui_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("UI Offscreen"),
            size: wgpu::Extent3d {
                width: 1920,
                height: 1080,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[format],
        });
        ui_texture.create_view(&Default::default())
    }

    pub fn update_frame(&mut self, queue: &wgpu::Queue, device: &Device) {
        let global = self.global_layout.as_mut().unwrap();
        let mut unitfrom_struct = global.global_unitform_struct.borrow_mut();

        unitfrom_struct.frame = unitfrom_struct.frame.wrapping_add(1);
        self.interaction_pipeline_cache
            .gpu_interaction_struct
            .as_mut()
            .unwrap()
            .copy_interaction_swap_frame(
                device,
                queue,
                &self
                    .interaction_pipeline_cache
                    .gpu_Interaction_buffer
                    .as_ref()
                    .unwrap(),
                unitfrom_struct.frame,
            );
        // 计算 mouse_state 在结构体中的偏移量

        // 取出单个字段的字节切片

        // queue.write_buffer(
        //     & self.global_layout.as_ref().unwrap().global_unitform_buffer,
        //     0,
        //     bytemuck::cast_slice(&[ self.global_layout.as_ref().unwrap().global_unitform_struct]),
        // );
        queue.submit([]); // ✅ 强制 flush
    }

    fn gpu_change_ui_state_with_transtion(
        &mut self,
        queue: &wgpu::Queue,
        device: &Device,
        state: &StateTransition,
    ) {
        let panel_size = std::mem::size_of::<Panel>();
        let offset = (state.panel_id * panel_size as u32) as wgpu::BufferAddress;
        let state_offset = offset_of!(Panel, state) as wgpu::BufferAddress;
        let new_state_sign = state.new_state.0;
        print!(
            "当前状态转变为 {:?}  {:?} 偏移量位{:?}",
            state.panel_id, state.new_state, state_offset
        );
        queue.write_buffer(
            &self.instance_buffer,
            offset + state_offset,
            bytemuck::bytes_of(&new_state_sign),
        );
    }

    pub fn update_network_dirty_entries(&mut self, queue: &wgpu::Queue, device: &Device) {
        let store = self.new_work_store.as_mut().unwrap();
        store.upload_dirty_to_gpu_batch(queue);
    }

    fn gpu_change_ui_collection_state_raw(
        &mut self,
        queue: &wgpu::Queue,
        device: &Device,
        state_mask: PanelCollectionState,
        panel_id: u32,
    ) {
        let panel_size = std::mem::size_of::<Panel>();
        let offset = (panel_id * panel_size as u32) as wgpu::BufferAddress;
        let state_offset = offset_of!(Panel, collection_state) as wgpu::BufferAddress;
        queue.write_buffer(
            &self.instance_buffer,
            offset + state_offset,
            bytemuck::bytes_of(&state_mask.bits()),
        );
    }

    pub fn change_static_panel_interaction(
        &mut self,
        queue: &wgpu::Queue,
        panel_id: u32,
        state: PanelInteraction,
    ) {
        let offsets =
            (panel_id as wgpu::BufferAddress * std::mem::size_of::<Panel>() as wgpu::BufferAddress);
        let field_offsets = offset_of!(Panel, interaction) as wgpu::BufferAddress;
        queue.write_buffer(
            &self.instance_buffer,
            offsets + field_offsets,
            bytemuck::bytes_of(&state.bits()),
        );
    }

    pub fn change_static_panel_texture(
        &mut self,
        queue: &wgpu::Queue,
        panel_id: u32,
        texture_id: u32,
    ) {
        let offsets =
            (panel_id as wgpu::BufferAddress * std::mem::size_of::<Panel>() as wgpu::BufferAddress);
        let field_offsets = offset_of!(Panel, texture_id) as wgpu::BufferAddress;
        queue.write_buffer(
            &self.instance_buffer,
            offsets + field_offsets,
            bytemuck::bytes_of(&texture_id),
        );
    }

    pub fn change_panel_kennel_des_idx(
        &mut self,
        queue: &wgpu::Queue,
        panel_id: u32,
        kennel_des_idx: u32,
    ) {
        let offsets =
            (panel_id as wgpu::BufferAddress * std::mem::size_of::<Panel>() as wgpu::BufferAddress);
        let field_offsets = offset_of!(Panel, kennel_des_id) as wgpu::BufferAddress;
        println!(
            "写入的kennel_des_idx panelid:{panel_id}=>{:?}",
            kennel_des_idx
        );
        queue.write_buffer(
            &self.instance_buffer,
            offsets + field_offsets,
            bytemuck::bytes_of(&kennel_des_idx),
        );
    }

    pub fn process_global_events(&mut self, queue: &wgpu::Queue, device: &Device) {
        for ev in self.global_hub.poll() {
            match ev {
                mile_api::ModuleEvent::KennelPushResultReadDes(parmas) => {
                    let panel_id = parmas.idx;
                    println!("{:?}", parmas);
                    // 证明是一个顶点回读
                    if (parmas._ty & ModuleEventType::Vertex.bits()) != 0 {
                        let link_plan_id = parmas.data;
                        // // 计算偏移并写入缓冲区
                        self.change_panel_kennel_des_idx(queue, panel_id, link_plan_id);
                    }

                    if (parmas._ty & ModuleEventType::Frag.bits()) != 0 {
                        let link_plan_id = parmas.data;
                        // // 计算偏移并写入缓冲区
                        self.change_panel_kennel_des_idx(queue, panel_id, link_plan_id);
                    }
                }
                _ => {}
            }
        }
    }

    #[inline]
    pub fn process_ui_events(&mut self, queue: &wgpu::Queue, device: &Device) {
        for ev in self.event_hub.poll() {
            match ev {
                CpuPanelEvent::Frag((panel_id, frag_event)) => {
                    println!("收到自定义片段动画事件 {:?}", frag_event);
                    for (callbacks) in self
                        .panel_interaction_trigger
                        .frag_callbacks
                        .get_mut(&frag_event)
                    {
                        for cb in callbacks.iter_mut() {
                            cb(panel_id);
                        }
                    }
                }
                CpuPanelEvent::Vertex((panel_id, vertex)) => {
                    println!("收到自定义顶点动画事件 {:?}", vertex);
                    for (callbacks) in self
                        .panel_interaction_trigger
                        .vertex_callbacks
                        .get_mut(&vertex)
                    {
                        for cb in callbacks.iter_mut() {
                            cb(panel_id);
                        }
                    }
                }
                CpuPanelEvent::StateTransition(state_event) => {
                    let emit = self.event_hub.sender.clone();

                    if state_event.state_config_des.is_open_frag {
                        let _ = emit.send(CpuPanelEvent::Frag((
                            state_event.panel_id,
                            UiInteractionScope {
                                panel_id: state_event.panel_id,
                                state: state_event.new_state.0,
                            },
                        )));
                    }

                    if state_event.state_config_des.is_open_vertex {
                        let _ = emit.send(CpuPanelEvent::Vertex((
                            state_event.panel_id,
                            UiInteractionScope {
                                panel_id: state_event.panel_id,
                                state: state_event.new_state.0,
                            },
                        )));
                    }

                    println!("当前UI元素改变了状态 {:?}", state_event.new_state);
                    self.gpu_change_ui_state_with_transtion(queue, device, &state_event);

                    let state_config = state_event.state_config_des;
                    let mut mask = PanelInteraction::DEFUALT;
                    for c in &state_config.open_api {
                        mask |= (*c).into(); // 将每个 Call 转成对应 bitflags 并合并
                    }

                    self.change_static_panel_interaction(queue, state_event.panel_id, mask);

                    if let Some(texture_name) = state_config.texture_id {
                        let raw_image_info = self
                            .ui_texture_map
                            .get_index_by_path(&texture_name)
                            .unwrap();
                        self.change_static_panel_texture(
                            queue,
                            state_event.panel_id,
                            raw_image_info.index,
                        )
                    }

                    if let Some(pos) = state_config.pos {
                        self.add_animation(
                            queue,
                            device,
                            state_event.panel_id,
                            TransformAnimFieldInfo {
                                field_id: (PanelField::POSITION_X | PanelField::POSITION_Y).bits(),
                                start_value: vec![0.0; 2],
                                target_value: vec![pos.x, pos.y],
                                duration: 1.0,
                                easing: EasingMask::LINEAR,
                                op: AnimOp::SET,
                                hold: 1,
                                delay: 0.0,
                                loop_count: 0,
                                ping_pong: 0,
                                on_complete: 1,
                            },
                        );
                    }

                    if let Some(size) = state_config.size {
                        self.add_animation(
                            queue,
                            device,
                            state_event.panel_id,
                            TransformAnimFieldInfo {
                                field_id: (PanelField::SIZE_X | PanelField::SIZE_Y).bits(),
                                start_value: vec![0.0; 2],
                                target_value: vec![size.x, size.y],
                                duration: 1.0,
                                easing: EasingMask::LINEAR,
                                op: AnimOp::SET,
                                hold: 1,
                                delay: 0.0,
                                loop_count: 0,
                                ping_pong: 0,
                                on_complete: 1,
                            },
                        );
                    }
                }

                CpuPanelEvent::SpecielAnim((panel_id, anim_info)) => {
                    self.add_animation(queue, device, panel_id, anim_info);
                }
                CpuPanelEvent::Click((Frame, scope)) => {
                    println!("点击 panel {:?}", scope);

                    for (callbacks) in self
                        .panel_interaction_trigger
                        .click_callbacks
                        .get_mut(&scope)
                    {
                        for cb in callbacks.iter_mut() {
                            cb(scope.panel_id);
                        }
                    }

                    for (callbacks) in self
                        .panel_interaction_trigger
                        .entry_callbacks
                        .get_mut(&scope)
                    {
                        for cb in callbacks.iter_mut() {
                            cb(scope.panel_id);
                        }
                    }

                    // let anim = TransformAnim {
                    //     field_id: (PanelField::SIZE_X | PanelField::SIZE_Y).bits(),
                    //     field_len: 1,
                    //     start_value: 0.0,
                    //     end_value: 100.0,
                    //     easing_mask: 1,
                    //     _pad1: 0,
                    //     duration: 1.0,
                    //     elapsed: 0.0,
                    //     instance_id: id,
                    //     op: AnimOp::ADD.bits(),
                    //     _pad2:0,
                    //     _pad3:0,
                    //     last_applied:0.0,
                    //     _pad4: [0u32;3],
                    // };
                    // self.add_animation(queue, anim);

                    // self.add_animation(
                    //     queue,
                    //     TransformAnim {
                    //         field_id: PanelField::TRANSPARENT.bits(),
                    //         field_len: 1,
                    //         start_value: 1.,
                    //         end_value: 0.,
                    //         easing_mask: 1,
                    //         _pad1: 0,
                    //         duration: 0.33,
                    //         elapsed: 0.0,
                    //         instance_id: id,
                    //         op: AnimOp::SET.bits(),
                    //     },
                    // );

                    // println!("当前给ui元素加了一个动画,元素为={}", id);
                }
                CpuPanelEvent::Drag((id, scope)) => {
                    for (callbacks) in self
                        .panel_interaction_trigger
                        .drag_callbacks
                        .get_mut(&scope)
                    {
                        for cb in callbacks.iter_mut() {
                            cb(scope.panel_id);
                        }
                    }
                }

                CpuPanelEvent::Hover((id, scope)) => {
                    println!("当前hover {:?}", scope);
                    for (callbacks) in self
                        .panel_interaction_trigger
                        .hover_callbacks
                        .get_mut(&scope)
                    {
                        for cb in callbacks.iter_mut() {
                            cb(scope.panel_id);
                        }
                    }
                }

                CpuPanelEvent::OUT((id, scope)) => {
                    for (callbacks) in self.panel_interaction_trigger.out_callbacks.get_mut(&scope)
                    {
                        for cb in callbacks.iter_mut() {
                            cb(scope.panel_id);
                        }
                    }
                }
                CpuPanelEvent::NetWorkTransition(net_work_evnt) => {
                    //     println!("当前有panel id 触发了关系网络事件 {:?}",net_work_evnt);

                    println!("网络事件 {:?}", net_work_evnt);
                    let to_net_work_collection_id =
                        net_work_evnt.state_config_des.insert_collection;

                    if let Some(exit_config) = net_work_evnt.state_config_des.exit_collection {
                        match exit_config {
                            ExitCollectionOp::ExitAllOldCollection => {}
                            ExitCollectionOp::ExitRangeOldCollection(items) => {}
                        }
                    }

                    if let Some(collection_id) = to_net_work_collection_id {
                        let store = self.new_work_store.as_mut().unwrap();
                        store.add_panel_to_collection(net_work_evnt.panel_id, collection_id);
                        println!(
                            "有面板加入了集合{:?} {:?}",
                            net_work_evnt.panel_id, collection_id
                        );
                    }

                    if (to_net_work_collection_id.is_some()) {
                        self.gpu_change_ui_collection_state_raw(
                            queue,
                            device,
                            PanelCollectionState::EntryCollection,
                            net_work_evnt.panel_id,
                        );
                        if net_work_evnt.state_config_des.immediately_anim
                            && to_net_work_collection_id.is_some()
                        {
                            self.add_animation(
                                queue,
                                device,
                                net_work_evnt.panel_id,
                                TransformAnimFieldInfo {
                                    field_id: (PanelField::AttchCollection).bits(),
                                    start_value: vec![to_net_work_collection_id.unwrap() as f32],
                                    target_value: vec![0.0],
                                    duration: 0.0,
                                    easing: EasingMask::LINEAR,
                                    op: AnimOp::Collectionimmediately,
                                    hold: 0,
                                    delay: 0.0,
                                    loop_count: 0,
                                    ping_pong: 0,
                                    on_complete: 0,
                                },
                            );
                        } else {
                            println!("前往的ID {:?}", to_net_work_collection_id);

                            self.add_animation(
                                queue,
                                device,
                                net_work_evnt.panel_id,
                                TransformAnimFieldInfo {
                                    field_id: (PanelField::AttchCollection).bits(),
                                    start_value: vec![to_net_work_collection_id.unwrap() as f32],
                                    target_value: vec![0.0],
                                    duration: 1.0,
                                    easing: EasingMask::LINEAR,
                                    op: AnimOp::CollectionTransfrom,
                                    hold: 1,
                                    delay: 0.0,
                                    loop_count: 0,
                                    ping_pong: 0,
                                    on_complete: 1,
                                },
                            );
                        }
                    }

                    //    let net_work = self.ui_net_work.borrow();
                    //    let ids_struct = net_work.net_work_panel_ids.borrow();
                }
                _ => {}
            }
        }
    }

    pub fn add_animation(
        &mut self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        panel_id: u32,
        anim_info: TransformAnimFieldInfo,
    ) {
        let offset_des = anim_info.split_write_field(panel_id);
        let mut index = 0;
        let mut to_remove: Vec<(AnimtionIdx, PanelIdx)> = Vec::new();

        for new_field in &offset_des {
            let new_mask = new_field.field_id;

            // 遍历 hash，找到与新动画字段冲突的旧动画
            for ((old_idx, panel_id_ref), &old_mask) in
                self.animation_pipe_line_cahce.field_cahce.hash.iter()
            {
                if (old_mask & new_mask) != 0 && *panel_id_ref == panel_id {
                    let offset = offset_of!(AnimtionFieldOffsetPtr, death);
                    // 标记旧动画死亡
                    // self.animation_des[old_idx].death = 1;
                    println!("old_idx 老的动画立即停止了 {} {}", old_mask, panel_id);
                    to_remove.push((*old_idx, panel_id));
                    queue.write_buffer(
                        self.animation_pipe_line_cahce
                            .animtion_field_offset_buffer
                            .as_ref()
                            .unwrap(),
                        *old_idx as u64 * std::mem::size_of::<AnimtionFieldOffsetPtr>() as u64
                            + offset as u64,
                        bytemuck::bytes_of(&1),
                    );
                }
            }

            // // 找到新动画写入位置
            // let new_idx = self.animation_des.len();
            // self.animation_des.push(*new_field);
            let id = self
                .animation_pipe_line_cahce
                .gpu_animation_des
                .animation_count
                + index;
            self.animation_pipe_line_cahce
                .field_cahce
                .hash
                .entry((id, panel_id))
                .and_modify(|m| *m |= new_mask)
                .or_insert(new_mask);
            index += 1;
            // 可选：立即写入 GPU buffer
            // self.write_anim_to_gpu(queue, device, new_idx);
        }

        for idx in to_remove {
            self.animation_pipe_line_cahce
                .field_cahce
                .hash
                .remove(&(idx));
        }

        queue.write_buffer(
            &self
                .animation_pipe_line_cahce
                .animtion_field_offset_buffer
                .as_ref()
                .unwrap(),
            self.animation_pipe_line_cahce
                .animtion_field_offset_ptr_point,
            bytemuck::cast_slice(offset_des.as_slice()),
        );

        // queue.write_buffer(
        //     &self.animation_pipe_line_cahce.animtion_raw_buffer.as_ref().unwrap(),
        //     self.animation_pipe_line_cahce.animtion_raw_buffer_point,
        //     bytemuck::cast_slice(raw_vec.as_slice())
        // );

        self.animation_pipe_line_cahce
            .animtion_field_offset_ptr_point +=
            (offset_des.len() * std::mem::size_of::<AnimtionFieldOffsetPtr>()) as u64;
        self.animation_pipe_line_cahce
            .gpu_animation_des
            .animation_count += offset_des.len() as u32;

        queue.write_buffer(
            &self
                .animation_pipe_line_cahce
                .gpu_animation_des_buffer
                .as_ref()
                .unwrap(),
            0,
            bytemuck::bytes_of(&self.animation_pipe_line_cahce.gpu_animation_des),
        );
    }

    pub fn update_mouse_state(&mut self, queue: &wgpu::Queue, mouse_state: MouseState) {
        // if (mouse_state.bits() & (MouseState::LEFT_UP.bits() | MouseState::RIGHT_UP.bits())) != 0 {
        //     self.global_unitform_struct.press_duration = 0.0;
        //     let offset = offset_of!(GlobalUniform, press_duration) as wgpu::BufferAddress;
        //     println!("press_duration offset = {}", offset);
        //     queue.write_buffer(
        //         &self.global_unitform_buffer,
        //         offset,
        //         bytemuck::bytes_of(&self.global_unitform_struct.press_duration),
        //     );
        // }
        let global = self.global_layout.as_mut().unwrap();
        let mut unitfrom_struct = global.global_unitform_struct.borrow_mut();

        unitfrom_struct.mouse_state = mouse_state.bits();

        let offset = offset_of!(GlobalUniform, mouse_state) as wgpu::BufferAddress;
        queue.write_buffer(
            &global.global_unitform_buffer,
            offset,
            bytemuck::bytes_of(&unitfrom_struct.mouse_state),
        );
    }

    pub fn mouse_press_tick_post(&mut self, queue: &wgpu::Queue) {
        let global = self.global_layout.as_mut().unwrap();
        let mut unitfrom_struct = global.global_unitform_struct.borrow_mut();

        let pressed = (unitfrom_struct.mouse_state
            & (MouseState::LEFT_DOWN.bits() | MouseState::RIGHT_DOWN.bits()))
            != 0;
        if !pressed {
            // 弹起，重置按下时间
            unitfrom_struct.press_duration = 0.0;
        }
        let offset = offset_of!(GlobalUniform, press_duration) as wgpu::BufferAddress;
        // 写入 GPU buffer
        queue.write_buffer(
            &global.global_unitform_buffer,
            offset,
            bytemuck::bytes_of(&unitfrom_struct.press_duration),
        );
    }

    pub fn global_unifrom_clear_tick(&mut self, queue: &wgpu::Queue) {
        let offset_id = offset_of!(GlobalUniform, click_layout_id) as wgpu::BufferAddress;
        let offset_z = offset_of!(GlobalUniform, click_layout_z) as wgpu::BufferAddress;
        // 写入 GPU buffer
        queue.write_buffer(
            &self.global_layout.as_ref().unwrap().global_unitform_buffer,
            offset_id,
            bytemuck::bytes_of(&0),
        );
        queue.write_buffer(
            &self.global_layout.as_ref().unwrap().global_unitform_buffer,
            offset_z,
            bytemuck::bytes_of(&0),
        );
    }

    pub fn mouse_press_tick_first(&mut self, queue: &wgpu::Queue) {
        let global = self.global_layout.as_mut().unwrap();
        let mut unitfrom_struct = global.global_unitform_struct.borrow_mut();

        let pressed = (unitfrom_struct.mouse_state
            & (MouseState::LEFT_DOWN.bits() | MouseState::RIGHT_DOWN.bits()))
            != 0;

        if pressed {
            // 持续按下累加时间
            unitfrom_struct.press_duration += 0.033;
        }

        let offset = offset_of!(GlobalUniform, press_duration) as wgpu::BufferAddress;
        // 写入 GPU buffer
        queue.write_buffer(
            &global.global_unitform_buffer,
            offset,
            bytemuck::bytes_of(&unitfrom_struct.press_duration),
        );
    }

    pub fn update_mouse_pos(&mut self, queue: &wgpu::Queue, mouse_pos: [f32; 2]) {
        let global = self.global_layout.as_mut().unwrap();
        let mut unitfrom_struct = global.global_unitform_struct.borrow_mut();
        // 更新 CPU 端状态
        unitfrom_struct.mouse_pos = mouse_pos;
        let offset = offset_of!(GlobalUniform, mouse_pos) as wgpu::BufferAddress;
        // 写入 GPU buffer
        queue.write_buffer(
            &global.global_unitform_buffer,
            offset,
            bytemuck::bytes_of(&unitfrom_struct.mouse_pos),
        );
    }
    pub fn read_img(&mut self, path: &Path) -> Option<UiTextureInfo> {
        // 1️⃣ 打开图片
        let img = ImageReader::open(path).ok()?.decode().ok()?.to_rgba8();
        let (orig_w, orig_h) = img.dimensions();
        println!("🖼️ 加载图片 {:?}, 大小: {}x{}", path, orig_w, orig_h);

        // 添加边距后的尺寸
        let img_width = orig_w + PADDING * 2;
        let img_height = orig_h + PADDING * 2;

        // 2️⃣ 选择可容纳图片的 atlas
        let atlas_id = if let Some((&id, _)) =
            self.ui_texture_map.data.iter_mut().find(|(_, atlas)| {
                let mut x = atlas.next_x;
                let mut y = atlas.next_y;
                let mut row_height = atlas.row_height;

                // 模拟多次换行，直到找到放得下的位置或确定放不下
                loop {
                    if x + img_width > atlas.width {
                        x = 0;
                        y += row_height;
                        row_height = 0;
                    }

                    if y + img_height > atlas.height {
                        return false; // 放不下
                    }

                    if x + img_width <= atlas.width {
                        return true; // 找到可以放的位置
                    }
                }
            }) {
            id
        } else {
            // 没有合适的 atlas，新建一个
            let atlas_size = DEFAULT_ATLAS_SIZE;
            let new_id = self.ui_texture_map.data.len() as u32;
            println!(
                "🆕 创建新的 Atlas #{} 尺寸 {}x{}",
                new_id, atlas_size, atlas_size
            );

            let atlas = TextureAtlas {
                width: atlas_size,
                height: atlas_size,
                data: RgbaImage::new(atlas_size, atlas_size),
                map: HashMap::new(),
                next_x: 0,
                next_y: 0,
                row_height: 0,
                texture: None,
                texture_view: None,
                sampler: None,
                index: new_id,
            };

            self.ui_texture_map.data.insert(new_id, atlas);
            new_id
        };

        // 3️⃣ 获取可用 atlas
        let atlas = self.ui_texture_map.data.get_mut(&atlas_id).unwrap();

        // 4️⃣ 计算插入坐标（支持自动换行）
        let (mut x, mut y) = (atlas.next_x, atlas.next_y);
        if x + img_width > atlas.width {
            x = 0;
            y += atlas.row_height;
            atlas.next_y = y;
            atlas.row_height = 0;
        }

        // 检查是否溢出
        if y + img_height > atlas.height {
            println!("⚠️ Atlas #{} 已满，无法放入 {:?}", atlas.index, path);
            return None;
        }

        // overlay
        image::imageops::overlay(&mut atlas.data, &img, x.into(), y.into());

        // 更新游标
        atlas.next_x = x + img_width;
        atlas.row_height = atlas.row_height.max(img_height);

        // 7️⃣ 计算UV（去除 padding）
        let uv_min = [
            (x + PADDING) as f32 / atlas.width as f32,
            (y + PADDING) as f32 / atlas.height as f32,
        ];
        let uv_max = [
            (x + PADDING + orig_w) as f32 / atlas.width as f32,
            (y + PADDING + orig_h) as f32 / atlas.height as f32,
        ];

        // 8️⃣ 生成或复用 UiTextureInfo
        let tex_name = path.file_name()?.to_string_lossy().to_string();
        if let Some(existing) = atlas.map.get(&tex_name) {
            println!("♻️ 已存在纹理 {:?} (atlas #{})", tex_name, atlas.index);
            return Some(existing.clone());
        }

        let tex_index = self.ui_texture_map.curr_ui_texture_info_index;
        self.ui_texture_map.curr_ui_texture_info_index += 1;

        let ui_info = UiTextureInfo {
            index: tex_index,
            uv_min,
            uv_max,
            path: tex_name.clone(),
            parent_index: atlas.index,
        };
        println!(
            "插入 {:?}: pos=({}, {}), next_x={}, row_height={}, atlas_size={}x{}",
            path, x, y, atlas.next_x, atlas.row_height, atlas.width, atlas.height
        );
        // 9️⃣ 注册缓存
        atlas.map.insert(tex_name.clone(), ui_info.clone());
        self.ui_texture_map.path_to_index.insert(
            tex_name.clone(),
            ImageRawInfo {
                index: tex_index,
                width: img_width,
                height: img_height,
            },
        );

        println!(
            "✅ 插入纹理 {:?} → index:{} Atlas:{} 坐标:({}, {})",
            tex_name, tex_index, atlas.index, x, y
        );

        Some(ui_info)
    }
    // pub fn add_instances_attach(
    //     &mut self,
    //     queue: &wgpu::Queue,
    //     instances: &[Panel],
    // ) {
    //     for ins in instances {
    //         let offset = ins.id as wgpu::BufferAddress
    //             * std::mem::size_of::<PanelAnimDelta>() as wgpu::BufferAddress;

    //         PanelAnimDelta::default().write_to_buffer(queue, &self.panel_anim_delta_buffer, offset);
    //     }
    // }

    pub fn add_instances_grid(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rows: u32,
        cols: u32,
        start_pos: [f32; 2],
        size: [f32; 2],
        gap: [f32; 2],
        texture_id: u32,
    ) {
        let panel_size_bytes = std::mem::size_of::<Panel>() as wgpu::BufferAddress;

        for row in 0..rows {
            for col in 0..cols {
                // 计算全局屏幕坐标
                let x = start_pos[0] + col as f32 * (size[0] + gap[0]);
                let y = start_pos[1] + row as f32 * (size[1] + gap[1]);

                // --- 用实例池长度作为 id，使 id 和 GPU buffer index 对齐 ---
                let buffer_index = self.instances.len();

                let interaction = if buffer_index == 0 {
                    PanelInteraction::CLICKABLE.bits() // 只给 0 号 panel 可拖拽
                } else {
                    PanelInteraction::CLICKABLE.bits() // 其他 panel 不可拖拽
                };

                let instance = Panel {
                    id: buffer_index as u32, // <-- 关键修改
                    position: [x, y],
                    size,
                    uv_offset: [0.0, 0.0],
                    uv_scale: [1.0, 1.0],
                    z_index: 0,
                    pass_through: 0,
                    interaction: interaction,
                    event_mask: 0,
                    state_mask: 0,
                    transparent: 1.0,
                    texture_id,
                    state: 0,
                    collection_state: 0,
                    kennel_des_id: 0,
                    pad_1: 0,
                };

                // 添加到 CPU 实例池
                self.instances.push(instance);
                self.num_instances = self.instances.len() as u32;

                // 写入 GPU buffer
                let offset = buffer_index as wgpu::BufferAddress * panel_size_bytes;
                queue.write_buffer(
                    &self.instance_buffer,
                    offset,
                    bytemuck::bytes_of(&self.instances[buffer_index]),
                );
                // self.add_instances_attach(queue,&[instance]);
            }
        }
        println!(
            "当前面板 CPU id 数据: {:?}",
            self.instances.iter().map(|e| { e.id }).collect::<Vec<_>>()
        );
    }

    pub fn add_instance(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        mut instance: Panel,
    ) -> u32 {
        // 分配唯一 ID
        instance.id = self.instance_pool_index;
        let curr_id = instance.id;

        // 添加到 CPU 端实例池
        self.instances.push(instance);
        self.num_instances = self.instances.len() as u32;

        // --- 写入 GPU ---
        let panel_size = std::mem::size_of::<Panel>() as wgpu::BufferAddress;
        let index = (self.instances.len() - 1) as wgpu::BufferAddress;
        let offset = index * panel_size;

        // 写入新增实例
        queue.write_buffer(
            &self.instance_buffer,
            offset,
            bytemuck::bytes_of(&self.instances[self.instances.len() - 1]),
        );

        self.instance_pool_index += 1;
        curr_id
    }

    pub fn clear_curr_frame(
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        offset: wgpu::BufferAddress,
        size: wgpu::BufferAddress,
    ) {
    }

    /// 更新实例（批量）
    pub fn update_instances(&mut self, queue: &wgpu::Queue, instances: &[Panel]) {
        self.instances = instances.to_vec();
        self.num_instances = instances.len() as u32;
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.instances),
        );
    }
}

// impl Computeable for GpuUi {
//     fn compute(&self, cpass: &mut wgpu::ComputePass<'_>) {
//         cpass.set_pipeline(&self.compute_pipeline.as_ref().unwrap());
//         cpass.set_bind_group(0, &self.compute_bind_group, &[]);

//         let workgroups = (self.num_instances + 63) / 64;
//         cpass.dispatch_workgroups(workgroups, 1, 1);
//     }

//     fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
//         DownloadBuffer::read_buffer(device, queue, &self.debug_buffer.slice(..), move |e| {
//             if let Ok(downloadBuffer) = e {
//                 let bytes = downloadBuffer;

//                 // cast bytes -> &[MyStruct]
//                 let data: &[GpuUiRelationDebugReadCallBack] = bytemuck::cast_slice(&bytes);

//                 // println!("当前回读 {:?}",data);

//             }
//         });
//         // let mut compute_buffer = self.ui_relation_network_buffers.as_ref().unwrap();
//         // DownloadBuffer::read_buffer(device, queue, &compute_buffer.collection_buf.slice(..), move |e| {
//         //     if let Ok(downloadBuffer) = e {
//         //         let bytes = downloadBuffer;

//         //         // cast bytes -> &[MyStruct]
//         //         let data: &[GpuUiCollection] = bytemuck::cast_slice(&bytes);

//         //         // 现在 data 就是 GPU buffer 里所有 MyStruct
//         //         for data in data {
//         //             println!("当前的 collection_buf = {:?}",data)
//         //         }
//         //     }
//         // });
//         // // DownloadBuffer::read_buffer(device, queue, &compute_buffer.id_buf.slice(..), move |e| {
//         // //     if let Ok(downloadBuffer) = e {
//         // //         let bytes = downloadBuffer;

//         // //         // cast bytes -> &[MyStruct]
//         // //         let data: &[UiRelationGpuBuffers] = bytemuck::cast_slice(&bytes);

//         // //         // 现在 data 就是 GPU buffer 里所有 MyStruct
//         // //         for anim in data {
//         // //         }
//         // //     }
//         // // });
//         // DownloadBuffer::read_buffer(device, queue, &compute_buffer.influence_buf.slice(..), move |e| {
//         //     if let Ok(downloadBuffer) = e {
//         //         let bytes = downloadBuffer;

//         //         // cast bytes -> &[MyStruct]
//         //         let data: &[GpuUiInfluence] = bytemuck::cast_slice(&bytes);

//         //         // 现在 data 就是 GPU buffer 里所有 MyStruct
//         //         for data in data {
//         //             println!("当前的 GpuUiInfluence = {:?}",data)
//         //         }
//         //     }
//         // });
//     }
// }

impl Renderable for GpuUi {
    fn render<'a>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        pass: &mut RenderPass<'a>,
    ) {
        {
            pass.set_pipeline(&self.pipeline.as_ref().unwrap());
            let indirects_count = self.indirects_len; // 你需要记录 update_indirect_buffer 生成了多少 draw
            for i in 0..indirects_count {
                let offset = (i as u64 * std::mem::size_of::<DrawIndexedIndirect>() as u64);
                pass.draw_indexed_indirect(&self.indirects_buffer, offset);
            }
            pass.set_bind_group(0, &self.render_bind_group, &[]);
            pass.set_bind_group(1, &self.ui_texture_bind_group, &[]);
            pass.set_bind_group(
                2,
                &self
                    .kennel
                    .borrow()
                    .render_binding_resources()
                    .as_ref()
                    .unwrap()
                    .bind_group,
                &[],
            );
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
        }
        // queue.submit(Some(encoder.finish()));
    }

    fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.gpu_debug.borrow_mut().debug(device, queue);
    }
}
