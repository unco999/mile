use bytemuck::{cast_slice, Pod, Zeroable};

use crate::structs::{AnimOp, EasingMask};


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}
// Quad 顶点数据（屏幕空间，-0.5 ~ 0.5）
const QUAD_VERTICES: &[Vertex] = &[
    Vertex {
        pos: [0.0, 0.0],
        uv: [0.0, 0.0],
    },
    Vertex {
        pos: [1.0, 0.0],
        uv: [1.0, 0.0],
    },
    Vertex {
        pos: [1.0, 1.0],
        uv: [1.0, 1.0],
    },
    Vertex {
        pos: [0.0, 1.0],
        uv: [0.0, 1.0],
    },
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GlobalUiState {
    // Mouse position
    pub mouse_pos: [f32; 2], // 8 bytes

    // Mouse button state mask
    pub mouse_state: u32, // 4 bytes
    pub _pad0: u32,       // 4 bytes padding 对齐

    // Hover panel ID
    pub hover_id: u32,      // 4 bytes, atomic
    pub hover_blocked: u32, // 4 bytes, atomic

    // Padding for alignment before hover_pos
    pub _pad1: [u32; 2], // 8 bytes

    // Hover position
    pub hover_pos: [f32; 2], // 8 bytes

    // Current depth under mouse
    pub current_depth: u32, // 4 bytes
    pub _pad2: u32,         // 4 bytes padding

    // Clicked panel ID (最后一次点击)
    pub click_id: u32,      // 4 bytes
    pub click_blocked: u32, // 4 bytes

    // Drag panel ID
    pub drag_id: u32,      // 4 bytes
    pub drag_blocked: u32, // 4 bytes

    // History panel ID
    pub history_id: u32, // 4 bytes
    pub _pad3: u32,      // 4 bytes padding

    // Final padding to 64 bytes
    pub _pad4: [u32; 2], // 8 bytes
}

const QUAD_INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

pub(super) fn quad_vertex_bytes() -> &'static [u8] {
    cast_slice(QUAD_VERTICES)
}

pub(super) fn quad_index_bytes() -> &'static [u8] {
    cast_slice(QUAD_INDICES)
}

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
    pub fragment_shader_id: u32, // 12, 补齐到 16
    pub vertex_shader_id: u32,   // 12, 补齐到 16

    // === 16-byte 块 6 ===
    pub color: [f32; 4],

    // === 16-byte 块 7 ===
    pub border_color: [f32; 4],

    // === 16-byte 块 8 ===
    pub border_width: f32,
    pub border_radius: f32,
    pub pad_border: [f32; 2],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
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
    pub event_point: [f32; 2],
    pub _pad5: [u32; 4],
}

impl GpuInteractionFrame {
    pub fn empty(frame: u32) -> Self {
        Self {
            frame,
            drag_id: u32::MAX,
            hover_id: u32::MAX,
            click_id: u32::MAX,
            mouse_pos: [0.0, 0.0],
            trigger_panel_state: 0,
            _pad1: 0,
            mouse_state: 0,
            _pad2: [0; 3],
            drag_delta: [0.0, 0.0],
            _pad3: [0.0, 0.0],
            pinch_delta: 0.0,
            pass_through_depth: 0,
            event_point: [0.0, 0.0],
            _pad5: [0; 4],
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
    pub fn zeroed() -> Self {
        let mut cache = Self::default();
        cache.pre.drag_id = u32::MAX;
        cache.pre.hover_id = u32::MAX;
        cache.pre.click_id = u32::MAX;
        cache.curr.drag_id = u32::MAX;
        cache.curr.hover_id = u32::MAX;
        cache.curr.click_id = u32::MAX;
        cache
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
pub struct GpuUiDebugReadCallBack {
    pub floats: [f32; 32],
    pub uints: [u32; 32],
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
