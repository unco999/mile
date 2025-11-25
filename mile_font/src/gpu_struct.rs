// Cargo.toml: bytemuck = "1", bitflags = "2", ahash = "0.8"

use std::{
    collections::{HashMap, btree_map::Range},
    fmt::Error,
    marker::PhantomData,
    sync::Arc,
};

use crate::DEFAULT_FONT_PATH;
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
pub enum FontStyleKind {
    Normal = 0,
    Italic = 1,
    Oblique = 2,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
pub enum TextAlign {
    Left = 0,
    Center = 1,
    Right = 2,
    Justify = 3,
}

impl Into<TextAlign> for u32 {
    fn into(self) -> TextAlign {
        match self {
            0 => TextAlign::Left,
            1 => TextAlign::Center,
            2 => TextAlign::Right,
            3 => TextAlign::Justify,
            _ => TextAlign::Center,
        }
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug)]
    pub struct TextDecoration: u32 {
        const NONE       = 0;
        const UNDERLINE  = 1 << 0;
        const OVERLINE   = 1 << 1;
        const STRIKE     = 1 << 2; // line-through
        // 预留位
    }
}

use mile_api::prelude::_ty::PanelId;
/**
 * 每个instance 负责一个字构造
 * 每个面板一个字 最主要的是动态大小的这个偏移索引
 * 所谓的索引 就是我们 字形印版 在  sdf纹理里面的坐标
 * 我们只要一个连续的索引在buffer中的位置
 * 如果 这个缓冲区满了 我们把长期有效记录 更新到持久缓存的位置
 */
use wgpu::BufferAddress;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuGlyphInfo<'a> {
    pub is_persistent: bool,              // 标记是否为长期有效字形
    pub offset_range: &'a GpuOffsetRange, // 引用 GpuOffsetRange，生命周期一致
}

#[repr(C)]
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct GpuOffsetRange {
    pub offset_start: BufferAddress, // 起始偏移
    pub end_start: BufferAddress,    // 结束偏移
}

impl GpuOffsetRange {
    pub fn new(start: u64, end: u64) -> GpuOffsetRange {
        GpuOffsetRange {
            offset_start: start,
            end_start: end,
        }
    }
}

#[derive(Clone)]
pub struct SdfInfo {
    char_with_texture: u32,
    texture_x: u32,
    texture_y: u32,
}

pub struct FontSDFIndexDynamicArea<'a> {
    pub area: Vec<[u32; 2]>,
    // 字形信息缓冲区（环形缓冲区）
    pub glyph_buffer: HashMap<&'a GpuOffsetRange, GpuGlyphInfo<'a>>,

    pub movable_type_printing: HashMap<char, SdfInfo>,
    // 持久缓存区
    pub persistent_cache: HashMap<&'a GpuOffsetRange, GpuGlyphInfo<'a>>,

    // 热缓存的头部和尾部索引
    pub head_idx: usize,
    pub tail_idx: usize,

    // 热缓存区的容量
    pub hot_cache_capacity: usize,

    // 当前有效数据的大小
    pub valid_size: usize,
}

#[derive(Debug)]
pub struct CpuText {
    //原始数据
    pub raw: Arc<str>,
    pub slice: Vec<GpuChar>,
    pub link_panel_quad: PanelId,
}

/**字体style */
#[derive(Debug)]
pub struct FontStyle {
    pub font_size: u32,
    pub font_file_path: Arc<str>,
    pub font_color: [f32; 4],
    pub font_weight: u32,
    pub font_line_height: u32,
    pub first_weight: f32,
    pub panel_size: [f32; 2], // 0~1
    pub text_align: TextAlign,
}

impl Default for FontStyle {
    fn default() -> Self {
        Self {
            font_size: 24,
            font_file_path: Arc::from(DEFAULT_FONT_PATH),
            font_color: [1.0, 1.0, 1.0, 1.0],
            font_weight: 0,
            font_line_height: 0,
            first_weight: 0.0,
            panel_size: [1.0, 1.0],
            text_align: TextAlign::Center,
        }
    }
}

/// Layout directives that accompany each glyph instance.
pub const GPU_CHAR_LAYOUT_FLAG_LINE_BREAK_BEFORE: u32 = 0x1;
pub const GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_SHIFT: u32 = 8;
pub const GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_MASK: u32 = 0x00ff_ff00;
pub const GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_MAX: u32 =
    GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_MASK >> GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_SHIFT;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct GpuChar {
    pub char_index: u32,
    pub gpu_text_index: u32,
    pub panel_index: u32,
    // relative order inside its owning text (0-based)
    pub self_index: u32,
    // per-glyph metrics (TTF units)
    pub glyph_advance_width: u32,
    pub glyph_left_side_bearing: i32,
    pub glyph_ver_advance: u32,
    pub glyph_ver_side_bearing: i32,
    /// Bitmask of `GPU_CHAR_LAYOUT_FLAG_*`.
    pub layout_flags: u32,
}
#[derive(Clone, Debug)]
pub struct GpuText {
    pub sdf_char_index_start_offset: u32, //gpu sdf_index offset 描述了怎么在统一buffer里面取gpu char
    pub sdf_char_index_end_offset: u32,   //这个实际上是 GpuChar这个gpu结构体的索引
    pub font_size: f32,
    pub size: u32,
    pub color: [f32; 4],
    /// Owning panel id to allow removal even if bookkeeping maps are missing
    pub panel: u32,
    // text origin (pixels or logical units depending on pipeline)
    pub position: [f32; 2],
    /// Optional line height in pixels (0 = derive from glyph metrics)
    pub line_height: f32,
    /// First-line indentation in pixels/em as provided by runtime
    pub first_line_indent: f32,
    /// Horizontal alignment hint from layout
    pub text_align: TextAlign,
    /// Bounding box (width, height) in pixels computed on CPU for alignment.
    pub text_bounds: [f32; 2],
}
