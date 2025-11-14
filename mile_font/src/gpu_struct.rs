// Cargo.toml: bytemuck = "1", bitflags = "2", ahash = "0.8"

use std::{collections::HashMap, fmt::Error, marker::PhantomData, sync::Arc};

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
    char_with_texture:u32,
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
pub struct CpuText{
    //原始数据
    pub raw:Arc<str>,
    pub slice:Vec<GpuChar>,
    pub link_panel_quad:PanelId
}

/**字体style */
#[derive(Debug)]
pub struct FontStyle {
    pub font_size: u32,
    pub font_file_path: &'static str,
    pub font_color: [f32; 4],
    pub font_weight: u32,
    pub font_line_height: u32,
}


#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable,Debug)]
pub struct GpuChar{
    pub char_index:u32,
    pub gpu_text_index:u32,
    pub panel_index:u32,
    // relative order inside its owning text (0-based)
    pub self_index:u32,
    // per-glyph metrics (TTF units)
    pub glyph_advance_width: u32,
    pub glyph_left_side_bearing: i32,
    pub glyph_ver_advance: u32,
    pub glyph_ver_side_bearing: i32,
}
pub struct GpuText{
    pub sdf_char_index_start_offset:u32, //gpu sdf_index offset 描述了怎么在统一buffer里面取gpu char
    pub sdf_char_index_end_offset:u32,  //这个实际上是 GpuChar这个gpu结构体的索引
    pub font_size:f32,
    pub size:u32,
    pub color:[f32;4],
    // text origin (pixels or logical units depending on pipeline)
    pub position:[f32;2],
}
