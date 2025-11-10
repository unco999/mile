// Cargo.toml: bytemuck = "1", bitflags = "2", ahash = "0.8"

use std::collections::HashMap;

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

/**
 * 每个instance 负责一个字构造
 * 每个面板一个字 最主要的是动态大小的这个偏移索引
 * 所谓的索引 就是我们 字形印版 在  sdf纹理里面的坐标
 * 我们只要一个连续的索引在buffer中的位置 
 * 如果 这个缓冲区满了 我们把长期有效记录 更新到持久缓存的位置
 */
pub struct GpuGlyphInfo<'a>{
    pub offset_range:&'a GpuOffsetRange
}


/**
 * 这个其实也是  字形文件的key  为什么呢  
 * 因为他的buffer的偏移是唯一的
 * 每个buffer 片段可以拿这个做唯一标识
 */
#[derive(Hash,PartialEq, Eq, PartialOrd, Ord)]
pub struct GpuOffsetRange{
    pub offset_start:wgpu::BufferAddress,
    pub end_start:wgpu::BufferAddress
}

pub struct FontSDFIndexDynamicArea<'a>{
      // 字形信息缓冲区（环形缓冲区）
    pub glyph_buffer: HashMap<GpuOffsetRange,GpuGlyphInfo<'a>>,  
      // 持久缓存区
    pub persistent_cache: HashMap<GpuOffsetRange,GpuGlyphInfo<'a>>,
    
    pub head_idx: usize,   // 热缓存的头部索引
    pub tail_idx: usize,   // 热缓存的尾部索引
    pub hot_cache_capacity: usize,  // 热缓存区的容量
    pub valid_size: usize, // 当前有效数据的大小
}

// ---- GPU 端可直接写入的样式（全部数值化）----
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct FontStyleGpu {
    // 16 bytes
    pub color: [f32; 4], // RGBA
    // 16 bytes
    pub size_px: f32,        // 字号
    pub line_height_px: f32, // 行高
    pub weight: u32,         // 100..900
    pub family_id: u32,      // 字体族 ID

    // 16 bytes
    pub file_id: u32,         // 字体文件 ID
    pub style_kind: u32,      // FontStyleKind as u32
    pub text_align: u32,      // TextAlign as u32
    pub decoration_bits: u32, // TextDecoration bits
}
// 总计 48 字节（按 16 对齐，适合 std140/std430）

// ---- 每个字形一条实例数据 ----

// 72 字节，按 16 对齐（std430 ok）

// ---- 文本块到字形区段的映射 ----
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct TextGpu {
    pub glyph_count: u32,     // 字形数量
    pub style_index: u32,     // 指向 FontStyleGpu 数组的索引
    pub wrap_width: f32,      // 换行宽度（像素）
    pub bounds_min: [f32; 2], // AABB min（用于裁剪/对齐）
    pub bounds_max: [f32; 2], // AABB max
}
// 48 字节
