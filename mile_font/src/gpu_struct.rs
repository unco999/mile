// Cargo.toml: bytemuck = "1", bitflags = "2", ahash = "0.8"

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
