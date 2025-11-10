// Cargo.toml: bytemuck = "1", bitflags = "2", ahash = "0.8"

use std::{collections::HashMap, fmt::Error, marker::PhantomData};

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
use wgpu::BufferAddress;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuGlyphInfo<'a> {
    pub is_persistent: bool,  // 标记是否为长期有效字形
    pub offset_range: &'a GpuOffsetRange,  // 引用 GpuOffsetRange，生命周期一致
}

#[repr(C)]
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct GpuOffsetRange {
    pub offset_start: BufferAddress,  // 起始偏移
    pub end_start: BufferAddress,     // 结束偏移
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
pub struct SdfInfo{
    texture_x:u32,
    texture_y:u32
}

pub struct FontSDFIndexDynamicArea<'a> {
    pub area:Vec<[u32;2]>,
    // 字形信息缓冲区（环形缓冲区）
    pub glyph_buffer: HashMap<&'a GpuOffsetRange, GpuGlyphInfo<'a>>,

    pub movable_type_printing:HashMap<char,SdfInfo>,
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

pub enum TextError {
    GlyphNotFound {
        character: char,
    },
}
impl<'a> FontSDFIndexDynamicArea<'a> {
    // 创建新的 FontSDFIndexDynamicArea
    pub fn new(hot_cache_capacity: usize) -> Self {
        Self {
            movable_type_printing:HashMap::new(),
            area:vec![],
            glyph_buffer: HashMap::with_capacity(hot_cache_capacity),
            persistent_cache: HashMap::new(),
            head_idx: 0,
            tail_idx: 0,
            hot_cache_capacity,
            valid_size: 0,
        }
    }

    /**
     * 他发一个字符串序列过来  然后我们来找到字符对应在gpu sdf里面的位置
     * 然后把这个变成一个新的range  range指定的位置是buffer
     */
    fn font_text_entry(&mut self, str: &'a str) -> Result<Vec<SdfInfo>, TextError> {
        let mut font_sdf_info: Vec<SdfInfo> = Vec::new();
        for c in str.chars() {
            let item = self.movable_type_printing
                .get(&c)
                .ok_or_else(|| TextError::GlyphNotFound { character: c.into() })?;
            font_sdf_info.push(item.clone());
        }
        Ok(font_sdf_info)
    }

    fn batch(&mut self, str: &'a str) -> Result<(), TextError> {
        let sdf_info = self.font_text_entry(str)?;
        for info in sdf_info{

        }


        Ok(())
    }

    /**
     * 插入字符在sdf里面的位置
     */
    pub fn font_record(&mut self,char:&char,x:u32,y:u32){
        self.movable_type_printing.insert(*char, SdfInfo{
            texture_x: x,
            texture_y: y,
        });
    }

    // 更新缓冲区并处理热缓存区满时的数据迁移
    pub fn update_glyph_buffer(&mut self, new_glyphs: &[GpuGlyphInfo<'a>]) {
        for glyph in new_glyphs {
            if self.valid_size >= self.hot_cache_capacity {
                // 缓存区满时，将长期有效的记录移到持久缓存
                self.move_to_persistent_cache();
                self.clear_hot_cache(); // 清空热缓存区
            }

            // 插入新数据到热缓存区，避免克隆
            self.glyph_buffer.insert(glyph.offset_range, *glyph); // 直接插入引用，不需要克隆
            self.tail_idx = (self.tail_idx + 1) % self.hot_cache_capacity;
            self.valid_size += 1;
        }
    }

    // 将长期有效的字形信息移动到持久缓存区
    fn move_to_persistent_cache(&mut self) {
        let mut keys_to_move = Vec::new();

        // 遍历 `glyph_buffer`，找到需要迁移到持久缓存区的数据
        for (key, glyph) in &self.glyph_buffer {
            if glyph.is_persistent {
                // 将长期有效的数据插入持久缓存区
                self.persistent_cache.insert(*key, *glyph);
                keys_to_move.push(*key);
            }
        }

        // 从热缓存区移除已迁移的数据
        for key in keys_to_move {
            self.glyph_buffer.remove(&key);
        }

        // 更新有效数据大小
        self.valid_size = self.glyph_buffer.len();
    }

    // 清空热缓存区
    fn clear_hot_cache(&mut self) {
        self.glyph_buffer.clear();  // 清空热缓存区
        self.head_idx = 0;  // 重置头部索引
        self.tail_idx = 0;  // 重置尾部索引
        self.valid_size = 0;  // 清空有效数据大小
    }

    // 获取持久缓存的字形
    pub fn get_persistent_cache(&self) -> Vec<GpuGlyphInfo<'a>> {
        self.persistent_cache
            .values()
            .cloned()
            .collect::<Vec<_>>()  // 将持久缓存区的内容转换为 `Vec`
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

#[test]
fn test(){
    let mut dynamic_area = FontSDFIndexDynamicArea::new(128);
    dynamic_area.update_glyph_buffer(&[
        GpuGlyphInfo{
            is_persistent: true,
            offset_range:&GpuOffsetRange{
                offset_start: 0,
                end_start: 128,
            },
        },
        GpuGlyphInfo{
            is_persistent: true,
            offset_range:&GpuOffsetRange{
                offset_start: 128,
                end_start: 256,
            },
        },
    ]);
    dynamic_area.move_to_persistent_cache();
    println!("当前的字形索引情况 {:?}",dynamic_area.persistent_cache)
}