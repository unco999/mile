use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use image::{DynamicImage, GenericImage, GenericImageView, ImageReader, RgbaImage};
use std::{collections::HashMap, path::Path};
use wgpu::{BufferAddress, TextureFormat, util::DeviceExt};

use crate::mui::Panel;

pub trait GpuPanelAttach: Pod + Zeroable + Send + Sync + 'static {
    /// 生成该 Panel 的扩展数据
    fn generate_for_panel(&self, idx: u32, panel: &Panel) -> Self;
}


/**
 * UI组件与组件之间的通信面板;
 */
pub struct UiMessage<Target, Payload> {
    pub payload: Payload,
    _marker: std::marker::PhantomData<Target>,
}

impl<Target, Payload> UiMessage<Target, Payload> {
    pub fn new(payload: Payload) -> Self {
        Self { payload, _marker: std::marker::PhantomData }
    }
}
pub struct PanelAttachContext<E: GpuPanelAttach> {
    panels: Vec<Panel>,
    exts: Vec<E>, // 每种扩展类型都是同一具体类型
}

impl<E: GpuPanelAttach> PanelAttachContext<E> {
    pub fn create_merged_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let mut merged_data = Vec::with_capacity(self.panels.len());

        for (i, panel) in self.panels.iter().enumerate() {
            for ext in &self.exts {
                let data = ext.generate_for_panel(i as u32, panel);
                merged_data.push(data);
            }
        }

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuPanelMergedBuffer"),
            contents: bytemuck::cast_slice(&merged_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }
}

bitflags! {
    pub struct KennelUiBlend: u32 {
        const Normal = 1 << 0;           // 1st bit
        const Additive = 1 << 1;          // 2nd bit
        const Multiplicative = 1 << 2;    // 3rd bit
        const Screen = 1 << 3;            // 4th bit
        const Difference = 1 << 4;        // 5th bit
        const Weighted = 1 << 5;          // 6th bit (可以扩展)
    }
}

bitflags! {
    /// Interaction Mask
    pub struct EntryState: u32 {
        const Enter = 0b00000001;
        const Exit    = 0b00000010;
        const Finish  = 0b00000100;
        const Tick  = 0b00001000;
    }
}

bitflags! {
    /// Interaction Mask
    pub struct PanelInteraction: u32 {
        const DEFUALT = 0b00000000;
        const VISIBLE    = 0b00000001;
        const CLICKABLE  = 0b00000010;
        const DRAGGABLE  = 0b00000100;
        const HOVER = 0b00001000;
        const Out = 0b00010000;
    }
}

bitflags! {
    /// Interaction Mask
    pub struct PanelInteractionHold: u32 {
        const Absolute = 0b00000000;
        const Current    = 0b00000010;
    }
}

bitflags! {
    /// Event Response Mask
    pub struct PanelEvent: u32 {
        const Defualt = 0b00000000;
        const MOUSE_CLICK  = 0b00000001;
        const MOUSE_HOVER  = 0b00000010;
        const MOUSE_SCROLL = 0b00000100;
        const KEY_PRESS    = 0b00001000;
    }
}

bitflags! {
    /// Panel State Mask
    pub struct PanelState: u32 {
        const Defualt = 0b00000000;
        const HOVER    = 0b00000001;
        const FOCUS    = 0b00000010;
        const SELECTED = 0b00000100;
        const ACTIVE   = 0b00001000;
    }
}

bitflags! {
    /// Panel State Mask
    pub struct PanelCollectionState: u32 {
        const EntryCollection = 0b00010000;
        const InCollection = 0b00100000;
        const ExitCollection = 0b01000000;
    }
}

bitflags::bitflags! {
    /// Mouse Button State Mask
    pub struct MouseState: u32 {
        const DEFAULT     = 0b0000_0000; // 无状态

        // 左键状态
        const LEFT_DOWN   = 0b0001_0000; // 左键按下
        const LEFT_UP     = 0b0010_0000; // 左键弹起

        // 右键状态
        const RIGHT_DOWN  = 0b0100_0000; // 右键按下
        const RIGHT_UP    = 0b1000_0000; // 右键弹起
    }
}

bitflags::bitflags! {
    /// Easing / interpolation type mask
    #[derive(Debug,Clone)]
    pub struct EasingMask: u32 {
        const LINEAR       = 0b0000_0001; // 线性插值
        const IN_QUAD      = 0b0000_0010; // 二次缓入
        const OUT_QUAD     = 0b0000_0100; // 二次缓出
        const IN_OUT_QUAD  = 0b0000_1000; // 二次缓入缓出
        const IN_CUBIC     = 0b0001_0000; // 三次缓入
        const OUT_CUBIC    = 0b0010_0000; // 三次缓出
        const IN_OUT_CUBIC = 0b0100_0000; // 三次缓入缓出
        const ELASTIC      = 0b1000_0000; // 弹性缓动
    }
}

bitflags::bitflags! {
    #[derive(Debug,Clone)]
    pub struct AnimOp: u32 {
        const SET = 0b0000_0001;   // 直接赋值
        const ADD = 0b0000_0010;   // 累加
        const MUL = 0b0000_0100;   // 乘法
        const LERP = 0b0000_1000;  // 线性插值
        const CollectionTransfrom = 0b0001_0000;  // 线性插值
        const Collectionimmediately = 0b0010_0000;  // 线性插值
    }
}

bitflags::bitflags! {
    #[derive(Hash,PartialEq, Eq)]
    pub struct PanelField: u32 {
        const POSITION_X     = 0b0000_0000_0001; // index 0
        const POSITION_Y     = 0b0000_0000_0010; // index 1
        const SIZE_X         = 0b0000_0000_0100; // index 2
        const SIZE_Y         = 0b0000_0000_1000; // index 3
        const UV_OFFSET_X    = 0b0000_0001_0000; // index 4
        const UV_OFFSET_Y    = 0b0000_0010_0000; // index 5
        const UV_SCALE_X     = 0b0000_0100_0000; // index 6
        const UV_SCALE_Y     = 0b0000_1000_0000; // index 7
        const TRANSPARENT    = 0b0001_0000_0000; // index 14
        const AttchCollection = 0b0010_0000_0000; // index 14
        const PREPOSITION_X  = 0b0100_0000_0000; // index 14
        const PREPOSITION_Y  = 0b1000_0000_0000; // index 14
        const ALL            = 0b1111_1111_1111; // index 14
        const Not            = 0b0000_0000_0000; // index 14
    }
}

bitflags::bitflags! {
    /// 集合采样策略（可组合）
    pub struct CollectionSampling: u32 {
        const FIRST    = 0b0000_0001; // 取首元素
        const AVERAGE  = 0b0000_0010; // 取平均值
        // 可以以后扩展其他采样策略，例如最大值、最小值等
    }
}

bitflags::bitflags! {
    pub struct RelLayoutMask: u32 {
        // 排列方向
        const HORIZONTAL       = 1 << 0;
        const VERTICAL         = 1 << 1;
        const GRID             = 1 << 2;
        const Ring            = 1 << 3;

        // 排列顺序
        const ORDER_FORWARD    = 1 << 4;
        const ORDER_REVERSE    = 1 << 5;
        const ORDER_RANDOM     = 1 << 6;

        // 布局行为
        const Defualt     = 1 << 7;

    }
}

bitflags::bitflags! {
    /// 集合采样策略（可组合）
    pub struct UiInfluenceType: u32 { //影响方式
        const Relatively    = 0b0000_0001; // 相对的
        const Absolute  = 0b0000_0010; // 绝对的
        // 可以以后扩展其他采样策略，例如最大值、最小值等
    }
}

pub type CollectionId = u32;

#[derive(Clone, Debug)]
pub struct UiCollection {
    pub collection_id: CollectionId,
    pub len: u32,
    pub sampling: u32, // 采样策略s
    pub ids: Vec<u32>,
}

/**
 * 关系影响两个集合之间的UI操作
 * 是从源头集合影响到target集合
 * 比如当源头集合的Influence监听的属性发生变化
 * 我们按照影响组件 来进行变化 可能是相对的 可能是绝对的
 */
#[derive(Clone, Debug)]
pub struct UiRelation {
    pub source_collection_id: CollectionId,
    pub target_collection_id: CollectionId,
    pub relation_mask: Vec<UiInfluence>,
}

#[derive(Clone, Debug)]
pub struct UiInfluence {
    pub field: u32,
    pub weight: f32,
    pub influence_type: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct GpuUiCollection {
    pub start_index: u32,
    pub len: u32,
    pub sampling: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct GpuUiInfluence {
    pub field: u32,
    pub weight: f32,
    pub influence_type: u32,
    pub reserved: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct GpuUiIdInfo {
    panel_id: u32,     // panel 的唯一 id
    is_source: u32,    // 0 = 普通 panel, 1 = source panel
    relation_idx: u32, // 对应的 relation buffer 索引，如果不是 source 可填 0xFFFFFFFF
    padding: u32,      // 对齐到 16 字节
}

#[derive(Clone, Default)]
pub struct UiRelationNetwork {
    pub collections: Vec<GpuUiCollection>,
    pub influences: Vec<GpuUiInfluence>,
    pub ids: Vec<GpuUiIdInfo>,
    pub next_collection_id: u32,
    pub id_offset: u32,
    pub influence_offset: u32,
}

#[derive(Debug, Clone)]
pub struct UiRelationGpuBuffers {
    pub collection_buf: wgpu::Buffer,
    pub relation_buf: wgpu::Buffer,
    pub influence_buf: wgpu::Buffer,
    pub id_buf: wgpu::Buffer,
}

// impl UiRelationNetwork {
//     pub fn global_init(device: &wgpu::Device) -> UiRelationGpuBuffers {
//         // 每种结构体大小
//         let collection_size = 1024 * std::mem::size_of::<GpuUiCollection>() as wgpu::BufferAddress;
//         let influence_size  = 1024 * std::mem::size_of::<GpuUiInfluence>()  as wgpu::BufferAddress;
//         let id_size         = 1024 * std::mem::size_of::<GpuUiIdInfo>()     as wgpu::BufferAddress;

//         // 辅助闭包：创建空 buffer
//         let make_empty_buffer = |label: &str, size: wgpu::BufferAddress| {
//             device.create_buffer(&wgpu::BufferDescriptor {
//                 label: Some(label),
//                 size,
//                 usage: wgpu::BufferUsages::STORAGE
//                      | wgpu::BufferUsages::COPY_DST
//                      | wgpu::BufferUsages::COPY_SRC,
//                 mapped_at_creation: false,
//             })
//         };

//         UiRelationGpuBuffers {
//             collection_buf: make_empty_buffer("UiRelationNetwork::collections", collection_size),
//             influence_buf:  make_empty_buffer("UiRelationNetwork::influences",  influence_size),
//             id_buf:         make_empty_buffer("UiRelationNetwork::ids",         id_size),
//         }
//     }

//     // pub fn write_collection_buf(&self, device: &wgpu::Device,offset:BufferAddress){

//     // }

//     pub fn create_gpu_buffers(&self, device: &wgpu::Device) -> UiRelationGpuBuffers {
//         // 辅助闭包：创建 buffer
//         let make_buffer = |label: &str, data: &[u8], usage: wgpu::BufferUsages| {
//             device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                 label: Some(label),
//                 contents: data,
//                 usage,
//             })
//         };

//         UiRelationGpuBuffers {
//             collection_buf: make_buffer(
//                 "UiRelationNetwork::collections",
//                 bytemuck::cast_slice(&self.collections),
//                 wgpu::BufferUsages::STORAGE
//                     | wgpu::BufferUsages::COPY_DST
//                     | wgpu::BufferUsages::COPY_SRC,
//             ),
//             relation_buf: make_buffer(
//                 "UiRelationNetwork::relations",
//                 bytemuck::cast_slice(&self.relations),
//                 wgpu::BufferUsages::STORAGE
//                     | wgpu::BufferUsages::COPY_DST
//                     | wgpu::BufferUsages::COPY_SRC,
//             ),
//             influence_buf: make_buffer(
//                 "UiRelationNetwork::influences",
//                 bytemuck::cast_slice(&self.influences),
//                 wgpu::BufferUsages::STORAGE
//                     | wgpu::BufferUsages::COPY_DST
//                     | wgpu::BufferUsages::COPY_SRC,
//             ),
//             id_buf: make_buffer(
//                 "UiRelationNetwork::ids",
//                 bytemuck::cast_slice(&self.ids),
//                 wgpu::BufferUsages::STORAGE
//                     | wgpu::BufferUsages::COPY_DST
//                     | wgpu::BufferUsages::COPY_SRC,
//             ),
//         }
//     }

// pub fn add_relation(&mut self, relation: &UiRelation) {
//     let influence_start = self.influence_offset;
//     let influence_count = relation.relation_mask.len() as u32;

//     for inf in &relation.relation_mask {
//         self.influences.push(GpuUiInfluence {
//             field: inf.field,
//             weight: inf.weight,
//             influence_type: inf.influence_type,
//             reserved: 0,
//         });
//     }

//     self.influence_offset += influence_count;

//     let id_start = self.id_offset;
//     self.ids.extend_from_slice(&relation.ids);
//     let id_count = relation.ids.len() as u32;
//     self.id_offset += id_count;

//     self.relations.push(GpuUiRelation {
//         source_collection_id: relation.source_collection_id,
//         target_collection_id: relation.target_collection_id,
//         influence_start,
//         influence_count,
//         id_start,
//         id_count,
//         reserved: 0,
//         padding: 0,
//     });
// }

// pub fn add_collection(&mut self, ui_collection: &UiCollection) -> u32 {
//     let collection_id = self.next_collection_id;
//     self.next_collection_id += 1;

//     let start_index = self.id_offset;
//     self.ids.extend_from_slice(&ui_collection.ids);
//     self.id_offset += ui_collection.ids.len() as u32;

//     self.collections.push(GpuUiCollection {
//         start_index,
//         len: ui_collection.ids.len() as u32,
//         sampling: ui_collection.sampling,
//         reserved: 0,
//     });

//     collection_id
// }
// pub fn from_ui_data(
//     &mut self,
//     collections: &[UiCollection],
//     relations: &[UiRelation],
//     queue: &wgpu::Queue,
//     gpu_buffers: &UiRelationGpuBuffers,
// ) -> Self {
//     let mut gpu_collections = Vec::new();
//     let mut gpu_relations = Vec::new();
//     let mut gpu_influences = Vec::new();
//     let mut gpu_id_infos = Vec::new();

//     // --- 序列化集合 ---
//     for col in collections {
//         let start_index = self.id_offset + self.ids.len() as u32 + gpu_id_infos.len() as u32;
//         gpu_collections.push(GpuUiCollection {
//             start_index,
//             len: col.len,
//             sampling: col.sampling,
//             reserved: 0,
//         });
//     }

//     // --- 序列化关系 ---
//     for (rel_idx, rel) in relations.iter().enumerate() {
//         let influence_start = self.influence_offset + self.influences.len() as u32 + gpu_influences.len() as u32;

//         gpu_influences.extend(rel.relation_mask.iter().map(|inf| GpuUiInfluence {
//             field: inf.field,
//             weight: inf.weight,
//             influence_type: inf.influence_type,
//             reserved: 0,
//         }));

//                 // --- 构建 GPU ID Info ---
//         let target_col = &collections[rel.target_collection_id as usize];
//         for &panel_id in &target_col.ids {
//             println!("当前的panel_id {:?}",panel_id);
//             gpu_id_infos.push(GpuUiIdInfo {
//                 panel_id,
//                 is_source: 0,
//                 relation_idx: u32::MAX,
//                 padding: 0,
//             });
//         }

//         let source_col = &gpu_collections[rel.source_collection_id as usize];
//         let id_start = source_col.start_index;
//         let id_count = source_col.len;

//         println!("当前的唯一关系的ID start {:?}",id_start);

//         gpu_relations.push(GpuUiRelation {
//             source_collection_id: rel.source_collection_id,
//             target_collection_id: rel.target_collection_id,
//             influence_start,
//             influence_count: rel.relation_mask.len() as u32,
//             id_start,
//             id_count,
//             reserved: 0,
//             padding: 0,
//         });

//         let source_col = &collections[rel.source_collection_id as usize];
//         println!("当前的源头集合 {:?}",source_col);
//         for &panel_id in &source_col.ids {
//             println!("当前的panel_id {:?}",panel_id);
//             gpu_id_infos.push(GpuUiIdInfo {
//                 panel_id,
//                 is_source: 1,
//                 relation_idx: rel_idx as u32,
//                 padding: 0,
//             });
//         }
//     }

//     // --- 写入 GPU buffer 按偏移 ---
//     if !gpu_collections.is_empty() {
//         let offset_bytes = self.collections.len() as wgpu::BufferAddress * std::mem::size_of::<GpuUiCollection>() as u64;
//         queue.write_buffer(
//             &gpu_buffers.collection_buf,
//             offset_bytes,
//             bytemuck::cast_slice(&gpu_collections),
//         );
//     }

//     if !gpu_relations.is_empty() {
//         let offset_bytes = self.relations.len() as wgpu::BufferAddress * std::mem::size_of::<GpuUiRelation>() as u64;
//         queue.write_buffer(
//             &gpu_buffers.relation_buf,
//             offset_bytes,
//             bytemuck::cast_slice(&gpu_relations),
//         );
//     }

//     if !gpu_influences.is_empty() {
//         let offset_bytes = self.influences.len() as wgpu::BufferAddress * std::mem::size_of::<GpuUiInfluence>() as u64;
//         queue.write_buffer(
//             &gpu_buffers.influence_buf,
//             offset_bytes,
//             bytemuck::cast_slice(&gpu_influences),
//         );
//         self.influence_offset += gpu_influences.len() as u32;
//     }

//     if !gpu_id_infos.is_empty() {
//         let offset_bytes = self.ids.len() as wgpu::BufferAddress * std::mem::size_of::<GpuUiIdInfo>() as u64;
//         queue.write_buffer(
//             &gpu_buffers.id_buf,
//             offset_bytes,
//             bytemuck::cast_slice(&gpu_id_infos),
//         );
//     }

//     // --- 更新 CPU 数据 ---
//     self.collections.extend_from_slice(&gpu_collections);
//     self.relations.extend_from_slice(&gpu_relations);
//     self.influences.extend_from_slice(&gpu_influences);
//     self.ids.extend_from_slice(&gpu_id_infos);

//     self.clone()
// }
// }

#[derive(Clone, Debug)]
pub struct UiTextureInfo {
    pub index: u32,
    pub parent_index: u32,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub path: String,
}

impl UiTextureInfo {
    pub fn to_gpu_struct(&self) -> GpuUiTextureInfo {
        GpuUiTextureInfo {
            index: self.index,
            uv_min: [self.uv_min[0], self.uv_min[1], 0.0, 0.0],
            uv_max: [self.uv_max[0], self.uv_max[1], 0.0, 0.0],
            parent_index: self.parent_index,
            _pad: [0u32; 2],
        }
    }
}

/// 集合间关系
#[derive(Debug, Clone)]
pub struct Relation {
    pub source_collection: u32, // 来源集合ID
    pub target_collection: u32, // 目标集合ID
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
pub struct GpuUiTextureInfo {
    pub index: u32,        // 4
    pub parent_index: u32, // 4
    pub _pad: [u32; 2],    // 8
    pub uv_min: [f32; 4],  // 16 (vec2 + padding)
    pub uv_max: [f32; 4],  // 16 (vec2 + padding)
}

#[derive(Clone)]
pub struct TextureAtlasSet {
    pub data: HashMap<u32, TextureAtlas>,
    pub curr_ui_texture_info_index: u32,
    pub path_to_index: HashMap<String, ImageRawInfo>,
}
#[derive(Clone)]
pub struct ImageRawInfo {
    pub index: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone)]
/// 动态纹理图集
///
pub struct TextureAtlas {
    pub width: u32,
    pub height: u32,
    pub data: RgbaImage, // CPU 大图
    pub map: HashMap<String, UiTextureInfo>,
    pub next_x: u32,
    pub next_y: u32,
    pub row_height: u32,
    pub texture: Option<wgpu::Texture>,
    pub texture_view: Option<wgpu::TextureView>,
    pub sampler: Option<wgpu::Sampler>,
    pub index: u32,
}

impl TextureAtlasSet {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            curr_ui_texture_info_index: 0,
            path_to_index: HashMap::new(),
        }
    }

    pub fn get_path_by_index(&self, index: u32) -> Option<String> {
        self.path_to_index.iter().find_map(|(k, v)| {
            if v.index == index {
                Some(k.clone())
            } else {
                None
            }
        })
    }

    /// 根据路径获取索引（若不存在则返回 None）
    pub fn get_index_by_path(&self, path: &str) -> Option<ImageRawInfo> {
        self.path_to_index.get(path).cloned()
    }

    /// 添加小图到指定 atlas（如果 atlas_id 不存在则创建）
    pub fn add_texture(
        &mut self,
        atlas_id: u32,
        name: &str,
        img: &RgbaImage,
        atlas_width: u32,
        atlas_height: u32,
    ) {
        let atlas = self
            .data
            .entry(atlas_id)
            .or_insert_with(|| TextureAtlas::new(atlas_width, atlas_height));
        self.curr_ui_texture_info_index += 1;
        atlas.add_sub_image(name, img, self.curr_ui_texture_info_index);
    }
}

impl TextureAtlas {
    /// 创建空 Atlas
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: RgbaImage::new(width, height),
            map: HashMap::new(),
            next_x: 0,
            next_y: 0,
            row_height: 0,
            texture: None,
            texture_view: None,
            sampler: None,
            index: 0,
        }
    }

    pub fn add_sub_image(
        &mut self,
        path: &str, // ✅ 新增参数
        img: &RgbaImage,
        index: u32,
    ) -> Option<UiTextureInfo> {
        let img_width = img.width();
        let img_height = img.height();

        // 检查是否换行
        if self.next_x + img_width > self.width {
            self.next_x = 0;
            self.next_y += self.row_height;
            self.row_height = 0;
        }

        // 超出 Atlas 大小
        if self.next_y + img_height > self.height {
            return None;
        }

        // 复制小图到大图
        self.data.copy_from(img, self.next_x, self.next_y).unwrap();

        // 更新行高
        if img_height > self.row_height {
            self.row_height = img_height;
        }

        // 计算 UV
        let uv_min = [
            self.next_x as f32 / self.width as f32,
            self.next_y as f32 / self.height as f32,
        ];
        let uv_max = [
            (self.next_x + img_width) as f32 / self.width as f32,
            (self.next_y + img_height) as f32 / self.height as f32,
        ];

        // 生成 UiTextureInfo
        let info = UiTextureInfo {
            index: self.map.len() as u32,
            uv_min,
            uv_max,
            path: path.to_string(), // ✅ 保存路径
            parent_index: self.index,
        };

        // 提取文件名作为 key（或直接用路径）
        let key = Path::new(path)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());

        self.map.insert(key, info.clone());

        // 移动下一个插入位置
        self.next_x += img_width;

        Some(info)
    }
    /// 上传大图到 GPU
    ///

    pub fn upload_to_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // 1️⃣ 创建 GPU 纹理
        let size = wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("UI Atlas Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // 2️⃣ 创建 TexelCopyTextureInfo
        let copy_texture = wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        };

        // 3️⃣ 创建 TexelCopyBufferLayout
        let buffer_layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * self.width), // RGBA8 每行字节数
            rows_per_image: Some(self.height),
        };

        // 4️⃣ Extent3d
        let extent = wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };

        // 5️⃣ 上传数据到 GPU
        queue.write_texture(copy_texture, &self.data, buffer_layout, extent);

        // 6️⃣ 创建纹理视图和采样器
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("UI Atlas Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // 7️⃣ 保存到结构体
        self.texture = Some(texture);
        self.texture_view = Some(view);
        self.sampler = Some(sampler);
    }

    /// 获取小图 UV
    pub fn get(&self, name: &str) -> Option<&UiTextureInfo> {
        self.map.get(name)
    }
}
