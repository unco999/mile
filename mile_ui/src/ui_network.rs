use std::{cell::RefCell, collections::HashMap, num::NonZeroU64, sync::{Arc, Mutex, OnceLock}};
use bytemuck::{Pod, Zeroable};
use mile_api::GpuDebug;
use wgpu::{naga::GatherMode, util::DeviceExt};

use crate::{structs::{RelLayoutMask, PanelField}, GlobalLayout, PanelAnimDelta};

type RelID = u32;
type CollectionId = u32;
type offset = u32;

static COLLECTIONS: OnceLock<Mutex<HashMap<String, u32>>> = OnceLock::new();
static RELS: OnceLock<Mutex<HashMap<String, u32>>> = OnceLock::new();
const PANEL_META_SIZE: usize = 6;
const COLLECTION_META_OFFSET: usize = 2;
const REL_META_OFFSET: usize = 8;
const IN_COLLECTION_INDEX_META_OFFSET:usize = 4;

const REL_SIM_OFFSET: usize = 8192;
const COLLECTION_SIM_OFFSET: usize = 16384;


const REL_ID_FIELD:usize = 0;
const REL_SOURCE_FIELD:usize = 1;
const REL_TARGET_FIELD:usize = 2;
const REL_ANIMTION_FIELD:usize = 3;
const REL_LAYOUT:usize = 4;
const REL_IMMEDIATELY_ANIM:usize = 5;
const REL_IMMEDIATELY_PARAMS1:usize = 6;
const REL_IMMEDIATELY_PARAMS2:usize = 7;
const REL_IMMEDIATELY_PARAMS3:usize = 8;

pub fn collection_by_name(name: &str) -> u32 {
    let map = COLLECTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().unwrap();
    if let Some(&id) = map.get(name) {
        id
    } else {
        let id = map.len() as u32;
        println!("name {:?} = {:}",name,id);
        map.insert(name.to_string(), id);
        id
    }
}

pub fn rel_by_name(name: &str) -> u32 {
    let map = RELS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().unwrap();
    if let Some(&id) = map.get(name) {
        id
    } else {
        let id = map.len() as u32;
        map.insert(name.to_string(), id);
        id
    }
}


pub type PanelId = u32;
pub type RelId = u32;

#[derive(Clone)]
pub struct Rel {
    pub id: u32,                     // REL_ID_FIELD
    pub source_collection: u32,      // REL_SOURCE_FIELD
    pub target_collection: u32,      // REL_TARGET_FIELD
    pub animation_field: u32,        // REL_ANIMTION_FIELD
    pub layout: u32,                 // REL_LAYOUT (例如 GRID / LIST)
    pub immediately_anim: u32,       // REL_IMMEDIATELY_ANIM (是否立即动画)
    pub immediately_params1: u32,    // REL_IMMEDIATELY_PARAMS1
    pub immediately_params2: u32,    // REL_IMMEDIATELY_PARAMS2
    pub immediately_params3: u32, // REL_IMMEDIATELY_PARAMS3
}

impl Rel {
    pub fn new(name: &str,source_collection:&str,target_collection:&str)->Self{
        Self{
            id: rel_by_name(name),
            source_collection: collection_by_name(source_collection),
            target_collection:collection_by_name(target_collection),
            animation_field: (PanelField::POSITION_X | PanelField::POSITION_Y).bits(),
            layout:RelLayoutMask::GRID.bits(),
            immediately_anim: 1,
            immediately_params1: 0,
            immediately_params2: 0,
            immediately_params3: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct DebugCallBack {
    pub floats: [f32; 32],
    pub uints: [u32; 32],
}

pub struct PanelRef {
    pub id: PanelId,
    pub collection_index_map: HashMap<CollectionId, usize>,
    pub rel_index_map: Vec<RelId>,
    pub dirty: bool,
}

pub struct Collection {
    pub id: CollectionId,
    pub children: Vec<PanelId>,
    pub children_offset: usize, // flat buffer offset
}

pub struct NetworkStore {
    pub panels_ref: HashMap<PanelId, PanelRef>,
    pub collections: HashMap<CollectionId, Collection>,
    pub rels: HashMap<RelId, Rel>,

    pub buffer_meta: Option<wgpu::Buffer>,
    pub buffer_rels: Option<wgpu::Buffer>,
    pub collection_flat_next: usize, // collection_flat 区下一个可用索引
    pub pipeline: Option<wgpu::ComputePipeline>,
    pub bind_group: Option<wgpu::BindGroup>,
    pub bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub last_uploaded_rel_end: usize, // 新增   
    pub rel_flat_next:usize,
    pub update_collection_sign:Vec<u32>,
    pub buffer_sim: Box<[u32; 139264]>, // CPU 上模拟
}

impl NetworkStore {
pub fn add_rel(
        &mut self,
        rel:Rel
    ) {
        self.rels.insert(rel.id, rel.clone());

        // --- 写入 buffer_sim 增量扁平存储 ---
        let rel_offset = self.rel_flat_next; // 下一个可用位置
        self.buffer_sim[rel_offset + REL_SIM_OFFSET + 0] = rel.id;
        self.buffer_sim[rel_offset + REL_SIM_OFFSET + 1] = rel.source_collection;
        self.buffer_sim[rel_offset + REL_SIM_OFFSET + 2] = rel.target_collection;
        self.buffer_sim[rel_offset + REL_SIM_OFFSET + 3] = rel.animation_field;
                self.buffer_sim[rel_offset + REL_SIM_OFFSET + 4] = rel.layout;
        self.buffer_sim[rel_offset + REL_SIM_OFFSET + 5] = rel.immediately_anim;
        self.buffer_sim[rel_offset + REL_SIM_OFFSET + 6] = rel.immediately_params1;
        self.buffer_sim[rel_offset + REL_SIM_OFFSET + 7] = rel.immediately_params2;
        self.buffer_sim[rel_offset + REL_SIM_OFFSET + 8] = rel.immediately_params3;
        self.rel_flat_next += 8; // 每个 Rel 占 4 个 u32


        // --- 更新相关 panel 的 rel_index_map ---
        for panel in self.panels_ref.values_mut() {
            if panel.collection_index_map.contains_key(&rel.source_collection.clone()) {
                panel.rel_index_map.push(rel_offset as u32); // 用 buffer_sim 偏移存储
                panel.dirty = true;

                // --- 同步更新 panel meta 的 REL_META_OFFSET start/end ---
                let base = panel.id as usize * PANEL_META_SIZE;
                let rel_start = *panel.rel_index_map.first().unwrap();
                let rel_end = *panel.rel_index_map.last().unwrap() + 4;
                self.buffer_sim[base] = rel_start as u32;
                self.buffer_sim[base + 1] = rel_end as u32;
            }
        }

        println!(
            "Rel added: id={} source={} target={} anim={} offset={} panels_updated={}",
            rel.id,
            rel.source_collection,
            rel.target_collection,
            rel.animation_field,
            rel_offset,
            self.panels_ref
                .values()
                .filter(|p| p.collection_index_map.contains_key(&rel.clone().source_collection.clone()))
                .count()
        );
    }

    pub fn new(device: &wgpu::Device,
        global_layout:&GlobalLayout,
        panel_anim_delta_buffer:&wgpu::Buffer,
        instance_panels_buffer:&wgpu::Buffer,
        gpu_debug_buffer:&wgpu::Buffer
    ) -> Self {
        const PANEL_META_SIZE: usize = 6;
    const MAX_PANELS: usize = 32;
    const REL_SIZE: usize = 1024;


    let buffer_meta = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("buffer_meta"),
        contents: bytemuck::cast_slice(&[u32::MAX;32768]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_rels = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("buffer_rels"),
        contents: bytemuck::cast_slice(&vec![u32::MAX; REL_SIZE * 512]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });


        // BindGroupLayout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("NetworkStore BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NetworkStore BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_rels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu_debug_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: panel_anim_delta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: instance_panels_buffer.as_entire_binding(),
                },
            ],
        });

          // ------------------ Compute Pipeline ------------------
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("NetworkStore Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("network_compute.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("NetworkStore PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("interaction_compute_pipeline_pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("interaction_compute_pipeline Layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: Default::default(),
            });


        Self {
            update_collection_sign:vec![],
            rel_flat_next:0,
            last_uploaded_rel_end:0,
            panels_ref: Default::default(),
            collections: Default::default(),
            rels: Default::default(),
            buffer_meta: Some(buffer_meta),
            buffer_rels: Some(buffer_rels),
            bind_group: Some(bind_group),
            bind_group_layout: Some(bind_group_layout),
            pipeline: Some(pipeline),
            buffer_sim: Box::new([0u32; 139264]),
            collection_flat_next: COLLECTION_SIM_OFFSET,
        }
    }
pub fn upload_dirty_collections(&mut self, queue: &wgpu::Queue) {
    const MAX_COLLECTION_ITEMS: usize = 128;
    if let Some(buffer_collection) = &self.buffer_meta {
        for &collection_id in &self.update_collection_sign {
            if let Some(collection) = self.collections.get(&collection_id) {
                let offset = collection.children_offset as usize;
                let len = collection.children.len();

                // 先填充整个区间为 u32::MAX
                self.buffer_sim[offset..offset + MAX_COLLECTION_ITEMS].fill(u32::MAX);

                // 写入实际 children
                let write_len = len.min(MAX_COLLECTION_ITEMS);
                self.buffer_sim[offset..offset + write_len].copy_from_slice(&collection.children[..write_len]);

         
                // 上传到 GPU buffer
                queue.write_buffer(
                    buffer_collection,
                    (offset * std::mem::size_of::<u32>()) as wgpu::BufferAddress, // u32 -> 4 bytes
                    bytemuck::cast_slice(&self.buffer_sim[offset..offset + MAX_COLLECTION_ITEMS]),
                );
            }
        }
    }

    // 上传完成后清空标记
    self.update_collection_sign.clear();
}

pub fn add_panel_to_collection(&mut self, panel_id: PanelId, collection_id: CollectionId) {
    // --- 获取或创建 collection ---
    let collection = self.collections.entry(collection_id)
        .or_insert_with(|| {
            let offset = COLLECTION_SIM_OFFSET + (collection_id * 128) as usize;
            Collection { id: collection_id, children: vec![], children_offset: offset}
        });

    self.update_collection_sign.push(collection_id);

    let idx = collection.children.len();
    collection.children.push(panel_id);


    // --- 获取或创建 panel ---
    let panel = self.panels_ref.entry(panel_id)
        .or_insert_with(|| PanelRef { id: panel_id, collection_index_map: HashMap::new(), rel_index_map: vec![], dirty: true });

    let in_index = collection.children.len();
    panel.collection_index_map.insert(collection_id, idx);
    panel.dirty = true;

    // 写入 panel meta buffer_sim
    let base = panel.id as usize * PANEL_META_SIZE;
    self.buffer_sim[base + COLLECTION_META_OFFSET] = collection.children_offset as u32;
    self.buffer_sim[base + COLLECTION_META_OFFSET + 1] = (collection.children_offset + 128) as u32;
    self.buffer_sim[base + IN_COLLECTION_INDEX_META_OFFSET] = in_index as u32;
    // --- 更新 panel 的 rel_index_map ---
    // 遍历所有 rel，如果 rel.source_collection == collection_id，则添加到 panel
    for rel in self.rels.values() {
        if rel.source_collection == collection_id {
            panel.rel_index_map.push(rel.id);

            // 同步更新 panel meta 的 REL_META_OFFSET start/end
            let rel_start = *panel.rel_index_map.first().unwrap() as usize;
            let rel_end = *panel.rel_index_map.last().unwrap() as usize + 1;
            self.buffer_sim[base ] = rel_start as u32;
            self.buffer_sim[base + 1] = rel_end as u32;
        }
    }
}
    pub fn upload_dirty_to_gpu_batch(&mut self, queue: &wgpu::Queue) {
        // 收集 dirty 面板
        self.upload_dirty_collections(queue);

        let mut dirty_panels: Vec<_> = self.panels_ref.values_mut().filter(|p| p.dirty).collect();
        if dirty_panels.is_empty() && self.last_uploaded_rel_end == self.rel_flat_next {
            return;
        }

        dirty_panels.sort_by_key(|p| p.id);

        // --- 上传 Meta ---
        if !dirty_panels.is_empty() {
            let meta_start = dirty_panels.first().unwrap().id as usize * PANEL_META_SIZE;
            let meta_end = dirty_panels.last().unwrap().id as usize * PANEL_META_SIZE + PANEL_META_SIZE;
            if let Some(buffer_meta) = &self.buffer_meta {
                println!("Meta upload for panels {}..{}: start={} end={}", 
                    dirty_panels.first().unwrap().id, dirty_panels.last().unwrap().id, meta_start, meta_end);
                queue.write_buffer(buffer_meta, 0, bytemuck::cast_slice(&self.buffer_sim[meta_start..meta_end]));
            }
        }


        // --- 上传 Rels 增量 ---
        if self.last_uploaded_rel_end < self.rel_flat_next {
            if let Some(buffer_rels) = &self.buffer_rels {
                let rel_slice = &self.buffer_sim[(self.last_uploaded_rel_end + REL_SIM_OFFSET)..(self.rel_flat_next + REL_SIM_OFFSET)];
                assert!(rel_slice.len() % 8 == 0, "Rel 数据必须是 4 的倍数");
                let rel_u32_count = rel_slice.len();
                println!("Rels upload: start={} end={} data={:?}", self.last_uploaded_rel_end, self.rel_flat_next,rel_slice);
                queue.write_buffer(buffer_rels, (self.last_uploaded_rel_end) as wgpu::BufferAddress, bytemuck::cast_slice(rel_slice));
            }
            self.last_uploaded_rel_end = self.rel_flat_next;
        }

        // --- 清除 dirty 标记 ---
        for panel in dirty_panels.iter_mut() {
            panel.dirty = false;
        }

        println!("Upload complete, dirty flags cleared.");
    }
    fn get_panel_rel_range_cpu(buffer_sim: &[u32], panel: &PanelRef) -> (usize, usize) {
        // 计算 panel 对应的 rel 区间
        let start = panel.rel_index_map.first().cloned().unwrap_or(0) as usize;
        let end = panel.rel_index_map.last().map(|v| v + 1).unwrap_or(0)as usize;
        (start, end)
    }
}

