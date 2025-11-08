use std::{
    cell::RefCell,
    collections::{HashMap, btree_map::Range},
    error::Error,
    fs,
    hash::Hash,
    io,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, Mutex},
};

use bytemuck::{Pod, Zeroable};
use mile_api::{
    global::global_event_bus,
    prelude::{_ty::PanelId, CpuGlobalUniform, EventBus, GpuDebug, ModEventStream, Renderable},
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use ttf_parser::{Face, GlyphId, OutlineBuilder, morx::InsertionEntryData};
use wgpu::{
    Buffer, BufferUsages, DepthBiasState, Extent3d, RenderPass, SamplerBindingType, TextureFormat,
    TextureViewDescriptor,
    hal::{TextureDescriptor, auxil::db},
    util::DeviceExt,
};

use crate::event::{BatchFontEntry, BatchRenderFont};

const QUAD_INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];
// Quad 顶点数据（屏幕空间，-0.5 ~ 0.5）
const QUAD_VERTICES: &[Vertex] = &[
    Vertex {
        pos: [-0.5, -0.5],
        uv: [0.0, 1.0],
    },
    Vertex {
        pos: [0.5, -0.5],
        uv: [1.0, 1.0],
    },
    Vertex {
        pos: [0.5, 0.5],
        uv: [1.0, 0.0],
    },
    Vertex {
        pos: [-0.5, 0.5],
        uv: [0.0, 0.0],
    },
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

/**字体style */
#[derive(Debug)]
pub struct FontStyle {
    pub font_size: u32,
    pub font_file_path: &'static str,
    pub font_color: [f32; 4],
    pub font_weight: u32,
    pub font_style: &'static str,
    pub font_family: &'static str,
    pub font_line_height: u32,
    pub font_text_align: &'static str,
    pub font_text_decoration: &'static str,
}

#[derive(Debug)]
struct Text<'a> {
    text: &'a str,
    style: &'a FontStyle,
}

struct TextGpu {}

#[derive(Clone, Debug)]
pub struct GlyphMetrics {
    // 字体级别度量
    pub units_per_em: u32,
    pub ascent: i32,
    pub descent: i32,
    pub line_gap: i32,
    pub advance_width: u32,
    pub left_side_bearing: i32,

    // 边界框
    pub x_min: i32,
    pub y_min: i32,
    pub x_max: i32,
    pub y_max: i32,

    // 字形特定度量
    pub glyph_advance_width: u32,
    pub glyph_left_side_bearing: i32,
    pub glyph_ver_advance: u32,
    pub glyph_ver_side_bearing: i32,
}

#[derive(Default)]
pub struct RenderPlanStore {
    pub render_text_plan_map: HashMap<RenderTextPlanIdx, RenderTextPlan>,
}

type RegisterEvent<'a> = ModEventStream<(BatchFontEntry, BatchRenderFont<'a, PanelId>)>;
pub struct MileFont {
    gpu_debug: RefCell<GpuDebug>,
    cache: FontPipeLineCache,
    fonts: HashMap<String, FontInstance>,
    entries: HashMap<FontKey, FontCacheEntry>,
    global_unifrom: Rc<CpuGlobalUniform>,
    current_version: u64,
    indirects_len: u32,
    num_instances: u32,
    is_update: bool,
    global_event_hub: &'static EventBus,
    register_global_event: RegisterEvent<'static>,
    store: RenderPlanStore,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

impl Renderable for MileFont {
    fn render<'a>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        pass: &mut RenderPass<'a>,
    ) {
        {
            pass.set_pipeline(&self.cache.render_pipeline.as_ref().unwrap());
            let indirects_count = self.indirects_len; // 你需要记录 update_indirect_buffer 生成了多少 draw
            for i in 0..indirects_count {
                let offset = (i as u64 * std::mem::size_of::<DrawIndexedIndirect>() as u64);
                pass.draw_indexed_indirect(&self.cache.indirects_buffer.as_ref().unwrap(), offset);
            }
            pass.set_bind_group(
                0,
                &self
                    .cache
                    .render_bind_cache
                    .as_ref()
                    .unwrap()
                    .render_bind_group,
                &[],
            );
            pass.set_vertex_buffer(
                0,
                self.cache
                    .render_buffer_cache
                    .as_ref()
                    .unwrap()
                    .vertex_buffer
                    .slice(..),
            );
            pass.set_index_buffer(
                self.cache
                    .render_buffer_cache
                    .as_ref()
                    .unwrap()
                    .index_buffer
                    .slice(..),
                wgpu::IndexFormat::Uint16,
            );
            pass.draw_indexed(
                0..QUAD_INDICES.len() as u32,
                0,
                0..self.num_instances as u32,
            );
        }
        // queue.submit(Some(encoder.finish()));
    }

    fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.gpu_debug.borrow_mut().debug(device, queue);
    }

    fn resize(
        &mut self,
        size: winit::dpi::PhysicalSize<u32>,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
    ) {
    }
}

pub struct FontCacheEntry {
    pub key: FontKey,
    pub glyphs: HashMap<char, FontGlyphResult>,
    pub last_used_frame: u64,
    pub version: u64,
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct FontKey {
    font_name: String,
    font_size: u32,
}

#[derive(Debug)]
pub struct FontBatch {
    pub results: Vec<FontGlyphResult>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct RenderTextPlan {
    font_with_board_index_offset_start: u32,
    font_with_board_index_offset_end: u32,
    font_size: u32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
    pad: u32,
}

pub type RenderTextPlanIdx = u32;

impl FontBatch {
    pub fn to_gpu_struct(
        &mut self,
        start_offset: u64,
    ) -> (Vec<FontGlyphDes>, Vec<GlyphInstruction>) {
        let mut glyph_descs = Vec::new();
        let mut gpu_instructions = Vec::new();

        let mut current_offset = start_offset as u32;

        for glyph in &self.results {
            let start_idx = current_offset;
            let end_idx = start_idx + glyph.outline.len() as u32;

            println!(
                "当前加入字形 {:?} gpu:start{} gpu:end{}",
                glyph.character, start_idx, end_idx
            );
            let m = &glyph.glyph_metrics;
            glyph_descs.push(FontGlyphDes {
                start_idx,
                end_idx,
                texture_idx_x: 0,
                texture_idx_y: 0,
                x_min: m.x_min,
                y_min: m.y_min,
                x_max: m.x_max,
                y_max: m.y_max,
                units_per_em: m.units_per_em,
                ascent: m.ascent,
                descent: m.descent,
                line_gap: m.line_gap,
                advance_width: m.advance_width,
                left_side_bearing: m.left_side_bearing,
                glyph_advance_width: m.advance_width,
                glyph_left_side_bearing: m.left_side_bearing,
            });

            println!("glyph.outline {:?}", glyph.outline);

            gpu_instructions.extend_from_slice(&glyph.outline);

            current_offset = end_idx;
        }

        (glyph_descs, gpu_instructions)
    }
}

#[derive(Debug, Clone)]
pub struct FontGlyphResult {
    pub character: char,
    pub glyph_id: GlyphId,
    pub outline: Vec<GlyphInstruction>,
    pub glyph_metrics: GlyphMetrics,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct FontGlyphDes {
    start_idx: u32,
    end_idx: u32,
    texture_idx_x: u32,
    texture_idx_y: u32,

    // 当前已有的边界框
    x_min: i32,
    y_min: i32,
    x_max: i32,
    y_max: i32,

    // 需要新增的关键度量字段
    units_per_em: u32,      // 每个em的字体单位数[citation:8]
    ascent: i32,            // 从基线到顶部的距离[citation:9]
    descent: i32,           // 从基线到底部的距离（通常为负值）[citation:9]
    line_gap: i32,          // 行间距[citation:9]
    advance_width: u32,     // 字形的总前进宽度[citation:9]
    left_side_bearing: i32, // 从原点到位图左边的距离[citation:9]

    // 字形特定的度量
    glyph_advance_width: u32,     // 特定字形的前进宽度
    glyph_left_side_bearing: i32, // 特定字形的左侧支撑
}

impl MileFont {
    pub fn new(global_unifrom: Rc<CpuGlobalUniform>) -> Self {
        Self {
            store: RenderPlanStore::default(),
            global_event_hub: global_event_bus(),
            gpu_debug: RefCell::new(GpuDebug::new("MileFont")),
            cache: Default::default(),
            fonts: Default::default(),
            global_unifrom: global_unifrom,
            entries: Default::default(),
            current_version: Default::default(),
            indirects_len: Default::default(),
            num_instances: Default::default(),
            is_update: false,
            register_global_event: RegisterEvent::new(global_event_bus()),
        }
    }

    pub fn load_font_file(&mut self) {
        self.load_to_face("../ttf/BIZUDPGothic-Regular.ttf");
    }

    pub fn evnet_polling(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (batch_render_fonts, batch_entry_fonts) = self.register_global_event.poll();

        let mut grouped: HashMap<&str, Vec<&str>> = HashMap::new();

        for e in &batch_entry_fonts {
            grouped.entry(&e.font_file_path).or_default().push(&e.str);
        }
        for (font_file_name, text) in grouped {
            let result = self.queue_batch_parse(font_file_name, text.iter().as_ref(), 16);
            let cache = self.cache.generic_buffer_cache.as_mut().unwrap();

            if let Ok(mut res) = result {
                println!("加入字体到印刷板 {:?}", text);
                let (des, instruction) = res.to_gpu_struct(cache.instruction_buffer_index);
                self.write_batch_buffer(queue, &des, &instruction);
            }
        }
        for batch_render_font in batch_render_fonts {
            println!("实际显示字体 {:?}", batch_render_font.str);
            self.test_entry_text(queue);
        }
    }

    pub fn test_entry_text(&mut self, queue: &wgpu::Queue) {
        let vertex_buffer = self
            .cache
            .render_buffer_cache
            .as_ref()
            .unwrap()
            .vertex_buffer
            .clone();
        let index_buffer = self
            .cache
            .render_buffer_cache
            .as_ref()
            .unwrap()
            .index_buffer
            .clone();
        let indirects_buffer = self.cache.indirects_buffer.as_ref().unwrap().clone();

        // 写入顶点 buffer
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(QUAD_VERTICES));

        // 写入索引 buffer
        queue.write_buffer(&index_buffer, 0, bytemuck::cast_slice(QUAD_INDICES));

        // 写入 indirect buffer (测试绘制)
        let indirect_data: [u32; 4] = [
            QUAD_INDICES.len() as u32, // index count
            1,                         // instance count
            0,                         // first index
            0,                         // base vertex
        ];

        self.num_instances += 1;
        queue.write_buffer(&indirects_buffer, 0, bytemuck::cast_slice(&indirect_data));

        println!("Quad vertex/index/indirect buffers uploaded for testing.");
    }

    pub fn test_entry(&mut self, queue: &wgpu::Queue) {
        let result = self.queue_batch_parse("../ttf/BIZUDPGothic-Regular.ttf", &["币"], 16);
        let cache = self.cache.generic_buffer_cache.as_mut().unwrap();

        if let Ok(mut res) = result {
            let (des, instruction) = res.to_gpu_struct(cache.instruction_buffer_index);
            self.write_batch_buffer(queue, &des, &instruction);
        }
    }

    pub fn write_batch_buffer(
        &mut self,
        queue: &wgpu::Queue,
        fond_des: &[FontGlyphDes],
        instruction: &[GlyphInstruction],
    ) {
        let cache = self.cache.generic_buffer_cache.as_mut().unwrap();
        let ins_offset =
            cache.instruction_buffer_index * std::mem::size_of::<GlyphInstruction>() as u64;
        let des_offset =
            cache.instruction_buffer_index as u64 * std::mem::size_of::<FontGlyphDes>() as u64;

        if (fond_des.len() == 0 && instruction.len() == 0) {
            return;
        }

        self.is_update = true;

        queue.write_buffer(
            cache.instruction_buffer.as_ref().unwrap(),
            ins_offset,
            bytemuck::cast_slice(instruction),
        );

        queue.write_buffer(
            cache.instruction_des_buffer.as_ref().unwrap(),
            des_offset,
            bytemuck::cast_slice(fond_des),
        );
        cache.instruction_buffer_index += fond_des.len() as u64;
    }

    pub fn batch_enqueue_compute(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batch_enqueue_compute Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("batch_enqueue_compute pass"),
                timestamp_writes: Default::default(),
            });
            let workgroup_count_x = (4096 + 7) / 8; // 512 个工作组
            let workgroup_count_y = (4096 + 7) / 8; // 512 个工作组
            cpass.set_pipeline(&self.cache.compute_pipeline.as_ref().unwrap());
            cpass.set_bind_group(
                0,
                &self.cache.compute_bind_cache.as_ref().unwrap().bind_0_group,
                &[],
            );
            let workgroups = 1;
            cpass.dispatch_workgroups(8, 8, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    pub fn copy_store_texture_to_render_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if self.is_update == false {
            return;
        }
        let cache = self.cache.generic_buffer_cache.as_ref().unwrap();

        let store_texture = &cache.storage_texture;
        let render_texture = &cache.render_texture;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SDF Texture Copy Encoder"),
        });

        let font_count = self.fonts.len() as u32;
        let glyph_size = 64u32;
        let atlas_width = 4096u32;

        // 每行最多放 glyph 个数
        let glyphs_per_row = atlas_width / glyph_size;
        // 动态计算 atlas 高度
        let atlas_height = ((font_count + glyphs_per_row - 1) / glyphs_per_row) * glyph_size;

        // 如果 compute 已经一次性输出整个 atlas，就直接拷贝整张纹理
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: store_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: render_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: 4096,  // atlas 宽度固定
                height: 4096, // 动态高度
                depth_or_array_layers: 1,
            },
        );

        println!(
            "拷贝sdf纹理到render width:{:?} height:{:?}",
            atlas_width, atlas_height
        );

        queue.submit(std::iter::once(encoder.finish()));
        self.is_update = false;
    }

    pub fn create_render_pipeline(&mut self, device: &wgpu::Device, format: TextureFormat) {
        // 创建 pipeline (Vertex + Fragment shader)
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader/font.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UI Pipeline Layout"),
            bind_group_layouts: &[self
                .cache
                .render_bind_cache
                .as_ref()
                .unwrap()
                .render_bind_group_layout
                .as_ref()
                .unwrap()],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main".into(),
                buffers: &[wgpu::VertexBufferLayout {
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
                }],
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float, // ✅ 必须与 render pass 的 depth_view 格式一致
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        self.cache.render_pipeline = Some(pipeline)
    }

    pub fn create_batch_enqueue_font_compute_cahce(&mut self, device: &wgpu::Device) {
        let compute_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute_shader_module Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader/batch_enqueue_font.wgsl").into()),
        });

        self.gpu_debug.borrow_mut().create_buffer(device);

        // === Bind Group Layout ===
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch_enqueue_compute Bind Group Layout"),
            entries: &[
                // 0️⃣ instance buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 0️⃣ instance buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });

        let generic_buffer_cache = self.cache.generic_buffer_cache.as_ref().unwrap();

        // === Bind Group ===
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self
                            .cache
                            .generic_buffer_cache
                            .as_ref()
                            .unwrap()
                            .storage_texture_view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: generic_buffer_cache
                        .instruction_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: generic_buffer_cache
                        .instruction_des_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self
                        .gpu_debug
                        .borrow()
                        .buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
            ],
        });
        // === Compute Pipeline ===
        let batch_enqueue_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("batch_enqueue_compute_pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("batch_enqueue_compute_pipeline Layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &compute_shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        self.cache.compute_bind_cache = Some(ComputeBindCache {
            bind_0_group: Some(compute_bind_group),
        });
        self.cache.compute_pipeline = Some(batch_enqueue_compute_pipeline);
    }

    pub fn create_template_render_texture_and_layout(
        &mut self,
        device: &wgpu::Device,
        format: Option<wgpu::TextureFormat>,
    ) {
        // ✅ 纹理格式 — RGBA8Unorm 是最常见的，也可以用浮点格式（例如 R16Float）以存储距离场

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
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
            label: Some("FontRenderBindGroupLayout"),
        });

        let global_buffer = self.global_unifrom.get_buffer();

        // 2️⃣ 创建 BindGroup
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Font  render BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &self
                            .cache
                            .generic_buffer_cache
                            .as_ref()
                            .unwrap()
                            .render_texture_view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        &self.cache.generic_buffer_cache.as_ref().unwrap().sampler,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self
                        .cache
                        .generic_buffer_cache
                        .as_ref()
                        .unwrap()
                        .instruction_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: global_buffer.as_entire_binding(),
                },
            ],
        });

        self.cache.render_bind_cache = Some(RenderBindCache {
            render_bind_group_layout: Some(bind_group_layout),
            render_bind_group: Some(group),
        })
    }

    pub fn init_buffer(&mut self, device: &wgpu::Device) {
        // self.cache.generic_buffer_cache = Some(GenericBufferCache {
        //     template_font_buffer: ()
        // })
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UI Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UI Quad Index Buffer"),
            contents: bytemuck::cast_slice(QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX | BufferUsages::COPY_DST,
        });

        let instruction_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instruction_buffer Global Buffer"),
            size: std::mem::size_of::<GlyphInstruction>() as u64 * 8096 as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instruction_des_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instruction_buffer Global Buffer"),
            size: std::mem::size_of::<FontGlyphDes>() as u64 * 8096 as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_layers = 32; // 或根据你的 UI 层数需求
        let indirects_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Draw Buffer"),
            size: (max_layers * std::mem::size_of::<DrawIndexedIndirect>()) as u64,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 创建采样器
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FontTemplateSampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,       // 用于深度比较，普通纹理设为 None
            anisotropy_clamp: 1, // 通常设为1，除非需要各向异性过滤
            border_color: None,
        });

        // 同时创建两个纹理（修正你的代码）
        let storage_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FontTemplateStorageTexture"),
            size: Extent3d {
                width: 4096,
                height: 4096,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let sampling_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FontSamplingTexture"),
            size: Extent3d {
                width: 4096,
                height: 4096,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float, // ✅ filterable
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let storage_texture_view = storage_texture.create_view(&TextureViewDescriptor::default());
        let sampling_texture_view = sampling_texture.create_view(&TextureViewDescriptor::default());

        self.cache.generic_buffer_cache = Some(GenericBufferCache {
            render_texture: sampling_texture,
            storage_texture: storage_texture,
            render_texture_view: sampling_texture_view,
            storage_texture_view: storage_texture_view,
            instruction_buffer: Some(instruction_buffer),
            instruction_buffer_index: 0,
            instruction_des_buffer: Some(instruction_des_buffer),
            sampler,
        });

        self.cache.indirects_buffer = Some(indirects_buffer);

        self.cache.render_buffer_cache = Some(RenderBufferCache {
            vertex_buffer: vertex_buffer,
            index_buffer: index_buffer,
        });

        // 确认它们是同一个对象
    }

    /// 批量解析字符串（多线程）
    /// 每个字符串将拆分为单个字形，并由 rayon 并行解析
    pub fn queue_batch_parse(
        &mut self,
        font_name: &str,
        text_batch: &[&str],
        font_size: u32,
    ) -> Result<FontBatch, Box<dyn Error>> {
        // --- Step 1: 获取字体句柄 ---
        let font_instance = self
            .fonts
            .get(font_name)
            .ok_or_else(|| format!("字体 '{}' 未加载", font_name))?;

        let key = FontKey {
            font_name: font_name.to_string(),
            font_size,
        };

        // --- Step 2: 获取或创建缓存 ---
        let cache = self
            .entries
            .entry(key.clone())
            .or_insert_with(|| FontCacheEntry {
                key: key.clone(),
                glyphs: HashMap::new(),
                last_used_frame: 0,
                version: self.current_version,
            });

        let face = &font_instance.face;

        let new_results: Vec<FontGlyphResult> = text_batch
            .par_iter()
            .flat_map(|text| {
                let mut local_glyphs = Vec::new();

                for ch in text.chars() {
                    if cache.glyphs.contains_key(&ch) {
                        continue;
                    }

                    if let Some(glyph_id) = face.glyph_index(ch) {
                        let mut builder = GlyphBuilder::new();
                        if face.outline_glyph(glyph_id, &mut builder).is_some() {
                            // 获取更多的字体度量信息
                            let bbox = face.global_bounding_box();
                            let metrics = GlyphMetrics {
                                units_per_em: face.units_per_em() as u32,
                                ascent: face.ascender() as i32, // 注意：ascent 可能是负数
                                descent: face.descender() as i32, // descent 通常是负数
                                line_gap: face.line_gap() as i32,
                                advance_width: face.glyph_hor_advance(glyph_id).unwrap_or(0) as u32,
                                left_side_bearing: face
                                    .glyph_hor_side_bearing(glyph_id)
                                    .unwrap_or(0)
                                    as i32,

                                // 边界框信息
                                x_min: bbox.x_min as i32,
                                y_min: bbox.y_min as i32,
                                x_max: bbox.x_max as i32,
                                y_max: bbox.y_max as i32,

                                // 字形特定信息
                                glyph_advance_width: face.glyph_hor_advance(glyph_id).unwrap_or(0)
                                    as u32,
                                glyph_left_side_bearing: face
                                    .glyph_hor_side_bearing(glyph_id)
                                    .unwrap_or(0)
                                    as i32,

                                // 垂直度量（如果需要）
                                glyph_ver_advance: face.glyph_ver_advance(glyph_id).unwrap_or(0)
                                    as u32,
                                glyph_ver_side_bearing: face
                                    .glyph_ver_side_bearing(glyph_id)
                                    .unwrap_or(0)
                                    as i32,
                            };

                            local_glyphs.push(FontGlyphResult {
                                character: ch,
                                glyph_id,
                                outline: builder.instructions.clone(),
                                glyph_metrics: metrics,
                            });
                        }
                    }
                }

                local_glyphs
            })
            .collect();

        // --- Step 4: 更新缓存 ---
        for glyph in &new_results {
            cache.glyphs.insert(glyph.character, glyph.clone());
        }

        // --- Step 5: 返回结果（可能为空） ---
        Ok(FontBatch {
            results: new_results,
        })
    }

    pub fn load_to_face<T: AsRef<str>>(&mut self, font_file: T) -> Result<(), Box<dyn Error>> {
        let path = font_file.as_ref();
        let buffer = self.load_font(path)?;

        // 关键：把 buffer 转成 'static 生命周期
        let leaked: &'static [u8] = Box::leak(buffer.clone().into_boxed_slice());

        let face = Face::parse(leaked, 0)?;

        // 打印基本信息
        println!(
            "✅ 成功加载字体: {}\n  - 大小: {:.2} KB\n - 字形总数: {:?}\n",
            path,
            buffer.len() as f32 / 1024.0,
            face.number_of_glyphs(),
        );
        self.fonts.insert(
            path.to_string(),
            FontInstance {
                data: buffer, // 实际持有数据
                face,
            },
        );

        Ok(())
    }
    /// 尝试多路径加载字体文件（相对路径、项目根目录、当前工作目录）
    /// 返回 Ok(Vec<u8>) 表示成功，Err(io::Error) 表示所有路径都失败
    fn load_font(&self, font_file: &str) -> Result<Vec<u8>, io::Error> {
        let mut tried_paths: Vec<PathBuf> = Vec::new();

        // 1️⃣ 尝试直接相对路径
        let path1 = Path::new(font_file);
        tried_paths.push(path1.to_path_buf());
        if let Ok(data) = fs::read(path1) {
            return Ok(data);
        }

        // 2️⃣ 尝试项目根目录
        if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
            let path2 = Path::new(manifest_dir).join(font_file);
            tried_paths.push(path2.clone());
            if let Ok(data) = fs::read(&path2) {
                return Ok(data);
            }
        }

        // 3️⃣ 尝试当前工作目录
        let path3 = Path::new(".").join(font_file);
        tried_paths.push(path3.clone());
        if let Ok(data) = fs::read(&path3) {
            return Ok(data);
        }

        // 4️⃣ 所有尝试都失败 -> 返回自定义错误
        let msg = tried_paths
            .iter()
            .map(|p| format!(" - {:?}", p))
            .collect::<Vec<_>>()
            .join("\n");

        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("无法找到字体文件 '{}'\n尝试过路径:\n{}", font_file, msg),
        ))
    }
}

impl MileFont {}

#[derive(Default)]
pub struct FontPipeLineCache {
    indirects_buffer: Option<wgpu::Buffer>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    compute_pipeline: Option<wgpu::ComputePipeline>,
    render_buffer_cache: Option<RenderBufferCache>,
    generic_buffer_cache: Option<GenericBufferCache>,
    render_bind_cache: Option<RenderBindCache>,
    compute_bind_cache: Option<ComputeBindCache>,
}

pub struct RenderBindCache {
    render_bind_group_layout: Option<wgpu::BindGroupLayout>,
    render_bind_group: Option<wgpu::BindGroup>,
}

pub struct ComputeBindCache {
    pub bind_0_group: Option<wgpu::BindGroup>,
}

pub struct GenericBindCache {
    pub template_texture_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub template_texture_bind_group: Option<wgpu::BindGroup>,
}

pub struct GenericBufferCache {
    /**模板复用字库文件 */
    storage_texture_view: wgpu::TextureView,
    render_texture_view: wgpu::TextureView,
    storage_texture: wgpu::Texture,
    render_texture: wgpu::Texture,
    instruction_buffer: Option<wgpu::Buffer>,
    instruction_buffer_index: u64,
    instruction_des_buffer: Option<wgpu::Buffer>,
    sampler: wgpu::Sampler,
}

pub struct ComputeBufferCache {}

pub struct RenderBufferCache {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

pub struct FontComputeCache {
    animation_compute_pipe: wgpu::ComputePipeline,
    batch_queue_compute_pipe: wgpu::ComputePipeline,
}

pub struct FontInstance {
    data: Vec<u8>,
    face: ttf_parser::Face<'static>,
}

/// Glyph 指令类型
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum GlyphCommand {
    MoveTo = 0,
    LineTo = 1,
    QuadTo = 2,
    CurveTo = 3,
    Close = 4,
}

/**
 * 字形轮廓数据
 */
pub struct GlyphBuilder {
    pub instructions: Vec<GlyphInstruction>,
}

impl GlyphBuilder {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }
}

impl OutlineBuilder for GlyphBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.instructions.push(GlyphInstruction::move_to(x, y));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.instructions.push(GlyphInstruction::line_to(x, y));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.instructions
            .push(GlyphInstruction::quad_to(x1, y1, x, y));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.instructions
            .push(GlyphInstruction::curve_to(x1, y1, x2, y2, x, y));
    }
    fn close(&mut self) {
        self.instructions.push(GlyphInstruction::close());
    }
}

/// GPU-friendly 字形指令
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlyphInstruction {
    pub command: u32,
    pub _pad: u32,      // <-- 匹配 Rust 的对齐补位
    pub data: [f32; 6], //  p0, p1, p2 展平
}
impl GlyphInstruction {
    #[inline]
    pub fn move_to(x: f32, y: f32) -> Self {
        Self {
            command: GlyphCommand::MoveTo as u32,
            data: [x, y, 0.0, 0.0, 0.0, 0.0],
            _pad: 0,
        }
    }

    #[inline]
    pub fn line_to(x: f32, y: f32) -> Self {
        Self {
            command: GlyphCommand::LineTo as u32,
            data: [x, y, 0.0, 0.0, 0.0, 0.0],
            _pad: 0,
        }
    }

    #[inline]
    pub fn quad_to(cx: f32, cy: f32, x: f32, y: f32) -> Self {
        Self {
            command: GlyphCommand::QuadTo as u32,
            data: [x, y, cx, cy, 0.0, 0.0],
            _pad: 0,
        }
    }

    #[inline]
    pub fn curve_to(cx1: f32, cy1: f32, cx2: f32, cy2: f32, x: f32, y: f32) -> Self {
        Self {
            command: GlyphCommand::CurveTo as u32,
            data: [x, y, cx1, cy1, cx2, cy2],
            _pad: 0,
        }
    }

    #[inline]
    pub fn close() -> Self {
        Self {
            command: GlyphCommand::Close as u32,
            data: [0.0; 6],
            _pad: 0,
        }
    }
}
