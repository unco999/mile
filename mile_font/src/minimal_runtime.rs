use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use mile_api::prelude::{GpuDebug, ModEventStream, Renderable, global_event_bus};
use ttf_parser::{Face, GlyphId, OutlineBuilder};
use wgpu::{
    DepthBiasState, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor, util::DeviceExt,
};
use winit::dpi::PhysicalSize;

use crate::{
    event::{BatchFontEntry, BatchRenderFont, RemoveRenderFont, ResetFontRuntime},
    prelude::{
        GPU_CHAR_LAYOUT_FLAG_LINE_BREAK_BEFORE, GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_MASK,
        GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_MAX, GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_SHIFT, GpuChar,
        GpuText,
    },
};

type RegisterEvent = ModEventStream<(
    BatchFontEntry,
    BatchRenderFont,
    RemoveRenderFont,
    ResetFontRuntime,
)>;

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

#[derive(Clone, Debug)]
pub struct GlyphMetrics {
    pub units_per_em: u32,
    pub ascent: i32,
    pub descent: i32,
    pub line_gap: i32,
    pub advance_width: u32,
    pub left_side_bearing: i32,
    pub x_min: i32,
    pub y_min: i32,
    pub x_max: i32,
    pub y_max: i32,
    pub glyph_advance_width: u32,
    pub glyph_left_side_bearing: i32,
    pub glyph_ver_advance: u32,
    pub glyph_ver_side_bearing: i32,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum GlyphCommand {
    MoveTo = 0,
    LineTo = 1,
    QuadTo = 2,
    CurveTo = 3,
    Close = 4,
}

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

impl MiniFontRuntime {
    pub fn render_texture_view(&self) -> Option<&wgpu::TextureView> {
        self.buffers.as_ref().map(|b| &b.render_texture_view)
    }
}

impl MiniFontRuntime {
    fn zero_char() -> GpuChar {
        GpuChar {
            char_index: 0,
            gpu_text_index: 0,
            panel_index: 0,
            self_index: 0,
            glyph_advance_width: 0,
            glyph_left_side_bearing: 0,
            glyph_ver_advance: 0,
            glyph_ver_side_bearing: 0,
            layout_flags: 0,
        }
    }

    fn round_pow2(n: u32) -> u32 {
        if n <= 1 {
            1
        } else {
            let mut v = n - 1;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v + 1
        }
    }

    fn coalesce_free_list(free: &mut Vec<FreeRange>) {
        if free.is_empty() {
            return;
        }
        free.sort_by_key(|r| r.start);
        let mut merged: Vec<FreeRange> = Vec::with_capacity(free.len());
        let mut cur = free[0];
        for &r in &free[1..] {
            if cur.start + cur.len == r.start {
                cur.len += r.len;
            } else {
                merged.push(cur);
                cur = r;
            }
        }
        merged.push(cur);
        *free = merged;
    }

    fn alloc_range(&mut self, cap: u32) -> u32 {
        // first-fit from free list
        for i in 0..self.free_list.len() {
            let r = self.free_list[i];
            if r.len >= cap {
                let start = r.start;
                if r.len == cap {
                    self.free_list.remove(i);
                } else {
                    self.free_list[i] = FreeRange {
                        start: r.start + cap,
                        len: r.len - cap,
                    };
                }
                // ensure CPU staging buffer large enough
                let need = (start + cap) as usize;
                if self.out_gpu_chars.len() < need {
                    self.out_gpu_chars.resize(need, Self::zero_char());
                }
                return start;
            }
        }
        // no free block: extend at end
        let start = self.out_gpu_chars.len() as u32;
        self.out_gpu_chars
            .resize((start + cap) as usize, Self::zero_char());
        start
    }

    fn free_range(&mut self, start: u32, len: u32) {
        if len == 0 {
            return;
        }
        self.free_list.push(FreeRange { start, len });
        Self::coalesce_free_list(&mut self.free_list);
    }

    /// Flag all tracked text slices for this panel as removed so stale strings disappear.
    fn retire_panel_text_indices(&mut self, panel_id: u32) -> bool {
        if let Some(indices) = self.panel_text_indices.remove(&panel_id) {
            let mut touched = false;
            for idx in indices {
                if let Some(flag) = self.text_removed.get_mut(idx) {
                    if !*flag {
                        *flag = true;
                        touched = true;
                    }
                }
            }
            touched
        } else {
            false
        }
    }

    /// Compact `out_gpu_texts` / `text_removed` by dropping flagged entries and rebuilding indices.
    fn compact_gpu_texts(&mut self) {
        if self.out_gpu_texts.is_empty() {
            return;
        }
        let mut kept_texts = Vec::with_capacity(self.out_gpu_texts.len());
        let mut kept_flags = Vec::with_capacity(self.text_removed.len());
        let mut rebuilt_indices: HashMap<u32, Vec<usize>> = HashMap::new();
        for (idx, text) in self.out_gpu_texts.iter().enumerate() {
            if self.text_removed.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let new_idx = kept_texts.len();
            kept_texts.push(text.clone());
            kept_flags.push(false);
            rebuilt_indices.entry(text.panel).or_default().push(new_idx);
        }
        self.out_gpu_texts = kept_texts;
        self.text_removed = kept_flags;
        self.panel_text_indices = rebuilt_indices;
    }

    fn reserve_for_text(&mut self, key: (u32, Arc<str>), needed: u32) -> (TextAlloc, u32) {
        // needed can be 0; treat as 1 for allocation accounting but set len later
        let want = Self::round_pow2(needed.max(1));
        if let Some(mut a) = self.text_allocs.get(&key).copied() {
            let prev_len = a.len;
            if want <= a.cap {
                a.len = needed;
                self.text_allocs.insert(key, a);
                return (a, prev_len);
            }
            // grow: alloc new, free old
            let new_start = self.alloc_range(want);
            self.free_range(a.start, a.cap);
            a = TextAlloc {
                start: new_start,
                len: needed,
                cap: want,
            };
            self.text_allocs.insert(key, a);
            return (a, prev_len);
        }
        let start = self.alloc_range(want);
        let a = TextAlloc {
            start,
            len: needed,
            cap: want,
        };
        self.text_allocs.insert(key, a);
        (a, 0)
    }

    fn write_chars_into_slots(&mut self, start: u32, chars: &[GpuChar]) {
        let end = start as usize + chars.len();
        if self.out_gpu_chars.len() < end {
            self.out_gpu_chars.resize(end, Self::zero_char());
        }
        self.out_gpu_chars[start as usize..end].copy_from_slice(chars);
    }
}

impl MiniFontRuntime {
    /// Create render pipeline and static buffers for font.wgsl. Call once after init_gpu().
    pub fn init_render_pipeline(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: TextureFormat,
    ) {
        if self.render.is_some() || self.buffers.is_none() {
            return;
        }
        let bufs = self.buffers.as_ref().unwrap();
        // Bind layout:
        // 0: texture, 1: sampler, 2: glyph_descs (RO storage),
        // 3: global_uniform (RO storage), 4: instances (RO storage),
        // 5: panels (RO storage), 6: panel_deltas (RO storage), 7: debug (RW storage)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mini.render.bgl"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mini.render.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Minimal global uniform storage buffer (zeroed; set screen_size=(1,1))
        let mut gu_bytes = [0u8; 256];
        // Set a safe default for screen_size = (1,1) to avoid div-by-zero in shader aspect ratio.
        // GlobalUniform layout puts screen_size at block 5 (offset 64 bytes).
        let screen_off = 64usize;
        gu_bytes[screen_off..screen_off + 4].copy_from_slice(&1u32.to_le_bytes());
        gu_bytes[screen_off + 4..screen_off + 8].copy_from_slice(&1u32.to_le_bytes());
        let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mini.render.global_uniform.staging"),
            contents: &gu_bytes,
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let global_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mini.render.global_uniform"),
            size: gu_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mini.render.init.copy"),
        });
        enc.copy_buffer_to_buffer(&staging, 0, &global_uniform, 0, gu_bytes.len() as u64);
        queue.submit(Some(enc.finish()));

        // Instance storage buffer (capacity for many glyphs)
        let instance_capacity: u32 = 65536;
        let instance = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mini.render.instance_buffer"),
            size: (instance_capacity as usize * std::mem::size_of::<GpuInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Fallback panel buffers (1 element, zeroed). Can be replaced with UI buffers later.
        let panel_capacity: u32 = 1;
        let panel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mini.render.ui_panels.fallback"),
            size: 256, // large enough for one UI Panel struct
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let panel_delta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mini.render.ui_panel_deltas.fallback"),
            size: 128, // large enough for one PanelAnimDelta struct
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mini.render.bg0"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bufs.render_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs.descriptor_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: global_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: instance.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: panel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: panel_delta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.gpu_debug.buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });

        const quad: [UiVertex; 4] = [
            UiVertex {
                pos: [0.0, 0.0],
                uv: [0.0, 0.0],
            },
            UiVertex {
                pos: [1.0, 0.0],
                uv: [1.0, 0.0],
            },
            UiVertex {
                pos: [1.0, 1.0],
                uv: [1.0, 1.0],
            },
            UiVertex {
                pos: [0.0, 1.0],
                uv: [0.0, 1.0],
            },
        ];

        let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];
        let vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mini.render.quad.vb"),
            contents: bytemuck::cast_slice(&quad),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mini.render.quad.ib"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mini.render.font.shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader/font.wgsl").into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mini.render.pipeline.layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mini.render.pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<UiVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float, // ? 必须与 render pass 的 depth_view 格式一致
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.render = Some(MiniRender {
            bgl,
            bg,
            pipeline,
            vertex,
            index,
            global_uniform,
            sampler,
            instance,
            instance_capacity,
            panel_buffer,
            panel_delta_buffer,
            panel_capacity,
        });
    }

    fn build_instance_slice(&self) -> Vec<GpuInstance> {
        // Build instances by walking each GpuText; compute pen_x using glyph metrics scaled to pixels.
        let mut out: Vec<GpuInstance> = Vec::new();
        for (t_idx, t) in self.out_gpu_texts.iter().enumerate() {
            if self.text_removed.get(t_idx).copied().unwrap_or(false) {
                continue;
            }
            println!("当前需要渲染的gpu text {:?}", t);
            let start = t.sdf_char_index_start_offset;
            let end = t.sdf_char_index_end_offset;
            let mut pen_x_px: f32 = 0.0;
            let mut line_break_acc: u32 = 0;
            let size_px: f32 = t.font_size;
            let origin = [t.position[0], t.position[1]];
            let color = t.color;
            for i in start..end {
                if let Some(ch) = self.out_gpu_chars.get(i as usize) {
                    let desc = self.cpu_glyph_metrics.get(ch.char_index as usize);
                    let (units_per_em, advance_units, line_units) = if let Some(m) = desc {
                        (
                            m.units_per_em.max(1) as f32,
                            m.glyph_advance_width as f32,
                            (m.ascent - m.descent + m.line_gap) as f32,
                        )
                    } else {
                        (1.0_f32, 0.0_f32, size_px)
                    };
                    let break_count = (ch.layout_flags & GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_MASK)
                        >> GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_SHIFT;
                    if break_count > 0 {
                        pen_x_px = 0.0;
                        line_break_acc = line_break_acc.saturating_add(break_count);
                    }
                    let cursor_x = pen_x_px;
                    let advance_px = if units_per_em > 0.0 {
                        advance_units * size_px / units_per_em
                    } else {
                        size_px
                    };
                    let mut line_height_px = if t.line_height > 0.0 {
                        t.line_height
                    } else if units_per_em > 0.0 {
                        line_units / units_per_em * size_px
                    } else {
                        size_px
                    };
                    if !line_height_px.is_finite() || line_height_px <= 0.0 {
                        line_height_px = size_px;
                    }

                    out.push(GpuInstance {
                        char_index: ch.char_index,
                        text_index: ch.gpu_text_index,
                        self_index: ch.self_index,
                        panel_index: ch.panel_index,
                        origin_cursor: [origin[0], origin[1], cursor_x, 0.0],
                        size_px,
                        line_height_px,
                        advance_px,
                        line_break_acc,
                        color,
                        flags: ch.layout_flags,
                        _pad: [0; 3],
                    });
                    pen_x_px += advance_px;
                }
            }
        }
        out
    }

    fn upload_instances_to_gpu(&mut self, queue: &wgpu::Queue) {
        let Some(render) = &self.render else {
            return;
        };
        let all = self.build_instance_slice();
        let cap = render.instance_capacity as usize;
        let count = all.len().min(cap);
        if count == 0 {
            self.draw_instance_count = 0;
            return;
        }
        queue.write_buffer(&render.instance, 0, bytemuck::cast_slice(&all[..count]));
        self.draw_instance_count = count as u32;
        // Fill debug buffer simple counters for readback printing:
        // uints[0] = instance_count, uints[1] = texts, uints[2] = chars
        if let Some(buf) = &self.gpu_debug.buffer {
            let mut uints = [0u32; 3];
            uints[0] = self.draw_instance_count;
            uints[1] = self.out_gpu_texts.len() as u32;
            uints[2] = self.out_gpu_chars.len() as u32;
            // uints array starts at offset 128 (32 floats * 4 bytes)
            queue.write_buffer(buf, 128, bytemuck::cast_slice(&uints));
        }
    }

    /// Optionally adopt UI runtime buffers for panels and panel_deltas.
    /// Pass `Some(panel_deltas)` if available; otherwise a zero fallback is used.
    pub fn set_panel_buffers_external(
        &mut self,
        device: &wgpu::Device,
        panels: &wgpu::Buffer,
        panel_deltas: Option<&wgpu::Buffer>,
    ) {
        // Grab copies of handles to avoid holding overlapping borrows on `self`.
        let (view, desc_buf) = match &self.buffers {
            Some(b) => (b.render_texture_view.clone(), b.descriptor_buffer.clone()),
            None => return,
        };
        let (layout, sampler, global_uniform, instance, panel_delta_fallback) = match &self.render {
            Some(r) => (
                r.bgl.clone(),
                r.sampler.clone(),
                r.global_uniform.clone(),
                r.instance.clone(),
                r.panel_delta_buffer.clone(),
            ),
            None => return,
        };
        let delta_res = match panel_deltas {
            Some(b) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: b,
                offset: 0,
                size: None,
            }),
            None => panel_delta_fallback.as_entire_binding(),
        };
        let new_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mini.render.bg0"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: desc_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: global_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: instance.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: panels,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: delta_res,
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.gpu_debug.buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });
        if let Some(render) = &mut self.render {
            render.bg = new_bg;
        }
    }
}

impl Renderable for MiniFontRuntime {
    fn render<'a>(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _frame_view: &wgpu::TextureView,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        let Some(render) = &self.render else {
            return;
        };
        pass.set_pipeline(&render.pipeline);
        pass.set_bind_group(0, &render.bg, &[]);
        pass.set_vertex_buffer(0, render.vertex.slice(..));
        pass.set_index_buffer(render.index.slice(..), wgpu::IndexFormat::Uint16);
        let instances = self.draw_instance_count;
        if instances > 0 {
            pass.draw_indexed(0..6, 0, 0..instances);
        }
    }

    fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // if(self.gpu_debug.check()) { return; }
        // if let Some(buf) = &self.gpu_debug.buffer {
        //     // Schedule an async readback and print via GpuDebugReadCallBack::print
        //     wgpu::util::DownloadBuffer::read_buffer(
        //         device,
        //         queue,
        //         &buf.slice(..),
        //         move |e| {
        //             if let Ok(bytes) = e {
        //                 let data: &[mile_api::interface::GpuDebugReadCallBack] =
        //                     bytemuck::cast_slice(&bytes);
        //                 for d in data {
        //                     mile_api::interface::GpuDebugReadCallBack::print("MiniFontRuntime.render", d);
        //                 }
        //             }
        //         },
        //     );
        // }
    }

    fn resize(&mut self, size: PhysicalSize<u32>, queue: &wgpu::Queue, _device: &wgpu::Device) {
        // Write screen_size into our local global_uniform so shader can do pixel->NDC conversion.
        if let Some(render) = &self.render {
            let w = size.width.to_le_bytes();
            let h = size.height.to_le_bytes();
            // screen_size is at offset 64 bytes in GlobalUniform
            queue.write_buffer(&render.global_uniform, 64, &w);
            queue.write_buffer(&render.global_uniform, 68, &h);
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

#[derive(Clone, Debug)]
pub struct FontGlyphResult {
    pub character: char,
    pub glyph_id: GlyphId,
    pub outline: Vec<GlyphInstruction>,
    pub glyph_metrics: GlyphMetrics,
}

pub struct FontBatch {
    pub results: Vec<FontGlyphResult>,
}

/// GPU-friendly flattened glyph instruction
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlyphInstruction {
    pub command: u32,
    pub _pad: u32,
    pub data: [f32; 6], // p0, p1, p2 pack
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default, Debug)]
struct MiniFontGlyphDes {
    start_idx: u32,
    end_idx: u32,
    texture_idx_x: u32,
    texture_idx_y: u32,
    // current bbox
    x_min: i32,
    y_min: i32,
    x_max: i32,
    y_max: i32,
    // key metrics
    units_per_em: u32,
    ascent: i32,
    descent: i32,
    line_gap: i32,
    advance_width: u32,
    left_side_bearing: i32,
    glyph_advance_width: u32,
    glyph_left_side_bearing: i32,
}

struct MiniBuffers {
    // compute IO
    instruction_buffer: wgpu::Buffer,
    descriptor_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    storage_texture: wgpu::Texture,
    storage_texture_view: wgpu::TextureView,
    render_texture: wgpu::Texture,
    render_texture_view: wgpu::TextureView,
    // cursor for appended instructions/descriptors
    instruction_cursor: u64,
}

struct MiniCompute {
    bgl: wgpu::BindGroupLayout,
    bg: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

// Minimal render resources for font.wgsl validation
struct MiniRender {
    bgl: wgpu::BindGroupLayout,
    bg: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    global_uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    instance: wgpu::Buffer,
    instance_capacity: u32,
    panel_buffer: wgpu::Buffer,
    panel_delta_buffer: wgpu::Buffer,
    panel_capacity: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
struct UiVertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
struct GpuInstance {
    // identity and indexing
    char_index: u32,
    text_index: u32,
    self_index: u32,
    panel_index: u32,
    // origin.xy + cursor_x + reserved slot
    origin_cursor: [f32; 4],
    size_px: f32,
    line_height_px: f32,
    advance_px: f32,
    line_break_acc: u32,
    color: [f32; 4],
    flags: u32,
    _pad: [u32; 3],
}

#[derive(Clone, Copy, Debug)]
struct TextAlloc {
    start: u32, // slot start in out_gpu_chars
    len: u32,   // current length used
    cap: u32,   // reserved capacity
}

#[derive(Clone, Copy, Debug)]
struct FreeRange {
    start: u32,
    len: u32,
}

/// Minimal text runtime that preserves shader-facing structs.
/// - Maintains per-font char -> glyph index mapping.
/// - Polls events: first BatchFontEntry (dedup + register), then BatchRenderFont (emit GpuChar range + one GpuText).
/// - This file does not perform real GPU buffer uploads yet; it only builds the CPU-side slices.
pub struct MiniFontRuntime {
    register: RegisterEvent,

    // glyph index per font; maps a character to a glyph index for that font
    glyph_index: HashMap<Arc<str>, HashMap<char, u32>>,

    // font faces cache
    fonts: HashMap<Arc<str>, FontInstance>,

    // CPU-side glyph descriptors and metrics by glyph index for layout
    cpu_glyph_descs: Vec<MiniFontGlyphDes>,
    cpu_glyph_metrics: Vec<GlyphMetrics>,

    // logical output buffers; in a full integration these would be uploaded to GPU buffers
    pub out_gpu_chars: Vec<GpuChar>,
    pub out_gpu_texts: Vec<GpuText>,
    // track active text indices per panel (multiple layouts per panel supported)
    panel_text_indices: HashMap<u32, Vec<usize>>,
    // logical deletion flags for out_gpu_texts (keeps indices stable)
    text_removed: Vec<bool>,

    // dynamic text storage for frequent updates
    text_allocs: HashMap<(u32, Arc<str>), TextAlloc>, // key: (panel_id, font_path)
    free_list: Vec<FreeRange>,                        // free ranges over out_gpu_chars

    // atlas tile cursor (64x64 tiles for 4096x4096 with 64px tile)
    tile_cursor: u32,

    // cursor into out_gpu_chars; also serves as start/end offsets in GpuText
    gpu_char_cursor: u32,
    // per-char one quad; linear counter for demo purposes
    quad_index: u32,

    // GPU compute resources
    buffers: Option<MiniBuffers>,
    compute: Option<MiniCompute>,
    render: Option<MiniRender>,
    // whether new data uploaded and requires compute
    is_update: bool,
    // debug readback
    gpu_debug: GpuDebug,
    // number of instances to draw (clamped to instance buffer capacity)
    draw_instance_count: u32,
}

impl MiniFontRuntime {
    pub fn new() -> Self {
        Self {
            register: RegisterEvent::new(global_event_bus()),
            glyph_index: HashMap::new(),
            fonts: HashMap::new(),
            panel_text_indices: HashMap::new(),
            text_removed: Vec::new(),
            cpu_glyph_descs: Vec::new(),
            cpu_glyph_metrics: Vec::new(),
            out_gpu_chars: Vec::new(),
            out_gpu_texts: Vec::new(),
            text_allocs: HashMap::new(),
            free_list: Vec::new(),
            tile_cursor: 0,
            gpu_char_cursor: 0,
            quad_index: 0,
            buffers: None,
            compute: None,
            render: None,
            is_update: false,
            gpu_debug: GpuDebug::new("MiniFontRuntime"),
            draw_instance_count: 0,
        }
    }

    /// Load a font file into the runtime's face cache.
    /// Same intent as previous `load_to_face` in structs.rs.
    pub fn load_to_face(&mut self, font_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let path: Arc<str> = Arc::from(font_path.to_string());
        self.ensure_face_loaded(path)
    }

    /// Convenience: load the demo font used in examples.
    pub fn load_font_file(&mut self) {
        // Prefer repository path under tf/ first; fall back handled in ensure_face_loaded
        if let Err(e) = self.load_to_face("tf/BIZUDPGothic-Regular.ttf") {
            eprintln!(
                "MiniFontRuntime.load_font_file: failed to load default face: {}",
                e
            );
        } else {
            // loaded
        }
    }

    /// Initialize minimal buffers and compute pipeline (idempotent).
    pub fn init_gpu(&mut self, device: &wgpu::Device) {
        if self.buffers.is_some() && self.compute.is_some() {
            return;
        }
        // debug buffer for readback
        self.gpu_debug.create_buffer(device);
        // buffers
        // Note: out_gpu_chars is CPU-side staging for now; a real instance buffer can be added later.
        let instruction_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mini.instruction_buffer"),
            size: (std::mem::size_of::<GlyphInstruction>() * 8096) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let descriptor_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mini.descriptor_buffer"),
            size: (std::mem::size_of::<MiniFontGlyphDes>() * 8096) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // params uniform buffer (base_index, glyph_count, pad0, pad1)
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mini.params_buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let storage_texture = device.create_texture(&TextureDescriptor {
            label: Some("mini.storage_sdf"),
            size: wgpu::Extent3d {
                width: 4096,
                height: 4096,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let render_texture = device.create_texture(&TextureDescriptor {
            label: Some("mini.render_sdf"),
            size: wgpu::Extent3d {
                width: 4096,
                height: 4096,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let storage_texture_view = storage_texture.create_view(&TextureViewDescriptor::default());
        let render_texture_view = render_texture.create_view(&TextureViewDescriptor::default());

        let buffers = MiniBuffers {
            instruction_buffer,
            descriptor_buffer,
            params_buffer,
            storage_texture,
            storage_texture_view,
            render_texture,
            render_texture_view,
            instruction_cursor: 0,
        };

        // compute pipeline
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mini.compute.bgl"),
            entries: &[
                // texture_storage_2d<rgba16float, write>
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // glyph_instructions
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
                // glyph_descs
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
                // debug buffer
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
                // params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mini.batch_enqueue_font"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader/batch_enqueue_font.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mini.compute.layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mini.compute.pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mini.compute.bg0"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&buffers.storage_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.instruction_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.descriptor_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.gpu_debug.buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.params_buffer.as_entire_binding(),
                },
            ],
        });

        self.buffers = Some(buffers);
        self.compute = Some(MiniCompute { bgl, bg, pipeline });
    }

    /// write descriptors/instructions into global SSBOs; return base index for this batch.
    fn write_batch_buffer(
        &mut self,
        queue: &wgpu::Queue,
        fond_des: &[MiniFontGlyphDes],
        instruction: &[GlyphInstruction],
    ) -> u32 {
        if fond_des.is_empty() && instruction.is_empty() {
            return 0;
        }
        let Some(bufs) = self.buffers.as_mut() else {
            // not initialized; nothing to do
            return 0;
        };
        let base = bufs.instruction_cursor as u32;
        let ins_offset = bufs.instruction_cursor * std::mem::size_of::<GlyphInstruction>() as u64;
        let des_offset = bufs.instruction_cursor * std::mem::size_of::<MiniFontGlyphDes>() as u64;

        queue.write_buffer(
            &bufs.instruction_buffer,
            ins_offset,
            bytemuck::cast_slice(instruction),
        );
        queue.write_buffer(
            &bufs.descriptor_buffer,
            des_offset,
            bytemuck::cast_slice(fond_des),
        );

        bufs.instruction_cursor += fond_des.len() as u64;
        self.is_update = true;
        base
    }

    fn batch_enqueue_compute(&self, device: &wgpu::Device, queue: &wgpu::Queue, glyph_count: u32) {
        if self.buffers.is_none() || self.compute.is_none() {
            return;
        }
        let compute = self.compute.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mini.compute.encoder.n"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mini.compute.pass.n"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute.pipeline);
            cpass.set_bind_group(0, &compute.bg, &[]);
            cpass.dispatch_workgroups(8, 8, glyph_count.max(1));
        }
        queue.submit(Some(encoder.finish()));
    }
    fn batch_enqueue_compute_n(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        glyph_count: u32,
    ) {
        if self.buffers.is_none() || self.compute.is_none() {
            return;
        }
        let compute = self.compute.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mini.compute.encoder.n"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mini.compute.pass.n"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute.pipeline);
            cpass.set_bind_group(0, &compute.bg, &[]);
            // Z dimension enumerates glyphs in this batch; one 64x64 tile per glyph.
            cpass.dispatch_workgroups(8, 8, glyph_count.max(1));
        }
        queue.submit(Some(encoder.finish()));
    }

    fn copy_store_texture_to_render_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if !self.is_update {
            return;
        }
        let Some(bufs) = self.buffers.as_ref() else {
            return;
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mini.copy.encoder"),
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &bufs.storage_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &bufs.render_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: 4096,
                height: 4096,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        self.is_update = false;
    }

    fn ensure_face_loaded(
        &mut self,
        font_path: Arc<str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::path::{Path, PathBuf};

        if self.fonts.contains_key(&font_path) {
            return Ok(());
        }

        // Resolve a few candidate paths to make running from different CWDs robust.
        let name_path = Path::new(font_path.as_ref());
        let file_name = name_path.file_name().map(|s| s.to_os_string());

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root: Option<PathBuf> = manifest_dir.parent().map(|p| p.to_path_buf());
        let cwd = std::env::current_dir().ok();

        let mut candidates: Vec<PathBuf> = Vec::new();
        // 1) As provided
        candidates.push(PathBuf::from(font_path.as_ref()));
        // 2) CWD joined
        if let Some(ref d) = cwd {
            candidates.push(d.join(font_path.as_ref()));
        }
        // 3) crate manifest dir joined
        candidates.push(manifest_dir.join(font_path.as_ref()));
        // 4) workspace root + provided
        if let Some(ref root) = workspace_root {
            candidates.push(root.join(font_path.as_ref()));
        }
        // 5) workspace root + "ttf/<file>"
        if let (Some(ref root), Some(ref fname)) = (workspace_root.as_ref(), file_name.as_ref()) {
            candidates.push(root.join("ttf").join(fname));
        }
        // 6) workspace root + "tf/<file>"
        if let (Some(ref root), Some(ref fname)) = (workspace_root.as_ref(), file_name.as_ref()) {
            candidates.push(root.join("tf").join(fname));
        }
        // 7) plain "ttf/<file>" from CWD
        if let Some(ref fname) = file_name {
            candidates.push(PathBuf::from("ttf").join(fname));
        }
        // 8) plain "tf/<file>" from CWD
        if let Some(ref fname) = file_name {
            candidates.push(PathBuf::from("tf").join(fname));
        }

        let selected = candidates.into_iter().find(|p| p.exists());
        let selected = match selected {
            Some(p) => p,
            None => {
                let cwd_dbg = std::env::current_dir()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| "<unknown>".into());
                return Err(
                    format!("font file not found: '{}' (cwd: {})", font_path, cwd_dbg).into(),
                );
            }
        };

        let buffer = std::fs::read(&selected)?;
        // Leak to 'static for ttf_parser::Face<'static>
        let leaked: &'static [u8] = Box::leak(buffer.clone().into_boxed_slice());
        let face = ttf_parser::Face::parse(leaked, 0)?;
        self.fonts
            .insert(font_path, FontInstance { data: buffer, face });
        Ok(())
    }

    fn queue_batch_parse_outlines<I>(
        &mut self,
        font_name: Arc<str>,
        text_batch: I,
        _font_size: u32,
    ) -> Result<FontBatch, Box<dyn std::error::Error>>
    where
        I: IntoIterator<Item = char>,
    {
        self.ensure_face_loaded(font_name.clone())?;
        let face = &self.fonts.get(&font_name).unwrap().face;

        let mut results: Vec<FontGlyphResult> = Vec::new();
        for ch in text_batch {
            if let Some(glyph_id) = face.glyph_index(ch) {
                let mut builder = GlyphBuilder::new();
                if face.outline_glyph(glyph_id, &mut builder).is_some() {
                    let bbox = face.global_bounding_box();
                    let metrics = GlyphMetrics {
                        units_per_em: face.units_per_em() as u32,
                        ascent: face.ascender() as i32,
                        descent: face.descender() as i32,
                        line_gap: face.line_gap() as i32,
                        advance_width: face.glyph_hor_advance(glyph_id).unwrap_or(0) as u32,
                        left_side_bearing: face.glyph_hor_side_bearing(glyph_id).unwrap_or(0)
                            as i32,

                        x_min: bbox.x_min as i32,
                        y_min: bbox.y_min as i32,
                        x_max: bbox.x_max as i32,
                        y_max: bbox.y_max as i32,

                        glyph_advance_width: face.glyph_hor_advance(glyph_id).unwrap_or(0) as u32,
                        glyph_left_side_bearing: face.glyph_hor_side_bearing(glyph_id).unwrap_or(0)
                            as i32,

                        glyph_ver_advance: face.glyph_ver_advance(glyph_id).unwrap_or(0) as u32,
                        glyph_ver_side_bearing: face.glyph_ver_side_bearing(glyph_id).unwrap_or(0)
                            as i32,
                    };

                    results.push(FontGlyphResult {
                        character: ch,
                        glyph_id,
                        outline: builder.instructions.clone(),
                        glyph_metrics: metrics,
                    });
                }
            }
        }
        Ok(FontBatch { results })
    }

    fn build_gpu_slices(
        &mut self,
        batch: &FontBatch,
        start_offset: u64,
    ) -> (Vec<MiniFontGlyphDes>, Vec<GlyphInstruction>) {
        let mut glyph_descs = Vec::with_capacity(batch.results.len());
        let mut gpu_instructions = Vec::new();
        let mut current_offset = start_offset as u32;
        for glyph in &batch.results {
            let start_idx = current_offset;
            let end_idx = start_idx + glyph.outline.len() as u32;
            let m = &glyph.glyph_metrics;
            // assign atlas tile sequentially
            let tex_x = self.tile_cursor % 64;
            let tex_y = self.tile_cursor / 64;
            self.tile_cursor = (self.tile_cursor + 1) % (64 * 64);
            glyph_descs.push(MiniFontGlyphDes {
                start_idx,
                end_idx,
                texture_idx_x: tex_x,
                texture_idx_y: tex_y,
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
                glyph_advance_width: m.glyph_advance_width,
                glyph_left_side_bearing: m.glyph_left_side_bearing,
            });
            gpu_instructions.extend_from_slice(&glyph.outline);
            current_offset = end_idx;
        }
        (glyph_descs, gpu_instructions)
    }

    /// Incrementally register glyphs for a font.
    /// Minimal behavior: dedup, skip cached, assign consecutive indices per font.
    // removed: legacy naive queue_batch_parse; glyph_index is only updated after GPU upload now.

    fn reset_runtime_state(&mut self) {
        self.glyph_index.clear();
        self.fonts.clear();
        self.cpu_glyph_descs.clear();
        self.cpu_glyph_metrics.clear();
        self.out_gpu_chars.clear();
        self.out_gpu_texts.clear();
        self.panel_text_indices.clear();
        self.text_removed.clear();
        self.text_allocs.clear();
        self.free_list.clear();
        self.tile_cursor = 0;
        self.gpu_char_cursor = 0;
        self.quad_index = 0;
        self.buffers = None;
        self.compute = None;
        self.render = None;
        self.is_update = false;
        self.draw_instance_count = 0;
        self.gpu_debug = GpuDebug::new("MiniFontRuntime");
    }

    /// Poll once: process batch font entries, then render texts.
    /// - For BatchFontEntry: group by font, dedup chars, register into glyph_index.
    /// - For BatchRenderFont: map chars to glyph indices, emit a contiguous GpuChar range and one GpuText.
    pub fn poll_global_event(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (batch_entry, batch_render, batch_remove, font_reset) = self.register.poll();
        if !font_reset.is_empty() {
            self.reset_runtime_state();
        }

        if batch_entry.is_empty() && batch_render.is_empty() && batch_remove.is_empty() {
            return;
        }
        println!(
            "events => add_glyph:{} render_text:{} remove_text:{}",
            batch_entry.len(),
            batch_render.len(),
            batch_remove.len()
        );

        // 0) Handle RemoveRenderFont first: clear per-panel texts/allocations (keep SDF/desc)
        if !batch_remove.is_empty() {
            let mut did_remove = false;
            for e in &batch_remove {
                did_remove |= self.remove_panel_texts(e.parent.0);
            }
            if did_remove {
                self.compact_gpu_texts();
                // 3) 实例缓冲重建
                self.upload_instances_to_gpu(queue);
            }
        }

        // 1) Handle BatchFontEntry
        {
            let mut grouped: HashMap<Arc<str>, HashSet<char>> = HashMap::new();
            for e in &batch_entry {
                let set = grouped.entry(e.font_file_path.clone()).or_default();
                for ch in e.text.chars() {
                    set.insert(ch);
                }
            }
            let mut did_any_upload = false;
            for (font, set) in grouped {
                let mut new_chars: Vec<char> = Vec::new();
                for &ch in set.iter() {
                    let present = self
                        .glyph_index
                        .get(&font)
                        .map_or(false, |m| m.contains_key(&ch));
                    if !present {
                        new_chars.push(ch);
                    }
                }
                if new_chars.is_empty() {
                    continue;
                }
                let batch = match self.queue_batch_parse_outlines(
                    font.clone(),
                    new_chars.iter().copied(),
                    16,
                ) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let start = self
                    .buffers
                    .as_ref()
                    .map(|b| b.instruction_cursor)
                    .unwrap_or(0);
                let (des, instr) = self.build_gpu_slices(&batch, start);
                let base = self.write_batch_buffer(queue, &des, &instr);
                {
                    let map = self.glyph_index.entry(font.clone()).or_default();
                    for (i, glyph) in batch.results.iter().enumerate() {
                        map.insert(glyph.character, base + i as u32);
                    }
                }
                // Keep CPU-side copies for layout
                {
                    let needed = (base as usize) + des.len();
                    if self.cpu_glyph_descs.len() < needed {
                        self.cpu_glyph_descs
                            .resize(needed, MiniFontGlyphDes::default());
                    }
                    if self.cpu_glyph_metrics.len() < needed {
                        self.cpu_glyph_metrics.resize(
                            needed,
                            GlyphMetrics {
                                units_per_em: 0,
                                ascent: 0,
                                descent: 0,
                                line_gap: 0,
                                advance_width: 0,
                                left_side_bearing: 0,
                                x_min: 0,
                                y_min: 0,
                                x_max: 0,
                                y_max: 0,
                                glyph_advance_width: 0,
                                glyph_left_side_bearing: 0,
                                glyph_ver_advance: 0,
                                glyph_ver_side_bearing: 0,
                            },
                        );
                    }
                    for (i, d) in des.iter().enumerate() {
                        self.cpu_glyph_descs[(base as usize) + i] = *d;
                    }
                    for (i, g) in batch.results.iter().enumerate() {
                        self.cpu_glyph_metrics[(base as usize) + i] = g.glyph_metrics.clone();
                    }
                }
                if let Some(bufs) = self.buffers.as_ref() {
                    let glyph_count = des.len() as u32;
                    let params: [u32; 4] = [base, glyph_count, 0, 0];
                    queue.write_buffer(&bufs.params_buffer, 0, bytemuck::cast_slice(&params));
                    self.batch_enqueue_compute_n(device, queue, glyph_count);
                    did_any_upload = true;
                }
            }

            if did_any_upload {
                self.copy_store_texture_to_render_texture(device, queue);
                self.gpu_debug.raw_debug(device, queue);
                self.is_update = false;
            }

            for e in &batch_render {
                let Some(char_map) = self.glyph_index.get(&e.font_file_path).cloned() else {
                    continue;
                };
                let panel_id = e.parent.0;
                // Ensure stale slices are removed even if RemoveRenderFont hasn't been handled yet
                self.retire_panel_text_indices(panel_id);

                // gather glyph indices + newline markers first
                let mut glyph_entries: Vec<(u32, u32)> = Vec::new();
                let mut pending_line_breaks: u32 = 0;
                let mut chars = e.text.chars().peekable();
                while let Some(ch) = chars.next() {
                    if ch == '\r' {
                        if matches!(chars.peek(), Some('\n')) {
                            chars.next();
                        }
                        pending_line_breaks = pending_line_breaks.saturating_add(1);
                        continue;
                    }
                    if ch == '\n' {
                        pending_line_breaks = pending_line_breaks.saturating_add(1);
                        continue;
                    }
                    if let Some(&idx) = char_map.get(&ch) {
                        let mut flags = 0u32;
                        if pending_line_breaks > 0 {
                            let encodeable =
                                pending_line_breaks.min(GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_MAX);
                            flags |= GPU_CHAR_LAYOUT_FLAG_LINE_BREAK_BEFORE;
                            flags |= encodeable << GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_SHIFT;
                            pending_line_breaks = pending_line_breaks.saturating_sub(encodeable);
                        }
                        glyph_entries.push((idx, flags));
                    }
                }
                let needed = glyph_entries.len() as u32;
                if needed == 0 {
                    continue;
                }

                // key by (panel_id, font_path) so the same panel updates in place when possible
                let key = (e.parent.0, e.font_file_path.clone());
                let (alloc, prev_len) = self.reserve_for_text(key, needed);

                // build chars for the reserved slot range
                let this_text_index = self.out_gpu_texts.len() as u32;
                let mut chars: Vec<GpuChar> = Vec::with_capacity(glyph_entries.len());
                for (i, &(ci, flags)) in glyph_entries.iter().enumerate() {
                    let metrics =
                        self.cpu_glyph_metrics
                            .get(ci as usize)
                            .cloned()
                            .unwrap_or(GlyphMetrics {
                                units_per_em: 1,
                                ascent: 0,
                                descent: 0,
                                line_gap: 0,
                                advance_width: 0,
                                left_side_bearing: 0,
                                x_min: 0,
                                y_min: 0,
                                x_max: 0,
                                y_max: 0,
                                glyph_advance_width: 0,
                                glyph_left_side_bearing: 0,
                                glyph_ver_advance: 0,
                                glyph_ver_side_bearing: 0,
                            });
                    chars.push(GpuChar {
                        char_index: ci,
                        gpu_text_index: this_text_index,
                        panel_index: e.parent.0,
                        self_index: i as u32,
                        glyph_advance_width: metrics.glyph_advance_width,
                        glyph_left_side_bearing: metrics.glyph_left_side_bearing,
                        glyph_ver_advance: metrics.glyph_ver_advance,
                        glyph_ver_side_bearing: metrics.glyph_ver_side_bearing,
                        layout_flags: flags,
                    });
                }
                self.write_chars_into_slots(alloc.start, &chars);
                let prev_len = prev_len as usize;
                if prev_len > chars.len() {
                    let start_zero = alloc.start as usize + chars.len();
                    let end_zero = alloc.start as usize + prev_len;
                    for slot in &mut self.out_gpu_chars[start_zero..end_zero] {
                        *slot = Self::zero_char();
                    }
                }

                // default small container size; position kept at origin for now
                let gpu_text = GpuText {
                    sdf_char_index_start_offset: alloc.start,
                    sdf_char_index_end_offset: alloc.start + needed,
                    font_size: e.font_style.font_size as f32,
                    size: 256,
                    color: e.font_style.font_color,
                    panel: panel_id,
                    position: [0.0, 0.0],
                    line_height: if e.font_style.font_line_height > 0 {
                        e.font_style.font_line_height as f32
                    } else {
                        0.0
                    },
                };
                // Debug print
                println!(
                    "GpuText generated -> start:{} end:{} font_size:{} color:[{:.2},{:.2},{:.2},{:.2}] text:\"{}\" panel:{}",
                    gpu_text.sdf_char_index_start_offset,
                    gpu_text.sdf_char_index_end_offset,
                    gpu_text.font_size,
                    gpu_text.color[0],
                    gpu_text.color[1],
                    gpu_text.color[2],
                    gpu_text.color[3],
                    &e.text,
                    e.parent.0
                );
                let new_index = self.out_gpu_texts.len();
                self.out_gpu_texts.push(gpu_text);
                self.text_removed.push(false);
                let entry = self.panel_text_indices.entry(e.parent.0).or_default();
                entry.clear();
                entry.push(new_index);
            }
            // After texts/chars updated, upload instance data for rendering
            self.upload_instances_to_gpu(queue);
        } // end fn poll_once
    }

    /// 清理指定 panel 的文本实例与动态分配，返回是否有修改
    fn remove_panel_texts(&mut self, panel_id: u32) -> bool {
        let mut touched = self.retire_panel_text_indices(panel_id);
        if !touched {
            for (idx, text) in self.out_gpu_texts.iter().enumerate() {
                if text.panel == panel_id {
                    if let Some(flag) = self.text_removed.get_mut(idx) {
                        if !*flag {
                            *flag = true;
                            touched = true;
                        }
                    }
                }
            }
        }
        let keys_to_remove: Vec<(u32, Arc<str>)> = self
            .text_allocs
            .keys()
            .filter(|(pid, _)| *pid == panel_id)
            .cloned()
            .collect();
        for key in keys_to_remove {
            if let Some(alloc) = self.text_allocs.remove(&key) {
                self.free_range(alloc.start, alloc.cap);
                touched = true;
            }
        }
        touched
    }
}
