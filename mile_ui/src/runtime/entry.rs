//! High level UI runtime orchestrator.
//!
//! This module exposes `MuiRuntime`, a lightweight façade that ties together the new
//! runtime building blocks (buffer arena, render batches, CPU state) while also wiring
//! in the global event bus and UI database.  The goal is to offer a centralized entry
//! point that gradually replaces the gigantic `GpuUi` struct in `mui.rs`.

use std::{
    any::TypeId,
    cell::RefCell,
    collections::{HashMap, HashSet},
    num::NonZeroU32,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

use super::{
    buffers::{BufferArena, BufferArenaConfig, BufferViewSet},
    compute::{ComputePipelines, FrameComputeContext},
    render::{QuadBatchKind, RenderPipelines},
    state::{
        CpuPanelEvent, FrameState, PanelEventRegistry, RuntimeState, StateTransition, UIEventHub,
    },
};
use crate::{
    mui_anim::{AnimProperty, AnimTargetValue, AnimationSpec, Easing},
    mui_prototype::{
        PanelBinding, PanelKey, PanelPayload, PanelRecord, PanelSnapshot, PanelStateOverrides,
        ShaderStage, UiState, install_runtime_event_bridge, registered_panel_keys,
        take_pending_shader,
    },
    runtime::_ty::{
        AnimtionFieldOffsetPtr, GpuAnimationDes, GpuInteractionFrame, Panel, TransformAnimFieldInfo,
    },
    structs::{AnimOp, EasingMask, MouseState, PanelField},
    util::texture_atlas_store::{self, GpuUiTextureInfo, TextureAtlasStore, UiTextureInfo},
};
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice, offset_of};
use mile_api::{
    event_bus::{EventBus, EventStream},
    global::{global_db, global_event_bus},
    interface::{Computeable, GlobalUniform},
    prelude::{GpuDebug, GpuDebugReadCallBack, Renderable},
};
use mile_db::{DbError, MileDb};
use mile_gpu_dsl::{
    gpu_ast_core::event::KennelResultIdxEvent,
    program_pipeline::render_binding::RenderBindingResources,
};
use serde::de;
use wgpu::{
    DepthBiasState, Device, Queue,
    util::{BufferInitDescriptor, DeviceExt, DownloadBuffer},
};

/// Tracks the previous and current frame interaction state so transitions
/// (e.g. click release, hover enter) can be processed without relying on GPU readback.
#[derive(Debug, Default, Clone)]
pub struct FrameHistory {
    pub previous: FrameSnapshot,
    pub current: FrameSnapshot,
}

#[derive(Debug, Default, Clone)]
pub struct FrameSnapshot {
    pub click_panel: Option<u32>,
    pub hover_panel: Option<u32>,
    pub frame_index: u32,
}

impl FrameHistory {
    pub fn advance(&mut self, frame_index: u32) {
        self.previous = self.current.clone();
        self.current.frame_index = frame_index;
    }
}

/// Maintains the CPU copy of `GlobalUniform` together with its GPU buffer.
pub struct GlobalUniformState {
    cpu: GlobalUniform,
    buffer: wgpu::Buffer,
}

impl GlobalUniformState {
    pub fn new(device: &Device) -> Self {
        let cpu = GlobalUniform::default();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ui::global-uniform"),
            contents: bytes_of(&cpu),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        Self { cpu, buffer }
    }

    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    #[inline]
    pub fn cpu(&self) -> &GlobalUniform {
        &self.cpu
    }

    #[inline]
    pub fn cpu_mut(&mut self) -> &mut GlobalUniform {
        &mut self.cpu
    }

    fn write_field<T>(&mut self, queue: &Queue, offset: wgpu::BufferAddress, value: &T)
    where
        T: bytemuck::Pod,
    {
        queue.write_buffer(&self.buffer, offset, bytes_of(value));
    }
}

#[derive(Default)]
pub struct TextureGpuData {
    pub atlas_ids: Vec<u32>,
    pub views: Vec<wgpu::TextureView>,
    pub samplers: Vec<wgpu::Sampler>,
    pub infos: Vec<GpuUiTextureInfo>,
}

impl TextureGpuData {
    pub fn is_ready(&self) -> bool {
        !self.views.is_empty() && !self.samplers.is_empty() && !self.infos.is_empty()
    }
}

pub struct TextureBindings {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub info_buffer: wgpu::Buffer,
    pub atlas_ids: Vec<u32>,
    pub texture_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PanelStateCpu {
    pub overrides: PanelStateOverrides,
    pub texture: Option<UiTextureInfo>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PanelCpuDescriptor {
    pub key: PanelKey,
    pub default_state: Option<UiState>,
    pub current_state: UiState,
    pub display_state: UiState,
    pub snapshot: PanelSnapshot,
    pub snapshot_texture: Option<UiTextureInfo>,
    pub states: HashMap<UiState, PanelStateCpu>,
    pub type_id: TypeId,
    pub quad_vertex: QuadBatchKind,
}

/// Central runtime orchestrator that will eventually replace `GpuUi`.
pub struct MuiRuntime {
    pub buffers: BufferArena,
    pub render: RenderPipelines,
    pub compute: RefCell<ComputePipelines>,
    pub state: RuntimeState,
    pub frame_history: FrameHistory,
    pub global_uniform: GlobalUniformState,
    pub texture_atlas_store: TextureAtlasStore,
    pub texture_bindings: Option<TextureBindings>,
    pub render_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub render_bind_group: Option<wgpu::BindGroup>,
    pub render_surface_format: Option<wgpu::TextureFormat>,
    pub kennel_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub kennel_bind_group: Option<wgpu::BindGroup>,
    pub global_event_bus: &'static EventBus,
    pub global_db: &'static MileDb,
    shader_events: EventStream<KennelResultIdxEvent>,
    pub panel_cache: HashMap<PanelKey, PanelCpuDescriptor>,
    pub panel_instances: Vec<Panel>,
    panel_instances_dirty: bool,
    transitioning_panels: HashMap<u32, f32>,
    animation_descriptor: GpuAnimationDes,
    animation_field_cache: HashMap<(u32, u32), u32>,
    pub trace: RefCell<GpuDebug>,
}

impl Renderable for MuiRuntime {
    fn render<'a>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_view: &wgpu::TextureView,
        mut pass: &mut wgpu::RenderPass<'a>,
    ) {
        self.encode_panels(&mut pass);
    }

    fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut trace = self.trace.borrow_mut();
        trace.debug(device, queue);
    }

    fn resize(
        &mut self,
        size: winit::dpi::PhysicalSize<u32>,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
    ) {
        let screen = self.global_uniform.cpu_mut();
        screen.screen_size = [size.width, size.height];
        let state_offset = offset_of!(GlobalUniform, screen_size) as wgpu::BufferAddress;
        queue.write_buffer(
            &self.global_uniform.buffer,
            state_offset,
            bytemuck::bytes_of(&[size.width, size.height]),
        );
    }
}

impl Computeable for MuiRuntime {
    fn encode(&self, pass: &mut wgpu::ComputePass<'_>) {
        let buffers = self.buffer_views();
        let ctx = FrameComputeContext {
            queue: None,
            frame_index: self.state.frame_state.frame_index,
            panel_count: self.panel_instances.len() as u32,
        };
        self.compute.borrow_mut().encode_all(pass, &buffers, &ctx);
    }

    fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let ctx = FrameComputeContext {
            queue: Some(queue),
            frame_index: self.state.frame_state.frame_index,
            panel_count: self.panel_instances.len() as u32,
        };
        self.compute.borrow_mut().readback_all(device, queue, &ctx);
    }

    fn is_dirty(&self) -> bool {
        self.compute.borrow().is_any_dirty()
    }
}

impl MuiRuntime {
    const PANEL_STRIDE: wgpu::BufferAddress = std::mem::size_of::<Panel>() as wgpu::BufferAddress;

    pub fn write_global_buffer<T: Pod + Zeroable>(
        &self,
        queue: &Queue,
        offset: wgpu::BufferAddress,
        value: T,
    ) {
        queue.write_buffer(
            &self.global_uniform.buffer,
            offset,
            bytemuck::bytes_of(&value),
        );
    }

    /// Construct a new runtime using the supplied GPU device.
    pub fn new(device: &Device, arena_cfg: BufferArenaConfig) -> Self {
        let buffers = BufferArena::new(device, &arena_cfg);
        let render = RenderPipelines::new(device);
        let global_uniform = GlobalUniformState::new(device);
        let mut state = RuntimeState::default();
        let hub = state.event_hub.clone();
        let registry = state.panel_events.clone();
        let compute = ComputePipelines::new(device, &buffers, global_uniform.buffer(), hub.clone());
        install_runtime_event_bridge(registry, hub.clone());
        let shader_events = global_event_bus().subscribe::<KennelResultIdxEvent>();

        Self {
            texture_atlas_store: TextureAtlasStore::default(),
            texture_bindings: None,
            render_bind_group_layout: None,
            render_bind_group: None,
            render_surface_format: None,
             kennel_bind_group_layout: None,
             kennel_bind_group: None,
            buffers,
            render,
            compute: RefCell::new(compute),
            state,
            frame_history: FrameHistory::default(),
            global_uniform,
            global_event_bus: global_event_bus(),
            global_db: global_db(),
            shader_events,
            panel_cache: HashMap::new(),
            panel_instances: Vec::new(),
            panel_instances_dirty: true,
            transitioning_panels: HashMap::new(),
            animation_descriptor: GpuAnimationDes::default(),
            animation_field_cache: HashMap::new(),
            trace: RefCell::new(GpuDebug::new("mui_runtime")),
        }
    }

    /**
     * 调用图片材质总合成入�?     */
    pub fn read_all_texture(&mut self) {
        self.texture_atlas_store.read_all_image();
    }

    #[inline]
    pub fn buffers(&self) -> &BufferArena {
        &self.buffers
    }

    #[inline]
    pub fn buffers_mut(&mut self) -> &mut BufferArena {
        &mut self.buffers
    }

    #[inline]
    pub fn render(&self) -> &RenderPipelines {
        &self.render
    }

    #[inline]
    pub fn render_mut(&mut self) -> &mut RenderPipelines {
        &mut self.render
    }

    #[inline]
    pub fn runtime_state(&self) -> &RuntimeState {
        &self.state
    }

    #[inline]
    pub fn runtime_state_mut(&mut self) -> &mut RuntimeState {
        &mut self.state
    }

    #[inline]
    pub fn global_uniform(&self) -> &GlobalUniformState {
        &self.global_uniform
    }

    #[inline]
    pub fn global_uniform_mut(&mut self) -> &mut GlobalUniformState {
        &mut self.global_uniform
    }

    #[inline]
    pub fn texture_store(&self) -> &TextureAtlasStore {
        &self.texture_atlas_store
    }

    #[inline]
    pub fn texture_store_mut(&mut self) -> &mut TextureAtlasStore {
        &mut self.texture_atlas_store
    }

    /// Load a single texture asset into the atlas store and return metadata.
    pub fn load_texture<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Option<texture_atlas_store::UiTextureInfo> {
        self.texture_atlas_store.read_img(path.as_ref())
    }

    pub fn upload_textures_to_gpu(&mut self, device: &Device, queue: &Queue) {
        self.texture_atlas_store.upload_all_to_gpu(device, queue);
    }

    pub fn texture_gpu_data(&self) -> TextureGpuData {
        let mut views = Vec::new();
        let mut samplers = Vec::new();
        let mut atlas_ids = Vec::new();
        let mut slot_map = HashMap::new();

        for atlas_id in self.texture_atlas_store.atlas_ids_sorted() {
            if let Some(atlas) = self.texture_atlas_store.atlas(atlas_id) {
                if let (Some(view), Some(sampler)) = (&atlas.texture_view, &atlas.sampler) {
                    let slot = views.len() as u32;
                    slot_map.insert(atlas.index, slot);
                    atlas_ids.push(atlas_id);
                    views.push(view.clone());
                    samplers.push(sampler.clone());
                }
            }
        }

        let infos = self
            .texture_atlas_store
            .build_gpu_texture_infos_with_slots(&slot_map);

        TextureGpuData {
            atlas_ids,
            views,
            samplers,
            infos,
        }
    }

    #[inline]
    pub fn texture_count(&self) -> u32 {
        self.texture_atlas_store.texture_count()
    }

    #[inline]
    pub fn texture_bindings(&self) -> Option<&TextureBindings> {
        self.texture_bindings.as_ref()
    }

    #[inline]
    pub fn texture_bindings_mut(&mut self) -> Option<&mut TextureBindings> {
        self.texture_bindings.as_mut()
    }

    pub fn install_kennel_bindings(&mut self, resources: &RenderBindingResources) {
        self.kennel_bind_group_layout = Some(resources.bind_group_layout.clone());
        self.kennel_bind_group = Some(resources.bind_group.clone());
    }

    pub fn panel_descriptor(&self, key: &PanelKey) -> Option<&PanelCpuDescriptor> {
        self.panel_cache.get(key)
    }

    pub fn render_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.render_bind_group.as_ref()
    }

    #[inline]
    pub fn buffer_views(&self) -> BufferViewSet<'_> {
        self.buffers.staging_guard()
    }

    #[inline]
    pub fn panel_stride() -> wgpu::BufferAddress {
        Self::PANEL_STRIDE
    }

    #[inline]
    fn panel_offset(index: u32) -> wgpu::BufferAddress {
        (index as wgpu::BufferAddress) * Self::PANEL_STRIDE
    }

    pub fn panel_keys_for_type(&self, type_id: TypeId) -> Vec<PanelKey> {
        let mut keys: HashSet<PanelKey> = self
            .panel_cache
            .iter()
            .filter(|(_, desc)| desc.type_id == type_id)
            .map(|(key, _)| key.clone())
            .collect();
        for key in registered_panel_keys(type_id) {
            keys.insert(key);
        }
        keys.into_iter().collect()
    }

    pub fn write_panel(&self, queue: &Queue, index: u32, panel: &Panel) {
        let offset = Self::panel_offset(index);
        queue.write_buffer(&self.buffers.instance, offset, bytes_of(panel));
        self.compute.borrow_mut().mark_interaction_dirty();
    }

    pub fn write_panels(&self, queue: &Queue, start_index: u32, panels: &[Panel]) {
        if panels.is_empty() {
            return;
        }
        let offset = Self::panel_offset(start_index);
        queue.write_buffer(&self.buffers.instance, offset, cast_slice(panels));
        self.compute.borrow_mut().mark_interaction_dirty();
    }

    pub fn refresh_registered_payloads(&mut self, device: &Device, queue: &Queue) {
        let registry = payload_registry().lock().unwrap();
        for (&type_id, refresh_fn) in registry.iter() {
            let keys = self.panel_keys_for_type(type_id);
            if keys.is_empty() {
                continue;
            }
            refresh_fn(self, &keys, device, queue);
        }
    }

    pub fn ensure_capacity(&mut self, cfg: &BufferArenaConfig) {
        self.buffers.ensure_capacity(cfg);
    }

    pub fn prepare_texture_bindings(&self, device: &Device) -> Option<TextureBindings> {
        let data = self.texture_gpu_data();
        if !data.is_ready() {
            return None;
        }

        let view_count = data.views.len() as u32;
        let sampler_count = data.samplers.len() as u32;
        let texture_count = data.infos.len() as u32;

        let info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ui::texture-infos"),
            contents: bytemuck::cast_slice(&data.infos),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui::texture-bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: NonZeroU32::new(view_count),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: NonZeroU32::new(sampler_count),
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
            ],
        });

        let texture_refs: Vec<&wgpu::TextureView> = data.views.iter().collect();
        let sampler_refs: Vec<&wgpu::Sampler> = data.samplers.iter().collect();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui::texture-bindings"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&texture_refs),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::SamplerArray(&sampler_refs),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: info_buffer.as_entire_binding(),
                },
            ],
        });

        Some(TextureBindings {
            layout,
            bind_group,
            info_buffer,
            atlas_ids: data.atlas_ids,
            texture_count,
        })
    }

    pub fn rebuild_texture_bindings(
        &mut self,
        device: &Device,
        queue: &Queue,
    ) -> Option<&TextureBindings> {
        self.upload_textures_to_gpu(device, queue);
        let bindings = self.prepare_texture_bindings(device);
        self.texture_bindings = bindings;
        self.texture_bindings.as_ref()
    }

    pub fn refresh_panel_cache<TPayload: PanelPayload>(
        &mut self,
        keys: &[PanelKey],
        _device: &Device,
        queue: &Queue,
    ) -> Result<(), DbError> {
        let table = self.global_db.bind_table::<PanelBinding<TPayload>>()?;
        for key in keys {
            match table.get(key)? {
                Some(mut record) => {
                    let pending = std::mem::take(&mut record.pending_animations);
                    let mut descriptor =
                        self.build_cpu_descriptor::<TPayload>(key.clone(), record.clone());
                    if let Some(existing) = self.panel_cache.get(key) {
                        descriptor.display_state = existing.display_state;
                    }
                    let needs_update = match self.panel_cache.get(key) {
                        Some(existing) => existing != &descriptor,
                        None => true,
                    };
                    if needs_update {
                        self.panel_cache.insert(key.clone(), descriptor.clone());
                        self.panel_instances_dirty = true;
                    }

                    if !pending.is_empty() {
                        self.schedule_panel_animations(queue, &descriptor, pending);
                    }
                    let mut entry = table.upsert_entry(key.clone(), record)?;
                    entry.value_mut().pending_animations.clear();
                    entry.commit()?;
                }
                None => {
                    if self.panel_cache.remove(key).is_some() {
                        self.panel_instances_dirty = true;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn upload_panel_instances(&mut self, device: &Device, queue: &Queue) {
        if !self.panel_instances_dirty {
            return;
        }
        if self.texture_bindings.is_none() {
            let _ = self.rebuild_texture_bindings(device, queue);
        }

        let mut descriptors: Vec<&PanelCpuDescriptor> = self.panel_cache.values().collect();
        descriptors.sort_by_key(|desc| desc.key.panel_id);

        let previous_instances: HashMap<u32, Panel> = self
            .panel_instances
            .iter()
            .copied()
            .map(|panel| (panel.id, panel))
            .collect();

        let mut panels = Vec::with_capacity(descriptors.len());

        const BATCH_ORDER: [QuadBatchKind; 3] = [
            QuadBatchKind::Normal,
            QuadBatchKind::MultiVertex,
            QuadBatchKind::UltraVertex,
        ];

        for kind in BATCH_ORDER {
            let start = panels.len() as u32;
            for desc in descriptors.iter().filter(|desc| desc.quad_vertex == kind) {
                let fallback = previous_instances.get(&desc.key.panel_id);
                let panel = self.descriptor_to_panel_with_base(desc, fallback);
                panels.push(panel);
            }
            let end = panels.len() as u32;
            self.render.set_instance_range(kind, start..end);
            self.render.set_indirect_count(kind, 0);
        }

        self.panel_instances = panels;

        if !self.panel_instances.is_empty() {
            self.write_panels(queue, 0, &self.panel_instances);
        }

        self.panel_instances_dirty = false;
    }

    pub fn event_poll(&mut self, _device: &Device, queue: &Queue) {
        let events = self.event_hub().poll();
        if events.is_empty() {
            self.process_shader_results(queue);
            return;
        }
        let registry = self.register_panel_events();
        let mut guard = registry.lock().unwrap();
        for event in events {
            if let CpuPanelEvent::StateTransition(transition) = &event {
                println!("transition {:?}", transition);
                self.apply_state_transition(transition.clone(), queue);
            }
            guard.emit(&event);
        }
        drop(guard);
        self.process_shader_results(queue);
    }

    fn apply_state_transition(&mut self, transition: StateTransition, queue: &Queue) {
        let panel_id = transition.panel_id;
        let panel_key = self
            .panel_cache
            .keys()
            .find(|key| key.panel_id == panel_id)
            .cloned();

        let Some(key) = panel_key else {
            eprintln!("state transition for unknown panel {}", panel_id);
            return;
        };

        let target_descriptor = {
            let Some(desc) = self.panel_cache.get_mut(&key) else {
                return;
            };
            if desc.display_state == transition.new_state {
                return;
            }
            desc.current_state = transition.new_state;
            let mut clone = desc.clone();
            clone.display_state = clone.current_state;
            clone
        };
        let (instance_index, previous_panel) = match self
            .panel_instances
            .iter()
            .enumerate()
            .find(|(_, panel)| panel.id == panel_id)
        {
            Some((idx, panel)) => (idx, *panel),
            None => {
                if let Some(desc) = self.panel_cache.get_mut(&key) {
                    desc.display_state = desc.current_state;
                }
                return;
            }
        };

        let target_panel =
            self.descriptor_to_panel_with_base(&target_descriptor, Some(&previous_panel));

        if let Some(duration) = self.enqueue_style_transition(queue, &previous_panel, &target_panel)
        {
            self.transitioning_panels.insert(panel_id, duration);
        } else if let Some(desc) = self.panel_cache.get_mut(&key) {
            desc.display_state = desc.current_state;
        }

        self.update_panel_state_field(instance_index as u32, transition.new_state.0, queue);
        if let Some(instance) = self.panel_instances.get_mut(instance_index) {
            instance.state = transition.new_state.0;
        }
    }

    fn enqueue_style_transition(
        &mut self,
        queue: &Queue,
        current: &Panel,
        target: &Panel,
    ) -> Option<f32> {
        const DURATION: f32 = 0.25;
        let mut enqueued = false;

        if current.position != target.position {
            let info = TransformAnimFieldInfo {
                field_id: (PanelField::POSITION_X | PanelField::POSITION_Y).bits(),
                start_value: current.position.to_vec(),
                target_value: target.position.to_vec(),
                duration: DURATION,
                easing: EasingMask::IN_OUT_QUAD,
                op: AnimOp::SET,
                hold: 1,
                delay: 0.0,
                loop_count: 0,
                ping_pong: 0,
                on_complete: 0,
            };
            self.enqueue_animation(queue, target.id, info);
            enqueued = true;
        }

        if current.size != target.size {
            let info = TransformAnimFieldInfo {
                field_id: (PanelField::SIZE_X | PanelField::SIZE_Y).bits(),
                start_value: current.size.to_vec(),
                target_value: target.size.to_vec(),
                duration: DURATION,
                easing: EasingMask::IN_OUT_QUAD,
                op: AnimOp::SET,
                hold: 1,
                delay: 0.0,
                loop_count: 0,
                ping_pong: 0,
                on_complete: 0,
            };
            self.enqueue_animation(queue, target.id, info);
            enqueued = true;
        }

        if (current.transparent - target.transparent).abs() > f32::EPSILON {
            let info = TransformAnimFieldInfo {
                field_id: PanelField::TRANSPARENT.bits(),
                start_value: vec![current.transparent],
                target_value: vec![target.transparent],
                duration: DURATION,
                easing: EasingMask::IN_OUT_QUAD,
                op: AnimOp::SET,
                hold: 1,
                delay: 0.0,
                loop_count: 0,
                ping_pong: 0,
                on_complete: 0,
            };
            self.enqueue_animation(queue, target.id, info);
            enqueued = true;
        }

        if current.color != target.color {
            let info = TransformAnimFieldInfo {
                field_id: (PanelField::COLOR_R
                    | PanelField::COLOR_G
                    | PanelField::COLOR_B
                    | PanelField::COLOR_A)
                    .bits(),
                start_value: current.color.to_vec(),
                target_value: target.color.to_vec(),
                duration: DURATION,
                easing: EasingMask::IN_OUT_QUAD,
                op: AnimOp::SET,
                hold: 1,
                delay: 0.0,
                loop_count: 0,
                ping_pong: 0,
                on_complete: 0,
            };
            self.enqueue_animation(queue, target.id, info);
            enqueued = true;
        }

        if enqueued { Some(DURATION) } else { None }
    }

    fn update_animation_time(&mut self, delta_time: f32, queue: &Queue) {
        if delta_time <= 0.0 {
            return;
        }
        if self.animation_descriptor.animation_count == 0 {
            return;
        }
        self.animation_descriptor.delta_time = delta_time;
        self.animation_descriptor.total_time += delta_time;
        self.animation_descriptor
            .write_to_buffer(queue, &self.buffers.animation_descriptor);
        self.compute.borrow_mut().mark_animation_dirty();
    }

    fn advance_panel_transitions(&mut self, delta_time: f32) {
        if self.transitioning_panels.is_empty() {
            return;
        }

        let mut finished = Vec::new();
        for (panel_id, remaining) in self.transitioning_panels.iter_mut() {
            if *remaining <= delta_time {
                finished.push(*panel_id);
            } else {
                *remaining -= delta_time;
            }
        }

        for panel_id in finished {
            self.transitioning_panels.remove(&panel_id);
            if let Some(desc) = self
                .panel_cache
                .values_mut()
                .find(|desc| desc.key.panel_id == panel_id)
            {
                desc.display_state = desc.current_state;
                self.panel_instances_dirty = true;
            }
        }
    }

    fn update_panel_state_field(&self, instance_index: u32, new_state: u32, queue: &Queue) {
        let panel_offset = Self::panel_offset(instance_index);
        let field_offset = offset_of!(Panel, state) as wgpu::BufferAddress;
        dbg!(
            "写入状态 {} => {} :{}",
            panel_offset,
            field_offset,
            new_state
        );
        queue.write_buffer(
            &self.buffers.instance,
            panel_offset + field_offset,
            bytemuck::bytes_of(&new_state),
        );
    }

    fn process_shader_results(&mut self, queue: &Queue) {
        while let Ok(event) = self.shader_events.try_recv() {
            let event = event.into_arc();
            self.apply_shader_result(&event, queue);
        }
    }

    fn apply_shader_result(&mut self, event: &KennelResultIdxEvent, queue: &Queue) {
        let Some((panel_key, state, stage)) = take_pending_shader(event.idx) else {
            return;
        };

        let display_state_matches = {
            let Some(descriptor) = self.panel_cache.get_mut(&panel_key) else {
                eprintln!(
                    "shader result for unknown panel {:?} (idx {})",
                    panel_key, event.idx
                );
                return;
            };

            let Some(state_cpu) = descriptor.states.get_mut(&state) else {
                eprintln!(
                    "shader result for panel {:?} missing state {:?}",
                    panel_key, state
                );
                return;
            };

            match stage {
                ShaderStage::Fragment => {
                    state_cpu.overrides.fragment_shader_id = Some(event.kennel_id)
                }
                ShaderStage::Vertex => state_cpu.overrides.vertex_shader_id = Some(event.kennel_id),
            }

            descriptor.display_state == state
        };

        if !display_state_matches {
            return;
        }

        let Some(index) = self
            .panel_instances
            .iter()
            .position(|panel| panel.id == panel_key.panel_id)
        else {
            self.panel_instances_dirty = true;
            return;
        };

        {
            let panel = &mut self.panel_instances[index];
            match stage {
                ShaderStage::Fragment => panel.fragment_shader_id = event.kennel_id,
                ShaderStage::Vertex => panel.vertex_shader_id = event.kennel_id,
            }
        }

        let panel = &self.panel_instances[index];
        self.write_panel(queue, index as u32, panel);
    }

    pub fn panel_instances(&self) -> &[Panel] {
        &self.panel_instances
    }

    pub fn ensure_render_pipeline(
        &mut self,
        device: &Device,
        queue: &Queue,
        surface_format: wgpu::TextureFormat,
    ) {
        if self.texture_bindings.is_none() {
            let _ = self.rebuild_texture_bindings(device, queue);
        }

        if self.render_bind_group_layout.is_none() {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ui::render-bind-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            self.render_bind_group_layout = Some(layout);
        }

        let mut trace = self.trace.borrow_mut();
        trace.create_buffer(device);

        if self.render_bind_group.is_none() {
            let layout = self
                .render_bind_group_layout
                .as_ref()
                .expect("render bind layout available");
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ui::render-bind-group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.global_uniform.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: trace
                            .buffer
                            .as_ref()
                            .expect("没有绑定trace-buffer")
                            .as_entire_binding(),
                    },
                ],
            });
            self.render_bind_group = Some(bind_group);
        }

        let Some(texture_bindings) = self.texture_bindings.as_ref() else {
            return;
        };
        let Some(kennel_layout) = self.kennel_bind_group_layout.as_ref() else {
            return;
        };

        let needs_pipeline = self
            .render
            .batches
            .get(QuadBatchKind::Normal)
            .pipeline
            .is_none()
            || self.render_surface_format != Some(surface_format);

        if needs_pipeline {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ui::basic-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("ui_basic.wgsl").into()),
            });

            let render_layout = self
                .render_bind_group_layout
                .as_ref()
                .expect("render bind layout available");
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ui::pipeline-layout"),
                bind_group_layouts: &[render_layout, &texture_bindings.layout, kennel_layout],
                push_constant_ranges: &[],
            });

            let vertex_layouts = [
                wgpu::VertexBufferLayout {
                    array_stride: 16,
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
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Panel>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 32,
                            shader_location: 6,
                            format: wgpu::VertexFormat::Uint32,
                        },
                        wgpu::VertexAttribute {
                            offset: 36,
                            shader_location: 7,
                            format: wgpu::VertexFormat::Uint32,
                        },
                        wgpu::VertexAttribute {
                            offset: 40,
                            shader_location: 8,
                            format: wgpu::VertexFormat::Uint32,
                        },
                        wgpu::VertexAttribute {
                            offset: 44,
                            shader_location: 9,
                            format: wgpu::VertexFormat::Uint32,
                        },
                        wgpu::VertexAttribute {
                            offset: 48,
                            shader_location: 10,
                            format: wgpu::VertexFormat::Uint32,
                        },
                        wgpu::VertexAttribute {
                            offset: 52,
                            shader_location: 11,
                            format: wgpu::VertexFormat::Uint32,
                        },
                        wgpu::VertexAttribute {
                            offset: 56,
                            shader_location: 12,
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: 60,
                            shader_location: 13,
                            format: wgpu::VertexFormat::Uint32,
                        },
                        wgpu::VertexAttribute {
                            offset: 64,
                            shader_location: 14,
                            format: wgpu::VertexFormat::Uint32,
                        },
                        wgpu::VertexAttribute {
                            offset: 68,
                            shader_location: 15,
                            format: wgpu::VertexFormat::Uint32,
                        },
                        wgpu::VertexAttribute {
                            offset: 72,
                            shader_location: 16,
                            format: wgpu::VertexFormat::Uint32,
                        },
                        wgpu::VertexAttribute {
                            offset: 76,
                            shader_location: 17,
                            format: wgpu::VertexFormat::Uint32,
                        },
                        wgpu::VertexAttribute {
                            offset: 80,
                            shader_location: 18,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 96,
                            shader_location: 19,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 112,
                            shader_location: 20,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                },
            ];

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ui::render-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

            self.render.set_pipeline(QuadBatchKind::Normal, pipeline);
            self.render_surface_format = Some(surface_format);
        }
    }

    pub fn encode_panels<'pass>(&self, pass: &mut wgpu::RenderPass<'pass>) {
        let Some(render_bind_group) = self.render_bind_group.as_ref() else {
            return;
        };
        let Some(texture_bindings) = self.texture_bindings.as_ref() else {
            return;
        };
        let Some(kennel_bind_group) = self.kennel_bind_group.as_ref() else {
            return;
        };

        let instance_count = self.panel_instances.len() as u32;
        if instance_count == 0 {
            return;
        }

        self.render.encode_main(
            pass,
            render_bind_group,
            &texture_bindings.bind_group,
            Some(kennel_bind_group),
            &self.buffers.instance,
            &self.buffers.indirect_draws,
            0,
            6,
            instance_count,
        );
    }

    fn build_cpu_descriptor<TPayload: PanelPayload>(
        &mut self,
        key: PanelKey,
        record: PanelRecord<TPayload>,
    ) -> PanelCpuDescriptor {
        let PanelRecord {
            default_state,
            states,
            current_state,
            snapshot,
            ..
        } = record;

        let snapshot_texture = snapshot
            .texture
            .as_ref()
            .and_then(|name| self.ensure_texture_loaded(name));

        let mut state_map = HashMap::with_capacity(states.len());
        for (state_id, overrides) in states.into_iter() {
            let texture = overrides
                .texture
                .as_ref()
                .and_then(|name| self.ensure_texture_loaded(name));
            state_map.insert(state_id, PanelStateCpu { overrides, texture });
        }

        let quad_vertex = snapshot.quad_vertex;

        PanelCpuDescriptor {
            key,
            default_state,
            current_state,
            display_state: current_state,
            snapshot,
            snapshot_texture,
            states: state_map,
            type_id: TypeId::of::<TPayload>(),
            quad_vertex,
        }
    }

    fn descriptor_to_panel(&self, desc: &PanelCpuDescriptor) -> Panel {
        self.descriptor_to_panel_with_base(desc, None)
    }

    fn descriptor_to_panel_with_base(
        &self,
        desc: &PanelCpuDescriptor,
        fallback: Option<&Panel>,
    ) -> Panel {
        let state_cpu = desc
            .states
            .get(&desc.display_state)
            .or_else(|| desc.states.get(&desc.current_state))
            .or_else(|| desc.default_state.and_then(|id| desc.states.get(&id)))
            .or_else(|| desc.states.values().next());

        let overrides = state_cpu.map(|state| &state.overrides);
        let texture_info = state_cpu
            .and_then(|state| state.texture.as_ref())
            .or(desc.snapshot_texture.as_ref());

        if let Some(base) = fallback {
            return self.apply_overrides_with_base(desc, overrides, texture_info, base);
        }

        let mut position = overrides
            .and_then(|o| o.position)
            .unwrap_or(desc.snapshot.position);
        if let Some(offset) = overrides.and_then(|o| o.offset) {
            position[0] += offset[0];
            position[1] += offset[1];
        }

        let mut size = overrides.and_then(|o| o.size);
        if size.is_none() {
            if overrides.and_then(|o| o.fit_to_texture).unwrap_or(false) {
                if let Some(info) = texture_info {
                    if let Some(raw) = self.texture_atlas_store.raw_image_info(&info.path) {
                        size = Some([raw.width as f32, raw.height as f32]);
                    }
                }
            }
        }
        let size = size.unwrap_or([100.0, 100.0]);

        let z_value = overrides
            .and_then(|o| o.z_index)
            .unwrap_or(desc.snapshot.z_index)
            .max(0) as u32;

        let pass_through = overrides.and_then(|o| o.pass_through).unwrap_or(0);
        let interaction = overrides.and_then(|o| o.interaction).unwrap_or(0);
        let event_mask = overrides.and_then(|o| o.event_mask).unwrap_or(0);
        let state_mask = overrides.and_then(|o| o.state_mask).unwrap_or(0);
        let vertex_shader_id = overrides
            .and_then(|o| o.vertex_shader_id)
            .unwrap_or(u32::MAX);
        let fragment_shader_id = overrides
            .and_then(|o| o.fragment_shader_id)
            .unwrap_or(u32::MAX);
        let collection_state = overrides.and_then(|o| o.collection_state).unwrap_or(0);

        let transparent = overrides
            .and_then(|o| o.transparent)
            .or_else(|| overrides.and_then(|o| o.color).map(|color| color[3]))
            .unwrap_or(desc.snapshot.color[3]);

        let (uv_offset, uv_scale, texture_id) = match texture_info {
            Some(info) => {
                let slot = self
                    .texture_slot_for_atlas(info.parent_index)
                    .unwrap_or(u32::MAX);
                (
                    [info.uv_min[0], info.uv_min[1]],
                    [
                        info.uv_max[0] - info.uv_min[0],
                        info.uv_max[1] - info.uv_min[1],
                    ],
                    slot,
                )
            }
            None => ([0.0, 0.0], [1.0, 1.0], u32::MAX),
        };

        let mut panel = Panel::default();
        panel.id = desc.key.panel_id;
        panel.position = position;
        panel.size = size;
        panel.uv_offset = uv_offset;
        panel.uv_scale = uv_scale;
        panel.z_index = z_value;
        panel.pass_through = pass_through;
        panel.interaction = interaction;
        panel.event_mask = event_mask;
        panel.state = desc.current_state.0;
        panel.state_mask = state_mask;
        panel.transparent = transparent;
        panel.texture_id = texture_id;
        panel.collection_state = collection_state;
        panel.vertex_shader_id = vertex_shader_id;
        panel.fragment_shader_id = fragment_shader_id;

        let color = overrides
            .and_then(|o| o.color)
            .unwrap_or(desc.snapshot.color);
        panel.color = color;

        if let Some(border) = overrides.and_then(|o| o.border.clone()) {
            panel.border_color = border.color;
            panel.border_width = border.width;
            panel.border_radius = border.radius;
        } else {
            panel.border_color = [0.0, 0.0, 0.0, 0.0];
            panel.border_width = 0.0;
            panel.border_radius = 0.0;
        }
        panel.pad_border = [0.0, 0.0];
        panel
    }

    fn apply_overrides_with_base(
        &self,
        desc: &PanelCpuDescriptor,
        overrides: Option<&PanelStateOverrides>,
        texture_info: Option<&UiTextureInfo>,
        base: &Panel,
    ) -> Panel {
        let mut panel = *base;
        panel.id = desc.key.panel_id;
        panel.state = desc.current_state.0;

        if let Some(position) = overrides.and_then(|o| o.position) {
            panel.position = position;
        }
        if let Some(offset) = overrides.and_then(|o| o.offset) {
            panel.position[0] += offset[0];
            panel.position[1] += offset[1];
        }

        if let Some(size) = overrides.and_then(|o| o.size) {
            panel.size = size;
        } else if overrides.and_then(|o| o.fit_to_texture).unwrap_or(false) {
            if let Some(info) = texture_info {
                panel.size = [
                    info.uv_max[0] - info.uv_min[0],
                    info.uv_max[1] - info.uv_min[1],
                ];
            }
        }

        if let Some(z) = overrides.and_then(|o| o.z_index) {
            panel.z_index = z.max(0) as u32;
        }
        if let Some(pass) = overrides.and_then(|o| o.pass_through) {
            panel.pass_through = pass;
        }
        if let Some(interaction) = overrides.and_then(|o| o.interaction) {
            panel.interaction = interaction;
        }
        if let Some(event_mask) = overrides.and_then(|o| o.event_mask) {
            panel.event_mask = event_mask;
        }
        if let Some(state_mask) = overrides.and_then(|o| o.state_mask) {
            panel.state_mask = state_mask;
        }
        if let Some(collection_state) = overrides.and_then(|o| o.collection_state) {
            panel.collection_state = collection_state;
        }

        if let Some(color) = overrides.and_then(|o| o.color) {
            panel.color = color;
        }
        if let Some(border) = overrides.and_then(|o| o.border.clone()) {
            panel.border_color = border.color;
            panel.border_width = border.width;
            panel.border_radius = border.radius;
        }

        if let Some(trans) = overrides
            .and_then(|o| o.transparent)
            .or_else(|| overrides.and_then(|o| o.color).map(|color| color[3]))
        {
            panel.transparent = trans;
        }

        if let Some(info) = texture_info {
            let slot = self
                .texture_slot_for_atlas(info.parent_index)
                .unwrap_or(u32::MAX);
            panel.texture_id = slot;
            panel.uv_offset = [info.uv_min[0], info.uv_min[1]];
            panel.uv_scale = [
                info.uv_max[0] - info.uv_min[0],
                info.uv_max[1] - info.uv_min[1],
            ];
        }

        if let Some(vertex) = overrides.and_then(|o| o.vertex_shader_id) {
            panel.vertex_shader_id = vertex;
        }
        if let Some(frag) = overrides.and_then(|o| o.fragment_shader_id) {
            panel.fragment_shader_id = frag;
        }

        panel
    }

    fn schedule_panel_animations(
        &mut self,
        queue: &Queue,
        descriptor: &PanelCpuDescriptor,
        specs: Vec<AnimationSpec>,
    ) {
        if specs.is_empty() {
            return;
        }
        let panel = self.descriptor_to_panel(descriptor);
        for spec in specs {
            if let Some(info) = animation_spec_to_transform(&panel, &spec) {
                self.enqueue_animation(queue, panel.id, info);
            } else {
                eprintln!(
                    "unsupported animation spec {:?} for panel {:?}",
                    spec.property, panel.id
                );
            }
        }
    }

    fn enqueue_animation(&mut self, queue: &Queue, panel_id: u32, info: TransformAnimFieldInfo) {
        let entries = info.split_write_field(panel_id);
        if entries.is_empty() {
            return;
        }

        let entry_size = std::mem::size_of::<AnimtionFieldOffsetPtr>() as wgpu::BufferAddress;
        let death_offset = offset_of!(AnimtionFieldOffsetPtr, death) as wgpu::BufferAddress;

        for entry in entries {
            self.kill_conflicting_fields(queue, panel_id, entry.field_id, death_offset, entry_size);

            let index = self.animation_descriptor.animation_count;
            if index >= self.buffers.animation_fields_capacity {
                eprintln!(
                    "animation buffer exhausted (capacity {}), skipping animation for panel {}",
                    self.buffers.animation_fields_capacity, panel_id
                );
                return;
            }

            let offset = index as wgpu::BufferAddress * entry_size;
            queue.write_buffer(
                &self.buffers.animation_fields,
                offset,
                bytemuck::cast_slice(std::slice::from_ref(&entry)),
            );

            self.animation_field_cache
                .insert((index, panel_id), entry.field_id);
            self.animation_descriptor.animation_count += 1;
        }

        self.animation_descriptor.delta_time = self.state.frame_state.delta_time;
        self.animation_descriptor.total_time += self.state.frame_state.delta_time;
        self.animation_descriptor
            .write_to_buffer(queue, &self.buffers.animation_descriptor);
        let mut compute = self.compute.borrow_mut();
        compute.update_animation_count(self.animation_descriptor.animation_count);
        compute.mark_animation_dirty();
    }

    fn kill_conflicting_fields(
        &mut self,
        queue: &Queue,
        panel_id: u32,
        field_mask: u32,
        death_offset: wgpu::BufferAddress,
        entry_size: wgpu::BufferAddress,
    ) {
        let mut stale = Vec::new();
        for (&(anim_idx, panel_idx), &mask) in self.animation_field_cache.iter() {
            if panel_idx == panel_id && (mask & field_mask) != 0 {
                stale.push((anim_idx, panel_idx));
            }
        }
        for (anim_idx, key) in stale {
            let offset = anim_idx as wgpu::BufferAddress * entry_size + death_offset;
            let death: u32 = 1;
            queue.write_buffer(
                &self.buffers.animation_fields,
                offset,
                bytemuck::bytes_of(&death),
            );
            self.animation_field_cache.remove(&(anim_idx, key));
        }
    }

    fn ensure_texture_loaded(&mut self, texture_name: &str) -> Option<UiTextureInfo> {
        if texture_name.trim().is_empty() {
            return None;
        }

        if let Some(info) = self.texture_atlas_store.texture_info(texture_name) {
            return Some(info.clone());
        }

        let mut candidates: Vec<PathBuf> = Vec::new();
        candidates.push(PathBuf::from(texture_name));
        candidates.push(PathBuf::from("texture").join(texture_name));

        for path in candidates {
            if path.exists() {
                if let Some(info) = self.texture_atlas_store.read_img(&path) {
                    return Some(info);
                } else {
                    eprintln!(
                        "failed to load texture '{:?}' into atlas (read_img returned None)",
                        path
                    );
                    return None;
                }
            }
        }

        eprintln!("texture '{texture_name}' not found on disk");
        None
    }

    fn texture_slot_for_atlas(&self, atlas_index: u32) -> Option<u32> {
        self.texture_bindings.as_ref().and_then(|bindings| {
            bindings
                .atlas_ids
                .iter()
                .position(|&id| id == atlas_index)
                .map(|idx| idx as u32)
        })
    }

    /// Expose the global database to callers that need to read/write persistent UI data.
    pub fn db(&self) -> &'static MileDb {
        self.global_db
    }

    /// Expose the shared event bus so higher level systems can publish events without
    /// creating additional handles.
    pub fn event_bus(&self) -> &'static EventBus {
        self.global_event_bus
    }

    /// Record frame timing information and clear transient CPU-side queues.
    pub fn begin_frame(&mut self, frame_index: u32, delta_time: f32) {
        self.frame_history.advance(frame_index);
        self.state.frame_state.frame_index = frame_index;
        self.state.frame_state.delta_time = delta_time;
        self.state.clear_frame();
        self.compute.borrow_mut().mark_interaction_dirty();
        self.advance_panel_transitions(delta_time);
    }

    /// Submit a CPU event to both the per-frame queue and the global event bus.
    pub fn push_cpu_event(&mut self, event: CpuPanelEvent) {
        self.state.push_event(event.clone());
        self.global_event_bus.publish(event);
    }

    /// Register a callback for a specific interaction scope.
    pub fn register_panel_events(&self) -> Arc<Mutex<PanelEventRegistry>> {
        Arc::clone(&self.state.panel_events)
    }

    #[inline]
    pub fn event_hub(&self) -> &Arc<UIEventHub> {
        &self.state.event_hub
    }

    #[inline]
    pub fn cpu_events(&self) -> &[CpuPanelEvent] {
        &self.state.cpu_events
    }

    pub fn mouse_press_tick_post(&mut self, queue: &wgpu::Queue) {
        let buffer = self.global_uniform.buffer.clone();
        let mut unitfrom_struct = self.global_uniform.cpu_mut();

        let pressed = (unitfrom_struct.mouse_state
            & (MouseState::LEFT_DOWN.bits() | MouseState::RIGHT_DOWN.bits()))
            != 0;
        if !pressed {
            // 弹起，重置按下时间
            unitfrom_struct.press_duration = 0.0;
            unitfrom_struct.mouse_state = MouseState::DEFAULT.bits();
        }
        let offset = offset_of!(GlobalUniform, press_duration) as wgpu::BufferAddress;
        // 写入 GPU buffer
        queue.write_buffer(
            &buffer,
            offset,
            bytemuck::bytes_of(&unitfrom_struct.press_duration),
        );
        queue.write_buffer(
            &buffer,
            offset_of!(GlobalUniform, mouse_state) as u64,
            bytemuck::bytes_of(&unitfrom_struct.mouse_state),
        );
    }

    pub fn mouse_press_tick_first(&mut self, queue: &wgpu::Queue) {
        let buffer = self.global_uniform.buffer().clone();
        let mut unitfrom_struct = self.global_uniform.cpu_mut();

        let pressed = (unitfrom_struct.mouse_state
            & (MouseState::LEFT_DOWN.bits() | MouseState::RIGHT_DOWN.bits()))
            != 0;

        if pressed {
            // 持续按下累加时间
            unitfrom_struct.press_duration += 0.033;
        }

        let offset = offset_of!(GlobalUniform, press_duration) as wgpu::BufferAddress;
        // 写入 GPU buffer
        queue.write_buffer(
            &buffer,
            offset,
            bytemuck::bytes_of(&unitfrom_struct.press_duration),
        );
    }

    pub fn copy_interaction_swap_frame(&mut self, device: &Device, queue: &Queue) {
        let frame_size = std::mem::size_of::<GpuInteractionFrame>() as wgpu::BufferAddress;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ui::interaction-frame-swap"),
        });

        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::interaction-temp"),
            size: frame_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let new_curr_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("ui::interaction-curr"),
            contents: cast_slice(&[GpuInteractionFrame::empty(
                self.state.frame_state.frame_index,
            )]),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        encoder.copy_buffer_to_buffer(
            &self.buffers.interaction_frames,
            frame_size,
            &temp_buffer,
            0,
            frame_size,
        );
        encoder.copy_buffer_to_buffer(
            &temp_buffer,
            0,
            &self.buffers.interaction_frames,
            0,
            frame_size,
        );
        encoder.copy_buffer_to_buffer(
            &new_curr_buffer,
            0,
            &self.buffers.interaction_frames,
            frame_size,
            frame_size,
        );

        queue.submit(Some(encoder.finish()));
    }

    pub fn update_mouse_state(&mut self, queue: &wgpu::Queue, mouse_state: MouseState) {
        let global = self.global_uniform.cpu_mut();
        global.mouse_state = mouse_state.bits();
        let offset = offset_of!(GlobalUniform, mouse_state) as wgpu::BufferAddress;
        queue.write_buffer(
            &self.global_uniform.buffer,
            offset,
            bytemuck::bytes_of(&self.global_uniform.cpu.mouse_state),
        );
    }

    pub fn tick_frame_update_data(&mut self, queue: &Queue) {
        let frame_index = self.state.frame_state.frame_index;
        let delta_time = self.state.frame_state.delta_time;

        let uniform = &mut self.global_uniform;

        // Mirror the legacy `Mui` frame ticking: keep dt/time/frame alive on both CPU & GPU.
        uniform.cpu_mut().dt = delta_time;
        let dt = uniform.cpu().dt;
        uniform.write_field(
            queue,
            offset_of!(GlobalUniform, dt) as wgpu::BufferAddress,
            &dt,
        );

        uniform.cpu_mut().time += delta_time;
        let time = uniform.cpu().time;
        uniform.write_field(
            queue,
            offset_of!(GlobalUniform, time) as wgpu::BufferAddress,
            &time,
        );

        uniform.cpu_mut().frame = frame_index;
        let frame = uniform.cpu().frame;
        uniform.write_field(
            queue,
            offset_of!(GlobalUniform, frame) as wgpu::BufferAddress,
            &frame,
        );

        self.update_animation_time(delta_time, queue);
    }

    /// Queue a write into the panel instance buffer by offset.
    pub fn write_panel_bytes(
        &self,
        queue: &Queue,
        panel_offset_bytes: wgpu::BufferAddress,
        data: &[u8],
    ) {
        queue.write_buffer(&self.buffers.instance, panel_offset_bytes, data);
    }

    /// Reads back the GPU interaction buffer so the CPU can inspect the latest
    /// frame pair and enqueue panel events without waiting on future frames.
    pub fn interaction_cpu_trigger(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // DownloadBuffer::read_buffer(
        //     device,
        //     queue,
        //     &self
        //         .interaction_pipeline_cache
        //         .gpu_Interaction_buffer
        //         .as_ref()
        //         .unwrap()
        //         .slice(..),
        //     move |e| {
        //         if let Ok(downloadBuffer) = e {
        //             let bytes = downloadBuffer;

        //             // cast bytes -> &[MyStruct]
        //             let data: &[GpuInteractionFrame] = bytemuck::cast_slice(&bytes);

        //             let old_frame = data[0];
        //             let new_frame = data[1];

        //             // println!("cpu记录�?{:?}", frame);
        //             // println!("显卡记录�?{:?}", new_frame.frame);

        //             if new_frame.click_id != u32::MAX {
        //                 // hub.push(CpuPanelEvent::Click((
        //                 //     new_frame.frame,
        //                 //     UiInteractionScope {
        //                 //         panel_id: new_frame.click_id,
        //                 //         state: new_frame.trigger_panel_state,
        //                 //     },
        //                 // )));
        //             }
        //             if new_frame.drag_id != u32::MAX {
        //                 // hub.push(CpuPanelEvent::Drag((
        //                 //     new_frame.frame,
        //                 //     UiInteractionScope {
        //                 //         panel_id: new_frame.drag_id,
        //                 //         state: new_frame.trigger_panel_state,
        //                 //     },
        //                 // )));
        //             }

        //             if new_frame.hover_id != u32::MAX && old_frame.hover_id != new_frame.hover_id {
        //                 // hub.push(CpuPanelEvent::Hover((
        //                 //     new_frame.frame,
        //                 //     UiInteractionScope {
        //                 //         panel_id: new_frame.hover_id,
        //                 //         state: new_frame.trigger_panel_state,
        //                 //     },
        //                 // )));
        //             }

        //             if (new_frame.hover_id != old_frame.hover_id) {
        //                 println!(
        //                     "当前退出了out {:?} {:?}",
        //                     old_frame.hover_id, old_frame.trigger_panel_state
        //                 );
        //                 // hub.push(CpuPanelEvent::OUT((
        //                 //     new_frame.frame,
        //                 //     UiInteractionScope {
        //                 //         panel_id: old_frame.hover_id,
        //                 //         state: old_frame.trigger_panel_state,
        //                 //     },
        //                 // )));
        //             }

        //             // if new_frame.hover_id != u32::MAX {
        //             //     hub.push(CpuPanelEvent::Hover((new_frame.frame, new_frame.hover_id)));
        //             // }
        //         }
        //     },
        // );
    }

    /// Convenience wrapper updating the render batch bookkeeping.
    pub fn set_render_batch_range(&mut self, kind: QuadBatchKind, range: std::ops::Range<u32>) {
        self.render.set_instance_range(kind, range);
    }

    /// Update the indirect draw count for a render batch.
    pub fn set_render_batch_indirect_count(&mut self, kind: QuadBatchKind, count: u32) {
        self.render.set_indirect_count(kind, count);
    }
}

fn animation_spec_to_transform(
    panel: &Panel,
    spec: &AnimationSpec,
) -> Option<TransformAnimFieldInfo> {
    let (field_bits, value_len) = match spec.property {
        AnimProperty::Position => ((PanelField::POSITION_X | PanelField::POSITION_Y).bits(), 2),
        AnimProperty::Size => ((PanelField::SIZE_X | PanelField::SIZE_Y).bits(), 2),
        AnimProperty::Opacity => (PanelField::TRANSPARENT.bits(), 1),
        AnimProperty::Color => (
            (PanelField::COLOR_R | PanelField::COLOR_G | PanelField::COLOR_B | PanelField::COLOR_A)
                .bits(),
            4,
        ),
        _ => return None,
    };

    let target_values = match spec.to {
        AnimTargetValue::Scalar(v) if value_len == 1 => vec![v],
        AnimTargetValue::Vec2(v) if value_len == 2 => v.to_vec(),
        AnimTargetValue::Vec4(v) if value_len == 2 => v[..2].to_vec(),
        AnimTargetValue::Vec4(v) if value_len == 4 => v.to_vec(),
        _ => return None,
    };

    let start_values = if let Some(ref from_value) = spec.from {
        match (from_value, value_len) {
            (AnimTargetValue::Scalar(v), 1) => vec![*v],
            (AnimTargetValue::Vec2(v), 2) => v.to_vec(),
            (AnimTargetValue::Vec4(v), 2) => v[..2].to_vec(),
            (AnimTargetValue::Vec4(v), 4) => v.to_vec(),
            _ => return None,
        }
    } else if spec.from_current {
        panel_property_as_vec(panel, spec.property)?
    } else {
        panel_property_as_vec(panel, spec.property)?
    };

    Some(TransformAnimFieldInfo {
        field_id: field_bits,
        start_value: start_values,
        target_value: target_values,
        duration: spec.duration.max(0.0),
        easing: easing_to_mask(spec.easing),
        op: AnimOp::SET,
        hold: 1,
        delay: spec.delay.max(0.0),
        loop_count: spec.loop_config.count.unwrap_or(0),
        ping_pong: if spec.loop_config.ping_pong { 1 } else { 0 },
        on_complete: 0,
    })
}

fn panel_property_as_vec(panel: &Panel, property: AnimProperty) -> Option<Vec<f32>> {
    match property {
        AnimProperty::Position => Some(panel.position.to_vec()),
        AnimProperty::Size => Some(panel.size.to_vec()),
        AnimProperty::Opacity => Some(vec![panel.transparent]),
        AnimProperty::Color => Some(panel.color.to_vec()),
        _ => None,
    }
}

fn easing_to_mask(easing: Easing) -> EasingMask {
    match easing {
        Easing::Linear => EasingMask::LINEAR,
        Easing::QuadraticIn => EasingMask::IN_QUAD,
        Easing::QuadraticOut => EasingMask::OUT_QUAD,
        Easing::QuadraticInOut => EasingMask::IN_OUT_QUAD,
        Easing::CubicIn => EasingMask::IN_CUBIC,
        Easing::CubicOut => EasingMask::OUT_CUBIC,
        Easing::CubicInOut => EasingMask::IN_OUT_CUBIC,
        _ => EasingMask::LINEAR,
    }
}

type PayloadRefreshFn = fn(&mut MuiRuntime, &[PanelKey], &wgpu::Device, &wgpu::Queue);

fn payload_registry() -> &'static Mutex<HashMap<TypeId, PayloadRefreshFn>> {
    static REGISTRY: OnceLock<Mutex<HashMap<TypeId, PayloadRefreshFn>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register_payload_refresh<T: PanelPayload + 'static>() {
    let mut registry = payload_registry().lock().unwrap();
    registry
        .entry(TypeId::of::<T>())
        .or_insert(refresh_payload::<T>);
}

fn refresh_payload<T: PanelPayload>(
    runtime: &mut MuiRuntime,
    keys: &[PanelKey],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let _ = runtime.refresh_panel_cache::<T>(keys, device, queue);
}


