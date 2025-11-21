//! Compute pipeline orchestration.
//!
//! Encapsulates the GPU compute stages used by the UI runtime.  Right now only the
//! interaction stage is implemented; network and animation stages remain no-ops but
//! keep the scaffolding ready for future work.

use std::{cell::RefCell, sync::Arc};

use bytemuck::{bytes_of, cast_slice};
use glam::Vec2;
use wgpu::{util::DownloadBuffer, wgc::device::queue};

use crate::runtime::{
    _ty::{GpuInteractionFrame, GpuRelationDispatchArgs, GpuRelationWorkItem},
    buffers::{BufferArena, BufferViewSet},
    relations::relation_registry,
    state::{CpuPanelEvent, UIEventHub, UiInteractionScope},
};
use mile_api::{GpuDebugReadCallBack, interface::GpuDebug};

/// Per-frame context values shared with compute stages.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameComputeContext<'a> {
    pub queue: Option<&'a wgpu::Queue>,
    pub frame_index: u32,
    pub panel_count: u32,
}

/// Holder for the three compute stages used by the UI runtime.
pub struct ComputePipelines {
    pub interaction: InteractionComputeStage,
    pub relations: RelationComputeStage,
    pub panel_delta: PanelDeltaStage,
    pub animation: AnimationComputeStage,
}

impl ComputePipelines {
    pub fn new(
        device: &wgpu::Device,
        buffers: &BufferArena,
        global_uniform: &wgpu::Buffer,
        event_hub: Arc<UIEventHub>,
    ) -> Self {
        Self {
            interaction: InteractionComputeStage::new(device, buffers, global_uniform, event_hub),
            relations: RelationComputeStage::new(device, buffers),
            panel_delta: PanelDeltaStage::new(device, buffers, global_uniform),
            animation: AnimationComputeStage::new(device, buffers, global_uniform),
        }
    }

    pub fn encode_all(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        buffers: &BufferViewSet<'_>,
        ctx: &FrameComputeContext<'_>,
    ) {
        let mut delta_needed = true;
        if self.interaction.is_dirty() {
            self.interaction.encode(pass, buffers, ctx);
            delta_needed = true;
        }
        if self.animation.is_dirty() {
            self.animation.encode(pass, buffers, ctx);
            delta_needed = true;
        }
        if self.relations.encode(pass, buffers, ctx) {
            delta_needed = true;
        }

        if delta_needed {
            self.panel_delta.set_dirty();
        }

        if self.panel_delta.is_dirty() {
            self.panel_delta.encode(pass, buffers, ctx);
        }
    }

    pub fn readback_all(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ctx: &FrameComputeContext<'_>,
    ) {
        self.interaction.readback(device, queue, ctx);
        self.relations.readback(device, queue, ctx);
        self.animation.readback(device, queue, ctx);
        self.panel_delta.readback(device, queue);
    }

    #[inline]
    pub fn mark_interaction_dirty(&mut self) {
        self.interaction.set_dirty();
    }

    #[inline]
    pub fn ingest_relation_work(&mut self, queue: &wgpu::Queue) {
        self.relations.ingest(queue);
    }

    #[inline]
    pub fn mark_animation_dirty(&mut self) {
        self.animation.set_dirty();
    }

    #[inline]
    pub fn mark_all_dirty(&mut self) {
        self.interaction.set_dirty();
        self.animation.set_dirty();
    }

    #[inline]
    pub fn is_any_dirty(&self) -> bool {
        self.interaction.is_dirty()
            || self.animation.is_dirty()
            || self.panel_delta.is_dirty()
            || self.relations.has_work()
    }

    pub fn rebuild_interaction_bind_group(
        &mut self,
        device: &wgpu::Device,
        buffers: &BufferArena,
        global_uniform: &wgpu::Buffer,
    ) {
        self.interaction
            .rebuild_bind_group(device, buffers, global_uniform);
        self.interaction.set_dirty();
        self.panel_delta
            .rebuild_bind_group(device, buffers, &global_uniform);
    }

    pub fn rebuild_animation_bind_group(
        &mut self,
        device: &wgpu::Device,
        buffers: &BufferArena,
        global_uniform: &wgpu::Buffer,
    ) {
        self.animation
            .rebuild_bind_groups(device, buffers, global_uniform);
        self.animation.set_dirty();
        self.panel_delta
            .rebuild_bind_group(device, buffers, global_uniform);
    }

    pub fn update_animation_count(&mut self, count: u32) {
        self.animation.set_animation_count(count);
    }
}

/// Interaction compute pipeline.
pub struct InteractionComputeStage {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    layout1: wgpu::BindGroupLayout,
    bind_group1: wgpu::BindGroup,
    interaction_buffer: wgpu::Buffer,
    trace: RefCell<GpuDebug>,
    trace_buffer: wgpu::Buffer,
    event_hub: Arc<UIEventHub>,
    workgroup_size: u32,
    dirty: bool,
}

impl InteractionComputeStage {
    const WORKGROUP_SIZE: u32 = 64;

    pub fn new(
        device: &wgpu::Device,
        buffers: &BufferArena,
        global_uniform: &wgpu::Buffer,
        event_hub: Arc<UIEventHub>,
    ) -> Self {
        let mut trace = GpuDebug::new("ui::interaction-compute");
        trace.create_buffer(device);
        let trace_buffer = trace
            .buffer
            .as_ref()
            .expect("interaction trace buffer not created")
            .clone();

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui::interaction-bind-layout"),
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
                // Clamp descriptor (count)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Clamp rules
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        // clamp_rules in WGSL is declared as `var<storage, read>`
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Second bind group layout: snapshots buffer at group(1), binding(3)
        let layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui::interaction-bind-layout-1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui::interaction-pipeline-layout"),
            bind_group_layouts: &[&layout, &layout1],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui::interaction-compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../interaction_compute.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ui::interaction-compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group =
            Self::create_bind_group(device, &layout, buffers, global_uniform, &trace_buffer);
        let bind_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui::interaction-bind-group-1"),
            layout: &layout1,
            entries: &[wgpu::BindGroupEntry {
                binding: 3,
                resource: buffers.snapshot.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            layout,
            bind_group,
            layout1,
            bind_group1,
            interaction_buffer: buffers.interaction_frames.clone(),
            trace: RefCell::new(trace),
            trace_buffer,
            event_hub,
            workgroup_size: Self::WORKGROUP_SIZE,
            dirty: true,
        }
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffers: &BufferArena,
        global_uniform: &wgpu::Buffer,
        trace_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui::interaction-bind-group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.instance.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: global_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.interaction_frames.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: trace_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.panel_anim_delta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.clamp_desc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.clamp_rules.as_entire_binding(),
                },
            ],
        })
    }

    pub fn rebuild_bind_group(
        &mut self,
        device: &wgpu::Device,
        buffers: &BufferArena,
        global_uniform: &wgpu::Buffer,
    ) {
        {
            let mut trace = self.trace.borrow_mut();
            if trace.buffer.is_none() {
                trace.create_buffer(device);
            }
            self.trace_buffer = trace
                .buffer
                .as_ref()
                .expect("interaction trace buffer not created")
                .clone();
        }

        self.bind_group = Self::create_bind_group(
            device,
            &self.layout,
            buffers,
            global_uniform,
            &self.trace_buffer,
        );
        self.bind_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui::interaction-bind-group-1"),
            layout: &self.layout1,
            entries: &[wgpu::BindGroupEntry {
                binding: 3,
                resource: buffers.snapshot.as_entire_binding(),
            }],
        });
        self.interaction_buffer = buffers.interaction_frames.clone();
    }

    #[inline]
    pub fn set_dirty(&mut self) {
        self.dirty = true;
    }

    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    pub fn encode(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        _buffers: &BufferViewSet<'_>,
        ctx: &FrameComputeContext<'_>,
    ) {
        if ctx.panel_count == 0 {
            self.clear_dirty();
            return;
        }

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_bind_group(1, &self.bind_group1, &[]);

        let workgroup_size = self.workgroup_size.max(1);
        let workgroups = (ctx.panel_count + workgroup_size - 1) / workgroup_size;
        pass.dispatch_workgroups(workgroups.max(1), 1, 1);

        self.clear_dirty();
    }

    pub fn readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _ctx: &FrameComputeContext<'_>,
    ) {
        if self.interaction_buffer.size() == 0 {
            return;
        }

        let mut trace_buffer = self.trace.borrow_mut();
        trace_buffer.debug(device, queue);

        let hub = self.event_hub.clone();
        DownloadBuffer::read_buffer(
            device,
            queue,
            &self.interaction_buffer.slice(..),
            move |result| {
                let Ok(bytes) = result else {
                    return;
                };

                let frame_size = std::mem::size_of::<GpuInteractionFrame>();
                if bytes.len() < frame_size * 2 {
                    return;
                }

                let frames: &[GpuInteractionFrame] = cast_slice(&bytes);
                if frames.len() < 2 {
                    return;
                }

                // println!("当前的frame 信息 {:?}",frames);

                let old_frame = frames[0];
                let new_frame: GpuInteractionFrame = frames[1];

                // Source drag start/drop transitions
                let drag_started = old_frame.drag_id == u32::MAX && new_frame.drag_id != u32::MAX;
                let drag_ended = old_frame.drag_id != u32::MAX && new_frame.drag_id == u32::MAX;

                if new_frame.click_id != u32::MAX {
                    hub.push(CpuPanelEvent::Click((
                        new_frame.frame,
                        UiInteractionScope {
                            panel_id: new_frame.click_id,
                            state: new_frame.trigger_panel_state,
                        },
                    )));
                }

                if drag_started {
                    hub.push(CpuPanelEvent::SourceDragStart((
                        new_frame.frame,
                        UiInteractionScope {
                            panel_id: new_frame.drag_id,
                            state: new_frame.trigger_panel_state,
                        },
                    )));
                }

                if new_frame.drag_id != u32::MAX {
                    hub.push(CpuPanelEvent::Drag((
                        Vec2::from_array(new_frame.drag_delta),
                        UiInteractionScope {
                            panel_id: new_frame.drag_id,
                            state: new_frame.trigger_panel_state,
                        },
                    )));
                    if new_frame.hover_id != u32::MAX {
                        hub.push(CpuPanelEvent::TargetDragOver((
                            Vec2::from_array(new_frame.drag_delta),
                            UiInteractionScope {
                                panel_id: new_frame.hover_id,
                                state: new_frame.trigger_panel_state,
                            },
                        )));
                    }
                }

                if new_frame.hover_id != u32::MAX && new_frame.hover_id != old_frame.hover_id {
                    hub.push(CpuPanelEvent::Hover((
                        new_frame.frame,
                        UiInteractionScope {
                            panel_id: new_frame.hover_id,
                            state: new_frame.trigger_panel_state,
                        },
                    )));
                }

                // Target drag enter/leave during drag
                if new_frame.drag_id != u32::MAX && new_frame.hover_id != old_frame.hover_id {
                    if old_frame.hover_id != u32::MAX {
                        hub.push(CpuPanelEvent::TargetDragLeave((
                            new_frame.frame,
                            UiInteractionScope {
                                panel_id: old_frame.hover_id,
                                state: old_frame.trigger_panel_state,
                            },
                        )));
                    }
                    if new_frame.hover_id != u32::MAX {
                        println!("拖拽进入了某个面板");
                        
                        hub.push(CpuPanelEvent::TargetDragEnter((
                            new_frame.frame,
                            UiInteractionScope {
                                panel_id: new_frame.hover_id,
                                state: new_frame.trigger_panel_state,
                            },
                        )));
                    }
                }

                if drag_ended {
                    hub.push(CpuPanelEvent::SourceDragDrop((
                        new_frame.frame,
                        UiInteractionScope {
                            panel_id: old_frame.drag_id,
                            state: old_frame.trigger_panel_state,
                        },
                    )));

                    if old_frame.hover_id != u32::MAX {
                        println!("拖拽并且落到了某个面板上 hover_id:{} : trigger_panel_state:{}",old_frame.hover_id,old_frame.trigger_panel_state);
                        hub.push(CpuPanelEvent::TargetDragDrop((
                            new_frame.frame,
                            UiInteractionScope {
                                panel_id: old_frame.hover_id,
                                state: old_frame.trigger_panel_state,
                            },
                        )));
                    }
                }

                if old_frame.hover_id != u32::MAX && new_frame.hover_id != old_frame.hover_id {
                    hub.push(CpuPanelEvent::OUT((
                        new_frame.frame,
                        UiInteractionScope {
                            panel_id: old_frame.hover_id,
                            state: old_frame.trigger_panel_state,
                        },
                    )));
                }
            },
        );
    }
}

pub struct AnimationComputeStage {
    pipeline: wgpu::ComputePipeline,
    layout0: wgpu::BindGroupLayout,
    layout1: wgpu::BindGroupLayout,
    bind_group0: wgpu::BindGroup,
    bind_group1: wgpu::BindGroup,
    animation_count: u32,
    dirty: bool,
    trace: GpuDebug,
}

impl AnimationComputeStage {
    const WORKGROUP_SIZE: u32 = 64;

    pub fn new(
        device: &wgpu::Device,
        buffers: &BufferArena,
        global_uniform: &wgpu::Buffer,
    ) -> Self {
        let layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui::animation-bind-group-0"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui::animation-bind-group-1"),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui::animation-pipeline-layout"),
            bind_group_layouts: &[&layout0, &layout1],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui::animation-compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../animtion_compute.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ui::animation-compute-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let mut trace = GpuDebug::new("animtion::compute");
        trace.create_buffer(device);
        let bind_group0 = Self::create_bind_group0(
            device,
            &layout0,
            buffers,
            global_uniform,
            &trace.buffer.clone().unwrap(),
        );
        let bind_group1 = Self::create_bind_group1(device, &layout1, buffers);

        Self {
            pipeline,
            layout0,
            layout1,
            bind_group0,
            bind_group1,
            animation_count: 0,
            dirty: false,
            trace,
        }
    }

    fn create_bind_group0(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffers: &BufferArena,
        global_uniform: &wgpu::Buffer,
        trace_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui::animation-bind-group-0"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.instance.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: global_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.animation_fields.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.panel_anim_delta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.animation_descriptor.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: trace_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_bind_group1(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffers: &BufferArena,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui::animation-bind-group-1"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.relation_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.relations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.instance.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.snapshot.as_entire_binding(),
                },
            ],
        })
    }

    pub fn rebuild_bind_groups(
        &mut self,
        device: &wgpu::Device,
        buffers: &BufferArena,
        global_uniform: &wgpu::Buffer,
    ) {
        self.bind_group0 = Self::create_bind_group0(
            device,
            &self.layout0,
            buffers,
            global_uniform,
            &self.trace.buffer.as_ref().unwrap(),
        );
        self.bind_group1 = Self::create_bind_group1(device, &self.layout1, buffers);
    }

    pub fn set_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn set_animation_count(&mut self, count: u32) {
        self.animation_count = count;
        if count > 0 {
            self.dirty = true;
        }
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn encode(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        _buffers: &BufferViewSet<'_>,
        _ctx: &FrameComputeContext<'_>,
    ) {
        if self.animation_count == 0 {
            self.dirty = false;
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group0, &[]);
        pass.set_bind_group(1, &self.bind_group1, &[]);
        let workgroups = (self.animation_count + Self::WORKGROUP_SIZE - 1) / Self::WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        self.dirty = false;
    }

    pub fn readback(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _ctx: &FrameComputeContext<'_>,
    ) {
        self.trace.debug(_device, _queue);
    }
}

/// Relation compute stage writes group offsets into the shared panel delta buffer.
pub struct RelationComputeStage {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    args_buffer: wgpu::Buffer,
    work_buffer: wgpu::Buffer,
    work_count: u32,
    capacity: u32,
    levels: Vec<GpuRelationDispatchArgs>,
    trace: GpuDebug,
}

impl RelationComputeStage {
    const WORKGROUP_SIZE: u32 = 64;

    pub fn new(device: &wgpu::Device, buffers: &BufferArena) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui::relation-bind-layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui::relation-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..4,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui::relation-compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rel.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ui::relation-compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut trace = GpuDebug::new("ui::relation-compute");
        trace.create_buffer(device);
        let trace_buffer = trace
            .buffer
            .as_ref()
            .expect("relation trace buffer not created")
            .clone();
        let args_buffer = buffers.relation_args.clone();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui::relation-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.panel_anim_delta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.relation_work.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.instance.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: args_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: trace_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.snapshot.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bind_group,
            args_buffer,
            work_buffer: buffers.relation_work.clone(),
            work_count: 0,
            capacity: buffers.capacities.max_relations,
            levels: Vec::new(),
            trace,
        }
    }

    pub fn ingest(&mut self, queue: &wgpu::Queue) {
        let mut registry = relation_registry().lock().unwrap();
        let mut work = registry.take_dirty();
        drop(registry);

        if work.is_empty() {
            self.work_count = 0;
            self.levels.clear();
            return;
        }

        work.sort_by(|a, b| {
            a.depth
                .cmp(&b.depth)
                .then(a.order.cmp(&b.order))
                .then(a.panel_id.cmp(&b.panel_id))
        });

        let limit = work.len().min(self.capacity as usize);
        let mut gpu_items = Vec::with_capacity(limit);
        self.levels.clear();

        for item in work.into_iter().take(limit) {
            if self
                .levels
                .last()
                .map(|level| level.depth != item.depth)
                .unwrap_or(true)
            {
                self.levels.push(GpuRelationDispatchArgs {
                    start: gpu_items.len() as u32,
                    count: 0,
                    depth: item.depth,
                    _pad: 0,
                });
            }
            if let Some(level) = self.levels.last_mut() {
                level.count += 1;
            }

            let gpu_item = GpuRelationWorkItem {
                panel_id: item.panel_id,
                container_id: item.container_id,
                relation_flags: item.layout_flags,
                order: item.order,
                total: if item.total == 0 { 1 } else { item.total },
                flags: item.flags,
                is_container: item.is_container as u32,
                _pad0: 0,
                origin: item.origin,
                container_size: item.size,
                slot_size: item.slot,
                spacing: item.spacing,
                padding: item.padding,
                percent: item.percent,
                scale: item.scale,
                entry_mode: item.entry_mode,
                entry_param: item.entry_param,
                exit_mode: item.exit_mode,
                exit_param: item.exit_param,
            };
            gpu_items.push(gpu_item);
        }

        if gpu_items.is_empty() {
            self.work_count = 0;
            self.levels.clear();
            return;
        }

        self.work_count = gpu_items.len() as u32;
        queue.write_buffer(&self.work_buffer, 0, bytemuck::cast_slice(&gpu_items));
        queue.write_buffer(&self.args_buffer, 0, bytemuck::cast_slice(&self.levels));
    }

    pub fn encode(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        _buffers: &BufferViewSet<'_>,
        _ctx: &FrameComputeContext<'_>,
    ) -> bool {
        if self.work_count == 0 || self.levels.is_empty() {
            return false;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let mut dispatched = false;
        for (level_index, level) in self.levels.iter().enumerate() {
            if level.count == 0 {
                continue;
            }
            let level_idx = level_index as u32;
            let bytes = bytemuck::bytes_of(&level_idx);
            pass.set_push_constants(0, bytes);
            let workgroups = (level.count + Self::WORKGROUP_SIZE - 1) / Self::WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
            dispatched = true;
        }
        dispatched
    }

    #[inline]
    pub fn has_work(&self) -> bool {
        self.work_count > 0
    }

    pub fn readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _ctx: &FrameComputeContext<'_>,
    ) {
        self.trace.debug(device, queue);
    }
}

pub struct PanelDeltaStage {
    trace: GpuDebug,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    dirty: bool,
}

impl PanelDeltaStage {
    const WORKGROUP_SIZE: u32 = 64;

    pub fn new(
        device: &wgpu::Device,
        buffers: &BufferArena,
        global_uniform: &wgpu::Buffer,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ui::panel-delta-layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui::panel-delta-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui::panel-delta-apply"),
            source: wgpu::ShaderSource::Wgsl(include_str!("panel_delta_apply.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ui::panel-delta-apply"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut trace = GpuDebug::new("panel-delta-stage");
        trace.create_buffer(device);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui::panel-delta-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.instance.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.panel_anim_delta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: trace.buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.snapshot.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.spawn_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: global_uniform.as_entire_binding(),
                },
            ],
        });

        Self {
            trace: trace,
            pipeline,
            bind_group,
            dirty: false,
        }
    }

    pub fn readback(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.trace.debug(device, queue);
    }

    pub fn rebuild_bind_group(
        &mut self,
        device: &wgpu::Device,
        buffers: &BufferArena,
        global_uniform: &wgpu::Buffer,
    ) {
        let layout = self.pipeline.get_bind_group_layout(0);
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui::panel-delta-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.instance.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.panel_anim_delta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.trace.buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.snapshot.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.spawn_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: global_uniform.as_entire_binding(),
                },
            ],
        });
        self.dirty = true;
    }

    #[inline]
    pub fn set_dirty(&mut self) {
        self.dirty = true;
    }

    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn encode(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        _buffers: &BufferViewSet<'_>,
        ctx: &FrameComputeContext<'_>,
    ) {
        if !self.dirty || ctx.panel_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let workgroups = (ctx.panel_count + Self::WORKGROUP_SIZE - 1) / Self::WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        self.dirty = false;
    }
}

/// CPU driven relation stage that writes group offsets into the animation delta buffer.
/// Fallback stage used while individual compute passes are not implemented.
#[derive(Default)]
pub struct NoopStage {
    dirty: bool,
}

impl NoopStage {
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    #[inline]
    pub fn set_dirty(&mut self) {
        self.dirty = true;
    }

    #[inline]
    fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    pub fn encode(
        &mut self,
        _pass: &mut wgpu::ComputePass<'_>,
        _buffers: &BufferViewSet<'_>,
        _ctx: &FrameComputeContext<'_>,
    ) {
        self.clear_dirty();
    }

    pub fn readback(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _ctx: &FrameComputeContext<'_>,
    ) {
    }
}

#[test]
fn output_gpu_struct_size() {
    println!(
        "当前的数据大小 {:?}",
        std::mem::size_of::<GpuRelationWorkItem>()
    )
}
