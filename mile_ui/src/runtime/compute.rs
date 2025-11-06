//! Compute pipeline orchestration.
//!
//! Encapsulates the GPU compute stages used by the UI runtime.  Right now only the
//! interaction stage is implemented; network and animation stages remain no-ops but
//! keep the scaffolding ready for future work.

use std::{cell::RefCell, sync::Arc};

use bytemuck::cast_slice;
use wgpu::util::DownloadBuffer;

use crate::runtime::{
    buffers::{BufferArena, BufferViewSet},
    _ty::GpuInteractionFrame,
    state::{CpuPanelEvent, UiInteractionScope, UIEventHub},
};
use mile_api::interface::GpuDebug;

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
    pub network: NoopStage,
    pub animation: NoopStage,
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
            network: NoopStage::default(),
            animation: NoopStage::default(),
        }
    }

    pub fn encode_all(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        buffers: &BufferViewSet<'_>,
        ctx: &FrameComputeContext<'_>,
    ) {
        if self.interaction.is_dirty() {
            self.interaction.encode(pass, buffers, ctx);
        }
        if self.network.is_dirty() {
            self.network.encode(pass, buffers, ctx);
        }
        if self.animation.is_dirty() {
            self.animation.encode(pass, buffers, ctx);
        }
    }

    pub fn readback_all(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ctx: &FrameComputeContext<'_>,
    ) {
        self.interaction.readback(device, queue, ctx);
        self.network.readback(device, queue, ctx);
        self.animation.readback(device, queue, ctx);
    }

    #[inline]
    pub fn mark_interaction_dirty(&mut self) {
        self.interaction.set_dirty();
    }

    #[inline]
    pub fn mark_all_dirty(&mut self) {
        self.interaction.set_dirty();
        self.network.set_dirty();
        self.animation.set_dirty();
    }

    #[inline]
    pub fn is_any_dirty(&self) -> bool {
        self.interaction.is_dirty() || self.network.is_dirty() || self.animation.is_dirty()
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
    }
}

/// Interaction compute pipeline.
pub struct InteractionComputeStage {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui::interaction-pipeline-layout"),
            bind_group_layouts: &[&layout],
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

        Self {
            pipeline,
            layout,
            bind_group,
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

                if new_frame.click_id != u32::MAX {
                    println!("点击事件 {:?}",new_frame.click_id);
                    hub.push(CpuPanelEvent::Click((
                        new_frame.frame,
                        UiInteractionScope {
                            panel_id: new_frame.click_id,
                            state: new_frame.trigger_panel_state,
                        },
                    )));
                }

                if new_frame.drag_id != u32::MAX {
                    println!("拖拽事件 {:?}",new_frame.drag_id);
                    hub.push(CpuPanelEvent::Drag((
                        new_frame.frame,
                        UiInteractionScope {
                            panel_id: new_frame.drag_id,
                            state: new_frame.trigger_panel_state,
                        },
                    )));
                }

                if new_frame.hover_id != u32::MAX && new_frame.hover_id != old_frame.hover_id {
                    println!("悬浮事件 {:?}",new_frame.hover_id);
                    hub.push(CpuPanelEvent::Hover((
                        new_frame.frame,
                        UiInteractionScope {
                            panel_id: new_frame.hover_id,
                            state: new_frame.trigger_panel_state,
                        },
                    )));
                }

                if old_frame.hover_id != u32::MAX && new_frame.hover_id != old_frame.hover_id {
                    println!("离开事件");
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
