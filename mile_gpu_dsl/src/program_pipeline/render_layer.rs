use std::ops::Range;

use bytemuck::{Pod, Zeroable};

use crate::mat::gpu_program::{RenderComponent, SerializableGpuProgram};

use super::ProgramHandle;

/// 描述渲染阶段每个 vec4 输出的来源
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderChannel {
    Constant(f32),
    ComputeResult { node_index: u32 },
    RenderImport { name: &'static str, mask: u32 },
}

pub const CHANNEL_CONSTANT: u32 = 0;
pub const CHANNEL_COMPUTE: u32 = 1;
pub const CHANNEL_RENDER_IMPORT: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RenderBindingComponent {
    pub channel_type: u32,
    pub source_index: u32,
    pub component_index: u32,
    pub reserved: u32,
    pub payload: [f32; 4],
}

impl RenderBindingComponent {
    pub fn from_channel(channel: &RenderChannel, component_index: usize) -> Self {
        match channel {
            RenderChannel::Constant(value) => Self {
                channel_type: CHANNEL_CONSTANT,
                source_index: 0,
                component_index: component_index as u32,
                reserved: 0,
                payload: [*value, 0.0, 0.0, 0.0],
            },
            RenderChannel::ComputeResult { node_index } => Self {
                channel_type: CHANNEL_COMPUTE,
                source_index: *node_index,
                component_index: component_index as u32,
                reserved: 0,
                payload: [0.0; 4],
            },
            RenderChannel::RenderImport { name: _, mask } => Self {
                channel_type: CHANNEL_RENDER_IMPORT,
                source_index: *mask,
                component_index: component_index as u32,
                reserved: 0,
                payload: [0.0; 4],
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RenderBindingLayer {
    pub compute_start: u32,
    pub compute_count: u32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub components: [RenderBindingComponent; 4],
}

/// 单个程序在渲染阶段的绑定描述，供外部渲染管线使用
#[derive(Debug, Clone)]
pub struct RenderLayerDescriptor {
    pub program: ProgramHandle,
    pub compute_range: Range<u32>,
    pub channels: [RenderChannel; 4],
}

impl RenderLayerDescriptor {
    pub fn from_program(
        program: &SerializableGpuProgram,
        program_handle: ProgramHandle,
        node_offset: u32,
    ) -> Self {
        let compute_range =
            node_offset..(node_offset + program.compute_nodes.len() as u32);

        let channels = std::array::from_fn(|idx| {
            let component = &program.render_plan.components[idx];
            match component {
                RenderComponent::Constant { value } => RenderChannel::Constant(*value),
                RenderComponent::ComputeNode { node_id } => RenderChannel::ComputeResult {
                    node_index: node_offset + *node_id,
                },
                RenderComponent::RenderImport { name, mask } => RenderChannel::RenderImport {
                    name: *name,
                    mask: *mask,
                },
            }
        });

        Self {
            program: program_handle,
            compute_range,
            channels,
        }
    }

    pub fn to_binding_layer(&self) -> RenderBindingLayer {
        let mut layer = RenderBindingLayer::zeroed();
        layer.compute_start = self.compute_range.start;
       layer.compute_count = self.compute_range.end - self.compute_range.start;
        for (idx, channel) in self.channels.iter().enumerate() {
            layer.components[idx] = RenderBindingComponent::from_channel(channel, idx);
        }
        layer
    }
}
