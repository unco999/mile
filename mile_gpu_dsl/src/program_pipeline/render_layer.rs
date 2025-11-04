use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use crate::prelude::{gpu_ast::{*},op::{*}, gpu_ast_compute_pipeline::*,manager::{*}, gpu_program::*, *};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderChannel {
    Constant(f32),
    ComputeResult { node_index: u32 },
    RenderImport { name: &'static str, mask: u32, component: u32 },
    RenderComposite {
        mask: u32,
        component: u32,
        factor_render_component: Option<u32>,
        factor_render_scale: f32,
        factor_inner_constant: f32,
        factor_inner_compute: Option<u32>,
        factor_outer_constant: f32,
        factor_outer_compute: Option<u32>,
        factor_unary: FactorUnary,
        offset: f32,
    },
    RenderExpression {
        expr_start: u32,
        expr_len: u32,
    },
}

pub const CHANNEL_CONSTANT: u32 = 0;
pub const CHANNEL_COMPUTE: u32 = 1;
pub const CHANNEL_RENDER_IMPORT: u32 = 2;
pub const CHANNEL_RENDER_COMPOSITE: u32 = 3;
pub const CHANNEL_RENDER_EXPR: u32 = 4;

pub const RENDER_EXPR_OP_CONSTANT: u32 = 0;
pub const RENDER_EXPR_OP_RENDER_IMPORT: u32 = 1;
pub const RENDER_EXPR_OP_COMPUTE_RESULT: u32 = 2;
pub const RENDER_EXPR_OP_UNARY_SIN: u32 = 3;
pub const RENDER_EXPR_OP_UNARY_COS: u32 = 4;
pub const RENDER_EXPR_OP_UNARY_TAN: u32 = 5;
pub const RENDER_EXPR_OP_UNARY_EXP: u32 = 6;
pub const RENDER_EXPR_OP_UNARY_LOG: u32 = 7;
pub const RENDER_EXPR_OP_UNARY_SQRT: u32 = 8;
pub const RENDER_EXPR_OP_UNARY_ABS: u32 = 9;
pub const RENDER_EXPR_OP_NEGATE: u32 = 10;
pub const RENDER_EXPR_OP_BINARY_ADD: u32 = 20;
pub const RENDER_EXPR_OP_BINARY_SUB: u32 = 21;
pub const RENDER_EXPR_OP_BINARY_MUL: u32 = 22;
pub const RENDER_EXPR_OP_BINARY_DIV: u32 = 23;
pub const RENDER_EXPR_OP_BINARY_MOD: u32 = 24;
pub const RENDER_EXPR_OP_BINARY_POW: u32 = 25;
pub const RENDER_EXPR_OP_BINARY_GT: u32 = 30;
pub const RENDER_EXPR_OP_BINARY_GE: u32 = 31;
pub const RENDER_EXPR_OP_BINARY_LT: u32 = 32;
pub const RENDER_EXPR_OP_BINARY_LE: u32 = 33;
pub const RENDER_EXPR_OP_BINARY_EQ: u32 = 34;
pub const RENDER_EXPR_OP_BINARY_NE: u32 = 35;
pub const RENDER_EXPR_OP_IF: u32 = 40;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RenderExprNodeGpu {
    pub op: u32,
    pub arg0: u32,
    pub arg1: u32,
    pub arg2: u32,
    pub data0: f32,
    pub data1: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RenderBindingComponent {
    pub channel_type: u32,
    pub source_index: u32,
    pub component_index: u32,
    pub source_component: u32,
    pub factor_component: u32,
    pub factor_inner_compute: u32,
    pub factor_outer_compute: u32,
    pub factor_unary: u32,
    pub payload: [f32; 4],
}

impl RenderBindingComponent {
    pub fn from_channel(channel: &RenderChannel, component_index: usize) -> Self {
        match channel {
            RenderChannel::Constant(value) => Self {
                channel_type: CHANNEL_CONSTANT,
                source_index: 0,
                component_index: component_index as u32,
                source_component: 0,
                factor_component: u32::MAX,
                factor_inner_compute: u32::MAX,
                factor_outer_compute: u32::MAX,
                factor_unary: 0,
                payload: [*value, 0.0, 0.0, 0.0],
            },
            RenderChannel::ComputeResult { node_index } => Self {
                channel_type: CHANNEL_COMPUTE,
                source_index: *node_index,
                component_index: component_index as u32,
                source_component: 0,
                factor_component: u32::MAX,
                factor_inner_compute: u32::MAX,
                factor_outer_compute: u32::MAX,
                factor_unary: 0,
                payload: [0.0; 4],
            },
            RenderChannel::RenderImport { name: _, mask, component } => Self {
                channel_type: CHANNEL_RENDER_IMPORT,
                source_index: *mask,
                component_index: component_index as u32,
                source_component: *component,
                factor_component: u32::MAX,
                factor_inner_compute: u32::MAX,
                factor_outer_compute: u32::MAX,
                factor_unary: 0,
                payload: [0.0; 4],
            },
            RenderChannel::RenderComposite {
                mask,
                component,
                factor_render_component,
                factor_render_scale,
                factor_inner_constant,
                factor_inner_compute,
                factor_outer_constant,
                factor_outer_compute,
                factor_unary,
                offset,
            } => Self {
                channel_type: CHANNEL_RENDER_COMPOSITE,
                source_index: *mask,
                component_index: component_index as u32,
                source_component: *component,
                factor_component: factor_render_component.unwrap_or(u32::MAX),
                factor_inner_compute: factor_inner_compute.unwrap_or(u32::MAX),
                factor_outer_compute: factor_outer_compute.unwrap_or(u32::MAX),
                factor_unary: factor_unary.to_u32(),
                payload: [
                    *factor_render_scale,
                    *factor_inner_constant,
                    *factor_outer_constant,
                    *offset,
                ],
            },
            RenderChannel::RenderExpression { expr_start, expr_len } => Self {
                channel_type: CHANNEL_RENDER_EXPR,
                source_index: *expr_start,
                component_index: *expr_len,
                source_component: expr_start + expr_len - 1,
                factor_component: u32::MAX,
                factor_inner_compute: u32::MAX,
                factor_outer_compute: u32::MAX,
                factor_unary: 0,
                payload: [0.0; 4],
            },
        }
    }
}

#[repr(C, align(16))]
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
        expr_offset: u32,
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
                RenderComponent::RenderImport { name, mask, component } =>
                    RenderChannel::RenderImport { name: *name, mask: *mask, component: *component },
                RenderComponent::RenderComposite { name: _, mask,
                    component,
                    factor_render_component,
                    factor_render_scale,
                    factor_inner_constant,
                    factor_inner_compute,
                    factor_outer_constant,
                    factor_outer_compute,
                    factor_unary,
                    offset,
                } =>
                    RenderChannel::RenderComposite {
                        mask: *mask,
                        component: *component,
                        factor_render_component: *factor_render_component,
                        factor_render_scale: *factor_render_scale,
                        factor_inner_constant: *factor_inner_constant,
                        factor_inner_compute: factor_inner_compute.map(|id| node_offset + id),
                        factor_outer_constant: *factor_outer_constant,
                        factor_outer_compute: factor_outer_compute.map(|id| node_offset + id),
                        factor_unary: *factor_unary,
                        offset: *offset,
                    },
                RenderComponent::RenderExpression { expr_start, expr_len } =>
                    RenderChannel::RenderExpression {
                        expr_start: expr_offset + *expr_start,
                        expr_len: *expr_len,
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

pub fn encode_render_expr_nodes(
    nodes: &[RenderExprNode],
    node_offset: u32,
    expr_offset: u32,
) -> Vec<RenderExprNodeGpu> {
    nodes
        .iter()
        .map(|node| match &node.op {
            RenderExprOp::Constant => RenderExprNodeGpu {
                op: RENDER_EXPR_OP_CONSTANT,
                arg0: 0,
                arg1: 0,
                arg2: 0,
                data0: node.data0,
                data1: node.data1,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            RenderExprOp::RenderImport => RenderExprNodeGpu {
                op: RENDER_EXPR_OP_RENDER_IMPORT,
                arg0: node.arg0,
                arg1: node.arg1,
                arg2: 0,
                data0: 0.0,
                data1: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            RenderExprOp::ComputeResult => RenderExprNodeGpu {
                op: RENDER_EXPR_OP_COMPUTE_RESULT,
                arg0: node.arg0 + node_offset,
                arg1: 0,
                arg2: 0,
                data0: 0.0,
                data1: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            RenderExprOp::Unary(op) => RenderExprNodeGpu {
                op: match *op {
                    SerializableUnaryOp::Sin => RENDER_EXPR_OP_UNARY_SIN,
                    SerializableUnaryOp::Cos => RENDER_EXPR_OP_UNARY_COS,
                    SerializableUnaryOp::Tan => RENDER_EXPR_OP_UNARY_TAN,
                    SerializableUnaryOp::Exp => RENDER_EXPR_OP_UNARY_EXP,
                    SerializableUnaryOp::Log => RENDER_EXPR_OP_UNARY_LOG,
                    SerializableUnaryOp::Sqrt => RENDER_EXPR_OP_UNARY_SQRT,
                    SerializableUnaryOp::Abs => RENDER_EXPR_OP_UNARY_ABS,
                },
                arg0: node.arg0 + expr_offset,
                arg1: 0,
                arg2: 0,
                data0: 0.0,
                data1: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            RenderExprOp::Binary(op) => RenderExprNodeGpu {
                op: match *op {
                    SerializableBinaryOp::Add => RENDER_EXPR_OP_BINARY_ADD,
                    SerializableBinaryOp::Subtract => RENDER_EXPR_OP_BINARY_SUB,
                    SerializableBinaryOp::Multiply => RENDER_EXPR_OP_BINARY_MUL,
                    SerializableBinaryOp::Divide => RENDER_EXPR_OP_BINARY_DIV,
                    SerializableBinaryOp::Modulo => RENDER_EXPR_OP_BINARY_MOD,
                    SerializableBinaryOp::Pow => RENDER_EXPR_OP_BINARY_POW,
                    SerializableBinaryOp::GreaterThan => RENDER_EXPR_OP_BINARY_GT,
                    SerializableBinaryOp::GreaterEqual => RENDER_EXPR_OP_BINARY_GE,
                    SerializableBinaryOp::LessThan => RENDER_EXPR_OP_BINARY_LT,
                    SerializableBinaryOp::LessEqual => RENDER_EXPR_OP_BINARY_LE,
                    SerializableBinaryOp::Equal => RENDER_EXPR_OP_BINARY_EQ,
                    SerializableBinaryOp::NotEqual => RENDER_EXPR_OP_BINARY_NE,
                    SerializableBinaryOp::Index => unreachable!("index operator not expected in render expression"),
                },
                arg0: node.arg0 + expr_offset,
                arg1: node.arg1 + expr_offset,
                arg2: 0,
                data0: 0.0,
                data1: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            RenderExprOp::Negate => RenderExprNodeGpu {
                op: RENDER_EXPR_OP_NEGATE,
                arg0: node.arg0 + expr_offset,
                arg1: 0,
                arg2: 0,
                data0: 0.0,
                data1: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            RenderExprOp::If => RenderExprNodeGpu {
                op: RENDER_EXPR_OP_IF,
                arg0: node.arg0 + expr_offset,
                arg1: node.arg1 + expr_offset,
                arg2: node.arg2 + expr_offset,
                data0: 0.0,
                data1: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
            },
        })
        .collect()
}


