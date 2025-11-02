use std::num::NonZeroU64;

use bytemuck::{Zeroable, cast_slice};
use wgpu::util::DeviceExt;

use super::render_layer::{RenderBindingLayer, RenderExprNodeGpu};

pub struct RenderBindingResources {
    pub layer_buffer: wgpu::Buffer,
    pub results_buffer: wgpu::Buffer,
    pub expr_nodes_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub layer_count: u32,
    pub expr_node_count: u32,
    layer_stride: u64,
    capacity: u32,
    expr_node_capacity: u32,
}

impl RenderBindingResources {
    pub fn with_capacity(
        device: &wgpu::Device,
        capacity: usize,
        compute_results: &wgpu::Buffer,
        expr_nodes: &[RenderExprNodeGpu],
    ) -> Self {
        let actual_capacity = capacity.max(1);
        let stride = std::mem::size_of::<RenderBindingLayer>() as u64;

        let zero_layers = vec![RenderBindingLayer::zeroed(); actual_capacity];
        let layer_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RenderBindingLayerBuffer"),
            contents: cast_slice(&zero_layers),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let layer_min_size = NonZeroU64::new(stride).unwrap();
        let expr_stride = std::mem::size_of::<RenderExprNodeGpu>() as u64;

        let mut expr_zero = Vec::new();
        let expr_slice = if expr_nodes.is_empty() {
            expr_zero.push(RenderExprNodeGpu::zeroed());
            &expr_zero[..]
        } else {
            expr_nodes
        };

        let expr_nodes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RenderExprNodesBuffer"),
            contents: cast_slice(expr_slice),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RenderBindingLayout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Some(layer_min_size),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(expr_stride).or_else(|| NonZeroU64::new(16)),
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RenderBindingBindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: layer_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: compute_results.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: expr_nodes_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            layer_buffer,
            results_buffer: compute_results.clone(),
            expr_nodes_buffer,
            bind_group_layout,
            bind_group,
            layer_count: 0,
            expr_node_count: expr_nodes.len() as u32,
            layer_stride: stride,
            capacity: actual_capacity as u32,
            expr_node_capacity: expr_slice.len() as u32,
        }
    }

    pub fn write_layers(&mut self, queue: &wgpu::Queue, layers: &[RenderBindingLayer]) {
        assert!(
            layers.len() as u32 <= self.capacity,
            "render binding layers exceed reserved capacity"
        );
        queue.write_buffer(&self.layer_buffer, 0, cast_slice(layers));
        self.layer_count = layers.len().min(self.capacity as usize) as u32;
    }

    pub fn layer_stride(&self) -> u64 {
        self.layer_stride
    }

    pub fn write_expr_nodes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        nodes: &[RenderExprNodeGpu],
    ) {
        let required = nodes.len().max(1) as u32;
        if required > self.expr_node_capacity {
            let mut init_data = Vec::new();
            let data_slice = if nodes.is_empty() {
                init_data.push(RenderExprNodeGpu::zeroed());
                &init_data[..]
            } else {
                nodes
            };

            self.expr_nodes_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("RenderExprNodesBuffer"),
                    contents: cast_slice(data_slice),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            self.expr_node_capacity = required;
            self.rebuild_bind_group(device);
        }

        let mut scratch = Vec::new();
        let upload_slice = if nodes.is_empty() {
            scratch.push(RenderExprNodeGpu::zeroed());
            &scratch[..]
        } else {
            nodes
        };

        queue.write_buffer(&self.expr_nodes_buffer, 0, cast_slice(upload_slice));
        self.expr_node_count = nodes.len() as u32;
    }

    fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RenderBindingBindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.layer_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.expr_nodes_buffer.as_entire_binding(),
                },
            ],
        });
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}
