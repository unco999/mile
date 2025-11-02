use std::num::NonZeroU64;

use bytemuck::{Zeroable, cast_slice};
use wgpu::util::DeviceExt;

use super::render_layer::RenderBindingLayer;

pub struct RenderBindingResources {
    pub layer_buffer: wgpu::Buffer,
    pub results_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub layer_count: u32,
    layer_stride: u64,
    capacity: u32,
}

impl RenderBindingResources {
    pub fn with_capacity(
        device: &wgpu::Device,
        capacity: usize,
        compute_results: &wgpu::Buffer,
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
            ],
        });

        Self {
            layer_buffer,
            results_buffer: compute_results.clone(),
            bind_group_layout,
            bind_group,
            layer_count: 0,
            layer_stride: stride,
            capacity: actual_capacity as u32,
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

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}
