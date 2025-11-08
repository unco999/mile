//! Render pipeline aggregation and per-tier batches.
//!
//! This module currently defines the shape of the render subsystem.  Actual pipeline
//! construction and draw submission will be implemented alongside the runtime refactor.

use std::{mem, ops::Range};

use crate::runtime::_ty::{quad_index_bytes, quad_index_len, quad_vertex_bytes};
use serde::{Deserialize, Serialize};
use wgpu::{
    IndexFormat,
    util::{BufferInitDescriptor, DeviceExt},
};

/// Enumeration of the vertex batch tiers we render each frame.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuadBatchKind {
    /// 正常顶点数量
    Normal,
    /// 较多顶点
    MultiVertex,
    /// 非常多的顶点
    UltraVertex,
}

impl Default for QuadBatchKind {
    fn default() -> Self {
        QuadBatchKind::Normal
    }
}

/// Holds the render pipeline and instance range for a single batch.
pub struct QuadBatch {
    pub pipeline: Option<wgpu::RenderPipeline>,
    pub bind_group: Option<wgpu::BindGroup>,
    pub instance_range: std::ops::Range<u32>,
    pub indirect_count: u32,
}

impl QuadBatch {
    pub fn ensure_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        let _ = (device, format);
        debug_assert!(
            self.pipeline.is_some(),
            "QuadBatch pipeline missing; configure RenderPipelines before encoding"
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        render_bind_group: &wgpu::BindGroup,
        texture_bind_group: &wgpu::BindGroup,
        kennel_bind_group: Option<&wgpu::BindGroup>,
        indirect_buffer: &wgpu::Buffer,
        fallback_instances: Range<u32>,
        fallback_indirect_count: u32,
        num_indices: u32,
    ) {
        let Some(pipeline) = &self.pipeline else {
            return;
        };

        if self.indirect_count == 0
            && self.instance_range.is_empty()
            && fallback_instances.is_empty()
        {
            return;
        }

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, render_bind_group, &[]);
        pass.set_bind_group(1, texture_bind_group, &[]);
        if let Some(kennel) = kennel_bind_group {
            pass.set_bind_group(2, kennel, &[]);
        }
        if let Some(batch_group) = &self.bind_group {
            pass.set_bind_group(3, batch_group, &[]);
        }

        let indirect_count = if self.indirect_count > 0 {
            self.indirect_count
        } else {
            fallback_indirect_count
        };

        if indirect_count > 0 {
            let stride = mem::size_of::<[u32; 5]>() as u64;
            for i in 0..indirect_count {
                let offset = i as u64 * stride;
                pass.draw_indexed_indirect(indirect_buffer, offset);
            }
            return;
        }

        let instance_range = if self.instance_range.is_empty() {
            fallback_instances
        } else {
            self.instance_range.clone()
        };

        if !instance_range.is_empty() && num_indices > 0 {
            pass.draw_indexed(0..num_indices, 0, instance_range);
        }
    }
}

/// Aggregates the three quad batches used by the runtime.
pub struct RenderBatches {
    pub normal_quads: QuadBatch,
    pub multi_vertex_quads: QuadBatch,
    pub ultra_vertex_quads: QuadBatch,
}

impl RenderBatches {
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (QuadBatchKind, &mut QuadBatch)> {
        [
            (QuadBatchKind::Normal, &mut self.normal_quads),
            (QuadBatchKind::MultiVertex, &mut self.multi_vertex_quads),
            (QuadBatchKind::UltraVertex, &mut self.ultra_vertex_quads),
        ]
        .into_iter()
    }

    pub fn iter(&self) -> impl Iterator<Item = (QuadBatchKind, &QuadBatch)> {
        [
            (QuadBatchKind::Normal, &self.normal_quads),
            (QuadBatchKind::MultiVertex, &self.multi_vertex_quads),
            (QuadBatchKind::UltraVertex, &self.ultra_vertex_quads),
        ]
        .into_iter()
    }

    pub fn get(&self, kind: QuadBatchKind) -> &QuadBatch {
        match kind {
            QuadBatchKind::Normal => &self.normal_quads,
            QuadBatchKind::MultiVertex => &self.multi_vertex_quads,
            QuadBatchKind::UltraVertex => &self.ultra_vertex_quads,
        }
    }

    pub fn get_mut(&mut self, kind: QuadBatchKind) -> &mut QuadBatch {
        match kind {
            QuadBatchKind::Normal => &mut self.normal_quads,
            QuadBatchKind::MultiVertex => &mut self.multi_vertex_quads,
            QuadBatchKind::UltraVertex => &mut self.ultra_vertex_quads,
        }
    }
}

struct QuadGeometry {
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    index_count: u32,
}

impl QuadGeometry {
    fn new(device: &wgpu::Device, kind: QuadBatchKind) -> Self {
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!("ui::quad-vertex-buffer::{kind:?}")),
            contents: quad_vertex_bytes(kind),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_data = quad_index_bytes(kind);
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!("ui::quad-index-buffer::{kind:?}")),
            contents: index_data,
            usage: wgpu::BufferUsages::INDEX,
        });
        let index_count = quad_index_len(kind);
        Self {
            vertex: vertex_buffer,
            index: index_buffer,
            index_count,
        }
    }
}

struct QuadGeometrySet {
    normal: QuadGeometry,
    multi_vertex: QuadGeometry,
    ultra_vertex: QuadGeometry,
}

impl QuadGeometrySet {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            normal: QuadGeometry::new(device, QuadBatchKind::Normal),
            multi_vertex: QuadGeometry::new(device, QuadBatchKind::MultiVertex),
            ultra_vertex: QuadGeometry::new(device, QuadBatchKind::UltraVertex),
        }
    }

    fn get(&self, kind: QuadBatchKind) -> &QuadGeometry {
        match kind {
            QuadBatchKind::Normal => &self.normal,
            QuadBatchKind::MultiVertex => &self.multi_vertex,
            QuadBatchKind::UltraVertex => &self.ultra_vertex,
        }
    }
}

/// Container wiring the batches with shared resources (bind layouts, kennel bindings).
pub struct RenderPipelines {
    geometries: QuadGeometrySet,
    pub batches: RenderBatches,
}

impl RenderPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            geometries: QuadGeometrySet::new(device),
            batches: RenderBatches {
                normal_quads: QuadBatch {
                    pipeline: None,
                    bind_group: None,
                    instance_range: 0..0,
                    indirect_count: 0,
                },
                multi_vertex_quads: QuadBatch {
                    pipeline: None,
                    bind_group: None,
                    instance_range: 0..0,
                    indirect_count: 0,
                },
                ultra_vertex_quads: QuadBatch {
                    pipeline: None,
                    bind_group: None,
                    instance_range: 0..0,
                    indirect_count: 0,
                },
            },
        }
    }

    pub fn set_pipeline(&mut self, kind: QuadBatchKind, pipeline: wgpu::RenderPipeline) {
        self.batches.get_mut(kind).pipeline = Some(pipeline);
    }

    pub fn set_instance_range(&mut self, kind: QuadBatchKind, range: Range<u32>) {
        self.batches.get_mut(kind).instance_range = range;
    }

    pub fn set_indirect_count(&mut self, kind: QuadBatchKind, count: u32) {
        self.batches.get_mut(kind).indirect_count = count;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_main(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        render_bind_group: &wgpu::BindGroup,
        texture_bind_group: &wgpu::BindGroup,
        kennel_bind_group: Option<&wgpu::BindGroup>,
        instance_buffer: &wgpu::Buffer,
        indirect_buffer: &wgpu::Buffer,
        indirect_count: u32,
        _legacy_num_indices: u32,
        num_instances: u32,
    ) {
        if num_instances == 0 {
            return;
        }

        pass.set_vertex_buffer(1, instance_buffer.slice(..));

        // Keep fallback disabled so only batches with explicit ranges draw.
        let fallback_instances = 0..0;

        for (kind, batch) in self.batches.iter() {
            let should_skip = batch.indirect_count == 0
                && batch.instance_range.is_empty()
                && fallback_instances.is_empty()
                && indirect_count == 0;
            if should_skip {
                continue;
            }

            let geometry: &QuadGeometry = self.geometries.get(kind);
            pass.set_vertex_buffer(0, geometry.vertex.slice(..));
            pass.set_index_buffer(geometry.index.slice(..), IndexFormat::Uint16);

            batch.encode(
                pass,
                render_bind_group,
                texture_bind_group,
                kennel_bind_group,
                indirect_buffer,
                fallback_instances.clone(),
                indirect_count,
                geometry.index_count,
            );
        }
    }
}
