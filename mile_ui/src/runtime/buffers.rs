//! GPU buffer provisioning and pool management.
//!
//! The current implementation simply provides the type skeleton that the refactor will
//! populate.  Methods return `todo!()` so that the compiler enforces implementation once
//! we plug the module into the runtime.

use crate::{
    runtime::_ty::{AnimtionFieldOffsetPtr, Panel, PanelAnimDelta, GpuInteractionFrameCache, GpuUiDebugReadCallBack},
    structs::{GpuUiCollection, GpuUiIdInfo, GpuUiInfluence}
};
use bytemuck::bytes_of;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

/// Tunable limits for GPU resource allocation.
#[derive(Debug, Clone)]
pub struct BufferArenaConfig {
    /// Maximum number of panel instances we pre-allocate.
    pub max_panels: u32,
    /// Maximum number of animation field entries.
    pub max_animation_fields: u32,
    /// Maximum number of collections (for network compute).
    pub max_collections: u32,
    /// Maximum number of relations/influences.
    pub max_relations: u32,
}

impl Default for BufferArenaConfig {
    fn default() -> Self {
        Self {
            max_panels: 1_024,
            max_animation_fields: 8_192,
            max_collections: 256,
            max_relations: 4_096,
        }
    }
}

/// Aggregates all persistent GPU buffers used by the UI runtime.
#[derive(Debug)]
pub struct BufferArena {
    pub instance: wgpu::Buffer,
    pub panel_anim_delta: wgpu::Buffer,
    pub animation_fields: wgpu::Buffer,
    pub animation_values: wgpu::Buffer,
    pub collections: wgpu::Buffer,
    pub relations: wgpu::Buffer,
    pub relation_ids: wgpu::Buffer,
    pub indirect_draws: wgpu::Buffer,
    pub interaction_frames: wgpu::Buffer,
    pub debug_buffer: wgpu::Buffer,
    pub capacities: BufferArenaConfig,
}

impl BufferArena {
    pub fn new(device: &wgpu::Device, cfg: &BufferArenaConfig) -> Self {
        let instance_size = (cfg.max_panels as u64 * std::mem::size_of::<Panel>() as u64).max(1);
        let instance = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::instances"),
            size: instance_size,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let panel_anim_delta = PanelAnimDelta::global_init(&device);

        let animation_field_size = (cfg.max_animation_fields as u64
            * std::mem::size_of::<AnimtionFieldOffsetPtr>() as u64)
            .max(1);
        let animation_fields = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::animation-fields"),
            size: animation_field_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let animation_values = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::animation-values"),
            size: (cfg.max_animation_fields as u64 * std::mem::size_of::<f32>() as u64).max(1),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let collections = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::collections"),
            size: (cfg.max_collections as u64 * std::mem::size_of::<GpuUiCollection>() as u64)
                .max(1),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let relations = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::relations"),
            size: (cfg.max_relations as u64 * std::mem::size_of::<GpuUiInfluence>() as u64).max(1),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let relation_ids = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::relation-ids"),
            size: (cfg.max_relations as u64 * std::mem::size_of::<GpuUiIdInfo>() as u64).max(1),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });


        let indirect_draws = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::indirect-draws"),
            size: (cfg.max_panels as u64 * std::mem::size_of::<[u32; 5]>() as u64).max(1),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let interaction_frames = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("ui::interaction-frames"),
            contents: bytes_of(&GpuInteractionFrameCache::default()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let debug_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("ui::interaction-debug"),
            contents: bytes_of(&GpuUiDebugReadCallBack::default()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        Self {
            instance,
            panel_anim_delta,
            animation_fields,
            animation_values,
            collections,
            relations,
            relation_ids,
            indirect_draws,
            interaction_frames,
            debug_buffer,
            capacities: cfg.clone(),
        }
    }

    pub fn ensure_capacity(&mut self, cfg: &BufferArenaConfig) {
        if cfg.max_panels > self.capacities.max_panels
            || cfg.max_animation_fields > self.capacities.max_animation_fields
            || cfg.max_collections > self.capacities.max_collections
            || cfg.max_relations > self.capacities.max_relations
        {
            // A full reallocation strategy will be implemented in a future step.
            panic!(
                "BufferArena::ensure_capacity called with larger requirements; growth not implemented yet"
            );
        }
    }

    pub fn staging_guard(&self) -> BufferViewSet<'_> {
        BufferViewSet {
            instance: self.instance.slice(..),
            panel_anim_delta: self.panel_anim_delta.slice(..),
            animation_fields: self.animation_fields.slice(..),
            animation_values: self.animation_values.slice(..),
            collections: self.collections.slice(..),
            relations: self.relations.slice(..),
            relation_ids: self.relation_ids.slice(..),
            indirect_draws: self.indirect_draws.slice(..),
            interaction_frames: self.interaction_frames.slice(..),
            debug_buffer: self.debug_buffer.slice(..),
        }
    }
}

/// Lightweight collection of buffer bindings passed into compute/render stages.
pub struct BufferViewSet<'a> {
    pub instance: wgpu::BufferSlice<'a>,
    pub panel_anim_delta: wgpu::BufferSlice<'a>,
    pub animation_fields: wgpu::BufferSlice<'a>,
    pub animation_values: wgpu::BufferSlice<'a>,
    pub collections: wgpu::BufferSlice<'a>,
    pub relations: wgpu::BufferSlice<'a>,
    pub relation_ids: wgpu::BufferSlice<'a>,
    pub indirect_draws: wgpu::BufferSlice<'a>,
    pub interaction_frames: wgpu::BufferSlice<'a>,
    pub debug_buffer: wgpu::BufferSlice<'a>,
}
