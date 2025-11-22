//! GPU buffer provisioning and pool management.
//!
//! The current implementation simply provides the type skeleton that the refactor will
//! populate.  Methods return `todo!()` so that the compiler enforces implementation once
//! we plug the module into the runtime.

use crate::{
    runtime::_ty::{
        AnimtionFieldOffsetPtr, GpuAnimationDes, GpuInteractionFrameCache, GpuRelationDispatchArgs,
        GpuUiDebugReadCallBack, Panel, PanelAnimDelta,
    },
    structs::{GpuUiCollection, GpuUiIdInfo, GpuUiInfluence},
};
use bytemuck::{bytes_of, cast_slice};
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
    pub snapshot: wgpu::Buffer,
    pub panel_anim_delta: wgpu::Buffer,
    pub spawn_flags: wgpu::Buffer,
    pub animation_fields: wgpu::Buffer,
    pub animation_values: wgpu::Buffer,
    pub animation_descriptor: wgpu::Buffer,
    pub collections: wgpu::Buffer,
    pub relations: wgpu::Buffer,
    pub relation_ids: wgpu::Buffer,
    pub relation_work: wgpu::Buffer,
    pub relation_args: wgpu::Buffer,
    pub indirect_draws: wgpu::Buffer,
    pub interaction_frames: wgpu::Buffer,
    pub debug_buffer: wgpu::Buffer,
    pub clamp_rules: wgpu::Buffer,
    pub clamp_desc: wgpu::Buffer,
    pub capacities: BufferArenaConfig,
    pub animation_fields_capacity: u32,
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

        let snapshot = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::panel-snapshots"),
            size: instance_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let panel_anim_delta = PanelAnimDelta::global_init(&device);
        let spawn_flags = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::spawn-flags"),
            size: (cfg.max_panels as u64 * std::mem::size_of::<u32>() as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let panel_index_stride = (cfg.max_panels as usize).max(1);
        let panel_index_init = vec![u32::MAX; panel_index_stride];

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

        let animation_descriptor = GpuAnimationDes::default().to_buffer(device);

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

        let relation_work = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::relation-work"),
            size: (cfg.max_relations as u64
                * std::mem::size_of::<crate::runtime::_ty::GpuRelationWorkItem>() as u64)
                .max(1),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let relation_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::relation-args"),
            size: (cfg.max_relations as u64
                * std::mem::size_of::<GpuRelationDispatchArgs>() as u64)
                .max(1),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

        // Clamp rules buffer and descriptor (count)
        let clamp_desc = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("ui::clamp-descriptor"),
            contents: bytes_of(&crate::runtime::_ty::GpuClampDescriptor::default()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let clamp_rules = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui::clamp-rules"),
            size: (cfg.max_relations as u64
                * std::mem::size_of::<crate::runtime::_ty::GpuClampRule>() as u64)
                .max(1),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            instance,
            snapshot,
            panel_anim_delta,
            spawn_flags,
            animation_fields,
            animation_values,
            animation_descriptor,
            collections,
            relations,
            relation_ids,
            relation_work,
            relation_args,
            indirect_draws,
            interaction_frames,
            debug_buffer,
            clamp_rules,
            clamp_desc,
            capacities: cfg.clone(),
            animation_fields_capacity: cfg.max_animation_fields,
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
            snapshot: self.snapshot.slice(..),
            panel_anim_delta: self.panel_anim_delta.slice(..),
            animation_fields: self.animation_fields.slice(..),
            animation_values: self.animation_values.slice(..),
            animation_descriptor: self.animation_descriptor.slice(..),
            collections: self.collections.slice(..),
            relations: self.relations.slice(..),
            relation_ids: self.relation_ids.slice(..),
            indirect_draws: self.indirect_draws.slice(..),
            interaction_frames: self.interaction_frames.slice(..),
            debug_buffer: self.debug_buffer.slice(..),
            clamp_rules: self.clamp_rules.slice(..),
            clamp_desc: self.clamp_desc.slice(..),
        }
    }
    pub fn reset_panel_memory(&self, queue: &wgpu::Queue) {
        fn zero_buffer(queue: &wgpu::Queue, buffer: &wgpu::Buffer, size: usize) {
            if size == 0 {
                return;
            }
            let zeros = vec![0u8; size];
            queue.write_buffer(buffer, 0, &zeros);
        }

        let panel_count = self.capacities.max_panels as usize;
        let panel_bytes = panel_count * std::mem::size_of::<Panel>();
        zero_buffer(queue, &self.instance, panel_bytes);
        zero_buffer(queue, &self.snapshot, panel_bytes);

        let delta_bytes = panel_count * std::mem::size_of::<PanelAnimDelta>();
        zero_buffer(queue, &self.panel_anim_delta, delta_bytes);
        zero_buffer(
            queue,
            &self.spawn_flags,
            panel_count * std::mem::size_of::<u32>(),
        );
        zero_buffer(
            queue,
            &self.indirect_draws,
            panel_count * std::mem::size_of::<[u32; 5]>(),
        );

        let anim_fields = self.capacities.max_animation_fields as usize;
        zero_buffer(
            queue,
            &self.animation_fields,
            anim_fields * std::mem::size_of::<AnimtionFieldOffsetPtr>(),
        );
        zero_buffer(
            queue,
            &self.animation_values,
            anim_fields * std::mem::size_of::<f32>(),
        );

        let collections = self.capacities.max_collections as usize;
        zero_buffer(
            queue,
            &self.collections,
            collections * std::mem::size_of::<GpuUiCollection>(),
        );

        let relations = self.capacities.max_relations as usize;
        zero_buffer(
            queue,
            &self.relations,
            relations * std::mem::size_of::<GpuUiInfluence>(),
        );
        zero_buffer(
            queue,
            &self.relation_ids,
            relations * std::mem::size_of::<GpuUiIdInfo>(),
        );
        zero_buffer(
            queue,
            &self.relation_work,
            relations * std::mem::size_of::<crate::runtime::_ty::GpuRelationWorkItem>(),
        );
        zero_buffer(
            queue,
            &self.relation_args,
            relations * std::mem::size_of::<GpuRelationDispatchArgs>(),
        );
        zero_buffer(
            queue,
            &self.clamp_rules,
            relations * std::mem::size_of::<crate::runtime::_ty::GpuClampRule>(),
        );

        let clamp_desc = crate::runtime::_ty::GpuClampDescriptor::default();
        queue.write_buffer(&self.clamp_desc, 0, bytes_of(&clamp_desc));

        let interaction_cache = crate::runtime::_ty::GpuInteractionFrameCache::zeroed();
        queue.write_buffer(&self.interaction_frames, 0, bytes_of(&interaction_cache));
        let debug = crate::runtime::_ty::GpuUiDebugReadCallBack::default();
        queue.write_buffer(&self.debug_buffer, 0, bytes_of(&debug));

        let anim_desc = GpuAnimationDes::default();
        queue.write_buffer(&self.animation_descriptor, 0, bytes_of(&anim_desc));
    }
}

/// Lightweight collection of buffer bindings passed into compute/render stages.
pub struct BufferViewSet<'a> {
    pub instance: wgpu::BufferSlice<'a>,
    pub snapshot: wgpu::BufferSlice<'a>,
    pub panel_anim_delta: wgpu::BufferSlice<'a>,
    pub animation_fields: wgpu::BufferSlice<'a>,
    pub animation_values: wgpu::BufferSlice<'a>,
    pub animation_descriptor: wgpu::BufferSlice<'a>,
    pub collections: wgpu::BufferSlice<'a>,
    pub relations: wgpu::BufferSlice<'a>,
    pub relation_ids: wgpu::BufferSlice<'a>,
    pub indirect_draws: wgpu::BufferSlice<'a>,
    pub interaction_frames: wgpu::BufferSlice<'a>,
    pub debug_buffer: wgpu::BufferSlice<'a>,
    pub clamp_rules: wgpu::BufferSlice<'a>,
    pub clamp_desc: wgpu::BufferSlice<'a>,
}
