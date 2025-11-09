//! New UI runtime decomposition.
//!
//! This module is currently a skeleton: it documents the shape of the new GPU runtime
//! (buffer arena, compute stages, render batches, CPU runtime state) without replacing
//! the existing `mui.rs` implementation yet.  Individual submodules can be fleshed out
//! and wired into `GpuUi` incrementally.

pub mod _ty;
pub mod buffers;
pub mod compute;
pub mod entry;
pub mod relations;
pub mod render;
pub mod state;

pub use buffers::{BufferArena, BufferArenaConfig, BufferViewSet};
pub use compute::{ComputePipelines, FrameComputeContext};
pub use entry::{FrameHistory, FrameSnapshot, MuiRuntime, register_payload_refresh};
pub use relations::{
    RelationWorkItem, clear_panel_relations, inject_relation_work, layout_flags,
    register_panel_relations, relation_registry, set_panel_active_state,
};
pub use render::{QuadBatchKind, RenderBatches, RenderPipelines};
pub use state::{
    ClickCallback, CpuPanelEvent, EntryCallBack, EntryFragBack, EntryVertexBack, FRAME, FrameState,
    HoverCallback, NetWorkTransition, OutCallBack, PanelEventRegistry, RuntimeState,
    StateConfigDes, StateNetWorkConfigDes, StateOpenCall, StateTransition, UIEventHub,
    UiInteractionScope, WgslResult,
};
