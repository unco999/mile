# Mile UI Runtime Rewrite Plan

Owned file: `mile_ui/src/mui.rs` (~3.2k LOC) currently mixes CPU state, GPU buffers,
pipelines, animation caches, interaction logic, network sync, etc. Our target is a
modular runtime that can feed the new panel layout, three render tiers, and GPU driven
animation/network passes without wasting bandwidth.

This document tracks the staged refactor.

---

## Stage A · Planning / Skeleton

### A1. BufferArena (显存资源调度)
- Manage **all persistent GPU buffers** (vertex/index/instance, animation fields, network
  descriptors, indirect draw commands, custom shader payloads).
- Provide `BufferArenaConfig` that derives sensible pool sizes from GPU limits
  (VRAM), with user overrides.
- Expose typed handles (`InstancePool`, `AnimationPool`, `NetworkPool`) that
  support:
  - contiguous range allocation (with staging writes & partial updates)
  - capacity growth via reallocation + copy
  - returning bind views for compute/render bind groups.
- Keep `Panel`/`PanelAnimDelta` struct definitions close to the pool so host/GPU
  stay in sync.

### A2. ComputePipelines
- Encapsulate three compute stages with uniform interface:
  1. `interaction` (reads panel + shared data → writes hover/click state)
  2. `network` (collections/relations → panel deltas)
  3. `animation` (field resolution + application)
- Each stage exposes:
  ```rust
  pub struct ComputeStage {
      pub fn ensure(&mut self, device, arena_cfg) -> Result<()>;
      pub fn encode(&mut self, pass, arena_views, frame_ctx);
      pub fn readback(&mut self, device, queue);
      pub fn mark_dirty(&mut self);
  }
  ```
- Bind group layouts align with existing WGSL (or updated versions) and consume
  `BufferArena` views.
- Stages run sequentially inside one compute pass for maximum coherence.

### A3. RenderPipelines / RenderBatches
- Manage three render tiers:
  1. `StaticQuadBatch`
  2. `VertexAnimatedQuadBatch`
  3. `OverlayQuadBatch`
- Each batch owns its render pipeline + instance range + indirect draw records.
- Provide `RenderContext::encode(pass, arena_views, frame_ctx)` for `GpuUi::render`.
- Single quad vertex buffer reused; per-tier instance buffers may have extra attributes.

### A4. RuntimeState
- CPU-side caches currently inside `GpuUi`: event hub, texture atlas, kennel bindings,
  global uniforms, animation queues, etc.
- `RuntimeState` exposes mutators invoked by builders/events; it stages data into
  `BufferArena` and marks compute stages dirty.

### A5. GpuUi Orchestrator
- Holds `BufferArena`, `ComputePipelines`, `RenderPipelines`, `RuntimeState`.
- Responsibilities per frame:
  1. flush CPU staged writes (`BufferArena::flush`)
  2. run compute stages sequentially
  3. update indirect commands
  4. issue render batches
- Public API remains similar (init, resize, update_mouse, process_events) but delegates internally.

### A6. Documentation & Traits
- Update `mile_api::Computeable` trait to supply `encode`/`is_dirty`.
  (Done.)
- Provide rustdoc diagrams and README sections so other crates integrate cleanly.

---

## Stage B · Implementation Steps

1. **Introduce skeleton modules** (`runtime/buffers.rs`, `runtime/compute.rs`,
   `runtime/render.rs`, `runtime/state.rs`) with TODO methods and unit tests explaining
   expected behaviour.
2. **Port existing buffer creation** logic into `BufferArena`, maintaining the old
   defaults (1000 panels, 8192 animations, etc.) behind the scenes.
3. **Wrap existing compute init** in `ComputePipelines` while keeping the WGSL unchanged;
   run them through the new orchestrator but write results into current structures.
4. **Gradually migrate `GpuUi::render`** to use render batches. Start with StaticQuad,
   then add vertex animated, overlay.
5. Remove old fields/functions from `GpuUi` once the new modules supply equivalents.
6. Introduce new WGSL or adjust existing ones to match the final buffer layout.
7. Finally, replace direct `queue.write_buffer` calls with `BufferArena` helpers and
   delete dead code.

Each step should compile independently; prefer feature flags or alternate constructors
if we need to keep the old path alive temporarily.

---

## Notes / Considerations

- Buffer growth must preserve content; prefer `wgpu::CommandEncoder::copy_buffer_to_buffer`.
- Keep `CustomWgsl` data accessible — likely as a separate pool inside BufferArena.
- `AnimationWriter` (introduced in `mui_prototype.rs`) will feed into the new animation stage; ensure pending animations are uploaded once per frame and cleared.
- Kitty Kennel integration: maintain current binding logic but move into Render/Compute modules for clarity.
- Tests/Demos (`mui_prototype`, `mui_build`) should work untouched, only the runtime internals change.

This document is updated alongside each milestone to reflect completion status and next tasks.
