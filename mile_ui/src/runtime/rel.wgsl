struct PanelAnimDelta {
    delta_position: vec2<f32>,
    delta_size: vec2<f32>,
    delta_uv_offset: vec2<f32>,
    delta_uv_scale: vec2<f32>,
    delta_z_index: i32,
    delta_pass_through: i32,
    panel_id: u32,
    _pad0: u32,
    delta_interaction: u32,
    delta_event_mask: u32,
    delta_state_mask: u32,
    _pad1: u32,
    delta_transparent: f32,
    delta_texture_id: i32,
    _pad2: vec2<f32>,
    start_position: vec2<f32>,
    container_origin: vec2<f32>,
};

struct RelWorkItem {
    panel_id: u32,
    container_id: u32,
    relation_flags: u32,
    order: u32,
    total: u32,
    _pad0: u32,
    origin: vec2<f32>,
    container_size: vec2<f32>,
    slot_size: vec2<f32>,
    spacing: vec2<f32>,
    padding: vec4<f32>,
    percent: vec2<f32>,
    scale: vec2<f32>,
    entry_mode: u32,
    exit_mode: u32,
    entry_param: f32,
    exit_param: f32,
};

struct RelArgs {
    work_count: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0)
var<storage, read_write> panel_deltas: array<PanelAnimDelta>;

@group(0) @binding(1)
var<storage, read> work_items: array<RelWorkItem>;

@group(0) @binding(2)
var<uniform> rel_args: RelArgs;

const REL_LAYOUT_FREE: u32 = 0u;
const REL_LAYOUT_HORIZONTAL: u32 = 1u;
const REL_LAYOUT_VERTICAL: u32 = 2u;
const REL_LAYOUT_GRID: u32 = 3u;
const REL_LAYOUT_RING: u32 = 4u;
const REL_LAYOUT_MASK: u32 = 0xFu;
const REL_TRANSITION_IMMEDIATE: u32 = 0u;
const REL_TRANSITION_TIMED: u32 = 1u;

fn layout_offset(item: RelWorkItem) -> vec2<f32> {
    let idx = f32(item.order);
    var pos = item.origin;
    switch (item.relation_flags & REL_LAYOUT_MASK) {
        case REL_LAYOUT_HORIZONTAL: {
            let step = item.slot_size.x + item.spacing.x;
            pos.x = item.origin.x + idx * step;
        }
        case REL_LAYOUT_VERTICAL: {
            let step = item.slot_size.y + item.spacing.y;
            pos.y = item.origin.y + idx * step;
        }
        case REL_LAYOUT_GRID: {
            // Placeholder: treat as free until grid layout is implemented.
        }
        case REL_LAYOUT_RING: {
            // Placeholder for future ring layouts.
        }
        default: {
            // free layout uses origin directly
        }
    }
    return pos;
}

fn fetch_container_delta(container_id: u32) -> vec2<f32> {
    if (container_id >= arrayLength(&panel_deltas)) {
        return vec2<f32>(0.0);
    }
    return panel_deltas[container_id].delta_position;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= rel_args.work_count) {
        return;
    }

    let item = work_items[idx];
    let panel_index = item.panel_id;
    if (panel_index >= arrayLength(&panel_deltas)) {
        return;
    }

    let layout_pos = layout_offset(item);
    let container_delta = fetch_container_delta(item.container_id);
    let final_delta = layout_pos + container_delta;

    panel_deltas[panel_index].delta_position = final_delta;
    panel_deltas[panel_index].delta_size = item.slot_size;
}
