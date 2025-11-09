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
    _pad3: vec2<f32>,
};

struct RelWorkItem {
    panel_id: u32,
    relation_flags: u32,
    order: u32,
    total: u32,
    origin: vec2<f32>,
    container_size: vec2<f32>,
    slot_size: vec2<f32>,
    spacing: vec2<f32>,
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

fn layout_position(item: RelWorkItem) -> vec2<f32> {
    let idx = f32(item.order);
    var pos = item.origin;
    switch (item.relation_flags & 0xFu) {
        case REL_LAYOUT_HORIZONTAL: {
            let step = item.slot_size.x + item.spacing.x;
            pos.x = item.origin.x + idx * step;
        }
        case REL_LAYOUT_VERTICAL: {
            let step = item.slot_size.y + item.spacing.y;
            pos.y = item.origin.y + idx * step;
        }
        default: {
            // free layout uses origin directly
        }
    }
    return pos;
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

    let _target = layout_position(item);
    panel_deltas[panel_index].delta_position = _target;
    panel_deltas[panel_index].delta_size = item.slot_size;
}
