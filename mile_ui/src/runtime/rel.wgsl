
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
struct Panel {
    // === 16-byte 区 1 ===
    position: vec2<f32>,    // 8 bytes
    size: vec2<f32>,        // 8 bytes

    // === 16-byte 区 2 ===
    uv_offset: vec2<f32>,   // 8 bytes
    uv_scale: vec2<f32>,    // 8 bytes

    // === 16-byte 区 3 ===
    z_index: u32,           // 4 bytes
    pass_through: u32,      // 4 bytes
    id: u32,                // 4 bytes
    interaction: u32,       // 4 bytes

    // === 16-byte 区 4 ===
    event_mask: u32,        // 4 bytes
    state_mask: u32,        // 4 bytes
    transparent: f32,       // 4 bytes
    texture_id: u32,        // 4 bytes

    // === 16-byte 区 5 ===
    state: u32,             // 4 bytes
    collection_state: u32,  // 4 bytes
    fragment_shader_id: u32,// 4 bytes
    vertex_shader_id: u32,  // 4 bytes

    // === 16-byte 区 6 ===
    rotation: vec4<f32>,

    // === 16-byte 区 7 ===
    scale: vec4<f32>,

    // === 16-byte 区 8 ===
    color: vec4<f32>,       // 16 bytes

    // === 16-byte 区 9 ===
    border_color: vec4<f32>,// 16 bytes

    // === 16-byte 区 10 ===
    border_width: f32,      // 4 bytes
    border_radius: f32,     // 4 bytes
    visible: u32,           // 4 bytes
    _pad_border: u32,       // 4 bytes (填充)
};

struct GpuUiDebugReadCallBack {
    floats: array<f32, 32>,
    uints: array<u32, 32>,
};

struct RelWorkItem {
    panel_id: u32,
    container_id: u32,
    relation_flags: u32,
    order: u32,
    total: u32,
    flags: u32,
    is_container: u32,
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

struct RelDispatchArgs {
    start: u32,
    count: u32,
    depth: u32,
    _pad: u32,
};

struct RelDispatchCursor {
    level: u32,
};

@group(0) @binding(0)
var<storage, read_write> panel_deltas: array<PanelAnimDelta>;

@group(0) @binding(1)
var<storage, read_write> work_items: array<RelWorkItem>;

@group(0) @binding(2)
var<storage, read_write> panels: array<Panel>;

@group(0) @binding(3)
var<storage, read> rel_dispatch_table: array<RelDispatchArgs>;

var<push_constant> rel_dispatch_cursor: RelDispatchCursor;

@group(0) @binding(4)
var<storage, read_write> debug_buffer: GpuUiDebugReadCallBack;

@group(0) @binding(5)
var<storage, read_write> panel_snapshots: array<Panel>;

const INVALID_ID: u32 = 4294967295u;
const REL_WORK_FLAG_ENTER_CONTAINER: u32 = 1u << 0u;
const REL_WORK_FLAG_EXIT_CONTAINER: u32 = 1u << 1u;

const REL_LAYOUT_TYPE_MASK: u32 = 0xffu;
const REL_LAYOUT_FREE: u32 = 0u;
const REL_LAYOUT_HORIZONTAL: u32 = 1u;
const REL_LAYOUT_VERTICAL: u32 = 2u;
const REL_LAYOUT_GRID: u32 = 3u;
const REL_LAYOUT_RING: u32 = 4u;
const REL_LAYOUT_ALIGN_CENTER: u32 = 1u << 12u;
const TAU: f32 = 6.28318530718;

fn is_valid_panel(id: u32) -> bool {
    return (panels[id - 1].size.x > 0.0) && (panels[id - 1].size.y > 0.0);
}

fn fetch_container_delta(container_id: u32) -> vec2<f32> {
    if (!is_valid_panel(container_id)) {
        return vec2<f32>(0.0);
    }
    return panel_deltas[container_id - 1].delta_position;
}

fn clamp_order(order: u32, total: u32) -> u32 {
    let safe_total = max(total, 1u);
    return min(order, safe_total - 1u);
}

fn resolve_container_size(item: RelWorkItem) -> vec2<f32> {
    var size = item.container_size;
    if (all(size == vec2<f32>(0.0)) && is_valid_panel(item.container_id)) {
        size = panels[item.container_id].size;
    }
    return size;
}

fn resolve_slot_size(item: RelWorkItem, panel_id: u32) -> vec2<f32> {
    var slot = item.slot_size;
    if (all(slot == vec2<f32>(0.0))) {
        slot = panels[panel_id].size;
    }
    return slot * item.scale;
}

fn align_axis_offset(
    start_pad: f32,
    end_pad: f32,
    extent: f32,
    slot_extent: f32,
    center: bool,
) -> f32 {
    let available = max(extent - start_pad - end_pad, 0.0);
    if (center) {
        let slack = max(available - slot_extent, 0.0);
        return start_pad + slack * 0.5;
    }
    return start_pad;
}

fn compute_layout_offset(panel_id: u32, item: RelWorkItem) -> vec2<f32> {

    let layout_kind = item.relation_flags & REL_LAYOUT_TYPE_MASK;
    if (layout_kind == REL_LAYOUT_FREE) {
        return vec2<f32>(0.0);
    }

    let slot = resolve_slot_size(item, panel_id);
    let container_size = resolve_container_size(item);
    let align_center = (item.relation_flags & REL_LAYOUT_ALIGN_CENTER) != 0u;
    let left_pad = item.padding.x;
    let top_pad = item.padding.y;
    let right_pad = item.padding.z;
    let bottom_pad = item.padding.w;
    let order = clamp_order(item.order, item.total);

    var offset = vec2<f32>(left_pad, top_pad);
    if (layout_kind == REL_LAYOUT_HORIZONTAL) {
        let stride = slot.x + item.spacing.x;
        offset.x += f32(order) * stride;
        offset.y = align_axis_offset(top_pad, bottom_pad, container_size.y, slot.y, align_center);
    } else if (layout_kind == REL_LAYOUT_VERTICAL) {
        let stride = slot.y + item.spacing.y;
        offset.y += f32(order) * stride;
        offset.x = align_axis_offset(left_pad, right_pad, container_size.x, slot.x, align_center);
    } else if (layout_kind == REL_LAYOUT_GRID) {
        let stride = slot + item.spacing;
        let available_width = max(container_size.x - left_pad - right_pad, stride.x);
        let columns = max(u32(available_width / stride.x), 1u);
        let col = order % columns;
        let row = order / columns;
        offset.x += f32(col) * stride.x;
        offset.y += f32(row) * stride.y;
    } else if (layout_kind == REL_LAYOUT_RING) {
        let radius = abs(item.spacing.x);
        let dir = select(-1.0, 1.0, item.spacing.x >= 0.0);
        let start_angle = item.spacing.y;
        let total = max(item.total, 1u);
        let angle_step = TAU / f32(total);
        let angle = start_angle + dir * angle_step * f32(order);
        let inner_width = max(container_size.x - left_pad - right_pad, radius * 2.0);
        let inner_height = max(container_size.y - top_pad - bottom_pad, radius * 2.0);
        let center = vec2<f32>(
            left_pad + inner_width * 0.5,
            top_pad + inner_height * 0.5,
        );
        offset = center + vec2<f32>(cos(angle), sin(angle)) * radius - slot * 0.5;
    }
    return offset;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let level = rel_dispatch_cursor.level;
    let dispatch = rel_dispatch_table[level];
    let local_idx = global_id.x;
    if (local_idx >= dispatch.count) {
        return;
    }
    let idx = dispatch.start + local_idx;
    var item = work_items[idx];
    let panel_id = item.panel_id;



 


    //if ((item.flags & REL_WORK_FLAG_EXIT_CONTAINER) != 0u) {
    //    panel_deltas[panel_id].container_origin = vec2<f32>(0.0);
    //    panel_deltas[panel_id].delta_position = vec2<f32>(0.0);
    //    item.flags &= ~REL_WORK_FLAG_EXIT_CONTAINER;
    //    work_items[idx] = item;
    //    return;
    //}
  

    if ((item.flags & REL_WORK_FLAG_ENTER_CONTAINER) != 0u) {


        if(panel_id == 50){
            debug_buffer.uints[9] = 99999;
        }

        let current_pos = panels[panel_id - 1].position;
        


        let container_pos = panels[item.container_id - 1].position;
        let container_origin = container_pos + work_items[item.container_id - 1].origin;
        let layout_offset = compute_layout_offset(panel_id - 1, item);
        let desired_pos = container_origin + layout_offset;
        work_items[idx].origin = layout_offset;
        
        panel_deltas[panel_id - 1].start_position = current_pos;
        //panel_deltas[panel_id].delta_position += delta;
        panels[panel_id - 1].position = desired_pos;
        panel_deltas[panel_id - 1].container_origin = container_origin;
        panel_snapshots[panel_id - 1].position = desired_pos;
        panels[panel_id - 1].collection_state = 1u;
        item.flags &= ~REL_WORK_FLAG_ENTER_CONTAINER;
        work_items[idx] = item;
        return;
    }

    if (!is_valid_panel(item.container_id)) {
        return;
    }

    let container_delta = panel_deltas[item.container_id].delta_position;
    panel_deltas[panel_id].delta_position += container_delta;
    panel_snapshots[panel_id - 1].position += container_delta;
    let container_pos = panels[item.container_id - 1].position;
    let container_origin = container_pos + item.origin;
    panel_deltas[panel_id - 1].container_origin = container_origin + work_items[item.container_id - 1].origin;
}
