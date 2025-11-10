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
    // === 16-byte 块 1 ===
    position: vec2<f32>,    // 8 bytes
    size: vec2<f32>,        // 8 bytes

    // === 16-byte 块 2 ===
    uv_offset: vec2<f32>,   // 8 bytes  
    uv_scale: vec2<f32>,    // 8 bytes

    // === 16-byte 块 3 ===
    z_index: u32,           // 4 bytes
    pass_through: u32,      // 4 bytes
    id: u32,                // 4 bytes
    interaction: u32,       // 4 bytes

    // === 16-byte 块 4 ===
    event_mask: u32,        // 4 bytes
    state_mask: u32,        // 4 bytes
    transparent: f32,       // 4 bytes
    texture_id: u32,        // 4 bytes

    // === 16-byte 块 5 ===
    state: u32,             // 4 bytes
    collection_state: u32,  // 4 bytes
    fragment_shader_id: u32,// 4 bytes
    vertex_shader_id: u32,  // 4 bytes

    // === 16-byte 块 6 ===
    color: vec4<f32>,       // 16 bytes

    // === 16-byte 块 7 ===
    border_color: vec4<f32>,// 16 bytes

    // === 16-byte 块 8 ===
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

struct RelArgs {
    data: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read_write> panel_deltas: array<PanelAnimDelta>;

@group(0) @binding(1)
var<storage, read_write> work_items: array<RelWorkItem>;

@group(0) @binding(2)
var<storage, read_write> panels: array<Panel>;

@group(0) @binding(3)
var<uniform> rel_args: RelArgs;

@group(0) @binding(4)
var<storage, read_write> debug_buffer: GpuUiDebugReadCallBack;

const INVALID_ID: u32 = 0xffffffffu;
const REL_WORK_FLAG_ENTER_CONTAINER: u32 = 1u << 0u;
const REL_WORK_FLAG_EXIT_CONTAINER: u32 = 1u << 1u;

fn is_valid_panel(id: u32) -> bool {
    return id != INVALID_ID && id < arrayLength(&panels);
}

fn is_valid_delta(id: u32) -> bool {
    return id != INVALID_ID && id < arrayLength(&panel_deltas);
}

fn fetch_container_delta(container_id: u32) -> vec2<f32> {
    if (!is_valid_delta(container_id)) {
        return vec2<f32>(0.0);
    }
    return panel_deltas[container_id].delta_position;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= rel_args.data.x) {
        return;
    }

    var item = work_items[idx];
    let panel_id = item.panel_id;


    if (!is_valid_delta(panel_id) || !is_valid_panel(panel_id)) {
        return;
    }

 


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
        let current_pos = panels[panel_id].position;
        let container_pos = panels[item.container_id].position;
        let desired_pos = container_pos + item.origin;
        let delta = current_pos - desired_pos ;
        panels[panel_id].position = desired_pos;
        panel_deltas[panel_id].delta_position = delta;
        panel_deltas[panel_id].container_origin = desired_pos;

        item.flags &= ~REL_WORK_FLAG_ENTER_CONTAINER;
        work_items[idx] = item;
        return;
    }

    if (!is_valid_delta(item.container_id)) {
        return;
    }

    let container_delta = fetch_container_delta(item.container_id);
    panel_deltas[panel_id].delta_position = container_delta;
    panel_deltas[panel_id].container_origin = container_delta;
}
