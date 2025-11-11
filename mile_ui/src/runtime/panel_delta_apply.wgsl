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

@group(0) @binding(0)
var<storage, read_write> panels: array<Panel>;

@group(0) @binding(1)
var<storage, read_write> panel_deltas: array<PanelAnimDelta>;

@group(0) @binding(2)
var<storage, read_write> debug_buffer: GpuUiDebugReadCallBack;

@group(0) @binding(3)
var<storage, read_write> panel_snapshots: array<Panel>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&panels)) {
        return;
    }

    let panel = panels[idx];
    let panel_id = panel.id;

    
    if (panel_id >= arrayLength(&panel_deltas)) {
        return;
    }

    let delta = panel_deltas[panel_id].delta_position;
    if (delta.x != 0.0 || delta.y != 0.0) {
        panels[idx].position += delta;
        panel_deltas[panel_id].delta_position = vec2<f32>(0.0);
    }
    panel_snapshots[idx] = panels[idx];
}
