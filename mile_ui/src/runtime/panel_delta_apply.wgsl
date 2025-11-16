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
    // Apply size delta
    let dsize = panel_deltas[panel_id].delta_size;
    if (dsize.x != 0.0 || dsize.y != 0.0) {
        panels[idx].size += dsize;
        panel_deltas[panel_id].delta_size = vec2<f32>(0.0);
    }
    // UV offsets
    let duv_off = panel_deltas[panel_id].delta_uv_offset;
    if (duv_off.x != 0.0 || duv_off.y != 0.0) {
        panels[idx].uv_offset += duv_off;
        panel_deltas[panel_id].delta_uv_offset = vec2<f32>(0.0);
    }
    let duv_scale = panel_deltas[panel_id].delta_uv_scale;
    if (duv_scale.x != 0.0 || duv_scale.y != 0.0) {
        panels[idx].uv_scale += duv_scale;
        panel_deltas[panel_id].delta_uv_scale = vec2<f32>(0.0);
    }
    // Transparent
    let dtrans = panel_deltas[panel_id].delta_transparent;
    if (dtrans != 0.0) {
        panels[idx].transparent += dtrans;
        panel_deltas[panel_id].delta_transparent = 0.0;
    }
    // Integer-like fields: z_index, pass_through
    let dz = panel_deltas[panel_id].delta_z_index;
    if (dz != 0) {
        let z = i32(panels[idx].z_index) + dz;
        panels[idx].z_index = u32(max(z, 0));
        panel_deltas[panel_id].delta_z_index = 0;
    }
    let dpass = panel_deltas[panel_id].delta_pass_through;
    if (dpass != 0) {
        let p = i32(panels[idx].pass_through) + dpass;
        panels[idx].pass_through = u32(max(p, 0));
        panel_deltas[panel_id].delta_pass_through = 0;
    }
    // Interaction/event/state mask: OR then clear
    let di = panel_deltas[panel_id].delta_interaction;
    if (di != 0u) {
        panels[idx].interaction = panels[idx].interaction | di;
        panel_deltas[panel_id].delta_interaction = 0u;
    }
    let dem = panel_deltas[panel_id].delta_event_mask;
    if (dem != 0u) {
        panels[idx].event_mask = panels[idx].event_mask | dem;
        panel_deltas[panel_id].delta_event_mask = 0u;
    }
    let dsm = panel_deltas[panel_id].delta_state_mask;
    if (dsm != 0u) {
        panels[idx].state_mask = panels[idx].state_mask | dsm;
        panel_deltas[panel_id].delta_state_mask = 0u;
    }
    // Texture id overwrite if non-zero delta
    let dtex = panel_deltas[panel_id].delta_texture_id;
    if (dtex != 0) {
        panels[idx].texture_id = u32(max(dtex, 0));
        panel_deltas[panel_id].delta_texture_id = 0;
    }
}
