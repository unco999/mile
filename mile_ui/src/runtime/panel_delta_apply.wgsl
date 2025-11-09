struct Panel {
    position: vec2<f32>,
    size: vec2<f32>,
    uv_offset: vec2<f32>,
    uv_scale: vec2<f32>,

    z_index: u32,
    pass_through: u32,
    id: u32,
    interaction: u32,

    event_mask: u32,
    state_mask: u32,
    transparent: f32,
    texture_id: u32,

    state: u32,
    collection_state: u32,
    fragment_shader_id: u32,
    vertex_shader_id: u32,

    color: vec4<f32>,
    border_color: vec4<f32>,
    border_width: f32,
    border_radius: f32,
    visible: u32,
    _pad_border: u32,
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
    _pad3: vec2<f32>,
};

@group(0) @binding(0)
var<storage, read_write> panels: array<Panel>;

@group(0) @binding(1)
var<storage, read_write> panel_deltas: array<PanelAnimDelta>;

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
}
