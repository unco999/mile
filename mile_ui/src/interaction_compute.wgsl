// Interaction compute shader rewritten for the new panel data layout.
// Panel position now represents the top-left corner in pixel space, and
// size is the width/height. Hover/click/drag results are written back into
// the frame cache buffer (two frames ring buffer) so the CPU can poll them.

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
    texture_slot: u32,

    state: u32,
    collection_state: u32,
    kennel_des_id: u32,
    flags: u32,

    color: vec4<f32>,
    border_color: vec4<f32>,
    border_width: f32,
    border_radius: f32,
    pad_border: vec2<f32>,
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
}

struct GlobalUniform {
    click_layout_z: atomic<u32>,
    click_layout_id: atomic<u32>,
    hover_layout_id: atomic<u32>,
    hover_layout_z: atomic<u32>,

    drag_layout_id: atomic<u32>,
    drag_layout_z: atomic<u32>,
    pad_atomic1: atomic<u32>,
    pad_atomic2: atomic<u32>,

    dt: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,

    mouse_pos: vec2<f32>,
    mouse_state: u32,
    frame: u32,

    screen_size: vec2<u32>,
    press_duration: f32,
    time: f32,

    event_point: vec2<f32>,
    extra1: vec2<f32>,

    extra2: vec2<f32>,
    pad_extra: vec2<f32>,
};

struct GpuInteractionFrame {
    frame: u32,
    drag_id: u32,
    hover_id: u32,
    click_id: u32,
    mouse_pos: vec2<f32>,
    trigger_panel_state: u32,
    _pad1: u32,
    mouse_state: u32,
    _pad2_0: u32,
    _pad2_1: u32,
    _pad2_2: u32,
    drag_delta: vec2<f32>,
    _pad3: vec2<f32>,
    pinch_delta: f32,
    pass_through_depth: u32,
    event_point: vec2<f32>,
    _pad4: array<u32, 4>,
};

struct GpuUiDebugReadCallBack {
    floats: array<f32, 32>,
    uints: array<u32, 32>,
};


const INVALID_ID: u32 = 0xffffffffu;
const MOUSE_LEFT_DOWN: u32 = 1u;
const MOUSE_LEFT_UP: u32 = 2u;
const MOUSE_LEFT_HELD: u32 = 16u;
const MOUSE_LEFT_RELEASED: u32 = 32u;
const INTERACTION_CLICK: u32 = 2u;
const INTERACTION_HOVER: u32 = 8u;
const INTERACTION_DRAG: u32 = 4u;
const DRAG_PRESS_THRESHOLD: f32 = 0.24;

@group(0) @binding(0) var<storage, read_write> panels: array<Panel>;
@group(0) @binding(1) var<storage, read_write> global_uniform: GlobalUniform;
@group(0) @binding(2) var<storage, read_write> frame_cache: array<GpuInteractionFrame, 2>;
@group(0) @binding(3) var<storage, read_write> debug_buffer: GpuUiDebugReadCallBack;
@group(0) @binding(4) var<storage, read_write> panel_anim_delta: array<PanelAnimDelta>;

fn mouse_inside(panel: Panel, mouse: vec2<f32>) -> bool {
    let min = panel.position;
    let max = panel.position + panel.size;
    return mouse.x >= min.x && mouse.x <= max.x && mouse.y >= min.y && mouse.y <= max.y;
}

fn claim_hover(candidate_z: u32, candidate_id: u32, pass_through: u32) -> bool {
    if (pass_through != 0u) {
        return false;
    }
    let prev_z = atomicMax(&global_uniform.hover_layout_z, candidate_z);
    let prev_id = atomicLoad(&global_uniform.hover_layout_id);
    if (candidate_z > prev_z || (candidate_z == prev_z && candidate_id >= prev_id)) {
        atomicStore(&global_uniform.hover_layout_id, candidate_id);
        return true;
    }
    return false;
}

fn claim_click(candidate_z: u32, candidate_id: u32, pass_through: u32) -> bool {
    if (pass_through != 0u) {
        return false;
    }
    let prev_z = atomicMax(&global_uniform.click_layout_z, candidate_z);
    let prev_id = atomicLoad(&global_uniform.click_layout_id);
    if (candidate_z > prev_z || (candidate_z == prev_z && candidate_id >= prev_id)) {
        atomicStore(&global_uniform.click_layout_id, candidate_id);
        return true;
    }
    return false;
}

fn claim_drag(candidate_z: u32, candidate_id: u32, pass_through: u32) -> bool {
    if (pass_through != 0u) {
        return false;
    }
    let prev_z = atomicMax(&global_uniform.drag_layout_z, candidate_z);
    let prev_id = atomicLoad(&global_uniform.drag_layout_id);
    if (candidate_z > prev_z || (candidate_z == prev_z && candidate_id >= prev_id)) {
        atomicStore(&global_uniform.drag_layout_id, candidate_id);
        return true;
    }
    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&panels)) {
        return;
    }

    let panel = panels[idx];
    let mouse = global_uniform.mouse_pos;
    let screen = vec2<f32>(global_uniform.screen_size);

    let hovered = mouse_inside(panel, mouse);
    let mouse_pressed = (global_uniform.mouse_state & MOUSE_LEFT_HELD) != 0u;
    let mouse_released = (global_uniform.mouse_state & MOUSE_LEFT_RELEASED) != 0u;
    let press_duration = global_uniform.press_duration;
        

    if (!hovered) {
        return;
    }


    // Hover claim
    if ((panel.interaction & INTERACTION_HOVER) != 0u) {
        if (claim_hover(panel.z_index, panel.id, panel.pass_through)) {
            frame_cache[1].hover_id = panel.id;
            frame_cache[1].trigger_panel_state = panel.state;
        }
    }

    // Click claim
    if ((panel.interaction & INTERACTION_CLICK) != 0u && mouse_released) {
        if (claim_click(panel.z_index, panel.id, panel.pass_through)) {
            frame_cache[1].click_id = panel.id;
            frame_cache[1].trigger_panel_state = panel.state;
            frame_cache[1].event_point = mouse - panel.position;
        }
    }

    // Drag (press)
    if ((panel.interaction & INTERACTION_DRAG) != 0u && mouse_pressed && press_duration >= DRAG_PRESS_THRESHOLD) {
        if (claim_drag(panel.z_index, panel.id, panel.pass_through)) {
            frame_cache[1].drag_id = panel.id;
            frame_cache[1].trigger_panel_state = panel.state;
            frame_cache[1].event_point = mouse - panel.position;
        }
    }
}
