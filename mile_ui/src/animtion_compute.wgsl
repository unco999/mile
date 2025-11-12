// Minimal animation compute kernel that lerps panel fields based on the CPU-side
// `AnimtionFieldOffsetPtr` buffer.  Each invocation processes exactly one field entry.

const PANEL_FIELD_POSITION_X : u32 = 0x1u;
const PANEL_FIELD_POSITION_Y : u32 = 0x2u;
const PANEL_FIELD_SIZE_X     : u32 = 0x4u;
const PANEL_FIELD_SIZE_Y     : u32 = 0x8u;
const PANEL_FIELD_TRANSPARENT: u32 = 0x100u;
const PANEL_FIELD_COLOR_R    : u32 = 0x1000u;
const PANEL_FIELD_COLOR_G    : u32 = 0x2000u;
const PANEL_FIELD_COLOR_B    : u32 = 0x4000u;
const PANEL_FIELD_COLOR_A    : u32 = 0x8000u;

const EASING_LINEAR      : u32 = 0x01u;
const EASING_IN_QUAD     : u32 = 0x02u;
const EASING_OUT_QUAD    : u32 = 0x04u;
const EASING_IN_OUT_QUAD : u32 = 0x08u;
const EASING_IN_CUBIC    : u32 = 0x10u;
const EASING_OUT_CUBIC   : u32 = 0x20u;
const EASING_IN_OUT_CUBIC: u32 = 0x40u;

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

struct AnimtionFieldOffsetPtr {
    field_id: u32,
    start_value: f32,
    target_value: f32,
    elapsed: f32,
    duration: f32,
    op: u32,
    hold: u32,
    delay: f32,
    loop_count: u32,
    ping_pong: u32,
    on_complete: u32,
    panel_id: u32,
    death: u32,
    easy_fn: u32,
    is_offset: u32,
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

struct AnimationDescriptor {
    animation_count: u32,
    frame_count: u32,
    start_index: u32,
    _pad0: u32,
    delta_time: f32,
    total_time: f32,
    _pad1: vec2<f32>,
};

struct DebugBuffer {
    floats: array<f32, 32>,
    uints: array<u32, 32>,
};

@group(0) @binding(0)
var<storage, read_write> panels: array<Panel>;

@group(0) @binding(1)
var<storage, read_write> global_uniform: GlobalUniform;

@group(0) @binding(2)
var<storage, read_write> animations: array<AnimtionFieldOffsetPtr>;

@group(0) @binding(3)
var<storage, read_write> panel_deltas: array<PanelAnimDelta>;

@group(0) @binding(4)
var<uniform> animation_meta: AnimationDescriptor;

@group(0) @binding(5)
var<storage, read_write> debug_buffer: DebugBuffer;

fn apply_easing(mask: u32, t: f32) -> f32 {
    let clamped = clamp(t, 0.0, 1.0);
    if ((mask & EASING_IN_QUAD) != 0u) {
        return clamped * clamped;
    }
    if ((mask & EASING_OUT_QUAD) != 0u) {
        let u = 1.0 - clamped;
        return 1.0 - u * u;
    }
    if ((mask & EASING_IN_OUT_QUAD) != 0u) {
        if (clamped < 0.5) {
            let u = clamped * 2.0;
            return 0.5 * u * u;
        }
        let u = (1.0 - clamped) * 2.0;
        return 1.0 - 0.5 * u * u;
    }
    if ((mask & EASING_IN_CUBIC) != 0u) {
        return clamped * clamped * clamped;
    }
    if ((mask & EASING_OUT_CUBIC) != 0u) {
        let u = clamped - 1.0;
        return u * u * u + 1.0;
    }
    if ((mask & EASING_IN_OUT_CUBIC) != 0u) {
        var u = clamped * 2.0;
        if (u < 1.0) {
            return 0.5 * u * u * u;
        }
        u = u - 2.0;
        return 0.5 * (u * u * u + 2.0);
    }
    // default linear
    return clamped;
}

fn write_panel_field(panel_index: u32, field: u32, value: f32) {
    if (field == PANEL_FIELD_POSITION_X) {
        panels[panel_index].position.x = value;
        return;
    }
    if (field == PANEL_FIELD_POSITION_Y) {
        panels[panel_index].position.y = value;
        return;
    }
    if (field == PANEL_FIELD_SIZE_X) {
        panels[panel_index].size.x = value;
        return;
    }
    if (field == PANEL_FIELD_SIZE_Y) {
        panels[panel_index].size.y = value;
        return;
    }
    if (field == PANEL_FIELD_TRANSPARENT) {
        panels[panel_index].transparent = value;
        return;
    }
    if (field == PANEL_FIELD_COLOR_R) {
        panels[panel_index].color.x = value;
        return;
    }
    if (field == PANEL_FIELD_COLOR_G) {
        panels[panel_index].color.y = value;
        return;
    }
    if (field == PANEL_FIELD_COLOR_B) {
        panels[panel_index].color.z = value;
        return;
    }
    if (field == PANEL_FIELD_COLOR_A) {
        panels[panel_index].color.w = value;
        return;
    }
}

fn write_delta(panel_index: u32, field: u32, delta: f32) {
    if (field == PANEL_FIELD_POSITION_X) {
        panel_deltas[panel_index].delta_position.x += delta;
    } else if (field == PANEL_FIELD_POSITION_Y) {
        panel_deltas[panel_index].delta_position.y += delta;
    } else if (field == PANEL_FIELD_SIZE_X) {
        panel_deltas[panel_index].delta_size.x += delta;
    } else if (field == PANEL_FIELD_SIZE_Y) {
        panel_deltas[panel_index].delta_size.y += delta;
    } else if (field == PANEL_FIELD_TRANSPARENT) {
        panel_deltas[panel_index].delta_transparent += delta;
    }
}

fn read_panel_field(panel_index: u32, field: u32) -> f32 {
    if (field == PANEL_FIELD_POSITION_X) {
        return panels[panel_index].position.x;
    }
    if (field == PANEL_FIELD_POSITION_Y) {
        return panels[panel_index].position.y;
    }
    if (field == PANEL_FIELD_SIZE_X) {
        return panels[panel_index].size.x;
    }
    if (field == PANEL_FIELD_SIZE_Y) {
        return panels[panel_index].size.y;
    }
    if (field == PANEL_FIELD_TRANSPARENT) {
        return panels[panel_index].transparent;
    }
    if (field == PANEL_FIELD_COLOR_R) {
        return panels[panel_index].color.x;
    }
    if (field == PANEL_FIELD_COLOR_G) {
        return panels[panel_index].color.y;
    }
    if (field == PANEL_FIELD_COLOR_B) {
        return panels[panel_index].color.z;
    }
    if (field == PANEL_FIELD_COLOR_A) {
        return panels[panel_index].color.w;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= animation_meta.animation_count) {
        return;
    }

    let _frame_debug = global_uniform.frame;

    let anim = animations[idx];
    if (anim.death != 0u) {
        return;
    }

    if (anim.panel_id == 0u) {
        animations[idx].death = 1u;
        return;
    }

    let panel_slot = anim.panel_id;
    if (panel_slot == 0u) {
        animations[idx].death = 1u;
        return;
    }
    let panel_index = panel_slot - 1u;
    let total_panels = arrayLength(&panels);
    if (panel_index >= total_panels) {
        animations[idx].death = 1u;
        return;
    }
    if (panel_slot >= arrayLength(&panel_deltas)) {
        animations[idx].death = 1u;
        return;
    }
    let delta_index = panel_slot;

    let delay = anim.delay;
    let duration = max(anim.duration, 0.00001);
    let dt = animation_meta.delta_time;

    let previous_elapsed = animations[idx].elapsed;
    let new_elapsed = min(previous_elapsed + dt, delay + duration);
    animations[idx].elapsed = new_elapsed;

    if (anim.hold != 0u && previous_elapsed == 0.0) {
        let current = read_panel_field(panel_index, anim.field_id);
        animations[idx].start_value = current;
        if (anim.field_id == PANEL_FIELD_POSITION_X) {
            panel_deltas[delta_index].start_position.x = panels[panel_index].position.x;
        } else if (anim.field_id == PANEL_FIELD_POSITION_Y) {
            panel_deltas[delta_index].start_position.y = panels[panel_index].position.y;
        }
    }

    if (new_elapsed < delay) {
        return;
    }

    let current_value = read_panel_field(panel_index, anim.field_id);
    var start_value = 0.0;
    if (anim.is_offset != 0u) {
        start_value = current_value;
    } else {
        start_value = animations[idx].start_value;
    };
    var target_value = animations[idx].target_value;
    if (anim.is_offset != 0u) {
        target_value = start_value + target_value;
    }
    let field_id = animations[idx].field_id;
    let easing = animations[idx].easy_fn;

    let raw_t = (new_elapsed - delay) / duration;
    let safe_t = select(raw_t, 0.0, raw_t != raw_t);
    let t = clamp(safe_t, 0.0, 1.0);
    let eased = apply_easing(easing, t);
    let value = mix(start_value, target_value, eased);
    let delta = value - current_value;
    let slot = global_uniform.frame & 31u; // frame % 32
    if (field_id == PANEL_FIELD_POSITION_X || field_id == PANEL_FIELD_POSITION_Y) {
        write_delta(delta_index, field_id, delta);
    } else {
        write_panel_field(panel_index, field_id, value);
        write_delta(delta_index, field_id, delta);
    }

    if (new_elapsed >= delay + duration) {
        animations[idx].death = 1u;
    }
}
