// Glyph layout compute shader: assigns per-glyph panel positions.

const GPU_CHAR_LAYOUT_FLAG_LINE_BREAK_BEFORE: u32 = 0x1u;
const GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_SHIFT: u32 = 8u;
const GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_MASK: u32 = 0x00ffff00u;

struct FontGlyphDes {
    start_idx: u32,
    end_idx: u32,
    texture_idx_x: u32,
    texture_idx_y: u32,
    x_min: i32,
    y_min: i32,
    x_max: i32,
    y_max: i32,
    units_per_em: u32,
    ascent: i32,
    descent: i32,
    line_gap: i32,
    advance_width: u32,
    left_side_bearing: i32,
    glyph_advance_width: u32,
    glyph_left_side_bearing: i32,
};

struct GpuChar {
    char_index: u32,
    gpu_text_index: u32,
    panel_index: u32,
    self_index: u32,
    glyph_advance_width: u32,
    glyph_left_side_bearing: i32,
    glyph_ver_advance: u32,
    glyph_ver_side_bearing: i32,
    layout_flags: u32,
};

struct FontTextSpan {
    glyph_start: u32,
    glyph_end: u32,
    font_size: f32,
    line_height: f32,
    first_line_indent: f32,
    padding: f32,
    panel_index: u32,
    text_align: u32,
    color: vec4<f32>,
    origin: vec2<f32>,
};

struct Panel {
    position: vec2<f32>,
    size: vec2<f32>,
    uv_offset: vec2<f32>,
    uv_scale: vec2<f32>,
    z_index: u32,
    interaction_passthrough: u32,
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
    rotation: vec4<f32>,
    scale: vec4<f32>,
    color: vec4<f32>,
    border_color: vec4<f32>,
    border_width: f32,
    border_radius: f32,
    visible: u32,
    _pad_border: u32,
}

struct PanelAnimDelta {
    delta_position: vec2<f32>,
    delta_size: vec2<f32>,
    delta_uv_offset: vec2<f32>,
    delta_uv_scale: vec2<f32>,
    delta_z_index: i32,
    delta_interaction_passthrough: i32,
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

struct GlyphInstance {
    char_index: u32,
    panel_index: u32,
    text_index: u32,
    self_index: u32,
    panel_origin: vec2<f32>,
    quad_size: vec2<f32>,
    color: vec4<f32>,
    font_size: f32,
    visibility: f32,
    _pad: vec2<f32>,
};

struct LayoutParams {
    glyph_count: u32,
    text_count: u32,
    padding: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> glyph_descs: array<FontGlyphDes>;
@group(0) @binding(1) var<storage, read> gpu_chars: array<GpuChar>;
@group(0) @binding(2) var<storage, read> text_spans: array<FontTextSpan>;
@group(0) @binding(3) var<storage, read> panels: array<Panel>;
@group(0) @binding(4) var<storage, read> panel_deltas: array<PanelAnimDelta>;
@group(0) @binding(5) var<storage, read_write> instances: array<GlyphInstance>;
@group(0) @binding(6) var<uniform> params: LayoutParams;

fn decode_line_break(flags: u32) -> u32 {
    if ((flags & GPU_CHAR_LAYOUT_FLAG_LINE_BREAK_BEFORE) == 0u) {
        return 0u;
    }
    return (flags & GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_MASK) >> GPU_CHAR_LAYOUT_LINE_BREAK_COUNT_SHIFT;
}

fn glyph_line_height(desc: FontGlyphDes, font_size: f32) -> f32 {
    let units = max(f32(desc.units_per_em), 1.0);
    let line_height_em = f32(desc.ascent - desc.descent + desc.line_gap);
    return line_height_em / units * font_size;
}

fn glyph_advance(desc: FontGlyphDes, font_size: f32) -> f32 {
    let units = max(f32(desc.units_per_em), 1.0);
    return f32(desc.glyph_advance_width) / units * font_size;
}

fn line_start_indent(line_index: u32, indent: f32) -> f32 {
    if (line_index == 0u) {
        return indent;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let glyph_idx = id.x;
    if (glyph_idx >= params.glyph_count) {
        return;
    }
    let glyph = gpu_chars[glyph_idx];
    let text_idx = glyph.gpu_text_index;
    if (text_idx >= params.text_count) {
        instances[glyph_idx].visibility = 0.0;
        return;
    }
    let span = text_spans[text_idx];
    if (span.panel_index == 0u) {
        instances[glyph_idx].visibility = 0.0;
        return;
    }
    let panel_index = span.panel_index - 1u;
    let panel = panels[panel_index];
    let delta = panel_deltas[panel_index];
    let container = panel.size + delta.delta_size;

    let wrap_padding = span.padding;
    let wrap_width = max(container.x - wrap_padding * 2.0, 1.0);
    let target_start = span.glyph_start + glyph.self_index;
    var line_index: u32 = 0u;
    var pen_x = line_start_indent(line_index, span.first_line_indent);

    var idx = span.glyph_start;
    loop {
        if (idx >= target_start) {
            break;
        }
        let prev = gpu_chars[idx];
        let prev_desc = glyph_descs[prev.char_index];
        let break_count = decode_line_break(prev.layout_flags);
        if (break_count > 0u) {
            line_index = line_index + break_count;
            pen_x = line_start_indent(line_index, span.first_line_indent);
        }
        let prev_adv = glyph_advance(prev_desc, span.font_size);
        let needed = pen_x + prev_adv;
        if (needed >= wrap_width) {
            line_index = line_index + 1u;
            let base = line_start_indent(line_index, span.first_line_indent);
            pen_x = base + prev_adv;
        } else {
            pen_x = needed;
        }
        idx = idx + 1u;
    }

    var current_pen = pen_x;
    let target_desc = glyph_descs[glyph.char_index];
    let glyph_break = decode_line_break(glyph.layout_flags);
    if (glyph_break > 0u) {
        line_index = line_index + glyph_break;
        current_pen = line_start_indent(line_index, span.first_line_indent);
    }
    let glyph_adv = glyph_advance(target_desc, span.font_size);
    if (current_pen + glyph_adv >= wrap_width) {
        line_index = line_index + 1u;
        current_pen = line_start_indent(line_index, span.first_line_indent);
    }

    var line_height_px = span.line_height;
    if (line_height_px <= 0.0) {
        line_height_px = glyph_line_height(target_desc, span.font_size);
    }
    if (!(line_height_px > 0.0)) {
        line_height_px = span.font_size;
    }

    let units = max(f32(target_desc.units_per_em), 1.0);
    let glyph_left_px =
        f32(target_desc.glyph_left_side_bearing) / units * span.font_size;
    let glyph_width_units = max(f32(target_desc.x_max - target_desc.x_min), 1.0);
    let glyph_height_units = max(f32(target_desc.y_max - target_desc.y_min), 1.0);
    let glyph_width_px = glyph_width_units / units * span.font_size;
    let glyph_height_px_raw = glyph_height_units / units * span.font_size;
    let glyph_min_height_px = max(span.font_size * 0.12, 1.5);
    let glyph_height_px = max(glyph_height_px_raw, glyph_min_height_px);
    let glyph_height_pad = (glyph_height_px - glyph_height_px_raw) * 0.5;
    let glyph_top_px = f32(target_desc.y_max) / units * span.font_size;

    let local_y =
        span.origin.y + wrap_padding + f32(line_index) * line_height_px;
    let wrapped_x = span.origin.x + wrap_padding + current_pen;
    let panel_origin = vec2<f32>(
        wrapped_x + glyph_left_px,
        local_y - glyph_top_px + span.font_size - glyph_height_pad,
    );

    let visible = select(
        0.0,
        1.0,
        local_y + span.font_size <= container.y - wrap_padding,
    );

    instances[glyph_idx] = GlyphInstance(
        glyph.char_index,
        span.panel_index,
        text_idx,
        glyph.self_index,
        panel_origin,
        vec2<f32>(glyph_adv, glyph_height_px),
        span.color,
        span.font_size,
        visible,
        vec2<f32>(0.0, 0.0),
    );
}
