struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) vis: f32,
    @location(2) color: vec4<f32>,
};

struct GlobalUniform {
      // === block 1: atomic z/layouts ===
    click_layout_z: u32,
    click_layout_id: u32,
    hover_layout_id: u32,
    hover_layout_z: u32, // 16 bytes

    // === block 2: atomic drag ===
    drag_layout_id: u32,
    drag_layout_z: u32,
    pad_atomic1: u32,
    pad_atomic2: u32,    // 16 bytes

    // === block 3: dt ===
    dt: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,                   // 16 bytes

    // === block 4: mouse ===
    mouse_pos: vec2<f32>,
    mouse_state: u32,
    frame: u32,                   // 16 bytes

    // === block 5: screen info ===
    screen_size: vec2<u32>,
    press_duration: f32,
    time: f32,                    // 16 bytes

    // === block 6: event points ===
    event_point: vec2<f32>,
    extra1: vec2<f32>,            // 16 bytes

    // === block 7: extra data ===
    extra2: vec2<f32>,
    pad_extra: vec2<f32>         // 16 bytes
};


struct FontGlyphDes {
    start_idx: u32,
    end_idx: u32,
    texture_idx_x: u32,
    texture_idx_y: u32,
    
    // 当前已有的边界框
    x_min: i32,
    y_min: i32, 
    x_max: i32,
    y_max: i32,
    
    // 需要新增的关键度量字段
    units_per_em: u32,        // 每个em的字体单位数[citation:8]
    ascent: i32,              // 从基线到顶部的距离[citation:9]
    descent: i32,             // 从基线到底部的距离（通常为负值）[citation:9]
    line_gap: i32,            // 行间距[citation:9]
    advance_width: u32,       // 字形的总前进宽度[citation:9]
    left_side_bearing: i32,   // 从原点到位图左边的距离[citation:9]
    
    // 字形特定的度量
    glyph_advance_width: u32, // 特定字形的前进宽度
    glyph_left_side_bearing: i32, // 特定字形的左侧支撑
};



struct Instance {
    char_index: u32,
    text_index: u32,
    self_index: u32,
    panel_index: u32,
    origin_cursor: vec4<f32>,
    size_px: f32,
    line_height_px: f32,
    advance_px: f32,
    line_break_acc: u32,
    color: vec4<f32>,
    flags: u32,
    _pad: array<u32, 3>,
};

@group(0) @binding(4)
var<storage, read> instances: array<Instance>;

// Optional link to UI panels/deltas. If not provided, runtime binds 1-element zero buffers.
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
    rotation: vec4<f32>,
    scale: vec4<f32>,
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
    container_origin: vec2<f32>,
};

@group(0) @binding(5)
var<storage, read> panels: array<Panel>;

@group(0) @binding(6)
var<storage, read> panel_deltas: array<PanelAnimDelta>;

struct GpuUiDebugReadCallBack {
    floats: array<f32, 32>,
    uints: array<u32, 32>,
};

@group(0) @binding(7)
var<storage, read_write> debug_buffer: GpuUiDebugReadCallBack;

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @builtin(instance_index) inst_id: u32
) -> VertexOutput {
    let screen_width = f32(global_uniform.screen_size.x);
    let screen_height = f32(global_uniform.screen_size.y);
    // pixel -> NDC conversion uses actual screen size
    // 通过实例索引选择 glyph，并将 tile 偏移叠加到 uv 上
    let inst = instances[inst_id];
    let des = glyph_descs[inst.char_index];
    let index = inst.self_index;
    let self_z_index = smoothstep(0,1024,f32(index));
    let tile_scale = vec2<f32>(f32(GLYPH_SIZE) / ATLAS_SIZE.x, f32(GLYPH_SIZE) / ATLAS_SIZE.y);
    let tile_origin = vec2<f32>(
        f32(des.texture_idx_x) * tile_scale.x,
        f32(des.texture_idx_y) * tile_scale.y
    );
    var out: VertexOutput;
    out.uv = tile_origin + uv * tile_scale;

    // Pixel-accurate layout with wrapping by panel.size:
    // - Wrap X when exceeding panel.size.x
    // - Drop rendering when exceeding panel.size.y
    // UI buffers index by (panel_id - 1); our instance.panel_index carries PanelId value.
    let pidx = inst.panel_index - 1u;
    let panel = panels[pidx];
    let delta = panel_deltas[pidx];
    let container = panel.size;
    // CPU provides baseline origin + cursor offset; fall back to glyph metrics if runtime data is invalid.
    let origin = inst.origin_cursor.xy;
    let cursor_x = inst.origin_cursor.z;
    var line_height_px = inst.line_height_px;
    if (line_height_px <= 0.0) {
        let units = max(f32(des.units_per_em), 1.0);
        let line_height_em = f32(des.ascent - des.descent + des.line_gap);
        line_height_px = line_height_em / units * inst.size_px;
    }
    debug_buffer.floats[pidx] = line_height_px;
    let wrap_width = max(container.x, 1.0);
    let base_line = floor(cursor_x / wrap_width);
    let x_in_line = cursor_x - base_line * wrap_width;
    let overflow = (x_in_line + inst.advance_px) > wrap_width;
    var line = f32(inst.line_break_acc) + base_line + select(0.0, 1.0, overflow);
    var wrapped_x = select(x_in_line, 0.0, overflow);

    // Accumulate leftover width when previous glyphs overflowed but still belong to this line.
    var carry = 0.0;
    var scan_idx = inst_id;
    var scans: u32 = 0u;
    loop {
        if (scan_idx == 0u || scans >= 64u) {
            break;
        }
        scan_idx -= 1u;
        scans += 1u;
        let prev = instances[scan_idx];
        if (prev.text_index != inst.text_index) {
            break;
        }
        let prev_cursor = prev.origin_cursor.z;
        let prev_base_line = floor(prev_cursor / wrap_width);
        let prev_x_in_line = prev_cursor - prev_base_line * wrap_width;
        let prev_overflow = (prev_x_in_line + prev.advance_px) > wrap_width;
        var prev_line = f32(prev.line_break_acc) + prev_base_line + select(0.0, 1.0, prev_overflow);
        if (prev_line != line) {
            break;
        }
        if (prev_overflow) {
            carry += wrap_width - prev_x_in_line;
        }
    }
    wrapped_x += carry;
    let local_y = origin.y + line * line_height_px;
    let wrapped_y = local_y;
    let wrapped_x_with_origin = origin.x + wrapped_x;
    // Visibility in container Y
    let visible = select(0.0, 1.0, wrapped_y + inst.size_px <= container.y);
    let px = panel.position + delta.delta_position + vec2<f32>(wrapped_x_with_origin, wrapped_y) + position * inst.size_px;
    debug_buffer.floats[min(inst_id, 31u)] = cursor_x;
    
    let ndc_x = px.x / screen_width * 2.0 - 1.0;
    let ndc_y = 1.0 - (px.y / screen_height) * 2.0;
    let z_norm = f32(panel.z_index) / 100.0 + self_z_index;
    let z = 0.99 - z_norm;
    out.position = vec4<f32>(ndc_x, ndc_y, z, 1.0);
    out.vis = visible;
    out.color = inst.color;
    return out;
}

@group(0) @binding(0)
var font_distance_texture: texture_2d<f32>;

@group(0) @binding(1)
var font_sampler: sampler;

@group(0) @binding(2)
var<storage, read> glyph_descs: array<FontGlyphDes>;

@group(0) @binding(3) var<storage, read> global_uniform: GlobalUniform;

const GLYPH_SIZE: u32 = 64u;
const ATLAS_SIZE: vec2<f32> = vec2<f32>(4096.0, 4096.0);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.vis < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let pixel_offset = vec2<f32>(0.5) / ATLAS_SIZE;
    let glyph_uv = in.uv + pixel_offset;
    let sdf_value = textureSample(font_distance_texture, font_sampler, glyph_uv).r;
    let edge_width = 0.5;
    let coverage = smoothstep(0.5 - edge_width, 0.5 + edge_width, sdf_value);
    let tint = in.color;
    let final_alpha = tint.a * coverage;
    let rgb = tint.rgb * coverage;
    return vec4<f32>(rgb, final_alpha);
}
