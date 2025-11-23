struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) vis: f32,
    @location(2) color: vec4<f32>,
    @location(3) font_size:f32
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
    first_line_indent: f32,
    text_align: u32,
    flags: u32,
    _pad: array<u32, 1>,
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
    let padding = 5.0;
    let wrap_width = max(container.x - padding * 2.0, 1.0);
    let units = max(f32(des.units_per_em), 1.0);
    let glyph_width_px = f32(des.x_max - des.x_min) / units * inst.size_px;
    let glyph_advance_px = f32(des.glyph_advance_width) / units * inst.size_px;
    let layout_width_px = select(
        max(glyph_advance_px, glyph_width_px),
        inst.advance_px,
        inst.advance_px > 0.0,
    );
    let glyph_left_units = select(
        des.left_side_bearing,
        des.glyph_left_side_bearing,
        des.glyph_left_side_bearing != 0,
    );
    let glyph_left_px = f32(glyph_left_units) / units * inst.size_px;
    let base_line = u32(cursor_x / wrap_width);
    let x_in_line = cursor_x - f32(base_line) * wrap_width;
    let overflow = (x_in_line + layout_width_px) >= wrap_width;
    let line_index: u32 =
        inst.line_break_acc + base_line + select(0u, 1u, overflow);
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
        let prev_des = glyph_descs[prev.char_index];
        let prev_units = max(f32(prev_des.units_per_em), 1.0);
        let prev_glyph_width_px = f32(prev_des.x_max - prev_des.x_min) / prev_units * prev.size_px;
        let prev_glyph_advance_px = f32(prev_des.glyph_advance_width) / prev_units * prev.size_px;
        let prev_layout_width_px = select(
            max(prev_glyph_advance_px, prev_glyph_width_px),
            prev.advance_px,
            prev.advance_px > 0.0,
        );
        let prev_base_line = u32(prev_cursor / wrap_width);
        let prev_x_in_line = prev_cursor - f32(prev_base_line) * wrap_width;
        let prev_overflow = (prev_x_in_line + prev_layout_width_px) >= wrap_width;
        let prev_line = prev.line_break_acc + prev_base_line + select(0u, 1u, prev_overflow);
        if (prev_line != line_index) {
            break;
        }
        if (prev_overflow) {
            carry += wrap_width - prev_x_in_line;
        }
    }
    wrapped_x += carry;
    // Clamp to the padded wrap width so long lines do not spill beyond the right edge.
    wrapped_x = min(wrapped_x, max(wrap_width - layout_width_px, 0.0));
    let local_y = origin.y + padding + f32(line_index) * line_height_px;
    let wrapped_y = local_y;
    let wrapped_x_with_origin = origin.x + padding + wrapped_x;
    // Visibility in container Y
    let visible = select(0.0, 1.0, wrapped_y + inst.size_px <= container.y - padding);
    let px = panel.position + delta.delta_position + vec2<f32>(wrapped_x_with_origin + glyph_left_px, wrapped_y) + position * inst.size_px;
    debug_buffer.floats[min(inst_id, 31u)] = cursor_x;
    
    let ndc_x = px.x / screen_width * 2.0 - 1.0;
    let ndc_y = 1.0 - (px.y / screen_height) * 2.0;
    let z_norm = f32(panel.z_index) / 100.0 + self_z_index;
    let z = 0.99 - z_norm;
    out.position = vec4<f32>(ndc_x, ndc_y, z, 1.0);
    out.vis = visible;
    out.color = inst.color;
    out.font_size = inst.size_px;
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

fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn font_size_normalized(size_px: f32) -> f32 {
    // Map roughly 12px..76px into 0..1. Values outside the range are clamped.
    return saturate((size_px - 12.0) / 78.0);
}

fn adaptive_edge_width(size_px: f32, px_range: f32) -> vec2<f32> {
    // 返回 (thin, wide) 两个宽度：小字号依赖 thin 保证亮度，大字号更多使用 wide 保留平滑。
    let norm = font_size_normalized(size_px);
    let thin_scale = mix(0.35, 0.6, norm);
    let thin_bias = mix(0.0006, 0.00025, norm);
    let wide_scale = mix(0.8, 1.25, norm);
    let wide_bias = mix(0.001, 0.00045, norm);
    let thin = max(px_range * thin_scale + thin_bias, 3e-5);
    let wide = max(px_range * wide_scale + wide_bias, thin * 1.2);
    return vec2<f32>(thin, wide);
}

fn adaptive_gamma(size_px: f32) -> f32 {
    // 小字号需要更亮的边缘，大字号保持锐利。
    let size_blend = font_size_normalized(size_px);
    return mix(0.3, 1.2, size_blend) + 0.15;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.vis < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let pixel_offset = vec2<f32>(0.5) / ATLAS_SIZE;
    let glyph_uv = in.uv + pixel_offset;
    let sdf_value = textureSample(font_distance_texture, font_sampler, glyph_uv).r;
    // 基于屏幕像素导数和梯度计算自适应边缘宽度。
    let dp = vec2<f32>(dpdx(sdf_value), dpdy(sdf_value));
    let grad = length(dp);
    let px_range = max(fwidth(sdf_value) / max(grad, 1e-3), 1e-4);
    let edge_width = adaptive_edge_width(in.font_size, px_range);
    let widths = adaptive_edge_width(in.font_size, px_range);
    let thin = widths.x;
    let wide = widths.y;
    let sharp_coverage = smoothstep(0.5 - thin, 0.5 + thin, sdf_value);
    let soft_coverage = smoothstep(0.5 - wide, 0.5 + wide, sdf_value);
    // Blend 两种 coverage：小字号靠 sharp，越大越接近 soft。
    let size_norm = font_size_normalized(in.font_size);
    let coverage = mix(max(sharp_coverage, soft_coverage * 0.9), soft_coverage, size_norm);
    // 对 coverage 做 gamma 调整，并对大字号稍微增强边缘对比。
    let gamma = adaptive_gamma(in.font_size);
    let shaped = pow(max(coverage, 1e-4), gamma);
    let fringe = shaped * (1.0 - shaped);
    let edge_boost = mix(0.2, 0.45, size_norm);
    let boosted = saturate(shaped + fringe * edge_boost);
    // 为小字号额外抬升内部亮度。
    let interior = smoothstep(0.52 + thin, 0.7 + thin, sdf_value);
    let smallness = smoothstep(0.0, 0.5, 0.5 - size_norm);
    let interior_boost = interior * smallness * 0.5;
    let final_coverage = saturate(boosted + interior_boost);
    let tint = in.color;
    let dist_from_edge = abs(sdf_value - 0.5);
    let edge_region = 1.0 - smoothstep(thin * 2.0, thin * 6.0, dist_from_edge);
    let whiteness = pow(1.0 - size_norm, 2.5) * edge_region * 0.85;
    let edge_tint = mix(tint.rgb, vec3<f32>(1.0, 1.0, 1.0), whiteness);
    let final_alpha = tint.a * final_coverage;
    let rgb = edge_tint * final_coverage;
    return vec4<f32>(rgb, final_alpha);
}
