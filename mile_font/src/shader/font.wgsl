struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) vis: f32,
    @location(2) color: vec4<f32>,
    @location(3) font_size: f32,
    @location(4) quad_coord: vec2<f32>,
    @location(5) glyph_bounds: vec2<f32>,
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
    
    // 字形特定的度�?
    glyph_advance_width: u32, // 特定字形的前进宽�?
    glyph_left_side_bearing: i32, // 特定字形的左侧支�?
};



struct Instance {
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

@group(0) @binding(4)
var<storage, read> instances: array<Instance>;

// Optional link to UI panels/deltas. If not provided, runtime binds 1-element zero buffers.
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
};

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

    // 通过实例索引选择 glyph，并叠加 tile 偏移到 uv
    let inst = instances[inst_id];
    let des = glyph_descs[inst.char_index];
    let index = inst.self_index;
    let self_z_index = smoothstep(0, 1024, f32(index));

    let tile_scale = vec2<f32>(
        f32(GLYPH_SIZE) / ATLAS_SIZE.x,
        f32(GLYPH_SIZE) / ATLAS_SIZE.y
    );
    let tile_origin = vec2<f32>(
        f32(des.texture_idx_x) * tile_scale.x,
        f32(des.texture_idx_y) * tile_scale.y
    );

    var out: VertexOutput;

    let pidx = inst.panel_index - 1u;
    let panel = panels[pidx];
    let delta = panel_deltas[pidx];
    let panel_origin = panel.position + delta.delta_position;
    let quad_offset = inst.panel_origin;
    let px = panel_origin + vec2<f32>(quad_offset.x, quad_offset.y)
        + position * inst.quad_size;

    out.uv = tile_origin + uv * tile_scale;

    // 这里不再用 glyph_bounds 来裁剪，只保留 quad 内部的归一化坐标
    out.quad_coord = position;
    out.glyph_bounds = vec2<f32>(0.0, 1.0);

    debug_buffer.floats[min(inst_id, 31u)] = inst.panel_origin.x;

    let ndc_x = px.x / screen_width * 2.0 - 1.0;
    let ndc_y = 1.0 - (px.y / screen_height) * 2.0;
    let base_z = clamp(f32(panel.z_index) / 100.0, 0.0, 1.0);
    let glyph_offset = self_z_index * 0.005 + 0.001;
    let z = clamp(base_z - glyph_offset, 0.0, 0.9999);

    out.position = vec4<f32>(ndc_x, ndc_y, z, 1.0);
    out.vis = inst.visibility;
    out.color = inst.color;
    out.font_size = inst.font_size;
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
    return saturate((size_px - 12.0) / 78.0);
}

fn adaptive_edge_width(size_px: f32, px_range: f32) -> vec2<f32> {
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
    let size_blend = font_size_normalized(size_px);
    return mix(0.6, 1.2, size_blend);
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 只用 vis 控制是否丢弃，不再按 glyph_bounds 横向裁剪
    if (in.vis < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let pixel_offset = vec2<f32>(0.5) / ATLAS_SIZE;
    let glyph_uv = in.uv + pixel_offset;

    let sdf_value = textureSample(font_distance_texture, font_sampler, glyph_uv).r;
    let dp = vec2<f32>(dpdx(sdf_value), dpdy(sdf_value));
    let grad = length(dp);
    let px_range_raw = fwidth(sdf_value) / max(grad, 1e-3);
    let px_range_min = mix(0.0025, 0.0045, font_size_normalized(in.font_size));
    let px_range = max(px_range_raw, px_range_min);

    let widths = adaptive_edge_width(in.font_size, px_range);
    let thin = widths.x;
    let wide = widths.y;

    let sharp_coverage = smoothstep(0.5 - thin, 0.5 + thin, sdf_value);
    let soft_coverage = smoothstep(0.5 - wide, 0.5 + wide, sdf_value);

    let size_norm = font_size_normalized(in.font_size);
    let coverage = mix(
        max(sharp_coverage, soft_coverage * 0.9),
        soft_coverage,
        size_norm,
    );

    let gamma = adaptive_gamma(in.font_size);
    let shaped = pow(max(coverage, 1e-4), gamma);

    let fringe = shaped * (1.0 - shaped);
    let edge_boost = mix(0.2, 0.45, size_norm);
    let boosted = saturate(shaped + fringe * edge_boost);

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
