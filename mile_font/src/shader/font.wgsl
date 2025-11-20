struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) vis: f32,
    @location(2) color: vec4<f32>
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
    pos_px: vec2<f32>,
    // Pixel height requested by CPU; quad size comes directly from this value.
    size_px: f32,
    _pad_size: f32,
    color: vec4<f32>,
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
    let pidx = select(inst.panel_index - 1u, 0u, inst.panel_index == 0u);
    let panel = panels[pidx];
    let delta = panel_deltas[pidx];
    let container = panel.size;
    // Estimate line height in pixels from font metrics
    let units = max(f32(des.units_per_em), 1.0);
    let line_height_em = f32(des.ascent - des.descent + des.line_gap);
    let line_height_px = line_height_em / units * inst.size_px;
    // Compute wrap with strict fit: if remaining width cannot include this glyph (even by 1px), force next line.
    let local_x = inst.pos_px.x;
    let wrap_width = max(container.x, 1.0);
    let base_line = floor(local_x / wrap_width);
    let x_in_line = local_x - base_line * wrap_width;
    let overflow = (x_in_line + inst.size_px) > wrap_width;
    let line = base_line + select(0.0, 1.0, overflow);
    let wrapped_x = select(x_in_line, 0.0, overflow);
    let wrapped_y = inst.pos_px.y + line * line_height_px;
    // Visibility in container Y
    let visible = select(0.0, 1.0, wrapped_y + inst.size_px <= container.y);
    let px = panel.position + delta.delta_position + vec2<f32>(wrapped_x, wrapped_y) + position * inst.size_px;
    debug_buffer.floats[min(inst_id, 31u)] = inst.size_px;
    
    let ndc_x = px.x / screen_width * 2.0 - 1.0;
    let ndc_y = 1.0 - (px.y / screen_height) * 2.0;
    let z_norm = f32(panel.z_index) / 100.0 + self_z_index;
    let z = 0.99 - z_norm;
    out.position = vec4<f32>(ndc_x, ndc_y, z, 1.0);
    out.vis = visible;
    out.color = inst.color;
    return out;
}

struct FragmentInput {
    @location(0) tex_coords: vec2<f32>,
};

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


    // 调试：可视化SDF值
    // return vec4<f32>(sdf_value, sdf_value, sdf_value, 1.0);
    
    // 正确的逻辑：
    // sdf_value 接近 1.0 -> 字体内部 -> 不透明
    // sdf_value 接近 0.0 -> 字体外部 -> 透明  
    // sdf_value 在 0.5 附近 -> 边缘 -> 平滑过渡
    
    let sdf_value = textureSample(font_distance_texture, font_sampler, glyph_uv).r;

    // 更锐利的边缘
    let edge_width = 0.5; // 更小的边缘宽度
    let alpha = smoothstep(0.5 - edge_width, 0.5 + edge_width, sdf_value);

    // 或者使用阶梯函数获得完全锐利的边缘
    //let alpha = select(0.0, 1.0, sdf_value > 0.5);
    

    return vec4<f32>(in.color.x,in.color.y,in.color.z,alpha);
}