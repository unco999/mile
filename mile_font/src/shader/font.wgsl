struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
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

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>
) -> VertexOutput {
    let screen_width = f32(global_uniform.screen_size.x);
    let screen_height = f32(global_uniform.screen_size.y);
    let aspect_ratio = f32(screen_width / screen_height);

    var out: VertexOutput;
    // 简单正交投影，x,y 范围 [-1,1]
    let scale: vec2<f32> = vec2<f32>(0.5, 0.5 * aspect_ratio); // 缩放到屏幕一半
    out.position = vec4<f32>(position.x,position.y * aspect_ratio, 0.0, 1.0);
    out.uv = uv;
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

@group(0) @binding(3) var<storage, read_write> global_uniform: GlobalUniform;

const GLYPH_SIZE: u32 = 64u;
const ATLAS_SIZE: vec2<f32> = vec2<f32>(4096.0, 4096.0); 
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let glyph_start_uv = vec2<f32>(0.0, 0.0);
    let glyph_end_uv = vec2<f32>(
        f32(GLYPH_SIZE) / ATLAS_SIZE.x,
        f32(GLYPH_SIZE) / ATLAS_SIZE.y
    );

    var glyph_uv = mix(glyph_start_uv, glyph_end_uv, in.uv);
    let pixel_offset = vec2<f32>(0.5) / ATLAS_SIZE;
    glyph_uv = glyph_uv + pixel_offset;


    // 调试：可视化SDF值
    // return vec4<f32>(sdf_value, sdf_value, sdf_value, 1.0);
    
    // 正确的逻辑：
    // sdf_value 接近 1.0 -> 字体内部 -> 不透明
    // sdf_value 接近 0.0 -> 字体外部 -> 透明  
    // sdf_value 在 0.5 附近 -> 边缘 -> 平滑过渡
    
    let sdf_value = textureSample(font_distance_texture, font_sampler, glyph_uv).r;

    // 更锐利的边缘
    let edge_width = 0.04; // 更小的边缘宽度
    let alpha = smoothstep(0.5 - edge_width, 0.5 + edge_width, sdf_value);

    // 或者使用阶梯函数获得完全锐利的边缘
    // let alpha = select(0.0, 1.0, sdf_value > 0.5);

    return vec4<f32>(1.0, 1.0, 1.0, alpha);
}