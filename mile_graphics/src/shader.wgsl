// shader.wgsl - 鼠标交互的辉光分形背景

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// 与 mile_api::interface::GlobalUniform 对齐的布局
struct GlobalUniform {
    // === block 1 ===
    click_layout_z: u32,
    click_layout_id: u32,
    hover_layout_id: u32,
    hover_layout_z: u32,
    // === block 2 ===
    drag_layout_id: u32,
    drag_layout_z: u32,
    pad_atomic1: u32,
    pad_atomic2: u32,
    // === block 3 ===
    dt: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
    // === block 4 ===
    mouse_pos: vec2<f32>,
    mouse_state: u32,
    frame: u32,
    // === block 5 ===
    screen_size: vec2<u32>,
    press_duration: f32,
    time: f32,
    // === block 6/7 ===
    event_point: vec2<f32>,
    extra1: vec2<f32>,
    extra2: vec2<f32>,
    pad_extra: vec2<f32>,
};

@group(0) @binding(0)
var<storage, read> global_uniform: GlobalUniform;

// HSV -> RGB（用于平滑调色板）
fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // 全屏三角形（不依赖顶点缓冲）
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vi], 0.999999, 1.0);
    out.uv = (out.position.xy + vec2<f32>(1.0)) * 0.5;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // 屏幕、时间、鼠标
    let screen = vec2<f32>(f32(global_uniform.screen_size.x), f32(global_uniform.screen_size.y));
    let sx = max(screen.x, 1.0);
    let sy = max(screen.y, 1.0);
    let aspect = sx / sy;
    let t = global_uniform.time;

    // 归一化坐标（保持比例）
    var uv = input.uv * 2.0 - vec2<f32>(1.0, 1.0);
    uv.x *= aspect;

    // 鼠标（转 uv 空间）
    let mouse_ndc = vec2<f32>(
        global_uniform.mouse_pos.x / sx * 2.0 - 1.0,
        1.0 - global_uniform.mouse_pos.y / sy * 2.0
    );
    let mouse_uv = vec2<f32>(mouse_ndc.x * aspect, mouse_ndc.y);

    // 深色基底（海洋）
    var color = vec3<f32>(0.01, 0.02, 0.035);

    // 多重正弦海浪（时间驱动）
    let w1 = sin(uv.x * 6.0 + t * 1.2);
    let w2 = sin(uv.y * 8.0 - t * 1.0);
    let w3 = sin((uv.x * 1.2 + uv.y * 1.1) * 5.0 + t * 0.8);
    var wave = (w1 * 0.5 + w2 * 0.5 + w3 * 0.5) / 1.5;

    // 鼠标涟漪
    let d = length(uv - mouse_uv);
    let ripple = sin(12.0 * d - t * 4.0) * exp(-d * 3.0);
    wave += ripple * 0.6;

    // 波峰（泡沫/霓虹高光）
    let crest = smoothstep(0.55, 0.9, (wave + 1.0) * 0.5);
    let neon = hsv2rgb(vec3<f32>(fract(t * 0.05 + crest * 0.15), 0.9, 1.0));
    color += neon * crest * 0.85;

    // 鼠标高亮（霓虹）
    let mouse_glow = pow(max(0.0, 0.5 - d), 2.0);
    let mouse_neon = hsv2rgb(vec3<f32>(fract(0.6 + t * 0.1), 0.95, 1.0));
    color += mouse_neon * mouse_glow * 0.9;

    // 微弱的波谷冷色
    let trough = smoothstep(0.0, 0.3, (wave + 1.0) * 0.5);
    color += vec3<f32>(0.05, 0.08, 0.12) * (1.0 - trough) * 0.2;

    // 暗角
    let vignette = smoothstep(1.4, 0.2, length(uv));
    color *= vignette;

    return vec4<f32>(color, 1.0);
}
