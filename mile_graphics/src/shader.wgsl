// shader.wgsl - 炫彩渐变 + 透明渐隐 + 光晕效果
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> time: f32; // 从主程序传入时间

// ------------------
// 顶点 shader：全屏三角形
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    out.position = vec4<f32>(pos[vertex_index], 0.9999999, 1.0);

    // UV 坐标映射到 [0,1]
    out.uv = (out.position.xy + vec2<f32>(1.0, 1.0)) * 0.5;
    return out;
}

// HSV -> RGB
fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

// 简单 2D 扰动函数
fn distort(uv: vec2<f32>, t: f32) -> vec2<f32> {
    var offset = vec2<f32>(
        sin(uv.y * 5.0 + t) * 0.03,
        cos(uv.x * 7.0 + t * 1.2) * 0.03
    );
    return uv + offset;
}


@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var uv = input.uv;
    uv = uv * 2.0 - vec2<f32>(1.0, 1.0);
    uv.x *= 1.77; // 屏幕比例
    let t = time * 0.8;

    // === 扭曲网格 ===
    uv = distort(uv, t);

    // === 网格参数 ===
    let grid_size = 0.1;
    let g = abs(fract(uv / grid_size - 0.5) - 0.5) / fwidth(uv / grid_size);
    let line = 1.0 - min(g.x, g.y); // 网格线

    // === 海洋波动 ===
    let wave = sin(uv.x * 10.0 + t) * 0.08 + cos(uv.y * 15.0 + t * 1.2) * 0.08;

    // === 光泽效果 ===
    let shine = pow(max(0.0, 0.25 - length(uv) * 0.35), 3.0);

    // === 冰冷蓝灰色渐变 ===
    let baseColor = vec3<f32>(0.2, 0.35, 0.55);   // 冰蓝底色
    let waveColor = vec3<f32>(0.4, 0.6, 0.8);    // 波纹高光
    var color = mix(baseColor, waveColor, 0.5 + 0.5 * sin(t * 1.2 + uv.x * 5.0));

    // 叠加网格线和波纹
    color += vec3<f32>(line) * 0.25;
    color += vec3<f32>(wave) * 0.15;
    color += vec3<f32>(shine) * 0.35;

    // 渐隐边缘
    let alpha = smoothstep(1.2, 0.0, length(uv));

    return vec4<f32>(color, alpha);
}