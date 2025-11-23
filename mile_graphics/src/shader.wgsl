// shader.wgsl - 交互式霓虹数据雾背景

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct GlobalUniform {
    click_layout_z: u32,
    click_layout_id: u32,
    hover_layout_id: u32,
    hover_layout_z: u32,
    drag_layout_id: u32,
    drag_layout_z: u32,
    pad_atomic1: u32,
    pad_atomic2: u32,
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

@group(0) @binding(0)
var<storage, read> global_uniform: GlobalUniform;

const PI: f32 = 3.14159265;

fn hash21(p: vec2<f32>) -> f32 {
    let h = sin(dot(p, vec2<f32>(127.1, 311.7)));
    return fract(h * 43758.5453123);
}

fn neon_palette(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.5, 0.2, 0.7);
    let b = vec3<f32>(0.4, 0.3, 0.4);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.0, 0.3, 0.7);
    return a + b * cos(2.0 * PI * (c * t + d));
}

fn rotate2(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vi], 0.99, 1.0);
    out.uv = (pos[vi] + vec2<f32>(1.0, 1.0)) * 0.5;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let screen = vec2<f32>(
        max(1.0, f32(global_uniform.screen_size.x)),
        max(1.0, f32(global_uniform.screen_size.y))
    );
    let aspect = screen.x / screen.y;
    let uv = input.uv;
    let centered = vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
    let time = global_uniform.time * 0.65;

    let mouse_uv = vec2<f32>(
        global_uniform.mouse_pos.x / screen.x,
        1.0 - global_uniform.mouse_pos.y / screen.y
    );
    let mouse_press = select(0.0, 1.0, global_uniform.mouse_state != 0u);

    var color = vec3<f32>(0.03, 0.02, 0.05);
    let vertical_grad = mix(vec3<f32>(0.02, 0.02, 0.05), vec3<f32>(0.12, 0.05, 0.18), uv.y);
    color = mix(color, vertical_grad, 0.7);

    let radial = exp(-length(centered * vec2<f32>(1.0, 0.7)) * 1.8);
    color += radial * vec3<f32>(0.05, 0.01, 0.08);

    let parallax = (mouse_uv - vec2<f32>(0.5, 0.5)) * 0.12;
    let grid_uv = (uv + parallax) * vec2<f32>(4.5, 3.0);
    let grid = exp(-42.0 * min(abs(fract(grid_uv) - 0.5).x, abs(fract(grid_uv) - 0.5).y));
    color += grid * vec3<f32>(0.02, 0.18, 0.36);

    let rotated = rotate2(centered, 0.2);
    let layer_uv = rotated * 3.0 + vec2<f32>(time * 0.2, -time * 0.15);
    let tile = floor(layer_uv);
    let spark = hash21(tile);
    let spark_trail = smoothstep(0.97, 1.0, spark) * exp(-30.0 * abs(fract(layer_uv.y) - 0.5));
    color += spark_trail * vec3<f32>(0.08, 0.35, 0.75);

    let scan = smoothstep(0.80, 1.0, 1.0 - abs(fract(uv.y * 1.6 - time * 0.25) - 0.5));
    color += scan * vec3<f32>(0.3, 0.1, 0.5) * 0.12;

    let diag = abs(fract((uv.x + uv.y) * 2.3 - time * 0.35) - 0.5);
    let glitch = smoothstep(0.48, 0.5, diag);
    color += glitch * vec3<f32>(0.08, 0.05, 0.15);

    let mouse_dir = uv - mouse_uv;
    let ripple_dist = length(mouse_dir);
    let ripple = exp(-ripple_dist * 20.0) *
        (0.6 + 0.4 * sin((ripple_dist - time * 0.6) * 45.0));
    let ripple_color = mix(vec3<f32>(0.05, 0.4, 0.8), vec3<f32>(0.9, 0.2, 0.8), mouse_press);
    color += ripple * ripple_color * (0.7 + mouse_press * 0.5);

    let pointer = clamp(
        1.0 - length(vec2<f32>(mouse_dir.x * aspect, mouse_dir.y) * 4.0),
        0.0,
        1.0
    );
    color += pointer * vec3<f32>(0.2, 0.5, 0.9) * 0.3;

    let node_scale = vec2<f32>(12.0, 6.0);
    let node_uv = floor((uv + parallax * 2.0) * node_scale);
    let flicker = pow(hash21(node_uv + floor(time * 2.0)), 12.0);
    color += flicker * neon_palette(hash21(node_uv * 1.7)) * 0.25;

    let grain = hash21(uv * screen);
    color += (grain - 0.5) * 0.008;

    let vignette = smoothstep(0.95, 0.35, length(centered * vec2<f32>(1.0, 1.4)));
    color *= vignette;

    color = clamp(color, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
    return vec4<f32>(color, 1.0);
}
