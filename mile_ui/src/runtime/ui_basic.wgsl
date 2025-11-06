struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,

    @location(2) instance_pos: vec2<f32>,
    @location(3) instance_size: vec2<f32>,
    @location(4) uv_offset: vec2<f32>,
    @location(5) uv_scale: vec2<f32>,

    @location(6) z_index: u32,
    @location(7) pass_through: u32,
    @location(8) instance_id: u32,
    @location(9) interaction: u32,

    @location(10) event_mask: u32,
    @location(11) state_mask: u32,
    @location(12) transparent: f32,
    @location(13) texture_slot: u32,

    @location(14) state: u32,
    @location(15) pad0: u32,
    @location(16) pad1: u32,
    @location(17) pad2: u32,
    @location(18) tint: vec4<f32>,
    @location(19) border_color: vec4<f32>,
    @location(20) border: vec2<f32>,
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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) texture_slot: u32,
    @location(2) alpha: f32,
    @location(3) tint: vec4<f32>,
    @location(4) border_color: vec4<f32>,
    @location(5) border: vec2<f32>,
    @location(6) local_pos: vec2<f32>,
    @location(7) instance_size: vec2<f32>,
};

struct GpuUiDebugReadCallBack {
    floats: array<f32, 32>,
    uints: array<u32, 32>,
};

@group(0) @binding(1) var<storage, read_write> debug_buffer: GpuUiDebugReadCallBack;


@group(0) @binding(0)
var<storage, read> global_uniform: GlobalUniform;

@group(1) @binding(0)
var ui_textures: binding_array<texture_2d<f32>>;

@group(1) @binding(1)
var ui_samplers: binding_array<sampler>;

fn to_clip_space(position: vec2<f32>) -> vec4<f32> {
    let screen = vec2<f32>(
        f32(global_uniform.screen_size.x),
        f32(global_uniform.screen_size.y)
    );
    let ndc = vec2<f32>(
        position.x / screen.x * 2.0 - 1.0,
        1.0 - position.y / screen.y * 2.0,
    );
    return vec4<f32>(ndc, 0.0, 1.0);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let quad_pos = input.instance_pos + input.pos * input.instance_size;
    out.clip_position = to_clip_space(quad_pos);
    out.clip_position.z = f32(0.0);
    out.uv = input.uv_offset + input.uv * input.uv_scale;
    out.texture_slot = input.texture_slot;
    out.alpha = input.transparent;
    out.tint = input.tint;
    out.border_color = input.border_color;
    out.border = input.border;
    out.local_pos = input.pos;
    out.instance_size = input.instance_size;
    return out;
}

fn mix_colors(base: vec4<f32>, overlay: vec4<f32>, mask: f32) -> vec4<f32> {
    let blended_rgb = mix(base.rgb, overlay.rgb, mask * overlay.a);
    let blended_a = mix(base.a, overlay.a, mask * overlay.a);
    return vec4<f32>(blended_rgb, blended_a);
}

fn rounded_rect_sdf(p: vec2<f32>, half_extents: vec2<f32>, radius: f32) -> f32 {
    let r = min(radius, min(half_extents.x, half_extents.y));
    let inner = max(half_extents - vec2<f32>(r, r), vec2<f32>(0.0));
    let q = abs(p) - inner;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let slot = input.texture_slot;
    var color = vec4<f32>(input.tint.rgb, input.tint.a);
    var alpha = color.a * input.alpha;

    debug_buffer.floats[0] = global_uniform.mouse_pos.x;
    debug_buffer.floats[1] = global_uniform.mouse_pos.y;
    debug_buffer.floats[2] = 99999.0;

    if (slot != 0xffffffffu) {
        let sampled = textureSample(ui_textures[slot], ui_samplers[slot], input.uv);
        color = vec4<f32>(sampled.rgb * input.tint.rgb, sampled.a * input.tint.a);
        alpha = color.a * input.alpha;
    }

    let border_width = input.border.x;
    if (border_width > 0.0) {
        let size_px = input.instance_size;
        let half = size_px * 0.5;
        let radius = input.border.y;
        let centered = (input.local_pos - vec2<f32>(0.5, 0.5)) * size_px;
        let sdf = rounded_rect_sdf(centered, half, radius);

        if (sdf > 0.0) {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }

        if (sdf >= -border_width) {
            color = input.border_color;
            alpha = input.border_color.a * input.alpha;
        }
    }

    return vec4<f32>(color.rgb, alpha);
}

