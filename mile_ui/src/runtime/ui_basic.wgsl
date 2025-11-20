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
    @location(13) texture_id: u32,

    @location(14) state: u32,
    @location(15) collection_state: u32,
    @location(16) fragment_shader_id: u32,
    @location(17) vertex_shader_id: u32,
    @location(18) rotation: vec4<f32>,
    @location(19) scale: vec4<f32>,
    @location(20) color: vec4<f32>,
    @location(21) border_color: vec4<f32>,
    @location(22) border: vec2<f32>,
    @location(23) visible: u32,
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

struct GpuUiDebugReadCallBack {
    floats: array<f32, 32>,
    uints: array<u32, 32>,
};

struct GpuUiTextureInfo {
    index: u32,
    parent_index: u32,
    _pad: vec2<u32>,
    uv_min: vec4<f32>,
    uv_max: vec4<f32>,
};

struct RenderBindingComponent {
    channel_type: u32,
    source_index: u32,
    component_index: u32,
    source_component: u32,
    factor_component: u32,
    factor_inner_compute: u32,
    factor_outer_compute: u32,
    factor_unary: u32,
    payload: vec4<f32>,
};

struct RenderExprNode {
    op: u32,
    arg0: u32,
    arg1: u32,
    arg2: u32,
    data0: f32,
    data1: f32,
    pad0: f32,
    pad1: f32,
};

struct RenderBindingLayer {
    compute_start: u32,
    compute_count: u32,
    reserved0: u32,
    reserved1: u32,
    components: array<RenderBindingComponent, 4>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) texture_id: u32,
    @location(2) transparent: f32,
    @location(3) color: vec4<f32>,
    @location(4) border_color: vec4<f32>,
    @location(5) border: vec2<f32>,
    @location(6) local_pos: vec2<f32>,
    @location(7) instance_size: vec2<f32>,
    @location(8) instance_pos: vec2<f32>,
    @location(9) fragment_shader_id: u32,
    @location(10) vertex_shader_id: u32,
};

const U32_MAX: u32 = 0xffffffffu;
const MAX_RENDER_EXPR_NODES: u32 = 64u;

const CHANNEL_CONSTANT: u32 = 0u;
const CHANNEL_COMPUTE: u32 = 1u;
const CHANNEL_RENDER_IMPORT: u32 = 2u;
const CHANNEL_RENDER_COMPOSITE: u32 = 3u;
const CHANNEL_RENDER_EXPR: u32 = 4u;

const RENDER_EXPR_OP_CONSTANT: u32 = 0u;
const RENDER_EXPR_OP_RENDER_IMPORT: u32 = 1u;
const RENDER_EXPR_OP_COMPUTE_RESULT: u32 = 2u;
const RENDER_EXPR_OP_UNARY_SIN: u32 = 3u;
const RENDER_EXPR_OP_UNARY_COS: u32 = 4u;
const RENDER_EXPR_OP_UNARY_TAN: u32 = 5u;
const RENDER_EXPR_OP_UNARY_EXP: u32 = 6u;
const RENDER_EXPR_OP_UNARY_LOG: u32 = 7u;
const RENDER_EXPR_OP_UNARY_SQRT: u32 = 8u;
const RENDER_EXPR_OP_UNARY_ABS: u32 = 9u;
const RENDER_EXPR_OP_NEGATE: u32 = 10u;
const RENDER_EXPR_OP_BINARY_ADD: u32 = 20u;
const RENDER_EXPR_OP_BINARY_SUB: u32 = 21u;
const RENDER_EXPR_OP_BINARY_MUL: u32 = 22u;
const RENDER_EXPR_OP_BINARY_DIV: u32 = 23u;
const RENDER_EXPR_OP_BINARY_MOD: u32 = 24u;
const RENDER_EXPR_OP_BINARY_POW: u32 = 25u;
const RENDER_EXPR_OP_BINARY_GT: u32 = 30u;
const RENDER_EXPR_OP_BINARY_GE: u32 = 31u;
const RENDER_EXPR_OP_BINARY_LT: u32 = 32u;
const RENDER_EXPR_OP_BINARY_LE: u32 = 33u;
const RENDER_EXPR_OP_BINARY_EQ: u32 = 34u;
const RENDER_EXPR_OP_BINARY_NE: u32 = 35u;
const RENDER_EXPR_OP_IF: u32 = 40u;
const RENDER_EXPR_OP_SMOOTHSTEP: u32 = 41u;

const FACTOR_UNARY_NONE: u32 = 0u;
const FACTOR_UNARY_SIN: u32 = 1u;
const FACTOR_UNARY_COS: u32 = 2u;

const RENDER_IMPORT_UV: u32 = 0x1u;
const RENDER_IMPORT_COLOR: u32 = 0x2u;
const RENDER_IMPORT_VERTEX_LOCAL: u32 = 0x4u;
const RENDER_IMPORT_INSTANCE_POS: u32 = 0x8u;
const RENDER_IMPORT_INSTANCE_SIZE: u32 = 0x10u;
const RENDER_IMPORT_RANDOM: u32 = 0x20u;

struct RenderImportInputs {
    uv: vec2<f32>,
    base_color: vec4<f32>,
    local_pos: vec2<f32>,
    instance_pos: vec2<f32>,
    instance_size: vec2<f32>,
};

@group(0) @binding(0)
var<storage, read> global_uniform: GlobalUniform;

@group(0) @binding(1)
var<storage, read_write> debug_buffer: GpuUiDebugReadCallBack;

@group(1) @binding(0)
var ui_textures: binding_array<texture_2d<f32>>;

@group(1) @binding(1)
var ui_samplers: binding_array<sampler>;

@group(1) @binding(2)
var<storage, read> sub_image_struct_array: array<GpuUiTextureInfo>;

@group(2) @binding(0)
var<storage, read> kennel_render_layers: array<RenderBindingLayer>;

@group(2) @binding(1)
var<storage, read> kennel_results_buffer: array<vec4<f32>>;

@group(2) @binding(2)
var<storage, read> render_expr_nodes: array<RenderExprNode>;

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

fn mode(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn apply_factor_unary(kind: u32, value: f32) -> f32 {
    switch kind {
        case FACTOR_UNARY_SIN: {
            return sin(value);
        }
        case FACTOR_UNARY_COS: {
            return cos(value);
        }
        default: {
            return value;
        }
    }
}

fn read_render_import(mask: u32, component_index: u32, inputs: RenderImportInputs) -> f32 {

    if (mask & RENDER_IMPORT_UV) != 0u {
        let uv_ext = vec4<f32>(inputs.uv, 0.0, 1.0);
        return uv_ext[component_index];
    }
    if (mask & RENDER_IMPORT_COLOR) != 0u {
        return inputs.base_color[component_index];
    }
    if (mask & RENDER_IMPORT_VERTEX_LOCAL) != 0u {
        let local = vec4<f32>(inputs.local_pos, 0.0, 1.0);
        return local[component_index];
    }
    if (mask & RENDER_IMPORT_INSTANCE_POS) != 0u {
        let pos = vec4<f32>(inputs.instance_pos, inputs.instance_size);
        return pos[component_index];
    }
    if (mask & RENDER_IMPORT_INSTANCE_SIZE) != 0u {
        let size = vec4<f32>(inputs.instance_size, 0.0, 1.0);
        return size[component_index];
    }
    if (mask & RENDER_IMPORT_RANDOM) != 0u {
        let seed = f32(global_uniform.frame) * 17.0 + f32(component_index) * 13.0 + global_uniform.time;
        return random(seed);
    }
    return 0.0;
}

fn eval_render_expression(start: u32, len: u32, lane: u32, inputs: RenderImportInputs) -> f32 {
    if (len == 0u) {
        return 0.0;
    }

    var values: array<f32, MAX_RENDER_EXPR_NODES>;
    for (var idx: u32 = 0u; idx < len; idx = idx + 1u) {
        let node = render_expr_nodes[start + idx];
        switch node.op {
            case RENDER_EXPR_OP_CONSTANT: {
                values[idx] = node.data0;
            }
            case RENDER_EXPR_OP_RENDER_IMPORT: {
                values[idx] = read_render_import(node.arg0, node.arg1, inputs);
            }
            case RENDER_EXPR_OP_COMPUTE_RESULT: {
                if (node.arg0 < arrayLength(&kennel_results_buffer)) {
                    values[idx] = kennel_results_buffer[node.arg0][lane];
                } else {
                    values[idx] = 0.0;
                }
            }
            case RENDER_EXPR_OP_UNARY_SIN: {
                values[idx] = sin(values[node.arg0 - start]);
            }
            case RENDER_EXPR_OP_UNARY_COS: {
                values[idx] = cos(values[node.arg0 - start]);
            }
            case RENDER_EXPR_OP_UNARY_TAN: {
                values[idx] = tan(values[node.arg0 - start]);
            }
            case RENDER_EXPR_OP_UNARY_EXP: {
                values[idx] = exp(values[node.arg0 - start]);
            }
            case RENDER_EXPR_OP_UNARY_LOG: {
                values[idx] = log(values[node.arg0 - start]);
            }
            case RENDER_EXPR_OP_UNARY_SQRT: {
                values[idx] = sqrt(values[node.arg0 - start]);
            }
            case RENDER_EXPR_OP_UNARY_ABS: {
                values[idx] = abs(values[node.arg0 - start]);
            }
            case RENDER_EXPR_OP_NEGATE: {
                values[idx] = -values[node.arg0 - start];
            }
            case RENDER_EXPR_OP_BINARY_ADD: {
                values[idx] = values[node.arg0 - start] + values[node.arg1 - start];
            }
            case RENDER_EXPR_OP_BINARY_SUB: {
                values[idx] = values[node.arg0 - start] - values[node.arg1 - start];
            }
            case RENDER_EXPR_OP_BINARY_MUL: {
                values[idx] = values[node.arg0 - start] * values[node.arg1 - start];
            }
            case RENDER_EXPR_OP_BINARY_DIV: {
                let right = values[node.arg1 - start];
                values[idx] = select(0.0, values[node.arg0 - start] / right, abs(right) > 1e-6);
            }
            case RENDER_EXPR_OP_BINARY_MOD: {
                let right = values[node.arg1 - start];
                values[idx] = select(0.0, mode(values[node.arg0 - start], right), abs(right) > 1e-6);
            }
            case RENDER_EXPR_OP_BINARY_POW: {
                values[idx] = pow(values[node.arg0 - start], values[node.arg1 - start]);
            }
            case RENDER_EXPR_OP_BINARY_GT: {
                values[idx] = select(0.0, 1.0, values[node.arg0 - start] > values[node.arg1 - start]);
            }
            case RENDER_EXPR_OP_BINARY_GE: {
                values[idx] = select(0.0, 1.0, values[node.arg0 - start] >= values[node.arg1 - start]);
            }
            case RENDER_EXPR_OP_BINARY_LT: {
                values[idx] = select(0.0, 1.0, values[node.arg0 - start] < values[node.arg1 - start]);
            }
            case RENDER_EXPR_OP_BINARY_LE: {
                values[idx] = select(0.0, 1.0, values[node.arg0 - start] <= values[node.arg1 - start]);
            }
            case RENDER_EXPR_OP_BINARY_EQ: {
                values[idx] = select(0.0, 1.0, abs(values[node.arg0 - start] - values[node.arg1 - start]) < 1e-6);
            }
            case RENDER_EXPR_OP_BINARY_NE: {
                values[idx] = select(1.0, 0.0, abs(values[node.arg0 - start] - values[node.arg1 - start]) < 1e-6);
            }
            case RENDER_EXPR_OP_IF: {
                let cond = values[node.arg0 - start];
                let then_val = values[node.arg1 - start];
                let else_val = values[node.arg2 - start];
                values[idx] = select(else_val, then_val, cond > 0.5);
            }
            case RENDER_EXPR_OP_SMOOTHSTEP: {
                let edge0 = values[node.arg0 - start];
                let edge1 = values[node.arg1 - start];
                let val = values[node.arg2 - start];
                values[idx] = smoothstep(edge0, edge1, val);
                debug_buffer.floats[8] = 999999.9;
            }
            default: {
                values[idx] = 0.0;
            }
        }
    }

    return values[len - 1u];
}

fn evaluate_render_layer(layer_index: u32, inputs: RenderImportInputs) -> vec4<f32> {
    if (layer_index == U32_MAX || layer_index >= arrayLength(&kennel_render_layers)) {
        return vec4<f32>(0.0);
    }

    let layer = kennel_render_layers[layer_index];
    var composed = vec4<f32>(0.0);

    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {
        let comp = layer.components[lane];
        switch comp.channel_type {
            case CHANNEL_CONSTANT: {
                composed[lane] = comp.payload.x;
            }
            case CHANNEL_COMPUTE: {
                if (comp.source_index < arrayLength(&kennel_results_buffer)) {
                    composed[lane] = kennel_results_buffer[comp.source_index][lane];
                }
            }
            case CHANNEL_RENDER_IMPORT: {
                let src_component = comp.source_component % 4u;
                composed[lane] = read_render_import(comp.source_index, src_component, inputs);
            }
            case CHANNEL_RENDER_COMPOSITE: {
                let src_component = comp.source_component % 4u;
                let base = read_render_import(comp.source_index, src_component, inputs);
                var inner = comp.payload.y;
                if (comp.factor_component != U32_MAX) {
                    let factor_base = read_render_import(comp.source_index, comp.factor_component, inputs);
                    inner = inner + factor_base * comp.payload.x;
                }
                if (comp.factor_inner_compute != U32_MAX
                    && comp.factor_inner_compute < arrayLength(&kennel_results_buffer)) {
                    let compute_val = kennel_results_buffer[comp.factor_inner_compute][lane];
                    inner = inner + compute_val;
                }
                inner = apply_factor_unary(comp.factor_unary, inner);
                var factor = inner * comp.payload.z;
                if (comp.factor_outer_compute != U32_MAX
                    && comp.factor_outer_compute < arrayLength(&kennel_results_buffer)) {
                    let compute_val = kennel_results_buffer[comp.factor_outer_compute][lane];
                    factor = factor * compute_val;
                }
                composed[lane] = base * factor + comp.payload.w;
            }
            case CHANNEL_RENDER_EXPR: {
                composed[lane] = eval_render_expression(
                    comp.source_index,
                    comp.component_index,
                    lane,
                    inputs,
                );
            }
            default: {
                composed[lane] = 0.0;
            }
        }
    }

    return composed;
}


fn random(seed: f32) -> f32 {
    // 使用一个简单的哈希函数生成伪随机数
    return fract(sin(seed) * 43758.5453);
}

fn smoothstep_official(edge0: f32, edge1: f32, value: f32) -> f32 {
    // WGSL 内置 smoothstep：对 value 进行 clamp，并使用 3x^2-2x^3 的 Hermite 插值
    return smoothstep(edge0, edge1, value);
}

fn rotate_xyz(p: vec3<f32>, rot: vec3<f32>) -> vec3<f32> {
    let cx = cos(rot.x);
    let sx = sin(rot.x);
    let cy = cos(rot.y);
    let sy = sin(rot.y);
    let cz = cos(rot.z);
    let sz = sin(rot.z);

    var v = p;
    v = vec3<f32>(v.x, v.y * cx - v.z * sx, v.y * sx + v.z * cx);
    v = vec3<f32>(v.x * cy + v.z * sy, v.y, -v.x * sy + v.z * cy);
    v = vec3<f32>(v.x * cz - v.y * sz, v.x * sz + v.y * cz, v.z);
    return v;
}

fn rounded_rect_sdf(p: vec2<f32>, half_extents: vec2<f32>, radius: f32) -> f32 {
    let r = min(radius, min(half_extents.x, half_extents.y));
    let inner = max(half_extents - vec2<f32>(r, r), vec2<f32>(0.0));
    let q = abs(p) - inner;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;


    var instance_pos = input.instance_pos;
    var instance_size = input.instance_size;
    var local_pos = input.pos;

    // 根据顶点落在的象限映射到调试槽，避免互相覆盖
    var slot: u32 = 0u;

    let uv = input.uv_offset + input.uv * input.uv_scale;

    if (input.visible == 0u) {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0);
        out.texture_id = U32_MAX;
        out.transparent = 0.0;
        out.color = vec4<f32>(0.0);
        out.border_color = vec4<f32>(0.0);
        out.border = vec2<f32>(0.0);
        out.local_pos = vec2<f32>(0.0);
        out.instance_size = vec2<f32>(0.0);
        out.instance_pos = vec2<f32>(0.0);
        out.fragment_shader_id = U32_MAX;
        out.vertex_shader_id = U32_MAX;
        return out;
    }

    if (input.vertex_shader_id != U32_MAX) {
        let vertex_inputs = RenderImportInputs(
            uv,
            vec4<f32>(0.0),
            input.pos,
            instance_pos,
            instance_size,
        );
        let vertex_adjust = evaluate_render_layer(input.vertex_shader_id, vertex_inputs);


        local_pos = input.pos + vertex_adjust.xy;


        instance_size = max(instance_size + vertex_adjust.zw, vec2<f32>(0.0));
    }
    let center = instance_pos + instance_size * 0.5;
    let centered = (local_pos - vec2<f32>(0.5, 0.5)) * instance_size;
    let scaled = centered * input.scale.xy;
    let rotated = rotate_xyz(vec3<f32>(scaled, 0.0), input.rotation.xyz);
    let screen_extent = max(f32(global_uniform.screen_size.x), f32(global_uniform.screen_size.y));
    let perspective_scale = select(0.0, 2.0 / screen_extent, screen_extent > 0.0);
    let depth = rotated.z * perspective_scale;
    let perspective = 1.0 / max(0.2, 1.0 + depth);
    let quad_pos = center + rotated.xy * perspective;

    out.clip_position = to_clip_space(quad_pos);

    let normalized_z = f32(input.z_index) / 100.0;
    out.clip_position.z = 1.0 - clamp(normalized_z, 0.0, 1.0);

    out.uv = uv;
    out.texture_id = input.texture_id;
    out.transparent = input.transparent;
    out.color = input.color;
    out.border_color = input.border_color;
    out.border = input.border;
    out.local_pos = local_pos;
    out.instance_size = instance_size * input.scale.xy;
    out.instance_pos = instance_pos;
    out.fragment_shader_id = input.fragment_shader_id;
    out.vertex_shader_id = input.vertex_shader_id;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var sampled = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    if (input.texture_id != U32_MAX) {
        let info = sub_image_struct_array[input.texture_id];
        let parent_index = info.parent_index;
        let tex = ui_textures[parent_index];
        let samp = ui_samplers[parent_index];

        let atlas_uv = input.uv;
        sampled = textureSample(tex, samp, atlas_uv);
    }

    var base_color = vec4<f32>(sampled.rgb * input.color.rgb, sampled.a * input.color.a);
    base_color.a = base_color.a * input.transparent;

    let border_width = input.border.x;
    if (border_width > 0.0) {
        let half_size = input.instance_size * 0.5;
        let radius = input.border.y;
        let centered = (input.local_pos - vec2<f32>(0.5, 0.5)) * input.instance_size;
        let sdf = rounded_rect_sdf(centered, half_size, radius);

        if (sdf > 0.0) {
            base_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        } else if (sdf >= -border_width) {
            base_color = vec4<f32>(input.border_color.rgb, input.border_color.a * input.transparent);
        }
    }

    let render_inputs = RenderImportInputs(
        input.uv,
        base_color,
        input.local_pos,
        input.instance_pos,
        input.instance_size,
    );
    let kennel_color = evaluate_render_layer(input.fragment_shader_id, render_inputs);
    
    var final_color = base_color;
    if any(kennel_color != vec4<f32>(0.0)) {
        final_color = kennel_color;
    }

    return final_color;
}
