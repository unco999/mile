struct VertexInput {
    // --- 椤剁偣鍩虹 ---
    @location(0) pos: vec2<f32>,        // 椤剁偣灞€閮ㄤ綅�?
    @location(1) uv: vec2<f32>,         // 椤剁�?UV

    // --- Panel 瀹炰緥鏁版嵁 ---
    @location(2) instance_pos: vec2<f32>,   // panel 浣嶇�?
    @location(3) instance_size: vec2<f32>,  // panel 灏哄�?
    @location(4) uv_offset: vec2<f32>,      // panel UV offset
    @location(5) uv_scale: vec2<f32>,       // panel UV scale

    // === Block 3 ===
    @location(6) z_index: u32,              // panel z_index
    @location(7) pass_through: u32,         // panel pass_through
    @location(8) instance_id: u32,          // panel id
    @location(9) interaction: u32,          // panel interaction mask

    // === Block 4 ===
    @location(10) event_mask: u32,          // panel event response mask
    @location(11) state_mask: u32,          // panel state mask
    @location(12) transparent: f32,         // panel transparent (瀵归�?
    @location(13) texture_id: u32,          // panel texture_id

    // === Block 5 ===
    @location(14) state: u32,               // panel state
    @location(15) kennel_des_id: u32,                // pad[0]
    @location(16) pad1: u32,                // pad[1]
};

struct GpuKennelPanelDes {
    color_input : array<u32, 4>,  // 16 bytes (4 * 4 bytes)
    vertex_input : array<u32, 4>, // 16 bytes (4 * 4 bytes)
    blender : u32,                // 4 bytes
    self_old_index : u32,         // 4 bytes
};


struct GlobalUniform {
      // === block 1: atomic z/layouts ===
    click_layout_z: atomic<u32>,
    click_layout_id: atomic<u32>,
    hover_layout_id: atomic<u32>,
    hover_layout_z: atomic<u32>, // 16 bytes

    // === block 2: atomic drag ===
    drag_layout_id: atomic<u32>,
    drag_layout_z: atomic<u32>,
    pad_atomic1: atomic<u32>,
    pad_atomic2: atomic<u32>,    // 16 bytes

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

struct SharedState {
    // Mouse position
    mouse_pos: vec2<f32>,        // 8 bytes

    // Mouse button state mask
    mouse_state: u32,            // 4 bytes
    _pad0: u32,                  // 4 bytes padding 瀵归�?

    // Hover panel ID
    hover_id: atomic<u32>,       // 4 bytes
    hover_blocked: atomic<u32>,  // 4 bytes
    _pad1: vec2<u32>,            // 8 bytes padding 瀵归�?hover_pos

    // Hover position
    hover_pos: vec2<f32>,        // 8 bytes

    // Current depth under mouse
    current_depth: u32,          // 4 bytes
    _pad2: u32,                  // 4 bytes padding

    // Clicked panel ID (鏈€鍚庝竴娆＄偣�?
    click_id: u32,               // 4 bytes
    click_blocked: u32,          // 4 bytes

    // Drag panel ID
    drag_id: u32,                // 4 bytes
    drag_blocked: u32,           // 4 bytes

    // History panel ID
    history_id: u32,             // 4 bytes
    _pad3: u32,                  // 4 bytes padding

    // Final padding to 64 bytes
    _pad4: vec2<u32>,            // 8 bytes
};

struct GpuUiTextureInfo {
    index: u32,
    parent_index: u32,
    _pad: vec2<u32>,
    uv_min: vec4<f32>, // 鍓嶄袱浣嶆槸鐪熷疄鍊?
    uv_max: vec4<f32>, // 鍓嶄袱浣嶆槸鐪熷疄鍊?
};


struct RenderOperation {
    op_type: u32,           // 鎿嶄綔绫诲�?
    source_type: u32,       // 鏁版嵁婧愮被�?
    buffer_offset: u32,     // 鍦╒_buffer涓殑瀛楄妭鍋忕Щ閲?
    component_count: u32,   // 鍒嗛噺鏁伴噺
    component_stride: u32,  // 鍒嗛噺姝ラ暱  
    data_format: u32,       // 鏁版嵁鏍煎紡
    blend_factor: f32,      // 娣峰悎鍥犲瓙
    custom_param: f32,      // 鑷畾涔夊弬�?
    condition_source: u32,  // 鏉′欢鏁版嵁婧?
    then_source: u32,       // then鍒嗘敮鏁版嵁�?
    else_source: u32,       // else鍒嗘敮鏁版嵁�?
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



@group(0) @binding(1)
var<storage, read_write> global_uniform: GlobalUniform;

@group(0) @binding(3)
var<storage, read> custom_wgsl: array<CustomWgsl>;





@group(1) @binding(0)
var ui_textures: binding_array<texture_2d<f32>>;

@group(1) @binding(1)
var ui_sampler:  binding_array<sampler>;

@group(1) @binding(2)
var<storage, read> sub_image_struct_array: array<GpuUiTextureInfo>;



@group(2) @binding(0)
var<storage, read> kennel_render_layers: array<RenderBindingLayer>;

@group(2) @binding(1)
var<storage, read> kennel_results_buffer: array<vec4<f32>>;
@group(2) @binding(2)
var<storage, read> render_expr_nodes: array<RenderExprNode>;


// 鎿嶄綔绫诲瀷甯搁噺
const OP_DIRECT: u32 = 0u;      // 鐩存帴浣跨敤
const OP_ADD: u32 = 1u;         // 鍔犳�?
const OP_MULTIPLY: u32 = 2u;    // 涔樻�? 
const OP_SUBTRACT: u32 = 4u;    // 鍑忔�?
const OP_DIVIDE: u32 = 5u;      // 闄ゆ�?
const OP_CONDITIONAL: u32 = 20u; // 鏉′欢娣峰�?

const CHANNEL_CONSTANT: u32 = 0u;
const CHANNEL_COMPUTE: u32 = 1u;
const CHANNEL_RENDER_IMPORT: u32 = 2u;
const CHANNEL_RENDER_COMPOSITE: u32 = 3u;

const CHANNEL_RENDER_EXPR: u32 = 4u;
const FACTOR_UNARY_NONE: u32 = 0u;
const FACTOR_UNARY_SIN: u32 = 1u;
const FACTOR_UNARY_COS: u32 = 2u;

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
const MAX_RENDER_EXPR_NODES: u32 = 64u;
const RENDER_IMPORT_UV: u32 = 0x1u;
const RENDER_IMPORT_COLOR: u32 = 0x2u;

// 鏁版嵁婧愮被鍨嬪父閲?
const SOURCE_COMPUTE_BUFFER: u32 = 0u;  // 璁＄畻缂撳啿�?
const SOURCE_RENDER_CALC: u32 = 1u;     // 娓叉煋璁＄畻
const SOURCE_RENDER_INPUT: u32 = 2u;    // 娓叉煋杈撳叆(UV�?

// 鏁版嵁鏍煎紡甯搁�?
const FORMAT_SCALAR: u32 = 0u;  // 鏍囬�?
const FORMAT_VEC2: u32 = 1u;    // vec2
const FORMAT_VEC3: u32 = 2u;    // vec3  
const FORMAT_VEC4: u32 = 3u;    // vec4


struct RenderPlanHeader {
    plan_count: u32,      // 瀹為檯璁″垝鏁伴�?
    dirty_flags: u32,     // 鑴忔爣璁?
    frame_index: u32,     // 甯х储�?
    _padding: u32,        // 濉�?
}

// 缁戝畾缁?

fn read_custom_frag(panel_id: u32, slot: u32) -> vec4<f32> {
    let frag_value = custom_wgsl[panel_id].frag[slot];
    var result: vec4<f32> = frag_value;

    // --- X ---
    if (frag_value.x == WGSL_TIME) {
        result.x = global_uniform.time;
    } else if (frag_value.x == WGSL_SIN_TIME) {
        result.x = sin(global_uniform.time);
    }

    // --- Y ---
    if (frag_value.y == WGSL_TIME) {
        result.y = global_uniform.time;
    } else if (frag_value.y == WGSL_SIN_TIME) {
        result.y = sin(global_uniform.time);
    }

    // --- Z ---
    if (frag_value.z == WGSL_TIME) {
        result.z = global_uniform.time;
    } else if (frag_value.z == WGSL_SIN_TIME) {
        result.z = sin(global_uniform.time);
    }

    // --- W ---
    if (frag_value.w == WGSL_TIME) {
        result.w = global_uniform.time;
    } else if (frag_value.w == WGSL_SIN_TIME) {
        result.w = sin(global_uniform.time);
    }

    return result;
}

struct CustomWgsl {
    frag: array<vec4<f32>,16>,
    vertex: array<vec4<f32>,16>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) instance_pos: vec2<f32>,
    @location(1) instance_size: vec2<f32>,
    @location(2) pass_through: u32,
    @location(3) z_index: u32,
    @location(4) instance_id: u32, // 浼犵�?fragment
    @location(5) transparent:f32,
    @location(6) texture_id:u32,
    @location(7) uv: vec2<f32>,
    @location(8) kennel_des_id:u32,
};

struct GpuAstNode {
    data: vec4<f32>,
    state: u32,
    op: u32,
    data_type: u32,
    left_child: u32,
    right_child: u32,
    import_info: u32,
    constant_value: f32,
    pad: u32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;


    let screen_size_f = vec2<f32>(global_uniform.screen_size);

    // 灞€閮ㄩ《鐐?[-0.5,0.5] �?鍍忕礌鍗曚綅
    let scaled_pos = input.pos * input.instance_size; // instance_size = 鍍忕礌瀹介�?

    // 椤剁偣鍍忕礌鍧愭�?= 闈㈡澘涓績 + 灞€閮ㄥ亸绉?
    let pixel_pos = input.instance_pos + scaled_pos;

    // 杞崲鍒?NDC [-1,1]
    let ndc = vec2<f32>(
        (pixel_pos.x / screen_size_f.x) * 2.0 - 1.0,
        1.0 - (pixel_pos.y / screen_size_f.y) * 2.0
    );

    out.position = vec4<f32>(ndc, 0.0, 1.0);


    out.instance_pos = input.instance_pos;
    out.instance_size = input.instance_size;
    out.pass_through = input.pass_through;
    out.z_index = input.z_index;
    out.instance_id = input.instance_id;
    out.transparent = input.transparent;
    out.texture_id = input.texture_id;
    out.uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);


    return out;
}

// @fragment
// fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
//     let tex_index = input.texture_id;


//     var color = vec4<f32>(0.0, 0.0, 0.0, input.transparent); // 榛樿鐧借壊 + alpha

//     // 妫€鏌ユ槸鍚﹁ hover
//     if (input.instance_id == sharedState.click_id) {
//         color = vec4<f32>(0.0, 1.0,1.0,input.transparent); // hover �?绾㈣�?+ alpha
//     }

//     return color;
// }
const WGSL_TIME: f32 = 9999999.0; // �?Rust �?MAX_TIME_SEC 瀵瑰�?
const WGSL_SIN_TIME: f32 = 9999999.1; // �?Rust �?MAX_TIME_SEC 瀵瑰�?
const U32_MAX: u32 = 4294967295u;

fn read_render_import(mask: u32, component_index: u32, uv: vec2<f32>, base_color: vec4<f32>) -> f32 {
    if (mask & RENDER_IMPORT_UV) != 0u {
        let uv_ext = vec4<f32>(uv, 0.0, 1.0);
        return uv_ext[component_index];
    }
    if (mask & RENDER_IMPORT_COLOR) != 0u {
        return base_color[component_index];
    }
    return 0.0;
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
fn eval_render_expression(start: u32, len: u32, lane: u32, input: VertexOutput, base_color: vec4<f32>) -> f32 {
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
                values[idx] = read_render_import(node.arg0, node.arg1, input.uv, base_color);
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
            default: {
                values[idx] = 0.0;
            }
        }
    }

    return values[len - 1u];
}

fn evaluate_render_layer(layer_index: u32, input: VertexOutput, base_color: vec4<f32>) -> vec4<f32> {
    if (layer_index >= arrayLength(&kennel_render_layers)) {
        return vec4<f32>(0.0);
    }

    let layer = kennel_render_layers[layer_index];
    var composed = vec4<f32>(0.0);

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let comp = layer.components[i];
        let lane = i;
        switch comp.channel_type {
            case CHANNEL_CONSTANT: {
                composed[lane] = comp.payload.x;
            }
            case CHANNEL_COMPUTE: {
                let node_index = comp.source_index;
                if (node_index < arrayLength(&kennel_results_buffer)) {
                    composed[lane] = kennel_results_buffer[node_index][lane];
                }
            }
            case CHANNEL_RENDER_IMPORT: {
                let src_component = comp.source_component % 4u;
                composed[lane] = read_render_import(comp.source_index, src_component, input.uv, base_color);
            }
            case CHANNEL_RENDER_COMPOSITE: {
                let src_component = comp.source_component % 4u;
                let base = read_render_import(comp.source_index, src_component, input.uv, base_color);
                var inner = comp.payload.y;
                if (comp.factor_component != U32_MAX) {
                    let factor_base = read_render_import(comp.source_index, comp.factor_component, input.uv, base_color);
                    inner = inner + factor_base * comp.payload.x;
                }
                if (comp.factor_inner_compute != U32_MAX && comp.factor_inner_compute < arrayLength(&kennel_results_buffer)) {
                    let compute_val = kennel_results_buffer[comp.factor_inner_compute][lane];
                    inner = inner + compute_val;
                }
                inner = apply_factor_unary(comp.factor_unary, inner);
                var factor = inner * comp.payload.z;
                if (comp.factor_outer_compute != U32_MAX && comp.factor_outer_compute < arrayLength(&kennel_results_buffer)) {
                    let compute_val = kennel_results_buffer[comp.factor_outer_compute][lane];
                    factor = factor * compute_val;
                }
                composed[lane] = base * factor + comp.payload.w;
            }
            case CHANNEL_RENDER_EXPR: {
                composed[lane] = eval_render_expression(comp.source_index, comp.component_index, lane, input, base_color);
            }
            default: {
                composed[lane] = 0.0;
            }
        }
    }

    return composed;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // 浣跨�?texture_id 绱㈠紩鏁扮粍杩涜閲囨牱


    let sub_image_struct = sub_image_struct_array[input.texture_id];
    let parent_index = sub_image_struct.parent_index;
    let sub_image_index = sub_image_struct.index;
    let sub_image_uv_min = vec2<f32>(sub_image_struct.uv_min.x,sub_image_struct.uv_min.y);
    let sub_image_uv_max = vec2<f32>(sub_image_struct.uv_max.x,sub_image_struct.uv_max.y);

    let tex = ui_textures[parent_index];
    let samp = ui_sampler[parent_index];


    let uv = input.uv; 
    let sub_uv = sub_image_uv_min + (sub_image_uv_max - sub_image_uv_min) * uv;
    let color = textureSample(tex, samp, sub_uv);

    
    let layer_index = input.kennel_des_id;
    let compute_color = evaluate_render_layer(layer_index, input, color);

    var final_color = color;
    if any(compute_color != vec4<f32>(0.0)) {
        final_color = compute_color;
    }

    return final_color;
    // 淇濇寔鍘熸湁閫忔槑搴?
}

// @fragment
// fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
//     // 绠€鍗曟�?UV 鏄犲皠鍒伴�?
//     let color = vec3<f32>(input.uv.x, input.uv.y, 0.0);

//     // 杈撳嚭棰滆壊锛屼繚鐣欓€忔槑�?
//     return vec4<f32>(color, input.transparent);
// }
