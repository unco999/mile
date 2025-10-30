struct VertexInput {
    // --- 顶点基础 ---
    @location(0) pos: vec2<f32>,        // 顶点局部位置
    @location(1) uv: vec2<f32>,         // 顶点 UV

    // --- Panel 实例数据 ---
    @location(2) instance_pos: vec2<f32>,   // panel 位置
    @location(3) instance_size: vec2<f32>,  // panel 尺寸
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
    @location(12) transparent: f32,         // panel transparent (对齐)
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
    _pad0: u32,                  // 4 bytes padding 对齐

    // Hover panel ID
    hover_id: atomic<u32>,       // 4 bytes
    hover_blocked: atomic<u32>,  // 4 bytes
    _pad1: vec2<u32>,            // 8 bytes padding 对齐 hover_pos

    // Hover position
    hover_pos: vec2<f32>,        // 8 bytes

    // Current depth under mouse
    current_depth: u32,          // 4 bytes
    _pad2: u32,                  // 4 bytes padding

    // Clicked panel ID (最后一次点击)
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
    uv_min: vec4<f32>, // 前两位是真实值
    uv_max: vec4<f32>, // 前两位是真实值
};


@group(0) @binding(1)
var<storage, read_write> global_uniform: GlobalUniform;

@group(0) @binding(3)
var<storage, read> custom_wgsl: array<CustomWgsl>;

@group(0) @binding(4) var<storage, read> V : array<f32>;

@group(0) @binding(5) var<storage, read> kennel_des_store : array<GpuKennelPanelDes>;

@group(1) @binding(0)
var ui_textures: binding_array<texture_2d<f32>>;

@group(1) @binding(1)
var ui_sampler:  binding_array<sampler>;

@group(1) @binding(2)
var<storage, read> sub_image_struct_array: array<GpuUiTextureInfo>;



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
    @location(4) instance_id: u32, // 传给 fragment
    @location(5) transparent:f32,
    @location(6) texture_id:u32,
    @location(7) uv: vec2<f32>,
    @location(8) kennel_des_id:u32,
};


@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;


    let screen_size_f = vec2<f32>(global_uniform.screen_size);

    // 局部顶点 [-0.5,0.5] → 像素单位
    let scaled_pos = input.pos * input.instance_size; // instance_size = 像素宽高

    // 顶点像素坐标 = 面板中心 + 局部偏移
    let pixel_pos = input.instance_pos + scaled_pos;

    // 转换到 NDC [-1,1]
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


//     var color = vec4<f32>(0.0, 0.0, 0.0, input.transparent); // 默认白色 + alpha

//     // 检查是否被 hover
//     if (input.instance_id == sharedState.click_id) {
//         color = vec4<f32>(0.0, 1.0,1.0,input.transparent); // hover → 红色 + alpha
//     }

//     return color;
// }
const WGSL_TIME: f32 = 9999999.0; // 与 Rust 的 MAX_TIME_SEC 对应
const WGSL_SIN_TIME: f32 = 9999999.1; // 与 Rust 的 MAX_TIME_SEC 对应
const U32_MAX: u32 = 4294967295u;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // 使用 texture_id 索引数组进行采样


    let sub_image_struct = sub_image_struct_array[input.texture_id];
    let parent_index = sub_image_struct.parent_index;
    let sub_image_index = sub_image_struct.index;
    let sub_image_uv_min = vec2<f32>(sub_image_struct.uv_min.x,sub_image_struct.uv_min.y);
    let sub_image_uv_max = vec2<f32>(sub_image_struct.uv_max.x,sub_image_struct.uv_max.y);

    let tex = ui_textures[parent_index];
    let samp = ui_sampler[parent_index];


    let uv = input.uv; // [0,1] 面板局部 UV
    let sub_uv = sub_image_uv_min + (sub_image_uv_max - sub_image_uv_min) * uv;
    let color = textureSample(tex, samp, sub_uv);

    let a = V[0];
    // let panel_frag: vec4<f32> = read_custom_frag(input.texture_id, 0);

    // mix 函数做线性混合
    // 这里用 panel_frag.a 控制混合比例
    // let final_color = mix(color, panel_frag, panel_frag.a); 

    if(input.kennel_des_id != U32_MAX){
        let kennel_des = kennel_des_store[input.kennel_des_id];
        let output_color_offset = kennel_des.color_input;
        let output_color_v = vec4<f32>(V[output_color_offset[0]],V[output_color_offset[1]],V[output_color_offset[2]],V[output_color_offset[3]]);
        return output_color_v;
    }

    // 保持原有透明度
    return vec4<f32>(color.rgb, color.a * input.transparent);
}

// @fragment
// fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
//     // 简单把 UV 映射到颜色
//     let color = vec3<f32>(input.uv.x, input.uv.y, 0.0);

//     // 输出颜色，保留透明度
//     return vec4<f32>(color, input.transparent);
// }