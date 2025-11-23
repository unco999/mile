struct Panel {
    // === 16-byte �?1 ===
    position: vec2<f32>,    // 8 bytes
    size: vec2<f32>,        // 8 bytes

    // === 16-byte �?2 ===
    uv_offset: vec2<f32>,   // 8 bytes
    uv_scale: vec2<f32>,    // 8 bytes

    // === 16-byte �?3 ===
    z_index: u32,           // 4 bytes
    interaction_passthrough: u32,      // 4 bytes
    id: u32,                // 4 bytes
    interaction: u32,       // 4 bytes

    // === 16-byte �?4 ===
    event_mask: u32,        // 4 bytes
    state_mask: u32,        // 4 bytes
    transparent: f32,       // 4 bytes
    texture_id: u32,        // 4 bytes

    // === 16-byte �?5 ===
    state: u32,             // 4 bytes
    collection_state: u32,  // 4 bytes
    fragment_shader_id: u32,// 4 bytes
    vertex_shader_id: u32,  // 4 bytes

    // === 16-byte �?6 ===
    rotation: vec4<f32>,

    // === 16-byte �?7 ===
    scale: vec4<f32>,

    // === 16-byte �?8 ===
    color: vec4<f32>,       // 16 bytes

    // === 16-byte �?9 ===
    border_color: vec4<f32>,// 16 bytes

    // === 16-byte �?10 ===
    border_width: f32,      // 4 bytes
    border_radius: f32,     // 4 bytes
    visible: u32,           // 4 bytes
    _pad_border: u32,       // 4 bytes (填充)
};


struct TransformAnim {
    field_id: u32,
    field_len: u32,
    start_value: f32,
    end_value: f32,

    easing_mask: u32,
    _pad1: u32,
    duration: f32,
    elapsed: f32,

    instance_id: u32,
    op: u32,
    _pad2: u32,
    _pad3: u32,

    last_applied: f32,
    _pad4: vec3<u32>, // 对齐�?16 字节
};

struct GpuUiCollection {
    start_index: u32,
    len: u32,
    sampling: u32,
    reserved: u32,
}

struct GpuUiRelation {
    source_collection_id: u32,
    target_collection_id: u32,
    influence_start: u32,
    influence_count: u32,
    id_start: u32,
    id_count: u32,
    reserved: u32,
    padding: u32, // 新增：保�?2字节对齐
}

struct GpuUiInfluence {
    field: u32,
    weight: f32,
    influence_type: u32,
    reserved: u32,
}
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

struct GpuUiIdInfo {
    panel_id: u32,     // panel 的唯一 id
    is_source: u32,    // 0 = 普�?panel, 1 = source panel
    relation_idx: u32, // 对应�?relation buffer 索引，如果不�?source 可填 0xFFFFFFFF
    padding: u32,      // 对齐�?16 字节
}


/**
**/
struct GpuInteractionFrame {
    frame: u32,
    drag_id: u32,
    hover_id: u32,
    click_id: atomic<u32>,

    mouse_pos: vec2<f32>,
    trigger_panel_state:u32,
    _pad1:u32,

    mouse_state: u32,
    _pad2: vec3<u32>,

    drag_delta: vec2<f32>,
    _pad3: vec2<f32>,

    pinch_delta: f32,
    interaction_passthrough_depth: u32,
    event_point: vec2<f32>,
    _pad5: vec4<u32>, // 保证 128 字节
};

fn update_hover_id(new_hover: u32) {
}


struct GpuUiDebugReadCallBack {
    floats: array<f32,32>, // 16 * 4 = 64 字节
    uints: array<u32,32>,  // 16 * 4 = 64 字节
}

fn ndc_to_pixel(ndc: vec2<f32>, screen_size: vec2<f32>) -> vec2<f32> {
    // X: [-1,1] -> [0, width]
    let x = (ndc.x + 1.0) * 0.5 * screen_size.x;

    // Y: [-1,1] -> [height, 0] （翻�?Y�?
    let y = (1.0 - ndc.y) * 0.5 * screen_size.y;

    return vec2<f32>(x, y);
}


struct PanelAnimDelta {
    // --- Position delta ---
     delta_position: vec2<f32>, // x, y

    // --- Size delta ---
     delta_size: vec2<f32>,     // width, height

    // --- UV delta ---
     delta_uv_offset: vec2<f32>,
     delta_uv_scale: vec2<f32>,

    // --- Panel attributes ---
     delta_z_index: i32,        // 可选，用于动画 z 层变�?
     delta_interaction_passthrough: i32,   // 可�?
     panel_id: u32,             // 对应 Panel �?id
     _pad0: u32,                // 补齐 16 字节

    // --- 状态相�?---
     delta_interaction: u32,    // mask
     delta_event_mask: u32,     // mask
     delta_state_mask: u32,     // mask
     _pad1: u32,                // 对齐

    // --- 透明�?texture ---
     delta_transparent: f32,
     delta_texture_id: i32,     // 可选，整型�?texture 变化
     _pad2: u32,           // 补齐 16 字节
     _pad3: u32,           // 补齐 16 字节

    // --- 起始位置 ---
     start_position: vec2<f32>,
     container_origin: vec2<f32>,
}

// fn try_set_click_layout(inst_id: u32, interaction_passthrough: u32) -> bool {
//     // 如果允许穿透，不参与竞�?
//     if (interaction_passthrough == 1u) {
//         return false;
//     }

//     // 原子地比较写入最大ID
//     let prev_id = atomicMax(&global_uniform.click_layout, inst_id);

//     if(prev_id == 0u){
//         return true;
//     }

//     // 如果当前id更大，说明自己是目前最上层的点击目�?
//     return inst_id > prev_id;
// }

fn try_set_click_layout(panel_id: u32, z_index: u32, interaction_passthrough: u32) -> bool {
    if (interaction_passthrough == 1u) {
        return false; // 穿透层不参与竞�?
    }

    let prev_z = atomicMax(&global_uniform.click_layout_z, z_index);
    let prev_id = atomicLoad(&global_uniform.click_layout_id);

         // debug 信息
    if (z_index > prev_z || (z_index == prev_z && panel_id >= prev_id)) {
        if ( atomicMax(&global_uniform.click_layout_id, panel_id) == panel_id){
            return true;
        }

        return true; 
    }

    return false; 
}

fn try_set_hover_layout(panel_id: u32, z_index: u32, interaction_passthrough: u32) -> bool {
    if (interaction_passthrough == 1u) {
        return false; 
    }

    let prev_z = atomicMax(&global_uniform.hover_layout_z, z_index);
    let prev_id = atomicLoad(&global_uniform.hover_layout_id);


    if (z_index > prev_z || (z_index == prev_z && panel_id >= prev_id)) {
        if (atomicMax(&global_uniform.hover_layout_id, panel_id) == panel_id) {
            return true; // 成功成为当前点击目标
        }
    }

    return false; // 被上层挡�?
}

fn try_set_drag_layout(panel_id: u32, z_index: u32, interaction_passthrough: u32) -> bool {
    if (interaction_passthrough == 1u) {
        return false; // 穿透层不参与竞�?
    }

    // 原子获取当前最�?z_index
    let prev_z = atomicMax(&global_uniform.drag_layout_z, z_index);
    let prev_id = atomicLoad(&global_uniform.drag_layout_id);

         // debug 信息
    // 如果自己是更高层，或者同�?id 更大
    if (z_index > prev_z || (z_index == prev_z && panel_id >= prev_id)) {
        if ( atomicMax(&global_uniform.drag_layout_id, panel_id) == panel_id){
            return true; // 成功成为当前点击目标
        }
    }

    return false; // 被上层挡�?
}



const U32_MAX: u32 = 4294967295u;

@group(0) @binding(0) var<storage, read_write> panels: array<Panel>;
@group(0) @binding(1) var<storage,read_write>  global_uniform: GlobalUniform;
//0号索引是当前�?
@group(0) @binding(2) var<storage,read_write> frame_cache_array:array<GpuInteractionFrame,2>;
@group(0) @binding(3) var<storage, read_write> debug_buffer: GpuUiDebugReadCallBack;
@group(0) @binding(4) var<storage, read_write> panel_anim_delta: array<PanelAnimDelta>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    var inst = panels[idx];
    
    let mouse = global_uniform.mouse_pos; 
    let state = global_uniform.mouse_state; 
    let panel_interaction = inst.interaction;
    let interaction_passthrough = inst.interaction_passthrough;
    let z_index = inst.z_index;

    debug_buffer.floats[0] = global_uniform.time;

    let half_size = inst.size / 2.0;
    let screen_size_f = vec2<f32>(global_uniform.screen_size); 
    let center_ndc = vec2<f32>(
    (inst.position.x / screen_size_f.x) * 2.0 - 1.0,
    1.0 - (inst.position.y / screen_size_f.y) * 2.0
    );
    let half_size_ndc = half_size / screen_size_f * 2.0;
    let min_pos = center_ndc - half_size_ndc;
    let max_pos = center_ndc + half_size_ndc;

    let mouse_pressed = (global_uniform.mouse_state & 16u) != 0u;
    let mouse_released = (global_uniform.mouse_state & 32u) != 0u;
    let mouse_pressed_time = global_uniform.press_duration;
    
    let pixel_pos = ndc_to_pixel(mouse, screen_size_f);


            // debug_buffer.floats[0] = min_pos.x;
            // debug_buffer.floats[1] = min_pos.y;
            // debug_buffer.floats[2] = max_pos.x;
            // debug_buffer.floats[3] = max_pos.y;

            // debug_buffer.floats[4] = mouse.x;
            // debug_buffer.floats[5] = mouse.y;


       

    // 如果鼠标在面板范围内
    if (mouse.x >= min_pos.x && mouse.x <= max_pos.x &&
        mouse.y >= min_pos.y && mouse.y <= max_pos.y) {

        if (
            ((panel_interaction & 4u) != 0u) && 
            mouse_pressed && 
            frame_cache_array[0].drag_id == inst.id 
         ) {
            // 计算面板新位�?
            let new_panel_pos = pixel_pos - global_uniform.event_point;

            // delta 可以直接用鼠标移动量
            let delta = new_panel_pos - panels[idx].position;


            // 更新动画缓冲
            panel_anim_delta[inst.id].delta_position = delta;

            // 更新面板位置
            panels[idx].position = new_panel_pos;

            // 保持 drag_id
            frame_cache_array[1].drag_id = inst.id;
        }else{
             panel_anim_delta[inst.id].delta_position = vec2<f32>(0.0);
        }
            
        let try_hover = try_set_hover_layout(inst.id,z_index,interaction_passthrough);
        // drag
        if(  
            ((panel_interaction & 2u) != 0u) && 
            try_hover
        ){
            frame_cache_array[1].hover_id = atomicLoad(&global_uniform.hover_layout_id); // 直接写入 click
            frame_cache_array[1].trigger_panel_state = panels[frame_cache_array[1].hover_id].state;
        }
        
        let try_drag = try_set_click_layout(inst.id,z_index,interaction_passthrough);
        // click
        if ( 
            ((panel_interaction & 2u) != 0u) 
            && mouse_released 
            && mouse_pressed_time > 0.035 
            && frame_cache_array[0].drag_id == 0xFFFFFFFFu
            && try_drag
            ) {
            frame_cache_array[1].click_id = atomicLoad(&global_uniform.click_layout_id); // 直接写入 click
            frame_cache_array[1].trigger_panel_state = panels[frame_cache_array[1].click_id].state;
            global_uniform.event_point = pixel_pos - panels[frame_cache_array[1].click_id].position;
            
        }

        // drag
        let try_click = try_set_drag_layout(inst.id,z_index,interaction_passthrough);

        if(  
            ((panel_interaction & 4u) != 0u) && 
            mouse_pressed && 
            mouse_released == false &&
            mouse_pressed_time > 0.113 &&
            try_click
        ){
            frame_cache_array[1].drag_id = atomicLoad(&global_uniform.drag_layout_id); // 直接写入 drag
            frame_cache_array[1].trigger_panel_state = panels[frame_cache_array[1].drag_id].state;
            global_uniform.event_point = pixel_pos - panels[frame_cache_array[1].drag_id].position;
            return;
        }




        if(mouse_released){
            frame_cache_array[1].drag_id = 0xFFFFFFFFu;
            global_uniform.hover_layout_id = 0u;
            global_uniform.hover_layout_z = 0u;
            global_uniform.click_layout_z = 0u;
            global_uniform.click_layout_id = 0u;
            global_uniform.drag_layout_id = 0u;
            global_uniform.drag_layout_id = 0u;

        }

    // }


        // if(((panel_interaction & 2u) != 0u) && frame_cache_array[1].hover_id == inst.id){
        //     frame_cache_array[1].trigger_panel_state = panels[frame_cache_array[1].hover_id].state;
        //     frame_cache_array[1].hover_id = U32_MAX;
        //     debug_buffer.uints[idx] =  66;
        // }
    }

    // debug_buffer.uints[0] = state;
}
