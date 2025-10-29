struct DebugCallBack {
    floats: array<f32, 32>,
    uints: array<u32, 32>,
};

struct Rel {
    id: u32,
    source_collection: u32,
    target_collection: u32,
    animation_field: u32,
};

struct Panel {
    position: vec2<f32>,       // 0..8
    size: vec2<f32>,           // 8..16
    uv_offset: vec2<f32>,      // 16..24
    uv_scale: vec2<f32>,       // 24..32

    z_index: u32,              // 32..36
    pass_through: u32,         // 36..40
    id: u32,                   // 40..44
    interaction: u32,          // 44..48

    event_mask: u32,           // 48..52
    state_mask: u32,           // 52..56
    transparent: f32,          // 56..60
    texture_id: u32,           // 60..64

    state: u32,                // 64..68
    collection_state:u32,
    pad1:u32,
    pad2:u32
};

 struct PanelAnimDelta {
    // --- Position / Size ---
    delta_position: vec2<f32>,   //  0  ~  8
    delta_size: vec2<f32>,       //  8  ~ 16

    // --- UV Delta ---
    delta_uv_offset: vec2<f32>,  // 16 ~ 24
    delta_uv_scale: vec2<f32>,   // 24 ~ 32

    // --- Panel Attributes ---
    delta_z_index: i32,          // 32 ~ 36
    delta_pass_through: i32,     // 36 ~ 40
    panel_id: u32,               // 40 ~ 44
    _pad0: u32,                  // 44 ~ 48

    // --- State Masks ---
    delta_interaction: u32,      // 48 ~ 52
    delta_event_mask: u32,       // 52 ~ 56
    delta_state_mask: u32,       // 56 ~ 60
    _pad1: u32,                  // 60 ~ 64

    // --- Transparency / Texture ---
    delta_transparent: f32,      // 64 ~ 68
    delta_texture_id: i32,       // 68 ~ 72
    _pad2: vec2<f32>,            // 72 ~ 80

    // --- Start Position ---
    start_position: vec2<f32>,   // 80 ~ 88
    _pad3: vec2<f32>,            // 88 ~ 96
};
@group(0) @binding(0)
var<storage, read_write> raw_data: array<u32>;          // meta
@group(0) @binding(1)
var<storage, read_write> rels_flat: array<u32>;         // rels_flat 改成 u32 数组
@group(0) @binding(2)
var<storage, read_write> debug: DebugCallBack;
@group(0) @binding(3) var<storage, read_write> panel_anim_delta: array<PanelAnimDelta>;
@group(0) @binding(4) var<storage, read_write> panels: array<Panel>;

const PANEL_META_SIZE: u32 = 6u;
const COLLECTION_META_OFFSET: u32 = 2u;
const REL_META_OFFSET: u32 = 0u;
const IN_COLLECTION_INDEX_META_OFFSET: u32 = 4u;
const U32_MAX: u32 = 4294967295u;
const COLLECTION_CHILDREN_LEN = 128u;

const REL_ID_FIELD = 0u;
const REL_SOURCE_FIELD = 1u;
const REL_TARGET_FIELD = 2u;
const REL_ANIMTION_FIELD = 3u;
const REL_LAYOUT = 4u;
const REL_IMMEDIATELY_ANIM = 5u;
const REL_IMMEDIATELY_PARAMS1 = 6u;
const REL_IMMEDIATELY_PARAMS2 = 7u;
const REL_IMMEDIATELY_PARAMS3 = 8u;


const REL_SIM_OFFSET:u32 = 8192u;        // rels_flat 起始偏移
const COLLECTION_SIM_OFFSET:u32 = 16384u; // collection_flat 起始偏移
// 读取 panel meta
// 获取 panel meta
fn panel_meta(panel_id: u32, field_idx: u32) -> u32 {
    return raw_data[panel_id * PANEL_META_SIZE + field_idx];
}

// 获取 panel 在 collection 中 index
fn panel_in_collection_index(panel_id: u32) -> u32 {
    return panel_meta(panel_id, IN_COLLECTION_INDEX_META_OFFSET);
}

fn get_source_first_panel(collection_id:u32) -> u32{

    return raw_data[COLLECTION_SIM_OFFSET + collection_id * COLLECTION_CHILDREN_LEN];
}

// 获取 panel 对应 rel 范围
fn get_panel_rel_range(panel_id: u32) -> vec2<u32> {
    return vec2<u32>(
        panel_meta(panel_id, REL_META_OFFSET),
        panel_meta(panel_id, REL_META_OFFSET + 1u)
    );
}

// 获取 rel 字段
fn get_rel_field(rel_idx: u32, field_offset: u32) -> u32 {
    return rels_flat[rel_idx * 4u + field_offset];
}

// 设置 rel source
fn set_rel_source(rel_idx: u32, value: u32) {
    rels_flat[rel_idx * 4u + REL_SOURCE_FIELD] = value;
}

// 设置 rel target
fn set_rel_target(rel_idx: u32, value: u32) {
    rels_flat[rel_idx * 4u + REL_TARGET_FIELD] = value;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let panel_idx = id.x;


    // // --- 输出 panel meta ---
    // for (var i: u32 = 0u; i < PANEL_META_SIZE; i = i + 1u) {
    //     debug.uints[i + panel_idx * PANEL_META_SIZE] = panel_meta(panel_idx, i);
    // }

     // 获取 panel rel 范围
    let rel_range = get_panel_rel_range(panel_idx);
    if rel_range.x == U32_MAX { return; }


    let collection_offset = get_rel_field(rel_range.x,REL_SOURCE_FIELD);
    let panel_field = get_rel_field(rel_range.x,REL_ANIMTION_FIELD);

    if(panel_field == 0u){
        return;
    }

    let source_first_panel_id = get_source_first_panel(collection_offset);
    if(source_first_panel_id == panel_idx){return; }
    if(source_first_panel_id == U32_MAX){ return; }
    
    let source_delta_position = panel_anim_delta[source_first_panel_id].delta_position;


    if(abs(source_delta_position.x) > 0.0 || abs(source_delta_position.y) > 0.0){
        panels[panel_idx].position += source_delta_position;
        panel_anim_delta[panel_idx].start_position = panels[panel_idx].position;
    }

}
