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

struct AnimtionFieldOffsetPtr {
   field_id: u32,        // 瀛楁鏍囪瘑
   start_value: f32,     // 璧峰鍊?
   target_value: f32,    // 鐩爣鍊?
   elapsed: f32,         // 宸茬粡杩囩殑鏃堕棿
   duration: f32,        // 鍔ㄧ敾鎸佺画鏃堕棿
   op: u32,              // 鎿嶄綔绫诲瀷锛圫ET/ADD/MUL/鈥︼級
   hold: u32,            // hold 鏃堕棿
   delay: f32,           // 寤惰繜鏃堕棿
   loop_count: u32,      // 寰幆娆℃暟
   ping_pong: u32,       // 寰€杩旀爣璁?
   on_complete: u32,     // 鍥炶皟鏍囪瘑
   panel_id: u32,        // Panel ID
   death: u32,           // 鏄惁缁撴潫
   easy_fn: u32,         // easing 鍑芥暟鏍囪瘑
   _pad: u32,       // 琛ラ綈16瀛楄妭瀵归綈
};

struct GpuUiCollection {
    collection_id: u32,
    items_offset: u32,
    items_len: u32,
    collection_layout_mask: u32,
    param0: f32,       // X 闂磋窛
    param1: f32,        // Y 闂磋窛
    _p1:u32,
    _p2:u32
}

struct GpuUiInfluence {
    field: u32,
    weight: f32,
    influence_type: u32,
    reserved: u32,
}


struct SharedState {
    mouse_pos: vec2<f32>,
    mouse_state: u32,
    _pad0: u32,

    hover_id: atomic<u32>,
    hover_blocked: atomic<u32>,
    _pad1: vec2<u32>,

    hover_pos: vec2<f32>,

    current_depth: u32,
    _pad2: u32,

    click_id: u32,
    click_blocked: u32,

    drag_id: u32,
    drag_blocked: u32,

    history_id: u32,
    _pad3: u32,

    _pad4: vec2<u32>,
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

 struct GpuAnimationDes {
     animation_count: u32,       // 褰撳墠甯у姩鐢绘暟閲?
     frame_count: u32,           // 姣忎釜鍔ㄧ敾甯ф暟
     start_index: u32,           // 鍔ㄧ敾鍦ㄥ叏灞€ buffer 鐨勮捣濮嬬储寮?
     _pad0: u32,                 // padding 16瀛楄妭瀵归綈

     delta_time: f32,            // 鍗曞抚鏃堕棿澧為噺
     total_time: f32,            // 鍔ㄧ敾鎬绘椂闀?
     _pad1: vec2<f32>,            // 琛ラ綈16瀛楄妭瀵归綈
}

struct GpuUiDebugReadCallBack {
    floats: array<f32,32>, // 16 * 4 = 64 瀛楄妭
    uints: array<u32,32>,  // 16 * 4 = 64 瀛楄妭
}

fn ease_linear(t: f32) -> f32 {
    return t;
}

fn ease_in_quad(t: f32) -> f32 {
    return t * t;
}

fn ease_out_quad(t: f32) -> f32 {
    return 1.0 - (1.0 - t) * (1.0 - t);
}

fn ease_in_out_quad(t: f32) -> f32 {
    if (t < 0.5) {
        return 2.0 * t * t;
    } else {
        return 1.0 - pow(-2.0 * t + 2.0, 2.0) / 2.0;
    }
}

fn ease_in_cubic(t: f32) -> f32 {
    return t * t * t;
}

fn ease_out_cubic(t: f32) -> f32 {
    let p = t - 1.0;
    return p * p * p + 1.0;
}

fn ease_in_out_cubic(t: f32) -> f32 {
    if (t < 0.5) {
        return 4.0 * t * t * t;
    } else {
        let p = 2.0 * t - 2.0;
        return 0.5 * p * p * p + 1.0;
    }
}

fn apply_easing(easing: u32, t: f32) -> f32 {
    switch (easing) {
        case 1u: { return ease_linear(t); }
        // case 2u: { return ease_in_quad(t); }
        // case 2u: { return ease_out_quad(t); }
        // case 3u: { return ease_in_out_quad(t); }
        // case 4u: { return ease_in_cubic(t); }
        // case 5u: { return ease_out_cubic(t); }
        // case 6u: { return ease_in_out_cubic(t); }
        default: { return t; } // fallback
    }
}
fn apply_op(cur: f32, old: f32, target_value: f32, eased: f32, op: u32) -> f32 {
    if ((op & 0x01u) != 0u) {
        // SET锛氫粠璧峰鍊兼彃鍊煎埌鐩爣鍊?
        return mix(old, target_value, eased);
    }
    if ((op & 0x02u) != 0u) {
        // ADD锛氶€愭笎鍦ㄥ綋鍓嶅熀纭€涓婂彔鍔?(eased 琛ㄧず姣斾緥)
        return old + target_value * eased;
    }
    if ((op & 0x04u) != 0u) {
        // MUL锛氶€愭笎缂╂斁
        return old * mix(1.0, target_value / max(old, 0.0001), eased);
    }
    if ((op & 0x08u) != 0u) {
        // LERP锛氬拰 SET 绫讳技锛屽彧鏄粠 old 鍑哄彂锛堜緥濡傚钩婊戣窡闅忥級
        return mix(old, target_value, eased);
    }
    return old;
}

struct GpuUiRelation {
    ui_relation_id: u32,       // 0
    sources: u32,              // 4
    targets: u32,              // 8
    transform_mask: u32,       // 12 -> 16 瀛楄妭杈圭晫瀵归綈
    weight: f32,               // 16
    delay: f32,                // 20
    collection_sampling: u32,  // 24
    pad: u32                  // 28 -> 鎬诲ぇ灏?32 瀛楄妭
};

@group(0) @binding(0) var<storage, read_write> panels: array<Panel>;
@group(0) @binding(1) var<storage, read_write> global_uniform: GlobalUniform;
@group(0) @binding(2) var<storage, read_write> anim_des: array<AnimtionFieldOffsetPtr>;
@group(0) @binding(3) var<storage, read_write> panel_anim_delta: array<PanelAnimDelta>;
@group(0) @binding(4) var<uniform> animtion_gpu_des:GpuAnimationDes;
@group(0) @binding(5) var<storage, read_write> debug_buffer: GpuUiDebugReadCallBack;


@group(1) @binding(0)
var<storage, read_write> raw_data: array<u32>;          
@group(1) @binding(1)
var<storage, read_write> rels_flat: array<u32>;

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



const COLLECTION_SIM_OFFSET:u32 = 16384u; // collection_flat 璧峰鍋忕Щ
fn panel_meta(panel_id: u32, field_idx: u32) -> u32 {
    return raw_data[panel_id * PANEL_META_SIZE + field_idx];
}

fn panel_in_collection_index(panel_id: u32) -> u32 {
    return panel_meta(panel_id, IN_COLLECTION_INDEX_META_OFFSET);
}

fn get_source_first_panel(collection_id:u32) -> u32{

    return raw_data[COLLECTION_SIM_OFFSET + collection_id * COLLECTION_CHILDREN_LEN];
}

fn get_panel_rel_range(panel_id: u32) -> vec2<u32> {
    return vec2<u32>(
        panel_meta(panel_id, REL_META_OFFSET),
        panel_meta(panel_id, REL_META_OFFSET + 1u)
    );
}

// 鑾峰彇 rel 瀛楁
fn get_rel_field(rel_idx: u32, field_offset: u32) -> u32 {
    return rels_flat[rel_idx * 4u + field_offset];
}

// 璁剧疆 rel source
fn set_rel_source(rel_idx: u32, value: u32) {
    rels_flat[rel_idx * 4u + REL_SOURCE_FIELD] = value;
}

// 璁剧疆 rel target
fn set_rel_target(rel_idx: u32, value: u32) {
    rels_flat[rel_idx * 4u + REL_TARGET_FIELD] = value;
}


fn calc_grid_position(index_in_collection: u32, spacing_x: f32, spacing_y: f32, items_len: u32) -> vec2<f32> {
    // 绠€鍗曟寜琛屾帓鍒楋紝鍋囪鍥哄畾琛屾暟
    let cols: u32 = 4u;
    let row: u32 = index_in_collection / cols;
    let col: u32 = index_in_collection % cols;
    return vec2(f32(col) * spacing_x, f32(row) * spacing_y);
}

fn calc_ring_position_with_center(
    index_in_collection: u32,
    radius: f32,
    start_angle: f32,
    items_len: u32,
    center: vec2<f32>
) -> vec2<f32> {
    // 姣忎釜鍏冪礌鐨勮搴︽杩?
    let angle = start_angle + 6.2831855 * f32(index_in_collection) / f32(items_len); // 2蟺

    let pos = vec2(cos(angle), sin(angle)) * radius;

    // 鍔犱笂涓績鐐瑰亸绉?
    return center + pos;
}

fn calc_ring_position(index_in_collection: u32, radius: f32, start_angle: f32, items_len: u32) -> vec2<f32> {
    let angle = start_angle + 6.2831855 * f32(index_in_collection) / f32(items_len); // 2pi = 6.2831855
    return vec2(cos(angle), sin(angle)) * radius;
}

fn get_source_first_panel_id(me_panel_idx:u32)->u32{
    let rel_range = get_panel_rel_range(me_panel_idx);
    if rel_range.x == U32_MAX { return U32_MAX; }


    let collection_offset = get_rel_field(rel_range.x,REL_SOURCE_FIELD);
    let source_first_panel_id = get_source_first_panel(collection_offset);
    if(source_first_panel_id == me_panel_idx){return U32_MAX;}
    if(source_first_panel_id == U32_MAX){ return U32_MAX;}
    return source_first_panel_id;
}

var<workgroup> center: vec2<f32>;
var<workgroup> stop_mask: atomic<u32>;

fn try_run_anim(anim_field: u32) -> bool {
    let prev_mask = atomicOr(&stop_mask, anim_field);

    if ((prev_mask & anim_field) != 0u) {
        return false;
    }

    return true;
}

const POSITION_X       : u32 = 1u;       // 0b0000_0000_0001
const POSITION_Y       : u32 = 2u;       // 0b0000_0000_0010
const SIZE_X           : u32 = 4u;       // 0b0000_0000_0100
const SIZE_Y           : u32 = 8u;       // 0b0000_0000_1000
const UV_OFFSET_X      : u32 = 16u;      // 0b0000_0001_0000
const UV_OFFSET_Y      : u32 = 32u;      // 0b0000_0010_0000
const UV_SCALE_X       : u32 = 64u;      // 0b0000_0100_0000
const UV_SCALE_Y       : u32 = 128u;     // 0b0000_1000_0000
const TRANSPARENT      : u32 = 256u;     // 0b0001_0000_0000
const ATTACH_COLLECTION: u32 = 512u;     // 0b0010_0000_0000
const PREPOSITION_X    : u32 = 1024u;    // 0b0100_0000_0000
const PREPOSITION_Y    : u32 = 2048u;    // 0b1000_0000_0000
const ALL_FIELDS       : u32 = 4095u;    // 0b1111_1111_1111
const NONE_FIELDS      : u32 = 0u;       // 0b0000_0000_0000
const PRE_POSITION_X: f32 = 9999988.0; // 涓?Rust 鐨?MAX_TIME_SEC 瀵瑰簲
const PRE_POSITION_Y: f32 = 9999989.0; // 涓?Rust 鐨?MAX_TIME_SEC 瀵瑰簲


@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;


    if (idx >= animtion_gpu_des.animation_count) {
        return;
    }
    
    let anim = anim_des[idx]; // anim: AnimtionFieldOffsetPtr


   // 璺宠繃姝讳骸鍔ㄧ敾
    if (anim_des[idx].death == 1u) {
        return;
    }

    if(anim.elapsed == 0.0 && PRE_POSITION_X != anim_des[idx].target_value && PRE_POSITION_Y != anim_des[idx].target_value){
        panel_anim_delta[anim_des[idx].panel_id].start_position = panels[anim_des[idx].panel_id].position;
    }


    if(PRE_POSITION_X == anim_des[idx].target_value){
        anim_des[idx].target_value = panel_anim_delta[anim_des[idx].panel_id].start_position.x;
    }

    
    if(PRE_POSITION_Y == anim_des[idx].target_value){
        anim_des[idx].target_value = panel_anim_delta[anim_des[idx].panel_id].start_position.y;
    }

 
    if (anim.field_id == 512u && anim.op == 32u && panels[anim.panel_id].collection_state == 16u) {

       let rel_range = get_panel_rel_range(anim.panel_id);
       let rel_layout = get_rel_field(rel_range.x,REL_LAYOUT);
       


       let source_first_panel_id = get_source_first_panel_id(anim.panel_id);

       if(source_first_panel_id == U32_MAX){
            return;
       }

       if (rel_layout == 4u) {

            let index_in_collection = panel_in_collection_index(anim.panel_id);
            let parent_pos: vec2<f32> = panels[source_first_panel_id].position + vec2<f32>(80.0,200.0);
            let parent_size: vec2<f32> = panels[source_first_panel_id].size * 0.8;
            
            let cols: u32 = 5u;
            let collection_size: u32 = 13u; // 浣犺嚜宸辫幏鍙栧瓙闈㈡澘鏁伴噺
            let rows: u32 = (collection_size + cols - 1u) / cols;
            
            let cell_size: vec2<f32> = vec2<f32>(
                parent_size.x / f32(cols),
                parent_size.y / f32(rows)
            );
            
            let row: u32 = index_in_collection / cols;
            let col: u32 = index_in_collection % cols;
            
            let target_pos: vec2<f32> = parent_pos
                + vec2<f32>(cell_size.x * (f32(col) + 0.5),
                            cell_size.y * (f32(row) + 0.5));
            
            let size: vec2<f32> = panels[source_first_panel_id].size / 2.0; // 鎴栬€呯敤瀛愰潰鏉胯嚜韬ぇ灏?
            
            panels[anim.panel_id].position = target_pos - size;
            anim_des[idx].death = 1u;
            panels[anim.panel_id].collection_state = 32u;
            panel_anim_delta[anim_des[idx].panel_id].start_position = panels[anim_des[idx].panel_id].position;
        }
    }

    // 鑾峰彇褰撳墠鍔ㄧ敾瀛楁



   // 鍒濆鍊煎～鍏咃紙鍙湪 elapsed 涓?0 鏃讹級
    if (anim_des[idx].elapsed == 0.0 && anim_des[idx].hold == 1u) {
        if (anim.field_id == 0x1u) {
            anim_des[idx].start_value = panels[anim.panel_id].position.x;
        } else if (anim.field_id == 0x2u) {
            anim_des[idx].start_value = panels[anim.panel_id].position.y;
        } else if (anim.field_id == 0x4u) {
            anim_des[idx].start_value = panels[anim.panel_id].size.x;
        } else if (anim.field_id == 0x8u) {
            anim_des[idx].start_value = panels[anim.panel_id].size.y;
        } else if (anim.field_id == 0x10u) {
            anim_des[idx].start_value = panels[anim.panel_id].uv_offset.x;
        } else if (anim.field_id == 0x20u) {
            anim_des[idx].start_value = panels[anim.panel_id].uv_offset.y;
        } else if (anim.field_id == 0x40u) {
            anim_des[idx].start_value = panels[anim.panel_id].uv_scale.x;
        } else if (anim.field_id == 0x80u) {
            anim_des[idx].start_value = panels[anim.panel_id].uv_scale.y;
        } else if (anim.field_id == 0x100u) {
            anim_des[idx].start_value = panels[anim.panel_id].transparent;
        }
    }

    // 鏇存柊 elapsed
    let new_elapsed = min(anim.elapsed + global_uniform.dt, anim.duration);

    anim_des[idx].elapsed = new_elapsed;



    // -------------------- 瀛楁鍖归厤 --------------------
    let t = clamp(anim.elapsed / max(anim.duration, 0.00001), 0.0, 1.0);
    let eased = apply_easing(anim.easy_fn, t);
    let start = anim_des[idx].start_value;
    let target_value = anim.target_value;
    let op = anim.op;

    switch anim.field_id {
        case 0x1u:  { panels[anim.panel_id].position.x = apply_op(panels[anim.panel_id].position.x, start, target_value, eased, op); }
        case 0x2u:  { panels[anim.panel_id].position.y = apply_op(panels[anim.panel_id].position.y, start, target_value, eased, op); }
        case 0x4u:  { panels[anim.panel_id].size.x     = apply_op(panels[anim.panel_id].size.x,     start, target_value, eased, op); }
        case 0x8u:  { panels[anim.panel_id].size.y     = apply_op(panels[anim.panel_id].size.y,     start, target_value, eased, op); }
        case 0x10u: { panels[anim.panel_id].uv_offset.x = apply_op(panels[anim.panel_id].uv_offset.x, start, target_value, eased, op); }
        case 0x20u: { panels[anim.panel_id].uv_offset.y = apply_op(panels[anim.panel_id].uv_offset.y, start, target_value, eased, op); }
        case 0x40u: { panels[anim.panel_id].uv_scale.x  = apply_op(panels[anim.panel_id].uv_scale.x,  start, target_value, eased, op); }
        case 0x80u: { panels[anim.panel_id].uv_scale.y  = apply_op(panels[anim.panel_id].uv_scale.y,  start, target_value, eased, op); }
        case 0x100u:{ panels[anim.panel_id].transparent = apply_op(panels[anim.panel_id].transparent, start, target_value, eased, op); }
        default: {}
    }
    // 鍔ㄧ敾瀹屾垚
    if (new_elapsed >= anim.duration) {
        anim_des[idx].death = 1u;
    }
    
  if (anim.field_id == 512u && anim.op == 32u && panels[anim.panel_id].collection_state == 16u) {
    // 绔嬪嵆璺熼殢妯″紡
            let rel_range = get_panel_rel_range(anim.panel_id);
            let rel_layout = get_rel_field(rel_range.x,REL_ANIMTION_FIELD);


            // // 璁＄畻鐜舰鍧愭爣
            // let pos: vec2<f32> = calc_ring_position_with_center(
            //     index_in_collection,
            //     collection.param0,
            //     collection.param1,
            //     collection.items_len,
            //     center
            // );

            // // debug_buffer.floats[idx] = rel[collection.items_offset].relation_idx;

            // // 鍐欏叆 panel buffer
            // panels[anim.panel_id].position = pos;
            // panels[anim.panel_id].state = 32u;
    }


    
    if (anim.field_id == 512u && anim.op == 16u && panels[anim.panel_id].collection_state == 16u) {
        // let collection = ui_collections[u32(anim.start_value)];
            let source_first_panel_id = get_source_first_panel_id(anim.panel_id);

            if(source_first_panel_id == U32_MAX){
                return;
            }


            if(anim.panel_id == source_first_panel_id) {
                anim_des[idx].death = 1u;
                panels[anim.panel_id].collection_state = 32u;
                return;
            }
            let rel_range = get_panel_rel_range(anim.panel_id);
            let rel_layout = get_rel_field(rel_range.x,REL_LAYOUT);




            let center = panels[source_first_panel_id].position;



        // let source_collection_offset = rel[collection.items_offset].sources;
        // let source_ids_offset = ui_collections[source_collection_offset].items_offset;
        // let center = panels[ui_ids[source_ids_offset]].position;
        // // 鐜舰甯冨眬
        if (rel_layout == 8u) {
            let index_in_collection = panel_in_collection_index(anim.panel_id);
            // debug_buffer.uints[anim.panel_id] = index_in_collection;
            // 鐩爣鍧愭爣
            let target_pos: vec2<f32> = calc_ring_position_with_center(
                index_in_collection,
                100.0, // radius
                5.0, // start angle
                10,
                center
            );

            let new_elapsed = min(anim.elapsed + global_uniform.dt, anim.duration);
            anim_des[idx].elapsed = new_elapsed;

            // 鍔ㄧ敾瀹屾垚鏍囪
            if (new_elapsed >= anim.duration) {
                anim_des[idx].death = 1u;
                panels[anim.panel_id].collection_state = 32u;
            }

            // 鍔ㄧ敾杩涘害
            let t = clamp(anim.elapsed / max(anim.duration, 0.00001), 0.0, 1.0);
            let eased = apply_easing(anim.easy_fn, t);

            // 璁＄畻澧為噺
            let delta: vec2<f32> = (target_pos - anim_des[idx].start_value) * eased;

            // 搴旂敤鍒?panel
            panels[anim.panel_id].position = anim_des[idx].start_value + delta;
            panel_anim_delta[anim.panel_id].delta_position = delta;
    }
    if (rel_layout == 4u) {
    // 褰撳墠 panel 鍦ㄩ泦鍚堜腑鐨勭储寮曪紙缁濆椤哄簭锛屼笉鍋氱浉瀵硅绠楋級
    let index_in_collection = panel_in_collection_index(anim.panel_id);

    // 宸︿笂瑙掕捣鐐?= source panel 鐨勪綅缃?
    let leftup: vec2<f32> = panels[source_first_panel_id].position;

    // 缃戞牸鍙傛暟
    let cols: u32 = 5u;              
    let spacing: vec2<f32> = vec2<f32>(120.0, 120.0);  // 妯旱闂磋窛

    // 鐩存帴鐢?index_in_collection 鎺掑竷
    let row: u32 = index_in_collection / cols;
    let col: u32 = index_in_collection % cols;

    // 鐩爣鍧愭爣 = 宸︿笂瑙掕捣鐐?+ 琛屽垪鍋忕Щ
    let target_pos: vec2<f32> = leftup + vec2<f32>(
        f32(col) * spacing.x,
        f32(row) * spacing.y
    );


    // === 鍔ㄧ敾閫昏緫鍚岀幆褰㈢増鏈?===
    let new_elapsed = min(anim.elapsed + global_uniform.dt, anim.duration);
    anim_des[idx].elapsed = new_elapsed;

    if (new_elapsed >= anim.duration) {
        anim_des[idx].death = 1u;
        panels[anim.panel_id].collection_state = 32u;
    }

    let t = clamp(anim.elapsed / max(anim.duration, 0.00001), 0.0, 1.0);
    let eased = apply_easing(anim.easy_fn, t);

        let delta: vec2<f32> = (target_pos - anim_des[idx].start_value) * eased;
        panels[anim.panel_id].position = anim_des[idx].start_value + delta;
        panel_anim_delta[anim.panel_id].delta_position = delta;
        }
    }
 




    // 璁＄畻鍔ㄧ敾杩涘害

    // 璁＄畻鐩爣鍊?
    

  // 璁＄畻鍔ㄧ敾杩涘害 (0~1)

    // 璁＄畻鐩爣宸€?


    // // //鍙€夛細鍐欏叆 panel_anim_delta 鍋氬閲忕疮鍔?
    // if (anim.field_id == 0x1u) {
    //     panel_anim_delta[anim.panel_id].delta_position.x += delta;
    // } else if (anim.field_id == 0x2u) {
    //     panel_anim_delta[anim.panel_id].delta_position.y += delta;
    // } else if (anim.field_id == 0x4u) {
    //     panel_anim_delta[anim.panel_id].delta_size.x += delta;
    // } else if (anim.field_id == 0x8u) {
    //     panel_anim_delta[anim.panel_id].delta_size.y += delta;
    // } else if (anim.field_id == 0x10u) {
    //     panel_anim_delta[anim.panel_id].delta_uv_offset.x += delta;
    // } else if (anim.field_id == 0x20u) {
    //     panel_anim_delta[anim.panel_id].delta_uv_offset.y += delta;
    // } else if (anim.field_id == 0x40u) {
    //     panel_anim_delta[anim.panel_id].delta_uv_scale.x += delta;
    // } else if (anim.field_id == 0x80u) {
    //     panel_anim_delta[anim.panel_id].delta_uv_scale.y += delta;
    // } else if (anim.field_id == 0x100u) {
    //     panel_anim_delta[anim.panel_id].delta_transparent += delta;
    // }



}