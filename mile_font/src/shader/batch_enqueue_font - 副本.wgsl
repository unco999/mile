struct GlyphInstruction {
    command: u32,
    _pad: u32,
    data: array<f32,6>
};

struct FontGlyphDes {
    start_idx: u32,
    end_idx: u32,
    texture_idx_x: u32,
    texture_idx_y: u32,
};

struct GpuUiDebugReadCallBack {
    floats: array<f32,32>,
    uints: array<u32,32>,
};

@group(0) @binding(0)
var font_distance_texture: texture_storage_2d<rgba16float, write>;

@group(0) @binding(1)
var<storage, read> glyph_instructions: array<GlyphInstruction>;

@group(0) @binding(2)
var<storage, read_write> glyph_descs: array<FontGlyphDes>;

@group(0) @binding(3)
var<storage, read_write> debug_buffer: GpuUiDebugReadCallBack;

const GLYPH_COMMAND_MOVE_TO: u32 = 0u;
const GLYPH_COMMAND_LINE_TO: u32 = 1u;
const GLYPH_COMMAND_QUAD_TO: u32 = 2u;
const GLYPH_COMMAND_CURVE_TO: u32 = 3u;
const GLYPH_COMMAND_CLOSE: u32 = 4u;

const GLYPH_SIZE: u32 = 64u;
const GLYPH_SIZE_F: f32 = 64.0;

// 点到线段距离
fn point_segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    let proj = a + ab * t;
    return length(p - proj);
}

// 坐标变换函数 - 统一处理
fn transform_point(raw_point: vec2<f32>) -> vec2<f32> {
    // 将2048设计空间的坐标映射到0-64纹理空间
    return raw_point * (GLYPH_SIZE_F / 2048.0);
}

fn is_point_inside_pixel(p: vec2<f32>, desc: FontGlyphDes) -> bool {
    var crossings: i32 = 0i;
    var current_point = vec2<f32>(0.0);
    var start_point = vec2<f32>(0.0);
    var has_start_point = false;
    var path_started = false;
    
    for (var i = desc.start_idx; i < desc.end_idx; i = i + 1u) {
        let instr = glyph_instructions[i];
        let command = instr.command;
        
        if (command == GLYPH_COMMAND_MOVE_TO) {
            let raw_point = vec2<f32>(instr.data[0], instr.data[1]);
            current_point = transform_point(raw_point);
            if (!path_started) {
                start_point = current_point;
                has_start_point = true;
                path_started = true;
            }
            
        } else if (command == GLYPH_COMMAND_LINE_TO) {
            let raw_point = vec2<f32>(instr.data[0], instr.data[1]);
            let next_point = transform_point(raw_point);
            
            // 检查线段与水平射线的交点
            if (abs(current_point.y - next_point.y) > 0.001) {
                if ((current_point.y < p.y && next_point.y >= p.y) || 
                    (current_point.y >= p.y && next_point.y < p.y)) {
                    // 计算交点X坐标
                    let t = (p.y - current_point.y) / (next_point.y - current_point.y);
                    let intersect_x = current_point.x + t * (next_point.x - current_point.x);
                    
                    // 如果交点在射线右侧，增加交叉计数
                    if (intersect_x > p.x) {
                        crossings = crossings + 1i;
                    }
                }
            }
            current_point = next_point;
            
        } else if (command == GLYPH_COMMAND_QUAD_TO) {
            let raw_control = vec2<f32>(instr.data[2], instr.data[3]);
            let raw_end = vec2<f32>(instr.data[0], instr.data[1]);
            
            let control = transform_point(raw_control);
            let end = transform_point(raw_end);
            
            // 将贝塞尔曲线离散成直线段
            let segments = 16u;
            var prev_point = current_point;
            for (var j = 1u; j <= segments; j = j + 1u) {
                let t = f32(j) / f32(segments);
                let next_segment_point = (1.0 - t) * (1.0 - t) * current_point + 
                                       2.0 * (1.0 - t) * t * control + 
                                       t * t * end;
                
                // 对每个线段段应用相同的射线检测
                if (abs(prev_point.y - next_segment_point.y) > 0.001) {
                    if ((prev_point.y < p.y && next_segment_point.y >= p.y) || 
                        (prev_point.y >= p.y && next_segment_point.y < p.y)) {
                        let t_intersect = (p.y - prev_point.y) / (next_segment_point.y - prev_point.y);
                        let intersect_x = prev_point.x + t_intersect * (next_segment_point.x - prev_point.x);
                        if (intersect_x > p.x) {
                            crossings = crossings + 1i;
                        }
                    }
                }
                prev_point = next_segment_point;
            }
            current_point = end;
            
        } else if (command == GLYPH_COMMAND_CLOSE) {
            if (has_start_point) {
                // 闭合路径：从当前点回到起点
                if (abs(current_point.y - start_point.y) > 0.001) {
                    if ((current_point.y < p.y && start_point.y >= p.y) || 
                        (current_point.y >= p.y && start_point.y < p.y)) {
                        let t = (p.y - current_point.y) / (start_point.y - current_point.y);
                        let intersect_x = current_point.x + t * (start_point.x - current_point.x);
                        if (intersect_x > p.x) {
                            crossings = crossings + 1i;
                        }
                    }
                }
                current_point = start_point;
            }
            path_started = false;
            has_start_point = false;
        }
    }
    
    // 奇偶规则：奇数个交叉点在内部，偶数个在外部
    return (crossings % 2) == 1;
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let glyph_idx: u32 = 0u;
    let desc = glyph_descs[glyph_idx];

    if (desc.start_idx == desc.end_idx) {
        return;
    }


    

    let local_x = id.x % GLYPH_SIZE;
    let local_y = id.y % GLYPH_SIZE;
    
    let px_local = vec2<f32>(f32(local_x), f32(local_y));
    let px_global = vec2<i32>(i32(local_x), i32(local_y));

    if (debug_buffer.floats[0] <= px_local.x) {
        debug_buffer.floats[0] = px_local.x;
    }
    
    if (debug_buffer.floats[1] <= px_local.y) {
        debug_buffer.floats[1] = px_local.y;
    }

    var min_dist: f32 = 1000.0;
    var current_point = vec2<f32>(0.0);
    var path_started = false;
    
    // 处理轮廓指令计算最小距离
    for (var i = desc.start_idx; i < desc.end_idx; i = i + 1u) {
        let instr = glyph_instructions[i];
        let command = instr.command;
        
        if (command == GLYPH_COMMAND_MOVE_TO) {
            let raw_point = vec2<f32>(instr.data[0], instr.data[1]);
            current_point = transform_point(raw_point);
            path_started = true;
            
        } else if (command == GLYPH_COMMAND_LINE_TO && path_started) {
            let raw_point = vec2<f32>(instr.data[0], instr.data[1]);
            let next_point = transform_point(raw_point);
            
            let dist = point_segment_distance(px_local, current_point, next_point);
            min_dist = min(min_dist, dist);
            current_point = next_point;
            
        } else if (command == GLYPH_COMMAND_QUAD_TO && path_started) {
            let raw_control = vec2<f32>(instr.data[2], instr.data[3]);
            let raw_end = vec2<f32>(instr.data[0], instr.data[1]);
            
            let control = transform_point(raw_control);
            let end = transform_point(raw_end);
            
            // 离散化贝塞尔曲线计算距离
            let segments = 16u;
            var prev_point = current_point;
            for (var j = 1u; j <= segments; j = j + 1u) {
                let t = f32(j) / f32(segments);
                let next_point = (1.0 - t) * (1.0 - t) * current_point + 
                                2.0 * (1.0 - t) * t * control + 
                                t * t * end;
                let dist = point_segment_distance(px_local, prev_point, next_point);
                min_dist = min(min_dist, dist);
                prev_point = next_point;
            }
            current_point = end;
            
        } else if (command == GLYPH_COMMAND_CLOSE) {
            path_started = false;
        }
    }

    let is_inside = is_point_inside_pixel(px_local, desc);
    var signed_distance: f32;
    if (is_inside) {
        signed_distance = -min_dist;
    } else {
        signed_distance = min_dist;
    }
    
    // 归一化SDF值到[0,1]范围
    let normalized_sdf = clamp(signed_distance / 2 + 0.5, 0.0, 1.0);
    
    // 存储到纹理
    textureStore(font_distance_texture, px_global, vec4<f32>(normalized_sdf, 0.0, 0.0, 1.0));

}