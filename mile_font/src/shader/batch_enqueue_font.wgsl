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
    
    // 边界框
    x_min: i32,
    y_min: i32, 
    x_max: i32,
    y_max: i32,
    
    // 关键度量字段
    units_per_em: u32,
    ascent: i32,
    descent: i32,
    line_gap: i32,
    advance_width: u32,
    left_side_bearing: i32,
    glyph_advance_width: u32,
    glyph_left_side_bearing: i32,
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
// 正确的坐标变换函数 - 处理TTF坐标系
fn transform_point(raw_point: vec2<f32>, desc: FontGlyphDes) -> vec2<f32> {
    let x_min = f32(desc.x_min);
    let y_min = f32(desc.y_min); 
    let x_max = f32(desc.x_max);
    let y_max = f32(desc.y_max);
    let ascent = f32(desc.ascent);
    let descent = f32(desc.descent);
    

    
    // 计算字形的实际宽度和高度
    let bbox_width = x_max - x_min;
    let bbox_height = y_max - y_min;
    
    // 计算缩放比例 - 基于实际字形大小，不是整个设计空间
    let scale_x = GLYPH_SIZE_F / max(bbox_width, 1.0);
    let scale_y = GLYPH_SIZE_F / max(bbox_height, 1.0);
    let scale = min(scale_x, scale_y);
    
    // 关键：将点从字形局部坐标系转换
    // 相对于字形边界框的左上角
    var px = vec2<f32>(
        (raw_point.x - x_min) * scale,  // x: 从左边界开始
        (y_max - raw_point.y) * scale   // y: 翻转y轴，从上边界开始
    );
    
    // 居中
    let scaled_width = bbox_width * scale;
    let scaled_height = bbox_height * scale;
    px.x += (GLYPH_SIZE_F - scaled_width) * 0.5;
    px.y += (GLYPH_SIZE_F - scaled_height) * 0.5;
    
    return px;
}

// 或者使用简化的固定缩放方案（如果上面的方法还有问题）：
fn transform_point_simple(raw_point: vec2<f32>, desc: FontGlyphDes) -> vec2<f32> {
    let units_per_em = f32(desc.units_per_em);
    
    // 将TTF坐标归一化到[0,1]范围
    let normalized_x = (raw_point.x + 200.0) / 1500.0; // 调整偏移和缩放
    let normalized_y = (raw_point.y + 200.0) / 1500.0;
    
    // 缩放到glyph大小
    var px = vec2<f32>(normalized_x * GLYPH_SIZE_F, normalized_y * GLYPH_SIZE_F);
    
    // 确保在纹理范围内
    px = clamp(px, vec2<f32>(2.0), vec2<f32>(GLYPH_SIZE_F - 2.0));
    
    return px;
}

// 修复的内部点检测函数
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
            current_point = transform_point(raw_point, desc);
            start_point = current_point;
            has_start_point = true;
            path_started = true;
            
        } else if (command == GLYPH_COMMAND_LINE_TO) {
            let raw_point = vec2<f32>(instr.data[0], instr.data[1]);
            let next_point = transform_point(raw_point, desc);
            
            // 修复的射线交叉检测
            if ((current_point.y <= p.y && next_point.y > p.y) || 
                (current_point.y > p.y && next_point.y <= p.y)) {
                
                // 计算交点X坐标
                let t = (p.y - current_point.y) / (next_point.y - current_point.y);
                if (t >= 0.0 && t <= 1.0) {
                    let intersect_x = current_point.x + t * (next_point.x - current_point.x);
                    
                    // 关键修复：只在交点严格在右侧时计数
                    if (intersect_x >= p.x) {
                        crossings = crossings + 1i;
                    }
                }
            }
            current_point = next_point;
            
        } else if (command == GLYPH_COMMAND_QUAD_TO) {
            let raw_control = vec2<f32>(instr.data[2], instr.data[3]);
            let raw_end = vec2<f32>(instr.data[0], instr.data[1]);
            
            let control = transform_point(raw_control, desc);
            let end = transform_point(raw_end, desc);
            
            // 将贝塞尔曲线离散成直线段
            let segments = 12u;
            var prev_point = current_point;
            for (var j = 1u; j <= segments; j = j + 1u) {
                let t = f32(j) / f32(segments);
                let next_segment_point = (1.0 - t) * (1.0 - t) * current_point + 
                                       2.0 * (1.0 - t) * t * control + 
                                       t * t * end;
                
                // 对每个线段段应用相同的射线检测
                if ((prev_point.y <= p.y && next_segment_point.y > p.y) || 
                    (prev_point.y > p.y && next_segment_point.y <= p.y)) {
                    
                    let t_intersect = (p.y - prev_point.y) / (next_segment_point.y - prev_point.y);
                    if (t_intersect >= 0.0 && t_intersect <= 1.0) {
                        let intersect_x = prev_point.x + t_intersect * (next_segment_point.x - prev_point.x);
                        if (intersect_x >= p.x) {
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
                if ((current_point.y <= p.y && start_point.y > p.y) || 
                    (current_point.y > p.y && start_point.y <= p.y)) {
                    
                    let t = (p.y - current_point.y) / (start_point.y - current_point.y);
                    if (t >= 0.0 && t <= 1.0) {
                        let intersect_x = current_point.x + t * (start_point.x - current_point.x);
                        if (intersect_x >= p.x) {
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



    var min_dist: f32 = 1000.0;
    var current_point = vec2<f32>(0.0);
    var start_point = vec2<f32>(0.0);
    var has_start_point = false;
    var path_started = false;
    
    // 处理轮廓指令计算最小距离
    for (var i = desc.start_idx; i < desc.end_idx; i = i + 1u) {
        let instr = glyph_instructions[i];
        let command = instr.command;
        
        if (command == GLYPH_COMMAND_MOVE_TO) {
            let raw_point = vec2<f32>(instr.data[0], instr.data[1]);
            current_point = transform_point(raw_point, desc);
            start_point = current_point;
            has_start_point = true;
            path_started = true;
            
        } else if (command == GLYPH_COMMAND_LINE_TO && path_started) {
            let raw_point = vec2<f32>(instr.data[0], instr.data[1]);
            let next_point = transform_point(raw_point, desc);
            
            let dist = point_segment_distance(px_local, current_point, next_point);
            min_dist = min(min_dist, dist);
            current_point = next_point;
            
        } else if (command == GLYPH_COMMAND_QUAD_TO && path_started) {
            let raw_control = vec2<f32>(instr.data[2], instr.data[3]);
            let raw_end = vec2<f32>(instr.data[0], instr.data[1]);
            
            let control = transform_point(raw_control, desc);
            let end = transform_point(raw_end, desc);
            
            // 离散化贝塞尔曲线计算距离
            let segments = 12u;
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
            
        } else if (command == GLYPH_COMMAND_CLOSE && path_started && has_start_point) {
            // 闭合路径：计算到起点的距离
            let dist = point_segment_distance(px_local, current_point, start_point);
            min_dist = min(min_dist, dist);
            current_point = start_point;
            path_started = false;
            has_start_point = false;
        }
    }

    let is_inside = is_point_inside_pixel(px_local, desc);
    var signed_distance: f32;
    if (is_inside) {
        signed_distance = -min_dist;
    } else {
        signed_distance = min_dist;
    }
    
    let max_dist = 1.5; // 更小的范围获得更锐利的边缘
    
    // 内部：接近1.0，外部：接近0.0
    let normalized_sdf = clamp(0.5 - signed_distance / (2.0 * max_dist), 0.0, 1.0);
    
    // 可选：非线性变换增强对比度
    let contrast = 1.2;
    let enhanced_sdf = clamp((normalized_sdf - 0.5) * contrast + 0.5, 0.0, 1.0);
    
    textureStore(font_distance_texture, px_global, vec4<f32>(enhanced_sdf, enhanced_sdf, enhanced_sdf, 1.0));
}