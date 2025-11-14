// ASCII only. No BOM.
// Batch SDF enqueue compute shader

struct GlyphInstruction {
    command: u32,
    _pad: u32,
    data: array<f32, 6>,
};

struct FontGlyphDes {
    start_idx: u32,
    end_idx: u32,
    texture_idx_x: u32,
    texture_idx_y: u32,

    // bbox
    x_min: i32,
    y_min: i32,
    x_max: i32,
    y_max: i32,

    // metrics
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
    floats: array<f32, 32>,
    uints: array<u32, 32>,
};

struct BatchParams {
    base_index: u32,
    glyph_count: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0)
var font_distance_texture: texture_storage_2d<rgba16float, write>;

@group(0) @binding(1)
var<storage, read> glyph_instructions: array<GlyphInstruction>;

@group(0) @binding(2)
var<storage, read_write> glyph_descs: array<FontGlyphDes>;

@group(0) @binding(3)
var<storage, read_write> debug_buffer: GpuUiDebugReadCallBack;

@group(0) @binding(4)
var<uniform> params: BatchParams;

const GLYPH_COMMAND_MOVE_TO: u32 = 0u;
const GLYPH_COMMAND_LINE_TO: u32 = 1u;
const GLYPH_COMMAND_QUAD_TO: u32 = 2u;
const GLYPH_COMMAND_CURVE_TO: u32 = 3u;
const GLYPH_COMMAND_CLOSE: u32 = 4u;

const GLYPH_SIZE: u32 = 64u;
const GLYPH_SIZE_F: f32 = 64.0;

fn point_segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    let proj = a + ab * t;
    return length(p - proj);
}

fn transform_point(raw_point: vec2<f32>, desc: FontGlyphDes) -> vec2<f32> {
    let x_min = f32(desc.x_min);
    let y_min = f32(desc.y_min);
    let x_max = f32(desc.x_max);
    let y_max = f32(desc.y_max);

    let bbox_width = x_max - x_min;
    let bbox_height = y_max - y_min;

    let scale_x = GLYPH_SIZE_F / max(bbox_width, 1.0);
    let scale_y = GLYPH_SIZE_F / max(bbox_height, 1.0);
    let scale = min(scale_x, scale_y);

    var px = vec2<f32>(
        (raw_point.x - x_min) * scale,
        (y_max - raw_point.y) * scale
    );

    let scaled_width = bbox_width * scale;
    let scaled_height = bbox_height * scale;
    px.x += (GLYPH_SIZE_F - scaled_width) * 0.5;
    px.y += (GLYPH_SIZE_F - scaled_height) * 0.5;

    return px;
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
            current_point = transform_point(raw_point, desc);
            start_point = current_point;
            has_start_point = true;
            path_started = true;

        } else if (command == GLYPH_COMMAND_LINE_TO && path_started) {
            let raw_point = vec2<f32>(instr.data[0], instr.data[1]);
            let end = transform_point(raw_point, desc);

            // crossing test
            if ((current_point.y <= p.y && end.y > p.y) ||
                (current_point.y > p.y && end.y <= p.y)) {
                let t = (p.y - current_point.y) / (end.y - current_point.y);
                if (t >= 0.0 && t <= 1.0) {
                    let intersect_x = current_point.x + t * (end.x - current_point.x);
                    if (intersect_x >= p.x) {
                        crossings = crossings + 1i;
                    }
                }
            }
            current_point = end;

        } else if (command == GLYPH_COMMAND_QUAD_TO && path_started) {
            let raw_control = vec2<f32>(instr.data[2], instr.data[3]);
            let raw_end = vec2<f32>(instr.data[0], instr.data[1]);

            let control = transform_point(raw_control, desc);
            let end = transform_point(raw_end, desc);

            let segments = 12u;
            var prev_point = current_point;
            for (var j = 1u; j <= segments; j = j + 1u) {
                let t = f32(j) / f32(segments);
                let next_point =
                    (1.0 - t) * (1.0 - t) * current_point +
                    2.0 * (1.0 - t) * t * control +
                    t * t * end;
                if ((prev_point.y <= p.y && next_point.y > p.y) ||
                    (prev_point.y > p.y && next_point.y <= p.y)) {
                    let tt = (p.y - prev_point.y) / (next_point.y - prev_point.y);
                    if (tt >= 0.0 && tt <= 1.0) {
                        let intersect_x = prev_point.x + tt * (next_point.x - prev_point.x);
                        if (intersect_x >= p.x) {
                            crossings = crossings + 1i;
                        }
                    }
                }
                prev_point = next_point;
            }
            current_point = end;

        } else if (command == GLYPH_COMMAND_CLOSE) {
            if (has_start_point) {
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

    return (crossings % 2) == 1;
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Use z slice per glyph
    if (id.z >= params.glyph_count) { return; }
 
    let local_x = id.x % GLYPH_SIZE;
    let local_y = id.y % GLYPH_SIZE;

    let glyph_idx: u32 = params.base_index + id.z;
    let desc = glyph_descs[glyph_idx];

    if (desc.start_idx == desc.end_idx) {
        return;
    }

    let px_local = vec2<f32>(f32(local_x), f32(local_y));
    // apply atlas tile offsets
    let tile_x: i32 = i32(desc.texture_idx_x) * i32(GLYPH_SIZE);
    let tile_y: i32 = i32(desc.texture_idx_y) * i32(GLYPH_SIZE);
    let px_global = vec2<i32>(tile_x + i32(local_x), tile_y + i32(local_y));

    var min_dist: f32 = 1e9;
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
            let segments = 12u;
            var prev_point = current_point;
            for (var j = 1u; j <= segments; j = j + 1u) {
                let t = f32(j) / f32(segments);
                let next_point =
                    (1.0 - t) * (1.0 - t) * current_point +
                    2.0 * (1.0 - t) * t * control +
                    t * t * end;
                let dist = point_segment_distance(px_local, prev_point, next_point);
                min_dist = min(min_dist, dist);
                prev_point = next_point;
            }
            current_point = end;

        } else if (command == GLYPH_COMMAND_CLOSE && path_started && has_start_point) {
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

    let max_dist = 1.5;
    let normalized_sdf = clamp(0.5 - signed_distance / (2.0 * max_dist), 0.0, 1.0);
    let enhanced_sdf = clamp((normalized_sdf - 0.5) * 1.2 + 0.5, 0.0, 1.0);

    textureStore(font_distance_texture, px_global, vec4<f32>(enhanced_sdf, enhanced_sdf, enhanced_sdf, 1.0));
}
