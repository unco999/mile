use bytemuck::{Pod, Zeroable};
use mile_api::Computeable;
use wgpu::{util::DeviceExt};

use crate::{core::{BinaryOp, UnaryFunc, dsl::vec3}, mat::op::{MatOp, Matrix, MatrixPlan, compile_to_matrix_plan}};

// -------------------- GPU ABI: Header / Stage / Recipe --------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MKHeaderSimple {
    pub L: u32,                 // lanes（列）
    pub final_v_len: u32,       // V 的行数
    pub topouts_count: u32,     // 顶层输出行个数
    pub flags: u32,             // 预留

    pub elm_stage_count: u32,   // 逐元素阶段数
    pub elm_stage_off: u32,     // Arena 偏移（u32 words）：逐元素阶段表起点
    pub recipe_off: u32,        // Arena 偏移（u32 words）：配方表起点
    pub recipe_stride_words: u32, // 每条配方的 u32 数（固定8）

    pub topouts_rows_off: u32,  // Arena 偏移：顶层输出行索引数组起点
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuElemwiseStage {
    pub out_start: u32,   // 写回 V 的起始行
    pub rows: u32,        // 本阶段输出行数
    pub recipe_base: u32, // 在配方表的起始偏移（以“条”为单位）
    pub _pad: u32,
}

/// 每行一条配方，固定 8 个 u32：
/// [code, arity, src0, src1, src2, p0_bits, p1_bits, p2_bits]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuRecipe {
    pub code: u32,
    pub arity: u32,
    pub src0: u32,
    pub src1: u32,
    pub src2: u32,
    pub p0_bits: u32,
    pub p1_bits: u32,
    pub p2_bits: u32,
}

// -------------------- 编译期辅助 --------------------

fn pick_src_row(m: &Matrix, r: usize) -> usize {
    // 每行 one-hot：找到第一个非零列
    let row = m.row_slice(r);
    if let Some(c) = row.iter().position(|&v| v != 0.0) { c } else { panic!("empty select row") }
}

fn map_unary_code(f: &UnaryFunc) -> u32 { use UnaryFunc::*; match f {
    Sin=>1, Cos=>2, Tan=>3, Exp=>4, Log=>5, Sqrt=>6, Abs=>7
}}

fn map_binary_code(b: &BinaryOp) -> u32 { use BinaryOp::*; match b {
    Add=>10, Subtract=>11, Multiply=>12, Divide=>13, Modulo=>14, Pow=>15,
    GreaterThan=>20, GreaterEqual=>21, LessThan=>22, LessEqual=>23, Equal=>24, NotEqual=>25
}}

// -------------------- Host 构建产物 --------------------

pub struct MileSimpleBuild {
    pub header: MKHeaderSimple,
    pub arena_u32: Vec<u32>,               // [elm_stages][recipes][topouts_rows]
    pub v_init_rows: Vec<(u32, Vec<f32>)>, // (row_idx, data[L]) 变量行
    pub const_rows: Vec<(u32, f32)>,       // (row_idx, value)   常量行（整行填 value）
    pub elm_rows_each: Vec<u32>,           // 每个 ElemwiseStage 的 rows（dispatch 计算用）
}

pub fn plan_to_mile_simple_empty(plan: &MatrixPlan, lanes: u32) -> MileSimpleBuild {
    // 和 plan_to_mile_simple 相同，但不写 inputs/consts
    let mut base = plan_to_mile_simple(plan, lanes, &[]);
    base.v_init_rows.clear();
    base.const_rows.clear();
    base
}

pub fn plan_to_mile_simple(
    plan: &MatrixPlan,
    lanes: u32,
    inputs: &[Vec<f32>], // 按变量行顺序
) -> MileSimpleBuild {
    // 1) 汇总逐元素阶段与配方
    let mut elm_stages: Vec<GpuElemwiseStage> = Vec::new();
    let mut recipes: Vec<GpuRecipe> = Vec::new();
    let mut elm_rows_each: Vec<u32> = Vec::new();

    for op in &plan.ops {
        match op {
            MatOp::UnaryMat { func, mat, out_start, rows } => {
                let recipe_base = recipes.len() as u32;
                for r in 0..*rows {
                    let src0 = pick_src_row(&plan.matrices[*mat], r) as u32;
                    recipes.push(GpuRecipe { code: map_unary_code(func), arity: 1,
                        src0, src1: 0, src2: 0, p0_bits: 0, p1_bits: 0, p2_bits: 0 });
                }
                elm_stages.push(GpuElemwiseStage {
                    out_start: *out_start as u32,
                    rows: *rows as u32,
                    recipe_base,
                    _pad: 0
                });
                elm_rows_each.push(*rows as u32);
            }
            MatOp::BinaryMat { op, left_mat, right_mat, out_start, rows } => {
                let recipe_base = recipes.len() as u32;
                for r in 0..*rows {
                    let s0 = pick_src_row(&plan.matrices[*left_mat], r) as u32;
                    let s1 = pick_src_row(&plan.matrices[*right_mat], r) as u32;
                    recipes.push(GpuRecipe { code: map_binary_code(op), arity: 2,
                        src0: s0, src1: s1, src2: 0, p0_bits: 0, p1_bits: 0, p2_bits: 0 });
                }
                elm_stages.push(GpuElemwiseStage {
                    out_start: *out_start as u32,
                    rows: *rows as u32,
                    recipe_base,
                    _pad: 0
                });
                elm_rows_each.push(*rows as u32);
            }
            MatOp::CondBlendMat { cond_mat, then_mat, else_mat, out_start, rows } => {
                let recipe_base = recipes.len() as u32;
                for r in 0..*rows {
                    let sc = pick_src_row(&plan.matrices[*cond_mat], r) as u32;
                    let st = pick_src_row(&plan.matrices[*then_mat], r) as u32;
                    let se = pick_src_row(&plan.matrices[*else_mat], r) as u32;
                    recipes.push(GpuRecipe { code: 30, arity: 3,
                        src0: sc, src1: st, src2: se, p0_bits: 0, p1_bits: 0, p2_bits: 0 });
                }
                elm_stages.push(GpuElemwiseStage {
                    out_start: *out_start as u32,
                    rows: *rows as u32,
                    recipe_base,
                    _pad: 0
                });
                elm_rows_each.push(*rows as u32);
            }
        }
    }

    // 2) 组装 Arena（u32 流）：[elm_stages][recipes][topouts_rows]
    let mut arena_u32: Vec<u32> = Vec::new();
    let elm_stage_off = arena_u32.len() as u32;
    bytemuck::cast_slice::<GpuElemwiseStage, u32>(&elm_stages)
        .iter().for_each(|w| arena_u32.push(*w));

    let recipe_off = arena_u32.len() as u32;
    let recipe_stride_words = (std::mem::size_of::<GpuRecipe>() / 4) as u32; // 固定 8
    bytemuck::cast_slice::<GpuRecipe, u32>(&recipes)
        .iter().for_each(|w| arena_u32.push(*w));

    let topouts_rows_off = arena_u32.len() as u32;
    let topouts_rows: Vec<u32> = plan.top_outputs.iter().map(|&i| i as u32).collect();
    arena_u32.extend_from_slice(&topouts_rows);

    // 3) Header
    let header = MKHeaderSimple {
        L: lanes,
        final_v_len: plan.final_v_len as u32,
        topouts_count: topouts_rows.len() as u32,
        flags: 0,

        elm_stage_count: elm_stages.len() as u32,
        elm_stage_off,
        recipe_off,
        recipe_stride_words,

        topouts_rows_off,
        _pad0: 0, _pad1: 0, _pad2: 0
    };

    // 4) 初始化 V：变量行、常量行
    let mut v_init_rows: Vec<(u32, Vec<f32>)> = Vec::new();
    for (i, row) in inputs.iter().enumerate() {
        assert_eq!(row.len() as u32, lanes, "input lanes mismatch");
        v_init_rows.push((i as u32, row.clone()));
    }
    let const_rows: Vec<(u32, f32)> =
        plan.constant_values.iter().map(|(idx, val)| (*idx as u32, *val)).collect();

    MileSimpleBuild { header, arena_u32, v_init_rows, const_rows, elm_rows_each }
}

// -------------------- GPU 资源打包 --------------------

pub struct MileSimpleGPU {
    pub header_cpu: MKHeaderSimple,  // 方便 readback/调度
    pub header_buf: wgpu::Buffer,
    pub arena_buf:  wgpu::Buffer,
    pub v_buf:      wgpu::Buffer,
    pub bg_layout:  wgpu::BindGroupLayout,
    pub bindgroup:  wgpu::BindGroup,
    pub pipeline:   wgpu::ComputePipeline,
    pub L: u32,
}

pub fn build_mile_simple_gpu(
    device: &wgpu::Device,
    queue:  &wgpu::Queue,
    shader_module: &wgpu::ShaderModule, // 你的 MK.elemwise.wgsl
    build: &MileSimpleBuild,            // plan_to_mile_simple 的产物
) -> MileSimpleGPU {
    // ---------------- 1) Buffers ----------------
    // Header 固定大小
    let header_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MK.header"),
        size: std::mem::size_of::<MKHeaderSimple>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&header_buf, 0, bytemuck::bytes_of(&build.header));

    // Arena 允许为空：空时给一个占位 u32=0，避免 0 字节绑定
    let arena_data: &[u32] = if build.arena_u32.is_empty() {
        &[0u32]
    } else {
        &build.arena_u32
    };
    let arena_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MK.arena"),
        contents: bytemuck::cast_slice(arena_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // V 至少 4 字节
    let v_size = (build.header.final_v_len as u64 * build.header.L as u64 * 4).max(4);
    let v_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MK.V"),
        size: v_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 初始化变量行
    for (row, data) in &build.v_init_rows {
        debug_assert_eq!(data.len() as u32, build.header.L);
        let byte_off = (*row as u64) * (build.header.L as u64) * 4;
        queue.write_buffer(&v_buf, byte_off, bytemuck::cast_slice(&data));
    }
    // 初始化常量行（整行填同一个常量）
    if !build.const_rows.is_empty() {
        let mut scratch = vec![0.0f32; build.header.L as usize];
        for (row, val) in &build.const_rows {
            scratch.fill(*val);
            let byte_off = (*row as u64) * (build.header.L as u64) * 4;
            queue.write_buffer(&v_buf, byte_off, bytemuck::cast_slice(&scratch));
        }
    }

    // ---------------- 2) BindGroup ----------------
    let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MK.bgl0"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MK.bg0"),
        layout: &bg_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: header_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: arena_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: v_buf.as_entire_binding() },
        ],
    });

    // ---------------- 3) Pipeline ----------------
    let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MK.pl"),
        bind_group_layouts: &[&bg_layout],
        push_constant_ranges: &[],
    });

    // wgpu 0.20: entry_point 是 &str；cache 用 None 即可
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MK.elemwise.pipeline"),
        layout: Some(&pl_layout),
        module: shader_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    MileSimpleGPU {
        header_cpu: build.header,
        header_buf,
        arena_buf,
        v_buf,
        bg_layout,
        bindgroup,
        pipeline,
        L: build.header.L,
    }
}

// -------------------- 内核持有者 + Computeable 实现 --------------------

pub struct GpuKennel {
    pub mk: MileSimpleGPU,
    pub elm_stage_count: u32,
    pub rows_each: Vec<u32>,
}

pub fn empty_build() -> MileSimpleBuild {
    MileSimpleBuild {
        header: MKHeaderSimple {
            L: 1, final_v_len: 1, topouts_count: 0, flags: 0,
            elm_stage_count: 0, elm_stage_off: 0, recipe_off: 0, recipe_stride_words: 8,
            topouts_rows_off: 0, _pad0:0, _pad1:0, _pad2:0,
        },
        arena_u32: vec![],         // 没有阶段/配方
        v_init_rows: vec![],       // 暂不写变量行
        const_rows: vec![],        // 暂不写常量行
        elm_rows_each: vec![],     // 0
    }
}

impl GpuKennel {

    pub fn test_entry(&mut self,device: &wgpu::Device,queue:  &wgpu::Queue){
        let expr = vec3(1.0,1.0,1.0) + vec3(2.0, 2.0, 2.0);
        let plan = compile_to_matrix_plan(&expr,&[]);
        self.set_plan(device, queue, &plan, 3, &[]);
    }

    #[inline]
    pub fn compute(&self,device: &wgpu::Device,queue:  &wgpu::Queue){
        let (wg_x, wg_y, wg_z) = self.workgroup_size();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("MK.run") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { 
            label: Some("MK.elemwise.pass"), 
            timestamp_writes: Default::default(),
            });

            cpass.set_pipeline(&self.mk.pipeline);
            cpass.set_bind_group(0, &self.mk.bindgroup, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y.max(1), wg_z.max(1));
         }
        queue.submit([encoder.finish()]);
    }

     pub fn set_plan(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &MatrixPlan,
        lanes: u32,
        inputs: &[Vec<f32>], // 可为空
    ) {
        // 1) 编译 plan -> build（Arena/Header/V 初始化列表）
        let build = plan_to_mile_simple(plan, lanes, inputs);

        // 2) 如果新数据能装进现有 buffer，就直接覆盖；否则重建 buffer
        // Header 大小固定：直接覆盖
        queue.write_buffer(&self.mk.header_buf, 0, bytemuck::bytes_of(&build.header));

        // Arena：如果容量不够（u32 数变化），重建；否则覆盖
        let new_arena_bytes = (build.arena_u32.len() * 4) as u64;
        let mut need_rebuild_arena = false;
        {
            // wgpu 没有直接取 buffer 容量的 API，通常你会把容量存在 struct 里
            // 这里假设你在 MileSimpleGPU 里再加一个 arena_capacity 字段去记录
        }

        // 简化：直接重建 arena（实现最简单）
        self.mk.arena_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MK.arena"),
            contents: bytemuck::cast_slice(&build.arena_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 3) V：判断大小，必要时重建（final_v_len * L * 4）
        let new_v_size = build.header.final_v_len as u64 * build.header.L as u64 * 4;
        // 同理，简化：直接重建 V
        self.mk.v_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MK.V"),
            size: new_v_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 初始化变量/常量行
        for (row, data) in &build.v_init_rows {
            queue.write_buffer(&self.mk.v_buf, (*row as u64) * build.header.L as u64 * 4, bytemuck::cast_slice(data));
        }
        for (row, val) in &build.const_rows {
            let vecv = vec![*val; build.header.L as usize];
            queue.write_buffer(&self.mk.v_buf, (*row as u64) * build.header.L as u64 * 4, bytemuck::cast_slice(&vecv));
        }

        // 4) 重新绑定（因为 arena/v 重建了）
        self.mk.bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MK.bg0"),
            layout: &self.mk.bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.mk.header_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.mk.arena_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.mk.v_buf.as_entire_binding() },
            ],
        });

        // 5) 更新本地状态
        self.mk.header_cpu = build.header;
        self.mk.L = build.header.L;
        self.elm_stage_count = build.header.elm_stage_count;
        self.rows_each = build.elm_rows_each;
    }

    pub fn new_empty(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let build = empty_build();
        let mk = build_mile_simple_gpu(device, queue, &create_mile_shader(device), &build);
        Self {
            mk,
            elm_stage_count: 0,
            rows_each: vec![],
        }
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shader_module: &wgpu::ShaderModule,
        build: MileSimpleBuild,
    ) -> Self {
        let mk = build_mile_simple_gpu(device, queue, shader_module, &build);
        Self {
            mk,
            elm_stage_count: build.header.elm_stage_count,
            rows_each: build.elm_rows_each,
        }
    }

    /// 计算 dispatch 大小（把 y 取各阶段 rows 的最大值，kernel 里会做 r<rows 的裁剪）
    pub fn workgroup_size(&self) -> (u32, u32, u32) {
        let max_rows = self.rows_each.iter().copied().max().unwrap_or(1);
        let wg_x = (self.mk.L + 255) / 256;
        let wg_y = max_rows; // wg_size.y=1，直接把 rows 放在 dispatch 的 y 维
        let wg_z = self.elm_stage_count; // z 维是 stage id
        (wg_x.max(1), wg_y.max(1), wg_z.max(1))
    }

    /// 可选：更新输入行（供 UI 动态交互调用）
    pub fn write_input_row(&self, queue: &wgpu::Queue, row: u32, data_lanes: &[f32]) {
        assert_eq!(data_lanes.len() as u32, self.mk.L);
        queue.write_buffer(&self.mk.v_buf, (row * self.mk.L * 4) as u64, bytemuck::cast_slice(data_lanes));
    }
}


// -------------------- WGSL（供创建 shader_module 用） --------------------
// 你已有同款 kernel 的话可复用；这里给一份与上面 ABI 匹配的最小核。
pub const MILE_WGSL: &str = r#"
struct MKHeaderSimple {
  L: u32,
  final_v_len: u32,
  topouts_count: u32,
  flags: u32,

  elm_stage_count: u32,
  elm_stage_off: u32,
  recipe_off: u32,
  recipe_stride_words: u32,

  topouts_rows_off: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read>      MK : MKHeaderSimple;
@group(0) @binding(1) var<storage, read>      ARENA : array<u32>;
@group(0) @binding(2) var<storage, read_write> V : array<f32>;

fn v_read(row: u32, lane: u32) -> f32 { return V[row * MK.L + lane]; }
fn v_write(row: u32, lane: u32, val: f32) { V[row * MK.L + lane] = val; }

fn elm_stage_entry(i: u32) -> vec4<u32> {
  let base = MK.elm_stage_off + i * 4u;
  return vec4<u32>(ARENA[base+0u], ARENA[base+1u], ARENA[base+2u], ARENA[base+3u]);
}
fn read_recipe(base: u32, index: u32, stride: u32) -> array<u32, 8> {
  let s = base + index * stride;
  return array<u32, 8>(
    ARENA[s+0u], ARENA[s+1u], ARENA[s+2u], ARENA[s+3u],
    ARENA[s+4u], ARENA[s+5u], ARENA[s+6u], ARENA[s+7u]
  );
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lane = gid.x;
  let r    = gid.y;
  let sid  = gid.z;

  if (sid >= MK.elm_stage_count || lane >= MK.L) { return; }

  // 读取阶段表
  let base = MK.elm_stage_off + sid * 4u;
  let out_start   = ARENA[base + 0u];
  let rows        = ARENA[base + 1u];
  let recipe_base = ARENA[base + 2u];

  if (r >= rows) { return; }

  let rec = read_recipe(MK.recipe_off, recipe_base + r, MK.recipe_stride_words);
  let code  : u32 = rec[0];
  let arity : u32 = rec[1];
  let src0  : u32 = rec[2];
  let src1  : u32 = rec[3];
  let src2  : u32 = rec[4];
  let p0    : f32 = bitcast<f32>(rec[5]);
  let p1    : f32 = bitcast<f32>(rec[6]);
  let p2    : f32 = bitcast<f32>(rec[7]);

  let a = V[src0 * MK.L + lane];

  var b: f32 = 0.0;
  if (arity >= 2u) {
    b = V[src1 * MK.L + lane];
  }

  var c: f32 = 0.0;
  if (arity >= 3u) {
    c = V[src2 * MK.L + lane];
  }

 var out: f32;

switch (code) {
  // 一元
  case 0u  { out = a; }
  case 1u  { out = sin(a); }
  case 2u  { out = cos(a); }
  case 3u  { out = tan(a); }
  case 4u  { out = exp(a); }
  case 5u  { out = log(a); }
  case 6u  { out = sqrt(a); }
  case 7u  { out = abs(a); }

  // 二元
  case 10u { out = a + b; }
  case 11u { out = a - b; }
  case 12u { out = a * b; }
  case 13u { out = a / b; }
  case 14u { out = a - b * floor(a / b); } // fmod(a,b)
  case 15u { out = pow(a, b); }

  case 20u { out = select(0.0, 1.0, a >  b); }
  case 21u { out = select(0.0, 1.0, a >= b); }
  case 22u { out = select(0.0, 1.0, a <  b); }
  case 23u { out = select(0.0, 1.0, a <= b); }
  case 24u { out = select(0.0, 1.0, abs(a - b) <  1e-6); }
  case 25u { out = select(0.0, 1.0, abs(a - b) >= 1e-6); }

  // 三元
  case 30u { out = select(c, b, a > 0.0); } // BLEND: cond>0?then:else

  default  { out = 0.0; }
}   

  V[(out_start + r) * MK.L + lane] = out;
}
"#;

// 小工具：创建 shader module
pub fn create_mile_shader(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MK.elemwise.wgsl"),
        source: wgpu::ShaderSource::Wgsl(MILE_WGSL.into()),
    })
}
