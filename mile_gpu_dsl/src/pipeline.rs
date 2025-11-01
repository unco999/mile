use std::{cell::RefCell, ptr::eq, rc::Rc, sync::Arc};

use bytemuck::{Pod, Zeroable};
use mile_api::{Computeable, CpuGlobalUniform, GlobalEventHub, ModuleEventType, ModuleParmas, Tick};
use wgpu::util::{DeviceExt, DownloadBuffer};
use mile_api::{MileResultDes,ModuleEvent};
use crate::{core::{BinaryOp, Expr, UnaryFunc, dsl::{self, var, wvec2, wvec3, wvec4}}, dsl::if_expr, mat::op::{ImportType, MatOp, Matrix, MatrixPlan, compile_to_matrix_plan_with_imports}};

// -------------------- GPU ABI: Header / Stage / Recipe --------------------


/**
 * 这里描述的是所有的结构情况
 */
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MileDesHeader {
    pub L: u32,          // 每行有多少元素（即 lane 数量）
    pub count: u32,      // `rows` 数量
    pub rows_off: u32,   // `rows` 在 `v_buf` 中的偏移（字节单位）
    pub _pad: u32,
}




#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable,Default)]
pub struct RenderOperation {
    pub op_type: u32,           // 操作类型
    pub source_type: u32,       // 数据源类型
    pub buffer_offset: u32,     // 在V_buffer中的字节偏移量
    pub component_count: u32,   // 分量数量
    pub component_stride: u32,  // 分量步长
    pub data_format: u32,       // 数据格式
    pub blend_factor: f32,      // 混合因子
    pub custom_param: f32,      // 自定义参数
    pub condition_source: u32,  // 条件数据源 (对于条件操作)
    pub then_source: u32,       // then分支数据源  
    pub else_source: u32,       // else分支数据源
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RenderPlan {
    pub operations: [RenderOperation; 8], // 最多支持8个操作
    pub operation_count: u32,
    pub final_output_mask: u32, // 输出掩码，决定哪些分量写入最终颜色
}





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
    GreaterThan=>20, GreaterEqual=>21, LessThan=>22, LessEqual=>23, Equal=>24, NotEqual=>25,
    Index => 26
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
fn encode_import_types( plan: &MatrixPlan, src0: u32, src1: u32) -> u32 {
    let type0 = get_import_type(plan, src0 as usize);
    let type1 = get_import_type(plan, src1 as usize);
    (type0 << 16) | type1  // 高16位是src0类型，低16位是src1类型
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
                recipes.push(GpuRecipe { 
                    code: map_unary_code(func), 
                    arity: 1,
                    src0, 
                    src1: 0, 
                    src2: 0, 
                    p0_bits: encode_import_types(plan, src0, 0), // 一元操作：src0类型 + 0
                    p1_bits: 0, 
                    p2_bits: 0 
                });
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
                recipes.push(GpuRecipe { 
                    code: map_binary_code(op), 
                    arity: 2,
                    src0: s0, 
                    src1: s1, 
                    src2: 0, 
                    p0_bits: encode_import_types(plan, s0, s1), // 二元操作：src0和src1类型
                    p1_bits: 0, 
                    p2_bits: 0 
                });
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
                recipes.push(GpuRecipe { 
                    code: 30, 
                    arity: 3,
                    src0: sc, 
                    src1: st, 
                    src2: se, 
                    p0_bits: encode_import_types_3way(plan, sc, st, se), // 三元操作：三个操作数类型
                    p1_bits: 0, 
                    p2_bits: 0 
                });
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
    
    // 处理计算导入的初始化
    for import in &plan.imports {
        if let crate::mat::op::ImportType::Compute(name) = &import.import_type {
            // 为计算导入提供默认值
            let default_data = vec![0.0; lanes as usize];
            v_init_rows.push((import.index as u32, default_data));
        }
    }
    
    // 原有的输入初始化
    for (i, row) in inputs.iter().enumerate() {
        assert_eq!(row.len() as u32, lanes, "input lanes mismatch");
        v_init_rows.push((i as u32, row.clone()));
    }
    
    let const_rows: Vec<(u32, f32)> =
        plan.constant_values.iter().map(|(idx, val)| (*idx as u32, *val)).collect();

    MileSimpleBuild { header, arena_u32, v_init_rows, const_rows, elm_rows_each }
}

// 编码两个操作数的导入类型

// 编码三个操作数的导入类型（使用p0和p1字段）
fn encode_import_types_3way(plan: &MatrixPlan, src0: u32, src1: u32, src2: u32) -> u32 {
    let type0 = get_import_type(plan, src0 as usize);
    let type1 = get_import_type(plan, src1 as usize);
    let type2 = get_import_type(plan, src2 as usize);
    // p0_bits: type0(16位) | type1(16位)
    // p1_bits: type2(16位) | 其他(16位) - 但p1是f32，需要特殊处理
    (type0 << 16) | type1
    // 注意：三元操作的第三个操作数类型需要在WGSL中特殊处理
}

// 获取单个操作数的导入类型
fn get_import_type(plan: &MatrixPlan, index: usize) -> u32 {
    for import in &plan.imports {
        if import.index == index {
            return match &import.import_type {
                ImportType::Compute("time") => 1,  // time从uniform读
                ImportType::Compute(_) => 2,       // 其他计算导入
                _ => 0,                           // 普通数据或渲染导入
            };
        }
    }
    0
}

// -------------------- GPU 资源打包 --------------------
pub fn create_compute_only_plan(
    matrices: Vec<Matrix>,
    ops: Vec<MatOp>,
    constant_values: Vec<(usize, f32)>,
    final_v_len: usize,
) -> MatrixPlan {
    MatrixPlan {
        matrices,
        ops,
        top_outputs: Vec::new(), // compute计划不需要顶层输出
        constant_values,
        final_v_len,
        imports: Vec::new(), // compute计划不包含渲染导入
        render_only_ops: Vec::new(),
        compute_only_ops: Vec::new(),
    }
}

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
    global_unitfrom:Rc<CpuGlobalUniform>
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
    let v_size: u64 = (build.header.final_v_len as u64 * build.header.L as u64 * 4).max(4);
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
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            wgpu::BindGroupEntry { binding: 3, resource: global_unitfrom.get_buffer().as_entire_binding() },

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
    pub util_tick:Tick,
    pub global_hub:Arc<GlobalEventHub<ModuleEvent<Expr,RenderPlan>>>,
    pub global_unitfrom:Rc<CpuGlobalUniform>
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

    

    pub fn expr_entry_plan_with_render(
        &mut self,
        idx: u32,
        expr: &Expr,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> RenderPlan {
        use crate::mat::op::{compile_to_matrix_plan_with_imports, ImportRegistry, ImportType};
        
        let mut registry = ImportRegistry::new();
        
        // 注册渲染导入（UV等）
        registry.register_render_import("uv", 0b01, Box::new(|input| {
            // 默认UV处理
            if input.is_empty() {
                vec![1.0, 1.0, 1.0, 1.0]
            } else {
                vec![1.0, 1.0, 1.0, 1.0]
            }
        }));

       registry.register_compute_import("time", 0b01, Box::new(|input| {
            // 默认UV处理
            if input.is_empty() {
                vec![1.0]
            } else {
                vec![1.0]
            }
        }));

        let plan = compile_to_matrix_plan_with_imports(expr, &registry);
        
        // 分析计划，生成渲染操作描述
        let render_plan = self.analyze_render_operations(&plan);
        
        // 设置计算计划
        let mut result = self.set_plan_layered(device, queue, &plan, 4, &[]); // 使用4个lanes对应RGBA
        
        // 设置渲染计划
        render_plan
    }

        // 更新渲染操作参数（供UI动态调整）
    pub fn update_render_operation(
        &self,
        queue: &wgpu::Queue,
        operation_index: usize,
        op_type: u32,
        blend_factor: f32,
    ) {
        // 这里可以更新uniform buffer或者通过其他方式传递到渲染管线
        // 具体实现取决于你的渲染管线架构
    }
    
    // 获取计算结果的缓冲区引用
    pub fn get_compute_buffer(&self) -> &wgpu::Buffer {
        &self.mk.v_buf
    }
    
    // 获取计算结果的布局信息
    pub fn get_compute_layout(&self) -> (u32, u32) {
        (self.mk.L, self.mk.header_cpu.final_v_len)
    }
        fn analyze_render_operations(&self, plan: &MatrixPlan) -> RenderPlan {
        let mut operations = [RenderOperation {
            op_type: 0,
            source_type: 0,
            buffer_offset: 0,
            component_count: 0,
            component_stride: 0,
            data_format: 0,
            blend_factor: 0.0,
            custom_param: 0.0,
            condition_source: 0,
            then_source: 0,
            else_source: 0,
        }; 8];
        
        let mut operation_count = 0;
        let lanes = 4;

        // 按组处理，而不是按单个行
        let output_groups = self.group_outputs_by_operation(plan);
        let (compute_final_ops, render_final_ops) = self.split_compute_render_operations(plan);
        
        println!("输出分组: {:?}", output_groups);
        println!("最终计算操作: {:?}", compute_final_ops);
        println!("最终渲染操作: {:?}", render_final_ops);

        // 1. 处理渲染操作（按组）
        for (op_type, first_output) in render_final_ops {
            if operation_count >= 8 { break; }
            
            // 找到这个输出对应的组
            if let Some(group) = output_groups.iter().find(|g| !g.is_empty() && g[0] == first_output) {
                let buffer_offset = (first_output as u32) * lanes * 4;
                let component_count = group.len() as u32;
                
                let mut operation = RenderOperation {
                    op_type,
                    source_type: 1, // 在渲染管线中计算
                    buffer_offset,
                    component_count,
                    component_stride: 4,
                    data_format: self.get_data_format(group.len()),
                    blend_factor: 1.0,
                    custom_param: 0.0,
                    condition_source: 0,
                    then_source: 0,
                    else_source: 0,
                };
                
                // 设置条件操作的分支信息
                if op_type == 20 {
                    if let Some((cond, then, else_)) = self.get_condition_branches(first_output, plan) {
                        operation.condition_source = cond as u32;
                        operation.then_source = then as u32;
                        operation.else_source = else_ as u32;
                    }
                }
                
                operations[operation_count] = operation;
                operation_count += 1;
            }
        }

        // 2. 处理compute缓存操作（按组）
        for &first_output in &compute_final_ops {
            if operation_count >= 8 { break; }
            
            // 找到这个输出对应的组
            if let Some(group) = output_groups.iter().find(|g| !g.is_empty() && g[0] == first_output) {
                let buffer_offset = (first_output as u32) * lanes * 4;
                let component_count = group.len() as u32;
                
                operations[operation_count] = RenderOperation {
                    op_type: 0, // 直接使用compute缓存
                    source_type: 0, // 来自计算缓存
                    buffer_offset,
                    component_count,
                    component_stride: 4,
                    data_format: self.get_data_format(group.len()),
                    blend_factor: 1.0,
                    custom_param: 0.0,
                    condition_source: 0,
                    then_source: 0,
                    else_source: 0,
                };
                operation_count += 1;
            }
        }

        // 3. 处理渲染导入节点（如UV输入）
        for import in &plan.imports {
            if operation_count >= 8 { break; }
            
            if let ImportType::Render(name) = &import.import_type {
                let buffer_offset = (import.index as u32) * lanes * 4;
                
                operations[operation_count] = RenderOperation {
                    op_type: match *name {
                        "uv" => 0,
                        "pos" => 1,
                        "normal" => 2,
                        _ => 0,
                    },
                    source_type: 2, // 渲染输入（如UV坐标）
                    buffer_offset,
                    component_count: lanes,
                    component_stride: 4,
                    data_format: 3, // vec4
                    blend_factor: 1.0,
                    custom_param: 0.0,
                    condition_source: 0,
                    then_source: 0,
                    else_source: 0,
                };
                operation_count += 1;
            }
        }

        // 计算最终输出掩码
        let final_output_mask = self.calculate_output_mask(plan);

        RenderPlan {
            operations,
            operation_count:operation_count as u32,
            final_output_mask,
        }
    }


     fn calculate_output_mask(&self, plan: &MatrixPlan) -> u32 {
        // 根据输出数量计算掩码
        // 例如：1个输出 -> 0b0001 (R), 2个输出 -> 0b0011 (RG), 3个输出 -> 0b0111 (RGB), 4个输出 -> 0b1111 (RGBA)
        match plan.top_outputs.len() {
            1 => 0b0001,
            2 => 0b0011, 
            3 => 0b0111,
            4 => 0b1111,
            _ => 0b1111, // 默认输出所有通道
        }
    }


 // 分离计算和渲染操作
    fn split_compute_render_operations(&self, plan: &MatrixPlan) -> (Vec<usize>, Vec<(u32, usize)>) {
        let mut compute_final_rows = Vec::new();
        let mut render_final_rows = Vec::new();
        
        // 按输出组处理，而不是按单个行处理
        let output_groups = self.group_outputs_by_operation(plan);
        
        for group in output_groups {
            if group.is_empty() { continue; }
            
            let first_output = group[0];
            let depends_on_render = self.output_depends_on_render(first_output, plan);
            
            if depends_on_render {
                // 整个组在render管线执行
                let op_type = self.get_final_op_type(first_output, plan);
                // 只记录组的第一个行，渲染管线会处理整个向量
                render_final_rows.push((op_type, first_output));
            } else {
                // 整个组在compute管线执行
                compute_final_rows.push(first_output);
            }
        }
        
        (compute_final_rows, render_final_rows)
    }
    
    // 将输出按操作分组（同一个操作的多个输出分为一组）
    fn group_outputs_by_operation(&self, plan: &MatrixPlan) -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let mut visited = std::collections::HashSet::new();
        
        for &output in &plan.top_outputs {
            if visited.contains(&output) { continue; }
            
            // 找到产生这个输出的操作
            if let Some(op_range) = self.find_operation_range(output, plan) {
                let group: Vec<usize> = (op_range.start..op_range.end)
                    .filter(|&row| plan.top_outputs.contains(&row))
                    .collect();
                
                for &row in &group {
                    visited.insert(row);
                }
                
                if !group.is_empty() {
                    groups.push(group);
                }
            }
        }
        
        groups
    }

     // 找到操作输出的范围
    fn find_operation_range(&self, output_row: usize, plan: &MatrixPlan) -> Option<std::ops::Range<usize>> {
        for op in &plan.ops {
            match op {
                MatOp::BinaryMat { out_start, rows, .. } => {
                    if (out_start..&(out_start + rows)).contains(&&output_row) {
                        return Some(*out_start..out_start + rows);
                    }
                }
                MatOp::UnaryMat { out_start, rows, .. } => {
                    if (out_start..&(out_start + rows)).contains(&&output_row) {
                        return Some(*out_start..out_start + rows);
                    }
                }
                MatOp::CondBlendMat { out_start, rows, .. } => {
                    if (out_start..&(out_start + rows)).contains(&&output_row) {
                        return Some(*out_start..out_start + rows);
                    }
                }
            }
        }
        None
    }
      // 获取最终操作的类型
    fn get_final_op_type(&self, output_row: usize, plan: &MatrixPlan) -> u32 {
        // 找到产生这个最终输出的操作
        for op in plan.ops.iter().rev() { // 从后往前找
            match op {
                MatOp::BinaryMat { out_start, rows, .. } => {
                    if (out_start..&(out_start + rows)).contains(&&output_row) {
                        return self.get_op_type(op);
                    }
                }
                MatOp::UnaryMat { out_start, rows, .. } => {
                    if (out_start..&(out_start + rows)).contains(&&output_row) {
                        return self.get_op_type(op);
                    }
                }
                MatOp::CondBlendMat { out_start, rows, .. } => {
                    if (out_start..&(out_start + rows)).contains(&&output_row) {
                        return 20; // 条件混合
                    }
                }
            }
        }
        0
    }
    // 检查最终输出是否依赖渲染输入
    fn output_depends_on_render(&self, output_row: usize, plan: &MatrixPlan) -> bool {
        // 遍历所有操作，找到产生这个输出的操作
        for op in &plan.ops {
            match op {
                MatOp::BinaryMat { out_start, rows, .. } => {
                    if (out_start..&(out_start + rows)).contains(&&output_row) {
                        return self.operation_depends_on_render(op, plan);
                    }
                }
                MatOp::UnaryMat { out_start, rows, .. } => {
                    if (out_start..&(out_start + rows)).contains(&&output_row) {
                        return self.operation_depends_on_render(op, plan);
                    }
                }
                MatOp::CondBlendMat { out_start, rows, .. } => {
                    if (out_start..&(out_start + rows)).contains(&&output_row) {
                        return self.operation_depends_on_render(op, plan);
                    }
                }
            }
        }
        false
    }
    
    // 检查操作是否依赖渲染输入
    fn operation_depends_on_render(&self, op: &MatOp, plan: &MatrixPlan) -> bool {
        match op {
            MatOp::BinaryMat { left_mat, right_mat, .. } => {
                self.matrix_depends_on_render(&plan.matrices[*left_mat], plan) ||
                self.matrix_depends_on_render(&plan.matrices[*right_mat], plan)
            }
            MatOp::UnaryMat { mat, .. } => {
                self.matrix_depends_on_render(&plan.matrices[*mat], plan)
            }
            MatOp::CondBlendMat { cond_mat, then_mat, else_mat, .. } => {
                self.matrix_depends_on_render(&plan.matrices[*cond_mat], plan) ||
                self.matrix_depends_on_render(&plan.matrices[*then_mat], plan) ||
                self.matrix_depends_on_render(&plan.matrices[*else_mat], plan)
            }
        }
    }

      fn analyze_plan_structure(&self, plan: &RenderPlan, test_name: &str) {
        println!("{}表达式分析:", test_name);
        println!("  总操作数: {}", plan.operation_count);
        
        let mut compute_ops = 0;
        let mut render_ops = 0;
        let mut render_inputs = 0;
        
        for i in 0..plan.operation_count as usize {
            let op = &plan.operations[i];
            match op.source_type {
                0 => {
                    compute_ops += 1;
                    println!("  操作{}: COMPUTE缓存 [偏移: {}, 类型: {}]", 
                        i, op.buffer_offset, op.op_type);
                }
                1 => {
                    render_ops += 1;
                    println!("  操作{}: RENDER计算 [偏移: {}, 类型: {}]", 
                        i, op.buffer_offset, op.op_type);
                }
                _ => {
                    render_inputs += 1;
                    println!("  操作{}: RENDER输入 [偏移: {}, 类型: {}]", 
                        i, op.buffer_offset, op.op_type);
                }
            }
        }
        
        println!("  统计: Compute操作={}, Render操作={}, Render输入={}", 
            compute_ops, render_ops, render_inputs);
    }

    // 检查矩阵是否依赖渲染输入
    fn matrix_depends_on_render(&self, matrix: &Matrix, plan: &MatrixPlan) -> bool {
        for r in 0..matrix.rows {
            let row = matrix.row_slice(r);
            for (col, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    // 检查这个列索引是否对应渲染导入
                    for import in &plan.imports {
                        if import.index == col && matches!(import.import_type, ImportType::Render(_)) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn get_op_type(&self, op: &MatOp) -> u32 {
        match op {
            MatOp::BinaryMat { op, .. } => {
                match op {
                    BinaryOp::Add => 1,
                    BinaryOp::Multiply => 2,
                    BinaryOp::Subtract => 4,
                    BinaryOp::Divide => 5,
                    _ => 0,
                }
            }
            MatOp::UnaryMat { func, .. } => {
                match func {
                    UnaryFunc::Sin => 10,
                    UnaryFunc::Cos => 11,
                    _ => 0,
                }
            }
            _ => 0,
        }
    }

     fn get_data_format(&self, output_count: usize) -> u32 {
        match output_count {
            1 => 0, // 标量
            2 => 1, // vec2  
            3 => 2, // vec3
            4 => 3, // vec4
            _ => 3, // 默认vec4
        }
    }

        fn analyze_operation_type(&self, ops: &[MatOp]) -> u32 {
        use crate::core::{BinaryOp, UnaryFunc};
        
        for op in ops.iter().rev() { // 从后往前分析，找到最后的操作
            match op {
                MatOp::BinaryMat { op, .. } => {
                    return match op {
                        BinaryOp::Add => 1,      // 加法
                        BinaryOp::Multiply => 2, // 乘法
                        BinaryOp::Subtract => 4, // 减法
                        BinaryOp::Divide => 5,   // 除法
                        BinaryOp::GreaterThan => 6,   // 大于比较
                        BinaryOp::LessThan => 7,      // 小于比较
                        BinaryOp::Equal => 8,         // 等于比较
                        _ => 0, // 其他操作默认处理
                    };
                }
                MatOp::UnaryMat { func, .. } => {
                    return match func {
                        UnaryFunc::Sin => 10,   // 正弦
                        UnaryFunc::Cos => 11,   // 余弦
                        UnaryFunc::Exp => 12,   // 指数
                        UnaryFunc::Log => 13,   // 对数
                        UnaryFunc::Sqrt => 14,  // 平方根
                        _ => 0,
                    };
                }
                MatOp::CondBlendMat { .. } => {
                    return 20; // 条件混合操作
                }
            }
        }
        0 // 默认操作
    }

    

    pub fn test_entry(&mut self,device: &wgpu::Device,queue:  &wgpu::Queue){
      // 测试1: 纯计算表达式
        use crate::core::dsl::*;
        
        println!("\n=== 测试分层计算 ===");
        
        // 测试1: 纯计算表达式 (应该在compute管线执行)
        let expr1 = var("time") * 10.0 + 10.0 * var("uv");
        let output1 = self.expr_entry_plan_with_render(0, &expr1, device, queue);
        self.analyze_plan_structure(&output1, "简单测试");
        
    }

    pub fn expr_entry_plan(&mut self,idx:u32,expr:&Expr,device: &wgpu::Device,queue:  &wgpu::Queue)->MileResultDes{
        // let plan = compile_to_matrix_plan_with_imports(expr);
        // let des = self.set_plan(device, queue, &plan, 3, &[]);
        // des
        todo!()
    }

      // 获取条件操作的分支信息
    fn get_condition_branches(&self, output_row: usize, plan: &MatrixPlan) -> Option<(usize, usize, usize)> {
        for op in &plan.ops {
            if let MatOp::CondBlendMat { cond_mat, then_mat, else_mat, out_start, rows } = op {
                if (out_start..&(out_start + rows)).contains(&&output_row) {
                    // 找到对应的条件、then、else数据行
                    let cond_row = self.get_first_nonzero_row(&plan.matrices[*cond_mat]);
                    let then_row = self.get_first_nonzero_row(&plan.matrices[*then_mat]);
                    let else_row = self.get_first_nonzero_row(&plan.matrices[*else_mat]);
                    
                    return Some((cond_row, then_row, else_row));
                }
            }
        }
        None
    }
    
    fn get_first_nonzero_row(&self, matrix: &Matrix) -> usize {
        for r in 0..matrix.rows {
            let row = matrix.row_slice(r);
            if let Some(col) = row.iter().position(|&v| v != 0.0) {
                return col;
            }
        }
        0
    }

    pub fn read_call_back_cpu(&mut self,device: &wgpu::Device,queue:  &wgpu::Queue){
        if(!self.util_tick.tick()) {return;}
        DownloadBuffer::read_buffer(
            device,
            queue,
            &self.mk.v_buf.slice(0..256),
            move |e|{
                if let Ok(downloadBuffer) = e {
                    let bytes = downloadBuffer;
                    // cast bytes -> &[MyStruct]
                    let data: &[f32] = bytemuck::cast_slice(&bytes);

                    data.iter().enumerate().for_each(|e|{
                        println!("{:?}",e);
                    });
                }
            }
        );
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

pub fn init_buffer(
    &mut self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    // 预分配大小，避免重新创建 buffer
    self.mk.arena_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MK.arena"),
        size: std::mem::size_of::<u32>() as u64 * 8192 as u64, // 预分配 arena 大小
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 预分配 V buffer
    let new_v_size = 8192 as u64 * 512 as u64 * 4; // 计算预分配的 V buffer 大小
    self.mk.v_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MK.V"),
        size: new_v_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
}

fn split_compute_render_plan(&self, plan: &MatrixPlan) -> (MatrixPlan, RenderPlan) {
        let mut compute_ops = Vec::new();
        let mut compute_matrices = Vec::new();
        let mut compute_constants = plan.constant_values.clone();
        
        // 收集不依赖渲染输入的操作
        for op in &plan.ops {
            if !self.operation_depends_on_render(op, plan) {
                compute_ops.push(op.clone());
                // 添加相关的矩阵
                match op {
                    MatOp::BinaryMat { left_mat, right_mat, .. } => {
                        compute_matrices.push(plan.matrices[*left_mat].clone());
                        compute_matrices.push(plan.matrices[*right_mat].clone());
                    }
                    MatOp::UnaryMat { mat, .. } => {
                        compute_matrices.push(plan.matrices[*mat].clone());
                    }
                    MatOp::CondBlendMat { cond_mat, then_mat, else_mat, .. } => {
                        compute_matrices.push(plan.matrices[*cond_mat].clone());
                        compute_matrices.push(plan.matrices[*then_mat].clone());
                        compute_matrices.push(plan.matrices[*else_mat].clone());
                    }
                }
            }
        }

        // 创建compute计划
        let compute_plan = MatrixPlan {
            matrices: compute_matrices,
            ops: compute_ops,
            top_outputs: self.get_compute_outputs(plan), // 只包含compute管线的输出
            constant_values: compute_constants,
            final_v_len: plan.final_v_len, // 可能需要调整
            imports: plan.imports.iter()
                .filter(|i| matches!(i.import_type, ImportType::Compute(_)))
                .cloned()
                .collect(),
            render_only_ops: Vec::new(),
            compute_only_ops: Vec::new(),
        };

        // 创建render计划
        let render_plan = self.analyze_render_operations(plan);

        (compute_plan, render_plan)
    }

        fn get_compute_outputs(&self, plan: &MatrixPlan) -> Vec<usize> {
        // 找出compute管线应该计算的输出
        // 这些是render管线需要读取的中间结果
        let mut outputs = Vec::new();
        
        for op in &plan.ops {
            if !self.operation_depends_on_render(op, plan) {
                match op {
                    MatOp::BinaryMat { out_start, rows, .. } => {
                        for i in 0..*rows {
                            outputs.push(out_start + i);
                        }
                    }
                    MatOp::UnaryMat { out_start, rows, .. } => {
                        for i in 0..*rows {
                            outputs.push(out_start + i);
                        }
                    }
                    MatOp::CondBlendMat { out_start, rows, .. } => {
                        for i in 0..*rows {
                            outputs.push(out_start + i);
                        }
                    }
                }
            }
        }
        
        outputs
    }

      pub fn set_plan_layered(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &MatrixPlan,
        lanes: u32,
        inputs: &[Vec<f32>],
    ) -> (MileResultDes, RenderPlan) {
        // 1. 分析哪些操作应该在compute管线执行
        let (compute_ops, render_plan) = self.prepare_layered_execution(plan);
        
        println!("Compute操作数: {}", compute_ops.len());
        println!("Render操作数: {}", render_plan.operation_count);

        // 2. 只为compute操作创建计划
        let compute_build = self.create_compute_build(&compute_ops, plan, lanes, inputs);
        
        // 更新compute管线的buffer
        queue.write_buffer(&self.mk.header_buf, 0, bytemuck::bytes_of(&compute_build.header));
        queue.write_buffer(&self.mk.arena_buf, 0, bytemuck::cast_slice(&compute_build.arena_u32));
        
        // 初始化compute数据
        for (row, data) in &compute_build.v_init_rows {
            queue.write_buffer(&self.mk.v_buf, (*row as u64) * compute_build.header.L as u64 * 4, bytemuck::cast_slice(data));
        }
        
        for (row, val) in &compute_build.const_rows {
            let vecv = vec![*val; compute_build.header.L as usize];
            queue.write_buffer(&self.mk.v_buf, (*row as u64) * compute_build.header.L as u64 * 4, bytemuck::cast_slice(&vecv));
        }

        // 3. 更新绑定组
        self.mk.bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MK.bg0"),
            layout: &self.mk.bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.mk.header_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.mk.arena_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.mk.v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.global_unitfrom.get_buffer().as_entire_binding() },
            ],
        });

        // 4. 更新本地状态
        self.mk.header_cpu = compute_build.header;
        self.mk.L = compute_build.header.L;
        self.elm_stage_count = compute_build.header.elm_stage_count;
        self.rows_each = compute_build.elm_rows_each;

        // 5. 返回结果
        let row_start_array = self.calculate_output_offsets(plan, lanes);

        (
            MileResultDes { row_start: row_start_array },
            render_plan
        )
    }

    fn prepare_layered_execution(&self, plan: &MatrixPlan) -> (Vec<MatOp>, RenderPlan) {
        let mut compute_ops = Vec::new();
        let mut render_ops = Vec::new();
        
        // 更智能的操作拆分
        for op in &plan.ops {
            self.split_operation_by_dependency(op, plan, &mut compute_ops, &mut render_ops);
        }

        println!("Compute操作数: {}", compute_ops.len());
        println!("Render操作数: {}", render_ops.len());
        
        let render_plan = self.analyze_render_operations_with_ops(plan, &render_ops);
        
        (compute_ops, render_plan)
    }
    
    // 严格检查：矩阵是否依赖任何渲染输入
    fn matrix_depends_on_any_render(&self, matrix: &Matrix, plan: &MatrixPlan) -> bool {
        for r in 0..matrix.rows {
            let row = matrix.row_slice(r);
            for (col, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    // 检查这个列索引是否对应任何渲染导入
                    for import in &plan.imports {
                        if import.index == col && matches!(import.import_type, ImportType::Render(_)) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
    
     fn split_operation_by_dependency(
        &self, 
        op: &MatOp, 
        plan: &MatrixPlan,
        compute_ops: &mut Vec<MatOp>,
        render_ops: &mut Vec<MatOp>,
    ) {
        match op {
            MatOp::BinaryMat { op: bin_op, left_mat, right_mat, out_start, rows } => {
                let left_depends = self.matrix_depends_on_any_render(&plan.matrices[*left_mat], plan);
                let right_depends = self.matrix_depends_on_any_render(&plan.matrices[*right_mat], plan);
                
                if !left_depends && !right_depends {
                    // 两个操作数都不依赖渲染 -> 完全在compute执行
                    compute_ops.push(MatOp::BinaryMat {
                        op: bin_op.clone(),
                        left_mat: *left_mat,
                        right_mat: *right_mat,
                        out_start: *out_start,
                        rows: *rows,
                    });
                } else if left_depends && right_depends {
                    // 两个操作数都依赖渲染 -> 完全在render执行
                    render_ops.push(MatOp::BinaryMat {
                        op: bin_op.clone(),
                        left_mat: *left_mat,
                        right_mat: *right_mat,
                        out_start: *out_start,
                        rows: *rows,
                    });
                } else {
                    // 混合依赖：一个依赖渲染，一个不依赖
                    if !left_depends {
                        // left是compute，right是render
                        // 在compute中预计算left操作数（直接传递）
                        compute_ops.push(MatOp::UnaryMat {
                            func: UnaryFunc::Sin, // 这里应该用Identity，但我们没有这个函数
                            mat: *left_mat,
                            out_start: *out_start,
                            rows: *rows,
                        });
                    } else {
                        // right是compute，left是render  
                        compute_ops.push(MatOp::UnaryMat {
                            func: UnaryFunc::Sin, // 同样的问题
                            mat: *right_mat,
                            out_start: *out_start,
                            rows: *rows,
                        });
                    }
                    // 最终操作在render执行
                    render_ops.push(MatOp::BinaryMat {
                        op: bin_op.clone(),
                        left_mat: *left_mat,
                        right_mat: *right_mat,
                        out_start: *out_start,
                        rows: *rows,
                    });
                }
            }
            MatOp::UnaryMat { func, mat, out_start, rows } => {
                if !self.matrix_depends_on_any_render(&plan.matrices[*mat], plan) {
                    compute_ops.push(MatOp::UnaryMat {
                        func: func.clone(),
                        mat: *mat,
                        out_start: *out_start,
                        rows: *rows,
                    });
                } else {
                    render_ops.push(MatOp::UnaryMat {
                        func: func.clone(),
                        mat: *mat,
                        out_start: *out_start,
                        rows: *rows,
                    });
                }
            }
            MatOp::CondBlendMat { cond_mat, then_mat, else_mat, out_start, rows } => {
                // 条件操作比较复杂，暂时整个在render执行
                render_ops.push(MatOp::CondBlendMat {
                    cond_mat: *cond_mat,
                    then_mat: *then_mat,
                    else_mat: *else_mat,
                    out_start: *out_start,
                    rows: *rows,
                });
            }
        }
    }

    fn analyze_render_operations_with_ops(&self, plan: &MatrixPlan, render_ops: &[MatOp]) -> RenderPlan {
        let mut operations = [RenderOperation::default(); 8];
        let mut operation_count = 0;
        let lanes = 4;

        // 处理渲染操作
        for op in render_ops {
            if operation_count >= 8 { break; }
            
            match op {
                MatOp::BinaryMat { op: bin_op, out_start, rows, .. } => {
                    let buffer_offset = (*out_start as u32) * lanes * 4;
                    
                    operations[operation_count] = RenderOperation {
                        op_type: match bin_op {
                            BinaryOp::Add => 1,
                            BinaryOp::Multiply => 2,
                            BinaryOp::Subtract => 4,
                            _ => 0,
                        },
                        source_type: 1, // 在渲染管线计算
                        buffer_offset,
                        component_count: (*rows as u32).min(4),
                        component_stride: 4,
                        data_format: self.get_data_format(*rows),
                        blend_factor: 1.0,
                        custom_param: 0.0,
                        condition_source: 0,
                        then_source: 0,
                        else_source: 0,
                    };
                    operation_count += 1;
                }
                // 其他操作类型...
                _ => {}
            }
        }

        // 添加渲染输入
        for import in &plan.imports {
            if operation_count >= 8 { break; }
            
            if let ImportType::Render(name) = &import.import_type {
                let buffer_offset = (import.index as u32) * lanes * 4;
                
                operations[operation_count] = RenderOperation {
                    op_type: 0,
                    source_type: 2, // 渲染输入
                    buffer_offset,
                    component_count: lanes,
                    component_stride: 4,
                    data_format: 3,
                    blend_factor: 1.0,
                    custom_param: 0.0,
                    condition_source: 0,
                    then_source: 0,
                    else_source: 0,
                };
                operation_count += 1;
            }
        }

        RenderPlan {
            operations,
            operation_count:operation_count as u32,
            final_output_mask: 0b1111,
        }
    }

    fn create_compute_build(
        &self,
        compute_ops: &[MatOp],
        original_plan: &MatrixPlan,
        lanes: u32,
        inputs: &[Vec<f32>],
    ) -> MileSimpleBuild {
        // 创建只包含compute操作的简化计划
        let compute_plan = MatrixPlan {
            matrices: original_plan.matrices.clone(), // 可能需要过滤
            ops: compute_ops.to_vec(),
            top_outputs: Vec::new(), // compute计划不需要顶层输出
            constant_values: original_plan.constant_values.clone(),
            final_v_len: original_plan.final_v_len,
            imports: original_plan.imports.iter()
                .filter(|i| matches!(i.import_type, crate::mat::op::ImportType::Compute(_)))
                .cloned()
                .collect(),
            render_only_ops: Vec::new(),
            compute_only_ops: Vec::new(),
        };

        plan_to_mile_simple(&compute_plan, lanes, inputs)
    }

    fn calculate_output_offsets(&self, plan: &MatrixPlan, lanes: u32) -> [u32; 4] {
        let mut row_start_array: [u32; 4] = [0; 4];
        
        // 计算compute结果的偏移量，供render管线使用
        for (i, &row) in plan.top_outputs.iter().enumerate().take(4) {
            row_start_array[i] = (row as u32 * lanes) as u32;
        }
        
        row_start_array
    }

    pub fn new_empty(
        device: &wgpu::Device, 
        queue: &wgpu::Queue,
        global_hub:Arc<GlobalEventHub<ModuleEvent<Expr,RenderPlan>>>,
        global_unitfrom:Rc<CpuGlobalUniform>) -> Self {
        let build = empty_build();
        let mk = build_mile_simple_gpu(device, queue, &create_mile_shader(device), &build,global_unitfrom.clone());
        Self {
            mk,
            elm_stage_count: 0,
            rows_each: vec![],
            util_tick:Tick::new(1),
            global_hub,
            global_unitfrom
        }
    }

    #[inline]
    pub fn process_ui_events(&mut self, device: &wgpu::Device,queue: &wgpu::Queue) {
        for ev in self.global_hub.poll() {
            match ev {


                mile_api::ModuleEvent::KennelPush(params) => {
                    let des = self.expr_entry_plan_with_render(
                        params.idx, 
                        &params.data, 
                        device, 
                        queue
                    );
                },

                _=>{

                }
            }
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

struct GlobalUniform {
      // === block 1: atomic z/layouts ===
    click_layout_z: u32,
    click_layout_id: u32,
    hover_layout_id: u32,
    hover_layout_z: u32, // 16 bytes

    // === block 2: atomic drag ===
    drag_layout_id: u32,
    drag_layout_z: u32,
    pad_atomic1: u32,
    pad_atomic2: u32,    // 16 bytes

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

@group(0) @binding(0) var<storage, read>      MK : MKHeaderSimple;
@group(0) @binding(1) var<storage, read>      ARENA : array<u32>;
@group(0) @binding(2) var<storage, read_write> V : array<f32>;
@group(0) @binding(3) var<storage, read> global_uniform: GlobalUniform;

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
  let p0_bits : u32 = rec[5];
  let p1_bits    : f32 = bitcast<f32>(rec[6]);
  let p2_bits    : f32 = bitcast<f32>(rec[7]);
  // 解码导入类型
  let import_type0 = (p0_bits >> 16u) & 0xFFFFu;
  let import_type1 = (p0_bits >> 0u) & 0xFFFFu;

  // 读取第一个操作数
  var a: f32;
  if (import_type0 == 1u) {
    // 从uniform读取time
    a = global_uniform.time;
  } else {
    // 从V缓冲区读取普通数据
    a = V[src0 * MK.L + lane];
  }

  // 读取第二个操作数
  var b: f32 = 0.0;
  if (arity >= 2u) {
    if (import_type1 == 1u) {
      b = global_uniform.time;
    } else {
      b = V[src1 * MK.L + lane];
    }
  }

var c: f32 = 0.0;
if (arity >= 3u) {
    let import_type2 = (u32(p1_bits) >> 16u) & 0xFFFFu; // 从p1的高16位读取第三个操作数类型
    if (import_type2 == 1u) {
        c = global_uniform.time;
    } else {
        c = V[src2 * MK.L + lane];
    }
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
