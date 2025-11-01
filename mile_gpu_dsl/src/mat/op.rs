use std::collections::HashMap;

use crate::core::{BinaryOp, Expr, UnaryFunc};



#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>, // row-major length = rows * cols
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.cols + c] = v;
    }
    pub fn row_slice(&self, r: usize) -> &[f32] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }
}


#[derive(Debug, Clone, PartialEq)]
pub enum ImportType {
    Render(String),    // 渲染管线导入，如UV、屏幕坐标等
    Compute(String),   // 计算管线导入，如缓存数据等
}

#[derive(Debug, Clone)]
pub struct ImportInfo {
    pub import_type: ImportType,
    pub mask: u32,     // 比如 01 是uv, 11 是pos+uv
    pub index: usize,  // 在V中的索引位置
}

pub struct ImportRegistry {
    render_imports: std::collections::HashMap<String, (u32, ImportHandler)>,
    compute_imports: std::collections::HashMap<String, (u32, ImportHandler)>,
}

pub type ImportHandler = Box<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>;

impl ImportRegistry {
    pub fn new() -> Self {
        Self {
            render_imports: HashMap::new(),
            compute_imports: HashMap::new(),
        }
    }

    // 注册渲染导入（如UV坐标）
    pub fn register_render_import(&mut self, name: &str, mask: u32, handler: ImportHandler) {
        self.render_imports.insert(name.to_string(), (mask, handler));
    }

    // 注册计算导入（如缓存数据）
    pub fn register_compute_import(&mut self, name: &str, mask: u32, handler: ImportHandler) {
        self.compute_imports.insert(name.to_string(), (mask, handler));
    }

    // 获取导入信息
    pub fn get_import_info(&self, name: &str) -> Option<(ImportType, u32)> {
        if let Some((mask, _)) = self.render_imports.get(name) {
            Some((ImportType::Render(name.to_string()), *mask))
        } else if let Some((mask, _)) = self.compute_imports.get(name) {
            Some((ImportType::Compute(name.to_string()), *mask))
        } else {
            None
        }
    }

    // 执行导入处理
    pub fn execute_import(&self, import_type: &ImportType, input: &[f32]) -> Vec<f32> {
        match import_type {
            ImportType::Render(name) => {
                if let Some((_, handler)) = self.render_imports.get(name) {
                    handler(input)
                } else {
                    vec![0.0; input.len()]
                }
            }
            ImportType::Compute(name) => {
                if let Some((_, handler)) = self.compute_imports.get(name) {
                    handler(input)
                } else {
                    vec![0.0; input.len()]
                }
            }
        }
    }
}

/// 每个矩阵化操作（op）:
/// - BinaryOpMat: 有 left_mat_index, right_mat_index, 输出起始索引 out_start, 行数 rows
/// - UnaryOpMat: 有 mat_index, out_start, rows
/// - CondBlendMat: cond_mat, then_mat, else_mat, out_start, rows
#[derive(Debug, Clone)]
pub enum MatOp {
    BinaryMat {
        op: BinaryOp,
        left_mat: usize,
        right_mat: usize,
        out_start: usize,
        rows: usize,
    },
    UnaryMat {
        func: UnaryFunc,
        mat: usize,
        out_start: usize,
        rows: usize,
    },
    CondBlendMat {
        cond_mat: usize,
        then_mat: usize,
        else_mat: usize,
        out_start: usize,
        rows: usize,
    },
}

/// 编译结果（矩阵计划）
#[derive(Debug, Clone)]
pub struct MatrixPlan {
    pub matrices: Vec<Matrix>,
    pub ops: Vec<MatOp>,
    pub top_outputs: Vec<usize>,
    pub constant_values: Vec<(usize, f32)>,
    pub final_v_len: usize,
    pub imports: Vec<ImportInfo>,           // 新增：所有导入节点信息
    pub render_only_ops: Vec<usize>,        // 新增：只能在渲染管线中执行的操作索引
    pub compute_only_ops: Vec<usize>,       // 新增：只能在计算管线中执行的操作索引
}

/// 把 AST 编译成矩阵计划（selection matrices + ops）
/// 原则：按自底向上遍历，遇到常量就把常量 append 到初始 v（并记录其 index），
/// 每遇到一个操作，先确定当前 v 长度 cur_len（这是矩阵的列数），根据子节点索引创建选择矩阵，
/// 然后为该 op 的 outputs 预留 out_start..out_start+rows-1（并 advance next_index）。
pub fn compile_to_matrix_plan_with_imports(
    expr: &Expr,
    registry: &ImportRegistry,
) -> MatrixPlan {
    let mut matrices: Vec<Matrix> = Vec::new();
    let mut ops: Vec<MatOp> = Vec::new();
    let mut constant_values: Vec<(usize, f32)> = Vec::new();
    let mut imports: Vec<ImportInfo> = Vec::new();
    let mut render_only_ops: Vec<usize> = Vec::new();
    let mut compute_only_ops: Vec<usize> = Vec::new();

    let mut next_index: usize = 0;
    let mut current_op_index = 0;

    // 递归编译函数
    fn rec(
        expr: &Expr,
        matrices: &mut Vec<Matrix>,
        ops: &mut Vec<MatOp>,
        constant_values: &mut Vec<(usize, f32)>,
        imports: &mut Vec<ImportInfo>,
        render_only_ops: &mut Vec<usize>,
        compute_only_ops: &mut Vec<usize>,
        registry: &ImportRegistry,
        next_index: &mut usize,
        current_op_index: &mut usize,
    ) -> Vec<usize> {
        match expr {
            Expr::RenderImport(name) => {
                // 处理渲染导入
                let idx = *next_index;
                *next_index += 1;
                
                if let Some((import_type, mask)) = registry.get_import_info(name) {
                    imports.push(ImportInfo {
                        import_type,
                        mask,
                        index: idx,
                    });
                    // 标记当前操作索引为渲染操作
                    render_only_ops.push(*current_op_index);
                }
                vec![idx]
            }
            Expr::ComputeImport(name) => {
                // 处理计算导入
                let idx = *next_index;
                *next_index += 1;
                
                if let Some((import_type, mask)) = registry.get_import_info(name) {
                    imports.push(ImportInfo {
                        import_type,
                        mask,
                        index: idx,
                    });
                    // 标记当前操作索引为计算操作
                    compute_only_ops.push(*current_op_index);
                }
                vec![idx]
            }
            Expr::Constant(val) => {
                let idx = *next_index;
                *next_index += 1;
                constant_values.push((idx, *val));
                vec![idx]
            }
            Expr::Vec2(v) => {
                let x = rec(&v.x, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                let y = rec(&v.y, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                vec![x[0], y[0]]
            }
            Expr::Vec3(v) => {
                let x = rec(&v.x, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                let y = rec(&v.y, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                let z = rec(&v.z, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                vec![x[0], y[0], z[0]]
            }
            Expr::Vec4(v) => {
                let x = rec(&v.x, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                let y = rec(&v.y, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                let z = rec(&v.z, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                let w = rec(&v.w, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                vec![x[0], y[0], z[0], w[0]]
            }
            Expr::BinaryOp(op, left, right) => {
                if let BinaryOp::Index = op {
                    // 索引操作的特殊处理
                    let l_idxs = rec(left, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                    let r_idxs = rec(right, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                    
                    if r_idxs.len() == 1 {
                        let index_idx = r_idxs[0];
                        let index_val = if let Some((_, val)) = constant_values.iter().find(|(idx, _)| *idx == index_idx) {
                            *val as usize
                        } else { 0 };
                        
                        if index_val < l_idxs.len() {
                            vec![l_idxs[index_val]]
                        } else {
                            vec![l_idxs[0]]
                        }
                    } else {
                        vec![l_idxs[0]]
                    }
                } else {
                    // 普通二元操作
                    let l_idxs = rec(left, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                    let r_idxs = rec(right, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);

                    let comps = l_idxs.len().max(r_idxs.len());
                    let mut outs = Vec::with_capacity(comps);
                    let cur_cols = *next_index;

                    let mut left_mat = Matrix::new(comps, cur_cols);
                    let mut right_mat = Matrix::new(comps, cur_cols);

                    for i in 0..comps {
                        let a_idx = if i < l_idxs.len() { l_idxs[i] } else { l_idxs[0] };
                        let b_idx = if i < r_idxs.len() { r_idxs[i] } else { r_idxs[0] };
                        left_mat.set(i, a_idx, 1.0);
                        right_mat.set(i, b_idx, 1.0);
                    }

                    let left_mat_idx = matrices.len();
                    matrices.push(left_mat);
                    let right_mat_idx = matrices.len();
                    matrices.push(right_mat);

                    let out_start = *next_index;
                    *next_index += comps;

                    // 添加操作并增加操作索引
                    ops.push(MatOp::BinaryMat { 
                        op: op.clone(), 
                        left_mat: left_mat_idx, 
                        right_mat: right_mat_idx, 
                        out_start, 
                        rows: comps 
                    });
                    *current_op_index += 1;

                    for i in 0..comps {
                        outs.push(out_start + i);
                    }
                    outs
                }
            }
            Expr::UnaryOp(func, sub) => {
                let s_idxs = rec(sub, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                let comps = s_idxs.len();
                let cur_cols = *next_index;
                let mut mat = Matrix::new(comps, cur_cols);
                for i in 0..comps {
                    let pick = s_idxs[i];
                    mat.set(i, pick, 1.0);
                }
                let mat_idx = matrices.len();
                matrices.push(mat);
                let out_start = *next_index;
                *next_index += comps;
                
                ops.push(MatOp::UnaryMat { 
                    func: func.clone(), 
                    mat: mat_idx, 
                    out_start, 
                    rows: comps 
                });
                *current_op_index += 1;
                
                (0..comps).map(|i| out_start + i).collect()
            }
            Expr::If { condition, then_branch, else_branch } => {
                let c_idxs = rec(condition, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                let t_idxs = rec(then_branch, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);
                let e_idxs = rec(else_branch, matrices, ops, constant_values, imports, render_only_ops, compute_only_ops, registry, next_index, current_op_index);

                let comps = t_idxs.len().max(e_idxs.len());
                let cur_cols = *next_index;
                let mut cond_mat = Matrix::new(comps, cur_cols);
                let mut then_mat = Matrix::new(comps, cur_cols);
                let mut else_mat = Matrix::new(comps, cur_cols);

                for i in 0..comps {
                    let c_pick = if i < c_idxs.len() { c_idxs[i] } else { c_idxs[0] };
                    let t_pick = if i < t_idxs.len() { t_idxs[i] } else { t_idxs[0] };
                    let e_pick = if i < e_idxs.len() { e_idxs[i] } else { e_idxs[0] };
                    cond_mat.set(i, c_pick, 1.0);
                    then_mat.set(i, t_pick, 1.0);
                    else_mat.set(i, e_pick, 1.0);
                }

                let cond_mat_idx = matrices.len(); matrices.push(cond_mat);
                let then_mat_idx = matrices.len(); matrices.push(then_mat);
                let else_mat_idx = matrices.len(); matrices.push(else_mat);

                let out_start = *next_index;
                *next_index += comps;

                ops.push(MatOp::CondBlendMat {
                    cond_mat: cond_mat_idx,
                    then_mat: then_mat_idx,
                    else_mat: else_mat_idx,
                    out_start,
                    rows: comps,
                });
                *current_op_index += 1;

                (0..comps).map(|i| out_start + i).collect()
            }
            _ => vec![] // 处理其他表达式类型
        }
    }

    let top_outputs = rec(
        expr, 
        &mut matrices, 
        &mut ops, 
        &mut constant_values, 
        &mut imports, 
        &mut render_only_ops, 
        &mut compute_only_ops, 
        registry, 
        &mut next_index,
        &mut current_op_index
    );
    let final_v_len = next_index;

    println!("编译结果 - 最终V长度: {:?}", final_v_len);
    println!("编译结果 - 顶层输出: {:?}", top_outputs);
    println!("编译结果 - 导入节点: {:?}", imports);
    println!("编译结果 - 渲染操作: {:?}", render_only_ops);
    println!("编译结果 - 计算操作: {:?}", compute_only_ops);

    MatrixPlan {
        matrices,
        ops,
        top_outputs,
        constant_values,
        final_v_len,
        imports,
        render_only_ops,
        compute_only_ops,
    }
}

// ---------- helper types: represent V as rows x cols (rows = final_v_len, cols = vector_len) ----------
type Mat2D = Vec<Vec<f32>>; // row-major: Mat2D[r][c]  (r: 0..rows-1, c: 0..cols-1)

/// simple matrix-matrix multiply: C = A (rowsA x colsA) * B (colsA x colsB)
/// A is Matrix (rows x cols), B is Mat2D (cols x colsB) represented as rows of length colsB.
/// Returns Mat2D with shape rowsA x colsB
fn mat_mul(A: &Matrix, B: &Mat2D) -> Mat2D {
    let rows = A.rows;
    let inner = A.cols; // == B.len()
    if inner == 0 {
        return vec![vec![0.0; if B.is_empty() { 0 } else { B[0].len() }]; rows];
    }
    let cols = B[0].len();
    let mut C: Mat2D = vec![vec![0.0; cols]; rows];

    // naive triple loop; good enough for moderate sizes; can be optimized later
    for r in 0..rows {
        let a_row = &A.data[r * A.cols .. r * A.cols + A.cols];
        let crow = &mut C[r];
        for k in 0..inner {
            let a = a_row[k];
            if a == 0.0 { continue; }
            let brow = &B[k]; // B's row k has length cols
            for c in 0..cols {
                crow[c] += a * brow[c];
            }
        }
    }
    C
}

/// elementwise binary op on two matrices of same shape (rows x cols)
fn mat_elemwise_binary(a: &Mat2D, b: &Mat2D, op: &BinaryOp) -> Mat2D {
    let rows = a.len();
    if rows == 0 { return vec![]; }
    let cols = a[0].len();
    let mut out = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let av = a[r][c];
            let bv = b[r][c];
            out[r][c] = match op {
                BinaryOp::Add => av + bv,
                BinaryOp::Subtract => av - bv,
                BinaryOp::Multiply => av * bv,
                BinaryOp::Divide => av / bv,
                BinaryOp::Modulo => av % bv,
                BinaryOp::Pow => av.powf(bv),
                BinaryOp::GreaterThan => if av > bv { 1.0 } else { 0.0 },
                BinaryOp::GreaterEqual => if av >= bv { 1.0 } else { 0.0 },
                BinaryOp::LessThan => if av < bv { 1.0 } else { 0.0 },
                BinaryOp::LessEqual => if av <= bv { 1.0 } else { 0.0 },
                BinaryOp::Equal => if (av - bv).abs() < std::f32::EPSILON { 1.0 } else { 0.0 },
                BinaryOp::NotEqual => if (av - bv).abs() >= std::f32::EPSILON { 1.0 } else { 0.0 },
                BinaryOp::Index => av,
            };
        }
    }
    out
}

/// elementwise unary op on matrix
fn mat_elemwise_unary(a: &Mat2D, func: &UnaryFunc) -> Mat2D {
    let rows = a.len();
    if rows == 0 { return vec![]; }
    let cols = a[0].len();
    let mut out = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let v = a[r][c];
            out[r][c] = match func {
                UnaryFunc::Sin => v.sin(),
                UnaryFunc::Cos => v.cos(),
                UnaryFunc::Tan => v.tan(),
                UnaryFunc::Exp => v.exp(),
                UnaryFunc::Log => v.ln(),
                UnaryFunc::Sqrt => v.sqrt(),
                UnaryFunc::Abs => v.abs(),
            };
        }
    }
    out
}

/// elementwise comparison > 0 => mask (0 or 1)
fn mat_greater_zero(a: &Mat2D) -> Mat2D {
    let rows = a.len();
    if rows == 0 { return vec![]; }
    let cols = a[0].len();
    let mut out = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r][c] = if a[r][c] > 0.0 { 1.0 } else { 0.0 };
        }
    }
    out
}

/// elementwise blend: mask * A + (1-mask) * B ; mask should be same shape
fn mat_blend(mask: &Mat2D, a: &Mat2D, b: &Mat2D) -> Mat2D {
    let rows = mask.len();
    if rows == 0 { return vec![]; }
    let cols = mask[0].len();
    let mut out = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let m = mask[r][c];
            out[r][c] = m * a[r][c] + (1.0 - m) * b[r][c];
        }
    }
    out
}

/// WRITE rows_mat (rows x cols) into V at row offset out_start
fn write_rows_to_V(V: &mut Mat2D, rows_mat: &Mat2D, out_start: usize) {
    let rows = rows_mat.len();
    if rows == 0 { return; }
    let cols = rows_mat[0].len();
    for r in 0..rows {
        let dst_r = out_start + r;
        V[dst_r].copy_from_slice(&rows_mat[r]);
    }
}

/// 批量矩阵流水线模拟：一次性对所有 lane 做矩阵运算（无 per-lane branch）
pub fn simulate_matrix_plan_batch(plan: &MatrixPlan, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if inputs.is_empty() { return vec![]; }
    let cols = inputs[0].len(); // vector_len (L)
    let rows = plan.final_v_len; // final_v_len
    // 构造 V (rows x cols)
    let mut V: Mat2D = vec![vec![0.0; cols]; rows];

    // // fill input variable rows
    // for i in 0..plan.variable_count {
    //     V[i].copy_from_slice(&inputs[i]);
    // }
    // fill constants
    for (idx, val) in &plan.constant_values {
        for c in 0..cols { V[*idx][c] = *val; }
    }
    // intermediate rows already zero

    // 执行 ops：每个 op 用对应的选择矩阵乘 V 得到 rows_mat (rows_op x cols)
    for op in &plan.ops {
        match op {
            MatOp::BinaryMat { op: bop, left_mat, right_mat, out_start, rows: op_rows } => {
                let left_selected = mat_mul(&plan.matrices[*left_mat], &V);   // op_rows x cols
                let right_selected = mat_mul(&plan.matrices[*right_mat], &V); // op_rows x cols
                let result = mat_elemwise_binary(&left_selected, &right_selected, bop);
                write_rows_to_V(&mut V, &result, *out_start);
            }
            MatOp::UnaryMat { func, mat, out_start, rows: op_rows } => {
                let sel = mat_mul(&plan.matrices[*mat], &V); // op_rows x cols
                let res = mat_elemwise_unary(&sel, func);
                write_rows_to_V(&mut V, &res, *out_start);
            }
            MatOp::CondBlendMat { cond_mat, then_mat, else_mat, out_start, rows: op_rows } => {
                let cond_sel = mat_mul(&plan.matrices[*cond_mat], &V); // rows x cols
                // 生成 mask（0/1），这里采用 >0 判定；如果需要不同阈值可改
                let mask = mat_greater_zero(&cond_sel);
                let then_sel = mat_mul(&plan.matrices[*then_mat], &V);
                let else_sel = mat_mul(&plan.matrices[*else_mat], &V);
                let blended = mat_blend(&mask, &then_sel, &else_sel);
                write_rows_to_V(&mut V, &blended, *out_start);
            }
        }
    }

    // 从 V 取出 top_outputs 行，返回为 Vec<Vec<f32>> 每个为 length cols
    plan.top_outputs.iter().map(|&ridx| V[ridx].clone()).collect()
}

/// 通用的逐元素变换函数类型
type ElemwiseFn = Box<dyn Fn(&[&Mat2D]) -> Mat2D>;

/// 通用的矩阵操作描述符
struct GenericMatOp {
    input_mats: Vec<usize>,    // 输入选择矩阵的索引
    output_start: usize,       // 输出在V中的起始位置
    rows: usize,               // 操作的行数
    transform: ElemwiseFn,     // 逐元素变换函数
}

/// 批量模拟的通用版本
pub fn simulate_matrix_plan_batch_generic(plan: &MatrixPlan) -> Vec<Vec<f32>> {
    let cols = 1;
    let rows = plan.final_v_len;
    let mut V: Mat2D = vec![vec![0.0; cols]; rows];

    // // 初始化V矩阵（变量和常量）
    // for i in 0..plan.variable_count {
    //     V[i].copy_from_slice(&inputs[i]);
    // }
    for (idx, val) in &plan.constant_values {
        for c in 0..cols { V[*idx][c] = *val; }
    }

    // 将原始操作转换为通用操作
    let generic_ops = convert_to_generic_ops(&plan.ops);

    // 统一执行所有通用操作
    for op in &generic_ops {
        // 1. 输入选择：用选择矩阵从V中提取输入数据
        let selected_inputs: Vec<Mat2D> = op.input_mats.iter()
            .map(|&mat_idx| mat_mul(&plan.matrices[mat_idx], &V))
            .collect();
        
        // 2. 逐元素变换：对每个lane执行相同的变换
        let refs: Vec<&Mat2D> = selected_inputs.iter().collect();
        let result = (op.transform)(&refs);
        
        // 3. 结果写回：将结果写入V的指定位置
        write_rows_to_V(&mut V, &result, op.output_start);
    }

    plan.top_outputs.iter().map(|&ridx| V[ridx].clone()).collect()
}


/// 将特定操作转换为通用操作
fn convert_to_generic_ops(ops: &[MatOp]) -> Vec<GenericMatOp> {
    ops.iter().map(|op| match op {
        MatOp::BinaryMat { op: bop, left_mat, right_mat, out_start, rows } => {
            let op_clone = bop.clone();
            GenericMatOp {
                input_mats: vec![*left_mat, *right_mat],
                output_start: *out_start,
                rows: *rows,
                transform: Box::new(move |inputs| {
                    let left = inputs[0];
                    let right = inputs[1];
                    mat_elemwise_binary(left, right, &op_clone)
                }),
            }
        },
        MatOp::UnaryMat { func, mat, out_start, rows } => {
            let func_clone = func.clone();
            GenericMatOp {
                input_mats: vec![*mat],
                output_start: *out_start,
                rows: *rows,
                transform: Box::new(move |inputs| {
                    mat_elemwise_unary(inputs[0], &func_clone)
                }),
            }
        },
        MatOp::CondBlendMat { cond_mat, then_mat, else_mat, out_start, rows } => {
            GenericMatOp {
                input_mats: vec![*cond_mat, *then_mat, *else_mat],
                output_start: *out_start,
                rows: *rows,
                transform: Box::new(|inputs| {
                    let cond = inputs[0];
                    let then = inputs[1];
                    let else_ = inputs[2];
                    let mask = mat_greater_zero(cond);
                    mat_blend(&mask, then, else_)
                }),
            }
        },
    }).collect()
}
#[test]
fn once_batch_matrix() {
    use crate::core::dsl::*;
    
    // 创建表达式
    let _if = Expr::If { 
        condition: Box::new(eq(Expr::RenderImport("uv"), Expr::Constant(2.0))), 
        then_branch: Box::new(Expr::Constant(5.0)), 
        else_branch: Box::new(Expr::Constant(11.0)) 
    };
    let expr = wvec3(_if.clone(), _if.clone(), Expr::RenderImport("uv"));

    // 创建并配置导入注册表
    let mut import_register = ImportRegistry::new();
    
    // 注册UV导入处理器 - 返回vec4格式的UV坐标
    import_register.register_render_import("uv", 0b01, Box::new(|input| {
        // 模拟UV坐标：假设输入是像素坐标，转换为0-1范围的RGBA
        if input.is_empty() {
            vec![0.5, 0.5, 0.0, 1.0] // 默认UV值
        } else {
            vec![
                input[0] / 1000.0,  // U
                input[1] / 1000.0,  // V  
                0.0,                // 固定值
                1.0                 // Alpha
            ]
        }
    }));

    // 编译计划
    let plan = compile_to_matrix_plan_with_imports(&expr, &import_register);

    println!("编译计划信息:");
    println!("  - 最终V长度: {}", plan.final_v_len);
    println!("  - 顶层输出数量: {}", plan.top_outputs.len());
    println!("  - 操作数量: {}", plan.ops.len());
    println!("  - 导入数量: {}", plan.imports.len());
    println!("  - 常量数量: {}", plan.constant_values.len());

    // 检查导入信息
    for import in &plan.imports {
        println!("导入: {:?}, 掩码: {}, 索引: {}", import.import_type, import.mask, import.index);
    }

    // 由于有RenderImport，我们需要模拟渲染输入
    // 假设我们有一个512x512的纹理，测试几个像素位置
    let test_uvs = vec![
        vec![0.0, 0.0],    // 左上角
        vec![256.0, 256.0], // 中心
        vec![511.0, 511.0], // 右下角
    ];

    // 对每个测试UV执行计算
    for (i, uv) in test_uvs.iter().enumerate() {
        println!("\n测试UV {}: {:?}", i, uv);
        
        // 执行导入处理
        let uv_values = import_register.execute_import(&ImportType::Render("uv".to_string()), uv);
        println!("UV处理结果: {:?}", uv_values);
        
        // 准备输入数据（这里需要根据实际的V结构来组织）
        // 由于有RenderImport，我们需要将UV值放在正确的位置
        let mut inputs = Vec::new();
        
        // 根据plan中的导入索引位置来组织输入
        // 这是一个简化的模拟 - 实际实现需要更精确的输入映射
        if let Some(uv_import) = plan.imports.iter().find(|i| matches!(&i.import_type, ImportType::Render(name) if name == "uv")) {
            // 创建一个足够大的输入向量，在UV导入的位置放入UV值
            let mut input_row = vec![0.0; plan.final_v_len];
            // 将UV值放入对应的位置（这里简化处理，只放第一个分量）
            if !uv_values.is_empty() {
                input_row[uv_import.index] = uv_values[0]; // 使用U分量
            }
            inputs.push(input_row);
        }
        
        // 执行计算
        let outputs = simulate_matrix_plan_batch(&plan, &inputs);
        println!("计算结果: {:?}", outputs);
    }

    // 也测试通用版本
    println!("\n使用通用版本模拟:");
    let outputs = simulate_matrix_plan_batch_generic(&plan);
    println!("通用版本输出: {:?}", outputs);
    
    assert_eq!(outputs.len(), 3);
    
    // 验证结果合理性
    // 由于条件 Expr::RenderImport("uv") == Constant(2.0) 应该总是false
    // 所以_if表达式应该返回else_branch的值11.0
    // 第三个分量是RenderImport("uv")本身
    
    println!("测试完成!");
}

// #[test]
// fn gpu_once_batch_matrix(){
//     use crate::core::dsl::*;
//     let _if = Expr::If { condition: Box::new(eq(Expr::Constant(1.0), Expr::Constant(2.0))), then_branch: Box::new(Expr::Constant(5.0)), else_branch: Box::new(Expr::Constant(11.0)) };
//     let expr = wvec3(_if.clone(), _if.clone(),33.0);
//     let plan = compile_to_matrix_plan_with_imports(&expr);
//     let inputs = vec![
//         vec![1.0_f32], // a
//         vec![3.0_f32], // b
//         vec![5.0_f32], // c
//         vec![7.0_f32], // d
//     ];
//     let outputs = simulate_matrix_plan_batch_generic(&plan);
//     println!("outputs {:?}",outputs);
// }