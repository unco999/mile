use crate::test::{BinaryOp, Expr, UnaryFunc};


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
    pub matrices: Vec<Matrix>,      // 选择矩阵集合 (rows x current_v_length_at_creation)
    pub ops: Vec<MatOp>,            // 按执行顺序
    pub top_outputs: Vec<usize>,    // 顶层表达式输出在最终 v 中的位置（分量化）
    pub variable_count: usize,      // 初始变量数量（inputs.len())
    pub constant_values: Vec<(usize, f32)>, // (index_in_v, value) for constants placed into initial v
    pub final_v_len: usize,         // 编译结束时 final v 的长度 (variable + constants + intermediates)
}

/// 把 AST 编译成矩阵计划（selection matrices + ops）
/// 原则：按自底向上遍历，遇到常量就把常量 append 到初始 v（并记录其 index），
/// 每遇到一个操作，先确定当前 v 长度 cur_len（这是矩阵的列数），根据子节点索引创建选择矩阵，
/// 然后为该 op 的 outputs 预留 out_start..out_start+rows-1（并 advance next_index）。
pub fn compile_to_matrix_plan(expr: &Expr, variables: &[&str]) -> MatrixPlan {
    let mut matrices: Vec<Matrix> = Vec::new();
    let mut ops: Vec<MatOp> = Vec::new();
    let mut constant_values: Vec<(usize, f32)> = Vec::new();

    // next_index 是当前 v 的长度（初始 = variable_count)
    let mut next_index: usize = variables.len();

    // 递归函数返回当前子表达式对应的索引列表（在 v 中）
    fn rec(
        expr: &Expr,
        variables: &[&str],
        matrices: &mut Vec<Matrix>,
        ops: &mut Vec<MatOp>,
        constant_values: &mut Vec<(usize, f32)>,
        next_index: &mut usize,
    ) -> Vec<usize> {
        match expr {
            Expr::Variable(name) => {
                let idx = variables.iter().position(|&v| v == name).expect("unknown variable");
                vec![idx]
            }
            Expr::Constant(val) => {
                // 把常量放到初始 v（append），并记录
                let idx = *next_index;
                *next_index += 1;
                constant_values.push((idx, *val));
                vec![idx]
            }
            Expr::Vec2(v) => {
                let x = rec(&v.x, variables, matrices, ops, constant_values, next_index);
                let y = rec(&v.y, variables, matrices, ops, constant_values, next_index);
                vec![x[0], y[0]]
            }
            Expr::Vec3(v) => {
                let x = rec(&v.x, variables, matrices, ops, constant_values, next_index);
                let y = rec(&v.y, variables, matrices, ops, constant_values, next_index);
                let z = rec(&v.z, variables, matrices, ops, constant_values, next_index);
                vec![x[0], y[0], z[0]]
            }
            Expr::Vec4(v) => {
                let x = rec(&v.x, variables, matrices, ops, constant_values, next_index);
                let y = rec(&v.y, variables, matrices, ops, constant_values, next_index);
                let z = rec(&v.z, variables, matrices, ops, constant_values, next_index);
                let w = rec(&v.w, variables, matrices, ops, constant_values, next_index);
                vec![x[0], y[0], z[0], w[0]]
            }
            Expr::BinaryOp(op, left, right) => {
                // 先递归子节点（它们会把自身的中间量/常量分配到 v 中）
                let l_idxs = rec(left, variables, matrices, ops, constant_values, next_index);
                let r_idxs = rec(right, variables, matrices, ops, constant_values, next_index);

                let comps = l_idxs.len().max(r_idxs.len());
                let mut outs = Vec::with_capacity(comps);

                // IMPORTANT: 列数 = 当前 next_index（此时还未为本 op 预留 output）
                let cur_cols = *next_index;

                // 为本 op 的每个并行分量创建选择矩阵的行（我们把每一行做成单独的 1xcur_cols 矩阵，方便管理）
                // 然后把这些行组合成一个 rows x cols 的矩阵
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

                // 现在为本 op 的 outputs 预留一段索引（这对应把中间量 append 到 v）
                let out_start = *next_index;
                *next_index += comps; // reserve

                // push op
                ops.push(MatOp::BinaryMat { op: op.clone(), left_mat: left_mat_idx, right_mat: right_mat_idx, out_start, rows: comps });

                for i in 0..comps {
                    outs.push(out_start + i);
                }
                outs
            }
            Expr::UnaryOp(func, sub) => {
                let s_idxs = rec(sub, variables, matrices, ops, constant_values, next_index);
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
                ops.push(MatOp::UnaryMat { func: func.clone(), mat: mat_idx, out_start, rows: comps });
                (0..comps).map(|i| out_start + i).collect()
            }
            Expr::If { condition, then_branch, else_branch } => {
                let c_idxs = rec(condition, variables, matrices, ops, constant_values, next_index);
                let t_idxs = rec(then_branch, variables, matrices, ops, constant_values, next_index);
                let e_idxs = rec(else_branch, variables, matrices, ops, constant_values, next_index);

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

                (0..comps).map(|i| out_start + i).collect()
            }
        }
    }

    let mut next_idx_local = next_index;
    let top_outputs = rec(expr, variables, &mut matrices, &mut ops, &mut constant_values, &mut next_idx_local);
    let final_v_len = next_idx_local;

    MatrixPlan {
        matrices,
        ops,
        top_outputs,
        variable_count: variables.len(),
        constant_values,
        final_v_len,
    }
}

/// 在 CPU 上用「矩阵流水线」一次性模拟（per-lane）执行 plan
/// - inputs: 每个输入变量对应一个 Vec<f32>（长度 vector_len）
/// - 返回：Vec<output_buffer> 每个 output_buffer 长度为 vector_len
pub fn simulate_matrix_plan_once(plan: &MatrixPlan, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if inputs.is_empty() {
        return vec![];
    }
    let vector_len = inputs[0].len();
    // 输出初始化
    let mut outputs: Vec<Vec<f32>> = vec![vec![0.0; vector_len]; plan.top_outputs.len()];

    for lane in 0..vector_len {
        // 构造初始 v：variables + constants (constants 填入对应索引)
        let mut v: Vec<f32> = vec![0.0; plan.final_v_len];
        // 填变量
        for i in 0..plan.variable_count {
            v[i] = inputs[i][lane];
        }
        // 填常量（如果常量 index < final_v_len）
        for (idx, val) in &plan.constant_values {
            v[*idx] = *val;
        }
        // 中间位初始 0 已经设置

        // 按 ops 顺序执行矩阵流水线：每个 op 先用选择矩阵乘 v 得到一列向量（rows 长度），
        // 然后对每行做非线性逐元素操作，写回 v 的预留位置 out_start + r
        for op in &plan.ops {
            match op {
                MatOp::BinaryMat { op: bop, left_mat, right_mat, out_start, rows } => {
                    let left = &plan.matrices[*left_mat];
                    let right = &plan.matrices[*right_mat];
                    for r in 0..*rows {
                        // compute left_val = dot(left.row(r), v)
                        let mut left_val = 0.0f32;
                        let row_slice = left.row_slice(r);
                        for (c, coeff) in row_slice.iter().enumerate() {
                            if *coeff != 0.0 {
                                left_val += coeff * v[c];
                            }
                        }
                        let mut right_val = 0.0f32;
                        let row_slice_r = right.row_slice(r);
                        for (c, coeff) in row_slice_r.iter().enumerate() {
                            if *coeff != 0.0 {
                                right_val += coeff * v[c];
                            }
                        }
                        let res = match bop {
                            BinaryOp::Add => left_val + right_val,
                            BinaryOp::Subtract => left_val - right_val,
                            BinaryOp::Multiply => left_val * right_val,
                            BinaryOp::Divide => left_val / right_val,
                            BinaryOp::Modulo => left_val % right_val,
                            BinaryOp::Pow => left_val.powf(right_val),
                            BinaryOp::GreaterThan => if left_val > right_val { 1.0 } else { 0.0 },
                            BinaryOp::GreaterEqual => if left_val >= right_val { 1.0 } else { 0.0 },
                            BinaryOp::LessThan => if left_val < right_val { 1.0 } else { 0.0 },
                            BinaryOp::LessEqual => if left_val <= right_val { 1.0 } else { 0.0 },
                            BinaryOp::Equal => if (left_val - right_val).abs() < std::f32::EPSILON { 1.0 } else { 0.0 },
                            BinaryOp::NotEqual => if (left_val - right_val).abs() >= std::f32::EPSILON { 1.0 } else { 0.0 },
                        };
                        v[out_start + r] = res;
                    }
                }
                MatOp::UnaryMat { func, mat, out_start, rows } => {
                    let m = &plan.matrices[*mat];
                    for r in 0..*rows {
                        let mut a_val = 0.0f32;
                        let row_slice = m.row_slice(r);
                        for (c, coeff) in row_slice.iter().enumerate() {
                            if *coeff != 0.0 {
                                a_val += coeff * v[c];
                            }
                        }
                        let res = match func {
                            UnaryFunc::Sin => a_val.sin(),
                            UnaryFunc::Cos => a_val.cos(),
                            UnaryFunc::Tan => a_val.tan(),
                            UnaryFunc::Exp => a_val.exp(),
                            UnaryFunc::Log => a_val.ln(),
                            UnaryFunc::Sqrt => a_val.sqrt(),
                            UnaryFunc::Abs => a_val.abs(),
                        };
                        v[out_start + r] = res;
                    }
                }
                MatOp::CondBlendMat { cond_mat, then_mat, else_mat, out_start, rows } => {
                    let cmat = &plan.matrices[*cond_mat];
                    let tmat = &plan.matrices[*then_mat];
                    let emat = &plan.matrices[*else_mat];
                    for r in 0..*rows {
                        let mut cond_val = 0.0f32;
                        for (c, coeff) in cmat.row_slice(r).iter().enumerate() {
                            if *coeff != 0.0 {
                                cond_val += coeff * v[c];
                            }
                        }
                        let mut then_val = 0.0f32;
                        for (c, coeff) in tmat.row_slice(r).iter().enumerate() {
                            if *coeff != 0.0 {
                                then_val += coeff * v[c];
                            }
                        }
                        let mut else_val = 0.0f32;
                        for (c, coeff) in emat.row_slice(r).iter().enumerate() {
                            if *coeff != 0.0 {
                                else_val += coeff * v[c];
                            }
                        }
                        // blend: cond * then + (1-cond) * else
                        v[out_start + r] = cond_val * then_val + (1.0 - cond_val) * else_val;
                    }
                }
            }
        }

        // 从 v 中读取 top outputs并写入 outputs
        for (out_i, &v_idx) in plan.top_outputs.iter().enumerate() {
            outputs[out_i][lane] = v[v_idx];
        }
    }

    outputs
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

    // fill input variable rows
    for i in 0..plan.variable_count {
        V[i].copy_from_slice(&inputs[i]);
    }
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

#[test]
fn once_batch_matrix() {
    use crate::test::dsl::*;
    // expr = a*b + c*d
    let _if = Expr::If { condition: Box::new(eq(Expr::Constant(1.0), Expr::Constant(2.0))), then_branch: Box::new(Expr::Constant(5.0)), else_branch: Box::new(Expr::Constant(11.0)) };
    let expr = var("a") * var("b") + var("c") * var("d");
    let plan = compile_to_matrix_plan(&_if, &["a","b","c","d"]);
    let inputs = vec![
        vec![1.0_f32], // a
        vec![3.0_f32], // b
        vec![5.0_f32], // c
        vec![7.0_f32], // d
    ];
    let outputs = simulate_matrix_plan_batch(&plan, &inputs);
    println!("batch 输出: {:?}", outputs);
    assert_eq!(outputs.len(), 1);
}