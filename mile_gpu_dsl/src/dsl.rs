
use std::fmt::Debug;

use std::collections::HashMap;
use ndarray::{Array2, Array1};
use bitflags::bitflags;

use crate::traits::VectorComponents;


bitflags::bitflags! {
    #[derive(Default)]
    pub struct GpuFuncFlag: u64 {
        const NONE        = 0;
        const SIN         = 1 << 0;
        const COS         = 1 << 1;
        const ABS         = 1 << 2;
        const MAX         = 1 << 3;
        const MIN         = 1 << 4;
        const CLAMP       = 1 << 5;
        const DOT         = 1 << 6;
        const NORMALIZE   = 1 << 7;
        const LERP        = 1 << 8;
        const X        = 1 << 9;
        const Y        = 1 << 10;
        const Z        = 1 << 11;
        const W        = 1 << 12;
    }
}

#[derive(Clone, Debug)]
pub enum GpuFunc {
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Abs(Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Clamp(Box<Expr>, Box<Expr>, Box<Expr>),
    Dot(Box<Expr>, Box<Expr>),
    Normalize(Box<Expr>),
    Lerp(Box<Expr>, Box<Expr>, Box<Expr>),
    X(Box<Expr>),    // 新增
    Y(Box<Expr>),    // 新增
    Z(Box<Expr>),    // 新增
    W(Box<Expr>),    // 新增

}

impl GpuFunc {
    #[inline]
    pub fn flag(&self) -> GpuFuncFlag {
        match self {
            GpuFunc::Sin(_)         => GpuFuncFlag::SIN,
            GpuFunc::Cos(_)         => GpuFuncFlag::COS,
            GpuFunc::Abs(_)         => GpuFuncFlag::ABS,
            GpuFunc::Max(_, _)      => GpuFuncFlag::MAX,
            GpuFunc::Min(_, _)      => GpuFuncFlag::MIN,
            GpuFunc::Clamp(_, _, _) => GpuFuncFlag::CLAMP,
            GpuFunc::Dot(_, _)      => GpuFuncFlag::DOT,
            GpuFunc::Normalize(_)   => GpuFuncFlag::NORMALIZE,
            GpuFunc::Lerp(_, _, _)  => GpuFuncFlag::LERP,
            GpuFunc::X(expr) => todo!(),
            GpuFunc::Y(expr) => todo!(),
            GpuFunc::Z(expr) => todo!(),
            GpuFunc::W(expr) => todo!(),
        }
    }

    #[inline]
    pub fn flag_bits(&self) -> u64 {
        self.flag().bits()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GpuVar {
    Uv,
    Color,
    Normal,
    Position,
    Time,
    GlobalIdX,
    GlobalIdY,
    BufferInput(u32),
    BufferOutput(u32),
}

#[derive(Clone, Debug)]
pub enum Expr {
    Var(String, usize),      // name, size (cpu-side var)
    Vec2(Box<Expr>,Box<Expr>),
    Vec3(Box<Expr>,Box<Expr>,Box<Expr>),
    Vec4(Box<Expr>,Box<Expr>,Box<Expr>,Box<Expr>),
    GpuVar(GpuVar),          // builtin GPU var (treated as input column)
    Scalar(f32),
    Call(GpuFunc),           // fixed-call functions
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),
    If {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
}

fn eval(expr: &Expr, vars: &HashMap<String, f32>, gpu_ctx: &HashMap<GpuVar, f32>) -> Vec<f32> {
    match expr {
        Expr::Scalar(v) => vec![*v],
        Expr::Var(name, _) => vec![*vars.get(name).unwrap_or(&0.0)],
        Expr::GpuVar(v) => vec![*gpu_ctx.get(v).unwrap_or(&0.0)],

        Expr::Add(a, b) => {
            let va = eval(a, vars, gpu_ctx);
            let vb = eval(b, vars, gpu_ctx);
            va.iter().zip(vb.iter().cycle()).map(|(x, y)| x + y).collect()
        },
        Expr::Sub(a, b) => {
            let va = eval(a, vars, gpu_ctx);
            let vb = eval(b, vars, gpu_ctx);
            va.iter().zip(vb.iter().cycle()).map(|(x, y)| x - y).collect()
        },
        Expr::Mul(a, b) => {
            let va = eval(a, vars, gpu_ctx);
            let vb = eval(b, vars, gpu_ctx);
            va.iter().zip(vb.iter().cycle()).map(|(x, y)| x * y).collect()
        },
        Expr::Div(a, b) => {
            let va = eval(a, vars, gpu_ctx);
            let vb = eval(b, vars, gpu_ctx);
            va.iter().zip(vb.iter().cycle()).map(|(x, y)| x / y).collect()
        },

        Expr::Vec2(a, b) => {
            let mut res = eval(a, vars, gpu_ctx);
            res.extend(eval(b, vars, gpu_ctx));
            res
        },
        Expr::Vec3(a, b, c) => {
            let mut res = eval(a, vars, gpu_ctx);
            res.extend(eval(b, vars, gpu_ctx));
            res.extend(eval(c, vars, gpu_ctx));
            res
        },
        Expr::Vec4(a, b, c, d) => {
            let mut res = eval(a, vars, gpu_ctx);
            res.extend(eval(b, vars, gpu_ctx));
            res.extend(eval(c, vars, gpu_ctx));
            res.extend(eval(d, vars, gpu_ctx));
            res
        },

        Expr::Call(func) => {
            // ⚠️ 对于函数，你也需要处理返回 Vec<f32>，如果是标量函数可以直接返回长度 1
            match func {
                GpuFunc::Sin(inner) => eval(inner, vars, gpu_ctx).into_iter().map(|v| v.sin()).collect(),
                GpuFunc::Cos(inner) => eval(inner, vars, gpu_ctx).into_iter().map(|v| v.cos()).collect(),
                // 其他同理
                _ => vec![0.0],
            }
        },

        _ => vec![0.0], // 其他暂时简化
    }
}



// ---------------- 新增：Compute 表/条目/输入引用 ----------------
#[derive(Clone, Copy, Debug)]
pub enum ComputeOpKind {
    Sin, Cos, Abs, Max, Min, Clamp, Dot, Normalize, Lerp,
    X,Y,Z,W
}

#[derive(Clone, Debug)]
pub enum InputRef {
    Node(usize),  // node index
    Col(usize),   // column index in X
    Const(f32),   // constant
}

#[derive(Clone, Debug)]
pub struct ComputeOpEntry {
    pub node_idx: usize,        // target node index (which node's bias to write)
    pub kind: ComputeOpKind,
    pub inputs: Vec<InputRef>,  // input refs in order
}

#[derive(Clone, Debug, Default)]
pub struct ComputeTable {
    pub entries: Vec<ComputeOpEntry>,
}
// -------------------------------------------------------------

/// 把表达式展开成 node_ws/node_bs，并收集 ComputeTable（Call 节点）
/// 返回 (X, node_ws, node_bs, node_names, compute_table)
pub fn build_all_nodes_as_linear(
    root: &Expr,
    env: &HashMap<String, Vec<f32>>,
    extra_gpu_inputs: &Vec<GpuVar>,
) -> (Array2<f32>, Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<String>, ComputeTable) {
    // collect columns
    let mut col_keys: Vec<String> = Vec::new();
    for k in env.keys() { col_keys.push(k.clone()); }
    for gv in extra_gpu_inputs {
        let name = gpuvar_name(gv);
        if !col_keys.contains(&name) { col_keys.push(name); }
    }

    let n = env.values().map(|v| v.len()).max().unwrap_or(1);
    let m = col_keys.len();
    let mut X = Array2::<f32>::zeros((n, m));
    for (j, col) in col_keys.iter().enumerate() {
        if let Some(v) = env.get(col) {
            for i in 0..n { X[[i,j]] = *v.get(i).unwrap_or(&v[0]); }
        } else {
            for i in 0..n { X[[i,j]] = 0.0; }
        }
    }

    let mut node_ws: Vec<Array2<f32>> = Vec::new();
    let mut node_bs: Vec<Array2<f32>> = Vec::new();
    let mut node_names: Vec<String> = Vec::new();
    let mut compute_table = ComputeTable::default();

    fn rec(
        expr: &Expr,
        X: &Array2<f32>,
        col_keys: &Vec<String>,
        node_ws: &mut Vec<Array2<f32>>,
        node_bs: &mut Vec<Array2<f32>>,
        node_names: &mut Vec<String>,
        compute_table: &mut ComputeTable,
    ) -> usize {
        let n = X.nrows();
        let m = col_keys.len();

        match expr {
            Expr::Var(name, _size) => {
                let mut W = Array2::<f32>::zeros((n, m));
                if let Some(col_idx) = col_keys.iter().position(|s| s == name) {
                    for i in 0..n { W[[i, col_idx]] = 1.0; }
                }
                let mut B = Array2::<f32>::zeros((n,1));
                node_ws.push(W); node_bs.push(B);
                node_names.push(format!("Var({})", name));
                return node_ws.len() - 1;
            }
            Expr::GpuVar(gv) => {
                let name = gpuvar_name(gv);
                let mut W = Array2::<f32>::zeros((n, m));
                if let Some(col_idx) = col_keys.iter().position(|s| s == &name) {
                    for i in 0..n { W[[i, col_idx]] = 1.0; }
                }
                let mut B = Array2::<f32>::zeros((n,1));
                node_ws.push(W); node_bs.push(B);
                node_names.push(format!("GpuVar({})", name));
                return node_ws.len() - 1;
            }
            Expr::Scalar(s) => {
                let W = Array2::<f32>::zeros((n, m));
                let mut B = Array2::<f32>::zeros((n,1));
                for i in 0..n { B[[i,0]] = *s; }
                node_ws.push(W); node_bs.push(B);
                node_names.push(format!("Const({})", s));
                return node_ws.len() - 1;
            }
            Expr::Add(a,b) => {
                let ia = rec(a, X, col_keys, node_ws, node_bs, node_names, compute_table);
                let ib = rec(b, X, col_keys, node_ws, node_bs, node_names, compute_table);
                let Wa = &node_ws[ia];
                let Wb = &node_ws[ib];
                let Ba = &node_bs[ia];
                let Bb = &node_bs[ib];
                let W = Wa + Wb;
                let B = Ba + Bb;
                node_ws.push(W); node_bs.push(B);
                node_names.push(format!("Add(n{},n{})", ia, ib));
                return node_ws.len() - 1;
            }
            Expr::Sub(a,b) => {
                let ia = rec(a, X, col_keys, node_ws, node_bs, node_names, compute_table);
                let ib = rec(b, X, col_keys, node_ws, node_bs, node_names, compute_table);
                let W = &node_ws[ia] - &node_ws[ib];
                let B = &node_bs[ia] - &node_bs[ib];
                node_ws.push(W); node_bs.push(B);
                node_names.push(format!("Sub(n{},n{})", ia, ib));
                return node_ws.len() - 1;
            }
            Expr::Mul(a,b) | Expr::Div(a,b) | Expr::Mod(a,b) => {
                let vals = compute_expr(expr, X, col_keys);
                let W = Array2::<f32>::zeros((n, m));
                let mut B = Array2::<f32>::zeros((n,1));
                for i in 0..n { B[[i,0]] = vals[i]; }
                node_ws.push(W); node_bs.push(B);
                node_names.push(format!("{:?}", expr_op_name(expr)));
                return node_ws.len() - 1;
            }
            Expr::If { cond, then_expr, else_expr } => {
                let ic = rec(cond, X, col_keys, node_ws, node_bs, node_names, compute_table);
                let it = rec(then_expr, X, col_keys, node_ws, node_bs, node_names, compute_table);
                let ie = rec(else_expr, X, col_keys, node_ws, node_bs, node_names, compute_table);
                let mask = compute_expr(cond, X, col_keys);
                let mut W = Array2::<f32>::zeros((n, m));
                let mut B = Array2::<f32>::zeros((n,1));
                for i in 0..n {
                    let mv = mask[i];
                    for j in 0..m {
                        W[[i, j]] = node_ws[it][[i, j]] * (1.0 - mv) + node_ws[ie][[i, j]] * mv;
                    }
                    B[[i, 0]] = node_bs[it][[i, 0]] * (1.0 - mv) + node_bs[ie][[i, 0]] * mv;
                }
                node_ws.push(W); node_bs.push(B);
                node_names.push(format!("If(n{},n{},n{})", ic, it, ie));
                return node_ws.len() - 1;
            }
            Expr::Eq(_,_) | Expr::Ne(_,_) | Expr::Lt(_,_) | Expr::Le(_,_) | Expr::Gt(_,_) | Expr::Ge(_,_) => {
                let vals = compute_expr(expr, X, col_keys);
                let W = Array2::<f32>::zeros((n, m));
                let mut B = Array2::<f32>::zeros((n,1));
                for i in 0..n { B[[i,0]] = vals[i]; }
                node_ws.push(W); node_bs.push(B);
                node_names.push(format!("Cmp({:?})", expr_op_name(expr)));
                return node_ws.len() - 1;
            }
Expr::Vec2(a, b) => {
    let ia = rec(a, X, col_keys, node_ws, node_bs, node_names, compute_table);
    let ib = rec(b, X, col_keys, node_ws, node_bs, node_names, compute_table);

    // 构造 node 输出矩阵
    let n = X.nrows();
    let m = col_keys.len();
    let mut W = Array2::<f32>::zeros((n, m));
    let mut B = Array2::<f32>::zeros((n, 2)); // Vec2 输出 2 列

    // 从子节点拷贝值到 B
    for i in 0..n {
        B[[i,0]] = node_bs[ia][[i,0]];
        B[[i,1]] = node_bs[ib][[i,0]];
    }

    node_ws.push(W);
    node_bs.push(B);
    node_names.push(format!("Vec2(n{},n{})", ia, ib));
    return node_ws.len() - 1;
},

Expr::Vec3(a, b, c) => {
    let ia = rec(a, X, col_keys, node_ws, node_bs, node_names, compute_table);
    let ib = rec(b, X, col_keys, node_ws, node_bs, node_names, compute_table);
    let ic = rec(c, X, col_keys, node_ws, node_bs, node_names, compute_table);

    let n = X.nrows();
    let m = col_keys.len();
    let mut W = Array2::<f32>::zeros((n, m));
    let mut B = Array2::<f32>::zeros((n, 3)); // Vec3 输出 3 列

    for i in 0..n {
        B[[i,0]] = node_bs[ia][[i,0]];
        B[[i,1]] = node_bs[ib][[i,0]];
        B[[i,2]] = node_bs[ic][[i,0]];
    }

    node_ws.push(W);
    node_bs.push(B);
    node_names.push(format!("Vec3(n{},n{},n{})", ia, ib, ic));
    return node_ws.len() - 1;
},

Expr::Vec4(a, b, c, d) => {
    let ia = rec(a, X, col_keys, node_ws, node_bs, node_names, compute_table);
    let ib = rec(b, X, col_keys, node_ws, node_bs, node_names, compute_table);
    let ic = rec(c, X, col_keys, node_ws, node_bs, node_names, compute_table);
    let id = rec(d, X, col_keys, node_ws, node_bs, node_names, compute_table);

    let n = X.nrows();
    let m = col_keys.len();
    let mut W = Array2::<f32>::zeros((n, m));
    let mut B = Array2::<f32>::zeros((n, 4)); // Vec4 输出 4 列

    for i in 0..n {
        B[[i,0]] = node_bs[ia][[i,0]];
        B[[i,1]] = node_bs[ib][[i,0]];
        B[[i,2]] = node_bs[ic][[i,0]];
        B[[i,3]] = node_bs[id][[i,0]];
    }

    node_ws.push(W);
    node_bs.push(B);
    node_names.push(format!("Vec4(n{},n{},n{},n{})", ia, ib, ic, id));
    return node_ws.len() - 1;
},
            // Call: 生成 node 占位，并把 entry push 到 compute_table
            Expr::Call(f) => {
                let input_nodes: Vec<usize> = match f {
                    GpuFunc::Sin(x) | GpuFunc::Cos(x) | GpuFunc::Abs(x) | GpuFunc::Normalize(x) => {
                        vec![rec(x, X, col_keys, node_ws, node_bs, node_names, compute_table)]
                    }
                    GpuFunc::Max(a,b) | GpuFunc::Min(a,b) | GpuFunc::Dot(a,b) => {
                        vec![
                            rec(a, X, col_keys, node_ws, node_bs, node_names, compute_table),
                            rec(b, X, col_keys, node_ws, node_bs, node_names, compute_table)
                        ]
                    }
                    GpuFunc::Clamp(x, low, high) | GpuFunc::Lerp(x, low, high) => {
                        vec![
                            rec(x, X, col_keys, node_ws, node_bs, node_names, compute_table),
                            rec(low, X, col_keys, node_ws, node_bs, node_names, compute_table),
                            rec(high, X, col_keys, node_ws, node_bs, node_names, compute_table)
                        ]
                    }
                    GpuFunc::X(expr) => vec![rec(&expr.clone(), X, col_keys, node_ws, node_bs, node_names, compute_table)],
                    GpuFunc::Y(expr) => vec![rec(&expr.clone(), X, col_keys, node_ws, node_bs, node_names, compute_table)],
                    GpuFunc::Z(expr) => vec![rec(&expr.clone(), X, col_keys, node_ws, node_bs, node_names, compute_table)],
                    GpuFunc::W(expr) => vec![rec(&expr.clone(), X, col_keys, node_ws, node_bs, node_names, compute_table)],
                };

                let W = Array2::<f32>::zeros((n, m));
                let B = Array2::<f32>::zeros((n,1));
                node_ws.push(W);
                node_bs.push(B);
                let node_idx = node_ws.len() - 1;

                let kind = match f {
                    GpuFunc::Sin(_) => ComputeOpKind::Sin,
                    GpuFunc::Cos(_) => ComputeOpKind::Cos,
                    GpuFunc::Abs(_) => ComputeOpKind::Abs,
                    GpuFunc::Max(_,_) => ComputeOpKind::Max,
                    GpuFunc::Min(_,_) => ComputeOpKind::Min,
                    GpuFunc::Clamp(_,_,_) => ComputeOpKind::Clamp,
                    GpuFunc::Dot(_,_) => ComputeOpKind::Dot,
                    GpuFunc::Normalize(_) => ComputeOpKind::Normalize,
                    GpuFunc::Lerp(_,_,_) => ComputeOpKind::Lerp,
                    GpuFunc::X(_) => ComputeOpKind::X,
                    GpuFunc::Y(_) => ComputeOpKind::Y,
                    GpuFunc::Z(_) => ComputeOpKind::Z,
                    GpuFunc::W(_) => ComputeOpKind::W,
                };
                let inputs_ref = input_nodes.iter().map(|&idx| InputRef::Node(idx)).collect::<Vec<_>>();
                compute_table.entries.push(ComputeOpEntry {
                    node_idx,
                    kind,
                    inputs: inputs_ref,
                });

                let input_str = input_nodes.iter()
                                           .map(|idx| format!("n{}", idx))
                                           .collect::<Vec<_>>()
                                           .join(",");
                node_names.push(format!("Call({} <- [{}])", gpufunc_name(f), input_str));

                return node_idx;
            }
        }
    }

    rec(root, &X, &col_keys, &mut node_ws, &mut node_bs, &mut node_names, &mut compute_table);

    (X, node_ws, node_bs, node_names, compute_table)
}

/// expr 操作名（debug）
fn expr_op_name(expr: &Expr) -> &'static str {
    match expr {
        Expr::Add(_,_) => "Add",
        Expr::Sub(_,_) => "Sub",
        Expr::Mul(_,_) => "Mul",
        Expr::Div(_,_) => "Div",
        Expr::Mod(_,_) => "Mod",
        Expr::If{..} => "If",
        Expr::Eq(_,_) => "Eq",
        Expr::Ne(_,_) => "Ne",
        Expr::Lt(_,_) => "Lt",
        Expr::Le(_,_) => "Le",
        Expr::Gt(_,_) => "Gt",
        Expr::Ge(_,_) => "Ge",
        Expr::Call(_) => "Call",
        Expr::Var(_,_) => "Var",
        Expr::GpuVar(_) => "GpuVar",
        Expr::Scalar(_) => "Const",
        Expr::Vec2(_,_) => "Vec2",
        Expr::Vec3(_,_,_) => "Vec3",
        Expr::Vec4(_,_,_,_) => "Vec4",
    }
}

fn gpuvar_name(gv: &GpuVar) -> String {
    match gv {
        GpuVar::Uv => "GPU_UV".into(),
        GpuVar::Color => "GPU_COLOR".into(),
        GpuVar::Normal => "GPU_NORMAL".into(),
        GpuVar::Position => "GPU_POSITION".into(),
        GpuVar::Time => "GPU_TIME".into(),
        GpuVar::GlobalIdX => "GlobalIdX".into(),
        GpuVar::GlobalIdY => "GlobalIdY".into(),
        GpuVar::BufferInput(i) => format!("GPU_BUF_IN_{}", i),
        GpuVar::BufferOutput(i) => format!("GPU_BUF_OUT_{}", i),
    }
}

fn gpufunc_name(f: &GpuFunc) -> &'static str {
    match f {
        GpuFunc::Sin(_) => "sin",
        GpuFunc::Cos(_) => "cos",
        GpuFunc::Abs(_) => "abs",
        GpuFunc::Max(_,_) => "max",
        GpuFunc::Min(_,_) => "min",
        GpuFunc::Clamp(_,_,_) => "clamp",
        GpuFunc::Dot(_,_) => "dot",
        GpuFunc::Normalize(_) => "normalize",
        GpuFunc::Lerp(_,_,_) => "lerp",
        GpuFunc::X(expr) => "X",
        GpuFunc::Y(expr) => "Y",
        GpuFunc::Z(expr) => "Z",
        GpuFunc::W(expr) => "W",
    }
}
/// 递归 CPU 求值（用于常量 fold / 模拟计算）
fn compute_expr(expr: &Expr, X: &Array2<f32>, col_keys: &Vec<String>) -> Vec<f32> {
    let n = X.nrows();
    let mut res = vec![0.0f32; n];

    match expr {
        Expr::Var(name, _sz) => {
            if let Some(idx) = col_keys.iter().position(|s| s==name) {
                for i in 0..n { res[i] = X[[i, idx]]; }
            }
        }
        Expr::GpuVar(gv) => {
            let name = gpuvar_name(gv);
            if let Some(idx) = col_keys.iter().position(|s| s==&name) {
                for i in 0..n { res[i] = X[[i, idx]]; }
            }
        }
        Expr::Scalar(s) => { for i in 0..n { res[i] = *s; } }

        Expr::Add(a,b) => {
            let va = compute_expr(a, X, col_keys);
            let vb = compute_expr(b, X, col_keys);
            for i in 0..n { res[i] = va[i] + vb[i]; }
        }
        Expr::Sub(a,b) => {
            let va = compute_expr(a, X, col_keys);
            let vb = compute_expr(b, X, col_keys);
            for i in 0..n { res[i] = va[i] - vb[i]; }
        }
        Expr::Mul(a,b) => {
            let va = compute_expr(a, X, col_keys);
            let vb = compute_expr(b, X, col_keys);
            for i in 0..n { res[i] = va[i] * vb[i]; }
        }
        Expr::Div(a,b) => {
            let va = compute_expr(a, X, col_keys);
            let vb = compute_expr(b, X, col_keys);
            for i in 0..n { res[i] = va[i] / vb[i]; }
        }
        Expr::Mod(a,b) => {
            let va = compute_expr(a, X, col_keys);
            let vb = compute_expr(b, X, col_keys);
            for i in 0..n { res[i] = va[i] % vb[i]; }
        }

        Expr::If{ cond, then_expr, else_expr } => {
            let vc = compute_expr(cond, X, col_keys);
            let vt = compute_expr(then_expr, X, col_keys);
            let ve = compute_expr(else_expr, X, col_keys);
            for i in 0..n { res[i] = if vc[i] != 0.0 { vt[i] } else { ve[i] }; }
        }

        Expr::Eq(a,b) => {
            let va = compute_expr(a, X, col_keys);
            let vb = compute_expr(b, X, col_keys);
            for i in 0..n { res[i] = if va[i] == vb[i] {1.0} else {0.0}; }
        }
        Expr::Ne(a,b) => {
            let va = compute_expr(a, X, col_keys);
            let vb = compute_expr(b, X, col_keys);
            for i in 0..n { res[i] = if va[i] != vb[i] {1.0} else {0.0}; }
        }
        Expr::Lt(a,b) => {
            let va = compute_expr(a, X, col_keys);
            let vb = compute_expr(b, X, col_keys);
            for i in 0..n { res[i] = if va[i] < vb[i] {1.0} else {0.0}; }
        }
        Expr::Le(a,b) => {
            let va = compute_expr(a, X, col_keys);
            let vb = compute_expr(b, X, col_keys);
            for i in 0..n { res[i] = if va[i] <= vb[i] {1.0} else {0.0}; }
        }
        Expr::Gt(a,b) => {
            let va = compute_expr(a, X, col_keys);
            let vb = compute_expr(b, X, col_keys);
            for i in 0..n { res[i] = if va[i] > vb[i] {1.0} else {0.0}; }
        }
        Expr::Ge(a,b) => {
            let va = compute_expr(a, X, col_keys);
            let vb = compute_expr(b, X, col_keys);
            for i in 0..n { res[i] = if va[i] >= vb[i] {1.0} else {0.0}; }
        }

        Expr::Call(f) => {
            match f {
                GpuFunc::Sin(x) => { let vx = compute_expr(x,X,col_keys); for i in 0..n { res[i] = vx[i].sin(); } }
                GpuFunc::Cos(x) => { let vx = compute_expr(x,X,col_keys); for i in 0..n { res[i] = vx[i].cos(); } }
                GpuFunc::Abs(x) => { let vx = compute_expr(x,X,col_keys); for i in 0..n { res[i] = vx[i].abs(); } }
                GpuFunc::Max(a,b) => { let va = compute_expr(a,X,col_keys); let vb = compute_expr(b,X,col_keys); for i in 0..n { res[i] = va[i].max(vb[i]); } }
                GpuFunc::Min(a,b) => { let va = compute_expr(a,X,col_keys); let vb = compute_expr(b,X,col_keys); for i in 0..n { res[i] = va[i].min(vb[i]); } }
                GpuFunc::Clamp(x,low,high) => { let vx = compute_expr(x,X,col_keys); let vl = compute_expr(low,X,col_keys); let vh = compute_expr(high,X,col_keys); for i in 0..n { res[i] = vx[i].min(vh[i]).max(vl[i]); } }
                GpuFunc::Dot(a,b) => { let va = compute_expr(a,X,col_keys); let vb = compute_expr(b,X,col_keys); for i in 0..n { res[i] = va[i]*vb[i]; } }
                GpuFunc::Normalize(x) => { let vx = compute_expr(x,X,col_keys); for i in 0..n { res[i] = vx[i]; } }
                                GpuFunc::Lerp(a,b,t) => { let va = compute_expr(a,X,col_keys); let vb = compute_expr(b,X,col_keys); let vt = compute_expr(t,X,col_keys); for i in 0..n { res[i] = va[i]*(1.0-vt[i]) + vb[i]*vt[i]; } }
                GpuFunc::X(expr) => {vec![compute_expr(expr,X,col_keys)];},
                GpuFunc::Y(expr) => {vec![compute_expr(expr,X,col_keys)];},
                GpuFunc::Z(expr) => {vec![compute_expr(expr,X,col_keys)];},
                GpuFunc::W(expr) => {vec![compute_expr(expr,X,col_keys)];},
            }
        }

        // --- 新增 Vec2 / Vec3 / Vec4 支持 ---
        Expr::Vec2(a,b) => {
            let va = compute_expr(a,X,col_keys);
            let vb = compute_expr(b,X,col_keys);
            let mut res2 = vec![0.0f32; n*2];
            for i in 0..n {
                res2[i*2+0] = va[i];
                res2[i*2+1] = vb[i];
            }
            return res2;
        }
        Expr::Vec3(a,b,c) => {
            let va = compute_expr(a,X,col_keys);
            let vb = compute_expr(b,X,col_keys);
            let vc = compute_expr(c,X,col_keys);
            let mut res3 = vec![0.0f32; n*3];
            for i in 0..n {
                res3[i*3+0] = va[i];
                res3[i*3+1] = vb[i];
                res3[i*3+2] = vc[i];
            }
            return res3;
        }
        Expr::Vec4(a,b,c,d) => {
            let va = compute_expr(a,X,col_keys);
            let vb = compute_expr(b,X,col_keys);
            let vc = compute_expr(c,X,col_keys);
            let vd = compute_expr(d,X,col_keys);
            let mut res4 = vec![0.0f32; n*4];
            for i in 0..n {
                res4[i*4+0] = va[i];
                res4[i*4+1] = vb[i];
                res4[i*4+2] = vc[i];
                res4[i*4+3] = vd[i];
            }
            return res4;
        }
    }

    res
}

/// 将 node_ws/node_bs 打平成 GPU buffer（节点主序）
/// 输出：(flat_W, flat_bias, node_count, n, m)
pub fn flatten_nodes_for_gpu(node_ws: &Vec<Array2<f32>>, node_bs: &Vec<Array2<f32>>) -> (Vec<f32>, Vec<f32>, usize, usize, usize) {
    let node_count = node_ws.len();
    if node_count == 0 { return (vec![], vec![], 0, 0, 0); }
    let n = node_ws[0].nrows();
    let m = node_ws[0].ncols();
    let mut flat_w = Vec::with_capacity(node_count * n * m);
    let mut flat_b = Vec::with_capacity(node_count * n);
    for node in node_ws {
        for i in 0..n {
            for j in 0..m {
                flat_w.push(node[[i,j]]);
            }
        }
    }
    for b in node_bs {
        for i in 0..n {
            flat_b.push(b[[i,0]]);
        }
    }
    (flat_w, flat_b, node_count, n, m)
}

// ---------- 构建一维 offset tag 表：op_index_for_flat_b ----------
pub fn build_op_index_map(table: &ComputeTable, n: usize, node_count: usize) -> Vec<i32> {
    let mut map = vec![-1i32; node_count * n];
    for (entry_idx, entry) in table.entries.iter().enumerate() {
        let base = entry.node_idx * n;
        for r in 0..n {
            map[base + r] = entry_idx as i32;
        }
    }
    map
}

// ---------- 用一维 tag 表按位置执行 compute（CPU 模拟 GPU） ----------
pub fn simulate_compute_table_by_tag(flat_b: &mut [f32], n: usize, tag: &[i32], table: &ComputeTable, X: &Array2<f32>) {
    let len = flat_b.len();
    for pos in 0..len {
        let op_idx = tag[pos];
        if op_idx < 0 { continue; } // -1 表示无需 compute
        let ei = op_idx as usize;
        let entry = &table.entries[ei];
        let row = pos % n;
        // gather inputs
        let mut vals: Vec<f32> = Vec::with_capacity(entry.inputs.len());
        for inp in &entry.inputs {
            match inp {
                InputRef::Node(idx) => {
                    let p = idx * n + row;
                    vals.push(flat_b[p]);
                }
                InputRef::Col(ci) => {
                    vals.push(X[[row, *ci]]);
                }
                InputRef::Const(c) => {
                    vals.push(*c);
                }
            }
        }
        // compute output
        let out = match entry.kind {
            ComputeOpKind::Sin => vals[0].sin(),
            ComputeOpKind::Cos => vals[0].cos(),
            ComputeOpKind::Abs => vals[0].abs(),
            ComputeOpKind::Max => vals[0].max(vals[1]),
            ComputeOpKind::Min => vals[0].min(vals[1]),
            ComputeOpKind::Clamp => {
                let x = vals[0]; let low = vals[1]; let high = vals[2];
                x.min(high).max(low)
            }
            ComputeOpKind::Dot => { vals[0] * vals[1] }
                        ComputeOpKind::Normalize => { vals[0] }
            ComputeOpKind::Lerp => { vals[0] * (1.0 - vals[2]) + vals[1] * vals[2] }
            ComputeOpKind::X => vals[0],
            ComputeOpKind::Y => vals[0],
            ComputeOpKind::Z => vals[0],
            ComputeOpKind::W => vals[0],
        };
        // write back
        flat_b[pos] = out;
    }
}
fn eval_expr(expr: &Expr, row: usize, X: &[Vec<f32>], col_keys: &[String]) -> Vec<f32> {
    match expr {
        Expr::Var(_, col) => vec![X[row][*col]],

        Expr::GpuVar(gv) => {
            let name = gpuvar_name(gv); // 返回 "Position", "Time" 等
            if let Some(idx) = col_keys.iter().position(|s| s == &name) {
                vec![X[row][idx]]
            } else {
                vec![0.0] // 找不到对应列就返回 0
            }
        }

        Expr::Scalar(v) => vec![*v],

        Expr::Vec2(a, b) => {
            let va = eval_expr(a, row, X, col_keys);
            let vb = eval_expr(b, row, X, col_keys);
            vec![va[0], vb[0]]
        },
        Expr::Vec3(a, b, c) => {
            let va = eval_expr(a, row, X, col_keys);
            let vb = eval_expr(b, row, X, col_keys);
            let vc = eval_expr(c, row, X, col_keys);
            vec![va[0], vb[0], vc[0]]
        },
        Expr::Vec4(a, b, c, d) => {
            let va = eval_expr(a, row, X, col_keys);
            let vb = eval_expr(b, row, X, col_keys);
            let vc = eval_expr(c, row, X, col_keys);
            let vd = eval_expr(d, row, X, col_keys);
            vec![va[0], vb[0], vc[0], vd[0]]
        },

        Expr::Add(a, b) => {
            let va = eval_expr(a, row, X, col_keys);
            let vb = eval_expr(b, row, X, col_keys);
            va.iter().zip(vb.iter()).map(|(x,y)| x + y).collect()
        },
        Expr::Sub(a, b) => {
            let va = eval_expr(a, row, X, col_keys);
            let vb = eval_expr(b, row, X, col_keys);
            va.iter().zip(vb.iter()).map(|(x,y)| x - y).collect()
        },
        Expr::Mul(a, b) => {
            let va = eval_expr(a, row, X, col_keys);
            let vb = eval_expr(b, row, X, col_keys);
            va.iter().zip(vb.iter()).map(|(x,y)| x * y).collect()
        },
        Expr::Div(a, b) => {
            let va = eval_expr(a, row, X, col_keys);
            let vb = eval_expr(b, row, X, col_keys);
            va.iter().zip(vb.iter()).map(|(x,y)| x / y).collect()
        },

        Expr::Call(func) => match func {
            GpuFunc::Sin(e) => eval_expr(e, row, X, col_keys).iter().map(|v| v.sin()).collect(),
            GpuFunc::Cos(e) => eval_expr(e, row, X, col_keys).iter().map(|v| v.cos()).collect(),
            GpuFunc::Abs(e) => eval_expr(e, row, X, col_keys).iter().map(|v| v.abs()).collect(),
            GpuFunc::Lerp(a, b, t) => {
                let va = eval_expr(a,row,X,col_keys);
                let vb = eval_expr(b,row,X,col_keys);
                let vt = eval_expr(t,row,X,col_keys);
                va.iter().zip(vb.iter()).zip(vt.iter())
                    .map(|((x,y),alpha)| x*(1.0-alpha)+y*alpha)
                    .collect()
            },
            GpuFunc::Max(a, b) => {
                let va = eval_expr(a,row,X,col_keys);
                let vb = eval_expr(b,row,X,col_keys);
                va.iter().zip(vb.iter()).map(|(x,y)| x.max(*y)).collect()
            },
            GpuFunc::Min(a, b) => {
                let va = eval_expr(a,row,X,col_keys);
                let vb = eval_expr(b,row,X,col_keys);
                va.iter().zip(vb.iter()).map(|(x,y)| x.min(*y)).collect()
            },
            GpuFunc::Clamp(v, min, max) => {
                let vv = eval_expr(v,row,X,col_keys);
                let vmin = eval_expr(min,row,X,col_keys);
                let vmax = eval_expr(max,row,X,col_keys);
                vv.iter().zip(vmin.iter()).zip(vmax.iter())
                    .map(|((x,lo),hi)| x.max(*lo).min(*hi))
                    .collect()
            },
            GpuFunc::Dot(a, b) => {
                let va = eval_expr(a,row,X,col_keys);
                let vb = eval_expr(b,row,X,col_keys);
                vec![va.iter().zip(vb.iter()).map(|(x,y)| x*y).sum()]
            },
            GpuFunc::Normalize(e) => {
                let ve = eval_expr(e,row,X,col_keys);
                let norm = ve.iter().map(|x| x*x).sum::<f32>().sqrt();
                if norm != 0.0 {
                    ve.iter().map(|x| x / norm).collect()
                } else {
                    ve.iter().map(|_| 0.0).collect()
                }
            },
            GpuFunc::X(expr) => eval_expr(expr,row,X,col_keys),
            GpuFunc::Y(expr) => eval_expr(expr,row,X,col_keys),
            GpuFunc::Z(expr) => eval_expr(expr,row,X,col_keys),
            GpuFunc::W(expr) => eval_expr(expr,row,X,col_keys),
        },

        Expr::If { cond, then_expr, else_expr } => {
            if eval_expr(cond, row, X, col_keys) != vec![0.0] {
                eval_expr(then_expr, row, X, col_keys)
            } else {
                eval_expr(else_expr, row, X, col_keys)
            }
        },

        Expr::Eq(a, b) => {
            if eval_expr(a, row, X, col_keys) == eval_expr(b, row, X, col_keys) { vec![1.0] } else { vec![0.0] }
        }
        Expr::Ne(a, b) => {
            if eval_expr(a, row, X, col_keys) != eval_expr(b, row, X, col_keys) { vec![1.0] } else { vec![0.0] }
        }
        Expr::Lt(a, b) => {
            if eval_expr(a, row, X, col_keys) < eval_expr(b, row, X, col_keys) { vec![1.0] } else { vec![0.0]  }
        }
        Expr::Le(a, b) => {
            if eval_expr(a, row, X, col_keys) <= eval_expr(b, row, X, col_keys) { vec![1.0] } else { vec![0.0]  }
        }
        Expr::Gt(a, b) => {
            if eval_expr(a, row, X, col_keys) > eval_expr(b, row, X, col_keys) {vec![1.0] } else { vec![0.0]  }
        }
        Expr::Ge(a, b) => {
            if eval_expr(a, row, X, col_keys) >= eval_expr(b, row, X, col_keys) { vec![1.0]} else { vec![0.0]  }
        }

        _ => vec![0.0], // 其他情况可继续补
    }
}
fn simulate_compute_table_simple(
    flat_b: &mut [f32],
    n: usize,
    compute_table: &[Expr],
    X: &Vec<Vec<f32>>,
    col_keys: &[String],  // ✅ 新增
) {
    let node_count = compute_table.len();
    let max_vec_len = 4; // Vec4 最大分量数

    for row in 0..n {
        for i in 0..node_count {
            // ✅ 传入 col_keys
            let val = eval_expr(&compute_table[i], row, X, col_keys); // Vec<f32>
            let base_idx = row * node_count * max_vec_len + i * max_vec_len;
            
            // 填充节点值
            for j in 0..val.len() {
                flat_b[base_idx + j] = val[j];
            }
            // 如果节点分量不足 Vec4，剩余填 0
            for j in val.len()..max_vec_len {
                flat_b[base_idx + j] = 0.0;
            }
        }
    }
}

// CPU 完整计算入口
pub fn cpu_compute(expr: &Expr, X: &Vec<Vec<f32>>,col_keys: &[String]) -> Vec<f32> {
    let n = X.len();
    let mut results = Vec::with_capacity(n);
    for row in 0..n {
        results.push(eval_expr(expr, row, X,col_keys));
    }
    results.iter().flatten().cloned().collect::<Vec<_>>()
}

// ================= AST (结构化) =================
#[derive(Debug, Clone)]
pub enum AstExpr {
    Var(String, usize),        // name, size (size 用于向量/标量注解)
    Scalar(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),

    // 运算节点（左右用 Box 封装 -- 记录子树）
    Add(Box<AstExpr>, Box<AstExpr>),
    Sub(Box<AstExpr>, Box<AstExpr>),
    Mul(Box<AstExpr>, Box<AstExpr>),
    Div(Box<AstExpr>, Box<AstExpr>),

    // 你可以 later 扩展 Call(GpuFunc, Vec<AstExpr>) 等
}


// ================= ToExpr trait =================
pub trait ToExpr {
    fn to_expr(self) -> Expr;
}
// ================= 原子类型 =================
#[derive(Debug, Clone)]
pub struct Wf32(pub f32);

#[derive(Debug, Clone)]
pub struct Wvec2f32(pub [f32; 2]);

#[derive(Debug, Clone)]
pub struct Wvec3f32(pub [f32; 3]);

#[derive(Debug, Clone)]
pub struct Wvec4f32(pub [f32; 4]);

#[derive(Debug, Clone)]
pub struct WRef(GpuVar);

#[derive(Debug, Clone)]
pub struct CRef(String);



// ================= 修复测试用例 =================


// ================= 测试 =================
#[test]
fn test_expr_ops() {


    // 直接可以传给 build_all_nodes_as_linear(&c, ...)
}
