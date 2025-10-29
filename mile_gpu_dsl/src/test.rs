use std::ops::{Add, Div, Mul, Sub};

pub mod dsl {
    use super::{Expr, UnaryFunc, BinaryOp};

    pub fn var(name: &str) -> Expr {
        Expr::Variable(name.to_string())
    }

    pub fn lit(v: f32) -> Expr {
        Expr::Constant(v)
    }

    pub fn vec2(a: Expr, b: Expr) -> Expr {
        Expr::Vector(vec![a, b])
    }

    pub fn vec3(a: Expr, b: Expr, c: Expr) -> Expr {
        Expr::Vector(vec![a, b, c])
    }

    pub fn vec4(a: Expr, b: Expr, c: Expr, d: Expr) -> Expr {
        Expr::Vector(vec![a, b, c, d])
    }

    pub fn sin(e: Expr) -> Expr {
        Expr::UnaryOp(UnaryFunc::Sin, Box::new(e))
    }
    pub fn cos(e: Expr) -> Expr {
        Expr::UnaryOp(UnaryFunc::Cos, Box::new(e))
    }
    pub fn sqrt(e: Expr) -> Expr {
        Expr::UnaryOp(UnaryFunc::Sqrt, Box::new(e))
    }

    // 便利构建 Equal / Modulo 等（按需扩展）
    pub fn eq(a: Expr, b: Expr) -> Expr {
        Expr::BinaryOp(BinaryOp::Equal, Box::new(a), Box::new(b))
    }

    pub fn modulo(a: Expr, b: Expr) -> Expr {
        Expr::BinaryOp(BinaryOp::Modulo, Box::new(a), Box::new(b))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Pow,
    GreaterThan,
    GreaterEqual,
    LessThan,
    LessEqual,
    Equal,
    NotEqual,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryFunc {
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
    Abs,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Variable(String),
    Constant(f32),
    BinaryOp(BinaryOp, Box<Expr>, Box<Expr>),
    UnaryOp(UnaryFunc, Box<Expr>),
    If {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },
    Vector(Vec<Expr>),
}

// 算术重载，便于 DSL 拼表达式
impl Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::BinaryOp(BinaryOp::Add, Box::new(self), Box::new(rhs))
    }
}
impl Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::BinaryOp(BinaryOp::Subtract, Box::new(self), Box::new(rhs))
    }
}
impl Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::BinaryOp(BinaryOp::Multiply, Box::new(self), Box::new(rhs))
    }
}
impl Div for Expr {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::BinaryOp(BinaryOp::Divide, Box::new(self), Box::new(rhs))
    }
}

#[derive(Debug, Clone)]
pub enum Kernel {
    InjectConstant { value: f32, out: usize },
    ElementwiseBinary { op: BinaryOp, a: usize, b: usize, out: usize },
    ElementwiseUnary { func: UnaryFunc, a: usize, out: usize },
    CondBlend { cond: usize, then_buf: usize, else_buf: usize, out: usize },
}

#[derive(Debug)]
pub struct GPUGraph {
    pub kernels: Vec<Kernel>,
    pub buffer_count: usize,
    pub input_count: usize,
    pub vector_len: usize,
}

impl GPUGraph {
    pub fn execute(&self, inputs: &[Vec<f32>]) -> Vec<f32> {
        if inputs.is_empty() {
            return vec![];
        }

        let mut buffers = inputs.to_vec();
        let max_out = self
            .kernels
            .iter()
            .map(|k| k.output_index())
            .max()
            .unwrap_or(0);
        buffers.resize(max_out + 1, vec![0.0; inputs[0].len()]);

        for kernel in &self.kernels {
            match kernel {
                Kernel::InjectConstant { value, out } => {
                    buffers[*out].iter_mut().for_each(|x| *x = *value);
                }
                Kernel::ElementwiseBinary { op, a, b, out } => {
                    for i in 0..buffers[0].len() {
                        buffers[*out][i] = match op {
                            BinaryOp::Add => buffers[*a][i] + buffers[*b][i],
                            BinaryOp::Subtract => buffers[*a][i] - buffers[*b][i],
                            BinaryOp::Multiply => buffers[*a][i] * buffers[*b][i],
                            BinaryOp::Divide => buffers[*a][i] / buffers[*b][i],
                            BinaryOp::Modulo => buffers[*a][i] % buffers[*b][i],
                            BinaryOp::Pow => buffers[*a][i].powf(buffers[*b][i]),
                            BinaryOp::GreaterThan => {
                                if buffers[*a][i] > buffers[*b][i] { 1.0 } else { 0.0 }
                            }
                            BinaryOp::GreaterEqual => {
                                if buffers[*a][i] >= buffers[*b][i] { 1.0 } else { 0.0 }
                            }
                            BinaryOp::LessThan => {
                                if buffers[*a][i] < buffers[*b][i] { 1.0 } else { 0.0 }
                            }
                            BinaryOp::LessEqual => {
                                if buffers[*a][i] <= buffers[*b][i] { 1.0 } else { 0.0 }
                            }
                            BinaryOp::Equal => {
                                if (buffers[*a][i] - buffers[*b][i]).abs() < std::f32::EPSILON { 1.0 } else { 0.0 }
                            }
                            BinaryOp::NotEqual => {
                                if (buffers[*a][i] - buffers[*b][i]).abs() >= std::f32::EPSILON { 1.0 } else { 0.0 }
                            }
                        };
                    }
                }
                Kernel::ElementwiseUnary { func, a, out } => {
                    for i in 0..buffers[0].len() {
                        buffers[*out][i] = match func {
                            UnaryFunc::Sin => buffers[*a][i].sin(),
                            UnaryFunc::Cos => buffers[*a][i].cos(),
                            UnaryFunc::Tan => buffers[*a][i].tan(),
                            UnaryFunc::Exp => buffers[*a][i].exp(),
                            UnaryFunc::Log => buffers[*a][i].ln(),
                            UnaryFunc::Sqrt => buffers[*a][i].sqrt(),
                            UnaryFunc::Abs => buffers[*a][i].abs(),
                        };
                    }
                }
                Kernel::CondBlend {
                    cond,
                    then_buf,
                    else_buf,
                    out,
                } => {
                    for i in 0..buffers[0].len() {
                        buffers[*out][i] = buffers[*cond][i] * buffers[*then_buf][i]
                            + (1.0 - buffers[*cond][i]) * buffers[*else_buf][i];
                    }
                }
            }
        }

        let last_out = self.kernels.last().unwrap().output_index();
        buffers[last_out].clone()
    }
}

trait HasOutputIndex {
    fn output_index(&self) -> usize;
}

impl HasOutputIndex for Kernel {
    fn output_index(&self) -> usize {
        match self {
            Kernel::InjectConstant { out, .. } => *out,
            Kernel::ElementwiseBinary { out, .. } => *out,
            Kernel::CondBlend { out, .. } => *out,
            Kernel::ElementwiseUnary { out, .. } => *out,
        }
    }
}

// ---------------- GPUCompiler ----------------
pub struct GPUCompiler {
    kernels: Vec<Kernel>,
    next_buffer: usize,
    input_count: usize,
    vector_len: usize,
    zero_buffer: Option<usize>, // 用于需要拷贝时复用一个 constant 0.0 buffer
}

impl GPUCompiler {
    pub fn new(input_count: usize, vector_len: usize) -> Self {
        Self {
            kernels: Vec::new(),
            next_buffer: input_count,
            input_count,
            vector_len,
            zero_buffer: None,
        }
    }

    // 如果需要一个 0.0 常量 buffer（用于 copy via add 0），就创建并缓存起来
    fn ensure_zero_buffer(&mut self) -> usize {
        if let Some(idx) = self.zero_buffer {
            idx
        } else {
            let idx = self.alloc_buffer();
            self.kernels.push(Kernel::InjectConstant { value: 0.0, out: idx });
            self.zero_buffer = Some(idx);
            idx
        }
    }

    // 把 src buffer 的值 "拷贝" 到 dest buffer（通过 a + 0 实现）
    fn emit_copy(&mut self, src: usize, dest: usize) {
        if src == dest {
            return;
        }
        let zero = self.ensure_zero_buffer();
        self.kernels.push(Kernel::ElementwiseBinary {
            op: BinaryOp::Add,
            a: src,
            b: zero,
            out: dest,
        });
    }

    /// 返回值：如果 expr 是标量，返回该标量的 buffer index；
    /// 如果 expr 是向量（n 分量），返回第一个分量 buffer index（并保证 n 个分量占用连续 buffer）。
    pub fn compile_expr(&mut self, expr: &Expr, variables: &[&str]) -> usize {
        match expr {
            Expr::Variable(name) => {
                variables.iter()
                    .position(|&v| v == name)
                    .unwrap_or_else(|| panic!("未知变量: {}", name))
            }
            Expr::Constant(val) => {
                let out = self.alloc_buffer();
                self.kernels.push(Kernel::InjectConstant { value: *val, out });
                out
            }
            Expr::Vector(comps) => {
                // 为整个向量分配连续的 outs
                let first_out = self.alloc_buffer();
                let mut outs = vec![first_out];
                for _ in 1..comps.len() {
                    outs.push(self.alloc_buffer());
                }
                // 编译每个分量并确保拷贝到 outs[i]
                for i in 0..comps.len() {
                    let src = self.compile_expr(&comps[i], variables);
                    self.emit_copy(src, outs[i]);
                }
                first_out
            }
            Expr::UnaryOp(func, sub) => {
                // 如果子表达式是向量（Vector 节点），我们可以直接遍历它的分量生成 unary kernel
                if let Expr::Vector(sub_comps) = &**sub {
                    // 先为输出分配连续 buffers
                    let first_out = self.alloc_buffer();
                    let mut outs = vec![first_out];
                    for _ in 1..sub_comps.len() {
                        outs.push(self.alloc_buffer());
                    }
                    // 对每个分量：编译子表达式分量，拷贝到临时（如果必要），再 emit unary
                    for i in 0..sub_comps.len() {
                        let a_idx = self.compile_expr(&sub_comps[i], variables);
                        // 确保 a_idx 的值在一个稳定的 buffer 上（不要求连续）
                        // 直接把 a_idx 的值 unary 到 outs[i]
                        self.kernels.push(Kernel::ElementwiseUnary {
                            func: func.clone(),
                            a: a_idx,
                            out: outs[i],
                        });
                    }
                    first_out
                } else {
                    let a = self.compile_expr(sub, variables);
                    let out = self.alloc_buffer();
                    self.kernels.push(Kernel::ElementwiseUnary {
                        func: func.clone(),
                        a,
                        out,
                    });
                    out
                }
            }
            Expr::BinaryOp(op, left, right) => {
                match (&**left, &**right) {
                    (Expr::Vector(lcomps), Expr::Vector(rcomps)) => {
                        assert!(lcomps.len() == rcomps.len(), "vector sizes must match for binary op");
                        let first_out = self.alloc_buffer();
                        let mut outs = vec![first_out];
                        for _ in 1..lcomps.len() {
                            outs.push(self.alloc_buffer());
                        }
                        for i in 0..lcomps.len() {
                            let a_idx = self.compile_expr(&lcomps[i], variables);
                            let b_idx = self.compile_expr(&rcomps[i], variables);
                            self.kernels.push(Kernel::ElementwiseBinary {
                                op: op.clone(),
                                a: a_idx,
                                b: b_idx,
                                out: outs[i],
                            });
                        }
                        first_out
                    }
                    (Expr::Vector(lcomps), _) => {
                        // left vector, right scalar -> broadcast right
                        let first_out = self.alloc_buffer();
                        let mut outs = vec![first_out];
                        for _ in 1..lcomps.len() {
                            outs.push(self.alloc_buffer());
                        }
                        let b_idx = self.compile_expr(right, variables); // 编译一次 scalar
                        for i in 0..lcomps.len() {
                            let a_idx = self.compile_expr(&lcomps[i], variables);
                            self.kernels.push(Kernel::ElementwiseBinary {
                                op: op.clone(),
                                a: a_idx,
                                b: b_idx,
                                out: outs[i],
                            });
                        }
                        first_out
                    }
                    (_, Expr::Vector(rcomps)) => {
                        // right vector, left scalar -> broadcast left
                        let first_out = self.alloc_buffer();
                        let mut outs = vec![first_out];
                        for _ in 1..rcomps.len() {
                            outs.push(self.alloc_buffer());
                        }
                        let a_idx = self.compile_expr(left, variables); // 编译一次 scalar
                        for i in 0..rcomps.len() {
                            let b_idx = self.compile_expr(&rcomps[i], variables);
                            self.kernels.push(Kernel::ElementwiseBinary {
                                op: op.clone(),
                                a: a_idx,
                                b: b_idx,
                                out: outs[i],
                            });
                        }
                        first_out
                    }
                    _ => {
                        // both scalar
                        let a = self.compile_expr(left, variables);
                        let b = self.compile_expr(right, variables);
                        let out = self.alloc_buffer();
                        self.kernels.push(Kernel::ElementwiseBinary {
                            op: op.clone(),
                            a,
                            b,
                            out,
                        });
                        out
                    }
                }
            }
            Expr::If { condition, then_branch, else_branch } => {
                match (&**condition, &**then_branch, &**else_branch) {
                    (Expr::Vector(cvec), Expr::Vector(tvec), Expr::Vector(evec)) => {
                        assert!(cvec.len() == tvec.len() && tvec.len() == evec.len());
                        let first_out = self.alloc_buffer();
                        let mut outs = vec![first_out];
                        for _ in 1..cvec.len() {
                            outs.push(self.alloc_buffer());
                        }
                        for i in 0..cvec.len() {
                            let c_idx = self.compile_expr(&cvec[i], variables);
                            let t_idx = self.compile_expr(&tvec[i], variables);
                            let e_idx = self.compile_expr(&evec[i], variables);
                            self.kernels.push(Kernel::CondBlend {
                                cond: c_idx,
                                then_buf: t_idx,
                                else_buf: e_idx,
                                out: outs[i],
                            });
                        }
                        first_out
                    }
                    (Expr::Vector(cvec), Expr::Vector(tvec), _) => {
                        assert!(cvec.len() == tvec.len());
                        let first_out = self.alloc_buffer();
                        let mut outs = vec![first_out];
                        for _ in 1..cvec.len() {
                            outs.push(self.alloc_buffer());
                        }
                        let e_scalar = self.compile_expr(else_branch, variables);
                        for i in 0..cvec.len() {
                            let c_idx = self.compile_expr(&cvec[i], variables);
                            let t_idx = self.compile_expr(&tvec[i], variables);
                            self.kernels.push(Kernel::CondBlend {
                                cond: c_idx,
                                then_buf: t_idx,
                                else_buf: e_scalar,
                                out: outs[i],
                            });
                        }
                        first_out
                    }
                    (Expr::Vector(cvec), _, Expr::Vector(evec)) => {
                        assert!(cvec.len() == evec.len());
                        let first_out = self.alloc_buffer();
                        let mut outs = vec![first_out];
                        for _ in 1..cvec.len() {
                            outs.push(self.alloc_buffer());
                        }
                        let t_scalar = self.compile_expr(then_branch, variables);
                        for i in 0..cvec.len() {
                            let c_idx = self.compile_expr(&cvec[i], variables);
                            let e_idx = self.compile_expr(&evec[i], variables);
                            self.kernels.push(Kernel::CondBlend {
                                cond: c_idx,
                                then_buf: t_scalar,
                                else_buf: e_idx,
                                out: outs[i],
                            });
                        }
                        first_out
                    }
                    _ => {
                        let cond_idx = self.compile_expr(condition, variables);
                        let then_idx = self.compile_expr(then_branch, variables);
                        let else_idx = self.compile_expr(else_branch, variables);
                        let out = self.alloc_buffer();
                        self.kernels.push(Kernel::CondBlend {
                            cond: cond_idx,
                            then_buf: then_idx,
                            else_buf: else_idx,
                            out,
                        });
                        out
                    }
                }
            }
        }
    }

    fn alloc_buffer(&mut self) -> usize {
        let idx = self.next_buffer;
        self.next_buffer += 1;
        idx
    }

    pub fn build(self) -> GPUGraph {
        GPUGraph {
            kernels: self.kernels,
            buffer_count: self.next_buffer,
            input_count: self.input_count,
            vector_len: self.vector_len,
        }
    }
}

pub fn compile_to_gpu_graph(expr: &Expr, variables: &[&str], vector_len: usize) -> GPUGraph {
    let mut compiler = GPUCompiler::new(variables.len(), vector_len);
    let _out = compiler.compile_expr(expr, variables);
    compiler.build()
}

// ----------------- 示例 / 测试 -----------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::dsl::*;

    #[test]
    fn test_vec_scalar_and_if() {
        // 构造： if ((a % 2) == 0) { sin(a * b) } else { sqrt(a + b) }
        let expr = Expr::If {
            condition: Box::new(Expr::BinaryOp(
                BinaryOp::Equal,
                Box::new(Expr::BinaryOp(
                    BinaryOp::Modulo,
                    Box::new(Expr::Variable("a".to_string())),
                    Box::new(Expr::Constant(2.0)),
                )),
                Box::new(Expr::Constant(0.0)),
            )),
            then_branch: Box::new(Expr::UnaryOp(
                UnaryFunc::Sin,
                Box::new(Expr::BinaryOp(
                    BinaryOp::Multiply,
                    Box::new(Expr::Variable("a".to_string())),
                    Box::new(Expr::Variable("b".to_string())),
                )),
            )),
            else_branch: Box::new(Expr::UnaryOp(
                UnaryFunc::Sqrt,
                Box::new(Expr::BinaryOp(
                    BinaryOp::Add,
                    Box::new(Expr::Variable("a".to_string())),
                    Box::new(Expr::Variable("b".to_string())),
                )),
            )),
        };

        let variables = vec!["a", "b", "c", "d"];

        // batch=1
        let batch1_inputs = vec![vec![12.0], vec![3.0], vec![2.0], vec![4.0]];
        let graph1 = compile_to_gpu_graph(&expr, &variables, 1);
        let out1 = graph1.execute(&batch1_inputs);
        println!("batch=1 输出: {:?}", out1); // 期望 sin(12*3) or sqrt depending on condition
        // batch=3 示例
        let batch3_inputs = vec![
            vec![5.0, 2.0, 7.0],
            vec![3.0, 4.0, 10.0],
            vec![2.0, 2.0, 2.0],
            vec![4.0, 4.0, 4.0],
        ];
        let graph3 = compile_to_gpu_graph(&expr, &variables, 3);
        let out3 = graph3.execute(&batch3_inputs);
        println!("batch=3 输出: {:?}", out3);
        // 这里只打印，断言留给你按需要加
    }

    #[test]
    fn test_vec3_ops() {
        // vec3 的加法与广播示例
        let v1 = vec3(var("a"), var("b"), var("c"));
        let v2 = vec3(lit(1.0), lit(2.0), lit(3.0));
        let sum = Expr::BinaryOp(BinaryOp::Add, Box::new(v1.clone()), Box::new(v2.clone()));

        let variables = vec!["a", "b", "c"];

        let inputs = vec![
            vec![1.0, 10.0], // a
            vec![2.0, 20.0], // b
            vec![3.0, 30.0], // c
        ];
        let graph = compile_to_gpu_graph(&sum, &variables, 2);
        let out = graph.execute(&inputs);
        println!("vec3 add 输出: {:?}", out);
        // 期望每个分量加上对应常量，并最终返回最后 kernel 的 out buffer（这里实现返回最后 kernel out）
    }
}
