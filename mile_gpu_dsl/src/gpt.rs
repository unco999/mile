use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add, Subtract, Multiply, Divide, Modulo, Pow,
    GreaterThan, GreaterEqual, LessThan, LessEqual,
    Equal, NotEqual,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryFunc {
    Sin, Cos, Tan,
    Exp, Log, Sqrt, Abs,
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
}
#[derive(Debug, Clone)]
pub enum Kernel {
    InjectConstant { value: f32, out: usize },
    ElementwiseBinary { op: BinaryOp, a: usize, b: usize, out: usize },
    ElementwiseUnary { func: UnaryFunc, a: usize, out: usize }, // 这里记录所有函数引用
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
            BinaryOp::GreaterThan => if buffers[*a][i] > buffers[*b][i] { 1.0 } else { 0.0 },
            BinaryOp::GreaterEqual => if buffers[*a][i] >= buffers[*b][i] { 1.0 } else { 0.0 },
            BinaryOp::LessThan => if buffers[*a][i] < buffers[*b][i] { 1.0 } else { 0.0 },
            BinaryOp::LessEqual => if buffers[*a][i] <= buffers[*b][i] { 1.0 } else { 0.0 },
            BinaryOp::Equal => if buffers[*a][i] == buffers[*b][i] { 1.0 } else { 0.0 },
            BinaryOp::NotEqual => if buffers[*a][i] != buffers[*b][i] { 1.0 } else { 0.0 },
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

// --- GPUCompiler: Expr -> GPUGraph ---
pub struct GPUCompiler {
    kernels: Vec<Kernel>,
    next_buffer: usize,
    input_count: usize,
    vector_len: usize,
}

impl GPUCompiler {
    pub fn new(input_count: usize, vector_len: usize) -> Self {
        Self {
            kernels: Vec::new(),
            next_buffer: input_count,
            input_count,
            vector_len,
        }
    }

    pub fn compile_expr(&mut self, expr: &Expr, variables: &[&str]) -> usize {
        match expr {
            Expr::Variable(name) => variables.iter().position(|&v| v == name).unwrap(),
            Expr::Constant(val) => {
                let out = self.alloc_buffer();
                self.kernels
                    .push(Kernel::InjectConstant { value: *val, out });
                out
            }
            Expr::BinaryOp(op, left, right) => {
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
Expr::UnaryOp(func, sub_expr) => {
    let a = self.compile_expr(sub_expr, variables);
    let out = self.alloc_buffer();
    self.kernels.push(Kernel::ElementwiseUnary {
        func: func.clone(),
        a,
        out,
    });
    out
}

Expr::BinaryOp(op, left, right) => {
    let a = self.compile_expr(left, variables);
    let b = self.compile_expr(right, variables);
    let out = self.alloc_buffer();

    // 所有二元操作统一走 ElementwiseBinary kernel
    self.kernels.push(Kernel::ElementwiseBinary {
        op: op.clone(),
        a,
        b,
        out,
    });

    out
}
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_buf = self.compile_expr(condition, variables);
                let then_buf = self.compile_expr(then_branch, variables);
                let else_buf = self.compile_expr(else_branch, variables);
                let out = self.alloc_buffer();
                self.kernels.push(Kernel::CondBlend {
                    cond: cond_buf,
                    then_buf,
                    else_buf,
                    out,
                });
                out
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
    let out = compiler.compile_expr(expr, variables);
    compiler.build()
}

#[test]
// === 测试 ===
fn main() {
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
    println!("batch=1 输出: {:?}", out1); // [10]

    // batch=3
    let batch3_inputs = vec![
        vec![5.0, 2.0, 7.0],
        vec![3.0, 4.0, 10.0],
        vec![2.0, 2.0, 2.0],
        vec![4.0, 4.0, 4.0],
    ];
    let graph3 = compile_to_gpu_graph(&expr, &variables, 3);
    let out3 = graph3.execute(&batch3_inputs);
    println!("batch=3 输出: {:?}", out3); // [10, 16, 40]
}
