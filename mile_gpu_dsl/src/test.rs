use std::ops::{Add, Div, Mul, Sub};

pub mod dsl {
    use super::{Expr, UnaryFunc, BinaryOp, Vec2, Vec3, Vec4};

    pub fn var(name: &str) -> Expr {
        Expr::Variable(name.to_string())
    }

    pub fn lit(v: f32) -> Expr {
        Expr::Constant(v)
    }

    pub fn vec2<X: Into<Expr>, Y: Into<Expr>>(x: X, y: Y) -> Expr {
        Expr::Vec2(Vec2::new(Box::new(x.into()), Box::new(y.into())))
    }

    pub fn vec3<X: Into<Expr>, Y: Into<Expr>, Z: Into<Expr>>(x: X, y: Y, z: Z) -> Expr {
        Expr::Vec3(Vec3::new(Box::new(x.into()), Box::new(y.into()), Box::new(z.into())))
    }

    pub fn vec4<X: Into<Expr>, Y: Into<Expr>, Z: Into<Expr>, W: Into<Expr>>(x: X, y: Y, z: Z, w: W) -> Expr {
        Expr::Vec4(Vec4::new(Box::new(x.into()), Box::new(y.into()), Box::new(z.into()), Box::new(w.into())))
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

    pub fn eq(a: Expr, b: Expr) -> Expr {
        Expr::BinaryOp(BinaryOp::Equal, Box::new(a), Box::new(b))
    }

    pub fn modulo(a: Expr, b: Expr) -> Expr {
        Expr::BinaryOp(BinaryOp::Modulo, Box::new(a), Box::new(b))
    }
}

// 向量结构体定义 - 使用 Box 打破循环
#[derive(Debug, Clone)]
pub struct Vec2 {
    pub x: Box<Expr>,
    pub y: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct Vec3 {
    pub x: Box<Expr>,
    pub y: Box<Expr>,
    pub z: Box<Expr>,
}

#[derive(Debug, Clone)]
pub struct Vec4 {
    pub x: Box<Expr>,
    pub y: Box<Expr>,
    pub z: Box<Expr>,
    pub w: Box<Expr>,
}

impl Vec2 {
    pub fn new(x: Box<Expr>, y: Box<Expr>) -> Self {
        Self { x, y }
    }
    
    pub fn eval(&self) -> Option<(f32, f32)> {
        match (&*self.x, &*self.y) {
            (Expr::Constant(a), Expr::Constant(b)) => Some((*a, *b)),
            _ => None,
        }
    }
}

impl Vec3 {
    pub fn new(x: Box<Expr>, y: Box<Expr>, z: Box<Expr>) -> Self {
        Self { x, y, z }
    }
    
    pub fn eval(&self) -> Option<(f32, f32, f32)> {
        match (&*self.x, &*self.y, &*self.z) {
            (Expr::Constant(a), Expr::Constant(b), Expr::Constant(c)) => Some((*a, *b, *c)),
            _ => None,
        }
    }
}

impl Vec4 {
    pub fn new(x: Box<Expr>, y: Box<Expr>, z: Box<Expr>, w: Box<Expr>) -> Self {
        Self { x, y, z, w }
    }
    
    pub fn eval(&self) -> Option<(f32, f32, f32, f32)> {
        match (&*self.x, &*self.y, &*self.z, &*self.w) {
            (Expr::Constant(a), Expr::Constant(b), Expr::Constant(c), Expr::Constant(d)) => {
                Some((*a, *b, *c, *d))
            }
            _ => None,
        }
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
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
}

impl From<f32> for Expr {
    fn from(v: f32) -> Self {
        Expr::Constant(v)
    }
}

impl From<&str> for Expr {
    fn from(v: &str) -> Self {
        Expr::Variable(v.to_string())
    }
}

impl From<i32> for Expr {
    fn from(v: i32) -> Self {
        Expr::Constant(v as f32)
    }
}

// 严格实现 Add trait - 更新为使用 Box
impl Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Expr {
        match (self, rhs) {
            (Expr::Vec2(lhs), Expr::Vec2(rhs)) => {
                Expr::Vec2(Vec2::new(
                    Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.x, rhs.x)),
                    Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.y, rhs.y)),
                ))
            }
            (Expr::Vec3(lhs), Expr::Vec3(rhs)) => {
                Expr::Vec3(Vec3::new(
                    Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.x, rhs.x)),
                    Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.y, rhs.y)),
                    Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.z, rhs.z)),
                ))
            }
            (Expr::Vec4(lhs), Expr::Vec4(rhs)) => {
                Expr::Vec4(Vec4::new(
                    Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.x, rhs.x)),
                    Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.y, rhs.y)),
                    Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.z, rhs.z)),
                    Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.w, rhs.w)),
                ))
            }
            (Expr::Vec2(vec), scalar) => {
                Expr::Vec2(Vec2::new(
                    Box::new(Expr::BinaryOp(BinaryOp::Add, vec.x, Box::new(scalar.clone()))),
                    Box::new(Expr::BinaryOp(BinaryOp::Add, vec.y, Box::new(scalar))),
                ))
            }
            (scalar, Expr::Vec2(vec)) => {
                Expr::Vec2(Vec2::new(
                    Box::new(Expr::BinaryOp(BinaryOp::Add, Box::new(scalar.clone()), vec.x)),
                    Box::new(Expr::BinaryOp(BinaryOp::Add, Box::new(scalar), vec.y)),
                ))
            }
            (l, r) => Expr::BinaryOp(BinaryOp::Add, Box::new(l), Box::new(r)),
        }
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

// Kernel 定义保持不变
#[derive(Debug, Clone)]
pub enum Kernel {
    InjectConstant { value: f32, out: usize },
    ElementwiseBinary { op: BinaryOp, a: usize, b: usize, out: usize },
    ElementwiseUnary { func: UnaryFunc, a: usize, out: usize },
    CondBlend { cond: usize, then_buf: usize, else_buf: usize, out: usize },
}

// GPUGraph 支持多个输出
#[derive(Debug)]
pub struct GPUGraph {
    pub kernels: Vec<Kernel>,
    pub buffer_count: usize,
    pub input_count: usize,
    pub vector_len: usize,
    pub output_buffers: Vec<usize>, // 所有输出buffer的索引
}
impl GPUGraph {
    pub fn execute(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if inputs.is_empty() {
            return vec![];
        }

        let mut buffers = inputs.to_vec();
        
        // 修正：考虑输出缓冲区的最大索引，而不仅仅是kernels的输出
        let max_output_index = self.output_buffers.iter().max().copied().unwrap_or(0);
        let max_kernel_out = self.kernels.iter().map(|k| k.output_index()).max().unwrap_or(0);
        let max_needed = max_output_index.max(max_kernel_out);
        
        buffers.resize(max_needed + 1, vec![0.0; inputs[0].len()]);

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

        // 返回所有输出buffer的内容
        self.output_buffers.iter()
            .map(|&idx| buffers[idx].clone())
            .collect()
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

// GPUCompiler 返回多个输出buffer
pub struct GPUCompiler {
    kernels: Vec<Kernel>,
    next_buffer: usize,
    input_count: usize,
    vector_len: usize,
    output_buffers: Vec<usize>, // 存储所有输出buffer
}

impl GPUCompiler {
    pub fn new(input_count: usize, vector_len: usize) -> Self {
        Self {
            kernels: Vec::new(),
            next_buffer: input_count,
            input_count,
            vector_len,
            output_buffers: Vec::new(),
        }
    }
   // 新增辅助方法：编译单个表达式，确保返回单个buffer索引
    fn compile_single_expr(&mut self, expr: &Expr, variables: &[&str]) -> usize {
        let buffers = self.compile_expr(expr, variables);
        // 对于向量分量，我们只需要第一个buffer
        buffers[0]
    }

    /// 编译表达式，返回所有输出buffer的索引
    pub fn compile_expr(&mut self, expr: &Expr, variables: &[&str]) -> Vec<usize> {
               match expr {
            Expr::Variable(name) => {
                let idx = variables.iter().position(|&v| v == name).unwrap();
                vec![idx]
            }
            Expr::Constant(val) => {
                let out = self.alloc_buffer();
                self.kernels.push(Kernel::InjectConstant { value: *val, out });
                vec![out]
            }
            Expr::Vec2(vec) => {
                // 分别编译x和y分量，确保分配独立的缓冲区
                let x_idx = self.compile_single_expr(&vec.x, variables);
                let y_idx = self.compile_single_expr(&vec.y, variables);
                // 返回所有分量的buffer索引
                vec![x_idx, y_idx]
            }
            Expr::Vec3(vec) => {
                let x_idx = self.compile_single_expr(&vec.x, variables);
                let y_idx = self.compile_single_expr(&vec.y, variables);
                let z_idx = self.compile_single_expr(&vec.z, variables);
                vec![x_idx, y_idx, z_idx]
            }
            Expr::Vec4(vec) => {
                let x_idx = self.compile_single_expr(&vec.x, variables);
                let y_idx = self.compile_single_expr(&vec.y, variables);
                let z_idx = self.compile_single_expr(&vec.z, variables);
                let w_idx = self.compile_single_expr(&vec.w, variables);
                vec![x_idx, y_idx, z_idx, w_idx]
            }
            Expr::BinaryOp(op, left, right) => {
                let left_buffers = self.compile_expr(left, variables);
                let right_buffers = self.compile_expr(right, variables);
                
                // 处理分量运算：支持向量与标量的广播
                let mut outputs = Vec::new();
                let max_components = left_buffers.len().max(right_buffers.len());
                
                for i in 0..max_components {
                    let a_idx = if i < left_buffers.len() { left_buffers[i] } else { left_buffers[0] };
                    let b_idx = if i < right_buffers.len() { right_buffers[i] } else { right_buffers[0] };
                    
                    let out = self.alloc_buffer();
                    self.kernels.push(Kernel::ElementwiseBinary {
                        op: op.clone(),
                        a: a_idx,
                        b: b_idx,
                        out,
                    });
                    outputs.push(out);
                }
                
                outputs
            }
            Expr::UnaryOp(func, sub) => {
                let input_buffers = self.compile_expr(sub, variables);
                let mut outputs = Vec::new();
                
                for &input_idx in &input_buffers {
                    let out = self.alloc_buffer();
                    self.kernels.push(Kernel::ElementwiseUnary {
                        func: func.clone(),
                        a: input_idx,
                        out,
                    });
                    outputs.push(out);
                }
                
                outputs
            }
            Expr::If { condition, then_branch, else_branch } => {
                let cond_buffers = self.compile_expr(condition, variables);
                let then_buffers = self.compile_expr(then_branch, variables);
                let else_buffers = self.compile_expr(else_branch, variables);
                
                let mut outputs = Vec::new();
                let max_components = then_buffers.len().max(else_buffers.len());
                
                for i in 0..max_components {
                    let cond_idx = if i < cond_buffers.len() { cond_buffers[i] } else { cond_buffers[0] };
                    let then_idx = if i < then_buffers.len() { then_buffers[i] } else { then_buffers[0] };
                    let else_idx = if i < else_buffers.len() { else_buffers[i] } else { else_buffers[0] };
                    
                    let out = self.alloc_buffer();
                    self.kernels.push(Kernel::CondBlend {
                        cond: cond_idx,
                        then_buf: then_idx,
                        else_buf: else_idx,
                        out,
                    });
                    outputs.push(out);
                }
                
                outputs
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
            output_buffers: self.output_buffers,
        }
    }

    pub fn set_output_buffers(&mut self, buffers: Vec<usize>) {
        self.output_buffers = buffers;
    }
}

pub fn compile_to_gpu_graph(expr: &Expr, variables: &[&str], vector_len: usize) -> GPUGraph {
    let mut compiler = GPUCompiler::new(variables.len(), vector_len);
    let output_buffers = compiler.compile_expr(expr, variables);
    compiler.set_output_buffers(output_buffers);
    compiler.build()
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::dsl::*;

    #[test]
    fn test_debug_vector_creation() {
        println!("=== Debug Vector Creation ===");
        
        let vec_expr = vec2(var("x"), var("y"));
        println!("Vector expression: {:?}", vec_expr);
        
        let variables = vec!["x", "y"];
        // 修正：输入数据应该有两个缓冲区，对应变量 x 和 y
        let inputs = vec![
            vec![100.0, 200.0], // 缓冲区 0: 变量 x 的数据
            vec![300.0, 400.0], // 缓冲区 1: 变量 y 的数据
        ];
        
        println!("Input buffers: {}", inputs.len());
        println!("Input[0]: {:?}", inputs[0]);
        println!("Input[1]: {:?}", inputs[1]);
        
        let mut compiler = GPUCompiler::new(variables.len(), 2);
        let output_buffers = compiler.compile_expr(&vec_expr, &variables);
        println!("Output buffers: {:?}", output_buffers);
        println!("Kernels: {:?}", compiler.kernels);
        println!("Next buffer: {}", compiler.next_buffer);
        
        compiler.set_output_buffers(output_buffers);
        let graph = compiler.build();
        
        println!("Graph output buffers: {:?}", graph.output_buffers);
        println!("Graph buffer count: {}", graph.buffer_count);
        println!("Graph input count: {}", graph.input_count);
        
        let outputs = graph.execute(&inputs);
        println!("Execution results: {:?}", outputs);
    }

    #[test]
    fn test_correct_input_structure() {
        // 正确的理解：每个变量对应一个输入缓冲区
        let vec_expr = vec2(var("position_x"), var("position_y"));
        let variables = vec!["position_x", "position_y"];
        
        // 输入数据：每个变量是一个独立的缓冲区
        let inputs = vec![
            vec![1.0, 2.0, 3.0],    // position_x 的三个元素
            vec![4.0, 5.0, 6.0],    // position_y 的三个元素
        ];
        
        let graph = compile_to_gpu_graph(&vec_expr, &variables, 3);
        let outputs = graph.execute(&inputs);
        
        println!("Correct input structure test:");
        println!("  Input buffers: {}", inputs.len());
        println!("  Output buffers: {}", outputs.len());
        println!("  X component: {:?}", outputs[0]); // 应该输出 [1.0, 2.0, 3.0]
        println!("  Y component: {:?}", outputs[1]); // 应该输出 [4.0, 5.0, 6.0]
        
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(outputs[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_single_variable() {
        // 单个变量的情况
        let expr = var("temperature");
        let variables = vec!["temperature"];
        let inputs = vec![
            vec![25.0, 30.0, 35.0], // temperature 数据
        ];
        
        let graph = compile_to_gpu_graph(&expr, &variables, 3);
        let outputs = graph.execute(&inputs);
        
        println!("Single variable test:");
        println!("  Input: {:?}", inputs[0]);
        println!("  Output: {:?}", outputs[0]);
        
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], vec![25.0, 30.0, 35.0]);
    }

    #[test]
    fn test_vector_addition_correct() {
        let v1 = vec2(var("a_x"), var("a_y"));
        let v2 = vec2(lit(1.0), lit(2.0));
        let sum = v1 + v2;

        let variables = vec!["a_x", "a_y"];
        let inputs = vec![
            vec![10.0, 20.0], // a_x 数据
            vec![5.0, 15.0],  // a_y 数据
        ];
        
        let graph = compile_to_gpu_graph(&sum, &variables, 2);
        let outputs = graph.execute(&inputs);
        
        println!("Vector addition test:");
        println!("  Input a_x: {:?}", inputs[0]);
        println!("  Input a_y: {:?}", inputs[1]);
        println!("  Output X: {:?}", outputs[0]); // 应该输出 [11.0, 21.0]
        println!("  Output Y: {:?}", outputs[1]); // 应该输出 [7.0, 17.0]
        
        assert_eq!(outputs[0], vec![11.0, 21.0]);
        assert_eq!(outputs[1], vec![7.0, 17.0]);
    }

    #[test]
    fn test_understanding_input_output() {
        // 这个测试帮助我们理解输入输出的结构
        println!("=== Understanding Input/Output Structure ===");
        
        // 表达式：vec2(x, y)
        // 变量：["x", "y"] 
        // 这意味着：
        // - 输入缓冲区 0: 变量 x 的数据
        // - 输入缓冲区 1: 变量 y 的数据
        // - 输出缓冲区 0: 向量 x 分量（直接来自输入缓冲区 0）
        // - 输出缓冲区 1: 向量 y 分量（直接来自输入缓冲区 1）
        
        let expr = vec2(var("x"), var("y"));
        let variables = vec!["x", "y"];
        let inputs = vec![
            vec![100.0, 200.0, 300.0], // x 数据
            vec![400.0, 500.0, 600.0], // y 数据
        ];
        
        let graph = compile_to_gpu_graph(&expr, &variables, 3);
        
        println!("Graph structure:");
        println!("  Input count: {}", graph.input_count);
        println!("  Output buffers: {:?}", graph.output_buffers);
        println!("  Total buffers: {}", graph.buffer_count);
        println!("  Kernels: {}", graph.kernels.len());
        
        let outputs = graph.execute(&inputs);
        
        println!("Results:");
        for (i, output) in outputs.iter().enumerate() {
            println!("  Output {}: {:?}", i, output);
        }
        
        // 对于简单的向量创建，输出应该直接是输入数据
        assert_eq!(outputs[0], vec![100.0, 200.0, 300.0]);
        assert_eq!(outputs[1], vec![400.0, 500.0, 600.0]);
    }
}