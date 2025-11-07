use std::ops::{Add, Div, Mul, Sub};

pub mod dsl {
    use super::{BinaryOp, Expr, UnaryFunc, Vec2, Vec3, Vec4};

    pub fn var(name: &'static str) -> Expr {
        Expr::RenderImport(name)
    }

    pub fn lit<X: Into<Expr>>(v: X) -> Expr {
        v.into()
    }

    pub fn wvec2<X: Into<Expr>, Y: Into<Expr>>(x: X, y: Y) -> Expr {
        Expr::Vec2(Vec2::new(Box::new(x.into()), Box::new(y.into())))
    }

    pub fn wvec3<X: Into<Expr>, Y: Into<Expr>, Z: Into<Expr>>(x: X, y: Y, z: Z) -> Expr {
        Expr::Vec3(Vec3::new(
            Box::new(x.into()),
            Box::new(y.into()),
            Box::new(z.into()),
        ))
    }

    pub fn wvec4<A: Into<Expr>, B: Into<Expr>, C: Into<Expr>, D: Into<Expr>>(
        a: A,
        b: B,
        c: C,
        d: D,
    ) -> Expr {
        Expr::Vec4(Vec4::new(
            Box::new(a.into()),
            Box::new(b.into()),
            Box::new(c.into()),
            Box::new(d.into()),
        ))
    }

    // ---------- unary functions ----------

    pub fn sin<E: Into<Expr>>(e: E) -> Expr {
        Expr::UnaryOp(UnaryFunc::Sin, Box::new(e.into()))
    }

    pub fn cos<E: Into<Expr>>(e: E) -> Expr {
        Expr::UnaryOp(UnaryFunc::Cos, Box::new(e.into()))
    }

    pub fn sqrt<E: Into<Expr>>(e: E) -> Expr {
        Expr::UnaryOp(UnaryFunc::Sqrt, Box::new(e.into()))
    }

    pub struct IF;
    // ---------- binary functions ----------

    impl IF {
        pub fn eq<A: Into<Expr>, B: Into<Expr>>(a: A, b: B) -> Expr {
            Expr::BinaryOp(BinaryOp::Equal, Box::new(a.into()), Box::new(b.into()))
        }

        pub fn ne<A: Into<Expr>, B: Into<Expr>>(a: A, b: B) -> Expr {
            Expr::BinaryOp(BinaryOp::NotEqual, Box::new(a.into()), Box::new(b.into()))
        }

        pub fn gt<A: Into<Expr>, B: Into<Expr>>(a: A, b: B) -> Expr {
            Expr::BinaryOp(
                BinaryOp::GreaterThan,
                Box::new(a.into()),
                Box::new(b.into()),
            )
        }

        pub fn lt<A: Into<Expr>, B: Into<Expr>>(a: A, b: B) -> Expr {
            Expr::BinaryOp(BinaryOp::LessThan, Box::new(a.into()), Box::new(b.into()))
        }

        pub fn ge<A: Into<Expr>, B: Into<Expr>>(a: A, b: B) -> Expr {
            Expr::BinaryOp(
                BinaryOp::GreaterThan,
                Box::new(a.into()),
                Box::new(b.into()),
            )
        }

        pub fn le<A: Into<Expr>, B: Into<Expr>>(a: A, b: B) -> Expr {
            Expr::BinaryOp(BinaryOp::LessEqual, Box::new(a.into()), Box::new(b.into()))
        }

        pub fn of<C: Into<Expr>, T: Into<Expr>, E: Into<Expr>>(
            cond: C,
            then_v: T,
            else_v: E,
        ) -> Expr {
            Expr::If {
                condition: Box::new(cond.into()),
                then_branch: Box::new(then_v.into()),
                else_branch: Box::new(else_v.into()),
            }
        }
    }

    pub fn modulo<A: Into<Expr>, B: Into<Expr>>(a: A, b: B) -> Expr {
        Expr::BinaryOp(BinaryOp::Modulo, Box::new(a.into()), Box::new(b.into()))
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
    Index,
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
    RenderImport(&'static str),
    ComputeImport(&'static str),
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

impl From<&'static str> for Expr {
    fn from(v: &'static str) -> Self {
        Expr::RenderImport(v)
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
            (Expr::Vec2(lhs), Expr::Vec2(rhs)) => Expr::Vec2(Vec2::new(
                Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.x, rhs.x)),
                Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.y, rhs.y)),
            )),
            (Expr::Vec3(lhs), Expr::Vec3(rhs)) => Expr::Vec3(Vec3::new(
                Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.x, rhs.x)),
                Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.y, rhs.y)),
                Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.z, rhs.z)),
            )),
            (Expr::Vec4(lhs), Expr::Vec4(rhs)) => Expr::Vec4(Vec4::new(
                Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.x, rhs.x)),
                Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.y, rhs.y)),
                Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.z, rhs.z)),
                Box::new(Expr::BinaryOp(BinaryOp::Add, lhs.w, rhs.w)),
            )),
            (Expr::Vec2(vec), scalar) => Expr::Vec2(Vec2::new(
                Box::new(Expr::BinaryOp(
                    BinaryOp::Add,
                    vec.x,
                    Box::new(scalar.clone()),
                )),
                Box::new(Expr::BinaryOp(BinaryOp::Add, vec.y, Box::new(scalar))),
            )),
            (scalar, Expr::Vec2(vec)) => Expr::Vec2(Vec2::new(
                Box::new(Expr::BinaryOp(
                    BinaryOp::Add,
                    Box::new(scalar.clone()),
                    vec.x,
                )),
                Box::new(Expr::BinaryOp(BinaryOp::Add, Box::new(scalar), vec.y)),
            )),
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
    InjectConstant {
        value: f32,
        out: usize,
    },
    ElementwiseBinary {
        op: BinaryOp,
        a: usize,
        b: usize,
        out: usize,
    },
    ElementwiseUnary {
        func: UnaryFunc,
        a: usize,
        out: usize,
    },
    CondBlend {
        cond: usize,
        then_buf: usize,
        else_buf: usize,
        out: usize,
    },
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
        let max_kernel_out = self
            .kernels
            .iter()
            .map(|k| k.output_index())
            .max()
            .unwrap_or(0);
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
                                if buffers[*a][i] > buffers[*b][i] {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            BinaryOp::GreaterEqual => {
                                if buffers[*a][i] >= buffers[*b][i] {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            BinaryOp::LessThan => {
                                if buffers[*a][i] < buffers[*b][i] {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            BinaryOp::LessEqual => {
                                if buffers[*a][i] <= buffers[*b][i] {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            BinaryOp::Equal => {
                                if (buffers[*a][i] - buffers[*b][i]).abs() < std::f32::EPSILON {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            BinaryOp::NotEqual => {
                                if (buffers[*a][i] - buffers[*b][i]).abs() >= std::f32::EPSILON {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            _ => 0.0,
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
        self.output_buffers
            .iter()
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
