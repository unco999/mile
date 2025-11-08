use crate::core::*;
use std::f32::EPSILON;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};
use std::rc::Rc;

impl From<f64> for Expr {
    fn from(v: f64) -> Self {
        Expr::Constant(v as f32)
    }
}

impl From<u32> for Expr {
    fn from(v: u32) -> Self {
        Expr::Constant(v as f32)
    }
}
impl From<bool> for Expr {
    fn from(b: bool) -> Self {
        Expr::Constant(if b { 1.0 } else { 0.0 })
    }
}

// VecN -> Expr (wrap a VecN as an Expr::VecN)
impl From<Vec2> for Expr {
    fn from(v: Vec2) -> Self {
        Expr::Vec2(v)
    }
}
impl From<Vec3> for Expr {
    fn from(v: Vec3) -> Self {
        Expr::Vec3(v)
    }
}
impl From<Vec4> for Expr {
    fn from(v: Vec4) -> Self {
        Expr::Vec4(v)
    }
}

// ---------- tuple -> Expr (vec2/vec3/vec4) by generics ----------
// (T,U) -> Vec2 Expr
impl<T, U> From<(T, U)> for Expr
where
    T: Into<Expr>,
    U: Into<Expr>,
{
    fn from(tu: (T, U)) -> Self {
        let (a, b) = tu;
        Expr::Vec2(Vec2::new(Box::new(a.into()), Box::new(b.into())))
    }
}

// (T,U,V) -> Vec3 Expr
impl<T, U, V> From<(T, U, V)> for Expr
where
    T: Into<Expr>,
    U: Into<Expr>,
    V: Into<Expr>,
{
    fn from(tuv: (T, U, V)) -> Self {
        let (a, b, c) = tuv;
        Expr::Vec3(Vec3::new(
            Box::new(a.into()),
            Box::new(b.into()),
            Box::new(c.into()),
        ))
    }
}

// (T,U,V,W) -> Vec4 Expr
impl<T, U, V, W> From<(T, U, V, W)> for Expr
where
    T: Into<Expr>,
    U: Into<Expr>,
    V: Into<Expr>,
    W: Into<Expr>,
{
    fn from(tuvw: (T, U, V, W)) -> Self {
        let (a, b, c, d) = tuvw;
        Expr::Vec4(Vec4::new(
            Box::new(a.into()),
            Box::new(b.into()),
            Box::new(c.into()),
            Box::new(d.into()),
        ))
    }
}

// 同时给 VecN 自身也加上 From<(T,...)> 以便直接构造 VecN（如果你也需要单独 VecN）
impl<T, U> From<(T, U)> for Vec2
where
    T: Into<Expr>,
    U: Into<Expr>,
{
    fn from((a, b): (T, U)) -> Self {
        Vec2::new(Box::new(a.into()), Box::new(b.into()))
    }
}
impl<T, U, V> From<(T, U, V)> for Vec3
where
    T: Into<Expr>,
    U: Into<Expr>,
    V: Into<Expr>,
{
    fn from((a, b, c): (T, U, V)) -> Self {
        Vec3::new(Box::new(a.into()), Box::new(b.into()), Box::new(c.into()))
    }
}
impl<T, U, V, W> From<(T, U, V, W)> for Vec4
where
    T: Into<Expr>,
    U: Into<Expr>,
    V: Into<Expr>,
    W: Into<Expr>,
{
    fn from((a, b, c, d): (T, U, V, W)) -> Self {
        Vec4::new(
            Box::new(a.into()),
            Box::new(b.into()),
            Box::new(c.into()),
            Box::new(d.into()),
        )
    }
}

// ---------- optionally: array -> Expr (if 2/3/4 sized arrays desired) ----------
impl From<[f32; 2]> for Expr {
    fn from(a: [f32; 2]) -> Self {
        Expr::Vec2(Vec2::new(
            Box::new(Expr::Constant(a[0])),
            Box::new(Expr::Constant(a[1])),
        ))
    }
}
impl From<[f32; 3]> for Expr {
    fn from(a: [f32; 3]) -> Self {
        Expr::Vec3(Vec3::new(
            Box::new(Expr::Constant(a[0])),
            Box::new(Expr::Constant(a[1])),
            Box::new(Expr::Constant(a[2])),
        ))
    }
}
impl From<[f32; 4]> for Expr {
    fn from(a: [f32; 4]) -> Self {
        Expr::Vec4(Vec4::new(
            Box::new(Expr::Constant(a[0])),
            Box::new(Expr::Constant(a[1])),
            Box::new(Expr::Constant(a[2])),
            Box::new(Expr::Constant(a[3])),
        ))
    }
}
// ---------- htructors that fold when possible ----------

pub fn vec2<X: Into<Expr>, Y: Into<Expr>>(x: X, y: Y) -> Expr {
    Expr::Vec2(Vec2::new(Box::new(x.into()), Box::new(y.into())))
}
pub fn vec3<X: Into<Expr>, Y: Into<Expr>, Z: Into<Expr>>(x: X, y: Y, z: Z) -> Expr {
    Expr::Vec3(Vec3::new(
        Box::new(x.into()),
        Box::new(y.into()),
        Box::new(z.into()),
    ))
}
pub fn vec4<A: Into<Expr>, B: Into<Expr>, C: Into<Expr>, D: Into<Expr>>(
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

pub fn cv(v: &'static str) -> Expr {
    Expr::ComputeImport(v)
}

pub fn rv(v: &'static str) -> Expr {
    Expr::RenderImport(v)
}

pub fn mix<A: Into<Expr>, B: Into<Expr>, T: Into<Expr>>(a: A, b: B, t: T) -> Expr {
    let a_expr = a.into();
    let b_expr = b.into();
    let t_expr = t.into();
    let one_minus_t = Expr::from(1.0f32) - t_expr.clone();
    let first = a_expr * one_minus_t;
    let second = b_expr * t_expr;
    first + second
}

pub fn smoothstep<A: Into<Expr>, B: Into<Expr>, C: Into<Expr>>(
    edge0: A,
    edge1: B,
    value: C,
) -> Expr {
    Expr::SmoothStep {
        edge0: Box::new(edge0.into()),
        edge1: Box::new(edge1.into()),
        value: Box::new(value.into()),
    }
}

impl Expr {
    pub fn x(&self) -> Expr {
        self.take(0)
    }

    pub fn y(&self) -> Expr {
        self.take(1)
    }

    pub fn z(&self) -> Expr {
        self.take(2)
    }

    pub fn w(&self) -> Expr {
        self.take(3)
    }

    pub fn take<T: Into<Expr>>(&self, index_expr: T) -> Expr {
        Expr::BinaryOp(
            BinaryOp::Index,
            Box::new(self.clone()), // 使用 Rc 共享而不是克隆
            Box::new(index_expr.into()),
        )
    }
}

// ... 同样提供 cos_expr, sqrt_expr 等（按需添加） ...

// ---------- override arithmetic traits with eager folding & vector broadcasting ----------

// Add convenience Expr + f32 (left expr)
impl Add<f32> for Expr {
    type Output = Expr;
    fn add(self, rhs: f32) -> Expr {
        self + Expr::Constant(rhs)
    }
}

impl Add<u32> for Expr {
    type Output = Expr;
    fn add(self, rhs: u32) -> Expr {
        self + Expr::Constant(rhs as f32)
    }
}

impl Add<i32> for Expr {
    type Output = Expr;
    fn add(self, rhs: i32) -> Expr {
        self + Expr::Constant(rhs as f32)
    }
}

impl Add<&'static str> for Expr {
    type Output = Expr;
    fn add(self, rhs: &'static str) -> Expr {
        self + Expr::RenderImport(rhs)
    }
}

// convenience: Expr * f32
impl Mul<f32> for Expr {
    type Output = Expr;
    fn mul(self, rhs: f32) -> Expr {
        self * Expr::Constant(rhs)
    }
}

impl Mul<Expr> for f32 {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Constant(self as f32) * rhs
    }
}

impl Mul<Expr> for u32 {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Constant(self as f32) * rhs
    }
}

impl Mul<Expr> for i32 {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Constant(self as f32) * rhs
    }
}

impl Add<Expr> for f32 {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Constant(self as f32) + rhs
    }
}

impl Add<Expr> for u32 {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Constant(self as f32) + rhs
    }
}

impl Add<Expr> for i32 {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Constant(self as f32) + rhs
    }
}

impl Div<Expr> for f32 {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::Constant(self as f32) / rhs
    }
}

impl Div<Expr> for u32 {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::Constant(self as f32) / rhs
    }
}

impl Div<Expr> for i32 {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::Constant(self as f32) / rhs
    }
}

impl Sub<Expr> for f32 {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::Constant(self as f32) - rhs
    }
}

impl Sub<Expr> for u32 {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::Constant(self as f32) - rhs
    }
}

impl Sub<Expr> for i32 {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::Constant(self as f32) - rhs
    }
}

impl Sub<f32> for Expr {
    type Output = Expr;

    fn sub(self, rhs: f32) -> Self::Output {
        Expr::BinaryOp(
            BinaryOp::Subtract,
            Box::new(self),
            Box::new(Expr::Constant(rhs)),
        )
    }
}

impl Mul<Expr> for &'static str {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::RenderImport(self) * rhs
    }
}

impl Div<Expr> for &'static str {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::RenderImport(self) / rhs
    }
}

impl Add<Expr> for &'static str {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::RenderImport(self) + rhs
    }
}

impl Sub<Expr> for &'static str {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::RenderImport(self) - rhs
    }
}

// convenience: Expr * f32
impl Mul<u32> for Expr {
    type Output = Expr;
    fn mul(self, rhs: u32) -> Expr {
        self * Expr::Constant(rhs as f32)
    }
}

// convenience: Expr * f32
impl Mul<i32> for Expr {
    type Output = Expr;
    fn mul(self, rhs: i32) -> Expr {
        self * Expr::Constant(rhs as f32)
    }
}
