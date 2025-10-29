use std::ops::{Add, Sub, Mul, Div};


use crate::dsl::GpuVar;
use crate::dsl::Expr;
use crate::dsl::GpuFunc;

// ================= 基础类型定义 =================

// 在 traits.rs 中添加向量分量访问
pub trait VectorComponents {
    fn x(self) -> Expr;
    fn y(self) -> Expr;
    fn z(self) -> Expr;
    fn w(self) -> Expr;
}

impl VectorComponents for WRef {
    fn x(self) -> Expr {
        Expr::Call(GpuFunc::X(Box::new(self.to_expr())))
    }
    
    fn y(self) -> Expr {
        Expr::Call(GpuFunc::Y(Box::new(self.to_expr())))
    }
    
    fn z(self) -> Expr {
        Expr::Call(GpuFunc::Z(Box::new(self.to_expr())))
    }
    
    fn w(self) -> Expr {
        Expr::Call(GpuFunc::W(Box::new(self.to_expr())))
    }
}

    
/// 包装 f32 标量
#[derive(Debug, Clone, Copy)]
pub struct Wf32(pub f32);

impl Into<Expr> for Wf32 {
    fn into(self) -> Expr {
        Expr::Scalar(self.0)
    }
}

impl Into<Expr> for u32 {
    fn into(self) -> Expr {
        Expr::Scalar(self as f32)
    }
}


impl Into<Expr> for i32 {
    fn into(self) -> Expr {
        Expr::Scalar(self as f32)
    }
}


impl Into<Expr> for f32 {
    fn into(self) -> Expr {
        Expr::Scalar(self)
    }
}


/// 包装 2D 向量
#[derive(Debug, Clone, Copy)]
pub struct Wvec2f32(pub [f32; 2]);

/// 包装 3D 向量  
#[derive(Debug, Clone, Copy)]
pub struct Wvec3f32(pub [f32; 3]);

/// 包装 4D 向量
#[derive(Debug, Clone, Copy)]
pub struct Wvec4f32(pub [f32; 4]);

/// GPU 变量引用
#[derive(Debug, Clone)]
pub struct WRef(pub GpuVar);

/// CPU 变量引用
#[derive(Debug, Clone)]
pub struct CRef(pub String, pub usize);

// ================= ToExpr Trait =================

/// 将类型转换为表达式树的 trait
pub trait ToExpr {
    fn to_expr(self) -> Expr;
}

// 为基本类型实现 ToExpr
impl ToExpr for f32 {
    fn to_expr(self) -> Expr {
        Expr::Scalar(self)
    }
}

impl ToExpr for u32 {
    fn to_expr(self) -> Expr {
        Expr::Scalar(self as f32)
    }
}

impl ToExpr for i32 {
    fn to_expr(self) -> Expr {
        Expr::Scalar(self as f32)
    }
}

// 为包装类型实现 ToExpr
impl ToExpr for Wf32 {
    fn to_expr(self) -> Expr {
        Expr::Scalar(self.0)
    }
}

impl ToExpr for Wvec2f32 {
    fn to_expr(self) -> Expr {
        let name = format!("__const_vec2_{}_{}", self.0[0], self.0[1]);
        Expr::Var(name, 2)
    }
}

impl ToExpr for Wvec3f32 {
    fn to_expr(self) -> Expr {
        let name = format!("__const_vec3_{}_{}_{}", self.0[0], self.0[1], self.0[2]);
        Expr::Var(name, 3)
    }
}

impl ToExpr for Wvec4f32 {
    fn to_expr(self) -> Expr {
        let name = format!("__const_vec4_{}_{}_{}_{}", self.0[0], self.0[1], self.0[2], self.0[3]);
        Expr::Var(name, 4)
    }
}

impl ToExpr for WRef {
    fn to_expr(self) -> Expr {
        Expr::GpuVar(self.0)
    }
}

impl ToExpr for CRef {
    fn to_expr(self) -> Expr {
        Expr::Var(self.0, self.1)
    }
}

// 为 Expr 自身实现 ToExpr
impl ToExpr for Expr {
    fn to_expr(self) -> Expr {
        self
    }
}

// ================= WRef 便捷构造方法 =================

impl WRef {
    pub fn uv() -> Self { WRef(GpuVar::Uv) }
    pub fn color() -> Self { WRef(GpuVar::Color) }
    pub fn normal() -> Self { WRef(GpuVar::Normal) }
    pub fn position() -> Self { WRef(GpuVar::Position) }
    pub fn time() -> Self { WRef(GpuVar::Time) }
    pub fn global_id_x() -> Self { WRef(GpuVar::GlobalIdX) }
    pub fn global_id_y() -> Self { WRef(GpuVar::GlobalIdY) }
    pub fn buffer_input(idx: u32) -> Self { WRef(GpuVar::BufferInput(idx)) }
    pub fn buffer_output(idx: u32) -> Self { WRef(GpuVar::BufferOutput(idx)) }
}
// ================= 运算宏定义 =================
// ================= 运算宏定义 =================

/// 为类型实现基本运算的宏
macro_rules! impl_bin_ops {
    ($type:ty) => {
        impl Add for $type {
            type Output = Expr;
            fn add(self, rhs: Self) -> Self::Output {
                Expr::Add(Box::new(self.to_expr()), Box::new(rhs.to_expr()))
            }
        }
        
        impl Sub for $type {
            type Output = Expr;
            fn sub(self, rhs: Self) -> Self::Output {
                Expr::Sub(Box::new(self.to_expr()), Box::new(rhs.to_expr()))
            }
        }
        
        impl Mul for $type {
            type Output = Expr;
            fn mul(self, rhs: Self) -> Self::Output {
                Expr::Mul(Box::new(self.to_expr()), Box::new(rhs.to_expr()))
            }
        }
        
        impl Div for $type {
            type Output = Expr;
            fn div(self, rhs: Self) -> Self::Output {
                Expr::Div(Box::new(self.to_expr()), Box::new(rhs.to_expr()))
            }
        }
    };
}

/// 为两个不同类型实现混合运算的宏
macro_rules! impl_mixed_ops {
    ($lhs:ty, $rhs:ty) => {
        impl Add<$rhs> for $lhs {
            type Output = Expr;
            fn add(self, rhs: $rhs) -> Self::Output {
                Expr::Add(Box::new(self.to_expr()), Box::new(rhs.to_expr()))
            }
        }
        
        impl Sub<$rhs> for $lhs {
            type Output = Expr;
            fn sub(self, rhs: $rhs) -> Self::Output {
                Expr::Sub(Box::new(self.to_expr()), Box::new(rhs.to_expr()))
            }
        }
        
        impl Mul<$rhs> for $lhs {
            type Output = Expr;
            fn mul(self, rhs: $rhs) -> Self::Output {
                Expr::Mul(Box::new(self.to_expr()), Box::new(rhs.to_expr()))
            }
        }
        
        impl Div<$rhs> for $lhs {
            type Output = Expr;
            fn div(self, rhs: $rhs) -> Self::Output {
                Expr::Div(Box::new(self.to_expr()), Box::new(rhs.to_expr()))
            }
        }
    };
}
// ================= 同类型运算 =================
impl_bin_ops!(Wf32);
impl_bin_ops!(WRef);
impl_bin_ops!(CRef);
impl_bin_ops!(Wvec2f32);
impl_bin_ops!(Wvec3f32);
impl_bin_ops!(Wvec4f32);

// ================= 混合类型运算 =================
// WRef 与向量
impl_mixed_ops!(WRef, Wvec2f32);
impl_mixed_ops!(WRef, Wvec3f32);
impl_mixed_ops!(WRef, Wvec4f32);

impl_mixed_ops!(Wvec2f32, WRef);
impl_mixed_ops!(Wvec3f32, WRef);
impl_mixed_ops!(Wvec4f32, WRef);

// Wf32 与向量
impl_mixed_ops!(Wf32, Wvec2f32);
impl_mixed_ops!(Wf32, Wvec3f32);
impl_mixed_ops!(Wf32, Wvec4f32);

impl_mixed_ops!(Wf32, f32);


impl_mixed_ops!(Wvec2f32, Wf32);
impl_mixed_ops!(Wvec3f32, Wf32);
impl_mixed_ops!(Wvec4f32, Wf32);

// CRef 与向量
impl_mixed_ops!(CRef, Wvec2f32);
impl_mixed_ops!(CRef, Wvec3f32);
impl_mixed_ops!(CRef, Wvec4f32);

impl_mixed_ops!(Wvec2f32, CRef);
impl_mixed_ops!(Wvec3f32, CRef);
impl_mixed_ops!(Wvec4f32, CRef);
impl_mixed_ops!(Expr, Expr);
// 视情况也可以加反向
impl_mixed_ops!(Wvec2f32, Expr);
impl_mixed_ops!(Wvec3f32, Expr);
impl_mixed_ops!(Wvec4f32, Expr);

// 视情况也可以加反向
impl_mixed_ops!(Expr,Wvec2f32);
impl_mixed_ops!(Expr,Wvec3f32);
impl_mixed_ops!(Expr,Wvec4f32);

impl_mixed_ops!(Expr,f32);
impl_mixed_ops!(Expr,u32);
impl_mixed_ops!(Expr,i32);
impl_mixed_ops!(Expr,WRef);


// ================= 便捷构造方法 =================

/// 创建标量常量
pub fn scalar(value: f32) -> Wf32 {
    Wf32(value)
}

/// 创建 2D 向量常量
pub fn vec2(x: f32, y: f32) -> Expr {
    Expr::Vec2(Box::new(Expr::Scalar(x)), Box::new(Expr::Scalar(y)))
}

/// 创建 2D 向量常量
pub fn vec3(x: f32, y: f32,z: f32) -> Expr {
    Expr::Vec3(Box::new(Expr::Scalar(x)), Box::new(Expr::Scalar(y)),Box::new(Expr::Scalar(z)))
}


/// 创建 4D 向量常量
pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> Expr {
    Expr::Vec4(Box::new(Expr::Scalar(x)), Box::new(Expr::Scalar(y)),Box::new(Expr::Scalar(z)),Box::new(Expr::Scalar(w)))
}

/// 创建 CPU 变量引用
pub fn var(name: &str, size: usize) -> CRef {
    CRef(name.to_string(), size)
}

// ================= GPU 变量便捷别名 =================

/// 获取 UV 坐标
pub fn uv() -> WRef {
    WRef::uv()
}

/// 获取颜色
pub fn color() -> WRef {
    WRef::color()
}

/// 获取法线
pub fn normal() -> WRef {
    WRef::normal()
}

/// 获取位置
pub fn position() -> WRef {
    WRef::position()
}
pub struct IF;

 // 比较操作
 pub fn eq<T: Into<Expr>, U: Into<Expr>>(a: T, b: U) -> Expr {
     Expr::Eq(Box::new(a.into()), Box::new(b.into()))
 }
 pub fn ne<T: Into<Expr>, U: Into<Expr>>(a: T, b: U) -> Expr {
     Expr::Ne(Box::new(a.into()), Box::new(b.into()))
 }
 pub fn lt<T: Into<Expr>, U: Into<Expr>>(a: T, b: U) -> Expr {
     Expr::Lt(Box::new(a.into()), Box::new(b.into()))
 }
 pub fn le<T: Into<Expr>, U: Into<Expr>>(a: T, b: U) -> Expr {
     Expr::Le(Box::new(a.into()), Box::new(b.into()))
 }
 pub fn gt<T: Into<Expr>, U: Into<Expr>>(a: T, b: U) -> Expr {
     Expr::Gt(Box::new(a.into()), Box::new(b.into()))
 }
 pub fn ge<T: Into<Expr>, U: Into<Expr>>(a: T, b: U) -> Expr {
     Expr::Ge(Box::new(a.into()), Box::new(b.into()))
 }

impl IF {
    // 构造 Expr::If
    pub fn if_or_else<T: Into<Expr>, U: Into<Expr>, V: Into<Expr>>(cond: T, then_expr: U, else_expr: V) -> Expr {
        Expr::If {
            cond: Box::new(cond.into()),
            then_expr: Box::new(then_expr.into()),
            else_expr: Box::new(else_expr.into()),
        }
    }

}
/// 获取时间
pub fn time() -> WRef {
    WRef::time()
}

/// 获取全局 X ID
pub fn global_id_x() -> WRef {
    WRef::global_id_x()
}

/// 获取全局 Y ID  
pub fn global_id_y() -> WRef {
    WRef::global_id_y()
}

/// 获取缓冲区输入
pub fn buffer_input(idx: u32) -> WRef {
    WRef::buffer_input(idx)
}

/// 获取缓冲区输出
pub fn buffer_output(idx: u32) -> WRef {
    WRef::buffer_output(idx)
}

// ================= 函数式方法扩展 =================

/// 为表达式添加数学函数支持
pub trait ExprExt {
    fn sin(self) -> Expr;
    fn cos(self) -> Expr;
    fn abs(self) -> Expr;
    fn normalize(self) -> Expr;
}

impl ExprExt for Expr {
    fn sin(self) -> Expr {
        Expr::Call(GpuFunc::Sin(Box::new(self)))
    }
    
    fn cos(self) -> Expr {
        Expr::Call(GpuFunc::Cos(Box::new(self)))
    }
    
    fn abs(self) -> Expr {
        Expr::Call(GpuFunc::Abs(Box::new(self)))
    }
    
    fn normalize(self) -> Expr {
        Expr::Call(GpuFunc::Normalize(Box::new(self)))
    }
}

/// 为包装类型添加数学函数支持
pub trait WrapperExt: ToExpr 
    where Self: Sized
{
    fn sin(self) -> Expr {
        self.to_expr().sin()
    }
    
    fn cos(self) -> Expr {
        self.to_expr().cos()
    }
    
    fn abs(self) -> Expr {
        self.to_expr().abs()
    }
    
    fn normalize(self) -> Expr {
        self.to_expr().normalize()
    }
}

// 为所有包装类型实现 WrapperExt
impl WrapperExt for Wf32 {}
impl WrapperExt for WRef {}
impl WrapperExt for CRef {}
impl WrapperExt for Wvec2f32 {}
impl WrapperExt for Wvec3f32 {}
impl WrapperExt for Wvec4f32 {}
// ================= 测试工具 =================


#[cfg(test)]
mod dsl_tests {
    use crate::dsl::{build_all_nodes_as_linear, build_op_index_map, cpu_compute, flatten_nodes_for_gpu, simulate_compute_table_by_tag};

    use super::*;
    use std::collections::HashMap;
// 修改测试部分，添加调试信息来定位问题
#[cfg(test)]
mod dsl_tests {
    use crate::dsl::{build_all_nodes_as_linear, build_op_index_map, cpu_compute, flatten_nodes_for_gpu, simulate_compute_table_by_tag};

    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_dsl_builders_new_style() {
        // 1️⃣ 环境变量
        let mut env: HashMap<String, Vec<f32>> = HashMap::new();
        env.insert("GPU_POSITION".into(), vec![1.0, 2.0]);
        env.insert("time".into(), vec![0.5, 0.5]);

        // 2️⃣ 构造 X 矩阵（行 = 样本，列 = 变量）
        let X = vec![
            vec![0.0, 0.0, 0.0, env["GPU_POSITION"][0], env["time"][0]],
            vec![0.0, 0.0, 0.0, env["GPU_POSITION"][1], env["time"][1]],
        ];

        let col_keys: Vec<String> = vec![
            "Uv".to_string(),
            "Color".to_string(),
            "Normal".to_string(),
            "GPU_POSITION".to_string(),
            "time".to_string(),
        ];

        let cpu_position = 3f32;

        // 3️⃣ 构造表达式（新 DSL 风格）
        let expr = vec2(2.0, 2.0) * cpu_position * 30.0 * vec2(2.0, 5.0) * position();
        
        println!("原始表达式结果:");
        let cpu_result = cpu_compute(&expr, &X, &col_keys);
        for (i, val) in cpu_result.iter().enumerate() {
            println!("Row {} -> result = {}", i, val);
        }

        // 4️⃣ 创建条件表达式 - 修复这里！
        // 原来的条件: eq(expr.clone(), Wf32(3.0)) 可能有问题
        // 因为 expr 是向量，Wf32(3.0) 是标量，比较可能不按预期工作
        
        // 尝试使用更合理的条件
        let condition = gt(position().x(), scalar(5.0)); // 使用 position 的 x 分量与 1.5 比较
        
        println!("条件表达式:");
        let condition_result = cpu_compute(&condition, &X, &col_keys);
        for (i, val) in condition_result.iter().enumerate() {
            println!("Row {} -> condition = {}", i, val);
        }

        let check = IF::if_or_else(
            condition, 
            5.0,  // then 分支
            3.0   // else 分支
        );

        println!("IF 表达式结构: {:?}", check);

        // 5️⃣ 线性化生成 GPU 数据
        let extra_gpu_inputs = vec![];
        let (X_nodes, node_ws, node_bs, node_names, compute_table) =
            build_all_nodes_as_linear(&check, &env, &extra_gpu_inputs);

        println!("计算表:");
        for (i, node) in compute_table.entries.iter().enumerate() {
            println!("Node {}: {:?}", i, node);
        }

        let (mut flat_w, mut flat_b, node_count, n, m) =
            flatten_nodes_for_gpu(&node_ws, &node_bs);

        println!("扁平化数据 - node_count: {}, n: {}, m: {}", node_count, n, m);
        println!("flat_b BEFORE simulate: {:?}", flat_b);

        // 6️⃣ 构造 op_index
        let op_index = build_op_index_map(&compute_table, n, node_count);

        // 7️⃣ GPU 模拟计算
        simulate_compute_table_by_tag(&mut flat_b, n, &op_index, &compute_table, &X_nodes);
        println!("flat_b AFTER simulate: {:?}", flat_b);

        // 8️⃣ 验证 IF 表达式结果
        let if_cpu_result = cpu_compute(&expr, &X, &col_keys);
        println!("IF 表达式 CPU 计算结果:");
        for (i, val) in if_cpu_result.iter().enumerate() {
            println!("Row {} -> if_result = {}", i, val);
        }

        // 验证一致性
        for i in 0..n {
            let expected = if_cpu_result[i];
            let actual = flat_b[i * node_count + (node_count - 1)]; // 最后一个节点的输出
            println!("样本 {}: 期望 = {}, 实际 = {}", i, expected, actual);
        }
    }
}
}