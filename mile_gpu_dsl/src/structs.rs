use std::marker::PhantomData;
use gpu_macro::TypeHash;
use bitflags::bitflags;

pub trait TypeHash {
    const HASH: u32;
}

// ------------------ Mask 类型 ------------------
bitflags! {
    #[derive(Clone, Copy, Debug, TypeHash)]
    pub struct TargetTeam: u32 {
        const DEFAULT = 0;
        const ENEMY   = 1 << 0;
        const ALLY    = 1 << 1;
        const BOSS    = 1 << 2;
    }

    #[derive(Clone, Copy, Debug, TypeHash)]
    pub struct PlayerType: u32 {
        const DEFAULT = 0;
        const LOCAL   = 1 << 0;
        const REMOTE  = 1 << 1;
    }
}

#[derive(TypeHash, Debug)]
pub struct Position;

// ------------------ 单个 Slot 类型 ------------------
#[derive(Debug)]
pub struct Slot<T: TypeHash, const HAS: u32, const NOT: u32> {
    _ty: PhantomData<T>,
}

impl<T: TypeHash, const HAS: u32, const NOT: u32> Slot<T, HAS, NOT> {
    pub const fn back<U>() -> U {
        // 这里类型层面的返回值，用于类型链
        todo!()
    }
}

// ------------------ Slot 列表 trait ------------------
pub trait ToFliterBlock {
    fn block_des();
}

impl<T: TypeHash, O: ToFilterBlockList> ToFliterBlock for ContextBlock<T, O> {
    fn block_des() {
        // 递归调用它包含的 Slot 列表
        O::block_des_list();
    }
}

pub trait ToFilterBlockList {
    fn block_des_list();
}

// 单个 Slot 实现 ToFliterBlock
impl<T: TypeHash, const HAS: u32, const NOT: u32> ToFliterBlock for Slot<T, HAS, NOT> {
    fn block_des() {
        println!(
            "Slot type: {:?}, HAS: {:#X}, NOT: {:#X}",
            T::HASH, HAS, NOT
        );
    }
}

// 元组实现 ToFilterBlockList，最多 N 个元素
macro_rules! impl_filter_list_for_tuples {
    ($($name:ident),+) => {
        impl<$($name: ToFliterBlock),+> ToFilterBlockList for ($($name,)+) {
            fn block_des_list() {
                $( $name::block_des(); )+
            }
        }
    };
}

// 自动生成 1~8 元素元组的实现
impl_filter_list_for_tuples!(T1);
impl_filter_list_for_tuples!(T1, T2);
impl_filter_list_for_tuples!(T1, T2, T3);
impl_filter_list_for_tuples!(T1, T2, T3, T4);
impl_filter_list_for_tuples!(T1, T2, T3, T4, T5);
impl_filter_list_for_tuples!(T1, T2, T3, T4, T5, T6);
impl_filter_list_for_tuples!(T1, T2, T3, T4, T5, T6, T7);
impl_filter_list_for_tuples!(T1, T2, T3, T4, T5, T6, T7, T8);


#[derive(Debug, Clone, Copy)]
pub enum OpCode {
    /// 条件判断：IF(condition)
    If,

    /// 调用某个函数或上下文：CALL(func)
    Call,

    /// 算术：加法
    Add,

    /// 算术：乘法
    Mul,

    /// 赋值：SET(value)
    Set,

    /// 比较运算
    Greater,
    Less,
    Equal,

    /// 逻辑操作
    And,
    Or,
    Not,

    /// 控制流
    BeginScope,
    EndScope,

    /// 常量或变量访问
    Load,
    Store,
}

// ------------------ TYPE 容器 ------------------
pub struct ContextBlock<T: TypeHash, O: ToFilterBlockList> {
    _ty: PhantomData<T>,
    _data: PhantomData<O>,
}

impl<T: TypeHash, O: ToFilterBlockList> ContextBlock<T, O> {
    pub fn build() {
        // 调用 Slot 列表输出信息
        O::block_des_list();
    }
}

type MySlots1 = ContextBlock<
    Position,
    (
        Slot::<PlayerType, {PlayerType::LOCAL.bits()},{PlayerType::DEFAULT.bits()}>,
        Slot::<PlayerType, {PlayerType::LOCAL.bits()},{PlayerType::DEFAULT.bits()}>,
    )
>;

type MySlots2 = ContextBlock<
    Position,
    (
        Slot::<PlayerType, {PlayerType::REMOTE.bits()},{PlayerType::DEFAULT.bits()}>,
    )
>;

type MyCombinedContext = CombinedContext<(MySlots1, MySlots2)>;


pub struct CombinedContext<O: ToFilterBlockList> {
    _data: PhantomData<O>,
}



impl<O: ToFilterBlockList> CombinedContext<O> {
    pub fn build() {
        O::block_des_list();
    }
}

#[test]
fn test_slots() {
    MyCombinedContext::build();
    
}
