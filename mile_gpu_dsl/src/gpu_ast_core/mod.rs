pub mod gpu_ast;
pub mod gpu_ast_compute_pipeline;
pub mod gpu_program;
pub mod kennel;
pub mod op;

pub mod event {
    use crate::core::Expr;

    /**
     * 发送指定表达式给监听者
     */
    pub struct ExprWithIdxEvent {
        pub idx: u32,
        pub expr: Expr,
    }

    /**
     * 返回Kennel编号给指定监听者
     */
    pub struct KennelResultIdxEvent {
        pub kennel_id: u32,
        pub idx: u32,
    }
}

use super::prelude::{Expr, gpu_ast_compute_pipeline::*, *};
