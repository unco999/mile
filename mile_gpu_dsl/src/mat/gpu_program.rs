use serde::Serialize;

use crate::{
    core::{BinaryOp, Expr, UnaryFunc, Vec4},
    mat::{
        gpu_ast::{DataType, GpuAstNode, GpuAstState, GpuOp},
        op::{ImportRegistry, ImportType},
    },
};

/// 描述构建 GPU 计算节点失败的原因
#[derive(Debug)]
pub enum ProgramBuildError {
    RootExpressionMustBeVec4,
    MissingImport { name: &'static str },
    RenderImportInCompute { name: &'static str },
    RenderStageExpression(&'static str),
    UnsupportedExpression(&'static str),
}

impl std::fmt::Display for ProgramBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProgramBuildError::RootExpressionMustBeVec4 => {
                write!(f, "root expression must be a Vec4 expression")
            }
            ProgramBuildError::MissingImport { name } => {
                write!(f, "import \"{}\" not registered in ImportRegistry", name)
            }
            ProgramBuildError::RenderImportInCompute { name } => {
                write!(
                    f,
                    "render import \"{}\" cannot be evaluated inside compute stage expressions",
                    name
                )
            }
            ProgramBuildError::RenderStageExpression(ctx) => {
                write!(f, "expression requires render stage support: {}", ctx)
            }
            ProgramBuildError::UnsupportedExpression(ctx) => {
                write!(f, "unsupported AST shape encountered while building program: {}", ctx)
            }
        }
    }
}

impl std::error::Error for ProgramBuildError {}

/// GPU 侧计算阶段节点的序列化表示
#[derive(Debug, Clone, Serialize)]
pub struct SerializableGpuProgram {
    pub compute_nodes: Vec<SerializableComputeNode>,
    pub render_plan: RenderPlan,
}

impl SerializableGpuProgram {
    /// 将序列化节点转换成 GPU 运行时所需的 `GpuAstNode`
    pub fn to_gpu_nodes(&self) -> Vec<GpuAstNode> {
        let mut nodes = Vec::with_capacity(self.compute_nodes.len());
        for node in &self.compute_nodes {
            let mut gpu_node = GpuAstNode::new();
            gpu_node.set_data_type(DataType::Scalar); // 目前所有计算节点都按标量输出
            match node.stage {
                ComputeStage::Precompute => {
                    gpu_node.add_state(GpuAstState::IS_COMPUTE | GpuAstState::PRE_COMPUTED);
                }
                ComputeStage::PerFrame => {
                    gpu_node.add_state(GpuAstState::IS_COMPUTE | GpuAstState::IS_TICK);
                }
            }

            match &node.kind {
                ComputeNodeKind::Constant { value } => {
                    gpu_node.set_constant(*value);
                    gpu_node.add_state(GpuAstState::IS_LEAF);
                }
                ComputeNodeKind::Import { name: _, mask, source } => {
                    match source {
                        ImportSource::Compute => gpu_node.set_import(1, *mask),
                        ImportSource::Render => gpu_node.set_import(0, *mask),
                    }
                    gpu_node.add_state(GpuAstState::IS_LEAF);
                }
                ComputeNodeKind::Unary { op, input } => {
                    gpu_node.set_op(op.to_gpu_op());
                    gpu_node.set_children(*input, u32::MAX);
                }
                ComputeNodeKind::Binary { op, left, right } => {
                    gpu_node.set_op(op.to_gpu_op());
                    gpu_node.set_children(*left, *right);
                }
            }

            nodes.push(gpu_node);
        }
        nodes
    }
}

/// 表示最终渲染阶段的颜色组装计划
#[derive(Debug, Clone, Serialize)]
pub struct RenderPlan {
    pub components: [RenderComponent; 4],
}

impl RenderPlan {
    pub fn new(components: [RenderComponent; 4]) -> Self {
        Self { components }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind")]
pub enum RenderComponent {
    Constant { value: f32 },
    ComputeNode { node_id: u32 },
    RenderImport { name: &'static str, mask: u32 },
}

/// 构建 GPU 计算节点的构建器
pub struct GpuProgramBuilder<'a> {
    registry: &'a ImportRegistry,
    nodes: Vec<SerializableComputeNode>,
}

impl<'a> GpuProgramBuilder<'a> {
    pub fn new(registry: &'a ImportRegistry) -> Self {
        Self {
            registry,
            nodes: Vec::new(),
        }
    }

    pub fn build_program(mut self, expr: &Expr) -> Result<SerializableGpuProgram, ProgramBuildError> {
        let vector = match expr {
            Expr::Vec4(vec4) => vec4,
            _ => return Err(ProgramBuildError::RootExpressionMustBeVec4),
        };

        let components = [
            self.build_component(&vector.x)?,
            self.build_component(&vector.y)?,
            self.build_component(&vector.z)?,
            self.build_component(&vector.w)?,
        ];

        Ok(SerializableGpuProgram {
            compute_nodes: self.nodes,
            render_plan: RenderPlan::new(components),
        })
    }

    fn build_component(&mut self, expr: &Expr) -> Result<RenderComponent, ProgramBuildError> {
        if let Some(value) = evaluate_constant(expr) {
            return Ok(RenderComponent::Constant { value });
        }

        if contains_render_import(expr) {
            return match expr {
                Expr::RenderImport(name) => self.plan_render_import(name),
                _ => Err(ProgramBuildError::RenderStageExpression(
                    "render stage arithmetic is not supported yet",
                )),
            };
        }

        let node = self.build_compute_expr(expr)?;
        Ok(RenderComponent::ComputeNode { node_id: node.id })
    }

    fn plan_render_import(
        &self,
        name: &'static str,
    ) -> Result<RenderComponent, ProgramBuildError> {
        let (import_type, mask) = self
            .registry
            .get_import_info(name)
            .ok_or(ProgramBuildError::MissingImport { name })?;

        match import_type {
            ImportType::Render(_) => Ok(RenderComponent::RenderImport {
                name,
                mask,
            }),
            ImportType::Compute(_) => Err(ProgramBuildError::RenderStageExpression(
                "compute import encountered while planning render stage",
            )),
        }
    }

    fn build_compute_expr(&mut self, expr: &Expr) -> Result<NodeHandle, ProgramBuildError> {
        if let Some(value) = evaluate_constant(expr) {
            return Ok(self.push_constant(value));
        }

        match expr {
            Expr::Constant(value) => Ok(self.push_constant(*value)),
            Expr::ComputeImport(name) => self.push_compute_import(*name),
            Expr::BinaryOp(op, left, right) => {
                let left_handle = self.build_compute_expr(left)?;
                let right_handle = self.build_compute_expr(right)?;
                let stage = left_handle.stage.combine(right_handle.stage);
                let id = self.push_binary(op, left_handle.id, right_handle.id, stage);
                Ok(NodeHandle { id, stage })
            }
            Expr::UnaryOp(func, input) => {
                let input_handle = self.build_compute_expr(input)?;
                let stage = input_handle.stage;
                let id = self.push_unary(func, input_handle.id, stage);
                Ok(NodeHandle { id, stage })
            }
            Expr::Vec4(Vec4 { .. })
            | Expr::Vec3(_)
            | Expr::Vec2(_) => Err(ProgramBuildError::UnsupportedExpression(
                "vector constructors cannot appear inside scalar compute expressions",
            )),
            Expr::RenderImport(name) => Err(ProgramBuildError::RenderImportInCompute { name: *name }),
            Expr::If { .. } => Err(ProgramBuildError::UnsupportedExpression(
                "conditional expressions are not supported in compute stage yet",
            )),
        }
    }

    fn push_constant(&mut self, value: f32) -> NodeHandle {
        let id = self.nodes.len() as u32;
        self.nodes.push(SerializableComputeNode {
            id,
            stage: ComputeStage::Precompute,
            kind: ComputeNodeKind::Constant { value },
        });
        NodeHandle { id, stage: ComputeStage::Precompute }
    }

    fn push_compute_import(
        &mut self,
        name: &'static str,
    ) -> Result<NodeHandle, ProgramBuildError> {
        let (import_type, mask) = self
            .registry
            .get_import_info(name)
            .ok_or(ProgramBuildError::MissingImport { name })?;

        match import_type {
            ImportType::Compute(_) => {
                let id = self.nodes.len() as u32;
                self.nodes.push(SerializableComputeNode {
                    id,
                    stage: ComputeStage::PerFrame,
                    kind: ComputeNodeKind::Import {
                        name,
                        mask: mask as u8,
                        source: ImportSource::Compute,
                    },
                });
                Ok(NodeHandle {
                    id,
                    stage: ComputeStage::PerFrame,
                })
            }
            ImportType::Render(_) => Err(ProgramBuildError::RenderImportInCompute { name }),
        }
    }

    fn push_binary(
        &mut self,
        op: &BinaryOp,
        left: u32,
        right: u32,
        stage: ComputeStage,
    ) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(SerializableComputeNode {
            id,
            stage,
            kind: ComputeNodeKind::Binary {
                op: SerializableBinaryOp::from(op),
                left,
                right,
            },
        });
        id
    }

    fn push_unary(
        &mut self,
        func: &UnaryFunc,
        input: u32,
        stage: ComputeStage,
    ) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(SerializableComputeNode {
            id,
            stage,
            kind: ComputeNodeKind::Unary {
                op: SerializableUnaryOp::from(func),
                input,
            },
        });
        id
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SerializableComputeNode {
    pub id: u32,
    pub stage: ComputeStage,
    pub kind: ComputeNodeKind,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ComputeNodeKind {
    Constant { value: f32 },
    Import { name: &'static str, mask: u8, source: ImportSource },
    Unary { op: SerializableUnaryOp, input: u32 },
    Binary { op: SerializableBinaryOp, left: u32, right: u32 },
}

#[derive(Debug, Clone, Serialize)]
pub enum ImportSource {
    Compute,
    Render,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum ComputeStage {
    Precompute,
    PerFrame,
}

impl ComputeStage {
    pub fn combine(self, other: Self) -> Self {
        if matches!(self, ComputeStage::PerFrame) || matches!(other, ComputeStage::PerFrame) {
            ComputeStage::PerFrame
        } else {
            ComputeStage::Precompute
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub enum SerializableBinaryOp {
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

impl SerializableBinaryOp {
    pub fn to_gpu_op(&self) -> GpuOp {
        match self {
            SerializableBinaryOp::Add => GpuOp::Add,
            SerializableBinaryOp::Subtract => GpuOp::Subtract,
            SerializableBinaryOp::Multiply => GpuOp::Multiply,
            SerializableBinaryOp::Divide => GpuOp::Divide,
            SerializableBinaryOp::Modulo => GpuOp::Modulo,
            SerializableBinaryOp::Pow => GpuOp::Pow,
            SerializableBinaryOp::GreaterThan => GpuOp::GreaterThan,
            SerializableBinaryOp::GreaterEqual => GpuOp::GreaterEqual,
            SerializableBinaryOp::LessThan => GpuOp::LessThan,
            SerializableBinaryOp::LessEqual => GpuOp::LessEqual,
            SerializableBinaryOp::Equal => GpuOp::Equal,
            SerializableBinaryOp::NotEqual => GpuOp::NotEqual,
            SerializableBinaryOp::Index => GpuOp::Index,
        }
    }
}

impl From<&BinaryOp> for SerializableBinaryOp {
    fn from(value: &BinaryOp) -> Self {
        match value {
            BinaryOp::Add => SerializableBinaryOp::Add,
            BinaryOp::Subtract => SerializableBinaryOp::Subtract,
            BinaryOp::Multiply => SerializableBinaryOp::Multiply,
            BinaryOp::Divide => SerializableBinaryOp::Divide,
            BinaryOp::Modulo => SerializableBinaryOp::Modulo,
            BinaryOp::Pow => SerializableBinaryOp::Pow,
            BinaryOp::GreaterThan => SerializableBinaryOp::GreaterThan,
            BinaryOp::GreaterEqual => SerializableBinaryOp::GreaterEqual,
            BinaryOp::LessThan => SerializableBinaryOp::LessThan,
            BinaryOp::LessEqual => SerializableBinaryOp::LessEqual,
            BinaryOp::Equal => SerializableBinaryOp::Equal,
            BinaryOp::NotEqual => SerializableBinaryOp::NotEqual,
            BinaryOp::Index => SerializableBinaryOp::Index,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub enum SerializableUnaryOp {
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
    Abs,
}

impl SerializableUnaryOp {
    pub fn to_gpu_op(&self) -> GpuOp {
        match self {
            SerializableUnaryOp::Sin => GpuOp::Sin,
            SerializableUnaryOp::Cos => GpuOp::Cos,
            SerializableUnaryOp::Tan => GpuOp::Tan,
            SerializableUnaryOp::Exp => GpuOp::Exp,
            SerializableUnaryOp::Log => GpuOp::Log,
            SerializableUnaryOp::Sqrt => GpuOp::Sqrt,
            SerializableUnaryOp::Abs => GpuOp::Abs,
        }
    }
}

impl From<&UnaryFunc> for SerializableUnaryOp {
    fn from(value: &UnaryFunc) -> Self {
        match value {
            UnaryFunc::Sin => SerializableUnaryOp::Sin,
            UnaryFunc::Cos => SerializableUnaryOp::Cos,
            UnaryFunc::Tan => SerializableUnaryOp::Tan,
            UnaryFunc::Exp => SerializableUnaryOp::Exp,
            UnaryFunc::Log => SerializableUnaryOp::Log,
            UnaryFunc::Sqrt => SerializableUnaryOp::Sqrt,
            UnaryFunc::Abs => SerializableUnaryOp::Abs,
        }
    }
}

struct NodeHandle {
    id: u32,
    stage: ComputeStage,
}

fn contains_render_import(expr: &Expr) -> bool {
    match expr {
        Expr::RenderImport(_) => true,
        Expr::ComputeImport(_) | Expr::Constant(_) => false,
        Expr::BinaryOp(_, left, right) => {
            contains_render_import(left) || contains_render_import(right)
        }
        Expr::UnaryOp(_, inner) => contains_render_import(inner),
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            contains_render_import(condition)
                || contains_render_import(then_branch)
                || contains_render_import(else_branch)
        }
        Expr::Vec2(vec) => contains_render_import(&vec.x) || contains_render_import(&vec.y),
        Expr::Vec3(vec) => {
            contains_render_import(&vec.x)
                || contains_render_import(&vec.y)
                || contains_render_import(&vec.z)
        }
        Expr::Vec4(vec) => {
            contains_render_import(&vec.x)
                || contains_render_import(&vec.y)
                || contains_render_import(&vec.z)
                || contains_render_import(&vec.w)
        }
    }
}

fn evaluate_constant(expr: &Expr) -> Option<f32> {
    match expr {
        Expr::Constant(value) => Some(*value),
        Expr::BinaryOp(op, left, right) => {
            let left = evaluate_constant(left)?;
            let right = evaluate_constant(right)?;
            Some(apply_binary_constant(op, left, right)?)
        }
        Expr::UnaryOp(func, input) => {
            let value = evaluate_constant(input)?;
            Some(apply_unary_constant(func, value)?)
        }
        _ => None,
    }
}

fn apply_binary_constant(op: &BinaryOp, left: f32, right: f32) -> Option<f32> {
    match op {
        BinaryOp::Add => Some(left + right),
        BinaryOp::Subtract => Some(left - right),
        BinaryOp::Multiply => Some(left * right),
        BinaryOp::Divide => Some(if right != 0.0 { left / right } else { 0.0 }),
        BinaryOp::Modulo => Some(if right != 0.0 { left % right } else { 0.0 }),
        BinaryOp::Pow => Some(left.powf(right)),
        BinaryOp::GreaterThan => Some(if left > right { 1.0 } else { 0.0 }),
        BinaryOp::GreaterEqual => Some(if left >= right { 1.0 } else { 0.0 }),
        BinaryOp::LessThan => Some(if left < right { 1.0 } else { 0.0 }),
        BinaryOp::LessEqual => Some(if left <= right { 1.0 } else { 0.0 }),
        BinaryOp::Equal => Some(if (left - right).abs() < f32::EPSILON {
            1.0
        } else {
            0.0
        }),
        BinaryOp::NotEqual => Some(if (left - right).abs() >= f32::EPSILON {
            1.0
        } else {
            0.0
        }),
        BinaryOp::Index => None, // 暂不支持常量索引折叠
    }
}

fn apply_unary_constant(func: &UnaryFunc, value: f32) -> Option<f32> {
    match func {
        UnaryFunc::Sin => Some(value.sin()),
        UnaryFunc::Cos => Some(value.cos()),
        UnaryFunc::Tan => Some(value.tan()),
        UnaryFunc::Exp => Some(value.exp()),
        UnaryFunc::Log => {
            if value > 0.0 {
                Some(value.ln())
            } else {
                None
            }
        }
        UnaryFunc::Sqrt => {
            if value >= 0.0 {
                Some(value.sqrt())
            } else {
                None
            }
        }
        UnaryFunc::Abs => Some(value.abs()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::dsl::wvec4, dsl::cv};

    fn create_test_registry() -> ImportRegistry {
        let mut registry = ImportRegistry::new();
        registry.register_compute_import(
            "time",
            0b0011,
            Box::new(|input| vec![input[0], 0.0, 0.0, 0.0]),
        );
        registry
    }

    #[test]
    fn build_simple_program() {
        let expr = wvec4(
            1.0,
            cv("time") * 3.14,
            1.0,
            1.0,
        );
        let registry = create_test_registry();
        let program = GpuProgramBuilder::new(&registry)
            .build_program(&expr)
            .expect("builder should succeed");

        assert_eq!(program.render_plan.components.len(), 4);
        match &program.render_plan.components[0] {
            RenderComponent::Constant { value } => assert!((*value - 1.0).abs() < f32::EPSILON),
            _ => panic!("expected constant component"),
        }

        match &program.render_plan.components[1] {
            RenderComponent::ComputeNode { node_id } => {
                assert!(*node_id < program.compute_nodes.len() as u32)
            }
            other => panic!("unexpected component: {:?}", other),
        }

        assert_eq!(program.compute_nodes.len(), 3);
        // import node should be marked per-frame
        let import_node = &program.compute_nodes[0];
        println!("import_node {:?}",import_node);
        assert!(matches!(
            import_node.kind,
            ComputeNodeKind::Import {
                name: "time",
                source: ImportSource::Compute,
                ..
            }
        ));
        assert_eq!(import_node.stage, ComputeStage::PerFrame);
    }

    #[test]
    fn serialize_to_json() {
        let expr = wvec4(
            1.0,
            cv("time") * 2.0,
            1.0,
            1.0,
        );
        let registry = create_test_registry();
        let program = GpuProgramBuilder::new(&registry)
            .build_program(&expr)
            .unwrap();

        let json = serde_json::to_string_pretty(&program).unwrap();
        println!("具体的情况{:?}",json);
        assert!(json.contains("\"compute_nodes\""));
        assert!(json.contains("\"render_plan\""));
    }
}
