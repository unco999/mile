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
    pub render_expr_nodes: Vec<RenderExprNode>,
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
    RenderImport { name: &'static str, mask: u32, component: u32 },
    RenderComposite {
        name: &'static str,
        mask: u32,
        component: u32,
        factor_render_component: Option<u32>,
        factor_render_scale: f32,
        factor_inner_constant: f32,
        factor_inner_compute: Option<u32>,
        factor_outer_constant: f32,
        factor_outer_compute: Option<u32>,
        factor_unary: FactorUnary,
        offset: f32,
    },
    RenderExpression {
        expr_start: u32,
        expr_len: u32,
    },
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum FactorUnary {
    None,
    Sin,
    Cos,
}

impl Default for FactorUnary {
    fn default() -> Self {
        FactorUnary::None
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum RenderExprOp {
    Constant,
    RenderImport,
    ComputeResult,
    Unary(SerializableUnaryOp),
    Binary(SerializableBinaryOp),
    Negate,
    If,
}

#[derive(Debug, Clone, Serialize)]
pub struct RenderExprNode {
    pub op: RenderExprOp,
    pub arg0: u32,
    pub arg1: u32,
    pub arg2: u32,
    pub data0: f32,
    pub data1: f32,
}

const MAX_RENDER_EXPR_NODES_PER_COMPONENT: usize = 64;

impl FactorUnary {
    fn from_unary(func: &UnaryFunc) -> Option<Self> {
        match func {
            UnaryFunc::Sin => Some(FactorUnary::Sin),
            UnaryFunc::Cos => Some(FactorUnary::Cos),
            _ => None,
        }
    }

    pub fn to_u32(self) -> u32 {
        match self {
            FactorUnary::None => 0,
            FactorUnary::Sin => 1,
            FactorUnary::Cos => 2,
        }
    }
}

/// 构建 GPU 计算节点的构建器
pub struct GpuProgramBuilder<'a> {
    registry: &'a ImportRegistry,
    nodes: Vec<SerializableComputeNode>,
    render_expr_nodes: Vec<RenderExprNode>,
}

struct FactorDetails {
    render_component: Option<u32>,
    render_scale: f32,
    inner_constant: f32,
    inner_compute_expr: Option<Expr>,
    outer_constant: f32,
    outer_compute_expr: Option<Expr>,
    unary: FactorUnary,
}

struct LinearRenderExpr {
    component: u32,
    scale: f32,
    constant: f32,
    compute_expr: Option<Expr>,
}

impl<'a> GpuProgramBuilder<'a> {
    pub fn new(registry: &'a ImportRegistry) -> Self {
        Self {
            registry,
            nodes: Vec::new(),
            render_expr_nodes: Vec::new(),
        }
    }

    pub fn build_program(mut self, expr: &Expr) -> Result<SerializableGpuProgram, ProgramBuildError> {
        let vector = match expr {
            Expr::Vec4(vec4) => vec4,
            _ => return Err(ProgramBuildError::RootExpressionMustBeVec4),
        };

        let components = [
            self.build_component(&vector.x, 0)?,
            self.build_component(&vector.y, 1)?,
            self.build_component(&vector.z, 2)?,
            self.build_component(&vector.w, 3)?,
        ];

        Ok(SerializableGpuProgram {
            compute_nodes: self.nodes,
            render_plan: RenderPlan::new(components),
            render_expr_nodes: self.render_expr_nodes,
        })
    }

    fn build_component(&mut self, expr: &Expr, lane: u32) -> Result<RenderComponent, ProgramBuildError> {
        if let Some(value) = evaluate_constant(expr) {
            return Ok(RenderComponent::Constant { value });
        }

        if let Some(render_component) = self.resolve_render_import(expr, lane)? {
            return Ok(render_component);
        }

        if let Some(render_component) = self.try_build_render_composite(expr, lane)? {
            return Ok(render_component);
        }

        if contains_render_import(expr) {
            if let Some(render_component) = self.try_build_render_expression(expr, lane)? {
                return Ok(render_component);
            }
            return match expr {
                Expr::RenderImport(name) => self.plan_render_import(name, 0),
                _ => Err(ProgramBuildError::RenderStageExpression(
                    "render stage arithmetic is not supported yet",
                )),
            };
        }

        let node = self.build_compute_expr(expr)?;
        Ok(RenderComponent::ComputeNode { node_id: node.id })
    }

    fn resolve_render_import(
        &self,
        expr: &Expr,
        default_lane: u32,
    ) -> Result<Option<RenderComponent>, ProgramBuildError> {
        if let Some((name, component)) = self.detect_render_import(expr) {
            let component = component.unwrap_or(default_lane);
            return self.plan_render_import(name, component).map(Some);
        }
        Ok(None)
    }

    fn detect_render_import(&self, expr: &Expr) -> Option<(&'static str, Option<u32>)> {
        match expr {
            Expr::RenderImport(name) => Some((*name, None)),
            Expr::BinaryOp(BinaryOp::Index, left, right) => {
                let (name, _) = self.detect_render_import(left)?;
                let idx = evaluate_index(right.as_ref())?;
                Some((name, Some(idx)))
            }
            _ => None,
        }
    }

    fn detect_render_factor(
        &self,
        expr: &'a Expr,
        default_lane: u32,
    ) -> Option<(&'static str, u32, Option<&'a Expr>)> {
        if let Some((name, component)) = self.detect_render_import(expr) {
            return Some((name, component.unwrap_or(default_lane), None));
        }

        if let Expr::BinaryOp(BinaryOp::Multiply, left, right) = expr {
            if let Some((name, component)) = self.detect_render_import(left) {
                return Some((
                    name,
                    component.unwrap_or(default_lane),
                    Some(right.as_ref()),
                ));
            }
            if let Some((name, component)) = self.detect_render_import(right) {
                return Some((
                    name,
                    component.unwrap_or(default_lane),
                    Some(left.as_ref()),
                ));
            }
        }

        None
    }

    fn try_build_render_composite(
        &mut self,
        expr: &Expr,
        lane: u32,
    ) -> Result<Option<RenderComponent>, ProgramBuildError> {
        let (core_expr, offset) = strip_constant_offset(expr);

        if let Some((name, component, factor_expr)) =
            self.detect_render_factor(core_expr, lane)
        {
            let mut factor_render_component = None;
            let mut factor_render_scale = 0.0f32;
            let mut factor_inner_constant = 1.0f32;
            let mut factor_inner_compute = None;
            let mut factor_outer_constant = 1.0f32;
            let mut factor_outer_compute = None;
            let mut factor_unary = FactorUnary::None;

            if let Some(factor_expr) = factor_expr {
                let factor_parts = self.extract_factor_terms(name, component, factor_expr)?;

                factor_render_component = factor_parts.render_component;
                factor_render_scale = factor_parts.render_scale;
                factor_inner_constant = factor_parts.inner_constant;
                factor_outer_constant = factor_parts.outer_constant;
                factor_unary = factor_parts.unary;

                if let Some(expr) = factor_parts.inner_compute_expr {
                    let node = self.build_compute_expr(&expr)?;
                    factor_inner_compute = Some(node.id);
                }

                if let Some(expr) = factor_parts.outer_compute_expr {
                    let node = self.build_compute_expr(&expr)?;
                    factor_outer_compute = Some(node.id);
                }
            }

            return self
                .build_render_composite(
                    name,
                    component,
                    factor_render_component,
                    factor_render_scale,
                    factor_inner_constant,
                    factor_inner_compute,
                    factor_outer_constant,
                    factor_outer_compute,
                    factor_unary,
                    offset,
                )
                .map(Some);
        }

        Ok(None)
    }

    fn try_build_render_expression(
        &mut self,
        expr: &Expr,
        lane: u32,
    ) -> Result<Option<RenderComponent>, ProgramBuildError> {
        let start_len = self.render_expr_nodes.len();
        let root = match self.build_render_expr(expr, lane) {
            Ok(index) => index,
            Err(err) => {
                self.render_expr_nodes.truncate(start_len);
                return Err(err);
            }
        };

        let expr_len = self.render_expr_nodes.len() - start_len;
        if expr_len == 0 {
            return Ok(None);
        }

        if expr_len > MAX_RENDER_EXPR_NODES_PER_COMPONENT {
            self.render_expr_nodes.truncate(start_len);
            return Err(ProgramBuildError::RenderStageExpression(
                "render expression exceeds supported complexity",
            ));
        }

        debug_assert_eq!(root as usize, self.render_expr_nodes.len() - 1);

        Ok(Some(RenderComponent::RenderExpression {
            expr_start: start_len as u32,
            expr_len: expr_len as u32,
        }))
    }

    fn build_render_expr(&mut self, expr: &Expr, lane: u32) -> Result<u32, ProgramBuildError> {
        if !contains_render_import(expr) {
            if let Some(value) = evaluate_constant(expr) {
                return Ok(self.push_render_expr_node(RenderExprNode {
                    op: RenderExprOp::Constant,
                    arg0: 0,
                    arg1: 0,
                    arg2: 0,
                    data0: value,
                    data1: 0.0,
                }));
            }

            let node = self.build_compute_expr(expr)?;
            return Ok(self.push_render_expr_node(RenderExprNode {
                op: RenderExprOp::ComputeResult,
                arg0: node.id,
                arg1: 0,
                arg2: 0,
                data0: 0.0,
                data1: 0.0,
            }));
        }

        match expr {
            Expr::Constant(value) => Ok(self.push_render_expr_node(RenderExprNode {
                op: RenderExprOp::Constant,
                arg0: 0,
                arg1: 0,
                arg2: 0,
                data0: *value,
                data1: 0.0,
            })),
            Expr::RenderImport(name) => {
                let (import_type, mask) = self
                    .registry
                    .get_import_info(*name)
                    .ok_or(ProgramBuildError::MissingImport { name: *name })?;
                match import_type {
                    ImportType::Render(_) => Ok(self.push_render_expr_node(RenderExprNode {
                        op: RenderExprOp::RenderImport,
                        arg0: mask,
                        arg1: lane,
                        arg2: 0,
                        data0: 0.0,
                        data1: 0.0,
                    })),
                    ImportType::Compute(_) => Err(ProgramBuildError::RenderStageExpression(
                        "compute import encountered while building render expression",
                    )),
                }
            }
            Expr::BinaryOp(BinaryOp::Index, left, right) => {
                let component = evaluate_index(right.as_ref()).ok_or_else(|| {
                    ProgramBuildError::RenderStageExpression("dynamic render index not supported")
                })?;
                self.build_render_expr(left, component)
            }
            Expr::Vec4(vec) => match lane {
                0 => self.build_render_expr(&vec.x, 0),
                1 => self.build_render_expr(&vec.y, 0),
                2 => self.build_render_expr(&vec.z, 0),
                3 => self.build_render_expr(&vec.w, 0),
                _ => Err(ProgramBuildError::RenderStageExpression(
                    "component out of range in render expression",
                )),
            },
            Expr::Vec3(vec) => match lane {
                0 => self.build_render_expr(&vec.x, 0),
                1 => self.build_render_expr(&vec.y, 0),
                2 => self.build_render_expr(&vec.z, 0),
                _ => Err(ProgramBuildError::RenderStageExpression(
                    "component out of range in render expression",
                )),
            },
            Expr::Vec2(vec) => match lane {
                0 => self.build_render_expr(&vec.x, 0),
                1 => self.build_render_expr(&vec.y, 0),
                _ => Err(ProgramBuildError::RenderStageExpression(
                    "component out of range in render expression",
                )),
            },
            Expr::ComputeImport(name) => {
                let node = self.push_compute_import(*name)?;
                Ok(self.push_render_expr_node(RenderExprNode {
                    op: RenderExprOp::ComputeResult,
                    arg0: node.id,
                    arg1: 0,
                    arg2: 0,
                    data0: 0.0,
                    data1: 0.0,
                }))
            }
            Expr::UnaryOp(func, inner) => {
                let child = self.build_render_expr(inner, lane)?;
                let op = RenderExprOp::Unary(match func {
                    UnaryFunc::Sin => SerializableUnaryOp::Sin,
                    UnaryFunc::Cos => SerializableUnaryOp::Cos,
                    UnaryFunc::Tan => SerializableUnaryOp::Tan,
                    UnaryFunc::Exp => SerializableUnaryOp::Exp,
                    UnaryFunc::Log => SerializableUnaryOp::Log,
                    UnaryFunc::Sqrt => SerializableUnaryOp::Sqrt,
                    UnaryFunc::Abs => SerializableUnaryOp::Abs,
                });
                Ok(self.push_render_expr_node(RenderExprNode {
                    op,
                    arg0: child,
                    arg1: 0,
                    arg2: 0,
                    data0: 0.0,
                    data1: 0.0,
                }))
            }
            Expr::BinaryOp(op, left, right) => {
                if matches!(op, BinaryOp::Index) {
                    unreachable!("index handled earlier");
                }

                let left_contains = contains_render_import(left);
                let right_contains = contains_render_import(right);

                let left_idx = if left_contains {
                    self.build_render_expr(left, lane)?
                } else if let Some(value) = evaluate_constant(left) {
                    self.push_render_expr_node(RenderExprNode {
                        op: RenderExprOp::Constant,
                        arg0: 0,
                        arg1: 0,
                        arg2: 0,
                        data0: value,
                        data1: 0.0,
                    })
                } else {
                    let node = self.build_compute_expr(left)?;
                    self.push_render_expr_node(RenderExprNode {
                        op: RenderExprOp::ComputeResult,
                        arg0: node.id,
                        arg1: 0,
                        arg2: 0,
                        data0: 0.0,
                        data1: 0.0,
                    })
                };

                let right_idx = if right_contains {
                    self.build_render_expr(right, lane)?
                } else if let Some(value) = evaluate_constant(right) {
                    self.push_render_expr_node(RenderExprNode {
                        op: RenderExprOp::Constant,
                        arg0: 0,
                        arg1: 0,
                        arg2: 0,
                        data0: value,
                        data1: 0.0,
                    })
                } else {
                    let node = self.build_compute_expr(right)?;
                    self.push_render_expr_node(RenderExprNode {
                        op: RenderExprOp::ComputeResult,
                        arg0: node.id,
                        arg1: 0,
                        arg2: 0,
                        data0: 0.0,
                        data1: 0.0,
                    })
                };

                let expr_op = match op {
                    BinaryOp::Add => RenderExprOp::Binary(SerializableBinaryOp::Add),
                    BinaryOp::Subtract => RenderExprOp::Binary(SerializableBinaryOp::Subtract),
                    BinaryOp::Multiply => RenderExprOp::Binary(SerializableBinaryOp::Multiply),
                    BinaryOp::Divide => RenderExprOp::Binary(SerializableBinaryOp::Divide),
                    BinaryOp::Modulo => RenderExprOp::Binary(SerializableBinaryOp::Modulo),
                    BinaryOp::Pow => RenderExprOp::Binary(SerializableBinaryOp::Pow),
                    BinaryOp::GreaterThan => RenderExprOp::Binary(SerializableBinaryOp::GreaterThan),
                    BinaryOp::GreaterEqual => RenderExprOp::Binary(SerializableBinaryOp::GreaterEqual),
                    BinaryOp::LessThan => RenderExprOp::Binary(SerializableBinaryOp::LessThan),
                    BinaryOp::LessEqual => RenderExprOp::Binary(SerializableBinaryOp::LessEqual),
                    BinaryOp::Equal => RenderExprOp::Binary(SerializableBinaryOp::Equal),
                    BinaryOp::NotEqual => RenderExprOp::Binary(SerializableBinaryOp::NotEqual),
                    BinaryOp::Index => unreachable!(),
                };

                Ok(self.push_render_expr_node(RenderExprNode {
                    op: expr_op,
                    arg0: left_idx,
                    arg1: right_idx,
                    arg2: 0,
                    data0: 0.0,
                    data1: 0.0,
                }))
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_idx = self.build_render_expr(condition, lane)?;
                let then_idx = self.build_render_expr(then_branch, lane)?;
                let else_idx = self.build_render_expr(else_branch, lane)?;
                Ok(self.push_render_expr_node(RenderExprNode {
                    op: RenderExprOp::If,
                    arg0: cond_idx,
                    arg1: then_idx,
                    arg2: else_idx,
                    data0: 0.0,
                    data1: 0.0,
                }))
            }
            _ => Err(ProgramBuildError::RenderStageExpression(
                "render expression contains unsupported construct",
            )),
        }
    }

    fn push_render_expr_node(&mut self, node: RenderExprNode) -> u32 {
        let index = self.render_expr_nodes.len() as u32;
        self.render_expr_nodes.push(node);
        index
    }


    fn extract_factor_terms(
        &self,
        name: &'static str,
        base_component: u32,
        expr: &Expr,
    ) -> Result<FactorDetails, ProgramBuildError> {
        if !contains_render_import(expr) {
            if let Some(value) = evaluate_constant(expr) {
                return Ok(FactorDetails {
                    render_component: None,
                    render_scale: 0.0,
                    inner_constant: value,
                    inner_compute_expr: None,
                    outer_constant: 1.0,
                    outer_compute_expr: None,
                    unary: FactorUnary::None,
                });
            }

            return Ok(FactorDetails {
                render_component: None,
                render_scale: 0.0,
                inner_constant: 0.0,
                inner_compute_expr: Some(expr.clone()),
                outer_constant: 1.0,
                outer_compute_expr: None,
                unary: FactorUnary::None,
            });
        }

        if contains_render_import_with_different_name(expr, name) {
            return Err(ProgramBuildError::RenderStageExpression(
                "render factor expression references multiple render inputs",
            ));
        }

        let (core_expr, outer_constant, outer_compute_expr) =
            peel_outer_multiply(expr.clone(), name, base_component)?;

        let (unary, inner_expr) = match core_expr {
            Expr::UnaryOp(func, inner) => {
                let factor = FactorUnary::from_unary(&func).ok_or(
                    ProgramBuildError::RenderStageExpression(
                        "render factor uses unsupported unary function",
                    ),
                )?;
                (factor, inner.as_ref().clone())
            }
            other => (FactorUnary::None, other),
        };

        let linear = extract_linear_render(&inner_expr, name, base_component).ok_or(
            ProgramBuildError::RenderStageExpression(
                "render factor expression is too complex for render pipeline",
            ),
        )?;

        let mut outer_constant = outer_constant;
        let mut outer_compute_expr = outer_compute_expr;
        if let Some(expr) = outer_compute_expr.take() {
            if let Some(value) = evaluate_constant(&expr) {
                outer_constant *= value;
            } else {
                outer_compute_expr = Some(expr);
            }
        }

        Ok(FactorDetails {
            render_component: Some(linear.component),
            render_scale: linear.scale,
            inner_constant: linear.constant,
            inner_compute_expr: linear.compute_expr,
            outer_constant,
            outer_compute_expr,
            unary,
        })
    }

    fn extract_compute_term(
        &mut self,
        expr: &Expr,
    ) -> Result<(Option<u32>, f32), ProgramBuildError> {
        if let Some(value) = evaluate_constant(expr) {
            return Ok((None, value));
        }
        if contains_render_import(expr) {
            return Err(ProgramBuildError::RenderStageExpression(
                "render expression is too complex for render pipeline"
            ));
        }
        let node = self.build_compute_expr(expr)?;
        Ok((Some(node.id), 0.0))
    }

    fn build_render_composite(
        &self,
        name: &'static str,
        component: u32,
        factor_render_component: Option<u32>,
        factor_render_scale: f32,
        factor_inner_constant: f32,
        factor_inner_compute: Option<u32>,
        factor_outer_constant: f32,
        factor_outer_compute: Option<u32>,
        factor_unary: FactorUnary,
        offset: f32,
    ) -> Result<RenderComponent, ProgramBuildError> {
        let (import_type, mask) = self
            .registry
            .get_import_info(name)
            .ok_or(ProgramBuildError::MissingImport { name })?;

        match import_type {
            ImportType::Render(_) => Ok(RenderComponent::RenderComposite {
                name,
                mask,
                component,
                factor_render_component,
                factor_render_scale,
                factor_inner_constant,
                factor_inner_compute,
                factor_outer_constant,
                factor_outer_compute,
                factor_unary,
                offset,
            }),
            ImportType::Compute(_) => Err(ProgramBuildError::RenderStageExpression(
                "compute import encountered while planning render stage",
            )),
        }
    }

    fn plan_render_import(
        &self,
        name: &'static str,
        component: u32,
    ) -> Result<RenderComponent, ProgramBuildError> {
        let (import_type, mask) = self
            .registry
            .get_import_info(name)
            .ok_or(ProgramBuildError::MissingImport { name })?;

        match import_type {
            ImportType::Render(_) => Ok(RenderComponent::RenderImport {
                name,
                mask,
                component,
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

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
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

fn contains_render_import_with_different_name(expr: &Expr, expected: &'static str) -> bool {
    match expr {
        Expr::RenderImport(name) => *name != expected,
        Expr::ComputeImport(_) | Expr::Constant(_) => false,
        Expr::BinaryOp(BinaryOp::Index, left, _) => {
            contains_render_import_with_different_name(left, expected)
        }
        Expr::BinaryOp(_, left, right) => {
            contains_render_import_with_different_name(left, expected)
                || contains_render_import_with_different_name(right, expected)
        }
        Expr::UnaryOp(_, inner) => contains_render_import_with_different_name(inner, expected),
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            contains_render_import_with_different_name(condition, expected)
                || contains_render_import_with_different_name(then_branch, expected)
                || contains_render_import_with_different_name(else_branch, expected)
        }
        Expr::Vec2(vec) => {
            contains_render_import_with_different_name(&vec.x, expected)
                || contains_render_import_with_different_name(&vec.y, expected)
        }
        Expr::Vec3(vec) => {
            contains_render_import_with_different_name(&vec.x, expected)
                || contains_render_import_with_different_name(&vec.y, expected)
                || contains_render_import_with_different_name(&vec.z, expected)
        }
        Expr::Vec4(vec) => {
            contains_render_import_with_different_name(&vec.x, expected)
                || contains_render_import_with_different_name(&vec.y, expected)
                || contains_render_import_with_different_name(&vec.z, expected)
                || contains_render_import_with_different_name(&vec.w, expected)
        }
    }
}

fn contains_named_render_import(expr: &Expr, expected: &'static str) -> bool {
    match expr {
        Expr::RenderImport(name) => *name == expected,
        Expr::ComputeImport(_) | Expr::Constant(_) => false,
        Expr::BinaryOp(BinaryOp::Index, left, _) => contains_named_render_import(left, expected),
        Expr::BinaryOp(_, left, right) => {
            contains_named_render_import(left, expected)
                || contains_named_render_import(right, expected)
        }
        Expr::UnaryOp(_, inner) => contains_named_render_import(inner, expected),
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            contains_named_render_import(condition, expected)
                || contains_named_render_import(then_branch, expected)
                || contains_named_render_import(else_branch, expected)
        }
        Expr::Vec2(vec) => {
            contains_named_render_import(&vec.x, expected)
                || contains_named_render_import(&vec.y, expected)
        }
        Expr::Vec3(vec) => {
            contains_named_render_import(&vec.x, expected)
                || contains_named_render_import(&vec.y, expected)
                || contains_named_render_import(&vec.z, expected)
        }
        Expr::Vec4(vec) => {
            contains_named_render_import(&vec.x, expected)
                || contains_named_render_import(&vec.y, expected)
                || contains_named_render_import(&vec.z, expected)
                || contains_named_render_import(&vec.w, expected)
        }
    }
}

fn detect_render_import_component(
    expr: &Expr,
    default_component: u32,
) -> Option<(&'static str, u32)> {
    match expr {
        Expr::RenderImport(name) => Some((*name, default_component)),
        Expr::BinaryOp(BinaryOp::Index, left, right) => {
            let (name, _) = detect_render_import_component(left, default_component)?;
            let idx = evaluate_index(right.as_ref())?;
            Some((name, idx))
        }
        _ => None,
    }
}

fn peel_outer_multiply(
    expr: Expr,
    expected_name: &'static str,
    _default_component: u32,
) -> Result<(Expr, f32, Option<Expr>), ProgramBuildError> {
    let mut current = expr;
    let mut outer_constant = 1.0f32;
    let mut compute_terms: Vec<Expr> = Vec::new();

    loop {
        match current {
            Expr::BinaryOp(BinaryOp::Multiply, left, right) => {
                let left_contains = contains_named_render_import(&left, expected_name);
                let right_contains = contains_named_render_import(&right, expected_name);

                if left_contains && right_contains {
                    return Err(ProgramBuildError::RenderStageExpression(
                        "render factor expression is too complex for render pipeline",
                    ));
                }

                if left_contains {
                    if contains_render_import(&right) {
                        return Err(ProgramBuildError::RenderStageExpression(
                            "render factor expression is too complex for render pipeline",
                        ));
                    }
                    if let Some(value) = evaluate_constant(&right) {
                        outer_constant *= value;
                    } else {
                        compute_terms.push(*right);
                    }
                    current = *left;
                    continue;
                }

                if right_contains {
                    if contains_render_import(&left) {
                        return Err(ProgramBuildError::RenderStageExpression(
                            "render factor expression is too complex for render pipeline",
                        ));
                    }
                    if let Some(value) = evaluate_constant(&left) {
                        outer_constant *= value;
                    } else {
                        compute_terms.push(*left);
                    }
                    current = *right;
                    continue;
                }

                if contains_render_import(&left) || contains_render_import(&right) {
                    return Err(ProgramBuildError::RenderStageExpression(
                        "render factor expression is too complex for render pipeline",
                    ));
                }

                compute_terms.push(Expr::BinaryOp(
                    BinaryOp::Multiply,
                    Box::new(*left),
                    Box::new(*right),
                ));
                current = Expr::Constant(1.0);
                break;
            }
            _ => break,
        }
    }

    let compute_expr = combine_multiply_terms(compute_terms);
    Ok((current, outer_constant, compute_expr))
}

fn combine_multiply_terms(mut terms: Vec<Expr>) -> Option<Expr> {
    if terms.is_empty() {
        return None;
    }

    let mut expr = terms.remove(0);
    for term in terms {
        expr = Expr::BinaryOp(BinaryOp::Multiply, Box::new(expr), Box::new(term));
    }

    if let Some(value) = evaluate_constant(&expr) {
        Some(Expr::Constant(value))
    } else {
        Some(expr)
    }
}

fn extract_linear_render(
    expr: &Expr,
    expected_name: &'static str,
    default_component: u32,
) -> Option<LinearRenderExpr> {
    match expr {
        Expr::RenderImport(name) => {
            if *name == expected_name {
                Some(LinearRenderExpr {
                    component: default_component,
                    scale: 1.0,
                    constant: 0.0,
                    compute_expr: None,
                })
            } else {
                None
            }
        }
        Expr::BinaryOp(BinaryOp::Index, _, _) => {
            let (name, component) = detect_render_import_component(expr, default_component)?;
            if name == expected_name {
                Some(LinearRenderExpr {
                    component,
                    scale: 1.0,
                    constant: 0.0,
                    compute_expr: None,
                })
            } else {
                None
            }
        }
        Expr::BinaryOp(BinaryOp::Multiply, left, right) => {
            if let Some(value) = evaluate_constant(left) {
                let mut linear =
                    extract_linear_render(right, expected_name, default_component)?;
                scale_linear_expr(&mut linear, value);
                Some(linear)
            } else if let Some(value) = evaluate_constant(right) {
                let mut linear =
                    extract_linear_render(left, expected_name, default_component)?;
                scale_linear_expr(&mut linear, value);
                Some(linear)
            } else {
                None
            }
        }
        Expr::BinaryOp(BinaryOp::Add, left, right) => {
            if let Some(mut linear) =
                extract_linear_render(left, expected_name, default_component)
            {
                if contains_render_import(right) {
                    return None;
                }
                add_compute_term(&mut linear, *(*right).clone(), 1.0);
                return Some(linear);
            }

            if let Some(mut linear) =
                extract_linear_render(right, expected_name, default_component)
            {
                if contains_render_import(left) {
                    return None;
                }
                add_compute_term(&mut linear, *(*left).clone(), 1.0);
                return Some(linear);
            }

            None
        }
        Expr::BinaryOp(BinaryOp::Subtract, left, right) => {
            if let Some(mut linear) =
                extract_linear_render(left, expected_name, default_component)
            {
                if contains_render_import(right) {
                    return None;
                }
                add_compute_term(&mut linear, *(*right).clone(), -1.0);
                return Some(linear);
            }

            if let Some(mut linear) =
                extract_linear_render(right, expected_name, default_component)
            {
                if contains_render_import(left) {
                    return None;
                }
                scale_linear_expr(&mut linear, -1.0);
                add_compute_term(&mut linear, *(*left).clone(), 1.0);
                return Some(linear);
            }

            None
        }
        _ => None,
    }
}

fn add_compute_term(linear: &mut LinearRenderExpr, expr: Expr, sign: f32) {
    if let Some(value) = evaluate_constant(&expr) {
        linear.constant += value * sign;
    } else {
        let signed_expr = if (sign - 1.0).abs() < f32::EPSILON {
            expr
        } else if (sign + 1.0).abs() < f32::EPSILON {
            multiply_expr_by_constant(expr, -1.0)
        } else {
            multiply_expr_by_constant(expr, sign)
        };

        if let Some(existing) = &linear.compute_expr {
            let combined = Expr::BinaryOp(
                BinaryOp::Add,
                Box::new(existing.clone()),
                Box::new(signed_expr),
            );
            linear.compute_expr = Some(combined);
        } else {
            linear.compute_expr = Some(signed_expr);
        }

        normalize_linear_compute(linear);
    }
}

fn scale_linear_expr(linear: &mut LinearRenderExpr, factor: f32) {
    linear.scale *= factor;
    linear.constant *= factor;
    if let Some(expr) = linear.compute_expr.take() {
        let scaled = multiply_expr_by_constant(expr, factor);
        if let Some(value) = evaluate_constant(&scaled) {
            linear.constant += value;
            linear.compute_expr = None;
        } else {
            linear.compute_expr = Some(scaled);
        }
    }
}

fn normalize_linear_compute(linear: &mut LinearRenderExpr) {
    if let Some(expr) = &linear.compute_expr {
        if let Some(value) = evaluate_constant(expr) {
            linear.constant += value;
            linear.compute_expr = None;
        }
    }
}

fn multiply_expr_by_constant(expr: Expr, scale: f32) -> Expr {
    if (scale - 1.0).abs() < f32::EPSILON {
        expr
    } else if scale.abs() < f32::EPSILON {
        Expr::Constant(0.0)
    } else {
        match expr {
            Expr::Constant(value) => Expr::Constant(value * scale),
            other => Expr::BinaryOp(
                BinaryOp::Multiply,
                Box::new(Expr::Constant(scale)),
                Box::new(other),
            ),
        }
    }
}

fn strip_constant_offset<'a>(expr: &'a Expr) -> (&'a Expr, f32) {
    let mut current = expr;
    let mut offset = 0.0;

    loop {
        match current {
            Expr::BinaryOp(BinaryOp::Add, left, right) => {
                if let Some(value) = evaluate_constant(left) {
                    offset += value;
                    current = right;
                    continue;
                }
                if let Some(value) = evaluate_constant(right) {
                    offset += value;
                    current = left;
                    continue;
                }
            }
            Expr::BinaryOp(BinaryOp::Subtract, left, right) => {
                if let Some(value) = evaluate_constant(right) {
                    offset -= value;
                    current = left;
                    continue;
                }
            }
            _ => {}
        }
        break;
    }

    (current, offset)
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

fn evaluate_index(expr: &Expr) -> Option<u32> {
    let value = evaluate_constant(expr)?;
    let clamped = value.clamp(0.0, 3.0);
    Some(clamped.round() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::dsl::{wvec4, sin}, dsl::{cv, rv}};

    fn create_test_registry() -> ImportRegistry {
        let mut registry = ImportRegistry::new();
        registry.register_compute_import(
            "time",
            0b0011,
            Box::new(|input| vec![input[0], 0.0, 0.0, 0.0]),
        );
        registry.register_render_import(
            "uv",
            0b0011,
            Box::new(|input| vec![input[0], input[1], 0.0, 0.0]),
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

    #[test]
    fn render_compute_mix_produces_composite() {
        let expr = wvec4(
            rv("uv").x() * (cv("time") * 0.5),
            0.0,
            0.0,
            0.0,
        );

        let registry = create_test_registry();
        let program = GpuProgramBuilder::new(&registry)
            .build_program(&expr)
            .expect("builder should support render/compute mixing");

        match &program.render_plan.components[0] {
            RenderComponent::RenderComposite {
                name,
                component,
                factor_render_component,
                factor_render_scale,
                factor_inner_constant,
                factor_inner_compute,
                factor_outer_constant,
                factor_outer_compute,
                factor_unary,
                offset,
                ..
            } => {
                assert_eq!(*name, "uv");
                assert_eq!(*component, 0);
                assert!(factor_render_component.is_none());
                assert!((*factor_render_scale - 0.0).abs() < f32::EPSILON);
                assert!((*factor_inner_constant - 0.0).abs() < f32::EPSILON);
                assert!(factor_inner_compute.is_some());
                assert!((*factor_outer_constant - 1.0).abs() < f32::EPSILON);
                assert!(factor_outer_compute.is_none());
                assert!(matches!(factor_unary, FactorUnary::None));
                assert_eq!(*offset, 0.0);
            }
            other => panic!("unexpected component: {:?}", other),
        }

        assert!(
            !program.compute_nodes.is_empty(),
            "expect compute nodes for factor expression"
        );
    }

    #[test]
    fn render_factor_can_reference_same_import() {
        let expr = wvec4(
            rv("uv").x() * (rv("uv").y() * cv("time")),
            0.0,
            0.0,
            0.0,
        );

        let registry = create_test_registry();
        let program = GpuProgramBuilder::new(&registry)
            .build_program(&expr)
            .expect("builder should support render factors involving render data");

        match &program.render_plan.components[0] {
            RenderComponent::RenderComposite {
                component,
                factor_render_component,
                factor_inner_compute,
                factor_outer_compute,
                factor_render_scale,
                factor_inner_constant,
                factor_unary,
                ..
            } => {
                assert_eq!(*component, 0);
                assert_eq!(*factor_render_component, Some(1));
                assert!((*factor_render_scale - 1.0).abs() < f32::EPSILON);
                assert!((*factor_inner_constant - 0.0).abs() < f32::EPSILON);
                assert!(factor_inner_compute.is_none());
                assert!(factor_outer_compute.is_some());
                assert!(matches!(factor_unary, FactorUnary::None));
            }
            other => panic!("unexpected component: {:?}", other),
        }
    }

    #[test]
    fn render_factor_with_unary() {
        let expr = wvec4(
            rv("uv").x() * (sin(rv("uv").y() * 6.0 - cv("time") * 2.0) * 0.5 + 0.5),
            0.0,
            0.0,
            0.0,
        );

        let registry = create_test_registry();
        let program = GpuProgramBuilder::new(&registry)
            .build_program(&expr)
            .expect("builder should support unary render factors");

        match &program.render_plan.components[0] {
            RenderComponent::RenderComposite {
                component,
                factor_render_component,
                factor_render_scale,
                factor_inner_constant,
                factor_inner_compute,
                factor_outer_constant,
                factor_outer_compute,
                factor_unary,
                offset,
                ..
            } => {
                assert_eq!(*component, 0);
                assert_eq!(*factor_render_component, Some(1));
                assert!((*factor_render_scale - 6.0).abs() < f32::EPSILON);
                assert!((*factor_inner_constant - 0.0).abs() < f32::EPSILON);
                assert!(factor_inner_compute.is_some()); // carries -time * 2.0
                assert!((*factor_outer_constant - 0.5).abs() < f32::EPSILON);
                assert!(factor_outer_compute.is_none());
                assert!(matches!(factor_unary, FactorUnary::Sin));
                assert!((*offset - 0.5).abs() < f32::EPSILON);
            }
            other => panic!("unexpected component: {:?}", other),
        }
    }
}
