use std::ops::Range;

use crate::{
    core::Expr,
    mat::{
        gpu_ast::{GpuAstNode, GpuAstState},
        gpu_ast_compute_pipeline::{ComputePipelineConfig, GpuComputePipeline},
        gpu_program::{
            ComputeStage, GpuProgramBuilder, ProgramBuildError, SerializableGpuProgram,
        },
        op::ImportRegistry,
    },
};

use super::render_layer::{encode_render_expr_nodes, RenderExprNodeGpu, RenderLayerDescriptor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgramHandle(pub u32);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageStats {
    pub precompute_nodes: u32,
    pub per_frame_nodes: u32,
}

impl StageStats {
    fn record(stage: ComputeStage) -> Self {
        match stage {
            ComputeStage::Precompute => Self {
                precompute_nodes: 1,
                per_frame_nodes: 0,
            },
            ComputeStage::PerFrame => Self {
                precompute_nodes: 0,
                per_frame_nodes: 1,
            },
        }
    }

    fn accumulate(&mut self, other: StageStats) {
        self.precompute_nodes += other.precompute_nodes;
        self.per_frame_nodes += other.per_frame_nodes;
    }
}

#[derive(Debug)]
pub struct ProgramSlot {
    handle: ProgramHandle,
    node_offset: u32,
    node_count: u32,
    expr_offset: u32,
    expr_count: u32,
    render_layer: RenderLayerDescriptor,
    stage_stats: StageStats,
    precompute_nodes: Vec<u32>,
    per_frame_nodes: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct ProgramSlotInfo {
    pub handle: ProgramHandle,
    pub compute_range: Range<u32>,
    pub expr_range: Range<u32>,
    pub stage_stats: StageStats,
    pub precompute_nodes: Vec<u32>,
    pub per_frame_nodes: Vec<u32>,
    pub render_layer: RenderLayerDescriptor,
}

#[derive(Debug)]
pub enum ProgramPipelineError {
    NodeCapacityExceeded { capacity: usize, required: usize },
    ProgramBuild(ProgramBuildError),
    GpuPipeline(String),
}

impl From<ProgramBuildError> for ProgramPipelineError {
    fn from(value: ProgramBuildError) -> Self {
        Self::ProgramBuild(value)
    }
}

pub struct ProgramPipeline {
    compute_pipeline: GpuComputePipeline,
    nodes: Vec<GpuAstNode>,
    render_expr_nodes: Vec<RenderExprNodeGpu>,
    programs: Vec<ProgramSlot>,
    next_program_id: u32,
    dirty: bool,
}

impl ProgramPipeline {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: ComputePipelineConfig,
        global_buffer: wgpu::Buffer,
    ) -> Self {
        let compute_pipeline =
            GpuComputePipeline::new(device, queue, config, global_buffer);

        Self {
            compute_pipeline,
            nodes: Vec::new(),
            render_expr_nodes: Vec::new(),
            programs: Vec::new(),
            next_program_id: 0,
            dirty: false,
        }
    }

    pub fn register_program(
        &mut self,
        expr: &Expr,
        registry: &ImportRegistry,
    ) -> Result<ProgramHandle, ProgramPipelineError> {
        let builder = GpuProgramBuilder::new(registry);
        let program = builder.build_program(expr)?;

        let node_count = program.compute_nodes.len();
        let required_total = self.nodes.len() + node_count;

        if required_total > self.compute_pipeline.max_nodes() as usize {
            return Err(ProgramPipelineError::NodeCapacityExceeded {
                capacity: self.compute_pipeline.max_nodes() as usize,
                required: required_total,
            });
        }

        let handle = ProgramHandle(self.next_program_id);
        self.next_program_id += 1;

        let node_offset = self.nodes.len() as u32;
        let mut gpu_nodes = convert_program_nodes(&program, node_offset);
        self.nodes.append(&mut gpu_nodes);

        let expr_offset = self.render_expr_nodes.len() as u32;
        let mut expr_nodes =
            encode_render_expr_nodes(&program.render_expr_nodes, node_offset, expr_offset);
        self.render_expr_nodes.append(&mut expr_nodes);

        let render_layer =
            RenderLayerDescriptor::from_program(&program, handle, node_offset, expr_offset);
        let stage_info = collect_stage_info(&program, node_offset);

        self.programs.push(ProgramSlot {
            handle,
            node_offset,
            node_count: node_count as u32,
            expr_offset,
            expr_count: program.render_expr_nodes.len() as u32,
            render_layer,
            stage_stats: stage_info.stats,
            precompute_nodes: stage_info.precompute_nodes,
            per_frame_nodes: stage_info.per_frame_nodes,
        });

        self.dirty = true;
        Ok(handle)
    }

    pub fn sync_gpu_buffers(
        &mut self,
        queue: &wgpu::Queue,
    ) -> Result<(), ProgramPipelineError> {
        if !self.dirty {
            return Ok(());
        }

        self.compute_pipeline
            .upload_raw_nodes(queue, &self.nodes)
            .map_err(|e| ProgramPipelineError::GpuPipeline(e.to_string()))?;
        self.dirty = false;
        Ok(())
    }

    pub fn run_compute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), ProgramPipelineError> {
        if self.nodes.is_empty() {
            return Ok(());
        }

        self.sync_gpu_buffers(queue)?;
        self.compute_pipeline
            .dispatch_node_count(device, queue, self.nodes.len() as u32);
        Ok(())
    }

    pub fn layers(&self) -> impl Iterator<Item = &RenderLayerDescriptor> {
        self.programs.iter().map(|slot| &slot.render_layer)
    }

    pub fn program_info(
        &self,
        handle: ProgramHandle,
    ) -> Option<ProgramSlotInfo> {
        self.programs
            .iter()
            .find(|slot| slot.handle == handle)
            .map(|slot| ProgramSlotInfo {
                handle: slot.handle,
                compute_range: slot.node_offset..(slot.node_offset + slot.node_count),
                expr_range: slot.expr_offset..(slot.expr_offset + slot.expr_count),
                stage_stats: slot.stage_stats,
                precompute_nodes: slot.precompute_nodes.clone(),
                per_frame_nodes: slot.per_frame_nodes.clone(),
                render_layer: slot.render_layer.clone(),
            })
    }

    pub fn total_node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn compute_nodes(&self) -> &[GpuAstNode] {
        &self.nodes
    }

    pub fn render_expr_nodes(&self) -> &[RenderExprNodeGpu] {
        &self.render_expr_nodes
    }

    pub fn node_buffer(&self) -> &wgpu::Buffer {
        &self.compute_pipeline.node_buffer
    }

    pub fn result_buffer(&self) -> &wgpu::Buffer {
        &self.compute_pipeline.result_buffer
    }
}

struct StageInfo {
    stats: StageStats,
    precompute_nodes: Vec<u32>,
    per_frame_nodes: Vec<u32>,
}

fn collect_stage_info(
    program: &SerializableGpuProgram,
    node_offset: u32,
) -> StageInfo {
    let mut stats = StageStats::default();
    let mut precompute_nodes = Vec::new();
    let mut per_frame_nodes = Vec::new();

    for node in &program.compute_nodes {
        match node.stage {
            ComputeStage::Precompute => {
                precompute_nodes.push(node_offset + node.id);
            }
            ComputeStage::PerFrame => {
                per_frame_nodes.push(node_offset + node.id);
            }
        }
        stats.accumulate(StageStats::record(node.stage));
    }

    StageInfo {
        stats,
        precompute_nodes,
        per_frame_nodes,
    }
}

fn convert_program_nodes(
    program: &SerializableGpuProgram,
    node_offset: u32,
) -> Vec<GpuAstNode> {
    let mut nodes = program.to_gpu_nodes();

    for node in &mut nodes {
        let (left, right) = node.get_children();
        let adjusted_left = if left != u32::MAX {
            left + node_offset
        } else {
            left
        };
        let adjusted_right = if right != u32::MAX {
            right + node_offset
        } else {
            right
        };
        node.set_children(adjusted_left, adjusted_right);

        // 将预计算节点标记为已完成，方便渲染时直接读取
        if node.has_state(GpuAstState::PRE_COMPUTED) {
            node.add_state(GpuAstState::COMPUTE_OVER);
        }
    }

    nodes
}
