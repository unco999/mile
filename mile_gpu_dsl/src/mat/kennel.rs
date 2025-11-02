use std::{collections::HashMap, rc::Rc};

use bytemuck::{cast_slice, Zeroable};
use mile_api::{CpuGlobalUniform, Tick};
use wgpu::util::DownloadBuffer;

pub use crate::program_pipeline::{
    ProgramHandle, ProgramPipelineError, ProgramSlotInfo, RenderBindingLayer,
    RenderBindingResources,
};

use crate::{
    core::Expr,
    mat::{
        gpu_ast::GpuAstNode,
        gpu_ast_compute_pipeline::ComputePipelineConfig,
        op::ImportRegistry,
    },
    program_pipeline::ProgramPipeline,
};

/// Kennel 初始化配置
pub struct KennelConfig {
    pub compute_config: ComputePipelineConfig,
    pub readback_interval_secs: u64,
}

impl Default for KennelConfig {
    fn default() -> Self {
        Self {
            compute_config: ComputePipelineConfig {
                max_nodes: 2048,
                max_imports: 32,
                workgroup_size: (64, 1, 1),
            },
            readback_interval_secs: 2,
        }
    }
}

/// 注册后的程序信息
#[derive(Clone)]
pub struct RegisteredProgram {
    pub handle: ProgramHandle,
    pub info: ProgramSlotInfo,
}

pub struct Kennel {
    pipeline: ProgramPipeline,
    programs: HashMap<u32, RegisteredProgram>,
    ordered_programs: Vec<u32>,
    readback_tick: Tick,
    render_binding_layers: Vec<RenderBindingLayer>,
    render_binding_resources: Option<RenderBindingResources>,
    render_binding_capacity: usize,
    _global_uniform: Rc<CpuGlobalUniform>,
}

impl Kennel {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        global_uniform: Rc<CpuGlobalUniform>,
        config: KennelConfig,
    ) -> Self {
        let tick_interval = config.readback_interval_secs.max(1);
        let pipeline = ProgramPipeline::new(
            device,
            queue,
            config.compute_config,
            global_uniform.get_buffer(),
        );

        Self {
            pipeline,
            programs: HashMap::new(),
            ordered_programs: Vec::new(),
            readback_tick: Tick::new(tick_interval),
            render_binding_layers: Vec::new(),
            render_binding_resources: None,
            render_binding_capacity: 0,
            _global_uniform: global_uniform,
        }
    }

    /// 注册一个表达式，返回程序句柄
    pub fn register_program(
        &mut self,
        expr: &Expr,
        registry: &ImportRegistry,
    ) -> Result<ProgramHandle, ProgramPipelineError> {
        let handle = self.pipeline.register_program(expr, registry)?;

        if let Some(info) = self.pipeline.program_info(handle) {
            self.programs.insert(handle.0, RegisteredProgram { handle, info });
            self.ordered_programs.push(handle.0);
        }

        self.render_binding_resources = None;

        Ok(handle)
    }

    pub fn reserve_render_layers(
        &mut self,
        device: &wgpu::Device,
        capacity: usize,
    ) {
        let cap = capacity.max(1);
        self.render_binding_capacity = cap;
        self.render_binding_layers
            .resize_with(cap, RenderBindingLayer::zeroed);
        self.render_binding_resources = Some(RenderBindingResources::with_capacity(
            device,
            cap,
            self.pipeline.result_buffer(),
        ));
    }

    /// 返回所有渲染层描述（保持注册顺序）
    pub fn render_layers(&self) -> Vec<&ProgramSlotInfo> {
        self.ordered_programs
            .iter()
            .filter_map(|key| self.programs.get(key))
            .map(|entry| &entry.info)
            .collect()
    }

    /// 查询指定程序的详细信息
    pub fn program_info(&self, handle: ProgramHandle) -> Option<&ProgramSlotInfo> {
        self.programs.get(&handle.0).map(|entry| &entry.info)
    }

    /// 执行所有已注册的 compute 程序
    pub fn compute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), ProgramPipelineError> {
        self.pipeline.run_compute(device, queue)
    }

    /// 当前节点数量
    pub fn total_node_count(&self) -> usize {
        self.pipeline.total_node_count()
    }

    /// 只读访问 GPU 节点缓冲
    pub fn node_buffer(&self) -> &wgpu::Buffer {
        self.pipeline.node_buffer()
    }

    /// 只读访问 GPU 结果缓冲
    pub fn result_buffer(&self) -> &wgpu::Buffer {
        self.pipeline.result_buffer()
    }

    /// 重新生成渲染绑定层数据与 GPU 资源
    pub fn rebuild_render_bindings(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<&RenderBindingResources> {
        if self.render_binding_resources.is_none() {
            let required = self
                .render_binding_capacity
                .max(self.ordered_programs.len())
                .max(1);
            self.render_binding_capacity = required;
            self.render_binding_layers
                .resize_with(required, RenderBindingLayer::zeroed);
            self.render_binding_resources = Some(RenderBindingResources::with_capacity(
                device,
                required,
                self.pipeline.result_buffer(),
            ));
        } else if self.render_binding_layers.len() != self.render_binding_capacity {
            self.render_binding_layers
                .resize_with(self.render_binding_capacity, RenderBindingLayer::zeroed);
        }

        for slot in &mut self.render_binding_layers {
            *slot = RenderBindingLayer::zeroed();
        }

        for (idx, key) in self
            .ordered_programs
            .iter()
            .take(self.render_binding_capacity)
            .enumerate()
        {
            if let Some(entry) = self.programs.get(key) {
                self.render_binding_layers[idx] = entry.info.render_layer.to_binding_layer();
            }
        }

        if let Some(resources) = &mut self.render_binding_resources {
            resources.write_layers(queue, &self.render_binding_layers);
            Some(resources)
        } else {
            None
        }
    }

    pub fn render_binding_layers(&self) -> &[RenderBindingLayer] {
        &self.render_binding_layers
    }

    pub fn render_binding_layers_mut(&mut self) -> &mut Vec<RenderBindingLayer>{
        &mut self.render_binding_layers
    }

    pub fn render_binding_resources(&self) -> Option<&RenderBindingResources> {
        self.render_binding_resources.as_ref()
    }

    /// 调试读取：按设定间隔打印前若干个节点与结果
    pub fn debug_readback(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let node_count = self.pipeline.total_node_count();
        if node_count == 0 {
            return;
        }
        if !self.readback_tick.tick() {
            return;
        }

        let nodes_to_read = 12;
        let node_bytes = (nodes_to_read as u64) * (GpuAstNode::SIZE as u64);
        let node_buffer = self.pipeline.node_buffer().clone();
        let node_slice = node_buffer.slice(0..node_bytes);

        DownloadBuffer::read_buffer(
            device,
            queue,
            &node_slice,
            move |result| {
                if let Ok(download) = result {
                    let nodes: &[GpuAstNode] = cast_slice(&download);
                    println!("=== 节点调试 (前 {} 个) ===", nodes.len());
                    for (idx, node) in nodes.iter().enumerate() {
                        println!("[node {idx}] {:?}", node);
                    }
                }
            },
        );

        let results_to_read = 12;
        let result_bytes =
            (results_to_read as u64) * (std::mem::size_of::<[f32; 4]>() as u64);
        let result_buffer = self.pipeline.result_buffer().clone();
        let result_slice = result_buffer.slice(0..result_bytes);

        DownloadBuffer::read_buffer(
            device,
            queue,
            &result_slice,
            move |result| {
                if let Ok(download) = result {
                    let values: &[[f32; 4]] = cast_slice(&download);
                    println!("=== 结果调试 (前 {} 项) ===", values.len());
                    for (idx, value) in values.iter().enumerate() {
                        println!("[result {idx}] {:?}", value);
                    }
                }
            },
        );
    }
}
