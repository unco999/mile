// 鍦?gpu_ast.rs 涓坊鍔犱互涓嬪唴瀹?

use crate::prelude::gpu_ast::*;
use crate::prelude::gpu_program::*;
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, BufferDescriptor, BufferUsages, CommandEncoder,
    ComputePass, ComputePipeline, Device, PipelineLayout, Queue, ShaderModule, util::DeviceExt,
};

pub struct GpuComputePipeline {
    pub node_buffer: Buffer,
    pub result_buffer: Buffer,
    import_buffer: Buffer,

    bind_group: BindGroup,
    bind_group_layout: BindGroupLayout,
    compute_pipeline: ComputePipeline,

    max_nodes: u32,
    max_imports: u32,
}

pub struct ComputePipelineConfig {
    pub max_nodes: u32,
    pub max_imports: u32,
    pub workgroup_size: (u32, u32, u32),
}

impl Default for ComputePipelineConfig {
    fn default() -> Self {
        Self {
            max_nodes: 1024,
            max_imports: 32,
            workgroup_size: (64, 1, 1),
        }
    }
}

/// 瀵煎叆鏁版嵁缁熶竴鏍煎紡
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuImportData {
    pub time: f32,
    pub delta_time: f32,
    pub resolution: [f32; 2],
    pub mouse: [f32; 2],
    pub custom: [f32; 4],
}

impl GpuComputePipeline {
    /// 鍒涘缓鏂扮殑 GPU 璁＄畻绠＄嚎
    pub fn new(
        device: &Device,
        queue: &Queue,
        config: ComputePipelineConfig,
        global_buffer: wgpu::Buffer,
    ) -> Self {
        let max_nodes = config.max_nodes;
        let max_imports = config.max_imports;

        // 鍒涘缓缂撳啿鍖?
        let node_buffer = create_node_buffer(&device, max_nodes);
        let result_buffer = create_result_buffer(&device, max_nodes);
        let import_buffer = create_import_buffer(&device, max_imports);

        // 鍒涘缓缁戝畾缁勫竷灞€
        let bind_group_layout = create_bind_group_layout(&device);

        // 鍒涘缓缁戝畾缁?
        let bind_group = create_bind_group(
            &device,
            &bind_group_layout,
            &node_buffer,
            &result_buffer,
            &import_buffer,
            &global_buffer,
        );

        // 鍒涘缓璁＄畻绠＄嚎
        let compute_pipeline =
            create_compute_pipeline(&device, &bind_group_layout, config.workgroup_size);

        Self {
            node_buffer,
            result_buffer,
            import_buffer,
            bind_group,
            bind_group_layout,
            compute_pipeline,
            max_nodes,
            max_imports,
        }
    }

    pub fn update_nodes(
        &mut self,
        graph: &GpuAstGraph,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if graph.nodes.len() > self.max_nodes as usize {
            return Err("测试".into());
        }

        self.write_nodes(queue, &graph.nodes)
    }

    /// 鏇存柊瀵煎叆鏁版嵁鍒?GPU 缂撳啿鍖?
    pub fn update_imports(
        &mut self,
        import_data: &[GpuImportData],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if import_data.len() > self.max_imports as usize {
            return Err(".into()".into());
        }

        let import_bytes = bytemuck::cast_slice(import_data);
        queue.write_buffer(&self.import_buffer, 0, import_bytes);

        Ok(())
    }

    /// 鎵ц璁＄畻绠＄嚎
    pub fn execute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        graph: &GpuAstGraph,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 鏇存柊鑺傜偣鏁版嵁
        self.update_nodes(graph, device, queue)?;
        self.dispatch(device, queue, graph.nodes.len() as u32);
        Ok(())
    }

    /// 鍩轰簬搴忓垪鍖?GPU 绋嬪簭鎵ц璁＄畻
    pub fn execute_program(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        program: &SerializableGpuProgram,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu_nodes = program.to_gpu_nodes();
        if gpu_nodes.len() > self.max_nodes as usize {
            return Err(".into()".into());
        }
        self.write_nodes(queue, &gpu_nodes)?;
        self.dispatch(device, queue, gpu_nodes.len() as u32);
        Ok(())
    }

    /// 浠?GPU 璇诲彇璁＄畻缁撴灉
    fn read_results(&self, graph: &GpuAstGraph) {}

    /// 鑾峰彇鏈€澶ц妭鐐瑰閲?
    pub fn max_nodes(&self) -> u32 {
        self.max_nodes
    }

    /// 鑾峰彇鏈€澶у鍏ュ閲?
    pub fn max_imports(&self) -> u32 {
        self.max_imports
    }

    pub fn upload_raw_nodes(
        &mut self,
        queue: &wgpu::Queue,
        nodes: &[GpuAstNode],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if nodes.len() > self.max_nodes as usize {
            return Err("严重错误".into());
        }
        self.write_nodes(queue, nodes)
    }

    pub fn dispatch_node_count(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        node_count: u32,
    ) {
        self.dispatch(device, queue, node_count);
    }
}

impl GpuComputePipeline {
    fn write_nodes(
        &mut self,
        queue: &wgpu::Queue,
        nodes: &[GpuAstNode],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_bytes = bytemuck::cast_slice(nodes);
        queue.write_buffer(&self.node_buffer, 0, node_bytes);
        Ok(())
    }

    fn dispatch(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, node_count: u32) {
        if node_count == 0 {
            return;
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);

            let workgroup_count = calculate_workgroup_count(node_count);
            compute_pass.dispatch_workgroups(
                workgroup_count.0,
                workgroup_count.1,
                workgroup_count.2,
            );
        }

        queue.submit(Some(encoder.finish()));
    }
}

// 杈呭姪鍑芥暟
fn create_node_buffer(device: &Device, max_nodes: u32) -> Buffer {
    let size = (max_nodes as u64) * (GpuAstNode::SIZE as u64);

    device.create_buffer(&BufferDescriptor {
        label: Some("Node Buffer"),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_result_buffer(device: &Device, max_nodes: u32) -> Buffer {
    let size = (max_nodes as u64) * (std::mem::size_of::<[f32; 4]>() as u64);

    device.create_buffer(&BufferDescriptor {
        label: Some("Result Buffer"),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_import_buffer(device: &Device, max_imports: u32) -> Buffer {
    let size = (max_imports as u64) * (std::mem::size_of::<GpuImportData>() as u64);

    device.create_buffer(&BufferDescriptor {
        label: Some("Import Buffer"),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            // 鑺傜偣缂撳啿鍖虹粦瀹?
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 缁撴灉缂撳啿鍖虹粦瀹?
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 瀵煎叆鏁版嵁缂撳啿鍖虹粦瀹?
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    node_buffer: &Buffer,
    result_buffer: &Buffer,
    import_buffer: &Buffer,
    cpu_global_buffer: &wgpu::Buffer,
) -> BindGroup {
    // 鍒涘缓缁熶竴甯搁噺

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: node_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cpu_global_buffer.as_entire_binding(),
            },
        ],
    })
}

fn create_compute_pipeline(
    device: &Device,
    bind_group_layout: &BindGroupLayout,
    workgroup_size: (u32, u32, u32),
) -> ComputePipeline {
    let shader_source = create_compute_shader(workgroup_size);

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: Default::default(),
    })
}

fn calculate_workgroup_count(node_count: u32) -> (u32, u32, u32) {
    let workgroup_size = 64; // 涓庣潃鑹插櫒涓殑 workgroup_size 鍖归厤
    let workgroups = (node_count + workgroup_size - 1) / workgroup_size;
    (workgroups, 1, 1)
}

// 鍒涘缓璁＄畻鐫€鑹插櫒
fn create_compute_shader(workgroup_size: (u32, u32, u32)) -> String {
    format!(
        r#"
struct GpuAstNode {{
    data: vec4<f32>,
    state: u32,
    op: u32,
    data_type: u32,
    left_child: u32,
    right_child: u32,
    import_info: u32,
    constant_value: f32,
    else_child: u32,
}};


struct GlobalUniform {{
      // === block 1: atomic z/layouts ===
    click_layout_z: u32,
    click_layout_id: u32,
    hover_layout_id: u32,
    hover_layout_z: u32, // 16 bytes

    // === block 2: atomic drag ===
    drag_layout_id: u32,
    drag_layout_z: u32,
    pad_atomic1: u32,
    pad_atomic2: u32,    // 16 bytes

    // === block 3: dt ===
    dt: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,                   // 16 bytes

    // === block 4: mouse ===
    mouse_pos: vec2<f32>,
    mouse_state: u32,
    frame: u32,                   // 16 bytes

    // === block 5: screen info ===
    screen_size: vec2<u32>,
    press_duration: f32,
    time: f32,                    // 16 bytes

    // === block 6: event points ===
    event_point: vec2<f32>,
    extra1: vec2<f32>,            // 16 bytes

    // === block 7: extra data ===
    extra2: vec2<f32>,
    pad_extra: vec2<f32>         // 16 bytes
}};

@group(0) @binding(0)
var<storage, read> nodes: array<GpuAstNode>;

@group(0) @binding(1) 
var<storage, read_write> results: array<vec4<f32>>;

@group(0) @binding(2) 
var<storage, read> global_uniform: GlobalUniform;

const OP_ADD: u32 = 0x1u;
const OP_SUBTRACT: u32 = 0x2u;
const OP_MULTIPLY: u32 = 0x4u;
const OP_DIVIDE: u32 = 0x8u;
const OP_MODULO: u32 = 0x10u;
const OP_POW: u32 = 0x20u;
const OP_GREATER_THAN: u32 = 0x40u;
const OP_GREATER_EQUAL: u32 = 0x80u;
const OP_LESS_THAN: u32 = 0x100u;
const OP_LESS_EQUAL: u32 = 0x200u;
const OP_EQUAL: u32 = 0x400u;
const OP_NOT_EQUAL: u32 = 0x800u;
const OP_INDEX: u32 = 0x1000u;
const OP_SIN: u32 = 0x2000u;
const OP_COS: u32 = 0x4000u;
const OP_TAN: u32 = 0x8000u;
const OP_EXP: u32 = 0x10000u;
const OP_LOG: u32 = 0x20000u;
const OP_SQRT: u32 = 0x40000u;
const OP_ABS: u32 = 0x80000u;
const OP_CONDITIONAL: u32 = 0x100000u;
const OP_SMOOTHSTEP: u32 = 0x200000u;

const TYPE_SCALAR: u32 = 0u;
const TYPE_VEC2: u32 = 1u;
const TYPE_VEC3: u32 = 2u;
const TYPE_VEC4: u32 = 3u;

const IS_COMPUTE: u32 = 0x1u;
const IS_RENDER: u32 = 0x2u;
const IS_LEAF: u32 = 0x10u;
const STATE_PRECOMPUTED: u32 = 0x200u;

fn is_valid_index(idx: u32) -> bool {{
    return idx != 0xFFFFFFFFu;
}}

fn load_constant(node: GpuAstNode) -> vec4<f32> {{
    switch(node.data_type) {{
        case TYPE_SCALAR: {{
            return vec4<f32>(node.constant_value);
        }}
        case TYPE_VEC2: {{
            return vec4<f32>(node.data.xy, 0.0, 1.0);
        }}
        case TYPE_VEC3: {{
            return vec4<f32>(node.data.xyz, 1.0);
        }}
        default: {{
            return node.data;
        }}
    }}
}}

fn load_compute_import(mask: u32) -> vec4<f32> {{
    var value = vec4<f32>(0.0);
    if ((mask & 0x1u) != 0u) {{
        value = vec4<f32>(global_uniform.time);
    }}
    if ((mask & 0x2u) != 0u) {{
        value = vec4<f32>(global_uniform.dt);
    }}
    if ((mask & 0x4u) != 0u) {{
        value = vec4<f32>(f32(global_uniform.screen_size.x), f32(global_uniform.screen_size.y), 0.0, 1.0);
    }}
    return value;
}}

fn evaluate_leaf(node: GpuAstNode) -> vec4<f32> {{
    if (node.import_info == 0u) {{
        return load_constant(node);
    }}
    let import_type = (node.import_info >> 16u) & 0xFFu;
    let mask = node.import_info & 0xFFu;
    if (import_type == 1u) {{
        return load_compute_import(mask);
    }}
    return node.data;
}}

fn apply_binary_op(op: u32, left: f32, right: f32) -> f32 {{
    switch(op) {{
        case OP_ADD: {{ return left + right; }}
        case OP_SUBTRACT: {{ return left - right; }}
        case OP_MULTIPLY: {{ return left * right; }}
        case OP_DIVIDE: {{
            return select(0.0, left / right, abs(right) > 1e-6);
        }}
        case OP_MODULO: {{
            return select(0.0, left % right, abs(right) > 1e-6);
        }}
        case OP_POW: {{ return pow(left, right); }}
        case OP_GREATER_THAN: {{ return select(0.0, 1.0, left > right); }}
        case OP_GREATER_EQUAL: {{ return select(0.0, 1.0, left >= right); }}
        case OP_LESS_THAN: {{ return select(0.0, 1.0, left < right); }}
        case OP_LESS_EQUAL: {{ return select(0.0, 1.0, left <= right); }}
        case OP_EQUAL: {{ return select(0.0, 1.0, abs(left - right) < 1e-4); }}
        case OP_NOT_EQUAL: {{ return select(0.0, 1.0, abs(left - right) >= 1e-4); }}
        default: {{ return left; }}
    }}
}}

fn apply_unary_op(op: u32, input: f32) -> f32 {{
    switch(op) {{
        case OP_SIN: {{ return sin(input); }}
        case OP_COS: {{ return cos(input); }}
        case OP_TAN: {{ return tan(input); }}
        case OP_EXP: {{ return exp(input); }}
        case OP_LOG: {{ return select(0.0, log(input), input > 0.0); }}
        case OP_SQRT: {{ return select(0.0, sqrt(input), input >= 0.0); }}
        case OP_ABS: {{ return abs(input); }}
        default: {{ return input; }}
    }}
}}

fn clamp_to_type(value: vec4<f32>, ty: u32) -> vec4<f32> {{
    switch(ty) {{
        case TYPE_SCALAR: {{
            return vec4<f32>(value.x);
        }}
        case TYPE_VEC2: {{
            return vec4<f32>(value.xy, 0.0, 1.0);
        }}
        case TYPE_VEC3: {{
            return vec4<f32>(value.xyz, 1.0);
        }}
        default: {{
            return value;
        }}
    }}
}}

@compute @workgroup_size({}, {}, {})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let node_index = global_id.x;
    if (node_index >= arrayLength(&results)) {{
        return;
    }}
    let node = nodes[node_index];
    if ((node.state & IS_COMPUTE) == 0u) {{
        return;
    }}

    if ((node.state & IS_LEAF) != 0u || (node.state & STATE_PRECOMPUTED) != 0u) {{
        results[node_index] = evaluate_leaf(node);
        return;
    }}

    let left_value = select(vec4<f32>(0.0), results[node.left_child], is_valid_index(node.left_child));
    let right_value = select(vec4<f32>(0.0), results[node.right_child], is_valid_index(node.right_child));
    let else_value = select(vec4<f32>(0.0), results[node.else_child], is_valid_index(node.else_child));

    var result = vec4<f32>(0.0);

    if (node.op == OP_CONDITIONAL) {{
        for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {{
            let cond = abs(left_value[lane]) > 1e-6;
            result[lane] = select(else_value[lane], right_value[lane], cond);
        }}
    }} else if (node.op == OP_SMOOTHSTEP) {{
        for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {{
            result[lane] = smoothstep(left_value[lane], right_value[lane], else_value[lane]);
        }}
    }} else if (node.op <= OP_NOT_EQUAL) {{
        for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {{
            result[lane] = apply_binary_op(node.op, left_value[lane], right_value[lane]);
        }}
    }} else if (node.op == OP_INDEX) {{
        let lane = u32(clamp(right_value.x, 0.0, 3.0));
        result = vec4<f32>(left_value[lane]);
    }} else {{
        for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {{
            result[lane] = apply_unary_op(node.op, left_value[lane]);
        }}
    }}

    results[node_index] = clamp_to_type(result, node.data_type);
}}
"#,
        workgroup_size.0, workgroup_size.1, workgroup_size.2
    )
}
