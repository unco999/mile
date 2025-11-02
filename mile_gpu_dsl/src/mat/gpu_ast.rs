use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

use crate::{core::{BinaryOp, UnaryFunc}, mat::op::{ImportType, MatOp, Matrix, MatrixPlan}};

bitflags! {
    /// GPU AST Node State
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct GpuAstState: u32 {
        const IS_COMPUTE     = 0b00000001;      // 计算管线节点
        const IS_RENDER      = 0b00000010;      // 渲染管线节点  
        const COMPUTE_OVER   = 0b00000100;      // 计算已完成
        const IS_TICK        = 0b00001000;      // 需要每帧更新
        const IS_LEAF        = 0b00010000;      // 叶子节点（常量/导入）
        const IS_BRANCH      = 0b00100000;      // 分支节点
        const NEEDS_UPDATE   = 0b01000000;      // 需要更新
        const IS_DIRTY       = 0b10000000;      // 数据已脏
        const IS_FINAL_OUTPUT = 0b100000000;    // 最终输出
        const PRE_COMPUTED   = 0b1000000000;    // 预先在compute计算
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuOp {
    Add            = 0b000000000001,
    Subtract       = 0b000000000010,
    Multiply       = 0b000000000100,
    Divide         = 0b000000001000,
    Modulo         = 0b000000010000,
    Pow            = 0b000000100000,
    GreaterThan    = 0b000001000000,
    GreaterEqual   = 0b000010000000,
    LessThan       = 0b000100000000,
    LessEqual      = 0b001000000000,
    Equal          = 0b010000000000,
    NotEqual       = 0b100000000000,
    Index          = 0b1000000000000,
    Sin            = 0b10000000000000,
    Cos            = 0b100000000000000,
    Tan            = 0b1000000000000000,
    Exp            = 0b10000000000000000,
    Log            = 0b100000000000000000,
    Sqrt           = 0b1000000000000000000,
    Abs            = 0b10000000000000000000,
}

#[derive(Debug,PartialEq, Eq, PartialOrd, Ord)]
/// 节点数据类型
pub enum DataType {
    Scalar = 0,    // 标量
    Vec2   = 1,    // 二维向量  
    Vec3   = 2,    // 三维向量
    Vec4   = 3,    // 四维向量
}

/// 优化的 GPU AST 节点
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug,Pod,Zeroable)]
pub struct GpuAstNode {
    // 数据部分 (16字节)
    data: [f32; 4],
    
    // 类型和控制部分 (16字节)  
    state: u32,
    op: u32,                    // 存储 GpuOp 的 u32 值
    data_type: u32,             // 存储 DataType 的 u32 值
    
    // 连接信息 (8字节)
    left_child: u32,
    right_child: u32,
    
    // 导入和常量信息 (8字节)
    import_info: u32,
    constant_value: f32,
    pad:u32,
}


// 为 GpuOp 实现一些辅助方法
impl GpuOp {
      pub fn is_arithmetic(&self) -> bool {
        matches!(self,
            GpuOp::Add | GpuOp::Subtract | GpuOp::Multiply | 
            GpuOp::Divide | GpuOp::Modulo | GpuOp::Pow
        )
    }

    pub fn is_comparison(&self) -> bool {
        matches!(self,
            GpuOp::GreaterThan | GpuOp::GreaterEqual | 
            GpuOp::LessThan | GpuOp::LessEqual | 
            GpuOp::Equal | GpuOp::NotEqual
        )
    }

    pub fn is_unary(&self) -> bool {
        matches!(self,
            GpuOp::Sin | GpuOp::Cos | GpuOp::Tan | 
            GpuOp::Exp | GpuOp::Log | GpuOp::Sqrt | GpuOp::Abs
        )
    }

    pub fn from_binary_op(op: &BinaryOp) -> Self {
        match op {
            BinaryOp::Add => GpuOp::Add,
            BinaryOp::Subtract => GpuOp::Subtract,
            BinaryOp::Multiply => GpuOp::Multiply,
            BinaryOp::Divide => GpuOp::Divide,
            BinaryOp::Modulo => GpuOp::Modulo,
            BinaryOp::Pow => GpuOp::Pow,
            BinaryOp::GreaterThan => GpuOp::GreaterThan,
            BinaryOp::GreaterEqual => GpuOp::GreaterEqual,
            BinaryOp::LessThan => GpuOp::LessThan,
            BinaryOp::LessEqual => GpuOp::LessEqual,
            BinaryOp::Equal => GpuOp::Equal,
            BinaryOp::NotEqual => GpuOp::NotEqual,
            BinaryOp::Index => GpuOp::Index,
        }
    }

    pub fn from_unary_func(func: &UnaryFunc) -> Self {
        match func {
            UnaryFunc::Sin => GpuOp::Sin,
            UnaryFunc::Cos => GpuOp::Cos,
            UnaryFunc::Tan => GpuOp::Tan,
            UnaryFunc::Exp => GpuOp::Exp,
            UnaryFunc::Log => GpuOp::Log,
            UnaryFunc::Sqrt => GpuOp::Sqrt,
            UnaryFunc::Abs => GpuOp::Abs,
        }
    }

    // 从 u32 转换回 GpuOp（安全版本）
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0b000000000001 => Some(GpuOp::Add),
            0b000000000010 => Some(GpuOp::Subtract),
            0b000000000100 => Some(GpuOp::Multiply),
            0b000000001000 => Some(GpuOp::Divide),
            0b000000010000 => Some(GpuOp::Modulo),
            0b000000100000 => Some(GpuOp::Pow),
            0b000001000000 => Some(GpuOp::GreaterThan),
            0b000010000000 => Some(GpuOp::GreaterEqual),
            0b000100000000 => Some(GpuOp::LessThan),
            0b001000000000 => Some(GpuOp::LessEqual),
            0b010000000000 => Some(GpuOp::Equal),
            0b100000000000 => Some(GpuOp::NotEqual),
            0b1000000000000 => Some(GpuOp::Index),
            0b10000000000000 => Some(GpuOp::Sin),
            0b100000000000000 => Some(GpuOp::Cos),
            0b1000000000000000 => Some(GpuOp::Tan),
            0b10000000000000000 => Some(GpuOp::Exp),
            0b100000000000000000 => Some(GpuOp::Log),
            0b1000000000000000000 => Some(GpuOp::Sqrt),
            0b10000000000000000000 => Some(GpuOp::Abs),
            _ => None,
        }
    }
}
// 为 DataType 实现转换方法
impl DataType {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(DataType::Scalar),
            1 => Some(DataType::Vec2),
            2 => Some(DataType::Vec3),
            3 => Some(DataType::Vec4),
            _ => None,
        }
    }
}

impl GpuAstNode {
    pub const SIZE: usize = std::mem::size_of::<GpuAstNode>();

      // 获取导入名称（简化版本，实际需要从导入信息中解析）
    pub fn get_import_name(&self) -> Option<&'static str> {
        if self.has_state(GpuAstState::IS_LEAF) && self.get_constant() == 0.0 {
            // 根据导入类型返回名称
            let (import_type, _) = self.get_import();
            match import_type {
                0 => Some("uv"),    // 渲染导入
                1 => Some("time"),  // 计算导入
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn new() -> Self {
        Self {
            data: [0.0; 4],
            state: 0,
            op: 0,
            data_type: DataType::Scalar as u32,
            left_child: u32::MAX,
            right_child: u32::MAX,
            import_info: 0,
            constant_value: 0.0,
            pad:0
        }
    }

    // 状态操作
    pub fn set_state(&mut self, state: GpuAstState) {
        self.state = state.bits();
    }

    pub fn get_state(&self) -> GpuAstState {
        GpuAstState::from_bits_truncate(self.state)
    }

    pub fn add_state(&mut self, state: GpuAstState) {
        self.state |= state.bits();
    }

    pub fn has_state(&self, state: GpuAstState) -> bool {
        self.get_state().contains(state)
    }

    // 操作设置
    pub fn set_op(&mut self, op: GpuOp) {
        self.op = op as u32;
    }

    pub fn get_op(&self) -> GpuOp {
        GpuOp::from_u32(self.op).unwrap_or(GpuOp::Add)
    }

    // 数据类型设置
    pub fn set_data_type(&mut self, data_type: DataType) {
        self.data_type = data_type as u32;
    }

    pub fn get_data_type(&self) -> DataType {
        DataType::from_u32(self.data_type).unwrap_or(DataType::Scalar)
    }

    // 数据访问方法保持不变...
    pub fn set_scalar(&mut self, value: f32) {
        self.data[0] = value;
        self.set_data_type(DataType::Scalar);
    }

    pub fn get_scalar(&self) -> f32 {
        self.data[0]
    }

    pub fn set_vec2(&mut self, x: f32, y: f32) {
        self.data[0] = x;
        self.data[1] = y;
        self.set_data_type(DataType::Vec2);
    }

    pub fn get_vec2(&self) -> [f32; 2] {
        [self.data[0], self.data[1]]
    }

    pub fn set_vec3(&mut self, x: f32, y: f32, z: f32) {
        self.data[0] = x;
        self.data[1] = y;
        self.data[2] = z;
        self.set_data_type(DataType::Vec3);
    }

    pub fn get_vec3(&self) -> [f32; 3] {
        [self.data[0], self.data[1], self.data[2]]
    }

    pub fn set_vec4(&mut self, x: f32, y: f32, z: f32, w: f32) {
        self.data = [x, y, z, w];
        self.set_data_type(DataType::Vec4);
    }

    pub fn get_vec4(&self) -> [f32; 4] {
        self.data
    }

    // 其他方法保持不变...
    pub fn set_children(&mut self, left: u32, right: u32) {
        self.left_child = left;
        self.right_child = right;
    }

    pub fn get_children(&self) -> (u32, u32) {
        (self.left_child, self.right_child)
    }

    pub fn has_children(&self) -> bool {
        self.left_child != u32::MAX || self.right_child != u32::MAX
    }

    pub fn set_import(&mut self, import_type: u8, mask: u8) {
        self.import_info = ((import_type as u32) << 16) | (mask as u32);
    }

    pub fn get_import(&self) -> (u8, u8) {
        let import_type = (self.import_info >> 16) as u8;
        let mask = (self.import_info & 0xFF) as u8;
        (import_type, mask)
    }

    pub fn set_constant(&mut self, value: f32) {
        self.constant_value = value;
        self.set_scalar(value);
    }

    pub fn get_constant(&self) -> f32 {
        self.constant_value
    }

    pub fn is_final_output(&self) -> bool {
        self.has_state(GpuAstState::IS_FINAL_OUTPUT) && 
        self.get_data_type() == DataType::Vec4
    }

    pub fn get_used_components(&self) -> usize {
        match self.get_data_type() {
            DataType::Scalar => 1,
            DataType::Vec2 => 2,
            DataType::Vec3 => 3,
            DataType::Vec4 => 4,
        }
    }
}

/// GPU AST 图 - 包含所有节点和最终输出信息
#[derive(Clone, Debug)]
pub struct GpuAstGraph {
    pub nodes: Vec<GpuAstNode>,
    pub final_outputs: Vec<u32>,  // 最终输出节点的索引
    pub compute_outputs: Vec<u32>, // 计算输出节点的索引
    pub render_outputs: Vec<u32>,  // 渲染输出节点的索引
}

impl GpuAstGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            final_outputs: Vec::new(),
            compute_outputs: Vec::new(),
            render_outputs: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: GpuAstNode) -> u32 {
        let index = self.nodes.len() as u32;
        self.nodes.push(node);
        index
    }

    pub fn mark_final_output(&mut self, node_index: u32) {
        if (node_index as usize) < self.nodes.len() {
            self.nodes[node_index as usize].add_state(GpuAstState::IS_FINAL_OUTPUT);
            self.final_outputs.push(node_index);
            
            // 根据状态分类
            let node = &self.nodes[node_index as usize];
            if node.has_state(GpuAstState::IS_COMPUTE) {
                self.compute_outputs.push(node_index);
            }
            if node.has_state(GpuAstState::IS_RENDER) {
                self.render_outputs.push(node_index);
            }
        }
    }

    pub fn get_final_outputs(&self) -> Vec<&GpuAstNode> {
        self.final_outputs.iter()
            .filter_map(|&idx| self.nodes.get(idx as usize))
            .collect()
    }

    pub fn validate(&self) -> bool {
        // 检查所有最终输出都是Vec4类型
        for &output_idx in &self.final_outputs {
            if let Some(node) = self.nodes.get(output_idx as usize) {
                if node.get_data_type() != DataType::Vec4 {
                    return false;
                }
            }
        }
        
        // 检查节点连接有效性
        for (i, node) in self.nodes.iter().enumerate() {
            let (left, right) = node.get_children();
            
            if left != u32::MAX && (left as usize) >= self.nodes.len() {
                return false;
            }
            if right != u32::MAX && (right as usize) >= self.nodes.len() {
                return false;
            }
        }
        
        true
    }
}

/// 针对最终输出vec4优化的CPU模拟器
pub struct FinalOutputSimulator {
    node_values: Vec<[f32; 4]>,    // 每个节点的当前值
    node_dirty: Vec<bool>,         // 脏标记，用于增量计算
    import_values: HashMap<String, [f32; 4]>,
    time: f32,
}

impl FinalOutputSimulator {

        fn compute_dirty_nodes(&mut self, graph: &GpuAstGraph) {
        println!("\n--- 计算脏节点 ---");
        
        let mut changed = true;
        let mut iterations = 0;
        
        while changed && iterations < graph.nodes.len() * 2 { // 防止无限循环
            changed = false;
            iterations += 1;
            
            for (i, node) in graph.nodes.iter().enumerate() {
                if self.node_dirty[i] {
                    let (left_idx, right_idx) = node.get_children();
                    
                    // 检查依赖是否就绪
                    let left_ready = left_idx == u32::MAX || !self.node_dirty[left_idx as usize];
                    let right_ready = right_idx == u32::MAX || !self.node_dirty[right_idx as usize];
                    
                    if left_ready && right_ready && node.has_children() {
                        let left_value = if left_idx != u32::MAX {
                            self.node_values[left_idx as usize]
                        } else {
                            [0.0; 4]
                        };
                        
                        let right_value = if right_idx != u32::MAX {
                            self.node_values[right_idx as usize]
                        } else {
                            [0.0; 4]
                        };
                        
                        // 执行操作
                        let result = self.apply_operation(node, left_value, right_value);
                        self.node_values[i] = result;
                        self.node_dirty[i] = false;
                        changed = true;
                        
                        let pipeline = if node.has_state(GpuAstState::IS_COMPUTE) { 
                            "COMPUTE" 
                        } else if node.has_state(GpuAstState::IS_RENDER) { 
                            "RENDER" 
                        } else { 
                            "UNKNOWN" 
                        };
                        
                        // println!("  计算节点[{}]({}): {:?} {} {:?} = {:?}", 
                        //     i, pipeline, left_value, format_op(node.get_op()), right_value, result);
                    }
                }
            }
        }
        
        // println!("计算完成，迭代次数: {}", iterations);
        
        // 检查是否还有脏节点
        let remaining_dirty = self.node_dirty.iter().filter(|&&dirty| dirty).count();
        if remaining_dirty > 0 {
            // println!("警告: 还有 {} 个节点未计算", remaining_dirty);
            for (i, &dirty) in self.node_dirty.iter().enumerate() {
                if dirty {
                    let node = &graph.nodes[i];
                    let (left, right) = node.get_children();
                    // println!("  未计算节点[{}]: op={:?}, children=({}, {})", 
                    //     i, node.get_op(), left, right);
                }
            }
        }
    }
    
    pub fn new() -> Self {
        Self {
            node_values: Vec::new(),
            node_dirty: Vec::new(),
            import_values: HashMap::new(),
            time: 0.0,
        }
    }
    
    pub fn set_import_value(&mut self, name: &str, value: [f32; 4]) {
        // println!("设置导入变量 '{}' = {:?}", name, value);
        self.import_values.insert(name.to_string(), value);
    }

    pub fn set_time(&mut self, time: f32) {
        // println!("设置时间: {}", time);
        self.time = time;
    }

    pub fn execute(&mut self, graph: &GpuAstGraph) -> Vec<[f32; 4]> {
        println!("\n=== 开始最终输出模拟运算 ===");
        println!("节点总数: {}", graph.nodes.len());
        println!("最终输出数: {}", graph.final_outputs.len());
        
        // 初始化存储
        self.node_values = vec![[0.0; 4]; graph.nodes.len()];
        self.node_dirty = vec![true; graph.nodes.len()]; // 标记所有节点为脏
        
        // 执行计算
        self.initialize_leaves(graph);
        self.compute_dirty_nodes(graph);
        
        // 收集最终输出
        self.collect_final_outputs(graph)
    }

          fn initialize_leaves(&mut self, graph: &GpuAstGraph) {
        println!("\n--- 初始化叶子节点 ---");
        
        for (i, node) in graph.nodes.iter().enumerate() {
            if !node.has_children() && node.has_state(GpuAstState::IS_LEAF) {
                if node.get_constant() != 0.0 {
                    // 常量节点 - 根据数据类型正确初始化
                    let constant_value = node.get_constant();
                    let value = match node.get_data_type() {
                        DataType::Scalar => [constant_value, constant_value, constant_value, constant_value],
                        DataType::Vec2 => [constant_value, constant_value, 0.0, 1.0],
                        DataType::Vec3 => [constant_value, constant_value, constant_value, 1.0],
                        DataType::Vec4 => [constant_value, constant_value, constant_value, constant_value],
                    };
                    self.node_values[i] = value;
                    self.node_dirty[i] = false;
                    println!("  常量节点[{}] = {:?} ({:?})", i, value, node.get_data_type());
                } else {
                    // 导入节点 - 修复同上
                    let (import_type, mask) = node.get_import();
                    let value = match import_type {
                        0 => *self.import_values.get("uv").unwrap_or(&[0.5, 0.5, 0.0, 1.0]),
                        1 => {
                            if let Some(name) = get_import_name_from_mask(mask) {
                                match name {
                                    "time" => [self.time, self.time, self.time, self.time], // 时间复制到所有分量
                                    "delta_time" => [0.016, 0.016, 0.016, 0.016],
                                    _ => [0.0, 0.0, 0.0, 0.0],
                                }
                            } else {
                                [self.time, self.time, self.time, self.time]
                            }
                        }
                        _ => [0.0, 0.0, 0.0, 1.0],
                    };
                    
                    self.node_values[i] = value;
                    self.node_dirty[i] = false;
                    
                    let node_type = if node.has_state(GpuAstState::IS_COMPUTE) { "计算导入" } else { "渲染导入" };
                    let import_name = get_import_name_from_mask(mask).unwrap_or("unknown");
                    println!("  {}节点[{}]({}) = {:?} ({:?})", node_type, i, import_name, value, node.get_data_type());
                }
            }
        }
    }


   fn apply_operation(&self, node: &GpuAstNode, left: [f32; 4], right: [f32; 4]) -> [f32; 4] {
        let op = node.get_op();
        let data_type = node.get_data_type();
        
        let mut result = [0.0; 4];
        
        // 根据操作类型执行计算
        match op {
            // 二元操作
            GpuOp::Add | GpuOp::Subtract | GpuOp::Multiply | GpuOp::Divide | 
            GpuOp::Modulo | GpuOp::Pow | GpuOp::GreaterThan | GpuOp::GreaterEqual |
            GpuOp::LessThan | GpuOp::LessEqual | GpuOp::Equal | GpuOp::NotEqual => {
                // 对于二元操作，根据数据类型决定如何计算
                match data_type {
                    DataType::Scalar => {
                        // 标量操作：只计算第一个分量
                        result[0] = self.apply_binary_op_scalar(op, left[0], right[0]);
                        // 标量复制到所有分量（对于颜色输出）
                        for i in 1..4 {
                            result[i] = result[0];
                        }
                    }
                    DataType::Vec2 => {
                        // Vec2操作：计算前两个分量
                        result[0] = self.apply_binary_op_scalar(op, left[0], right[0]);
                        result[1] = self.apply_binary_op_scalar(op, left[1], right[1]);
                        result[2] = 0.0;
                        result[3] = 1.0;
                    }
                    DataType::Vec3 => {
                        // Vec3操作：计算前三个分量
                        result[0] = self.apply_binary_op_scalar(op, left[0], right[0]);
                        result[1] = self.apply_binary_op_scalar(op, left[1], right[1]);
                        result[2] = self.apply_binary_op_scalar(op, left[2], right[2]);
                        result[3] = 1.0;
                    }
                    DataType::Vec4 => {
                        // Vec4操作：计算所有四个分量
                        for i in 0..4 {
                            result[i] = self.apply_binary_op_scalar(op, left[i], right[i]);
                        }
                    }
                }
            }
            
            // 一元操作
            GpuOp::Sin | GpuOp::Cos | GpuOp::Tan | GpuOp::Exp | 
            GpuOp::Log | GpuOp::Sqrt | GpuOp::Abs => {
                // 一元操作只使用左操作数
                match data_type {
                    DataType::Scalar => {
                        result[0] = self.apply_unary_op_scalar(op, left[0]);
                        for i in 1..4 {
                            result[i] = result[0];
                        }
                    }
                    DataType::Vec2 => {
                        result[0] = self.apply_unary_op_scalar(op, left[0]);
                        result[1] = self.apply_unary_op_scalar(op, left[1]);
                        result[2] = 0.0;
                        result[3] = 1.0;
                    }
                    DataType::Vec3 => {
                        result[0] = self.apply_unary_op_scalar(op, left[0]);
                        result[1] = self.apply_unary_op_scalar(op, left[1]);
                        result[2] = self.apply_unary_op_scalar(op, left[2]);
                        result[3] = 1.0;
                    }
                    DataType::Vec4 => {
                        for i in 0..4 {
                            result[i] = self.apply_unary_op_scalar(op, left[i]);
                        }
                    }
                }
            }
            
            GpuOp::Index => {
                // 索引操作：返回左操作数
                result = left;
            }
        }
        
        result
    }
   fn apply_binary_op_scalar(&self, op: GpuOp, left: f32, right: f32) -> f32 {
        match op {
            GpuOp::Add => left + right,
            GpuOp::Subtract => left - right,
            GpuOp::Multiply => left * right,
            GpuOp::Divide => if right != 0.0 { left / right } else { 0.0 },
            GpuOp::Modulo => if right != 0.0 { left % right } else { 0.0 },
            GpuOp::Pow => left.powf(right),
            GpuOp::GreaterThan => if left > right { 1.0 } else { 0.0 },
            GpuOp::GreaterEqual => if left >= right { 1.0 } else { 0.0 },
            GpuOp::LessThan => if left < right { 1.0 } else { 0.0 },
            GpuOp::LessEqual => if left <= right { 1.0 } else { 0.0 },
            GpuOp::Equal => if (left - right).abs() < f32::EPSILON { 1.0 } else { 0.0 },
            GpuOp::NotEqual => if (left - right).abs() >= f32::EPSILON { 1.0 } else { 0.0 },
            _ => left, // 其他操作返回左操作数
        }
    }

        fn apply_unary_op_scalar(&self, op: GpuOp, input: f32) -> f32 {
        match op {
            GpuOp::Sin => input.sin(),
            GpuOp::Cos => input.cos(),
            GpuOp::Tan => input.tan(),
            GpuOp::Exp => input.exp(),
            GpuOp::Log => if input > 0.0 { input.ln() } else { 0.0 },
            GpuOp::Sqrt => if input >= 0.0 { input.sqrt() } else { 0.0 },
            GpuOp::Abs => input.abs(),
            _ => input,
        }
    }
    fn collect_final_outputs(&self, graph: &GpuAstGraph) -> Vec<[f32; 4]> {
        println!("\n--- 收集最终输出 ---");
        
        let mut outputs = Vec::new();
        
        for &output_idx in &graph.final_outputs {
            if (output_idx as usize) < self.node_values.len() {
                let value = self.node_values[output_idx as usize];
                outputs.push(value);
                
                if let Some(node) = graph.nodes.get(output_idx as usize) {
                    println!("  最终输出[{}] = {:?} ({:?})", output_idx, value, node.get_data_type());
                    
                    if node.is_final_output() {
                        println!("    → 渲染颜色: rgba({:.3}, {:.3}, {:.3}, {:.3})", 
                            value[0], value[1], value[2], value[3]);
                    }
                }
            }
        }
        
        outputs
    }
}
// 辅助函数：从导入掩码获取导入名称
fn get_import_name_from_mask(mask: u8) -> Option<&'static str> {
    match mask {
        0b01 => Some("uv"),           // UV坐标
        0b11 => Some("time"),         // 时间
        0b10 => Some("delta_time"),   // 增量时间
        0b100 => Some("buffer"),      // 缓存数据
        _ => None,
    }
}
// 更新辅助函数
fn format_op(op: GpuOp) -> &'static str {
    match op {
        GpuOp::Add => "+",
        GpuOp::Subtract => "-",
        GpuOp::Multiply => "*",
        GpuOp::Divide => "/",
        GpuOp::Modulo => "%",
        GpuOp::Pow => "**",
        GpuOp::GreaterThan => ">",
        GpuOp::GreaterEqual => ">=",
        GpuOp::LessThan => "<",
        GpuOp::LessEqual => "<=",
        GpuOp::Equal => "==",
        GpuOp::NotEqual => "!=",
        GpuOp::Sin => "sin",
        GpuOp::Cos => "cos",
        GpuOp::Tan => "tan",
        GpuOp::Exp => "exp",
        GpuOp::Log => "log",
        GpuOp::Sqrt => "sqrt",
        GpuOp::Abs => "abs",
        GpuOp::Index => "index",
    }
}
// 改进的转换函数，正确建立节点连接
pub fn convert_to_final_output_ast(plan: &MatrixPlan) -> GpuAstGraph {
    let mut graph = GpuAstGraph::new();
    
    // 第一步：创建所有节点并记录索引映射
    let node_indices = create_nodes_from_plan(&mut graph, plan);
    
    // 第二步：根据矩阵操作建立精确的连接
    build_precise_connections(&mut graph, plan, &node_indices);
    
    // 第三步：标记计算和渲染管线
    mark_compute_render_pipeline(&mut graph, plan);
    
    // 第四步：标记最终输出
    mark_final_outputs(&mut graph, plan, &node_indices);
    
    graph
}

// 标记计算和渲染管线
fn mark_compute_render_pipeline(graph: &mut GpuAstGraph, plan: &MatrixPlan) {
    
    // 首先标记所有导入节点的管线类型
    for (i, node) in graph.nodes.iter_mut().enumerate() {
        if node.has_state(GpuAstState::IS_LEAF) && node.get_constant() == 0.0 {
            // 这是导入节点，保持原有的管线标记
            let (import_type, _) = node.get_import();
            if import_type == 1 { // 计算导入
                node.add_state(GpuAstState::IS_COMPUTE | GpuAstState::PRE_COMPUTED);
            }
        }
    }
    
    // 标记计算操作
    for &op_index in &plan.compute_only_ops {
        if op_index < plan.ops.len() {
            if let Some(output_indices) = get_op_output_nodes(&plan.ops[op_index], plan.final_v_len) {
                for output_idx in output_indices {
                    if (output_idx as usize) < graph.nodes.len() {
                        let node = &mut graph.nodes[output_idx as usize];
                        node.add_state(GpuAstState::IS_COMPUTE | GpuAstState::PRE_COMPUTED);
                    }
                }
            }
        }
    }
    
    // 标记渲染操作
    for &op_index in &plan.render_only_ops {
        if op_index < plan.ops.len() {
            if let Some(output_indices) = get_op_output_nodes(&plan.ops[op_index], plan.final_v_len) {
                for output_idx in output_indices {
                    if (output_idx as usize) < graph.nodes.len() {
                        let node = &mut graph.nodes[output_idx as usize];
                        node.add_state(GpuAstState::IS_RENDER);
                    }
                }
            }
        }
    }
    
    // 修复：先收集需要推断的节点信息，然后再修改
    let mut nodes_to_infer: Vec<(usize, u32, u32)> = Vec::new();
    
    // 第一步：收集需要推断的节点信息（只读）
    for (i, node) in graph.nodes.iter().enumerate() {
        if !node.has_state(GpuAstState::IS_COMPUTE | GpuAstState::IS_RENDER) && node.has_children() {
            let (left_idx, right_idx) = node.get_children();
            nodes_to_infer.push((i, left_idx, right_idx));
        }
    }
    
    // 第二步：根据收集的信息进行推断和修改
    for (i, left_idx, right_idx) in nodes_to_infer {
        // 检查输入节点的管线类型（只读访问）
        let mut has_compute_input = false;
        let mut has_render_input = false;
        
        if left_idx != u32::MAX {
            let left_node = &graph.nodes[left_idx as usize];
            if left_node.has_state(GpuAstState::IS_COMPUTE) {
                has_compute_input = true;
            }
            if left_node.has_state(GpuAstState::IS_RENDER) {
                has_render_input = true;
            }
        }
        
        if right_idx != u32::MAX {
            let right_node = &graph.nodes[right_idx as usize];
            if right_node.has_state(GpuAstState::IS_COMPUTE) {
                has_compute_input = true;
            }
            if right_node.has_state(GpuAstState::IS_RENDER) {
                has_render_input = true;
            }
        }
        
        // 根据输入推断管线类型（可变访问）
        let node = &mut graph.nodes[i];
        if has_compute_input && !has_render_input {
            node.add_state(GpuAstState::IS_COMPUTE);
        } else if has_render_input && !has_compute_input {
            node.add_state(GpuAstState::IS_RENDER);
        } else if has_compute_input && has_render_input {
            // 混合管线，标记为渲染（因为最终输出到渲染）
            node.add_state(GpuAstState::IS_RENDER);
        }
    }
}

// 获取操作对应的输出节点索引
fn get_op_output_nodes(mat_op: &MatOp, final_v_len: usize) -> Option<Vec<u32>> {
    match mat_op {
        MatOp::BinaryMat { out_start, rows, .. } => {
            Some((*out_start..*out_start + *rows).map(|i| i as u32).collect())
        }
        MatOp::UnaryMat { out_start, rows, .. } => {
            Some((*out_start..*out_start + *rows).map(|i| i as u32).collect())
        }
        MatOp::CondBlendMat { out_start, rows, .. } => {
            Some((*out_start..*out_start + *rows).map(|i| i as u32).collect())
        }
    }
}

fn build_precise_connections(graph: &mut GpuAstGraph, plan: &MatrixPlan, node_indices: &[u32]) {
    
    for (op_index, mat_op) in plan.ops.iter().enumerate() {
        match mat_op {
            MatOp::BinaryMat { op, left_mat, right_mat, out_start, rows } => {
                
                let left_matrix = &plan.matrices[*left_mat];
                let right_matrix = &plan.matrices[*right_mat];
                
                for row in 0..*rows {
                    let output_idx = out_start + row;
                    if output_idx < node_indices.len() {
                        let output_node_idx = node_indices[output_idx];
                        let output_node = &mut graph.nodes[output_node_idx as usize];
                        
                        // 设置操作类型
                        output_node.set_op(GpuOp::from_binary_op(op));
                        
                        // 根据选择矩阵找到实际的输入节点
                        let left_inputs = find_inputs_from_matrix(left_matrix, row, node_indices);
                        let right_inputs = find_inputs_from_matrix(right_matrix, row, node_indices);
                        
                        // 使用第一个输入作为连接（简化处理）
                        if let Some(&left_input) = left_inputs.first() {
                            if let Some(&right_input) = right_inputs.first() {
                                output_node.set_children(left_input, right_input);
            
                            }
                        }
                    }
                }
            }
            
            MatOp::UnaryMat { func, mat, out_start, rows } => {
                
                let matrix = &plan.matrices[*mat];
                
                for row in 0..*rows {
                    let output_idx = out_start + row;
                    if output_idx < node_indices.len() {
                        let output_node_idx = node_indices[output_idx];
                        let output_node = &mut graph.nodes[output_node_idx as usize];
                        
                        output_node.set_op(GpuOp::from_unary_func(func));
                        
                        let inputs = find_inputs_from_matrix(matrix, row, node_indices);
                        if let Some(&input) = inputs.first() {
                            output_node.set_children(input, u32::MAX);
                        }
                    }
                }
            }
            
            MatOp::CondBlendMat { cond_mat, then_mat, else_mat, out_start, rows } => {
                
                let cond_matrix = &plan.matrices[*cond_mat];
                let then_matrix = &plan.matrices[*then_mat];
                let else_matrix = &plan.matrices[*else_mat];
                
                for row in 0..*rows {
                    let output_idx = out_start + row;
                    if output_idx < node_indices.len() {
                        let output_node_idx = node_indices[output_idx];
                        let output_node = &mut graph.nodes[output_node_idx as usize];
                        
                        output_node.set_op(GpuOp::Add); // 简化处理
                        
                        let cond_inputs = find_inputs_from_matrix(cond_matrix, row, node_indices);
                        let then_inputs = find_inputs_from_matrix(then_matrix, row, node_indices);
                        let else_inputs = find_inputs_from_matrix(else_matrix, row, node_indices);
                        
                        // 使用第一个输入作为连接
                        if let (Some(&cond), Some(&then), Some(&else_)) = 
                            (cond_inputs.first(), then_inputs.first(), else_inputs.first()) {
                            // 简化：只使用前两个输入
                            output_node.set_children(cond, then);
      
                        }
                    }
                }
            }
        }
    }
}

// 根据选择矩阵找到输入节点
fn find_inputs_from_matrix(matrix: &Matrix, row: usize, node_indices: &[u32]) -> Vec<u32> {
    let mut inputs = Vec::new();
    
    if row < matrix.rows {
        for col in 0..matrix.cols {
            if matrix.data[row * matrix.cols + col] != 0.0 && col < node_indices.len() {
                inputs.push(node_indices[col]);
            }
        }
    }
    
    inputs
}
// 修复转换函数，正确设置导入掩码
fn create_nodes_from_plan(graph: &mut GpuAstGraph, plan: &MatrixPlan) -> Vec<u32> {
    let mut node_indices = Vec::new();
    
    // 为每个V位置创建节点
    for i in 0..plan.final_v_len {
        let mut node = GpuAstNode::new();
        
        // 检查是否为常量
        if let Some(&(_, value)) = plan.constant_values.iter().find(|&&(idx, _)| idx == i) {
            node.set_constant(value);
            node.set_state(GpuAstState::IS_LEAF);
            node.set_data_type(DataType::Scalar);
        }
        
        // 检查是否为导入 - 正确设置导入类型和掩码
        if let Some(import_info) = plan.imports.iter().find(|imp| imp.index == i) {
            let (import_type, import_name, mask) = match &import_info.import_type {
                ImportType::Render(name) => (0, *name, import_info.mask as u8),
                ImportType::Compute(name) => (1, *name, import_info.mask as u8),
            };
            
            node.set_import(import_type, mask);
            node.set_state(GpuAstState::IS_LEAF);
            
            // 根据导入类型正确设置状态
            match &import_info.import_type {
                ImportType::Render(_) => {
                    node.add_state(GpuAstState::IS_RENDER);
                    // println!("  创建渲染导入节点[{}]: {} (mask: {})", i, import_name, mask);
                }
                ImportType::Compute(_) => {
                    node.add_state(GpuAstState::IS_COMPUTE);
                    // println!("  创建计算导入节点[{}]: {} (mask: {})", i, import_name, mask);
                }
            }
            
            node.set_data_type(DataType::Vec4);
        }
        
        node_indices.push(graph.add_node(node));
    }
    
    node_indices
}


// 更新转换函数中的操作设置
fn build_connections(graph: &mut GpuAstGraph, plan: &MatrixPlan, node_indices: &[u32]) {
    for mat_op in &plan.ops {
        match mat_op {
            MatOp::BinaryMat { op, left_mat, right_mat, out_start, rows } => {
                for i in 0..*rows {
                    let output_idx = out_start + i;
                    if output_idx < node_indices.len() {
                        let node_idx = node_indices[output_idx];
                        let node = &mut graph.nodes[node_idx as usize];
                        
                        // 使用新的转换方法
                        let gpu_op = GpuOp::from_binary_op(op);
                        node.set_op(gpu_op);
                        
                        // ... 其余代码保持不变 ...
                    }
                }
            }
            MatOp::UnaryMat { func, mat, out_start, rows } => {
                for i in 0..*rows {
                    let output_idx = out_start + i;
                    if output_idx < node_indices.len() {
                        let node_idx = node_indices[output_idx];
                        let node = &mut graph.nodes[node_idx as usize];
                        
                        // 使用新的转换方法
                        let gpu_op = GpuOp::from_unary_func(func);
                        node.set_op(gpu_op);
                        
                        // ... 其余代码保持不变 ...
                    }
                }
            }

            // 处理其他操作类型...
            _ => {}
        }
    }
}

fn mark_final_outputs(graph: &mut GpuAstGraph, plan: &MatrixPlan, node_indices: &[u32]) {
    for &output_idx in &plan.top_outputs {
        if output_idx < node_indices.len() {
            let node_idx = node_indices[output_idx];
            let node = &mut graph.nodes[node_idx as usize];
            
            // 最终输出总是Vec4类型
            node.set_data_type(DataType::Vec4);
            node.add_state(GpuAstState::IS_FINAL_OUTPUT | GpuAstState::IS_TICK);
            
            graph.mark_final_output(node_idx);
            
            // println!("标记最终输出节点[{}]为Vec4", output_idx);
        }
    }
}


#[cfg(test)]
mod ast_computation_tests {
    use super::*;
    use crate::core::{BinaryOp, Expr, UnaryFunc, Vec2, Vec3, Vec4, dsl::{var, wvec4}};

  // 修复导入注册，确保掩码正确设置
fn create_test_registry() -> crate::mat::op::ImportRegistry {
    let mut registry = crate::mat::op::ImportRegistry::new();
    
    // 使用不同的掩码来区分不同的导入类型
    registry.register_render_import(
        "uv",
        0b01,  // 掩码01表示UV
        Box::new(|input| vec![input[0], input[1], 0.0, 1.0])
    );
    
    registry.register_compute_import(
        "time", 
        0b11,  // 掩码11表示时间
        Box::new(|input| vec![input[0], 0.0, 0.0, 0.0])
    );
    
    registry.register_compute_import(
        "delta_time",
        0b10,  // 掩码10表示增量时间
        Box::new(|input| vec![input[0], 0.0, 0.0, 0.0])
    );
    
    registry
}

// 添加调试测试来验证导入值设置
#[test]
fn test_import_values_debug() {
    use crate::dsl::*;
    // println!("\n=== 调试导入值设置 ===");
    
    let registry = create_test_registry();
    
    // 测试表达式: time * 3.14
    let expr = wvec4(1.0, cv("time") * 3.14, 2.0, 3.0);

    let matrix_plan = crate::mat::op::compile_to_matrix_plan_with_imports(&expr, &registry);
    
    // println!("导入信息: {:?}", matrix_plan.imports);
    
    // 转换为 GPU AST Graph
    let graph = convert_to_final_output_ast(&matrix_plan);
    
    // 执行计算，设置时间
    let mut simulator = FinalOutputSimulator::new();
    simulator.set_time(2.5); // 设置时间值
    
    let results = simulator.execute(&graph);
    
    // 应该得到: 2.5 * 3.14 = 7.85
    // println!("计算结果: {:?}", results);
    if !results.is_empty() {
        let expected = 2.5 * 3.14;
        let actual = results[0][0];
        // println!("期望: {}, 实际: {}", expected, actual);
        assert!((actual - expected).abs() < 0.001, "计算结果不正确");
    }
}
// 更新测试，添加调试信息
#[test]
fn test_vec4_output_correctness() {
    use crate::dsl::*;
    println!("\n=== 测试Vec4输出正确性 ===");
    
    let registry = create_test_registry();
    
    // 创建明确的Vec4表达式
    let expr = wvec4(
        cv("time") * 3.14,  // x分量: time * 3.14
        cv("time") * 2.0,   // y分量: time * 2.0  
        cv("time") * 1.5,   // z分量: time * 1.5
        1.0             // w分量: 1.0
    );

    let matrix_plan = crate::mat::op::compile_to_matrix_plan_with_imports(&expr, &registry);
    
    // 转换为 GPU AST Graph
    let graph = convert_to_final_output_ast(&matrix_plan);
    println!("当前GPUAstGraph {:?}",graph);
    // 执行计算，设置时间
    let mut simulator = FinalOutputSimulator::new();
    simulator.set_time(2.0); // 设置时间值
    
    let results = simulator.execute(&graph);
    
    // 验证结果: vec4(2.0*3.14, 2.0*2.0, 2.0*1.5, 1.0)
    assert_eq!(results.len(), 4); // Vec4有4个分量输出
    
    let expected_x = 2.0 * 3.14;
    let expected_y = 2.0 * 2.0;
    let expected_z = 2.0 * 1.5;
    let expected_w = 1.0;
    
    // println!("期望: vec4({}, {}, {}, {})", expected_x, expected_y, expected_z, expected_w);
    // println!("实际: vec4({}, {}, {}, {})", results[0][0], results[1][0], results[2][0], results[3][0]);
    
    assert!((results[0][0] - expected_x).abs() < 0.001, "x分量不正确");
    assert!((results[1][0] - expected_y).abs() < 0.001, "y分量不正确");
    assert!((results[2][0] - expected_z).abs() < 0.001, "z分量不正确");
    assert!((results[3][0] - expected_w).abs() < 0.001, "w分量不正确");
    
    println!("Vec4输出测试通过!");
}
}