use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

use crate::{
    core::{BinaryOp, UnaryFunc},
    mat::op::{ImportType, MatOp, Matrix, MatrixPlan},
};

bitflags! {
    /// GPU AST Node State
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct GpuAstState: u32 {
        const IS_COMPUTE     = 0b00000001;      // 璁＄畻绠＄嚎鑺傜偣
        const IS_RENDER      = 0b00000010;      // 娓叉煋绠＄嚎鑺傜偣
        const COMPUTE_OVER   = 0b00000100;      // 璁＄畻宸插畬鎴?
        const IS_TICK        = 0b00001000;      // 闇€瑕佹瘡甯ф洿鏂?
        const IS_LEAF        = 0b00010000;      // 鍙跺瓙鑺傜偣锛堝父閲?瀵煎叆锛?
        const IS_BRANCH      = 0b00100000;      // 鍒嗘敮鑺傜偣
        const NEEDS_UPDATE   = 0b01000000;      // 闇€瑕佹洿鏂?
        const IS_DIRTY       = 0b10000000;      // 鏁版嵁宸茶剰
        const IS_FINAL_OUTPUT = 0b100000000;    // 鏈€缁堣緭鍑?
        const PRE_COMPUTED   = 0b1000000000;    // 棰勫厛鍦╟ompute璁＄畻
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuOp {
    Add = 0b000000000001,
    Subtract = 0b000000000010,
    Multiply = 0b000000000100,
    Divide = 0b000000001000,
    Modulo = 0b000000010000,
    Pow = 0b000000100000,
    GreaterThan = 0b000001000000,
    GreaterEqual = 0b000010000000,
    LessThan = 0b000100000000,
    LessEqual = 0b001000000000,
    Equal = 0b010000000000,
    NotEqual = 0b100000000000,
    Index = 0b1000000000000,
    Sin = 0b10000000000000,
    Cos = 0b100000000000000,
    Tan = 0b1000000000000000,
    Exp = 0b10000000000000000,
    Log = 0b100000000000000000,
    Sqrt = 0b1000000000000000000,
    Abs = 0b10000000000000000000,
    Conditional = 0b100000000000000000000,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
/// 鑺傜偣鏁版嵁绫诲瀷
pub enum DataType {
    Scalar = 0, // 鏍囬噺
    Vec2 = 1,   // 浜岀淮鍚戦噺
    Vec3 = 2,   // 涓夌淮鍚戦噺
    Vec4 = 3,   // 鍥涚淮鍚戦噺
}

/// 浼樺寲鐨?GPU AST 鑺傜偣
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuAstNode {
    // 鏁版嵁閮ㄥ垎 (16瀛楄妭)
    data: [f32; 4],

    // 绫诲瀷鍜屾帶鍒堕儴鍒?(16瀛楄妭)
    state: u32,
    op: u32,        // 瀛樺偍 GpuOp 鐨?u32 鍊?
    data_type: u32, // 瀛樺偍 DataType 鐨?u32 鍊?

    // 杩炴帴淇℃伅 (8瀛楄妭)
    left_child: u32,
    right_child: u32,

    // 瀵煎叆鍜屽父閲忎俊鎭?(8瀛楄妭)
    import_info: u32,
    constant_value: f32,
    else_child: u32,
}

// 涓?GpuOp 瀹炵幇涓€浜涜緟鍔╂柟娉?
impl GpuOp {
    pub fn is_arithmetic(&self) -> bool {
        matches!(
            self,
            GpuOp::Add
                | GpuOp::Subtract
                | GpuOp::Multiply
                | GpuOp::Divide
                | GpuOp::Modulo
                | GpuOp::Pow
        )
    }

    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            GpuOp::GreaterThan
                | GpuOp::GreaterEqual
                | GpuOp::LessThan
                | GpuOp::LessEqual
                | GpuOp::Equal
                | GpuOp::NotEqual
        )
    }

    pub fn is_unary(&self) -> bool {
        matches!(
            self,
            GpuOp::Sin
                | GpuOp::Cos
                | GpuOp::Tan
                | GpuOp::Exp
                | GpuOp::Log
                | GpuOp::Sqrt
                | GpuOp::Abs
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

    // 浠?u32 杞崲鍥?GpuOp锛堝畨鍏ㄧ増鏈級
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
            0b100000000000000000000 => Some(GpuOp::Conditional),
            _ => None,
        }
    }
}
// 涓?DataType 瀹炵幇杞崲鏂规硶
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

    // 鑾峰彇瀵煎叆鍚嶇О锛堢畝鍖栫増鏈紝瀹為檯闇€瑕佷粠瀵煎叆淇℃伅涓В鏋愶級
    pub fn get_import_name(&self) -> Option<&'static str> {
        if self.has_state(GpuAstState::IS_LEAF) && self.get_constant() == 0.0 {
            // 鏍规嵁瀵煎叆绫诲瀷杩斿洖鍚嶇О
            let (import_type, _) = self.get_import();
            match import_type {
                0 => Some("uv"),   // 娓叉煋瀵煎叆
                1 => Some("time"), // 璁＄畻瀵煎叆
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
            else_child: u32::MAX,
        }
    }

    // 鐘舵€佹搷浣?
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

    // 鎿嶄綔璁剧疆
    pub fn set_op(&mut self, op: GpuOp) {
        self.op = op as u32;
    }

    pub fn get_op(&self) -> GpuOp {
        GpuOp::from_u32(self.op).unwrap_or(GpuOp::Add)
    }

    // 鏁版嵁绫诲瀷璁剧疆
    pub fn set_data_type(&mut self, data_type: DataType) {
        self.data_type = data_type as u32;
    }

    pub fn get_data_type(&self) -> DataType {
        DataType::from_u32(self.data_type).unwrap_or(DataType::Scalar)
    }

    // 鏁版嵁璁块棶鏂规硶淇濇寔涓嶅彉...
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

    // 鍏朵粬鏂规硶淇濇寔涓嶅彉...
    pub fn set_children(&mut self, left: u32, right: u32) {
        self.left_child = left;
        self.right_child = right;
        self.else_child = u32::MAX;
    }

    pub fn get_children(&self) -> (u32, u32) {
        (self.left_child, self.right_child)
    }

    pub fn has_children(&self) -> bool {
        self.left_child != u32::MAX || self.right_child != u32::MAX || self.else_child != u32::MAX
    }

    pub fn set_else_child(&mut self, else_child: u32) {
        self.else_child = else_child;
    }

    pub fn get_else_child(&self) -> u32 {
        self.else_child
    }

    pub fn set_conditional_children(&mut self, condition: u32, then_child: u32, else_child: u32) {
        self.left_child = condition;
        self.right_child = then_child;
        self.else_child = else_child;
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
        self.has_state(GpuAstState::IS_FINAL_OUTPUT) && self.get_data_type() == DataType::Vec4
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

/// GPU AST 鍥?- 鍖呭惈鎵€鏈夎妭鐐瑰拰鏈€缁堣緭鍑轰俊鎭?
#[derive(Clone, Debug)]
pub struct GpuAstGraph {
    pub nodes: Vec<GpuAstNode>,
    pub final_outputs: Vec<u32>,   // 鏈€缁堣緭鍑鸿妭鐐圭殑绱㈠紩
    pub compute_outputs: Vec<u32>, // 璁＄畻杈撳嚭鑺傜偣鐨勭储寮?
    pub render_outputs: Vec<u32>,  // 娓叉煋杈撳嚭鑺傜偣鐨勭储寮?
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

            // 鏍规嵁鐘舵€佸垎绫?
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
        self.final_outputs
            .iter()
            .filter_map(|&idx| self.nodes.get(idx as usize))
            .collect()
    }

    pub fn validate(&self) -> bool {
        // 妫€鏌ユ墍鏈夋渶缁堣緭鍑洪兘鏄疺ec4绫诲瀷
        for &output_idx in &self.final_outputs {
            if let Some(node) = self.nodes.get(output_idx as usize) {
                if node.get_data_type() != DataType::Vec4 {
                    return false;
                }
            }
        }

        // 妫€鏌ヨ妭鐐硅繛鎺ユ湁鏁堟€?
        for (i, node) in self.nodes.iter().enumerate() {
            let (left, right) = node.get_children();
            let else_child = node.get_else_child();

            if left != u32::MAX && (left as usize) >= self.nodes.len() {
                return false;
            }
            if right != u32::MAX && (right as usize) >= self.nodes.len() {
                return false;
            }
            if else_child != u32::MAX && (else_child as usize) >= self.nodes.len() {
                return false;
            }
        }

        true
    }
}

/// 閽堝鏈€缁堣緭鍑簐ec4浼樺寲鐨凜PU妯℃嫙鍣?
pub struct FinalOutputSimulator {
    node_values: Vec<[f32; 4]>, // 姣忎釜鑺傜偣鐨勫綋鍓嶅€?
    node_dirty: Vec<bool>,      // 鑴忔爣璁帮紝鐢ㄤ簬澧為噺璁＄畻
    import_values: HashMap<String, [f32; 4]>,
    time: f32,
}

impl FinalOutputSimulator {
    fn compute_dirty_nodes(&mut self, graph: &GpuAstGraph) {
        let mut changed = true;
        let mut iterations = 0;

        while changed && iterations < graph.nodes.len() * 2 {
            changed = false;
            iterations += 1;

            for (i, node) in graph.nodes.iter().enumerate() {
                if self.node_dirty[i] {
                    let (left_idx, right_idx) = node.get_children();
                    let else_idx = node.get_else_child();

                    let left_ready = left_idx == u32::MAX || !self.node_dirty[left_idx as usize];
                    let right_ready = right_idx == u32::MAX || !self.node_dirty[right_idx as usize];
                    let else_ready = else_idx == u32::MAX || !self.node_dirty[else_idx as usize];

                    if left_ready && right_ready && else_ready && node.has_children() {
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

                        let else_value = if else_idx != u32::MAX {
                            self.node_values[else_idx as usize]
                        } else {
                            [0.0; 4]
                        };

                        let result = if node.get_op() == GpuOp::Conditional {
                            self.apply_conditional(node, left_value, right_value, else_value)
                        } else {
                            self.apply_operation(node, left_value, right_value)
                        };

                        self.node_values[i] = result;
                        self.node_dirty[i] = false;
                        changed = true;
                    }
                }
            }
        }

        let remaining_dirty = self.node_dirty.iter().filter(|&&dirty| dirty).count();
        if remaining_dirty > 0 {
            for (i, &dirty) in self.node_dirty.iter().enumerate() {
                if dirty {
                    let node = &graph.nodes[i];
                    let (left, right) = node.get_children();
                    let else_child = node.get_else_child();
                    let _ = (left, right, else_child);
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
        // println!("璁剧疆瀵煎叆鍙橀噺 '{}' = {:?}", name, value);
        self.import_values.insert(name.to_string(), value);
    }

    pub fn set_time(&mut self, time: f32) {
        // println!("璁剧疆鏃堕棿: {}", time);
        self.time = time;
    }

    pub fn execute(&mut self, graph: &GpuAstGraph) -> Vec<[f32; 4]> {
        println!("\n=== 寮€濮嬫渶缁堣緭鍑烘ā鎷熻繍绠?===");
        println!("鑺傜偣鎬绘暟: {}", graph.nodes.len());
        println!("鏈€缁堣緭鍑烘暟: {}", graph.final_outputs.len());

        // 鍒濆鍖栧瓨鍌?
        self.node_values = vec![[0.0; 4]; graph.nodes.len()];
        self.node_dirty = vec![true; graph.nodes.len()]; // 鏍囪鎵€鏈夎妭鐐逛负鑴?

        // 鎵ц璁＄畻
        self.initialize_leaves(graph);
        self.compute_dirty_nodes(graph);

        // 鏀堕泦鏈€缁堣緭鍑?
        self.collect_final_outputs(graph)
    }

    fn initialize_leaves(&mut self, graph: &GpuAstGraph) {
        println!("\n--- 鍒濆鍖栧彾瀛愯妭鐐?---");

        for (i, node) in graph.nodes.iter().enumerate() {
            if !node.has_children() && node.has_state(GpuAstState::IS_LEAF) {
                if node.get_constant() != 0.0 {
                    // 甯搁噺鑺傜偣 - 鏍规嵁鏁版嵁绫诲瀷姝ｇ‘鍒濆鍖?
                    let constant_value = node.get_constant();
                    let value = match node.get_data_type() {
                        DataType::Scalar => [
                            constant_value,
                            constant_value,
                            constant_value,
                            constant_value,
                        ],
                        DataType::Vec2 => [constant_value, constant_value, 0.0, 1.0],
                        DataType::Vec3 => [constant_value, constant_value, constant_value, 1.0],
                        DataType::Vec4 => [
                            constant_value,
                            constant_value,
                            constant_value,
                            constant_value,
                        ],
                    };
                    self.node_values[i] = value;
                    self.node_dirty[i] = false;
                    println!(
                        "  甯搁噺鑺傜偣[{}] = {:?} ({:?})",
                        i,
                        value,
                        node.get_data_type()
                    );
                } else {
                    // 瀵煎叆鑺傜偣 - 淇鍚屼笂
                    let (import_type, mask) = node.get_import();
                    let value = match import_type {
                        0 => *self
                            .import_values
                            .get("uv")
                            .unwrap_or(&[0.5, 0.5, 0.0, 1.0]),
                        1 => {
                            if let Some(name) = get_import_name_from_mask(mask) {
                                match name {
                                    "time" => [self.time, self.time, self.time, self.time], // 鏃堕棿澶嶅埗鍒版墍鏈夊垎閲?
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

                    let node_type = if node.has_state(GpuAstState::IS_COMPUTE) {
                        "璁＄畻瀵煎叆"
                    } else {
                        "娓叉煋瀵煎叆"
                    };
                    let import_name = get_import_name_from_mask(mask).unwrap_or("unknown");
                    println!(
                        "  {}鑺傜偣[{}]({}) = {:?} ({:?})",
                        node_type,
                        i,
                        import_name,
                        value,
                        node.get_data_type()
                    );
                }
            }
        }
    }

    fn apply_operation(&self, node: &GpuAstNode, left: [f32; 4], right: [f32; 4]) -> [f32; 4] {
        let op = node.get_op();
        let data_type = node.get_data_type();

        let mut result = [0.0; 4];

        // 鏍规嵁鎿嶄綔绫诲瀷鎵ц璁＄畻
        match op {
            GpuOp::Add
            | GpuOp::Subtract
            | GpuOp::Multiply
            | GpuOp::Divide
            | GpuOp::Modulo
            | GpuOp::Pow
            | GpuOp::GreaterThan
            | GpuOp::GreaterEqual
            | GpuOp::LessThan
            | GpuOp::LessEqual
            | GpuOp::Equal
            | GpuOp::NotEqual => {
                // 瀵逛簬浜屽厓鎿嶄綔锛屾牴鎹暟鎹被鍨嬪喅瀹氬浣曡绠?
                match data_type {
                    DataType::Scalar => {
                        // 鏍囬噺鎿嶄綔锛氬彧璁＄畻绗竴涓垎閲?
                        result[0] = self.apply_binary_op_scalar(op, left[0], right[0]);
                        // 鏍囬噺澶嶅埗鍒版墍鏈夊垎閲忥紙瀵逛簬棰滆壊杈撳嚭锛?
                        for i in 1..4 {
                            result[i] = result[0];
                        }
                    }
                    DataType::Vec2 => {
                        // Vec2鎿嶄綔锛氳绠楀墠涓や釜鍒嗛噺
                        result[0] = self.apply_binary_op_scalar(op, left[0], right[0]);
                        result[1] = self.apply_binary_op_scalar(op, left[1], right[1]);
                        result[2] = 0.0;
                        result[3] = 1.0;
                    }
                    DataType::Vec3 => {
                        // Vec3鎿嶄綔锛氳绠楀墠涓変釜鍒嗛噺
                        result[0] = self.apply_binary_op_scalar(op, left[0], right[0]);
                        result[1] = self.apply_binary_op_scalar(op, left[1], right[1]);
                        result[2] = self.apply_binary_op_scalar(op, left[2], right[2]);
                        result[3] = 1.0;
                    }
                    DataType::Vec4 => {
                        // Vec4鎿嶄綔锛氳绠楁墍鏈夊洓涓垎閲?
                        for i in 0..4 {
                            result[i] = self.apply_binary_op_scalar(op, left[i], right[i]);
                        }
                    }
                }
            }
            GpuOp::Sin
            | GpuOp::Cos
            | GpuOp::Tan
            | GpuOp::Exp
            | GpuOp::Log
            | GpuOp::Sqrt
            | GpuOp::Abs => {
                // 涓€鍏冩搷浣滃彧浣跨敤宸︽搷浣滄暟
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
                // 绱㈠紩鎿嶄綔锛氳繑鍥炲乏鎿嶄綔鏁?
                result = left;
            }
            GpuOp::Conditional => unreachable!("conditional evaluation handled separately"),
        }

        result
    }

    fn apply_conditional(
        &self,
        node: &GpuAstNode,
        condition: [f32; 4],
        then_value: [f32; 4],
        else_value: [f32; 4],
    ) -> [f32; 4] {
        match node.get_data_type() {
            DataType::Scalar => {
                let cond_active = condition[0].abs() > f32::EPSILON;
                let value = if cond_active {
                    then_value[0]
                } else {
                    else_value[0]
                };
                [value; 4]
            }
            DataType::Vec2 => {
                let mut result = [0.0; 4];
                for lane in 0..2 {
                    let cond_active = condition[lane].abs() > f32::EPSILON;
                    result[lane] = if cond_active {
                        then_value[lane]
                    } else {
                        else_value[lane]
                    };
                }
                result[2] = 0.0;
                result[3] = 1.0;
                result
            }
            DataType::Vec3 => {
                let mut result = [0.0; 4];
                for lane in 0..3 {
                    let cond_active = condition[lane].abs() > f32::EPSILON;
                    result[lane] = if cond_active {
                        then_value[lane]
                    } else {
                        else_value[lane]
                    };
                }
                result[3] = 1.0;
                result
            }
            DataType::Vec4 => {
                let mut result = [0.0; 4];
                for lane in 0..4 {
                    let cond_active = condition[lane].abs() > f32::EPSILON;
                    result[lane] = if cond_active {
                        then_value[lane]
                    } else {
                        else_value[lane]
                    };
                }
                result
            }
        }
    }
    fn apply_binary_op_scalar(&self, op: GpuOp, left: f32, right: f32) -> f32 {
        match op {
            GpuOp::Add => left + right,
            GpuOp::Subtract => left - right,
            GpuOp::Multiply => left * right,
            GpuOp::Divide => {
                if right != 0.0 {
                    left / right
                } else {
                    0.0
                }
            }
            GpuOp::Modulo => {
                if right != 0.0 {
                    left % right
                } else {
                    0.0
                }
            }
            GpuOp::Pow => left.powf(right),
            GpuOp::GreaterThan => {
                if left > right {
                    1.0
                } else {
                    0.0
                }
            }
            GpuOp::GreaterEqual => {
                if left >= right {
                    1.0
                } else {
                    0.0
                }
            }
            GpuOp::LessThan => {
                if left < right {
                    1.0
                } else {
                    0.0
                }
            }
            GpuOp::LessEqual => {
                if left <= right {
                    1.0
                } else {
                    0.0
                }
            }
            GpuOp::Equal => {
                if (left - right).abs() < f32::EPSILON {
                    1.0
                } else {
                    0.0
                }
            }
            GpuOp::NotEqual => {
                if (left - right).abs() >= f32::EPSILON {
                    1.0
                } else {
                    0.0
                }
            }
            _ => left, // 鍏朵粬鎿嶄綔杩斿洖宸︽搷浣滄暟
        }
    }

    fn apply_unary_op_scalar(&self, op: GpuOp, input: f32) -> f32 {
        match op {
            GpuOp::Sin => input.sin(),
            GpuOp::Cos => input.cos(),
            GpuOp::Tan => input.tan(),
            GpuOp::Exp => input.exp(),
            GpuOp::Log => {
                if input > 0.0 {
                    input.ln()
                } else {
                    0.0
                }
            }
            GpuOp::Sqrt => {
                if input >= 0.0 {
                    input.sqrt()
                } else {
                    0.0
                }
            }
            GpuOp::Abs => input.abs(),
            _ => input,
        }
    }
    fn collect_final_outputs(&self, graph: &GpuAstGraph) -> Vec<[f32; 4]> {
        println!("\n--- 鏀堕泦鏈€缁堣緭鍑?---");

        let mut outputs = Vec::new();

        for &output_idx in &graph.final_outputs {
            if (output_idx as usize) < self.node_values.len() {
                let value = self.node_values[output_idx as usize];
                outputs.push(value);

                if let Some(node) = graph.nodes.get(output_idx as usize) {
                    println!(
                        "  鏈€缁堣緭鍑篬{}] = {:?} ({:?})",
                        output_idx,
                        value,
                        node.get_data_type()
                    );

                    if node.is_final_output() {
                        println!(
                            "    鈫?娓叉煋棰滆壊: rgba({:.3}, {:.3}, {:.3}, {:.3})",
                            value[0], value[1], value[2], value[3]
                        );
                    }
                }
            }
        }

        outputs
    }
}
// 杈呭姪鍑芥暟锛氫粠瀵煎叆鎺╃爜鑾峰彇瀵煎叆鍚嶇О
fn get_import_name_from_mask(mask: u8) -> Option<&'static str> {
    match mask {
        0b01 => Some("uv"),         // UV鍧愭爣
        0b11 => Some("time"),       // 鏃堕棿
        0b10 => Some("delta_time"), // 澧為噺鏃堕棿
        0b100 => Some("buffer"),    // 缂撳瓨鏁版嵁
        _ => None,
    }
}
// 鏇存柊杈呭姪鍑芥暟
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
        GpuOp::Conditional => "conditional",
    }
}
// 鏀硅繘鐨勮浆鎹㈠嚱鏁帮紝姝ｇ‘寤虹珛鑺傜偣杩炴帴
pub fn convert_to_final_output_ast(plan: &MatrixPlan) -> GpuAstGraph {
    let mut graph = GpuAstGraph::new();

    // 绗竴姝ワ細鍒涘缓鎵€鏈夎妭鐐瑰苟璁板綍绱㈠紩鏄犲皠
    let node_indices = create_nodes_from_plan(&mut graph, plan);

    // 绗簩姝ワ細鏍规嵁鐭╅樀鎿嶄綔寤虹珛绮剧‘鐨勮繛鎺?
    build_precise_connections(&mut graph, plan, &node_indices);

    // 绗笁姝ワ細鏍囪璁＄畻鍜屾覆鏌撶绾?
    mark_compute_render_pipeline(&mut graph, plan);

    // 绗洓姝ワ細鏍囪鏈€缁堣緭鍑?
    mark_final_outputs(&mut graph, plan, &node_indices);

    graph
}

// 鏍囪璁＄畻鍜屾覆鏌撶绾?
fn mark_compute_render_pipeline(graph: &mut GpuAstGraph, plan: &MatrixPlan) {
    // 棣栧厛鏍囪鎵€鏈夊鍏ヨ妭鐐圭殑绠＄嚎绫诲瀷
    for (i, node) in graph.nodes.iter_mut().enumerate() {
        if node.has_state(GpuAstState::IS_LEAF) && node.get_constant() == 0.0 {
            // 杩欐槸瀵煎叆鑺傜偣锛屼繚鎸佸師鏈夌殑绠＄嚎鏍囪
            let (import_type, _) = node.get_import();
            if import_type == 1 {
                // 璁＄畻瀵煎叆
                node.add_state(GpuAstState::IS_COMPUTE | GpuAstState::PRE_COMPUTED);
            }
        }
    }

    // 鏍囪璁＄畻鎿嶄綔
    for &op_index in &plan.compute_only_ops {
        if op_index < plan.ops.len() {
            if let Some(output_indices) = get_op_output_nodes(&plan.ops[op_index], plan.final_v_len)
            {
                for output_idx in output_indices {
                    if (output_idx as usize) < graph.nodes.len() {
                        let node = &mut graph.nodes[output_idx as usize];
                        node.add_state(GpuAstState::IS_COMPUTE | GpuAstState::PRE_COMPUTED);
                    }
                }
            }
        }
    }

    // 鏍囪娓叉煋鎿嶄綔
    for &op_index in &plan.render_only_ops {
        if op_index < plan.ops.len() {
            if let Some(output_indices) = get_op_output_nodes(&plan.ops[op_index], plan.final_v_len)
            {
                for output_idx in output_indices {
                    if (output_idx as usize) < graph.nodes.len() {
                        let node = &mut graph.nodes[output_idx as usize];
                        node.add_state(GpuAstState::IS_RENDER);
                    }
                }
            }
        }
    }

    // 淇锛氬厛鏀堕泦闇€瑕佹帹鏂殑鑺傜偣淇℃伅锛岀劧鍚庡啀淇敼
    let mut nodes_to_infer: Vec<(usize, u32, u32, u32)> = Vec::new();

    // 绗竴姝ワ細鏀堕泦闇€瑕佹帹鏂殑鑺傜偣淇℃伅锛堝彧璇伙級
    for (i, node) in graph.nodes.iter().enumerate() {
        if !node.has_state(GpuAstState::IS_COMPUTE | GpuAstState::IS_RENDER) && node.has_children()
        {
            let (left_idx, right_idx) = node.get_children();
            let else_idx = node.get_else_child();
            nodes_to_infer.push((i, left_idx, right_idx, else_idx));
        }
    }

    // 绗簩姝ワ細鏍规嵁鏀堕泦鐨勪俊鎭繘琛屾帹鏂拰淇敼
    for (i, left_idx, right_idx, else_idx) in nodes_to_infer {
        // 妫€鏌ヨ緭鍏ヨ妭鐐圭殑绠＄嚎绫诲瀷锛堝彧璇昏闂級
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

        if else_idx != u32::MAX {
            let else_node = &graph.nodes[else_idx as usize];
            if else_node.has_state(GpuAstState::IS_COMPUTE) {
                has_compute_input = true;
            }
            if else_node.has_state(GpuAstState::IS_RENDER) {
                has_render_input = true;
            }
        }

        // 鏍规嵁杈撳叆鎺ㄦ柇绠＄嚎绫诲瀷锛堝彲鍙樿闂級
        let node = &mut graph.nodes[i];
        if has_compute_input && !has_render_input {
            node.add_state(GpuAstState::IS_COMPUTE);
        } else if has_render_input && !has_compute_input {
            node.add_state(GpuAstState::IS_RENDER);
        } else if has_compute_input && has_render_input {
            // 娣峰悎绠＄嚎锛屾爣璁颁负娓叉煋锛堝洜涓烘渶缁堣緭鍑哄埌娓叉煋锛?
            node.add_state(GpuAstState::IS_RENDER);
        }
    }
}

// 鑾峰彇鎿嶄綔瀵瑰簲鐨勮緭鍑鸿妭鐐圭储寮?
fn get_op_output_nodes(mat_op: &MatOp, final_v_len: usize) -> Option<Vec<u32>> {
    match mat_op {
        MatOp::BinaryMat {
            out_start, rows, ..
        } => Some((*out_start..*out_start + *rows).map(|i| i as u32).collect()),
        MatOp::UnaryMat {
            out_start, rows, ..
        } => Some((*out_start..*out_start + *rows).map(|i| i as u32).collect()),
        MatOp::CondBlendMat {
            out_start, rows, ..
        } => Some((*out_start..*out_start + *rows).map(|i| i as u32).collect()),
    }
}

fn build_precise_connections(graph: &mut GpuAstGraph, plan: &MatrixPlan, node_indices: &[u32]) {
    for (op_index, mat_op) in plan.ops.iter().enumerate() {
        match mat_op {
            MatOp::BinaryMat {
                op,
                left_mat,
                right_mat,
                out_start,
                rows,
            } => {
                let left_matrix = &plan.matrices[*left_mat];
                let right_matrix = &plan.matrices[*right_mat];

                for row in 0..*rows {
                    let output_idx = out_start + row;
                    if output_idx < node_indices.len() {
                        let output_node_idx = node_indices[output_idx];
                        let output_node = &mut graph.nodes[output_node_idx as usize];

                        // 璁剧疆鎿嶄綔绫诲瀷
                        output_node.set_op(GpuOp::from_binary_op(op));

                        // 鏍规嵁閫夋嫨鐭╅樀鎵惧埌瀹為檯鐨勮緭鍏ヨ妭鐐?
                        let left_inputs = find_inputs_from_matrix(left_matrix, row, node_indices);
                        let right_inputs = find_inputs_from_matrix(right_matrix, row, node_indices);

                        // 浣跨敤绗竴涓緭鍏ヤ綔涓鸿繛鎺ワ紙绠€鍖栧鐞嗭級
                        if let Some(&left_input) = left_inputs.first() {
                            if let Some(&right_input) = right_inputs.first() {
                                output_node.set_children(left_input, right_input);
                            }
                        }
                    }
                }
            }

            MatOp::UnaryMat {
                func,
                mat,
                out_start,
                rows,
            } => {
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

            MatOp::CondBlendMat {
                cond_mat,
                then_mat,
                else_mat,
                out_start,
                rows,
            } => {
                let cond_matrix = &plan.matrices[*cond_mat];
                let then_matrix = &plan.matrices[*then_mat];
                let else_matrix = &plan.matrices[*else_mat];

                for row in 0..*rows {
                    let output_idx = out_start + row;
                    if output_idx < node_indices.len() {
                        let output_node_idx = node_indices[output_idx];
                        let output_node = &mut graph.nodes[output_node_idx as usize];

                        output_node.set_op(GpuOp::Conditional);
                        output_node.add_state(GpuAstState::IS_BRANCH);

                        let cond_inputs = find_inputs_from_matrix(cond_matrix, row, node_indices);
                        let then_inputs = find_inputs_from_matrix(then_matrix, row, node_indices);
                        let else_inputs = find_inputs_from_matrix(else_matrix, row, node_indices);

                        // 浣跨敤绗竴涓緭鍏ヤ綔涓鸿繛鎺?
                        if let (Some(&cond), Some(&then), Some(&else_)) = (
                            cond_inputs.first(),
                            then_inputs.first(),
                            else_inputs.first(),
                        ) {
                            // 绠€鍖栵細鍙娇鐢ㄥ墠涓や釜杈撳叆
                            output_node.set_conditional_children(cond, then, else_);
                        }
                    }
                }
            }
        }
    }
}

// 鏍规嵁閫夋嫨鐭╅樀鎵惧埌杈撳叆鑺傜偣
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
// 淇杞崲鍑芥暟锛屾纭缃鍏ユ帺鐮?
fn create_nodes_from_plan(graph: &mut GpuAstGraph, plan: &MatrixPlan) -> Vec<u32> {
    let mut node_indices = Vec::new();

    // 涓烘瘡涓猇浣嶇疆鍒涘缓鑺傜偣
    for i in 0..plan.final_v_len {
        let mut node = GpuAstNode::new();

        // 妫€鏌ユ槸鍚︿负甯搁噺
        if let Some(&(_, value)) = plan.constant_values.iter().find(|&&(idx, _)| idx == i) {
            node.set_constant(value);
            node.set_state(GpuAstState::IS_LEAF);
            node.set_data_type(DataType::Scalar);
        }

        // 妫€鏌ユ槸鍚︿负瀵煎叆 - 姝ｇ‘璁剧疆瀵煎叆绫诲瀷鍜屾帺鐮?
        if let Some(import_info) = plan.imports.iter().find(|imp| imp.index == i) {
            let (import_type, import_name, mask) = match &import_info.import_type {
                ImportType::Render(name) => (0, *name, import_info.mask as u8),
                ImportType::Compute(name) => (1, *name, import_info.mask as u8),
            };

            node.set_import(import_type, mask);
            node.set_state(GpuAstState::IS_LEAF);

            // 鏍规嵁瀵煎叆绫诲瀷姝ｇ‘璁剧疆鐘舵€?
            match &import_info.import_type {
                ImportType::Render(_) => {
                    node.add_state(GpuAstState::IS_RENDER);
                    // println!("  鍒涘缓娓叉煋瀵煎叆鑺傜偣[{}]: {} (mask: {})", i, import_name, mask);
                }
                ImportType::Compute(_) => {
                    node.add_state(GpuAstState::IS_COMPUTE);
                    // println!("  鍒涘缓璁＄畻瀵煎叆鑺傜偣[{}]: {} (mask: {})", i, import_name, mask);
                }
            }

            node.set_data_type(DataType::Vec4);
        }

        node_indices.push(graph.add_node(node));
    }

    node_indices
}

// 鏇存柊杞崲鍑芥暟涓殑鎿嶄綔璁剧疆
fn build_connections(graph: &mut GpuAstGraph, plan: &MatrixPlan, node_indices: &[u32]) {
    for mat_op in &plan.ops {
        match mat_op {
            MatOp::BinaryMat {
                op,
                left_mat,
                right_mat,
                out_start,
                rows,
            } => {
                for i in 0..*rows {
                    let output_idx = out_start + i;
                    if output_idx < node_indices.len() {
                        let node_idx = node_indices[output_idx];
                        let node = &mut graph.nodes[node_idx as usize];

                        // 浣跨敤鏂扮殑杞崲鏂规硶
                        let gpu_op = GpuOp::from_binary_op(op);
                        node.set_op(gpu_op);

                        // ... 鍏朵綑浠ｇ爜淇濇寔涓嶅彉 ...
                    }
                }
            }
            MatOp::UnaryMat {
                func,
                mat,
                out_start,
                rows,
            } => {
                for i in 0..*rows {
                    let output_idx = out_start + i;
                    if output_idx < node_indices.len() {
                        let node_idx = node_indices[output_idx];
                        let node = &mut graph.nodes[node_idx as usize];

                        // 浣跨敤鏂扮殑杞崲鏂规硶
                        let gpu_op = GpuOp::from_unary_func(func);
                        node.set_op(gpu_op);

                        // ... 鍏朵綑浠ｇ爜淇濇寔涓嶅彉 ...
                    }
                }
            }

            // 澶勭悊鍏朵粬鎿嶄綔绫诲瀷...
            _ => {}
        }
    }
}

fn mark_final_outputs(graph: &mut GpuAstGraph, plan: &MatrixPlan, node_indices: &[u32]) {
    for &output_idx in &plan.top_outputs {
        if output_idx < node_indices.len() {
            let node_idx = node_indices[output_idx];
            let node = &mut graph.nodes[node_idx as usize];

            // 鏈€缁堣緭鍑烘€绘槸Vec4绫诲瀷
            node.set_data_type(DataType::Vec4);
            node.add_state(GpuAstState::IS_FINAL_OUTPUT | GpuAstState::IS_TICK);

            graph.mark_final_output(node_idx);

            // println!("鏍囪鏈€缁堣緭鍑鸿妭鐐筟{}]涓篤ec4", output_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_node(value: f32) -> GpuAstNode {
        let mut node = GpuAstNode::new();
        node.set_constant(value);
        node.set_state(GpuAstState::IS_LEAF | GpuAstState::IS_COMPUTE);
        node
    }

    #[test]
    fn conditional_branch_selects_then_value() {
        let mut graph = GpuAstGraph::new();

        let condition_idx = graph.add_node(constant_node(1.0));
        let then_idx = graph.add_node(constant_node(5.0));
        let else_idx = graph.add_node(constant_node(-5.0));

        let mut node = GpuAstNode::new();
        node.set_op(GpuOp::Conditional);
        node.set_conditional_children(condition_idx, then_idx, else_idx);
        node.set_data_type(DataType::Scalar);
        node.set_state(GpuAstState::IS_COMPUTE);
        let conditional_idx = graph.add_node(node);
        graph.mark_final_output(conditional_idx);

        let mut simulator = FinalOutputSimulator::new();
        let results = simulator.execute(&graph);

        assert!((results[0][0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn conditional_branch_selects_else_value() {
        let mut graph = GpuAstGraph::new();

        let condition_idx = graph.add_node(constant_node(f32::EPSILON / 2.0));
        let then_idx = graph.add_node(constant_node(5.0));
        let else_idx = graph.add_node(constant_node(-3.0));

        let mut node = GpuAstNode::new();
        node.set_op(GpuOp::Conditional);
        node.set_conditional_children(condition_idx, then_idx, else_idx);
        node.set_data_type(DataType::Scalar);
        node.set_state(GpuAstState::IS_COMPUTE);
        let conditional_idx = graph.add_node(node);
        graph.mark_final_output(conditional_idx);

        let mut simulator = FinalOutputSimulator::new();
        let results = simulator.execute(&graph);

        assert!((results[0][0] + 3.0).abs() < 1e-6);
    }
}
