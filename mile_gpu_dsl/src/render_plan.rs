// 在 pipeline.rs 中添加

// -------------------- 动态 RenderPlan 存储结构 --------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RenderPlanHeader {
    pub plan_count: u32,      // 实际计划数量
    pub dirty_flags: u32,     // 脏标记
    pub frame_index: u32,     // 帧索引
    pub _padding: u32,        // 填充
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct RenderPlanStorage {
    header: RenderPlanHeader,
    plans: Vec<RenderPlan>,   // 动态数量的渲染计划
    max_plans: usize,         // 最大支持的计划数量
    buffer_size: usize,       // 总缓冲区大小（字节）
}

impl RenderPlanStorage {
    pub fn new(max_plans: usize) -> Self {
        let buffer_size = std::mem::size_of::<RenderPlanHeader>() + 
                         max_plans * std::mem::size_of::<RenderPlan>();
        
        Self {
            header: RenderPlanHeader {
                plan_count: 0,
                dirty_flags: 0,
                frame_index: 0,
                _padding: 0,
            },
            plans: Vec::with_capacity(max_plans),
            max_plans,
            buffer_size,
        }
    }

    // 添加或更新渲染计划
    pub fn update_plan(&mut self, index: usize, plan: RenderPlan) -> bool {
        if index >= self.max_plans {
            return false;
        }

        // 确保有足够的容量
        if index >= self.plans.len() {
            self.plans.resize(index + 1, RenderPlan::default());
            self.header.plan_count = self.plans.len() as u32;
        }

        self.plans[index] = plan;
        self.set_dirty_flag(index);
        true
    }

    // 设置脏标记
    pub fn set_dirty_flag(&mut self, index: usize) {
        if index < 32 {
            self.header.dirty_flags |= 1 << index;
        }
    }

    // 清除所有脏标记
    pub fn clear_dirty_flags(&mut self) {
        self.header.dirty_flags = 0;
    }

    // 检查是否有脏标记
    pub fn has_dirty_plans(&self) -> bool {
        self.header.dirty_flags != 0
    }

    // 获取需要更新的计划索引
    pub fn get_dirty_plan_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        for i in 0..self.max_plans.min(32) {
            if (self.header.dirty_flags & (1 << i)) != 0 {
                indices.push(i);
            }
        }
        indices
    }

    // 递增帧索引
    pub fn increment_frame(&mut self) {
        self.header.frame_index = self.header.frame_index.wrapping_add(1);
    }

    // 获取序列化后的字节数据（使用 bytemuck::cast_slice）
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.buffer_size);

        // 1. 添加头部
        bytes.extend_from_slice(bytemuck::cast_slice(&[self.header]));

        // 2. 添加所有渲染计划
        bytes.extend_from_slice(bytemuck::cast_slice(&self.plans));

        // 3. 填充到总缓冲区大小
        if bytes.len() < self.buffer_size {
            bytes.extend_from_slice(&vec![0u8; self.buffer_size - bytes.len()]);
        }

        bytes
    }

    // 获取当前数据大小（字节）
    pub fn data_size(&self) -> usize {
        self.buffer_size
    }

    // 获取计划数量
    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }

    // 获取最大计划数量
    pub fn max_plans(&self) -> usize {
        self.max_plans
    }

    // 获取帧索引
    pub fn frame_index(&self) -> u32 {
        self.header.frame_index
    }
}

// -------------------- RenderPlan 管理器 --------------------

pub struct RenderPlanManager {
    pub storage: RenderPlanStorage,
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub needs_rebind: bool,
}

impl RenderPlanManager {
    pub fn new(device: &wgpu::Device, max_plans: usize) -> Self {
        let storage = RenderPlanStorage::new(max_plans);
        let buffer_size = storage.data_size() as u64;

        // 创建存储缓冲区
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_plan_storage_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 创建绑定组布局
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render_plan_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(buffer_size).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        // 创建绑定组
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_plan_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            storage,
            buffer,
            bind_group,
            bind_group_layout,
            needs_rebind: false,
        }
    }

    // 更新渲染计划
    pub fn update_plan(&mut self, index: usize, plan: RenderPlan) -> bool {
        if self.storage.update_plan(index, plan) {
            self.needs_rebind = true;
            true
        } else {
            false
        }
    }

    // 上传数据到GPU
    pub fn upload_to_gpu(&mut self, queue: &wgpu::Queue) {
        if self.storage.has_dirty_plans() {
            let data = self.storage.as_bytes();
            queue.write_buffer(&self.buffer, 0, &data);
            self.storage.clear_dirty_flags();
            self.storage.increment_frame();
            self.needs_rebind = false;
            
            println!("上传了 {} 个渲染计划到GPU, 帧索引: {}", 
                self.storage.plan_count(), self.storage.frame_index());
        }
    }

    // 检查是否需要重新绑定
    pub fn needs_rebind(&self) -> bool {
        self.needs_rebind
    }

    // 获取绑定组
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    // 获取绑定组布局
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &bind_group_layout
    }
}

// -------------------- 在 GpuKennel 中集成 --------------------

impl GpuKennel {
    pub fn create_render_plan_manager(device: &wgpu::Device, max_plans: usize) -> RenderPlanManager {
        RenderPlanManager::new(device, max_plans)
    }

    // 在设置计划时更新渲染计划管理器
    pub fn set_plan_with_manager(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &MatrixPlan,
        lanes: u32,
        inputs: &[Vec<f32>],
        plan_manager: &mut RenderPlanManager,
        plan_index: usize,
    ) -> (MileResultDes, RenderPlan) {
        let (result, render_plan) = self.set_plan_layered(device, queue, plan, lanes, inputs);
        
        // 更新渲染计划管理器
        plan_manager.update_plan(plan_index, render_plan);
        
        (result, render_plan)
    }
}