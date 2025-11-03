use crate::{
    CpuPanelEvent, GpuUi, NetWorkTransition, PANEL_ID, Panel, StateTransition,
    TransformAnimFieldInfo, UIEventHub, UiInteractionScope,
    structs::{
        AnimOp, CollectionId, CollectionSampling, EasingMask, EntryState, PanelEvent, PanelField,
        PanelInteraction, RelLayoutMask,
    },
    ui_network::{Collection, Rel, collection_by_name, rel_by_name},
};
use flume::Sender;
use glam::{Vec2, vec2, vec4};
use mile_api::{ModuleEventType, ModuleParmas};
use mile_gpu_dsl::{
    core::{
        Expr, dsl::{IF, sin, wvec4},
    },
    dsl::*,
};
use std::{
    any::Any,
    cell::{Cell, RefCell},
    collections::{self, HashMap},
    default,
    fmt::{self, Debug},
    marker::PhantomData,
    path::Path,
    rc::Rc,
    sync::Arc,
};
use wgpu::naga::keywords::wgsl::RESERVED;

pub type StateId = u32;

const WGSL_TIME: f32 = 9999999.0; // 与 Rust 的 MAX_TIME_SEC 对应
const WGSL_SIN_TIME: f32 = 9999999.1; // 与 Rust 的 MAX_TIME_SEC 对应
const PRE_POSITION_X: f32 = 9999988.0; // 与 Rust 的 MAX_TIME_SEC 对应
const PRE_POSITION_Y: f32 = 9999989.0; // 与 Rust 的 MAX_TIME_SEC 对应

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum Call {
    DRAG,
    CLICK,
    HOVER,
    Defualt,
    VISIBLE,
    OUT,
}

#[derive(Debug, Clone, Default)]
pub enum ExitCollectionOp {
    #[default]
    ExitAllOldCollection,
    ExitRangeOldCollection(Vec<String>),
}

pub struct StateNetWorkBinding<'a, T> {
    pub state_id: u32,
    pub mui: &'a Mui<T>,
}

impl<'b, T> StateNetWorkBinding<'b, T>
where
    T: 'static,
{
    fn exit(mut self) -> &'b Mui<T> {
        self.mui
    }

    // pub fn register_all(&self) {
    //     let mui = self.mui;
    //     let state_id = self.state_id;
    //     let send  = mui.emit.clone();
    //     let mut configs = mui.pending_net_work.state_net_work.borrow_mut();
    //     if let Some(config) = configs.get_mut(&state_id){
    //         let wrapped_cb: Box<dyn FnMut(u32)> = Box::new(move |panel_id| {
    //             send.send(CpuPanelEvent::NetWorkTransition(
    //                 NetWorkTransition {
    //                     state_config_des: config.to_des(),
    //                     curr_state: UiState(state_id),
    //                     panel_id,
    //                 }
    //             ));
    //         });
    //         config.call.push(wrapped_cb);
    //     }
    // }

    fn set_collection(mut self, id: u32) -> Self {
        let mui = self.mui;
        {
            let mut configs = mui.pending_net_work.state_net_work.borrow_mut();
            let state_network_config = configs.get_mut(&self.state_id).unwrap();
            state_network_config.insert_collection = Some(id);
        }
        self
    }

    pub fn transform_entry_anim_type(mut self) -> Self {
        let mui = self.mui;
        {
            let mut configs = mui.pending_net_work.state_net_work.borrow_mut();
            let state_network_config = configs.get_mut(&self.state_id).unwrap();
            state_network_config.immediately_anim = false;
        }
        self
    }

    fn exit_collection(mut self, op: ExitCollectionOp) -> Self {
        let mui = self.mui;
        {
            let mut configs = mui.pending_net_work.state_net_work.borrow_mut();
            let state_network_config = configs.get_mut(&self.state_id).unwrap();
            state_network_config.exit_collection = Some(op);
        }
        self
    }
}

#[derive(Default)]
/// PendingCallbacks 按状态存储
pub struct PendingStateNetWorkBinding {
    pub current_state: Cell<StateId>,
    pub state_net_work: Rc<RefCell<HashMap<StateId, StateNetWorkConfig>>>,
}

pub struct StateNetWorkConfig {
    pub call: Vec<Box<dyn FnMut(u32) + 'static>>,
    insert_collection: Option<u32>,
    exit_collection: Option<ExitCollectionOp>,
    immediately_anim: bool,
}

impl Default for StateNetWorkConfig {
    fn default() -> Self {
        Self {
            call: Default::default(),
            insert_collection: Default::default(),
            exit_collection: Default::default(),
            immediately_anim: true,
        }
    }
}

impl fmt::Debug for StateNetWorkConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateNetWorkConfig")
            .field("insert_collection", &self.insert_collection)
            .finish()
    }
}

impl StateNetWorkConfig {
    fn to_des(&self) -> StateNetWorkConfigDes {
        StateNetWorkConfigDes {
            insert_collection: self.insert_collection.clone(),
            exit_collection: self.exit_collection.clone(),
            immediately_anim: self.immediately_anim,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StateNetWorkConfigDes {
    pub insert_collection: Option<u32>,
    pub exit_collection: Option<ExitCollectionOp>,
    pub immediately_anim: bool,
}

impl From<Call> for PanelInteraction {
    fn from(c: Call) -> Self {
        match c {
            Call::DRAG => PanelInteraction::DRAGGABLE,
            Call::CLICK => PanelInteraction::CLICKABLE,
            Call::HOVER => PanelInteraction::HOVER,
            Call::HOVER => PanelInteraction::HOVER,
            Call::Defualt => PanelInteraction::DEFUALT,
            Call::VISIBLE => PanelInteraction::VISIBLE,
            Call::OUT => PanelInteraction::Out,
        }
    }
}

///
pub struct StateConfig<T> {
    pub state_id: StateId,
    pub size: Option<Vec2>,
    pub with_image_size: bool,
    pub pos: Option<Vec2>,
    pub offset: Option<Vec2>,
    pub z_index: Option<u32>,
    pub texture_id: Option<String>,
    pub call: HashMap<Call, Box<dyn FnMut(u32) + 'static>>,
    pub entry_frag_vertex: Vec<Box<dyn FnMut(u32) + 'static>>,
    pub vertex: Option<Box<dyn FnMut(&mut T, u32) -> Expr + 'static>>,
    pub frag: Option<Box<dyn FnMut(&mut T, u32) -> Expr + 'static>>,
}

#[derive(Debug, Clone)]
pub struct StateConfigDes {
    pub state_id: StateId,
    pub size: Option<Vec2>,
    pub pos: Option<Vec2>,
    pub offset: Option<Vec2>,
    pub texture_id: Option<String>,
    pub open_api: Vec<Call>,
    pub is_open_frag: bool,
    pub is_open_vertex: bool,
}

impl<T> StateConfig<T> {
    fn to_des(&self) -> StateConfigDes {
        StateConfigDes {
            state_id: self.state_id,
            size: self.size,
            pos: self.pos,
            texture_id: self.texture_id.clone(),
            open_api: self.call.keys().cloned().collect::<Vec<_>>(),
            offset: self.offset,
            is_open_frag: self.frag.is_some(),
            is_open_vertex: self.vertex.is_some(),
        }
    }
}

// 手动实现 Debug，忽略 `call`
impl<T> fmt::Debug for StateConfig<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateConfig")
            .field("size", &self.size)
            .field("position", &self.pos)
            .field("texture_id", &self.texture_id)
            .finish()
    }
}

impl<T> Default for StateConfig<T> {
    fn default() -> Self {
        Self {
            size: None,
            call: HashMap::new(),
            ..Default::default()
        }
    }
}

/// PendingCallbacks 按状态存储
pub struct PendingCallbacks<T> {
    pub current_state: Cell<StateId>,
    pub states: RefCell<HashMap<StateId, StateConfig<T>>>,
}

impl<T> Default for PendingCallbacks<T> {
    fn default() -> Self {
        Self {
            current_state: Cell::new(0),
            states: RefCell::new(HashMap::new()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UiState(pub StateId);

pub struct Mui<T> {
    pub pending: PendingCallbacks<T>,
    pub pending_net_work: PendingStateNetWorkBinding,
    pub panel_id: u32,
    pub data: Rc<RefCell<T>>,
    pub default_state: UiState,
    pub emit: Sender<CpuPanelEvent>,
    pub state_config: Rc<RefCell<HashMap<StateId, StateConfigDes>>>,
    pub ui_out: Option<Box<dyn FnMut(u32) + 'static>>,
}

#[derive(Clone, Debug)]
pub enum WgslResult {
    FragResult(FragResult),
    VertexResult(VertexResult),
}

#[derive(Clone, Debug)]
pub struct FragResult {
    pub panel_id: u32,
    pub slot: usize,
    pub color: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct VertexResult {
    pub panel_id: u32,
    pub slot: usize,
    pub vertex: [f32; 4],
}

pub struct MuiGrid<T> {
    item: Vec<Mui<T>>,
}

impl<T> Mui<T>
where
    T: std::fmt::Debug + 'static,
{
    /// 获取状态字段，字段可以是 size / position / texture_id 等
    /// 如果当前状态没有设置，就 fallback 到默认状态
    pub fn get_field<F, R>(&self, state_id: StateId, field: F) -> R
    where
        F: Fn(&StateConfig<T>) -> Option<R>,
        R: Clone,
    {
        let states = self.pending.states.borrow();
        let default_cfg = states.get(&0u32).expect("默认状态必须存在");

        let cfg = states.get(&state_id).unwrap_or(default_cfg);

        field(cfg)
            .or_else(|| field(default_cfg))
            .expect("字段在默认状态必须存在")
    }

    /// 批量生成网格面板
    /// rows, cols: 行列数量
    /// spacing: x/y 间距
    // pub fn grid(
    //     input: Rc<RefCell<T>>,
    //     send: Sender<CpuPanelEvent>,
    //     rows: u32,
    //     cols: u32,
    //     spacing: Vec2,
    //     start_id: u32,
    // ) -> MuiGrid<T> {
    //     let mut muis = Vec::new();

    //     for r in 0..rows {
    //         for c in 0..cols {
    //             let mut mui = Mui::new(input.clone(), send.clone());
    //             mui.panel_id = start_id + r * cols + c;

    //             // 默认位置按网格排列
    //             // 假设 State 0 默认状态
    //             let pos = vec2(c as f32 * spacing.x, r as f32 * spacing.y);
    //             mui
    //                 .pos(pos);

    //             muis.push(mui);
    //         }
    //     }

    //     MuiGrid { item: muis }
    // }

    pub fn new(input: Rc<RefCell<T>>, send: Sender<CpuPanelEvent>) -> Self {
        Self {
            pending: PendingCallbacks::default(),
            panel_id: 0,
            data: input,
            default_state: UiState(0),
            emit: send,
            state_config: Rc::new(RefCell::new(HashMap::new())),
            pending_net_work: PendingStateNetWorkBinding::default(),
            ui_out: None,
        }
    }

    pub fn vertex<F>(&self, f: F) -> &Self
    where
        F: Fn(&mut T, u32) -> Expr + 'static,
    {
        let curr_state = self.pending.current_state.get();
        let mut pending = self.pending.states.borrow_mut();
        let mut config_o = pending.get_mut(&curr_state);
        if let Some(config) = config_o {
            config.vertex = Some(Box::new(f));
        };
        self
    }

    pub fn frag<F>(&self, f: F) -> &Self
    where
        F: Fn(&mut T, u32) -> Expr + 'static,
    {
        let curr_state = self.pending.current_state.get();
        let mut pending = self.pending.states.borrow_mut();
        let mut config_o = pending.get_mut(&curr_state);
        if let Some(config) = config_o {
            config.frag = Some(Box::new(f));
        };
        self
    }

    pub fn default_state(&mut self, state: UiState) -> &Self {
        self.default_state = state;
        self
    }

    /// 切换当前状态
    pub fn state(&self, state: UiState) -> &Self {
        self.pending.current_state.set(state.0);
        self.pending_net_work.current_state.set(state.0);

        let mut _state = self.pending.states.borrow_mut();

        _state.insert(
            state.0,
            StateConfig {
                call: HashMap::new(),
                offset: None,
                size: None,
                pos: None,
                texture_id: None,
                state_id: state.0,
                entry_frag_vertex: vec![],
                z_index: Some(0),
                with_image_size: false,
                vertex: None,
                frag: None,
            },
        );
        // self
        self
    }

    pub fn texture(&self, path: &str) -> &Self {
        let state = &self.pending.current_state.get();
        if let Some(cfg) = self.pending.states.borrow_mut().get_mut(state) {
            cfg.texture_id = Some(path.to_string());
        }
        self
    }

    pub fn pos(&self, vec: Vec2) -> &Self {
        let state = &self.pending.current_state.get();
        if let Some(cfg) = self.pending.states.borrow_mut().get_mut(state) {
            cfg.pos = Some(vec);
        }
        self
    }

    /**
     * 相对之前的状态坐标偏移多少
     */
    pub fn offset(&self, vec: Vec2) -> &Self {
        let state = &self.pending.current_state.get();
        if let Some(cfg) = self.pending.states.borrow_mut().get_mut(state) {
            cfg.offset = Some(vec);
        }
        self
    }

    pub fn z_index(&self, z_index: u32) -> &Self {
        let state = &self.pending.current_state.get();
        if let Some(cfg) = self.pending.states.borrow_mut().get_mut(state) {
            cfg.z_index = Some(z_index);
        }
        self
    }

    /// 设置 size
    pub fn size_with_image(&self) -> &Self {
        let state = &self.pending.current_state.get();
        if let Some(cfg) = self.pending.states.borrow_mut().get_mut(state) {
            cfg.with_image_size = true
        }
        self
    }

    /// 设置 size
    pub fn size(&self, width: f32, height: f32) -> &Self {
        let state = &self.pending.current_state.get();
        if let Some(cfg) = self.pending.states.borrow_mut().get_mut(state) {
            cfg.size = Some(Vec2::new(width, height));
        }
        self
    }

    pub fn on<'b>(&'b self) -> CallbackBuilder<T> {
        CallbackBuilder {
            state_id: self.pending.current_state.get(),
            mui: self,
            next_state: vec![],
            pending_callbacks: RefCell::new(Vec::new()),
        }
    }

    pub fn net_work<'b>(&'b self) -> StateNetWorkBinding<T> {
        let mui = &self.pending_net_work;
        let curr_state_id = self.pending_net_work.current_state.get();
        {
            let mut configs = mui.state_net_work.borrow_mut();
            configs.insert(curr_state_id, StateNetWorkConfig::default());
        }
        StateNetWorkBinding {
            state_id: self.pending_net_work.current_state.get(),
            mui: self,
        }
    }

    // pub fn on_drag<'b>(&'b mut self) -> CallbackBuilder<T>
    // {
    //     CallbackBuilder {
    //         state_id: self.pending.current_state,
    //         mui: self,
    //         next_state: None,
    //     }
    // }

    fn register_all_call(&self) {
        let states = self.pending.states.borrow_mut();

        for (state, config) in states.iter() {}
    }

    pub fn net_work_build(&self, panel_id: u32) {
        // let defualt_state = &self.default_state.0;
        // let net_work_config = self.pending_net_work.state_net_work.borrow();
        // let config_o: Option<&StateNetWorkConfig> = net_work_config.get(defualt_state);

        // let mut gpu_ui_clone = gpu_ui.clone();
        // let gpu_ui = gpu_ui_clone.borrow_mut();
        // let mut net_work = gpu_ui.ui_net_work.borrow_mut();
        // let mut gpu_net_work_ids = gpu_ui.gpu_network_ids.borrow_mut();
        // if let Some(config_o) = config_o{
        //     for insert_collection_name in &config_o.insert_collection{
        //         net_work.add_item_to_collection_by_name(&insert_collection_name, panel_id,&mut gpu_net_work_ids, queue);
        //     }
        // }

        let default_state = self.default_state.0;
        let net_work_config = self.pending_net_work.state_net_work.borrow_mut();
        let config = net_work_config.get(&default_state);

        if let Some(init_state_config) = config {
            self.emit
                .send(CpuPanelEvent::NetWorkTransition(NetWorkTransition {
                    state_config_des: init_state_config.to_des(),
                    curr_state: UiState(default_state),
                    panel_id: panel_id,
                }));
        }
    }

    pub fn build(&self, gui_ui: Arc<RefCell<GpuUi>>, queue: &wgpu::Queue, device: &wgpu::Device) {
        let defualt_state = &self.default_state.0;
        let mut transfrom_config_des: StateConfigDes;
        {
            let des_mut = self.state_config.borrow();
            let config_des_copy = des_mut.get(&self.default_state.0).cloned().unwrap();
            transfrom_config_des = config_des_copy.clone();
            drop(des_mut);
        }

        let states = self.pending.states.borrow();

        let mut ui = gui_ui.borrow_mut();

        let texture_path = self.get_field(*defualt_state, |e| e.texture_id.clone());

        let raw_image_info = ui
            .ui_texture_map
            .get_index_by_path(texture_path.as_str())
            .unwrap();

        let position = self.get_field(*defualt_state, |e| e.pos);

        let with_image_size = self.get_field(*defualt_state, |e| Some(e.with_image_size));

        let size = if with_image_size {
            vec2(raw_image_info.width as f32, raw_image_info.height as f32)
        } else {
            self.get_field(*defualt_state, |e| e.size)
        };

        let z_index = self.get_field(*defualt_state, |e| e.z_index);

        let config_des = self.state_config.borrow_mut();

        let default_des = config_des.get(defualt_state).expect("Mui没有默认配置");

        let mut mask = PanelInteraction::DEFUALT;
        for c in &default_des.open_api {
            mask |= (*c).into(); // 将每个 Call 转成对应 bitflags 并合并
        }

        // println!("纹理数据 {:?}",texture_path);

        // println!("我们拿到的id {:?}",texture_id);

        let panel = Panel {
            position: position.into(),
            size: size.into(),
            uv_offset: [0.0, 0.0],
            uv_scale: [1.0, 1.0],
            z_index,
            pass_through: 0,
            id: 0,
            interaction: mask.bits(),
            event_mask: PanelEvent::Defualt.bits(),
            state_mask: PanelEvent::Defualt.bits(),
            transparent: 1.0,
            texture_id: raw_image_info.index,
            state: *defualt_state,
            collection_state: 0,
            kennel_des_id: u32::MAX,
            pad_1: 0,
        };

        println!("创造了新的面板 {:?}", panel);

        drop(states);
        {
            let curr_id = ui.add_instance(device, queue, panel);

            // 临时收集所有要注册的回调
            let mut to_register: Vec<(Call, Box<dyn FnMut(u32) + 'static>, u32)> = Vec::new();
            let mut to_entry_register: Vec<(Box<dyn FnMut(u32) + 'static>, u32)> = Vec::new();
            let mut to_out_register: Vec<(Box<dyn FnMut(u32) + 'static>, u32)> = Vec::new();

            let mut to_frag_register = RefCell::new(Vec::new());
            let mut to_vertex_register = RefCell::new(Vec::new());

            let mut states = self.pending.states.borrow_mut();

            for (_state_id, state_cfg) in states.iter_mut() {
                let offset_o = state_cfg.offset.clone();
                let emit = self.emit.clone();
                let state_des = state_cfg.to_des();

                if let Some(frag) = state_cfg.frag.take() {
                    {
                        let mut to_frag_register = to_frag_register.borrow_mut();
                        to_frag_register.push((frag, *_state_id));
                    }
                }

                if let Some(vertex) = state_cfg.vertex.take() {
                    let mut to_vertex_register = to_vertex_register.borrow_mut();
                    to_vertex_register.push((vertex, *_state_id));
                }

                to_out_register.push((
                    Box::new(move |panel_id| {
                        let mut des = state_des.clone();
                        println!("当前触发的panel_id {:?}", panel_id);

                        if let Some(offset) = offset_o {
                            let _ = emit.send(CpuPanelEvent::SpecielAnim((
                                panel_id,
                                crate::TransformAnimFieldInfo {
                                    field_id: (PanelField::POSITION_X | PanelField::POSITION_Y)
                                        .bits(),
                                    start_value: vec![0.0; 2],
                                    target_value: vec![PRE_POSITION_X, PRE_POSITION_Y],
                                    duration: 0.33,
                                    easing: EasingMask::LINEAR,
                                    op: AnimOp::SET,
                                    hold: 1,
                                    delay: 0.0,
                                    loop_count: 0,
                                    ping_pong: 0,
                                    on_complete: 1,
                                },
                            )));
                        }
                    }),
                    *_state_id,
                ));

                for (call_type, cb) in state_cfg.call.drain() {
                    println!("在这个状态{} 注册了call {:?}", *_state_id, call_type);
                    to_register.push((call_type, cb, *_state_id));
                }
                for (cb) in state_cfg.entry_frag_vertex.drain(..) {
                    to_entry_register.push((cb, *_state_id));
                }
            }

            // 退出 states 的 RefMut
            drop(states);

            for (mut cb, state) in to_entry_register {
                ui.panel_interaction_trigger
                    .entry_callbacks
                    .entry(UiInteractionScope {
                        panel_id: curr_id,
                        state,
                    })
                    .or_insert_with(Vec::new)
                    .push(cb);
            }

            for (mut cb, state) in to_out_register {
                ui.panel_interaction_trigger
                    .out_callbacks
                    .entry(UiInteractionScope {
                        panel_id: curr_id,
                        state,
                    })
                    .or_insert_with(Vec::new)
                    .push(cb);
            }

            for (mut cb, state) in to_frag_register.borrow_mut().drain(..) {
                let data_ref = self.data.clone();
                println!("在{}状态注册了自定义frag", state);
                let emit = ui.global_hub.sender.clone();
                let wrap: Box<dyn FnMut(u32)> = Box::new(move |panel_id: PANEL_ID| {
                    // 数据的可变借用

                    let mut data = data_ref.borrow_mut();
                    let expr: Expr = cb(&mut data, panel_id); // 使用闭包
                    let _ = emit.send(mile_api::ModuleEvent::KennelPush(ModuleParmas {
                        module_name: "mile_ui_wgsl",
                        idx: panel_id,
                        data: expr,
                        _ty: (ModuleEventType::Push | ModuleEventType::Frag).bits(),
                    }));
                });
                ui.panel_interaction_trigger
                    .frag_callbacks
                    .entry(UiInteractionScope {
                        panel_id: curr_id,
                        state,
                    })
                    .or_insert_with(Vec::new)
                    .push(wrap);
            }

            for (mut cb, state) in to_vertex_register.borrow_mut().drain(..) {
                let data_ref = self.data.clone();
                println!("在{}状态注册了自定义vertex", state);
                let emit = ui.global_hub.sender.clone();
                let wrap: Box<dyn FnMut(u32)> = Box::new(move |panel_id: PANEL_ID| {
                    // 数据的可变借用
                    let mut data = data_ref.borrow_mut();

                    let expr: Expr = cb(&mut data, panel_id); // 使用闭包
                    let _ = emit.send(mile_api::ModuleEvent::KennelPush(ModuleParmas {
                        module_name: "mile_ui_wgsl",
                        idx: panel_id,
                        data: expr,
                        _ty: (ModuleEventType::Push | ModuleEventType::Vertex).bits(),
                    }));
                });
                ui.panel_interaction_trigger
                    .vertex_callbacks
                    .entry(UiInteractionScope {
                        panel_id: curr_id,
                        state,
                    })
                    .or_insert_with(Vec::new)
                    .push(wrap);
            }

            // 现在安全地注册回调
            for (call_type, mut cb, state) in to_register {
                match call_type {
                    Call::CLICK => {
                        ui.panel_interaction_trigger
                            .click_callbacks
                            .entry(UiInteractionScope {
                                panel_id: curr_id,
                                state,
                            })
                            .or_insert_with(Vec::new)
                            .push(cb);
                    }
                    Call::DRAG => {
                        ui.panel_interaction_trigger
                            .drag_callbacks
                            .entry(UiInteractionScope {
                                panel_id: curr_id,
                                state,
                            })
                            .or_insert_with(Vec::new)
                            .push(cb);
                    }
                    Call::HOVER => {
                        ui.panel_interaction_trigger
                            .hover_callbacks
                            .entry(UiInteractionScope {
                                panel_id: curr_id,
                                state,
                            })
                            .or_insert_with(Vec::new)
                            .push(cb);
                    }
                    default => {}
                }
            }

            self.net_work_build(curr_id);

            let _ = ui
                .event_hub
                .sender
                .send(CpuPanelEvent::StateTransition(StateTransition {
                    state_config_des: transfrom_config_des,
                    new_state: self.default_state,
                    panel_id: self.panel_id,
                }));
        }
    }
}

pub struct CallbackBuilder<'b, T>
where
    T: 'static,
{
    state_id: StateId,
    mui: &'b Mui<T>,
    next_state: Vec<(UiState, Call)>,
    pending_callbacks: RefCell<Vec<(Call, Box<dyn FnMut(&mut T, u32) + 'static>)>>,
}

impl<'b, T> CallbackBuilder<'b, T>
where
    T: 'static,
{
    pub fn next_state(mut self, call: Call, state: UiState) -> Self {
        self.next_state.push((state, call));
        self
    }

    pub fn call<F>(mut self, cb_type: Call, cb: F) -> Self
    where
        F: Fn(&mut T, u32) + 'static,
    {
        self.pending_callbacks
            .borrow_mut()
            .push((cb_type, Box::new(cb)));
        self
    }

    /// 延迟注册 — 由 build 阶段或统一注册器调用
    pub fn register_all(&self) {
        let mut mui = self.mui;
        let data_ref = self.mui.data.clone();
        let emit = self.mui.emit.clone();

        // let curr_state = self.state_id;
        // let configs = mui.state_config.borrow_mut();

        let mut states = self.mui.pending.states.borrow_mut();
        let cfg = states
            .get_mut(&self.state_id)
            .expect("没找到当前配置上下文");

        for (cb_type, mut cb_logic) in self.pending_callbacks.borrow_mut().drain(..) {
            let data_ref = data_ref.clone();
            let emit = emit.clone();
            let next_state = self.next_state.clone();
            println!("外部 {:?}", next_state);
            let net_work_config = mui.pending_net_work.state_net_work.clone();
            let curr_state = self.state_id;
            let states_config = mui.state_config.clone();
            let wrapped_cb: Box<dyn FnMut(u32)> = Box::new(move |panel_id| {
                let mut data = data_ref.borrow_mut();
                cb_logic(&mut data, panel_id);

                if cb_type == Call::HOVER {
                    let mut state_config = states_config.borrow_mut();
                    let curr_state_config = state_config.get(&curr_state).unwrap();
                    if let Some(offset) = curr_state_config.offset {
                        let _ = emit.send(CpuPanelEvent::SpecielAnim((
                            panel_id,
                            TransformAnimFieldInfo {
                                field_id: (PanelField::POSITION_X | PanelField::POSITION_Y).bits(),
                                start_value: vec![0.0; 2],
                                target_value: vec![offset.x, offset.y],
                                duration: 2.0,
                                easing: EasingMask::IN_OUT_CUBIC,
                                op: AnimOp::ADD,
                                hold: 1,
                                delay: 0.0,
                                loop_count: 0,
                                ping_pong: 0,
                                on_complete: 1,
                            },
                        )));
                    }
                }

                for ((uistate, call)) in &next_state {
                    let curr_state_config = states_config.borrow_mut();
                    let next_config = curr_state_config.get(&uistate.0);
                    let next_state_net_work_config_mut = net_work_config.borrow_mut();

                    let next_state_net_work_config: Option<&StateNetWorkConfig> =
                        next_state_net_work_config_mut.get(&uistate.0).clone();

                    if (cb_type != *call) {
                        return;
                    }

                    if let Some(net_work_config) = next_state_net_work_config {
                        let _ = emit.send(CpuPanelEvent::NetWorkTransition(NetWorkTransition {
                            state_config_des: net_work_config.to_des(),
                            curr_state: *uistate,
                            panel_id: panel_id,
                        }));
                    }

                    if let Some(next) = next_config {
                        let _ = emit.send(CpuPanelEvent::StateTransition(StateTransition {
                            state_config_des: next.clone(),
                            new_state: *uistate,
                            panel_id,
                        }));
                    }
                }
            });
            cfg.call.insert(cb_type, wrapped_cb);
        }

        let mut states_config = mui.state_config.borrow_mut();
        states_config.insert(self.state_id, cfg.to_des());
    }

    // pub fn call<F>(mut self, cb_type: Call,mut cb: F) -> Self
    // where
    //     F: FnMut(&mut T,u32) + 'static, // 不需要 Send + Sync
    // {
    //     let data_ref = self.mui.data.clone(); // Rc<RefCell<T>>
    //     let mut next_state = self.next_state;
    //     let emit = self.mui.emit.clone();
    //     let old_state = self.mui.curr_state;
    //     let panel_id = self.mui.panel_id;

    //     print!("当前的状态触发有下一个状态 {:?}",next_state);

    //     // 闭包内部只在事件触发时借用
    //     let wrapped_cb: Box<dyn FnMut(u32) + 'static> = Box::new(move |panel_id| {
    //         let mut data = data_ref.borrow_mut();
    //         // 注意这里 cb 也要是 FnMut，传递一个 &mut T
    //         cb(&mut data,panel_id);

    //         if let Some(next_state) = next_state{
    //             let _ = emit.send(CpuPanelEvent::StateTransition(StateTransition{
    //                 old_state,
    //                 new_state:next_state,
    //                 panel_id
    //             }));
    //         }
    //     });

    //     // 插入到当前状态
    //     let curr_state = &self.state_id;
    //     let mut state_map = self.mui.pending.states.borrow_mut();
    //     if let Some(cfg) = state_map.get_mut(curr_state) {
    //         cfg.call.insert(cb_type, wrapped_cb);
    //     }

    //     self
    // }

    pub fn exit(self) -> &'b Mui<T> {
        self.register_all();
        self.mui
    }
}

// ---------------------- 使用示例 ----------------------

#[derive(Debug)]
struct Data {
    pub index: f32,
}

//    {
//         let mut gpu_ui_b = gpu_ui.borrow_mut();
//         {
//             // let net_work =gpu_ui_b.new_work_store.as_mut().unwrap();

//             // let mut rel = Rel::new("测试关系","背包","背包元素");
//             // rel.layout = RelLayoutMask::GRID.bits();
//             // // rel.animation_field = PanelField::Not.bits();

//             // net_work.add_rel(rel);

//         }
//     }

#[derive(Debug)]
struct Counter {
    count: u32,
}

pub fn mile_test(gpu_ui: Arc<RefCell<GpuUi>>, queue: &wgpu::Queue, device: &wgpu::Device) {
    let counter = Rc::new(RefCell::new(Counter { count: 0 }));
    let emit = gpu_ui.borrow().event_hub.sender.clone();

    for x in 0..3 {
        for y in 0..3 {
            Mui::new(counter.clone(), emit.clone())
                .default_state(UiState(0))
                .state(UiState(0))
                .texture("caton (2).png")
                .size_with_image()
                .pos(vec2(x as f32 * 200.0 + 300.0, y as f32 * 200.0 + 300.0))
                .frag(|data, panel_id: u32| {
                        let uv = rv("uv");
                        let scan_line = sin(uv.y() * 15.0 + cv("time") * 2.0);
                        let color = rv("color");
                        wvec4(scan_line.clone() + color.x(), scan_line.clone() + color.y(), scan_line.clone() + color.z(), color.w())
                })
                .on()
                .call(Call::CLICK, move |data, panelid| {
                    data.count += 1;
                    println!("当前的点击次数 {:?}", data.count);
                })
                .next_state(Call::CLICK, UiState(1))
                .exit()
                .state(UiState(1))
                .texture("caton (3).png")
                .size_with_image()
                .on()
                .call(Call::CLICK, move |data, panelid| {
                    data.count += 1;
                    println!("当前的点击次数 {:?}", data.count);
                })
                .next_state(Call::CLICK, UiState(0))
                .exit()
                .build(gpu_ui.clone(), queue, device);
        }
    }
}
