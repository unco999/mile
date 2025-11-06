//! CPU side runtime state and frame scheduling helpers.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use flume::{Receiver, Sender};
use glam::Vec2;

use crate::{
    mui_anim::AnimationSpec,
    mui_prototype::{PanelStateOverrides, UiState},
    runtime::_ty::TransformAnimFieldInfo,
    structs::PanelInteraction,
};

/// Frame index alias used throughout the UI runtime.
pub type FRAME = u32;

/// Result propagated from WGSL compilation / gpu kernels.
#[derive(Debug, Clone, Default)]
pub struct WgslResult {
    pub label: Option<String>,
    pub success: bool,
    pub log: Option<String>,
}

/// Callbacks registered for UI interaction scopes.
pub type ClickCallback = Box<dyn FnMut(u32) + Send>;
pub type DragCallback = Box<dyn FnMut(u32) + Send>;
pub type HoverCallback = Box<dyn FnMut(u32) + Send>;
pub type EntryCallBack = Box<dyn FnMut(u32) + Send>;
pub type OutCallBack = Box<dyn FnMut(u32) + Send>;
pub type EntryFragBack = Box<dyn FnMut(u32) + Send>;
pub type EntryVertexBack = Box<dyn FnMut(u32) + Send>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UiInteractionScope {
    pub panel_id: u32,
    pub state: u32,
}

#[derive(Default)]
pub struct PanelEventRegistry {
    click_callbacks: HashMap<UiInteractionScope, Vec<ClickCallback>>,
    drag_callbacks: HashMap<UiInteractionScope, Vec<DragCallback>>,
    hover_callbacks: HashMap<UiInteractionScope, Vec<HoverCallback>>,
    entry_callbacks: HashMap<UiInteractionScope, Vec<EntryCallBack>>,
    out_callbacks: HashMap<UiInteractionScope, Vec<OutCallBack>>,
    frag_callbacks: HashMap<UiInteractionScope, Vec<EntryFragBack>>,
    vertex_callbacks: HashMap<UiInteractionScope, Vec<EntryVertexBack>>,
}

impl PanelEventRegistry {
    fn registry_for<T>(
        map: &mut HashMap<UiInteractionScope, Vec<T>>,
        scope: UiInteractionScope,
    ) -> &mut Vec<T> {
        map.entry(scope).or_default()
    }

    fn emit_callbacks<T>(callbacks: Option<&mut Vec<T>>, panel_id: u32)
    where
        T: FnMut(u32),
    {
        if let Some(callbacks) = callbacks {
            callbacks.iter_mut().for_each(|cb| cb(panel_id));
        }
    }

    pub fn unregister_scope(&mut self, scope: &UiInteractionScope) {
        self.click_callbacks.remove(scope);
        self.drag_callbacks.remove(scope);
        self.hover_callbacks.remove(scope);
        self.entry_callbacks.remove(scope);
        self.out_callbacks.remove(scope);
        self.frag_callbacks.remove(scope);
        self.vertex_callbacks.remove(scope);
    }

    pub fn register_click<F>(&mut self, scope: UiInteractionScope, callback: F)
    where
        F: FnMut(u32) + Send + 'static,
    {
        Self::registry_for(&mut self.click_callbacks, scope).push(Box::new(callback));
    }

    pub fn register_drag<F>(&mut self, scope: UiInteractionScope, callback: F)
    where
        F: FnMut(u32) + Send + 'static,
    {
        Self::registry_for(&mut self.drag_callbacks, scope).push(Box::new(callback));
    }

    pub fn register_hover<F>(&mut self, scope: UiInteractionScope, callback: F)
    where
        F: FnMut(u32) + Send + 'static,
    {
        Self::registry_for(&mut self.hover_callbacks, scope).push(Box::new(callback));
    }

    pub fn register_entry<F>(&mut self, scope: UiInteractionScope, callback: F)
    where
        F: FnMut(u32) + Send + 'static,
    {
        Self::registry_for(&mut self.entry_callbacks, scope).push(Box::new(callback));
    }

    pub fn register_out<F>(&mut self, scope: UiInteractionScope, callback: F)
    where
        F: FnMut(u32) + Send + 'static,
    {
        Self::registry_for(&mut self.out_callbacks, scope).push(Box::new(callback));
    }

    pub fn register_frag<F>(&mut self, scope: UiInteractionScope, callback: F)
    where
        F: FnMut(u32) + Send + 'static,
    {
        Self::registry_for(&mut self.frag_callbacks, scope).push(Box::new(callback));
    }

    pub fn register_vertex<F>(&mut self, scope: UiInteractionScope, callback: F)
    where
        F: FnMut(u32) + Send + 'static,
    {
        Self::registry_for(&mut self.vertex_callbacks, scope).push(Box::new(callback));
    }

    pub fn emit(&mut self, event: &CpuPanelEvent) {
        match event {
            CpuPanelEvent::Click((_frame, scope)) => {
                Self::emit_callbacks(self.click_callbacks.get_mut(scope), scope.panel_id);
                Self::emit_callbacks(self.entry_callbacks.get_mut(scope), scope.panel_id);
            }
            CpuPanelEvent::Drag((_frame, scope)) => {
                Self::emit_callbacks(self.drag_callbacks.get_mut(scope), scope.panel_id);
            }
            CpuPanelEvent::Hover((_frame, scope)) => {
                Self::emit_callbacks(self.hover_callbacks.get_mut(scope), scope.panel_id);
            }
            CpuPanelEvent::OUT((_frame, scope)) => {
                Self::emit_callbacks(self.out_callbacks.get_mut(scope), scope.panel_id);
            }
            CpuPanelEvent::Frag((_frame, scope)) => {
                Self::emit_callbacks(self.frag_callbacks.get_mut(scope), scope.panel_id);
            }
            CpuPanelEvent::Vertex((_frame, scope)) => {
                Self::emit_callbacks(self.vertex_callbacks.get_mut(scope), scope.panel_id);
            }
            CpuPanelEvent::StateTransition(state) => {
                let scope = UiInteractionScope {
                    panel_id: state.panel_id,
                    state: state.new_state.0,
                };
                Self::emit_callbacks(self.entry_callbacks.get_mut(&scope), scope.panel_id);
            }
            _ => {}
        }
    }
}

#[derive(Clone)]
pub struct UIEventHub {
    pub sender: Sender<CpuPanelEvent>,
    pub receiver: Receiver<CpuPanelEvent>,
    pub pre_hover_panel_id: Option<u32>,
}

impl UIEventHub {
    pub fn new() -> Self {
        let (sender, receiver) = flume::unbounded();
        Self {
            sender,
            receiver,
            pre_hover_panel_id: None,
        }
    }

    pub fn push(&self, event: CpuPanelEvent) {
        let _ = self.sender.send(event);
    }

    pub fn poll(&self) -> Vec<CpuPanelEvent> {
        let mut events = Vec::new();
        while let Ok(ev) = self.receiver.try_recv() {
            events.push(ev);
        }
        events
    }
}

#[derive(Debug, Clone)]
pub struct StateConfigDes {
    pub is_open_frag: bool,
    pub is_open_vertex: bool,
    pub open_api: Vec<StateOpenCall>,
    pub texture_id: Option<String>,
    pub pos: Option<Vec2>,
    pub size: Option<Vec2>,
}

impl Default for StateConfigDes {
    fn default() -> Self {
        Self {
            is_open_frag: false,
            is_open_vertex: false,
            open_api: Vec::new(),
            texture_id: None,
            pos: None,
            size: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StateOpenCall {
    Interaction(PanelInteraction),
}

impl From<StateOpenCall> for PanelInteraction {
    fn from(value: StateOpenCall) -> Self {
        match value {
            StateOpenCall::Interaction(mask) => mask,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct StateNetWorkConfigDes {
    pub insert_collection: Option<u32>,
    pub immediately_anim: bool,
}

#[derive(Debug, Clone)]
pub struct StateTransition {
    pub state_config_des: StateConfigDes,
    pub new_state: UiState,
    pub panel_id: u32,
}

#[derive(Debug, Clone)]
pub struct NetWorkTransition {
    pub state_config_des: StateNetWorkConfigDes,
    pub curr_state: UiState,
    pub panel_id: u32,
}

#[derive(Debug, Clone)]
pub enum CpuPanelEvent {
    OUT((FRAME, UiInteractionScope)),
    Hover((FRAME, UiInteractionScope)),
    Click((FRAME, UiInteractionScope)),
    StateTransition(StateTransition),
    Drag((FRAME, UiInteractionScope)),
    NetWorkTransition(NetWorkTransition),
    TotalUpdate(FRAME),
    SwapInteractionFrame(FRAME),
    WgslResult(WgslResult),
    SpecielAnim((u32, TransformAnimFieldInfo)),
    Frag((FRAME, UiInteractionScope)),
    Vertex((FRAME, UiInteractionScope)),
}

/// Per-frame data passed into compute/render stages.
#[derive(Debug, Default)]
pub struct FrameState {
    pub frame_index: u32,
    pub delta_time: f32,
    pub hover_id: Option<u32>,
    pub click_id: Option<u32>,
    pub drag_id: Option<u32>,
    pub hover_panel: Option<UiInteractionScope>,
    pub click_panel: Option<UiInteractionScope>,
    pub drag_panel: Option<UiInteractionScope>,
}

/// Collects CPU owned state that ultimately drives GPU uploads.
pub struct RuntimeState {
    pub event_hub: Arc<UIEventHub>,
    pub panel_events: Arc<Mutex<PanelEventRegistry>>,
    pub pending_animations: Vec<AnimationSpec>,
    pub pending_panel_overrides: Vec<(u32, PanelStateOverrides)>,
    pub frame_state: FrameState,
    pub cpu_events: Vec<CpuPanelEvent>,
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            event_hub: Arc::new(UIEventHub::new()),
            panel_events: Arc::new(Mutex::new(PanelEventRegistry::default())),
            pending_animations: Vec::new(),
            pending_panel_overrides: Vec::new(),
            frame_state: FrameState::default(),
            cpu_events: Vec::new(),
        }
    }
}

impl RuntimeState {
    pub fn clear_frame(&mut self) {
        self.pending_animations.clear();
        self.pending_panel_overrides.clear();
        self.cpu_events.clear();
    }

    pub fn push_event(&mut self, event: CpuPanelEvent) {
        self.cpu_events.push(event);
    }

    pub fn enqueue_animation(&mut self, animation: AnimationSpec) {
        self.pending_animations.push(animation);
    }
}
