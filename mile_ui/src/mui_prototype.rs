use crate::{
    mui_anim::{AnimBuilder, AnimProperty, AnimationSpec},
    mui_group::{
        GroupCenterMode, GroupRelationSpec, MuiGroupDefinition, configure_group, group_definition,
        group_type_id,
    },
    mui_style::{PanelStylePatch, StyleError, load_panel_style},
    runtime::{
        register_payload_refresh,
        state::{
            CpuPanelEvent, PanelEventRegistry, StateConfigDes, StateOpenCall, StateTransition,
            UIEventHub, UiInteractionScope,
        },
        QuadBatchKind,
    },
    structs::PanelInteraction,
};
use glam::{Vec2, Vec4, vec2};
use mile_api::{
    global::{global_db, global_event_bus},
    prelude::_ty::PanelId,
};
use mile_db::{DbError, TableBinding, TableHandle};
use mile_gpu_dsl::{
    core::{Expr, dsl::{wvec2, wvec4}}, dsl::rv, gpu_ast_core::event::{ExprTy, ExprWithIdxEvent}
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    any::{Any, TypeId, type_name},
    collections::{HashMap, HashSet},
    fmt,
    marker::PhantomData,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU32, AtomicU64, Ordering},
    },
};

static EVENT_REGISTRY: OnceLock<Arc<Mutex<PanelEventRegistry>>> = OnceLock::new();
static EVENT_HUB: OnceLock<Arc<UIEventHub>> = OnceLock::new();
static PENDING_REGISTRATIONS: OnceLock<Mutex<Vec<Box<dyn Fn(&mut PanelEventRegistry) + Send>>>> =
    OnceLock::new();
static PENDING_STATE_EVENTS: OnceLock<Mutex<Vec<StateTransition>>> = OnceLock::new();
static PANEL_KEY_REGISTRY: OnceLock<Mutex<HashMap<TypeId, HashSet<PanelKey>>>> = OnceLock::new();
static PENDING_SHADERS: OnceLock<Mutex<HashMap<u32, PendingShaderRequest>>> = OnceLock::new();
static NEXT_SHADER_IDX: AtomicU32 = AtomicU32::new(1);

fn pending_registrations() -> &'static Mutex<Vec<Box<dyn Fn(&mut PanelEventRegistry) + Send>>> {
    PENDING_REGISTRATIONS.get_or_init(|| Mutex::new(Vec::new()))
}

fn pending_state_events() -> &'static Mutex<Vec<StateTransition>> {
    PENDING_STATE_EVENTS.get_or_init(|| Mutex::new(Vec::new()))
}

fn panel_key_registry() -> &'static Mutex<HashMap<TypeId, HashSet<PanelKey>>> {
    PANEL_KEY_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn pending_shaders() -> &'static Mutex<HashMap<u32, PendingShaderRequest>> {
    PENDING_SHADERS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct PendingShaderRequest {
    panel_key: PanelKey,
    state: UiState,
    stage: ShaderStage,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ShaderStage {
    Fragment,
    Vertex,
}

impl ShaderStage {
    fn to_expr_ty(self) -> ExprTy {
        match self {
            ShaderStage::Fragment => ExprTy::Frag,
            ShaderStage::Vertex => ExprTy::Vertex,
        }
    }
}

fn next_shader_idx() -> u32 {
    NEXT_SHADER_IDX.fetch_add(1, Ordering::Relaxed)
}

fn register_pending_shader(idx: u32, key: &PanelKey, state: UiState, stage: ShaderStage) {
    let mut guard = pending_shaders().lock().unwrap();
    guard.insert(
        idx,
        PendingShaderRequest {
            panel_key: key.clone(),
            state,
            stage,
        },
    );
}

fn submit_shader_request(
    key: &PanelKey,
    state: UiState,
    shader: &Arc<dyn Fn(&ShaderScope) -> Expr + Send + Sync + 'static>,
    stage: ShaderStage,
) {
    let scope = ShaderScope;
    let expr = shader(&scope);
    let idx = next_shader_idx();
    register_pending_shader(idx, key, state, stage);
    global_event_bus().publish(ExprWithIdxEvent {
        idx,
        expr,
        _ty: stage.to_expr_ty(),
    });
}

pub(crate) fn take_pending_shader(idx: u32) -> Option<(PanelKey, UiState, ShaderStage)> {
    pending_shaders()
        .lock()
        .unwrap()
        .remove(&idx)
        .map(|request| (request.panel_key, request.state, request.stage))
}

pub(crate) fn register_panel_key<TPayload: PanelPayload>(key: &PanelKey) {
    let mut guard = panel_key_registry().lock().unwrap();
    guard
        .entry(TypeId::of::<TPayload>())
        .or_insert_with(HashSet::new)
        .insert(key.clone());
}

pub(crate) fn registered_panel_keys(type_id: TypeId) -> Vec<PanelKey> {
    panel_key_registry()
        .lock()
        .unwrap()
        .get(&type_id)
        .map(|set| set.iter().cloned().collect())
        .unwrap_or_default()
}

pub(crate) fn install_runtime_event_bridge(
    registry: Arc<Mutex<PanelEventRegistry>>,
    event_hub: Arc<UIEventHub>,
) {
    let stored = EVENT_REGISTRY.get_or_init(|| Arc::clone(&registry)).clone();
    let hub = EVENT_HUB.get_or_init(|| Arc::clone(&event_hub)).clone();

    let mut pending = pending_registrations().lock().unwrap();
    if !pending.is_empty() {
        let mut guard = stored.lock().unwrap();
        for task in pending.drain(..) {
            task(&mut guard);
        }
    }

    let mut pending_events = pending_state_events().lock().unwrap();
    for event in pending_events.drain(..) {
        hub.push(CpuPanelEvent::StateTransition(event));
    }
}

fn runtime_event_registry() -> Option<Arc<Mutex<PanelEventRegistry>>> {
    EVENT_REGISTRY.get().cloned()
}

fn runtime_event_hub() -> Option<Arc<UIEventHub>> {
    EVENT_HUB.get().cloned()
}

fn enqueue_state_transition(event: StateTransition) {
    if let Some(hub) = runtime_event_hub() {
        dbg!("发送转换事件");
        hub.push(CpuPanelEvent::StateTransition(event));
    } else {
        pending_state_events().lock().unwrap().push(event);
    }
}

/// Unique key used to address a panel inside the DB/runtime.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PanelKey {
    pub panel_id: u32,
    pub panel_uuid: String,
    pub scope: String,
}

impl PanelKey {
    pub fn new(panel_uuid: &'static str, scope: impl Into<String>) -> Self {
        let panel_id = panel_id_pool().id_for(panel_uuid);
        Self {
            panel_id,
            panel_uuid: panel_uuid.to_owned(),
            scope: scope.into(),
        }
    }

    pub fn numeric_id(&self) -> u32 {
        self.panel_id
    }

    pub fn uuid(&self) -> &str {
        &self.panel_uuid
    }
}

/// Simple wrapper to keep API parity with the existing builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UiState(pub u32);

/// Interaction types supported by the prototype.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UiEventKind {
    Click,
    Drag,
    Hover,
    Out,
}

/// Visual/Layout overrides persisted for each state.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct PanelStateOverrides {
    #[serde(default)]
    pub texture: Option<String>,
    #[serde(default)]
    pub fit_to_texture: Option<bool>,
    #[serde(default)]
    pub position: Option<[f32; 2]>,
    #[serde(default)]
    pub size: Option<[f32; 2]>,
    #[serde(default)]
    pub offset: Option<[f32; 2]>,
    #[serde(default)]
    pub color: Option<[f32; 4]>,
    #[serde(default)]
    pub border: Option<BorderStyle>,
    #[serde(default)]
    pub z_index: Option<i32>,
    #[serde(default)]
    pub pass_through: Option<u32>,
    #[serde(default)]
    pub interaction: Option<u32>,
    #[serde(default)]
    pub event_mask: Option<u32>,
    #[serde(default)]
    pub state_mask: Option<u32>,
    #[serde(default)]
    pub transparent: Option<f32>,
    #[serde(default)]
    pub collection_state: Option<u32>,
    #[serde(default)]
    pub state_transform_fade: Option<f32>,
    #[serde(default)]
    pub fragment_shader_id: Option<u32>,
    #[serde(default)]
    pub vertex_shader_id: Option<u32>,
    #[serde(default)]
    pub transitions: HashMap<UiEventKind, UiState>,
}

/// Resolved visual/layout snapshot applied to a panel.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PanelSnapshot {
    #[serde(default)]
    pub texture: Option<String>,
    pub fit_to_texture: bool,
    pub position: [f32; 2],
    pub color: [f32; 4],
    #[serde(default)]
    pub border: Option<BorderStyle>,
    pub z_index: i32,
    #[serde(default)]
    pub fragment_shader_id: Option<u32>,
    #[serde(default)]
    pub vertex_shader_id: Option<u32>,
    #[serde(default)]
    pub quad_vertex: QuadBatchKind,
}

impl Default for PanelSnapshot {
    fn default() -> Self {
        Self {
            texture: None,
            fit_to_texture: false,
            position: [0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            border: None,
            z_index: 0,
            fragment_shader_id: None,
            vertex_shader_id: None,
            quad_vertex: QuadBatchKind::Normal,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BorderStyle {
    pub color: [f32; 4],
    pub width: f32,
    pub radius: f32,
}

impl Default for BorderStyle {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            width: 0.0,
            radius: 0.0,
        }
    }
}

/// Users provide their own payload type by implementing this trait (auto-implemented for compatible types).
pub trait PanelPayload:
    Serialize + DeserializeOwned + Clone + Default + PartialEq + Send + Sync + 'static
{
    fn register_payload_type()
    where
        Self: Sized,
    {
        register_payload_refresh::<Self>();
    }
}

impl<T> PanelPayload for T where
    T: Serialize + DeserializeOwned + Clone + Default + PartialEq + Send + Sync + 'static
{
}

/// Full record stored in mile_db for a panel.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(bound(
    serialize = "TPayload: Serialize",
    deserialize = "TPayload: DeserializeOwned"
))]
pub struct PanelRecord<TPayload: PanelPayload> {
    pub default_state: Option<UiState>,
    #[serde(default)]
    pub states: HashMap<UiState, PanelStateOverrides>,
    pub current_state: UiState,
    pub snapshot: PanelSnapshot,
    #[serde(default)]
    pub pending_animations: Vec<AnimationSpec>,
    pub data: TPayload,
}

impl<TPayload: PanelPayload> Default for PanelRecord<TPayload> {
    fn default() -> Self {
        Self {
            default_state: None,
            states: HashMap::new(),
            current_state: UiState(0),
            snapshot: PanelSnapshot::default(),
            pending_animations: Vec::new(),
            data: TPayload::default(),
        }
    }
}

/// Event emitted whenever a panel record is updated.
#[derive(Clone, Debug)]
pub struct PanelStateChanged<TPayload: PanelPayload> {
    pub key: PanelKey,
    pub old: Option<PanelRecord<TPayload>>,
    pub new: PanelRecord<TPayload>,
}

/// DB binding so we can persist `PanelRecord`.
pub struct PanelBinding<TPayload: PanelPayload>(PhantomData<TPayload>);

impl<TPayload: PanelPayload> TableBinding for PanelBinding<TPayload> {
    type Key = PanelKey;
    type Value = PanelRecord<TPayload>;

    fn descriptor() -> &'static str {
        "mile_ui/panel_record@v1"
    }

    fn emit_change(key: &Self::Key, old: Option<&Self::Value>, new: &Self::Value) {
        global_event_bus().publish(PanelStateChanged::<TPayload> {
            key: key.clone(),
            old: old.cloned(),
            new: new.clone(),
        });

        let listeners = listeners_map::<TPayload>();
        if let Some(entries) = listeners.lock().unwrap().get(key).cloned() {
            let handle = PanelHandle::<TPayload>::for_key(key.clone());
            let change = PanelStyleChange {
                key,
                old: old.cloned(),
                new: new.clone(),
                handle,
                _marker: PhantomData,
            };
            for entry in entries {
                entry.listener.on_change(&change);
            }
        }
    }
}

/// Handle that owns the ability to mutate panel data.
struct PanelHandle<TPayload: PanelPayload> {
    table: TableHandle<PanelBinding<TPayload>>,
    key: PanelKey,
}

impl<TPayload: PanelPayload> Clone for PanelHandle<TPayload> {
    fn clone(&self) -> Self {
        let table = global_db()
            .bind_table::<PanelBinding<TPayload>>()
            .expect("panel table available");
        Self {
            table,
            key: self.key.clone(),
        }
    }
}

impl<TPayload: PanelPayload> PanelHandle<TPayload> {
    fn new(panel_uuid: &'static str, scope: impl Into<String>) -> Result<Self, DbError> {
        let db = global_db();
        let table = db.bind_table::<PanelBinding<TPayload>>()?;
        Ok(Self {
            table,
            key: PanelKey::new(panel_uuid, scope),
        })
    }

    fn for_key(key: PanelKey) -> Self {
        let db = global_db();
        let table = db
            .bind_table::<PanelBinding<TPayload>>()
            .expect("panel table available");
        Self { table, key }
    }

    fn mutate<F>(&self, mutator: F) -> Result<(), DbError>
    where
        F: FnOnce(&mut PanelRecord<TPayload>),
    {
        let mut entry = self
            .table
            .upsert_entry(self.key.clone(), PanelRecord::<TPayload>::default())?;
        mutator(entry.value_mut());
        entry.commit()
    }

    fn read(&self) -> Result<PanelRecord<TPayload>, DbError> {
        Ok(self
            .table
            .get(&self.key)?
            .unwrap_or_else(PanelRecord::<TPayload>::default))
    }

    fn register_listener(
        &self,
        listener: Arc<dyn PanelStyleListener<TPayload>>,
    ) -> PanelListenerGuard<TPayload> {
        let map_arc = listeners_map::<TPayload>();
        let mut map = map_arc.lock().unwrap();
        let entry = ListenerEntry::<TPayload> {
            id: next_listener_id(),
            listener,
        };
        let key = self.key.clone();
        map.entry(key.clone()).or_default().push(entry.clone());
        PanelListenerGuard {
            key,
            id: entry.id,
            _marker: PhantomData,
        }
    }

    fn remove_listener(key: &PanelKey, id: u64) {
        let map_arc = listeners_map::<TPayload>();
        if let Some(entries) = map_arc.lock().unwrap().get_mut(key) {
            entries.retain(|entry| entry.id != id);
        }
    }
}

/// Snapshot delivered to style listeners.
pub struct PanelStyleChange<'a, TPayload: PanelPayload> {
    pub key: &'a PanelKey,
    pub old: Option<PanelRecord<TPayload>>,
    pub new: PanelRecord<TPayload>,
    handle: PanelHandle<TPayload>,
    _marker: PhantomData<&'a TPayload>,
}

impl<'a, TPayload: PanelPayload> PanelStyleChange<'a, TPayload> {
    /// Mutate the live panel record via the DB handle.
    pub fn mutate<F>(&self, mutator: F)
    where
        F: FnOnce(&mut PanelRecord<TPayload>),
    {
        if let Err(err) = self.handle.mutate(mutator) {
            eprintln!("failed to apply listener mutation: {err:?}");
        }
    }
}

/// Listener trait that gets called whenever a panel record changes.
pub trait PanelStyleListener<TPayload: PanelPayload>: Send + Sync + 'static {
    fn on_change(&self, change: &PanelStyleChange<'_, TPayload>);
}

#[derive(Clone)]
struct ListenerEntry<TPayload: PanelPayload> {
    id: u64,
    listener: Arc<dyn PanelStyleListener<TPayload>>,
}

pub struct PanelListenerGuard<TPayload: PanelPayload> {
    key: PanelKey,
    id: u64,
    _marker: PhantomData<TPayload>,
}

impl<TPayload: PanelPayload> Drop for PanelListenerGuard<TPayload> {
    fn drop(&mut self) {
        PanelHandle::<TPayload>::remove_listener(&self.key, self.id);
    }
}

struct PanelDataObserver<TPayload, TObserved, Extract, Handler>
where
    TPayload: PanelPayload,
    TObserved: Send + Sync + 'static,
    Extract: Fn(&TPayload) -> Option<&TObserved> + Send + Sync + 'static,
    Handler: Fn(&TObserved, &PanelStyleChange<'_, TPayload>) + Send + Sync + 'static,
{
    extractor: Extract,
    handler: Handler,
    _marker: PhantomData<(TPayload, TObserved)>,
}

impl<TPayload, TObserved, Extract, Handler> PanelDataObserver<TPayload, TObserved, Extract, Handler>
where
    TPayload: PanelPayload,
    TObserved: Send + Sync + 'static,
    Extract: Fn(&TPayload) -> Option<&TObserved> + Send + Sync + 'static,
    Handler: Fn(&TObserved, &PanelStyleChange<'_, TPayload>) + Send + Sync + 'static,
{
    fn new(extractor: Extract, handler: Handler) -> Arc<dyn PanelStyleListener<TPayload>> {
        Arc::new(Self {
            extractor,
            handler,
            _marker: PhantomData,
        })
    }
}

impl<TPayload, TObserved, Extract, Handler> PanelStyleListener<TPayload>
    for PanelDataObserver<TPayload, TObserved, Extract, Handler>
where
    TPayload: PanelPayload,
    TObserved: Send + Sync + 'static,
    Extract: Fn(&TPayload) -> Option<&TObserved> + Send + Sync + 'static,
    Handler: Fn(&TObserved, &PanelStyleChange<'_, TPayload>) + Send + Sync + 'static,
{
    fn on_change(&self, change: &PanelStyleChange<'_, TPayload>) {
        if let Some(observed) = (self.extractor)(&change.new.data) {
            (self.handler)(observed, change);
        }
    }
}

fn attach_observers<TPayload: PanelPayload>(
    handle: &PanelHandle<TPayload>,
    observers: &[Arc<dyn PanelStyleListener<TPayload>>],
) -> Vec<PanelListenerGuard<TPayload>> {
    if observers.is_empty() {
        return Vec::new();
    }
    observers
        .iter()
        .map(|observer| handle.register_listener(observer.clone()))
        .collect()
}

fn apply_runtime_callbacks<TPayload: PanelPayload>(
    registry: &mut PanelEventRegistry,
    key: &PanelKey,
    callbacks: &HashMap<UiState, HashMap<UiEventKind, Arc<EventFn<TPayload>>>>,
    transitions: &HashMap<UiState, HashMap<UiEventKind, UiState>>,
) {
    let mut states: HashSet<UiState> = HashSet::new();
    states.extend(callbacks.keys().copied());
    states.extend(transitions.keys().copied());
    for targets in transitions.values() {
        states.extend(targets.values().copied());
    }

    for state in states {
        let scope = UiInteractionScope {
            panel_id: key.panel_id,
            state: state.0,
        };
        registry.unregister_scope(&scope);

        let mut events: HashSet<UiEventKind> = HashSet::new();
        if let Some(map) = callbacks.get(&state) {
            events.extend(map.keys().copied());
        }
        if let Some(map) = transitions.get(&state) {
            events.extend(map.keys().copied());
        }

        if events.is_empty() {
            continue;
        }

        for event in events {
            let key_clone = key.clone();
            let state_copy = state;
            let scope_copy = scope;
            match event {
                UiEventKind::Click => registry.register_click(scope_copy, move |_panel_id| {
                    trigger_event_internal::<TPayload>(
                        &key_clone,
                        Some(state_copy),
                        UiEventKind::Click,
                    );
                }),
                UiEventKind::Drag => registry.register_drag(scope_copy, move |_panel_id| {
                    trigger_event_internal::<TPayload>(
                        &key_clone,
                        Some(state_copy),
                        UiEventKind::Drag,
                    );
                }),
                UiEventKind::Hover => registry.register_hover(scope_copy, move |_panel_id| {
                    trigger_event_internal::<TPayload>(
                        &key_clone,
                        Some(state_copy),
                        UiEventKind::Hover,
                    );
                }),
                UiEventKind::Out => registry.register_out(scope_copy, move |_panel_id| {
                    trigger_event_internal::<TPayload>(
                        &key_clone,
                        Some(state_copy),
                        UiEventKind::Out,
                    );
                }),
            }
        }
    }
}

fn install_runtime_callbacks<TPayload: PanelPayload>(
    key: &PanelKey,
    callbacks: &HashMap<UiState, HashMap<UiEventKind, Arc<EventFn<TPayload>>>>,
    transitions: &HashMap<UiState, HashMap<UiEventKind, UiState>>,
) {
    if let Some(registry_arc) = runtime_event_registry() {
        let mut registry = registry_arc.lock().unwrap();
        apply_runtime_callbacks::<TPayload>(&mut registry, key, callbacks, transitions);
    } else {
        let key_clone = key.clone();
        let callbacks_clone = callbacks.clone();
        let transitions_clone = transitions.clone();
        let mut pending = pending_registrations().lock().unwrap();
        pending.push(Box::new(move |registry: &mut PanelEventRegistry| {
            apply_runtime_callbacks::<TPayload>(
                registry,
                &key_clone,
                &callbacks_clone,
                &transitions_clone,
            );
        }));
    }
}

/// Runtime registry storing event handlers and the current state of each panel.
struct PanelRuntime<TPayload: PanelPayload> {
    handle: PanelHandle<TPayload>,
    current_state: UiState,
    transitions: HashMap<UiState, HashMap<UiEventKind, UiState>>,
    callbacks: HashMap<UiState, HashMap<UiEventKind, Arc<EventFn<TPayload>>>>,
    observer_guards: Vec<PanelListenerGuard<TPayload>>,
}
type EventFn<TPayload> = dyn for<'a> Fn(&mut EventFlow<'a, TPayload>) + Send + Sync + 'static;

#[derive(Clone, Debug)]
pub struct PanelEventArgs<TPayload: PanelPayload> {
    pub panel_key: PanelKey,
    pub state: UiState,
    pub event: UiEventKind,
    pub record_snapshot: PanelRecord<TPayload>,
}

pub struct EventFlow<'a, TPayload: PanelPayload> {
    record: &'a mut PanelRecord<TPayload>,
    args: &'a PanelEventArgs<TPayload>,
    current_state: &'a mut UiState,
    transitions: &'a HashMap<UiState, HashMap<UiEventKind, UiState>>,
    override_state: Option<UiState>,
}

impl<'a, TPayload: PanelPayload> EventFlow<'a, TPayload> {
    fn new(
        record: &'a mut PanelRecord<TPayload>,
        args: &'a PanelEventArgs<TPayload>,
        current_state: &'a mut UiState,
        transitions: &'a HashMap<UiState, HashMap<UiEventKind, UiState>>,
    ) -> Self {
        Self {
            record,
            args,
            current_state,
            transitions,
            override_state: None,
        }
    }

    pub fn payload(&mut self) -> &mut TPayload {
        &mut self.record.data
    }

    pub fn payload_ref(&self) -> &TPayload {
        &self.record.data
    }

    pub fn push_animation(&mut self, animation: AnimationSpec) {
        self.record.pending_animations.push(animation);
    }

    pub fn args(&self) -> &PanelEventArgs<TPayload> {
        self.args
    }

    pub fn state(&self) -> UiState {
        *self.current_state
    }

    pub fn set_state(&mut self, state: UiState) {
        self.record.current_state = state;
        *self.current_state = state;
        self.override_state = Some(state);
        let transition = build_state_transition_event(self.record, self.args, state);
        enqueue_state_transition(transition);
    }

    pub fn transition(&mut self, event: UiEventKind) -> Option<UiState> {
        let current = *self.current_state;
        if let Some(next) = self
            .transitions
            .get(&current)
            .and_then(|map| map.get(&event))
            .copied()
        {
            self.set_state(next);
            Some(next)
        } else {
            None
        }
    }

    fn take_override(&mut self) -> Option<UiState> {
        self.override_state.take()
    }
}

fn overrides_for_state<'a, TPayload: PanelPayload>(
    record: &'a PanelRecord<TPayload>,
    state: UiState,
) -> Option<&'a PanelStateOverrides> {
    record
        .states
        .get(&state)
        .or_else(|| {
            record
                .default_state
                .and_then(|default| record.states.get(&default))
        })
        .or_else(|| record.states.values().next())
}

fn state_config_from_overrides(overrides: Option<&PanelStateOverrides>) -> StateConfigDes {
    let mut config = StateConfigDes::default();
    if let Some(override_ref) = overrides {
        if let Some(texture) = &override_ref.texture {
            config.texture_id = Some(texture.clone());
        }
        if let Some(position) = override_ref.position {
            config.pos = Some(Vec2::new(position[0], position[1]));
        }
        if let Some(size) = override_ref.size {
            config.size = Some(Vec2::new(size[0], size[1]));
        }
        if let Some(mask_bits) = override_ref.interaction {
            if let Some(mask) = PanelInteraction::from_bits(mask_bits) {
                config.open_api.push(StateOpenCall::Interaction(mask));
            }
        }
        config.is_open_frag = override_ref.fragment_shader_id.is_some();
        config.is_open_vertex = override_ref.vertex_shader_id.is_some();
    }
    config
}

fn build_state_transition_event<TPayload: PanelPayload>(
    record: &PanelRecord<TPayload>,
    args: &PanelEventArgs<TPayload>,
    state: UiState,
) -> StateTransition {
    let overrides = overrides_for_state(record, state);
    let config = state_config_from_overrides(overrides);
    StateTransition {
        state_config_des: config,
        new_state: state,
        panel_id: args.panel_key.panel_id,
    }
}

fn listeners_map<TPayload: PanelPayload>()
-> Arc<Mutex<HashMap<PanelKey, Vec<ListenerEntry<TPayload>>>>> {
    static MAPS: OnceLock<Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>> = OnceLock::new();
    let maps = MAPS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = maps.lock().unwrap();
    let entry = guard.entry(TypeId::of::<TPayload>()).or_insert_with(|| {
        Arc::new(Mutex::new(
            HashMap::<PanelKey, Vec<ListenerEntry<TPayload>>>::new(),
        )) as Arc<dyn Any + Send + Sync>
    });
    let arc = entry.clone();
    drop(guard);
    arc.downcast::<Mutex<HashMap<PanelKey, Vec<ListenerEntry<TPayload>>>>>()
        .unwrap()
}

fn runtime_map<TPayload: PanelPayload>() -> Arc<Mutex<HashMap<PanelKey, PanelRuntime<TPayload>>>> {
    static MAPS: OnceLock<Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>> = OnceLock::new();
    let maps = MAPS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = maps.lock().unwrap();
    let entry = guard.entry(TypeId::of::<TPayload>()).or_insert_with(|| {
        Arc::new(Mutex::new(
            HashMap::<PanelKey, PanelRuntime<TPayload>>::new(),
        )) as Arc<dyn Any + Send + Sync>
    });
    let arc = entry.clone();
    drop(guard);
    arc.downcast::<Mutex<HashMap<PanelKey, PanelRuntime<TPayload>>>>()
        .unwrap()
}

fn register_runtime<TPayload: PanelPayload>(key: PanelKey, runtime: PanelRuntime<TPayload>) {
    let arc = runtime_map::<TPayload>();
    register_panel_key::<TPayload>(&key);
    arc.lock().unwrap().insert(key, runtime);
}

fn trigger_event_internal<TPayload: PanelPayload>(
    key: &PanelKey,
    forced_state: Option<UiState>,
    event: UiEventKind,
) {
    let arc = runtime_map::<TPayload>();
    let mut registry = arc.lock().unwrap();
    let Some(runtime) = registry.get_mut(key) else {
        eprintln!(
            "attempted to trigger event {:?} on unknown panel {:?}",
            event, key
        );
        return;
    };

    let state = if let Some(state_override) = forced_state {
        runtime.current_state = state_override;
        state_override
    } else {
        runtime.current_state
    };

    let snapshot = match runtime.handle.read() {
        Ok(record) => record,
        Err(err) => {
            eprintln!("failed to read panel record: {err:?}");
            PanelRecord::<TPayload>::default()
        }
    };

    let Some(callbacks) = runtime.callbacks.get(&state) else {
        return;
    };
    let args = PanelEventArgs {
        panel_key: key.clone(),
        state,
        event,
        record_snapshot: snapshot,
    };

    let mut applied_override = false;
    if let Some(handler) = callbacks.get(&event) {
        let handle = &runtime.handle;
        let transitions = &runtime.transitions;
        let current_state_ref = &mut runtime.current_state;
        if let Err(err) = handle.mutate(|record| {
            let mut flow = EventFlow::new(record, &args, current_state_ref, transitions);
            handler(&mut flow);
            applied_override = flow.take_override().is_some();
        }) {
            eprintln!("event mutation failed: {err:?}");
        }
    }

    if !applied_override {
        if let Some(next_state) = runtime
            .transitions
            .get(&state)
            .and_then(|map| map.get(&event))
        {
            runtime.current_state = *next_state;
        }
    }
}

pub fn trigger_event<TPayload: PanelPayload>(key: &PanelKey, event: UiEventKind) {
    trigger_event_internal::<TPayload>(key, None, event);
}

/// Handle returned by `build` so tests can trigger events.
pub struct PanelRuntimeHandle {
    key: PanelKey,
}

impl PanelRuntimeHandle {
    pub fn trigger<TPayload: PanelPayload>(&self, event: UiEventKind) {
        trigger_event::<TPayload>(&self.key, event);
    }

    pub fn key(&self) -> &PanelKey {
        &self.key
    }
}

/// Flow mode for the builder.
enum FlowMode<TPayload: PanelPayload> {
    Stateful {
        default_state: Option<UiState>,
        states: HashMap<UiState, PanelStateDefinition<TPayload>>,
    },
    Stateless {
        state: PanelStateDefinition<TPayload>,
    },
}

/// Definition captured during state building.
struct PanelStateDefinition<TPayload: PanelPayload> {
    overrides: PanelStateOverrides,
    frag_shader: Option<FragClosure>,
    vertex_shader: Option<VertexClosure>,
    callbacks: HashMap<UiEventKind, Arc<EventFn<TPayload>>>,
    animations: Vec<AnimationSpec>,
    groups: Vec<MuiGroupDefinition>,
    group_relations: Vec<GroupRelationSpec>,
}

impl<TPayload: PanelPayload> Default for PanelStateDefinition<TPayload> {
    fn default() -> Self {
        Self {
            overrides: PanelStateOverrides::default(),
            frag_shader: None,
            vertex_shader: None,
            callbacks: HashMap::new(),
            animations: Vec::new(),
            groups: Vec::new(),
            group_relations: Vec::new(),
        }
    }
}

/// Entry point for the new builder.
pub struct Mui<TPayload: PanelPayload> {
    handle: PanelHandle<TPayload>,
    mode: FlowMode<TPayload>,
    observers: Vec<Arc<dyn PanelStyleListener<TPayload>>>,
    quad_vertex: QuadBatchKind,
}

impl<TPayload: PanelPayload> Mui<TPayload> {
    fn derived_scope(panel_uuid: &'static str) -> String {
        format!("{}::{}", type_name::<TPayload>(), panel_uuid)
    }

    pub fn stateful(panel_uuid: &'static str) -> Result<Self, DbError> {
        Self::stateful_with_scope(panel_uuid, Self::derived_scope(panel_uuid))
    }

    pub fn stateful_with_scope(
        panel_uuid: &'static str,
        scope: impl Into<String>,
    ) -> Result<Self, DbError> {
        TPayload::register_payload_type();
        let handle = PanelHandle::<TPayload>::new(panel_uuid, scope)?;
        Ok(Self {
            handle,
            mode: FlowMode::Stateful {
                default_state: None,
                states: HashMap::new(),
            },
            observers: Vec::new(),
            quad_vertex: QuadBatchKind::Normal,
        })
    }

    pub fn stateless(panel_uuid: &'static str) -> Result<Self, DbError> {
        Self::stateless_with_scope(panel_uuid, Self::derived_scope(panel_uuid))
    }

    pub fn stateless_with_scope(
        panel_uuid: &'static str,
        scope: impl Into<String>,
    ) -> Result<Self, DbError> {
        TPayload::register_payload_type();
        let handle = PanelHandle::<TPayload>::new(panel_uuid, scope)?;
        Ok(Self {
            handle,
            mode: FlowMode::Stateless {
                state: PanelStateDefinition::default(),
            },
            observers: Vec::new(),
            quad_vertex: QuadBatchKind::Normal,
        })
    }

    /// Observe panel data changes by extracting a reference of type `TObserved`.
    pub fn observe<TObserved, Extract, Handler>(
        mut self,
        extractor: Extract,
        handler: Handler,
    ) -> Self
    where
        TObserved: Send + Sync + 'static,
        Extract: Fn(&TPayload) -> Option<&TObserved> + Send + Sync + 'static,
        Handler: Fn(&TObserved, &PanelStyleChange<'_, TPayload>) + Send + Sync + 'static,
    {
        let listener =
            PanelDataObserver::<TPayload, TObserved, Extract, Handler>::new(extractor, handler);
        self.observers.push(listener);
        self
    }

    /// Observe the entire payload whenever it changes.
    pub fn observe_payload<Handler>(mut self, handler: Handler) -> Self
    where
        Handler: Fn(&TPayload, &PanelStyleChange<'_, TPayload>) + Send + Sync + 'static,
    {
        let listener =
            PanelDataObserver::<TPayload, TPayload, _, _>::new(|payload| Some(payload), handler);
        self.observers.push(listener);
        self
    }

    pub fn quad_vertex(mut self, quad_vertex: QuadBatchKind) -> Self {
        self.quad_vertex = quad_vertex;
        self
    }

    pub fn default_state(mut self, state: UiState) -> Self {
        if let FlowMode::Stateful {
            ref mut default_state,
            ..
        } = self.mode
        {
            *default_state = Some(state);
        }
        self
    }

    pub fn state<F>(mut self, state_id: UiState, configure: F) -> Self
    where
        F: FnOnce(StateStageBuilder<TPayload>) -> StateStageBuilder<TPayload>,
    {
        let builder = configure(StateStageBuilder::new(state_id));
        let definition = builder.into_definition();

        match self.mode {
            FlowMode::Stateful { ref mut states, .. } => {
                states.insert(state_id, definition);
            }
            FlowMode::Stateless { .. } => {
                panic!("stateless panels cannot register multiple states")
            }
        }
        self
    }

    pub fn configure_stateless<F>(mut self, configure: F) -> Self
    where
        F: FnOnce(StateStageBuilder<TPayload>) -> StateStageBuilder<TPayload>,
    {
        match self.mode {
            FlowMode::Stateless { ref mut state } => {
                let builder = configure(StateStageBuilder::new(UiState(0)));
                *state = builder.into_definition();
            }
            FlowMode::Stateful { .. } => panic!("stateful panels should use `state`"),
        }
        self
    }

    pub fn build(self) -> Result<PanelRuntimeHandle, DbError> {
        let handle = self.handle;
        let quad_vertex = self.quad_vertex;
        match self.mode {
            FlowMode::Stateful {
                default_state,
                states,
            } => build_stateful::<TPayload>(handle, default_state, states, self.observers, quad_vertex),
            FlowMode::Stateless { state } => {
                build_stateless::<TPayload>(handle, state, self.observers, quad_vertex)
            }
        }
    }

    pub fn register_style_listener<L>(&self, listener: L) -> PanelListenerGuard<TPayload>
    where
        L: PanelStyleListener<TPayload>,
    {
        self.handle.register_listener(Arc::new(listener))
    }
}

/// Builder dedicated to a single state.
pub struct StateStageBuilder<TPayload: PanelPayload> {
    state_id: UiState,
    definition: PanelStateDefinition<TPayload>,
}

impl<TPayload: PanelPayload> StateStageBuilder<TPayload> {
    fn new(state_id: UiState) -> Self {
        Self {
            state_id,
            definition: PanelStateDefinition::default(),
        }
    }

    fn into_definition(self) -> PanelStateDefinition<TPayload> {
        self.definition
    }

    pub fn texture(mut self, path: &str) -> Self {
        self.definition.overrides.texture = Some(path.to_owned());
        self
    }

    pub fn size_with_image(mut self) -> Self {
        self.definition.overrides.fit_to_texture = Some(true);
        self
    }

    pub fn size(mut self, size: Vec2) -> Self {
        self.definition.overrides.size = Some([size.x, size.y]);
        self
    }

    pub fn position(mut self, pos: Vec2) -> Self {
        self.definition.overrides.position = Some([pos.x, pos.y]);
        self
    }

    pub fn color(mut self, tint: Vec4) -> Self {
        self.definition.overrides.color = Some([tint.x, tint.y, tint.z, tint.w]);
        self
    }

    pub fn border(mut self, border: BorderStyle) -> Self {
        self.definition.overrides.border = Some(border);
        self
    }

    pub fn z_index(mut self, z: i32) -> Self {
        self.definition.overrides.z_index = Some(z);
        self
    }

    pub fn state_transform_fade(mut self, fade: f32) -> Self {
        self.definition.overrides.state_transform_fade = Some(fade.max(0.0));
        self
    }

    pub fn fragment_shader<F>(mut self, shader: F) -> Self
    where
        F: Fn(&ShaderScope) -> Expr + Send + Sync + 'static,
    {
        self.definition.frag_shader = Some(Arc::new(shader));
        self
    }

    pub fn vertex_shader<F>(mut self, shader: F) -> Self
    where
        F: Fn(&ShaderScope) -> Expr + Send + Sync + 'static,
    {
        self.definition.vertex_shader = Some(Arc::new(shader));
        self
    }

    pub(crate) fn group_with<T: 'static, F>(mut self, configure: F) -> Self
    where
        F: FnOnce(&mut MuiGroupDefinition),
    {
        let id = group_type_id::<T>();
        if self
            .definition
            .groups
            .iter()
            .any(|existing| existing.id == id)
        {
            return self;
        }

        let mut definition = group_definition::<T>();
        configure(&mut definition);
        let order = self.definition.groups.len() as u32;
        definition.order = order;
        definition.cpu_index = order;
        self.definition.groups.push(definition);
        self
    }

    pub fn group<T: 'static>(self) -> Self {
        self.group_with::<T, _>(|_| {})
    }

    pub fn try_style(mut self, key: &str) -> Result<Self, StyleError> {
        let patch = load_panel_style(key)?;
        apply_style_patch(&mut self.definition.overrides, &patch);
        Ok(self)
    }

    pub fn style(self, key: &str) -> Self {
        self.try_style(key)
            .unwrap_or_else(|err| panic!("failed to load style '{key}': {err}"))
    }

    pub fn events(self) -> InteractionStageBuilder<TPayload> {
        InteractionStageBuilder {
            parent: self,
            stage: InteractionStage::default(),
            last_event: None,
        }
    }
}

fn apply_style_patch(overrides: &mut PanelStateOverrides, patch: &PanelStylePatch) {
    if let Some(texture) = &patch.texture {
        overrides.texture = Some(texture.clone());
    }
    if let Some(fit) = patch.fit_to_texture {
        overrides.fit_to_texture = Some(fit);
    }
    if let Some(position) = patch.position {
        overrides.position = Some(position);
    }
    if let Some(z) = patch.z_index {
        overrides.z_index = Some(z as i32);
    }
}

#[derive(Default)]
struct InteractionStage<TPayload: PanelPayload> {
    transitions: HashMap<UiEventKind, UiState>,
    callbacks: HashMap<UiEventKind, Arc<EventFn<TPayload>>>,
}

pub struct InteractionStageBuilder<TPayload: PanelPayload> {
    parent: StateStageBuilder<TPayload>,
    stage: InteractionStage<TPayload>,
    last_event: Option<UiEventKind>,
}

impl<TPayload: PanelPayload> InteractionStageBuilder<TPayload> {
    pub fn on_event<F>(mut self, event: UiEventKind, handler: F) -> Self
    where
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>) + Send + Sync + 'static,
    {
        let event_fn: Arc<EventFn<TPayload>> = Arc::new(handler);
        self.stage.callbacks.insert(event, event_fn);
        self.last_event = Some(event);
        self
    }

    pub fn transition_to(mut self, event: UiEventKind, next: UiState) -> Self {
        self.stage.transitions.insert(event, next);
        self.last_event = Some(event);
        self
    }

    pub fn transition_to_next(mut self, next: UiState) -> Self {
        if let Some(event) = self.last_event {
            self.stage.transitions.insert(event, next);
        }
        self
    }

    pub fn finish(mut self) -> StateStageBuilder<TPayload> {
        let overrides = &mut self.parent.definition.overrides;

        let mut interaction_mask = overrides
            .interaction
            .unwrap_or(PanelInteraction::DEFUALT.bits());

        for kind in self
            .stage
            .callbacks
            .keys()
            .chain(self.stage.transitions.keys())
        {
            interaction_mask |= match kind {
                UiEventKind::Click => PanelInteraction::CLICKABLE.bits(),
                UiEventKind::Drag => PanelInteraction::DRAGGABLE.bits(),
                UiEventKind::Hover => PanelInteraction::HOVER.bits(),
                UiEventKind::Out => PanelInteraction::Out.bits(),
            };
        }

        if interaction_mask != PanelInteraction::DEFUALT.bits() {
            overrides.interaction = Some(interaction_mask);
        }

        let overrides = &mut self.parent.definition.overrides;

        let mut interaction_mask = overrides
            .interaction
            .unwrap_or(PanelInteraction::DEFUALT.bits());

        for kind in self
            .stage
            .callbacks
            .keys()
            .chain(self.stage.transitions.keys())
        {
            interaction_mask |= match kind {
                UiEventKind::Click => PanelInteraction::CLICKABLE.bits(),
                UiEventKind::Drag => PanelInteraction::DRAGGABLE.bits(),
                UiEventKind::Hover => PanelInteraction::HOVER.bits(),
                UiEventKind::Out => PanelInteraction::Out.bits(),
            };
        }

        if interaction_mask != PanelInteraction::DEFUALT.bits() {
            overrides.interaction = Some(interaction_mask);
        }

        let transitions = std::mem::take(&mut self.stage.transitions);
        let callbacks = std::mem::take(&mut self.stage.callbacks);
        overrides.transitions.extend(transitions);
        self.parent.definition.callbacks.extend(callbacks);
        self.parent
    }
}

fn build_stateful<TPayload: PanelPayload>(
    handle: PanelHandle<TPayload>,
    default_state: Option<UiState>,
    states: HashMap<UiState, PanelStateDefinition<TPayload>>,
    observers: Vec<Arc<dyn PanelStyleListener<TPayload>>>,
    quad_vertex: QuadBatchKind,
) -> Result<PanelRuntimeHandle, DbError> {
    let panel_key = handle.key.clone();
    let default = default_state
        .or_else(|| states.keys().copied().next())
        .expect("at least one state required");

    handle.mutate(|record| {
        record.default_state = Some(default);
        record.current_state = default;
        record.states.clear();
        record.snapshot.quad_vertex = quad_vertex;
        for (state_id, definition) in &states {
            record
                .states
                .insert(*state_id, definition.overrides.clone());
        }
    })?;

    let mut transitions = HashMap::new();
    let mut callbacks = HashMap::new();

    for (state_id, definition) in states {
        let PanelStateDefinition {
            overrides,
            frag_shader,
            vertex_shader,
            callbacks: state_callbacks,
            animations: _animations,
            groups: _groups,
            group_relations: _group_relations,
        } = definition;

        if let Some(shader) = frag_shader.as_ref() {
            submit_shader_request(&panel_key, state_id, shader, ShaderStage::Fragment);
        }
        if let Some(shader) = vertex_shader.as_ref() {
            submit_shader_request(&panel_key, state_id, shader, ShaderStage::Vertex);
        }

        transitions.insert(state_id, overrides.transitions.clone());
        callbacks.insert(state_id, state_callbacks);
    }

    install_runtime_callbacks::<TPayload>(&panel_key, &callbacks, &transitions);
    let mut runtime = PanelRuntime {
        handle,
        current_state: default,
        transitions,
        callbacks,
        observer_guards: Vec::new(),
    };
    runtime.observer_guards = attach_observers(&runtime.handle, &observers);
    register_runtime(panel_key.clone(), runtime);

    Ok(PanelRuntimeHandle { key: panel_key })
}

fn build_stateless<TPayload: PanelPayload>(
    handle: PanelHandle<TPayload>,
    definition: PanelStateDefinition<TPayload>,
    observers: Vec<Arc<dyn PanelStyleListener<TPayload>>>,
    quad_vertex: QuadBatchKind,
) -> Result<PanelRuntimeHandle, DbError> {
    let state_id = UiState(0);
    let panel_key = handle.key.clone();

    handle.mutate(|record| {
        record.default_state = Some(state_id);
        record.current_state = state_id;
        record.states.clear();
        record.snapshot.quad_vertex = quad_vertex;
        record.states.insert(state_id, definition.overrides.clone());
    })?;

    let PanelStateDefinition {
        overrides,
        vertex_shader,
        frag_shader,
        callbacks,
        animations: _animations,
        groups: _groups,
        group_relations: _group_relations,
    } = definition;

    if let Some(shader) = frag_shader.as_ref() {
        submit_shader_request(&panel_key, state_id, shader, ShaderStage::Fragment);
    }
    if let Some(shader) = vertex_shader.as_ref() {
        submit_shader_request(&panel_key, state_id, shader, ShaderStage::Vertex);
    }

    let transitions = HashMap::from([(state_id, overrides.transitions.clone())]);
    let callbacks = HashMap::from([(state_id, callbacks)]);

    install_runtime_callbacks::<TPayload>(&panel_key, &callbacks, &transitions);
    let mut runtime = PanelRuntime {
        handle,
        current_state: state_id,
        transitions,
        callbacks,
        observer_guards: Vec::new(),
    };
    runtime.observer_guards = attach_observers(&runtime.handle, &observers);
    register_runtime(panel_key.clone(), runtime);

    Ok(PanelRuntimeHandle { key: panel_key })
}

/// Shader closure placeholder to maintain API parity.
pub type FragClosure = Arc<dyn Fn(&ShaderScope) -> Expr + Send + Sync + 'static>;
pub type VertexClosure = FragClosure;

#[derive(Default)]
pub struct ShaderScope;

/// Example listener that can only mutate its own panel via the guard method.
pub struct CountResetListener;

impl PanelStyleListener<UiPanelData> for CountResetListener {
    fn on_change(&self, change: &PanelStyleChange<'_, UiPanelData>) {
        if let Some(old) = &change.old {
            if old.data.count != change.new.data.count && change.new.data.count > 5 {
                change.mutate(|record| record.data.count = 0);
            }
        }
    }
}

/// Example payload type used in the demo.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct UiPanelData {
    pub count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct TestCustomData {
    pub count: u32,
}

struct CounterHeader;

fn build_demo_panel_with_uuid(panel_uuid: &'static str) -> Result<PanelRuntimeHandle, DbError> {
    configure_group::<CounterHeader, _>(|group| group.center(GroupCenterMode::FirstElement));
    let runtime = Mui::<TestCustomData>::stateful(panel_uuid)?
        .default_state(UiState(0))
        .quad_vertex(QuadBatchKind::UltraVertex)
        .state(UiState(0), |state| {
            let state = state
                .group::<CounterHeader>()
                .size(vec2(100.0, 100.0))
                .position(vec2(333.0, 333.0))
                .border(BorderStyle {
                    color: [1.0, 0.0, 0.0, 1.0],
                    width: 10.0,
                    radius: 0.0,
                })
                .z_index(5)
                .fragment_shader(|_flow| {
                    let r= rv("uv").x();
                    let g= rv("uv").y();
                    wvec4(r, g,1.0,1.0)
                }
                )
                .events()
                .on_event(
                    UiEventKind::Click,
                    |flow: &mut EventFlow<'_, TestCustomData>| {
                        let data = flow.payload();
                        data.count += 1;
                        println!("内部点击事件 {}", data.count);
                        flow.set_state(UiState(1));
                    },
                )
                .finish()
                .state_transform_fade(0.2);
            state
                .events()
                .on_event(UiEventKind::Drag, |flow| {
                    {
                        let data = flow.payload();
                        data.count += 1;
                        flow.set_state(UiState(1));
                    }
                    flow.push_animation(
                        AnimBuilder::new(AnimProperty::Size)
                            .to(vec2(3.0, 3.0))
                            .build(),
                    );
                })
                .finish()
        })
        .state(UiState(1), |state| {
            state
                .size(vec2(200.0, 100.0))
                .position(vec2(433.0, 333.0))
                .color(Vec4::new(1.0, 0.0, 0.8, 1.0))
                .z_index(5)
                .border(BorderStyle {
                    color: [1.0, 0.0, 0.0, 0.0],
                    width: 5.0,
                    radius: 1.0,
                })
                .events()
                .on_event(UiEventKind::Click, |flow| {
                    flow.payload().count += 1;
                    flow.set_state(UiState(2));
                })
                .finish()
        })
        .state(UiState(2), |state| {
            state
                .position(vec2(0.0, 0.0))
                .events()
                .on_event(UiEventKind::Click, |flow| {
                    flow.payload().count += 1;
                    flow.set_state(UiState(0));
                })
                .finish()
        })
        .build()?;

    Ok(runtime)
}

/// Demonstration that mirrors the existing builder usage.
pub fn build_demo_panel() -> Result<PanelRuntimeHandle, DbError> {
    build_demo_panel_with_uuid("inventory_panel")
}

/// Pretty print helper to inspect panel record contents.
impl<TPayload: PanelPayload> fmt::Display for PanelRecord<TPayload> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PanelRecord(default_state={:?})", self.default_state)?;
        writeln!(f, "  data: ... (payload omitted)")?;
        for (state, data) in &self.states {
            writeln!(
                f,
                "  state {:?}: texture={:?}, fit={:?}, z={:?}, pos={:?}",
                state, data.texture, data.fit_to_texture, data.z_index, data.position,
            )?;
        }
        Ok(())
    }
}

/// Panel ID allocator mapping UUID strings to incremental numeric IDs.
struct PanelIdPool {
    map: Mutex<HashMap<String, u32>>,
    next_id: AtomicU32,
}

impl PanelIdPool {
    fn id_for(&self, uuid: &str) -> u32 {
        let mut guard = self.map.lock().unwrap();
        if let Some(&id) = guard.get(uuid) {
            return id;
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        guard.insert(uuid.to_owned(), id);
        id
    }
}

fn panel_id_pool() -> &'static PanelIdPool {
    static POOL: OnceLock<PanelIdPool> = OnceLock::new();
    POOL.get_or_init(|| PanelIdPool {
        map: Mutex::new(HashMap::new()),
        next_id: AtomicU32::new(0),
    })
}

pub fn panel_numeric_id(uuid: &str) -> u32 {
    panel_id_pool().id_for(uuid)
}

fn next_listener_id() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Example listener registration.
pub fn register_demo_listener(handle: &Mui<UiPanelData>) -> PanelListenerGuard<UiPanelData> {
    handle.register_style_listener(CountResetListener)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PANEL_UUID: &str = "inventory_panel_test_case";

    fn read_counter_record(key: &PanelKey) -> PanelRecord<UiPanelData> {
        PanelHandle::<UiPanelData>::for_key(key.clone())
            .read()
            .expect("panel record to exist")
    }

    #[test]
    fn demo_panel_states_and_events_roundtrip() {
        let handle = build_demo_panel_with_uuid(TEST_PANEL_UUID).expect("demo panel builds");

        let mut record = read_counter_record(handle.key());
        assert_eq!(record.default_state, Some(UiState(0)));
        assert_eq!(record.current_state, UiState(0));
        assert_eq!(record.data.count, 0);
        assert_eq!(record.states.len(), 2);

        let primary = record
            .states
            .get(&UiState(0))
            .expect("primary state registered");
        assert_eq!(primary.texture.as_deref(), Some("caton (2).png"));
        assert_eq!(primary.fit_to_texture, Some(true));
        assert_eq!(primary.state_transform_fade, Some(0.2));
        assert_eq!(
            primary.transitions.get(&UiEventKind::Click),
            Some(&UiState(1))
        );

        let secondary = record
            .states
            .get(&UiState(1))
            .expect("secondary state registered");
        assert_eq!(secondary.texture.as_deref(), Some("caton (3).png"));
        assert_eq!(
            secondary.transitions.get(&UiEventKind::Click),
            Some(&UiState(0))
        );

        handle.trigger::<UiPanelData>(UiEventKind::Click);
        record = read_counter_record(handle.key());
        assert_eq!(record.current_state, UiState(1));
        assert_eq!(record.data.count, 1);
        assert_eq!(record.pending_animations.len(), 1);

        handle.trigger::<UiPanelData>(UiEventKind::Click);
        record = read_counter_record(handle.key());
        assert_eq!(record.current_state, UiState(0));
        assert_eq!(record.data.count, 2);
    }
}


