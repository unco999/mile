use crate::{
    mui_anim::{AnimBuilder, AnimProperty, AnimationSpec, Easing},
    mui_rel::{
        Field, RelComposer, RelContainerSpec, RelGraphDefinition, RelLayoutKind, RelScrollAxis,
        RelSpace, RelViewKey, panel_field,
    },
    mui_style::{PanelStylePatch, StyleError, load_panel_style},
    runtime::{
        QuadBatchKind, panel_position, register_payload_refresh,
        relations::register_panel_relations,
        state::{
            CpuPanelEvent, DataChangeEnvelope, PanelEventRegistry, PanelStyleRewrite,
            StateConfigDes, StateOpenCall, StateTransition, UIEventHub, UiInteractionScope,
        },
    },
    structs::{PanelField, PanelInteraction},
};
use glam::{Vec2, Vec3, Vec4, vec2, vec3, vec4};
use mile_api::{
    global::{global_db, global_event_bus},
    prelude::_ty::PanelId,
};
use mile_db::{DbError, TableBinding, TableHandle};
use mile_font::{
    event::{BatchFontEntry, BatchRenderFont, RemoveRenderFont},
    prelude::FontStyle,
};
use mile_gpu_dsl::{
    core::{
        Expr,
        dsl::{IF, modulo, sin, sqrt, wvec2, wvec3, wvec4},
    },
    dsl::{cv, rv, smoothstep},
    gpu_ast_core::event::{ExprTy, ExprWithIdxEvent},
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
static PENDING_REGISTRATIONS: OnceLock<
    Mutex<Vec<Box<dyn Fn(&mut PanelEventRegistry) + Send + 'static>>>,
> = OnceLock::new();
static PENDING_STATE_EVENTS: OnceLock<Mutex<Vec<StateTransition>>> = OnceLock::new();
static PANEL_KEY_REGISTRY: OnceLock<Mutex<HashMap<TypeId, HashSet<PanelKey>>>> = OnceLock::new();
static PENDING_SHADERS: OnceLock<Mutex<HashMap<u32, PendingShaderRequest>>> = OnceLock::new();
static NEXT_SHADER_IDX: AtomicU32 = AtomicU32::new(1);

// Typed observers (erased): subscribe to payload type by TypeId, handler receives &dyn Any payload
type ErasedTypeCb = Arc<dyn Fn(&PanelKey, &dyn Any) + Send + Sync + 'static>;
static TYPE_OBSERVERS: OnceLock<Mutex<HashMap<TypeId, Vec<ErasedTypeCb>>>> = OnceLock::new();

fn type_observers() -> &'static Mutex<HashMap<TypeId, Vec<ErasedTypeCb>>> {
    TYPE_OBSERVERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_type_observer_erased(ty: TypeId, handler: ErasedTypeCb) {
    let mut map = type_observers().lock().unwrap();
    map.entry(ty).or_default().push(handler);
}

fn notify_type_observers_erased(ty: TypeId, source: &PanelKey, payload: &dyn Any) {
    let map = type_observers().lock().unwrap();
    if let Some(list) = map.get(&ty) {
        for cb in list.iter() {
            cb(source, payload);
        }
    }
}

fn pending_registrations()
-> &'static Mutex<Vec<Box<dyn Fn(&mut PanelEventRegistry) + Send + 'static>>> {
    PENDING_REGISTRATIONS.get_or_init(|| Mutex::new(Vec::new()))
}

fn pending_state_events() -> &'static Mutex<Vec<StateTransition>> {
    PENDING_STATE_EVENTS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Install any pending event registrations into the runtime registry.
/// This should be called at a safe point (e.g. at the start of event polling)
/// to avoid re-entrant locking during user callbacks.
pub fn drain_pending_event_registrations() {
    if let Some(registry_arc) = runtime_event_registry() {
        let mut pending = pending_registrations().lock().unwrap();
        if pending.is_empty() {
            return;
        }
        if let Ok(mut registry) = registry_arc.lock() {
            for task in pending.drain(..) {
                task(&mut registry);
            }
        }
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderStage {
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

    pub fn from_expr_ty(expr_ty: ExprTy) -> Self {
        match expr_ty {
            ExprTy::Frag => ShaderStage::Fragment,
            ExprTy::Vertex => ShaderStage::Vertex,
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
    pub fn new(panel_uuid: impl AsRef<str>, scope: impl Into<String>) -> Self {
        let s = panel_uuid.as_ref();
        let panel_id = panel_id_pool().id_for(s);
        Self {
            panel_id,
            panel_uuid: s.to_owned(),
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
    /// Panel 初始化时调用（构建完成后立即触发一次）
    Init,
    Click,
    /// 通用拖拽位移（仍保留兼容旧逻辑）
    Drag,
    /// 拖拽源：开始/持续/离开/放下
    SourceDragStart,
    SourceDragOver,
    SourceDragLeave,
    SourceDragDrop,
    /// 拖拽目标：进入/停留/离开/放下
    TargetDragEnter,
    TargetDragOver,
    TargetDragLeave,
    TargetDragDrop,
    Hover,
    Out,
}

/// Generic event payload used by `on_event_with`.
#[derive(Clone, Debug)]
pub enum UiEventData {
    None,
    Vec2(glam::Vec2),
    U32(u32),
    Bool(bool),
}

/// Identifies a drag payload by its Rust type and an optional user-provided tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DragPayloadId {
    type_id: TypeId,
    tag: Option<u64>,
}

impl DragPayloadId {
    pub fn of<T: 'static>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            tag: None,
        }
    }

    pub fn with_tag(mut self, tag: u64) -> Self {
        self.tag = Some(tag);
        self
    }

    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    pub fn tag(&self) -> Option<u64> {
        self.tag
    }
}

/// Type-erased payload transferred between drag source/targets.
#[derive(Clone, Debug)]
pub struct DragPayload {
    id: DragPayloadId,
    data: Arc<dyn Any + Send + Sync>,
}

impl DragPayload {
    pub fn new<T>(value: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        Self::with_id(DragPayloadId::of::<T>(), value)
    }

    pub fn with_id<T>(id: DragPayloadId, value: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        Self {
            id,
            data: Arc::new(value),
        }
    }

    pub fn id(&self) -> DragPayloadId {
        self.id
    }

    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: 'static,
    {
        self.data.as_ref().downcast_ref::<T>()
    }

    pub fn downcast_arc<T>(&self) -> Option<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        self.data.clone().downcast::<T>().ok()
    }
}

/// Captures the drag payload and its originating panel.
#[derive(Clone, Debug)]
pub struct DragContext {
    pub source: PanelKey,
    pub payload: DragPayload,
}

impl DragContext {
    pub fn new(source: PanelKey, payload: DragPayload) -> Self {
        Self { source, payload }
    }
}

static GLOBAL_DRAG_CONTEXT: OnceLock<Mutex<Option<DragContext>>> = OnceLock::new();

fn drag_context_store() -> &'static Mutex<Option<DragContext>> {
    GLOBAL_DRAG_CONTEXT.get_or_init(|| Mutex::new(None))
}

fn current_drag_context() -> Option<DragContext> {
    drag_context_store().lock().unwrap().clone()
}

fn set_global_drag_context(ctx: Option<DragContext>) {
    *drag_context_store().lock().unwrap() = ctx;
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
    pub rotation: Option<[f32; 3]>,
    #[serde(default)]
    pub scale: Option<[f32; 3]>,
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
    #[serde(default)]
    pub visible: Option<bool>,
    /// If true, runtime will initialize position from current mouse pos once and then clear this flag.
    #[serde(default)]
    pub trigger_mouse_pos: bool,
    /// Clamp/offset rules applied to specific fields in this state.
    /// - For dims == 1: only X component is used (min[0], max[0], step[0]).
    /// - For dims == 2: X/Y components are applied independently.
    /// - Step <= 0 disables snapping. Otherwise snap to min + round((v-min)/step)*step.
    #[serde(default)]
    pub clamp_offsets: Vec<ClampOffset>,
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
    #[serde(default = "panel_snapshot_rotation_default")]
    pub rotation: [f32; 3],
    #[serde(default = "panel_snapshot_scale_default")]
    pub scale: [f32; 3],
    pub z_index: i32,
    #[serde(default)]
    pub fragment_shader_id: Option<u32>,
    #[serde(default)]
    pub vertex_shader_id: Option<u32>,
    #[serde(default)]
    pub quad_vertex: QuadBatchKind,
    #[serde(default = "panel_snapshot_visible_default")]
    pub visible: bool,
}

impl Default for PanelSnapshot {
    fn default() -> Self {
        Self {
            texture: None,
            fit_to_texture: false,
            position: [0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            border: None,
            rotation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            z_index: 0,
            fragment_shader_id: None,
            vertex_shader_id: None,
            quad_vertex: QuadBatchKind::Normal,
            visible: true,
        }
    }
}

/// Marks that initial position should be taken from current mouse position at spawn.
/// This is a CPU-side convenience; runtime will translate it into a concrete position on first upsert
/// and then clear the flag so future refreshes won't override user changes.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum InitialPosition {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "mouse")]
    Mouse,
}

const fn panel_snapshot_visible_default() -> bool {
    true
}

const fn panel_snapshot_rotation_default() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}

const fn panel_snapshot_scale_default() -> [f32; 3] {
    [1.0, 1.0, 1.0]
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

/// Describes a clamp/offset rule for a specific field.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClampOffset {
    /// Target field to clamp (e.g., Field::PositionX or Field::Position).
    pub field: Field,
    /// Minimum allowed value; Y is used only when `dims == 2`.
    pub min: [f32; 2],
    /// Maximum allowed value; Y is used only when `dims == 2`.
    pub max: [f32; 2],
    /// Step size (quantization); Y is used only when `dims == 2`.
    pub step: [f32; 2],
    /// 1 for scalar fields (use X only), 2 for vec2 fields (use X/Y).
    pub dims: u8,
    /// Flags:
    /// - bit0: relative budget (use `max` as +/- budget around origin, ignore `min`)
    /// - bit1: axis_x_only (freeze Y at current panel.position.y)
    /// - bit2: axis_y_only (freeze X at current panel.position.x)
    pub flags: u32,
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
    #[serde(default)]
    pub change_epoch: u64,
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
            change_epoch: 0,
            data: TPayload::default(),
        }
    }
}

/// Apply initial visual snapshot from a state's overrides so CPU-side snapshot
/// matches what GPU will resolve on first upload.
fn apply_initial_snapshot_from_overrides<TPayload: PanelPayload>(
    record: &mut PanelRecord<TPayload>,
    overrides: &PanelStateOverrides,
) {
    if let Some(texture) = &overrides.texture {
        record.snapshot.texture = Some(texture.clone());
    }
    if let Some(fit) = overrides.fit_to_texture {
        record.snapshot.fit_to_texture = fit;
    }
    if let Some(pos) = overrides.position {
        record.snapshot.position = pos;
    }
    if let Some(col) = overrides.color {
        record.snapshot.color = col;
    }
    if let Some(border) = overrides.border.clone() {
        record.snapshot.border = Some(border);
    }
    if let Some(rotation) = overrides.rotation {
        record.snapshot.rotation = rotation;
    }
    if let Some(scale) = overrides.scale {
        record.snapshot.scale = scale;
    }
    if let Some(z) = overrides.z_index {
        record.snapshot.z_index = z;
    }
    if let Some(frag) = overrides.fragment_shader_id {
        record.snapshot.fragment_shader_id = Some(frag);
    }
    if let Some(vs) = overrides.vertex_shader_id {
        record.snapshot.vertex_shader_id = Some(vs);
    }
    if let Some(vis) = overrides.visible {
        record.snapshot.visible = vis;
    }
    // quad_vertex 已在构建流程中单独设置
}

/// Event emitted whenever a panel record is updated.
#[derive(Clone, Debug)]
pub struct PanelStateChanged<TPayload: PanelPayload> {
    pub key: PanelKey,
    pub old: Option<PanelRecord<TPayload>>,
    pub new: PanelRecord<TPayload>,
}

/// Erased DB change event that carries runtime type info and a JSON snapshot of the payload.
/// - `payload_type` is the Rust type name (e.g., "mile_ui::mui_prototype::UiPanelData")
/// - `payload_json` is serialized from `new.data` for dynamic consumers
#[derive(Clone, Debug)]
pub struct PanelStateChangedErased {
    pub key: PanelKey,
    pub payload_type: &'static str,
    pub payload_json: String,
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
        // Also publish an erased variant so dynamic listeners (unknown T at compile time)
        // can react based on `payload_type` and `payload_json`.
        let payload_json = match serde_json::to_string(&new.data) {
            Ok(s) => s,
            Err(_) => "null".to_string(),
        };
        global_event_bus().publish(PanelStateChangedErased {
            key: key.clone(),
            payload_type: type_name::<TPayload>(),
            payload_json,
        });
        // Notify typed observers with erased payload dispatch.
        let ty = TypeId::of::<TPayload>();
        let payload_any: &dyn Any = &new.data;
        notify_type_observers_erased(ty, key, payload_any);
        // Bridge into UI event hub so panels can receive on_data_change during event processing.
        if let Some(hub) = runtime_event_hub() {
            hub.push(CpuPanelEvent::DataChange(DataChangeEnvelope {
                source_uuid: key.panel_uuid.clone(),
                payload_type: ty,
                payload: Arc::new(new.data.clone()),
            }));
        }

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
    fn new(panel_uuid: impl AsRef<str>, scope: impl Into<String>) -> Result<Self, DbError> {
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
    data_callbacks: &HashMap<UiState, Vec<DataChangeCbEntry<TPayload>>>,
    callbacks_with: &HashMap<UiState, HashMap<UiEventKind, Arc<EventFnWith<TPayload>>>>,
) {
    let mut states: HashSet<UiState> = HashSet::new();
    states.extend(callbacks.keys().copied());
    states.extend(callbacks_with.keys().copied());
    states.extend(data_callbacks.keys().copied());
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
        if let Some(map) = callbacks_with.get(&state) {
            events.extend(map.keys().copied());
        }
        if let Some(map) = transitions.get(&state) {
            events.extend(map.keys().copied());
        }

        for event in events {
            let key_clone = key.clone();
            let state_copy = state;
            let scope_copy = scope;
            match event {
                UiEventKind::Init => { /* init is triggered programmatically once; no runtime registry */
                }
                UiEventKind::Click => registry.register_click(scope_copy, move |_ignored| {
                    trigger_event_internal_with::<TPayload>(
                        &key_clone,
                        Some(state_copy),
                        UiEventKind::Click,
                        UiEventData::None,
                    );
                }),
                UiEventKind::Drag => registry.register_drag(scope_copy, move |vec2| {
                    trigger_event_internal_with::<TPayload>(
                        &key_clone,
                        Some(state_copy),
                        UiEventKind::Drag,
                        UiEventData::Vec2(vec2),
                    );
                }),
                UiEventKind::SourceDragStart => {
                    registry.register_source_drag_start(scope_copy, move |_id| {
                        trigger_event_internal_with::<TPayload>(
                            &key_clone,
                            Some(state_copy),
                            UiEventKind::SourceDragStart,
                            UiEventData::None,
                        );
                    })
                }
                UiEventKind::SourceDragOver => registry.register_drag(scope_copy, move |vec2| {
                    trigger_event_internal_with::<TPayload>(
                        &key_clone,
                        Some(state_copy),
                        UiEventKind::SourceDragOver,
                        UiEventData::Vec2(vec2),
                    );
                }),
                UiEventKind::SourceDragLeave => (),
                UiEventKind::SourceDragDrop => {
                    registry.register_source_drag_drop(scope_copy, move |_id| {
                        trigger_event_internal_with::<TPayload>(
                            &key_clone,
                            Some(state_copy),
                            UiEventKind::SourceDragDrop,
                            UiEventData::None,
                        );
                    })
                }
                UiEventKind::TargetDragEnter => {
                    registry.register_target_drag_enter(scope_copy, move |_id| {
                        trigger_event_internal_with::<TPayload>(
                            &key_clone,
                            Some(state_copy),
                            UiEventKind::TargetDragEnter,
                            UiEventData::None,
                        );
                    })
                }
                UiEventKind::TargetDragOver => {
                    registry.register_target_drag_over(scope_copy, move |vec2| {
                        trigger_event_internal_with::<TPayload>(
                            &key_clone,
                            Some(state_copy),
                            UiEventKind::TargetDragOver,
                            UiEventData::Vec2(vec2),
                        );
                    })
                }
                UiEventKind::TargetDragLeave => {
                    registry.register_target_drag_leave(scope_copy, move |_id| {
                        trigger_event_internal_with::<TPayload>(
                            &key_clone,
                            Some(state_copy),
                            UiEventKind::TargetDragLeave,
                            UiEventData::None,
                        );
                    })
                }
                UiEventKind::TargetDragDrop => {
                    registry.register_target_drag_drop(scope_copy, move |_id| {
                        trigger_event_internal_with::<TPayload>(
                            &key_clone,
                            Some(state_copy),
                            UiEventKind::TargetDragDrop,
                            UiEventData::None,
                        );
                    })
                }
                UiEventKind::Hover => registry.register_hover(scope_copy, move |_ignored| {
                    trigger_event_internal_with::<TPayload>(
                        &key_clone,
                        Some(state_copy),
                        UiEventKind::Hover,
                        UiEventData::None,
                    );
                }),
                UiEventKind::Out => registry.register_out(scope_copy, move |_ignored| {
                    trigger_event_internal_with::<TPayload>(
                        &key_clone,
                        Some(state_copy),
                        UiEventKind::Out,
                        UiEventData::None,
                    );
                }),
            }
        }
        // Register data change listeners for this scope/state
        if let Some(list) = data_callbacks.get(&state) {
            for entry in list {
                let key_clone = key.clone();
                let state_copy = state;
                let ty = entry.ty;
                let src = entry.source_uuid.clone();
                registry.register_data_change(scope, ty, src, move |env| {
                    trigger_data_change_for::<TPayload>(&key_clone, state_copy, env);
                });
            }
        }
    }
}

fn install_runtime_callbacks<TPayload: PanelPayload + 'static>(
    key: &PanelKey,
    callbacks: &HashMap<UiState, HashMap<UiEventKind, Arc<EventFn<TPayload>>>>,
    transitions: &HashMap<UiState, HashMap<UiEventKind, UiState>>,
    data_callbacks: &HashMap<UiState, Vec<DataChangeCbEntry<TPayload>>>,
    callbacks_with: &HashMap<UiState, HashMap<UiEventKind, Arc<EventFnWith<TPayload>>>>,
) {
    // 统一入队，避免在回调持锁时重入；在 event_poll 或 runtime 初始化桥接时统一安装。
    let key_clone = key.clone();
    let callbacks_cloned = callbacks.clone();
    let transitions_cloned = transitions.clone();
    let data_map_cloned = data_callbacks.clone();
    let callbacks_with_cloned = callbacks_with.clone();
    let mut pending = pending_registrations().lock().unwrap();
    pending.push(Box::new(move |registry: &mut PanelEventRegistry| {
        apply_runtime_callbacks::<TPayload>(
            registry,
            &key_clone,
            &callbacks_cloned,
            &transitions_cloned,
            &data_map_cloned,
            &callbacks_with_cloned,
        );
    }));
}

fn trigger_data_change_for<TPayload: PanelPayload>(
    key: &PanelKey,
    state: UiState,
    env: &crate::runtime::state::DataChangeEnvelope,
) {
    // Lookup runtime
    let arc = runtime_map::<TPayload>();
    let mut registry = arc.lock().unwrap();
    let Some(runtime) = registry.get_mut(key) else {
        return;
    };
    // Prepare snapshot arguments
    let mut snapshot = match runtime.handle.read() {
        Ok(record) => record,
        Err(_) => PanelRecord::<TPayload>::default(),
    };
    if let Some(pos) = panel_position(key.panel_id) {
        snapshot.snapshot.position = pos;
    }
    // Build args
    let args = PanelEventArgs {
        panel_key: key.clone(),
        state,
        event: UiEventKind::Init, // synthetic
        record_snapshot: snapshot,
    };
    // Execute handlers for this state
    if let Some(list) = runtime.data_callbacks.get_mut(&state) {
        let handle = &runtime.handle;
        let transitions = &runtime.transitions;
        let current_state_ref = &mut runtime.current_state;
        let mut updated_drag: Option<DragContext> = None;
        if let Err(err) = handle.mutate(|record| {
            let mut flow = EventFlow::new(
                record,
                &args,
                current_state_ref,
                transitions,
                current_drag_context(),
            );
            for entry in list {
                if entry.ty == env.payload_type {
                    if let Some(ref src) = entry.source_uuid {
                        if src != &env.source_uuid {
                            continue;
                        }
                    }
                    (entry.handler)(env.payload.as_ref(), &mut flow);
                }
            }
            updated_drag = flow.take_drag_context();
        }) {
            eprintln!("data change mutation failed: {err:?}");
        }
        if let Some(ctx) = updated_drag {
            set_global_drag_context(Some(ctx.clone()));
            runtime.active_drag = Some(ctx);
        }
    }
}

/// Runtime registry storing event handlers and the current state of each panel.
struct PanelRuntime<TPayload: PanelPayload> {
    handle: PanelHandle<TPayload>,
    current_state: UiState,
    transitions: HashMap<UiState, HashMap<UiEventKind, UiState>>,
    callbacks: HashMap<UiState, HashMap<UiEventKind, Arc<EventFn<TPayload>>>>,
    callbacks_with: HashMap<UiState, HashMap<UiEventKind, Arc<EventFnWith<TPayload>>>>,
    observer_guards: Vec<PanelListenerGuard<TPayload>>,
    data_callbacks: HashMap<UiState, Vec<DataChangeCbEntry<TPayload>>>,
    /// Active drag context shared across handlers (source writes, targets read).
    active_drag: Option<DragContext>,
}
type EventFn<TPayload> = dyn for<'a> Fn(&mut EventFlow<'a, TPayload>) + Send + Sync + 'static;
type EventFnWith<TPayload> =
    dyn for<'a> Fn(&mut EventFlow<'a, TPayload>, UiEventData) + Send + Sync + 'static;
type DataChangeErasedFn<TPayload> =
    dyn for<'a> Fn(&dyn Any, &mut EventFlow<'a, TPayload>) + Send + Sync + 'static;

#[derive(Clone)]
struct DataChangeCbEntry<TPayload: PanelPayload> {
    ty: TypeId,
    source_uuid: Option<String>,
    handler: Arc<DataChangeErasedFn<TPayload>>,
}

#[derive(Clone, Debug)]
pub struct PanelEventArgs<TPayload: PanelPayload> {
    pub panel_key: PanelKey,
    pub state: UiState,
    pub event: UiEventKind,
    pub record_snapshot: PanelRecord<TPayload>,
}

#[derive(Debug)]
pub struct EventFlow<'a, TPayload: PanelPayload> {
    pub record: &'a mut PanelRecord<TPayload>,
    pub args: &'a PanelEventArgs<TPayload>,
    pub current_state: &'a mut UiState,
    pub transitions: &'a HashMap<UiState, HashMap<UiEventKind, UiState>>,
    pub override_state: Option<UiState>,
    /// Active drag context if any (source sets, targets read).
    drag_context: Option<DragContext>,
}

impl<'a, TPayload: PanelPayload> EventFlow<'a, TPayload> {
    fn new(
        record: &'a mut PanelRecord<TPayload>,
        args: &'a PanelEventArgs<TPayload>,
        current_state: &'a mut UiState,
        transitions: &'a HashMap<UiState, HashMap<UiEventKind, UiState>>,
        drag_context: Option<DragContext>,
    ) -> Self {
        Self {
            record,
            args,
            current_state,
            transitions,
            override_state: None,
            drag_context,
        }
    }

    /// Convenience: publish font events to build text for this panel.
    /// - text: 文本
    /// - font_path: 字体文件路径（建议 tf/...）
    /// - font_size: 像素大小
    /// - color: RGBA
    /// - weight/line_height: 预留参数（直接传入 FontStyle）
    pub fn text(&self, text: &str, style: FontStyle) {
        let pid = PanelId(self.args.panel_key.panel_id);
        // Always clear previous texts for this panel before queuing new one
        global_event_bus().publish(RemoveRenderFont { parent: pid });

        global_event_bus().publish(BatchFontEntry {
            text: Arc::from(text.to_string()),
            font_file_path: style.font_file_path.clone(),
        });
        global_event_bus().publish(BatchRenderFont {
            text: Arc::from(text.to_string()),
            font_file_path: style.font_file_path.clone(),
            parent: pid,
            font_style: Arc::new(style),
        });
    }

    /// 清空当前面板的文字渲染（不影响 SDF/字体缓存）
    pub fn clear_texts(&self) {
        let pid = PanelId(self.args.panel_key.panel_id);
        global_event_bus().publish(mile_font::event::RemoveRenderFont { parent: pid });
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

    /// Queue a CPU panel rewrite signal: set specific panel fields to given values.
    /// `field_bits` use `PanelField` bitflags (see `crate::structs::PanelField`).
    pub fn style_set(&self, field_bits: u32, values: [f32; 4]) {
        if let Some(hub) = runtime_event_hub() {
            hub.push(CpuPanelEvent::PanelStyleRewrite(PanelStyleRewrite {
                panel_id: self.args.panel_key.panel_id,
                field_id: field_bits,
                op: 0,
                values,
            }));
        }
    }

    /// Queue a CPU panel rewrite signal: add offsets to specific panel fields.
    pub fn style_add(&self, field_bits: u32, values: [f32; 4]) {
        if let Some(hub) = runtime_event_hub() {
            hub.push(CpuPanelEvent::PanelStyleRewrite(PanelStyleRewrite {
                panel_id: self.args.panel_key.panel_id,
                field_id: field_bits,
                op: 1,
                values,
            }));
        }
    }

    pub fn position_anim(&mut self) -> PositionAnimFlow<TPayload> {
        let base = Vec2::from_array(self.args.record_snapshot.snapshot.position);
        PositionAnimFlow::<TPayload> {
            builder: AnimBuilder::new(AnimProperty::Position).from_current(),
            base,
            target: None,
            offset_target: false,
            _marker: PhantomData,
            mark_from_snapshot: false,
            mark_to_snapshot: false,
        }
    }

    pub fn color_anim(&mut self) -> ColorAnimFlow<TPayload> {
        let base = Vec4::from_array(self.args.record_snapshot.snapshot.color);
        ColorAnimFlow::<TPayload> {
            builder: AnimBuilder::new(AnimProperty::Color).from_current(),
            base,
            target: None,
            offset_target: false,
            _marker: PhantomData,
            mark_from_snapshot: false,
            mark_to_snapshot: false,
        }
    }

    pub fn args(&self) -> &PanelEventArgs<TPayload> {
        self.args
    }

    /// Get a typed view of the current drag payload if it matches type `T`.
    pub fn drag_payload<T: 'static>(&self) -> Option<&T> {
        self.drag_context
            .as_ref()
            .and_then(|ctx| ctx.payload.downcast_ref::<T>())
    }

    /// Set the active drag payload for this flow; targets can read via `drag_payload`.
    pub fn set_drag_payload<T>(&mut self, payload: T)
    where
        T: Send + Sync + 'static,
    {
        let id = DragPayloadId::of::<T>();
        self.drag_payload_with_id(id, payload);
    }

    /// Set drag payload with an explicit id (e.g., custom tag for matching).
    pub fn drag_payload_with_id<T>(&mut self, id: DragPayloadId, payload: T)
    where
        T: Send + Sync + 'static,
    {
        let payload = DragPayload::with_id(id, payload);
        let source = self.args.panel_key.clone();
        self.drag_context = Some(DragContext::new(source, payload));
    }

    /// Consume and return the current drag context so callers can persist it.
    pub fn take_drag_context(&mut self) -> Option<DragContext> {
        self.drag_context.take()
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

    /// Force this flow to be treated as changed even if payload comparisons are equal.
    /// Useful when external DB mutations need to trigger data change observers.
    pub fn mark_changed(&mut self) {
        self.record.change_epoch = self.record.change_epoch.wrapping_add(1);
    }

    pub fn request_fragment_shader<F>(&self, shader: F)
    where
        F: Fn(&ShaderScope) -> Expr + Send + Sync + 'static,
    {
        let shader: Arc<dyn Fn(&ShaderScope) -> Expr + Send + Sync + 'static> = Arc::new(shader);
        submit_shader_request(
            &self.args.panel_key,
            *self.current_state,
            &shader,
            ShaderStage::Fragment,
        );
    }

    pub fn request_vertex_shader<F>(&self, shader: F)
    where
        F: Fn(&ShaderScope) -> Expr + Send + Sync + 'static,
    {
        let shader: Arc<dyn Fn(&ShaderScope) -> Expr + Send + Sync + 'static> = Arc::new(shader);
        submit_shader_request(
            &self.args.panel_key,
            *self.current_state,
            &shader,
            ShaderStage::Vertex,
        );
    }

    /// Mutate current panel's payload and persist via DB commit (observers will be notified).
    /// This should be used inside event callbacks; the outer mutate will commit and emit.
    pub fn update_self_payload(&mut self, mutator: impl FnOnce(&mut TPayload)) {
        mutator(&mut self.record.data);
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

/// Lightweight flow used inside type-based observers. It allows mutating the
/// receiving panel's payload and enqueueing animations, similar to EventFlow.
pub struct ObserveFlow<'a, TPayload: PanelPayload> {
    record: &'a mut PanelRecord<TPayload>,
}

impl<'a, TPayload: PanelPayload> ObserveFlow<'a, TPayload> {
    pub fn payload(&mut self) -> &mut TPayload {
        &mut self.record.data
    }

    pub fn push_animation(&mut self, animation: AnimationSpec) {
        self.record.pending_animations.push(animation);
    }
}

pub struct PositionAnimFlow<TPayload: PanelPayload> {
    builder: AnimBuilder,
    base: Vec2,
    target: Option<Vec2>,
    offset_target: bool,
    mark_from_snapshot: bool,
    mark_to_snapshot: bool,
    _marker: PhantomData<TPayload>,
}

impl<TPayload: PanelPayload> PositionAnimFlow<TPayload> {
    pub fn offset(mut self, delta: Vec2) -> Self {
        self.target = Some(delta);
        self.offset_target = true;
        self
    }

    pub fn to_offset(self, delta: Vec2) -> Self {
        self.offset(delta)
    }

    pub fn to_snapshot(mut self) -> Self {
        self.target = Some(self.base);
        self.offset_target = false;
        self.mark_to_snapshot = true;
        self
    }

    pub fn to(mut self, target: Vec2) -> Self {
        self.target = Some(target);
        self.offset_target = false;
        self
    }

    pub fn from_snapshot(mut self) -> Self {
        self.builder = self.builder.from(self.base);
        self.mark_from_snapshot = true;
        self
    }

    pub fn from_current(mut self) -> Self {
        self.builder = self.builder.from_current();
        self.mark_from_snapshot = false;
        self
    }

    pub fn from_offset(mut self, delta: Vec2) -> Self {
        self.builder = self.builder.from(self.base + delta);
        self.mark_from_snapshot = false;
        self
    }

    pub fn duration(mut self, seconds: f32) -> Self {
        self.builder = self.builder.duration(seconds);
        self
    }

    pub fn delay(mut self, seconds: f32) -> Self {
        self.builder = self.builder.delay(seconds);
        self
    }

    pub fn easing(mut self, easing: Easing) -> Self {
        self.builder = self.builder.easing(easing);
        self
    }

    pub fn loop_count(mut self, count: u32) -> Self {
        self.builder = self.builder.loop_count(count);
        self
    }

    pub fn infinite(mut self) -> Self {
        self.builder = self.builder.infinite();
        self
    }

    pub fn ping_pong(mut self, enabled: bool) -> Self {
        self.builder = self.builder.ping_pong(enabled);
        self
    }

    pub fn push(mut self, flow: &mut EventFlow<'_, TPayload>) {
        let mut builder = if self.offset_target {
            let delta = self.target.unwrap_or(Vec2::ZERO);
            self.builder.to_offset(delta)
        } else {
            let target = self.target.unwrap_or(self.base);
            self.builder.to(target)
        };
        if self.mark_from_snapshot {
            builder = builder.mark_from_snapshot();
        }
        if self.mark_to_snapshot {
            builder = builder.mark_to_snapshot();
        }
        flow.push_animation(builder.build());
    }
}

pub struct ColorAnimFlow<TPayload: PanelPayload> {
    builder: AnimBuilder,
    base: Vec4,
    target: Option<Vec4>,
    offset_target: bool,
    mark_from_snapshot: bool,
    mark_to_snapshot: bool,
    _marker: PhantomData<TPayload>,
}

impl<TPayload: PanelPayload> ColorAnimFlow<TPayload> {
    pub fn offset(mut self, delta: Vec4) -> Self {
        self.target = Some(delta);
        self.offset_target = true;
        self
    }

    pub fn to_offset(self, delta: Vec4) -> Self {
        self.offset(delta)
    }

    pub fn to_snapshot(mut self) -> Self {
        self.target = Some(self.base);
        self.offset_target = false;
        self.mark_to_snapshot = true;
        self
    }

    pub fn to(mut self, target: Vec4) -> Self {
        self.target = Some(target);
        self.offset_target = false;
        self
    }

    pub fn from_snapshot(mut self) -> Self {
        self.builder = self.builder.from(self.base);
        self.mark_from_snapshot = true;
        self
    }

    pub fn from_current(mut self) -> Self {
        self.builder = self.builder.from_current();
        self.mark_from_snapshot = false;
        self
    }

    pub fn duration(mut self, seconds: f32) -> Self {
        self.builder = self.builder.duration(seconds);
        self
    }

    pub fn delay(mut self, seconds: f32) -> Self {
        self.builder = self.builder.delay(seconds);
        self
    }

    pub fn easing(mut self, easing: Easing) -> Self {
        self.builder = self.builder.easing(easing);
        self
    }

    pub fn loop_count(mut self, count: u32) -> Self {
        self.builder = self.builder.loop_count(count);
        self
    }

    pub fn infinite(mut self) -> Self {
        self.builder = self.builder.infinite();
        self
    }

    pub fn ping_pong(mut self, enabled: bool) -> Self {
        self.builder = self.builder.ping_pong(enabled);
        self
    }

    pub fn push(mut self, flow: &mut EventFlow<'_, TPayload>) {
        let mut builder = if self.offset_target {
            let delta = self.target.unwrap_or(Vec4::ZERO);
            self.builder.to_offset(delta)
        } else {
            let target = self.target.unwrap_or(self.base);
            self.builder.to(target)
        };
        if self.mark_from_snapshot {
            builder = builder.mark_from_snapshot();
        }
        if self.mark_to_snapshot {
            builder = builder.mark_to_snapshot();
        }
        flow.push_animation(builder.build());
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
        if let Some(color) = override_ref.color {
            config.color = Some(color);
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

static MAPS: OnceLock<Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>> = OnceLock::new();

fn runtime_map<TPayload: PanelPayload>() -> Arc<Mutex<HashMap<PanelKey, PanelRuntime<TPayload>>>> {
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
    println!("当前构建UI元素 {:?}", key);
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

    let mut snapshot = match runtime.handle.read() {
        Ok(record) => record,
        Err(err) => {
            eprintln!("failed to read panel record: {err:?}");
            PanelRecord::<TPayload>::default()
        }
    };
    if let Some(pos) = panel_position(key.panel_id) {
        snapshot.snapshot.position = pos;
    }

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
        let mut updated_drag: Option<DragContext> = None;
        if let Err(err) = handle.mutate(|record| {
            let mut flow = EventFlow::new(
                record,
                &args,
                current_state_ref,
                transitions,
                current_drag_context(),
            );
            handler(&mut flow);
            applied_override = flow.take_override().is_some();
            updated_drag = flow.take_drag_context();
        }) {
            eprintln!("event mutation failed: {err:?}");
        }
        if let Some(ctx) = updated_drag {
            set_global_drag_context(Some(ctx.clone()));
            runtime.active_drag = Some(ctx);
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

fn trigger_event_internal_with<TPayload: PanelPayload>(
    key: &PanelKey,
    forced_state: Option<UiState>,
    event: UiEventKind,
    data: UiEventData,
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

    let mut snapshot = match runtime.handle.read() {
        Ok(record) => record,
        Err(err) => {
            eprintln!("failed to read panel record: {err:?}");
            PanelRecord::<TPayload>::default()
        }
    };
    if let Some(pos) = panel_position(key.panel_id) {
        snapshot.snapshot.position = pos;
    }

    let args = PanelEventArgs {
        panel_key: key.clone(),
        state,
        event,
        record_snapshot: snapshot,
    };

    if let Some(map) = runtime.callbacks_with.get(&state) {
        if let Some(handler) = map.get(&event) {
            let handle = &runtime.handle;
            let transitions = &runtime.transitions;
            let current_state_ref = &mut runtime.current_state;
            let mut updated_drag: Option<DragContext> = None;
            if let Err(err) = handle.mutate(|record| {
                let mut flow = EventFlow::new(
                    record,
                    &args,
                    current_state_ref,
                    transitions,
                    current_drag_context(),
                );
                handler(&mut flow, data);
                updated_drag = flow.take_drag_context();
            }) {
                eprintln!("event mutation failed: {err:?}");
            }
            if let Some(ctx) = updated_drag {
                set_global_drag_context(Some(ctx.clone()));
                runtime.active_drag = Some(ctx);
            } else if matches!(
                event,
                UiEventKind::SourceDragDrop | UiEventKind::SourceDragLeave
            ) {
                // 清理拖拽上下文
                runtime.active_drag = None;
                set_global_drag_context(None);
            }
            return;
        }
    }
    // Fallback
    drop(registry);
    trigger_event_internal::<TPayload>(key, Some(state), event);
}

pub fn trigger_event<TPayload: PanelPayload>(key: &PanelKey, event: UiEventKind) {
    trigger_event_internal::<TPayload>(key, None, event);
}

/// Handle returned by `build` so tests can trigger events.
#[derive(Clone)]
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
    callbacks_with: HashMap<UiEventKind, Arc<EventFnWith<TPayload>>>,
    // Data change callbacks registered via events().on_data_change
    data_callbacks: Vec<DataChangeCbEntry<TPayload>>,
    animations: Vec<AnimationSpec>,
    relations: Option<RelGraphDefinition>,
}

impl<TPayload: PanelPayload> Default for PanelStateDefinition<TPayload> {
    fn default() -> Self {
        Self {
            overrides: PanelStateOverrides::default(),
            frag_shader: None,
            vertex_shader: None,
            callbacks: HashMap::new(),
            callbacks_with: HashMap::new(),
            data_callbacks: Vec::new(),
            animations: Vec::new(),
            relations: None,
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
    fn derived_scope<S: AsRef<str>>(panel_uuid: S) -> String {
        format!("{}::{}", type_name::<TPayload>(), panel_uuid.as_ref())
    }

    pub fn new<S: AsRef<str>>(panel_uuid: S) -> Result<Self, DbError> {
        let scope = Self::derived_scope(&panel_uuid);
        Self::stateful_with_scope(panel_uuid, scope)
    }

    pub fn stateful_with_scope(
        panel_uuid: impl AsRef<str>,
        scope: impl Into<String>,
    ) -> Result<Self, DbError> {
        TPayload::register_payload_type();
        let handle = PanelHandle::<TPayload>::new(panel_uuid.as_ref(), scope)?;
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

    pub fn stateless<S: AsRef<str>>(panel_uuid: S) -> Result<Self, DbError> {
        let scope = Self::derived_scope(&panel_uuid);
        Self::stateless_with_scope(panel_uuid, scope)
    }

    pub fn stateless_with_scope(
        panel_uuid: impl AsRef<str>,
        scope: impl Into<String>,
    ) -> Result<Self, DbError> {
        TPayload::register_payload_type();
        let handle = PanelHandle::<TPayload>::new(panel_uuid.as_ref(), scope)?;
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
        let builder = configure(StateStageBuilder::new(state_id, &self.handle.key));
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
                let builder = configure(StateStageBuilder::new(UiState(0), &self.handle.key));
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
            } => build_stateful::<TPayload>(
                handle,
                default_state,
                states,
                self.observers,
                quad_vertex,
            ),
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

    /// Observe DB changes for any panels with payload type `TObserved`.
    /// The handler receives the observed payload (new value) and a flow to mutate THIS panel.
    pub fn observe_type<TObserved, Handler>(mut self, handler: Handler) -> Self
    where
        TObserved: PanelPayload,
        Handler: Fn(&TObserved, &mut ObserveFlow<TPayload>) + Send + Sync + 'static,
    {
        let self_handle = self.handle.clone();
        let ty = TypeId::of::<TObserved>();
        // Erased wrapper: downcast payload to TObserved then invoke user handler
        let cb: ErasedTypeCb = Arc::new(move |source, any_payload| {
            if let Some(payload) = any_payload.downcast_ref::<TObserved>() {
                // Mutate THIS panel to provide a flow, then run handler
                let _ = self_handle.mutate(|record| {
                    let mut flow = ObserveFlow { record };
                    handler(payload, &mut flow);
                });
            }
        });
        register_type_observer_erased(ty, cb);
        self
    }
}

/// Builder dedicated to a single state.
pub struct StateStageBuilder<TPayload: PanelPayload> {
    definition: PanelStateDefinition<TPayload>,
    rel: RelComposer,
}

impl<TPayload: PanelPayload> StateStageBuilder<TPayload> {
    fn new(state_id: UiState, panel_key: &PanelKey) -> Self {
        let owner = RelViewKey::for_owner::<TPayload>(
            panel_key.panel_uuid.clone(),
            panel_key.scope.clone(),
            state_id.0,
        );
        Self {
            definition: PanelStateDefinition::default(),
            rel: RelComposer::new(owner),
        }
    }

    fn into_definition(mut self) -> PanelStateDefinition<TPayload> {
        self.definition.relations = self.rel.into_definition();
        self.definition
    }

    pub fn rel(&mut self) -> &mut RelComposer {
        &mut self.rel
    }

    pub fn texture(mut self, path: &str) -> Self {
        self.definition.overrides.texture = Some(path.to_owned());
        self
    }

    pub fn size_with_image(mut self) -> Self {
        self.definition.overrides.fit_to_texture = Some(true);
        self
    }

    pub fn size(self, size: Vec2) -> Self {
        self.rel_size(size)
    }

    pub fn rel_size(mut self, size: Vec2) -> Self {
        let value = [size.x, size.y];
        self.definition.overrides.size = Some(value);
        self.rel.size_fixed(value);
        self
    }

    pub fn position(self, pos: Vec2) -> Self {
        self.rel_position_in(RelSpace::Local, pos)
    }

    pub fn rotation(mut self, angles: Vec3) -> Self {
        self.definition.overrides.rotation = Some([angles.x, angles.y, angles.z]);
        self
    }

    pub fn scale(mut self, scale: Vec3) -> Self {
        self.definition.overrides.scale = Some([scale.x, scale.y, scale.z]);
        self
    }

    /// Initialize position from current mouse position once at spawn.
    /// This sets a flag on the overrides; runtime will translate it to a concrete position
    /// using the CPU copy of GlobalUniform.mouse_pos during the first GPU write, then clear the flag.
    pub fn with_trigger_mouse_pos(mut self) -> Self {
        self.definition.overrides.trigger_mouse_pos = true;
        self
    }

    pub fn rel_position_in(mut self, space: RelSpace, pos: Vec2) -> Self {
        let value = [pos.x, pos.y];
        self.definition.overrides.position = Some(value);
        self.rel.position_fixed(space, value);
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

    pub fn visible(mut self, visible: bool) -> Self {
        self.definition.overrides.visible = Some(visible);
        self
    }

    /// Clamp and optional step-quantize a scalar field.
    /// - Absolute mode (most fields): `range = [min, max, step]` with `step <= 0` meaning no snapping.
    ///   Snap anchor is `min`: `min + round((v - min)/step)*step`.
    /// - Relative budget mode (for `Field::OnlyPositionX/OnlyPositionY`): interpret as
    ///   `range = [0, max_offset, step]` measured from the panel's snapshot/origin captured at drag start.
    ///   Axis is locked accordingly (X-only or Y-only).
    pub fn clamp_offset(mut self, field: Field, range: [f32; 3]) -> Self {
        let mut rule = ClampOffset {
            field: field.clone(),
            min: [range[0], 0.0],
            max: [range[1], 0.0],
            step: [range[2], 0.0],
            dims: 1,
            flags: 0,
        };
        // Special-cased semantics: OnlyPositionX/OnlyPositionY -> relative budget around origin.
        match field {
            Field::OnlyPositionX => {
                let max_off = (range[1] - range[0]).max(0.0);
                rule.min = [0.0, 0.0];
                rule.max = [max_off, 0.0];
                rule.flags = 0x1 | 0x2; // relative budget + axis_x_only
            }
            Field::OnlyPositionY => {
                let max_off = (range[1] - range[0]).max(0.0);
                rule.min = [0.0, 0.0];
                rule.max = [max_off, 0.0];
                rule.flags = 0x1 | 0x4; // relative budget + axis_y_only
            }
            _ => {}
        }
        self.definition.overrides.clamp_offsets.push(rule);
        self
    }

    /// Clamp and optional step-quantize a vec2 field (per-axis).
    /// - `step <= 0` per-component means no snapping on that axis.
    /// - Snap anchor is `min` per axis.
    pub fn clamp_offset_v2(
        mut self,
        field: Field,
        min_v: [f32; 2],
        max_v: [f32; 2],
        step_v: [f32; 2],
    ) -> Self {
        let rule = ClampOffset {
            field,
            min: min_v,
            max: max_v,
            step: step_v,
            dims: 2,
            flags: 0,
        };
        self.definition.overrides.clamp_offsets.push(rule);
        self
    }

    /// Budget clamp (relative to origin) for scalar axis. `max_offset = [max_x, step]`
    pub fn clamp_offset_budget(mut self, field: Field, max_offset_and_step: [f32; 2]) -> Self {
        // flags: relative budget; axis locks derive from field variant
        let mut flags = 1u32; // relative budget
        match field {
            Field::OnlyPositionX => {
                flags |= 1 << 1;
            } // axis_x_only
            Field::OnlyPositionY => {
                flags |= 1 << 2;
            } // axis_y_only
            _ => {}
        }
        let rule = ClampOffset {
            field,
            min: [0.0, 0.0],
            max: [max_offset_and_step[0], 0.0],
            step: [max_offset_and_step[1], 0.0],
            dims: 1,
            flags,
        };
        self.definition.overrides.clamp_offsets.push(rule);
        self
    }

    /// Budget clamp (relative to origin) for vec2 axis. `max_offset = [max_x, max_y]`
    pub fn clamp_offset_budget_v2(
        mut self,
        field: Field,
        max_offset: [f32; 2],
        step: [f32; 2],
    ) -> Self {
        let mut flags = 1u32; // relative budget
        match field {
            Field::OnlyPositionX => {
                flags |= 1 << 1;
            }
            Field::OnlyPositionY => {
                flags |= 1 << 2;
            }
            _ => {}
        }
        let rule = ClampOffset {
            field,
            min: [0.0, 0.0],
            max: max_offset,
            step,
            dims: 2,
            flags,
        };
        self.definition.overrides.clamp_offsets.push(rule);
        self
    }

    pub fn container_layout(mut self, layout: RelLayoutKind) -> Self {
        self.rel.container_self(|spec| spec.layout = layout.clone());
        self
    }

    pub fn configure_container_layout<F>(mut self, configure: F) -> Self
    where
        F: FnOnce(&mut RelLayoutKind),
    {
        self.rel.container_self(|spec| configure(&mut spec.layout));
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

    pub fn try_style(mut self, key: &str) -> Result<Self, StyleError> {
        let patch = load_panel_style(key)?;
        apply_style_patch(&mut self.definition.overrides, &patch);
        Ok(self)
    }

    pub fn style(self, key: &str) -> Self {
        self.try_style(key)
            .unwrap_or_else(|err| panic!("failed to load style '{key}': {err}"))
    }

    pub fn container_style(mut self) -> ContainerStyleBuilder<TPayload> {
        let spec = self.rel.container_spec().cloned();
        ContainerStyleBuilder {
            parent: self,
            spec,
            dirty: false,
        }
    }

    pub fn events(self) -> InteractionStageBuilder<TPayload> {
        InteractionStageBuilder {
            parent: self,
            stage: InteractionStage::default(),
            last_event: None,
        }
    }
}

pub struct ContainerStyleBuilder<TPayload: PanelPayload> {
    parent: StateStageBuilder<TPayload>,
    spec: Option<RelContainerSpec>,
    dirty: bool,
}

impl<TPayload: PanelPayload> ContainerStyleBuilder<TPayload> {
    fn spec_mut(&mut self) -> &mut RelContainerSpec {
        self.dirty = true;
        self.spec.get_or_insert_with(RelContainerSpec::default)
    }

    pub fn space(mut self, space: RelSpace) -> Self {
        self.spec_mut().space = space;
        self
    }

    pub fn origin(mut self, origin: Vec2) -> Self {
        self.spec_mut().origin = [origin.x, origin.y];
        self
    }

    pub fn size_container(mut self, size: Vec2) -> Self {
        self.spec_mut().size = Some([size.x, size.y]);
        self
    }

    pub fn clear_size(mut self) -> Self {
        if let Some(spec) = self.spec.as_mut() {
            if spec.size.is_some() {
                spec.size = None;
                self.dirty = true;
            }
        }
        self
    }

    pub fn size_percent_of_parent(mut self, percent: Vec2) -> Self {
        self.spec_mut().size_percent_of_parent = Some([percent.x, percent.y]);
        self
    }

    pub fn clear_size_percent(mut self) -> Self {
        if let Some(spec) = self.spec.as_mut() {
            if spec.size_percent_of_parent.is_some() {
                spec.size_percent_of_parent = None;
                self.dirty = true;
            }
        }
        self
    }

    pub fn padding(mut self, padding: [f32; 4]) -> Self {
        self.spec_mut().padding = padding;
        self
    }

    pub fn clip_content(mut self, clip: bool) -> Self {
        self.spec_mut().clip_content = clip;
        self
    }

    pub fn scroll_axis(mut self, axis: RelScrollAxis) -> Self {
        self.spec_mut().scroll_axis = axis;
        self
    }

    pub fn layout(mut self, layout: RelLayoutKind) -> Self {
        self.spec_mut().layout = layout;
        self
    }

    pub fn configure_layout<F>(mut self, configure: F) -> Self
    where
        F: FnOnce(&mut RelLayoutKind),
    {
        configure(&mut self.spec_mut().layout);
        self
    }

    pub fn slot_size(mut self, slot: Vec2) -> Self {
        self.spec_mut().slot_size = Some([slot.x, slot.y]);
        self
    }

    pub fn clear_slot_size(mut self) -> Self {
        if let Some(spec) = self.spec.as_mut() {
            if spec.slot_size.is_some() {
                spec.slot_size = None;
                self.dirty = true;
            }
        }
        self
    }

    pub fn element_scale(mut self, scale: Vec2) -> Self {
        self.spec_mut().element_scale = [scale.x, scale.y];
        self
    }

    pub fn reset_element_scale(mut self) -> Self {
        self.spec_mut().element_scale = [1.0, 1.0];
        self
    }

    pub fn disable(mut self) -> Self {
        self.spec = None;
        self.dirty = true;
        self
    }

    pub fn finish(mut self) -> StateStageBuilder<TPayload> {
        if self.dirty {
            match self.spec.take() {
                Some(spec) => {
                    self.parent.rel.container_self(move |target| *target = spec);
                }
                None => {
                    self.parent.rel.clear_container_self();
                }
            }
        }
        self.parent
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
    if let Some(rotation) = patch.rotation {
        overrides.rotation = Some(rotation);
    }
    if let Some(scale) = patch.scale {
        overrides.scale = Some(scale);
    }
    if let Some(z) = patch.z_index {
        overrides.z_index = Some(z as i32);
    }
}

#[derive(Default)]
struct InteractionStage<TPayload: PanelPayload> {
    transitions: HashMap<UiEventKind, UiState>,
    callbacks: HashMap<UiEventKind, Arc<EventFn<TPayload>>>,
    callbacks_with: HashMap<UiEventKind, Arc<EventFnWith<TPayload>>>,
    data_callbacks: Vec<DataChangeCbEntry<TPayload>>,
}

pub struct InteractionStageBuilder<TPayload: PanelPayload> {
    parent: StateStageBuilder<TPayload>,
    stage: InteractionStage<TPayload>,
    last_event: Option<UiEventKind>,
}

impl<TPayload: PanelPayload> InteractionStageBuilder<TPayload> {
    /// 注册面板初始化回调（构建完成后立即触发一次）
    pub fn on_init<F>(self, handler: F) -> Self
    where
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>) + Send + Sync + 'static,
    {
        self.on_event(UiEventKind::Init, handler)
    }
    pub fn on_event<F>(mut self, event: UiEventKind, handler: F) -> Self
    where
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>) + Send + Sync + 'static,
    {
        let event_fn: Arc<EventFn<TPayload>> = Arc::new(handler);
        self.stage.callbacks.insert(event, event_fn);
        self.last_event = Some(event);
        self
    }
    /// Variant of `on_event` that also receives an event payload.
    pub fn on_event_with<F>(mut self, event: UiEventKind, handler: F) -> Self
    where
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>, UiEventData) + Send + Sync + 'static,
    {
        let event_fn: Arc<EventFnWith<TPayload>> = Arc::new(handler);
        self.stage.callbacks_with.insert(event, event_fn);
        self.last_event = Some(event);
        self
    }

    /// Drag 专用：仅当当前 drag_payload 能 downcast 为 `T` 时才触发。
    /// 回调收到拖拽 delta 和强类型 payload。
    pub fn on_drag_with_payload<T, F>(self, handler: F) -> Self
    where
        T: Send + Sync + Clone + 'static,
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>, glam::Vec2, T) + Send + Sync + 'static,
    {
        self.register_target_with_payload_delta(UiEventKind::Drag, handler)
    }

    fn register_target_with_payload<T, F>(mut self, event: UiEventKind, handler: F) -> Self
    where
        T: Send + Sync + Clone + 'static,
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>, T) + Send + Sync + 'static,
    {
        let event_fn: Arc<EventFnWith<TPayload>> = Arc::new(move |flow, data| {
            let _ = data;
            let Some(payload) = flow.drag_payload::<T>() else {
                return;
            };
            handler(flow, payload.clone());
        });
        self.stage.callbacks_with.insert(event, event_fn);
        self.last_event = Some(event);
        self
    }

    fn register_target_with_payload_delta<T, F>(mut self, event: UiEventKind, handler: F) -> Self
    where
        T: Send + Sync + Clone + 'static,
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>, glam::Vec2, T) + Send + Sync + 'static,
    {
        let event_fn: Arc<EventFnWith<TPayload>> = Arc::new(move |flow, data| {
            let UiEventData::Vec2(delta) = data else {
                return;
            };
            let Some(payload) = flow.drag_payload::<T>() else {
                return;
            };
            handler(flow, delta, payload.clone());
        });
        self.stage.callbacks_with.insert(event, event_fn);
        self.last_event = Some(event);
        self
    }

    /// 拖拽源：开始阶段。
    pub fn source_drag_start<F>(self, handler: F) -> Self
    where
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>) + Send + Sync + 'static,
    {
        self.on_event(UiEventKind::SourceDragStart, handler)
    }

    /// 拖拽源：开始阶段（并设置 payload）。
    pub fn source_drag_start_with_payload<T, F>(self, builder: F) -> Self
    where
        T: Send + Sync + 'static,
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>) -> T + Send + Sync + 'static,
    {
        self.on_event(UiEventKind::SourceDragStart, move |flow| {
            let payload = builder(flow);
            flow.set_drag_payload(payload);
        })
    }

    /// 拖拽源：开始阶段（指定自定义 DragPayloadId）。
    pub fn source_drag_start_with_id<T, F>(self, id: DragPayloadId, builder: F) -> Self
    where
        T: Send + Sync + 'static,
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>) -> T + Send + Sync + 'static,
    {
        self.on_event(UiEventKind::SourceDragStart, move |flow| {
            let payload = builder(flow);
            flow.drag_payload_with_id(id, payload);
        })
    }

    /// 拖拽源：拖拽过程中持续更新（带 Vec2 delta）。
    pub fn source_drag_over<F>(self, handler: F) -> Self
    where
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>, glam::Vec2) + Send + Sync + 'static,
    {
        self.on_event_with(UiEventKind::SourceDragOver, move |flow, data| {
            let UiEventData::Vec2(delta) = data else {
                return;
            };
            handler(flow, delta);
        })
    }

    /// 拖拽源：drag 离开。
    pub fn source_drag_leave<F>(self, handler: F) -> Self
    where
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>) + Send + Sync + 'static,
    {
        self.on_event(UiEventKind::SourceDragLeave, handler)
    }

    /// 拖拽源：drag drop/结束。
    pub fn source_drag_drop<F>(self, handler: F) -> Self
    where
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>) + Send + Sync + 'static,
    {
        self.on_event(UiEventKind::SourceDragDrop, handler)
    }

    /// 拖拽目标：drag enter。
    pub fn target_drag_enter<T, F>(self, handler: F) -> Self
    where
        T: Send + Sync + Clone + 'static,
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>, T) + Send + Sync + 'static,
    {
        self.register_target_with_payload(UiEventKind::TargetDragEnter, handler)
    }

    /// 拖拽目标：drag over（带 Vec2 delta）。
    pub fn target_drag_over<T, F>(self, handler: F) -> Self
    where
        T: Send + Sync + Clone + 'static,
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>, glam::Vec2, T) + Send + Sync + 'static,
    {
        self.register_target_with_payload_delta(UiEventKind::TargetDragOver, handler)
    }

    /// 拖拽目标：drag leave。
    pub fn target_drag_leave<T, F>(self, handler: F) -> Self
    where
        T: Send + Sync + Clone + 'static,
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>, T) + Send + Sync + 'static,
    {
        self.register_target_with_payload(UiEventKind::TargetDragLeave, handler)
    }

    /// 拖拽目标：drag drop。
    pub fn target_drag_drop<T, F>(self, handler: F) -> Self
    where
        T: Send + Sync + Clone + 'static,
        F: for<'a> Fn(&mut EventFlow<'a, TPayload>, T) + Send + Sync + 'static,
    {
        self.register_target_with_payload(UiEventKind::TargetDragDrop, handler)
    }
    /// Register a data-change listener for any panels with payload type `TObserved`.
    /// If `source_uuid` is Some(uuid), only changes from that panel trigger the callback.
    pub fn on_data_change<TObserved, F>(mut self, source_uuid: Option<&str>, handler: F) -> Self
    where
        TObserved: PanelPayload,
        F: for<'a> Fn(&TObserved, &mut EventFlow<'a, TPayload>) + Send + Sync + 'static,
    {
        let ty = TypeId::of::<TObserved>();
        let src = source_uuid.map(|s| s.to_string());
        let wrapped: Arc<DataChangeErasedFn<TPayload>> = Arc::new(move |any, flow| {
            if let Some(payload) = any.downcast_ref::<TObserved>() {
                handler(payload, flow);
            }
        });
        self.stage.data_callbacks.push(DataChangeCbEntry {
            ty,
            source_uuid: src,
            handler: wrapped,
        });
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
            .chain(self.stage.callbacks_with.keys())
            .chain(self.stage.transitions.keys())
        {
            interaction_mask |= match kind {
                UiEventKind::Init => 0,
                UiEventKind::Click => PanelInteraction::CLICKABLE.bits(),
                UiEventKind::Drag
                | UiEventKind::SourceDragStart
                | UiEventKind::SourceDragOver
                | UiEventKind::SourceDragLeave
                | UiEventKind::SourceDragDrop
                | UiEventKind::TargetDragEnter
                | UiEventKind::TargetDragOver
                | UiEventKind::TargetDragLeave
                | UiEventKind::TargetDragDrop => PanelInteraction::DRAGGABLE.bits(),
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
            .chain(self.stage.callbacks_with.keys())
            .chain(self.stage.transitions.keys())
        {
            interaction_mask |= match kind {
                UiEventKind::Init => 0,
                UiEventKind::Click => PanelInteraction::CLICKABLE.bits(),
                UiEventKind::Drag
                | UiEventKind::SourceDragStart
                | UiEventKind::SourceDragOver
                | UiEventKind::SourceDragLeave
                | UiEventKind::SourceDragDrop
                | UiEventKind::TargetDragEnter
                | UiEventKind::TargetDragOver
                | UiEventKind::TargetDragLeave
                | UiEventKind::TargetDragDrop => PanelInteraction::DRAGGABLE.bits(),
                UiEventKind::Hover => PanelInteraction::HOVER.bits(),
                UiEventKind::Out => PanelInteraction::Out.bits(),
            };
        }

        if interaction_mask != PanelInteraction::DEFUALT.bits() {
            overrides.interaction = Some(interaction_mask);
        }

        let transitions = std::mem::take(&mut self.stage.transitions);
        let callbacks = std::mem::take(&mut self.stage.callbacks);
        let callbacks_with = std::mem::take(&mut self.stage.callbacks_with);
        let data_callbacks = std::mem::take(&mut self.stage.data_callbacks);
        overrides.transitions.extend(transitions);
        self.parent.definition.callbacks.extend(callbacks);
        self.parent.definition.callbacks_with.extend(callbacks_with);
        self.parent.definition.data_callbacks.extend(data_callbacks);
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
        // 初始化 snapshot，使之与默认状态的 overrides 对齐，方便与 GPU snapshot 对接
        let ov = record.states.get(&default).cloned().unwrap();
        apply_initial_snapshot_from_overrides(record, &ov);
    })?;

    let mut transitions = HashMap::new();
    let mut callbacks = HashMap::new();
    let mut callbacks_with = HashMap::new();

    let mut data_map: HashMap<UiState, Vec<DataChangeCbEntry<TPayload>>> = HashMap::new();
    for (state_id, definition) in states {
        let PanelStateDefinition {
            overrides,
            frag_shader,
            vertex_shader,
            callbacks: state_callbacks,
            callbacks_with: state_callbacks_with,
            data_callbacks,
            animations: _animations,
            relations,
        } = definition;

        if let Some(shader) = frag_shader.as_ref() {
            submit_shader_request(&panel_key, state_id, shader, ShaderStage::Fragment);
        }
        if let Some(shader) = vertex_shader.as_ref() {
            submit_shader_request(&panel_key, state_id, shader, ShaderStage::Vertex);
        }

        transitions.insert(state_id, overrides.transitions.clone());
        callbacks.insert(state_id, state_callbacks);
        callbacks_with.insert(state_id, state_callbacks_with);
        // Store data change callbacks into runtime map after runtime is created
        data_map.insert(state_id, data_callbacks);

        if let Some(rel_def) = relations {
            register_panel_relations(panel_key.panel_id, state_id, rel_def);
        }
    }

    install_runtime_callbacks::<TPayload>(
        &panel_key,
        &callbacks,
        &transitions,
        &data_map,
        &callbacks_with,
    );
    let mut runtime = PanelRuntime {
        handle,
        current_state: default,
        transitions,
        callbacks,
        callbacks_with,
        observer_guards: Vec::new(),
        data_callbacks: data_map,
        active_drag: None,
    };
    runtime.observer_guards = attach_observers(&runtime.handle, &observers);
    register_runtime(panel_key.clone(), runtime);

    // 初始化阶段触发一次 Init 事件（用于面板创建时的初始化逻辑，例如字体注册）
    trigger_event_internal::<TPayload>(&panel_key, Some(default), UiEventKind::Init);

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
        let ov = record.states.get(&state_id).cloned().unwrap();
        apply_initial_snapshot_from_overrides(record, &ov);
    })?;

    let PanelStateDefinition {
        overrides,
        vertex_shader,
        frag_shader,
        callbacks,
        callbacks_with,
        data_callbacks,
        animations: _animations,
        relations,
    } = definition;

    if let Some(shader) = frag_shader.as_ref() {
        submit_shader_request(&panel_key, state_id, shader, ShaderStage::Fragment);
    }
    if let Some(shader) = vertex_shader.as_ref() {
        submit_shader_request(&panel_key, state_id, shader, ShaderStage::Vertex);
    }

    let transitions = HashMap::from([(state_id, overrides.transitions.clone())]);
    let callbacks = HashMap::from([(state_id, callbacks)]);
    let callbacks_with = HashMap::from([(state_id, callbacks_with)]);
    let data_map: HashMap<UiState, Vec<DataChangeCbEntry<TPayload>>> =
        HashMap::from([(state_id, data_callbacks)]);

    install_runtime_callbacks::<TPayload>(
        &panel_key,
        &callbacks,
        &transitions,
        &data_map,
        &callbacks_with,
    );
    // Note: for stateless we also pass callbacks_with
    let mut runtime = PanelRuntime {
        handle,
        current_state: state_id,
        transitions,
        callbacks,
        callbacks_with,
        observer_guards: Vec::new(),
        data_callbacks: data_map,
        active_drag: None,
    };

    runtime.observer_guards = attach_observers(&runtime.handle, &observers);
    register_runtime(panel_key.clone(), runtime);

    if let Some(rel_def) = relations {
        register_panel_relations(panel_key.panel_id, state_id, rel_def);
    }

    // 初始化阶段触发一次 Init 事件
    trigger_event_internal::<TPayload>(&panel_key, Some(state_id), UiEventKind::Init);

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
    /// Optional scalar used by demos to represent brightness/amount in [0,1].
    pub brightness: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct TestCustomData {
    pub count: u32,
}

//这是一个在rust 写好的shader函数dsl 最终会编译成gpu ast分解计算
fn frag_template(intensity: f32) -> Expr {
    let uv = rv("uv");
    let t = cv("time");
    let wave = (sin(uv.x() * 8.0 + t.clone() * 1.2) + sin(uv.y() * 9.0 - t * 1.1)) * 0.5;
    let crest = smoothstep(0.55, 0.9, (wave + 1.0) * 0.5) * intensity;
    wvec4(
        0.02 + 0.8 * crest.clone(),
        0.05 + 0.3 * crest.clone(),
        0.12 + crest.clone(),
        1.0,
    )
}

//新增了全局DB数据中心  这个数据中心
//UI可以在数据流里操作  比如事件响应 点击 拖拽等
//也可以在别的系统里面 同步获取 &mut
//让我们来创造这个数据

//    Mui::<DataTest>::new("test_panel")?
//         .default_state(UiState(0))
//         .state(UiState(0), |state| {
//             let state = state
//                 .position(vec2(200.0, 200.0))
//                 .size(vec2(500.0, 500.0))
//                 .z_index(5)
//                 .color(vec4(0.3, 0.3, 0.5, 0.55))
//                 .container_style()
//                 .layout(RelLayoutKind::grid([0.0, 0.0]))
//                 .size_container(vec2(500.0, 500.0))
//                 .slot_size(vec2(80.0, 50.0)) //网格的每个大小
//                 .finish() //这里退出容器设置的上下文
//                 .border(BorderStyle {
//                     color: [1.0, 1.0, 1.0, 0.66],
//                     width: 3.0,
//                     radius: 0.0,
//                 })
//                 .events()
//                 .on_init(|flow|{
//                     flow.text(
//                         "枯枝探新芽，
//                               细雨吻旧窗。
//                               时光轻驻足，
//                               春意悄然藏。",
//                         Arc::from("tf/STXIHEI.ttf"),
//                         60,
//                         [1.0,1.0,1.0,1.0],
//                         1,
//                         1
//                     );
//                 })
//                 .on_event(UiEventKind::Click, |flow| {
//                     flow.text(
//                         "枯枝悄悄抽出新芽，细雨温柔地敲打着旧窗。时光仿佛在这一刻驻足，冬日的萧瑟悄然褪去，泥土的芬芳在空气中弥漫。我听见冰凌融化的轻响，看见屋檐下蜘蛛编织新的网。春意就这样无声无息地，在每道裂缝里生根发芽，把积蓄一季的力量，化作枝头第一抹鹅黄。",
//                         Arc::from("tf/STXIHEI.ttf"),
//                         60,
//                         [1.0,1.0,1.0,1.0],
//                         1,
//                         1
//                     );
//                 })
//                 .finish();
//             state
//         })
//         .build()?;

//PanelPayload  是他绑定的全局数据流
//使用DB 可以让程序从同步中的任何hook阶段 让UI和实际执行操作一份数据

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct DataTest {
    count: u32,
}
//这个数据是一个绑定数据  我们可以给ui当作flow 比如在点击的时候改变count

//这里创造了一个绑定DataTest数据流的UI  名字叫test_container
//并且把他的样式写清楚了
//支持透明度
//我们来给面板渲染一点字体
fn demo_panel() -> Result<(), DbError> {
    let _ = Mui::<DataTest>::new("test_container")?
        //我们的UI设计 遵循状态设计  他可以在交互事件中自由的切换状态  比如
        //下次会修复这个bug
        //让我们给他附加 超多面板吧
        //接着上回  我们的面板可以写入实时frag
        //让我们给他加入小面板
        //按grid排列
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            //我们要把这个panel当作一个容器 然后我们去写他的容器配置
            //好吧  因为新的更改 排列似乎出问题了  我需要回头检查一下
            let state = state
                .z_index(1)
                .container_style()
                //小问题 没有设置容器大小 再来看看
                .slot_size(vec2(108.0, 52.0))
                .size_container(vec2(500.0, 500.0))
                .layout(RelLayoutKind::grid([0.0, 0.0]))
                .finish() //这里要退出容器设置的上下文;
                .position(vec2(300.0, 200.0))
                .color(vec4(0.5, 0.7, 0.6, 0.3))
                .border(BorderStyle {
                    color: [0.1, 0.3, 0.2, 1.0],
                    width: 8.0,
                    radius: 0.0,
                })
                //我们在状态里面加入这个接口 加入frag实时计算buffer
                .fragment_shader(|e| frag_template(1.0))
                .size(vec2(500.0, 500.0))
                .events() //进入event上下文构建
                .on_event(UiEventKind::Click, |flow| {
                    let mut data_test = flow.payload(); //这里是取出DataTest的可变引用
                    data_test.count += 1; //给他增加值;
                    flow.set_state(UiState(1)); //如果点击 我们切换到状态1
                })
                .finish();
            state
        })
        .state(UiState(1), |state| {
            //在点击后  就会转换到状态1 这里我们再加一个状态轮换
            let state = state
                .size(vec2(400.0, 400.0))
                .events()
                .on_event(UiEventKind::Click, |flow| {
                    flow.set_state(UiState(0)); //我们来看看效果
                })
                .finish();
            state
        })
        .build();

    //  让我们创造子面板

    //加24个小面板
    for idx in 0..24 {
        let uuid = format!("demo_entry_{idx}");
        let panel = Mui::<TestCustomData>::new(Box::leak(uuid.into_boxed_str()))?
            .default_state(UiState(0))
            .state(
                UiState(0),
                move |mut state: StateStageBuilder<TestCustomData>| {
                    //访问关系组件 rel  让他依附 一个panel 当作容器
                    state.rel().container_with::<DataTest>("test_container");

                    //通过子类 确定自己要进入的容器  这里指定DataTest 和test_container 就可以绑定了

                    state
                        .z_index(4 + idx)
                        .position(vec2(0.0, 0.0))
                        .color(vec4(0.1, 0.1, 0.1, 0.5))
                        .size(vec2(108.0, 52.0))
                        .border(BorderStyle {
                            color: [0.15, 0.8, 0.45, 1.0],
                            width: 1.0,
                            radius: 9.0,
                        })
                        .events()
                        .on_event(UiEventKind::Hover, |flow| {
                            flow.position_anim()
                                .from_current()
                                .offset(vec2(0.0, -14.0))
                                .duration(0.18)
                                .easing(Easing::BackOut)
                                .push(flow);
                        })
                        .on_event(UiEventKind::Out, |flow| {
                            //所有的交互操作 全部在flow里面进行
                            flow.position_anim()
                                .from_current()
                                .to_snapshot()
                                .duration(0.22)
                                .easing(Easing::BackIn)
                                .push(flow);
                        })
                        .finish()
                },
            )
            .build()?;
    }
    Ok(())
}

fn build_demo_panel_with_uuid(
    panel_uuid: &'static str,
) -> Result<Vec<PanelRuntimeHandle>, DbError> {
    const ITEM_COUNT: usize = 1;
    let mut handles = Vec::with_capacity(ITEM_COUNT + 1);

    for idx in 0..ITEM_COUNT {
        let slot_position = vec2(
            48.0 + (idx as f32 % 6.0) * 118.0,
            72.0 + (idx as f32 / 6.0).floor() * 70.0,
        );
        let uuid = format!("demo_entry_{idx}");
        let panel = Mui::<TestCustomData>::new(Box::leak(uuid.into_boxed_str()))?
            .default_state(UiState(0))
            .state(
                UiState(0),
                move |mut state: StateStageBuilder<TestCustomData>| {
                    state.rel().container_with::<TestCustomData>("test_back");

                    state
                        .z_index(4 + (idx as i32 % 3))
                        .position(vec2(0.0, 0.0))
                        .size(vec2(108.0, 52.0))
                        .border(BorderStyle {
                            color: [0.15, 0.8, 0.45, 1.0],
                            width: 1.0,
                            radius: 9.0,
                        })
                        .fragment_shader(|_| frag_template(1.0))
                        .events()
                        .on_event(UiEventKind::Hover, |flow| {
                            flow.position_anim()
                                .from_current()
                                .offset(vec2(0.0, -14.0))
                                .duration(0.18)
                                .easing(Easing::BackOut)
                                .push(flow);
                        })
                        .on_event(UiEventKind::Out, |flow| {
                            flow.position_anim()
                                .from_current()
                                .to_snapshot()
                                .duration(0.22)
                                .easing(Easing::BackIn)
                                .push(flow);
                        })
                        .finish()
                },
            )
            .build()?;

        handles.push(panel);
    }

    let test_container = Mui::<TestCustomData>::new("test_back")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            state
                .position(vec2(300.0, 300.0))
                .container_style()
                .origin(vec2(32.0, 32.0))
                .size_container(vec2(560.0, 360.0))
                .slot_size(vec2(120.0, 60.0))
                .layout(RelLayoutKind::grid([0.0, 0.0]))
                .finish()
                .z_index(1)
                .position(vec2(0.0, 0.0))
                .texture("backgound.png")
                .size(vec2(640.0, 420.0))
                .events()
                .on_event(UiEventKind::Drag, |_| {})
                .finish()
        })
        .build()?;
    handles.push(test_container);

    Ok(handles)
}
/// Demonstration that mirrors the existing builder usage.
pub fn build_demo_panel() -> Result<(), DbError> {
    // build_demo_panel_with_uuid("inventory_panel")
    build_slider_color_demo()?;
    Ok(())
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
struct TestUi {
    pub test_count: f32,
    pub lock_state: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
struct SliderLock {
    pub lock: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
struct UIDefault;

/// Build a small demo: one draggable slider (with X-axis clamp) and a color panel
/// whose color responds to events (hover: lighten; out: restore). The slider's drag
/// also nudges the color panel using an offset color animation to show cross-panel
/// event response using existing data flow.
pub fn build_slider_color_demo() -> Result<Vec<PanelRuntimeHandle>, DbError> {
    let mut handles: Vec<PanelRuntimeHandle> = Vec::new();

    // Target color panel
    let color_panel = Mui::<TestUi>::new("demo_color_target")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            let state = state
                .z_index(2)
                .position(vec2(320.0, 180.0))
                .size(vec2(240.0, 140.0))
                .color(vec4(0.12, 0.28, 0.60, 1.0))
                // Respond to brightness changes via observers (data-driven)
                .events()
                .on_data_change::<UiPanelData, _>(None, |target, flow| {
                    println!("当前color面版监听的事件 {:?}", target);
                    if (flow.payload_ref().lock_state) {
                        return;
                    }
                    flow.style_set(
                        PanelField::TRANSPARENT.bits(),
                        [target.brightness, 0.0, 0.0, 0.0],
                    );
                })
                .on_event(UiEventKind::Click, |flow| {
                    let position = Vec2::from_array(flow.record.snapshot.position);
                    println!("当前position {:?}", position);
                    let _ = Mui::<UIDefault>::new("nb")
                        .unwrap()
                        .default_state(UiState(0))
                        .state(UiState(0), |state| {
                            let state = state
                                .color(vec4(0.0, 1.0, 1.0, 1.0))
                                .size(vec2(300.0, 300.0))
                                .with_trigger_mouse_pos();
                            state
                        })
                        .build();
                })
                .finish();
            state
        })
        .build()?;
    handles.push(color_panel.clone());

    // Slider "thumb" – draggable only on X, clamped into [120, 580] with 5px steps
    let x_min = 120.0_f32;
    let x_max = 580.0_f32;
    let slider_thumb = Mui::<UiPanelData>::new("demo_slider_thumb")?
        .default_state(UiState(0))
        .state(UiState(0), move |state: StateStageBuilder<UiPanelData>| {
            let state = state
                .z_index(3)
                .position(vec2(x_min, 420.0))
                .size(vec2(36.0, 36.0))
                .color(vec4(0.85, 0.55, 0.22, 1.0))
                // X-only with relative budget [0..(x_max-x_min)], step 5px; origin from snapshot
                .clamp_offset(Field::OnlyPositionX, [0.0, (x_max - x_min - 36.0), 5.0])
                .events()
                .on_data_change::<SliderLock, _>(None, |target, flow| {
                    if (target.lock) {
                        flow.set_state(UiState(1));
                    }
                })
                .on_event_with(UiEventKind::Drag, move |flow, drag_detla| {
                    // 将滑块位置映射为 [0,1] 的进度，并写入“颜色面板”的 payload.brightness，
                    // 触发本面板的 DB 提交（只允许改自己）。
                    let pos = Vec2::from_array(flow.args().record_snapshot.snapshot.position);
                    let len = (x_max - x_min - 36.0).max(1.0);
                    let t = ((pos.x - x_min) / len).clamp(0.0, 1.0);

                    match drag_detla {
                        UiEventData::Vec2(vec2) => {
                            flow.update_self_payload(|data| {
                                data.brightness += vec2.x / x_max;
                            });
                        }
                        _ => {}
                    }
                })
                .finish();
            state
        })
        .state(UiState(1), move |state| {
            let state = state
                .color(vec4(1.0, 0.0, 0.22, 1.0))
                .clamp_offset(Field::OnlyPositionX, [0.0, (x_max - x_min - 36.0), 5.0])
                .events()
                .on_data_change::<SliderLock, _>(None, |target, flow| {
                    if (target.lock == false) {
                        println!("回到state 0");
                        flow.set_state(UiState(0));
                    }
                    flow.update_self_payload(|payload| {
                        payload.brightness = 0.0;
                    });
                })
                .finish();
            state
        })
        .build()?;
    handles.push(slider_thumb);

    // Slider track: a background bar aligned with the slider thumb's origin and height,
    // width equals the clamp budget (x_max - x_min).
    let slider_track = Mui::<UiPanelData>::new("demo_slider_track")?
        .default_state(UiState(0))
        .state(UiState(0), move |state| {
            let state = state
                .z_index(2)
                .position(vec2(x_min, 420.0))
                .size(vec2((x_max - x_min), 36.0))
                .color(vec4(0.2, 0.2, 0.2, 0.65))
                .events()
                .finish();
            state
        })
        .build()?;
    handles.push(slider_track);

    // Update button: clicking it will bump the color target's brightness and trigger its observers.
    let update_button = Mui::<SliderLock>::new("demo_update_button")?
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            let state = state
                .z_index(4)
                .position(vec2(x_min + (x_max - x_min) + 20.0, 420.0))
                .size(vec2(80.0, 36.0))
                .color(vec4(0.25, 0.6, 0.3, 0.9))
                .events()
                .on_event(UiEventKind::Click, |flow| {
                    // 只允许改自己的 payload，提交 DB 触发观察者
                    flow.update_self_payload(|data| {
                        data.lock = !data.lock;
                    });
                })
                .finish();
            state
        })
        .build()?;
    handles.push(update_button);

    Ok(handles)
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
        next_id: AtomicU32::new(1),
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
        let handles = build_demo_panel_with_uuid(TEST_PANEL_UUID).expect("demo panel builds");
        let runtime_handle = handles.first().expect("runtime handle present");

        let mut record = read_counter_record(runtime_handle.key());
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

        runtime_handle.trigger::<UiPanelData>(UiEventKind::Click);
        record = read_counter_record(runtime_handle.key());
        assert_eq!(record.current_state, UiState(1));
        assert_eq!(record.data.count, 1);
        assert_eq!(record.pending_animations.len(), 1);

        runtime_handle.trigger::<UiPanelData>(UiEventKind::Click);
        record = read_counter_record(runtime_handle.key());
        assert_eq!(record.current_state, UiState(0));
        assert_eq!(record.data.count, 2);
    }
}
