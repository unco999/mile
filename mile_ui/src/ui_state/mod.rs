use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
    sync::Arc,
    time::Duration,
};

use glam::{Vec2, Vec3, Vec4};

use mile_api::prelude::Event;

/// Unique identifier for a UI state. Accepts any hashable + cloneable key type.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct StateId<S>(pub S);

impl<S> StateId<S> {
    pub fn into_inner(self) -> S {
        self.0
    }
}

impl<S> From<S> for StateId<S> {
    fn from(value: S) -> Self {
        StateId(value)
    }
}

/// Marker trait for events that a state or transition wants to observe.
pub trait UiStateEvent: Event {}

/// Describes how a UI state is laid out in 2D.
#[derive(Clone, Debug, Default)]
pub struct LayoutSpec {
    pub position: Vec2,
    pub size: Vec2,
    pub anchor: Vec2,
    pub pivot: Vec2,
    pub rotation_deg: f32,
    pub scale: Vec2,
    pub z_index: i32,
    pub padding: Vec4,
    pub margin: Vec4,
    pub clip_rect: Option<Vec4>,
    pub visible: bool,
}

/// Visual resources bound to a state.
#[derive(Clone, Debug, Default)]
pub struct VisualSpec {
    pub texture: Option<String>,
    pub nine_slice: Option<[f32; 4]>,
    pub tint: Vec4,
    pub opacity: f32,
    pub shader: Option<String>,
    pub shader_params: HashMap<String, Vec4>,
}

/// Arbitrary payload stored alongside a state. This allows the caller to attach
/// business-specific configuration without polluting the core spec.
#[derive(Clone, Debug)]
pub struct Content<T> {
    pub data: T,
}

impl<T> Content<T> {
    pub fn new(data: T) -> Self {
        Self { data }
    }
}

/// Interaction behaviour for a state.
#[derive(Clone)]
pub struct InteractionSpec<TCtx> {
    pub capture_mouse: bool,
    pub capture_keyboard: bool,
    pub hoverable: bool,
    pub focusable: bool,
    pub draggable: bool,
    pub priority: i32,
    pub on_click: Vec<StateCallback<TCtx>>,
    pub on_hover: Vec<StateCallback<TCtx>>,
    pub on_drag: Vec<StateCallback<TCtx>>,
    pub on_focus: Vec<StateCallback<TCtx>>,
    pub on_blur: Vec<StateCallback<TCtx>>,
}

impl<TCtx> std::fmt::Debug for InteractionSpec<TCtx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let on_click = self.on_click.len();
        let on_hover = self.on_hover.len();
        let on_drag = self.on_drag.len();
        let on_focus = self.on_focus.len();
        let on_blur = self.on_blur.len();

        f.debug_struct("InteractionSpec")
            .field("capture_mouse", &self.capture_mouse)
            .field("capture_keyboard", &self.capture_keyboard)
            .field("hoverable", &self.hoverable)
            .field("focusable", &self.focusable)
            .field("draggable", &self.draggable)
            .field("priority", &self.priority)
            .field("on_click_handlers", &on_click)
            .field("on_hover_handlers", &on_hover)
            .field("on_drag_handlers", &on_drag)
            .field("on_focus_handlers", &on_focus)
            .field("on_blur_handlers", &on_blur)
            .finish()
    }
}

impl<TCtx> Default for InteractionSpec<TCtx> {
    fn default() -> Self {
        Self {
            capture_mouse: false,
            capture_keyboard: false,
            hoverable: false,
            focusable: false,
            draggable: false,
            priority: 0,
            on_click: Vec::new(),
            on_hover: Vec::new(),
            on_drag: Vec::new(),
            on_focus: Vec::new(),
            on_blur: Vec::new(),
        }
    }
}

pub type StateCallback<TCtx> = Arc<dyn Fn(&mut TCtx) + Send + Sync>;

/// Named animation tracks for a state, grouped by a semantic.
#[derive(Clone, Debug)]
pub struct AnimationSpec<S> {
    pub enter: Vec<AnimationClip>,
    pub exit: Vec<AnimationClip>,
    pub idle: Vec<AnimationClip>,
    pub conditional: HashMap<StateId<S>, Vec<AnimationClip>>,
}

impl<S> Default for AnimationSpec<S> {
    fn default() -> Self {
        Self {
            enter: Vec::new(),
            exit: Vec::new(),
            idle: Vec::new(),
            conditional: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: Option<String>,
    pub duration: Duration,
    pub easing: Easing,
    pub bindings: Vec<AnimationBinding>,
}

#[derive(Clone, Debug)]
pub struct AnimationBinding {
    pub target: AnimationTarget,
    pub keyframes: Vec<Keyframe>,
}

#[derive(Clone, Debug)]
pub enum AnimationTarget {
    Position,
    Size,
    Scale,
    Rotation,
    Opacity,
    Custom(String),
}

#[derive(Clone, Debug)]
pub struct Keyframe {
    pub time: f32,
    pub values: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub enum Easing {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Custom(fn(f32) -> f32),
}

impl Default for Easing {
    fn default() -> Self {
        Easing::Linear
    }
}

/// Describes how a state participates in networked panel collections.
#[derive(Clone, Debug)]
pub struct NetworkSpec<S> {
    pub attach_collection: Option<u32>,
    pub exit_collections: HashSet<u32>,
    pub immediate_transition: bool,
    pub transition_to: HashMap<StateId<S>, NetworkTransitionRule>,
}


impl<S> Default for NetworkSpec<S> {
    fn default() -> Self {
        Self {
            attach_collection: None,
            exit_collections: HashSet::new(),
            immediate_transition:false,
            transition_to: HashMap::new()
        }
    }
}

#[derive(Clone, Debug)]
pub struct NetworkTransitionRule {
    pub delay: Duration,
    pub easing: Easing,
    pub collection: Option<u32>,
}

/// GPU bindings that should be resolved when this state is active.
#[derive(Clone, Debug, Default)]
pub struct GpuBindingSpec {
    pub storage_buffers: HashMap<String, u32>,
    pub textures: HashMap<String, u32>,
    pub uniform_blocks: HashMap<String, u32>,
}

/// Audio cues tied to this state.
#[derive(Clone, Debug)]
pub struct AudioCue {
    pub event: AudioEvent,
    pub volume: f32,
    pub delay: Duration,
}

#[derive(Clone, Debug)]
pub enum AudioEvent {
    Play(String),
    Stop(String),
    StopAll,
}

bitflags::bitflags! {
    #[derive(Default, Debug, Clone, Copy)]
    pub struct LogicFlags: u32 {
        const BLOCK_INPUT = 1 << 0;
        const SKIP_LAYOUT = 1 << 1;
        const SKIP_RENDER = 1 << 2;
        const CACHE_GEOMETRY = 1 << 3;
        const CACHE_TEXTURE = 1 << 4;
    }
}

#[derive(Clone, Debug, Default)]
pub struct MetaInfo {
    pub name: Option<String>,
    pub tags: Vec<String>,
    pub debug_notes: Option<String>,
}

/// Comprehensive state configuration.
#[derive(Clone, Debug)]
pub struct StateConfig<S, TCtx, TPayload = ()> {
    pub id: StateId<S>,
    pub layout: LayoutSpec,
    pub visual: VisualSpec,
    pub content: Content<TPayload>,
    pub interaction: InteractionSpec<TCtx>,
    pub animation: AnimationSpec<S>,
    pub network: NetworkSpec<S>,
    pub gpu_bindings: GpuBindingSpec,
    pub audio: Option<AudioCue>,
    pub flags: LogicFlags,
    pub meta: MetaInfo,
}

impl<S, TCtx, TPayload> StateConfig<S, TCtx, TPayload>
where
    S: Clone + Eq + Hash + Ord,
{
    pub fn builder(id: impl Into<StateId<S>>, payload: TPayload) -> StateConfigBuilder<S, TCtx, TPayload> {
        StateConfigBuilder {
            id: id.into(),
            layout: LayoutSpec::default(),
            visual: VisualSpec::default(),
            content: Content::new(payload),
            interaction: InteractionSpec::default(),
            animation: AnimationSpec::default(),
            network: NetworkSpec::default(),
            gpu_bindings: GpuBindingSpec::default(),
            audio: None,
            flags: LogicFlags::default(),
            meta: MetaInfo::default(),
        }
    }
}

pub struct StateConfigBuilder<S, TCtx, TPayload> {
    id: StateId<S>,
    layout: LayoutSpec,
    visual: VisualSpec,
    content: Content<TPayload>,
    interaction: InteractionSpec<TCtx>,
    animation: AnimationSpec<S>,
    network: NetworkSpec<S>,
    gpu_bindings: GpuBindingSpec,
    audio: Option<AudioCue>,
    flags: LogicFlags,
    meta: MetaInfo,
}

impl<S, TCtx, TPayload> StateConfigBuilder<S, TCtx, TPayload>
where
    S: Clone + Eq + Hash + Ord,
{
    pub fn layout(mut self, layout: LayoutSpec) -> Self {
        self.layout = layout;
        self
    }

    pub fn visual(mut self, visual: VisualSpec) -> Self {
        self.visual = visual;
        self
    }

    pub fn interaction(mut self, interaction: InteractionSpec<TCtx>) -> Self {
        self.interaction = interaction;
        self
    }

    pub fn animation(mut self, animation: AnimationSpec<S>) -> Self {
        self.animation = animation;
        self
    }

    pub fn network(mut self, network: NetworkSpec<S>) -> Self {
        self.network = network;
        self
    }

    pub fn gpu_bindings(mut self, gpu_bindings: GpuBindingSpec) -> Self {
        self.gpu_bindings = gpu_bindings;
        self
    }

    pub fn audio(mut self, audio: AudioCue) -> Self {
        self.audio = Some(audio);
        self
    }

    pub fn flags(mut self, flags: LogicFlags) -> Self {
        self.flags = flags;
        self
    }

    pub fn meta(mut self, meta: MetaInfo) -> Self {
        self.meta = meta;
        self
    }

    pub fn finalize(self) -> StateConfig<S, TCtx, TPayload> {
        StateConfig {
            id: self.id,
            layout: self.layout,
            visual: self.visual,
            content: self.content,
            interaction: self.interaction,
            animation: self.animation,
            network: self.network,
            gpu_bindings: self.gpu_bindings,
            audio: self.audio,
            flags: self.flags,
            meta: self.meta,
        }
    }
}

/// Trait implemented by anything that can be applied to a UI context when a state becomes active.
pub trait UiStateBehaviour<TCtx> {
    fn on_enter(&self, ctx: &mut TCtx);
    fn on_exit(&self, ctx: &mut TCtx);
    fn on_update(&self, ctx: &mut TCtx, dt: Duration);
}

/// Graph of states owned by a component or panel.
pub struct StateGraph<S, TCtx, TPayload> {
    configs: BTreeMap<StateId<S>, StateConfig<S, TCtx, TPayload>>,
    transitions: HashMap<StateId<S>, Vec<StateTransition<S>>>,
}

impl<S, TCtx, TPayload> Default for StateGraph<S, TCtx, TPayload> {
    fn default() -> Self {
        Self {
            configs: BTreeMap::new(),
            transitions: HashMap::new(),
        }
    }
}

impl<S, TCtx, TPayload> StateGraph<S, TCtx, TPayload>
where
    S: Clone + Eq + Hash + Ord,
{
    pub fn register_state(&mut self, config: StateConfig<S, TCtx, TPayload>) {
        self.configs.insert(config.id.clone(), config);
    }

    pub fn add_transition(&mut self, transition: StateTransition<S>) {
        self.transitions
            .entry(transition.from.clone())
            .or_default()
            .push(transition);
    }

    pub fn state(&self, id: &StateId<S>) -> Option<&StateConfig<S, TCtx, TPayload>> {
        self.configs.get(id)
    }

    pub fn transitions_from(&self, id: &StateId<S>) -> impl Iterator<Item = &StateTransition<S>> {
        self.transitions.get(id).into_iter().flatten()
    }
}

/// A transition between two states with optional guard/effect hooks.
pub struct StateTransition<S> {
    pub from: StateId<S>,
    pub to: StateId<S>,
    pub guard: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    pub effect: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl<S> StateTransition<S> {
    pub fn new(from: impl Into<StateId<S>>, to: impl Into<StateId<S>>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            guard: None,
            effect: None,
        }
    }

    pub fn with_guard(mut self, guard: Arc<dyn Fn() -> bool + Send + Sync>) -> Self {
        self.guard = Some(guard);
        self
    }

    pub fn with_effect(mut self, effect: Arc<dyn Fn() + Send + Sync>) -> Self {
        self.effect = Some(effect);
        self
    }
}

