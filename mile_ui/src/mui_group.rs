use crate::mui_anim::{AnimationSpec, Easing};
use glam::Vec2;
use serde::{Deserialize, Serialize};
use std::any::{TypeId, type_name};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicU32, Ordering},
};

pub type GroupId = u32;

fn group_registry() -> &'static Mutex<HashMap<TypeId, GroupId>> {
    static REGISTRY: OnceLock<Mutex<HashMap<TypeId, GroupId>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn group_definition_registry() -> &'static Mutex<HashMap<GroupId, MuiGroupDefinition>> {
    static DEFINITIONS: OnceLock<Mutex<HashMap<GroupId, MuiGroupDefinition>>> = OnceLock::new();
    DEFINITIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn group_type_id<T: 'static>() -> GroupId {
    static NEXT_ID: AtomicU32 = AtomicU32::new(1);
    let type_id = TypeId::of::<T>();
    let registry = group_registry();
    let mut map = registry.lock().unwrap();
    *map.entry(type_id)
        .or_insert_with(|| NEXT_ID.fetch_add(1, Ordering::SeqCst))
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GroupCenterMode {
    FirstElement,
    LastElement,
    Average,
}

impl Default for GroupCenterMode {
    fn default() -> Self {
        GroupCenterMode::Average
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GroupOffsets {
    #[serde(default)]
    pub first: Option<[f32; 2]>,
    #[serde(default)]
    pub last: Option<[f32; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum GroupLayout {
    Unordered,
    Grid(GridLayout),
    Horizontal(LineLayout),
    Vertical(LineLayout),
    Ring(RingLayout),
    Curve(CurveLayout),
    Custom { descriptor: String },
}

impl Default for GroupLayout {
    fn default() -> Self {
        GroupLayout::Unordered
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GridLayout {
    #[serde(default)]
    pub spacing: [f32; 2],
    #[serde(default)]
    pub columns: Option<u32>,
    #[serde(default)]
    pub rows: Option<u32>,
}

impl Default for GridLayout {
    fn default() -> Self {
        Self {
            spacing: [0.0, 0.0],
            columns: None,
            rows: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LineLayout {
    pub spacing: f32,
    #[serde(default)]
    pub align_to_center: bool,
    #[serde(default)]
    pub wrap: Option<u32>,
}

impl Default for LineLayout {
    fn default() -> Self {
        Self {
            spacing: 0.0,
            align_to_center: false,
            wrap: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RingLayout {
    pub radius: f32,
    #[serde(default)]
    pub start_angle: f32,
    #[serde(default)]
    pub clockwise: bool,
}

impl Default for RingLayout {
    fn default() -> Self {
        Self {
            radius: 0.0,
            start_angle: 0.0,
            clockwise: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CurveLayout {
    pub sample_count: u32,
    #[serde(default)]
    pub tension: f32,
}

impl Default for CurveLayout {
    fn default() -> Self {
        Self {
            sample_count: 16,
            tension: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GroupAnimationScope {
    Whole,
    AttributeOffset,
    ContainerPipeline,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GroupAnimationSpec {
    pub scope: GroupAnimationScope,
    pub animation: AnimationSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TransitionAnchor {
    GroupCenter(GroupCenterMode),
    GroupSlot(u32),
    Pointer,
    Absolute([f32; 2]),
    LastKnownPosition,
    LinkedGroup {
        group: GroupId,
        center: GroupCenterMode,
    },
    Custom {
        descriptor: String,
    },
}

impl Default for TransitionAnchor {
    fn default() -> Self {
        TransitionAnchor::GroupCenter(GroupCenterMode::Average)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TransitionTimeline {
    pub duration: f32,
    pub delay: f32,
    pub easing: Easing,
    pub overshoot: f32,
}

impl Default for TransitionTimeline {
    fn default() -> Self {
        Self {
            duration: 0.0,
            delay: 0.0,
            easing: Easing::Linear,
            overshoot: 0.0,
        }
    }
}

impl TransitionTimeline {
    pub fn with_duration(seconds: f32) -> Self {
        Self {
            duration: seconds.max(0.0),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GroupTransitionMotion {
    pub scope: GroupAnimationScope,
    pub from: TransitionAnchor,
    pub to: TransitionAnchor,
    #[serde(default)]
    pub timeline: TransitionTimeline,
    #[serde(default)]
    pub snap_on_complete: bool,
}

impl Default for GroupTransitionMotion {
    fn default() -> Self {
        Self {
            scope: GroupAnimationScope::Whole,
            from: TransitionAnchor::Pointer,
            to: TransitionAnchor::GroupCenter(GroupCenterMode::Average),
            timeline: TransitionTimeline::default(),
            snap_on_complete: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum GroupTransitionBehavior {
    Immediate,
    Motion(GroupTransitionMotion),
    Custom { descriptor: String },
}

impl Default for GroupTransitionBehavior {
    fn default() -> Self {
        GroupTransitionBehavior::Immediate
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct GroupTransitionSet {
    #[serde(default)]
    pub entry: GroupTransitionBehavior,
    #[serde(default)]
    pub update: GroupTransitionBehavior,
    #[serde(default)]
    pub exit: GroupTransitionBehavior,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MuiGroupDefinition {
    pub id: GroupId,
    pub order: u32,
    pub cpu_index: u32,
    #[serde(default)]
    pub layout: GroupLayout,
    #[serde(default)]
    pub center: GroupCenterMode,
    #[serde(default)]
    pub offsets: GroupOffsets,
    #[serde(default)]
    pub transitions: GroupTransitionSet,
    #[serde(default)]
    pub animations: Vec<GroupAnimationSpec>,
}

impl Default for MuiGroupDefinition {
    fn default() -> Self {
        Self {
            id: 0,
            order: 0,
            cpu_index: 0,
            layout: GroupLayout::default(),
            center: GroupCenterMode::default(),
            offsets: GroupOffsets::default(),
            transitions: GroupTransitionSet::default(),
            animations: Vec::new(),
        }
    }
}

impl MuiGroupDefinition {
    pub fn for_type<T: 'static>() -> Self {
        let mut def = Self::default();
        def.id = group_type_id::<T>();
        def
    }
}

pub fn configure_group<T: 'static, F>(configure: F)
where
    F: FnOnce(MuiGroupBuilder<T>) -> MuiGroupBuilder<T>,
{
    let builder = configure(MuiGroupBuilder::<T>::new());
    let definition = builder.build();
    let mut map = group_definition_registry().lock().unwrap();
    map.insert(definition.id, definition);
}

pub fn group_definition<T: 'static>() -> MuiGroupDefinition {
    let id = group_type_id::<T>();
    let map = group_definition_registry();
    let guard = map.lock().unwrap();
    guard
        .get(&id)
        .cloned()
        .unwrap_or_else(|| MuiGroupBuilder::<T>::new().build())
}

pub struct MuiGroupBuilder<TLabel: 'static> {
    definition: MuiGroupDefinition,
    _marker: PhantomData<TLabel>,
}

impl<TLabel: 'static> MuiGroupBuilder<TLabel> {
    pub fn new() -> Self {
        Self {
            definition: MuiGroupDefinition {
                id: group_type_id::<TLabel>(),
                ..MuiGroupDefinition::default()
            },
            _marker: PhantomData,
        }
    }

    pub fn layout(mut self, layout: GroupLayout) -> Self {
        self.definition.layout = layout;
        self
    }

    pub fn configure_definition(mut self, configure: impl FnOnce(&mut MuiGroupDefinition)) -> Self {
        configure(&mut self.definition);
        self
    }

    pub fn order(mut self, order: u32) -> Self {
        self.definition.order = order;
        self
    }

    pub fn cpu_index(mut self, cpu_index: u32) -> Self {
        self.definition.cpu_index = cpu_index;
        self
    }

    pub fn unordered(self) -> Self {
        self.layout(GroupLayout::Unordered)
    }

    pub fn grid_layout(mut self, grid: GridLayout) -> Self {
        self.definition.layout = GroupLayout::Grid(grid);
        self
    }

    pub fn horizontal_layout(mut self, line: LineLayout) -> Self {
        self.definition.layout = GroupLayout::Horizontal(line);
        self
    }

    pub fn vertical_layout(mut self, line: LineLayout) -> Self {
        self.definition.layout = GroupLayout::Vertical(line);
        self
    }

    pub fn ring_layout(mut self, ring: RingLayout) -> Self {
        self.definition.layout = GroupLayout::Ring(ring);
        self
    }

    pub fn curve_layout(mut self, curve: CurveLayout) -> Self {
        self.definition.layout = GroupLayout::Curve(curve);
        self
    }

    pub fn custom_layout(mut self, descriptor: impl Into<String>) -> Self {
        self.definition.layout = GroupLayout::Custom {
            descriptor: descriptor.into(),
        };
        self
    }

    pub fn center(mut self, center: GroupCenterMode) -> Self {
        self.definition.center = center;
        self
    }

    pub fn first_offset(mut self, offset: impl Into<Vec2>) -> Self {
        self.definition.offsets.first = Some(offset.into().to_array());
        self
    }

    pub fn last_offset(mut self, offset: impl Into<Vec2>) -> Self {
        self.definition.offsets.last = Some(offset.into().to_array());
        self
    }

    pub fn entry_transition(mut self, behavior: GroupTransitionBehavior) -> Self {
        self.definition.transitions.entry = behavior;
        self
    }

    pub fn entry_motion<F>(mut self, scope: GroupAnimationScope, configure: F) -> Self
    where
        F: FnOnce(GroupTransitionMotionBuilder) -> GroupTransitionMotionBuilder,
    {
        let builder = GroupTransitionMotionBuilder::new(scope);
        let motion = configure(builder).build();
        self.definition.transitions.entry = GroupTransitionBehavior::Motion(motion);
        self
    }

    pub fn entry_immediate(mut self) -> Self {
        self.definition.transitions.entry = GroupTransitionBehavior::Immediate;
        self
    }

    pub fn entry_custom(mut self, descriptor: impl Into<String>) -> Self {
        self.definition.transitions.entry = GroupTransitionBehavior::Custom {
            descriptor: descriptor.into(),
        };
        self
    }

    pub fn update_transition(mut self, behavior: GroupTransitionBehavior) -> Self {
        self.definition.transitions.update = behavior;
        self
    }

    pub fn update_motion<F>(mut self, scope: GroupAnimationScope, configure: F) -> Self
    where
        F: FnOnce(GroupTransitionMotionBuilder) -> GroupTransitionMotionBuilder,
    {
        let builder = GroupTransitionMotionBuilder::new(scope);
        let motion = configure(builder).build();
        self.definition.transitions.update = GroupTransitionBehavior::Motion(motion);
        self
    }

    pub fn update_immediate(mut self) -> Self {
        self.definition.transitions.update = GroupTransitionBehavior::Immediate;
        self
    }

    pub fn update_custom(mut self, descriptor: impl Into<String>) -> Self {
        self.definition.transitions.update = GroupTransitionBehavior::Custom {
            descriptor: descriptor.into(),
        };
        self
    }

    pub fn exit_transition(mut self, behavior: GroupTransitionBehavior) -> Self {
        self.definition.transitions.exit = behavior;
        self
    }

    pub fn exit_motion<F>(mut self, scope: GroupAnimationScope, configure: F) -> Self
    where
        F: FnOnce(GroupTransitionMotionBuilder) -> GroupTransitionMotionBuilder,
    {
        let builder = GroupTransitionMotionBuilder::new(scope);
        let motion = configure(builder).build();
        self.definition.transitions.exit = GroupTransitionBehavior::Motion(motion);
        self
    }

    pub fn exit_immediate(mut self) -> Self {
        self.definition.transitions.exit = GroupTransitionBehavior::Immediate;
        self
    }

    pub fn exit_custom(mut self, descriptor: impl Into<String>) -> Self {
        self.definition.transitions.exit = GroupTransitionBehavior::Custom {
            descriptor: descriptor.into(),
        };
        self
    }

    pub fn push_animation(mut self, scope: GroupAnimationScope, animation: AnimationSpec) -> Self {
        self.definition
            .animations
            .push(GroupAnimationSpec { scope, animation });
        self
    }

    pub fn build(self) -> MuiGroupDefinition {
        self.definition
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum GroupConstraint {
    None,
    Rect { min: [f32; 2], max: [f32; 2] },
    Radius { center: [f32; 2], radius: f32 },
}

impl Default for GroupConstraint {
    fn default() -> Self {
        GroupConstraint::None
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum GroupRelationMode {
    ParentToChild,
    SharedOffset,
    Weighted { weight: f32 },
    Physics,
}

impl Default for GroupRelationMode {
    fn default() -> Self {
        GroupRelationMode::ParentToChild
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GroupRelationSpec {
    pub source: GroupId,
    pub target: GroupId,
    pub enabled: bool,
    #[serde(default)]
    pub constraint: GroupConstraint,
    #[serde(default)]
    pub mode: GroupRelationMode,
}

impl Default for GroupRelationSpec {
    fn default() -> Self {
        Self {
            source: 0,
            target: 0,
            enabled: true,
            constraint: GroupConstraint::default(),
            mode: GroupRelationMode::default(),
        }
    }
}

pub struct GroupRelationBuilder<Source: 'static, Target: 'static> {
    relation: GroupRelationSpec,
    _marker: PhantomData<(Source, Target)>,
}

impl<Source, Target> GroupRelationBuilder<Source, Target> {
    pub fn new() -> Self {
        Self {
            relation: GroupRelationSpec {
                source: group_type_id::<Source>(),
                target: group_type_id::<Target>(),
                ..GroupRelationSpec::default()
            },
            _marker: PhantomData,
        }
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.relation.enabled = enabled;
        self
    }

    pub fn no_constraint(mut self) -> Self {
        self.relation.constraint = GroupConstraint::None;
        self
    }

    pub fn constraint_rect(mut self, min: impl Into<Vec2>, max: impl Into<Vec2>) -> Self {
        self.relation.constraint = GroupConstraint::Rect {
            min: min.into().to_array(),
            max: max.into().to_array(),
        };
        self
    }

    pub fn constraint_radius(mut self, center: impl Into<Vec2>, radius: f32) -> Self {
        self.relation.constraint = GroupConstraint::Radius {
            center: center.into().to_array(),
            radius,
        };
        self
    }

    pub fn parent_to_child(mut self) -> Self {
        self.relation.mode = GroupRelationMode::ParentToChild;
        self
    }

    pub fn shared_offset(mut self) -> Self {
        self.relation.mode = GroupRelationMode::SharedOffset;
        self
    }

    pub fn weighted(mut self, weight: f32) -> Self {
        self.relation.mode = GroupRelationMode::Weighted {
            weight: weight.clamp(0.0, 1.0),
        };
        self
    }

    pub fn physics(mut self) -> Self {
        self.relation.mode = GroupRelationMode::Physics;
        self
    }

    pub fn configure_definition(mut self, configure: impl FnOnce(&mut GroupRelationSpec)) -> Self {
        configure(&mut self.relation);
        self
    }

    pub fn build(self) -> GroupRelationSpec {
        self.relation
    }
}

pub struct GroupTransitionMotionBuilder {
    motion: GroupTransitionMotion,
}

impl GroupTransitionMotionBuilder {
    pub fn new(scope: GroupAnimationScope) -> Self {
        Self {
            motion: GroupTransitionMotion {
                scope,
                ..GroupTransitionMotion::default()
            },
        }
    }

    pub fn from(mut self, anchor: TransitionAnchor) -> Self {
        self.motion.from = anchor;
        self
    }

    pub fn to(mut self, anchor: TransitionAnchor) -> Self {
        self.motion.to = anchor;
        self
    }

    pub fn timeline(mut self, timeline: TransitionTimeline) -> Self {
        self.motion.timeline = timeline;
        self
    }

    pub fn duration(mut self, seconds: f32) -> Self {
        self.motion.timeline.duration = seconds.max(0.0);
        self
    }

    pub fn delay(mut self, seconds: f32) -> Self {
        self.motion.timeline.delay = seconds.max(0.0);
        self
    }

    pub fn easing(mut self, easing: Easing) -> Self {
        self.motion.timeline.easing = easing;
        self
    }

    pub fn overshoot(mut self, value: f32) -> Self {
        self.motion.timeline.overshoot = value.max(0.0);
        self
    }

    pub fn snap_on_complete(mut self, enabled: bool) -> Self {
        self.motion.snap_on_complete = enabled;
        self
    }

    pub fn build(self) -> GroupTransitionMotion {
        self.motion
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GroupGraphDefinition {
    #[serde(default)]
    pub groups: Vec<MuiGroupDefinition>,
    #[serde(default)]
    pub relations: Vec<GroupRelationSpec>,
}

pub struct GroupGraphBuilder {
    groups: Vec<MuiGroupDefinition>,
    relations: Vec<GroupRelationSpec>,
}

impl GroupGraphBuilder {
    pub fn new() -> Self {
        Self {
            groups: Vec::new(),
            relations: Vec::new(),
        }
    }

    pub fn group<L, F>(&mut self, configure: F) -> &mut Self
    where
        L: 'static,
        F: FnOnce(MuiGroupBuilder<L>) -> MuiGroupBuilder<L>,
    {
        let builder = MuiGroupBuilder::<L>::new();
        let definition = configure(builder).build();
        self.groups.push(definition);
        self
    }

    pub fn relation<S, T, F>(&mut self, configure: F) -> &mut Self
    where
        S: 'static,
        T: 'static,
        F: FnOnce(GroupRelationBuilder<S, T>) -> GroupRelationBuilder<S, T>,
    {
        let builder = GroupRelationBuilder::<S, T>::new();
        let relation = configure(builder).build();
        self.relations.push(relation);
        self
    }

    pub fn finish(self) -> GroupGraphDefinition {
        GroupGraphDefinition {
            groups: self.groups,
            relations: self.relations,
        }
    }
}
