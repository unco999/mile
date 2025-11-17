use serde::{Deserialize, Serialize};
use std::{
    any::{TypeId, type_name},
    collections::{HashMap, HashSet},
};

/// Field identifiers that can be targeted by relation rules.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelPanelField {
    Position,
    PositionX,
    PositionY,
    /// Lock movement to X axis (Y should remain unchanged during interaction).
    OnlyPositionX,
    /// Lock movement to Y axis (X should remain unchanged during interaction).
    OnlyPositionY,
    Size,
    SizeX,
    SizeY,
    /// User-defined field name (owned so we can serialize/deserialize safely).
    Custom(String),
}

/// Frequently used field constants to match the relation DSL style.
#[allow(non_upper_case_globals)]
pub mod panel_field {
    use super::RelPanelField;

    pub const position: RelPanelField = RelPanelField::Position;
    pub const position_x: RelPanelField = RelPanelField::PositionX;
    pub const position_y: RelPanelField = RelPanelField::PositionY;
    pub const only_position_x: RelPanelField = RelPanelField::OnlyPositionX;
    pub const only_position_y: RelPanelField = RelPanelField::OnlyPositionY;
    pub const size: RelPanelField = RelPanelField::Size;
    pub const size_x: RelPanelField = RelPanelField::SizeX;
    pub const size_y: RelPanelField = RelPanelField::SizeY;

    /// Create a custom field key by name.
    pub fn custom(name: impl Into<String>) -> RelPanelField {
        RelPanelField::Custom(name.into())
    }
}

/// Public alias used by higher-level UI builders.
/// Keeps call sites concise (e.g., `Field::PositionX`) while still mapping
/// to the internal `RelPanelField` enum.
pub type Field = RelPanelField;

/// Unique key describing a concrete panel view (payload type + UUID + optional scope/state).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RelViewKey {
    pub payload: TypeId,
    pub payload_name: &'static str,
    pub panel_uuid: String,
    pub scope: Option<String>,
    pub state: Option<u32>,
}

impl RelViewKey {
    pub fn for_panel<T: 'static>(panel_uuid: impl Into<String>) -> Self {
        Self {
            payload: TypeId::of::<T>(),
            payload_name: type_name::<T>(),
            panel_uuid: panel_uuid.into(),
            scope: None,
            state: None,
        }
    }

    pub fn for_owner<T: 'static>(
        panel_uuid: impl Into<String>,
        scope: impl Into<String>,
        state: u32,
    ) -> Self {
        Self {
            payload: TypeId::of::<T>(),
            payload_name: type_name::<T>(),
            panel_uuid: panel_uuid.into(),
            scope: Some(scope.into()),
            state: Some(state),
        }
    }

    pub fn scoped(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    pub fn with_state(mut self, state: u32) -> Self {
        self.state = Some(state);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RelRuleId(u32);

impl RelRuleId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RelRuleHandle {
    pub id: RelRuleId,
    pub target: RelViewKey,
}

/// Spatial context describing how container coordinates map into the global UI graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelSpace {
    Screen,
    Parent,
    Local,
}

impl Default for RelSpace {
    fn default() -> Self {
        RelSpace::Local
    }
}

/// Axis helpers for scrollable container regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelScrollAxis {
    None,
    Horizontal,
    Vertical,
    Both,
}

impl Default for RelScrollAxis {
    fn default() -> Self {
        RelScrollAxis::None
    }
}

/// Describes how a view transitions when entering or exiting a container.
#[derive(Clone, Debug, PartialEq)]
pub enum RelTransition {
    Immediate,
    Timed(f32),
}

impl RelTransition {
    pub fn timed(seconds: f32) -> Self {
        RelTransition::Timed(seconds.max(0.0))
    }
}

impl Default for RelTransition {
    fn default() -> Self {
        RelTransition::Immediate
    }
}

/// Layout metadata a relation owner can expose for consumers that call `container_with`.
#[derive(Clone, Debug, PartialEq)]
pub struct RelContainerSpec {
    pub space: RelSpace,
    pub origin: [f32; 2],
    pub size: Option<[f32; 2]>,
    pub size_percent_of_parent: Option<[f32; 2]>,
    pub padding: [f32; 4],
    pub clip_content: bool,
    pub scroll_axis: RelScrollAxis,
    pub layout: RelLayoutKind,
    pub slot_size: Option<[f32; 2]>,
    pub element_scale: [f32; 2],
}

impl Default for RelContainerSpec {
    fn default() -> Self {
        Self {
            space: RelSpace::default(),
            origin: [0.0, 0.0],
            size: None,
            size_percent_of_parent: None,
            padding: [0.0; 4],
            clip_content: true,
            scroll_axis: RelScrollAxis::None,
            layout: RelLayoutKind::free(),
            slot_size: None,
            element_scale: [1.0, 1.0],
        }
    }
}

impl RelContainerSpec {
    pub fn space(mut self, space: RelSpace) -> Self {
        self.space = space;
        self
    }

    pub fn origin(mut self, origin: [f32; 2]) -> Self {
        self.origin = origin;
        self
    }

    pub fn size(mut self, size: [f32; 2]) -> Self {
        self.size = Some(size);
        self
    }

    pub fn clear_size(mut self) -> Self {
        self.size = None;
        self
    }

    pub fn padding(mut self, padding: [f32; 4]) -> Self {
        self.padding = padding;
        self
    }

    pub fn clip_content(mut self, clip: bool) -> Self {
        self.clip_content = clip;
        self
    }

    pub fn scroll_axis(mut self, axis: RelScrollAxis) -> Self {
        self.scroll_axis = axis;
        self
    }

    pub fn layout(mut self, layout: RelLayoutKind) -> Self {
        self.layout = layout;
        self
    }

    pub fn configure_layout(mut self, configure: impl FnOnce(&mut RelLayoutKind)) -> Self {
        configure(&mut self.layout);
        self
    }

    pub fn size_percent_of_parent(mut self, percent: [f32; 2]) -> Self {
        self.size_percent_of_parent = Some(percent);
        self
    }

    pub fn clear_size_percent(mut self) -> Self {
        self.size_percent_of_parent = None;
        self
    }

    pub fn slot_size(mut self, slot: [f32; 2]) -> Self {
        self.slot_size = Some(slot);
        self
    }

    pub fn clear_slot_size(mut self) -> Self {
        self.slot_size = None;
        self
    }

    pub fn element_scale(mut self, scale: [f32; 2]) -> Self {
        self.element_scale = scale;
        self
    }

    pub fn reset_element_scale(mut self) -> Self {
        self.element_scale = [1.0, 1.0];
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RelLayoutKind {
    Free,
    Horizontal {
        spacing: f32,
        align_center: bool,
    },
    Vertical {
        spacing: f32,
        align_center: bool,
    },
    Grid {
        spacing: [f32; 2],
        columns: Option<u32>,
        rows: Option<u32>,
    },
    Ring {
        radius: f32,
        start_angle: f32,
        clockwise: bool,
    },
    Custom {
        descriptor: String,
    },
}

impl RelLayoutKind {
    pub fn free() -> Self {
        RelLayoutKind::Free
    }

    pub fn horizontal(spacing: f32) -> Self {
        RelLayoutKind::Horizontal {
            spacing,
            align_center: false,
        }
    }

    pub fn vertical(spacing: f32) -> Self {
        RelLayoutKind::Vertical {
            spacing,
            align_center: false,
        }
    }

    pub fn grid(spacing: [f32; 2]) -> Self {
        RelLayoutKind::Grid {
            spacing,
            columns: None,
            rows: None,
        }
    }

    pub fn ring(radius: f32) -> Self {
        RelLayoutKind::Ring {
            radius,
            start_angle: 0.0,
            clockwise: true,
        }
    }
}

/// Relationship rule declared for or against other panel views.
#[derive(Clone, Debug)]
pub enum RelRule {
    Mutex {
        target: RelViewKey,
    },
    Dependency {
        target: RelViewKey,
    },
    AttachField {
        target: RelViewKey,
        field: RelPanelField,
    },
    ContainerLink {
        target: RelViewKey,
        entry: RelTransition,
        exit: RelTransition,
    },
}

impl RelRule {
    pub fn target_key(&self) -> Option<&RelViewKey> {
        match self {
            RelRule::Mutex { target }
            | RelRule::Dependency { target }
            | RelRule::AttachField { target, .. }
            | RelRule::ContainerLink { target, .. } => Some(target),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RelRuleEntry {
    pub id: RelRuleId,
    pub rule: RelRule,
}

/// Snapshot containing every rule authored for a specific panel state.
#[derive(Clone, Debug)]
pub struct RelGraphDefinition {
    pub owner: RelViewKey,
    pub rules: Vec<RelRuleEntry>,
    pub container: Option<RelContainerSpec>,
    pub properties: Vec<RelProperty>,
}

impl RelGraphDefinition {
    pub fn new(
        owner: RelViewKey,
        rules: Vec<RelRuleEntry>,
        container: Option<RelContainerSpec>,
        properties: Vec<RelProperty>,
    ) -> Self {
        Self {
            owner,
            rules,
            container,
            properties,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty() && self.container.is_none() && self.properties.is_empty()
    }

    pub fn parse(&self) -> RelParsedGraph {
        RelParser::parse(self)
    }
}

#[derive(Clone, Debug)]
pub struct RelGraphState {
    pub owner: RelViewKey,
    pub rules: Vec<RelRuleEntry>,
    pub rule_status: HashMap<RelRuleId, RelRuleStatus>,
    pub container: Option<RelContainerSpec>,
    pub properties: Vec<RelProperty>,
}

impl RelGraphState {
    pub fn new(definition: RelGraphDefinition) -> Self {
        let rule_status = definition
            .rules
            .iter()
            .map(|entry| (entry.id, RelRuleStatus::Active))
            .collect();
        Self {
            owner: definition.owner,
            rules: definition.rules,
            rule_status,
            container: definition.container,
            properties: definition.properties,
        }
    }

    pub fn set_status(&mut self, id: RelRuleId, status: RelRuleStatus) {
        if let Some(entry) = self.rule_status.get_mut(&id) {
            *entry = status;
        }
    }
}

#[derive(Clone, Debug)]
pub struct RelParsedGraph {
    pub owner: RelViewKey,
    pub nodes: HashMap<RelViewKey, RelNodeState>,
    pub container: Option<RelContainerSpec>,
    pub diagnostics: RelDiagnostics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelRuleStatus {
    Active,
    Suspended,
}

impl Default for RelRuleStatus {
    fn default() -> Self {
        RelRuleStatus::Active
    }
}

/// Helper owned by the state builder so DSL calls can push relation rules.
#[derive(Clone, Debug)]
pub struct RelComposer {
    owner: RelViewKey,
    rules: Vec<RelRuleEntry>,
    container: Option<RelContainerSpec>,
    next_rule_id: u32,
    properties: Vec<RelProperty>,
}

impl RelComposer {
    pub fn new(owner: RelViewKey) -> Self {
        Self {
            owner,
            rules: Vec::new(),
            container: None,
            next_rule_id: 0,
            properties: Vec::new(),
        }
    }

    pub fn owner(&self) -> &RelViewKey {
        &self.owner
    }

    pub fn mutex_view<T: 'static>(&mut self, panel_uuid: impl Into<String>) -> &mut Self {
        let panel_uuid = panel_uuid.into();
        let target = RelViewKey::for_panel::<T>(panel_uuid);
        self.push_rule(RelRule::Mutex { target });
        self
    }

    pub fn remove_mutex_view<T: 'static>(&mut self, panel_uuid: impl Into<String>) -> &mut Self {
        let panel_uuid = panel_uuid.into();
        let key = RelViewKey::for_panel::<T>(panel_uuid);
        self.remove_matching(|rule| matches!(rule, RelRule::Mutex { target } if target == &key));
        self
    }

    pub fn dep_view<T: 'static>(&mut self, panel_uuid: impl Into<String>) -> &mut Self {
        let panel_uuid = panel_uuid.into();
        let target = RelViewKey::for_panel::<T>(panel_uuid);
        self.push_rule(RelRule::Dependency { target });
        self
    }

    pub fn remove_dep_view<T: 'static>(&mut self, panel_uuid: impl Into<String>) -> &mut Self {
        let panel_uuid = panel_uuid.into();
        let key = RelViewKey::for_panel::<T>(panel_uuid);
        self.remove_matching(
            |rule| matches!(rule, RelRule::Dependency { target } if target == &key),
        );
        self
    }

    pub fn attach<T: 'static>(
        &mut self,
        panel_uuid: impl Into<String>,
        field: RelPanelField,
    ) -> &mut Self {
        let panel_uuid = panel_uuid.into();
        let target = RelViewKey::for_panel::<T>(panel_uuid);
        self.push_rule(RelRule::AttachField { target, field });
        self
    }

    pub fn remove_attach<T: 'static>(
        &mut self,
        panel_uuid: impl Into<String>,
        field: RelPanelField,
    ) -> &mut Self {
        let panel_uuid = panel_uuid.into();
        let key = RelViewKey::for_panel::<T>(panel_uuid);
        self.remove_matching(|rule| {
            matches!(
                rule,
                RelRule::AttachField {
                    target,
                    field: existing,
                } if target == &key && *existing == field
            )
        });
        self
    }

    pub fn container_with<T: 'static>(&mut self, panel_uuid: impl Into<String>) -> &mut Self {
        let panel_uuid = panel_uuid.into();
        let target = RelViewKey::for_panel::<T>(panel_uuid);
        self.push_rule(RelRule::ContainerLink {
            target,
            entry: RelTransition::Immediate,
            exit: RelTransition::Immediate,
        });
        self
    }

    pub fn with_container_entry<T: 'static>(
        &mut self,
        panel_uuid: impl Into<String>,
        transition: RelTransition,
    ) -> &mut Self {
        let panel_uuid = panel_uuid.into();
        let key = RelViewKey::for_panel::<T>(panel_uuid);
        if !self.modify_container_link(&key, |entry, _| *entry = transition.clone()) {
            self.push_rule(RelRule::ContainerLink {
                target: key,
                entry: transition,
                exit: RelTransition::Immediate,
            });
        }
        self
    }

    pub fn with_container_exit<T: 'static>(
        &mut self,
        panel_uuid: impl Into<String>,
        transition: RelTransition,
    ) -> &mut Self {
        let panel_uuid = panel_uuid.into();
        let key = RelViewKey::for_panel::<T>(panel_uuid);
        if !self.modify_container_link(&key, |_, exit| *exit = transition.clone()) {
            self.push_rule(RelRule::ContainerLink {
                target: key,
                entry: RelTransition::Immediate,
                exit: transition,
            });
        }
        self
    }

    pub fn remove_container_with<T: 'static>(
        &mut self,
        panel_uuid: impl Into<String>,
    ) -> &mut Self {
        let panel_uuid = panel_uuid.into();
        let key = RelViewKey::for_panel::<T>(panel_uuid);
        self.remove_matching(|rule| {
            matches!(
                rule,
                RelRule::ContainerLink { target, .. } if target == &key
            )
        });
        self
    }

    pub fn has_rules(&self) -> bool {
        !self.rules.is_empty() || self.container.is_some() || !self.properties.is_empty()
    }

    pub fn clear(&mut self) {
        self.rules.clear();
        self.container = None;
        self.next_rule_id = 0;
        self.properties.clear();
    }

    pub fn into_definition(self) -> Option<RelGraphDefinition> {
        if self.rules.is_empty() && self.container.is_none() {
            None
        } else {
            Some(RelGraphDefinition::new(
                self.owner,
                self.rules,
                self.container,
                self.properties,
            ))
        }
    }

    pub fn container_spec(&self) -> Option<&RelContainerSpec> {
        self.container.as_ref()
    }

    pub fn container_self<F>(&mut self, configure: F) -> &mut Self
    where
        F: FnOnce(&mut RelContainerSpec),
    {
        let mut spec = self.container.take().unwrap_or_default();
        configure(&mut spec);
        self.container = Some(spec);
        self
    }

    pub fn clear_container_self(&mut self) -> &mut Self {
        self.container = None;
        self
    }

    pub fn position_fixed(&mut self, space: RelSpace, value: [f32; 2]) -> &mut Self {
        self.set_property(RelProperty::Position {
            space,
            value: RelVec2::Fixed(value),
        });
        self
    }

    pub fn position_signal(&mut self, space: RelSpace, signal: &'static str) -> &mut Self {
        self.set_property(RelProperty::Position {
            space,
            value: RelVec2::Signal(signal),
        });
        self
    }

    pub fn clear_position(&mut self) -> &mut Self {
        self.remove_property(|prop| matches!(prop, RelProperty::Position { .. }));
        self
    }

    pub fn size_fixed(&mut self, value: [f32; 2]) -> &mut Self {
        self.set_property(RelProperty::Size(RelVec2::Fixed(value)));
        self
    }

    pub fn size_signal(&mut self, signal: &'static str) -> &mut Self {
        self.set_property(RelProperty::Size(RelVec2::Signal(signal)));
        self
    }

    pub fn clear_size(&mut self) -> &mut Self {
        self.remove_property(|prop| matches!(prop, RelProperty::Size(_)));
        self
    }

    pub fn scalar_fixed(&mut self, field: RelPanelField, value: f32) -> &mut Self {
        self.set_property(RelProperty::Custom {
            field,
            value: RelScalar::Fixed(value),
        });
        self
    }

    pub fn scalar_signal(&mut self, field: RelPanelField, signal: &'static str) -> &mut Self {
        self.set_property(RelProperty::Custom {
            field,
            value: RelScalar::Signal(signal),
        });
        self
    }

    pub fn clear_scalar(&mut self, field: RelPanelField) -> &mut Self {
        self.remove_property(|prop| {
            matches!(
                prop,
                RelProperty::Custom {
                    field: existing, ..
                } if *existing == field
            )
        });
        self
    }

    pub fn properties(&self) -> &[RelProperty] {
        &self.properties
    }

    pub fn rule_entries(&self) -> &[RelRuleEntry] {
        &self.rules
    }

    pub fn rule(&self, id: RelRuleId) -> Option<&RelRule> {
        self.rules
            .iter()
            .find(|entry| entry.id == id)
            .map(|entry| &entry.rule)
    }

    pub fn rule_handle(&self, id: RelRuleId) -> Option<RelRuleHandle> {
        self.rules
            .iter()
            .find(|entry| entry.id == id)
            .and_then(|entry| {
                entry.rule.target_key().map(|target| RelRuleHandle {
                    id: entry.id,
                    target: target.clone(),
                })
            })
    }

    pub fn handles(&self) -> Vec<RelRuleHandle> {
        self.rules
            .iter()
            .filter_map(|entry| {
                entry.rule.target_key().map(|target| RelRuleHandle {
                    id: entry.id,
                    target: target.clone(),
                })
            })
            .collect()
    }

    fn remove_matching<F>(&mut self, predicate: F)
    where
        F: Fn(&RelRule) -> bool,
    {
        self.rules.retain(|entry| !predicate(&entry.rule));
    }

    fn modify_container_link<F>(&mut self, key: &RelViewKey, mut apply: F) -> bool
    where
        F: FnOnce(&mut RelTransition, &mut RelTransition),
    {
        for rule_entry in &mut self.rules {
            if let RelRule::ContainerLink {
                target,
                entry,
                exit,
            } = &mut rule_entry.rule
            {
                if target == key {
                    apply(entry, exit);
                    return true;
                }
            }
        }
        false
    }

    fn push_rule(&mut self, rule: RelRule) -> RelRuleId {
        let id = RelRuleId::new(self.next_rule_id);
        self.next_rule_id += 1;
        self.rules.push(RelRuleEntry { id, rule });
        id
    }

    fn set_property(&mut self, property: RelProperty) {
        self.remove_property(|prop| {
            std::mem::discriminant(prop) == std::mem::discriminant(&property)
                && match (&property, prop) {
                    (
                        RelProperty::Custom { field, .. },
                        RelProperty::Custom {
                            field: existing, ..
                        },
                    ) => field == existing,
                    _ => true,
                }
        });
        self.properties.push(property);
    }

    fn remove_property<F>(&mut self, predicate: F)
    where
        F: Fn(&RelProperty) -> bool,
    {
        self.properties.retain(|prop| !predicate(prop));
    }
}
#[derive(Clone, Debug)]
pub struct RelNodeState {
    pub key: RelViewKey,
    pub dependencies: Vec<RelRuleId>,
    pub mutex_group: Vec<RelRuleId>,
    pub attached_fields: Vec<RelRuleId>,
    pub container_links: Vec<RelContainerLinkState>,
    pub properties: Vec<RelProperty>,
}

impl RelNodeState {
    pub fn new(key: RelViewKey) -> Self {
        Self {
            key,
            dependencies: Vec::new(),
            mutex_group: Vec::new(),
            attached_fields: Vec::new(),
            container_links: Vec::new(),
            properties: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RelContainerLinkState {
    pub rule: RelRuleId,
    pub target: RelViewKey,
    pub entry: RelTransition,
    pub exit: RelTransition,
}

#[derive(Clone, Debug, Default)]
pub struct RelContainerRegistry {
    pub specs: HashMap<RelViewKey, RelContainerSpec>,
}

impl RelContainerRegistry {
    pub fn register(&mut self, key: RelViewKey, spec: RelContainerSpec) {
        self.specs.insert(key, spec);
    }

    pub fn deregister(&mut self, key: &RelViewKey) {
        self.specs.remove(key);
    }

    pub fn get(&self, key: &RelViewKey) -> Option<&RelContainerSpec> {
        self.specs.get(key)
    }
}

#[derive(Clone, Debug, Default)]
pub struct RelDiagnostics {
    pub warnings: Vec<String>,
    pub conflicts: Vec<RelConflict>,
    pub visited: HashSet<RelViewKey>,
}

#[derive(Clone, Debug)]
pub struct RelConflict {
    pub rule: RelRuleId,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RelScalar {
    Fixed(f32),
    Signal(&'static str),
}

#[derive(Clone, Debug, PartialEq)]
pub enum RelVec2 {
    Fixed([f32; 2]),
    Signal(&'static str),
}

#[derive(Clone, Debug, PartialEq)]
pub enum RelProperty {
    Position {
        space: RelSpace,
        value: RelVec2,
    },
    Size(RelVec2),
    Custom {
        field: RelPanelField,
        value: RelScalar,
    },
}

pub struct RelParser;

impl RelParser {
    pub fn parse(definition: &RelGraphDefinition) -> RelParsedGraph {
        let mut diagnostics = RelDiagnostics::default();
        let mut nodes: HashMap<RelViewKey, RelNodeState> = HashMap::new();
        let mut seen_ids: HashSet<RelRuleId> = HashSet::new();

        {
            let owner_entry = nodes
                .entry(definition.owner.clone())
                .or_insert_with(|| RelNodeState::new(definition.owner.clone()));
            owner_entry.properties.extend(definition.properties.clone());
        }

        for entry in &definition.rules {
            if !seen_ids.insert(entry.id) {
                diagnostics.conflicts.push(RelConflict {
                    rule: entry.id,
                    message: format!("duplicate relation rule id {}", entry.id.raw()),
                });
                continue;
            }

            let Some(target_key) = entry.rule.target_key().cloned() else {
                diagnostics.warnings.push(format!(
                    "relation rule {} missing target key",
                    entry.id.raw()
                ));
                continue;
            };

            let node = nodes
                .entry(target_key.clone())
                .or_insert_with(|| RelNodeState::new(target_key));

            match &entry.rule {
                RelRule::Mutex { .. } => node.mutex_group.push(entry.id),
                RelRule::Dependency { .. } => node.dependencies.push(entry.id),
                RelRule::AttachField { .. } => node.attached_fields.push(entry.id),
                RelRule::ContainerLink {
                    target,
                    entry: entry_transition,
                    exit: exit_transition,
                } => node.container_links.push(RelContainerLinkState {
                    rule: entry.id,
                    target: target.clone(),
                    entry: entry_transition.clone(),
                    exit: exit_transition.clone(),
                }),
            }
        }

        RelParsedGraph {
            owner: definition.owner.clone(),
            nodes,
            container: definition.container.clone(),
            diagnostics,
        }
    }
}
