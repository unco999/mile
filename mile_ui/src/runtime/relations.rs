use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
    sync::{Mutex, OnceLock},
};

use crate::{
    mui_prototype::{PanelKey, UiState, panel_numeric_id, registered_panel_keys},
    mui_rel::{
        RelContainerLinkState, RelContainerSpec, RelGraphDefinition, RelLayoutKind, RelParsedGraph,
        RelNodeState, RelSpace, RelTransition, RelViewKey,
    },
};

#[derive(Default)]
pub struct RelationRegistry {
    panels: HashMap<u32, PanelRelationEntry>,
    dirty: HashSet<u32>,
    manual: Vec<RelationWorkItem>,
    container_members: HashMap<u32, Vec<u32>>,
}

#[derive(Default)]
struct PanelRelationEntry {
    states: HashMap<u32, RelParsedGraph>,
    active_state: Option<u32>,
}

impl PanelRelationEntry {
    fn active_graph(&self) -> Option<&RelParsedGraph> {
        let state = self.active_state?;
        self.states.get(&state)
    }
}

#[derive(Clone)]
pub struct RelationWorkItem {
    pub panel_id: u32,
    pub container_id: u32,
    pub layout_flags: u32,
    pub order: u32,
    pub total: u32,
    pub _pad0: u32,
    pub origin: [f32; 2],
    pub size: [f32; 2],
    pub slot: [f32; 2],
    pub spacing: [f32; 2],
    pub padding: [f32; 4],
    pub percent: [f32; 2],
    pub scale: [f32; 2],
    pub entry_mode: u32,
    pub entry_param: f32,
    pub exit_mode: u32,
    pub exit_param: f32,
}

impl Default for RelationWorkItem {
    fn default() -> Self {
        Self {
            panel_id: 0,
            container_id: 0,
            layout_flags: 0,
            order: 0,
            total: 0,
             _pad0: 0,
            origin: [0.0, 0.0],
            size: [0.0, 0.0],
            slot: [0.0, 0.0],
            spacing: [0.0, 0.0],
            padding: [0.0; 4],
            percent: [0.0, 0.0],
            scale: [1.0, 1.0],
            entry_mode: 0,
            entry_param: 0.0,
            exit_mode: 0,
            exit_param: 0.0,
        }
    }
}

impl RelationRegistry {
    pub fn register(&mut self, panel_id: u32, state: UiState, definition: RelGraphDefinition) {
        let graph = definition.parse();
        self.emit_registration_feedback(panel_id, state, &graph);

        let entry = self.panels.entry(panel_id).or_default();
        entry.states.insert(state.0, graph);
        if entry.active_state.is_none() {
            entry.active_state = Some(state.0);
        }
        self.dirty.insert(panel_id);
        self.rebuild_memberships();
        self.mark_all_container_children_dirty();
    }

    pub fn clear_panel(&mut self, panel_id: u32) {
        self.panels.remove(&panel_id);
        self.dirty.insert(panel_id);
        self.rebuild_memberships();
        self.mark_all_container_children_dirty();
    }

    pub fn set_active_state(&mut self, panel_id: u32, state: UiState) {
        if let Some(entry) = self.panels.get_mut(&panel_id) {
            if entry.active_state != Some(state.0) {
                entry.active_state = Some(state.0);
                self.dirty.insert(panel_id);
                self.rebuild_memberships();
                self.mark_all_container_children_dirty();
            }
        }
    }

    pub fn take_dirty(&mut self) -> Vec<RelationWorkItem> {
        let dirty_ids: Vec<u32> = self.dirty.drain().collect();
        let mut items: Vec<RelationWorkItem> = Vec::new();
        for panel_id in dirty_ids {
            if let Some(entry) = self.panels.get(&panel_id) {
                if let Some(graph) = entry.active_graph() {
                    items.extend(self.build_work_items(panel_id, graph));
                }
            }
        }
        if !self.manual.is_empty() {
            items.extend(self.manual.drain(..));
        }
        items
    }

    fn build_work_items(
        &self,
        panel_id: u32,
        graph: &RelParsedGraph,
    ) -> Vec<RelationWorkItem> {
        let mut items = Vec::new();
        for node in graph.nodes.values() {
            for link in &node.container_links {
                let Some(container_panel_id) = resolve_panel_id(&link.target) else {
                    continue;
                };
                let Some(spec) = self.active_container_spec(container_panel_id) else {
                    eprintln!(
                        "[mui::rel] panel {panel_id}: container '{}' missing spec; \
                     consider calling container_self on that panel state.",
                        link.target.panel_uuid
                    );
                    continue;
                };
                items.push(self.build_container_work_item(
                    panel_id,
                    container_panel_id,
                    spec,
                    link,
                ));
            }
        }
        items
    }

    fn build_container_work_item(
        &self,
        child_panel_id: u32,
        container_panel_id: u32,
        spec: &RelContainerSpec,
        link: &RelContainerLinkState,
    ) -> RelationWorkItem {
        let (mut layout_flags, spacing) = encode_layout(spec);
        if spec.size_percent_of_parent.is_some() {
            layout_flags |= layout_flags::HAS_PERCENT;
        }
        layout_flags |= encode_space(spec.space);

        let slot = spec
            .slot_size
            .or(spec.size)
            .unwrap_or([0.0f32, 0.0f32]);
        let size = spec.size.unwrap_or(slot);

        let (entry_mode, entry_param) = encode_transition(&link.entry);
        let (exit_mode, exit_param) = encode_transition(&link.exit);
        RelationWorkItem {
            panel_id: child_panel_id,
            container_id: container_panel_id,
            layout_flags,
            order: self.container_order(container_panel_id, child_panel_id) as u32,
            total: self
                .container_members
                .get(&container_panel_id)
                .map(|children| children.len() as u32)
                .unwrap_or(1),
            _pad0: 0,
            origin: spec.origin,
            size,
            slot,
            spacing,
            padding: spec.padding,
            percent: spec.size_percent_of_parent.unwrap_or([0.0, 0.0]),
            scale: spec.element_scale,
            entry_mode,
            entry_param,
            exit_mode,
            exit_param,
        }
    }

    fn active_container_spec(&self, panel_id: u32) -> Option<&RelContainerSpec> {
        self.panels
            .get(&panel_id)
            .and_then(|entry| entry.active_graph())
            .and_then(|graph| graph.container.as_ref())
    }

    fn container_order(&self, container_panel_id: u32, child: u32) -> usize {
        self.container_members
            .get(&container_panel_id)
            .and_then(|children| children.iter().position(|id| *id == child))
            .unwrap_or(0)
    }

    fn rebuild_memberships(&mut self) {
        self.container_members.clear();
        for (&panel_id, entry) in &self.panels {
            let Some(graph) = entry.active_graph() else {
                continue;
            };
            for node in graph.nodes.values() {
                for link in &node.container_links {
                    if let Some(container_panel_id) = resolve_panel_id(&link.target) {
                        self.container_members
                            .entry(container_panel_id)
                            .or_default()
                            .push(panel_id);
                    }
                }
            }
        }
        for children in self.container_members.values_mut() {
            children.sort_unstable();
            children.dedup();
        }
    }

    fn mark_all_container_children_dirty(&mut self) {
        for children in self.container_members.values() {
            for &child in children {
                self.dirty.insert(child);
            }
        }
    }

    pub fn inject_manual(&mut self, items: Vec<RelationWorkItem>) {
        self.manual.extend(items);
    }

    fn emit_registration_feedback(&self, panel_id: u32, state: UiState, graph: &RelParsedGraph) {
        self.log_parser_diagnostics(panel_id, state, graph);
        self.warn_unknown_targets(panel_id, state, graph);
    }

    fn log_parser_diagnostics(&self, panel_id: u32, state: UiState, graph: &RelParsedGraph) {
        for warning in &graph.diagnostics.warnings {
            eprintln!(
                "[mui::rel] panel {panel_id} state {} warning: {warning}",
                state.0
            );
        }
        for conflict in &graph.diagnostics.conflicts {
            eprintln!(
                "[mui::rel] panel {panel_id} state {} conflict (rule {}): {}",
                state.0,
                conflict.rule.raw(),
                conflict.message
            );
        }
    }

    fn warn_unknown_targets(&self, panel_id: u32, state: UiState, graph: &RelParsedGraph) {
        let mut cache: HashMap<TypeId, Vec<PanelKey>> = HashMap::new();
        for node in graph.nodes.values() {
            if node.key == graph.owner {
                continue;
            }
            if self.view_exists(&mut cache, &node.key) {
                continue;
            }

            let scope = node.key.scope.as_deref().unwrap_or("<default>");
            let kinds = summarize_relation_kinds(node);
            eprintln!(
                "[mui::rel] panel {panel_id} state {}: missing relation target '{}'<{}> scope={scope}. \
                 kinds={kinds}. Relation will remain inactive until the target panel is registered.",
                state.0,
                node.key.panel_uuid,
                node.key.payload_name,
            );
        }
    }

    fn view_exists(
        &self,
        cache: &mut HashMap<TypeId, Vec<PanelKey>>,
        key: &RelViewKey,
    ) -> bool {
        let entries = cache
            .entry(key.payload)
            .or_insert_with(|| registered_panel_keys(key.payload));
        entries.iter().any(|panel| {
            panel.panel_uuid == key.panel_uuid
                && key
                    .scope
                    .as_ref()
                    .map(|scope| *scope == panel.scope)
                    .unwrap_or(true)
        })
    }
}

fn summarize_relation_kinds(node: &RelNodeState) -> String {
    let mut parts = Vec::new();
    if !node.dependencies.is_empty() {
        parts.push("dep_view");
    }
    if !node.mutex_group.is_empty() {
        parts.push("mutex_view");
    }
    if !node.attached_fields.is_empty() {
        parts.push("attach");
    }
    if !node.container_links.is_empty() {
        parts.push("container_with");
    }
    if parts.is_empty() {
        "rule".to_string()
    } else {
        parts.join("/")
    }
}

fn encode_layout(spec: &RelContainerSpec) -> (u32, [f32; 2]) {
    match &spec.layout {
        RelLayoutKind::Free => (layout_flags::FREE, [0.0, 0.0]),
        RelLayoutKind::Horizontal {
            spacing,
            align_center,
        } => {
            let mut flags = layout_flags::HORIZONTAL;
            if *align_center {
                flags |= layout_flags::ALIGN_CENTER;
            }
            (flags, [*spacing, 0.0])
        }
        RelLayoutKind::Vertical {
            spacing,
            align_center,
        } => {
            let mut flags = layout_flags::VERTICAL;
            if *align_center {
                flags |= layout_flags::ALIGN_CENTER;
            }
            (flags, [0.0, *spacing])
        }
        RelLayoutKind::Grid { spacing, .. } => (layout_flags::GRID, *spacing),
        RelLayoutKind::Ring {
            radius,
            start_angle,
            clockwise,
        } => {
            let signed_radius = if *clockwise { *radius } else { -*radius };
            (layout_flags::RING, [signed_radius, *start_angle])
        }
        RelLayoutKind::Custom { .. } => (layout_flags::FREE, [0.0, 0.0]),
    }
}

fn encode_space(space: RelSpace) -> u32 {
    match space {
        RelSpace::Screen => layout_flags::SPACE_SCREEN,
        RelSpace::Parent => layout_flags::SPACE_PARENT,
        RelSpace::Local => layout_flags::SPACE_LOCAL,
    }
}

const TRANSITION_MODE_IMMEDIATE: u32 = 0;
const TRANSITION_MODE_TIMED: u32 = 1;

fn encode_transition(transition: &RelTransition) -> (u32, f32) {
    match transition {
        RelTransition::Immediate => (TRANSITION_MODE_IMMEDIATE, 0.0),
        RelTransition::Timed(duration) => (TRANSITION_MODE_TIMED, *duration),
    }
}

fn resolve_panel_id(key: &RelViewKey) -> Option<u32> {
    if key.panel_uuid.is_empty() {
        None
    } else {
        Some(panel_numeric_id(&key.panel_uuid))
    }
}

pub fn relation_registry() -> &'static Mutex<RelationRegistry> {
    static REGISTRY: OnceLock<Mutex<RelationRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(RelationRegistry::default()))
}

pub fn register_panel_relations(panel_id: u32, state: UiState, definition: RelGraphDefinition) {
    relation_registry()
        .lock()
        .unwrap()
        .register(panel_id, state, definition);
}

pub fn clear_panel_relations(panel_id: u32) {
    relation_registry().lock().unwrap().clear_panel(panel_id);
}

pub fn set_panel_active_state(panel_id: u32, state: UiState) {
    relation_registry()
        .lock()
        .unwrap()
        .set_active_state(panel_id, state);
}

pub fn inject_relation_work(items: Vec<RelationWorkItem>) {
    relation_registry().lock().unwrap().inject_manual(items);
}

pub fn active_panel_relations(panel_id: u32) -> Option<RelParsedGraph> {
    relation_registry()
        .lock()
        .ok()
        .and_then(|registry| {
            registry
                .panels
                .get(&panel_id)
                .and_then(|entry| entry.active_graph().cloned())
        })
}

pub mod layout_flags {
    pub const FREE: u32 = 0;
    pub const HORIZONTAL: u32 = 1;
    pub const VERTICAL: u32 = 2;
    pub const GRID: u32 = 3;
    pub const RING: u32 = 4;

    pub const SPACE_SCREEN: u32 = 1 << 8;
    pub const SPACE_PARENT: u32 = 1 << 9;
    pub const SPACE_LOCAL: u32 = 1 << 10;

    pub const ALIGN_CENTER: u32 = 1 << 12;
    pub const HAS_PERCENT: u32 = 1 << 13;
}
