use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
    sync::{Mutex, OnceLock},
};

use crate::{
    mui_prototype::{PanelKey, UiState, registered_panel_keys},
    mui_rel::{RelGraphDefinition, RelParsedGraph, RelNodeState, RelViewKey},
};

#[derive(Default)]
pub struct RelationRegistry {
    panels: HashMap<u32, PanelRelationEntry>,
    dirty: HashSet<u32>,
    manual: Vec<RelationWorkItem>,
}

#[derive(Default)]
struct PanelRelationEntry {
    states: HashMap<u32, RelParsedGraph>,
    active_state: Option<u32>,
}

#[derive(Clone)]
pub struct RelationWorkItem {
    pub panel_id: u32,
    pub graph: Option<RelParsedGraph>,
    pub layout_flags: u32,
    pub order: u32,
    pub total: u32,
    pub origin: [f32; 2],
    pub size: [f32; 2],
    pub slot: [f32; 2],
    pub spacing: [f32; 2],
}

impl Default for RelationWorkItem {
    fn default() -> Self {
        Self {
            panel_id: 0,
            graph: None,
            layout_flags: 0,
            order: 0,
            total: 0,
            origin: [0.0, 0.0],
            size: [0.0, 0.0],
            slot: [0.0, 0.0],
            spacing: [0.0, 0.0],
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
    }

    pub fn clear_panel(&mut self, panel_id: u32) {
        self.panels.remove(&panel_id);
        self.dirty.insert(panel_id);
    }

    pub fn set_active_state(&mut self, panel_id: u32, state: UiState) {
        if let Some(entry) = self.panels.get_mut(&panel_id) {
            if entry.active_state != Some(state.0) {
                entry.active_state = Some(state.0);
                self.dirty.insert(panel_id);
            }
        }
    }

    pub fn take_dirty(&mut self) -> Vec<RelationWorkItem> {
        let dirty_ids: Vec<u32> = self.dirty.drain().collect();
        let mut items: Vec<RelationWorkItem> = dirty_ids
            .into_iter()
            .map(|panel_id| {
                let graph = self.panels.get(&panel_id).and_then(|entry| {
                    entry
                        .active_state
                        .and_then(|state| entry.states.get(&state).cloned())
                });
                RelationWorkItem {
                    panel_id,
                    graph,
                    ..RelationWorkItem::default()
                }
            })
            .collect();
        if !self.manual.is_empty() {
            items.extend(self.manual.drain(..));
        }
        items
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

pub mod layout_flags {
    pub const FREE: u32 = 0;
    pub const HORIZONTAL: u32 = 1;
    pub const VERTICAL: u32 = 2;
}
