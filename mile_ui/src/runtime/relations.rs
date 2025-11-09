use std::{
    collections::{HashMap, HashSet},
    sync::{Mutex, OnceLock},
};

use crate::{
    mui_prototype::UiState,
    mui_rel::{RelGraphDefinition, RelParsedGraph},
};

#[derive(Default)]
pub struct RelationRegistry {
    panels: HashMap<u32, PanelRelationEntry>,
    dirty: HashSet<u32>,
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
}

impl RelationRegistry {
    pub fn register(&mut self, panel_id: u32, state: UiState, definition: RelGraphDefinition) {
        let entry = self.panels.entry(panel_id).or_default();
        entry.states.insert(state.0, definition.parse());
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
        dirty_ids
            .into_iter()
            .map(|panel_id| {
                let graph = self.panels.get(&panel_id).and_then(|entry| {
                    entry
                        .active_state
                        .and_then(|state| entry.states.get(&state).cloned())
                });
                RelationWorkItem { panel_id, graph }
            })
            .collect()
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
