use std::collections::HashMap;
use std::sync::Mutex;

use std::sync::OnceLock;

fn registry() -> &'static Mutex<HashMap<u32, [f32; 2]>> {
    static REGISTRY: OnceLock<Mutex<HashMap<u32, [f32; 2]>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn set_panel_position(panel_id: u32, position: [f32; 2]) {
    let mut guard = registry().lock().unwrap();
    guard.insert(panel_id, position);
}

pub fn panel_position(panel_id: u32) -> Option<[f32; 2]> {
    let guard = registry().lock().unwrap();
    guard.get(&panel_id).copied()
}

pub fn clear() {
    registry().lock().unwrap().clear();
}
