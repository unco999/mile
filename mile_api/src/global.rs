use std::sync::{Arc, Mutex, OnceLock};

use mile_db::MileDb;
use mlua::Lua;

use crate::prelude::{EventBus, ImportRegistry, KeyedEventBus};

static GLOBAL_BUS: OnceLock<EventBus> = OnceLock::new();

static GLOBAL_KEY_BUS: OnceLock<KeyedEventBus> = OnceLock::new();

pub fn global_event_bus() -> &'static EventBus {
    GLOBAL_BUS.get_or_init(EventBus::new)
}

pub fn global_key_event_bus() -> &'static KeyedEventBus {
    GLOBAL_KEY_BUS.get_or_init(KeyedEventBus::new)
}

static GLOBAL_DB: OnceLock<MileDb> = OnceLock::new();

pub fn global_db() -> &'static MileDb {
    GLOBAL_DB.get_or_init(|| MileDb::in_memory().expect("你当前环境无法开启本地db数据库"))
}

static LUA_RUNTIME: OnceLock<Mutex<Arc<Lua>>> = OnceLock::new();

fn runtime_cell() -> &'static Mutex<Arc<Lua>> {
    LUA_RUNTIME.get_or_init(|| Mutex::new(Arc::new(Lua::new())))
}

pub fn get_lua_runtime() -> Arc<Lua> {
    runtime_cell().lock().expect("lua runtime poisoned").clone()
}

pub fn update_lua_runtime() -> Arc<Lua> {
    let mut guard = runtime_cell().lock().expect("lua runtime poisoned");
    let new_runtime = Arc::new(Lua::new());
    *guard = new_runtime.clone();
    new_runtime
}

static GLOBAL_WGSL_REGISTER: OnceLock<ImportRegistry> = OnceLock::new();

pub fn global_wgsl_register() -> &'static ImportRegistry {
    GLOBAL_WGSL_REGISTER.get_or_init(|| {
        let mut import = ImportRegistry::new();
        import.register_render_import("uv", 1, Box::new(|_| vec![0.0]));
        import.register_render_import("color", 2, Box::new(|_| vec![0.0]));
        import.register_compute_import("time", 1, Box::new(|_| vec![0.0]));
        import.register_render_import("pos", 4, Box::new(|_| vec![0.0]));
        import.register_render_import("instance_pos", 8, Box::new(|_| vec![0.0]));
        import.register_render_import("instance_size", 16, Box::new(|_| vec![0.0]));
        import.register_render_import("random", 32, Box::new(|_| vec![0.0]));
        import
    })
}
