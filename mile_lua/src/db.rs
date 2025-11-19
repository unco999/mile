use mlua::{Lua, Result as LuaResult, UserData, Value};
use mlua::prelude::LuaSerdeExt;
use once_cell::sync::OnceCell;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

static NEXT_DB_INDEX: AtomicU32 = AtomicU32::new(1);
static DB_REGISTRY: OnceCell<Mutex<HashMap<u32, JsonValue>>> = OnceCell::new();

fn registry() -> &'static Mutex<HashMap<u32, JsonValue>> {
    DB_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn store_json(json: JsonValue) -> u32 {
    let idx = NEXT_DB_INDEX.fetch_add(1, Ordering::Relaxed);
    registry().lock().unwrap().insert(idx, json);
    idx
}

pub fn load_json(idx: u32) -> Option<JsonValue> {
    registry().lock().unwrap().get(&idx).cloned()
}

#[derive(Clone, Debug)]
pub struct LuaTableDb {
    pub index: u32,
}

impl UserData for LuaTableDb {}

pub fn register_db_globals(lua: &Lua) -> LuaResult<()> {
    // db(value) -> userdata(idx), stores JSON snapshot
    let create = lua.create_function(|lua, value: Value| {
        let json = lua.from_value::<JsonValue>(value)?;
        let idx = store_json(json);
        lua.create_userdata(LuaTableDb { index: idx })
    })?;

    // db_get(idx) -> Lua value (table/userdata)
    let get = lua.create_function(|lua, idx: u32| {
        if let Some(json) = load_json(idx) {
            lua.to_value(&json)
        } else {
            Ok(Value::Nil)
        }
    })?;

    let globals = lua.globals();
    globals.set("db", create)?;
    globals.set("db_get", get)?;
    Ok(())
}
