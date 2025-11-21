use mlua::prelude::LuaSerdeExt;
use mlua::{Lua, Result as LuaResult, UserData, Value};
use once_cell::sync::OnceCell;
use serde_json::{Value as JsonValue, json};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

static NEXT_DB_INDEX: AtomicU32 = AtomicU32::new(1);
static DB_REGISTRY: OnceCell<Mutex<HashMap<u32, JsonValue>>> = OnceCell::new();
static DB_REVISIONS: OnceCell<Mutex<HashMap<u32, u64>>> = OnceCell::new();

fn registry() -> &'static Mutex<HashMap<u32, JsonValue>> {
    DB_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn revisions() -> &'static Mutex<HashMap<u32, u64>> {
    DB_REVISIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn bump_revision(idx: u32) -> u64 {
    let mut guard = revisions().lock().unwrap();
    let entry = guard.entry(idx).or_insert(0);
    *entry += 1;
    *entry
}

fn current_revision(idx: u32) -> u64 {
    *revisions().lock().unwrap().get(&idx).unwrap_or(&0)
}

pub fn store_json(json: JsonValue) -> u32 {
    let idx = NEXT_DB_INDEX.fetch_add(1, Ordering::Relaxed);
    {
        registry().lock().unwrap().insert(idx, json);
    }
    {
        revisions().lock().unwrap().insert(idx, 0);
    }
    idx
}

pub fn update_json(idx: u32, json: JsonValue) {
    registry().lock().unwrap().insert(idx, json);
    bump_revision(idx);
}

pub fn load_json(idx: u32) -> Option<JsonValue> {
    registry().lock().unwrap().get(&idx).cloned()
}

pub fn clear_all() -> usize {
    NEXT_DB_INDEX.store(1, Ordering::Relaxed);
    let mut cleared = 0;
    if let Some(map) = DB_REGISTRY.get() {
        let mut guard = map.lock().unwrap();
        cleared = guard.len();
        guard.clear();
    }
    if let Some(map) = DB_REVISIONS.get() {
        map.lock().unwrap().clear();
    }
    cleared
}

#[derive(Clone, Debug)]
pub struct LuaTableDb {
    pub index: u32,
}

impl UserData for LuaTableDb {}

pub fn db_payload_with_revision(idx: u32) -> JsonValue {
    json!({
        "db_index": idx,
        "db_rev": current_revision(idx),
    })
}

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

    let clear = lua.create_function(|_, ()| Ok(clear_all() as u32))?;

    let globals = lua.globals();
    globals.set("db", create)?;
    globals.set("db_get", get)?;
    globals.set("db_clear_all", clear)?;
    Ok(())
}
