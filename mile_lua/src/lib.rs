mod db;
pub mod watch;

use crate::db::{
    LuaTableDb, db_payload_with_revision, load_json, register_db_globals, store_json, update_json,
};
use flume::TryRecvError;
use glam::{vec2, vec3, vec4};
use mile_api::prelude::{
    _ty::PanelId, KeyedEventStream, global_db, global_event_bus, global_key_event_bus,
};
use mile_font::{event::{RemoveRenderFont, ResetFontRuntime}, prelude::FontStyle};
use mile_gpu_dsl::gpu_ast_core::event::ResetKennel;
use mile_ui::{
    mui_prototype::{BorderStyle, EventFlow, Mui, PanelBinding, PanelKey, UiEventKind, UiState},
    mui_rel::{RelContainerSpec, RelLayoutKind, RelScrollAxis, RelSpace, apply_container_alias},
    runtime::entry::ResetUiRuntime,
    runtime::relations::clear_panel_relations,
};
use mlua::prelude::LuaSerdeExt;
use mlua::{
    AnyUserData, Error, Function, Lua, RegistryKey, Result as LuaResult, Table, UserData,
    UserDataMethods, Value, Variadic,
};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Value as JsonValue, json};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

type SharedLua = Arc<Mutex<Lua>>;
static LUA_PANEL_REGISTRY: OnceCell<Mutex<Vec<PanelKey>>> = OnceCell::new();
const DEFAULT_FONT_PATH: &str = "tf/STXIHEI.ttf";

fn panel_registry() -> &'static Mutex<Vec<PanelKey>> {
    LUA_PANEL_REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

fn track_panel_key(key: &PanelKey) {
    let mut registry = panel_registry().lock().unwrap();
    if registry.iter().any(|existing| existing == key) {
        return;
    }
    println!(
        "[lua_panel] register {} (id={}) scope={}",
        key.panel_uuid, key.panel_id, key.scope
    );
    registry.push(key.clone());
}

fn snapshot_panel_keys() -> Vec<PanelKey> {
    panel_registry().lock().unwrap().clone()
}

fn drain_panel_keys() -> Vec<PanelKey> {
    panel_registry().lock().unwrap().drain(..).collect()
}

// Generated Lua DB structs (build.rs executes Lua entry script; default: lua/main.lua)
#[cfg(has_out_dir)]
mod lua_registered_types {
    include!(concat!(env!("OUT_DIR"), "/lua_registered_types.rs"));
}
#[cfg(has_out_dir)]
pub use lua_registered_types::*;

// Fallback for editors/analysis that skip build.rs: provide empty module so rest of file still
// parses, but note that real builds must run the build script.
#[cfg(not(has_out_dir))]
mod lua_registered_types {}
#[cfg(not(has_out_dir))]
pub use lua_registered_types::*;

#[derive(Clone, Debug, PartialEq)]
pub struct LuaPayload(pub JsonValue);

#[derive(Clone, Debug)]
pub struct LuaStruct {
    pub type_name: String,
    pub json: JsonValue,
}

impl UserData for LuaStruct {}

#[derive(Clone)]
struct LuaKeyEventStream {
    key: String,
    stream: Arc<KeyedEventStream>,
}

#[derive(Clone)]
struct LuaMultiKeyEventStream {
    streams: Vec<LuaKeyEventStream>,
}

impl UserData for LuaKeyEventStream {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("try_recv", |lua, this, ()| match this.stream.try_recv() {
            Ok(event) => json_to_lua_value(lua, keyed_delivery_to_json(event, &this.key)),
            Err(TryRecvError::Empty) => Ok(Value::Nil),
            Err(TryRecvError::Disconnected) => {
                Err(mlua::Error::external("keyed event channel closed"))
            }
        });

        methods.add_method("drain", |lua, this, ()| {
            let table = lua.create_table()?;
            for (idx, event) in drain_keyed_stream(lua, this)?.into_iter().enumerate() {
                table.set(idx + 1, event)?;
            }
            Ok(table)
        });
    }
}

impl UserData for LuaMultiKeyEventStream {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        methods.add_method("drain", |lua, this, ()| {
            let table = lua.create_table()?;
            let mut idx = 1;
            for stream in &this.streams {
                for event in drain_keyed_stream(lua, stream)? {
                    table.set(idx, event)?;
                    idx += 1;
                }
            }
            Ok(table)
        });

        methods.add_method("try_recv", |lua, this, ()| {
            for stream in &this.streams {
                if let Ok(event) = stream.stream.try_recv() {
                    return json_to_lua_value(lua, keyed_delivery_to_json(event, &stream.key));
                }
            }
            Ok(Value::Nil)
        });
    }
}

fn inject_data_ty(mut json: JsonValue, ty: &str) -> JsonValue {
    if ty.is_empty() {
        return json;
    }
    match json {
        JsonValue::Object(mut map) => {
            map.entry("data_ty")
                .or_insert_with(|| JsonValue::String(ty.to_string()));
            JsonValue::Object(map)
        }
        other => {
            let mut map = serde_json::Map::new();
            map.insert("data_ty".into(), JsonValue::String(ty.to_string()));
            map.insert("value".into(), other);
            JsonValue::Object(map)
        }
    }
}

fn lua_value_to_json(lua: &Lua, value: Value) -> LuaResult<JsonValue> {
    match value {
        Value::Table(tbl) => {
            if let Ok(Some(idx)) = tbl.get::<Option<u32>>("db_index") {
                return Ok(db_payload_with_revision(idx));
            }
            lua.from_value(Value::Table(tbl))
        }
        Value::UserData(ud) => {
            if let Ok(ls) = ud.borrow::<LuaStruct>() {
                let ty = ls.type_name.clone();
                Ok(inject_data_ty(ls.json.clone(), &ty))
            } else if let Ok(db_ref) = ud.borrow::<LuaTableDb>() {
                Ok(db_payload_with_revision(db_ref.index))
            } else {
                lua.from_value(Value::UserData(ud))
            }
        }
        other => lua.from_value(other),
    }
}

fn ensure_db_payload(lua: &Lua, value: Value) -> LuaResult<JsonValue> {
    let json = lua_value_to_json(lua, value)?;
    let has_index = json
        .as_object()
        .and_then(|map| map.get("db_index"))
        .and_then(|idx| idx.as_u64())
        .is_some();
    if has_index {
        Ok(json)
    } else {
        let idx = store_json(json);
        Ok(db_payload_with_revision(idx))
    }
}

fn json_to_lua_value(lua: &Lua, json: JsonValue) -> LuaResult<Value> {
    if let JsonValue::Object(map) = &json {
        if let Some(idx) = map.get("db_index").and_then(|v| v.as_u64()) {
            let idx_u32 = idx as u32;
            if let Some(stored) = load_json(idx_u32) {
                let value = lua.to_value(&stored)?;
                if let Value::Table(table) = value {
                    table.set("db_index", idx_u32)?;
                    return Ok(Value::Table(table));
                }
                return Ok(value);
            }
            let tbl = lua.create_table()?;
            tbl.set("db_index", idx_u32)?;
            return Ok(Value::Table(tbl));
        }
    }
    lua.to_value(&json)
}

fn keyed_delivery_to_json(
    delivery: mile_api::event_bus::KeyedEventDelivery,
    key: &str,
) -> JsonValue {
    let payload = delivery
        .into_owned()
        .unwrap_or_else(|shared| (*shared).clone());

    match payload {
        JsonValue::Object(mut map) => {
            map.entry("key")
                .or_insert_with(|| JsonValue::String(key.to_string()));
            JsonValue::Object(map)
        }
        other => {
            let mut map = serde_json::Map::new();
            map.insert("key".into(), JsonValue::String(key.to_string()));
            map.insert("value".into(), other);
            JsonValue::Object(map)
        }
    }
}

fn drain_keyed_stream(lua: &Lua, stream: &LuaKeyEventStream) -> LuaResult<Vec<Value>> {
    let events = stream.stream.drain();
    events
        .into_iter()
        .map(|event| json_to_lua_value(lua, keyed_delivery_to_json(event, &stream.key)))
        .collect()
}

fn clear_db_snapshots() -> u32 {
    crate::db::clear_all() as u32
}

fn clear_font_for_panels(panels: &[PanelKey]) {
    if panels.is_empty() {
        return;
    }
    let bus = global_event_bus();
    for key in panels {
        bus.publish(RemoveRenderFont {
            parent: PanelId(key.panel_id),
        });
    }
}

fn clear_ui_panels() -> LuaResult<u32> {
    let panels = drain_panel_keys();
    if panels.is_empty() {
        return Ok(0);
    }
    let table = global_db()
        .bind_table::<PanelBinding<LuaPayload>>()
        .map_err(|err| mlua::Error::external(format!("panel table access failed: {err}")))?;
    for key in &panels {
        if let Err(err) = table.remove(key) {
            eprintln!(
                "[lua_reset] failed to remove panel '{}': {err}",
                key.panel_uuid
            );
        }
        clear_panel_relations(key.panel_id);
    }
    clear_font_for_panels(&panels);
    global_event_bus().publish(ResetUiRuntime);
    Ok(panels.len() as u32)
}

fn clear_font_runtime_state() -> u32 {
    let panels = snapshot_panel_keys();
    clear_font_for_panels(&panels);
    global_event_bus().publish(ResetFontRuntime);
    panels.len() as u32
}

fn clear_kennel_state() -> u32 {
    global_event_bus().publish(ResetKennel);
    0
}

fn register_runtime_reset(lua: &Lua) -> LuaResult<()> {
    let reset = lua.create_table()?;
    reset.set("db", lua.create_function(|_, ()| Ok(clear_db_snapshots()))?)?;
    reset.set("ui", lua.create_function(|_, ()| clear_ui_panels())?)?;
    reset.set(
        "font",
        lua.create_function(|_, ()| Ok(clear_font_runtime_state()))?,
    )?;
    reset.set(
        "kennel",
        lua.create_function(|_, ()| Ok(clear_kennel_state()))?,
    )?;
    let globals = lua.globals();
    globals.set("mile_runtime_reset", reset)?;
    Ok(())
}

fn register_key_event_bus(lua: &Lua) -> LuaResult<()> {
    let emit = lua.create_function(|lua, (key, payload): (String, Option<Value>)| {
        let json = match payload {
            Some(value) => lua_value_to_json(lua, value)?,
            None => JsonValue::Null,
        };
        println!(
            "[lua][event_bus] emit key={key}, payload={payload}",
            payload = json
        );
        global_key_event_bus().publish(key, json);
        Ok(())
    })?;

    let subscribe = lua.create_function(|lua, keys: Variadic<String>| {
        if keys.is_empty() {
            return Err(mlua::Error::external(
                "mile_event.on requires at least one key",
            ));
        }

        let streams: Vec<LuaKeyEventStream> = keys
            .into_iter()
            .map(|key| LuaKeyEventStream {
                key: key.clone(),
                stream: Arc::new(global_key_event_bus().subscribe(key)),
            })
            .collect();

        if streams.len() == 1 {
            lua.create_userdata(streams.into_iter().next().unwrap())
        } else {
            lua.create_userdata(LuaMultiKeyEventStream { streams })
        }
    })?;

    let events = lua.create_table()?;
    events.set("emit", emit)?;
    events.set("on", subscribe)?;
    lua.globals().set("mile_event", events)?;
    Ok(())
}

impl Default for LuaPayload {
    fn default() -> Self {
        Self(JsonValue::Null)
    }
}

impl Serialize for LuaPayload {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let text = serde_json::to_string(&self.0).map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(&text)
    }
}

impl<'de> Deserialize<'de> for LuaPayload {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text: String = Deserialize::deserialize(deserializer)?;
        let value = serde_json::from_str(&text).map_err(serde::de::Error::custom)?;
        Ok(LuaPayload(value))
    }
}

#[derive(Clone, Default)]
struct StateSpec {
    position: Option<[f32; 2]>,
    size: Option<[f32; 2]>,
    rotation: Option<[f32; 3]>,
    scale: Option<[f32; 3]>,
    color: Option<[f32; 4]>,
    texture: Option<String>,
    size_with_image: bool,
    trigger_mouse_pos: bool,
    visible: Option<bool>,
    container: Option<RelContainerSpec>,
    z_index: Option<i32>,
    border: Option<BorderStyle>,
    texts: Vec<LuaTextSpec>,
}

#[derive(Clone)]
struct LuaTextSpec {
    text: String,
    font_path: Option<String>,
    font_size: u32,
    color: [f32; 4],
    panel_size: [f32; 2],
    weight: u32,
    line_height: u32,
    first_weight: f32,
    text_align: u32,
}

impl Default for LuaTextSpec {
    fn default() -> Self {
        Self {
            text: String::new(),
            font_path: None,
            font_size: 16,
            color: [1.0; 4],
            panel_size: [1.0; 2],
            weight: 400,
            line_height: 0,
            first_weight: 0.0,
            text_align: 0,
        }
    }
}

impl LuaTextSpec {
    fn to_font_style(&self) -> FontStyle {
        FontStyle {
            font_size: self.font_size,
            font_file_path: Arc::from(self
                .font_path
                .clone()
                .unwrap_or_else(|| DEFAULT_FONT_PATH.to_string())),
            font_color: self.color,
            font_weight: self.weight,
            font_line_height: self.line_height,
            first_weight: self.first_weight,
            panel_size: self.panel_size,
            text_align: self.text_align.into(),
        }
    }
}

#[derive(Clone)]
struct StateEntry {
    spec: StateSpec,
    handlers: HashMap<UiEventKind, Arc<RegistryKey>>,
    data_handlers: HashMap<u32, Arc<RegistryKey>>,
    container_links: Vec<String>,
    container_aliases: Vec<(String, String)>,
}

impl Default for StateEntry {
    fn default() -> Self {
        Self {
            spec: StateSpec::default(),
            handlers: HashMap::new(),
            data_handlers: HashMap::new(),
            container_links: Vec::new(),
            container_aliases: Vec::new(),
        }
    }
}

#[derive(Clone)]
struct LuaMuiBuilder {
    lua: SharedLua,
    id: String,
    payload: LuaPayload,
    states: HashMap<u32, StateEntry>,
    current_state: u32,
    default_state: Option<u32>,
}

impl LuaMuiBuilder {
    fn new(lua: SharedLua, id: String, payload: LuaPayload) -> Self {
        let mut states = HashMap::new();
        states.insert(0, StateEntry::default());
        Self {
            lua,
            id,
            payload,
            states,
            current_state: 0,
            default_state: None,
        }
    }

    fn current_entry_mut(&mut self) -> &mut StateEntry {
        self.states
            .entry(self.current_state)
            .or_insert_with(StateEntry::default)
    }

    fn build_internal(&self) -> LuaResult<()> {
        let payload_for_init = self.payload.clone();
        let lua = self.lua.clone();
        let default_state = self.default_state.unwrap_or(0);

        let mut state_ids: Vec<u32> = self.states.keys().copied().collect();
        state_ids.sort_unstable();

        let mut builder = Mui::<LuaPayload>::new(&self.id)
            .map_err(|err| mlua::Error::external(format!("Mui.new failed: {err}")))?;
        builder = builder.default_state(UiState(default_state));

        for state_id in state_ids {
            let entry = self
                .states
                .get(&state_id)
                .cloned()
                .unwrap_or_else(StateEntry::default);
            let state_is_default = state_id == default_state;
            let lua = lua.clone();
            let payload = payload_for_init.clone();
            builder = builder.state(UiState(state_id), move |state| {
                let mut s = state;
                if let Some(pos) = entry.spec.position {
                    s = s.position(vec2(pos[0], pos[1]));
                }
                if let Some(size) = entry.spec.size {
                    s = s.size(vec2(size[0], size[1]));
                }
                if let Some(rotation) = entry.spec.rotation {
                    s = s.rotation(vec3(rotation[0], rotation[1], rotation[2]));
                }
                if let Some(scale) = entry.spec.scale {
                    s = s.scale(vec3(scale[0], scale[1], scale[2]));
                }
                if let Some(color) = entry.spec.color {
                    s = s.color(vec4(color[0], color[1], color[2], color[3]));
                }
                if let Some(border) = entry.spec.border.as_ref() {
                    s = s.border(border.clone());
                }
                if let Some(tex) = entry.spec.texture.as_ref() {
                    s = s.texture(tex);
                }
                if entry.spec.size_with_image {
                    s = s.size_with_image();
                }
                if entry.spec.trigger_mouse_pos {
                    s = s.with_trigger_mouse_pos();
                }
                if let Some(vis) = entry.spec.visible {
                    s = s.visible(vis);
                }
                if let Some(z) = entry.spec.z_index {
                    s = s.z_index(z);
                }
                if let Some(container) = entry.spec.container.as_ref() {
                    let spec = container.clone();
                    println!("有个单位执行了容器设定 {:?}",spec);
                    s.rel().container_self(move |target| *target = spec);
                }
                for (alias, panel_uuid) in entry.container_aliases.iter() {
                    if !apply_container_alias(s.rel(), alias, panel_uuid) {
                        eprintln!(
                            "Lua container_with_alias failed: alias='{}', panel='{}'",
                            alias, panel_uuid
                        );
                    }
                }
                for panel_uuid in entry.container_links.iter() {
                    s.rel().container_with::<LuaPayload>(panel_uuid);
                }

                let mut events = s.events();
                if state_is_default {
                    let payload = payload.clone();
                    events = events.on_event(UiEventKind::Init, move |flow| {
                        *flow.payload() = payload.clone();
                    });
                }
                if !entry.spec.texts.is_empty() {
                    let texts = entry.spec.texts.clone();
                    events = events.on_event(UiEventKind::Init, move |flow| {
                        flow.clear_texts();
                        for spec in &texts {
                            flow.text(&spec.text, spec.to_font_style());
                        }
                    });
                }

                for (kind, key) in entry.handlers.iter() {
                    let lua = lua.clone();
                    let key = key.clone();
                    events = events.on_event(*kind, move |flow| {
                        if let Err(err) =
                            with_lua(&lua, |lua_ctx| dispatch_lua_event(lua_ctx, &key, flow))
                        {
                            eprintln!("lua on_event error: {err}");
                        }
                    });
                }

                for (&source_db, key) in entry.data_handlers.iter() {
                    let lua = lua.clone();
                    let key = key.clone();
                    events =
                        events.on_data_change::<LuaPayload, _>(None, move |src_payload, flow| {
                            if payload_db_index(src_payload) != Some(source_db) {
                                return;
                            }
                            let call_result = with_lua(&lua, |lua_ctx| {
                                dispatch_lua_on_data(
                                    lua_ctx,
                                    &key,
                                    Some(source_db),
                                    src_payload.clone(),
                                    flow,
                                )
                            });
                            if let Err(err) = call_result {
                                eprintln!("lua on_target_data error: {err}");
                            }
                        });
                }

                events.finish()
            });
        }

        let handle = builder
            .build()
            .map_err(|err| mlua::Error::external(format!("Mui.build failed: {err}")))?;
        track_panel_key(handle.key());

        Ok(())
    }
}

fn with_lua<R, F>(lua: &SharedLua, f: F) -> LuaResult<R>
where
    F: FnOnce(&Lua) -> LuaResult<R>,
{
    let guard = lua
        .lock()
        .map_err(|_| mlua::Error::external("lua runtime poisoned"))?;
    f(&*guard)
}

fn dispatch_lua_event(
    lua: &Lua,
    key: &RegistryKey,
    flow: &mut EventFlow<'_, LuaPayload>,
) -> LuaResult<()> {
    let panel_id = flow.args().panel_key.panel_uuid.clone();
    let state_id = flow.state().0;
    let event_name = format!("{:#?}", flow.args().event);
    let payload = flow.payload_ref().clone();
    let drag_source_panel = flow
        .drag_source_panel()
        .map(|panel| panel.panel_uuid.clone());

    let func: Function = lua.registry_value(key)?;
    let tbl = lua.create_table()?;
    tbl.set("panel_id", panel_id)?;
    tbl.set("state", state_id)?;
    tbl.set("event", event_name)?;
    tbl.set("payload", materialize_payload_value(lua, &payload)?)?;
    if let Some(drag_payload) = flow.drag_payload::<LuaPayload>() {
        tbl.set(
            "drag_payload",
            materialize_payload_value(lua, drag_payload)?,
        )?;
    }

    let ret: Option<Value> = func.call(tbl.clone())?;
    let lua_ref = lua;
    if let Some(Value::Table(new_tbl)) = ret {
        apply_flow_directives(lua_ref, &new_tbl, flow)?;
        return Ok(())
    }
    apply_flow_directives(lua_ref, &tbl, flow)?;
    Ok(())
}

fn dispatch_lua_on_data(
    lua: &Lua,
    key: &RegistryKey,
    source_db_index: Option<u32>,
    source_payload: LuaPayload,
    flow: &mut EventFlow<'_, LuaPayload>,
) -> LuaResult<()> {
    let func: Function = lua.registry_value(key)?;
    let tbl = lua.create_table()?;
    tbl.set("panel_id", flow.args().panel_key.panel_uuid.clone())?;
    tbl.set("state", flow.state().0)?;
    tbl.set("event", "on_target_data")?;
    tbl.set(
        "payload",
        materialize_payload_value(lua, flow.payload_ref())?,
    )?;
    if let Some(idx) = source_db_index {
        tbl.set("source_db_index", idx)?;
    }
    tbl.set(
        "source_payload",
        materialize_payload_value(lua, &source_payload)?,
    )?;

    let ret: Option<Value> = func.call(tbl.clone())?;
    let lua_ref = lua;
    if let Some(Value::Table(new_tbl)) = ret {
        apply_flow_directives(lua_ref, &new_tbl, flow)?;
    }
    apply_flow_directives(lua_ref, &tbl, flow)?;
    Ok(())
}

fn apply_flow_directives(
    lua: &Lua,
    table: &Table,
    flow: &mut EventFlow<'_, LuaPayload>,
) -> LuaResult<()> {
    if let Ok(value) = table.get::<Value>("payload") {
        match value {
            Value::Table(tbl) => {
                if let Some((idx, mutated)) = commit_db_table(lua, &tbl)? {
                    *flow.payload() = LuaPayload(db_payload_with_revision(idx));
                    if mutated {
                        flow.mark_changed();
                    }
                } else {
                    let new_payload = lua_value_to_json(lua, Value::Table(tbl.clone()))?;
                    *flow.payload() = LuaPayload(new_payload);
                }
            }
            other => {
                let new_payload = lua_value_to_json(lua, other)?;
                *flow.payload() = LuaPayload(new_payload);
            }
        }
    }
    if let Ok(value) = table.get::<Value>("drag_payload") {
        if !matches!(value, Value::Nil) {
            let payload_json = ensure_db_payload(lua, value)?;
            flow.set_drag_payload(LuaPayload(payload_json));
        }
    }
    if let Ok(value) = table.get::<Value>("text") {
        if !matches!(value, Value::Nil) {
            flow.clear_texts();
            apply_text_from_lua(flow, &value)?;
            flow.mark_changed();
        }
    }
    if let Ok(Some(state)) = table.get::<Option<u32>>("state") {
        let state = UiState(state);
        if flow.state() != state {
            flow.set_state(state);
        }
    }
    if let Ok(Some(next_state)) = table.get::<Option<u32>>("next_state") {
        flow.set_state(UiState(next_state));
    }
    if let Ok(Some(drag_state)) = table.get::<Option<u32>>("drag_source_state") {
        if flow.set_drag_source_state(UiState(drag_state)) {
            flow.mark_changed();
        }
    }
    Ok(())
}

fn commit_db_table(lua: &Lua, table: &Table) -> LuaResult<Option<(u32, bool)>> {
    let Some(idx) = table.get::<Option<u32>>("db_index")? else {
        return Ok(None);
    };
    let Some(JsonValue::Object(mut stored)) = load_json(idx) else {
        return Ok(Some((idx, false)));
    };
    let mut mutated = false;
    for pair in table.clone().pairs::<Value, Value>() {
        let (key, value) = pair?;
        let key_str = match key {
            Value::String(s) => s.to_string_lossy().to_string(),
            Value::Integer(i) => i.to_string(),
            _ => continue,
        };
        if key_str == "db_index" {
            continue;
        }
        if !stored.contains_key(&key_str) {
            continue;
        }
        if matches!(value, Value::Nil) {
            if stored.remove(&key_str).is_some() {
                mutated = true;
            }
        } else {
            let json_value = lua_value_to_json(lua, value)?;
            if stored.get(&key_str) != Some(&json_value) {
                stored.insert(key_str, json_value);
                mutated = true;
            }
        }
    }
    if mutated {
        update_json(idx, JsonValue::Object(stored));
    }
    Ok(Some((idx, mutated)))
}

fn format_lua_value(
    lua: &Lua,
    value: Value,
    depth: usize,
    visited: &mut HashSet<usize>,
) -> LuaResult<String> {
    const MAX_RECURSION_DEPTH: usize = 8;

    match value {
        Value::Nil => Ok("nil".to_string()),
        Value::Boolean(b) => Ok(b.to_string()),
        Value::Integer(i) => Ok(i.to_string()),
        Value::Number(n) => Ok(n.to_string()),
        Value::String(s) => Ok(s.to_string_lossy().to_string()),
        Value::Table(table) => {
            if depth >= MAX_RECURSION_DEPTH {
                return Ok("<max depth reached>".to_string());
            }

            let ptr = table.to_pointer() as usize;
            if !visited.insert(ptr) {
                return Ok("<recursion>".to_string());
            }

            let indent = "  ".repeat(depth + 1);
            let mut fields = Vec::new();
            for pair in table.pairs::<Value, Value>() {
                let (key, value) = pair?;
                let key = format_lua_value(lua, key, depth + 1, visited)?;
                let value = format_lua_value(lua, value, depth + 1, visited)?;
                fields.push(format!("{indent}{key} = {value}"));
            }
            visited.remove(&ptr);

            let closing_indent = "  ".repeat(depth);
            if fields.is_empty() {
                Ok("{}".to_string())
            } else {
                Ok(format!("{{\n{}\n{closing_indent}}}", fields.join(",\n")))
            }
        }
        other => {
            if let Some(text) = lua.coerce_string(other.clone())? {
                Ok(text.to_string_lossy().to_string())
            } else {
                Ok(format!("{other:?}"))
            }
        }
    }
}

fn payload_db_index(payload: &LuaPayload) -> Option<u32> {
    if let JsonValue::Object(map) = &payload.0 {
        map.get("db_index")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
    } else {
        None
    }
}

fn materialize_payload_value(lua: &Lua, payload: &LuaPayload) -> LuaResult<Value> {
    if let Some(idx) = payload_db_index(payload) {
        if let Some(stored) = load_json(idx) {
            if let JsonValue::Object(map) = stored.clone() {
                let value = lua.to_value(&JsonValue::Object(map.clone()))?;
                if let Value::Table(table) = value {
                    if table.get::<Option<u32>>("db_index")?.is_none() {
                        table.set("db_index", idx)?;
                    }
                    attach_db_proxy_metatable(lua, &table, idx, map)?;
                    return Ok(Value::Table(table));
                }
            }
            return Ok(lua.to_value(&stored)?);
        }
    }
    json_to_lua_value(lua, payload.0.clone())
}

fn attach_db_proxy_metatable(
    lua: &Lua,
    table: &Table,
    idx: u32,
    keys_map: serde_json::Map<String, JsonValue>,
) -> LuaResult<()> {
    use std::collections::HashSet;
    let allowed: HashSet<String> = keys_map.keys().cloned().collect();
    if allowed.is_empty() {
        return Ok(());
    }
    let allowed_arc = Arc::new(allowed);
    let setter = {
        let allowed = allowed_arc.clone();
        lua.create_function(move |lua_ctx, (tbl, key, value): (Table, Value, Value)| {
            let key_clone = key.clone();
            let value_clone = value.clone();
            tbl.raw_set(key_clone, value)?;
            let key_str = match key {
                Value::String(s) => s.to_string_lossy().to_string(),
                Value::Integer(i) => i.to_string(),
                _ => return Ok(()),
            };
            if !allowed.contains(&key_str) {
                return Ok(());
            }
            let mut current = match load_json(idx) {
                Some(JsonValue::Object(obj)) => obj,
                _ => serde_json::Map::new(),
            };
            if matches!(value_clone, Value::Nil) {
                current.remove(&key_str);
            } else {
                let json_value = lua_value_to_json(lua_ctx, value_clone)?;
                current.insert(key_str, json_value);
            }
            update_json(idx, JsonValue::Object(current));
            Ok(())
        })?
    };
    let mt = match table.metatable() {
        Some(mt) => mt,
        None => lua.create_table()?,
    };
    mt.set("__newindex", setter)?;
    table.set_metatable(Some(mt));
    Ok(())
}

fn panel_uuid_from_db_index(idx: u32) -> String {
    format!("lua_db_panel_{}", idx)
}

fn lua_value_to_panel_uuid(lua: &Lua, value: Value) -> LuaResult<String> {
    match value {
        Value::String(s) => Ok(s.to_string_lossy().to_string()),
        other => {
            if let Some(idx) = extract_db_index_from_value(lua, other)? {
                Ok(panel_uuid_from_db_index(idx))
            } else {
                Err(mlua::Error::external(
                    "container_with expects userdata/table with db_index or a panel uuid string",
                ))
            }
        }
    }
}

fn parse_event_kind(name: &str) -> Option<UiEventKind> {
    let normalized = name.trim().to_lowercase().replace('-', "_");
    match normalized.as_str() {
        "init" => Some(UiEventKind::Init),
        "click" => Some(UiEventKind::Click),
        "drag" => Some(UiEventKind::Drag),
        "drag_start" | "source_drag_start" => Some(UiEventKind::SourceDragStart),
        "drag_over" | "source_drag_over" => Some(UiEventKind::SourceDragOver),
        "drag_leave" | "source_drag_leave" => Some(UiEventKind::SourceDragLeave),
        "drag_drop" | "source_drag_drop" => Some(UiEventKind::SourceDragDrop),
        "target_drag_enter" | "drag_enter" => Some(UiEventKind::TargetDragEnter),
        "target_drag_over" => Some(UiEventKind::TargetDragOver),
        "target_drag_leave" | "drag_exit" => Some(UiEventKind::TargetDragLeave),
        "target_drag_drop" => Some(UiEventKind::TargetDragDrop),
        "hover" => Some(UiEventKind::Hover),
        "out" | "leave" => Some(UiEventKind::Out),
        _ => None,
    }
}

fn extract_db_index_from_value(lua: &Lua, value: Value) -> LuaResult<Option<u32>> {
    match value {
        Value::Nil => Ok(None),
        Value::UserData(ud) => {
            if let Ok(db_ref) = ud.borrow::<LuaTableDb>() {
                Ok(Some(db_ref.index))
            } else {
                Err(mlua::Error::external(
                    "expected db userdata for on_target_data",
                ))
            }
        }
        Value::Table(tbl) => {
            let idx: Option<u32> = tbl.get("db_index").ok().flatten();
            Ok(idx)
        }
        Value::Integer(i) => Ok(Some(i as u32)),
        Value::Number(n) => Ok(Some(n as u32)),
        other => Err(mlua::Error::external(format!(
            "expected db userdata or table, got {}",
            other.type_name()
        ))),
    }
}

fn apply_text_from_lua(flow: &mut EventFlow<'_, LuaPayload>, value: &Value) -> LuaResult<()> {
    let table = match value {
        Value::Table(t) => t.clone(),
        other => {
            return Err(mlua::Error::external(format!(
                "flow:text expects table, got {}",
                other.type_name()
            )));
        }
    };

    // 支持字段：text, font_path, font_size, color (array[4]), weight, line_height
    let text: String = table
        .get("text")
        .map_err(|_| mlua::Error::external("flow.text requires field 'text'"))?;
    let font_path: Option<String> = table.get("font_path").ok();
    let font_size: u32 = table.get("font_size").unwrap_or(16);
    let color: Option<Vec<f32>> = table.get("color").ok();
    let panel_size: Option<Vec<f32>> = table.get("panel_size").ok();
    let weight: u32 = table.get("weight").unwrap_or(400);
    let line_height: u32 = table.get("line_height").unwrap_or(0);
    let first_weight:f32 =  table.get("first_weight").unwrap_or(0.0);
    let text_align:u32 =  table.get("text_align").unwrap_or(0);
    let final_color = if let Some(c) = color {
        let mut arr = [1.0, 1.0, 1.0, 1.0];
        for (i, v) in c.iter().copied().take(4).enumerate() {
            arr[i] = v;
        }
        arr
    } else {
        [1.0, 1.0, 1.0, 1.0]
    };

    let panel_size = if let Some(c) = panel_size {
        let mut arr = [1.0, 1.0];
        for (i, v) in c.iter().copied().take(2).enumerate() {
            arr[i] = v;
        }
        arr
    } else {
        [1.0, 1.0]
    };

    let path = font_path
        .map(|p| p.into())
        .unwrap_or_else(|| DEFAULT_FONT_PATH.into());

    let style = FontStyle{
        font_size,
        font_file_path:path,
        font_color: final_color,
        font_weight:weight,
        font_line_height: line_height,
        first_weight,
        panel_size,
        text_align:text_align.into(),
    };

    flow.clear_texts();
    flow.text(&text, style);
    Ok(())
}

fn parse_builder_text_spec(lua: &Lua, value: Value) -> LuaResult<LuaTextSpec> {
    match value {
        Value::Nil => Err(mlua::Error::external("text expects string or table")),
        Value::String(s) => Ok(LuaTextSpec {
            text: s.to_string_lossy().to_string(),
            ..LuaTextSpec::default()
        }),
        Value::Table(tbl) => parse_builder_text_spec_table(&tbl),
        other => Err(mlua::Error::external(format!(
            "text expects string or table, got {}",
            other.type_name()
        ))),
    }
}

fn parse_builder_text_spec_table(tbl: &Table) -> LuaResult<LuaTextSpec> {
    let text: Option<String> = tbl.get::<Option<String>>("text")?;
    let mut spec = LuaTextSpec::default();
    spec.text = text.ok_or_else(|| mlua::Error::external("text table requires field 'text'"))?;
    spec.font_path = tbl.get::<Option<String>>("font_path")?;
    spec.font_size = tbl.get::<Option<u32>>("font_size")?.unwrap_or(16);
    spec.weight = tbl.get::<Option<u32>>("weight")?.unwrap_or(400);
    spec.line_height = tbl.get::<Option<u32>>("line_height")?.unwrap_or(0);
    spec.first_weight = tbl.get::<Option<f32>>("first_weight")?.unwrap_or(0.0);
    spec.text_align = tbl.get::< Option<u32>>("text_align")?.unwrap_or(0);
    if let Some(color) = tbl.get::<Option<Vec<f32>>>("color")? {
        for (idx, value) in color.into_iter().take(4).enumerate() {
            spec.color[idx] = value;
        }
    }
    if let Some(panel_size) = tbl.get::<Option<Vec<f32>>>("panel_size")? {
        for (idx, value) in panel_size.into_iter().take(2).enumerate() {
            spec.panel_size[idx] = value;
        }
    }
    Ok(spec)
}

impl UserData for LuaMuiBuilder {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        // 设置位置
        methods.add_method_mut("position", |lua, this, (x, y): (f32, f32)| {
            this.current_entry_mut().spec.position = Some([x, y]);
            lua.create_userdata(this.clone())
        });

        // 设置尺寸
        methods.add_method_mut("size", |lua, this, (w, h): (f32, f32)| {
            this.current_entry_mut().spec.size = Some([w, h]);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("rotation", |lua, this, (x, y, z): (f32, f32, f32)| {
            this.current_entry_mut().spec.rotation = Some([x, y, z]);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("scale", |lua, this, (x, y, z): (f32, f32, f32)| {
            this.current_entry_mut().spec.scale = Some([x, y, z]);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("color", |lua, this, (r, g, b, a): (f32, f32, f32, f32)| {
            this.current_entry_mut().spec.color = Some([r, g, b, a]);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("text", |lua, this, value: Value| {
            let spec = parse_builder_text_spec(lua, value)?;
            this.current_entry_mut().spec.texts.push(spec);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("border", |lua, this, value: Value| {
            let border = match value {
                Value::Nil => None,
                Value::Table(tbl) => Some(parse_border_style(&tbl)?),
                other => {
                    return Err(mlua::Error::external(format!(
                        "border expects table or nil, got {}",
                        other.type_name()
                    )));
                }
            };
            this.current_entry_mut().spec.border = border;
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("texture", |lua, this, path: String| {
            this.current_entry_mut().spec.texture = Some(path);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("size_with_image", |lua, this, ()| {
            this.current_entry_mut().spec.size_with_image = true;
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("with_trigger_mouse_pos", |lua, this, ()| {
            this.current_entry_mut().spec.trigger_mouse_pos = true;
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("visible", |lua, this, visible: bool| {
            this.current_entry_mut().spec.visible = Some(visible);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("z_index", |lua, this, z: i32| {
            this.current_entry_mut().spec.z_index = Some(z);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("container", |lua, this, value: Value| {
            let new_spec = match value {
                Value::Nil => None,
                Value::Table(tbl) => Some(parse_container_spec(&tbl)?),
                other => {
                    return Err(mlua::Error::external(format!(
                        "container expects table or nil, got {}",
                        other.type_name()
                    )));
                }
            };
            this.current_entry_mut().spec.container = new_spec;
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("container_with", |lua, this, target: Value| {
            let panel_uuid = lua_value_to_panel_uuid(lua, target)?;
            this.current_entry_mut().container_links.push(panel_uuid);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut(
            "container_with_alias",
            |lua, this, (alias, target): (String, Value)| {
                let panel_uuid = lua_value_to_panel_uuid(lua, target)?;
                this.current_entry_mut()
                    .container_aliases
                    .push((alias, panel_uuid));
                lua.create_userdata(this.clone())
            },
        );

        // 事件回调：目前只支持 click（可按需扩展）
        methods.add_method_mut("on_event", |lua: &Lua, this, (name, func): (String, Function)| {
            let kind = parse_event_kind(&name).ok_or_else(|| {
                mlua::Error::external(format!(
                    "unsupported event '{}'; expected click/drag/source_/target_ drag variants/hover/out",
                    name
                ))
            })?;
            let key = Arc::new(lua.create_registry_value(func)?);
            this.current_entry_mut().handlers.insert(kind, key);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("state", |lua, this, state_id: u32| {
            this.current_state = state_id;
            this.states
                .entry(state_id)
                .or_insert_with(StateEntry::default);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("default_state", |lua, this, state_id: u32| {
            this.default_state = Some(state_id);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut(
            "on_target_data",
            |lua, this, (source, func): (Value, Function)| {
                let source_db = match extract_db_index_from_value(lua, source)? {
                    Some(idx) => idx,
                    None => {
                        return Err(mlua::Error::external(
                            "on_target_data expects db userdata or table with db_index",
                        ));
                    }
                };
                let key = Arc::new(lua.create_registry_value(func)?);
                this.current_entry_mut()
                    .data_handlers
                    .insert(source_db, key);
                lua.create_userdata(this.clone())
            },
        );

        // 构建面板
        methods.add_method("build", |_lua, this, ()| {
            this.build_internal()?;
            Ok(())
        });
    }
}

/// 向 Lua 注册全局 `Mui.new` 接口（最小实现：id + data + position/size + click 事件）
pub fn register_lua_api(lua: &Lua) -> LuaResult<()> {
    let lua_shared = Arc::new(Mutex::new(lua.clone()));

    register_db_globals(lua)?;
    register_runtime_reset(lua)?;
    register_key_event_bus(lua)?;

    {
        let make_struct = lua.create_function(|lua, (ty, value): (String, Value)| {
            let json = lua_value_to_json(lua, value)?;
            let json = inject_data_ty(json, &ty);
            let ud = lua.create_userdata(LuaStruct {
                type_name: ty,
                json: json.clone(),
            })?;
            Ok::<AnyUserData, mlua::Error>(ud)
        })?;
        let globals = lua.globals();
        globals.set("struct", make_struct.clone())?;
        globals.set("register_db_type", make_struct)?;
    }

    let new_fn = {
        let lua_shared_handle = lua_shared.clone();
        lua.create_function(move |lua, value: Value| {
            let (id, payload_json) = match value {
                Value::Table(tbl) => {
                    let id: String = tbl
                        .get("id")
                        .map_err(|_| mlua::Error::external("Mui.new requires field 'id'"))?;
                    let data_value: Option<Value> = tbl.get("data").ok();
                    let payload = match data_value {
                        Some(v) => ensure_db_payload(lua, v)?,
                        None => JsonValue::Null,
                    };
                    (id, payload)
                }
                Value::UserData(ud) => {
                    if let Ok(db_ref) = ud.borrow::<LuaTableDb>() {
                        let id = panel_uuid_from_db_index(db_ref.index);
                        (id, db_payload_with_revision(db_ref.index))
                    } else {
                        return Err(mlua::Error::external(
                            "Mui.new expects table or db userdata",
                        ));
                    }
                }
                other => {
                    return Err(mlua::Error::external(format!(
                        "Mui.new expects table or db userdata, got {}",
                        other.type_name()
                    )));
                }
            };

            let builder =
                LuaMuiBuilder::new(lua_shared_handle.clone(), id, LuaPayload(payload_json));
            lua.create_userdata(builder)
        })?
    };

    let globals = lua.globals();
    let mui = lua.create_table()?;
    mui.set("new", new_fn)?;
    globals.set("Mui", mui)?;

    // 重写 Lua 的 print，走 Rust println! 以避免 Windows 控制台乱码
    let print_fn = lua.create_function(|lua, values: Variadic<Value>| {
        let mut parts = Vec::new();
        for value in values {
            let text = match value {
                Value::String(s) => s.to_string_lossy(),
                Value::Nil => "nil".to_string(),
                Value::Boolean(b) => b.to_string(),
                Value::Integer(i) => i.to_string(),
                Value::Number(n) => n.to_string(),
                other => {
                    if let Some(coerced) = lua.coerce_string(other.clone())? {
                        coerced.to_string_lossy()
                    } else {
                        format!("{other:?}")
                    }
                }
            };
            parts.push(text);
        }
        if parts.is_empty() {
            println!("[lua]");
        } else {
            println!("[lua] {}", parts.join("\t"));
        }
        Ok(())
    })?;
    globals.set("print", print_fn)?;

    let print_recursive_fn = lua.create_function(|lua, values: Variadic<Value>| {
        if values.is_empty() {
            println!("[lua][print_r]");
            return Ok(());
        }

        let mut rendered = Vec::new();
        for value in values {
            let mut visited = HashSet::new();
            rendered.push(format_lua_value(lua, value, 0, &mut visited)?);
        }

        println!("[lua][print_r] {}", rendered.join("\t"));
        Ok(())
    })?;
    globals.set("print_r", print_recursive_fn)?;
    Ok(())
}

fn parse_container_spec(table: &Table) -> LuaResult<RelContainerSpec> {
    let mut spec = RelContainerSpec::default();
    if let Some(space) = table.get::<Option<String>>("space")? {
        spec.space = parse_space_name(&space)?;
    }
    if let Some(origin) = parse_vec2_field(table, "origin")? {
        spec.origin = origin;
    }
    if let Some(size) = parse_vec2_field(table, "size")? {
        spec.size = Some(size);
    }
    if let Some(slot) = parse_vec2_field(table, "slot_size")? {
        spec.slot_size = Some(slot);
    }
    if let Some(padding) = parse_vec4_field(table, "padding")? {
        spec.padding = padding;
    }
    if let Some(clip) = table.get::<Option<bool>>("clip_content")? {
        spec.clip_content = clip;
    } else if let Some(clip) = table.get::<Option<bool>>("clip")? {
        spec.clip_content = clip;
    }
    if let Some(axis) = table.get::<Option<String>>("scroll_axis")? {
        spec.scroll_axis = parse_scroll_axis_name(&axis)?;
    }
    if let Some(layout_value) = table.get::<Option<Value>>("layout")? {
        spec.layout = parse_layout_value(layout_value)?;
    }
    Ok(spec)
}

fn parse_border_style(table: &Table) -> LuaResult<BorderStyle> {
    let mut style = BorderStyle::default();
    if let Some(color) = parse_vec4_field(table, "color")? {
        style.color = color;
    }
    if let Some(width) = table.get::<Option<f32>>("width")? {
        style.width = width;
    }
    if let Some(radius) = table.get::<Option<f32>>("radius")? {
        style.radius = radius;
    }
    Ok(style)
}

fn parse_layout_value(value: Value) -> LuaResult<RelLayoutKind> {
    match value {
        Value::Nil => Ok(RelLayoutKind::free()),
        Value::String(name) => parse_layout_kind(&name.to_string_lossy(), None),
        Value::Table(tbl) => {
            let kind: String = tbl
                .get("kind")
                .map_err(|_| mlua::Error::external("layout.kind required"))?;
            parse_layout_kind(&kind, Some(&tbl))
        }
        other => Err(mlua::Error::external(format!(
            "layout expects string or table, got {}",
            other.type_name()
        ))),
    }
}

fn parse_layout_kind(kind: &str, table: Option<&Table>) -> LuaResult<RelLayoutKind> {
    let lower = kind.to_lowercase();
    match lower.as_str() {
        "free" => Ok(RelLayoutKind::free()),
        "horizontal" => {
            let spacing = table
                .map(|tbl| tbl.get::<Option<f32>>("spacing"))
                .transpose()?
                .flatten()
                .unwrap_or(0.0);
            Ok(RelLayoutKind::horizontal(spacing))
        }
        "vertical" => {
            let spacing = table
                .map(|tbl| tbl.get::<Option<f32>>("spacing"))
                .transpose()?
                .flatten()
                .unwrap_or(0.0);
            Ok(RelLayoutKind::vertical(spacing))
        }
        "grid" => {
            let spacing = table
                .and_then(|tbl| parse_vec2_field(tbl, "spacing").transpose())
                .transpose()?
                .unwrap_or([0.0, 0.0]);
            let columns = table
                .map(|tbl| tbl.get::<Option<u32>>("columns"))
                .transpose()?
                .flatten();
            let rows = table
                .map(|tbl| tbl.get::<Option<u32>>("rows"))
                .transpose()?
                .flatten();
            Ok(RelLayoutKind::Grid {
                spacing,
                columns,
                rows,
            })
        }
        "ring" => {
            let tbl = table.ok_or_else(|| {
                mlua::Error::external("layout.kind 'ring' requires radius/start_angle")
            })?;
            let radius: f32 = tbl
                .get("radius")
                .map_err(|_| mlua::Error::external("ring layout requires radius"))?;
            let start_angle: f32 = tbl.get("start_angle").unwrap_or(0.0);
            let clockwise: bool = tbl.get("clockwise").unwrap_or(true);
            Ok(RelLayoutKind::Ring {
                radius,
                start_angle,
                clockwise,
            })
        }
        other => Err(mlua::Error::external(format!(
            "unsupported layout kind '{other}'"
        ))),
    }
}

fn parse_vec2_field(table: &Table, key: &str) -> LuaResult<Option<[f32; 2]>> {
    match table.get::<Value>(key)? {
        Value::Nil => Ok(None),
        Value::Table(seq) => {
            let mut values = [0.0f32; 2];
            for (idx, item) in seq.sequence_values::<f32>().enumerate() {
                if idx >= 2 {
                    break;
                }
                values[idx] = item?;
            }
            Ok(Some(values))
        }
        _ => Ok(None),
    }
}

fn parse_vec4_field(table: &Table, key: &str) -> LuaResult<Option<[f32; 4]>> {
    match table.get::<Value>(key)? {
        Value::Nil => Ok(None),
        Value::Table(seq) => {
            let mut values = [0.0f32; 4];
            for (idx, item) in seq.sequence_values::<f32>().enumerate() {
                if idx >= 4 {
                    break;
                }
                values[idx] = item?;
            }
            Ok(Some(values))
        }
        _ => Ok(None),
    }
}

fn parse_scroll_axis_name(name: &str) -> LuaResult<RelScrollAxis> {
    match name.to_lowercase().as_str() {
        "none" => Ok(RelScrollAxis::None),
        "horizontal" | "x" => Ok(RelScrollAxis::Horizontal),
        "vertical" | "y" => Ok(RelScrollAxis::Vertical),
        "both" | "xy" => Ok(RelScrollAxis::Both),
        other => Err(mlua::Error::external(format!(
            "unsupported scroll axis '{other}'"
        ))),
    }
}

fn parse_space_name(name: &str) -> LuaResult<RelSpace> {
    match name.to_lowercase().as_str() {
        "screen" => Ok(RelSpace::Screen),
        "parent" => Ok(RelSpace::Parent),
        "local" => Ok(RelSpace::Local),
        other => Err(mlua::Error::external(format!(
            "unsupported container space '{other}'"
        ))),
    }
}
