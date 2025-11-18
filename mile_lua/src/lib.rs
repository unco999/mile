use glam::{vec2, vec4};
use mile_ui::{
    mui_prototype::{EventFlow, Mui, UiEventKind, UiState},
    mui_rel::{apply_container_alias, RelContainerSpec, RelLayoutKind, RelScrollAxis, RelSpace},
};
use mlua::prelude::LuaSerdeExt;
use mlua::{
    Function, Lua, RegistryKey, Result as LuaResult, Table, UserData, UserDataMethods, Value,
    Variadic,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;

// Generated Lua DB structs (build.rs executes Lua entry script; default: lua/main.lua)
include!(concat!(env!("OUT_DIR"), "/lua_registered_types.rs"));

#[derive(Clone, Debug, PartialEq)]
pub struct LuaPayload(pub JsonValue);

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
    color: Option<[f32; 4]>,
    texture: Option<String>,
    size_with_image: bool,
    trigger_mouse_pos: bool,
    visible: Option<bool>,
    container: Option<RelContainerSpec>,
    z_index: Option<i32>,
}

#[derive(Clone)]
struct StateEntry {
    spec: StateSpec,
    handlers: HashMap<UiEventKind, Arc<RegistryKey>>,
    data_handlers: Vec<LuaDataHandler>,
    container_links: Vec<String>,
    container_aliases: Vec<(String, String)>,
}

impl Default for StateEntry {
    fn default() -> Self {
        Self {
            spec: StateSpec::default(),
            handlers: HashMap::new(),
            data_handlers: Vec::new(),
            container_links: Vec::new(),
            container_aliases: Vec::new(),
        }
    }
}

#[derive(Clone)]
struct LuaDataHandler {
    ty: Option<String>,
    source: Option<String>,
    callback: Arc<RegistryKey>,
}

#[derive(Clone)]
struct LuaMuiBuilder {
    lua: Arc<Lua>,
    id: String,
    payload: LuaPayload,
    states: HashMap<u32, StateEntry>,
    current_state: u32,
    default_state: Option<u32>,
}

impl LuaMuiBuilder {
    fn new(lua: Arc<Lua>, id: String, payload: LuaPayload) -> Self {
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
                if let Some(color) = entry.spec.color {
                    s = s.color(vec4(color[0], color[1], color[2], color[3]));
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

                for (kind, key) in entry.handlers.iter() {
                    let lua = lua.clone();
                    let key = key.clone();
                    events = events.on_event(*kind, move |flow| {
                        if let Err(err) = dispatch_lua_event(&lua, &key, flow) {
                            eprintln!("lua on_event error: {err}");
                        }
                    });
                }

                for handler in entry.data_handlers.iter() {
                    let lua = lua.clone();
                    let key = handler.callback.clone();
                    let source_owned = handler.source.clone();
                    let source_for_reg = source_owned.clone();
                    let ty_filter = handler.ty.clone();
                    let ty_filter_closure = ty_filter.clone();
                    events = events.on_data_change::<LuaPayload, _>(
                        source_for_reg.as_deref(),
                        move |src_payload, flow| {
                            if !payload_matches_ty(src_payload, ty_filter_closure.as_deref()) {
                                return;
                            }
                            if let Err(err) = dispatch_lua_on_data(
                                &lua,
                                &key,
                                source_owned.clone(),
                                src_payload.clone(),
                                flow,
                            ) {
                                eprintln!("lua on_target_data error: {err}");
                            }
                        },
                    );
                }

                events.finish()
            });
        }

        builder
            .build()
            .map_err(|err| mlua::Error::external(format!("Mui.build failed: {err}")))?;

        Ok(())
    }
}

fn dispatch_lua_event(
    lua: &Arc<Lua>,
    key: &RegistryKey,
    flow: &mut EventFlow<'_, LuaPayload>,
) -> LuaResult<()> {
    let panel_id = flow.args().panel_key.panel_uuid.clone();
    let state_id = flow.state().0;
    let event_name = format!("{:#?}", flow.args().event);
    let payload = flow.payload_ref().clone();

    let func: Function = lua.registry_value(key)?;
    let tbl = lua.create_table()?;
    tbl.set("panel_id", panel_id)?;
    tbl.set("state", state_id)?;
    tbl.set("event", event_name)?;
    tbl.set("payload", lua.to_value(&payload.0)?)?;

    let ret: Option<Value> = func.call(tbl.clone())?;
    let lua_ref = lua.as_ref();
    if let Some(Value::Table(new_tbl)) = ret {
        apply_flow_directives(lua_ref, &new_tbl, flow)?;
    }
    apply_flow_directives(lua_ref, &tbl, flow)?;
    Ok(())
}

fn dispatch_lua_on_data(
    lua: &Arc<Lua>,
    key: &RegistryKey,
    source_uuid: Option<String>,
    source_payload: LuaPayload,
    flow: &mut EventFlow<'_, LuaPayload>,
) -> LuaResult<()> {
    let func: Function = lua.registry_value(key)?;
    let tbl = lua.create_table()?;
    tbl.set("panel_id", flow.args().panel_key.panel_uuid.clone())?;
    tbl.set("state", flow.state().0)?;
    tbl.set("event", "on_target_data")?;
    tbl.set("payload", lua.to_value(&flow.payload_ref().0)?)?;
    if let Some(src) = source_uuid.as_ref() {
        tbl.set("source_uuid", src.clone())?;
    }
    tbl.set("source_payload", lua.to_value(&source_payload.0)?)?;

    let ret: Option<Value> = func.call(tbl.clone())?;
    let lua_ref = lua.as_ref();
    if let Some(Value::Table(new_tbl)) = ret {
        apply_flow_directives(lua_ref, &new_tbl, flow)?;
    }
    apply_flow_directives(lua_ref, &tbl, flow)?;
    Ok(())
}

fn payload_matches_ty(payload: &LuaPayload, ty: Option<&str>) -> bool {
    let Some(expected) = ty else {
        return true;
    };
    if expected.is_empty() {
        return true;
    }
    match &payload.0 {
        JsonValue::Object(map) => map
            .get("data_ty")
            .and_then(|value| value.as_str())
            .map_or(false, |value| value == expected),
        _ => false,
    }
}

fn apply_flow_directives(
    lua: &Lua,
    table: &Table,
    flow: &mut EventFlow<'_, LuaPayload>,
) -> LuaResult<()> {
    if let Ok(Value::Table(payload_tbl)) = table.get("payload") {
        let new_payload: JsonValue = lua.from_value(Value::Table(payload_tbl.clone()))?;
        *flow.payload() = LuaPayload(new_payload);
    }
    if let Ok(value) = table.get::<Value>("text") {
        if !matches!(value, Value::Nil) {
            apply_text_from_lua(flow, &value)?;
        }
    }
    if let Ok(Some(next_state)) = table.get::<Option<u32>>("next_state") {
        flow.set_state(UiState(next_state));
    }
    Ok(())
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
    let weight: u32 = table.get("weight").unwrap_or(400);
    let line_height: u32 = table.get("line_height").unwrap_or(0);

    let final_color = if let Some(c) = color {
        let mut arr = [1.0, 1.0, 1.0, 1.0];
        for (i, v) in c.iter().copied().take(4).enumerate() {
            arr[i] = v;
        }
        arr
    } else {
        [1.0, 1.0, 1.0, 1.0]
    };

    let path = font_path
        .map(|p| p.into())
        .unwrap_or_else(|| "tf/Alibaba-PuHuiTi-Regular.ttf".into());
    flow.text(&text, path, font_size, final_color, weight, line_height);
    Ok(())
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

        methods.add_method_mut("color", |lua, this, (r, g, b, a): (f32, f32, f32, f32)| {
            this.current_entry_mut().spec.color = Some([r, g, b, a]);
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

        methods.add_method_mut("container_with", |lua, this, panel_uuid: String| {
            this.current_entry_mut().container_links.push(panel_uuid);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut(
            "container_with_alias",
            |lua, this, (alias, panel_uuid): (String, String)| {
                this.current_entry_mut()
                    .container_aliases
                    .push((alias, panel_uuid));
                lua.create_userdata(this.clone())
            },
        );

        // 事件回调：目前只支持 click（可按需扩展）
        methods.add_method_mut("on_event", |lua, this, (name, func): (String, Function)| {
            let kind = match name.to_lowercase().as_str() {
                "click" => UiEventKind::Click,
                other => {
                    return Err(mlua::Error::external(format!(
                        "unsupported event '{}' (only click)",
                        other
                    )));
                }
            };
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
            |lua, this, (ty, source, func): (Option<String>, Option<String>, Function)| {
                let key = Arc::new(lua.create_registry_value(func)?);
                this.current_entry_mut().data_handlers.push(LuaDataHandler {
                    ty,
                    source,
                    callback: key,
                });
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
    let lua_arc = Arc::new(lua.clone());

    let new_fn = {
        let lua_arc = lua_arc.clone();
        lua.create_function(move |lua, table: Value| {
            let tbl = match table {
                Value::Table(t) => t,
                other => {
                    return Err(mlua::Error::external(format!(
                        "Mui.new expects table, got {}",
                        other.type_name()
                    )));
                }
            };

            let id: String = tbl
                .get("id")
                .map_err(|_| mlua::Error::external("Mui.new requires field 'id'"))?;

            let data_value: Option<Value> = tbl.get("data").ok();
            let payload_json: JsonValue = match data_value {
                Some(v) => lua.from_value(v)?,
                None => JsonValue::Null,
            };

            let builder = LuaMuiBuilder::new(lua_arc.clone(), id, LuaPayload(payload_json));
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
