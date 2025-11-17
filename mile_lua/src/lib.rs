use glam::{vec2, vec4};
use mile_ui::mui_prototype::{EventFlow, Mui, UiEventKind, UiState};
use mlua::prelude::LuaSerdeExt;
use mlua::{Function, Lua, RegistryKey, Result as LuaResult, UserData, UserDataMethods, Value};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;

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
    color: Option<[f32; 4]>
}

#[derive(Clone)]
struct LuaMuiBuilder {
    lua: Arc<Lua>,
    id: String,
    payload: LuaPayload,
    state: StateSpec,
    handlers: HashMap<UiEventKind, Arc<RegistryKey>>,
}

impl LuaMuiBuilder {
    fn new(lua: Arc<Lua>, id: String, payload: LuaPayload) -> Self {
        Self {
            lua,
            id,
            payload,
            state: StateSpec::default(),
            handlers: HashMap::new(),
        }
    }

    fn build_internal(&self) -> LuaResult<()> {
        let payload_for_init = self.payload.clone();
        let state_spec = self.state.clone();
        let handlers = self.handlers.clone();
        let lua = self.lua.clone();

        let builder = Mui::<LuaPayload>::new(&self.id)
            .map_err(|err| mlua::Error::external(format!("Mui.new failed: {err}")))?;

        let builder = builder
            .default_state(UiState(0))
            .state(UiState(0), move |state| {
                let mut s = state;
                if let Some(pos) = state_spec.position {
                    s = s.position(vec2(pos[0], pos[1]));
                }
                if let Some(size) = state_spec.size {
                    s = s.size(vec2(size[0], size[1]));
                }
                if let Some(color) = state_spec.color {
                    s = s.color(vec4(color[0], color[1], color[2], color[3]));
                }

                let mut events = s.events().on_event(UiEventKind::Init, {
                    let payload = payload_for_init.clone();
                    move |flow| {
                        *flow.payload() = payload.clone();
                    }
                });

                for (kind, key) in handlers.iter() {
                    let lua = lua.clone();
                    let key = key.clone();
                    events = events.on_event(*kind, move |flow| {
                        if let Err(err) = dispatch_lua_event(&lua, &key, flow) {
                            eprintln!("lua on_event error: {err}");
                        }
                    });
                }

                events.finish()
            });

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

    let ret: Option<Value> = func.call(tbl)?;
    if let Some(Value::Table(new_tbl)) = ret {
        let new_payload: JsonValue = lua.from_value(Value::Table(new_tbl))?;
        *flow.payload() = LuaPayload(new_payload);
    }
        Ok(())
}

impl UserData for LuaMuiBuilder {
    fn add_methods<M: UserDataMethods<Self>>(methods: &mut M) {
        // 设置位置
        methods.add_method_mut("position", |lua, this, (x, y): (f32, f32)| {
            this.state.position = Some([x, y]);
            lua.create_userdata(this.clone())
        });

        // 设置尺寸
        methods.add_method_mut("size", |lua, this, (w, h): (f32, f32)| {
            this.state.size = Some([w, h]);
            lua.create_userdata(this.clone())
        });

        methods.add_method_mut("color", |lua, this, (r,g,b,a): (f32, f32,f32,f32)| {
            this.state.color = Some([r,g,b,a]);
            lua.create_userdata(this.clone())
        });


        // 事件回调：目前只支持 click（可按需扩展）
        methods.add_method_mut("on_event", |lua, this, (name, func): (String, Function)| {
            let kind = match name.to_lowercase().as_str() {
                "click" => UiEventKind::Click,
                other => {
                    return Err(mlua::Error::external(format!(
                        "unsupported event '{}' (only click)",
                        other
                    )))
                }
            };
            let key = Arc::new(lua.create_registry_value(func)?);
            this.handlers.insert(kind, key);
            lua.create_userdata(this.clone())
        });

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
                    )))
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
    Ok(())
}
