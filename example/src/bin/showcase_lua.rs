use mile_core::Mile;
use mile_lua::register_lua_api;
use mlua::{Function, Lua, Table, prelude::LuaSerdeExt};
use serde_json::{json, Value as JsonValue};


fn launch_lua_entry(lua: &Lua) -> mlua::Result<()> {
    // 载入 main.lua（可用 include_str! 或读取文件）
    let script_path = format!("lua/main.lua");
    println!("lua entry start => {}",script_path);
    let script = std::fs::read_to_string(script_path)?;
    lua.load(&script).set_name("main.lua").exec()?;

    // 获取 mile_entry，全局可选传入 context（lua.create_table()? 里可写配置）
    let globals = lua.globals();
    let entry: Function = globals.get("mile_entry")?;
    let context: Table = lua.create_table()?; // 如果无需参数，也可传 Value::Nil
    let handle: Table = entry.call(context)?;

    // 调用返回值里的 run（或你自定义的其他函数）
    if let Ok(run_fn) = handle.get::<Function>("run") {
        run_fn.call::<()>(())?;
    }
    Ok(())
}

fn main() {
    let lua = Lua::new();
    register_lua_api(&lua).expect("register lua api");

    Mile::new()
        .add_demo(move ||{
           launch_lua_entry(&lua).expect("lua runtime start error");
        })
        .run();
}

fn typed_payload_json(mut json: JsonValue, data_ty: &str) -> JsonValue {
    if let JsonValue::Object(ref mut map) = json {
        map.insert("data_ty".into(), JsonValue::String(data_ty.to_string()));
        return json;
    }

    let mut map = serde_json::Map::new();
    map.insert("data_ty".into(), JsonValue::String(data_ty.to_string()));
    map.insert("value".into(), json);
    JsonValue::Object(map)
}
