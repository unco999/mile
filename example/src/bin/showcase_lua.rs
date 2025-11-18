use mile_core::Mile;
use mile_lua::register_lua_api;
use mlua::Lua;

fn main() {
    let lua = Lua::new();
    register_lua_api(&lua).expect("register lua api");

    Mile::new()
        .add_demo(move || {
            let _ = register_lua_api(&lua);

            println!("开始执行 Lua demo");
            let script = r#"
print("[lua] 构建 Flow.text 测试")

local panel = Mui.new({
        id = "lua_text_panel",
        data = { count = 0 }
    })
    :position(200, 200)
    :size(320, 180)
    :color(0.2, 0.5, 0.8, 1.0)
    :on_event("click", function(flow)
        local payload = flow.payload
        payload.count = payload.count + 1
        flow.payload = payload
        return {
            text = {
                text = string.format("点击次数: %d", payload.count),
                font_path = "tf/STXIHEI.ttf",
                font_size = 32,
                color = { 1.0, 0.95, 0.8, 1.0 },
                weight = 600,
                line_height = 36
            }
        }
    end)
    :build()
"#;

            lua.load(script).exec().expect("lua exec");
        })
        .run();
}
