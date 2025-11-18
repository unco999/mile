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
print("[lua] 注册点击计数面板")

local counter = Mui.new({
        id = "lua_counter", 
        data = { count = 0, label = "Lua Counter" }
    })
    :default_state(0)
    :state(0)
    :position(120, 160)
    :size(260, 140)
    :color(0.2, 0.6, 0.9, 1.0)
    :on_event("click", function(flow)
        local payload = flow.payload
        payload.count = payload.count + 1
        print(string.format("[lua] %s clicked %d times", payload.label, payload.count))
        flow.payload = payload
        flow.next_state = 1
    end)
    :state(1)
    :position(420, 160)
    :size(260, 140)
    :color(0.9, 0.4, 0.2, 1.0)
    :on_event("click", function(flow)
        local payload = flow.payload
        payload.count = payload.count + 1
        print(string.format("[lua] %s (state %d) clicked %d times", payload.label, flow.state, payload.count))
        flow.payload = payload
        flow.next_state = 0
    end)
    :build()

print("[lua] 构建多状态面板")

local state_panel = Mui.new({
        id = "lua_state_demo",
        data = { message = "State Demo", current = 0 }
    })
    :default_state(0)
    :state(0)
    :position(120, 360)
    :size(220, 120)
    :color(0.3, 0.8, 0.4, 1.0)
    :on_event("click", function(flow)
        print(string.format("[lua] %s -> state 1", flow.payload.message))
        flow.payload.current = 1
        flow.next_state = 1
    end)
    :state(1)
    :position(360, 360)
    :size(220, 120)
    :color(0.8, 0.3, 0.4, 1.0)
    :on_event("click", function(flow)
        print(string.format("[lua] %s -> state 2", flow.payload.message))
        flow.payload.current = 2
        flow.next_state = 2
    end)
    :state(2)
    :position(600, 360)
    :size(220, 120)
    :color(0.4, 0.4, 0.9, 1.0)
    :on_event("click", function(flow)
        print(string.format("[lua] %s -> state 0", flow.payload.message))
        flow.payload.current = 0
        flow.next_state = 0
    end)
    :build()
"#;

            lua.load(script).exec().expect("lua exec");
        })
        .run();
}
