use mile_core::Mile;
use mile_lua::register_lua_api;
use mlua::Lua;

fn main() {
    let lua = Lua::new();
    register_lua_api(&lua).expect("register lua api");

    Mile::new()
        .add_demo(move || {
            register_lua_api(&lua);

            println!("开始执行lua");
            lua.load(
                r#"
                    local ui = Mui.new({test = 10,id = "uitest"})
                        :position(50,50)
                        :size(500,500)
                        :color(1.0,1.0,1.0,1.0)
                        :on_event("click",function(self,flow) print("click ui") end)
                        :build()

                    
                    
"#,
            )
            .exec()
            .expect("lua exec");
        })
        .run();
}
