-- Running this script should print the emitted payload from Rust and show the
-- table contents here. Multiple keys can be subscribed at once.
local stream = mile_event.on("lua_event_test", "hook")

-- Emit a payload that will be echoed by the Rust side logger and received here.
mile_event.emit("lua_event_test", { source = "lua", message = "hello from lua" })
mile_event.emit("hook", { game_event = "game_init" })


mile_event.emit("hook",{ game_event = "game_init"});

local drained = stream:drain()
print("[lua][test_event_bus] drained events:", #drained)
for idx, event in ipairs(drained) do
    -- Each event is converted back into a Lua table; print a summary.
    print("当前的lua侧打印",event.key)
end

return drained
