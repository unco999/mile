-- Running this script should print the emitted payload from Rust and show the
-- table contents here. Multiple keys can be subscribed at once.
local stream = mile_event.on("lua_event_test", "hook")

-- Emit a payload that will be echoed by the Rust side logger and received here.
mile_event.emit("lua_event_test", { source = "lua", message = "hello from lua" })
mile_event.emit("hook", { game_event = "game_init" })

local drained = stream:drain()
print("[lua][test_event_bus] drained events:", #drained)
for idx, event in ipairs(drained) do
    -- Each event is converted back into a Lua table; print a summary.
    local key = event.__key or "unknown"
    local from = event.source or "unknown"
    local message = event.message or event.game_event or event.value or "(missing payload)"
    print(string.format("[lua][test_event_bus] #%d key=%s from=%s message=%s", idx, key, from, message))
end

return drained
