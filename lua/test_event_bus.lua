-- Demonstrates the keyed event bus from Lua by emitting and draining a message.
-- Running this script should print the emitted payload from Rust and show the
-- table contents here.
local stream = mile_event.on("lua_event_test")

-- Emit a payload that will be echoed by the Rust side logger and received here.
mile_event.emit("lua_event_test", { source = "lua", message = "hello from lua" })

local drained = stream:drain()
print("[lua][test_event_bus] drained events:", #drained)
for idx, event in ipairs(drained) do
    -- Each event is converted back into a Lua table; print a summary.
    local from = event.source or "unknown"
    local message = event.message or "(missing message)"
    print(string.format("[lua][test_event_bus] #%d from=%s message=%s", idx, from, message))
end

return drained
