print("[lua] Demo             ڵ  10         (             ֵ)")

local PARENT_TAG = "lua_container_parent"
local ITEM_TAG = "lua_container_child"

local parent_bind = db({
    tag = PARENT_TAG,
    title = "test",
})

local parent_bind2 = db({
    tag = PARENT_TAG,
    title = "test",
})

Mui.new(parent_bind2)
    :default_state(0)
    :state(0)
        :size(560, 500)
        :position(100, 620)
        :color(0.08, 0.50, 0.16, 0.95)
        :border({
            color = { 0.15, 0.45, 0.85, 0.8 },
            width = 3.0,
            radius = 10.0,
        })
        :on_event("target_drag_drop",function()
            print("被拖入了")
        end)
        :on_event("hover",function()
            print("悬浮")
        end)
        :container({
            origin = { 0.0, 0.0 },
            size = { 560.0, 500.0 },
            padding = { 20.0, 20.0, 20.0, 20.0 },
            slot_size = {50,50},
            layout = {
                kind = "grid",
                columns = 2,
                spacing = { 16.0, 16.0 },
            },
        })
    :build()
Mui.new(parent_bind)
    :default_state(0)
    :state(0)
        :size(560, 500)
        :position(120, 120)
        :color(0.08, 0.10, 0.16, 0.95)
        :border({
            color = { 0.15, 0.45, 0.85, 0.8 },
            width = 3.0,
            radius = 10.0,
        })
        :container({
            origin = { 0.0, 0.0 },
            size = { 560.0, 500.0 },
            padding = { 20.0, 20.0, 20.0, 20.0 },
            slot_size = {50,50},
            layout = {
                kind = "grid",
                columns = 2,
                spacing = { 16.0, 16.0 },
            },
        })
    :build()

for i = 1, 10 do
    local child_binding = db({
        index = i
    })
    Mui.new(child_binding)
        :default_state(0)
        :state(0)
            :position(50 * i,50 * i)
            :size(50, 50)
            :color(0.12 + i * 0.015, 0.35 + i * 0.02, 0.25, 0.88)
            :border({
                color = { 0.95, 0.95, 0.95, 0.9 },
                width = 2.0,
                radius = 8.0,
            })
            :z_index(10)
            :on_event("drag",function(ctx) 
                local payload = ctx.payload
                ctx.drag_payload = payload;
                print("当前拖拽开始了");
            end)
            :container_with(parent_bind)
        :build()
end

