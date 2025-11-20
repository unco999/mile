print("[lua] Demo：最简容器挂载示例")

local Demo = {}

local parent_data = db({ tag = "demo_parent" })

local function build_parent()
    Mui.new({
        id = "lua_demo_container",
        data = parent_data,
    })
        :default_state(1)
        :state(1)
            :position(120, 120)
            :size(360, 200)
            :color(0.15, 0.20, 0.28, 0.95)
            :border({
                color = { 0.9, 0.7, 0.3, 1.0 },
                width = 3.0,
                radius = 8.0,
            })
            :container({
                space = "parent",
                origin = { 0.0, 0.0 },
                size = { 360, 200 },
                slot_size = { 100, 60 },
                padding = { 16, 16, 16, 16 },
                layout = {
                    kind = "grid",
                    columns = 3,
                    rows = 1,
                    spacing = { 12, 0 },
                },
            })
        :build()
end

local function build_child(index, color)
    Mui.new({
        id = string.format("lua_demo_child_%02d", index),
        data = db({ tag = "demo_child", index = index }),
    })
        :size(100, 60)
        :color(color[1], color[2], color[3], 0.95)
        :border({
            color = { 1.0, 1.0, 1.0, 0.9 },
            width = 2.0,
            radius = 10.0,
        })
        :container_with(parent_data)
        :build()
end

function Demo.run()
    build_parent()
    build_child(1, { 0.45, 0.65, 0.85 })
    build_child(2, { 0.75, 0.45, 0.65 })
    build_child(3, { 0.55, 0.85, 0.55 })
end

return Demo
