print("[lua] Demo：单子面板容器挂载示例")

local Demo = {}

-- 准备父面板的数据句柄
local parent_data = db({
    tag = "single_parent",
    title = "父容器",
})

local function build_parent()
    Mui.new(parent_data)
        :default_state(0)
        :state(0)
            :position(100, 100)
            :size(320, 160)
            :color(0.16, 0.20, 0.32, 0.95)
            :border({
                color = { 0.95, 0.75, 0.35, 1.0 },
                width = 3.0,
                radius = 10.0,
            })
            :container({
                space = "parent",
                origin = { 0.0, 0.0 },
                size = { 320, 160 },
                slot_size = { 140, 100 },
                padding = { 20, 20, 20, 20 },
                layout = {
                    kind = "grid",
                    columns = 1,
                    rows = 1,
                    spacing = { 0, 0 },
                },
            })
        :build()
end

local function build_child()
    Mui.new({
        id = "lua_demo_child",
        data = db({ tag = "single_child" }),
    })
        :default_state(0)
        :state(0)
            :z_index(5)
            :size(140, 100)
            :color(0.45, 0.70, 0.88, 0.95)
            :border({
                color = { 1.0, 1.0, 1.0, 0.9 },
                width = 2.0,
                radius = 12.0,
            })
            :container_with(parent_data)
        :build()
end

function Demo.run()
    build_parent()
    build_child()
end

Demo.run()
print("lua端构建")