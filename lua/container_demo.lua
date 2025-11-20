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
            :position(100, 300)
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
                slot_size = { 30, 30 },
                padding = { 0, 0, 0, 0 },
                layout = {
                    kind = "grid",
                    columns = 1,
                    rows = 1,
                    spacing = { 0, 0 },
                },
            })
        :build()
end

local function build_child(id)
    Mui.new({
        id = "lua_demo_child" .. id,
        data = db({ tag = "single_child" }),
    })
        :default_state(0)
        :state(0)
            :position(1000,300)
            :z_index(10)
            :size(30, 30)
            :color(0.45, 0.70, 0.88, 0.95)
            :border({
                color = { 0.0, 1.0, 1.0, 0.9 },
                width = 2.0,
                radius = 1.0,
            })
            :container_with(parent_data)
        :build()
end

function Demo.run()
    build_parent()
    for i=20,1,-1 do
        build_child(i) 
    end
end

Demo.run()
print("lua端构建")