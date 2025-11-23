
local panel_parent =  db({
})


local inventory_parent =  db({
})


Mui.new(panel_parent)
        :default_state(0)
        :state(0)
            :position(200,100)
            :size(400,150)
            :color(0.0,0.0,0.0,0.8)
            :container({
                size = {400,120},
                origin = {50,10},
                layout = { kind = "float", axis = "y" },
                spacing = { 4.0, 4.0 },         -- 需要的话
            })
            :border({
                width = 5,
                color = {1,1,1,5}
            })
        :build()

Mui.new(inventory_parent)
        :text("你把东西放过来")
        :default_state(0)
        :state(0)
            :position(200,300)
            :size(400,150)
            :color(0.0,0.0,0.0,0.8)
            :hover(true)
            :on_event("target_drag_drop", function(flow)
                if flow.drag_payload then
                    flow.drag_source_state = 1;
                    -- flow.drag_source_state = 1
                    print("确实有他的面板");
                end
                end)
            :container({
                size = {400,120},
                origin = {50,10},
                layout = { kind = "float", axis = "y" },
                spacing = { 4.0, 4.0 },         -- 需要的话
            })
            :border({
                width = 5,
                color = {0,1,1,5}
            })
        :build()
        
local texture_arr = {
    [0] = "XPlus_1_Q_8_00673_.png",  -- 索引 1
    [1] = "XPlus_1_Q_8_00674_.png",  -- 索引 2
    [2] = "XPlus_1_Q_8_00675_.png",  -- 索引 3
    [3] = "XPlus_1_Q_8_00676_.png",  -- 索引 4
    [4] = "XPlus_1_Q_8_00677_.png"   -- 索引 5
}


for i = 20,0,-1 do

    local panel_a =  db({

    })

    Mui.new(panel_a)
        :default_state(0)
        :state(1)
            :container_with(inventory_parent)
        :state(0)
            :position(0,0,0)
            :z_index(1)
            :drag(true)
            :container_with(panel_parent)
            :texture(texture_arr[i % 4])
            :border({
                color = {1,0,0,1},  
                width = 1
            })
            :on_event("source_drag_start", function(flow)
                print("[lua] 源面板开始拖拽")
                flow.drag_payload = panel_a;
            end)
            :on_event("source_drag_drop", function(flow)
                print("[lua] 源面板结束拖拽")
            end)
            :size(40,30)
        :build()
end