local payload_counter = 0

local source_bind = db({
    tag = "drag_payload_source",
    label = "源面板"
})

local target_bind = db({
    tag = "drag_payload_target",
    last_message = "等待拖入"
})

Mui.new(source_bind)
    :default_state(0)
    :state(0)
        :z_index(1)
        :position(120, 120)
        :size(220, 140)
        :color(0.15, 0.35, 0.65, 0.92)
        :border({
            color = { 0.95, 0.95, 0.95, 0.4 },
            width = 2.0,
            radius = 12.0,
        })
        :on_event("hover", function(flow)end)
        :text({ text = "拖拽这里", font_size = 32 })
        :on_event("source_drag_start", function(flow)
            payload_counter = payload_counter + 1
            flow.drag_payload = {
                message = string.format("这是第 %d 次拖拽", payload_counter),
                count = payload_counter,
            }
            print(string.format("[lua] 源面板开始拖拽 -> payload.count=%d", payload_counter))
        end)
        :on_event("source_drag_drop", function()
            print("[lua] 源面板拖拽完成")
        end)
    :build()

Mui.new(target_bind)
    :default_state(0)
    :state(0)
        :z_index(4)
        :position(420, 160)
        :size(260, 220)
        :color(0.09, 0.12, 0.18, 0.95)
        :border({
            color = { 0.35, 0.75, 0.55, 0.8 },
            width = 2.0,
            radius = 14.0,
        })
        :text({ text = "目标区域", font_size = 30 })
            :on_event("hover", function(flow)end)
            :on_event("target_drag_enter", function(flow)
                local data = flow.drag_payload
                if data then
                    print(string.format("[lua] 拖入 -> %s", data.message or "(nil)"))
                    flow.text = {
                        text = "收到: " .. (data.message or "未知"),
                        font_size = 26,
                    }
                else
                    print("[lua] 拖入但没有 payload")
                end
            end)
            :on_event("target_drag_over", function(flow)
                if flow.drag_payload then
                    flow.border = {
                        color = { 0.9, 0.8, 0.2, 1.0 },
                        width = 3.5,
                        radius = 16.0,
                    }
                end
            end)
            :on_event("target_drag_leave", function(flow)
                flow.border = {
                    color = { 0.35, 0.75, 0.55, 0.8 },
                    width = 2.0,
                    radius = 14.0,
                }
            end)
            :on_event("target_drag_drop", function(flow)
                local data = flow.drag_payload
                if data then
                    print(string.format("[lua] 在目标处落下 -> count=%d", data.count or -1))
                end
            end)
    :build()
