print("[lua] Demo：左侧两个按钮控制计数，右侧显示数值并带边框")

local SOURCE_TAG = "lua_counter_state"
local textures = {
    "caton (1).png",
    "caton (2).png",
    "caton (3).png",
}

local shared_state = {
    value = 15,
}


local global_bind = db({
        tag = SOURCE_TAG,
        value = shared_state.value,
})

local display_bind = db({
        tag = SOURCE_TAG,
        value = shared_state.value,
})


Mui.new(global_bind)
    :default_state(0)
    :state(0)
        :size(140, 80)
        :position(200, 200)
        :color(0, 0.5, 0, 0.7)
        :border({
            color = { 0.95, 0.65, 0.35, 1.0 },
            width = 3.0,
            radius = 0.0,
        })
        :on_event("click", function(ctx)
            local payload = ctx.payload;
            print("当前数据绑定",payload);
            ctx.payload.value = payload.value + 15;
            print("lua触发了点击",payload.value)
            ctx.text = {
                text = tostring(payload.value),
                font_path = "tf/STXIHEI.ttf",
                font_size = 24,
                color = { 0.95, 0.95, 0.95, 1.0 },
            }
        end)
    :build()


-- Mui.new(display_bind)
--     :default_state(0)
--     :state(0)
--         :size(140, 80)
--         :position(500, 200)
--         :color(0, 0.5, 0, 0.7)
--         :border({
--             color = { 0.95, 0.65, 0.35, 1.0 },
--             width = 3.0,
--             radius = 0.0,
--         })
--         :on_event("click", function(ctx)
--             local payload = ctx.payload;
--             print("当前数据绑定",payload);
--             ctx.payload.value = payload.value + 15;
--             print("lua触发了点击",payload.value)
--             ctx.text = {
--                 text = tostring(payload.value),
--                 font_path = "tf/STXIHEI.ttf",
--                 font_size = 24,
--                 color = { 0.95, 0.95, 0.95, 1.0 },
--             }
--         end)
--     :build()

Mui.new(display_bind)
    :default_state(0)
    :state(0)
        :size(140, 80)
        :position(200, 500)
        :color(0, 0.5,0.3, 0.7)
        :border({
            color = { 0.95, 0.65, 0.35, 1.0 },
            width = 3.0,
            radius = 0.0,
        })
        :on_target_data(global_bind,function(ctx)
            local target = ctx.source_payload;
            print("最新的绑定数据发生变更",target.value)
            ctx.text = {
                text = tostring(target.value) .. "你好",
                font_path = "tf/STXIHEI.ttf",
                font_size = 36,
                color = { 0.95, 0.95, 0.95, 1.0 },
            }
        end)
    :build()