print("[lua] Demo：左侧两个按钮控制计数，右侧显示数值并带边框")

local SOURCE_TAG = "lua_counter_state"
local textures = {
    "caton (1).png",
    "caton (2).png",
    "caton (3).png",
}

local shared_state = {
    value = 0,
}


local function mutate_state(delta)
    shared_state.value = shared_state.value + delta
end

local function publish_state()
    return db({
        tag = SOURCE_TAG,
        value = shared_state.value,
    })
end

local function create_button(id, label, offset_x, delta)
    Mui.new({
        id = id,
        data = publish_state(),
    })
        :size(140, 80)
        :position(offset_x, 80)
        :color(0, 0, 0, 0.5)
        :border({
            color = { 0.95, 0.65, 0.35, 1.0 },
            width = 3.0,
            radius = 0.0,
        })
        :on_event("click", function(ctx)
            mutate_state(delta)
            local payload = publish_state()
            ctx.payload = payload
            ctx.text = {
                text = string.format("%s -> %d", label, shared_state.value),
                font_path = "tf/STXIHEI.ttf",
                font_size = 24,
                color = { 0.95, 0.95, 0.95, 1.0 },
            }
        end)
        :build()
end

create_button("lua_btn_dec", "递减 -", 40, -1)
create_button("lua_btn_inc", "递增 +", 220, 1)

local display = Mui.new({
    id = "lua_counter_display",
    data = publish_state(),
})
    :default_state(1)
    :state(1)
        :texture("caton (1).png")
        :on_event("click", function(ctx)
            ctx.text = {
                text = "实时更新就是爽",
                font_path = "tf/STXIHEI.ttf",
                font_size = 24,
                color = { 0.95, 0.95, 0.95, 1.0 },
            }
        end)
        :size(460, 320)
        :position(380, 60)
        :rotation(0,180,90)
    :build()


