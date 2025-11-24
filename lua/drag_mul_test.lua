
local panel_parent =  db({
    counter = 0
})


local inventory_parent =  db({
})


Mui.new(panel_parent)
        :default_state(0)
        :state(0)
            :position(200,100)
            :size(400,150)
            :color(0.0,0.0,0.0,0.9)
            :border({
                width = 5,
                color = {1,1,1,5}
            })
            :on_event("click",function(ctx)
                ctx.payload.counter = ctx.payload.counter + 1;
                ctx.text = {
                    text = tostring(ctx.payload.counter)
                }
                print("当前点击" .. ctx.payload.counter)
            end)
        :build()
