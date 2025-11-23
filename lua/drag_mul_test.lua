
local panel_parent =  db({
})

Mui.new(panel_parent)
        :default_state(0)
        :state(0)
        :size(400,100)
        :color(0.0,0.0,0.0,0.8)
        :drag(true)
        :container({
            layout = "float",
            axis = "x"
        })
        :border({
            width = 5,
            color = {1,1,1,5}
        })
        :build()


for i = 3,0,-1 do
    local panel_a =  db({

    })

    Mui.new(panel_a)
        :default_state(0)
        :state(0)
            :z_index(1)
            :container_with(panel_parent)
            :texture("XPlus_1_Q_8_00673_.png")
            :size(40,30)
        :build()
end