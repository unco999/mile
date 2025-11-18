register_db_type("LuaUserProfile", {
    id = 9527,
    name = "Lua Wanderer",
    level = 42,
    online = true,
    stats = {
        hp = 180,
        mp = 64,
        crit = 0.25,
    },
    preferences = {
        theme = "dark",
        notifications = 1,
    }
})

register_db_type("LuaInventoryEntry", {
    item_id = 10010,
    name = "Phoenix Tonic",
    stack = 3,
    weight = 1.2,
    price = 56.5,
    owner = {
        id = 9527,
        alias = "Lua Wanderer",
    },
})