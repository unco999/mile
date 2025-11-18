use mile_core::Mile;
use mile_lua::{
    register_lua_api, LuaInventoryEntry, LuaInventoryEntry_Owner, LuaUserProfile,
    LuaUserProfile_Preferences, LuaUserProfile_Stats,
};
use mlua::{prelude::LuaSerdeExt, Lua};
use serde::Serialize;
use serde_json::Value as JsonValue;

const LUA_DEMO_SCRIPT: &str = r#"
local function deep_clone(value)
    if type(value) ~= "table" then
        return value
    end
    local copy = {}
    for key, entry in pairs(value) do
        copy[key] = deep_clone(entry)
    end
    return copy
end

local profile_payload = deep_clone(RUST_PROFILE_DATA or { data_ty = "LuaUserProfile" })
local inventory_payload = {}
if type(RUST_INVENTORY_DATA) == "table" then
    for idx, entry in ipairs(RUST_INVENTORY_DATA) do
        inventory_payload[idx] = deep_clone(entry)
    end
end

print("[lua] 数据驱动 UI：LuaUserProfile / LuaInventoryEntry 绑定示例")

Mui.new({
    id = "lua_profile_card",
    data = profile_payload
})
:position(60, 120)
:size(420, 230)
:color(0.12, 0.16, 0.26, 0.95)
:container({
    space = "screen",
    size = { 420, 230 },
    padding = { 18, 18, 18, 18 },
    layout = {
        kind = "vertical",
        spacing = 12
    }
})
:on_event("click", function(flow)
    local payload = flow.payload or {}
    payload.level = (payload.level or 1) + 1
    payload.online = not payload.online
    if payload.stats then
        payload.stats.hp = (payload.stats.hp or 0) + 8
        payload.stats.mp = math.max(8, (payload.stats.mp or 0) - 2)
    end
    payload.data_ty = "LuaUserProfile"
    flow.payload = payload
    return { payload = payload }
end)
:build()

Mui.new({ id = "lua_profile_observer" })
:position(76, 138)
:size(388, 198)
:z_index(10)
:on_target_data("LuaUserProfile", "lua_profile_card", function(flow)
    local payload = flow.source_payload or {}
    local stats = payload.stats or {}
    local state = payload.online and "在线" or "离线"
    local text = string.format(
        "Lv.%d %s\nHP:%d  MP:%d  Crit:%.2f\n状态: %s\n点击卡片会刷新 payload 并触发本面板",
        payload.level or 0,
        payload.name or "??",
        stats.hp or 0,
        stats.mp or 0,
        stats.crit or 0,
        state
    )
    return {
        text = {
            text = text,
            font_path = "tf/STXIHEI.ttf",
            font_size = 24,
            line_height = 28
        }
    }
end)
:build()

local inventory_cursor = 1
local function cycle_inventory()
    if #inventory_payload == 0 then
        return { data_ty = "LuaInventoryEntry" }
    end
    if inventory_cursor > #inventory_payload then
        inventory_cursor = 1
    end
    local snapshot = deep_clone(inventory_payload[inventory_cursor])
    inventory_cursor = inventory_cursor + 1
    snapshot.data_ty = "LuaInventoryEntry"
    return snapshot
end

Mui.new({
    id = "lua_inventory_source",
    data = cycle_inventory()
})
:on_target_data("LuaUserProfile", "lua_profile_card", function(flow)
    local next_entry = cycle_inventory()
    flow.payload = next_entry
    return { payload = next_entry }
end)
:build()

Mui.new({
    id = "lua_inventory_observer"
})
:position(520, 120)
:size(420, 230)
:color(0.10, 0.12, 0.18, 0.95)
:container({
    space = "screen",
    size = { 420, 230 },
    padding = { 18, 18, 18, 18 }
})
:on_target_data("LuaInventoryEntry", "lua_inventory_source", function(flow)
    local item = flow.source_payload or {}
    local owner = item.owner or {}
    local text = string.format(
        "物品: %s (#%d)\n数量:%d  单价:%.1f  重量:%.1f\n持有者:%s (%d)\n\n提示：左侧卡片被点击时，这里会轮换展示下一个库存条目。",
        item.name or "??",
        item.item_id or 0,
        item.stack or 0,
        item.price or 0,
        item.weight or 0,
        owner.alias or "??",
        owner.id or 0
    )
    return {
        text = {
            text = text,
            font_path = "tf/STXIHEI.ttf",
            font_size = 22,
            line_height = 26
        }
    }
end)
:build()
"#;

fn main() {
    let lua = Lua::new();
    register_lua_api(&lua).expect("register lua api");

    let profile = LuaUserProfile {
        online: true,
        name: "Lua Wanderer".into(),
        stats: LuaUserProfile_Stats {
            mp: 64,
            hp: 180,
            crit: 0.25,
        },
        level: 42,
        preferences: LuaUserProfile_Preferences {
            notifications: 1,
            theme: "dark".into(),
        },
        id: 9527,
    };

    let inventory = vec![
        LuaInventoryEntry {
            price: 56.5,
            name: "Phoenix Tonic".into(),
            item_id: 10010,
            owner: LuaInventoryEntry_Owner {
                alias: profile.name.clone(),
                id: profile.id,
            },
            weight: 1.2,
            stack: 3,
        },
        LuaInventoryEntry {
            price: 128.0,
            name: "Starlit Compass".into(),
            item_id: 10042,
            owner: LuaInventoryEntry_Owner {
                alias: "Quest Broker".into(),
                id: 7,
            },
            weight: 4.0,
            stack: 1,
        },
        LuaInventoryEntry {
            price: 12.0,
            name: "Traveler's Patch".into(),
            item_id: 10073,
            owner: LuaInventoryEntry_Owner {
                alias: "Lua Wanderer".into(),
                id: profile.id,
            },
            weight: 0.2,
            stack: 12,
        },
    ];

    let profile_payload = typed_payload_json(profile, "LuaUserProfile");
    let inventory_payload: Vec<JsonValue> = inventory
        .into_iter()
        .map(|entry| typed_payload_json(entry, "LuaInventoryEntry"))
        .collect();

    Mile::new()
        .add_demo({
            let lua = lua.clone();
            let profile_payload = profile_payload.clone();
            let inventory_payload = inventory_payload.clone();
            move || {
                let globals = lua.globals();
                globals
                    .set(
                        "RUST_PROFILE_DATA",
                        lua.to_value(&profile_payload).expect("profile to lua"),
                    )
                    .expect("set profile data");
                globals
                    .set(
                        "RUST_INVENTORY_DATA",
                        lua.to_value(&inventory_payload).expect("inventory to lua"),
                    )
                    .expect("set inventory data");

                lua.load(LUA_DEMO_SCRIPT)
                    .exec()
                    .expect("lua demo exec");
            }
        })
        .run();
}

fn typed_payload_json<T: Serialize>(value: T, data_ty: &str) -> JsonValue {
    let mut json = serde_json::to_value(value).expect("serialize typed payload");
    if let JsonValue::Object(ref mut map) = json {
        map.insert("data_ty".into(), JsonValue::String(data_ty.to_string()));
    }
    json
}
