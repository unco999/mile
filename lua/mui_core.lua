local reset_api = rawget(_G, "mile_runtime_reset") or {}

local function resolve_handler(kind)
    local fn = reset_api[kind]
    if type(fn) ~= "function" then
        error(string.format("未提供 %s 重置接口", kind))
    end
    return fn
end

local function call_reset(kind)
    local fn = resolve_handler(kind)
    local ok, result = pcall(fn)
    if not ok then
        error(string.format("重置 %s 失败: %s", kind, result))
    end
    return result or 0
end

local function perform(kind, label)
    local cleared = call_reset(kind)
    print(string.format("[mui.reset] %s -> %d", label, cleared))
    return cleared
end

local function build_interface(kind, label)
    local iface = {}

    function iface.reset()
        return perform(kind, label)
    end

    -- 别名：部分模块语义上叫 clear
    iface.clear = iface.reset

    function iface.describe()
        return { kind = kind, label = label }
    end

    return iface
end

local M = {
    reset = {},
    db = build_interface("db", "db"),
    ui = build_interface("ui", "ui"),
    font = build_interface("font", "font"),
    kennel = build_interface("kennel", "kennel"),
}

M.reset.db = M.db.reset
M.reset.ui = M.ui.reset
M.reset.font = M.font.reset
M.reset.kennel = M.kennel.reset

function M.reset.all()
    return {
        db = M.reset.db(),
        ui = M.reset.ui(),
        font = M.reset.font(),
        kennel = M.reset.kennel(),
    }
end

return M
