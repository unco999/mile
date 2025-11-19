-- Entry Lua script for build-time type registration + runtime entry.
-- Add/require your own Lua modules here.

require("test_require")
require("mui_core")
require("script")

-- 运行时入口函数（可被 Rust 调用）
-- 约定：Rust 获取全局函数 mile_entry 并调用它
function mile_entry(context)
    -- context 可由 Rust 传入（例如全局表、配置等），默认忽略即可
    return {
        run = function()
            -- 在这里启动你的 Lua 逻辑，或 require 其他模块
            print("[lua] mile_entry.run invoked")
        end
    }
end
