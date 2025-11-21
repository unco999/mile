use mlua::{Lua, Result as LuaResult};
use std::thread;

/// Register a minimal threading helper for Lua scripts.
///
/// We create a **fresh** Lua runtime per worker thread because the primary
/// runtime is not `Sync` and cannot be accessed concurrently from multiple
/// OS threads. If you need to reuse the caller's environment, serialize the
/// state you need into the code chunk or provide configuration via globals
/// before spawning.
///
/// This exposes `mile_thread.spawn(code_string)` to Lua, which spawns a detached
/// OS thread executing the provided Lua chunk on the isolated Lua state. The
/// chunk is expected to be self contained; no state is shared with the caller.
pub fn register_thread_api(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let thread_api = lua.create_table()?;

    let spawn = lua.create_function(|_, source: String| {
        thread::spawn(move || {
            let lua = Lua::new();

            let chunk = lua.load(&source).set_name("mile_thread_spawn");
            if let Err(err) = chunk.and_then(|code| code.exec()) {
                eprintln!("[mile_thread] worker failed: {err}");
            }
        });
        Ok(())
    })?;

    thread_api.set("spawn", spawn)?;
    globals.set("mile_thread", thread_api)?;
    Ok(())
}
