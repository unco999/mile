use mlua::{Lua, Result as LuaResult, Table};
use std::thread;

use crate::register_lua_api;

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

    let spawn = lua.create_function(move |lua, source: String| {
        let package: Table = lua.globals().get("package")?;
        let package_path: String = package.get("path")?;
        let package_cpath: Option<String> = package.get("cpath").ok();
        let package_path = package_path.clone();
        let package_cpath = package_cpath.clone();
        thread::spawn(move || {
            if let Err(err) = spawn_worker_runtime(source, package_path, package_cpath) {
                eprintln!("[mile_thread] worker failed: {err}");
            }
        });
        Ok(())
    })?;

    thread_api.set("spawn", spawn)?;
    globals.set("mile_thread", thread_api)?;
    Ok(())
}

fn spawn_worker_runtime(
    source: String,
    package_path: String,
    package_cpath: Option<String>,
) -> LuaResult<()> {
    let lua = Lua::new();
    register_lua_api(&lua)?;
    configure_package_paths(&lua, &package_path, package_cpath.as_deref())?;
    lua.load(&source)
        .set_name("mile_thread_spawn")
        .exec()
}

fn configure_package_paths(lua: &Lua, path: &str, cpath: Option<&str>) -> LuaResult<()> {
    let globals = lua.globals();
    let package: Table = globals.get("package")?;
    package.set("path", path)?;
    if let Some(cpath_value) = cpath {
        package.set("cpath", cpath_value)?;
    }
    Ok(())
}
