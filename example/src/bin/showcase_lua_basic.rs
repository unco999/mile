use std::{
    fs, io,
    path::{Path, PathBuf},
    thread,
};

use mile_api::prelude::global_event_bus;
use mile_core::Mile;
use mile_lua::{
    register_lua_api,
    watch::{LuaDeployEvent, LuaDeployStatus, spawn_lua_watch},
};
use mlua::{Function, Lua, Table};

const LUA_SOURCE_DIR: &str = "lua";
const LUA_DEPLOY_DIR: &str = "target/lua_runtime";

fn run_lua_entry() -> mlua::Result<()> {
    let lua = Lua::new();
    register_lua_api(&lua)?;
    let deploy_root = resolved_deploy_root();
    configure_package_path(&lua, &deploy_root)?;
    trigger_runtime_reset(&lua)?;

    let script_path = deploy_root.join("main.lua");
    println!("[lua] entry start => {}", script_path.display());
    let script = fs::read_to_string(&script_path)?;
    lua.load(&script).set_name("main.lua").exec()?;

    let globals = lua.globals();
    let entry: Function = globals.get("mile_entry")?;
    let context: Table = lua.create_table()?;
    let handle: Table = entry.call(context)?;

    if let Ok(run_fn) = handle.get::<Function>("run") {
        run_fn.call::<()>(())?;
    }

    Ok(())
}

fn configure_package_path(lua: &Lua, deploy_root: &Path) -> mlua::Result<()> {
    let globals = lua.globals();
    let package: Table = globals.get("package")?;
    let current_path: String = package.get("path")?;
    let deploy = path_to_lua_str(deploy_root);
    let search_roots = format!(
        "{deploy}/?.lua;{deploy}/?/init.lua;{current}",
        deploy = deploy,
        current = current_path
    );
    package.set("path", search_roots)?;
    Ok(())
}

fn main() {
    bootstrap_lua_assets().expect("sync lua assets into deploy dir");
    spawn_lua_deploy_logger();

    run_lua_entry().expect("initial lua launch");

    let _lua_watch = spawn_lua_watch(LUA_SOURCE_DIR, LUA_DEPLOY_DIR, move || {
        println!("[lua_watch] change detected -> reloading scripts");
        if let Err(err) = run_lua_entry() {
            eprintln!("[lua_watch] reload failed: {err}");
        }
    })
    .expect("start lua file watcher");

    Mile::new()
        .add_demo(move || {
            if let Err(err) = run_lua_entry() {
                eprintln!("[lua] launch failed: {err}");
            }
        })
        .run();
}

fn spawn_lua_deploy_logger() {
    let bus = global_event_bus().clone();
    thread::spawn(move || {
        let stream = bus.subscribe::<LuaDeployEvent>();
        while let Ok(delivery) = stream.recv() {
            let event = delivery.into_owned().unwrap_or_else(|arc| (*arc).clone());
            log_deploy_event(event);
        }
    });
}

fn log_deploy_event(event: LuaDeployEvent) {
    match event.status {
        LuaDeployStatus::Copied => {
            println!(
                "[lua_deploy] copied {:?} -> {:?}",
                event.source, event.target
            );
        }
        LuaDeployStatus::Removed => {
            println!(
                "[lua_deploy] removed target {:?} (source {:?})",
                event.target, event.source
            );
        }
        LuaDeployStatus::Failed(reason) => {
            eprintln!("[lua_deploy] failed to sync {:?}: {}", event.source, reason);
        }
    }
}

fn bootstrap_lua_assets() -> io::Result<()> {
    copy_lua_tree(Path::new(LUA_SOURCE_DIR), Path::new(LUA_DEPLOY_DIR))
}

fn copy_lua_tree(src: &Path, dst: &Path) -> io::Result<()> {
    if !src.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("lua source dir {:?} not found", src),
        ));
    }

    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        let target_path = dst.join(entry.file_name());

        if file_type.is_dir() {
            copy_lua_tree(&path, &target_path)?;
        } else if is_lua_file(&path) {
            if let Some(parent) = target_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&path, &target_path)?;
        }
    }
    Ok(())
}

fn is_lua_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("lua"))
        .unwrap_or(false)
}

fn resolved_deploy_root() -> PathBuf {
    fs::canonicalize(LUA_DEPLOY_DIR).unwrap_or_else(|_| Path::new(LUA_DEPLOY_DIR).to_path_buf())
}

fn path_to_lua_str(path: &Path) -> String {
    let replaced = path.to_string_lossy().replace('\\', "/");
    if let Some(stripped) = replaced.strip_prefix("//?/") {
        stripped.to_string()
    } else {
        replaced
    }
}

fn trigger_runtime_reset(lua: &Lua) -> mlua::Result<()> {
    let globals = lua.globals();
    if let Ok(reset_table) = globals.get::<Table>("mile_runtime_reset") {
        if let Ok(reset_fn) = reset_table.get::<Function>("all") {
            let _: () = reset_fn.call(())?;
        }
    }
    Ok(())
}
