use std::{
    fs, io,
    path::{Path, PathBuf},
    thread,
};

use mile_api::prelude::global_event_bus;
use mile_core::{
    LUA_DEPLOY_DIR, LUA_SOURCE_DIR, Mile, bootstrap_lua_assets, log_deploy_event, run_lua_entry,
};
use mile_lua::{
    register_lua_api,
    watch::{LuaDeployEvent, LuaDeployStatus, spawn_lua_watch},
};
use mlua::{Function, Lua, Table};

fn main() {
    bootstrap_lua_assets().expect("sync lua assets into deploy dir");
    spawn_lua_deploy_logger();

    let mut binding = Mile::new();
    let mile = binding.add_demo(move || {
        if let Err(err) = run_lua_entry() {
            eprintln!("[lua] launch failed: {err}");
        }
    });

    let event_arc = mile.user_event.clone();
    let _lua_watch = spawn_lua_watch(LUA_SOURCE_DIR, LUA_DEPLOY_DIR, move || {
        println!("[lua_watch] change detected -> reloading scripts");
        _ = event_arc.send_event(mile_core::AppEvent::Reset);
    })
    .expect("start lua file watcher");
    mile.run();
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
