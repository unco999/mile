mod lua_db_codegen;

use lua_db_codegen::generate_types;
use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

fn main() -> Result<(), Box<dyn Error>> {
    println!("开始编译lua注册结构体 (入口脚本)");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    let entry_path = env::var("LUA_DB_ENTRY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("lua").join("main.lua"));

    let output_path = out_dir.join("lua_registered_types.rs");
    if !entry_path.exists() {
        fs::write(&output_path, "// auto-generated: no lua entry file found\n")?;
        println!("cargo:rerun-if-env-changed=LUA_DB_ENTRY");
        return Ok(());
    }

    let code = generate_types(&entry_path)?;
    fs::write(&output_path, code)?;

    println!("cargo:rerun-if-env-changed=LUA_DB_ENTRY");
    if let Some(dir) = entry_path.parent() {
        emit_rerun_for_dir(dir)?;
    }

    Ok(())
}

fn emit_rerun_for_dir(dir: &Path) -> Result<(), Box<dyn Error>> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                emit_rerun_for_dir(&path)?;
            } else if path.extension().is_some_and(|ext| ext == "lua") {
                eprintln!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    Ok(())
}
