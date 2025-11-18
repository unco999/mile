mod lua_db_codegen;

use lua_db_codegen::generate_types;
use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

fn main() -> Result<(), Box<dyn Error>> {
    println!("开始编译lua注册结构体");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let lua_dir = env::var("LUA_DB_TYPE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("lua_types"));

    let output_path = out_dir.join("lua_registered_types.rs");
    if !lua_dir.exists() {
        fs::write(&output_path, "// auto-generated: no lua types found\n")?;
        println!("cargo:rerun-if-env-changed=LUA_DB_TYPE_DIR");
        return Ok(());
    }

    let code = generate_types(&lua_dir)?;
    fs::write(&output_path, code)?;

    println!("cargo:rerun-if-env-changed=LUA_DB_TYPE_DIR");
    emit_rerun_for_dir(&lua_dir)?;

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
