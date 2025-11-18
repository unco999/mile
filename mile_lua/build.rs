mod lua_db_codegen;

use lua_db_codegen::generate_types;
use mlua::Lua;
use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

fn read_lua_script_file(){
 // 获取当前包的目录 (mile_lua/)
    let current_pkg_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:warning=当前包目录: {}", current_pkg_dir);
    
    // 直接使用当前包目录的父目录作为工作区根目录
    let workspace_root = Path::new(&current_pkg_dir).parent().unwrap();
    println!("cargo:warning=Workspace 根目录: {:?}", workspace_root);
    
    // Lua 源目录应该在 workspace 根目录下的 lua 文件夹
    let lua_src_dir = workspace_root.join("lua");
    println!("cargo:warning=Lua 源目录: {:?}", lua_src_dir);
    
    // 检查源目录是否存在
    if !lua_src_dir.exists() {
        println!("cargo:warning=❌ Lua 源目录不存在: {:?}", lua_src_dir);
        
        // 列出目录内容来调试
        if let Ok(entries) = fs::read_dir(workspace_root) {
            println!("cargo:warning=Workspace 根目录内容:");
            for entry in entries.flatten() {
                println!("cargo:warning=  - {:?}", entry.file_name());
            }
        }
        return;
    }
    
    let out_dir = env::var("OUT_DIR").unwrap();
    
    // 目标目录
    let target_dir = Path::new(&out_dir)
        .parent().unwrap()  // target/debug/build/mile_lua-hash
        .parent().unwrap()  // target/debug/build
        .parent().unwrap(); // target/debug
    
    println!("cargo:warning=目标目录: {:?}", target_dir);
    
    let lua_target_dir = target_dir.join("lua");
    println!("cargo:warning=Lua 目标目录: {:?}", lua_target_dir);
    
    // 复制文件
    if let Err(e) = copy_dir_all(&lua_src_dir, &lua_target_dir) {
        println!("cargo:warning=复制失败: {}", e);
    } else {
        println!("cargo:warning=✅ Lua 文件复制成功!");
        
        // 验证复制结果
        if lua_target_dir.join("main.lua").exists() {
            println!("cargo:warning=✅ main.lua 复制成功");
        }
        if lua_target_dir.join("test_require.lua").exists() {
            println!("cargo:warning=✅ test_require.lua 复制成功");
        }
    }
}

fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> std::io::Result<()> {
    fs::create_dir_all(&dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()))?;
        } else {
            fs::copy(entry.path(), dst.as_ref().join(entry.file_name()))?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    read_lua_script_file();
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
