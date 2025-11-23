use mile_api::prelude::global_event_bus;
use notify::{
    Config, Event, RecommendedWatcher, RecursiveMode, Result as NotifyResult, Watcher,
    recommended_watcher,
};
use std::any::Any;
use std::env;
use std::fs;
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub enum LuaDeployStatus {
    Copied,
    Removed,
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct LuaDeployEvent {
    pub source: PathBuf,
    pub target: PathBuf,
    pub status: LuaDeployStatus,
}

/// Spawn a filesystem watcher on `lua_root`. Whenever a `.lua` file changes the updated file is
/// copied into the mirrored path under `deploy_root`, a [`LuaDeployEvent`] is published on the
/// global event bus, and `on_reload` is invoked after the copy/removal succeeds.
pub fn spawn_lua_watch<S, D, F>(
    lua_root: S,
    deploy_root: D,
    mut on_reload: F,
) -> NotifyResult<RecommendedWatcher>
where
    S: AsRef<Path>,
    D: AsRef<Path>,
    F: FnMut() + Send + 'static,
{
    let source_root = canonicalize_dir(lua_root.as_ref());
    let deploy_root = ensure_dir(deploy_root.as_ref());

    let (tx, rx) = mpsc::channel();
    let mut watcher = recommended_watcher(move |res| {
        let _ = tx.send(res);
    })?;
    watcher.configure(Config::default().with_compare_contents(true))?;
    watcher.watch(&source_root, RecursiveMode::Recursive)?;

    let watch_root = source_root.clone();
    let deploy_root_clone = deploy_root.clone();

    thread::spawn(move || {
        let debounce = Duration::from_millis(200);
        let mut on_reload = on_reload;
        let mut pending_reload = false;
        let mut deadline: Option<Instant> = None;

        loop {
            let message = if let Some(deadline_instant) = deadline {
                match rx.recv_timeout(deadline_instant.saturating_duration_since(Instant::now())) {
                    Ok(res) => Some(res),
                    Err(RecvTimeoutError::Timeout) => {
                        if pending_reload {
                            invoke_reload(&mut on_reload);
                            pending_reload = false;
                        }
                        deadline = None;
                        continue;
                    }
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            } else {
                match rx.recv() {
                    Ok(res) => Some(res),
                    Err(_) => break,
                }
            };

            match message {
                Some(Ok(event)) => {
                    if handle_event(event, &watch_root, &deploy_root_clone) {
                        pending_reload = true;
                        deadline = Some(Instant::now() + debounce);
                    }
                }
                Some(Err(err)) => {
                    eprintln!("[lua_watch] notify error: {err}");
                }
                None => {}
            }
        }

        if pending_reload {
            invoke_reload(&mut on_reload);
        }
    });

    Ok(watcher)
}

fn invoke_reload<F: FnMut()>(handler: &mut F) {
    if let Err(err) = panic::catch_unwind(AssertUnwindSafe(|| handler())) {
        let message = describe_panic(err.as_ref());
        eprintln!("[lua_watch] reload handler panicked: {message}");
    }
}

fn describe_panic(err: &(dyn Any + Send + 'static)) -> String {
    if let Some(msg) = err.downcast_ref::<&str>() {
        (*msg).to_string()
    } else if let Some(msg) = err.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

fn handle_event(event: Event, source_root: &Path, deploy_root: &Path) -> bool {
    let mut should_reload = false;
    for path in event.paths {
        if let Some((deploy_event, trigger_reload)) = sync_lua_path(&path, source_root, deploy_root)
        {
            global_event_bus().publish(deploy_event);
            should_reload |= trigger_reload;
        }
    }
    should_reload
}

fn sync_lua_path(
    path: &Path,
    source_root: &Path,
    deploy_root: &Path,
) -> Option<(LuaDeployEvent, bool)> {
    if !is_lua_file(path) {
        return None;
    }

    let rel_path = match path.strip_prefix(source_root) {
        Ok(rel) => rel.to_path_buf(),
        Err(_) => {
            let fallback_target = path
                .file_name()
                .map(|name| deploy_root.join(name))
                .unwrap_or_else(|| deploy_root.to_path_buf());
            return Some((
                LuaDeployEvent {
                    source: path.to_path_buf(),
                    target: fallback_target,
                    status: LuaDeployStatus::Failed(format!(
                        "path {:?} outside lua root {:?}",
                        path, source_root
                    )),
                },
                false,
            ));
        }
    };

    let target = deploy_root.join(&rel_path);
    if path.exists() {
        match copy_to_target(path, &target) {
            Ok(()) => Some((
                LuaDeployEvent {
                    source: path.to_path_buf(),
                    target,
                    status: LuaDeployStatus::Copied,
                },
                true,
            )),
            Err(err) => Some((
                LuaDeployEvent {
                    source: path.to_path_buf(),
                    target,
                    status: LuaDeployStatus::Failed(format!("copy failed: {err}")),
                },
                false,
            )),
        }
    } else {
        match remove_from_target(&target) {
            Ok(()) => Some((
                LuaDeployEvent {
                    source: path.to_path_buf(),
                    target,
                    status: LuaDeployStatus::Removed,
                },
                true,
            )),
            Err(err) => Some((
                LuaDeployEvent {
                    source: path.to_path_buf(),
                    target,
                    status: LuaDeployStatus::Failed(format!("remove failed: {err}")),
                },
                false,
            )),
        }
    }
}

fn is_lua_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("lua"))
        .unwrap_or(false)
}

fn copy_to_target(source: &Path, target: &Path) -> io::Result<()> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    if source == target {
        return Ok(());
    }
    fs::copy(source, target)?;
    Ok(())
}

fn remove_from_target(target: &Path) -> io::Result<()> {
    match fs::remove_file(target) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

fn canonicalize_dir(path: &Path) -> PathBuf {
    match fs::canonicalize(path) {
        Ok(p) => strip_verbatim_prefix(p),
        Err(_) => absolute_fallback(path),
    }
}

fn absolute_fallback(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

#[cfg(windows)]
fn strip_verbatim_prefix(path: PathBuf) -> PathBuf {
    let raw = path.to_string_lossy().into_owned();
    if let Some(stripped) = raw.strip_prefix(r"\\?\") {
        PathBuf::from(stripped)
    } else {
        PathBuf::from(raw)
    }
}

#[cfg(not(windows))]
fn strip_verbatim_prefix(path: PathBuf) -> PathBuf {
    path
}

fn ensure_dir(path: &Path) -> PathBuf {
    if let Err(err) = fs::create_dir_all(path) {
        eprintln!("[lua_watch] failed to ensure deploy dir {:?}: {err}", path);
    }
    canonicalize_dir(path)
}
