use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PanelStylePatch {
    pub texture: Option<String>,
    pub fit_to_texture: Option<bool>,
    pub position: Option<[f32; 2]>,
    pub size: Option<[f32; 2]>,
    pub offset: Option<[f32; 2]>,
    pub rotation: Option<[f32; 3]>,
    pub scale: Option<[f32; 3]>,
    pub color: Option<[f32; 4]>,
    pub border: Option<BorderStylePatch>,
    pub z_index: Option<u32>,
    pub pass_through: Option<u32>,
    pub transparent: Option<f32>,
    pub state_transform_fade: Option<f32>,
    pub fragment_shader_id: Option<u32>,
    pub vertex_shader_id: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BorderStylePatch {
    pub color: Option<[f32; 4]>,
    pub width: Option<f32>,
    pub radius: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum StyleError {
    UnknownKey(String),
    Io(PathBuf, String),
    Parse(PathBuf, String),
}

impl std::fmt::Display for StyleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StyleError::UnknownKey(key) => write!(f, "unknown style key: {key}"),
            StyleError::Io(path, msg) => {
                write!(f, "failed to read style {}: {msg}", path.display())
            }
            StyleError::Parse(path, msg) => {
                write!(f, "failed to parse style {}: {msg}", path.display())
            }
        }
    }
}

impl std::error::Error for StyleError {}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct PanelStyleSchema {
    texture: Option<String>,
    fit_to_texture: Option<bool>,
    position: Option<[f32; 2]>,
    size: Option<[f32; 2]>,
    offset: Option<[f32; 2]>,
    rotation: Option<[f32; 3]>,
    scale: Option<[f32; 3]>,
    color: Option<[f32; 4]>,
    border: Option<BorderStylePatch>,
    z_index: Option<u32>,
    pass_through: Option<u32>,
    transparent: Option<f32>,
    interaction: Option<u32>,
    event_mask: Option<u32>,
    state_mask: Option<u32>,
    collection_state: Option<u32>,
    state_transform_fade: Option<f32>,
    fragment_shader_id: Option<u32>,
    vertex_shader_id: Option<u32>,
}

impl From<PanelStyleSchema> for PanelStylePatch {
    fn from(schema: PanelStyleSchema) -> Self {
        PanelStylePatch {
            texture: schema.texture,
            fit_to_texture: schema.fit_to_texture,
            position: schema.position,
            size: schema.size,
            offset: schema.offset,
            rotation: schema.rotation,
            scale: schema.scale,
            color: schema.color,
            border: schema.border,
            z_index: schema.z_index,
            pass_through: schema.pass_through,
            transparent: schema.transparent,
            state_transform_fade: schema.state_transform_fade,
            fragment_shader_id: schema.fragment_shader_id,
            vertex_shader_id: schema.vertex_shader_id,
        }
    }
}

static STYLE_REGISTRY: OnceLock<Result<HashMap<String, PathBuf>, StyleError>> = OnceLock::new();

fn style_registry() -> &'static Result<HashMap<String, PathBuf>, StyleError> {
    STYLE_REGISTRY.get_or_init(|| {
        let index_path = PathBuf::from("styles/index.json");
        if !index_path.exists() {
            return Ok(HashMap::new());
        }
        let text = fs::read_to_string(&index_path)
            .map_err(|err| StyleError::Io(index_path.clone(), format!("{}", err)))?;
        let mapping: HashMap<String, String> = serde_json::from_str(&text)
            .map_err(|err| StyleError::Parse(index_path.clone(), format!("{}", err)))?;
        let resolved = mapping
            .into_iter()
            .map(|(key, value)| (key, PathBuf::from(value)))
            .collect();
        Ok(resolved)
    })
}

fn resolve_style_path(key: &str) -> Result<PathBuf, StyleError> {
    if key.ends_with(".json") {
        let path = PathBuf::from(key);
        if path.exists() {
            return Ok(path);
        }
    }

    match style_registry() {
        Ok(registry) => registry
            .get(key)
            .cloned()
            .or_else(|| {
                let candidate = PathBuf::from(format!("styles/{key}.json"));
                if candidate.exists() {
                    Some(candidate)
                } else {
                    None
                }
            })
            .ok_or_else(|| StyleError::UnknownKey(key.to_string())),
        Err(err) => Err(err.clone()),
    }
}

pub fn load_panel_style(key: &str) -> Result<PanelStylePatch, StyleError> {
    let path = resolve_style_path(key)?;
    let text = fs::read_to_string(&path)
        .map_err(|err| StyleError::Io(path.clone(), format!("{}", err)))?;
    let schema: PanelStyleSchema = serde_json::from_str(&text)
        .map_err(|err| StyleError::Parse(path.clone(), format!("{}", err)))?;
    Ok(schema.into())
}
