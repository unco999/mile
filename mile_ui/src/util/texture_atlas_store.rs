use std::{
    collections::{
        HashMap,
        hash_map::{Iter as HashIter, IterMut as HashIterMut},
    },
    fs,
    path::Path,
};

use image::{GenericImage, ImageReader, RgbaImage};

#[derive(Clone, Default)]
pub struct TextureAtlasSet {
    pub data: HashMap<u32, TextureAtlas>,
    pub curr_ui_texture_info_index: u32,
    pub path_to_index: HashMap<String, ImageRawInfo>,
}
#[derive(Clone)]
pub struct ImageRawInfo {
    pub index: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone)]
pub struct TextureAtlas {
    pub width: u32,
    pub height: u32,
    pub data: RgbaImage, // CPU 大图
    pub map: HashMap<String, UiTextureInfo>,
    pub next_x: u32,
    pub next_y: u32,
    pub row_height: u32,
    pub texture: Option<wgpu::Texture>,
    pub texture_view: Option<wgpu::TextureView>,
    pub sampler: Option<wgpu::Sampler>,
    pub index: u32,
}

impl TextureAtlasSet {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            curr_ui_texture_info_index: 0,
            path_to_index: HashMap::new(),
        }
    }

    pub fn get_path_by_index(&self, index: u32) -> Option<String> {
        self.path_to_index.iter().find_map(|(k, v)| {
            if v.index == index {
                Some(k.clone())
            } else {
                None
            }
        })
    }

    /// 根据路径获取索引（若不存在则返回 None）
    pub fn get_index_by_path(&self, path: &str) -> Option<ImageRawInfo> {
        self.path_to_index.get(path).cloned()
    }

    /// 添加小图到指定 atlas（如果 atlas_id 不存在则创建）
    pub fn add_texture(
        &mut self,
        atlas_id: u32,
        name: &str,
        img: &RgbaImage,
        atlas_width: u32,
        atlas_height: u32,
    ) {
        let atlas = self
            .data
            .entry(atlas_id)
            .or_insert_with(|| TextureAtlas::new(atlas_width, atlas_height));
        self.curr_ui_texture_info_index += 1;
        atlas.add_sub_image(name, img, self.curr_ui_texture_info_index);
    }
}

impl TextureAtlas {
    /// 创建空 Atlas
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: RgbaImage::new(width, height),
            map: HashMap::new(),
            next_x: 0,
            next_y: 0,
            row_height: 0,
            texture: None,
            texture_view: None,
            sampler: None,
            index: 0,
        }
    }

    pub fn add_sub_image(
        &mut self,
        path: &str, // ✅ 新增参数
        img: &RgbaImage,
        index: u32,
    ) -> Option<UiTextureInfo> {
        let img_width = img.width();
        let img_height = img.height();

        // 检查是否换行
        if self.next_x + img_width > self.width {
            self.next_x = 0;
            self.next_y += self.row_height;
            self.row_height = 0;
        }

        // 超出 Atlas 大小
        if self.next_y + img_height > self.height {
            return None;
        }

        // 复制小图到大图
        self.data.copy_from(img, self.next_x, self.next_y).unwrap();

        // 更新行高
        if img_height > self.row_height {
            self.row_height = img_height;
        }

        // 计算 UV
        let uv_min = [
            self.next_x as f32 / self.width as f32,
            self.next_y as f32 / self.height as f32,
        ];
        let uv_max = [
            (self.next_x + img_width) as f32 / self.width as f32,
            (self.next_y + img_height) as f32 / self.height as f32,
        ];

        // 生成 UiTextureInfo
        let info = UiTextureInfo {
            index: self.map.len() as u32,
            uv_min,
            uv_max,
            path: path.to_string(), // ✅ 保存路径
            parent_index: self.index,
        };

        // 提取文件名作为 key（或直接用路径）
        let key = Path::new(path)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());

        self.map.insert(key, info.clone());

        // 移动下一个插入位置
        self.next_x += img_width;

        Some(info)
    }
    /// 上传大图到 GPU
    ///

    pub fn upload_to_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // 1️⃣ 创建 GPU 纹理
        let size = wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("UI Atlas Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // 2️⃣ 创建 TexelCopyTextureInfo
        let copy_texture = wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        };

        // 3️⃣ 创建 TexelCopyBufferLayout
        let buffer_layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * self.width), // RGBA8 每行字节数
            rows_per_image: Some(self.height),
        };

        // 4️⃣ Extent3d
        let extent = wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };

        // 5️⃣ 上传数据到 GPU
        queue.write_texture(copy_texture, &self.data, buffer_layout, extent);

        // 6️⃣ 创建纹理视图和采样器
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("UI Atlas Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // 7️⃣ 保存到结构体
        self.texture = Some(texture);
        self.texture_view = Some(view);
        self.sampler = Some(sampler);
    }

    /// 获取小图 UV
    pub fn get(&self, name: &str) -> Option<&UiTextureInfo> {
        self.map.get(name)
    }
}

const PADDING: u32 = 2; // 每张图像间的像素间距，防止GPU采样溢出
const DEFAULT_ATLAS_SIZE: u32 = 2048;

#[derive(Clone, Debug, PartialEq)]
pub struct UiTextureInfo {
    pub index: u32,
    pub parent_index: u32,
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub path: String,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
pub struct GpuUiTextureInfo {
    pub index: u32,        // 4
    pub parent_index: u32, // 4
    pub _pad: [u32; 2],    // 8
    pub uv_min: [f32; 4],  // 16 (vec2 + padding)
    pub uv_max: [f32; 4],  // 16 (vec2 + padding)
}

impl UiTextureInfo {
    pub fn to_gpu_struct(&self) -> GpuUiTextureInfo {
        GpuUiTextureInfo {
            index: self.index,
            uv_min: [self.uv_min[0], self.uv_min[1], 0.0, 0.0],
            uv_max: [self.uv_max[0], self.uv_max[1], 0.0, 0.0],
            parent_index: self.parent_index,
            _pad: [0u32; 2],
        }
    }
}

#[derive(Default)]
pub struct TextureAtlasStore {
    pub ui_texture_map: TextureAtlasSet,
}

impl TextureAtlasStore {
    pub fn atlas(&self, id: u32) -> Option<&TextureAtlas> {
        self.ui_texture_map.data.get(&id)
    }

    pub fn atlas_mut(&mut self, id: u32) -> Option<&mut TextureAtlas> {
        self.ui_texture_map.data.get_mut(&id)
    }

    pub fn atlases(&self) -> HashIter<'_, u32, TextureAtlas> {
        self.ui_texture_map.data.iter()
    }

    pub fn atlases_mut(&mut self) -> HashIterMut<'_, u32, TextureAtlas> {
        self.ui_texture_map.data.iter_mut()
    }

    pub fn texture_info(&self, name: &str) -> Option<&UiTextureInfo> {
        self.ui_texture_map
            .data
            .values()
            .find_map(|atlas| atlas.get(name))
    }

    pub fn raw_image_info(&self, name: &str) -> Option<&ImageRawInfo> {
        self.ui_texture_map.path_to_index.get(name)
    }

    pub fn atlas_ids_sorted(&self) -> Vec<u32> {
        let mut keys: Vec<u32> = self.ui_texture_map.data.keys().copied().collect();
        keys.sort_unstable();
        keys
    }

    pub fn texture_count(&self) -> u32 {
        self.ui_texture_map.curr_ui_texture_info_index
    }

    pub fn upload_all_to_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        for (_id, atlas) in self.ui_texture_map.data.iter_mut() {
            atlas.upload_to_gpu(device, queue);
        }
    }

    pub fn collect_texture_views(&self) -> Vec<wgpu::TextureView> {
        self.ui_texture_map
            .data
            .values()
            .filter_map(|atlas| atlas.texture_view.clone())
            .collect()
    }

    pub fn collect_samplers(&self) -> Vec<wgpu::Sampler> {
        self.ui_texture_map
            .data
            .values()
            .filter_map(|atlas| atlas.sampler.clone())
            .collect()
    }

    pub fn build_gpu_texture_infos(&self) -> Vec<GpuUiTextureInfo> {
        let len = self.ui_texture_map.curr_ui_texture_info_index as usize;
        if len == 0 {
            return Vec::new();
        }

        let mut infos = vec![GpuUiTextureInfo::default(); len];
        for atlas in self.ui_texture_map.data.values() {
            for info in atlas.map.values() {
                let idx = info.index as usize;
                if idx < len {
                    infos[idx] = info.to_gpu_struct();
                }
            }
        }
        infos
    }

    pub fn build_gpu_texture_infos_with_slots(
        &self,
        slot_map: &HashMap<u32, u32>,
    ) -> Vec<GpuUiTextureInfo> {
        let len = self.ui_texture_map.curr_ui_texture_info_index as usize;
        if len == 0 {
            return Vec::new();
        }

        let mut infos = vec![GpuUiTextureInfo::default(); len];
        for atlas in self.ui_texture_map.data.values() {
            let slot = slot_map.get(&atlas.index).copied();
            for info in atlas.map.values() {
                let idx = info.index as usize;
                if idx >= len {
                    continue;
                }
                let mut gpu_info = info.to_gpu_struct();
                gpu_info.parent_index = slot.unwrap_or(u32::MAX);
                infos[idx] = gpu_info;
            }
        }
        infos
    }

    pub fn read_all_image(&mut self) {
        // 遍历 ./texture 目录
        let texture_dir = Path::new("./texture");
        if !texture_dir.exists() {
            eprintln!("纹理目录 {:?} 不存在", texture_dir);
            return;
        }

        // 收集所有支持的图片文件
        let supported_ext = ["png", "jpg", "jpeg", "bmp"];

        let mut image_paths = Vec::new();
        if let Ok(entries) = fs::read_dir(texture_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if supported_ext.contains(&ext.to_lowercase().as_str()) {
                        image_paths.push(path);
                    }
                }
            }
        }

        if image_paths.is_empty() {
            println!("未找到任何纹理文件");
            return;
        }

        // 逐个调用 gpu_ui.read_img
        for path in image_paths {
            println!("读取纹理文件: {:?}", path);
            self.read_img(path.as_path());
        }
    }

    pub fn read_img(&mut self, path: &Path) -> Option<UiTextureInfo> {
        // 1️⃣ 打开图片
        let img = ImageReader::open(path).ok()?.decode().ok()?.to_rgba8();
        let (orig_w, orig_h) = img.dimensions();
        println!("🖼️ 加载图片 {:?}, 大小: {}x{}", path, orig_w, orig_h);

        // 添加边距后的尺寸
        let img_width = orig_w + PADDING * 2;
        let img_height = orig_h + PADDING * 2;

        // 2️⃣ 选择可容纳图片的 atlas
        let atlas_id = if let Some((&id, _)) =
            self.ui_texture_map.data.iter_mut().find(|(_, atlas)| {
                let mut x = atlas.next_x;
                let mut y = atlas.next_y;
                let mut row_height = atlas.row_height;

                // 模拟多次换行，直到找到放得下的位置或确定放不下
                loop {
                    if x + img_width > atlas.width {
                        x = 0;
                        y += row_height;
                        row_height = 0;
                    }

                    if y + img_height > atlas.height {
                        return false; // 放不下
                    }

                    if x + img_width <= atlas.width {
                        return true; // 找到可以放的位置
                    }
                }
            }) {
            id
        } else {
            // 没有合适的 atlas，新建一个
            let atlas_size = DEFAULT_ATLAS_SIZE;
            let new_id = self.ui_texture_map.data.len() as u32;
            println!(
                "🆕 创建新的 Atlas #{} 尺寸 {}x{}",
                new_id, atlas_size, atlas_size
            );

            let atlas = TextureAtlas {
                width: atlas_size,
                height: atlas_size,
                data: RgbaImage::new(atlas_size, atlas_size),
                map: HashMap::new(),
                next_x: 0,
                next_y: 0,
                row_height: 0,
                texture: None,
                texture_view: None,
                sampler: None,
                index: new_id,
            };

            self.ui_texture_map.data.insert(new_id, atlas);
            new_id
        };

        // 3️⃣ 获取可用 atlas
        let atlas = self.ui_texture_map.data.get_mut(&atlas_id).unwrap();

        // 4️⃣ 计算插入坐标（支持自动换行）
        let (mut x, mut y) = (atlas.next_x, atlas.next_y);
        if x + img_width > atlas.width {
            x = 0;
            y += atlas.row_height;
            atlas.next_y = y;
            atlas.row_height = 0;
        }

        // 检查是否溢出
        if y + img_height > atlas.height {
            println!("⚠️ Atlas #{} 已满，无法放入 {:?}", atlas.index, path);
            return None;
        }

        // overlay
        image::imageops::overlay(&mut atlas.data, &img, x.into(), y.into());

        // 更新游标
        atlas.next_x = x + img_width;
        atlas.row_height = atlas.row_height.max(img_height);

        // 7️⃣ 计算UV（去除 padding）
        let uv_min = [
            (x + PADDING) as f32 / atlas.width as f32,
            (y + PADDING) as f32 / atlas.height as f32,
        ];
        let uv_max = [
            (x + PADDING + orig_w) as f32 / atlas.width as f32,
            (y + PADDING + orig_h) as f32 / atlas.height as f32,
        ];

        // 8️⃣ 生成或复用 UiTextureInfo
        let tex_name = path.file_name()?.to_string_lossy().to_string();
        if let Some(existing) = atlas.map.get(&tex_name) {
            println!("♻️ 已存在纹理 {:?} (atlas #{})", tex_name, atlas.index);
            return Some(existing.clone());
        }

        let tex_index = self.ui_texture_map.curr_ui_texture_info_index;
        self.ui_texture_map.curr_ui_texture_info_index += 1;

        let ui_info = UiTextureInfo {
            index: tex_index,
            uv_min,
            uv_max,
            path: tex_name.clone(),
            parent_index: atlas.index,
        };
        println!(
            "插入 {:?}: pos=({}, {}), next_x={}, row_height={}, atlas_size={}x{}",
            path, x, y, atlas.next_x, atlas.row_height, atlas.width, atlas.height
        );
        // 9️⃣ 注册缓存
        atlas.map.insert(tex_name.clone(), ui_info.clone());
        self.ui_texture_map.path_to_index.insert(
            tex_name.clone(),
            ImageRawInfo {
                index: tex_index,
                width: img_width,
                height: img_height,
            },
        );

        println!(
            "✅ 插入纹理 {:?} → index:{} Atlas:{} 坐标:({}, {})",
            tex_name, tex_index, atlas.index, x, y
        );

        Some(ui_info)
    }
}
