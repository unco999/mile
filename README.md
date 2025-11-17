# mile

现代化的 Rust UI/GPU 实验项目，提供实时渲染、UI 交互与 GPU DSL 示例。workspace 包含核心引擎（mile_core）、UI（mile_ui）、字体（mile_font）、API（mile_api）、图形（mile_graphics）、GPU DSL（mile_gpu_dsl）等 crate。

A modern Rust UI/GPU playground with real-time rendering, UI interaction, and a small GPU DSL. Workspace includes core engine, UI, font, API, graphics, and GPU DSL crates.

## 快速开始 / Quick Start
- 构建：`cargo build --workspace`
- 运行主引擎：`cargo run -p mile_core`
- 单独运行示例（example crate）：
  - 基础布局 Basic Layout: `cargo run -p example --bin showcase_basic_layout`
  - 动画 Animation: `cargo run -p example --bin showcase_animation`
  - 响应式 Responsive: `cargo run -p example --bin showcase_responsive`
  - 文本 Text: `cargo run -p example --bin showcase_text`
  - AST Shader (GPU DSL): `cargo run -p example --bin showcase_ast_shader`
- 代码格式：`cargo fmt --all`
- 静态检查：`cargo clippy --workspace --all-targets -- -D warnings`

## 案例概览 / Showcase Overview

### 1) 基础布局 / Basic Layout
- 描述：容器 + 子项的基础布局演示，展示位置、尺寸、颜色和事件绑定。
- 亮点：容器约束、子项 hover 动画。
- 入口：`example/src/bin/showcase_basic_layout.rs`

### 2) 动画 / Animation
- 描述：面板在 hover/click 时执行位置与颜色动画，展示缓动曲线。
- 亮点：`position_anim` / `color_anim` 与 Quadratic/Cubic easing。
- 入口：`example/src/bin/showcase_animation.rs`

### 3) 响应式 / Responsive
- 描述：利用容器百分比和 clamp 规则，实现横向拖拽滑块与动态布局。
- 亮点：`clamp_offset(Field::OnlyPositionX, …)`、拖拽事件 `on_event_with(UiEventKind::Drag, ...)`。
- 入口：`example/src/bin/showcase_responsive.rs`

### 4) 文本 / Text
- 描述：SDF 字体渲染与事件驱动文本更新。
- 亮点：`flow.text(...)` 调度字体加载与渲染，点击更新内容与样式。
- 入口：`example/src/bin/showcase_text.rs`

### 5) AST Shader (GPU DSL)
- 描述：使用 mile_gpu_dsl 构建片元着色器 AST，运行时生成波形渐变。
- 亮点：DSL 表达式（`sin`、`smoothstep`、`wvec4`）、运行时 shader 请求。
- 入口：`example/src/bin/showcase_ast_shader.rs`

### 6) 状态轮换 / Text
- 描述：状态轮换机制。
- 亮点：`state(UiState(x))` 状态轮换自动会在样式描述中插值。
- 入口：`example/src/bin/showcase_state.rs`


## 架构概览 / Architecture
- `mile_core`: 事件循环、WGPU 上下文、UI/字体/DSL 运行时装配。
- `mile_ui`: UI 原型系统，包含面板 DSL、事件流、动画、交互 compute pipeline。
- `mile_font`: SDF 字体运行时与渲染。
- `mile_gpu_dsl`: GPU AST/管线定义，支持 compute/fragment shader 生成。
- `mile_graphics`: 渲染原语与 wgpu 封装。
- `mile_api`: 共享类型与接口。
- 资产：`tf/` (字体)、`texture/` (纹理)、`markdown/` (文档/截图)。

## 开发提示 / Dev Notes
- 交互：拖拽/点击/悬停事件通过 GPU 交互缓冲回传 CPU，再触发 `on_event`/`on_event_with`。
- 动画：动画字段与缓动参数在 UI DSL 内声明，runtime 每帧刷新。
- 关系/布局：`mui_rel` 提供容器与相对布局；`clamp_offset` 支持相对预算与轴向锁定。
- 调试：如未收到交互事件，检查 panel 的 `interaction` 掩码（由事件声明自动设置）和 `PanelInteraction` 写入是否刷新。
