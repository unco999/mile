# Mile Engine 🌍
[[使用过程]([https://www.bilibili.com/video/视频ID](https://www.bilibili.com/video/BV14n1ABvEes/))](https://www.bilibili.com/video/BV14n1ABvEes/)
> **Language**: [English](#english-version) | [中文](#chinese-version)
<div id="english-version">
**Mile Engine** is a game and graphics development engine that heavily relies on GPU parallel computing.  
Currently in an unstable early construction phase.

## 🚀 Running the Engine

Project entry point is at `mile_core/src/main.rs`:

```bash
# Enter the core module
cd mile/mile_core
```

![Demo GIF](https://github.com/unco999/mile/blob/master/markdown/01.gif)

## 🖌 Mile UI – GPU-Parallel Stateful Interface
Mile UI is an innovative UI framework that enables high-performance, composable, stateful interface elements and collection operations through GPU parallel computing.

### Core Concepts
- **GPU-Driven**: All UI state, animation, and layout computations are executed in parallel for maximum performance.
- **State Machine Panels**: Each Panel is bound to a state machine, responding to events like clicks, hovers, and drags.
- **Collections**: Independently existing UI element groups supporting center point calculation, group animations, and multiple arrangement methods.
- **Relationship Components**: Define spatial, animation, constraint, and weight relationships between collections to create complex UI networks.
- **Custom Animations & Entrance/Exit Modes**: Supports user-defined offsets, entrance/exit animations, and state transitions, with instant mode as default.

## 🖋 Mile Font – GPU-Accelerated Distance Field Font Rendering
Mile Font is a GPU-accelerated font rendering module, currently based on Signed Distance Field (SDF) technology, providing high-performance text display and basic typesetting.

### Core Concepts
- **GPU Parallel Computing**: All font rendering operations are processed through GPU parallel computing, supporting dynamic adjustments and large-scale text rendering.
- **Distance Field Rendering**: Utilizes SDF technology for smooth edges and scaling without quality loss.
- **Future Effects Expansion**: Currently focuses on basic fonts, with plans to support special effect fonts, gradients, outlines, shadows, and animation effects.
- **Flexible Texture Management**: Supports font texture uploading, caching, and dynamic composition, providing underlying font support for UI and graphics modules.

## ⚡ Mile GPU DSL – Arbitrary Expression Compilation & Execution on GPU DAG
Mile GPU DSL is a GPU-accelerated computing module that compiles arbitrary mathematical expressions and algorithms into Directed Acyclic Graphs (DAG) for efficient parallel computation on GPUs.

### Core Concepts
- **Universal Expressions**: Supports any nested mathematical formulas including:
  - Binary operations: addition, subtraction, multiplication, division, modulus, exponentiation, comparison (> >= < <= == !=)
  - Unary operations: trigonometric functions, exponents, logarithms, square roots, absolute values
  - Conditional logic: if/else implemented via matrix routing (CondBlend) for batch parallel processing
- **GPU DAG Compilation**: Converts expressions into GPU kernel sequences, mapping each intermediate result to buffers, ensuring correct computation order through DAG dependencies.
- **Lazy Function Calls**: All functions and operations are registered as kernels in the DAG, with unified GPU parallel scheduling during actual execution.
- **Batch Computing**: The same DAG can process multiple input vectors, enabling vectorization and parallel processing.

## 📄 License
Mile Engine is developed by unco999 and open-sourced under the MIT License.
</div>
<div id="chinese-version"># 🌀 Mile 引擎

Mile 引擎 是一个高度依赖 GPU 并行运算的游戏和图形开发引擎。  
目前处于不稳定的前期构造环节。

## 🚀 运行引擎
项目入口在 `mile_core/src/main.rs`：

```bash
# 进入核心模块
cd mile/mile_core
```

![Demo GIF](https://github.com/unco999/mile/blob/master/markdown/01.gif)

## 🖌 Mile UI – GPU 并行驱动的状态化界面
Mile UI 是一套创新的 UI 框架，通过 GPU 并行计算实现高性能、可组合、状态化的界面元素与集合操作。

### 核心理念
- **GPU 驱动**：所有 UI 状态、动画和布局运算均可并行执行，性能极高。
- **状态机面板**：每个 Panel 都绑定状态机，响应点击、悬浮、拖拽等事件。
- **集合 (Collection)**：独立存在的 UI 元素组合，支持中心点计算、群体动画和多种排列方式。
- **关系组件**：定义集合间的空间、动画、约束与权重关系，实现复杂的 UI 网络。
- **自定义动画与进出模式**：支持用户自定义偏移、进出动画和状态过渡，默认即时模式。

## 🖋 Mile Font – GPU 驱动的距离场字体渲染
Mile Font 是一个 GPU 加速的字体渲染模块，目前基于距离场 (Signed Distance Field, SDF) 技术，实现高性能的文本显示与基础排版。

### 核心理念
- **GPU 并行计算**：所有字体渲染运算通过 GPU 并行处理，支持动态调整和大规模文本渲染。
- **距离场渲染**：利用 SDF 技术，实现平滑边缘和缩放无损的字体渲染。
- **未来特效扩展**：目前以基础字体为主，后续计划支持特效字库、渐变、描边、阴影和动画效果。
- **灵活纹理管理**：支持字体纹理上传、缓存和动态组合，为 UI 与图形模块提供底层字体支持。

## ⚡ Mile GPU DSL – 任意表达式的 GPU DAG 编译与执行
Mile GPU DSL 是一个 GPU 加速计算模块，核心是将任意数学表达式和算法公式编译成 有向无环图 (DAG, Directed Acyclic Graph)，并在 GPU 上进行高效并行计算。

### 核心理念
- **通用表达式**：支持任意嵌套的数学公式，包括：
  - 二元运算：加减乘除、模、幂、比较（> >= < <= == !=）
  - 一元运算：三角函数、指数、对数、开方、绝对值等
  - 条件判断：if/else 通过矩阵路由 (CondBlend) 实现批量并行
- **GPU DAG 编译**：将表达式转换为 GPU kernel 序列，每个中间结果映射到 buffer，通过 DAG 依赖保证计算顺序正确。
- **延迟函数调用**：所有函数和操作在 DAG 内注册为 kernel，真正执行时统一调度 GPU 并行处理。
- **批量计算 (Batch)**：同一个 DAG 可处理多个输入向量，实现向量化和并行处理。

## 📄 许可证
Mile 引擎由 unco999 开发，开源许可遵循 MIT License。
</div> 
