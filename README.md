# Mile Engine 🌍

> **Language**: [English](#README.md) | [中文](#README_zh.md)

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
