# Repository Guidelines

## Project Structure & Module Organization
This workspace spans six crates defined in Cargo.toml. mile_core/src/main.rs orchestrates the application event loop and is the entry point for running demos. GPU compute infrastructure lives in mile_gpu_dsl/src, with AST, kernel, and pipeline logic inside mat/. Rendering primitives and windowing glue sit in mile_graphics, while mile_ui and mile_font provide UI composition and SDF font rendering. Shared APIs are in mile_api. Asset packs and showcase media reside under 	exture/, 	tf/, and markdown/; keep large experimental assets outside the repository.

## Build, Test, and Development Commands
cargo build --workspace compiles every crate with shared dependencies resolved. Use cargo run -p mile_core to launch the engine shell from this root. cargo test -p mile_gpu_dsl currently covers the expression compiler; add -p <crate> for crate-specific suites. Run cargo fmt --all before committing to enforce rustfmt defaults, and cargo clippy --workspace --all-targets -- -D warnings to catch GPU and UI regressions early.

## Coding Style & Naming Conventions
Default rustfmt settings (4-space indent, trailing commas) are the source of truth. Follow Rust 2024 idioms: snake_case for functions and modules, PascalCase for types, screaming SCREAMING_SNAKE_CASE for constants. Keep modules small enough to mirror the directory layout, and prefer explicit pub(crate) visibility. When extending GPU kernels, document buffer layouts with concise comments near the struct or enum definition.

## Testing Guidelines
Unit tests live alongside implementation files using #[cfg(test)]; mirror that pattern for new modules. Prefer deterministic buffer fixtures so CI-safe tests do not depend on GPU timing. Expose new DSL features through focused tests in mile_gpu_dsl/src/mat, and add integration smoke tests in crate-level 	ests/ directories when host/device coordination is required. Validate failures with RUST_BACKTRACE=1 cargo test before opening a review.

## Commit & Pull Request Guidelines
Commits in this repository use short present-tense summaries (often Chinese) such as ³É¹¦°ó¶¨ render. Keep subjects under ~50 characters and elaborate in the body when behavior changes. Each PR should describe the user-facing impact, list touched crates, link related issues, and include screenshots or GIFs when UI or rendering output shifts. Call out GPU shader or asset changes explicitly so reviewers can re-run the correct demo.

## Assets & Configuration Tips
Check new fonts into 	tf/ and textures into 	exture/, maintaining lightweight source formats. Update markdown/ with refreshed captures when demos change. Avoid committing generated content under 	arget/; clean builds with cargo clean if artifacts slip into diffs.