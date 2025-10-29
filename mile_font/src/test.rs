use ttf_parser::{Face, GlyphId, OutlineBuilder};
use std::fs;
use std::path::Path;

use crate::structs::MileFont;

/// OutlineBuilder 实现：打印字形轮廓
struct PrintOutline;

impl OutlineBuilder for PrintOutline {
    fn move_to(&mut self, x: f32, y: f32) {
        print!("M({:.1},{:.1}) ", x, y);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        print!("L({:.1},{:.1}) ", x, y);
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        print!("Q({:.1},{:.1} -> {:.1},{:.1}) ", x1, y1, x, y);
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        print!("C({:.1},{:.1} -> {:.1},{:.1} -> {:.1},{:.1}) ", x1, y1, x2, y2, x, y);
    }
    fn close(&mut self) {
        print!("Z ");
    }
}

#[test]
fn main() {

}

