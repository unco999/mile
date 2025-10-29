use gpu_macro::wgsl;

use crate::{mat::op::simulate_matrix_plan_batch};

#[test]

fn simulate_gpu_test(){
    use crate::core::dsl::*; // 或按你原来路径
    use crate::mat::op::compile_to_matrix_plan;
    use crate::core::*;
// 使用 e! 宏构造比较复杂表达式：
    let ex = vec3(3.0f32, 1.0f32, 3.0f32) * 1 * 1 * vec3(2.0, 3.0,3.0);
    let plan = compile_to_matrix_plan(&ex, &["a","b","c","d"]);
    let inputs = vec![
        vec![1.0_f32], // a
        vec![3.0_f32], // b
        vec![5.0_f32], // c
        vec![7.0_f32], // d
    ];
    let outputs = simulate_matrix_plan_batch(&plan, &inputs);
    println!("batch 输出: {:?}", outputs);
}