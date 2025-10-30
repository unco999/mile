use gpu_macro::wgsl;

use crate::{dsl::{ if_expr}, mat::op::{simulate_matrix_plan_batch, simulate_matrix_plan_batch_generic}};


#[test]
fn vec4_test(){
    use crate::core::dsl::*; // 或按你原来路径
    use crate::mat::op::compile_to_matrix_plan;
    use crate::core::*;

    let expr = wvec4(1.0,1.0,1.0,5.0) + wvec4(2.0, 2.0, 2.0,5.0);
    
    let plan = compile_to_matrix_plan(&expr, &["a","b","c","d"]);
    let inputs = vec![
        vec![-2.0_f32], // a
        vec![3.0_f32], // b
        vec![5.0_f32], // c
        vec![7.0_f32], // d
    ];
    let outputs = simulate_matrix_plan_batch_generic(&plan, &inputs);
    println!("batch 输出: {:?}", outputs);
}

// #[test]
// fn simulate_gpu_test(){
//     use crate::core::dsl::*; // 或按你原来路径
//     use crate::mat::op::compile_to_matrix_plan;
//     use crate::core::*;


//     let ex = vec3(3.0f32, 1.0f32, 
//         if_expr(var("a").eq(3.0 - 1.0 * var("c")), 3.0, 5.0))
//          * 1 * 1 * vec3(2.0, 3.0,3.0) + var("a");

//     let plan = compile_to_matrix_plan(&ex, &["a","b","c","d"]);
//     let inputs = vec![
//         vec![-2.0_f32], // a
//         vec![3.0_f32], // b
//         vec![5.0_f32], // c
//         vec![7.0_f32], // d
//     ];
//     let outputs = simulate_matrix_plan_batch_generic(&plan, &inputs);
//     println!("batch 输出: {:?}", outputs);
// }