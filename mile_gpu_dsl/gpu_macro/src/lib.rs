// gpu_macro/src/lib.rs
extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};


// 宏生成 trait impl
#[proc_macro_derive(TypeHash)]
pub fn type_hash_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let type_str = name.to_string();
    let bytes: Vec<u8> = type_str.bytes().collect();

    let mut hash_expr = quote! { 0x811C9DC5u32 };
    for b in bytes {
        let b_val = b as u32;
        hash_expr = quote! {
            (#hash_expr ^ #b_val).wrapping_mul(0x01000193)
        };
    }

    let expanded = quote! {
        impl TypeHash for #name {
            const HASH: u32 = #hash_expr;
        }
    };
    TokenStream::from(expanded)
}
