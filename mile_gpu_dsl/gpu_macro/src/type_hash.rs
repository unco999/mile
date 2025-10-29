/// 宏：为一个 struct 或 bitflags 自动生成类型 hash
#[proc_macro_derive(TypeHash)]
pub fn type_hash_derive(input: TokenStream) -> TokenStream {
    // 解析输入 TokenStream 为 AST
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    // 使用类型名字生成 const HASH
    let expanded = quote! {
        impl #name {
            pub const HASH: u32 = {
                const BYTES: &[u8] = stringify!(#name).as_bytes();
                let mut hash = 0x811C9DC5u32;
                let mut i = 0;
                while i < BYTES.len() {
                    hash ^= BYTES[i] as u32;
                    hash = hash.wrapping_mul(0x01000193);
                    i += 1;
                }
                hash
            };
        }
    };

    // 转回 TokenStream 输出
    TokenStream::from(expanded)
}