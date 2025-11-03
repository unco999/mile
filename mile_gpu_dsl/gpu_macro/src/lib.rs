// wgsl_macro/src/lib.rs
// Procedural macro crate: wgsl!("...")
// Usage: add this crate as a proc-macro dependency, then in your main crate:
// use wgsl_macro::wgsl;
// let ex = wgsl!("vec3(3.0,1.0,3.0) * 1.0 + vec3(3.0,1.0, if((3.0>1.0),3.0,10.0))");

extern crate proc_macro;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{LitStr, parse_macro_input};

#[proc_macro]
pub fn wgsl(input: TokenStream) -> TokenStream {
    // This macro expects a single string literal containing a WGSL-like expression.
    // Reason: allows using `if(...)` and other WGSL-style tokens easily inside the string.
    let s = parse_macro_input!(input as LitStr).value();
    match parse_wgsl(&s) {
        Ok(pe) => {
            let ts = pe.to_tokenstream();
            TokenStream::from(ts)
        }
        Err(e) => panic!("wgsl! parse error: {}", e),
    }
}

// ----------------- Lexer -----------------
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Number(f32),
    Ident(String),
    Op(String),
    LParen,
    RParen,
    Comma,
    EOF,
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}
fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn lex(s: &str) -> Vec<Tok> {
    let mut i = 0usize;
    let bytes = s.as_bytes();
    let n = s.len();
    let mut out = Vec::new();
    while i < n {
        let c = s[i..].chars().next().unwrap();
        if c.is_whitespace() {
            i += c.len_utf8();
            continue;
        }
        if is_ident_start(c) {
            let mut j = i + c.len_utf8();
            while j < n {
                let cc = s[j..].chars().next().unwrap();
                if !is_ident_continue(cc) {
                    break;
                }
                j += cc.len_utf8();
            }
            let ident = &s[i..j];
            out.push(Tok::Ident(ident.to_string()));
            i = j;
            continue;
        }
        if c.is_ascii_digit()
            || (c == '.' && i + 1 < n && s[i + 1..].chars().next().unwrap().is_ascii_digit())
        {
            // number (simple parser floats & ints)
            let mut j = i;
            let mut seen_dot = false;
            while j < n {
                let cc = s[j..].chars().next().unwrap();
                if cc == '.' {
                    if seen_dot {
                        break;
                    }
                    seen_dot = true;
                    j += cc.len_utf8();
                    continue;
                }
                if cc.is_ascii_digit() || cc == 'e' || cc == 'E' || cc == '+' || cc == '-' {
                    j += cc.len_utf8();
                    continue;
                }
                break;
            }
            let num_str = &s[i..j];
            if let Ok(v) = num_str.parse::<f32>() {
                out.push(Tok::Number(v));
            } else {
                // fallback: try removing trailing chars
                let trimmed = num_str.trim_end_matches(|ch: char| {
                    !ch.is_numeric()
                        && ch != '.'
                        && ch != 'e'
                        && ch != 'E'
                        && ch != '+'
                        && ch != '-'
                });
                let v = trimmed.parse::<f32>().unwrap_or(0.0);
                out.push(Tok::Number(v));
            }
            i = j;
            continue;
        }
        // two-char ops
        if i + 1 < n {
            let two = &s[i..i + 2];
            match two {
                ">=" | "<=" | "==" | "!=" => {
                    out.push(Tok::Op(two.to_string()));
                    i += 2;
                    continue;
                }
                _ => {}
            }
        }
        match c {
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            ',' => {
                out.push(Tok::Comma);
                i += 1;
            }
            '+' | '-' | '*' | '/' | '%' | '>' | '<' => {
                out.push(Tok::Op(c.to_string()));
                i += 1;
            }
            _ => {
                // unknown - treat as ident char sequence
                let mut j = i + c.len_utf8();
                while j < n {
                    let cc = s[j..].chars().next().unwrap();
                    if cc.is_whitespace() || cc == '(' || cc == ')' || cc == ',' {
                        break;
                    }
                    j += cc.len_utf8();
                }
                let chunk = &s[i..j];
                out.push(Tok::Ident(chunk.to_string()));
                i = j;
            }
        }
    }
    out.push(Tok::EOF);
    out
}

// ----------------- Parser -----------------
#[derive(Debug, Clone)]
enum PExpr {
    Const(f32),
    Var(String),
    Binary {
        op: String,
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    UnaryFunc {
        name: String,
        a: Box<PExpr>,
    },
    Vec2(Box<PExpr>, Box<PExpr>),
    Vec3(Box<PExpr>, Box<PExpr>, Box<PExpr>),
    Vec4(Box<PExpr>, Box<PExpr>, Box<PExpr>, Box<PExpr>),
    If(Box<PExpr>, Box<PExpr>, Box<PExpr>),
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}
impl Parser {
    fn new(toks: Vec<Tok>) -> Self {
        Self { toks, pos: 0 }
    }
    fn peek(&self) -> &Tok {
        &self.toks[self.pos]
    }
    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos].clone();
        self.pos += 1;
        t
    }
    fn expect_ident(&mut self) -> Result<String, String> {
        match self.bump() {
            Tok::Ident(s) => Ok(s),
            t => Err(format!("expected ident, got {:?}", t)),
        }
    }

    fn parse(&mut self) -> Result<PExpr, String> {
        self.parse_expr()
    }

    // expr -> cmp
    fn parse_expr(&mut self) -> Result<PExpr, String> {
        self.parse_cmp()
    }

    // cmp -> add ((==|!=|>|>=|<|<=) add)*
    fn parse_cmp(&mut self) -> Result<PExpr, String> {
        let mut node = self.parse_add()?;
        loop {
            match self.peek() {
                Tok::Op(op) if ["==", "!=", ">=", "<=", ">", "<"].contains(&op.as_str()) => {
                    let op = if let Tok::Op(s) = self.bump() {
                        s
                    } else {
                        "".to_string()
                    };
                    let right = self.parse_add()?;
                    node = PExpr::Binary {
                        op,
                        a: Box::new(node),
                        b: Box::new(right),
                    };
                }
                _ => break,
            }
        }
        Ok(node)
    }

    // add -> mul ( (+|-) mul )*
    fn parse_add(&mut self) -> Result<PExpr, String> {
        let mut node = self.parse_mul()?;
        loop {
            match self.peek() {
                Tok::Op(op) if op == "+" || op == "-" => {
                    let op = if let Tok::Op(s) = self.bump() {
                        s
                    } else {
                        "".to_string()
                    };
                    let right = self.parse_mul()?;
                    node = PExpr::Binary {
                        op,
                        a: Box::new(node),
                        b: Box::new(right),
                    };
                }
                _ => break,
            }
        }
        Ok(node)
    }

    // mul -> unary ( (*|/|%) unary )*
    fn parse_mul(&mut self) -> Result<PExpr, String> {
        let mut node = self.parse_unary()?;
        loop {
            match self.peek() {
                Tok::Op(op) if op == "*" || op == "/" || op == "%" => {
                    let op = if let Tok::Op(s) = self.bump() {
                        s
                    } else {
                        "".to_string()
                    };
                    let right = self.parse_unary()?;
                    node = PExpr::Binary {
                        op,
                        a: Box::new(node),
                        b: Box::new(right),
                    };
                }
                _ => break,
            }
        }
        Ok(node)
    }

    // unary -> ( - | + ) unary | primary
    fn parse_unary(&mut self) -> Result<PExpr, String> {
        match self.peek() {
            Tok::Op(op) if op == "-" => {
                self.bump();
                let v = self.parse_unary()?;
                Ok(PExpr::Binary {
                    op: "*".to_string(),
                    a: Box::new(PExpr::Const(-1.0)),
                    b: Box::new(v),
                })
            }
            Tok::Op(op) if op == "+" => {
                self.bump();
                self.parse_unary()
            }
            _ => self.parse_postfix(),
        }
    }

    // postfix handles function calls like ident(...)
    fn parse_postfix(&mut self) -> Result<PExpr, String> {
        let mut node = self.parse_primary()?;
        // no postfix operators for now
        Ok(node)
    }

    fn parse_primary(&mut self) -> Result<PExpr, String> {
        match self.peek().clone() {
            Tok::Number(v) => {
                self.bump();
                Ok(PExpr::Const(v))
            }
            Tok::Ident(ref s) => {
                // function call vs variable
                let name = s.clone();
                self.bump();
                match self.peek() {
                    Tok::LParen => {
                        self.bump(); // consume '('
                        // parse comma separated args until RParen
                        let mut args = Vec::new();
                        if let Tok::RParen = self.peek() {
                            self.bump();
                        } else {
                            loop {
                                let e = self.parse_expr()?;
                                args.push(e);
                                match self.peek() {
                                    Tok::Comma => {
                                        self.bump();
                                        continue;
                                    }
                                    Tok::RParen => {
                                        self.bump();
                                        break;
                                    }
                                    t => return Err(format!("expected , or ), got {:?}", t)),
                                }
                            }
                        }
                        // handle special builtins
                        match name.as_str() {
                            "vec2" => {
                                if args.len() != 2 {
                                    return Err("vec2 expects 2 args".into());
                                }
                                Ok(PExpr::Vec2(
                                    Box::new(args.remove(0)),
                                    Box::new(args.remove(0)),
                                ))
                            }
                            "vec3" => {
                                if args.len() != 3 {
                                    return Err("vec3 expects 3 args".into());
                                }
                                Ok(PExpr::Vec3(
                                    Box::new(args.remove(0)),
                                    Box::new(args.remove(0)),
                                    Box::new(args.remove(0)),
                                ))
                            }
                            "vec4" => {
                                if args.len() != 4 {
                                    return Err("vec4 expects 4 args".into());
                                }
                                Ok(PExpr::Vec4(
                                    Box::new(args.remove(0)),
                                    Box::new(args.remove(0)),
                                    Box::new(args.remove(0)),
                                    Box::new(args.remove(0)),
                                ))
                            }
                            "if" => {
                                if args.len() != 3 {
                                    return Err("if(cond,then,else) expects 3 args".into());
                                }
                                Ok(PExpr::If(
                                    Box::new(args.remove(0)),
                                    Box::new(args.remove(0)),
                                    Box::new(args.remove(0)),
                                ))
                            }
                            "sin" | "cos" | "sqrt" | "tan" | "exp" | "log" | "abs" => {
                                if args.len() != 1 {
                                    return Err(format!("{} expects 1 arg", name));
                                }
                                Ok(PExpr::UnaryFunc {
                                    name,
                                    a: Box::new(args.remove(0)),
                                })
                            }
                            _ => {
                                // generic function call not supported -> treat as variable? or error
                                // We'll treat unknown identifier with parentheses as an error for now
                                return Err(format!("unknown function: {}", name));
                            }
                        }
                    }
                    _ => {
                        // plain variable
                        Ok(PExpr::Var(name))
                    }
                }
            }
            Tok::LParen => {
                self.bump();
                let e = self.parse_expr()?;
                match self.peek() {
                    Tok::RParen => {
                        self.bump();
                        Ok(e)
                    }
                    _ => Err("expected )".into()),
                }
            }
            t => Err(format!("unexpected token: {:?}", t)),
        }
    }
}

fn parse_wgsl(s: &str) -> Result<PExpr, String> {
    let toks = lex(s);
    let mut p = Parser::new(toks);
    let e = p.parse()?;
    Ok(e)
}

// ---------------- codegen ----------------
impl PExpr {
    fn to_tokenstream(&self) -> proc_macro2::TokenStream {
        match self {
            PExpr::Const(v) => {
                let lit = proc_macro2::Literal::f64_suffixed((*v) as f64);
                quote! { crate::test::Expr::Constant(#lit as f32) }
            }
            PExpr::Var(name) => {
                quote! { crate::test::Expr::Variable(#name.to_string()) }
            }
            PExpr::Binary { op, a, b } => {
                let a_ts = a.to_tokenstream();
                let b_ts = b.to_tokenstream();
                let bop = match op.as_str() {
                    "+" => quote! { crate::test::BinaryOp::Add },
                    "-" => quote! { crate::test::BinaryOp::Subtract },
                    "*" => quote! { crate::test::BinaryOp::Multiply },
                    "/" => quote! { crate::test::BinaryOp::Divide },
                    "%" => quote! { crate::test::BinaryOp::Modulo },
                    ">" => quote! { crate::test::BinaryOp::GreaterThan },
                    ">=" => quote! { crate::test::BinaryOp::GreaterEqual },
                    "<" => quote! { crate::test::BinaryOp::LessThan },
                    "<=" => quote! { crate::test::BinaryOp::LessEqual },
                    "==" => quote! { crate::test::BinaryOp::Equal },
                    "!=" => quote! { crate::test::BinaryOp::NotEqual },
                    _ => panic!("unsupported binary op {}", op),
                };
                quote! {
                    crate::test::Expr::BinaryOp(#bop, Box::new(#a_ts), Box::new(#b_ts))
                }
            }
            PExpr::UnaryFunc { name, a } => {
                let a_ts = a.to_tokenstream();
                let u = match name.as_str() {
                    "sin" => quote! { crate::test::UnaryFunc::Sin },
                    "cos" => quote! { crate::test::UnaryFunc::Cos },
                    "sqrt" => quote! { crate::test::UnaryFunc::Sqrt },
                    "tan" => quote! { crate::test::UnaryFunc::Tan },
                    "exp" => quote! { crate::test::UnaryFunc::Exp },
                    "log" => quote! { crate::test::UnaryFunc::Log },
                    "abs" => quote! { crate::test::UnaryFunc::Abs },
                    _ => panic!("unknown unary func {}", name),
                };
                quote! { crate::test::Expr::UnaryOp(#u, Box::new(#a_ts)) }
            }
            PExpr::Vec2(x, y) => {
                let xs = x.to_tokenstream();
                let ys = y.to_tokenstream();
                quote! { crate::test::Expr::Vec2(crate::test::Vec2::new(Box::new(#xs), Box::new(#ys))) }
            }
            PExpr::Vec3(x, y, z) => {
                let xs = x.to_tokenstream();
                let ys = y.to_tokenstream();
                let zs = z.to_tokenstream();
                quote! { crate::test::Expr::Vec3(crate::test::Vec3::new(Box::new(#xs), Box::new(#ys), Box::new(#zs))) }
            }
            PExpr::Vec4(a, b, c, d) => {
                let a1 = a.to_tokenstream();
                let b1 = b.to_tokenstream();
                let c1 = c.to_tokenstream();
                let d1 = d.to_tokenstream();
                quote! { crate::test::Expr::Vec4(crate::test::Vec4::new(Box::new(#a1), Box::new(#b1), Box::new(#c1), Box::new(#d1))) }
            }
            PExpr::If(c, t, e) => {
                let cts = c.to_tokenstream();
                let tts = t.to_tokenstream();
                let ets = e.to_tokenstream();
                quote! { crate::test::Expr::If { condition: Box::new(#cts), then_branch: Box::new(#tts), else_branch: Box::new(#ets) } }
            }
        }
    }
}

// ---------------- Cargo.toml (example) ----------------
// [package]
// name = "wgsl_macro"
// version = "0.1.0"
// edition = "2021"
//
// [lib]
// proc-macro = true
//
// [dependencies]
// proc-macro2 = "1"
// quote = "1"
// syn = { version = "2", features = ["full"] }

// ---------------- Example usage (in your crate) ----------------
// // Cargo.toml: add dependency to the proc-macro crate
// wgsl_macro = { path = "../wgsl_macro" }
//
// // in code:
// use wgsl_macro::wgsl;
//
// let ex = wgsl!("vec3(3.0, 1.0, 3.0) * 1.0 + vec3(3.0, 1.0, if((3.0 > 1.0), 3.0, 10.0))");
// // ex is an Expr value (constructed at compile time)
