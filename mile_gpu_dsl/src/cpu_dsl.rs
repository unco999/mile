// cpu_dsl_extended.rs
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::{Add, Sub, Mul, Div};

// ==================== 基础类型定义 ====================
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Scalar(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
}

impl Value {
    pub fn as_scalar(&self) -> f32 {
        match self {
            Value::Scalar(s) => *s,
            Value::Vec2(v) => v[0],
            Value::Vec3(v) => v[0],
            Value::Vec4(v) => v[0],
        }
    }
    
    pub fn dimensions(&self) -> usize {
        match self {
            Value::Scalar(_) => 1,
            Value::Vec2(_) => 2,
            Value::Vec3(_) => 3,
            Value::Vec4(_) => 4,
        }
    }
}

// ==================== 表达式 DSL ====================
#[derive(Debug, Clone)]
pub struct Expr {
    pub code: String,
    pub value_type: ValueType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueType {
    Scalar,
    Vec2,
    Vec3,
    Vec4,
}

impl Expr {
    pub fn new<S: Into<String>>(s: S, value_type: ValueType) -> Self { 
        Expr { code: s.into(), value_type } 
    }
    
    pub fn call(func: &str, args: Vec<impl ToExpr>) -> Self {
        let args_str = args.iter()
            .map(|arg| arg.to_expr().code.clone())
            .collect::<Vec<_>>()
            .join(", ");
        Expr::new(format!("{}({})", func, args_str), ValueType::Scalar) // 假设函数返回标量
    }
    
    pub fn to_wgsl(&self) -> String { self.code.clone() }
}

// 包装类型
#[derive(Debug, Clone)]
pub struct Wf32(f32);
#[derive(Debug, Clone)]
pub struct Wvec2f32([f32;2]);
#[derive(Debug, Clone)]
pub struct Wvec3f32([f32;3]);
#[derive(Debug, Clone)]
pub struct Wvec4f32([f32;4]);
#[derive(Debug, Clone)]
pub enum WRef { 
    Var(&'static str, ValueType),
    ComponentAccess(Box<WRef>, &'static str), // 支持分量访问，如 a.x
}

// ToExpr trait 实现
pub trait ToExpr { 
    fn to_expr(&self) -> Expr; 
    fn value_type(&self) -> ValueType;
}

impl ToExpr for Expr { 
    fn to_expr(&self) -> Expr { self.clone() } 
    fn value_type(&self) -> ValueType { self.value_type.clone() }
}

impl ToExpr for Wf32 { 
    fn to_expr(&self) -> Expr { Expr::new(format!("{}", self.0), ValueType::Scalar) } 
    fn value_type(&self) -> ValueType { ValueType::Scalar }
}

impl ToExpr for Wvec2f32 { 
    fn to_expr(&self) -> Expr { Expr::new(format!("vec2({},{})", self.0[0], self.0[1]), ValueType::Vec2) } 
    fn value_type(&self) -> ValueType { ValueType::Vec2 }
}

impl ToExpr for Wvec3f32 { 
    fn to_expr(&self) -> Expr { Expr::new(format!("vec3({},{},{})", self.0[0], self.0[1], self.0[2]), ValueType::Vec3) } 
    fn value_type(&self) -> ValueType { ValueType::Vec3 }
}

impl ToExpr for Wvec4f32 { 
    fn to_expr(&self) -> Expr { Expr::new(format!("vec4({},{},{},{})", self.0[0], self.0[1], self.0[2], self.0[3]), ValueType::Vec4) } 
    fn value_type(&self) -> ValueType { ValueType::Vec4 }
}

impl ToExpr for WRef { 
    fn to_expr(&self) -> Expr { 
        match self { 
            WRef::Var(s, t) => Expr::new(*s, t.clone()),
            WRef::ComponentAccess(var, comp) => {
                let base = var.to_expr();
                Expr::new(format!("{}.{}", base.code, comp), ValueType::Scalar)
            }
        }
    }
    fn value_type(&self) -> ValueType { 
        match self {
            WRef::Var(_, t) => t.clone(),
            WRef::ComponentAccess(_, _) => ValueType::Scalar,
        }
    }
}

// 运算符重载宏
macro_rules! impl_ops {
    ($lhs:ty, $rhs:ty) => {
        impl Add<$rhs> for $lhs { 
            type Output = Expr; 
            fn add(self, rhs: $rhs) -> Self::Output { 
                let lhs_expr = self.to_expr();
                let rhs_expr = rhs.to_expr();
                let result_type = determine_result_type(&lhs_expr.value_type, &rhs_expr.value_type);
                Expr::new(format!("({} + {})", lhs_expr.code, rhs_expr.code), result_type)
            } 
        }
        impl Sub<$rhs> for $lhs { 
            type Output = Expr; 
            fn sub(self, rhs: $rhs) -> Self::Output { 
                let lhs_expr = self.to_expr();
                let rhs_expr = rhs.to_expr();
                let result_type = determine_result_type(&lhs_expr.value_type, &rhs_expr.value_type);
                Expr::new(format!("({} - {})", lhs_expr.code, rhs_expr.code), result_type)
            } 
        }
        impl Mul<$rhs> for $lhs { 
            type Output = Expr; 
            fn mul(self, rhs: $rhs) -> Self::Output { 
                let lhs_expr = self.to_expr();
                let rhs_expr = rhs.to_expr();
                let result_type = determine_result_type(&lhs_expr.value_type, &rhs_expr.value_type);
                Expr::new(format!("({} * {})", lhs_expr.code, rhs_expr.code), result_type)
            } 
        }
        impl Div<$rhs> for $lhs { 
            type Output = Expr; 
            fn div(self, rhs: $rhs) -> Self::Output { 
                let lhs_expr = self.to_expr();
                let rhs_expr = rhs.to_expr();
                let result_type = determine_result_type(&lhs_expr.value_type, &rhs_expr.value_type);
                Expr::new(format!("({} / {})", lhs_expr.code, rhs_expr.code), result_type)
            } 
        }
    };
}

fn determine_result_type(lhs: &ValueType, rhs: &ValueType) -> ValueType {
    match (lhs, rhs) {
        (ValueType::Scalar, _) => rhs.clone(),
        (_, ValueType::Scalar) => lhs.clone(),
        (a, b) if a == b => a.clone(),
        _ => ValueType::Scalar, // 默认回退
    }
}

// 实现所有运算符组合
impl_ops!(Expr, Expr);
impl_ops!(Expr, Wf32);
impl_ops!(Expr, Wvec2f32);
impl_ops!(Expr, Wvec3f32);
impl_ops!(Expr, Wvec4f32);
impl_ops!(Expr, WRef);
impl_ops!(WRef, Expr);
impl_ops!(WRef, Wf32);
impl_ops!(WRef, Wvec2f32);
impl_ops!(WRef, Wvec3f32);
impl_ops!(WRef, Wvec4f32);
impl_ops!(WRef, WRef);
impl_ops!(Wf32, Expr);
impl_ops!(Wf32, Wf32);
impl_ops!(Wf32, Wvec2f32);
impl_ops!(Wf32, Wvec3f32);
impl_ops!(Wf32, Wvec4f32);
impl_ops!(Wf32, WRef);
impl_ops!(Wvec2f32, Expr);
impl_ops!(Wvec2f32, Wf32);
impl_ops!(Wvec2f32, Wvec2f32);
impl_ops!(Wvec2f32, Wvec3f32);
impl_ops!(Wvec2f32, Wvec4f32);
impl_ops!(Wvec2f32, WRef);
impl_ops!(Wvec3f32, Expr);
impl_ops!(Wvec3f32, Wf32);
impl_ops!(Wvec3f32, Wvec2f32);
impl_ops!(Wvec3f32, Wvec3f32);
impl_ops!(Wvec3f32, Wvec4f32);
impl_ops!(Wvec3f32, WRef);
impl_ops!(Wvec4f32, Expr);
impl_ops!(Wvec4f32, Wf32);
impl_ops!(Wvec4f32, Wvec2f32);
impl_ops!(Wvec4f32, Wvec3f32);
impl_ops!(Wvec4f32, Wvec4f32);
impl_ops!(Wvec4f32, WRef);

// ==================== 扩展的 AST ====================
#[derive(Debug, Clone)]
pub enum AST {
    Var(String, ValueType),
    Const(f32),
    Vec2Const([f32; 2]),
    Vec3Const([f32; 3]),
    Vec4Const([f32; 4]),
    Add(Box<AST>, Box<AST>),
    Sub(Box<AST>, Box<AST>),
    Mul(Box<AST>, Box<AST>),
    Div(Box<AST>, Box<AST>),
    ComponentAccess(Box<AST>, String), // 分量访问，如 a.x
    Vec2(Box<AST>, Box<AST>),
    Vec3(Box<AST>, Box<AST>, Box<AST>),
    Vec4(Box<AST>, Box<AST>, Box<AST>, Box<AST>),
    FunctionCall(String, Vec<AST>), // 函数调用
}

impl AST {
    pub fn value_type(&self) -> ValueType {
        match self {
            AST::Var(_, t) => t.clone(),
            AST::Const(_) => ValueType::Scalar,
            AST::Vec2Const(_) => ValueType::Vec2,
            AST::Vec3Const(_) => ValueType::Vec3,
            AST::Vec4Const(_) => ValueType::Vec4,
            AST::Add(l, r) | AST::Sub(l, r) | AST::Mul(l, r) | AST::Div(l, r) => {
                determine_result_type(&l.value_type(), &r.value_type())
            }
            AST::ComponentAccess(_, _) => ValueType::Scalar,
            AST::Vec2(_, _) => ValueType::Vec2,
            AST::Vec3(_, _, _) => ValueType::Vec3,
            AST::Vec4(_, _, _, _) => ValueType::Vec4,
            AST::FunctionCall(_, _) => ValueType::Scalar, // 假设函数返回标量
        }
    }
}

// ==================== 解析器扩展 ====================
pub struct Parser<'a> {
    s: &'a str,
    pos: usize,
    len: usize,
}

impl<'a> Parser<'a> {
    pub fn new(s: &'a str) -> Self { Self { s, pos: 0, len: s.len() } }
    
    fn peek(&self) -> Option<char> { self.s[self.pos..].chars().next() }
    fn bump(&mut self) { if let Some(c) = self.peek() { self.pos += c.len_utf8(); } }
    fn eat_ws(&mut self) { while let Some(c) = self.peek() { if c.is_whitespace() { self.bump(); } else { break; } } }

    fn parse_number(&mut self) -> Option<f32> {
        self.eat_ws();
        let mut num = String::new();
        let mut seen_dot = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || (c=='.' && !seen_dot) {
                if c=='.' { seen_dot = true; }
                num.push(c); self.bump();
            } else { break; }
        }
        if num.is_empty() { None } else { num.parse::<f32>().ok() }
    }
    
    fn parse_ident(&mut self) -> Option<String> {
        self.eat_ws();
        let mut id = String::new();
        if let Some(c) = self.peek() {
            if c.is_alphabetic() || c=='_' {
                id.push(c); self.bump();
                while let Some(c2) = self.peek() {
                    if c2.is_alphanumeric() || c2=='_' { id.push(c2); self.bump(); } else { break; }
                }
                return Some(id);
            }
        }
        None
    }

    fn parse_vec_constructor(&mut self, dim: usize) -> Result<AST, String> {
        self.bump(); // 吃掉 '('
        self.eat_ws();
        
        let mut components = Vec::new();
        for i in 0..dim {
            if i > 0 {
                self.eat_ws();
                if let Some(',') = self.peek() {
                    self.bump();
                } else {
                    return Err(format!("expected ',' in vec{} constructor", dim));
                }
            }
            components.push(self.parse_expr()?);
            self.eat_ws();
        }
        
        if let Some(')') = self.peek() {
            self.bump();
            match dim {
                2 => Ok(AST::Vec2(Box::new(components[0].clone()), Box::new(components[1].clone()))),
                3 => Ok(AST::Vec3(Box::new(components[0].clone()), Box::new(components[1].clone()), Box::new(components[2].clone()))),
                4 => Ok(AST::Vec4(Box::new(components[0].clone()), Box::new(components[1].clone()), Box::new(components[2].clone()), Box::new(components[3].clone()))),
                _ => Err("unsupported vector dimension".into()),
            }
        } else {
            Err(format!("missing ')' in vec{} constructor", dim))
        }
    }

    fn parse_function_call(&mut self, func_name: String) -> Result<AST, String> {
        self.bump(); // 吃掉 '('
        self.eat_ws();
        
        let mut args = Vec::new();
        if let Some(')') = self.peek() {
            self.bump();
            return Ok(AST::FunctionCall(func_name, args));
        }
        
        loop {
            args.push(self.parse_expr()?);
            self.eat_ws();
            
            match self.peek() {
                Some(',') => { self.bump(); self.eat_ws(); }
                Some(')') => { self.bump(); break; }
                _ => return Err("expected ',' or ')' in function call".into()),
            }
        }
        
        Ok(AST::FunctionCall(func_name, args))
    }

    // factor := number | ident | ident'('... | '(' expr ')' | ident '.' component
    fn parse_factor(&mut self) -> Result<AST, String> {
        self.eat_ws();
        
        if let Some('(') = self.peek() {
            self.bump();
            let node = self.parse_expr()?;
            self.eat_ws();
            if let Some(')') = self.peek() { self.bump(); Ok(node) } else { Err("missing )".into()) }
        } else if let Some(n) = self.parse_number() {
            Ok(AST::Const(n))
        } else if let Some(id) = self.parse_ident() {
            self.eat_ws();
            
            // 检查是否是函数调用
            if let Some('(') = self.peek() {
                return self.parse_function_call(id);
            }
            
            // 检查是否是向量构造函数
            if id.starts_with("vec") {
                let dim = id.chars().nth(3).and_then(|c| c.to_digit(10)).unwrap_or(0) as usize;
                if dim >= 2 && dim <= 4 {
                    return self.parse_vec_constructor(dim);
                }
            }
            
            // 检查是否是分量访问
            if let Some('.') = self.peek() {
                self.bump();
                if let Some(comp) = self.parse_ident() {
                    let valid_comps = match comp.as_str() {
                        "x" | "y" | "z" | "w" | "r" | "g" | "b" | "a" => true,
                        _ => false,
                    };
                    if valid_comps {
                        return Ok(AST::ComponentAccess(Box::new(AST::Var(id, ValueType::Scalar)), comp));
                    }
                }
                return Err("invalid component access".into());
            }
            
            // 普通变量
            Ok(AST::Var(id, ValueType::Scalar)) // 默认为标量，实际类型应该在上下文中确定
        } else {
            Err(format!("unexpected token at pos {}", self.pos))
        }
    }

    // term := factor (('*'|'/') factor)*
    fn parse_term(&mut self) -> Result<AST, String> {
        let mut node = self.parse_factor()?;
        loop {
            self.eat_ws();
            match self.peek() {
                Some('*') => { self.bump(); let rhs = self.parse_factor()?; node = AST::Mul(Box::new(node), Box::new(rhs)); }
                Some('/') => { self.bump(); let rhs = self.parse_factor()?; node = AST::Div(Box::new(node), Box::new(rhs)); }
                _ => break,
            }
        }
        Ok(node)
    }

    // expr := term (('+'|'-') term)*
    fn parse_expr(&mut self) -> Result<AST, String> {
        let mut node = self.parse_term()?;
        loop {
            self.eat_ws();
            match self.peek() {
                Some('+') => { self.bump(); let rhs = self.parse_term()?; node = AST::Add(Box::new(node), Box::new(rhs)); }
                Some('-') => { self.bump(); let rhs = self.parse_term()?; node = AST::Sub(Box::new(node), Box::new(rhs)); }
                _ => break,
            }
        }
        Ok(node)
    }
    
    pub fn parse_all(&mut self) -> Result<AST, String> {
        let node = self.parse_expr()?; 
        self.eat_ws();
        if self.pos < self.len { Err(format!("trailing chars at {}", self.pos)) } else { Ok(node) }
    }
}

// ==================== 矩阵计算系统 ====================
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self { 
        Matrix { rows, cols, data: vec![0.0; rows * cols] } 
    }
    
    pub fn from_rows(rows: Vec<Vec<f32>>) -> Self {
        let rows_count = rows.len();
        let cols_count = if rows.is_empty() { 0 } else { rows[0].len() };
        let mut data = Vec::with_capacity(rows_count * cols_count);
        for row in rows {
            data.extend(row);
        }
        Matrix { rows: rows_count, cols: cols_count, data }
    }
    
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }
    
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }
    
    pub fn mul_vec(&self, v: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.get(i, j) * v[j];
            }
        }
        result
    }
    
    pub fn print(&self, name: &str) {
        println!("{} ({}x{}):", name, self.rows, self.cols);
        for i in 0..self.rows {
            let row: Vec<f32> = (0..self.cols).map(|j| self.get(i, j)).collect();
            println!("  {:?}", row);
        }
    }
    
    pub fn to_wgsl(&self, name: &str) -> String {
        let mut wgsl = format!("var<private> {}: mat{}x{}<f32> = mat{}x{}<f32>(\n", 
            name, self.rows, self.cols, self.rows, self.cols);
        
        for i in 0..self.rows {
            let row: Vec<String> = (0..self.cols).map(|j| format!("{:.6}", self.get(i, j))).collect();
            wgsl += &format!("    vec{}<f32>({}),\n", self.cols, row.join(", "));
        }
        wgsl += ");\n";
        wgsl
    }
}

#[derive(Debug)]
pub struct ComputationResult {
    pub m1: Matrix,
    pub m2: Matrix, 
    pub m3: Matrix,
    pub intermediate_results: Vec<Vec<f32>>,
    pub final_result: Vec<f32>,
    pub wgsl_code: String,
}

// 项收集系统
#[derive(Debug, Clone)]
pub struct Term {
    pub coefficient: f32,
    pub variables: Vec<String>,
    pub value_type: ValueType,
}

pub fn flatten_ast_to_terms(ast: &AST) -> Vec<Term> {
    let mut terms = Vec::new();
    collect_terms_recursive(ast, &mut terms, 1.0);
    terms
}

fn collect_terms_recursive(ast: &AST, terms: &mut Vec<Term>, coefficient: f32) {
    match ast {
        AST::Add(a, b) => {
            collect_terms_recursive(a, terms, coefficient);
            collect_terms_recursive(b, terms, coefficient);
        }
        AST::Sub(a, b) => {
            collect_terms_recursive(a, terms, coefficient);
            collect_terms_recursive(b, terms, -coefficient);
        }
        AST::Mul(a, b) => {
            // 对于乘法，我们需要分别处理左右子树
            match (&**a, &**b) {
                (AST::Const(c), right) => {
                    collect_terms_recursive(right, terms, coefficient * c);
                }
                (left, AST::Const(c)) => {
                    collect_terms_recursive(left, terms, coefficient * c);
                }
                _ => {
                    // 复杂乘法，创建新项
                    let mut term_vars = Vec::new();
                    extract_variables(ast, &mut term_vars);
                    terms.push(Term {
                        coefficient,
                        variables: term_vars,
                        value_type: ast.value_type(),
                    });
                }
            }
        }
        AST::Div(a, b) => {
            if let AST::Const(divisor) = &**b {
                collect_terms_recursive(a, terms, coefficient / divisor);
            } else {
                // 复杂除法
                let mut term_vars = Vec::new();
                extract_variables(ast, &mut term_vars);
                terms.push(Term {
                    coefficient,
                    variables: term_vars,
                    value_type: ast.value_type(),
                });
            }
        }
        AST::Var(name, value_type) => {
            terms.push(Term {
                coefficient,
                variables: vec![name.clone()],
                value_type: value_type.clone(),
            });
        }
        AST::Const(c) => {
            terms.push(Term {
                coefficient: coefficient * c,
                variables: Vec::new(),
                value_type: ValueType::Scalar,
            });
        }
        AST::ComponentAccess(expr, comp) => {
            if let AST::Var(name, _) = &**expr {
                terms.push(Term {
                    coefficient,
                    variables: vec![format!("{}.{}", name, comp)],
                    value_type: ValueType::Scalar,
                });
            }
        }
        _ => {
            // 其他情况，提取所有变量
            let mut term_vars = Vec::new();
            extract_variables(ast, &mut term_vars);
            terms.push(Term {
                coefficient,
                variables: term_vars,
                value_type: ast.value_type(),
            });
        }
    }
}

fn extract_variables(ast: &AST, vars: &mut Vec<String>) {
    match ast {
        AST::Var(name, _) => vars.push(name.clone()),
        AST::ComponentAccess(expr, comp) => {
            if let AST::Var(name, _) = &**expr {
                vars.push(format!("{}.{}", name, comp));
            } else {
                extract_variables(expr, vars);
            }
        }
        AST::Add(a, b) | AST::Sub(a, b) | AST::Mul(a, b) | AST::Div(a, b) => {
            extract_variables(a, vars);
            extract_variables(b, vars);
        }
        AST::Vec2(a, b) | AST::Vec3(a, b, _) | AST::Vec4(a, b, _, _) => {
            extract_variables(a, vars);
            extract_variables(b, vars);
        }
        AST::FunctionCall(_, args) => {
            for arg in args {
                extract_variables(arg, vars);
            }
        }
        _ => {}
    }
}

/// 主构建函数：将表达式转换为矩阵计算流程
pub fn build_computation_pipeline(ast: &AST, var_values: &HashMap<&str, Value>) -> ComputationResult {
    // 1) 收集项
    let terms = flatten_ast_to_terms(ast);
    println!("Collected terms: {:#?}", terms);

    // 2) 构建变量表（包含所有变量和分量）
    let mut all_vars = HashSet::new();
    for term in &terms {
        for var in &term.variables {
            all_vars.insert(var.clone());
        }
    }
    let var_order: Vec<String> = all_vars.into_iter().collect();
    println!("Variable order: {:?}", var_order);

    // 3) 构建变量值向量
    let mut vars_vec = Vec::new();
    for var_name in &var_order {
        if let Some(dot_pos) = var_name.find('.') {
            let base_var = &var_name[..dot_pos];
            let component = &var_name[dot_pos+1..];
            if let Some(value) = var_values.get(base_var) {
                let comp_value = match (value, component) {
                    (Value::Vec2(v), "x") => v[0],
                    (Value::Vec2(v), "y") => v[1],
                    (Value::Vec3(v), "x") => v[0],
                    (Value::Vec3(v), "y") => v[1],
                    (Value::Vec3(v), "z") => v[2],
                    (Value::Vec4(v), "x") => v[0],
                    (Value::Vec4(v), "y") => v[1],
                    (Value::Vec4(v), "z") => v[2],
                    (Value::Vec4(v), "w") => v[3],
                    _ => 0.0,
                };
                vars_vec.push(comp_value);
            } else {
                vars_vec.push(0.0);
            }
        } else if let Some(value) = var_values.get(var_name.as_str()) {
            vars_vec.push(value.as_scalar());
        } else {
            vars_vec.push(0.0);
        }
    }
    println!("Variable values: {:?}", vars_vec);

    // 4) 构建 M1：变量选择矩阵
    let m1_rows = var_order.len();
    let m1_cols = var_order.len();
    let mut m1 = Matrix::new(m1_rows, m1_cols);
    for i in 0..m1_rows {
        m1.set(i, i, 1.0);
    }

    // 5) 构建 M2：操作数选择矩阵
    let mut m2_rows = Vec::new();
    let mut term_slots = Vec::new();
    
    for (term_idx, term) in terms.iter().enumerate() {
        let mut slots_for_term = Vec::new();
        
        for var in &term.variables {
            if let Some(var_idx) = var_order.iter().position(|v| v == var) {
                let mut row = vec![0.0; var_order.len()];
                row[var_idx] = 1.0;
                m2_rows.push(row);
                slots_for_term.push(m2_rows.len() - 1);
            }
        }
        
        // 纯常量项
        if term.variables.is_empty() && term.coefficient != 0.0 {
            let mut row = vec![0.0; var_order.len()];
            m2_rows.push(row);
            slots_for_term.push(m2_rows.len() - 1);
        }
        
        term_slots.push((term_idx, slots_for_term, term.coefficient));
    }
    
    let m2 = Matrix::from_rows(m2_rows);

    // 6) 模拟计算过程
    let v_after_m1 = m1.mul_vec(&vars_vec);
    let v_slots = m2.mul_vec(&vars_vec);
    
    // 7) 逐项计算乘积
    let mut products = Vec::new();
    for (term_idx, slots, coefficient) in &term_slots {
        if slots.is_empty() {
            // 纯常量项
            products.push(*coefficient);
        } else {
            let mut product = *coefficient;
            for &slot_idx in slots {
                product *= v_slots[slot_idx];
            }
            products.push(product);
        }
    }

    // 8) 构建 M3：聚合矩阵
    let mut m3_data = Vec::new();
    let output_dims = match ast.value_type() {
        ValueType::Scalar => 1,
        ValueType::Vec2 => 2,
        ValueType::Vec3 => 3,
        ValueType::Vec4 => 4,
    };
    
    // 简单实现：将所有项相加
    for _ in 0..output_dims {
        let mut row = vec![0.0; products.len()];
        for i in 0..products.len() {
            row[i] = 1.0; // 简单相加
        }
        m3_data.push(row);
    }
    let m3 = Matrix::from_rows(m3_data);

    let final_result = m3.mul_vec(&products);

    // 9) 生成 WGSL 代码
    let wgsl_code = generate_wgsl_code(&m1, &m2, &m3, &var_order, &products, &final_result);

    ComputationResult {
        m1,
        m2,
        m3,
        intermediate_results: vec![v_after_m1, v_slots, products.clone()],
        final_result,
        wgsl_code,
    }
}

fn generate_wgsl_code(m1: &Matrix, m2: &Matrix, m3: &Matrix, var_order: &[String], 
                     products: &[f32], final_result: &[f32]) -> String {
    let mut wgsl = String::new();
    
    wgsl += "// Generated WGSL code from matrix computation pipeline\n";
    wgsl += "// Variables and matrices for computation\n\n";
    
    // 输入变量
    wgsl += "// Input variables\n";
    for (i, var) in var_order.iter().enumerate() {
        wgsl += &format!("var<private> {}: f32 = {:.6};\n", var.replace('.', "_"), 0.0);
    }
    wgsl += "\n";
    
    // 矩阵定义
    wgsl += &m1.to_wgsl("M1");
    wgsl += "\n";
    wgsl += &m2.to_wgsl("M2");
    wgsl += "\n";
    wgsl += &m3.to_wgsl("M3");
    wgsl += "\n";
    
    // 计算步骤
    wgsl += "// Computation steps\n";
    wgsl += "fn compute() -> f32 {\n";
    wgsl += "    // Step 1: M1 * vars\n";
    wgsl += "    var v_after_m1: array<f32, ";
    wgsl += &m1.rows.to_string();
    wgsl += ">;\n";
    wgsl += "    for (var i: u32 = 0u; i < ";
    wgsl += &m1.rows.to_string();
    wgsl += "u; i = i + 1u) {\n";
    wgsl += "        v_after_m1[i] = 0.0;\n";
    wgsl += "        for (var j: u32 = 0u; j < ";
    wgsl += &m1.cols.to_string();
    wgsl += "u; j = j + 1u) {\n";
    wgsl += "            v_after_m1[i] = v_after_m1[i] + M1[i][j] * ... ; // variable access here\n";
    wgsl += "        }\n";
    wgsl += "    }\n\n";
    
    wgsl += "    // Step 2: M2 * v_after_m1\n";
    wgsl += "    var v_slots: array<f32, ";
    wgsl += &m2.rows.to_string();
    wgsl += ">;\n";
    wgsl += "    for (var i: u32 = 0u; i < ";
    wgsl += &m2.rows.to_string();
    wgsl += "u; i = i + 1u) {\n";
    wgsl += "        v_slots[i] = 0.0;\n";
    wgsl += "        for (var j: u32 = 0u; j < ";
    wgsl += &m2.cols.to_string();
    wgsl += "u; j = j + 1u) {\n";
    wgsl += "            v_slots[i] = v_slots[i] + M2[i][j] * v_after_m1[j];\n";
    wgsl += "        }\n";
    wgsl += "    }\n\n";
    
    wgsl += "    // Step 3: Compute products\n";
    wgsl += "    var products: array<f32, ";
    wgsl += &products.len().to_string();
    wgsl += ">;\n";
    // 这里需要根据实际的项计算逻辑来生成代码
    
    wgsl += "    // Step 4: M3 * products\n";
    wgsl += "    var result: f32 = 0.0;\n";
    wgsl += "    for (var i: u32 = 0u; i < ";
    wgsl += &m3.cols.to_string();
    wgsl += "u; i = i + 1u) {\n";
    wgsl += "        result = result + M3[0][i] * products[i];\n";
    wgsl += "    }\n\n";
    
    wgsl += "    return result;\n";
    wgsl += "}\n";
    
    wgsl
}

// ==================== 测试和示例 ====================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_operations() {
        // 创建向量变量
        let a = WRef::Var("a", ValueType::Vec2);
        let b = WRef::Var("b", ValueType::Vec2);
        let scalar = Wf32(2.0);
        
        // 向量运算
        let expr = a + b * scalar;
        println!("Vector expression: {}", expr.to_wgsl());
        
        // 解析
        let binding = expr.to_wgsl();
        let mut parser = Parser::new(&binding);
        let ast = parser.parse_all().expect("Parse error");
        println!("AST: {:?}", ast);
        
        // 变量值
        let mut var_values = HashMap::new();
        var_values.insert("a", Value::Vec2([1.0, 2.0]));
        var_values.insert("b", Value::Vec2([3.0, 4.0]));
        
        // 构建计算管道
        let result = build_computation_pipeline(&ast, &var_values);
        
        println!("Final result: {:?}", result.final_result);
        println!("WGSL code:\n{}", result.wgsl_code);
    }

    #[test]
    fn test_component_access() {
        // 测试分量访问
        let vec_var = WRef::Var("position", ValueType::Vec3);
        let x_component = WRef::ComponentAccess(Box::new(vec_var), "x");
        let expr = x_component + Wf32(1.0);
        
        println!("Component access expression: {}", expr.to_wgsl());
        
        let binding = expr.to_wgsl();
        let mut parser = Parser::new(&binding);
        let ast = parser.parse_all().expect("Parse error");
        
        let mut var_values = HashMap::new();
        var_values.insert("position", Value::Vec3([1.0, 2.0, 3.0]));
        
        let result = build_computation_pipeline(&ast, &var_values);
        println!("Result: {:?}", result.final_result);
    }
}

// fn main() {
//     // 运行测试
//     println!("=== Testing Vector Operations ===");
//     test_vector_operations();
    
//     println!("\n=== Testing Component Access ===");
//     test_component_access();
    
//     // 保存完整的 WGSL 代码到文件
//     use std::fs::File;
//     use std::io::Write;
    
//     let mut parser = Parser::new("a * b + c - d * e");
//     let ast = parser.parse_all().expect("Parse error");
    
//     let mut var_values = HashMap::new();
//     var_values.insert("a", Value::Scalar(1.0));
//     var_values.insert("b", Value::Scalar(2.0));
//     var_values.insert("c", Value::Scalar(3.0));
//     var_values.insert("d", Value::Scalar(4.0));
//     var_values.insert("e", Value::Scalar(5.0));
    
//     let result = build_computation_pipeline(&ast, &var_values);
    
//     // 输出矩阵信息
//     result.m1.print("M1");
//     result.m2.print("M2"); 
//     result.m3.print("M3");
//     println!("Intermediate results: {:?}", result.intermediate_results);
//     println!("Final result: {:?}", result.final_result);
    
//     // 保存 WGSL 代码
//     let mut file = File::create("computation.wgsl").expect("Unable to create file");
//     file.write_all(result.wgsl_code.as_bytes()).expect("Unable to write file");
//     println!("WGSL code saved to computation.wgsl");
// }