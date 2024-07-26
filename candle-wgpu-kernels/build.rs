use std::collections::{HashMap, HashSet};
use std::{env, fs};
use std::path::{Path, PathBuf};

fn get_all_wgsl_files<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
    let mut files = Vec::new();
    visit_dirs(dir.as_ref(), &mut files).unwrap();
    files
}

fn visit_dirs(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, files)?;
            } else if path.extension().map(|ext| ext == "pwgsl").unwrap_or(false) {
                files.push(path);
            }
        }
    }
    Ok(())
}

fn to_upper_camel_case(snake_case: &str) -> String {
    snake_case
        .split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first_char) => first_char.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set");

    let candle_core_path = PathBuf::from(&manifest_dir);
    let absolute_candle_core_path = fs::canonicalize(&candle_core_path).expect("Failed to get absolute outer folder path");

    let shader_dir = candle_core_path.join("src");
    if !shader_dir.exists(){
        panic!("could not find shader path {:?}", shader_dir);
    }
    println!("Searching Shaders in: {:?}", shader_dir);
    
    let mut shader_map = HashMap::new();
    let mut virtual_id_counter = 1;

    let mut modules : HashSet<String> = HashSet::new();

    for file in get_all_wgsl_files(shader_dir){
        println!("Found File: {:?}", file);
        let shader_content = preprocess_shader(&file, &mut shader_map, &mut virtual_id_counter);
        
        shader_map.insert(virtual_id_counter, shader_content);
        

        let parent_dir = file.parent().unwrap();
        let parent_name = parent_dir.file_name().unwrap().to_str().unwrap().to_string();
        if parent_name == "src" {
            continue;
        }
        modules.insert(parent_name);
        let generated_dir = parent_dir.join("generated");
        fs::create_dir_all(&generated_dir).unwrap();
        let original_file_name = file.file_name().unwrap().to_str().unwrap();
       

        let mut global_functions = HashMap::new();
        //create the File:
        let debug_s = shader_loader::load_shader(virtual_id_counter, &shader_map, &vec!["f32"], &mut global_functions);
        let new_file_name = format!("{}_generated_f32.wgsl", original_file_name);
        let new_file_path = generated_dir.join(new_file_name);
        fs::write(new_file_path, debug_s).expect("Failed to write shader file");
     

        let debug_s = shader_loader::load_shader(virtual_id_counter, &shader_map, &vec!["u32"],&mut global_functions);
        let new_file_name = format!("{}_generated_u32.wgsl", original_file_name);
        let new_file_path = generated_dir.join(new_file_name);
        fs::write(new_file_path, debug_s).expect("Failed to write shader file");

        let debug_s = shader_loader::load_shader(virtual_id_counter, &shader_map, &vec!["u8"],&mut global_functions);
        let new_file_name = format!("{}_generated_u8.wgsl", original_file_name);
        let new_file_path = generated_dir.join(new_file_name);
        fs::write(new_file_path, debug_s).expect("Failed to write shader file");
        
        let functions : Vec<String> = global_functions.iter().map(|(key, _)| to_upper_camel_case(key)).collect();

        let functions_match : Vec<String> = global_functions.iter().map(|(key, value)| format!("Functions::{} => \"g{}\"", to_upper_camel_case(key), value)).collect();


        let content =  
        format!("/// *********** This File Is Genereted! **********///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Functions{{{}}}
impl crate::EntryPoint for Functions{{
    fn get_entry_point(&self) -> &'static str{{
        match self{{
            {}
        }}
    }} 
}}
pub fn load_shader(typ : crate::DType) -> &'static str {{
    match typ{{
        crate::DType::F32 => include_str!(\"generated/shader.pwgsl_generated_f32.wgsl\"),
        crate::DType::U32 => include_str!(\"generated/shader.pwgsl_generated_u32.wgsl\"),
        crate::DType::U8 => include_str!(\"generated/shader.pwgsl_generated_u8.wgsl\"),
    }}
}}
    ",functions.join(","), functions_match.join(","));

        let new_file_path = parent_dir.join("mod.rs");

        fs::write(new_file_path, content).unwrap();


        let absolute_file_path = fs::canonicalize(&file).expect("Failed to get absolute file path");
        match absolute_file_path.strip_prefix(&absolute_candle_core_path) {
            Ok(relative_path) =>  println!("cargo::rerun-if-changed={}", relative_path.to_string_lossy()),
            Err(_) => println!("File path is not inside the outer folder"),
        }
        virtual_id_counter += 1;
    }

    //create src/generated.rs
    let mut shader_content = "use crate::*; \n
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pipelines{\n".to_string();

    for m in modules.iter(){
        shader_content.push_str(&format!("\t{}(DType, {}::Functions),\n", to_upper_camel_case(m), m)); 
    }
    shader_content.push_str("}");

    let functions_match : Vec<String> = modules.iter().map(|m| format!("\t\t\tPipelines::{}(_, f) => f.get_entry_point()", to_upper_camel_case(m))).collect();

    shader_content.push_str(
        &format!("
impl crate::EntryPoint for Pipelines{{
    fn get_entry_point(&self) -> &'static str{{
        match self{{
{}
        }}
    }} 
}}
    ",functions_match.join(",\n")));

shader_content.push_str(
"#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Shaders{\n");
for m in modules.iter(){
    shader_content.push_str(&format!("\t{}(DType),\n", to_upper_camel_case(m))); 
}
shader_content.push_str("}");

shader_content.push_str(
    &format!("
impl Pipelines {{
    pub fn get_shader(&self) -> Shaders{{
        match self{{
{}
        }}
    }}

    pub fn load_shader(&self) -> &'static str{{
        match self{{
{}        
        }}
    }}
}} 
",
modules.iter().map(|m| format!("\t\t\tPipelines::{}(typ, _) => Shaders::{}(typ.clone())", to_upper_camel_case(m), to_upper_camel_case(m))).collect::<Vec<String>>().join(",\n"),
modules.iter().map(|m| format!("\t\tPipelines::{}(typ, _) => {}::load_shader(typ.clone())", to_upper_camel_case(&m), m)).collect::<Vec<String>>().join(",\n")
));


shader_content.push_str(
    &format!("
impl Shaders {{
    pub fn get_shader(&self) -> Shaders{{
        match self{{
{}
        }}
    }}

    pub fn load_shader(&self) -> &'static str{{
        match self{{
{}        
        }}
    }}
}} 
",
modules.iter().map(|m| format!("\t\t\tShaders::{}(typ) => Shaders::{}(typ.clone())", to_upper_camel_case(m), to_upper_camel_case(m))).collect::<Vec<String>>().join(",\n"),
modules.iter().map(|m| format!("\t\tShaders::{}(typ) => {}::load_shader(typ.clone())", to_upper_camel_case(&m), m)).collect::<Vec<String>>().join(",\n")
));

    fs::write("src/generated.rs", shader_content).expect("Failed to write shader map file");
}

fn preprocess_shader(
    path: &PathBuf,
    shader_map: &mut HashMap<u32, String>,
    virtual_id_counter: &mut u32,
) -> String {
    println!("Process File: {:?}", path);
    let contents = fs::read_to_string(path).expect("Failed to read WGSL file");
    let mut result = String::new();
    let mut lines = contents.lines();

    while let Some(line) = lines.next() {
        if line.starts_with("#include") {
          
            let included_file = line.split_whitespace().nth(1).expect("No file specified in #include");
            let included_file = included_file.trim_matches('"');
            let included_path = path.parent().unwrap().join(included_file);
            println!("Found Include: {:?}", included_path);
            let included_shader_content = preprocess_shader(&included_path, shader_map, virtual_id_counter);
            result.push_str(&format!("#include_virtual {}\n", virtual_id_counter));
            shader_map.insert(*virtual_id_counter, included_shader_content);



            *virtual_id_counter += 1;
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}



mod shader_loader{
    use fancy_regex::Regex;
    use std::collections::HashMap;

    pub fn load_shader<T : AsRef<str>>(virtual_id: u32, shader_map: &std::collections::HashMap<u32, T>, defines: &Vec<&str>, global_functions : &mut HashMap<String, String>) -> String {
        println!("Load Shader {virtual_id}");
        let mut shader_code : String= (*shader_map.get(&virtual_id).expect("Shader ID not found")).as_ref().to_string();
        while let Some(include_pos) = shader_code.find("#include_virtual") {
            let start_pos = include_pos + "#include_virtual".len();
            let end_pos = shader_code[start_pos..].find('\n').unwrap() + start_pos;
            let virtual_id_str = &shader_code[start_pos..end_pos].trim();
            let include_id: u32 = virtual_id_str.parse().expect("Invalid virtual ID");
            let include_content = (*shader_map.get(&include_id).expect("Included shader ID not found")).as_ref();
            shader_code.replace_range(include_pos..end_pos, include_content);
        }
        
        let result = apply_defines(&shader_code, defines);
        
        let result = Regex::new(r"//.*\n", ).unwrap().replace_all(&result, "");
        let result = Regex::new(r"\n\s*(?=\n)", ).unwrap().replace_all(&result, "");
        //let result = Regex::new(r"((\s+)(?![\w\s])|(?<!\w)(\s+))", ).unwrap().replace_all(&result, "");
        
        let result = shader_shortener::shorten_variable_names(&result, global_functions);
        
        //let result = result.replace("\n", "");

        result
    }
    
    fn apply_defines(shader_code: &str, global_defines: &Vec<&str>) -> String {
        let mut result = String::new();
        let mut lines = shader_code.lines().peekable();
        let mut inside_ifdef = false;
        let mut defines : HashMap<String, String> = HashMap::new();
        while let Some(line) = lines.next() {
            if line.starts_with("#ifdef") {
                let key = line.split_whitespace().nth(1).expect("No key specified in #ifdef");
                if defines.contains_key(key) || global_defines.contains(&key) {
                    inside_ifdef = true;
                } else {
                    skip_until_else_or_endif(&mut lines);
                }
            } else if line.starts_with("#else") {
                if inside_ifdef {
                    inside_ifdef = false;
                    skip_until_endif(&mut lines);
                }
            } else if line.starts_with("#endif") {
                inside_ifdef = false;
            } else if line.starts_with("#define") {
                let mut parts = line.split_whitespace();
                parts.next(); // Skip the directive
                let key = parts.next().expect("No key specified in #define").to_string();
                let value = parts.next().unwrap_or("").to_string();
                defines.insert(key, value);
            } else {
                let processed_line = replace_defines(line, &defines);
                result.push_str(&processed_line);
                result.push('\n');
            }
        }
    
        result
    }
    
    fn replace_defines(line: &str, defines: &HashMap<String, String>) -> String {
        let mut processed_line = String::from(line);
        for (key, value) in defines {
            processed_line = processed_line.replace(key, value);
        }
        for (key, value) in defines {
            processed_line = processed_line.replace(key, value);
        }
        processed_line
    }
    
    fn skip_until_else_or_endif<'a, I: Iterator<Item = &'a str>>(lines: &mut std::iter::Peekable<I>) {
        while let Some(line) = lines.next() {
            if line.starts_with("#else") || line.starts_with("#endif") {
                break;
            }
        }
    }
    
    fn skip_until_endif<'a, I: Iterator<Item = &'a str>>(lines: &mut std::iter::Peekable<I>) {
        while let Some(line) = lines.next() {
            if line.starts_with("#endif") {
                break;
            }
        }
    }

    mod shader_shortener{
        use std::collections::HashMap;
        use std::str::Chars;

        #[derive(Debug, PartialEq, Eq)]
        enum Token<'a> {
            Word(&'a str),
            Symbol(char)
        }

        struct Tokenizer<'a> {
            chars: Chars<'a>,
            input: &'a str,
            pos: usize,
        }

        impl<'a> Tokenizer<'a> {
            fn new(input: &'a str) -> Self {
                Tokenizer {
                    chars: input.chars(),
                    input,
                    pos: 0,
                }
            }
        }

        impl<'a> Iterator for Tokenizer<'a> {
            type Item = Token<'a>;

            fn next(&mut self) -> Option<Self::Item> {
                while let Some(c) = self.chars.next() {
                    self.pos += c.len_utf8();
                    if c.is_alphabetic() || c == '_' {
                        let start = self.pos - c.len_utf8();
                        while let Some(&next_c) = self.chars.clone().peekable().peek() {
                            if next_c.is_alphanumeric() || next_c == '_' {
                                self.chars.next();
                                self.pos += next_c.len_utf8();
                            } else {
                                break;
                            }
                        }
                        return Some(Token::Word(&self.input[start..self.pos]));
                    } else {
                        return Some(Token::Symbol(c));
                    }
                }
                None
            }
        }

        fn generate_short_name(counter: usize) -> (String,usize) {
            let alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            let mut name = String::new();
            let mut count = counter;
            let reserved_words = ["as", "if", "do", "fn", "of"];

            while count > 0 {
                name.push(alphabet.chars().nth((count - 1) % alphabet.len()).unwrap());
                count = (count - 1) / alphabet.len();
            }
            
            let name = name.chars().rev().collect::<String>();

            if reserved_words.contains(&name.as_str()) {
                generate_short_name(counter + 1)
            } else {
                (name, counter + 1)
            }

        }

        pub fn shorten_variable_names(shader_code: &str, global_functions : &mut HashMap<String, String>) -> String {
            let tokenizer = Tokenizer::new(shader_code);
            let mut result = String::new();
            let mut variables = HashMap::new();
            let mut functions = HashMap::new();
            let mut var_counter = 1;
            let mut is_compute_fn = false;

            let mut tokens = tokenizer.peekable();
            let mut prev_token = None;
            while let Some(token) = tokens.next() {
                match token {
                    Token::Word(word) if word == "let" || word=="var" || word=="const" => {
                        result.push_str(word);
                        let mut generic_counter = 0;
                        while let Some(token) = tokens.next() {
                            match token{
                                Token::Word(var_name) => {
                                    if generic_counter == 0{
                                        let short_name = variables.entry(var_name).or_insert_with(||
                                            {
                                            let (name, new_counter ) = generate_short_name(var_counter);
                                            var_counter = new_counter;
                                            name
                                        });
                                       
                                        result.push_str(short_name);
                                        break;
                                    }
                                    else{
                                        result.push_str(var_name);
                                    }
                                },
                                Token::Symbol(c) => {
                                    if c=='<'{
                                        generic_counter+=1;
                                    }
                                    else if c=='>'{
                                        generic_counter-=1;
                                    }
                                    result.push(c);
                                }
                                
                                
                            }
                        }
                    }
                    Token::Word(word) if word == "fn" => {
                        result.push_str(word);
                        while let Some(token) = tokens.next() {
                            match token{
                                Token::Word(var_name) => {
                                    if is_compute_fn{

                                        let short_name;
                                        if global_functions.contains_key(var_name){
                                            short_name = global_functions.get(var_name).unwrap().to_string();
                                        }
                                        else{
                                            loop {
                                                let (n, new_counter ) = generate_short_name(var_counter);
                                                var_counter = new_counter;
                                                if !global_functions.values().any(|c| *c == n){
                                                    short_name = n;
                                                    global_functions.insert(var_name.to_string(), short_name.to_string());
                                                    break;
                                                }
                                            }                
                                        }
                                        result.push('g');//ga, gb, etc...
                                        result.push_str(&short_name);
                                        is_compute_fn = false;
                                    }
                                    else{
                                        let short_name = functions.entry(var_name).or_insert_with(||
                                            {
                                                let (name, new_counter ) = generate_short_name(var_counter);
                                                var_counter = new_counter;
                                                name
                                            });
                                            
                                        result.push_str(short_name);
                                    }
                                    break;
                                },
                                Token::Symbol(c) => {
                                    result.push(c);
                                }
                            }
                        }
                    }
                    Token::Word(word) => {
                        let mut valid = true;
                        if word == "default"{
                            result.push_str(word);
                        }
                        else{
                            if let Some(p_token) = prev_token{
                                if let Token::Symbol(c) = p_token{
                                    if c == '.'{
                                        result.push_str(word);
                                        valid = false;
                                    }
                                    else if c.is_alphanumeric(){
                                        result.push_str(word);
                                        valid = false;
                                    }
                                }
                            }
                            if valid{
                                if let Some(nt) = tokens.peek(){ //this may be a function call
                                    if let Token::Symbol(c) = nt{
                                        if *c == '('{ //this is a function call
                                            if let Some(short_name) = functions.get(word) {
                                                result.push_str(short_name); 
                                            } else {
                                                result.push_str(word);
                                            } 
                                            valid = false;
                                        }
                                        else if *c == ':'{ //this is a variable definition, e.g. v1 : u32, v2 : u32
                                            let short_name = variables.entry(word).or_insert_with(||
                                            {
                                                let (name, new_counter ) = generate_short_name(var_counter);
                                                var_counter = new_counter;
                                                name
                                            });
                                            
                                            result.push_str(short_name);
                                            valid=false;
                                        }
                                    }
                                }
                                if valid{
                                    if let Some(short_name) = variables.get(word) {
                                        result.push_str(short_name); 
                                    } else {
                                        result.push_str(word);
                                    } 
                                }
                            }
                        }
                    }
                    Token::Symbol(c) => {
                        if c == '@'{
                            if let Some(word) = tokens.peek(){
                                if let Token::Word(word) = word{
                                    if *word == "compute"{
                                        is_compute_fn = true;
                                    }
                                }
                            }
                        }
                        result.push(c);}
                }
                prev_token = Some(token);
            }

            result
        }

    }
}

