// https://github.com/karpathy/llama2.c

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle_transformers::models::llama2_c as model;
use candle_transformers::models::llama2_c_weights as weights;
use candle_transformers::models::quantized_llama2_c as qmodel;
mod training;
use clap::{Parser, Subcommand};

use anyhow::{Error as E, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use candle::{IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::io::Write;
use tokenizers::Tokenizer;

use model::{Cache, Config, Llama};
use qmodel::QLlama;
use weights::TransformerWeights;

#[derive(Parser, Debug, Clone)]
struct InferenceCmd {
    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long, default_value = "")]
    prompt: String,

    /// Config file in binary or safetensors format.
    #[arg(long)]
    config: Option<String>,

    #[arg(long, default_value = "karpathy/tinyllamas")]
    model_id: String,

    /// The model to be used when getting it from the hub. Possible
    /// values are 'stories15M.bin', 'stories42M.bin', see more at:
    /// https://huggingface.co/karpathy/tinyllamas/tree/main
    #[arg(long, default_value = "stories15M.bin")]
    which_model: String,
}

#[derive(Parser, Debug, Clone)]
struct EvaluationCmd {
    /// A directory with the pre-tokenized dataset in the format generated by the tinystories.py
    /// script from llama2.c https://github.com/karpathy/llama2.c
    #[arg(long)]
    pretokenized_dir: Option<String>,

    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Config file in binary format.
    #[arg(long)]
    config: Option<String>,

    #[arg(long, default_value = "karpathy/tinyllamas")]
    model_id: String,

    /// The model to be used when getting it from the hub. Possible
    /// values are 'stories15M.bin', 'stories42M.bin', see more at:
    /// https://huggingface.co/karpathy/tinyllamas/tree/main
    #[arg(long, default_value = "stories15M.bin")]
    which_model: String,
}

#[derive(Parser, Debug, Clone)]
pub struct TrainingCmd {
    /// A directory with the pre-tokenized dataset in the format generated by the tinystories.py
    /// script from llama2.c https://github.com/karpathy/llama2.c
    #[arg(long)]
    pretokenized_dir: String,

    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    #[arg(long, default_value_t = 0.001)]
    learning_rate: f64,
}

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Inference(InferenceCmd),
    Eval(EvaluationCmd),
    Train(TrainingCmd),
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The task to be performed, inference, training or evaluation.
    #[command(subcommand)]
    task: Option<Task>,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Tokenizer config file.
    #[arg(long)]
    tokenizer: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

impl Args {
    fn tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_path = match &self.tokenizer {
            Some(config) => std::path::PathBuf::from(config),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("hf-internal-testing/llama-tokenizer".to_string());
                api.get("tokenizer.json")?
            }
        };
        Tokenizer::from_file(tokenizer_path).map_err(E::msg)
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match &args.task {
        None => {
            let cmd = InferenceCmd {
                temperature: None,
                top_p: None,
                prompt: "".to_string(),
                config: None,
                model_id: "karpathy/tinyllamas".to_string(),
                which_model: "stories15M.bin".to_string(),
            };
            run_inference(&cmd, &args)?
        }
        Some(Task::Inference(cmd)) => run_inference(cmd, &args)?,
        Some(Task::Eval(cmd)) => run_eval(cmd, &args)?,
        Some(Task::Train(cmd)) => training::run(cmd, &args)?,
    }
    Ok(())
}

enum Model {
    Llama(Llama),
    QLlama(QLlama),
}

impl Model {
    fn forward(&self, xs: &Tensor, pos: usize, cache: &mut Cache) -> anyhow::Result<Tensor> {
        match self {
            Self::Llama(l) => Ok(l.forward(xs, pos, cache)?),
            Self::QLlama(l) => Ok(l.forward(xs, pos, cache)?),
        }
    }
}

fn run_eval(args: &EvaluationCmd, common_args: &Args) -> Result<()> {
    use std::io::BufRead;

    let config_path = match &args.config {
        Some(config) => std::path::PathBuf::from(config),
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            println!("loading the model weights from {}", args.model_id);
            let api = api.model(args.model_id.clone());
            api.get(&args.which_model)?
        }
    };

    let tokenizer = common_args.tokenizer()?;

    let device = candle_examples::device(common_args.cpu)?;
    let mut file = std::fs::File::open(config_path)?;
    let config = Config::from_reader(&mut file)?;
    let weights = TransformerWeights::from_reader(&mut file, &config, &device)?;
    let vb = weights.var_builder(&config, &device)?;
    let mut cache = Cache::new(false, &config, vb.pp("rot"))?;
    let model = Llama::load(vb, config)?;

    let tokens = match &args.pretokenized_dir {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let model_id = "roneneldan/TinyStories"; // TODO: Make this configurable.
            println!("loading the evaluation dataset from {}", model_id);
            let api = api.dataset(model_id.to_string());
            let dataset_path = api.get("TinyStories-valid.txt")?;
            let file = std::fs::File::open(dataset_path)?;
            let file = std::io::BufReader::new(file);
            let mut tokens = vec![];
            for line in file.lines() {
                let line = line?.replace("<|endoftext|>", "<s>");
                let line = tokenizer.encode(line, false).map_err(E::msg)?;
                tokens.push(line.get_ids().to_vec())
            }
            tokens.concat()
        }
        Some(pretokenized_dir) => {
            // Use shard 0 for the test split, similar to llama2.c
            // https://github.com/karpathy/llama2.c/blob/ce05cc28cf1e3560b873bb21837638a434520a67/tinystories.py#L121
            let path = std::path::PathBuf::from(pretokenized_dir).join("data00.bin");
            let bytes = std::fs::read(path)?;
            // Tokens are encoded as u16.
            let mut tokens = vec![0u16; bytes.len() / 2];
            std::io::Cursor::new(bytes).read_u16_into::<LittleEndian>(&mut tokens)?;
            tokens.into_iter().map(|u| u as u32).collect::<Vec<u32>>()
        }
    };
    println!("dataset loaded and encoded: {} tokens", tokens.len());

    let seq_len = model.config.seq_len;
    let iter = (0..tokens.len()).step_by(seq_len).flat_map(|start_idx| {
        if start_idx + seq_len + 1 > tokens.len() {
            None
        } else {
            let tokens = &tokens[start_idx..start_idx + seq_len + 1];
            let inputs = Tensor::new(&tokens[..seq_len], &device);
            let targets = Tensor::new(&tokens[1..], &device);
            Some(inputs.and_then(|inputs| targets.map(|targets| (inputs, targets))))
        }
    });
    let batch_iter = candle_datasets::Batcher::new_r2(iter).batch_size(args.batch_size);
    for inp_tgt in batch_iter {
        let (inp, tgt) = inp_tgt?;
        let logits = model.forward(&inp, 0, &mut cache)?;
        let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
        println!("{}", loss.to_vec0::<f32>()?);
    }
    Ok(())
}

fn run_inference(args: &InferenceCmd, common_args: &Args) -> Result<()> {
    let config_path = match &args.config {
        Some(config) => std::path::PathBuf::from(config),
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            println!("loading the model weights from {}", args.model_id);
            let api = api.model(args.model_id.clone());
            api.get(&args.which_model)?
        }
    };

    let tokenizer = common_args.tokenizer()?;

    //let device = candle_examples::device(common_args.cpu)?;
    let device = pollster::block_on(candle::Device::new_webgpu(0))?;

    let is_gguf = config_path.extension().map_or(false, |v| v == "gguf");
    let is_safetensors = config_path
        .extension()
        .map_or(false, |v| v == "safetensors");
    let (model, config, mut cache) = if is_gguf {
        let vb = qmodel::VarBuilder::from_gguf(config_path, &device)?;
        let (_vocab_size, dim) = vb
            .get_no_shape("model.embed_tokens.weight")?
            .shape()
            .dims2()?;
        let config = match dim {
            64 => Config::tiny_260k(),
            288 => Config::tiny_15m(),
            512 => Config::tiny_42m(),
            768 => Config::tiny_110m(),
            _ => anyhow::bail!("no config for dim {dim}"),
        };
        let freq_cis_real = vb
            .get(
                (config.seq_len, config.head_size() / 2),
                "rot.freq_cis_real",
            )?
            .dequantize(&device)?;
        let freq_cis_imag = vb
            .get(
                (config.seq_len, config.head_size() / 2),
                "rot.freq_cis_imag",
            )?
            .dequantize(&device)?;

        let fake_vb = candle_nn::VarBuilder::from_tensors(
            [
                ("freq_cis_real".to_string(), freq_cis_real),
                ("freq_cis_imag".to_string(), freq_cis_imag),
            ]
            .into_iter()
            .collect(),
            candle::DType::F32,
            &device,
        );
        let cache = model::Cache::new(true, &config, fake_vb)?;
        let model = Model::QLlama(QLlama::load(vb, config.clone())?);
        (model, config, cache)
    } else if is_safetensors {
        let config = Config::tiny_15m();
        let tensors = candle::safetensors::load(config_path, &device)?;
        let vb = candle_nn::VarBuilder::from_tensors(tensors, candle::DType::F32, &device);
        let cache = model::Cache::new(true, &config, vb.pp("rot"))?;
        let model = Model::Llama(Llama::load(vb, config.clone())?);
        (model, config, cache)
    } else {
        let mut file = std::fs::File::open(config_path)?;
        let config = Config::from_reader(&mut file)?;
        println!("{config:?}");
        let weights = TransformerWeights::from_reader(&mut file, &config, &device)?;
        let vb = weights.var_builder(&config, &device)?;
        let cache = model::Cache::new(true, &config, vb.pp("rot"))?;
        let model = Model::Llama(Llama::load(vb, config.clone())?);
        (model, config, cache)
    };

    println!("starting the inference loop");
    let mut logits_processor = LogitsProcessor::new(299792458, args.temperature, args.top_p);
    let mut index_pos = 0;

    print!("{}", args.prompt);
    let mut tokens = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);

    let start_gen = std::time::Instant::now();
    for index in 0.. {
        if tokens.len() >= config.seq_len {
            break;
        }
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, index_pos, &mut cache)?;
        let logits = logits.i((0, logits.dim(1)? - 1))?;
        let logits = if common_args.repeat_penalty == 1. || tokens.is_empty() {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(common_args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                common_args.repeat_penalty,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    let dt = start_gen.elapsed();
    println!(
        "\n{} tokens generated ({:.2} token/s)\n",
        tokens.len(),
        tokens.len() as f64 / dt.as_secs_f64(),
    );
    #[cfg(feature = "wgpu_debug")]
    match device {
        candle::Device::WebGpu(gpu) => {
            let info = pollster::block_on(gpu.get_debug_info()).unwrap();
            let map2 = candle::wgpu::debug_info::calulate_measurment(&info);
            candle::wgpu::debug_info::save_list(&map2, "wgpu_infollama2-c_small.json").unwrap();
        },
        _ => {},
    };

    Ok(())
}
