//! EfficientNet implementation.
//!
//! https://arxiv.org/abs/1905.11946

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::{Parser, ValueEnum};
use lantern::{DType, IndexOp, D};
use lantern_nn::{Module, VarBuilder};
use lantern_transformers::models::efficientnet::{EfficientNet, MBConvConfig};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Which {
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Variant of the model to use.
    #[arg(value_enum, long, default_value_t = Which::B2)]
    which: Which,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = lantern_examples::device(args.cpu)?;

    let image = lantern_examples::imagenet::load_image224(args.image)?.to_device(&device)?;
    println!("loaded image {image:?}");

    let model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/lantern-efficientnet".into());
            let filename = match args.which {
                Which::B0 => "efficientnet-b0.safetensors",
                Which::B1 => "efficientnet-b1.safetensors",
                Which::B2 => "efficientnet-b2.safetensors",
                Which::B3 => "efficientnet-b3.safetensors",
                Which::B4 => "efficientnet-b4.safetensors",
                Which::B5 => "efficientnet-b5.safetensors",
                Which::B6 => "efficientnet-b6.safetensors",
                Which::B7 => "efficientnet-b7.safetensors",
            };
            api.get(filename)?
        }
        Some(model) => model.into(),
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let cfg = match args.which {
        Which::B0 => MBConvConfig::b0(),
        Which::B1 => MBConvConfig::b1(),
        Which::B2 => MBConvConfig::b2(),
        Which::B3 => MBConvConfig::b3(),
        Which::B4 => MBConvConfig::b4(),
        Which::B5 => MBConvConfig::b5(),
        Which::B6 => MBConvConfig::b6(),
        Which::B7 => MBConvConfig::b7(),
    };
    let model = EfficientNet::new(vb, cfg, lantern_examples::imagenet::CLASS_COUNT as usize)?;
    println!("model built");
    let logits = model.forward(&image.unsqueeze(0)?)?;
    let prs = lantern_nn::ops::softmax(&logits, D::Minus1)?
        .i(0)?
        .to_vec1::<f32>()?;
    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for &(category_idx, pr) in prs.iter().take(5) {
        println!(
            "{:24}: {:.2}%",
            lantern_examples::imagenet::CLASSES[category_idx],
            100. * pr
        );
    }
    Ok(())
}
