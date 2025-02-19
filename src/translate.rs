use anyhow::Error as E;
use candle_core::{Device, Tensor, DType};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::marian::{self, MTModel};
use tokenizers::Tokenizer;

pub struct Translator {
    model: MTModel,
    config: marian::Config,
    tokenizer: Tokenizer,      // 用于对英文文本进行编码
    tokenizer_dec: Tokenizer,  // 用于对生成的 token 进行解码（中文）
    device: Device,
}

impl Translator {
    /// 创建一个新的 `Translator`。
    ///
    /// - `model_path` 指向 safetensors 文件路径（例如 "opus-mt-en-zh/model.safetensors"）。
    /// - `en_token` 为英文分词器文件，比如 "opus-mt-en-zh/tokenizer-en.json"。
    /// - `zh_token` 为中文分词器文件，比如 "opus-mt-en-zh/tokenizer-zh.json"。
    pub fn new(model_path: &str, en_token: &str, zh_token: &str) -> anyhow::Result<Self> {
        println!("Initializing Translator with model_path: {}, en_token: {}, zh_token: {}", model_path, en_token, zh_token);

        // 选择设备，这里使用 CPU，如需 GPU/MPS 可改为 `Device::new_metal(0)` 等
        let device = Device::Cpu;

        // 从 safetensors 文件创建 VarBuilder，注意这里使用了 unsafe，
        // 但其含义只是“零拷贝”映射，不会真正不安全。
        println!("Creating VarBuilder from model_path...");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)?
        };

        // Marian 配置，可根据您的实际模型配置进行修改
        // 这里是 Helsinki-NLP/opus-mt-en-zh 的典型参数
        let config = marian::Config {
            vocab_size: 65001,
            decoder_vocab_size: Some(65001),
            max_position_embeddings: 512,
            encoder_layers: 6,
            encoder_ffn_dim: 2048,
            encoder_attention_heads: 8,
            decoder_layers: 6,
            decoder_ffn_dim: 2048,
            decoder_attention_heads: 8,
            use_cache: true,
            is_encoder_decoder: true,
            activation_function: Activation::Swish,
            d_model: 512,
            decoder_start_token_id: 65000,
            scale_embedding: true,
            pad_token_id: 65000,
            eos_token_id: 0,
            forced_eos_token_id: 0,
            share_encoder_decoder_embeddings: true,
        };

        // 分别加载英文分词器和中文分词器
        println!("Loading English tokenizer from file: {}", en_token);
        let tokenizer = Tokenizer::from_file(en_token).map_err(E::msg)?;
        println!("Loading Chinese tokenizer from file: {}", zh_token);
        let tokenizer_dec = Tokenizer::from_file(zh_token).map_err(E::msg)?;

        // 创建 Marian 模型
        println!("Creating Marian model...");
        let model = MTModel::new(&config, vb)?;

        println!("Translator initialized successfully.");
        Ok(Self {
            model,
            config,
            tokenizer,
            tokenizer_dec,
            device,
        })
    }

    /// 翻译：若文本是英文，则进行翻译；否则原样返回
    pub fn translate(&mut self, text: &str) -> anyhow::Result<String> {
        println!("Translating text: {}", text);

        // 1. 判断是否英文
        if !is_english(text) {
            println!("Text is not in English, returning original text.");
            return Ok(text.to_string());
        }

        // 2. 先经过 encoder 编码
        println!("Encoding text with English tokenizer...");
        let mut tokens = self
            .tokenizer
            .encode(text, /* add_special_tokens = */ true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // 将 encoder 端的 eos token 写入序列
        tokens.push(self.config.eos_token_id);
        let tokens = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let encoder_xs = self.model.encoder().forward(&tokens, /* start_pos = */ 0)?;

        // 3. decoder 端逐 token 解码
        let mut token_ids = vec![self.config.decoder_start_token_id];
        // 随机数种子、其他采样参数在这里指定
        let mut logits_processor = LogitsProcessor::new(/*seed=*/1337, /*top_k=*/None, /*top_p=*/None);

        // 这里设一个最大循环步数，以免无法收敛时死循环。可视需求调整
        for index in 0..128 {
            // decoder 部分：可以只拿最后一个 token 送入，也可以拿整个序列
            // 下面的逻辑是“第一步用全部，后续一步只输入一个新 token”。
            let context_size = if index >= 1 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&token_ids[start_pos..], &self.device)?.unsqueeze(0)?;

            // decoder forward
            let logits = self.model.decode(&input_ids, &encoder_xs, start_pos)?;
            // logits shape: [batch=1, seq_len, vocab_size] => 先移除 batch=1
            let logits = logits.squeeze(0)?;
            // 取序列最后一个 step 的 logits
            let logits = logits.get(logits.dim(0)? - 1)?;

            // 用采样策略拿到下一个 token
            let next_token = logits_processor.sample(&logits)?;

            // 如果是 eos 或 forced_eos，就提前结束
            if next_token == self.config.eos_token_id
                || next_token == self.config.forced_eos_token_id
            {
                break;
            }

            token_ids.push(next_token);
        }

        // 4. 用 tokenizer_dec 解码生成后的序列（去掉开头的 decoder_start_token_id）
        println!("Decoding generated tokens with Chinese tokenizer...");
        let translation = self
            .tokenizer_dec
            .decode(&token_ids[1..], /* skip_special_tokens = */ true)
            .map_err(E::msg)?;

        // 5. 注意要清空 KV 缓存，避免下次翻译时冲突
        self.model.reset_kv_cache();

        println!("Translation completed successfully.");
        Ok(translation)
    }
}

/// 判断文本是否主要由英文字母构成
fn is_english(text: &str) -> bool {
    let en_chars = text.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let total_chars = text.chars().filter(|c| !c.is_whitespace()).count();

    if total_chars == 0 {
        return false;
    }
    en_chars as f32 / total_chars as f32 > 0.5
}