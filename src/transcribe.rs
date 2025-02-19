use samplerate::{convert, ConverterType};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

/// Whisper 结构体封装了 Whisper 状态，
/// 并提供从 WAV 文件转录文本的接口。
pub struct Whisper {
    /// Whisper 内部状态，用于执行转录操作
    whisper_state: WhisperState,
    /// 固定的目标采样率为 16000Hz
    sample_rate_target: u32,
}

impl Whisper {
    /// 根据指定的模型文件路径创建一个新的 Whisper 转录器。
    ///
    /// # 参数
    ///
    /// * `whisper_model_path` - Whisper 模型文件路径（例如 "models/ggml-whisper.bin"）
    ///
    /// # Panics
    ///
    /// 如果创建 WhisperContext 或状态失败，则会直接 panic。
    pub fn new(whisper_model_path: &str) -> Self {
        let ctx = WhisperContext::new_with_params(
            whisper_model_path,
            WhisperContextParameters {
                use_gpu: true,
                flash_attn: false,
                ..Default::default()
            },
        )
        .expect("failed to create WhisperContext");
        let state = ctx.create_state().expect("failed to create Whisper state");
        Self {
            whisper_state: state,
            sample_rate_target: 16000,
        }
    }

     /// 对音频数据进行转录，并返回识别的文本。
    ///
    /// 如果输入数据的采样率不是 16000Hz，则会自动进行重采样。
    /// 注意：仅支持单声道音频数据。
    ///
    /// # 参数
    ///
    /// * `samples` - 单声道音频数据（f32 格式，范围[-1.0, 1.0]）
    /// * `input_sample_rate` - 输入数据的原始采样率
    /// * `channels` - 音频通道数（必须为 1）
    ///
    /// # 返回值
    ///
    /// 成功时返回 `Some(转录文本)`；如果转录过程中出现问题，则会 panic 或返回 None。
    pub fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        input_sample_rate: u32,
        channels: u16,
    ) -> Option<String> {
        // 检查通道数
        if channels != 1 {
            panic!("Only mono audio data is supported, current number of channels: {}", channels);
        }
    
        // 如果采样率不匹配，则进行重采样
        let mut samples = if input_sample_rate != self.sample_rate_target {
            println!(
                "need audio_resample, since input_sample_rate is {} and self.sample_rate_target {}",
                input_sample_rate, self.sample_rate_target
            );
            audio_resample(&samples, input_sample_rate, self.sample_rate_target)
        } else {
            samples
        };
    
        // 如果输入数据不足 1.1 秒，则补充静音数据（0.0）以满足最小时长要求
        let min_samples = (self.sample_rate_target as f64 * 1.1) as usize;
        if samples.len() < min_samples {
            let pad_count = min_samples - samples.len();
            samples.extend(vec![0.0; pad_count]);
            println!("Padded to {} samples ({} seconds)", 
            samples.len(),
            samples.len() as f32 / self.sample_rate_target as f32);        
        }
    
        // 配置转录参数
        let mut params = FullParams::new(SamplingStrategy::default());
        params.set_debug_mode(false);
        params.set_language(Some("auto")); // 自动检测语言
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params.set_logprob_thold(-2.0);
        params.set_temperature(0.0);
        // 执行转录
        self.whisper_state
            .full(params, &samples)
            .expect("transcription failed");
    
        // 收集识别结果
        let mut result = String::new();
        let num_segments = self
            .whisper_state
            .full_n_segments()
            .expect("Failed to get number of segments");
        for i in 0..num_segments {
            if let Ok(segment_text) = self.whisper_state.full_get_segment_text_lossy(i) {
                result.push_str(&segment_text);
                result.push('\n');
            }
        }
        println!("Transcription result: {}", result);
        Some(result)
    }
    
}

/// 对音频数据进行重采样，从原始采样率转换到目标采样率。
///
/// 使用 SincBestQuality 算法进行转换，且仅支持单声道音频数据。
///
/// # 参数
///
/// * `data` - 输入音频数据（f32 数组）
/// * `sample_rate0` - 原始采样率
/// * `sample_rate` - 目标采样率
///
/// # Panics
///
/// 如果重采样失败，则会 panic。
pub fn audio_resample(data: &[f32], sample_rate0: u32, sample_rate: u32) -> Vec<f32> {
    convert(
        sample_rate0,
        sample_rate,
        1, // 单声道
        ConverterType::SincBestQuality,
        data,
    )
    .expect("failed to resample")
}