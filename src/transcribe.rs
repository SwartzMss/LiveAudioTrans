use log::{info, debug, error};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

/// Whisper 结构体封装了 Whisper 状态，
/// 并提供从 WAV 文件转录文本的接口。
pub struct Whisper {
    /// Whisper 内部状态，用于执行转录操作
    whisper_state: WhisperState,
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
    ) -> Option<String> {

        // 配置转录参数
        let mut params = FullParams::new(SamplingStrategy::default());
        params.set_debug_mode(false);
        params.set_language(Some("en")); // 自动检测语言
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
        Some(result)
    }
    
}

