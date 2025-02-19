use std::path::Path;
use std::time::Duration;
use std::thread;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use std::collections::VecDeque;

mod download_model;
use download_model::download_file;

mod capture;
use capture::AudioCapture;

mod transcribe;
use transcribe::Whisper;

mod translate;
use translate::Translator;

/// 确保模型文件存在，如果不存在则下载
fn ensure_model_exists(model_path: &str, download_url: &str) {
    if !Path::new(model_path).exists() {
        println!("Model file not found at {}. Downloading...", model_path);
        download_file(download_url, model_path);
    } 
}

fn main() {
    // 确保 Whisper 模型存在
    let whisper_model_path = "models/ggml-base-q5_1.bin";
    let whisper_download_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin";
    ensure_model_exists(whisper_model_path, whisper_download_url);

    // 确保翻译模型存在
    let translator_model_path = "models/model.safetensors";
    let translator_download_url = "https://huggingface.co/Helsinki-NLP/opus-mt-en-zh/resolve/refs%2Fpr%2F26/model.safetensors";
    ensure_model_exists(translator_model_path, translator_download_url);

    println!("Loading Whisper model...");
    let mut whisper = Whisper::new(whisper_model_path);
    println!("Whisper model loaded.");

    // 初始化翻译器
    let tokenizer_path_en = "models/tokenizer-marian-base-en.json";
    let tokenizer_path_zh = "models/tokenizer-marian-base-zh.json";
    let mut translator = Translator::new(translator_model_path, tokenizer_path_en, tokenizer_path_zh)
        .expect("Failed to load translator model");

    // 在 main.rs 中创建音频数据传输的 channel
    let (audio_sender, audio_receiver): (Sender<Vec<i16>>, Receiver<Vec<i16>>) = unbounded();

    // 将 Sender 传递给 capture 模块，由 capture 模块采集音频后通过该 channel 发送数据
    let _audio_capture = AudioCapture::new_stream_with_sender(audio_sender);
    let sample_rate = _audio_capture.sample_rate() as u32;

    // 用于传递转录结果的 channel
    let (result_sender, result_receiver): (Sender<String>, Receiver<String>) = bounded(10);

    // --- 滑动窗口参数 --- 
    // 窗口大小为 1 秒的数据（满足 Whisper 的最低要求）
    let window_size = sample_rate as usize; 
    // 每隔 300ms更新一次（即步长为 300ms对应的采样点数）
    let step_size = sample_rate as usize / 3;

    // 用 VecDeque 来维护滑动窗口
    let mut sliding_buffer: VecDeque<i16> = VecDeque::new();

    // 转录线程：使用滑动窗口策略，每次从窗口中取出 1 秒数据进行转录
    let sliding_result_sender = result_sender.clone();
    let sliding_sample_rate = sample_rate;
    // 将 Whisper 实例移入线程中（如果需要在同一线程使用，可使用 Arc+Mutex 包裹）
    // 这里为简单演示，将其直接 move 进闭包（注意：此示例中 whisper 不再在 main 中使用）
    thread::spawn(move || {
        // 循环等待音频数据
        loop {
            // 尝试从 channel 中接收音频数据块
            if let Ok(chunk) = audio_receiver.recv() {
                // 将新采集的音频数据加入滑动窗口
                for sample in chunk {
                    sliding_buffer.push_back(sample);
                }

                // 保持滑动窗口的长度不超过 window_size + step_size，
                // 这样能保证每次提取最近 window_size 的数据即可
                while sliding_buffer.len() > window_size + step_size {
                    sliding_buffer.pop_front();
                }

                // 如果滑动窗口中的数据已经达到 1 秒，则取最近 1 秒的数据进行转录
                if sliding_buffer.len() >= window_size {
                    let window: Vec<i16> = sliding_buffer
                        .iter()
                        .cloned()
                        .skip(sliding_buffer.len() - window_size)
                        .collect();
                    // 转换为 f32 并归一化（注意：Whisper 需要 [-1.0, 1.0] 的数据）
                    let samples_f32: Vec<f32> = window.iter().map(|&s| s as f32 / 32768.0).collect();

                    if let Some(text) = whisper.transcribe_samples(samples_f32, sliding_sample_rate, 1) {
                        sliding_result_sender.send(text.trim().to_string()).unwrap();
                    }
                }
            }
            // 避免 CPU 占用过高，可稍微 sleep 一下
            thread::sleep(Duration::from_millis(10));
        }
    });

    // 主线程：处理转录结果，并进行翻译
    println!("Starting real-time transcription loop...");
    loop {
        while let Ok(text) = result_receiver.try_recv() {
            println!("Raw transcription: {}", text);
            // 调用翻译模块
            match translator.translate(&text) {
                Ok(translated) => {
                    // 如果翻译结果与原文相同，则认为原文为中文，直接输出
                    if text.trim() == translated.trim() {
                        println!("[实时转录] {}", text);
                    } else {
                        println!("[实时转录] 英文: {}\n          中文: {}", text, translated);
                    }
                }
                Err(e) => eprintln!("Translation error: {:?}", e),
            }
        }
        thread::sleep(Duration::from_millis(50));
    }
}
