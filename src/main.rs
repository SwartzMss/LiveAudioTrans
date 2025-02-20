use std::path::Path;
use std::time::Duration;
use std::thread;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};

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

    // 创建音频数据传输的 channel
    let (audio_sender, audio_receiver): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();

    // 将 Sender 传递给 capture 模块，采集到的数据会通过该 channel 发送
    let _audio_capture = AudioCapture::new_stream_with_sender(audio_sender);

    // 用于传递转录结果的 channel
    let (result_sender, result_receiver): (Sender<String>, Receiver<String>) = bounded(10);

    // 启动转录线程：直接处理每个收到的音频数据块
    let transcribe_result_sender = result_sender.clone();
    thread::spawn(move || {
        loop {
            if let Ok(chunk) = audio_receiver.recv() {
                // 此处直接使用采集到的 f32 数据，不再做归一化处理
                if let Some(text) = whisper.transcribe_samples(chunk) {
                    transcribe_result_sender.send(text).unwrap();
                }
            }
        }
    });

    // 主线程：处理转录结果，并进行翻译
    println!("Starting real-time transcription loop...");
    loop {
        while let Ok(text) = result_receiver.try_recv() {
            println!("[live translate] {}", text);
            // match translator.translate(&text) {
            //     Ok(translated) => {
            //         if text.trim() == translated.trim() {
            //             println!("[live translate] {}", text);
            //         } else {
            //             println!("[live translate] 英文: {}\n          中文: {}", text, translated);
            //         }
            //     }
            //     Err(e) => eprintln!("Translation error: {:?}", e),
            // }
        }
        thread::sleep(Duration::from_millis(50));
    }
}
