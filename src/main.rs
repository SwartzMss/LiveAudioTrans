use std::path::Path;
use std::time::Duration;
use std::thread;
use std::io;
use std::sync::{Arc, Mutex};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use log::{info, error,Level, LevelFilter};
use fern::Dispatch;
use std::fs::File;
use chrono::Local;
use std::panic;
use threadpool::ThreadPool;

mod download_model;
use download_model::download_file;

mod capture;
use capture::AudioCapture;

mod transcribe;
use transcribe::Whisper;

mod translate;
use translate::Translator;



fn setup_logging(log_to_file: bool) {
    // 全局设置为 Debug，保证 debug 日志也能通过
    let base_dispatch = Dispatch::new()
        .level(LevelFilter::Debug)
        .format(|out, message, record| {
            let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
            out.finish(format_args!(
                "[{}] [{}] [{}] {}",
                timestamp,
                record.target(),
                record.level(),
                message
            ))
        });
    
    // 创建终端输出：只显示 Info 及以上的日志
    let stdout_dispatch = Dispatch::new()
        .filter(|metadata| metadata.level() >= Level::Info)
        .chain(io::stdout());
    
    // 如果需要输出到文件，则配置文件日志（记录 Debug 及以上级别日志）
    if log_to_file {
        let file_dispatch = Dispatch::new()
            .chain(File::create("app.log").unwrap());
        
        // 合并终端和文件输出
        base_dispatch
            .chain(stdout_dispatch)
            .chain(file_dispatch)
            .apply()
            .unwrap();
    } else {
        base_dispatch.chain(stdout_dispatch).apply().unwrap();
    }
}

/// 确保模型文件存在，如果不存在则下载
fn ensure_model_exists(model_path: &str, download_url: &str) {
    if !Path::new(model_path).exists() {
        info!("Model file not found at {}. Downloading...", model_path);
        download_file(download_url, model_path);
    } 
}

fn main() {
    panic::set_hook(Box::new(|panic_info| {
        error!("Panic occurred: {:?}", panic_info);
    }));
    setup_logging(true);
    // 确保 Whisper 模型存在
    let whisper_model_path = "models/ggml-base-q5_1.bin";
    let whisper_download_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin";
    ensure_model_exists(whisper_model_path, whisper_download_url);

    // 确保翻译模型存在
    let translator_model_path = "models/model.safetensors";
    let translator_download_url = "https://huggingface.co/Helsinki-NLP/opus-mt-en-zh/resolve/refs%2Fpr%2F26/model.safetensors";
    ensure_model_exists(translator_model_path, translator_download_url);

    info!("Loading Whisper model...");
    let whisper = Arc::new(Mutex::new(Whisper::new(whisper_model_path)));
    info!("Whisper model loaded.");

    let thread_id = std::thread::current().id();
    info!("main thread: {:?}", thread_id);

    // 初始化翻译器
    let tokenizer_path_en = "models/tokenizer-marian-base-en.json";
    let tokenizer_path_zh = "models/tokenizer-marian-base-zh.json";
    let mut translator = Translator::new(translator_model_path, tokenizer_path_en, tokenizer_path_zh)
        .expect("Failed to load translator model");

        let thread_id = std::thread::current().id();
        info!("Audio callback in thread: {:?}", thread_id);
    // 创建音频数据传输的 channel
    let (audio_sender, audio_receiver): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = unbounded();

    // 将 Sender 传递给 capture 模块，采集到的数据会通过该 channel 发送
    let _audio_capture = AudioCapture::new_stream_with_sender(audio_sender);

    // 用于传递转录结果的 channel
    let (result_sender, result_receiver): (Sender<String>, Receiver<String>) = bounded(10);

    // 创建一个线程池，大小为 10
    let pool = ThreadPool::new(1);

    // 启动一个监听线程：从 audio_receiver 中读取音频块，然后丢到线程池中处理
    let transcribe_result_sender = result_sender.clone();
    let whisper_for_thread = Arc::clone(&whisper);
    thread::spawn(move || {
        while let Ok(chunk) = audio_receiver.recv() {
            let whisper_handle = Arc::clone(&whisper_for_thread);
            let result_handle = transcribe_result_sender.clone();

            // 提交一个任务给线程池
            pool.execute(move || {
                let mut whisper_locked = whisper_handle.lock().unwrap();
                info!(" start");
                if let Some(text) = whisper_locked.transcribe_samples(chunk) {
                    let _ = result_handle.send(text.trim().to_string());
                }
                info!(" end");
            });
        }
    });

    // 主线程：处理转录结果，并进行翻译
    info!("Starting real-time transcription loop...");
    loop {
        while let Ok(text) = result_receiver.try_recv() {
            info!("result_receiver{}", text);
            // match translator.translate(&text) {
            //     Ok(translated) => {
            //         if text.trim() == translated.trim() {
            //             info!("[live translate] {}", text);
            //         } else {
            //             info!("[live translate] 英文: {}\n          中文: {}", text, translated);
            //         }
            //     }
            //     Err(e) => eprintln!("Translation error: {:?}", e),
            // }
        }
        thread::sleep(Duration::from_millis(50));
    }
}
