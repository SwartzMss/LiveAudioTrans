use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use samplerate::{convert, ConverterType};
use num_traits::ToPrimitive;
use crossbeam_channel::{Sender, unbounded};
use std::thread;

pub struct AudioCapture {
    stream: Option<Stream>,
}

impl AudioCapture {
    /// 创建一个 AudioCapture 实例，启动采集，并将采集到的数据通过传入的 Sender 发送出去
    /// 这里我们先将原始数据通过 raw_sender 从 CPAL 回调传出，
    /// 然后在单独的线程中做混合/重采样等处理，最终发送到 audio_sender
    pub fn new_stream_with_sender(audio_sender: Sender<Vec<f32>>) -> Self {
        let host = cpal::default_host();
        // 使用默认输出设备采集音频
        let device = host
            .default_output_device()
            .expect("Failed to get default output device");
        println!(
            "Using device: {}",
            device.name().unwrap_or_else(|_| "Unknown".to_string())
        );
        
        // 获取设备的默认输出配置
        let config = device
            .default_output_config()
            .expect("Failed to get default output config");
        println!("Default output config: {:?}", config);
        
        let sample_format = config.sample_format();
        // 转换为 StreamConfig（消耗 config）
        let stream_config: StreamConfig = config.into();
        
        let err_fn = |err| eprintln!("Stream error: {}", err);
        
        // 创建一个中间 channel，用于在回调中发送原始数据
        let (raw_tx, raw_rx) = unbounded::<Vec<f32>>();
        
        // 根据采样数据格式构建 stream，这里只支持 f32
        let stream = match sample_format {
            SampleFormat::F32 => {
                Self::build_stream::<f32>(&device, &stream_config, raw_tx, err_fn)
            },
            _ => panic!("Unsupported sample format"),
        }
        .expect("Failed to build stream");
        
        stream.play().expect("Failed to play stream");

        // 单独启动一个线程，处理原始数据（比如通道混合、重采样等）
        thread::spawn(move || {
            let channels = stream_config.channels as usize;
            let input_sample_rate = stream_config.sample_rate.0;
            let target_sample_rate = 16000; // 目标采样率 16kHz
        
            // 本地缓存和计数器
            let mut buffer: Vec<f32> = Vec::new();
            let mut counter: usize = 0;
            // 参考代码中阈值：(16000 / 320 * 0.6) ≈ 30
            let send_threshold = (16000.0 / 320.0 * 0.6) as usize;
        
            for raw_samples in raw_rx.iter() {
                // 1. 处理通道：如果是双通道，则混合为单通道
                let mono_samples = if channels == 1 {
                    raw_samples
                } else if channels == 2 {
                    let mut mono = Vec::with_capacity(raw_samples.len() / 2);
                    for frame in raw_samples.chunks(2) {
                        if frame.len() < 2 { break; }
                        let mixed = (frame[0] + frame[1]) / 2.0;
                        mono.push(mixed);
                    }
                    mono
                } else {
                    panic!("Unsupported number of channels: {}", channels);
                };
        
                // 2. 重采样处理：如果采样率不匹配，则进行重采样
                let processed_samples = if input_sample_rate != target_sample_rate {
                    Self::audio_resample(&mono_samples, input_sample_rate, target_sample_rate)
                } else {
                    mono_samples
                };
        
                // 将处理后的数据追加到缓存中
                buffer.extend(processed_samples);
                counter += 1;
        
                // 当累计数据超过 1.1s（17600 个采样点）且计数器超过 send_threshold 后进行发送
                if counter > send_threshold && (buffer.len() as f64) >= 1.1 * (target_sample_rate as f64) {
                    let samples = buffer.clone();
                    if let Err(err) = audio_sender.send(samples) {
                        eprintln!("Failed to send processed audio data: {}", err);
                    }
                    // 重置缓存和计数器
                    buffer.clear();
                    counter = 0;
                }
            }
        });
        
        
        Self {
            stream: Some(stream),
        }
    }
    
    /// 停止采集流
    pub fn stop(&mut self) {
        if let Some(stream) = self.stream.take() {
            stream.pause().expect("Failed to pause stream");
        }
    }
    
    /// 构建采集流，将采集到的原始数据通过 raw_sender 发送出去
    fn build_stream<T>(
        device: &cpal::Device,
        config: &StreamConfig,
        raw_sender: Sender<Vec<f32>>,
        err_fn: impl Fn(cpal::StreamError) + Send + 'static,
    ) -> Result<Stream, cpal::BuildStreamError>
    where
        T: cpal::Sample + cpal::SizedSample + ToPrimitive + 'static,
    {
        device.build_input_stream(
            config,
            move |data: &[T], _| {
                // 将采样数据转换为 f32 数组
                let samples: Vec<f32> = data.iter()
                    .map(|&sample| sample.to_f32().unwrap())
                    .collect();
                
                // 将原始数据发送给处理线程
                if let Err(err) = raw_sender.send(samples) {
                    eprintln!("Failed to send raw audio data: {}", err);
                }
            },
            err_fn,
            None,
        )
    }

    /// 对音频数据进行重采样，从原始采样率转换到目标采样率。
    /// 使用 SincBestQuality 算法进行转换，且仅支持单通道音频数据。
    fn audio_resample(data: &[f32], sample_rate0: u32, sample_rate: u32) -> Vec<f32> {
        convert(
            sample_rate0,
            sample_rate,
            1, // 单声道
            ConverterType::SincBestQuality,
            data,
        )
        .expect("failed to resample")
    }
}
