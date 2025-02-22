use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use samplerate::{convert, ConverterType};
use crossbeam_channel::Sender;
use std::sync::{Arc, Mutex};
use log::{info, error};

/// 一个简单的 `AudioCapture` 结构，持有一个可选的 `Stream`
pub struct AudioCapture {
    stream: Option<Stream>,
}

impl AudioCapture {
    /// 创建并启动音频采集，将处理后的音频（16k 单声道）通过 `audio_sender` 发送出去。
    pub fn new_stream_with_sender(audio_sender: Sender<Vec<f32>>) -> Self {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("Failed to get default output device");
        info!("Using device: {}", device.name().unwrap_or_else(|_| "Unknown".into()));

        let supported_config = device
            .default_output_config()
            .expect("Failed to get default output config");
        info!("Default output config: {:?}", supported_config);

        // 把 `SupportedStreamConfig` 转成通用的 `StreamConfig`
        let sample_format = supported_config.sample_format();
        let stream_config: StreamConfig = supported_config.into();

        let err_fn = |err| error!("Stream error: {}", err);

        let data_accumulator = Arc::new(Mutex::new(DataAccumulator {
            buffer: Vec::new(),
            counter: 0,
        }));

        let input_channels = stream_config.channels as usize;
        let input_sample_rate = stream_config.sample_rate.0;
        let target_sample_rate = 16_000;

        // 阈值大约为 1.1 秒、且累计调用次数也到达一定量再发送
        let send_frames_threshold = (target_sample_rate as f64 * 1.1) as usize;
        let send_counter_threshold = (16_000.0 / 320.0 * 0.6) as usize;

        let stream = match sample_format {
            SampleFormat::F32 => {
                device.build_input_stream(
                    &stream_config,
                    {
                        let data_accumulator = Arc::clone(&data_accumulator);
                        move |data: &[f32], _| {
                            // 如果是双声道则混合，否则直接拷贝
                            let mono_samples = if input_channels == 1 {
                                data.to_vec()
                            } else if input_channels == 2 {
                                stereo_to_mono(data)
                            } else {
                                panic!("Unsupported number of channels: {}", input_channels);
                            };

                            // 重采样到 16k
                            let processed = if input_sample_rate != target_sample_rate {
                                audio_resample(&mono_samples, input_sample_rate, target_sample_rate)
                            } else {
                                mono_samples
                            };

                            // 累计数据到共享 buffer 里
                            let mut acc = data_accumulator.lock().unwrap();
                            acc.buffer.extend(processed);
                            acc.counter += 1;

                            // 检查是否达到发送门槛
                            if acc.counter > send_counter_threshold
                                && acc.buffer.len() >= send_frames_threshold
                            {
                                // 把数据拷贝一份，再清空 buffer
                                let to_send = acc.buffer.clone();
                                acc.buffer.clear();
                                acc.counter = 0;
                                drop(acc);

                                // 将数据发送到主线程或其他处理端
                                if let Err(e) = audio_sender.send(to_send) {
                                    eprintln!("Failed to send processed audio data: {}", e);
                                }
                            }
                        }
                    },
                    err_fn,
                    None,
                )
            }
            other => {
                panic!("Unsupported sample format: {:?}", other);
            }
        }
        .expect("Failed to build input stream");

        // 开始播放（录音实际上就是“启动回调”）
        stream.play().expect("Failed to start stream");

        Self {
            stream: Some(stream),
        }
    }

    #[allow(dead_code)]
    /// 停止录音
    pub fn stop(&mut self) {
        if let Some(s) = self.stream.take() {
            if let Err(e) = s.pause() {
                error!("Failed to pause stream: {}", e);
            }
        }
    }
}

/// 一个简单的结构，用来在回调中累计数据
struct DataAccumulator {
    buffer: Vec<f32>,
    counter: usize,
}

/// 将立体声数据（f32）混合为单声道
fn stereo_to_mono(stereo_data: &[f32]) -> Vec<f32> {
    let mut mono = Vec::with_capacity(stereo_data.len() / 2);
    for chunk in stereo_data.chunks(2) {
        let m = (chunk[0] + chunk[1]) * 0.5;
        mono.push(m);
    }
    mono
}

/// 用 `samplerate` crate 进行重采样
fn audio_resample(data: &[f32], sr_in: u32, sr_out: u32) -> Vec<f32> {
    convert(
        sr_in,
        sr_out,
        1, // 单声道
        ConverterType::SincBestQuality,
        data,
    )
    .expect("failed to resample")
}
