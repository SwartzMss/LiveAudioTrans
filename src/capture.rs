use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use num_traits::ToPrimitive;
use crossbeam_channel::Sender;

/// AudioCapture 使用 cpal 采集音频，并通过外部传入的 channel 实时发送采集数据
pub struct AudioCapture {
    stream: Option<Stream>,
    sample_rate: usize, // 当前采样率（Hz）
}

impl AudioCapture {
    /// 创建一个 AudioCapture 实例，启动采集，并将采集到的数据通过传入的 Sender 发送出去
    pub fn new_stream_with_sender(audio_sender: Sender<Vec<i16>>) -> Self {
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
        
        let sample_rate = config.sample_rate().0 as usize;
        let sample_format = config.sample_format();
        // 转换为 StreamConfig（消耗 config）
        let stream_config: StreamConfig = config.into();
        
        let err_fn = |err| eprintln!("Stream error: {}", err);
        let stream = match sample_format {
            SampleFormat::I16 => {
                Self::build_stream::<i16>(&device, &stream_config, audio_sender.clone(), err_fn)
            },
            SampleFormat::F32 => {
                Self::build_stream::<f32>(&device, &stream_config, audio_sender.clone(), err_fn)
            },
            SampleFormat::F64 => {
                Self::build_stream::<f64>(&device, &stream_config, audio_sender.clone(), err_fn)
            },
            _ => panic!("Unsupported sample format"),
        }
        .expect("Failed to build stream");
        
        stream.play().expect("Failed to play stream");
        
        Self {
            stream: Some(stream),
            sample_rate,
        }
    }
    
    /// 返回当前设备采样率
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }
    
    /// 停止采集流
    pub fn stop(&mut self) {
        if let Some(stream) = self.stream.take() {
            stream.pause().expect("Failed to pause stream");
        }
    }
    
    /// 内部函数：构建采集流，将采集到的数据直接通过传入的 Sender 发送出去
    fn build_stream<T>(
        device: &cpal::Device,
        config: &StreamConfig,
        audio_sender: Sender<Vec<i16>>,
        err_fn: impl Fn(cpal::StreamError) + Send + 'static,
    ) -> Result<Stream, cpal::BuildStreamError>
    where
        T: cpal::Sample + cpal::SizedSample + ToPrimitive + 'static,
    {
        let channels = config.channels as usize;
        device.build_input_stream(
            config,
            move |data: &[T], _| {
                // println!("Captured {} frames", data.len() / channels);
                let mut samples: Vec<i16> = Vec::with_capacity(data.len());
                if channels == 1 {
                    // 单声道：直接转换采样数据
                    for &sample in data.iter() {
                        let sample_i16 = if std::mem::size_of::<T>() == 2 {
                            sample.to_i16().unwrap()
                        } else {
                            (sample.to_f32().unwrap() * 32767.0)
                                .clamp(-32768.0, 32767.0) as i16
                        };
                        samples.push(sample_i16);
                    }
                } else if channels == 2 {
                    // 立体声：将左右通道混合为单声道
                    for frame in data.chunks(2) {
                        if frame.len() < 2 {
                            break;
                        }
                        let left = frame[0].to_f32().unwrap();
                        let right = frame[1].to_f32().unwrap();
                        // 若采样为 i16，则先归一化
                        let (l, r) = if std::mem::size_of::<T>() == 2 {
                            (left / 32768.0, right / 32768.0)
                        } else {
                            (left, right)
                        };
                        let mixed = (l + r) / 2.0;
                        let sample_i16 = (mixed * 32767.0)
                            .clamp(-32768.0, 32767.0) as i16;
                        samples.push(sample_i16);
                    }
                } else {
                    panic!("Unsupported number of channels: {}", channels);
                }
                
                // 直接将采集到的数据发送出去
                if let Err(err) = audio_sender.send(samples) {
                    eprintln!("Failed to send audio data: {}", err);
                }
            },
            err_fn,
            None,
        )
    }
}
