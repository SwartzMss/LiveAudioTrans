# Live Audio Transcription & Translation

[![Rust](https://img.shields.io/badge/Rust-1.72%2B-blue)](https://www.rust-lang.org/)

实时音频流转录（基于Whisper）与翻译（基于MarianNMT）工具，支持CPU/GPU加速。

## 功能特性

- 🎤 实时音频采集（16kHz 单声道）
- 🔊 自动重采样和声道转换
- 🤖 Whisper语音识别（支持中英文）
- 🌍 MarianNMT 机器翻译（英译中）
- ⚡ 多线程处理架构
- 📊 日志记录与异常捕获
- 🚀 支持CUDA加速（需要NVIDIA GPU）


## 安装 & 依赖

本项目支持 **CPU 加速模式** 和 **CUDA 加速模式** 两种编译方式，默认情况下采用 CPU 加速模式。如果需要使用 CUDA 加速，请启用 `cuda` feature。

## CPU 加速模式（默认）

```bash
cargo build --release
```

## CUDA 加速模式

```bash
cargo build --release --features cuda
```

**依赖库** 

**Whisper 模型文件**  
   - 默认会在 `models` 目录下查找 `ggml-base-q5_1.bin`，若不存在则会自动从 Hugging Face 下载。
   - 若需使用其他 Whisper 模型文件，可修改 `main.rs` 中 `Transcribe` 命令的 `model_path` 与对应的下载地址 `download_url`。

**Marian 翻译模型文件**  
   - 默认会在 `models` 目录下查找 `model.safetensors` (以及对应分词器 `tokenizer-marian-base-en.json`、`tokenizer-marian-base-zh.json`)。  
   - 不存在时会从 Hugging Face 下载一份示例模型文件并存储到 `models` 文件夹下。  
   - 若想替换成其他支持英->中翻译的 Marian 模型，可在代码中调整相关配置。

**注意事项**
   - CPU加速模式：本项目使用 OpenMP 进行并行计算，运行时需要 vcomp140.dll 文件。若缺少该文件，请确保已安装支持的 Visual C++ Redistributable 包。

