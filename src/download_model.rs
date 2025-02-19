use reqwest::blocking::Client;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

/// 下载文件，并保存到指定路径。如果下载失败则 panic。
pub fn download_file(url: &str, output_path: &str) {
    println!("Downloading from {} ...", url);

    // 使用 Client 构建器设置超时时间（例如 600 秒）
    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .expect("Failed to build client");

    let response = client.get(url).send().expect("Failed to send request");

    // 检查响应状态码
    if !response.status().is_success() {
        panic!("Failed to download file: HTTP {}", response.status());
    }

    // 如果输出目录不存在，则创建
    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).expect("Failed to create directories");
    }

    let mut file = File::create(output_path).expect("Failed to create file");
    // 直接将响应体复制到文件中
    let content = response.bytes().expect("Failed to read response bytes");
    file.write_all(&content).expect("Failed to write to file");
    println!("Download completed and saved to {}", output_path);
}
