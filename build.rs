use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn main() -> io::Result<()> {
    // 调试输出：开始执行 build.rs
    println!("cargo:warning=Build script started!");

    // 1. 获取 OUT_DIR（形如 target/release/build/xxx/out）
    let out_dir = env::var("OUT_DIR").unwrap();
    println!("cargo:warning=OUT_DIR: {}", out_dir);

    let mut dest_dir = PathBuf::from(&out_dir);

    // 2. 往上退三级，把路径修正到真正的可执行文件输出目录 (target/release)
    dest_dir.pop(); // out
    dest_dir.pop(); // build/xxx
    dest_dir.pop(); // build
    // 现在 dest_dir 理论上就是 target/release
    println!("cargo:warning=Destination directory: {:?}", dest_dir);

    // 3. 准备要复制的源目录和目标目录
    let models_src = PathBuf::from("models"); // 项目根目录下的 models
    let models_dst = dest_dir.join("models"); // target/release/models

    // 调试输出：源目录和目标目录
    println!("cargo:warning=Source directory: {:?}", models_src);
    println!("cargo:warning=Destination directory: {:?}", models_dst);

    // 检查源目录是否存在
    if !models_src.exists() {
        println!("cargo:warning=models directory does not exist!");
        return Ok(());
    }

    // 4. 递归复制整个目录
    println!("cargo:warning=Copying models directory...");
    if let Err(e) = copy_dir_all(&models_src, &models_dst) {
        println!("cargo:warning=Failed to copy models directory: {}", e);
        return Err(e);
    }
    println!("cargo:warning=Models directory copied successfully!");

    // 5. 如果 models 目录里的内容有变动，就让 Cargo 重新执行构建脚本
    println!("cargo:rerun-if-changed=models");

    // 调试输出：构建脚本完成
    println!("cargo:warning=Build script finished!");

    Ok(())
}

/// 递归复制整个目录
fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    if src.is_dir() {
        fs::create_dir_all(dst)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let file_name = entry.file_name();
            let src_path = entry.path();
            let dst_path = dst.join(file_name);

            if file_type.is_dir() {
                copy_dir_all(&src_path, &dst_path)?;
            } else {
                fs::copy(src_path, dst_path)?;
            }
        }
    }
    Ok(())
}