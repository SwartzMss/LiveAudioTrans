use std::io::{self, Write};
use crossterm::{
    style::{Color, Print, ResetColor, SetForegroundColor},
    ExecutableCommand,
};

pub struct ScrollingDisplay {
    current_english: String,
    current_chinese: String,
}

impl ScrollingDisplay {
    pub fn new() -> Self {
        Self {
            current_english: String::new(),
            current_chinese: String::new(),
        }
    }

    // 检查是否是特殊标记（如[音乐]等）
    fn is_special_mark(text: &str) -> bool {
        text.starts_with('[') && text.ends_with(']')
    }

    pub fn add_text(&mut self, english: &str, chinese: &str) {
        // 如果是特殊标记，不显示
        if Self::is_special_mark(english) || Self::is_special_mark(chinese) {
            return;
        }

        // 直接更新当前文本
        self.current_english = english.to_string();
        self.current_chinese = chinese.to_string();

        // 显示文本
        self.display_text();
    }

    fn display_text(&self) {
        // 显示英文（黄色）
        io::stdout()
            .execute(SetForegroundColor(Color::Yellow))
            .unwrap()
            .execute(Print(&self.current_english))
            .unwrap()
            .execute(ResetColor)
            .unwrap()
            .execute(Print("\n"))
            .unwrap();

        // 显示中文（绿色）
        io::stdout()
            .execute(SetForegroundColor(Color::Green))
            .unwrap()
            .execute(Print(&self.current_chinese))
            .unwrap()
            .execute(ResetColor)
            .unwrap()
            .execute(Print("\n"))
            .unwrap();

        io::stdout().flush().unwrap();
    }
} 