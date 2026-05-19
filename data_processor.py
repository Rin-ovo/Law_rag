import os
import re
import fitz  # PyMuPDF
from docx import Document
from pathlib import Path


class UniversalParser:
    def __init__(self, output_dir="./markdown"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 法律条文识别正则 - 修正了字符集
        self.article_pattern = re.compile(r'^第[一二三四五六七八九十百零0-9]+[条章节]')

    def _clean_text(self, text, law_name):
        """通用清理逻辑：处理噪声、断行并注入元数据"""
        # 修正：移除转义字符
        text = re.sub(r"\\'", '', text)  # 修正这里
        text = re.sub(r'---\s*PAGE\s*\d+\s*---', '', text)

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_chunks = []
        current_chunk = ""

        for i, line in enumerate(lines):
            if self.article_pattern.match(line):
                if current_chunk:
                    cleaned_chunks.append(current_chunk)
                current_chunk = f"### [{law_name}] {line}"
            else:
                if current_chunk:
                    # 改进拼接逻辑，避免过度紧凑
                    current_chunk += " " + line if not line.startswith(('。', '，', '；', '、')) else line
                else:
                    # 处理开头的非条文内容
                    if i == 0:
                        current_chunk = f"### [{law_name}] 导言\n{line}"
                    else:
                        current_chunk = line

        if current_chunk:
            cleaned_chunks.append(current_chunk)
        return "\n\n".join(cleaned_chunks)

    def parse_pdf(self, path):
        """解析PDF文件"""
        try:
            doc = fitz.open(path)
            text_parts = []
            for page in doc:
                text = page.get_text()
                if text.strip():  # 跳过空页
                    text_parts.append(text)
            return "\n".join(text_parts)
        except Exception as e:
            print(f"解析PDF失败 {path}: {e}")
            return ""

    def parse_docx(self, path):
        """解析DOCX文件"""
        try:
            doc = Document(path)
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)
            return "\n".join(paragraphs)
        except Exception as e:
            print(f"解析DOCX失败 {path}: {e}")
            return ""

    def parse_txt(self, path):
        """解析TXT文件"""
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            print(f"无法解码文件: {path}")
            return ""
        except Exception as e:
            print(f"读取文件失败 {path}: {e}")
            return ""

    def process_file(self, file_path):
        """处理单个文件"""
        path = Path(file_path)
        if not path.exists():
            print(f"文件不存在: {file_path}")
            return

        ext = path.suffix.lower()
        law_name = path.stem  # 以文件名作为法律标签

        print(f"正在解析 {ext} 文件: {path.name}...")

        raw_text = ""
        if ext == '.pdf':
            raw_text = self.parse_pdf(path)
        elif ext == '.docx':
            raw_text = self.parse_docx(path)
        elif ext == '.txt':
            raw_text = self.parse_txt(path)
        else:
            print(f"跳过不支持的格式: {ext}")
            return

        if not raw_text.strip():
            print(f"⚠️ 警告: {path.name} 内容为空，跳过处理")
            return

        formatted_md = self._clean_text(raw_text, law_name)

        output_path = self.output_dir / f"{law_name}.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_md)
        print(f"✅ 转换成功: {output_path}")


# 使用示例
if __name__ == "__main__":
    parser = UniversalParser()

    # 创建示例目录结构
    data_dir = Path("./docs")
    data_dir.mkdir(exist_ok=True)

    # 提示用户放置文件
    print("=" * 50)
    print("法律文档转换工具")
    print("=" * 50)
    print(f"请将PDF/DOCX/TXT文件放置在: {data_dir.absolute()}")
    print("支持的格式: .pdf, .docx, .txt")
    print("=" * 50)

    # 检查文件
    files_found = False
    for file in os.listdir(data_dir):
        if file.lower().endswith(('.pdf', '.docx', '.txt')):
            files_found = True
            parser.process_file(data_dir / file)
    if not files_found:
        print("未找到支持的文档文件。")
        print("请创建测试文件:")
        print("1. 创建 docs/ 目录")
        print("2. 放置PDF/DOCX/TXT文件到该目录")
        print("3. 重新运行本程序")