#!/usr/bin/env python3
"""
Document Loader - 支持从终端参数选择读取 txt、pdf、网址并打印内容
使用方法:
    python main.py <file_path_or_url>
    python main.py document.txt
    python main.py document.pdf
    python main.py https://example.com
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.error import URLError

try:
    import fitz  # PyMuPDF for PDF reading
except ImportError:
    print("错误: 需要安装 PyMuPDF 库")
    print("请运行: uv add pymupdf")
    sys.exit(1)


def read_txt_file(file_path):
    """读取文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            with open(file_path, 'r', encoding='gbk') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                return f"无法读取文件 {file_path}: {e}"
    except Exception as e:
        return f"读取文件 {file_path} 时出错: {e}"


def read_pdf_file(file_path):
    """读取PDF文件"""
    try:
        doc = fitz.open(file_path)
        text_content = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content += f"\n--- 第 {page_num + 1} 页 ---\n"
            text_content += page.get_text()
        
        doc.close()
        return text_content
    except Exception as e:
        return f"读取PDF文件 {file_path} 时出错: {e}"


def read_url(url):
    """读取网页内容"""
    try:
        with urlopen(url, timeout=10) as response:
            content_type = response.headers.get('content-type', '')
            
            if 'text/html' in content_type.lower():
                # HTML内容，简单提取文本
                html_content = response.read().decode('utf-8', errors='ignore')
                # 简单的HTML标签去除
                import re
                text_content = re.sub(r'<[^>]+>', '', html_content)
                return text_content
            else:
                # 其他类型内容，直接返回
                return response.read().decode('utf-8', errors='ignore')
                
    except URLError as e:
        return f"无法访问URL {url}: {e}"
    except Exception as e:
        return f"读取URL {url} 时出错: {e}"


def is_url(string):
    """检查字符串是否为URL"""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Document Loader - 读取txt、pdf、网址内容',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    python main.py document.txt
    python main.py document.pdf  
    python main.py https://example.com
        """
    )
    
    parser.add_argument('source', help='文件路径或URL')
    parser.add_argument('--encoding', default='utf-8', help='文本文件编码 (默认: utf-8)')
    
    args = parser.parse_args()
    
    source = args.source
    
    # 判断输入类型
    if is_url(source):
        print(f"正在读取URL: {source}")
        print("=" * 50)
        content = read_url(source)
    else:
        file_path = Path(source)
        if not file_path.exists():
            print(f"错误: 文件不存在: {source}")
            sys.exit(1)
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.txt':
            print(f"正在读取文本文件: {source}")
            print("=" * 50)
            content = read_txt_file(source)
        elif file_extension == '.pdf':
            print(f"正在读取PDF文件: {source}")
            print("=" * 50)
            content = read_pdf_file(source)
        else:
            print(f"错误: 不支持的文件类型: {file_extension}")
            print("支持的文件类型: .txt, .pdf")
            sys.exit(1)
    
    # 打印内容
    print(content)
    print("\n" + "=" * 50)
    print("读取完成")


if __name__ == "__main__":
    main()