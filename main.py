#!/usr/bin/env python3
"""
Document Loader - 支持从终端参数选择读取 txt、pdf、网址并打印内容
使用方法:
    python main.py <file_path_or_url>
    python main.py document.txt
    python main.py document.pdf
    python main.py https://example.com
"""

# ===== 标准库导入 =====
import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

# ===== 第三方库导入 =====
try:
    import fitz  # type: ignore # PyMuPDF for PDF reading
except ImportError:
    print("错误: 需要安装 PyMuPDF 库")
    print("请运行: uv add pymupdf")
    sys.exit(1)

# ===== 项目自定义模块导入 =====
# 向量嵌入和搜索功能
from embeddings import HybridSearch, SimpleEmbeddings, simple_text_search

# 文本分割功能
from text_splitter import create_text_splitter

# 向量存储功能
# from vector_store_integration import (
#     add_chunks_to_vector_store,
#     create_vector_store_integration,
#     search_vector_store,
# )


def read_txt_file(file_path):
    """读取文本文件"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            with open(file_path, "r", encoding="gbk") as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as file:
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
        # 验证URL scheme
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ["http", "https"]:
            raise ValueError(f"不支持的URL scheme: {parsed_url.scheme}")

        with urlopen(url, timeout=10) as response:  # nosec B310 - URL scheme validated above
            content_type = response.headers.get("content-type", "")

            if "text/html" in content_type.lower():
                # HTML内容，简单提取文本
                html_content = response.read().decode("utf-8", errors="ignore")
                # 简单的HTML标签去除
                import re

                text_content = re.sub(r"<[^>]+>", "", html_content)
                return text_content
            else:
                # 其他类型内容，直接返回
                return response.read().decode("utf-8", errors="ignore")

    except URLError as e:
        return f"无法访问URL {url}: {e}"
    except Exception as e:
        return f"读取URL {url} 时出错: {e}"


def is_url(string):
    """检查字符串是否为URL"""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Document Loader - 读取txt、pdf、网址内容并支持文本分割",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 基本文档读取
    python main.py document.txt
    python main.py document.pdf
    python main.py https://example.com

    # 文本分割
    python main.py document.txt --split --chunk-size 500 --splitter recursive

    # 搜索功能
    python main.py document.txt --search-mode keyword --search-query "Python 编程"
    python main.py document.txt --search-mode semantic --search-query "人工智能"
    python main.py document.txt --search-mode hybrid --search-query "数据 算法"

    # 分割+搜索组合
    python main.py large_file.txt --split --search-mode semantic --search-query "机器学习"
        """,
    )

    parser.add_argument("source", help="文件路径或URL")
    parser.add_argument("--encoding", default="utf-8", help="文本文件编码 (默认: utf-8)")
    parser.add_argument("--split", action="store_true", help="启用文本分割")
    parser.add_argument("--chunk-size", type=int, default=1000, help="分割块大小 (默认: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="分割块重叠大小 (默认: 200)")
    parser.add_argument(
        "--splitter",
        choices=["character", "recursive", "streaming", "token", "semantic"],
        default="recursive",
        help="分割器类型 (默认: recursive)",
    )
    parser.add_argument(
        "--search-mode",
        choices=["keyword", "semantic", "hybrid"],
        help="搜索模式: keyword(关键词), semantic(语义), hybrid(混合)",
    )
    parser.add_argument("--search-query", help="搜索查询内容")
    parser.add_argument("--top-k", type=int, default=5, help="搜索结果数量 (默认: 5)")

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

        if file_extension == ".txt":
            print(f"正在读取文本文件: {source}")
            print("=" * 50)
            content = read_txt_file(source)
        elif file_extension == ".pdf":
            print(f"正在读取PDF文件: {source}")
            print("=" * 50)
            content = read_pdf_file(source)
        else:
            print(f"错误: 不支持的文件类型: {file_extension}")
            print("支持的文件类型: .txt, .pdf")
            sys.exit(1)

    # 如果启用搜索功能
    if args.search_mode and args.search_query:
        print(f"正在使用 {args.search_mode} 搜索模式...")
        print(f"搜索查询: {args.search_query}")
        print(f"返回结果数量: {args.top_k}")
        print("=" * 50)

        # 准备搜索文档（如果同时启用了分割）
        search_documents = []

        if args.split:
            # 先分割文本，然后在分割后的块中搜索
            print(f"正在使用 {args.splitter} 分割器预处理文本...")

            # 创建分割器
            splitter = create_text_splitter(args.splitter, args.chunk_size, args.chunk_overlap)

            # 分割文档
            documents = splitter.create_documents(
                content,
                {
                    "source": source,
                    "total_length": len(content),
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": args.chunk_overlap,
                },
            )

            search_documents = [doc["page_content"] for doc in documents]
            print(f"文本已分割为 {len(search_documents)} 个块进行搜索")
        else:
            # 直接在整个文档中搜索
            search_documents = [content]
            print("在整个文档中进行搜索")

        # 执行搜索
        try:
            if args.search_mode == "keyword":
                results = simple_text_search(args.search_query, search_documents, args.top_k)
            elif args.search_mode == "semantic":
                embedder = SimpleEmbeddings()
                results = embedder.similarity_search(
                    args.search_query, search_documents, args.top_k
                )
            elif args.search_mode == "hybrid":
                hybrid_search = HybridSearch()
                results = hybrid_search.search(args.search_query, search_documents, args.top_k)

            # 显示搜索结果
            print(f"\n找到 {len(results)} 个相关结果:")
            print("-" * 50)

            for i, result in enumerate(results, 1):
                if args.search_mode == "keyword":
                    print(f"结果 {i}: 分数={result['score']:.3f}")
                    content_preview = result["document"][:300]
                    if len(result["document"]) > 300:
                        content_preview += "..."
                    print(f"内容: {content_preview}")
                elif args.search_mode == "semantic":
                    print(f"结果 {i}: 相似度={result['similarity']:.3f}")
                    content_preview = result["document"][:300]
                    if len(result["document"]) > 300:
                        content_preview += "..."
                    print(f"内容: {content_preview}")
                elif args.search_mode == "hybrid":
                    print(
                        f"结果 {i}: 综合分数={result['combined_score']:.3f} "
                        f"(关键词={result['keyword_score']:.3f}, 语义={result['semantic_score']:.3f})"
                    )
                    content_preview = result["document"][:300]
                    if len(result["document"]) > 300:
                        content_preview += "..."
                    print(f"内容: {content_preview}")
                print()

        except Exception as e:
            print(f"搜索时出错: {e}")
            # 降级到关键词搜索
            print("降级到关键词搜索...")
            results = simple_text_search(args.search_query, search_documents, args.top_k)
            for i, result in enumerate(results, 1):
                print(f"结果 {i}: 分数={result['score']:.3f}")
                content_preview = result["document"][:300]
                if len(result["document"]) > 300:
                    content_preview += "..."
                print(f"内容: {content_preview}")
                print()

    # 如果启用文本分割但不搜索
    elif args.split:
        print(f"正在使用 {args.splitter} 分割器分割文本...")
        print(f"分割参数: chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}")
        print("=" * 50)

        # 创建分割器
        splitter = create_text_splitter(args.splitter, args.chunk_size, args.chunk_overlap)

        # 创建文档对象
        metadata = {
            "source": source,
            "total_length": len(content),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        }

        documents = splitter.create_documents(content, metadata)

        # 打印分割结果
        print(f"总共分割为 {len(documents)} 个块:")
        print("-" * 50)

        for i, doc in enumerate(documents):
            print(f"块 {i + 1} (长度: {len(doc['page_content'])}):")
            print(f"元数据: {doc['metadata']}")
            content_preview = doc["page_content"][:200]
            if len(doc["page_content"]) > 200:
                content_preview += "..."
            print(f"内容: {content_preview}")
            print("-" * 50)
    else:
        # 直接打印内容
        print(content)

    print("\n" + "=" * 50)
    print("处理完成")


if __name__ == "__main__":
    main()
