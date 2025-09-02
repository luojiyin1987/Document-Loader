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
import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.error import URLError
from typing import List, Dict, Any, Optional
from embeddings import SimpleEmbeddings, HybridSearch, simple_text_search

try:
    import fitz  # PyMuPDF for PDF reading
except ImportError:
    print("错误: 需要安装 PyMuPDF 库")
    print("请运行: uv add pymupdf")
    sys.exit(1)


class TextSplitter:
    """文本分割器基类"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """分割文本，子类需要实现此方法"""
        raise NotImplementedError("子类必须实现 split_text 方法")
    
    def create_documents(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """创建文档对象"""
        chunks = self.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc_metadata = (metadata or {}).copy()
            doc_metadata.update({
                'chunk_index': i,
                'chunk_size': len(chunk),
                'splitter': self.__class__.__name__
            })
            documents.append({
                'page_content': chunk,
                'metadata': doc_metadata
            })
        
        return documents


class CharacterTextSplitter(TextSplitter):
    """按字符数分割文本"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separator: str = ''):
        super().__init__(chunk_size, chunk_overlap)
        self.separator = separator
    
    def split_text(self, text: str) -> List[str]:
        """按字符数分割文本"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # 如果不是最后一个chunk，尝试在分隔符处分割
            if end < len(text) and self.separator:
                last_separator = chunk.rfind(self.separator)
                if last_separator != -1 and last_separator > self.chunk_size // 2:
                    chunk = chunk[:last_separator + len(self.separator)]
                    end = start + last_separator + len(self.separator)
            
            chunks.append(chunk)
            start = end - self.chunk_overlap
            
            # 避免无限循环
            if start >= len(text) - self.chunk_overlap:
                break
        
        return chunks


class RecursiveCharacterTextSplitter(TextSplitter):
    """迭代字符文本分割器 - 按段落、句子、词迭代分割，避免递归内存问题"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """迭代分割文本，避免递归内存问题"""
        return list(self._iterative_split(text))
    
    def _iterative_split(self, text: str) -> List[str]:
        """迭代分割方法"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        remaining_text = text
        
        while remaining_text:
            chunk = self._extract_next_chunk(remaining_text)
            chunks.append(chunk)
            
            # 计算剩余文本，考虑重叠
            chunk_end = len(chunk) - self.chunk_overlap
            if chunk_end <= 0:
                chunk_end = len(chunk)
            
            remaining_text = remaining_text[chunk_end:]
            
            # 防止无限循环
            if not remaining_text or len(remaining_text) < 10:
                break
        
        return chunks
    
    def _extract_next_chunk(self, text: str) -> str:
        """提取下一个chunk"""
        if len(text) <= self.chunk_size:
            return text
        
        # 尝试在各种分隔符处分割
        for separator in self.separators:
            if not separator:
                continue
                
            # 在chunk_size范围内寻找最后一个分隔符
            search_area = text[:self.chunk_size]
            last_separator_pos = -1
            
            # 从后向前查找分隔符
            pos = len(search_area) - 1
            while pos >= 0:
                if search_area.startswith(separator, pos):
                    last_separator_pos = pos
                    break
                pos -= 1
            
            # 如果找到合适的分隔符
            if last_separator_pos != -1 and last_separator_pos > self.chunk_size // 4:
                return text[:last_separator_pos + len(separator)]
        
        # 如果没有合适的分隔符，按字符分割
        return text[:self.chunk_size]


class StreamingTextSplitter(TextSplitter):
    """流式文本分割器 - 适用于超大文件的内存友好分割"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """流式分割文本"""
        return list(self._streaming_split_generator(text))
    
    def _streaming_split_generator(self, text: str):
        """流式分割生成器"""
        if len(text) <= self.chunk_size:
            yield text
            return
        
        position = 0
        buffer = ""
        
        while position < len(text):
            # 读取下一个段落或句子
            next_segment = self._read_next_segment(text, position)
            
            if not next_segment:
                break
            
            # 如果添加下一个segment不会超过chunk_size，就添加到buffer
            if len(buffer) + len(next_segment) <= self.chunk_size:
                buffer += next_segment
                position += len(next_segment)
            else:
                # 如果buffer不为空，先输出buffer
                if buffer:
                    yield buffer
                    # 重置buffer，考虑重叠
                    overlap_start = max(0, len(buffer) - self.chunk_overlap)
                    buffer = buffer[overlap_start:]
                
                # 检查新的segment是否太大
                if len(next_segment) > self.chunk_size:
                    # 大segment需要进一步分割
                    for chunk in self._split_large_segment(next_segment):
                        yield chunk
                else:
                    buffer = next_segment
                    position += len(next_segment)
        
        # 输出最后的buffer
        if buffer:
            yield buffer
    
    def _read_next_segment(self, text: str, position: int) -> str:
        """从指定位置读取下一个段落或句子"""
        if position >= len(text):
            return ""
        
        # 尝试按段落分割
        for separator in ["\n\n", "\n"]:
            if separator in text[position:]:
                next_pos = text.find(separator, position)
                if next_pos != -1:
                    return text[position:next_pos + len(separator)]
        
        # 尝试按句子分割
        for separator in ["。", "！", "？", ".", "!", "?"]:
            if separator in text[position:]:
                next_pos = text.find(separator, position)
                if next_pos != -1:
                    return text[position:next_pos + len(separator)]
        
        # 如果没有找到分隔符，返回剩余部分（不超过chunk_size）
        remaining = text[position:]
        return remaining[:self.chunk_size]
    
    def _split_large_segment(self, segment: str) -> List[str]:
        """分割过大的segment"""
        chunks = []
        start = 0
        
        while start < len(segment):
            end = start + self.chunk_size
            chunk = segment[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
            
            if start >= len(segment) - self.chunk_overlap:
                break
        
        return chunks


class TokenTextSplitter(TextSplitter):
    """按token数分割文本（基于单词）"""
    
    def split_text(self, text: str) -> List[str]:
        """按token数分割文本"""
        # 简单的tokenization（按空格和标点分割）
        tokens = re.findall(r'\w+|[^\w\s]', text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            
            # 重建文本
            chunk_text = ' '.join(chunk_tokens)
            chunks.append(chunk_text)
            
            start = end - self.chunk_overlap
            
            if start >= len(tokens) - self.chunk_overlap:
                break
        
        return chunks


class SemanticTextSplitter(TextSplitter):
    """语义文本分割器 - 基于句子边界和语义连贯性"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        # 中文句子结束符
        self.chinese_sentence_endings = ['。', '！', '？', '；', '…']
        # 英文句子结束符
        self.english_sentence_endings = ['.', '!', '?', ';', '...']
    
    def split_text(self, text: str) -> List[str]:
        """基于语义分割文本"""
        if len(text) <= self.chunk_size:
            return [text]
        
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 如果单个句子太长，按字符分割
                if len(sentence) > self.chunk_size:
                    char_chunks = self._split_long_sentence(sentence)
                    chunks.extend(char_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """分割成句子"""
        sentences = []
        current = ""
        
        i = 0
        while i < len(text):
            char = text[i]
            current += char
            
            # 检查是否是句子结束符
            if (char in self.chinese_sentence_endings or 
                char in self.english_sentence_endings):
                
                # 检查是否有引号
                if i + 1 < len(text) and text[i + 1] in ['"', '"', ''', ''']:
                    current += text[i + 1]
                    i += 1
                
                sentences.append(current)
                current = ""
            
            i += 1
        
        if current:
            sentences.append(current)
        
        return sentences
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """分割长句子"""
        chunks = []
        start = 0
        
        while start < len(sentence):
            end = start + self.chunk_size
            chunk = sentence[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
            
            if start >= len(sentence) - self.chunk_overlap:
                break
        
        return chunks


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
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Document Loader - 读取txt、pdf、网址内容并支持文本分割',
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
        """
    )
    
    parser.add_argument('source', help='文件路径或URL')
    parser.add_argument('--encoding', default='utf-8', help='文本文件编码 (默认: utf-8)')
    parser.add_argument('--split', action='store_true', help='启用文本分割')
    parser.add_argument('--chunk-size', type=int, default=1000, help='分割块大小 (默认: 1000)')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='分割块重叠大小 (默认: 200)')
    parser.add_argument('--splitter', choices=['character', 'recursive', 'streaming', 'token', 'semantic'], 
                       default='recursive', help='分割器类型 (默认: recursive)')
    parser.add_argument('--search-mode', choices=['keyword', 'semantic', 'hybrid'], 
                       help='搜索模式: keyword(关键词), semantic(语义), hybrid(混合)')
    parser.add_argument('--search-query', help='搜索查询内容')
    parser.add_argument('--top-k', type=int, default=5, help='搜索结果数量 (默认: 5)')
    
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
            if args.splitter == 'character':
                splitter = CharacterTextSplitter(args.chunk_size, args.chunk_overlap)
            elif args.splitter == 'recursive':
                splitter = RecursiveCharacterTextSplitter(args.chunk_size, args.chunk_overlap)
            elif args.splitter == 'streaming':
                splitter = StreamingTextSplitter(args.chunk_size, args.chunk_overlap)
            elif args.splitter == 'token':
                splitter = TokenTextSplitter(args.chunk_size, args.chunk_overlap)
            elif args.splitter == 'semantic':
                splitter = SemanticTextSplitter(args.chunk_size, args.chunk_overlap)
            
            # 分割文档
            documents = splitter.create_documents(content, {
                'source': source,
                'total_length': len(content),
                'chunk_size': args.chunk_size,
                'chunk_overlap': args.chunk_overlap
            })
            
            search_documents = [doc['page_content'] for doc in documents]
            print(f"文本已分割为 {len(search_documents)} 个块进行搜索")
        else:
            # 直接在整个文档中搜索
            search_documents = [content]
            print("在整个文档中进行搜索")
        
        # 执行搜索
        try:
            if args.search_mode == 'keyword':
                results = simple_text_search(args.search_query, search_documents, args.top_k)
            elif args.search_mode == 'semantic':
                embedder = SimpleEmbeddings()
                results = embedder.similarity_search(args.search_query, search_documents, args.top_k)
            elif args.search_mode == 'hybrid':
                hybrid_search = HybridSearch()
                results = hybrid_search.search(args.search_query, search_documents, args.top_k)
            
            # 显示搜索结果
            print(f"\n找到 {len(results)} 个相关结果:")
            print("-" * 50)
            
            for i, result in enumerate(results, 1):
                if args.search_mode == 'keyword':
                    print(f"结果 {i}: 分数={result['score']:.3f}")
                    print(f"内容: {result['document'][:300]}{'...' if len(result['document']) > 300 else ''}")
                elif args.search_mode == 'semantic':
                    print(f"结果 {i}: 相似度={result['similarity']:.3f}")
                    print(f"内容: {result['document'][:300]}{'...' if len(result['document']) > 300 else ''}")
                elif args.search_mode == 'hybrid':
                    print(f"结果 {i}: 综合分数={result['combined_score']:.3f} "
                          f"(关键词={result['keyword_score']:.3f}, 语义={result['semantic_score']:.3f})")
                    print(f"内容: {result['document'][:300]}{'...' if len(result['document']) > 300 else ''}")
                print()
        
        except Exception as e:
            print(f"搜索时出错: {e}")
            # 降级到关键词搜索
            print("降级到关键词搜索...")
            results = simple_text_search(args.search_query, search_documents, args.top_k)
            for i, result in enumerate(results, 1):
                print(f"结果 {i}: 分数={result['score']:.3f}")
                print(f"内容: {result['document'][:300]}{'...' if len(result['document']) > 300 else ''}")
                print()
    
    # 如果启用文本分割但不搜索
    elif args.split:
        print(f"正在使用 {args.splitter} 分割器分割文本...")
        print(f"分割参数: chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}")
        print("=" * 50)
        
        # 创建分割器
        if args.splitter == 'character':
            splitter = CharacterTextSplitter(args.chunk_size, args.chunk_overlap)
        elif args.splitter == 'recursive':
            splitter = RecursiveCharacterTextSplitter(args.chunk_size, args.chunk_overlap)
        elif args.splitter == 'streaming':
            splitter = StreamingTextSplitter(args.chunk_size, args.chunk_overlap)
        elif args.splitter == 'token':
            splitter = TokenTextSplitter(args.chunk_size, args.chunk_overlap)
        elif args.splitter == 'semantic':
            splitter = SemanticTextSplitter(args.chunk_size, args.chunk_overlap)
        
        # 创建文档对象
        metadata = {
            'source': source,
            'total_length': len(content),
            'chunk_size': args.chunk_size,
            'chunk_overlap': args.chunk_overlap
        }
        
        documents = splitter.create_documents(content, metadata)
        
        # 打印分割结果
        print(f"总共分割为 {len(documents)} 个块:")
        print("-" * 50)
        
        for i, doc in enumerate(documents):
            print(f"块 {i + 1} (长度: {len(doc['page_content'])}):")
            print(f"元数据: {doc['metadata']}")
            print(f"内容: {doc['page_content'][:200]}{'...' if len(doc['page_content']) > 200 else ''}")
            print("-" * 50)
    else:
        # 直接打印内容
        print(content)
    
    print("\n" + "=" * 50)
    print("处理完成")


if __name__ == "__main__":
    main()