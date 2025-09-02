#!/usr/bin/env python3
"""
Text Splitter Module - 提供多种文本分割策略
支持字符、递归、流式、token和语义分割
"""

# ===== 标准库导入 =====
import re
from typing import Any, Dict, List, Optional


class TextSplitter:
    """文本分割器基类"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """分割文本，子类需要实现此方法"""
        raise NotImplementedError("子类必须实现 split_text 方法")

    def create_documents(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """创建文档对象"""
        chunks = self.split_text(text)
        documents = []

        for i, chunk in enumerate(chunks):
            doc_metadata = (metadata or {}).copy()
            doc_metadata.update(
                {
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "splitter": self.__class__.__name__,
                }
            )
            documents.append({"page_content": chunk, "metadata": doc_metadata})

        return documents


class CharacterTextSplitter(TextSplitter):
    """按字符数分割文本"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separator: str = ""):
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
                    chunk = chunk[: last_separator + len(self.separator)]
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
            search_area = text[: self.chunk_size]
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
                return text[: last_separator_pos + len(separator)]

        # 如果没有合适的分隔符，按字符分割
        return text[: self.chunk_size]


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
                    return text[position : next_pos + len(separator)]

        # 尝试按句子分割
        for separator in ["。", "！", "？", ".", "!", "?"]:
            if separator in text[position:]:
                next_pos = text.find(separator, position)
                if next_pos != -1:
                    return text[position : next_pos + len(separator)]

        # 如果没有找到分隔符，返回剩余部分（不超过chunk_size）
        remaining = text[position:]
        return remaining[: self.chunk_size]

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
        tokens = re.findall(r"\w+|[^\w\s]", text)

        if len(tokens) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            # 重建文本
            chunk_text = " ".join(chunk_tokens)
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
        self.chinese_sentence_endings = ["。", "！", "？", "；", "…"]
        # 英文句子结束符
        self.english_sentence_endings = [".", "!", "?", ";", "..."]

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
            if char in self.chinese_sentence_endings or char in self.english_sentence_endings:

                # 检查是否有引号
                if i + 1 < len(text) and text[i + 1] in ['"', '"', """, """]:
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


# 工厂函数，方便创建不同类型的分割器
def create_text_splitter(
    splitter_type: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs,
) -> TextSplitter:
    """
    创建文本分割器的工厂函数

    Args:
        splitter_type: 分割器类型 ('character', 'recursive', 'streaming', 'token', 'semantic')
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
        **kwargs: 其他分割器特定参数

    Returns:
        TextSplitter: 对应类型的文本分割器实例
    """
    splitter_map = {
        "character": CharacterTextSplitter,
        "recursive": RecursiveCharacterTextSplitter,
        "streaming": StreamingTextSplitter,
        "token": TokenTextSplitter,
        "semantic": SemanticTextSplitter,
    }

    if splitter_type not in splitter_map:
        raise ValueError(
            f"不支持的分割器类型: {splitter_type}. 支持的类型: {list(splitter_map.keys())}"
        )

    return splitter_map[splitter_type](chunk_size, chunk_overlap, **kwargs)
