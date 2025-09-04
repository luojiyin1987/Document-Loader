#!/usr/bin/env python3
"""
RetrievalQA 集成脚本 - 将文档加载、处理和问答功能整合
支持从文件或URL加载文档并构建问答系统
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from advanced_retrieval_qa import AdvancedRetrievalQA, create_advanced_retrieval_qa

# 导入项目模块
from main import is_url, read_pdf_file, read_txt_file, read_url
from retrieval_qa import RetrievalQA, create_retrieval_qa
from text_splitter import create_text_splitter


class DocumentProcessor:
    """文档处理器"""

    def __init__(self):
        self.supported_formats = {".txt", ".pdf"}

    def load_document(self, source: str) -> str:
        """加载文档内容"""
        if is_url(source):
            print(f"正在从URL加载文档: {source}")
            return read_url(source)
        else:
            file_path = Path(source)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {source}")

            file_extension = file_path.suffix.lower()
            if file_extension == ".txt":
                print(f"正在加载文本文件: {source}")
                return read_txt_file(source)
            elif file_extension == ".pdf":
                print(f"正在加载PDF文件: {source}")
                return read_pdf_file(source)
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}")

    def process_documents(
        self,
        sources: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitter_type: str = "recursive",
    ) -> List[Dict[str, Any]]:
        """批量处理文档"""
        all_documents = []

        for source in sources:
            try:
                # 加载文档内容
                content = self.load_document(source)

                if not content.strip():
                    print(f"警告: 文档 {source} 内容为空，跳过处理")
                    continue

                # 创建文本分割器
                splitter = create_text_splitter(
                    splitter_type, chunk_size, chunk_overlap
                )

                # 分割文档
                metadata = {
                    "source": source,
                    "total_length": len(content),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "processed_at": datetime.now().isoformat(),
                }

                documents = splitter.create_documents(content, metadata)

                # 添加到总文档列表
                all_documents.extend(documents)

                print(f"文档 {source} 已分割为 {len(documents)} 个块")

            except Exception as e:
                print(f"处理文档 {source} 时出错: {e}")
                continue

        print(f"总共处理了 {len(all_documents)} 个文档块")
        return all_documents


class QASystemManager:
    """问答系统管理器"""

    def __init__(self, system_type: str = "basic"):
        self.system_type = system_type
        self.qa_system: Optional[Union[RetrievalQA, AdvancedRetrievalQA]] = None
        self.document_processor = DocumentProcessor()

    def create_system(
        self,
        retriever_type: str = "hybrid",
        context_tokens: int = 2000,
        answer_generator: str = "template",
    ) -> None:
        """创建问答系统"""
        if self.system_type == "basic":
            self.qa_system = create_retrieval_qa(
                retriever_type=retriever_type,
                context_max_tokens=context_tokens,
                answer_generator_type=answer_generator,
            )
        elif self.system_type == "advanced":
            self.qa_system = create_advanced_retrieval_qa()
        else:
            raise ValueError(f"不支持的系统类型: {self.system_type}")

        print(f"已创建 {self.system_type} 类型的问答系统")

    def load_knowledge_base(
        self,
        sources: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitter_type: str = "recursive",
    ) -> None:
        """加载知识库"""
        if not self.qa_system:
            raise RuntimeError("请先创建问答系统")

        print("开始加载知识库...")

        # 处理文档
        documents = self.document_processor.process_documents(
            sources, chunk_size, chunk_overlap, splitter_type
        )

        if not documents:
            raise ValueError("没有成功处理任何文档")

        # 添加文档到问答系统
        self.qa_system.add_documents(documents)

        print(f"知识库加载完成，共 {len(documents)} 个文档块")

    def query(
        self, question: str, top_k: int = 5, context_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """执行查询"""
        if not self.qa_system:
            raise RuntimeError("请先创建并初始化问答系统")

        print(f"正在查询: {question}")

        try:
            if self.system_type == "advanced" and context_strategy:
                result = self.qa_system.query(
                    question, top_k, context_strategy=context_strategy
                )
            else:
                # 基础系统不支持context_strategy参数
                result = self.qa_system.query(question, top_k)

            # 格式化结果
            formatted_result = {
                "query": question,
                "answer": result.answer,
                "confidence": result.confidence,
                "sources": list(set(result.sources)),
                "context_docs_count": len(result.context.relevant_docs),
                "processing_time": result.metadata.get("total_time", 0),
                "retrieval_time": result.metadata.get("retrieval_time", 0),
                "context_score": result.context.context_score,
                "total_tokens": result.context.total_tokens,
            }

            return formatted_result

        except Exception as e:
            print(f"查询出错: {e}")
            return {
                "query": question,
                "answer": f"查询处理失败: {e}",
                "confidence": 0.0,
                "sources": [],
                "error": str(e),
            }

    def batch_query(self, questions: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """批量查询"""
        results = []
        for question in questions:
            result = self.query(question, top_k)
            results.append(result)
        return results

    def interactive_mode(self):
        """交互式问答模式"""
        print("\n=== 交互式问答模式 ===")
        print("输入您的问题，输入 'quit' 或 'exit' 退出")
        print("输入 'stats' 查看系统统计信息")
        print("输入 'help' 查看帮助信息")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n请输入您的问题: ").strip()

                if user_input.lower() in ["quit", "exit", "退出"]:
                    print("感谢使用！")
                    break

                elif user_input.lower() in ["stats", "统计"]:
                    self.show_stats()
                    continue

                elif user_input.lower() in ["help", "帮助"]:
                    self.show_help()
                    continue

                elif not user_input:
                    print("请输入有效的问题")
                    continue

                # 执行查询
                result = self.query(user_input)

                # 显示结果
                print(f"\n问题: {result['query']}")
                print(f"答案: {result['answer']}")
                print(f"置信度: {result['confidence']:.3f}")
                if result["sources"]:
                    print(f"来源: {', '.join(result['sources'])}")
                print(f"处理时间: {result['processing_time']:.3f}秒")

            except KeyboardInterrupt:
                print("\n\n感谢使用！")
                break
            except Exception as e:
                print(f"处理查询时出错: {e}")

    def show_stats(self):
        """显示系统统计信息"""
        if not self.qa_system:
            print("系统未初始化")
            return

        try:
            if hasattr(self.qa_system, "get_system_stats"):
                stats = self.qa_system.get_system_stats()
            else:
                stats = self.qa_system.get_stats()

            print("\n=== 系统统计信息 ===")
            if "performance" in stats:
                perf = stats["performance"]
                print(f"总查询数: {perf['total_queries']}")
                print(f"平均响应时间: {perf['avg_time']:.3f}秒")
                print(f"缓存命中率: {perf['cache_hit_rate']:.2%}")

            if "retriever" in stats:
                retriever_stats = stats["retriever"]
                print(f"检索次数: {retriever_stats.get('retrieval_count', 0)}")
                print(
                    f"平均检索时间: {retriever_stats.get('avg_retrieval_time', 0):.3f}秒"
                )

            if "total_queries" in stats:
                print(f"总查询数: {stats['total_queries']}")
                print(f"平均响应时间: {stats['avg_answer_time']:.3f}秒")

        except Exception as e:
            print(f"获取统计信息时出错: {e}")

    def show_help(self):
        """显示帮助信息"""
        print("\n=== 帮助信息 ===")
        print("可用命令:")
        print("  quit/exit/退出    - 退出交互模式")
        print("  stats/统计       - 查看系统统计信息")
        print("  help/帮助        - 显示此帮助信息")
        print("\n查询提示:")
        print("  - 直接输入您的问题")
        print("  - 支持中文和英文查询")
        print("  - 可以询问事实性、解释性或总结性问题")
        print("  - 系统会基于加载的文档内容回答")


def main():
    parser = argparse.ArgumentParser(
        description="RetrievalQA 集成系统 - 文档加载与问答",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 基本使用 - 加载文档并进行问答
    python retrieval_qa_integration.py document.txt --interactive

    # 加载多个文档
    python retrieval_qa_integration.py doc1.txt doc2.pdf --interactive

    # 使用高级系统
    python retrieval_qa_integration.py document.txt --system advanced --interactive

    # 指定查询
    python retrieval_qa_integration.py document.txt --query "Python是什么？"

    # 批量查询
    python retrieval_qa_integration.py document.txt --batch-query "问题1" "问题2" "问题3"

    # 自定义参数
    python retrieval_qa_integration.py document.txt --chunk-size 1500 --retriever hybrid
    --interactive
        """,
    )

    # 必需参数
    parser.add_argument("sources", nargs="+", help="文档文件路径或URL")

    # 系统类型
    parser.add_argument(
        "--system",
        choices=["basic", "advanced"],
        default="basic",
        help="问答系统类型 (默认: basic)",
    )

    # 检索器配置
    parser.add_argument(
        "--retriever",
        choices=["vector", "hybrid"],
        default="hybrid",
        help="检索器类型 (默认: hybrid)",
    )

    # 文档处理配置
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="文本分割块大小 (默认: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="文本分割重叠大小 (默认: 200)"
    )
    parser.add_argument(
        "--splitter",
        choices=["character", "recursive", "streaming", "token", "semantic"],
        default="recursive",
        help="文本分割器类型 (默认: recursive)",
    )

    # 上下文配置
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=2000,
        help="上下文最大token数 (默认: 2000)",
    )
    parser.add_argument(
        "--context-strategy",
        choices=["relevant_first", "diverse", "comprehensive"],
        help="上下文构建策略 (仅高级系统)",
    )

    # 查询模式
    parser.add_argument("--query", help="单个查询问题")
    parser.add_argument("--batch-query", nargs="+", help="批量查询问题")
    parser.add_argument("--interactive", action="store_true", help="交互式问答模式")

    # 输出配置
    parser.add_argument("--top-k", type=int, default=5, help="检索文档数量 (默认: 5)")
    parser.add_argument("--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    try:
        # 创建问答系统管理器
        manager = QASystemManager(args.system)

        # 创建问答系统
        manager.create_system(
            retriever_type=args.retriever, context_tokens=args.context_tokens
        )

        # 加载知识库
        manager.load_knowledge_base(
            args.sources,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            splitter_type=args.splitter,
        )

        # 执行查询
        if args.query:
            result = manager.query(args.query, args.top_k, args.context_strategy)
            print("\n=== 查询结果 ===")
            print(f"问题: {result['query']}")
            print(f"答案: {result['answer']}")
            print(f"置信度: {result['confidence']:.3f}")
            if result["sources"]:
                print(f"来源: {', '.join(result['sources'])}")
            print(f"处理时间: {result['processing_time']:.3f}秒")

            if args.verbose:
                print(f"上下文文档数: {result['context_docs_count']}")
                print(f"检索时间: {result['retrieval_time']:.3f}秒")
                print(f"上下文分数: {result['context_score']:.3f}")
                print(f"总Token数: {result['total_tokens']}")

        elif args.batch_query:
            print("\n=== 批量查询结果 ===")
            results = manager.batch_query(args.batch_query, args.top_k)

            for i, result in enumerate(results, 1):
                print(f"\n查询 {i}: {result['query']}")
                print(f"答案: {result['answer']}")
                print(f"置信度: {result['confidence']:.3f}")
                if result["sources"]:
                    print(f"来源: {', '.join(result['sources'])}")
                print("-" * 50)

        elif args.interactive:
            manager.interactive_mode()

        else:
            print("请指定查询模式: --query, --batch-query, 或 --interactive")
            parser.print_help()

    except Exception as e:
        print(f"程序执行出错: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
