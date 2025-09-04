#!/usr/bin/env python3
"""
示例代理实现
基于项目现有功能创建的智能代理
"""

from typing import Any, List

from agents import AgentTask, BaseAgent, ToolInterface
from embeddings import HybridSearch, SimpleEmbeddings, simple_text_search
from search_engine import create_search_engine_manager
from text_splitter import create_text_splitter


class DocumentProcessingTool(ToolInterface):
    """文档处理工具"""

    @property
    def name(self) -> str:
        return "document_processor"

    @property
    def description(self) -> str:
        return "处理文档：读取、分割和分析文本内容"

    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        if isinstance(input_data, dict):
            return "source" in input_data and "action" in input_data
        return False

    def execute(self, input_data: Any) -> Any:
        """执行文档处理"""
        if not self.validate_input(input_data):
            raise ValueError("无效的输入数据")

        source = input_data["source"]
        action = input_data["action"]

        # 导入文档处理函数
        from main import is_url, read_pdf_file, read_txt_file, read_url

        try:
            if is_url(source):
                content = read_url(source)
                doc_type = "url"
            else:
                from pathlib import Path

                file_path = Path(source)
                if not file_path.exists():
                    return {"error": f"文件不存在: {source}"}

                if file_path.suffix.lower() == ".txt":
                    content = read_txt_file(source)
                    doc_type = "txt"
                elif file_path.suffix.lower() == ".pdf":
                    content = read_pdf_file(source)
                    doc_type = "pdf"
                elif file_path.suffix.lower() == ".md":
                    content = read_txt_file(source)  # Markdown文件也是文本文件
                    doc_type = "markdown"
                else:
                    return {"error": f"不支持的文件类型: {source}"}

            if action == "read":
                return {
                    "content": content,
                    "doc_type": doc_type,
                    "source": source,
                    "length": len(content),
                }

            elif action == "split":
                splitter_type = input_data.get("splitter", "recursive")
                chunk_size = input_data.get("chunk_size", 1000)
                chunk_overlap = input_data.get("chunk_overlap", 200)

                splitter = create_text_splitter(
                    splitter_type, chunk_size, chunk_overlap
                )
                documents = splitter.create_documents(content, {"source": source})

                return {
                    "documents": documents,
                    "doc_type": doc_type,
                    "source": source,
                    "total_chunks": len(documents),
                }

            else:
                return {"error": f"不支持的操作: {action}"}

        except Exception as e:
            return {"error": f"处理文档时出错: {str(e)}"}


class TextSearchTool(ToolInterface):
    """文本搜索工具"""

    @property
    def name(self) -> str:
        return "text_search"

    @property
    def description(self) -> str:
        return "在文档中搜索文本内容，支持关键词、语义和混合搜索"

    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        if isinstance(input_data, dict):
            return "query" in input_data and "documents" in input_data
        return False

    def execute(self, input_data: Any) -> Any:
        """执行文本搜索"""
        if not self.validate_input(input_data):
            raise ValueError("无效的输入数据")

        query = input_data["query"]
        documents = input_data["documents"]
        search_mode = input_data.get("search_mode", "keyword")
        top_k = input_data.get("top_k", 5)

        try:
            if isinstance(documents, list):
                # 如果是文档列表
                search_documents = [
                    doc["page_content"] if isinstance(doc, dict) else doc
                    for doc in documents
                ]
            else:
                # 如果是单个文本
                search_documents = [documents]

            if search_mode == "keyword":
                results = simple_text_search(query, search_documents, top_k)
            elif search_mode == "semantic":
                embedder = SimpleEmbeddings()
                results = embedder.similarity_search(query, search_documents, top_k)
            elif search_mode == "hybrid":
                hybrid_search = HybridSearch()
                results = hybrid_search.search(query, search_documents, top_k)
            else:
                return {"error": f"不支持的搜索模式: {search_mode}"}

            return {
                "query": query,
                "search_mode": search_mode,
                "results": results,
                "total_results": len(results),
            }

        except Exception as e:
            return {"error": f"搜索时出错: {str(e)}"}


class WebSearchTool(ToolInterface):
    """网络搜索工具"""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "使用搜索引擎进行网络搜索"

    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        if isinstance(input_data, dict):
            return "query" in input_data
        return False

    def execute(self, input_data: Any) -> Any:
        """执行网络搜索"""
        if not self.validate_input(input_data):
            raise ValueError("无效的输入数据")

        query = input_data["query"]
        engine = input_data.get("engine", "web")
        num_results = input_data.get("num_results", 10)

        try:
            search_manager = create_search_engine_manager()
            results = search_manager.search(query, engine, num_results)

            return {
                "query": query,
                "engine": engine,
                "results": results,
                "total_results": len(results),
            }

        except Exception as e:
            return {"error": f"网络搜索时出错: {str(e)}"}


class DocumentProcessingAgent(BaseAgent):
    """文档处理代理"""

    def __init__(self):
        super().__init__(
            name="DocumentProcessor",
            description="专门处理文档的智能代理，支持读取、分割和分析各种格式的文档",
            tools=[DocumentProcessingTool()],
        )

    @property
    def supported_tasks(self) -> List[str]:
        return ["document_processing", "text_analysis", "content_extraction"]

    def can_handle_task(self, task: AgentTask) -> bool:
        """判断是否能处理任务"""
        task_type = task.metadata.get("task_type", "").lower()
        # 支持的任务类型映射
        if task_type == "document":
            return True
        return any(supported in task_type for supported in self.supported_tasks)

    async def _execute_task_impl(self, task: AgentTask) -> Any:
        """执行文档处理任务"""
        action = task.metadata.get("action", "read")

        if not task.input_data:
            raise ValueError("任务缺少输入数据")

        # 使用文档处理工具
        tool = self.tools[0]  # DocumentProcessingTool
        input_data = {"source": task.input_data, "action": action, **task.metadata}

        result = tool.execute(input_data)

        if "error" in result:
            raise Exception(result["error"])

        return result


class TextSearchAgent(BaseAgent):
    """文本搜索代理"""

    def __init__(self):
        super().__init__(
            name="TextSearcher",
            description="专门进行文本搜索的智能代理，支持多种搜索模式",
            tools=[TextSearchTool()],
        )

    @property
    def supported_tasks(self) -> List[str]:
        return ["text_search", "content_search", "semantic_search"]

    def can_handle_task(self, task: AgentTask) -> bool:
        """判断是否能处理任务"""
        task_type = task.metadata.get("task_type", "").lower()
        # 支持的任务类型映射
        if task_type == "search":
            return True
        return any(supported in task_type for supported in self.supported_tasks)

    async def _execute_task_impl(self, task: AgentTask) -> Any:
        """执行文本搜索任务"""
        if not task.input_data or not isinstance(task.input_data, dict):
            raise ValueError("任务输入数据格式错误")

        query = task.input_data.get("query")
        documents = task.input_data.get("documents")

        if not query or not documents:
            raise ValueError("搜索任务缺少查询或文档")

        # 使用文本搜索工具
        tool = self.tools[0]  # TextSearchTool
        input_data = {"query": query, "documents": documents, **task.metadata}

        result = tool.execute(input_data)

        if "error" in result:
            raise Exception(result["error"])

        return result


class WebSearchAgent(BaseAgent):
    """网络搜索代理"""

    def __init__(self):
        super().__init__(
            name="WebSearcher",
            description="专门进行网络搜索的智能代理，获取最新信息",
            tools=[WebSearchTool()],
        )

    @property
    def supported_tasks(self) -> List[str]:
        return ["web_search", "information_gathering", "research"]

    def can_handle_task(self, task: AgentTask) -> bool:
        """判断是否能处理任务"""
        task_type = task.metadata.get("task_type", "").lower()
        # 支持的任务类型映射
        if task_type == "web":
            return True
        return any(supported in task_type for supported in self.supported_tasks)

    async def _execute_task_impl(self, task: AgentTask) -> Any:
        """执行网络搜索任务"""
        if not task.input_data:
            raise ValueError("任务缺少输入数据")

        # 使用网络搜索工具
        tool = self.tools[0]  # WebSearchTool
        input_data = {"query": task.input_data, **task.metadata}

        result = tool.execute(input_data)

        if "error" in result:
            raise Exception(result["error"])

        return result


class AnalysisAgent(BaseAgent):
    """分析代理 - 综合分析代理"""

    def __init__(self):
        super().__init__(
            name="Analyst",
            description="综合分析代理，能够协调其他代理完成复杂的分析任务",
            tools=[],
        )

    @property
    def supported_tasks(self) -> List[str]:
        return ["analysis", "research", "comprehensive_analysis"]

    def can_handle_task(self, task: AgentTask) -> bool:
        """判断是否能处理任务"""
        task_type = task.metadata.get("task_type", "").lower()
        # 支持的任务类型映射
        if task_type == "analysis":
            return True
        return any(supported in task_type for supported in self.supported_tasks)

    async def _execute_task_impl(self, task: AgentTask) -> Any:
        """执行综合分析任务"""
        # 这是一个高级代理，可以协调其他代理
        analysis_type = task.metadata.get("analysis_type", "general")

        if analysis_type == "document_analysis":
            # 文档分析任务
            if not task.input_data:
                raise ValueError("文档分析任务缺少输入数据")

            # 创建子任务
            subtasks = []

            # 1. 处理文档
            processing_task = AgentTask(
                id=f"{task.id}_processing",
                description=f"处理文档: {task.input_data}",
                input_data=task.input_data,
                metadata={
                    "task_type": "document_processing",
                    "action": "split",
                    "splitter": "recursive",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                },
            )
            subtasks.append(processing_task)

            # 这里可以添加更多的子任务...

            # 返回分析结果
            return {
                "analysis_type": "document_analysis",
                "input_source": task.input_data,
                "subtasks": [subtask.id for subtask in subtasks],
                "status": "ready_for_execution",
            }

        elif analysis_type == "web_research":
            # 网络研究任务
            if not task.input_data:
                raise ValueError("网络研究任务缺少输入数据")

            return {
                "analysis_type": "web_research",
                "research_query": task.input_data,
                "strategy": "multi_source_analysis",
                "status": "ready_for_execution",
            }

        else:
            # 通用分析
            return {
                "analysis_type": "general",
                "task_description": task.description,
                "input_data": task.input_data,
                "metadata": task.metadata,
                "recommendation": "需要更多具体信息来执行分析",
            }


# 代理工厂函数
def create_document_agent() -> DocumentProcessingAgent:
    """创建文档处理代理"""
    return DocumentProcessingAgent()


def create_search_agent() -> TextSearchAgent:
    """创建文本搜索代理"""
    return TextSearchAgent()


def create_web_search_agent() -> WebSearchAgent:
    """创建网络搜索代理"""
    return WebSearchAgent()


def create_analysis_agent() -> AnalysisAgent:
    """创建分析代理"""
    return AnalysisAgent()


# 预定义的代理配置
AGENT_CONFIGS = {
    "document_processor": create_document_agent,
    "text_searcher": create_search_agent,
    "web_searcher": create_web_search_agent,
    "analyst": create_analysis_agent,
}


def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """根据类型创建代理"""
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(
            f"不支持的代理类型: {agent_type}. 支持的类型: {list(AGENT_CONFIGS.keys())}"
        )

    return AGENT_CONFIGS[agent_type](**kwargs)
