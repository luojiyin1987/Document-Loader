#!/usr/bin/env python3
"""
Agents 和 AgentExecutor 系统
提供智能代理框架，支持多种任务类型和工具调用
"""

import asyncio

# ===== 标准库导入 =====
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(Enum):
    """代理状态枚举"""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentTask:
    """代理任务数据结构"""

    id: str
    description: str
    input_data: Any = None
    expected_output: Optional[str] = None
    priority: int = 1
    max_retries: int = 3
    timeout: int = 300  # 5分钟超时
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """代理执行结果数据结构"""

    task_id: str
    status: AgentStatus
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_info: Dict[str, Any] = field(default_factory=dict)


class ToolInterface(ABC):
    """工具接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass

    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """执行工具"""
        pass

    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        pass


class AgentInterface(ABC):
    """代理接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """代理名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """代理描述"""
        pass

    @property
    @abstractmethod
    def supported_tasks(self) -> List[str]:
        """支持的任务类型"""
        pass

    @property
    @abstractmethod
    def status(self) -> AgentStatus:
        """代理状态"""
        pass

    @abstractmethod
    def can_handle_task(self, task: AgentTask) -> bool:
        """判断是否能处理任务"""
        pass

    @abstractmethod
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """执行任务"""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """获取代理能力"""
        pass


class BaseAgent(AgentInterface):
    """基础代理实现"""

    def __init__(self, name: str, description: str, tools: Optional[List[ToolInterface]] = None):
        self._name = name
        self._description = description
        self.tools = tools or []
        self._status = AgentStatus.IDLE
        self.execution_history: List[AgentResult] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def supported_tasks(self) -> List[str]:
        return ["general"]

    @property
    def status(self) -> AgentStatus:
        return self._status

    @status.setter
    def status(self, value: AgentStatus):
        self._status = value

    def can_handle_task(self, task: AgentTask) -> bool:
        """基础实现：检查任务类型是否支持"""
        return "general" in self.supported_tasks

    def add_tool(self, tool: ToolInterface):
        """添加工具"""
        self.tools.append(tool)

    def get_capabilities(self) -> Dict[str, Any]:
        """获取代理能力"""
        return {
            "name": self.name,
            "description": self.description,
            "supported_tasks": self.supported_tasks,
            "tools": [{"name": tool.name, "description": tool.description} for tool in self.tools],
            "status": self.status.value,
            "total_executions": len(self.execution_history),
        }

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """执行任务的通用实现"""
        start_time = time.time()
        self._status = AgentStatus.RUNNING

        try:
            # 子类需要实现具体的任务执行逻辑
            result = await self._execute_task_impl(task)

            self._status = AgentStatus.COMPLETED
            execution_time = time.time() - start_time

            agent_result = AgentResult(
                task_id=task.id, status=self._status, output=result, execution_time=execution_time, agent_info={"name": self.name, "description": self.description}
            )

        except Exception as e:
            self._status = AgentStatus.ERROR
            execution_time = time.time() - start_time

            agent_result = AgentResult(
                task_id=task.id, status=self._status, error=str(e), execution_time=execution_time, agent_info={"name": self.name, "description": self.description}
            )

        self.execution_history.append(agent_result)
        return agent_result

    @abstractmethod
    async def _execute_task_impl(self, task: AgentTask) -> Any:
        """子类需要实现的具体任务执行逻辑"""
        pass


class AgentExecutor:
    """代理执行器 - 负责任务调度和代理管理"""

    def __init__(self):
        self.agents: Dict[str, AgentInterface] = {}
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: Dict[str, AgentResult] = {}
        self.running_tasks: Dict[str, AgentInterface] = {}
        self.max_concurrent_tasks = 3

    def register_agent(self, agent: AgentInterface):
        """注册代理"""
        self.agents[agent.name] = agent
        print(f"已注册代理: {agent.name}")

    def unregister_agent(self, agent_name: str):
        """注销代理"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            print(f"已注销代理: {agent_name}")

    def submit_task(self, task: AgentTask) -> str:
        """提交任务"""
        self.task_queue.append(task)
        print(f"已提交任务: {task.id} - {task.description}")
        return task.id

    def get_task_result(self, task_id: str) -> Optional[AgentResult]:
        """获取任务结果"""
        return self.completed_tasks.get(task_id)

    def get_agent_status(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """获取代理状态"""
        if agent_name in self.agents:
            return self.agents[agent_name].get_capabilities()
        return None

    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有代理及其状态"""
        return [agent.get_capabilities() for agent in self.agents.values()]

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """执行单个任务"""
        # 选择合适的代理
        selected_agent = self._select_agent_for_task(task)
        if not selected_agent:
            return AgentResult(task_id=task.id, status=AgentStatus.ERROR, error="没有找到合适的代理来处理此任务")

        print(f"代理 {selected_agent.name} 开始执行任务: {task.id}")
        result = await selected_agent.execute_task(task)
        print(f"代理 {selected_agent.name} 完成任务: {task.id}, 状态: {result.status.value}")

        return result

    async def execute_next_task(self) -> Optional[AgentResult]:
        """执行下一个任务"""
        if not self.task_queue:
            return None

        # 检查并发限制
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            print(f"达到最大并发任务数限制: {self.max_concurrent_tasks}")
            return None

        # 获取下一个任务（按优先级排序）
        self.task_queue.sort(key=lambda x: x.priority, reverse=True)
        task = self.task_queue.pop(0)

        # 执行任务
        result = await self.execute_task(task)
        self.completed_tasks[task.id] = result

        # 从运行中任务列表移除
        if task.id in self.running_tasks:
            del self.running_tasks[task.id]

        return result

    async def run_all_tasks(self) -> List[AgentResult]:
        """执行所有任务"""
        results = []

        while self.task_queue or self.running_tasks:
            result = await self.execute_next_task()
            if result:
                results.append(result)
            else:
                # 没有任务可以执行，等待一下
                await asyncio.sleep(0.1)

        return results

    def _select_agent_for_task(self, task: AgentTask) -> Optional[AgentInterface]:
        """为任务选择合适的代理"""
        available_agents = []

        for agent in self.agents.values():
            if agent.can_handle_task(task) and agent.status != AgentStatus.RUNNING:
                available_agents.append(agent)

        if not available_agents:
            return None

        # 简单策略：选择第一个可用的代理
        # 可以扩展为更复杂的负载均衡策略
        return available_agents[0]

    def get_statistics(self) -> Dict[str, Any]:
        """获取执行器统计信息"""
        total_tasks = len(self.completed_tasks)
        successful_tasks = sum(1 for result in self.completed_tasks.values() if result.status == AgentStatus.COMPLETED)
        failed_tasks = sum(1 for result in self.completed_tasks.values() if result.status == AgentStatus.ERROR)

        avg_execution_time = 0
        if successful_tasks > 0:
            total_time = sum(result.execution_time for result in self.completed_tasks.values() if result.status == AgentStatus.COMPLETED)
            avg_execution_time = total_time / successful_tasks

        return {
            "total_agents": len(self.agents),
            "pending_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "average_execution_time": avg_execution_time,
        }
