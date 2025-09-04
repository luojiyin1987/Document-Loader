#!/usr/bin/env python3
"""
测试Agents系统
"""

import asyncio

from agents import AgentTask
from example_agents import create_document_agent


async def test_document_agent():
    """测试文档处理代理"""
    print("=== 测试文档处理代理 ===")

    # 创建代理
    agent = create_document_agent()
    print(f"代理名称: {agent.name}")
    print(f"代理描述: {agent.description}")

    # 创建任务
    task = AgentTask(
        id="test_001",
        description="测试文档处理",
        input_data="README.md",
        metadata={"task_type": "document_processing", "action": "read"},
    )

    print(f"任务ID: {task.id}")
    print(f"任务描述: {task.description}")

    # 执行任务
    try:
        result = await agent.execute_task(task)
        print(f"执行结果: {result.status}")
        if result.error:
            print(f"错误: {result.error}")
        else:
            print(f"输出: {result.output}")
    except Exception as e:
        print(f"执行异常: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_document_agent())
