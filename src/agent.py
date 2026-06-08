import logging

from langchain_core.messages import SystemMessage, HumanMessage

from src.llm import llm
from src import toolset
from src.memory import ShortTerm

log = logging.getLogger(__name__)

system_prompt = (
    '你是一个智能助手，可以反复使用工具来逐步完成复杂任务。\n'
    '行动前先简要说明你当前的理解和下一步计划。\n'
    '如果信息足够，直接回答用户。\n'
    '如果有不确定的地方，直接向用户提问澄清。\n'
    '不要用相同参数重复调用同一个工具。\n'
    '搜索结果为中文时优先采用中文关键词。'
)

tools = [
    toolset.get_weather,
    toolset.web_search,
    toolset.get_current_time,
    toolset.calculate,
]
tools_map = {t.name: t for t in tools}


def run_agent(user_input: str, memory: ShortTerm, max_steps: int = 10) -> str:
    memory.add(HumanMessage(content=user_input))

    llm_with_tools = llm.bind_tools(tools)

    for i in range(max_steps):
        messages = [SystemMessage(content=system_prompt)] + memory.get_messages()
        response = llm_with_tools.invoke(messages)
        memory.add(response)

        if response.content:
            # 思考过程
            log.info(f'[thought] {response.content[:200]}')

        if not response.tool_calls:
            log.info(f'无需工具调用，第[{i}]轮结束')
            return response.content or ''

        log.info(f'工具调用第[{i + 1}]轮')

        for tc in response.tool_calls:
            tool_func = tools_map.get(tc['name'])
            if tool_func is None:
                log.warning(f'未知工具: {tc["name"]}')
                continue

            log.info(f'执行工具: {tc["name"]}，参数: {tc["args"]}')

            result = tool_func.invoke(tc)

            log.info(f'工具结果: {str(result)[:100]}')

            memory.add(result)

    log.warning(f'工具调用达到上限{max_steps}轮，强制总结')

    messages = [SystemMessage(content=system_prompt)] + memory.get_messages()
    messages.append(SystemMessage(content='请根据已有的工具返回信息，简洁地回答用户的问题。'))
    response = llm.invoke(messages)
    memory.add(response)
    return response.content or ''
