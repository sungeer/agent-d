import json

from langchain_core.messages import BaseMessage, ToolMessage


class ShortTerm:
    """短期记忆，按 token 数控制上限，最大化利用上下文窗口。"""

    def __init__(self, max_tokens: int = 1_000_000) -> None:
        self.max_tokens = max_tokens
        self._messages: list[BaseMessage] = []

    def add(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self._trim()

    @staticmethod
    def _count_tokens(message: BaseMessage) -> int:
        """估算单条消息的 token 数。"""
        parts: list[str] = []
        # 消息内容
        content = message.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # 多模态内容块
            for block in content:
                if isinstance(block, dict) and 'text' in block:
                    parts.append(block['text'])
        # tool_calls 的开销
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                parts.append(json.dumps(tc, ensure_ascii=False))
        # ToolMessage 的 tool_call_id
        if isinstance(message, ToolMessage):
            parts.append(message.tool_call_id)
        text = ''.join(parts)
        return max(1, len(text) // 2)

    def _trim(self) -> None:
        """超出 token 上限时，从旧消息开始丢弃。"""
        while self._messages:
            total = sum(self._count_tokens(m) for m in self._messages)
            if total <= self.max_tokens:
                break
            self._messages.pop(0)

    def get_messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def clear(self) -> None:
        """清空全部历史（开始新对话时使用）。"""
        self._messages = []
