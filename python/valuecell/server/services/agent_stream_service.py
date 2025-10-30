"""
Agent stream service for handling streaming agent interactions.
"""

import logging
import os
import time
from datetime import datetime
from typing import AsyncGenerator, Optional

from valuecell.core.coordinate.orchestrator import AgentOrchestrator
from valuecell.core.types import UserInput, UserInputMetadata
from valuecell.utils.uuid import generate_conversation_id

logger = logging.getLogger(__name__)


class AgentStreamService:
    """Service for handling streaming agent queries."""

    def __init__(self):
        """Initialize the agent stream service."""
        self.orchestrator = AgentOrchestrator()
        logger.info("Agent stream service initialized")

    @staticmethod
    def _resolve_model_info(agent_name: Optional[str]) -> tuple[str, str]:
        env = os.getenv
        if not agent_name:
            mid = env("RESEARCH_AGENT_MODEL_ID", "unknown")
        else:
            name = agent_name.lower()
            if "research" in name:
                mid = env("RESEARCH_AGENT_MODEL_ID", "unknown")
            elif "auto" in name and "trading" in name:
                mid = env("TRADING_PARSER_MODEL_ID", env("TRADING_PARSER_MODEL_ID", "unknown"))
            elif "sec" in name:
                mid = env("SEC_ANALYSIS_MODEL_ID", env("SEC_PARSER_MODEL_ID", "unknown"))
            else:
                mid = env("RESEARCH_AGENT_MODEL_ID", "unknown")

        provider = "未知供应商"
        openrouter_key = env("OPENROUTER_API_KEY")
        openai_key = env("OPENAI_API_KEY")
        if openrouter_key:
            provider = "OpenRouter"
        if mid.startswith("google/") or mid.startswith("gemini"):
            provider = "Google"
        elif mid.startswith("openai/") or openai_key:
            provider = "OpenAI"
        elif mid.startswith("anthropic/"):
            provider = "Anthropic"
        elif mid.startswith("deepseek/") and openrouter_key:
            provider = "OpenRouter"
        elif mid.startswith("deepseek/"):
            provider = "DeepSeek"
        return provider, mid

    async def stream_query_agent(
        self,
        query: str,
        agent_name: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream agent responses for a given query.

        Args:
            query: User query to process
            agent_name: Optional specific agent name to use. If provided, takes precedence over query parsing.
            conversation_id: Optional conversation ID for context tracking.

        Yields:
            str: Content chunks from the agent response
        """
        try:
            logger.info(f"Processing streaming query: {query[:100]}...")

            user_id = "default_user"
            target_agent_name = agent_name

            conversation_id = conversation_id or generate_conversation_id()

            user_input_meta = UserInputMetadata(
                user_id=user_id, conversation_id=conversation_id
            )

            user_input = UserInput(
                query=query, target_agent_name=target_agent_name, meta=user_input_meta
            )

            provider, model_id = self._resolve_model_info(target_agent_name)

            start_ts = time.monotonic()
            start_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 先发一条元信息（中文）：供应商 / 模型 / 开始时间
            yield {
                "event": "message_chunk",
                "content": (
                    f"模型供应商：{provider}\n"
                    f"模型：{model_id}\n"
                    f"开始时间：{start_human}"
                ),
            }

            # Use the orchestrator's process_user_input method for streaming
            async for response_chunk in self.orchestrator.process_user_input(
                user_input
            ):
                yield response_chunk.model_dump(exclude_none=True)

            # 结束后补一条耗时信息
            elapsed = time.monotonic() - start_ts
            yield {
                "event": "message_chunk",
                "content": f"生成完成，用时：{elapsed:.2f} 秒",
            }

        except Exception as e:
            logger.error(f"Error in stream_query_agent: {str(e)}")
            yield f"Error processing query: {str(e)}"
