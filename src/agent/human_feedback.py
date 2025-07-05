import logging
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from src.agent.state import ResearchState

logger = logging.getLogger(__name__)


def human_feedback(
    state: ResearchState, config: RunnableConfig
) -> Command[Literal["lead_agent", "sub_agent", "__end__"]]:
    """Human feedback"""

    feedback = interrupt("Please Review the Plan.")
    if feedback and feedback.get("accept"):
        logger.info("accepted")
        return Command(goto="generate_multi_sub_agent")
    logger.info("feedback: %s", feedback)

    return Command(goto="__end__")
